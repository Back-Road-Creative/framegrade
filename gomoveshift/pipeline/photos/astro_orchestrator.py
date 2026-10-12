"""Astro orchestrator prepare half: light (+ dark)
RAW paths -> calibrated scene-linear frames, triage metrics, a registration reference."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .astro_calibrate import (
    _STRICT_IDENTITY_FIELDS,  # read here so this file's readability rule cannot drift from it
    AstroCalibrationError,
    AstroHighEfficiencyError,
    DecodedAstroFrame,
    build_master_dark,
    decode_linear,
    hot_pixel_map,
    read_capture_identity,
    read_raw_mosaic,
)
from .astro_develop import (
    AstroDevelopResult,
    chroma_sigma,
    denoise_chroma,
    develop_stack,
    develop_stack_with_params,
    preview_render,
)
from .astro_geometry import (
    MIN_MATCHES_PER_FRAME,
    FrameMatches,
    GeometrySolveError,
    solve_night,
)
from .astro_lens import LensParams, build_remap
from .astro_linear import linear_algorithm_hash, write_linear_artifact
from .astro_meteors import STREAK_FEATHER_PX, reinsert
from .astro_register import (
    MATCH_WIDE_TOLERANCE_PX,
    RegistrationError,
    fit_similarity,
    match_stars,
    radial_inlier_residuals_px,
    register,
    warp_to_reference,
    warp_validity,
)
from .astro_sidecar import (
    UsedFrame,
    _star_shape_payload,
    stack_sidecar_payload,
    write_stack_sidecar,
)
from .astro_stack import (
    MIN_AUTO_HORIZON_STARS,
    MIN_STACK_FRAMES,
    StackResult,
    horizon_mask,
    stack,
)
from .astro_triage import (
    STAR_DETECT_SIGMA,
    FrameMetrics,
    TriageReport,
    classify_streaks,
    detect_stars,
    measure_frame,
    triage,
)
from .exif_gps import extract_exif_capture_metadata

logger = logging.getLogger(__name__)

# BT.709 luma weights, as elsewhere in this codebase (color_correction.py). Kept
# local rather than imported from astro_develop.py's private, later-stage ``_luma``.
_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


class AstroOrchestratorError(RuntimeError):
    """Not enough usable frames to hand off to registration/stacking."""


class MeteorPolicyError(ValueError):
    """``MeteorPolicy.mode`` named an unsupported posture -- only off/reviewed ship until a precision gate exists."""


# Excessive-candidate guard (PROVISIONAL): a very large number of "meteor"-labelled
# streaks in one run is a misclassification storm, not genuine meteors -- above this, skip all.
MAX_METEOR_CANDIDATES = 50

# prepare_frames' default decode concurrency. 4, not the core count: RAM is this path's
# binding constraint, not CPU (an out-of-memory failure ends the whole stack), and each
# worker past the first costs another live frame plus its demosaic buffers -- on the
# order of 16 bytes per pixel, about 0.3 GB for a 20 MP frame. 4 buys most of the speed-up
# that 8 workers reach while charging a fraction of the extra memory, and the preflight is
# told the count either way so a bigger --decode-workers is refused up front rather than met
# as an OOM.
DECODE_WORKERS_DEFAULT = 4

# How much poorer than the chunk's richest frame a frame may be and still count as its
# equal when the reference is chosen. 5%: star count climbs
# monotonically as the sky darkens, so a plain "most stars" rule really picks "the
# latest frame", and the runner-up frame is nearly always within a few percent of the
# best. Inside this band the counts say nothing; above it a
# frame is genuinely richer and still wins outright.
REFERENCE_STAR_COUNT_TIE_FRACTION = 0.05


def pick_reference_index(metrics, candidates, total_frames: int) -> int:
    """The index in ``metrics`` every other frame registers against: the frame nearest the
    chunk's MIDDLE among those statistically tied with the richest on ``star_count``.

    Every frame is warped onto the reference, so the reference's position sets the worst-
    case sky rotation any single fit must absorb -- a reference at the end asks the first
    frame to bridge the chunk's WHOLE span, the middle asks for half. That span is what
    the outer-third inlier floor measures: at 10 x 30 s the ends of a chunk are ~2.5
    minutes of rotation apart, far enough that corner stars leave the search windows and
    the fit collapses to the frame centre. Choosing an end frame as the
    reference therefore loses frames to ``too few outer-third inliers``, and the
    rejections are positional: first frames are lost, middle frames are not.

    ``total_frames`` (not ``len(candidates)``) defines the centre: rejected frames still
    occupy their slice of time, so the middle of the CHUNK is what halves the rotation,
    not the middle of whatever survived triage.

    Ordering inside the band: nearest the centre, then richer, then sharper (lower
    ``fwhm_px``), then lower index so a run is reproducible."""
    best_star_count = max(metrics[i].star_count for i in candidates)
    floor = best_star_count * (1.0 - REFERENCE_STAR_COUNT_TIE_FRACTION)
    centre = (total_frames - 1) / 2.0
    tied = [i for i in candidates if metrics[i].star_count >= floor]
    return min(tied, key=lambda i: (abs(i - centre), -metrics[i].star_count, metrics[i].fwhm_px, i))


@dataclass(frozen=True)
class MeteorPolicy:
    """Meteor-reinsertion posture for one ``stack_run`` call. ``mode="off"`` (default
    -- one streak is not proof of a meteor) reinserts nothing. ``"reviewed"``
    reinserts only ``allowlist``'s named ``(frame_index, streak_index)`` identities
    (``streak_index``: a streak's 0-based position among its OWN frame's entries, in
    ``classify_streaks``'s deterministic emission order). No "auto" policy exists --
    ``stack_run`` raises ``MeteorPolicyError`` for one."""

    mode: str = "off"
    allowlist: frozenset[tuple[int, int]] = frozenset()


@dataclass(frozen=True)
class PreparedFrames:
    """Output of ``prepare_frames``: everything register/stack/develop needs."""

    frames: tuple[DecodedAstroFrame, ...]  # scene-linear, calibrated, in input order
    metrics: tuple[FrameMetrics, ...]  # parallel to frames, same order
    report: TriageReport
    reference_index: int
    failed: tuple[tuple[Path, str], ...]  # (path, reason) for frames that could not decode
    # The darks this chunk was actually calibrated against, carried so ``stack_run`` --
    # which sees only a ``PreparedFrames`` -- can record them in the provenance sidecar.
    # ``()`` is the honest "no dark was subtracted" claim, never an unknown.
    dark_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class AstroReport:
    """Output of ``stack_run``: the written export, its provenance sidecar (``None`` on a
    (non-fatal, by contract) sidecar write failure), the raw stage results, and every
    frame dropped along the way -- decode failures inherited from ``PreparedFrames.failed``
    plus any triage-verdict rejection (``gate_triage``) or registration failure
    ``stack_run`` itself hit."""

    image_path: Path | None
    sidecar_path: Path | None
    stack: StackResult
    develop: AstroDevelopResult | None
    rejected_frames: tuple[tuple[Path, str], ...]
    meteors_reinserted: int  # count of "meteor"-labelled streaks actually composited
    photometry: tuple[FramePhotometry, ...] = ()  # per-frame offset/scale/weight, ref first
    linear_artifact_path: Path | None = None
    # Which warp model this chunk's stacking path actually used, and why. See
    # ``_chunk_geometry_payload``; ``None`` only on paths that never reach the warp.
    geometry: dict | None = None
    # Colour speckle in the written master, before and after the writer's chroma blur
    # (``astro_develop.denoise_chroma``). Recorded rather than assumed: the step is
    # cosmetic and irreversible, so a run says what it actually changed.
    chroma_sigma_before: float = 0.0
    chroma_sigma_after: float = 0.0
    # The display JPEG written beside the master (``astro_develop.preview_render``).
    # ``None`` when that render failed, which never fails the run.
    preview_path: Path | None = None


def _luma(rgb: np.ndarray) -> np.ndarray:
    """Single-channel luma off a decoded frame's linear RGB, for triage only."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    weighted = _LUMA_WEIGHTS[0] * r + _LUMA_WEIGHTS[1] * g + _LUMA_WEIGHTS[2] * b
    return weighted.astype(np.float32)


# Photometric normalisation: equal exposure/ISO never implies equal sky
# pedestal. Measured PRE-warp at the SAME sky_stars_mask pixel positions in every frame,
# never a per-frame-transformed mask: a static rig's gradient is fixed in pixel space
# while only the stars drift, so tracking the mask fabricates an offset (pinned by
# test_photometry_is_a_noop_under_genuine_star_field_turnover).
#
# PEDESTAL-ONLY: ``scale`` is pinned at 1.0, never estimated. Two earlier
# throughput (scale) estimators -- a 99th-percentile sky-pixel proxy, and a star-matched
# flux ratio as its fallback -- are deliberately not used, same posture as everything else
# PROVISIONAL in this file. Only the masked-median pedestal survives: a median tolerates up to 50% pixel contamination BY
# CONSTRUCTION, so it stays correct whether star-lit pixels are 0.1% or 20% of the mask
# -- unlike a percentile proxy, which needs a non-trivial star-lit share before it reads
# stars at all and otherwise reads sky noise/gradient, producing a false rejection or a
# fabricated correction on a real (sparse) night sky.


@dataclass(frozen=True)
class FramePhotometry:
    """One frame's photometric normalisation: ``corrected = frame - offset``.
    ``scale`` is always 1.0 -- no throughput/gain correction is applied (module
    comment above). ``weight`` is always 1.0."""

    path: Path
    offset: float
    scale: float
    weight: float


def _sky_pedestal(luma, sky_mask):
    """Robust median sky level over ``sky_mask``-ed pixels."""
    return float(np.median(np.asarray(luma, dtype=np.float64)[sky_mask]))


def _frame_photometry(frame_luma, ref_pedestal, sky_mask):
    """Offset mapping ``frame_luma``'s masked sky pedestal onto the reference's,
    measured at the SAME ``sky_mask`` positions the reference used (module comment
    above). ``scale`` is always 1.0 (pinned, never estimated -- module comment above).
    Raises ``ValueError`` only if the offset comes out non-finite (an empty ``sky_mask``
    -- never produced by a real horizon_mask, but never trusted blind)."""
    offset = _sky_pedestal(frame_luma, sky_mask) - ref_pedestal
    if not np.isfinite(offset):
        raise ValueError(f"derived offset {offset!r} is non-finite")
    return offset, 1.0


def decode_with_he_fallback(path, master_dark, hotmap):
    """``decode_linear``, retried through DNG conversion on an HE/HE* NEF. The DNG
    always goes BACK to ``decode_linear`` (scene-linear) -- never
    ``develop._decode_he_via_dng`` (display-referred sRGB), which would silently
    corrupt the stack. Returns ``(DecodedAstroFrame, None)`` or ``(None, reason)``.
    """
    try:
        return decode_linear(path, master_dark, hotmap), None
    except AstroHighEfficiencyError as exc:
        from .develop import _convert_he_to_dng  # lazy: no develop<->astro import cycle

        dng_path = _convert_he_to_dng(path)
        if dng_path is None:
            return None, str(exc)
        try:
            dng_decoded = decode_linear(dng_path, master_dark, hotmap)
        except AstroHighEfficiencyError as dng_exc:
            return None, str(dng_exc)
        frame = DecodedAstroFrame(
            dng_decoded.rgb, Path(path), "he-dng-linear", dng_decoded.bit_depth
        )
        return frame, None


def _require_readable_identity(identity, role: str) -> None:
    """Refuse an identity whose strict fields didn't read. ``validate_dark_identity``
    SKIPS a field that is ``None`` on either side, so an unreadable ISO/exposure would
    otherwise pass as if it matched -- and "I could not read the exposure" is not
    evidence that a dark matches the lights."""
    unreadable = [f for f in _STRICT_IDENTITY_FIELDS if getattr(identity, f) is None]
    if unreadable:
        raise AstroCalibrationError(
            f"{identity.path}: this {role}'s {', '.join(unreadable)} could not be read, so "
            "no dark can be proven to match the lights -- drop this file from the run, or "
            "re-run without --darks to skip dark subtraction entirely"
        )


def build_validated_master_dark(light_paths, dark_paths) -> np.ndarray:
    """Master dark for ``light_paths``, refusing any dark whose capture identity
    disagrees with the lights. The single seam both astro paths use --
    ``prepare_frames`` and ``astro_trails.render_trail`` -- so neither can build an
    unvalidated master. A refusal is fatal to the chunk, never per-dark: ``--darks`` is
    one per-run set, so dropping the odd frame silently would hand back a master built
    from an unexamined subset; the message names the file and the field instead.

    The light identity is the FIRST light path given (a chunk's earliest frame in
    capture order on the CLI path), and the refusal names it. It is representative by
    construction: ``astro_sequence`` splits a run the moment ISO changes at all, and
    camera/raw_shape/cfa_pattern are body properties fixed for one card, so four of the
    five compared fields cannot vary within a chunk. The fifth, ``exposure_s``, is held
    only within ``astro_sequence.MAX_HOLD_RATIO`` (1.15x, measured between neighbouring
    frames of one detected run) while the comparison is exact equality -- so judging the
    darks against every light in turn would refuse runs the sequencer deliberately
    admits, and one anchor frame is the honest reference.

    An unreadable field is refused before the first mosaic is read; a mismatch is caught
    inside ``build_master_dark``, after the dark mosaics but before the median and
    before any light is decoded (on the order of a minute per frame).
    """
    first_light = next(iter(light_paths), None)
    if first_light is None:
        raise AstroCalibrationError(
            "build_validated_master_dark: no light frames to validate the darks against"
        )
    light_identity = read_capture_identity(first_light)
    _require_readable_identity(light_identity, "light")
    dark_identities = [read_capture_identity(p) for p in dark_paths]
    for dark_identity in dark_identities:
        _require_readable_identity(dark_identity, "dark")
    return build_master_dark(
        [read_raw_mosaic(p) for p in dark_paths],
        light_identity=light_identity,
        dark_identities=dark_identities,
    )


def prepare_frames(
    light_paths,
    *,
    dark_paths=(),
    horizon_mask=None,
    cloud_baseline=None,
    decode_workers=1,
    detect_meteors=True,
) -> PreparedFrames:
    """Calibrate ``light_paths`` and triage them into a ``PreparedFrames``. Raises
    ``AstroOrchestratorError`` when every light failed to decode, or fewer than
    ``astro_stack.MIN_STACK_FRAMES`` usable frames remain to register/stack, and
    ``AstroCalibrationError`` (``AstroDarkIdentityMismatchError`` on a disagreement)
    when a supplied dark can't be proven to match the lights -- see
    ``build_validated_master_dark``. No ``dark_paths`` reads no identity at all.

    ``detect_meteors=False`` skips each frame's streak scan -- by far the largest part of
    triage's cost -- and leaves ``streaks`` as ``None``
    rather than ``[]``, so ``classify_streaks`` refuses the run instead of reporting a
    meteor-free night it never looked for.

    ``decode_workers`` > 1 decodes and triages frames on that many THREADS. Results stay in
    SOURCE ORDER regardless (see the loop) -- ``frames``/``metrics`` are paired positionally
    and ``reference_index`` indexes into both. Each worker beyond the first holds another
    frame and its demosaic buffers live, which ``preflight.estimate_astro_stack_ram_gb``
    must be told about via its own ``decode_workers``; sizing the gate for serial decode and
    then running concurrent decode is how a run OOMs past a preflight that passed.

    ``cloud_baseline`` (optional, ``(star, background)``, a caller's running
    ``astro_triage.update_run_cloud_baseline`` result) threads straight through to
    ``triage`` as its ``run_baseline`` -- omitted, one chunk's cloud triage is
    judged against itself alone, same as before this parameter existed."""
    master_dark = None
    hotmap = None
    if dark_paths:
        master_dark = build_validated_master_dark(light_paths, dark_paths)
        hotmap = hot_pixel_map(master_dark)

    def _prepare_one(light_path):
        """One frame's decode + EXIF + triage, as a unit. Returns
        ``(path, decoded_or_None, metric_or_None, reason_or_None)``."""
        light_path = Path(light_path)
        decoded, reason = decode_with_he_fallback(light_path, master_dark, hotmap)
        if decoded is None:
            return light_path, None, None, reason
        meta = extract_exif_capture_metadata(light_path)
        metric = measure_frame(
            light_path,
            _luma(decoded.rgb),
            mask=horizon_mask,
            exposure_s=meta.exposure_s,
            detect_meteors=detect_meteors,
        )
        return light_path, decoded, metric, None

    # THREADS, not processes: rawpy/LibRaw's unpack+postprocess and the HE path's DNG
    # converter subprocess both release the GIL, so threads parallelise for real --
    # per-frame time drops sharply with a few workers on a high-resolution body. Processes would
    # have to pickle each ~250 MB decoded frame back to the parent, paying a copy this
    # shares for free.
    #
    # ``executor.map`` yields in SUBMISSION order, not completion order, which is load-
    # bearing rather than incidental: ``frames`` and ``metrics`` are paired positionally,
    # ``reference_index`` indexes into both, and the CLI names each chunk by its first and
    # last source frame. A completion-ordered collection would pair every frame with a
    # different frame's triage and silently mislabel every output.
    if decode_workers > 1:
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            prepared_rows = list(pool.map(_prepare_one, light_paths))
    else:
        prepared_rows = [_prepare_one(p) for p in light_paths]

    frames: list[DecodedAstroFrame] = []
    metrics: list[FrameMetrics] = []
    failed: list[tuple[Path, str]] = []
    for light_path, decoded, metric, reason in prepared_rows:
        if decoded is None:
            failed.append((light_path, reason))
            continue
        frames.append(decoded)
        metrics.append(metric)

    if not frames:
        raise AstroOrchestratorError(
            f"prepare_frames: every light frame failed to decode ({len(failed)} of {len(light_paths)})"
        )

    frames_t, metrics_t = tuple(frames), tuple(metrics)
    report = triage(metrics_t, run_baseline=cloud_baseline)

    usable = [i for i, m in enumerate(metrics_t) if m.usable]
    if len(usable) < MIN_STACK_FRAMES:
        raise AstroOrchestratorError(
            f"prepare_frames: only {len(usable)} usable of {len(frames_t)} decoded "
            f"({MIN_STACK_FRAMES} required to register/stack)"
        )
    reference_index = pick_reference_index(metrics_t, usable, len(frames_t))

    return PreparedFrames(
        frames_t,
        metrics_t,
        report,
        reference_index,
        tuple(failed),
        tuple(Path(p) for p in dark_paths),
    )


def _warp_streak(streak, transform):
    """A streak's ``(x1, y1, x2, y2)`` endpoints, mapped from its OWN frame's
    coordinates into reference coordinates via that frame's registration
    ``transform`` (frame xy -> reference xy, ``astro_register``'s convention). The
    stack -- and the layer a meteor gets composited onto -- is in reference
    coordinates, so an un-warped streak lands in the wrong place. ``transform is
    None`` is the reference frame itself: already reference coordinates, no warp."""
    if transform is None:
        return streak
    matrix = np.asarray(transform, dtype=np.float64)
    x1, y1, x2, y2 = streak
    p1 = matrix[:, :2] @ np.array([x1, y1]) + matrix[:, 2]
    p2 = matrix[:, :2] @ np.array([x2, y2]) + matrix[:, 2]
    return (float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))


def _resolve_meteor_selection(streaks, policy, aligned_by_index):
    """Pure decision: which of ``streaks`` (``classify_streaks``'s output) ``policy``
    reinserts, as ``(selected, candidate_count, skip_reason)`` -- see ``MeteorPolicy``
    for the allowlist identity. ``selected`` excludes an unregistered frame's streak
    (already in ``rejected``, never guessed at here); ``skip_reason`` is set
    (``selected`` forced empty) for mode "off" or the excessive-candidate guard.
    Raises ``MeteorPolicyError`` for any ``policy.mode`` other than off/reviewed."""
    if policy.mode not in ("off", "reviewed"):
        raise MeteorPolicyError(
            f"stack_run: meteors.mode={policy.mode!r} is not supported -- only "
            "'off'/'reviewed' ship until a precision gate exists"
        )
    candidates = [s for s in streaks if s.label == "meteor"]
    if policy.mode == "off":
        return [], len(candidates), "meteors.mode='off' (safe default -- see MeteorPolicy)"
    if len(candidates) > MAX_METEOR_CANDIDATES:
        reason = (
            f"{len(candidates)} meteor candidates exceeds the excessive-candidate guard "
            f"({MAX_METEOR_CANDIDATES}); classification looks unreliable, skipping all reinsertion"
        )
        return [], len(candidates), reason
    counters: dict[int, int] = {}
    selected = []
    for streak in streaks:
        idx = counters.get(streak.frame_index, 0)
        counters[streak.frame_index] = idx + 1
        if (
            streak.label == "meteor"
            and (streak.frame_index, idx) in policy.allowlist
            and streak.frame_index in aligned_by_index
        ):
            selected.append(streak)
    return selected, len(candidates), None


def _stream_meteor_layers(selected, aligned_by_index, transform_by_index, develop_result):
    """Lazily yield ``(layer, [warped streak])`` pairs for ``reinsert``'s streaming
    form. ``selected`` is frame-ordered, so same-``frame_index`` streaks are adjacent:
    each unique source frame is developed exactly once (not once per streak) and
    at most two full-res layers (the one being composited, the one being developed) are resident."""
    layer = None
    layer_frame_index = None
    for streak in selected:
        if streak.frame_index != layer_frame_index:
            layer = develop_stack_with_params(aligned_by_index[streak.frame_index], develop_result)
            layer_frame_index = streak.frame_index
        warped = _warp_streak(streak.streak, transform_by_index[streak.frame_index])
        yield layer, [warped]


def _triage_reject_reason(metric: FrameMetrics, report: TriageReport) -> str | None:
    """Which triage verdict makes ``metric`` unfit to stack, or ``None`` if it's clean.
    Priority -- too-few-stars (``usable=False``) outranks cloud, which outranks dew --
    picks exactly one cause for a frame that trips more than one flag."""
    if not metric.usable:
        return "unusable"
    if metric.path in report.cloud_frames:
        return "cloud"
    if metric.path in report.dew_frames:
        return "dew"
    return None


def _gate_triage_verdicts(prepared: PreparedFrames):
    """``(accepted_indices, rejected, reason_counts)`` for ``gate_triage``:
    ``accepted_indices`` is a ``set[int]`` into ``prepared.frames``; ``rejected`` is
    ``(path, reason)`` pairs ready to fold into ``stack_run``'s own rejected list;
    ``reason_counts`` always carries all three keys (0 when a verdict removed nothing)
    so a floor-breach error can say exactly how many frames each reason cost."""
    accepted: set[int] = set()
    rejected: list[tuple[Path, str]] = []
    counts = {"unusable": 0, "cloud": 0, "dew": 0}
    for i, metric in enumerate(prepared.metrics):
        reason = _triage_reject_reason(metric, prepared.report)
        if reason is None:
            accepted.add(i)
        else:
            counts[reason] += 1
            rejected.append((prepared.frames[i].path, f"triage: {reason}"))
    return accepted, rejected, counts


# --- lens-corrected warp for the stacking path ----------------------------------
#
# A 2x3 similarity cannot undo barrel/pincushion distortion, so stars smear radially
# toward the corners however well the fit converges. ``astro_geometry.solve_night``
# fits one shared lens for the whole chunk plus a rotation per frame, and
# ``astro_lens.build_remap`` composes that into ONE ``cv2.remap`` pair -- the same
# number of resamples the affine path costs. Trails/timelapse keep the 2x3: their
# callers live in astro_trails/astro_timelapse and are untouched here.

# Seed focal length for the joint fit, as a fraction of the frame's pixel width: the
# rig astro_lens documents (Nikon Z DX 18-140) at its wide end, 18 mm on a 23.5 mm-wide
# DX sensor. It is a STARTING POINT only -- solve_night fits f_px itself, and a seed
# that sends the fit somewhere wrong is caught by the held-out gate below, which then
# keeps the 2x3. It is not a claim about what focal length any given night was shot at.
LENS_SEED_F_PX_PER_FRAME_WIDTH = 18.0 / 23.5

# Tolerance for re-pairing stars under register()'s own fitted transform, to hand the
# joint fit its matched pairs. Deliberately the WIDE tolerance, not the tight 3 px
# final one: the outer stars the affine misses by more than a few px are exactly the
# evidence the lens model exists to use, and a tight window would drop them and starve
# the fit of the signal it is being asked to measure.
LENS_PAIR_TOLERANCE_PX = MATCH_WIDE_TOLERANCE_PX

# Held-out outer-third stars a residual must rest on before it is evidence at all --
# below this, either model's outer number is one or two stars' luck, on either side of
# the comparison.
MIN_LENS_HOLDOUT_OUTER_STARS = 4

# The lens remap is used only when its held-out outer-third p95 is at most this share
# of the 2x3's on the SAME held-out stars. A tie is not evidence, so the margin is
# real: a lens path that merely matches the affine buys nothing and still risks the
# fit being wrong in a way the holdout happens not to see.
LENS_WARP_MAX_RESIDUAL_RATIO = 0.9


@dataclass(frozen=True)
class _PendingWarp:
    """One registered frame, held between registration and the warp: the chunk's lens
    fit needs EVERY frame's matched stars before ANY frame can be warped."""

    index: int
    frame: DecodedAstroFrame
    registration: object
    offset: float
    scale: float
    ref_xy: np.ndarray
    frame_xy: np.ndarray


def _lens_payload(lens: LensParams) -> dict:
    return {
        "k1": float(lens.k1),
        "k2": float(lens.k2),
        "cx": float(lens.cx),
        "cy": float(lens.cy),
        "f_px": float(lens.f_px),
    }


def _matched_star_pairs(stars_ref, stars_frame, transform):
    """``(ref_xy, frame_xy)`` in each frame's OWN raw pixel grid -- what
    ``astro_geometry.FrameMatches`` wants -- paired under ``register``'s own fitted 2x3.

    ``register`` does not hand its matched pairs back, so they are re-paired here
    rather than re-running its whole coarse-shift + progressive cascade: the fitted
    transform already places every frame star within a few px of its counterpart, so
    one mutual-best pass under it recovers the same pairing far more cheaply.
    """
    stars_ref = np.asarray(stars_ref, dtype=np.float64)
    stars_frame = np.asarray(stars_frame, dtype=np.float64)
    projected = stars_frame.copy()
    transform = np.asarray(transform, dtype=np.float64)
    projected[:, :2] = projected[:, :2] @ transform[:, :2].T + transform[:, 2]
    pairs = match_stars(stars_ref, projected, (0.0, 0.0), tolerance_px=LENS_PAIR_TOLERANCE_PX)
    if len(pairs) == 0:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    return stars_ref[pairs[:, 0], :2], stars_frame[pairs[:, 1], :2]


def _similarity_holdout_outer_third(ref_xy, frame_xy, frame_shape):
    """The CURRENT 2x3 path's held-out outer-third residual, measured exactly the way
    ``astro_geometry`` measures the lens model's: even-indexed pairs train, odd-indexed
    pairs held out, residual in the frame's own pixel space, binned on the frame
    position. Refitting on the train half is what makes the two numbers comparable --
    ``register``'s own residuals are in-sample and would flatter the affine.

    ``None`` when the train half cannot support a fit at all.
    """
    n = len(ref_xy)
    index = np.arange(n)
    train, holdout = index[index % 2 == 0], index[index % 2 == 1]
    if len(train) < MIN_LENS_HOLDOUT_OUTER_STARS or len(holdout) == 0:
        return None
    try:
        ref_to_frame = fit_similarity(frame_xy[train], ref_xy[train])
    except RegistrationError:
        return None
    bins = radial_inlier_residuals_px(
        ref_to_frame,
        frame_xy[holdout],
        ref_xy[holdout],
        np.ones(len(holdout), dtype=bool),
        frame_shape,
    )
    return bins["outer"]


def _solve_chunk_geometry(pending, frame_shape):
    """``(SolveResult | None, seed lens, refusal reason | None)`` for the whole chunk.

    A frame with fewer than ``MIN_MATCHES_PER_FRAME`` matched stars is left OUT of the
    joint fit rather than submitted -- ``solve_night`` refuses the entire night over one
    such frame, and one star-poor frame must not cost every other frame its lens
    correction. Those frames keep the 2x3.
    """
    height, width = frame_shape[:2]
    seed = LensParams(
        k1=0.0,
        k2=0.0,
        cx=(width - 1) / 2,
        cy=(height - 1) / 2,
        f_px=LENS_SEED_F_PX_PER_FRAME_WIDTH * width,
    )
    matches = [
        FrameMatches(frame_id=str(item.frame.path), ref_xy=item.ref_xy, frame_xy=item.frame_xy)
        for item in pending
        if len(item.ref_xy) >= MIN_MATCHES_PER_FRAME
    ]
    if not matches:
        return (
            None,
            seed,
            (
                f"no frame carried the {MIN_MATCHES_PER_FRAME} matched stars the joint lens fit needs"
            ),
        )
    try:
        return solve_night(matches, frame_shape, seed), seed, None
    except GeometrySolveError as exc:
        return None, seed, f"lens fit refused: {exc}"


def _warp_record(item, solution, frame_shape, chunk_reason):
    """The per-frame switch decision, gated on measurement rather than on the fit
    having converged: the lens+rotation model must BEAT the 2x3 on the SAME held-out
    stars, in the same outer radial third, by ``LENS_WARP_MAX_RESIDUAL_RATIO``.

    Upstream of the warp on purpose -- a gate read off the stacked image could only
    fire after the whole chunk had already been resampled and stacked.
    """
    similarity_outer = _similarity_holdout_outer_third(item.ref_xy, item.frame_xy, frame_shape)
    similarity_p95 = None if similarity_outer is None else similarity_outer["p95_px"]
    similarity_n = 0 if similarity_outer is None else similarity_outer["n"]
    lens_p95 = None
    if solution is None:
        reason = chunk_reason or (
            f"frame carried fewer than {MIN_MATCHES_PER_FRAME} matched stars for the joint fit"
        )
    elif solution.method != "rotation":
        reason = f"the joint fit described this frame with a {solution.method}, not a rotation"
    else:
        lens_outer = (solution.held_out_residuals_px or {}).get("outer") or {}
        lens_p95 = lens_outer.get("p95_px")
        if lens_p95 is None or lens_outer.get("n", 0) < MIN_LENS_HOLDOUT_OUTER_STARS:
            reason = (
                f"the fit's held-out outer third rests on {lens_outer.get('n', 0)} stars, "
                f"need >= {MIN_LENS_HOLDOUT_OUTER_STARS}"
            )
        elif similarity_p95 is None or similarity_n < MIN_LENS_HOLDOUT_OUTER_STARS:
            reason = "the 2x3's own held-out outer third is unmeasurable, so nothing to beat"
        elif lens_p95 > similarity_p95 * LENS_WARP_MAX_RESIDUAL_RATIO:
            reason = (
                f"held-out outer-third p95 {lens_p95:.2f} px does not beat the 2x3's "
                f"{similarity_p95:.2f} px by {LENS_WARP_MAX_RESIDUAL_RATIO}"
            )
        else:
            reason = None
    return {
        "model": "similarity" if reason is not None else "lens_rotation",
        "reason": reason,
        "held_out_outer_p95_px": {"lens": lens_p95, "similarity": similarity_p95},
    }


def _chunk_geometry_payload(solution, seed, chunk_reason, records) -> dict:
    """The chunk-level audit record: which warp this chunk's frames took, and why."""
    models = [record["model"] for record in records.values()]
    return {
        "seed_lens": _lens_payload(seed),
        "lens": None if solution is None else _lens_payload(solution.lens),
        "converged": solution is not None,
        "reason": chunk_reason,
        "frames_lens_warped": models.count("lens_rotation"),
        "frames_affine_warped": models.count("similarity"),
    }


def stack_run(
    prepared: PreparedFrames,
    out_path,
    *,
    horizon_frac=None,
    meteors: MeteorPolicy | None = None,
    gate_triage: bool = True,
    linear_artifact_path: Path | None = None,
    linear_only: bool = False,
) -> AstroReport:
    """``prepared`` (``prepare_frames``'s output) -> register -> stack -> develop ->
    export -> sidecar. A frame that fails registration is dropped into
    ``rejected_frames`` rather than killing the run, same posture as
    ``prepare_frames``'s decode failures; the run only raises
    ``AstroOrchestratorError`` when fewer than ``MIN_STACK_FRAMES`` survive. The
    sidecar write is never fatal (``write_stack_sidecar``'s own contract) --
    ``sidecar_path`` is ``None`` on that failure, the export still stands.

    ``horizon_frac`` is the user's own ``--horizon`` override; absent it, the sky
    boundary is placed from the reference frame's own detected star density -- but only
    when that frame yields at least ``MIN_AUTO_HORIZON_STARS`` detections, and
    ``AstroOrchestratorError`` naming ``--horizon FRACTION`` when it does not (see that
    constant for the measurement). Either way, that same sky mask gates every
    ``detect_stars`` call below -- ground lights/trees never seed or survive
    into the star set fed to registration.

    ``meteors`` (default off, see ``MeteorPolicy``) is the reinsertion posture; the
    base master always exports regardless -- reinsertion only ever adds to it.

    ``gate_triage`` (default ON -- triage verdicts used to gate nothing, so
    a cloudy/dewy frame poured straight into the mean): a cloud/dew-flagged or
    star-starved (``usable=False``) frame is excluded before registration is ever
    attempted, landing in ``rejected_frames`` with its verdict as the reason.
    ``prepared.reference_index`` is re-picked from the accepted frames (through the same
    ``pick_reference_index`` ``prepare_frames`` uses, against the full chunk length) when
    triage rejected it, so the reference is never a frame this same run excluded. Set
    ``gate_triage=False`` to deliberately stack every decoded frame regardless of
    verdict (diagnostics only). Triage's own thresholds and cohort-relative logic are
    unchanged here -- a majority-cloudy run still defines its own baseline; gating on
    that verdict is a separate, known limitation, not something this flag fixes.
    """
    if linear_only and linear_artifact_path is None:
        raise AstroOrchestratorError("stack_run: linear_only requires linear_artifact_path")
    if gate_triage:
        accepted_indices, triage_rejected, reason_counts = _gate_triage_verdicts(prepared)
        if len(accepted_indices) < MIN_STACK_FRAMES:
            raise AstroOrchestratorError(
                f"stack_run: triage gating left {len(accepted_indices)} of "
                f"{len(prepared.frames)} frames usable to stack (unusable="
                f"{reason_counts['unusable']}, cloud={reason_counts['cloud']}, "
                f"dew={reason_counts['dew']} removed; {MIN_STACK_FRAMES} required) -- "
                "pass gate_triage=False to stack every decoded frame regardless of verdict"
            )
        reference_index = (
            prepared.reference_index
            if prepared.reference_index in accepted_indices
            else pick_reference_index(
                prepared.metrics, sorted(accepted_indices), len(prepared.frames)
            )
        )
    else:
        accepted_indices = set(range(len(prepared.frames)))
        triage_rejected = []
        reference_index = prepared.reference_index

    ref_frame = prepared.frames[reference_index]
    luma_ref = _luma(ref_frame.rgb)

    # The mask must exist before ANY detect_stars call feeds registration. horizon_frac
    # needs no other input, so it's built first. Auto mode derives the boundary FROM
    # star density, so one unmasked detect_stars pass on the reference is unavoidable
    # to seed that density map -- accepted here as a deliberate one-extra-pass cost;
    # the reference is then re-detected under its own mask like every other frame.
    if horizon_frac is not None:
        mask = horizon_mask(luma_ref.shape, horizon_frac=horizon_frac)
    else:
        stars_ref_unmasked = detect_stars(luma_ref, sigma=STAR_DETECT_SIGMA)
        # Evidence gate, upstream of the estimate and of every irreversible artifact:
        # below MIN_AUTO_HORIZON_STARS detections the density map cannot separate "the
        # sky ends here" from "the only stars happen to be up here" (see that constant
        # for the 3600-frame measurement), and no statistic derived from that density
        # recovers the difference -- so refuse rather than place a boundary this run
        # would then stack, develop and export behind.
        if len(stars_ref_unmasked) < MIN_AUTO_HORIZON_STARS:
            raise AstroOrchestratorError(
                f"stack_run: automatic horizon needs >= {MIN_AUTO_HORIZON_STARS} detected "
                f"stars in the reference frame, got {len(stars_ref_unmasked)} -- at that "
                "count the star-density boundary cannot be told apart from a sky whose "
                "stars merely happen to sit high in the frame. Pass --horizon FRACTION to "
                "place the boundary yourself -- that clears THIS refusal, and says nothing "
                "about registration, which gates the chunk separately."
            )
        # A raw centroid scatter is an impulse, not a density -- most rows read exactly
        # zero, so the boundary would snap to the lowest single detected star's row.
        # Blur first so a row between stars still reads populated; a star-free band
        # stays 0 and reads as ground under horizon_mask's strict clearance test.
        impulses = np.zeros(luma_ref.shape, dtype=np.float32)
        rows = np.clip(stars_ref_unmasked[:, 1].round().astype(int), 0, luma_ref.shape[0] - 1)
        cols = np.clip(stars_ref_unmasked[:, 0].round().astype(int), 0, luma_ref.shape[1] - 1)
        impulses[rows, cols] = stars_ref_unmasked[:, 2]
        kernel_h = max(3, luma_ref.shape[0] // 20)
        kernel_w = max(3, luma_ref.shape[1] // 20)
        density = cv2.blur(impulses, (kernel_w, kernel_h))
        mask = horizon_mask(luma_ref.shape, star_density=density)
    # `mask` stays the feathered float blend weight `stack()` needs below;
    # `sky_stars_mask` is its boolean threshold, detect_stars's own contract.
    sky_stars_mask = mask >= 0.5
    stars_ref = detect_stars(luma_ref, sigma=STAR_DETECT_SIGMA, mask=sky_stars_mask)
    ref_pedestal = _sky_pedestal(luma_ref, sky_stars_mask)

    used_frames = [UsedFrame(prepared.metrics[reference_index], None)]
    photometry = [FramePhotometry(ref_frame.path, 0.0, 1.0, 1.0)]
    aligned_frames: list[np.ndarray] = [ref_frame.rgb]
    unaligned_frames: list[np.ndarray] = [ref_frame.rgb]
    # Coverage/contributor contract: the reference is never warped -- every one of
    # its pixels is a real sample, so its validity is unconditionally full coverage.
    # Every other frame's validity is the ACTUAL warp_validity of the transform
    # register() fit for it, so a frame that lands far off-reference measurably
    # contributes nothing where its warp vacated the frame (see stack()'s sky_validity).
    sky_validity: list[np.ndarray] = [np.ones(luma_ref.shape[:2], dtype=bool)]
    rejected: list[tuple[Path, str]] = [*prepared.failed, *triage_rejected]
    # Original frame index -> its ALIGNED array / registration transform, so a later
    # meteor can be developed from the same warp already computed above, and its
    # streak endpoints warped by the same transform, without redoing either.
    aligned_by_index: dict[int, np.ndarray] = {reference_index: ref_frame.rgb}
    transform_by_index: dict[int, np.ndarray | None] = {reference_index: None}

    pending: list[_PendingWarp] = []
    for i, frame in enumerate(prepared.frames):
        if i == reference_index or i not in accepted_indices:
            continue
        luma_frame = _luma(frame.rgb)
        stars_frame = detect_stars(luma_frame, sigma=STAR_DETECT_SIGMA, mask=sky_stars_mask)
        try:
            registration = register(
                luma_ref, stars_ref, luma_frame, stars_frame, sky_mask=sky_stars_mask
            )
        except RegistrationError as exc:
            rejected.append((frame.path, str(exc)))
            continue
        # photometry, pre-warp, on frame.rgb's own raw pixels (module comment above).
        try:
            offset, scale = _frame_photometry(luma_frame, ref_pedestal, sky_stars_mask)
        except ValueError as exc:
            rejected.append((frame.path, f"photometry: {exc}"))
            continue
        ref_xy, frame_xy = _matched_star_pairs(stars_ref, stars_frame, registration.transform)
        pending.append(_PendingWarp(i, frame, registration, offset, scale, ref_xy, frame_xy))

    # The chunk's shared lens is fitted across EVERY registered frame at once, so
    # it has to sit between registration and the warp -- no frame can be warped until
    # every frame's matched stars have been offered to the fit.
    geometry_solution, seed_lens, chunk_reason = _solve_chunk_geometry(pending, luma_ref.shape)
    frame_solutions = (
        {}
        if geometry_solution is None
        else {solution.frame_id: solution for solution in geometry_solution.frames}
    )
    warp_records: dict[str, dict] = {}
    for item in pending:
        frame, registration = item.frame, item.registration
        record = _warp_record(
            item, frame_solutions.get(str(frame.path)), luma_ref.shape, chunk_reason
        )
        warp_records[str(frame.path)] = record
        if record["model"] == "lens_rotation":
            warp_transform = build_remap(
                luma_ref.shape,
                geometry_solution.lens,
                frame_solutions[str(frame.path)].rotation,
            )
        else:
            warp_transform = registration.transform
        aligned = warp_to_reference(frame.rgb, warp_transform, luma_ref.shape)
        corrected = ((aligned.astype(np.float64) - item.offset) * item.scale).astype(np.float32)
        aligned_frames.append(corrected)
        unaligned_frames.append(frame.rgb)
        used_frames.append(UsedFrame(prepared.metrics[item.index], registration))
        photometry.append(FramePhotometry(frame.path, item.offset, item.scale, 1.0))
        aligned_by_index[item.index] = corrected
        # Stays the 2x3 whichever warp the image took: its consumer is the meteor
        # step, which transforms streak ENDPOINT coordinates, not pixels.
        transform_by_index[item.index] = registration.transform
        # Validity describes the warp that actually ran, remap or affine.
        sky_validity.append(warp_validity(warp_transform, frame.rgb.shape, luma_ref.shape))
    geometry_payload = _chunk_geometry_payload(
        geometry_solution, seed_lens, chunk_reason, warp_records
    )

    if len(aligned_frames) < MIN_STACK_FRAMES:
        raise AstroOrchestratorError(
            f"stack_run: only {len(aligned_frames)} of {len(prepared.frames)} frames "
            f"registered against the reference ({MIN_STACK_FRAMES} required to stack)"
        )

    stack_result = stack(aligned_frames, unaligned_frames, mask, sky_validity=sky_validity)
    # Last use: stack() has already consumed every array. Clearing in place (not just
    # dropping the name) lets each frame-sized buffer be collected as soon as
    # aligned_by_index's remaining references to the same arrays are cleared below --
    # otherwise peak RSS ratchets upward across consecutive chunks in one long-lived
    # process (preflight.py sizes each chunk against CURRENT free memory).
    aligned_frames.clear()
    unaligned_frames.clear()
    sky_validity.clear()
    written_linear_path = None
    if linear_artifact_path is not None:
        boundary_row = float(np.median(np.sum(mask >= 0.5, axis=0)))
        registration_provenance = []
        for used in used_frames:
            registration = used.registration
            metrics = used.metrics
            exif = extract_exif_capture_metadata(metrics.path)
            registration_provenance.append(
                {
                    "frame": str(metrics.path.resolve()),
                    "transform": "identity"
                    if registration is None
                    else registration.transform.tolist(),
                    "n_matched": 0 if registration is None else registration.n_matched,
                    "inlier_fraction": 1.0
                    if registration is None
                    else registration.inlier_fraction,
                    "residual_median_px": 0.0
                    if registration is None
                    else registration.residual_median_px,
                    "residual_p95_px": 0.0
                    if registration is None
                    else registration.residual_p95_px,
                    # In-sample: RANSAC-inlier residuals under the fitted transform,
                    # per radial sensor third -- not held-out geometry evidence.
                    "radial_inlier_residuals_px_in_sample": None
                    if registration is None
                    else registration.radial_inlier_residuals_px,
                    # Per-frame triage quality (astro_triage.measure_frame), reusing
                    # astro_sidecar's own field shape rather than re-serialising it.
                    "triage": {
                        "star_count": int(metrics.star_count),
                        "background_median": float(metrics.background_median),
                        "fwhm_px": float(metrics.fwhm_px),
                        "shape": _star_shape_payload(metrics.shape),
                    },
                    # Which warp this frame's pixels actually went through, and the
                    # held-out numbers that decided it. The reference is never warped.
                    "warp": warp_records.get(str(metrics.path), {"model": "reference"}),
                    "exposure_s": exif.exposure_s,
                    "captured_at": exif.capture_utc.isoformat() if exif.capture_utc else None,
                }
            )
        identity = read_capture_identity(ref_frame.path)
        write_linear_artifact(
            linear_artifact_path,
            stack_result,
            reference_raw=str(ref_frame.path.resolve()),
            capture_identity={
                "camera": identity.camera,
                "raw_shape": list(identity.raw_shape),
                "cfa_pattern": list(identity.cfa_pattern)
                if identity.cfa_pattern is not None
                else None,
                "iso": identity.iso,
                "exposure_s": identity.exposure_s,
                "temperature_c": identity.temperature_c,
            },
            mask_provenance={
                "mode": "explicit" if horizon_frac is not None else "automatic",
                "requested_horizon_frac": horizon_frac,
                "resolved_boundary_row": boundary_row,
                "resolved_boundary_frac": boundary_row / mask.shape[0],
            },
            source_frames=[str(frame.path.resolve()) for frame in prepared.frames],
            registration_provenance=registration_provenance,
            photometry_provenance=[
                {
                    "frame": str(item.path.resolve()),
                    "offset": item.offset,
                    "scale": item.scale,
                    "weight": item.weight,
                }
                for item in photometry
            ],
            algorithm_version=linear_algorithm_hash(),
        )
        written_linear_path = Path(linear_artifact_path)
    if linear_only:
        # No meteor step runs on this path (it returns before develop), so these two
        # per-frame buffers have no further consumer -- release them here too.
        aligned_by_index.clear()
        transform_by_index.clear()
        return AstroReport(
            image_path=None,
            sidecar_path=None,
            stack=stack_result,
            develop=None,
            rejected_frames=tuple(rejected),
            meteors_reinserted=0,
            photometry=tuple(photometry),
            linear_artifact_path=written_linear_path,
            geometry=geometry_payload,
        )
    # Same boolean mask detect_stars/_sky_pedestal/_frame_photometry already use:
    # ground and horizon glow never reach the gradient fit, black point or stretch targets.
    develop_result = develop_stack(stack_result.rgb, sky_mask=sky_stars_mask)
    policy = meteors if meteors is not None else MeteorPolicy()
    # Frames prepared with detect_meteors=False carry streaks=None. Asking a policy that
    # would reinsert to act on frames that never searched is a silent false negative, so
    # classify_streaks raises on them; an "off" policy consumes nothing and skips instead.
    if any(frame.streaks is None for frame in prepared.metrics):
        if policy.mode != "off":
            raise MeteorPolicyError(
                f"meteor policy mode={policy.mode!r} needs streak data, but these frames "
                "were prepared with detect_meteors=False -- re-run prepare_frames with "
                "detect_meteors=True"
            )
        streaks = ()
    else:
        streaks = classify_streaks(prepared.metrics)
    selected, candidate_count, skip_reason = _resolve_meteor_selection(
        streaks, policy, aligned_by_index
    )
    reinserted_rgb = develop_result.rgb_uint16
    reinserted_count = 0
    if selected:
        reinserted_rgb = reinsert(
            develop_result.rgb_uint16,
            _stream_meteor_layers(selected, aligned_by_index, transform_by_index, develop_result),
            feather_px=STREAK_FEATHER_PX,
        )
        reinserted_count = len(selected)
    # Last use either way: reinsert() (when selected) or _resolve_meteor_selection's
    # membership check (when not) already consumed these -- release before develop's
    # own working buffers are allocated below.
    aligned_by_index.clear()
    transform_by_index.clear()
    # The master a person looks at, after any meteor layer is composited onto it so
    # the layer is never speckled against a sky that is not. develop_result.rgb_uint16
    # stays as the stretch produced it -- astro_contact.recover_balanced needs an
    # invertible artifact, and a blur is not invertible.
    chroma_before = chroma_sigma(reinserted_rgb)
    reinserted_rgb = denoise_chroma(reinserted_rgb)
    chroma_after = chroma_sigma(reinserted_rgb)

    from . import develop as _develop  # lazy: no develop<->astro import cycle
    from .develop import DEVELOP_ALGO_VERSION

    image_path = _develop.export_tiff16(reinserted_rgb, out_path)
    # A .tif at the archival 0.12 background reads as a black rectangle in a file
    # browser, so every run also writes the JPEG a person can actually open. It is
    # unconditional rather than a flag: the user always gets a picture, and a
    # hand-rolled render stops being the only way to see the night's work.
    preview_path: Path | None = Path(out_path).with_suffix(".preview.jpg")
    try:
        _develop.export_jpeg(preview_render(reinserted_rgb), preview_path)
    except Exception as exc:  # noqa: BLE001 - a cosmetic render never costs the master
        # The .tif is what hours of stacking bought; the preview is a convenience.
        # A full disk or a codec fault loses the convenience and nothing else.
        logger.warning("preview render failed for %s: %s", image_path, exc)
        preview_path = None

    decoder = f"{ref_frame.decoder}+astrostack{stack_result.n_frames}"
    payload = stack_sidecar_payload(
        used_frames=used_frames,
        rejected_frames=rejected,
        develop=develop_result,
        stack=stack_result,
        decoder=decoder,
        algo_version=DEVELOP_ALGO_VERSION,
        streaks=streaks,
        meteor_mode=policy.mode,
        meteor_candidate_count=candidate_count,
        meteor_reinserted_count=reinserted_count,
        meteor_skip_reason=skip_reason,
        # The sidecar is the audit record for how this master was built: a run that
        # subtracted a validated master dark must not record ``inputs.dark: []``, which
        # is what this callsite claimed while the artifact's own name hash encoded the
        # darks. ``prepared.dark_paths`` is the run's own, never re-derived here.
        dark_paths=prepared.dark_paths,
    )
    sidecar_path = write_stack_sidecar(payload, image_path)

    return AstroReport(
        image_path=image_path,
        sidecar_path=sidecar_path,
        stack=stack_result,
        develop=develop_result,
        rejected_frames=tuple(rejected),
        meteors_reinserted=reinserted_count,
        photometry=tuple(photometry),
        linear_artifact_path=written_linear_path,
        geometry=geometry_payload,
        chroma_sigma_before=chroma_before,
        chroma_sigma_after=chroma_after,
        preview_path=preview_path,
    )
