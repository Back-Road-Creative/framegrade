"""Privacy-blur hard gate.

Guarantees a blur-enabled run never ships an unblurred master. The stage-9
encode composites faces/plates iff a blur mask is present
(``single_pass._run_encode_step``), so "mask present at encode time" is
equivalent to "FINAL is blurred". Two failure modes this closes:

  * **Resume drops the mask.** ``BlurDetectionStage`` is skipped on resume once
    its stage is complete, so ``ctx.blur_mask_path`` comes back ``None`` and the
    encode would silently fall through to the no-blur ``-vf`` path.
    ``resolve_mask`` repopulates it from disk; ``check_mask_ready`` hard-fails if
    it is genuinely absent.
  * **A pre-blur FINAL is accepted.** A FINAL encoded before blur was enabled
    (no re-encode on resume) would ship unblurred. ``mark_applied`` stamps a
    durable marker beside every blurred FINAL; ``check_applied`` refuses a
    resume that accepts a FINAL without it, and — because a marker only ever
    attested that some encode *claimed* to composite — also refuses one whose
    pixels contradict it (``masked_regions_look_blurred``).

Escape hatch for both: ``config.drive.skip_blur`` — settable via the ``drive``
command's ``--skip-blur`` flag, or ``drive.skip_blur: true`` in config.yaml for
commands (like ``pipeline``) that register no such flag.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from .cache import file_fingerprint, load_cached_json, save_cache_json

logger = logging.getLogger(__name__)

_MARKER_SUFFIX = ".blurred"

#: Frames sampled from the FINAL for the pixel spot-check. Enough that one
#: naturally-flat background does not decide the answer, few enough that the
#: check costs a second on a multi-hour master.
_SPOT_CHECK_SAMPLES = 3
#: Both videos are decoded to this width before comparing. Blur is a
#: low-frequency property; full resolution buys nothing and costs seconds.
_SPOT_CHECK_WIDTH = 480
_SPOT_CHECK_TIMEOUT_S = 60
#: A mask covering less of the frame than this has nothing worth measuring.
_MIN_MASK_COVERAGE = 0.004
#: The masked region must carry less than this share of the surrounding
#: region's high-frequency detail. Deliberately loose: the composite blurs at a
#: radius that collapses detail by an order of magnitude, so a real blur clears
#: this easily, while an UNBLURRED region sits at ~1.0. The gap is what makes a
#: one-sided threshold safe.
_BLUR_DETAIL_RATIO = 0.5


class BlurGateError(RuntimeError):
    """A blur-enabled run cannot prove its master is blurred.

    Raised on the drive path (``day_master_encoder``) when blur is required but the
    day master can't be shown to carry the privacy composite — a resume reusing a
    cached master that predates blur (no ``.blurred`` marker), or a detection failure
    that would otherwise fall through to an unblurred encode. It is a pipeline-control
    hard-fail (like ``OstAuditPendingError``): ``run_all_days`` re-raises it instead of
    swallowing it into a per-day skip, so an operator never silently ships an
    unblurred long-form/short. Escape: ``--skip-blur`` / re-run ``--fresh``.
    """


def _overlaps(a: list, b: list) -> bool:
    """True when two boxes share any pixel — the cheapest 'same object' test."""
    ax, ay, aw, ah = a[0], a[1], a[2], a[3]
    bx, by, bw, bh = b[0], b[1], b[2], b[3]
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def backfill_track_births(frame_boxes: dict, *, pre_roll_frames: int) -> dict:
    """Extend a newly-appeared box backward over the frames before it.

    ``detect_and_track`` scores an object several frames after it becomes
    visible, so the frames just before a track's FIRST detection are the ones
    that ship unmasked. Typical case: one frame holds no box at all, the next
    frame covers an oncoming cabin, and nothing had been detected there for a
    long run of earlier frames.

    ``blur_pre_roll_frames`` is the intended mechanism but is a
    ``detect_and_track`` argument, so tuning it costs a full re-detection pass
    on a long day, and at its default it can emit no entry for the frame
    before the first detection. This runs on the already-computed detections
    instead.

    A box is a birth when it overlaps NO box in the previous populated frame.
    It is held rather than shrunk going back: an approaching vehicle is
    smaller earlier, so the later, larger box over-covers instead of
    under-covering. Depth stays bounded by ``pre_roll_frames`` — a fraction of
    a second — which is what keeps a newly-detected roadside object from
    blurring open road.

    Returns a new dict keyed by int frame index; the input is not mutated.
    """
    normalized = {int(k): v for k, v in frame_boxes.items()}
    if pre_roll_frames <= 0:
        return normalized

    populated = sorted(f for f, boxes in normalized.items() if boxes)
    if not populated:
        return normalized

    filled = {f: [list(b) for b in boxes] for f, boxes in normalized.items()}
    for idx, frame in enumerate(populated):
        prev = normalized[populated[idx - 1]] if idx else []
        births = [b for b in normalized[frame] if not any(_overlaps(b, p) for p in prev)]
        if not births:
            continue
        for back in range(1, pre_roll_frames + 1):
            target = frame - back
            if target < 0:
                break
            entry = filled.setdefault(target, [])
            for b in births:
                if list(b) not in entry:
                    entry.append(list(b))
    return filled


def fill_detection_gaps(frame_boxes: dict, *, max_gap_frames: int) -> dict:
    """Give every frame between two detected frames a mask.

    ``detect_and_track`` emits entries only for frames it sampled or
    flow-tracked. At ``blur_sample_fps=8`` against 59.94 fps video that leaves
    ~7 frames between samples with NO entry, and the renderer reads a missing
    entry as "nothing to blur". Wherever an object first appears between two
    samples, its leading frames ship unmasked.

    This is NOT the case ``backfill_track_births`` covers: a first detection
    that follows a long void, far past ``max_gap_frames``. The guard below
    correctly refuses to fill such a void — interpolating there would blur
    seconds of open road. ``backfill_track_births`` closes that case; this
    one closes the ordinary inter-sample gap, where the bracketing detections
    really do describe the same objects.

    Each gap frame inherits the UNION of its bracketing entries: a gap frame
    may hold an object that is appearing OR one that is leaving, and nothing
    in the data distinguishes them, so cover both. Gaps longer than
    ``max_gap_frames`` are left alone — past the sample interval the
    bracketing boxes describe unrelated objects, and filling between them
    would blur bare road.

    Returns a new dict keyed by int frame index; the input is not mutated.
    """
    normalized: dict = {}
    for k, v in frame_boxes.items():
        normalized[int(k)] = v

    detected = sorted(f for f, boxes in normalized.items() if boxes)
    if len(detected) < 2:
        return normalized

    filled = dict(normalized)
    for lo, hi in zip(detected, detected[1:]):
        if hi - lo <= 1 or hi - lo > max_gap_frames:
            continue
        union: list = []
        for box in list(normalized[lo]) + list(normalized[hi]):
            if box not in union:
                union.append(box)
        for f in range(lo + 1, hi):
            filled[f] = [list(b) for b in union]
    return filled


def blur_required(config) -> bool:
    """True when this run must produce a blurred master."""
    drive = config.drive
    return bool(getattr(drive, "blur_enabled", False)) and not bool(
        getattr(drive, "skip_blur", False)
    )


def _marker_path(final_output: Path) -> Path:
    return Path(final_output).with_name(Path(final_output).name + _MARKER_SUFFIX)


def resolve_mask(ctx) -> Optional[Path]:
    """Return the blur mask, repopulating ``ctx.blur_mask_path`` from the known
    stage path on resume (the detection stage doesn't re-run to set it)."""
    if ctx.blur_mask_path is not None:
        return ctx.blur_mask_path
    expected = ctx.paths.stages_dir / f"{ctx.project_name}_blur_mask.mkv"
    if expected.exists():
        ctx.blur_mask_path = expected
    return ctx.blur_mask_path


def check_mask_ready(ctx) -> Optional[str]:
    """Pre-encode gate: error string if blur is required but no usable mask is
    present, else ``None``. Also repopulates the mask path for the encode."""
    if not blur_required(ctx.config):
        return None
    mask = resolve_mask(ctx)
    if mask is None or not Path(mask).exists() or Path(mask).stat().st_size == 0:
        return (
            "BLUR GATE: blur_enabled but no privacy mask is present for the master "
            "encode — faces/plates would ship unblurred. Re-run from blur_detection "
            "(--fresh), or set drive.skip_blur: true in config.yaml (this command "
            "has no --skip-blur flag; that flag exists only on `gomoveshift drive`)."
        )
    return None


def cabin_leak_report_path(mask: Path) -> Path:
    """Where the occupant-coverage gate persists its FULL leak list — beside the
    mask it judged, so the two travel together across a resume."""
    return mask.with_name(f"{mask.stem}_cabin_leaks.json")


def _write_cabin_leak_report(path: Path, rep) -> None:
    """Persist every leak. The error string can only name a handful, and
    re-deriving the rest means re-scanning a multi-hour master — so the report the
    gate already computed is written down rather than thrown away."""
    payload = {
        "vehicle_frames": rep.vehicle_frames,
        "leaks": [
            {"frame": leak.frame, "coverage": round(leak.coverage, 4), "box": list(leak.box)}
            for leak in sorted(rep.leaks, key=lambda v: v.frame)
        ],
    }
    try:
        path.write_text(json.dumps(payload, indent=1))
    except OSError as exc:  # a report is diagnostics; never mask the leak itself
        logger.warning("could not write cabin leak report %s: %s", path, exc)


def cabin_verdict_path(mask: Path) -> Path:
    """Where the cabin-coverage PASS/FAIL verdict is cached, beside the mask it
    judged, so a relaunch that reuses the identical mask+source skips the
    multi-hour re-verify (``_build_day_blur_mask`` caches the mask itself the
    same way; this caches the gate that runs right after it)."""
    return mask.with_name(f"{mask.stem}_cabin_verdict.json")


def _safe_fingerprint(path: Optional[Path]) -> Optional[str]:
    """``file_fingerprint`` needs a real, readable file. Any input change (or a
    path that doesn't exist/isn't stat-able) must miss the cache rather than
    crash the gate, so a missing/unreadable input fingerprints as ``None`` —
    which never equals a previously-cached real fingerprint."""
    if path is None:
        return None
    try:
        return file_fingerprint(Path(path))
    except OSError:
        return None


def _cabin_coverage_signature(drive, video: Path, mask: Path, veh: Path, plate) -> dict:
    """Every input the coverage check reads: the video/mask/model bytes it
    verifies against, plus every config knob that changes what a PASS means."""
    return {
        "video_fp": _safe_fingerprint(video),
        "mask_fp": _safe_fingerprint(mask),
        "vehicle_model_fp": _safe_fingerprint(veh),
        "plate_model_fp": _safe_fingerprint(plate),
        "cabin_frac": drive.blur_vehicle_cabin_frac,
        "conf": drive.blur_vehicle_conf,
        "sample_fps": getattr(drive, "blur_cabin_gate_sample_fps", 4.0),
        "min_coverage": getattr(drive, "blur_cabin_gate_min_coverage", 0.5),
        "min_height_frac": drive.blur_vehicle_min_height_frac,
        "ignore_bottom_frac": drive.blur_vehicle_ignore_bottom_frac,
        "plate_max_area_frac": drive.blur_plate_max_area_frac,
    }


def _cabin_coverage_error(config, video: Path, mask: Optional[Path]) -> Optional[str]:
    """Shared core for ``check_cabin_coverage`` (legacy per-episode path) and
    ``check_day_master_cabin_coverage`` (drive day-master path): error string if
    any plate-corroborated vehicle's cabin region in ``video`` is left uncovered
    by ``mask`` (a real car's driver would ship exposed), else ``None``. No-op
    when blur isn't required, the occupant cabin blur is off, or the mask/
    vehicle-model is absent (each caller's own mask-readiness gate and the
    detection stage own those failures). Verifies against the MASK, before the
    irreversible encode."""
    if not blur_required(config):
        return None
    drive = config.drive
    if getattr(drive, "blur_vehicle_cabin_frac", 0.0) <= 0:
        return None
    if mask is None:
        return None
    from ..analysis import resolve_plate_model, resolve_vehicle_model, verify_cabin_coverage

    veh = resolve_vehicle_model(drive)
    if veh is None:
        return None  # blur_detection hard-fails the missing-model case

    plate = resolve_plate_model(drive)
    signature = _cabin_coverage_signature(drive, video, mask, veh, plate)
    verdict_path = cabin_verdict_path(mask)
    cached = load_cached_json(verdict_path)
    if cached is not None and cached.get("signature") == signature:
        return cached.get("error")

    try:
        rep = verify_cabin_coverage(
            video,
            mask,
            veh,
            plate_model=plate,
            cabin_frac=drive.blur_vehicle_cabin_frac,
            conf=drive.blur_vehicle_conf,
            sample_fps=getattr(drive, "blur_cabin_gate_sample_fps", 4.0),
            min_coverage=getattr(drive, "blur_cabin_gate_min_coverage", 0.5),
            min_height_frac=drive.blur_vehicle_min_height_frac,
            ignore_bottom_frac=drive.blur_vehicle_ignore_bottom_frac,
            plate_max_area_frac=drive.blur_plate_max_area_frac,
        )
    except Exception as exc:  # never let the gate itself crash the run silently
        return f"BLUR GATE (cabin): coverage check could not run: {exc}"
    report = cabin_leak_report_path(mask)
    if rep.ok:
        report.unlink(missing_ok=True)  # never let a passing run inherit a stale leak list
        save_cache_json(verdict_path, {"signature": signature, "error": None})
        return None
    _write_cabin_leak_report(report, rep)
    worst = sorted(rep.leaks, key=lambda v: v.coverage)[:5]
    frames = ", ".join(f"f{leak.frame}({leak.coverage:.0%})" for leak in worst)
    error = (
        f"BLUR GATE (cabin): {len(rep.leaks)} of {rep.vehicle_frames} corroborated "
        f"vehicle-frame(s) have an EXPOSED occupant cabin the mask misses — e.g. {frames} "
        f"(all {len(rep.leaks)} in {report.name}). "
        "A real car's driver would ship unblurred. Re-run blur_detection (--fresh), "
        "or disable via drive.skip_blur (the `drive` command's --skip-blur flag; "
        "other commands set drive.skip_blur: true in config.yaml)."
    )
    save_cache_json(verdict_path, {"signature": signature, "error": error})
    return error


def check_cabin_coverage(ctx) -> Optional[str]:
    """Pre-encode gate (legacy per-episode path): thin ctx-adapter over
    ``_cabin_coverage_error`` — see its docstring for the actual semantics,
    shared with the drive day-master path's ``check_day_master_cabin_coverage``."""
    return _cabin_coverage_error(ctx.config, ctx.current_video, resolve_mask(ctx))


def check_day_master_cabin_coverage(config, video: Path, mask: Optional[Path]) -> Optional[str]:
    """Pre-encode gate (drive day-master path): same occupant-coverage check as
    ``check_cabin_coverage``, for a caller that builds its mask directly
    (``day_master_encoder._build_day_blur_mask``) rather than through a stage
    ctx. Call this AFTER the day mask is built and BEFORE the master encode
    composites it — a leak must abort the run pre-encode, never post-hoc."""
    return _cabin_coverage_error(config, video, mask)


def mark_applied(final_output: Path) -> None:
    """Stamp a durable marker proving THIS FINAL was encoded with blur."""
    _marker_path(final_output).write_text("blurred\n")


def master_blur_proven(config, master_path: Path) -> bool:
    """True when a blur-required run's ``master_path`` carries the ``.blurred`` marker
    (or blur isn't required). The marker is stamped by ``mark_applied`` only after the
    privacy composite lands, so its presence proves the master — and every LF/short/
    thumbnail cut from it — is blurred."""
    if not blur_required(config):
        return True
    return _marker_path(master_path).exists()


def require_master_blurred(config, master_path: Path) -> None:
    """Drive-path hard gate: raise ``BlurGateError`` when a blur-enabled run is about to
    REUSE a cached day master that isn't provably blurred (no ``.blurred`` marker — it
    predates blur). Fires upstream of the LF/short cuts (and so upstream of any upload),
    never a silent unblurred ship. No-op when blur isn't required (``--skip-blur``)."""
    if master_blur_proven(config, master_path):
        return
    raise BlurGateError(
        f"BLUR GATE: blur_enabled but the cached day master {Path(master_path).name} "
        "carries no .blurred marker — it was encoded before blur was on and its LFs/"
        "shorts/thumbnails would ship unblurred. Re-encode with --fresh, or bypass with "
        "--skip-blur."
    )


def _gray_frames(video: Path, timestamps, width: int, *, timeout: float = _SPOT_CHECK_TIMEOUT_S):
    """One gray frame per timestamp, scaled to ``width``, or ``[]`` if any read fails.

    Shared by every pixel proof in the pipeline — this gate's blur spot-check and
    the shorts post-burn OST check — so both read frames by the same rules. Each
    caller passes its own ``width``: blur is a low-frequency property that survives
    a downscale, while counting glyph strokes does not.

    ``float32`` because callers difference these arrays: in ``uint8`` a
    dark-minus-bright neighbour wraps to a large positive and reads as exactly the
    high-frequency detail a blur is supposed to have removed. Height comes from the
    first frame and a later short read returns ``[]`` rather than a frame reshaped
    to geometry the first one did not have — every caller treats an empty list as
    unmeasurable and passes, which is the safe direction for a one-sided gate.
    """
    import numpy as np

    height = None
    frames = []
    for t in timestamps:
        try:
            out = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
                 "-frames:v", "1", "-vf", f"scale={width}:-2,format=gray",
                 "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                capture_output=True, timeout=timeout,
            )  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            return []
        if out.returncode != 0 or not out.stdout:
            return []
        if height is None:
            height = len(out.stdout) // width
        if height <= 0 or len(out.stdout) < width * height:
            return []
        buf = np.frombuffer(out.stdout[: width * height], dtype=np.uint8)
        frames.append(buf.reshape(height, width).astype(np.float32))
    return frames


def _detail(gray, selector) -> float:
    """Mean absolute Laplacian over the selected pixels — high-frequency energy,
    which is exactly what a blur removes."""
    import numpy as np

    lap = np.abs(np.diff(gray, axis=0)[:, :-1]) + np.abs(np.diff(gray, axis=1)[:-1, :])
    sel = selector[:-1, :-1]
    return float(lap[sel].mean()) if sel.any() else 0.0


def masked_regions_look_blurred(final_output, mask_path) -> tuple[bool, str]:
    """Does the FINAL's masked region actually carry less detail than the rest?

    Samples a few frames from the FINAL and the frame-aligned mask, and compares
    high-frequency energy inside the mask against the same frame outside it. One
    frame that is clearly blurred is proof the composite ran; the check fails
    only when EVERY measurable frame shows the masked region as sharp as its
    surroundings.

    Returns ``(True, reason)`` when the pixels corroborate the blur AND when the
    question is unanswerable — no mask, unreadable video, a mask covering
    nothing. Absence of measurement is not evidence of an unblurred ship, and a
    privacy gate that hard-fails a six-hour encode over a measurement it could
    not take is worse than the hole it closes. Callers keep the ``.blurred``
    marker as the necessary condition; this only ever adds corroboration.
    """

    if mask_path is None or not Path(mask_path).exists():
        return True, "no mask available to compare against"
    fractions = [(i + 0.5) / _SPOT_CHECK_SAMPLES for i in range(_SPOT_CHECK_SAMPLES)]
    duration = _probe_duration(mask_path)
    if duration <= 0:
        return True, "mask duration unreadable"
    times = [f * duration for f in fractions]

    mask_frames = _gray_frames(Path(mask_path), times, _SPOT_CHECK_WIDTH)
    final_frames = _gray_frames(Path(final_output), times, _SPOT_CHECK_WIDTH)
    if not mask_frames or len(final_frames) != len(mask_frames):
        return True, "frames unreadable"

    ratios = []
    for m, f in zip(mask_frames, final_frames):
        if m.shape != f.shape:
            return True, "mask and FINAL disagree on geometry"
        inside = m > 127
        if inside.mean() < _MIN_MASK_COVERAGE or (~inside).mean() < _MIN_MASK_COVERAGE:
            continue
        outside_detail = _detail(f, ~inside)
        if outside_detail <= 0:
            continue
        ratio = _detail(f, inside) / outside_detail
        ratios.append(ratio)
        if ratio < _BLUR_DETAIL_RATIO:
            return True, f"masked region carries {ratio:.0%} of the surrounding detail"
    if not ratios:
        return True, "no sampled frame carried enough mask coverage to measure"
    return False, (
        f"masked regions are as sharp as the rest of the frame "
        f"(detail ratio {min(ratios):.0%}, expected under {_BLUR_DETAIL_RATIO:.0%}) "
        f"across {len(ratios)} sampled frame(s)"
    )


def _probe_duration(video: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(video)],
            capture_output=True, text=True, timeout=_SPOT_CHECK_TIMEOUT_S,
        )  # fmt: skip
        return float((out.stdout or "0").strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def check_applied(ctx, final_output: Path) -> Optional[str]:
    """Resume gate: error string if a pre-existing FINAL cannot be shown to be
    blurred, else ``None``.

    Two conditions, cheapest first. The ``.blurred`` marker must be there — it
    is what a pre-blur FINAL lacks. Then the PIXELS must corroborate it: a
    marker is an attestation that an encode once claimed to composite the mask,
    and a stale, copied, or hand-created one attests just as loudly, so on its
    own it never proved anything about the artefact being accepted.
    ``masked_regions_look_blurred`` is one-sided — it fails only on a clear
    negative and passes anything it cannot measure.
    """
    if not blur_required(ctx.config):
        return None
    if not _marker_path(final_output).exists():
        return (
            "BLUR GATE: blur_enabled but the existing FINAL.mp4 carries no blur "
            "marker — it was encoded before blur was on and would ship unblurred. "
            "Re-encode with --fresh, or set drive.skip_blur: true in config.yaml "
            "(this command has no --skip-blur flag; that flag exists only on "
            "`gomoveshift drive`)."
        )
    blurred, why = masked_regions_look_blurred(final_output, resolve_mask(ctx))
    if not blurred:
        return (
            f"BLUR GATE: the existing FINAL.mp4 carries a blur marker but its pixels "
            f"contradict it — {why}. The marker is stale or was copied from another "
            "encode; faces/plates would ship unblurred. Re-encode with --fresh, or "
            "set drive.skip_blur: true in config.yaml (this command has no "
            "--skip-blur flag; that flag exists only on `gomoveshift drive`)."
        )
    logger.debug("blur pixel spot-check on %s: %s", Path(final_output).name, why)
    return None
