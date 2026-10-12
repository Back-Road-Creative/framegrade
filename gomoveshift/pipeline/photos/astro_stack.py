"""Sky/ground stacking for astro composites.

``stack_sky`` takes frames ``astro_register`` already warped onto one reference — stars
sit on the SAME pixel across frames, so a per-pixel MEAN buys the full sqrt(N) noise win,
and a per-n-calibrated leave-one-out rejector drops a streak (present at one pixel
in only a frame or two) as an outlier rather than averaging it in; a real meteor is
reinserted later by ``astro_meteors`` from its own calibrated frame. ``stack_ground`` takes
the RAW, unaligned frames: the foreground is fixed in sensor coordinates and converges
under a plain per-pixel MEDIAN, while unaligned stars land at a different pixel every
frame and the median discards the resulting minority for free. ``horizon_mask``/``blend``
stitch the two results at the ridgeline.

Both stack functions process ``BAND_ROWS``-row bands rather than the whole frame at once —
the caller's frame LIST is already the memory budget, and since the per-pixel
math has no cross-row dependency, banding changes nothing about the answer, only the peak
extra memory (~one band x N frames) used to get there. Constants are provisional pending
the reference-card tuning pass, ``bracket.py``'s measured-against-a-real-card
discipline.
"""

from dataclasses import dataclass

import numpy as np

from gomoveshift.pipeline.photos.astro_quality import CoverageStats, coverage_stats
from gomoveshift.pipeline.photos.astro_register import contributor_count

# PER-PIXEL kappa from valid count (``_kappa_for_count``), calibrated ONE-SIDED: the <=1%
# false rejection holds for bright-only; ``bright_only=False`` roughly doubles it.
SHORT_STACK_KAPPA_BY_N: dict[int, float] = {5: 8.8, 8: 4.8, 20: 3.3}
# Below this: no kappa holds <= 1% -- plain validity-masked mean.
SHORT_STACK_MIN_REJECT_FRAMES = 5
# Floor on the leave-one-out std (division-by-zero, not reliability).
SIGMA_CLIP_STD_EPS = 1e-6
# A CHANNEL is degenerate (no z evidence) when its MAD~0 vs |median|: a censored majority.
DEGENERACY_REL_EPS = 8.0 * np.finfo(np.float32).eps
# Row-band height: a band's temporaries stay a small multiple of one frame's size.
BAND_ROWS = 128
# Below 3 frames "reject the odd one out" is undefined -- 2 frames can only disagree.
MIN_STACK_FRAMES = 3
# Minimum per-PIXEL contributor count (coverage contract) below which a pixel has
# no meaningful stacking gain, however clean its individual samples are -- the edge-wedge
# case: real inter-frame drift trims a different border off each warped frame, so the
# outer band is backed by only a handful of frames while the centre is backed by all of
# them. JUDGEMENT CALL, not a measured SNR threshold like SHORT_STACK_KAPPA_BY_N: reuses
# the same ">= 3 samples" size call MIN_STACK_FRAMES already makes for a whole run,
# applied per pixel instead -- below it, "averaged" isn't meaningfully different from
# "one raw sample" for this pipeline's purposes.
MIN_STACK_CONTRIBUTORS = MIN_STACK_FRAMES
# horizon_mask's soft transition width, in rows -- a hard 0/1 edge would seam any
# ridgeline that isn't perfectly flat (it never is).
FEATHER_ROWS = 24.0
# star_density mode: fraction of the row profile's OWN upper-quartile value used as the
# "still reads as sky" threshold -- frame-relative, so one fraction fits both a dense
# Milky Way frame and a sparse light-polluted one.
STAR_DENSITY_THRESHOLD_FRAC = 0.25
# Evidence floor for the AUTOMATIC (star_density) boundary: below this many DETECTED
# stars in the reference frame the density map cannot tell "the sky ends here" from "the
# only stars happen to be up here", so ``astro_orchestrator.stack_run`` refuses instead
# of estimating. Measured on 3600 synthetic 200x200 fields driven through the shipped
# path (detect_stars -> blurred impulse density -> horizon_mask), vertical extinction
# falloff 0.5, ridge at 0.65 and 0.80 of frame height, 120 seeds per cell, boundary
# error as % of frame height: at 40-44 detections it is median 1.5 / p90 6.5 / p99 10.0
# with 0.8% of frames off by more than 10% of frame height, against 2.3% gross at 35-39,
# 5.5% at 30-34, 8.5% at 20-24, 16.0% at 15-19, 24.8% at 10-14 and 44.5% below 10 -- 40
# is the smallest bin edge at which that gross-error rate drops under 1%.
MIN_AUTO_HORIZON_STARS = 40


class StackInputError(ValueError):
    """Too few frames, or frames of mixed shape, handed to a stack function."""


def _validated_frames(frames, fn_name: str) -> list[np.ndarray]:
    """``frames`` validated (>= ``MIN_STACK_FRAMES``, one shape) and cast to float32, as a
    LIST -- never combined into one ``(N, H, W, C)`` array. Both stack functions band
    directly off this list, ``np.stack``-ing only one row band (all frames, that band) at
    a time, so peak memory never includes a whole extra sequence-equivalent copy.

    Precondition callers must hold: frames should already be float32 (as
    ``astro_calibrate.decode_linear`` and ``astro_register.warp_to_reference`` produce) --
    ``np.asarray(f, dtype=np.float32)`` is then a no-op view, not a copy. Any other dtype
    makes this comprehension materialize a whole new float32 sequence, costing one extra
    full sequence-equivalent at peak -- see ``preflight._ASTRO_PEAK_SEQUENCE_EQUIVALENTS``.
    """
    frames = [np.asarray(f, dtype=np.float32) for f in frames]
    if len(frames) < MIN_STACK_FRAMES:
        raise StackInputError(f"{fn_name} needs >= {MIN_STACK_FRAMES} frames, got {len(frames)}")
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise StackInputError(f"{fn_name} requires one shape, got {sorted(shapes)}")
    return frames


def _kappa_for_count(count: np.ndarray) -> np.ndarray:
    """Per-pixel kappa: keyed on THIS pixel's own valid COUNT, never ``len(frames)``."""
    lowest = SHORT_STACK_KAPPA_BY_N[min(SHORT_STACK_KAPPA_BY_N)]
    kappa = np.full(count.shape, lowest, dtype=np.float32)
    for key in sorted(SHORT_STACK_KAPPA_BY_N):
        kappa = np.where(count >= key, SHORT_STACK_KAPPA_BY_N[key], kappa)
    return kappa


def _require_a_valid_pixel(validity: list[np.ndarray]) -> None:
    """Hard-fail when ``validity`` says NO pixel, anywhere in the frame, has even one
    valid contributor. Left unchecked, the masked-mean tail BOTH rejection modes in
    ``_short_stack_reject`` share -- leave-one-out reject at ``n >=
    SHORT_STACK_MIN_REJECT_FRAMES``, plain masked mean below it -- still runs to
    completion on an entirely-False mask and returns a numerically well-defined but
    meaningless all-zero (black) result. Below the reject threshold that is truly
    silent: no exception, no warning, no NaN. AT OR ABOVE it, the nanmedian/MAD
    degeneracy block runs on an all-NaN slice per pixel-channel and emits an
    UNATTRIBUTED ``RuntimeWarning`` ("All-NaN slice encountered") that names no
    frame, no stage, and no failure. It RAISES ~2 per pixel-channel (measured
    all-invalid at n=8: 384 at 8x8x3, 24576 at 64x64x3, 393216 at 256x256x3 under
    ``simplefilter("always")``), but under the DEFAULT filters the CLI actually runs
    under, ``__warningregistry__`` dedups per warn-site and stderr gets exactly TWO
    lines at every one of those sizes -- the count does not scale with the frame.
    So: not silence, but not an operator signal either. Neither regime tells an
    operator a multi-hour render produced nothing.

    Deliberately NOT ``MIN_STACK_CONTRIBUTORS``: that constant is the
    ``thin_coverage`` POLICY's floor -- a per-PIXEL "keep it but mark it" decision
    for a pixel with SOME real contributors, just fewer than the floor (see
    ``StackResult``/``_thin_coverage``). Zero contributors, GLOBALLY (every pixel,
    every frame), is a different case: there is no partial signal to keep, average,
    or mark -- there is nothing. This check is a structural zero-vs-nonzero test,
    never a tunable floor, and it stays that way on purpose."""
    counts = contributor_count(validity)
    if not np.any(counts):
        raise StackInputError(
            f"stack: all {len(validity)} frame(s)' validity mask is False at every "
            "pixel -- no frame contributed a single valid sample anywhere in the sky "
            "stack, so there is nothing to average. Refusing rather than silently "
            "writing a black master."
        )


def _masked_median(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median over the short frame axis without ``nanmedian``'s full-frame NaN copies."""
    valid = np.asarray(valid, dtype=bool)
    ordered = np.sort(np.where(valid[..., None], values, np.inf), axis=0)
    count = valid.sum(axis=0)
    lower_index = np.maximum((count - 1) // 2, 0)[None, ..., None]
    upper_index = (count // 2)[None, ..., None]
    index_shape = (1, *count.shape, values.shape[-1])
    lower = np.take_along_axis(ordered, np.broadcast_to(lower_index, index_shape), axis=0)[0]
    upper = np.take_along_axis(ordered, np.broadcast_to(upper_index, index_shape), axis=0)[0]
    result = (lower + upper) * np.float32(0.5)
    return np.where(count[..., None] > 0, result, np.nan)


def _short_stack_reject(
    frames: list[np.ndarray],
    validity: list[np.ndarray] | None,
    *,
    bright_only: bool = True,
    return_counts: bool = False,
):
    """Row-banded outlier reject; ``validity``: per-frame bool.

    Returns ``(mean, frac, mode)`` by default. ``return_counts=True`` additionally returns the
    post-rejection, per-pixel contributor count as ``(mean, count, frac, mode)``. The default
    shape is preserved for existing callers of this deliberately private seam.

    Rejects a BRIGHT (high-side) outlier by default; ``bright_only=False`` uses ``|z|`` and
    rejects dark outliers too, at ~DOUBLE the false rejection (kappa is calibrated one-sided).
    Hard-fails via ``_require_a_valid_pixel`` when ``validity`` is supplied and marks
    every pixel invalid in every frame -- see that function for why."""
    if validity is not None:
        _require_a_valid_pixel(validity)
    n = len(frames)
    reject_enabled = n >= SHORT_STACK_MIN_REJECT_FRAMES
    policy = "leave_one_out_bright" if bright_only else "leave_one_out_two_sided"
    mode = policy if reject_enabled else "no_reject_fallback"
    height, width, channels = frames[0].shape
    out = np.empty((height, width, channels), dtype=np.float32)
    counts_out = np.empty((height, width), dtype=np.uint16) if return_counts else None
    rejected_total, sample_total = 0, 0
    for row0 in range(0, height, BAND_ROWS):
        row1 = min(row0 + BAND_ROWS, height)
        band = np.stack([f[row0:row1] for f in frames], axis=0)
        if validity is None:
            valid = np.ones(band.shape[:3], dtype=bool)
        else:
            valid = np.stack([v[row0:row1] for v in validity], axis=0).astype(bool)
        valid_f = valid[..., None].astype(np.float32)
        if reject_enabled:
            total = (band * valid_f).sum(axis=0)
            count = valid_f.sum(axis=0)
            kappa = _kappa_for_count(count)
            others_count = count - valid_f
            others_count_floor = np.maximum(others_count, 1.0)
            # Center before squaring -- never E[x^2]-E[x]^2, which cancels below sigma ~5e-4.
            mu = total / np.maximum(count, 1.0)
            centered = band - mu
            centered *= valid_f
            sq_masked = centered * centered
            total_sq = sq_masked.sum(axis=0)
            np.subtract(total_sq, sq_masked, out=sq_masked)
            sq_masked /= others_count_floor
            sq_masked -= (centered / others_count_floor) ** 2
            np.maximum(sq_masked, 0.0, out=sq_masked)
            np.sqrt(sq_masked, out=sq_masked)
            # Per-CHANNEL degeneracy via median/MAD, excluded from `z`, not veto/vote.
            median = _masked_median(band, valid)
            mad = _masked_median(np.abs(band - median), valid)
            chan_ok = mad > DEGENERACY_REL_EPS * np.abs(median)
            del median, mad
            sq_masked += SIGMA_CLIP_STD_EPS
            z = centered * (count / others_count_floor) / sq_masked
            del centered, sq_masked
            cz = np.where(chan_ok, z if bright_only else np.abs(z), -np.inf).max(-1, keepdims=True)
            del z, chan_ok
            enough = count >= SHORT_STACK_MIN_REJECT_FRAMES
            reject = valid_f.astype(bool) & enough & (cz > kappa)
            mask = valid[..., None] & ~reject
        else:
            mask = valid[..., None]
        mask_full = np.broadcast_to(mask, band.shape)
        count_final = np.maximum(mask_full.sum(axis=0), 1)
        out[row0:row1] = (np.where(mask_full, band, 0.0).sum(axis=0) / count_final).astype(
            np.float32
        )
        if counts_out is not None:
            counts_out[row0:row1] = mask[..., 0].sum(axis=0, dtype=np.uint16)
        # rejected_total/sample_total answer "what fraction of VALID samples did the
        # sigma clip itself throw away" -- an invalid (out-of-footprint) sample is
        # excluded from BOTH sides, never counted as a rejection. ``reject`` (when
        # computed) already implies ``valid`` (see its definition above), so summing
        # it alone is exactly "valid & reject".
        valid_full = np.broadcast_to(valid[..., None], band.shape)
        sample_total += int(valid_full.sum())
        if reject_enabled:
            rejected_total += int(np.broadcast_to(reject, band.shape).sum())
    rejected_fraction = rejected_total / sample_total if sample_total else 0.0
    if counts_out is not None:
        return out, counts_out, rejected_fraction, mode
    return out, rejected_fraction, mode


def stack_sky(aligned_frames, validity=None, *, bright_only=True):
    """Per-pixel MEAN rejecting an outlier SAMPLE per frame-pixel; see ``_short_stack_reject``."""
    frames = _validated_frames(aligned_frames, "stack_sky")
    mean, _rejected_fraction, _mode = _short_stack_reject(frames, validity, bright_only=bright_only)
    return mean


def stack_ground(frames) -> np.ndarray:
    """Per-pixel MEDIAN of the RAW, unaligned frames -- the fixed foreground converges,
    an unaligned star lands at a different pixel each frame and gets discarded."""
    frames = _validated_frames(frames, "stack_ground")
    height, width, channels = frames[0].shape
    out = np.empty((height, width, channels), dtype=np.float32)
    for row0 in range(0, height, BAND_ROWS):
        row1 = min(row0 + BAND_ROWS, height)
        band = np.stack([f[row0:row1] for f in frames], axis=0)
        out[row0:row1] = np.median(band, axis=0).astype(np.float32)
    return out


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Clamped cubic ease, flat-tangented at both ends — ``masks.py``'s shared shape."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def horizon_mask(shape_hw, *, horizon_frac: float | None = None, star_density=None) -> np.ndarray:
    """An ``(H, W)`` float32 mask in [0, 1]: 1.0 sky, 0.0 ground, feathered across
    ``FEATHER_ROWS`` at the boundary. Exactly one of two ways to place that boundary:
    ``horizon_frac`` is the operator's own ``--horizon FRACTION`` (0..1 of the height
    from the top), authoritative; ``star_density`` is an ``(H, W)`` map whose row-wise
    mean profile is thresholded at ``STAR_DENSITY_THRESHOLD_FRAC`` of its OWN
    upper-quartile value, boundary one row past the LOWEST row that still clears it (a
    profile that never clears it reads "no sky found": ground everywhere).

    That clearance test is STRICT (``>``), which is what makes the last clause true. A
    sky shallower than 0.25 of the frame height leaves the profile's own upper quartile
    at 0, degenerating the threshold to 0 -- and ``profile >= 0`` holds for every row of
    any non-negative density, so the old non-strict test returned the frame HEIGHT
    there: "sky everywhere, no masking at all", failing OPEN (not the "ground
    everywhere" this docstring used to imply), with the empty-``sky_rows`` branch
    unreachable. Measured over the full 1..199 ridge sweep of
    ``test_horizon_mask_star_density_places_the_ridge_at_every_sky_depth`` (200 rows,
    uniform density above the ridge, none below): non-strict was wrong for every ridge
    <= 49; strict is exact for all 199."""
    if (horizon_frac is None) == (star_density is None):
        raise ValueError("horizon_mask needs exactly one of horizon_frac or star_density")
    height, width = shape_hw
    if horizon_frac is not None:
        if not 0.0 <= horizon_frac <= 1.0:
            raise ValueError(f"horizon_frac must be in [0, 1], got {horizon_frac}")
        boundary_row = horizon_frac * height
    else:
        density = np.asarray(star_density, dtype=np.float32)
        if density.shape != (height, width):
            raise ValueError(f"star_density shape {density.shape} must match {(height, width)}")
        profile = density.mean(axis=1)
        threshold = STAR_DENSITY_THRESHOLD_FRAC * float(np.percentile(profile, 75.0))
        sky_rows = np.flatnonzero(profile > threshold)
        boundary_row = float(sky_rows[-1] + 1) if sky_rows.size else 0.0
    rows = np.arange(height, dtype=np.float32)
    column = _smoothstep((boundary_row - rows) / FEATHER_ROWS + 0.5)
    return np.repeat(column[:, None], width, axis=1).astype(np.float32)


def blend(sky: np.ndarray, ground: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``mask * sky + (1 - mask) * ground``, pure, shape-checked."""
    sky = np.asarray(sky, dtype=np.float32)
    ground = np.asarray(ground, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if sky.shape != ground.shape:
        raise ValueError(f"blend: sky shape {sky.shape} != ground shape {ground.shape}")
    if mask.shape != sky.shape[:2]:
        raise ValueError(f"blend: mask shape {mask.shape} must match frame HxW {sky.shape[:2]}")
    weight = mask[..., None]
    return (weight * sky + (1.0 - weight) * ground).astype(np.float32)


@dataclass(frozen=True)
class StackResult:
    """``stack()``'s output: the blend plus the honesty metrics a ``.stack.json``
    sidecar wants without re-deriving them from the pixels."""

    rgb: np.ndarray
    n_frames: int
    mean_rejected_fraction: float
    rejection_mode: str = "leave_one_out_bright"  # _two_sided, or no_reject_fallback under n
    # Coverage/contributor contract -- both ``None`` unless ``sky_validity`` was
    # supplied (nothing to summarise without real per-frame validity). ``coverage`` is
    # astro_quality.coverage_stats(contributor_count(sky_validity), MIN_STACK_CONTRIBUTORS):
    # min/median contributor count and the fraction of pixels below floor, so a run can be
    # judged rather than guessed at. ``thin_coverage`` is the same floor applied per pixel,
    # an (H, W) bool -- POLICY is "keep the pixel but mark it": ``rgb`` is left untouched at
    # these pixels (see stack()'s docstring for why), the mark lives here instead.
    coverage: CoverageStats | None = None
    thin_coverage: np.ndarray | None = None
    # A later deep combine needs the two pre-blend components plus the ACTUAL samples that
    # survived validity/rejection. Optional defaults preserve every existing StackResult
    # constructor while this internal contract is introduced ahead of persistence.
    sky_sum: np.ndarray | None = None  # float32 RGB numerator, scene-linear/reference geometry
    sky_count: np.ndarray | None = None  # uint16 (H, W), post-rejection contributors
    ground: np.ndarray | None = None  # float32 RGB, fixed camera geometry and never sky-warped


def _thin_coverage(sky_validity) -> tuple[CoverageStats | None, np.ndarray | None]:
    """``(coverage_stats(...), below-floor mask)`` from ``sky_validity``, or ``(None,
    None)`` when there's no validity to summarise -- see ``StackResult`` for the fields."""
    if sky_validity is None:
        return None, None
    counts = contributor_count(sky_validity)
    return coverage_stats(counts, MIN_STACK_CONTRIBUTORS), counts < MIN_STACK_CONTRIBUTORS


def stack(
    aligned_frames, unaligned_frames, mask: np.ndarray, *, sky_validity=None, bright_only=True
) -> StackResult:
    """Thin composition: short-stack-rejected sky + median ground, blended at ``mask``. The
    two frame lists are the SAME captures, warped and raw -- a count mismatch is
    refused rather than silently paired wrong.

    Coverage/contributor contract: a pixel backed by fewer than
    ``MIN_STACK_CONTRIBUTORS`` real frames has no meaningful stacking gain -- the edge-wedge
    case real drift produces. POLICY is "keep the pixel but mark it", not crop or blank:
    ``rgb`` is computed exactly as before (``stack_sky``'s leave-one-out math already
    recovers a real single-sample pixel correctly --
    see test_validity_mask_recovers_sky_from_zero_filled_warp_borders -- overwriting it
    here would fight that), and the thin pixels are surfaced on the result instead via
    ``StackResult.thin_coverage``/``coverage``. Cropping is deliberately NOT done here: it
    would change ``rgb``'s shape, which every caller and ``blend``'s own shape check assume
    stays fixed to the input frame shape. This is the consumer CONTRACT for
    coverage/thin_coverage, not the crop/reject decision, which belongs where real
    warp validity actually gets threaded through. ``sky_validity=None`` leaves both
    ``coverage`` and ``thin_coverage`` ``None`` -- there is nothing to summarise
    without real per-frame validity, whoever the caller is.

    ``sky_validity`` supplied but False at EVERY pixel of EVERY frame -- the all-invalid
    stack, distinct from the per-pixel thin-coverage case above -- raises
    ``StackInputError`` (see ``_require_a_valid_pixel``) instead of silently returning an
    all-zero ``rgb``. Below ``SHORT_STACK_MIN_REJECT_FRAMES`` that pre-fix path was
    truly silent (no exception, warning, or NaN); at or above it there was still no
    raise, only an unattributed per-pixel-channel ``RuntimeWarning`` -- see
    ``_require_a_valid_pixel`` for the detail either way."""
    if len(aligned_frames) != len(unaligned_frames):
        raise StackInputError(
            f"stack needs equal-length aligned/unaligned lists, "
            f"got {len(aligned_frames)} and {len(unaligned_frames)}"
        )
    aligned = _validated_frames(aligned_frames, "stack")
    sky, sky_count, frac, mode = _short_stack_reject(
        aligned, sky_validity, bright_only=bright_only, return_counts=True
    )
    ground = stack_ground(unaligned_frames)
    rgb = blend(sky, ground, mask)
    coverage, thin_coverage = _thin_coverage(sky_validity)
    return StackResult(
        rgb=rgb,
        n_frames=len(aligned_frames),
        mean_rejected_fraction=frac,
        rejection_mode=mode,
        coverage=coverage,
        thin_coverage=thin_coverage,
        sky_sum=(sky * sky_count[..., None]).astype(np.float32),
        sky_count=sky_count,
        ground=ground,
    )
