"""Astro develop: per-channel gradient removal, background white balance,
and the midtone stretch that turns a stacked linear float RGB into a
display-referred uint16 master.

Runs after stacking, before ``develop()``'s ordinary chain. Per-channel
gradient fitting neutralizes a sodium/LED colour cast; ``background_wb``
then equalizes the three channels' sky-background levels (measured off the
background, never a star); the stretch's ``m`` is solved once per step and
shared across channels so it never re-tints. Thresholds are PROVISIONAL
pending a real night card.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .astro_triage import CENTER_REGION_FRACTION, CORNER_REGION_FRACTION

# Cells per axis for the gradient sample grid — coarse enough that a lone
# star can't move a cell's MEDIAN, dense enough for a degree-4 (15-term) fit.
GRADIENT_GRID = 16

# Accepted/default polynomial degrees for the gradient fit.
GRADIENT_DEGREES = (2, 3, 4)
GRADIENT_DEGREE_DEFAULT = 2

# Sigma-reject a sample past this many sigma of the current fit; a
# Milky-Way-core-sized patch of cells reads as a smooth bump across many
# samples (not one outlier) and gets rejected over these iterations.
GRADIENT_REJECT_SIGMA = 3.0
GRADIENT_REJECT_ITERATIONS = 4

# Multiple of n_terms a masked fit needs before it's trusted -- exactly
# n_terms is DETERMINED but still extrapolates wildly. PROVISIONAL.
UNDERDETERMINED_MARGIN = 3

# Condition number past which a masked design matrix is untrustworthy even with
# enough samples (e.g. 1-2 sampled grid rows starve the y/xy/y^2 terms of variety).
# 1e8 is float64's standard rule of thumb: cond 10**k costs ~k of ~15-16 digits.
CONDITION_NUMBER_MAX = 1e8

# Sky pixels a masked median/std needs before it is trusted; below it a handful of
# pixels set the black point for the whole frame. PROVISIONAL, like every threshold here.
MIN_SKY_SAMPLES = 256

BLACK_POINT_SIGMAS = 2.8  # black point = luminance median - this many sigma
STRETCH_TARGET_BACKGROUND = 0.12  # design target: a 0.10-0.15 post-stretch window
STRETCH_STEPS = 3  # several small stretches beat one violent one

LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)  # BT.709
_UINT16_MAX = 65535


def _poly_terms(degree):
    """``[(i, j), ...]`` for every ``x**i * y**j`` with ``i + j <= degree``,
    lowest total degree first — index 0 is always the constant term."""
    return [(i, total - i) for total in range(degree + 1) for i in range(total + 1)]


def _design_matrix(x, y, degree):
    return np.stack([np.power(x, i) * np.power(y, j) for i, j in _poly_terms(degree)], axis=-1)


def _grid_samples(channel, grid, mask=None):
    """Per-cell ``(x, y, median)`` over a ``grid``x``grid`` partition,
    coordinates normalised to ``[0, 1]``. The per-cell MEDIAN already shrugs
    off a lone star; only structure spanning many cells needs rejection.
    ``mask``: a cell with no True pixels is dropped, others use those only."""
    h, w = channel.shape
    row_bands = [b for b in np.array_split(np.arange(h), grid) if b.size]
    col_bands = [b for b in np.array_split(np.arange(w), grid) if b.size]
    xs, ys, vals = [], [], []
    for rows in row_bands:
        for cols in col_bands:
            cell = channel[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
            if mask is not None:
                cell_mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
                if not cell_mask.any():
                    continue
                cell = cell[cell_mask]
            vals.append(np.median(cell))
            ys.append((rows[0] + rows[-1]) / 2.0 / max(h - 1, 1))
            xs.append((cols[0] + cols[-1]) / 2.0 / max(w - 1, 1))
    return np.array(xs), np.array(ys), np.array(vals)


def _fit_gradient(channel, degree, mask=None):
    """Sigma-rejecting least-squares fit; returns ``(coeffs, n_samples_used)``.
    Two-sided always (regression test below). Masked + underdetermined OR
    ill-conditioned (too few samples or too little coordinate spread) falls back to a
    constant surface and WARNS (zero raises); unmasked always takes the original lstsq."""
    xs, ys, vals = _grid_samples(channel, GRADIENT_GRID, mask)
    n_terms = len(_poly_terms(degree))
    if mask is not None:
        if vals.size == 0:
            raise ValueError("sky_mask leaves no sample cells for the gradient fit")
        underdetermined = vals.size < n_terms * UNDERDETERMINED_MARGIN
        ill_conditioned = np.linalg.cond(_design_matrix(xs, ys, degree)) > CONDITION_NUMBER_MAX
        if underdetermined or ill_conditioned:
            reason = "too few samples" if underdetermined else "ill-conditioned design matrix"
            warnings.warn(
                f"sky_mask gradient fit fell back to a constant surface ({reason}, "
                f"n_samples={vals.size}, n_terms={n_terms})",
                RuntimeWarning,
                stacklevel=3,
            )
            coeffs = np.zeros(n_terms)
            coeffs[0] = float(np.median(vals))
            return coeffs, int(vals.size)
    sample_mask = np.ones(vals.shape, dtype=bool)
    coeffs, *_ = np.linalg.lstsq(_design_matrix(xs, ys, degree), vals, rcond=None)
    for _ in range(GRADIENT_REJECT_ITERATIONS):
        residuals = vals - _design_matrix(xs, ys, degree) @ coeffs
        sigma = np.std(residuals[sample_mask])
        if sigma <= 0:
            break
        new_mask = np.abs(residuals) <= GRADIENT_REJECT_SIGMA * sigma
        if new_mask.sum() < n_terms or np.array_equal(new_mask, sample_mask):
            break
        sample_mask = new_mask
        coeffs, *_ = np.linalg.lstsq(
            _design_matrix(xs[sample_mask], ys[sample_mask], degree), vals[sample_mask], rcond=None
        )
    return coeffs, int(sample_mask.sum())


@dataclass(frozen=True)
class GradientResidual:
    """How well the per-channel gradient surface tracked the frame at its centre vs.
    its four corners (astro_triage's centre-box-vs-corner-boxes convention,
    ``CENTER_REGION_FRACTION`` / ``CORNER_REGION_FRACTION`` of the shorter side; the
    four corner boxes are pooled into one region, matching ``_region_shapes``).

    Each value is ``median(|luma residual|)`` over that region's pixels, divided by the
    ONE frame-wide ``median(|luma fitted surface|)`` -- a fraction of the fitted
    background level, not a raw ADU number, so it means the same thing across exposures.
    A single frame-wide normaliser (rather than each region's own local surface level) is
    deliberate: the fitted surface can cross zero within a region -- centred on a night
    sky's actual dark corners -- and dividing by a local level near that crossing would
    blow the fraction up for reasons that have nothing to do with fit quality. ``None``
    when the region has no valid (unmasked) pixels, or the frame's overall fitted level
    is non-positive and a fraction cannot be meaningfully formed -- an honest absence,
    never a fabricated zero."""

    center: float | None
    corners: float | None


def _gradient_regions(h, w, mask):
    """Boolean ``(center, corners)`` pixel masks over an ``h``x``w`` frame, matching
    ``astro_triage._region_shapes``'s convention exactly (a pixel in the centre box is
    never also counted as a corner pixel). ``mask`` (optional) is ANDed into both."""
    side = min(h, w)
    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2.0, h / 2.0
    center_half = side * CENTER_REGION_FRACTION / 2.0
    corner_side = side * CORNER_REGION_FRACTION
    center = (np.abs(x_idx - cx) <= center_half) & (np.abs(y_idx - cy) <= center_half)
    near_x = (x_idx <= corner_side) | (x_idx >= w - corner_side)
    near_y = (y_idx <= corner_side) | (y_idx >= h - corner_side)
    corners = (near_x & near_y) & ~center
    if mask is not None:
        center = center & mask
        corners = corners & mask
    return center, corners


def _region_residual_fraction(region_mask, residual_luma, background_level):
    """``None`` for an empty region or a non-positive frame-wide level -- see
    ``GradientResidual``."""
    if not region_mask.any() or background_level <= 0:
        return None
    return float(np.median(np.abs(residual_luma[region_mask])) / background_level)


def _remove_gradient_impl(rgb, degree, sky_mask):
    if degree not in GRADIENT_DEGREES:
        raise ValueError(f"gradient degree must be one of {GRADIENT_DEGREES}, got {degree}")
    rgb = np.asarray(rgb, dtype=np.float64)
    h, w = rgb.shape[:2]
    if sky_mask is not None:
        sky_mask = np.asarray(sky_mask, dtype=bool)
        if sky_mask.shape != (h, w):
            raise ValueError(f"sky_mask shape {sky_mask.shape} != frame shape {(h, w)}")
    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float64)
    x_norm, y_norm = x_idx / max(w - 1, 1), y_idx / max(h - 1, 1)
    full_design = _design_matrix(x_norm, y_norm, degree)
    out = np.empty_like(rgb)
    residual_luma = np.zeros((h, w), dtype=np.float64)
    surface_luma = np.zeros((h, w), dtype=np.float64)
    for c in range(3):
        channel = rgb[:, :, c]
        coeffs, _n_samples = _fit_gradient(channel, degree, sky_mask)
        surface = full_design @ coeffs
        out[:, :, c] = channel - surface + surface.mean()
        residual_luma += LUMA_WEIGHTS[c] * (channel - surface)
        surface_luma += LUMA_WEIGHTS[c] * surface
    center_mask, corners_mask = _gradient_regions(h, w, sky_mask)
    background_level = float(np.median(np.abs(surface_luma)))
    residual = GradientResidual(
        center=_region_residual_fraction(center_mask, residual_luma, background_level),
        corners=_region_residual_fraction(corners_mask, residual_luma, background_level),
    )
    return out.astype(np.float32), residual


def remove_gradient(rgb, degree=GRADIENT_DEGREE_DEFAULT, sky_mask=None):
    """Per-channel sample-grid gradient fit + subtract; each channel's
    fitted surface MEAN is added back so its overall level survives.

    The mean, not ``coeffs[0]``. The design matrix normalises x and y to [0, 1]
    rather than centring them, so ``coeffs[0]`` is the surface at the (0, 0)
    CORNER — the darkest point of a vignetted night frame. Adding a corner back
    while subtracting a whole surface drops every pixel by the difference, which can
    push whole channels below zero. The damage is silent and total:
    ``background_wb`` reads a non-positive channel median, takes
    its divide-by-zero guard and returns gains of exactly 1.0 (no white balance),
    then the stretch clips the negatives to black and the render is monochrome
    blue. Subtracting a surface and adding back that same surface's mean leaves
    the channel mean unchanged by construction, which is what the pinning test
    asserts.

    ``sky_mask`` (optional): boolean, shaped like ``rgb``'s first
    two axes; False excludes ground/foreground/bright sources from the fit.
    Default ``None`` reproduces today's whole-frame output byte-for-byte.

    See ``remove_gradient_detailed`` for the same fit plus its ``GradientResidual``."""
    out, _residual = _remove_gradient_impl(rgb, degree, sky_mask)
    return out


def remove_gradient_detailed(rgb, degree=GRADIENT_DEGREE_DEFAULT, sky_mask=None):
    """``remove_gradient``, plus the fit's own ``GradientResidual`` (centre vs. corners)
    from the SAME per-channel fit -- never a second, separate fitting pass."""
    return _remove_gradient_impl(rgb, degree, sky_mask)


def background_wb(rgb):
    """White balance off the sky BACKGROUND, never a star.

    Each channel's background is its whole-frame MEDIAN — not the mean, not
    the max. A star, or the Milky Way core, is a small minority of a
    frame's pixels and cannot move a median the way it moves a mean or a
    max; the mean/max would let the brightest, most saturated pixels set
    the white point, which is exactly what this step avoids.

    Each channel is then scaled so its background matches the MEAN of the
    three measured backgrounds, rather than pinning to one reference
    channel (e.g. green) — that keeps the overall image level from
    lurching when a single channel happens to be the outlier.

    LINEAR-light operation: must run before the stretch, never after — the
    stretch is a one-way door ("everything up to the stretch happens
    in linear light"). ``remove_gradient`` already adds each channel's own
    constant back, which only partially levels a cast; this is the step
    that actually equalizes the three channel backgrounds.

    Returns ``(rgb, gains)``: the balanced float32 frame and the
    per-channel gains applied, for the ``.stack.json`` sidecar.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    backgrounds = np.array([np.median(rgb[:, :, c]) for c in range(3)])
    positive = backgrounds[backgrounds > 0]
    target = float(np.mean(positive)) if positive.size else 0.0
    # A non-positive background can't produce a meaningful ratio — leave
    # that channel untouched rather than dividing by zero or guessing.
    gains = tuple(float(target / bg) if bg > 0 and target > 0 else 1.0 for bg in backgrounds)
    out = rgb * np.array(gains)
    return out.astype(np.float32), gains


def _luma(rgb):
    return np.asarray(rgb, dtype=np.float64) @ LUMA_WEIGHTS


def _sky_samples(luma, sky_mask):
    """Flat luminance samples restricted to ``sky_mask``: a bright ground
    silhouette, lit building or horizon glow sitting in the WHOLE-FRAME stats can shift
    the median toward its own level while leaving std deceptively small, driving the
    black point above the sky's own faint signal and crushing it to zero. ``None`` is the
    whole frame and reproduces today's output exactly. A degenerate mask does NOT fall
    back to the whole frame -- that would silently restore the very statistic this mask
    exists to remove -- so an empty one RAISES (as ``_fit_gradient`` does) and one under
    ``MIN_SKY_SAMPLES`` warns (as its too-few-samples branch does) and returns what it has."""
    if sky_mask is None:
        return luma
    samples = luma[np.asarray(sky_mask, dtype=bool)]
    if samples.size == 0:
        raise ValueError("sky_mask leaves no sky pixels for the black point / background stats")
    if samples.size < MIN_SKY_SAMPLES:
        warnings.warn(
            f"sky_mask leaves too few sky pixels for a stable black point "
            f"(n_sky={samples.size}, minimum {MIN_SKY_SAMPLES})",
            RuntimeWarning,
            stacklevel=3,
        )
    return samples


def black_point(rgb, sky_mask=None):
    """Unified black point from luminance stats — the SAME value
    ``stretch`` applies to all three channels, never a per-channel one.

    ``sky_mask`` (optional): restrict the median/std this is measured from to sky
    pixels only (see ``_sky_samples``). Default ``None`` reproduces today's
    whole-frame statistics byte-for-byte."""
    samples = _sky_samples(_luma(rgb), sky_mask)
    return float(max(np.median(samples) - BLACK_POINT_SIGMAS * np.std(samples), 0.0))


def mtf(x, m):
    """Midtone transfer function: maps ``m`` -> 0.5, fixes both endpoints."""
    x = np.asarray(x, dtype=np.float64)
    return ((m - 1.0) * x) / (((2.0 * m) - 1.0) * x - m)


def solve_m(background_in, target=STRETCH_TARGET_BACKGROUND):
    """Closed-form inverse of ``mtf``: the ``m`` solving
    ``mtf(background_in, m) == target``."""
    b, t = float(background_in), float(target)
    denom = 2.0 * b * t - b - t
    return 0.5 if denom == 0 else b * (t - 1.0) / denom


def _geometric(a, b, t):
    return max(a, 1e-6) ** (1.0 - t) * max(b, 1e-6) ** t


def _stretch_detailed(rgb, steps=STRETCH_STEPS, sky_mask=None):
    rgb = np.asarray(rgb, dtype=np.float64)
    bp = black_point(rgb, sky_mask)
    shifted = np.clip(rgb - bp, 0.0, None)
    peak = float(shifted.max())
    working = shifted / peak if peak > 0 else shifted
    background_before = float(np.median(_sky_samples(_luma(working), sky_mask)))
    m_values = []
    for step in range(1, steps + 1):
        step_target = _geometric(background_before, STRETCH_TARGET_BACKGROUND, step / steps)
        current = float(np.median(_sky_samples(_luma(working), sky_mask)))
        m = solve_m(current, step_target)
        working = mtf(working, m)
        m_values.append(m)
    background_after = float(np.median(_sky_samples(_luma(working), sky_mask)))
    rgb_uint16 = (np.clip(working, 0.0, 1.0) * _UINT16_MAX).round().astype(np.uint16)
    return rgb_uint16, bp, m_values, background_before, background_after, peak


def stretch(rgb, steps=STRETCH_STEPS, sky_mask=None):
    """Subtract the unified black point, then ``steps`` small MTF stretches —
    the SAME ``m`` on all three channels each step (per-channel would shift
    colour) — each solved for a geometric intermediate target. ``sky_mask``
    restricts the black point and the background reads that drive each step's
    target to sky pixels only; default ``None`` is whole-frame."""
    rgb_uint16, *_ = _stretch_detailed(rgb, steps, sky_mask)
    return rgb_uint16


# Radius, in pixels, of the blur applied to the colour difference planes. Chroma
# carries almost no real spatial detail at this scale -- nebulosity and dust lanes
# are broad -- while sensor colour noise is per-pixel, so a blur this wide erases
# the speckle and leaves the picture. On clear sky away from the band this cuts the
# Lab chroma sigma substantially while leaving the luma sigma, and so star detail,
# unchanged.
CHROMA_BLUR_SIGMA = 3.0


def chroma_sigma(rgb_uint16) -> float:
    """How much colour speckle an image carries: the spread of each pixel's
    departure from its own grey. Zero for a perfectly neutral frame, whatever
    its brightness or its luma noise."""
    f = np.asarray(rgb_uint16, dtype=np.float64) / _UINT16_MAX
    return float((f - f.mean(axis=2, keepdims=True)).std())


def denoise_chroma(rgb_uint16, sigma=CHROMA_BLUR_SIGMA):
    """Blur the COLOUR of a developed master without touching its brightness.

    Splits each pixel into its grey level and its colour departure from that
    grey, blurs only the second, and recombines. Luma is preserved exactly and
    not merely approximately: the colour planes sum to zero across channels at
    every pixel by construction, and a Gaussian blur is linear, so the blurred
    planes still sum to zero and the recombined grey is the original grey.

    Runs AFTER the stretch, on display-referred uint16 -- unlike gradient
    removal and white balance, which must happen in linear light. Colour noise
    is what makes a clean stack look speckled, and it is the one kind of noise
    stacking cannot average away when the frames share a sensor's colour
    response.

    Deliberately NOT part of ``develop_stack``: every step there is invertible
    from the numbers on ``AstroDevelopResult``, which is what lets
    ``astro_contact.recover_balanced`` walk a master back to balanced linear.
    A blur is not invertible. So this is the WRITER's step -- the caller applies
    it to the image it is about to hand a viewer, and ``rgb_uint16`` stays the
    reversible artifact every other consumer relies on.
    """
    import cv2

    f = np.asarray(rgb_uint16, dtype=np.float32) / _UINT16_MAX
    luma = f.mean(axis=2, keepdims=True)
    smoothed = cv2.GaussianBlur(f - luma, (0, 0), float(sigma))
    return (np.clip(luma + smoothed, 0.0, 1.0) * _UINT16_MAX).round().astype(np.uint16)


# The display render's three knobs. They are DELIBERATELY not the master's: the .tif
# lands its sky at STRETCH_TARGET_BACKGROUND (0.12) because that is the archival
# choice -- recoverable back to balanced linear, nothing clipped, faint signal intact
# on a darker night. That same frame shown small looks black, so an
# operator judging framing or picking a keeper cannot see their own picture.
# The black point sits INSIDE the sky distribution, on purpose. A master has already
# been stretched to put its background at STRETCH_TARGET_BACKGROUND, so its darkest
# pixels are nowhere near zero; a black point down at the 1st percentile lifts that
# background a second time and returns a washed grey, a picture with nothing in it.
# Cutting at the 30th percentile keeps the sky floor low and the band well above it,
# which is the separation a viewer is actually looking for. The cost is
# that the darkest 30% -- foreground, and the faintest corner of sky -- goes to black,
# and that is the correct trade for a display render whose whole job is to be looked
# at. Nothing downstream reads this, and the .tif keeps every one of those pixels.
PREVIEW_BLACK_PERCENTILE = 30.0
PREVIEW_WHITE_PERCENTILE = 99.8  # above this is a handful of star cores
PREVIEW_ARCSINH_GAIN = 10.0  # arcsinh: steep on the faint end, gentle on the bright
PREVIEW_SATURATION = 1.35  # dust lanes and airglow are real colour, shown honestly


def preview_render(rgb_uint16) -> np.ndarray:
    """An 8-bit render of a master meant for a human to LOOK at, not to process.

    Explicitly and irreversibly lossy, which is why it is a separate artifact and
    never replaces the .tif: the percentile black point discards the bottom of the
    sky, the arcsinh curve is not one of the ``m_values`` any sidecar records, and
    the saturation lift is a choice about taste rather than a measurement. Nothing
    downstream reads it -- no stacking, no photometry, no contact sheet analysis --
    so no later stage can inherit those choices by accident.

    Does not modify its input.
    """
    import cv2

    f = np.asarray(rgb_uint16, dtype=np.float32) / _UINT16_MAX
    lo = float(np.percentile(f, PREVIEW_BLACK_PERCENTILE))
    hi = float(np.percentile(f, PREVIEW_WHITE_PERCENTILE))
    g = np.clip((f - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    g = np.arcsinh(g * PREVIEW_ARCSINH_GAIN) / np.arcsinh(PREVIEW_ARCSINH_GAIN)
    u8 = (g * 255.0).round().clip(0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * PREVIEW_SATURATION, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


@dataclass(frozen=True)
class AstroDevelopResult:
    """The numbers the orchestrator's ``.stack.json`` sidecar records."""

    rgb_uint16: np.ndarray
    black_point: float
    m_values: list
    background_before: float
    background_after: float
    background_gains: tuple
    # ``shifted.max()`` from the stretch step -- the stretch's third, otherwise-hidden
    # parameter. Replaying black_point/m_values alone (``develop_stack_with_params``)
    # would still normalise a meteor layer to ITS OWN peak rather than base's.
    peak: float = 0.0
    # The mask this result was developed under, so ``develop_stack_with_params`` refits a
    # meteor layer's gradient the same way rather than trusting a caller to re-supply it.
    sky_mask: np.ndarray | None = None
    # The gradient fit's own centre-vs-corners residual (see ``GradientResidual``); a
    # later full-frame product needs this to see whether a low-degree fit held up.
    gradient_residual: GradientResidual | None = None


def develop_stack(rgb, sky_mask=None) -> AstroDevelopResult:
    """Thin composition: gradient removal -> background white balance ->
    stretch. Both colour steps run in linear light, before the stretch's
    one-way door (everything up to the stretch happens in linear
    light). ``sky_mask`` passes straight through to the gradient fit AND to
    the stretch's black point / background reads — ground/foreground
    pixels can no longer inflate either. Default ``None`` reproduces today's
    whole-frame output byte-for-byte."""
    degraded, gradient_residual = remove_gradient_detailed(rgb, GRADIENT_DEGREE_DEFAULT, sky_mask)
    balanced, gains = background_wb(degraded)
    rgb_uint16, bp, m_values, before, after, peak = _stretch_detailed(balanced, sky_mask=sky_mask)
    return AstroDevelopResult(
        rgb_uint16, bp, m_values, before, after, gains, peak, sky_mask, gradient_residual
    )


def develop_stack_with_params(rgb, base: AstroDevelopResult) -> np.ndarray:
    """Develop ``rgb`` -- a frame already calibrated and warped into the SAME geometry
    as whatever stack produced ``base`` -- using ``base``'s OWN tone rather than fitting
    a fresh one (the ``astro_meteors`` module-docstring contract).

    Gradient removal still fits ``rgb``'s own pixels (gradient is per-frame
    -- and ``AstroDevelopResult`` never records a gradient model, so this cannot
    accidentally reuse ``base``'s) but under ``base.sky_mask`` -- the same mask
    ``develop_stack`` fitted under, carried on the result so no caller can forget it and
    leave the layer on a different tone. Background gain, black point, peak normalisation,
    and every ``m`` then reuse ``base``'s recorded values, applied in the same order
    ``develop_stack``/``_stretch_detailed`` used to derive them.

    Exact, not approximate: feeding ``develop_stack``'s own input stack back through
    here reproduces its ``rgb_uint16`` byte-for-byte, because every step below is a
    deterministic function of inputs that are, in that case, identical to the ones
    ``develop_stack`` used to build ``base``.
    """
    degraded = remove_gradient(rgb, GRADIENT_DEGREE_DEFAULT, base.sky_mask)
    balanced = (np.asarray(degraded, dtype=np.float64) * np.array(base.background_gains)).astype(
        np.float32
    )
    shifted = np.clip(np.asarray(balanced, dtype=np.float64) - base.black_point, 0.0, None)
    working = shifted / base.peak if base.peak > 0 else shifted
    for m in base.m_values:
        working = mtf(working, m)
    return (np.clip(working, 0.0, 1.0) * _UINT16_MAX).round().astype(np.uint16)
