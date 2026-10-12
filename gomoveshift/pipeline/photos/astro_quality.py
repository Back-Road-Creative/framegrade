"""Signal/rejection/geometry quality metrics for astro stacking (metric-helper half).

Pure numpy, no I/O, no test-fixture import (callers wire these to real stack output and
pin thresholds at the call site). Degenerate input (empty mask/array, n=0) returns a
documented finite value, never raises or emits NaN.

``contrast_retention`` TRUSTS its optional ``known_nuisance`` argument; there is no
gate that verifies it against ``reference``. A wrong or incomplete ``known_nuisance`` silently
produces a wrong retention value -- see that function's docstring and
astro-quality-contract.md for the evidence and the follow-up design.

``a3_by_radial_third`` is the one exception to "no other astro_* imports": it reuses
astro_triage's star detector/shape measurer rather than reimplementing star detection and
second-moment PSF shape here -- see its own docstring.
"""

from collections import namedtuple

import numpy as np

from gomoveshift.pipeline.photos.astro_triage import (
    _isotropic_fwhm,
    _measure_star_shapes,
    _median_shape,
    _robust_stats,
    detect_stars,
)

_MAD_TO_SIGMA = 1.4826  # same convention as astro_calibrate.py / astro_triage.py.
_DENOM_EPS = 1e-9  # floor for a legitimately-zero denominator (flat synthetic input).

# PROVISIONAL acceptance contracts -- call-site policy, never enforced inside this module.
CONTRAST_RETENTION_MIN = 0.90
FALSE_REJECTION_RATE_MAX = 0.01

# One namedtuple per multi-value result -- attribute access, like sibling astro_* dataclasses.
NoiseImprovement = namedtuple("NoiseImprovement", "measured_factor ideal_factor ratio_to_ideal")
RejectionStats = namedtuple("RejectionStats", "rate any_rejected_fraction")
StellarColorRatios = namedtuple(
    "StellarColorRatios", "rg_ratios bg_ratios rg_spread bg_spread valid"
)
CoverageStats = namedtuple("CoverageStats", "min_count median_count below_floor_fraction")


def broad_feature_contrast(image, feature_mask, background_mask) -> float:
    """(median(image[feature_mask]) - median(image[background_mask])) /
    median(image[background_mask]); median (not mean) for outlier robustness.
    0.0 (never NaN/raise) if image/either mask is empty, or background median is 0."""
    image = np.asarray(image, dtype=np.float64)
    feature_mask = np.asarray(feature_mask, dtype=bool)
    background_mask = np.asarray(background_mask, dtype=bool)
    if image.size == 0 or not feature_mask.any() or not background_mask.any():
        return 0.0
    background_level = float(np.median(image[background_mask]))
    if background_level == 0.0:
        return 0.0
    feature_level = float(np.median(image[feature_mask]))
    return (feature_level - background_level) / background_level


def contrast_retention(
    reference, after, feature_mask, background_mask, known_nuisance=None
) -> float:
    """INJECTED-SIGNAL FIXTURE METRIC ONLY -- broad_feature_contrast(after,...) /
    broad_feature_contrast(reference,...), the input to the CONTRAST_RETENTION_MIN gate. `reference` MUST be gradient-free (a synthetic fixture's clean base frame
    before the nuisance is injected); no production caller can supply that -- UNLESS
    `known_nuisance` is given, in which case `reference - known_nuisance` stands in as
    the nuisance-free denominator (retention measured against a corrected
    reference, not the raw contaminated one). Do NOT pass the pre-correction frame as
    `reference` with `known_nuisance=None` -- geometry alone
    flips the ratio's sign.

    TRUSTS `known_nuisance` -- does not verify it against `reference` in any way. A
    wrong or incomplete `known_nuisance` silently produces a wrong retention value.
    Six rounds tried a correlation-based gate to catch that; every version either
    false-rejected a clean reference carrying ordinary sky structure (stars, ground,
    cloud) or, once tuned not to, lost detection sensitivity by 1-2 orders of magnitude
    and still failed on structurally plain references (see astro-quality-contract.md
    for the measured numbers and the untried matched-filter follow-up).

    Even with a correct `known_nuisance`, the ratio is exact (1.0) only for a perfect
    correction of `after`; a partial one stays geometry-sensitive and can go negative."""
    reference = np.asarray(reference, dtype=np.float64)
    if known_nuisance is not None:
        reference = reference - np.asarray(known_nuisance, dtype=np.float64)
    reference_contrast = broad_feature_contrast(reference, feature_mask, background_mask)
    if reference_contrast == 0.0:
        return 0.0
    after_contrast = broad_feature_contrast(after, feature_mask, background_mask)
    return after_contrast / reference_contrast


def background_noise(image, mask) -> float:
    """1.4826 * median(abs(image[mask] - median(image[mask]))) -- same MAD-to-sigma
    convention as astro_calibrate.py. 0.0 if image or mask is empty."""
    image = np.asarray(image, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if image.size == 0 or not mask.any():
        return 0.0
    values = image[mask]
    median = np.median(values)
    return float(_MAD_TO_SIGMA * np.median(np.abs(values - median)))


def noise_improvement(single_frame_noise, stacked_noise, n) -> NoiseImprovement:
    """measured_factor = single_frame_noise / stacked_noise (denom floored at
    _DENOM_EPS); ideal_factor = sqrt(n) (0.0 if n <= 0); ratio_to_ideal =
    measured/ideal (0.0 if ideal is 0). Never raises, never NaN/inf."""
    denom = float(stacked_noise) if stacked_noise > 0 else _DENOM_EPS
    measured_factor = float(single_frame_noise) / denom
    ideal_factor = float(np.sqrt(n)) if n > 0 else 0.0
    ratio_to_ideal = measured_factor / ideal_factor if ideal_factor > 0 else 0.0
    return NoiseImprovement(measured_factor, ideal_factor, ratio_to_ideal)


def false_rejection_rate(rejection_mask) -> RejectionStats:
    """rate = True count / total count over a boolean (n, ...) mask (True ==
    rejected); any_rejected_fraction = mean(mask.any(axis=0)). Input to the
    <= FALSE_REJECTION_RATE_MAX gate (docs/astro-quality-contract.md, gate 4). Both 0.0 if mask is empty."""
    mask = np.asarray(rejection_mask, dtype=bool)
    if mask.size == 0:
        return RejectionStats(0.0, 0.0)
    rate = float(mask.sum()) / float(mask.size)
    any_rejected_fraction = float(np.mean(mask.any(axis=0)))
    return RejectionStats(rate, any_rejected_fraction)


def stellar_color_ratios(rgb, star_xy, aperture_px) -> StellarColorRatios:
    """Per-star R/G, B/G aperture sums; ddof=0 spread over MEASURED stars only (`valid`
    True) so an unmeasured star can't inflate the spread. Empty arrays + 0.0 spreads if
    star_xy is empty; a star's ratio stays 0.0 and `valid` stays False (excluded from
    the spread) if its aperture is empty or its G sum is 0. Spread is 0.0 with fewer
    than 2 valid stars."""
    rgb = np.asarray(rgb, dtype=np.float64)
    star_xy = np.asarray(star_xy, dtype=np.float64).reshape(-1, 2)
    n = star_xy.shape[0]
    empty = np.zeros((0,), dtype=np.float64)
    if n == 0 or rgb.size == 0:
        return StellarColorRatios(empty, empty, 0.0, 0.0, np.zeros((0,), dtype=bool))
    h, w = rgb.shape[0], rgb.shape[1]
    yy, xx = np.mgrid[0:h, 0:w]
    radius_sq = float(aperture_px) * float(aperture_px)
    rg = np.zeros(n, dtype=np.float64)
    bg = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for i, (x, y) in enumerate(star_xy):
        aperture = (xx - x) ** 2 + (yy - y) ** 2 <= radius_sq
        if not aperture.any():
            continue
        r_sum = float(rgb[..., 0][aperture].sum())
        g_sum = float(rgb[..., 1][aperture].sum())
        b_sum = float(rgb[..., 2][aperture].sum())
        if g_sum == 0.0:
            continue
        rg[i] = r_sum / g_sum
        bg[i] = b_sum / g_sum
        valid[i] = True
    rg_spread = float(np.std(rg[valid])) if valid.sum() >= 2 else 0.0
    bg_spread = float(np.std(bg[valid])) if valid.sum() >= 2 else 0.0
    return StellarColorRatios(rg, bg, rg_spread, bg_spread, valid)


def coverage_stats(contributor_counts, floor) -> CoverageStats:
    """min/median of contributor_counts, plus fraction of pixels strictly below
    floor. (0, 0.0, 0.0) if contributor_counts is empty (nothing to summarise)."""
    counts = np.asarray(contributor_counts)
    if counts.size == 0:
        return CoverageStats(0, 0.0, 0.0)
    return CoverageStats(
        int(np.min(counts)), float(np.median(counts)), float(np.mean(counts < floor))
    )


def boundary_seam(image, boundary_mask) -> float:
    """abs(mean(image[boundary_mask]) - mean(image[~boundary_mask])) / sigma, sigma =
    min(background_noise(image, boundary_mask), background_noise(image, ~boundary_mask))
    -- each side's OWN sigma, never the combined-band sigma (saturates near 1.349
    regardless of step size). Caller restricts to a band straddling the seam. 0.0 if
    empty/all True/False; sigma floors at _DENOM_EPS."""
    image = np.asarray(image, dtype=np.float64)
    mask = np.asarray(boundary_mask, dtype=bool)
    if image.size == 0 or mask.size == 0 or not mask.any() or mask.all():
        return 0.0
    mean_a = float(np.mean(image[mask]))
    mean_b = float(np.mean(image[~mask]))
    sigma = min(background_noise(image, mask), background_noise(image, ~mask))
    denom = sigma if sigma > 0 else _DENOM_EPS
    return abs(mean_a - mean_b) / denom


# Star-shape gate: the deep product ships only the region that passes it (see
# astro_combine.py's SCOPED_FRAME_FRACTION comment for the one-time measurement this makes
# recomputable: +55.2% unpaired / +60.7% paired outer-third growth against this same boundary).
A3_RADIAL_GROWTH_BOUNDARY = 0.10  # max fractional stacked-FWHM growth per radial third.

RadialThirdShape = namedtuple("RadialThirdShape", "fwhm_px growth_fraction n passed")


def a3_by_radial_third(stacked_luma, frame_median_fwhm_px, mask=None):
    """Star-shape gate input: for each of three concentric radial thirds of
    ``stacked_luma`` (radius from the frame centre, normalised by the centre-to-corner
    distance; thirds at < 1/3, 1/3-2/3, >= 2/3, band names "center"/"middle"/"outer" --
    exactly astro_register.radial_inlier_residuals_px's convention, so the two are
    comparable side by side), the median stacked star FWHM, its fractional growth against
    ``frame_median_fwhm_px`` (the per-frame median FWHM that went into the stack), the star
    count that FWHM rests on, and pass/fail against ``A3_RADIAL_GROWTH_BOUNDARY``.

    Built on astro_triage's canonical star detector (``detect_stars``) and second-moment
    shape measurer (``_measure_star_shapes``/``_median_shape``/``_isotropic_fwhm``) rather
    than a second detector/measurer -- reaching into those underscore-prefixed names is a
    deliberate exception (see module docstring); a follow-up should promote them to public
    names in astro_triage.py.

    A band with no detected stars reports ``n=0`` and ``fwhm_px``/``growth_fraction``/
    ``passed`` all ``None`` -- an honest absence, never a fabricated zero or a silent pass
    -- same convention as ``radial_inlier_residuals_px``."""
    luma = np.asarray(stacked_luma, dtype=np.float32)
    mask_arr = np.asarray(mask, dtype=bool) if mask is not None else None
    stars = detect_stars(luma, mask=mask_arr)
    valid = luma[mask_arr] if mask_arr is not None else luma
    background, noise_sigma = _robust_stats(valid)

    height, width = luma.shape[:2]
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    max_radius = np.linalg.norm(center)
    if stars.shape[0]:
        radius = np.linalg.norm(stars[:, :2] - center, axis=1) / max(max_radius, 1.0)
    else:
        radius = np.zeros((0,), dtype=np.float64)
    bins = {
        "center": radius < 1 / 3,
        "middle": (radius >= 1 / 3) & (radius < 2 / 3),
        "outer": radius >= 2 / 3,
    }
    result = {}
    for name, selected in bins.items():
        band_stars = stars[selected]
        shapes = _measure_star_shapes(luma, band_stars, background, noise_sigma)
        if not shapes:
            result[name] = RadialThirdShape(fwhm_px=None, growth_fraction=None, n=0, passed=None)
            continue
        median_shape = _median_shape([s for _, _, s in shapes])
        fwhm_px = _isotropic_fwhm(median_shape)
        growth_fraction = (
            (fwhm_px - float(frame_median_fwhm_px)) / float(frame_median_fwhm_px)
            if frame_median_fwhm_px
            else None
        )
        passed = (
            growth_fraction <= A3_RADIAL_GROWTH_BOUNDARY if growth_fraction is not None else None
        )
        result[name] = RadialThirdShape(
            fwhm_px=fwhm_px, growth_fraction=growth_fraction, n=len(shapes), passed=passed
        )
    return result
