"""Per-frame astro quality metrics: star count, FWHM, PSF shape, streak detection.

Reads one float32 linear luminance frame (0..1, from ``astro_calibrate``); ``triage``
folds a run's ``FrameMetrics`` into cloud/dew flags relative to the clearest (cloud) and
opening (dew) frames of the group it is handed -- report flags, never a refusal.
Pure and deterministic; no I/O beyond the ``path`` it carries. Every constant below is
PROVISIONAL — tuned against synthetic star fields pending a reference night card, same discipline as
``bracket.py``. ``classify_streaks`` labels a run's streaks satellite/aircraft/artifact,
falling through to unknown -- never meteor -- absent positive evidence;
``classify_shape`` reads one frame's PSF shape into a likely-cause label --
never touches ``usable``/cloud/dew. ``classify_exposure`` is the one classification here
that DOES refuse a frame: a cap-on frame (darks shot at the end of a night and left on
the card) is ``usable=False`` and defines neither baseline, because its black-floor
background would otherwise be read as the run's clearest sky.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

MAD_TO_SIGMA = 1.4826  # MAD -> Gaussian-sigma scale factor (standard, not a tunable).
GAUSSIAN_SIGMA_TO_FWHM = 2.3548200450309493  # 2*sqrt(2*ln 2): sigma -> FWHM (standard).

# Star detection (PROVISIONAL, see module docstring).
STAR_DETECT_SIGMA = 6.0  # MAD-sigma above background to seed a star.
STAR_SMOOTH_SIGMA_PX = 1.0  # Pre-blur: stabilizes the centroid vs. shot noise.
STAR_MAXIMA_WINDOW_PX = 5  # Odd local-maximum neighbourhood, ~PSF footprint.
STAR_CENTROID_WINDOW_PX = 9  # Odd centroid window: PSF wings, no neighbour bleed.
FWHM_STAR_SAMPLE_COUNT = 50  # Median FWHM/shape over the brightest N stars.
FWHM_WINDOW_PX = 11  # Odd second-moment window, starting point for the growth below.
MIN_USABLE_STAR_COUNT = 5  # Below this a frame has no usable signal alone.

# Adaptive shape metrics (PROVISIONAL): FWHM_WINDOW_PX ceilinged the reference
# frames at a few pixels.
FWHM_WINDOW_GROWTH_FACTOR = 2.0  # window roughly doubles (odd-clamped) per growth step.
FWHM_WINDOW_MAX_PX = 61  # hard growth bound: a compute-cost ceiling, not a physical limit.
FWHM_WINDOW_GROWTH_SIGNAL_RATIO = 3.0  # PROVISIONAL: a ring's flux vs. its noise pedestal.
FWHM_NOISE_FLOOR_SIGMA = 2.0  # px below background + this * MAD-sigma don't count as signal.
CENTER_REGION_FRACTION = 0.3  # central box side, as a fraction of the frame's shorter side.
CORNER_REGION_FRACTION = 0.25  # each corner box side, as a fraction of the frame's shorter side.

# Shape classification (PROVISIONAL): a label only, never rejects a frame.
TRAILING_ELLIPTICITY_MIN = 0.3  # major/minor asymmetry read as directional trailing.
DEFOCUS_ROUNDNESS_MAX_ELLIPTICITY = 0.2  # below this a star reads "round" for these purposes.
DEFOCUS_FWHM_MIN_PX = 8.0  # round AND above this isotropic size reads "soft" (out of focus).
CORNER_ABERRATION_SIZE_RATIO = 1.5  # corner median size over this * center -> field aberration.

# Streak detection (PROVISIONAL).
STREAK_MEDIAN_KSIZE_PX = 31  # Odd; erases stars/streaks, keeps sky glow.
STREAK_SIGMA_K = 7.0  # MAD-sigma for a high-pass pixel to count as streak.
# rescale-window half-width, in raw-luma MAD-sigma, before the uint8 median
# blur below (cv2.medianBlur has no float/16-bit path at this kernel size). A fixed
# [0, 1] -> uint8 scale gives a dark sky (~0.02 linear) one code per ~0.0039 -- comparable
# to a subtle sky-glow gradient across a 31px window, so the median blur's per-code
# banding leaks into the recovered noise floor (measured ~5x inflation on a synthetic
# dark-sky gradient test), which can raise the detection threshold enough to hide a real
# faint trail. Anchoring the rescale window on the frame's own robust spread keeps the
# quantization step a small fraction of the real dynamic range at any exposure.
STREAK_RESCALE_SIGMA_SPAN = 20.0
STREAK_RESCALE_MIN_SPAN = 1e-3  # floor on the rescale half-width: a near-zero-noise
# frame can't collapse the window to zero width (divide-by-zero guard).
STREAK_MORPH_CLOSE_PX = 3  # odd; bridges a 1px threshold gap from noise along
# a real trail -- small enough that two separate nearby trails never bridge together.
STREAK_MIN_ELONGATION_RATIO = 3.0  # length/width floor -- a component this squat
# is a blob of salt, not a trail.
STREAK_MIN_COMPONENT_AREA_PX = 5  # component pixel-count floor -- a handful of
# touching pixels is noise, not a trail.
STREAK_MIN_LENGTH_PX_FALLBACK = 20.0  # No-EXIF fallback.
# px/mm-focal/s from sidereal rate at a representative pixel scale; scales up longer exposures/lenses.
STREAK_TRAIL_PX_PER_MM_PER_S = 0.02

# Triage flags (PROVISIONAL), relative to the run's own numbers.
CLOUD_STAR_COUNT_DROP_RATIO = 0.4  # star_count under this * clear baseline -> cloud.
CLOUD_BACKGROUND_JUMP_RATIO = 1.5  # background over this * clear baseline -> cloud.
CLOUD_BASELINE_FRAME_COUNT = 3  # Clearest frames defining the cloud baseline.
DEW_FWHM_INFLATION_RATIO = 1.5  # fwhm_px over this * run-start baseline -> dew.
FWHM_BASELINE_FRAME_COUNT = 3  # Opening frames defining the dew baseline.

# Cap-on classification (PROVISIONAL): see ``classify_exposure`` for what each encodes.
# Both tests are ratios against the frame's OWN pixel distribution, so neither carries a
# black level, an ISO, an exposure time or a sensor in it.
EXPOSURE_SPREAD_PERCENTILE = 84.13  # median..here = 1 sigma of a Gaussian; the UPPER half,
# which black-clipping (astro_calibrate's postprocess subtracts the pedestal and clips
# negatives at 0) leaves intact, unlike the MAD -- over half a capped frame's pixels can
# land on the clip floor, which collapses the MAD toward zero and makes any ratio to it
# explode. Measured on a simulated capped frame: MAD-sigma 1e-8 against a true 1e-3.
EXPOSURE_BRIGHT_PERCENTILE = 99.9  # the brightest 0.1% of pixels: the stars, if any.
CAP_ON_STRUCTURE_SPREAD_MAX = 5.0  # (p99.9 - median) / spread at or below this = no spatial
# structure: a Gaussian's own 99.9th percentile is 3.09 sigma, so a frame whose brightest
# 0.1% is only a few sigma up holds nothing but noise. Simulated cap-on frames measure
# 3.0-3.6 and star fields 90-205, so 5.0 sits in a wide empty gap rather than on a slope.
CAP_ON_LEVEL_SPREAD_MAX = 2.0  # median / spread at or below this = the frame's level is
# still at the sensor's black floor: its noise distribution straddles the calibrated zero
# point instead of riding on sky. A half-clipped Gaussian reads 0.67 whatever its noise
# amplitude (measured 0.0-1.6 across simulated capped frames); overcast sky -- flat, but
# lit -- reads 10.0, which is what this test is here to tell apart from a cap.
EXPOSURE_STATS_MAX_SAMPLES = 2_000_000  # stride-subsample cap for the two percentiles
# above: a full-resolution percentile pass costs several times all of triage with the
# streak scan off. 2e6 samples still leave ~2000 beyond the 99.9th.

# Streak classification (PROVISIONAL): satellite / aircraft / artifact / unknown.
SATELLITE_COLLINEAR_COS_MIN = 0.97  # cos(angle) floor between frame-to-frame streak directions.
SATELLITE_MAX_GAP_RATIO = 3.0  # max endpoint gap, as a multiple of the shorter streak's length.
SATELLITE_PROGRESSION_COS_MIN = 0.9  # cos(angle) floor: centroid displacement vs. own direction.
SATELLITE_MAX_CADENCE_S = 30.0  # max real gap (s) chaining trusts; PROVISIONAL.
AIRCRAFT_MIN_STROBE_COUNT = 3  # dashes needed before spacing regularity means anything.
AIRCRAFT_COLLINEAR_COS_MIN = 0.97  # cos(angle) floor between within-frame streak directions.
AIRCRAFT_LINE_OFFSET_PX = 6.0  # max perpendicular offset from a common line to group as strobes.
AIRCRAFT_GAP_REGULARITY_RATIO = 0.35  # gap stdev/mean ceiling to call spacing "regular".
ARTIFACT_POSITION_TOLERANCE_PX = 3.0  # px of slack for two streaks' LINE SEGMENTS
# (not bounding boxes) to count as the same fixed photosite defect footprint.
ARTIFACT_MIN_OCCURRENCES = 3  # PROVISIONAL floor below the reference 5-frame recurrence.


class StreakDetectionNotRunError(ValueError):
    """Streak data was asked for on a frame measured with ``detect_meteors=False``."""


@dataclass(frozen=True)
class StarShape:  # one star's (or a run's median) second-moment PSF shape
    fwhm_major_px: float
    fwhm_minor_px: float
    ellipticity: float  # 0 = round, -> 1 = a line.
    orientation_deg: float  # major-axis angle, degrees, mod 180 (an axis has no direction).
    hfr_px: float  # half-flux radius.
    saturated: bool  # still window-limited at FWHM_WINDOW_MAX_PX, not converged.


@dataclass(frozen=True)
class RegionShapeSummary:  # median StarShape: frame center vs. the four corners
    center: StarShape | None
    corners: StarShape | None


@dataclass(frozen=True)
class StreakComponent:  # one detected streak's full geometry (detect_streak_components)
    x1: float
    y1: float
    x2: float
    y2: float
    length_px: float  # pixel extent along the fitted major axis.
    width_px: float  # cross-trail extent: sqrt(12 * minor-axis variance), a uniform strip's width.
    mean_brightness: float  # mean high-pass value over the component's pixels.
    peak_brightness: float  # brightest high-pass pixel in the component.


@dataclass(frozen=True)
class FrameMetrics:  # one frame's triage numbers
    path: Path
    star_count: int
    background_median: float
    fwhm_px: float
    # None = the streak scan was skipped (detect_meteors=False), NOT 'searched and found
    # none' -- [] is that positive claim. classify_streaks refuses None rather than
    # silently reporting a meteor-free run it never looked for.
    streaks: list[tuple[float, float, float, float]] | None
    usable: bool
    shape: StarShape | None = None  # median PSF shape over the sampled stars.
    region_shape: RegionShapeSummary | None = None  # center-vs-corner split.
    streak_components: tuple[StreakComponent, ...] | None = ()  # full per-streak geometry.
    captured_at: datetime | None = None  # missing -> untrustworthy, declines chaining.
    exposure_class: str = "light"  # classify_exposure: light / cap_on / indeterminate.


@dataclass(frozen=True)
class TriageReport:  # a run's FrameMetrics + flags relative to the run itself
    frames: tuple[FrameMetrics, ...]
    cloud_frames: tuple[Path, ...]
    dew_frames: tuple[Path, ...]
    cap_on_frames: tuple[Path, ...] = ()  # lens cap left on -- never a light, never a baseline.
    indeterminate_frames: tuple[Path, ...] = ()  # neither population; reported, not acted on.


@dataclass(frozen=True)
class StreakClassification:  # one streak: its frame, geometry, and discriminator label
    frame_index: int
    path: Path
    streak: tuple[float, float, float, float]
    label: str  # "satellite" | "aircraft" | "artifact" | "unknown" -- see classify_streaks
    reason: str | None = None  # run-level decline reason, set on every label.


def _robust_stats(values):
    """(median, MAD-derived sigma) of ``values``, flattened."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    background = float(np.median(flat))
    sigma = float(np.median(np.abs(flat - background))) * MAD_TO_SIGMA
    return background, sigma


def detect_stars(luma, sigma=STAR_DETECT_SIGMA, mask=None):
    """(N,3) float32 (x, y, flux): local maxima above background + sigma*MAD-sigma,
    sub-pixel centroided. Canonical detector (column order/dtype contractual).
    ``mask`` (bool, True=usable) excludes pixels from both stats and detection."""
    luma = np.asarray(luma, dtype=np.float32)
    smoothed = cv2.GaussianBlur(luma, (0, 0), STAR_SMOOTH_SIGMA_PX)
    valid = smoothed[mask] if mask is not None else smoothed
    background, noise_sigma = _robust_stats(valid)
    threshold = background + sigma * noise_sigma
    win = STAR_MAXIMA_WINDOW_PX
    dilated = cv2.dilate(smoothed, np.ones((win, win), dtype=np.uint8))
    peaks = (smoothed >= dilated) & (smoothed > threshold)
    if mask is not None:
        peaks &= mask
    ys, xs = np.nonzero(peaks)
    half = STAR_CENTROID_WINDOW_PX // 2
    h, w = luma.shape
    stars = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0, y1 = max(0, y - half), min(h, y + half + 1)
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        patch = np.clip(luma[y0:y1, x0:x1] - background, 0.0, None)
        total = float(patch.sum())
        if total <= 0.0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        stars.append((float((xx * patch).sum() / total), float((yy * patch).sum() / total), total))
    if not stars:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(stars, dtype=np.float32)


def _second_moments(luma, background, noise_sigma, cx, cy, window):
    """(ixx, iyy, ixy, total, patch, px_cx, px_cy, x0, y0) of flux above a noise floor
    (``FWHM_NOISE_FLOOR_SIGMA`` * ``noise_sigma``) in a ``window``-px box; None if empty."""
    h, w = luma.shape
    half = window // 2
    xi, yi = int(round(cx)), int(round(cy))
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    raw = luma[y0:y1, x0:x1] - background
    patch = np.where(raw > FWHM_NOISE_FLOOR_SIGMA * noise_sigma, raw, 0.0)
    total = float(patch.sum())
    if total <= 0.0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    px_cx, px_cy = (xx * patch).sum() / total, (yy * patch).sum() / total
    dx, dy = xx - px_cx, yy - px_cy
    ixx = float((patch * dx * dx).sum() / total)
    iyy = float((patch * dy * dy).sum() / total)
    ixy = float((patch * dx * dy).sum() / total)
    return ixx, iyy, ixy, total, patch, float(px_cx), float(px_cy), x0, y0


def _grow_second_moments(luma, background, noise_sigma, cx, cy):
    """``_second_moments`` grown from ``FWHM_WINDOW_PX`` while each ring beats
    ``FWHM_WINDOW_GROWTH_SIGNAL_RATIO`` times its noise-only expectation over the added
    area (a relative TOTAL-flux test never beats a faint star's own small total).
    Returns (moments, saturated); moments is None only if there's no flux at the start."""
    window = FWHM_WINDOW_PX
    moments = _second_moments(luma, background, noise_sigma, cx, cy, window)
    if moments is None:
        return None, False
    prev_total, prev_px = moments[3], moments[4].size
    noise_px_flux = noise_sigma * np.exp(-(FWHM_NOISE_FLOOR_SIGMA**2) / 2.0) / np.sqrt(2.0 * np.pi)
    while window < FWHM_WINDOW_MAX_PX:
        window = min(FWHM_WINDOW_MAX_PX, int(window * FWHM_WINDOW_GROWTH_FACTOR) | 1)
        grown = _second_moments(luma, background, noise_sigma, cx, cy, window)
        if grown is None:
            break
        total, added_px = grown[3], grown[4].size - prev_px
        if total - prev_total < FWHM_WINDOW_GROWTH_SIGNAL_RATIO * noise_px_flux * added_px:
            return moments, False
        moments, prev_total, prev_px = grown, total, grown[4].size
    return moments, window >= FWHM_WINDOW_MAX_PX


def _half_flux_radius(patch, px_cx, px_cy, x0, y0):
    """Radius (px) enclosing half of ``patch``'s flux around (px_cx, px_cy)."""
    yy, xx = np.mgrid[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]]
    r = np.sqrt((xx - px_cx) ** 2 + (yy - px_cy) ** 2).ravel()
    flux = patch.ravel()
    total = float(flux.sum())
    if total <= 0.0:
        return 0.0
    order = np.argsort(r)
    cum = np.cumsum(flux[order])
    idx = min(int(np.searchsorted(cum, total / 2.0)), r.size - 1)
    return float(r[order][idx])


def _tensor_shape(ixx, iyy, ixy, saturated, hfr_px):
    """``StarShape`` from a second-moment tensor: eigen-decomposed for major/minor
    axis FWHM, ellipticity, and orientation (degrees, mod 180)."""
    mean = (ixx + iyy) / 2.0
    spread = np.sqrt(max(((ixx - iyy) / 2.0) ** 2 + ixy**2, 0.0))
    major_var, minor_var = max(mean + spread, 0.0), max(mean - spread, 0.0)
    fwhm_major = np.sqrt(major_var) * GAUSSIAN_SIGMA_TO_FWHM
    fwhm_minor = np.sqrt(minor_var) * GAUSSIAN_SIGMA_TO_FWHM
    ellipticity = 1.0 - (fwhm_minor / fwhm_major) if fwhm_major > 0.0 else 0.0
    orientation = np.degrees(0.5 * np.arctan2(2.0 * ixy, ixx - iyy)) % 180.0
    return StarShape(
        fwhm_major_px=float(fwhm_major),
        fwhm_minor_px=float(fwhm_minor),
        ellipticity=float(ellipticity),
        orientation_deg=float(orientation),
        hfr_px=float(hfr_px),
        saturated=bool(saturated),
    )


def _isotropic_fwhm(shape):
    """Single isotropic FWHM (px) equivalent to a ``StarShape``'s trace."""
    return float(np.sqrt((shape.fwhm_major_px**2 + shape.fwhm_minor_px**2) / 2.0))


def _measure_star_shapes(luma, stars, background, noise_sigma):
    """[(x, y, StarShape)] for the brightest ``FWHM_STAR_SAMPLE_COUNT`` stars with
    measurable flux, via the adaptive-window second-moment tensor + half-flux radius."""
    if stars.shape[0] == 0:
        return []
    top = stars[np.argsort(stars[:, 2])[::-1][:FWHM_STAR_SAMPLE_COUNT]]
    out = []
    for x, y, _ in top:
        cx, cy = float(x), float(y)
        moments, saturated = _grow_second_moments(luma, background, noise_sigma, cx, cy)
        if moments is None:
            continue
        ixx, iyy, ixy, _total, patch, px_cx, px_cy, x0, y0 = moments
        hfr = _half_flux_radius(patch, px_cx, px_cy, x0, y0)
        out.append((cx, cy, _tensor_shape(ixx, iyy, ixy, saturated, hfr)))
    return out


def _median_shape(shapes):
    """Component-wise median ``StarShape`` over ``shapes`` (``saturated`` by
    majority vote); None if ``shapes`` is empty."""
    if not shapes:
        return None
    return StarShape(
        fwhm_major_px=float(np.median([s.fwhm_major_px for s in shapes])),
        fwhm_minor_px=float(np.median([s.fwhm_minor_px for s in shapes])),
        ellipticity=float(np.median([s.ellipticity for s in shapes])),
        orientation_deg=float(np.median([s.orientation_deg for s in shapes])),
        hfr_px=float(np.median([s.hfr_px for s in shapes])),
        saturated=bool(np.median([1.0 if s.saturated else 0.0 for s in shapes]) >= 0.5),
    )


def _region_shapes(height, width, star_shapes):
    """Median shape of stars in the frame's central box vs. its four corner boxes
    (``CENTER_REGION_FRACTION`` / ``CORNER_REGION_FRACTION`` of the shorter side)."""
    side = min(height, width)
    cx, cy = width / 2.0, height / 2.0
    center_half = side * CENTER_REGION_FRACTION / 2.0
    corner_side = side * CORNER_REGION_FRACTION
    center, corners = [], []
    for x, y, shape in star_shapes:
        if abs(x - cx) <= center_half and abs(y - cy) <= center_half:
            center.append(shape)
            continue
        near_x = x <= corner_side or x >= width - corner_side
        near_y = y <= corner_side or y >= height - corner_side
        if near_x and near_y:
            corners.append(shape)
    return RegionShapeSummary(center=_median_shape(center), corners=_median_shape(corners))


def classify_shape(metrics):
    """One frame's likely-cause label (likely_trailing / likely_defocus / corner_aberration /
    indeterminate) -- never raises or touches ``usable``/cloud/dew (PROVISIONAL thresholds)."""
    shape = metrics.shape
    if shape is None:
        return "indeterminate"
    region = metrics.region_shape
    if region is not None and region.center is not None and region.corners is not None:
        center_size = _isotropic_fwhm(region.center)
        corner_size = _isotropic_fwhm(region.corners)
        if (
            region.center.ellipticity < DEFOCUS_ROUNDNESS_MAX_ELLIPTICITY
            and center_size > 0.0
            and corner_size > CORNER_ABERRATION_SIZE_RATIO * center_size
        ):
            return "corner_aberration"
    if shape.ellipticity >= TRAILING_ELLIPTICITY_MIN:
        return "likely_trailing"
    if (
        shape.ellipticity < DEFOCUS_ROUNDNESS_MAX_ELLIPTICITY
        and _isotropic_fwhm(shape) > DEFOCUS_FWHM_MIN_PX
    ):
        return "likely_defocus"
    return "indeterminate"


def _robust_rescale_u8(scaled, valid):
    """(uint8 array, lo, hi): ``scaled`` rescaled onto ``valid``'s own robust
    [lo, hi] window (``STREAK_RESCALE_SIGMA_SPAN`` sigma around its median) so the
    8-bit step ``cv2.medianBlur`` requires at this kernel size stays a small
    fraction of the frame's real dynamic range instead of a blind [0, 1] cast."""
    level, sigma = _robust_stats(valid)
    span = max(sigma * STREAK_RESCALE_SIGMA_SPAN, STREAK_RESCALE_MIN_SPAN)
    lo, hi = level - span, level + span
    u8 = np.clip((scaled - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return u8.astype(np.uint8), lo, hi


def _streak_component_geometry(ys, xs, highpass):
    """``StreakComponent`` for one connected component's pixel coordinates, via a
    closed-form 2x2 eigen-fit (the same construction as ``_tensor_shape``): major-axis
    direction and its pixel extent (length + endpoints), minor-axis spread read as a
    uniform strip's width (variance = width**2 / 12). None if the component has no
    elongation to fit (a single point)."""
    xf, yf = xs.astype(np.float64), ys.astype(np.float64)
    cx, cy = float(xf.mean()), float(yf.mean())
    dx, dy = xf - cx, yf - cy
    ixx, iyy, ixy = float((dx * dx).mean()), float((dy * dy).mean()), float((dx * dy).mean())
    mean_v = (ixx + iyy) / 2.0
    spread = np.sqrt(max(((ixx - iyy) / 2.0) ** 2 + ixy**2, 0.0))
    major_var = max(mean_v + spread, 0.0)
    if major_var <= 0.0:
        return None
    minor_var = max(mean_v - spread, 0.0)
    angle = 0.5 * np.arctan2(2.0 * ixy, ixx - iyy) if spread > 0.0 else 0.0
    direction = np.array([np.cos(angle), np.sin(angle)])
    t = dx * direction[0] + dy * direction[1]
    length = float(t.max() - t.min())
    width = float(np.sqrt(12.0 * minor_var)) or 1.0  # a 1px-wide line has no measurable spread.
    p1 = np.array([cx, cy]) + direction * float(t.min())
    p2 = np.array([cx, cy]) + direction * float(t.max())
    pixels = highpass[ys, xs]
    return StreakComponent(
        x1=float(p1[0]),
        y1=float(p1[1]),
        x2=float(p2[0]),
        y2=float(p2[1]),
        length_px=length,
        width_px=width,
        mean_brightness=float(pixels.mean()),
        peak_brightness=float(pixels.max()),
    )


def detect_streak_components(luma, min_length_px, mask=None):
    """[StreakComponent]: connected foreground blobs of a robust high-pass threshold,
    morphologically closed and elongation/area-gated so ONE physical trail yields ONE
    detection (raw ``HoughLinesP`` strung non-contiguous hot pixels into 10-25
    near-duplicate segments per real trail; scattered salt pixels that never touch can
    never share a component -- rejected by construction, not a late filter). The
    background estimate runs on ``luma``'s own robustly rescaled range rather than a
    blind 8-bit cast of [0, 1]. ``mask`` (bool, True=usable) excludes pixels
    from both stats and detection, the same contract as ``detect_stars``."""
    luma = np.asarray(luma, dtype=np.float32)
    scaled = np.clip(luma, 0.0, 1.0)
    valid = scaled[mask] if mask is not None else scaled
    u8, lo, hi = _robust_rescale_u8(scaled, valid)
    background_u8 = cv2.medianBlur(u8, STREAK_MEDIAN_KSIZE_PX)
    background_rescaled = lo + background_u8.astype(np.float32) / 255.0 * (hi - lo)
    # Repair: a bright region wider than the median kernel (a lit foreground, a
    # horizon-glow band) saturates every pixel in its window to u8=255, so the
    # rescaled estimate above clamps to the window's ceiling `hi` instead of the
    # true bright level -- reading as a large spurious high-pass "streak". Where
    # the rescaled median is saturated, fall back to a full-[0,1]-range 8-bit
    # median instead: coarser quantization at dark-sky levels (why it isn't used
    # everywhere -- see STREAK_RESCALE_SIGMA_SPAN above), but accurate once
    # pixels are actually bright.
    background_full = (
        cv2.medianBlur((scaled * 255.0).astype(np.uint8), STREAK_MEDIAN_KSIZE_PX).astype(np.float32)
        / 255.0
    )
    background = np.where(background_u8 >= 255, background_full, background_rescaled)
    highpass = scaled - background
    highpass_valid = highpass[mask] if mask is not None else highpass
    level, noise_sigma = _robust_stats(highpass_valid)
    binary = ((highpass > level + STREAK_SIGMA_K * noise_sigma) * 255).astype(np.uint8)
    if mask is not None:
        binary[~mask] = 0
    close_kernel = np.ones((STREAK_MORPH_CLOSE_PX, STREAK_MORPH_CLOSE_PX), dtype=np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    # connectedComponentsWithStats (not connectedComponents): gives each label's
    # bounding box up front, so extracting its pixels is O(box area) instead of a
    # full H*W rescan per label -- O(n_labels * H * W) took minutes on a full-resolution
    # frame with thousands of components (invisible at this suite's 200-300px
    # fixtures).
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    components = []
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] < STREAK_MIN_COMPONENT_AREA_PX:
            continue
        x0 = stats[label, cv2.CC_STAT_LEFT]
        y0 = stats[label, cv2.CC_STAT_TOP]
        box_w = stats[label, cv2.CC_STAT_WIDTH]
        box_h = stats[label, cv2.CC_STAT_HEIGHT]
        sub_ys, sub_xs = np.nonzero(labels[y0 : y0 + box_h, x0 : x0 + box_w] == label)
        ys, xs = sub_ys + y0, sub_xs + x0
        component = _streak_component_geometry(ys, xs, highpass)
        if component is None:
            continue
        min_span = max(min_length_px, STREAK_MIN_ELONGATION_RATIO * component.width_px)
        if component.length_px < min_span:
            continue
        components.append(component)
    return components


def detect_streaks(luma, min_length_px, mask=None):
    """List of (x1,y1,x2,y2) float tuples -- ``detect_streak_components``'s endpoints
    only, the contract every caller predating streak components still expects."""
    return [
        (c.x1, c.y1, c.x2, c.y2) for c in detect_streak_components(luma, min_length_px, mask=mask)
    ]


def _exposure_stats(values):
    """(median, spread, bright_level) of ``values`` for ``classify_exposure``: the
    median, the ``EXPOSURE_SPREAD_PERCENTILE`` upper half-width, and the
    ``EXPOSURE_BRIGHT_PERCENTILE`` level. Stride-subsampled to
    ``EXPOSURE_STATS_MAX_SAMPLES`` (see that constant for the measured cost)."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size > EXPOSURE_STATS_MAX_SAMPLES:
        flat = flat[:: int(np.ceil(flat.size / EXPOSURE_STATS_MAX_SAMPLES))]
    median, spread_level, bright = np.percentile(
        flat, [50.0, EXPOSURE_SPREAD_PERCENTILE, EXPOSURE_BRIGHT_PERCENTILE]
    )
    return float(median), float(spread_level - median), float(bright)


def classify_exposure(median, spread, bright_level):
    """One frame's exposure class -- "light" / "cap_on" / "indeterminate" -- from the
    three numbers ``_exposure_stats`` measures on the frame itself.

    A cap-on frame (darks shot at the end of the night and left on the
    card) is the one failure this exists for: it has the LOWEST background physically
    possible, so ``_cloud_baseline`` takes it as the run's "clearest sky", every genuine
    light then reads as a background jump, and ``astro_select`` REJECTS them -- silently,
    and forward across chunks through ``update_run_cloud_baseline``.

    Two independent tests, each a ratio against the frame's own spread, so neither
    carries a black level or a sensor in it:

    - STRUCTURE, ``(bright_level - median) / spread``: light fell on the sensor iff
      something in the frame stands above its own noise. A capped frame is read noise on
      a pedestal -- its brightest 0.1% of pixels sit at a Gaussian's 3.09 sigma and no
      further. A star field puts its brightest 0.1% tens to hundreds of sigma up.
    - LEVEL, ``median / spread``: a capped frame sits at the sensor's black floor, which
      ``astro_calibrate``'s postprocess has already moved to zero, so its noise still
      straddles the zero point. Sky -- even thick overcast -- rides well above it.

    Both on the cap side -> "cap_on". Both on the light side -> "light". Anything else
    is "indeterminate" and is REPORTED, never guessed into a population:

    - structure but at the floor is the cap going on MID-exposure (part sky,
      part floor). It is not a dark -- it holds real stars -- and it is not a
      clean light either.
    - flat but lit is overcast, fog, or a lens actually pointed at something uniform.

    ``spread <= 0`` (a frame with no measurable spread at all -- every pixel identical)
    reads as no structure, and as at-the-floor only if the level is at zero too.
    """
    if spread <= 0.0:
        at_floor, structured = median <= 0.0, False
    else:
        at_floor = median / spread <= CAP_ON_LEVEL_SPREAD_MAX
        structured = (bright_level - median) / spread > CAP_ON_STRUCTURE_SPREAD_MAX
    if at_floor and not structured:
        return "cap_on"
    if structured and not at_floor:
        return "light"
    return "indeterminate"


def measure_frame(
    path,
    luma,
    mask=None,
    exposure_s=None,
    focal_mm=None,
    captured_at=None,
    detect_meteors=True,
):
    """One frame's ``FrameMetrics``. Streak length threshold derives from
    ``exposure_s``/``focal_mm`` when given, else the fallback.
    ``captured_at`` threads straight through to ``FrameMetrics.captured_at``.

    ``detect_meteors=False`` skips the streak scan, which dominated the cost of this
    function on a star-rich frame (over 90% of triage), on every frame of a run
    whose CLI passes ``stack_run`` no meteor policy at all and therefore cannot
    reinsert anything. Skipping records ``streaks``/``streak_components`` as ``None``,
    never ``[]``: an empty list is the positive claim "searched, found none", and
    ``classify_streaks`` refuses metrics that never looked rather than silently
    reporting no meteors."""
    luma = np.asarray(luma, dtype=np.float32)
    stars = detect_stars(luma, mask=mask)
    valid = luma[mask] if mask is not None else luma
    background, noise_sigma = _robust_stats(valid)
    star_shapes = _measure_star_shapes(luma, stars, background, noise_sigma)
    fwhm = float(np.median([_isotropic_fwhm(s) for _, _, s in star_shapes])) if star_shapes else 0.0
    shape = _median_shape([s for _, _, s in star_shapes])
    region_shape = _region_shapes(luma.shape[0], luma.shape[1], star_shapes)
    if exposure_s and focal_mm:
        min_length = max(
            STREAK_TRAIL_PX_PER_MM_PER_S * focal_mm * exposure_s, STREAK_MIN_LENGTH_PX_FALLBACK
        )
    else:
        min_length = STREAK_MIN_LENGTH_PX_FALLBACK
    star_count = int(stars.shape[0])
    exposure_class = classify_exposure(*_exposure_stats(valid))
    streak_components = (
        detect_streak_components(luma, min_length, mask=mask) if detect_meteors else None
    )
    return FrameMetrics(
        path=Path(path),
        star_count=star_count,
        background_median=background,
        fwhm_px=fwhm,
        streaks=(
            None
            if streak_components is None
            else [(c.x1, c.y1, c.x2, c.y2) for c in streak_components]
        ),
        usable=star_count >= MIN_USABLE_STAR_COUNT and exposure_class != "cap_on",
        shape=shape,
        region_shape=region_shape,
        streak_components=(None if streak_components is None else tuple(streak_components)),
        captured_at=captured_at,
        exposure_class=exposure_class,
    )


def _cloud_baseline(frames):
    """(star, background) baseline: the CLEAREST condition this group actually shows --
    the median of its ``CLOUD_BASELINE_FRAME_COUNT`` highest star counts and, taken
    independently, of its same-many lowest backgrounds.

    Not the group's own median: a night that clouds over and STAYS over drags
    that median down with it, so nothing reads as anomalous. On a night that clouds
    over, the star count collapses to a small fraction of its clear-sky level while
    the background rises only slightly -- the star half catches it, the background
    half never fires that low. Not the opening N frames
    either (the ``FWHM_BASELINE_FRAME_COUNT`` shape dew uses): a run whose cloud is
    already there at frame 0 and lifts later would take the cloud as its baseline.
    3 is the smallest window whose median ignores one anomalous frame, and it needs 2
    clear frames anywhere in the group rather than 2 at the start.

    A group with no clear frame at all -- an entirely overcast chunk -- has nothing to
    be judged against and is NOT flagged; that needs a run-level baseline threaded
    across chunks, which this function never sees (``prepare_frames`` triages one chunk,
    ``astro_sequence.STACK_LEN_DEFAULT`` = 20 frames).

    Baseline >= the group median for stars and <= it for background by construction, so
    every frame the old group-median rule flagged is still flagged: this widens the test,
    it does not move it.

    Frames ``classify_exposure`` reads as ``cap_on`` are dropped first: a capped frame has
    the lowest background physically possible, so "lowest background" would hand the
    baseline -- and, through ``update_run_cloud_baseline``, the rest of the night -- to a
    frame that never saw sky. An ``indeterminate`` frame stays in: the median-of-3 above
    is already built to ignore one outlier, and dropping frames on a maybe would shrink
    the baseline on exactly the nights it matters most.
    """
    frames = [f for f in frames if f.exposure_class != "cap_on"]
    if not frames:
        return 0.0, 0.0
    n = min(len(frames), CLOUD_BASELINE_FRAME_COUNT)
    stars = sorted((f.star_count for f in frames), reverse=True)[:n]
    backgrounds = sorted(f.background_median for f in frames)[:n]
    return float(np.median(stars)), float(np.median(backgrounds))


class RunCloudBaseline(NamedTuple):
    """A run-level cloud baseline plus the provenance that decides how it merges."""

    star: float
    background: float
    from_focused: bool  # False = derived from defocused frames, held provisionally.


def update_run_cloud_baseline(existing, metrics):
    """Fold one chunk's ``FrameMetrics`` into a RUN-level cloud baseline: the
    clearest (star, background) condition seen anywhere in the run SO FAR, carried
    forward across chunks (``existing`` is ``None`` before the first contribution,
    else a prior return of this function) -- closes the residual named in
    ``_cloud_baseline``'s docstring without a second decode pass.

    Frames ``classify_shape`` reads as ``likely_defocus`` are PREFERRED for a chunk's
    contribution: star count alone conflates defocus with cloud, since a defocused
    frame's noise blobs count as "stars" the same as real ones.

    But a chunk with NO in-focus frame falls back to its own frames rather than
    contributing nothing, and the returned ``from_focused`` records which it was.
    That fallback is the difference between working and inert when a run's CLEAR
    opening frames are also soft-focused (both ``likely_defocus``) while the
    CLOUDED later frames read ``indeterminate``. Filtering defocus out unconditionally
    discarded the only clear sky in the run and let the cloud define the baseline.

    ``from_focused`` then bounds the risk that fallback reintroduces, because a
    defocus-derived baseline is held provisionally rather than permanently:

    - focused + focused -> merge, keeping the CLEAREST of each (max star, min
      background), so a later cloudier chunk never erodes a clearer one.
    - focused baseline, defocused chunk -> REFUSED. Inflated blob counts can never
      raise a baseline real stars already set.
    - defocused baseline, focused chunk -> REPLACED outright, not merged, so if the
      defocused numbers were inflated the inflation cannot persist past the first
      genuinely in-focus chunk.
    - defocused + defocused -> merge, still provisional.

    Returns a ``RunCloudBaseline``; ``triage`` reads it positionally, so a caller
    passing a bare ``(star, background)`` pair still works (and is read as focused)."""
    # Cap-on frames are dropped before anything else (``_cloud_baseline``'s docstring):
    # a chunk that is ONLY darks then contributes nothing at all rather than carrying a
    # black-floor background forward as the run's clearest sky.
    frames = [f for f in metrics if f.exposure_class != "cap_on"]
    if not frames:
        return existing
    focused = [f for f in frames if classify_shape(f) != "likely_defocus"]
    from_focused = bool(focused)
    star, background = _cloud_baseline(focused or frames)
    if existing is None:
        return RunCloudBaseline(star, background, from_focused)
    was_focused = existing[2] if len(existing) > 2 else True
    if was_focused and not from_focused:
        return existing
    if from_focused and not was_focused:
        return RunCloudBaseline(star, background, True)
    return RunCloudBaseline(max(star, existing[0]), min(background, existing[1]), from_focused)


def triage(metrics, *, run_baseline=None):
    """Fold a run's ``FrameMetrics`` into cloud/dew flags: cloud against the group's
    clearest frames (``_cloud_baseline``), dew against its opening ones. Report flags
    only -- nothing here refuses a frame or a run.

    Frames ``classify_exposure`` reads as ``cap_on`` are reported in their own
    ``cap_on_frames`` -- they are already ``usable=False``, they define neither baseline,
    and they are never relabelled cloud or dew. ``indeterminate_frames`` reports the
    frames that matched neither population (the cap going on mid-exposure); they are
    counted, not acted on.

    ``run_baseline`` (optional, ``(star, background)``, from
    ``update_run_cloud_baseline``) folds in the clearest condition seen elsewhere in
    the run: the group's own star baseline is raised to ``max(group, run)`` and its
    background baseline lowered to ``min(group, run)``, so a chunk with no clear
    frame of its own still judges against the run's real clear sky. Omitted, this is
    exactly the prior per-chunk-only behaviour."""
    frames = tuple(metrics)
    star_baseline, background_baseline = _cloud_baseline(frames)
    if run_baseline is not None:
        # Positional, not unpacked: update_run_cloud_baseline returns a 3-field
        # RunCloudBaseline, and a caller's bare (star, background) pair must keep working.
        run_star, run_background = run_baseline[0], run_baseline[1]
        star_baseline = max(star_baseline, run_star)
        background_baseline = min(background_baseline, run_background)
    # Both baselines see only frames that saw sky. A cap-on frame carries fwhm_px 0
    # (no stars to measure), so leaving it in the opening frames would zero the dew
    # baseline and switch dew detection off for the whole group.
    sky_frames = [f for f in frames if f.exposure_class != "cap_on"]
    baseline_n = min(len(sky_frames), FWHM_BASELINE_FRAME_COUNT)
    fwhm_baseline = (
        float(np.median([f.fwhm_px for f in sky_frames][:baseline_n])) if baseline_n else 0.0
    )
    cloud, dew, cap_on, indeterminate = [], [], [], []
    for f in frames:
        if f.exposure_class == "cap_on":
            # Reported as its own cause, never as cloud: "you left the lens cap on" and
            # "it clouded over" are different nights, and a capped frame would trip the
            # star-collapse test trivially.
            cap_on.append(f.path)
            continue
        if f.exposure_class != "light":
            indeterminate.append(f.path)
        star_collapsed = (
            star_baseline > 0 and f.star_count < star_baseline * CLOUD_STAR_COUNT_DROP_RATIO
        )
        background_jumped = (
            background_baseline > 0
            and f.background_median > background_baseline * CLOUD_BACKGROUND_JUMP_RATIO
        )
        if star_collapsed or background_jumped:
            cloud.append(f.path)
        if fwhm_baseline > 0 and f.fwhm_px > fwhm_baseline * DEW_FWHM_INFLATION_RATIO:
            dew.append(f.path)
    return TriageReport(
        frames=frames,
        cloud_frames=tuple(cloud),
        dew_frames=tuple(dew),
        cap_on_frames=tuple(cap_on),
        indeterminate_frames=tuple(indeterminate),
    )


def _geom(streak):
    """(p1, p2, unit direction (sign-canonical, or None if zero-length), length, centroid)."""
    x1, y1, x2, y2 = streak
    p1, p2 = np.array([x1, y1], dtype=np.float64), np.array([x2, y2], dtype=np.float64)
    v = p2 - p1
    length = float(np.linalg.norm(v))
    if length == 0.0:
        return p1, p2, None, 0.0, (p1 + p2) / 2.0
    unit = v / length
    if unit[0] < 0.0 or (unit[0] == 0.0 and unit[1] < 0.0):
        unit = -unit  # canonical sign so two parallel streaks agree on "direction"
    return p1, p2, unit, length, (p1 + p2) / 2.0


def _satellite_step(prev, curr, prior_progression):
    """Progression unit vector prev->curr if ``curr`` continues ``prev``'s track
    (collinear, endpoints close, progressing along the track, and -- once a chain
    already exists -- still heading the same way as ``prior_progression``); else None."""
    p1, p2, dir_prev, len_prev, cen_prev = _geom(prev)
    q1, q2, dir_curr, len_curr, cen_curr = _geom(curr)
    if dir_prev is None or dir_curr is None:
        return None
    if float(np.dot(dir_prev, dir_curr)) < SATELLITE_COLLINEAR_COS_MIN:
        return None
    gap = min(np.linalg.norm(a - b) for a in (p1, p2) for b in (q1, q2))
    if gap > SATELLITE_MAX_GAP_RATIO * max(min(len_prev, len_curr), 1.0):
        return None
    progression = cen_curr - cen_prev
    prog_len = float(np.linalg.norm(progression))
    if prog_len == 0.0:
        return None
    progression = progression / prog_len
    if abs(float(np.dot(progression, dir_prev))) < SATELLITE_PROGRESSION_COS_MIN:
        return None
    if prior_progression is not None and float(np.dot(progression, prior_progression)) <= 0.0:
        return None  # reversed course mid-chain: not the same object continuing its track
    return progression


def _cadence_reason(frames):
    """``None`` when every adjacent pair's capture gap (``FrameMetrics.captured_at``) is
    known, strictly increasing, and within ``SATELLITE_MAX_CADENCE_S``; else the decline
    reason. A non-increasing pair declines at ANY cadence (catches an out-of-order
    list); a gap over the ceiling also declines, but only catches a dropped frame when
    its own gap crosses that ceiling -- not in general. PROVISIONAL, fail-closed."""
    if len(frames) < 2:
        return None
    max_gap = 0.0
    for prev, curr in zip(frames, frames[1:]):
        if prev.captured_at is None or curr.captured_at is None:
            return "capture timestamp missing on at least one frame; satellite chaining declines"
        gap = (curr.captured_at - prev.captured_at).total_seconds()
        if gap <= 0.0:
            return "capture times are not strictly increasing; satellite chaining declines"
        max_gap = max(max_gap, gap)
    if max_gap > SATELLITE_MAX_CADENCE_S:
        return (
            f"cadence {max_gap:.2f}s exceeds the {SATELLITE_MAX_CADENCE_S:.0f}s "
            "the geometry gate assumes"
        )
    return None


def _point_segment_distance(p, a, b):
    """Distance (px) from point ``p`` to segment ``a``-``b``."""
    ab = b - a
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 == 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / ab_len2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _bbox(streak):
    """(minx, miny, maxx, maxy) of one streak -- a cheap reject before the four point-segment distance calls in ``_footprint_agrees``."""
    x1, y1, x2, y2 = streak
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _footprint_agrees(streak_a, streak_b, box_a, box_b):
    """True iff EVERY endpoint of each streak lies within ``ARTIFACT_POSITION_TOLERANCE_PX`` of the OTHER streak's segment -- both extents must occupy the same footprint, not just share a closest point, so a crosser is rejected once either streak runs past the tolerance footprint (2x ``ARTIFACT_POSITION_TOLERANCE_PX``), with no length ratio involved -- two streaks both shorter than that footprint are indistinguishable from one defect by position alone. ``box_a``/``box_b`` (tolerance-expanded) skip the four distance calls for a pair too far apart to agree."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    tol = ARTIFACT_POSITION_TOLERANCE_PX
    if ax1 + tol < bx0 or bx1 + tol < ax0 or ay1 + tol < by0 or by1 + tol < ay0:
        return False
    p1, p2 = np.array(streak_a[:2], dtype=np.float64), np.array(streak_a[2:], dtype=np.float64)
    q1, q2 = np.array(streak_b[:2], dtype=np.float64), np.array(streak_b[2:], dtype=np.float64)
    return (
        _point_segment_distance(p1, q1, q2) <= tol
        and _point_segment_distance(p2, q1, q2) <= tol
        and _point_segment_distance(q1, p1, p2) <= tol
        and _point_segment_distance(q2, p1, p2) <= tol
    )


def _artifact_groups(per_frame):
    """Index-pair tuples in a group where EVERY pair mutually satisfies ``_footprint_agrees`` (re-checked per candidate, never inferred through a chain, so one streak can't bridge two unrelated defects into a group) spanning >= ``ARTIFACT_MIN_OCCURRENCES`` DISTINCT FRAMES: a frame that already matched can't lock its other repeat out."""
    entries = [(fi, si, s) for fi, streaks in enumerate(per_frame) for si, s in enumerate(streaks)]
    boxes = [_bbox(s) for _, _, s in entries]
    groups = []
    for i in range(len(entries)):
        for group in groups:
            if all(
                _footprint_agrees(entries[i][2], entries[k][2], boxes[i], boxes[k]) for k in group
            ):
                group.append(i)
                break
        else:
            groups.append([i])
    return [
        tuple((entries[k][0], entries[k][1]) for k in group)
        for group in groups
        if len({entries[k][0] for k in group}) >= ARTIFACT_MIN_OCCURRENCES
    ]


def _chain_satellites(per_frame):
    """Set of (frame_index, streak_index) pairs belonging to a chain that persists
    across >= 2 consecutive frames -- the satellite signature."""
    members = set()
    for start_frame, streaks in enumerate(per_frame):
        for start_idx in range(len(streaks)):
            if (start_frame, start_idx) in members:
                continue
            chain = [(start_frame, start_idx)]
            frame_i, idx_i, prior_progression = start_frame, start_idx, None
            while frame_i + 1 < len(per_frame):
                prev = per_frame[frame_i][idx_i]
                match = None
                for ci, curr in enumerate(per_frame[frame_i + 1]):
                    progression = _satellite_step(prev, curr, prior_progression)
                    if progression is not None:
                        match = (ci, progression)
                        break  # first match; ambiguous multi-candidate frames are rare
                if match is None:
                    break
                idx_i, prior_progression = match
                frame_i += 1
                chain.append((frame_i, idx_i))
            if len(chain) >= 2:
                members.update(chain)
    return members


def _aircraft_groups(streaks):
    """Index tuples of ``streaks`` (one frame) that form a dashed strobe: mutually
    collinear, on a common line, and regularly spaced along it."""
    geoms = [_geom(s) for s in streaks]
    used = [False] * len(streaks)
    groups = []
    for i, (_, _, base_dir, _, base_cen) in enumerate(geoms):
        if used[i] or base_dir is None:
            continue
        members = [i]
        for j, (_, _, dir_j, _, cen_j) in enumerate(geoms):
            if j == i or used[j] or dir_j is None:
                continue
            if abs(float(np.dot(base_dir, dir_j))) < AIRCRAFT_COLLINEAR_COS_MIN:
                continue
            offset = cen_j - base_cen
            offset = offset - float(np.dot(offset, base_dir)) * base_dir
            if float(np.linalg.norm(offset)) <= AIRCRAFT_LINE_OFFSET_PX:
                members.append(j)
        if len(members) < AIRCRAFT_MIN_STROBE_COUNT:
            continue
        members.sort(key=lambda k: float(np.dot(geoms[k][4], base_dir)))
        gaps = np.asarray(
            [
                min(np.linalg.norm(a - b) for a in geoms[m][:2] for b in geoms[n][:2])
                for m, n in zip(members, members[1:])
            ],
            dtype=np.float64,
        )
        if gaps.mean() == 0.0 or float(gaps.std() / gaps.mean()) > AIRCRAFT_GAP_REGULARITY_RATIO:
            continue
        for k in members:
            used[k] = True
        groups.append(tuple(members))
    return groups


def classify_streaks(metrics):
    """Label each frame's streaks satellite / aircraft / artifact / unknown by
    cross-frame persistence and real capture cadence.

    - satellite: >=2 consecutive, collinear, progressing frames, ONLY when every
      capture gap is known, increasing, and within ``SATELLITE_MAX_CADENCE_S``
      (``_cadence_reason``) -- else the run declines rather than trust index
      adjacency.
    - artifact: mutual position agreement (every pair's endpoints within tolerance of the other's segment, not a length ratio or bounding box) across >= ``ARTIFACT_MIN_OCCURRENCES`` DISTINCT FRAMES; decided before aircraft.
    - aircraft: a within-frame dashed-strobe pattern, among streaks not claimed above.
    - unknown: everything else, including a single-exposure streak with no strobe or repeat evidence: a single exposure alone is never positive proof of a meteor, and this module has no other single-frame signal that would establish one.

    ``metrics``: a run's ``list[FrameMetrics]`` in capture order. Returns a flat
    ``tuple[StreakClassification, ...]``, frame order then streak order;
    ``reason`` is set on every result (any label) of a declined run, else ``None``.
    """
    frames = tuple(metrics)
    unsearched = [f.path for f in frames if f.streaks is None]
    if unsearched:
        # Refusing beats returning (): a frame measured with detect_meteors=False never
        # looked, and an empty classification would read downstream as "this run has no
        # meteors" -- the exact false negative the None/[] distinction exists to prevent.
        raise StreakDetectionNotRunError(
            f"classify_streaks: {len(unsearched)} frame(s) were measured with "
            f"detect_meteors=False and carry no streak data (first: {unsearched[0]}) -- "
            "re-measure them with detect_meteors=True before classifying"
        )
    per_frame = [list(f.streaks) for f in frames]
    cadence_reason = _cadence_reason(frames)
    satellite = _chain_satellites(per_frame) if cadence_reason is None else set()
    artifact = {pair for group in _artifact_groups(per_frame) for pair in group}
    aircraft = set()
    for i, streaks in enumerate(per_frame):
        remaining = [
            si for si in range(len(streaks)) if (i, si) not in satellite and (i, si) not in artifact
        ]
        for group in _aircraft_groups([streaks[si] for si in remaining]):
            aircraft.update((i, remaining[k]) for k in group)
    results = []
    for i, f in enumerate(frames):
        for si, streak in enumerate(f.streaks):
            if (i, si) in satellite:
                label = "satellite"
            elif (i, si) in artifact:
                label = "artifact"
            elif (i, si) in aircraft:
                label = "aircraft"
            else:
                label = "unknown"
            # cadence_reason (run-level) stamps every result, any label.
            results.append(StreakClassification(i, f.path, streak, label, cadence_reason))
    return tuple(results)
