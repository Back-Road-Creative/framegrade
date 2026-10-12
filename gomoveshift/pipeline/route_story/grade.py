"""Still-photo grading: haze removal, exposure, vibrance and lens-corner crop.

Applied once, at the end of ``hud._load_still``, to every inserted photo
regardless of source (camera preview or RAW developer). Mountain views shot
through haze read flat and blue-grey; a saturation boost alone cannot restore
the lost contrast, so haze removal runs first, then a gentle exposure and
vibrance pass, then a light local-contrast curve with a pre-quantisation
dither so a lifted sky does not band. A separate pass removes the dark,
rounded lens-hood vignetting some frames carry in their corners, which the
straight-edge black-border crop cannot see.

All tuned constants live here, named, with the tuning story in
``docs/pipeline/17-route-story-still-grade.md``.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# --- haze removal (dark-channel prior, Ko/He-style) ---
_HAZE_PATCH = 15
_HAZE_DOWNSCALE_MAX = 480
_HAZE_OMEGA_MAX = 0.65
_HAZE_T_MIN = 0.45
_HAZE_DARK_LOW = 0.05
_HAZE_DARK_HIGH = 0.55
_HAZE_VARIANCE_MIN = 0.0008
_HAZE_ATMOSPHERE_MIN = 0.15
_HAZE_ATMOSPHERE_TOP_FRACTION = 0.001
# Dark-channel transmission alone cannot tell a distant hazy veil from a dark
# foreground object (forest shadow, foliage) -- both have a low dark-channel
# value. So correction is only fully trusted where a pixel ALSO looks like
# haze on its own: bright enough (haze lifts luma toward the atmosphere
# colour, it never darkens a scene) AND not richly saturated (a veil
# desaturates colour; ordinary scene saturation on a hazy ridge is still
# fairly low, so the low end of this band is set to pass a real ridge through
# untouched and only the genuinely rich end -- sky, close foliage -- is
# protected). Measurements on real footage: near-ground forest ~0.33-0.41 luma,
# distant hazy ridges ~0.62-0.65 luma, both around 0.25-0.33 saturation.
# Confidence is the PRODUCT of the two bands, so a pixel needs to look hazy on
# both axes to get full correction; either axis alone pulls transmission back
# toward 1 (no change).
_HAZE_LUMA_LOW = 0.42
_HAZE_LUMA_HIGH = 0.55
_HAZE_SAT_LOW = 0.35
_HAZE_SAT_HIGH = 0.55
# The transmission-map recovery above corrects EXPOSURE (it pulls a hazy
# pixel's mean back toward its true radiance) but a dark-channel prior's
# per-pixel transmission is itself correlated with local brightness, so
# lower-albedo texture inside a hazy region (a shaded fold in a ridge) gets
# systematically LESS correction than brighter texture right next to it --
# the recovery formula alone nets out to *less* local contrast in that band,
# not more (measured on real fixtures: distant-ridge luma std dropped after
# haze removal alone). What was missing was restoring the fine spatial detail
# a hazy veil actually erases, not just its mean brightness. A confidence-
# gated large-radius unsharp pass does that: it boosts local contrast only
# where the same haze-confidence mask says the pixel IS hazy, scaled by how
# much haze this frame actually has, so a clear or near-ground pixel is left
# alone (confidence ~0) and a hazy ridge gets its texture back.
# The amount is deliberately modest. A camera preview is an 8-bit JPEG: a
# strong boost here (6.0 was tried) passes every contrast/haze number while
# drawing white halos along each ridgeline, posterising the far field into
# electric blue and amplifying JPEG blocks -- it reads as a bad HDR filter,
# not a clear day. The dehaze ceiling/transmission floor above are eased for
# the same reason. Natural beats maximal; the over-sharpening guard in the
# real-fixture tests pins this.
_CLARITY_SIGMA_FRACTION = 0.02
_CLARITY_AMOUNT = 1.5
# Regulariser of the clarity base's edge-following fit: small against a
# ridge-to-haze luma step (~0.3, squared 0.09), so such an edge is kept whole.
_CLARITY_EDGE_EPS = 1e-3
# A genuinely hazy ridge legitimately picks up a *few* newly-bright highlight
# pixels once its veil is cleared -- that is the point of the pass, not a
# fault. What must not happen is an already-clear, already-bright frame
# (little haze, so its clarity confidence is high only because it happens to
# be light-coloured, not because it's a hazy veil) getting pushed into a wall
# of clipped highlights. This cap is looser than exposure's own
# ``_EXPOSURE_MAX_CLIP_FRACTION`` for exactly that reason.
_CLARITY_MAX_CLIP_FRACTION = 0.025

# --- exposure ---
_EXPOSURE_TARGET_LUMA = 0.45
_EXPOSURE_GAMMA_MIN = 0.75
_EXPOSURE_GAMMA_MAX = 1.15
_EXPOSURE_MAX_CLIP_FRACTION = 0.01
# A haze-removed frame reads darker overall (the bright veil that inflated its
# mean luma is gone) without the near ground having changed at all, so the
# exposure target is raised in proportion to how much dehaze actually did.
# But chasing that target as an unbounded whole-frame goal is wrong: it
# darkens rich frames and over-lifts a dim, haze-free frame (a shadowed lake)
# by 0.2. The near ground is already
# protected by the transmission guard above, not by exposure, so exposure's
# own move is always capped by how much haze was actually found -- a small,
# named amount on a clear frame, more on a heavily hazed one, never
# unbounded -- and a genuinely underexposed clear frame gets a little extra
# room since it has nothing to do with haze at all.
_EXPOSURE_DEHAZE_TARGET_BOOST = 0.12
_EXPOSURE_CLEAR_MAX_DELTA = 0.05
_EXPOSURE_HAZY_MAX_DELTA = 0.15
_EXPOSURE_UNDEREXPOSED_LUMA = 0.30
_EXPOSURE_UNDEREXPOSED_MAX_DELTA = 0.10

# --- vibrance ---
_VIBRANCE_TARGET_COLOURFULNESS = 55.0
_VIBRANCE_K_MAX = 0.6
# The stages above are each individually colourfulness-non-decreasing on their
# own target frame, but a later stage (exposure's gamma, the S-curve) can
# still cost a small amount of the chroma an earlier stage added, netting a
# small overall loss on a frame that started already fairly colourful. Rather
# than chase that interaction through every stage, guarantee the final output
# never reads duller than the source: blend back toward the untouched frame
# by just enough to hold the floor.
_COLOURFULNESS_NEVER_DULLER_SEARCH_STEPS = 8

# --- local punch / dither ---
_SCURVE_MIX = 0.18
_DITHER_AMPLITUDE = 1.0 / 255.0
# Centred (+0.5) so the offsets are zero-mean and stay strictly inside half a
# level: a value already on an 8-bit level never moves, and the frame is never
# biased darker. Uncentred, the -0.5 cell turned pure 255 into 254.
_BAYER4 = (
    np.array(
        [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
        dtype=np.float32,
    )
    + 0.5
) / 16.0 - 0.5

# --- lens-hood corner crop ---
_CORNER_FRACTION = 0.04
_CORNER_DARK_RATIO = 0.55
_STRAIGHT_EDGE_DARK_RATIO = 0.85
# Real lens-hood vignetting is a small margin (measured on true vignette
# fixtures: ~0.07-0.13 of the short side). A dark, textured scene corner --
# trees or a riverbank framing the shot -- can pass the corners-dark/
# edges-bright test with a much larger extent. Past this bound the "dark
# corner" reads as scene content, not vignetting, so it is left uncropped
# rather than clamped: clamping a 40%-extent read as if it were a real
# small-margin vignette still discards real content.
_CORNER_CROP_MAX_FRACTION = 0.22


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _colourfulness(rgb01: np.ndarray) -> float:
    """Hasler-Susstrunk colourfulness metric, on the conventional 0-255 scale."""
    r, g, b = rgb01[..., 0] * 255.0, rgb01[..., 1] * 255.0, rgb01[..., 2] * 255.0
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, std_yb = float(np.std(rg)), float(np.std(yb))
    mean_rg, mean_yb = float(np.mean(rg)), float(np.mean(yb))
    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))


def _downscale(rgb: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return rgb
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _dark_channel(rgb: np.ndarray, patch: int) -> np.ndarray:
    minc = np.min(rgb, axis=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, patch), max(1, patch)))
    return cv2.erode(minc, kernel)


def _edge_following(guide: np.ndarray, values: np.ndarray, radius: int) -> np.ndarray:
    """``values`` smoothed over ``radius`` px without crossing ``guide``'s edges.

    A guided filter: inside each window ``values`` is fitted as a straight-line
    function of ``guide``, so a strong edge in the guide stays a step in the
    result instead of being averaged across the way a Gaussian would.
    """
    size = (radius * 2 + 1, radius * 2 + 1)

    def mean(array: np.ndarray) -> np.ndarray:
        return cv2.blur(array, size, borderType=cv2.BORDER_REFLECT)

    guide_mean, values_mean = mean(guide), mean(values)
    covariance = mean(guide * values) - guide_mean * values_mean
    variance = mean(guide * guide) - guide_mean * guide_mean
    slope = covariance / (variance + _CLARITY_EDGE_EPS)
    offset = values_mean - slope * guide_mean
    return mean(slope) * guide + mean(offset)


def _haze_strength(rgb01: np.ndarray) -> float:
    small = _downscale(rgb01, _HAZE_DOWNSCALE_MAX)
    return float(np.mean(_dark_channel(small, _HAZE_PATCH)))


def _haze_confidence(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel 0-1 trust that a pixel is genuine distant haze, not a near,
    dark or richly saturated object the transmission map alone can't tell
    apart from haze."""
    saturation = cv2.cvtColor(np.clip(rgb, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2HSV)[..., 1]
    confidence_luma = np.clip(
        (_luma(rgb) - _HAZE_LUMA_LOW) / (_HAZE_LUMA_HIGH - _HAZE_LUMA_LOW), 0.0, 1.0
    )
    confidence_sat = np.clip(
        (_HAZE_SAT_HIGH - saturation) / (_HAZE_SAT_HIGH - _HAZE_SAT_LOW), 0.0, 1.0
    )
    return confidence_luma * confidence_sat


def measure(image: Image.Image) -> dict:
    """Colourfulness, mean luma, clip fraction and haze estimate of ``image``."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luma = _luma(rgb)
    return {
        "colourfulness": _colourfulness(rgb),
        "mean_luma": float(np.mean(luma)),
        "clip_fraction": float(np.mean(luma >= 0.995)),
        "haze_estimate": _haze_strength(rgb),
    }


def _remove_haze(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Dehazed image and the 0-1 strength (``omega``/``_HAZE_OMEGA_MAX``) applied."""
    small = _downscale(rgb, _HAZE_DOWNSCALE_MAX)
    if float(np.var(small)) < _HAZE_VARIANCE_MIN:
        # No spatial texture at all: nothing to derive a depth cue from, so a
        # flat card/backdrop is left alone rather than corrupted by noise.
        return rgb, 0.0
    # ``_HAZE_PATCH`` is sized for the downscaled copy directly (same as
    # ``_haze_strength``'s ``measure()`` reading) -- scaling it down further
    # by the downscale ratio (previous code) left a 1-3px window on a
    # 5568px-wide source, too small to pool a meaningful neighbourhood, which
    # under-corrected distant haze relative to what ``measure()`` reported.
    dark_small = _dark_channel(small, _HAZE_PATCH)
    dark_mean = float(np.mean(dark_small))
    if dark_mean <= _HAZE_DARK_LOW:
        return rgb, 0.0
    strength = min(1.0, (dark_mean - _HAZE_DARK_LOW) / (_HAZE_DARK_HIGH - _HAZE_DARK_LOW))
    omega = _HAZE_OMEGA_MAX * strength
    if omega <= 1e-3:
        return rgb, 0.0

    flat_dark = dark_small.reshape(-1)
    flat_rgb = small.reshape(-1, 3)
    count = max(1, int(_HAZE_ATMOSPHERE_TOP_FRACTION * flat_dark.size))
    idx = np.argpartition(flat_dark, -count)[-count:]
    atmosphere = np.clip(np.mean(flat_rgb[idx], axis=0), _HAZE_ATMOSPHERE_MIN, 1.0)

    dark_norm_small = _dark_channel(small / atmosphere[None, None, :], _HAZE_PATCH)
    transmission_small = 1.0 - omega * dark_norm_small
    transmission_small = cv2.bilateralFilter(
        transmission_small.astype(np.float32), d=9, sigmaColor=0.15, sigmaSpace=15
    )
    transmission = cv2.resize(
        transmission_small, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    confidence = _haze_confidence(rgb)
    transmission = 1.0 - (1.0 - transmission) * confidence
    transmission = np.clip(transmission, _HAZE_T_MIN, 1.0)

    recovered = (rgb - atmosphere[None, None, :]) / transmission[..., None] + atmosphere[
        None, None, :
    ]
    recovered = _apply_haze_clarity(rgb, np.clip(recovered, 0.0, 1.0), strength, confidence)
    return np.clip(recovered, 0.0, 1.0), strength


def _apply_haze_clarity(
    rgb: np.ndarray, dehazed: np.ndarray, strength: float, confidence: np.ndarray
) -> np.ndarray:
    """Restore the fine local contrast a hazy veil erases (see ``_CLARITY_*``).

    A large-radius unsharp pass, added back only where ``confidence`` says a
    pixel is genuine haze and scaled by how hazy this frame is overall -- a
    clear frame or a near-ground pixel gets ~0 boost.
    """
    if strength <= 1e-3:
        return dehazed
    h, w = dehazed.shape[:2]
    sigma = max(3.0, min(h, w) * _CLARITY_SIGMA_FRACTION)
    # The base must not average ACROSS a ridgeline. A Gaussian base does: beside
    # a dark near ridge it reads darker than the haze really is, so "detail"
    # comes out positive along the whole edge and the boost paints a white glow
    # hugging the ridge (measured on a synthetic ridge: +0.105 luma in the 20px
    # beside it, against 0.001 with this base).
    guide = _luma(dehazed).astype(np.float32)
    base = np.stack(
        [
            _edge_following(guide, dehazed[..., channel].astype(np.float32), round(sigma))
            for channel in range(3)
        ],
        axis=-1,
    )
    detail = dehazed - base
    amount = _CLARITY_AMOUNT
    boosted = dehazed
    base_clip = float(np.mean(_luma(dehazed) >= 0.995))
    # A confident-hazy sky or bright ridge highlight can already sit close to
    # white; boosting its local contrast can push it over. Taper the amount
    # down (never below zero) rather than let clarity clip highlights the
    # rest of the pipeline was careful not to.
    for _ in range(4):
        boosted = np.clip(dehazed + amount * strength * confidence[..., None] * detail, 0.0, 1.0)
        clip_fraction = float(np.mean(_luma(boosted) >= 0.995))
        if clip_fraction <= max(base_clip, _CLARITY_MAX_CLIP_FRACTION):
            break
        amount *= 0.5
    return boosted


def _apply_exposure(rgb: np.ndarray, haze_strength: float = 0.0) -> np.ndarray:
    mean_luma = float(np.mean(_luma(rgb)))
    if mean_luma <= 1e-6:
        return rgb
    target = _EXPOSURE_TARGET_LUMA + _EXPOSURE_DEHAZE_TARGET_BOOST * haze_strength
    allowed = _EXPOSURE_CLEAR_MAX_DELTA + (
        _EXPOSURE_HAZY_MAX_DELTA - _EXPOSURE_CLEAR_MAX_DELTA
    ) * float(np.clip(haze_strength, 0.0, 1.0))
    if mean_luma < _EXPOSURE_UNDEREXPOSED_LUMA:
        allowed = max(allowed, _EXPOSURE_UNDEREXPOSED_MAX_DELTA)
    target = mean_luma + float(np.clip(target - mean_luma, -allowed, allowed))
    gamma = float(np.log(target + 1e-6) / np.log(mean_luma + 1e-6))
    gamma = min(_EXPOSURE_GAMMA_MAX, max(_EXPOSURE_GAMMA_MIN, gamma))
    candidate = np.clip(rgb, 0.0, 1.0) ** gamma
    if gamma < 1.0:
        for taper in (0.75, 0.5, 0.25):
            if float(np.mean(_luma(candidate) >= 0.995)) <= _EXPOSURE_MAX_CLIP_FRACTION:
                break
            tapered_gamma = 1.0 + (gamma - 1.0) * taper
            candidate = np.clip(rgb, 0.0, 1.0) ** tapered_gamma
    # A single global gamma solved from ``mean_luma ** gamma == target`` is
    # only an approximation on a wide, non-uniform histogram (a bright sky
    # over dark water): the ACTUAL resulting mean can overshoot the ``target``
    # solved for above. Blend back toward the original by exactly enough to
    # cap the true delta -- this is what makes ``allowed`` a real guarantee
    # rather than a hint the gamma solve can miss.
    actual_delta = float(np.mean(_luma(candidate))) - mean_luma
    if abs(actual_delta) > allowed and abs(actual_delta) > 1e-6:
        alpha = allowed / abs(actual_delta)
        candidate = rgb * (1.0 - alpha) + candidate * alpha
    return candidate


def _apply_vibrance(rgb: np.ndarray) -> np.ndarray:
    colourfulness = _colourfulness(rgb)
    gap = (_VIBRANCE_TARGET_COLOURFULNESS - colourfulness) / _VIBRANCE_TARGET_COLOURFULNESS
    k = float(np.clip(gap, 0.0, 1.0)) * _VIBRANCE_K_MAX
    if k <= 1e-3:
        return rgb
    hsv = cv2.cvtColor(np.clip(rgb, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2HSV)
    s = hsv[..., 1]
    hsv[..., 1] = np.clip(s * (1.0 + k * (1.0 - s)), 0.0, 1.0)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)


def _never_duller(original: np.ndarray, graded: np.ndarray) -> np.ndarray:
    """Blend ``graded`` back toward ``original`` just enough that grading
    never leaves its own input duller -- see
    ``_COLOURFULNESS_NEVER_DULLER_SEARCH_STEPS``. Guards the grading stages
    only; a separate crop before grading (``crop_lens_corners``) is outside
    this guarantee since it structurally changes the frame's content, not
    just its tone."""
    before = _colourfulness(original)
    if _colourfulness(graded) >= before:
        return graded
    lo, hi = 0.0, 1.0
    for _ in range(_COLOURFULNESS_NEVER_DULLER_SEARCH_STEPS):
        mid = (lo + hi) / 2.0
        blended = original * (1.0 - mid) + graded * mid
        if _colourfulness(blended) >= before:
            lo = mid
        else:
            hi = mid
    return np.clip(original * (1.0 - lo) + graded * lo, 0.0, 1.0)


def _apply_s_curve(rgb: np.ndarray) -> np.ndarray:
    x = np.clip(rgb, 0.0, 1.0)
    s = 3.0 * x**2 - 2.0 * x**3
    return np.clip(x * (1.0 - _SCURVE_MIX) + s * _SCURVE_MIX, 0.0, 1.0)


def _dither(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    tile = np.tile(_BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]
    return np.clip(rgb + tile[..., None] * _DITHER_AMPLITUDE, 0.0, 1.0)


def grade_still(image: Image.Image, *, enabled: bool = True) -> Image.Image:
    """Haze removal, exposure, vibrance and local punch, in that order."""
    if not enabled:
        return image
    mode = image.mode
    original = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    rgb, haze_strength = _remove_haze(original)
    rgb = _apply_exposure(rgb, haze_strength)
    rgb = _apply_vibrance(rgb)
    rgb = _apply_s_curve(rgb)
    rgb = _never_duller(original, rgb)
    rgb = _dither(rgb)
    pixels = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    graded = Image.fromarray(pixels, "RGB")
    return graded if mode == "RGB" else graded.convert(mode)


def _corner_vignette_extent(luma: np.ndarray) -> float | None:
    """Fraction (of the short side) to crop off each edge, or ``None`` if clean."""
    h, w = luma.shape
    short = min(h, w)
    patch = max(4, int(short * _CORNER_FRACTION))
    mid_w, mid_h = w // 2, h // 2

    def mean(a: np.ndarray) -> float:
        return float(np.mean(a))

    corners = (
        luma[:patch, :patch],
        luma[:patch, w - patch :],
        luma[h - patch :, :patch],
        luma[h - patch :, w - patch :],
    )
    edges = (
        luma[:patch, mid_w - patch : mid_w + patch],
        luma[h - patch :, mid_w - patch : mid_w + patch],
        luma[mid_h - patch : mid_h + patch, :patch],
        luma[mid_h - patch : mid_h + patch, w - patch :],
    )
    edge_mean = mean(np.concatenate([e.ravel() for e in edges]))
    corners_dark = all(mean(c) < _CORNER_DARK_RATIO * (edge_mean + 1e-6) for c in corners)
    edges_bright = edge_mean > _STRAIGHT_EDGE_DARK_RATIO * mean(luma)
    if not (corners_dark and edges_bright):
        return None

    threshold = _STRAIGHT_EDGE_DARK_RATIO * mean(luma)
    max_walk = short // 2

    def walk(y0: int, x0: int, dy: int, dx: int) -> int:
        for step in range(1, max_walk):
            y, x = y0 + dy * step, x0 + dx * step
            if luma[y, x] >= threshold:
                return step
        return max_walk

    extent = max(
        walk(0, 0, 1, 1),
        walk(0, w - 1, 1, -1),
        walk(h - 1, 0, -1, 1),
        walk(h - 1, w - 1, -1, -1),
    )
    fraction = extent / short
    return fraction if fraction <= _CORNER_CROP_MAX_FRACTION else None


def crop_lens_corners(image: Image.Image) -> Image.Image:
    """Crop the largest same-aspect rectangle excluding dark rounded corners.

    Detects dark, rounded lens-hood vignetting in all four corners while the
    straight edges between them stay bright — unlike ``crop_black_border``,
    which only removes unambiguously straight black edges. A clean frame is
    returned unchanged.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luma = _luma(rgb)
    fraction = _corner_vignette_extent(luma)
    if not fraction:
        return image
    h, w = luma.shape
    margin_w, margin_h = round(fraction * w), round(fraction * h)
    if margin_w <= 0 or margin_h <= 0:
        return image
    return image.crop((margin_w, margin_h, w - margin_w, h - margin_h))
