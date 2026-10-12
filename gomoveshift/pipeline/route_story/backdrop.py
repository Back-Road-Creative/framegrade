"""Vivid, cloud-free grading for NASA GIBS Blue Marble backdrop rasters.

Both Blue Marble opening-shot bands (``blue_marble`` z2-4, and the default,
non-trip-date ``gibs`` z5-8 fallback -- see ``prepare.py``) come back from
GIBS dark and desaturated: measured mean pixel value 37-47 of 255 over the
lower 48. Left alone, the resulting opening reads as "very dark", "fuzzy"
and not recognisably the USA; the target is "vibrant, recognizable and
clear". :func:`grade_backdrop` is a fixed recipe -- judged by eye
against real GIBS samples, not adaptive -- applied to every Blue Marble
raster before it is tiled.
"""

from __future__ import annotations

import numpy as np

# Bumped whenever the recipe below changes pixel output -- folded into a
# bundle's cache key (``prepare.py``'s ``_band_source_versions``) so graded
# and un-graded tiles can never share a bundle id.
BACKDROP_RECIPE_VERSION = "backdrop-grade-1"

_MIDTONE_GAMMA = 0.62
_SATURATION_GAIN = 1.45
_BLACK_POINT = 0.04
_WHITE_SCALE = 0.92
_SCURVE_WEIGHT = 0.35


def grade_backdrop(rgb: np.ndarray) -> np.ndarray:
    """Lift, saturate and contrast-stretch a Blue Marble raster.

    ``rgb`` and the return value are both ``uint8`` ``HxWx3``. Applies, in
    order: a midtone gamma lift, a saturation boost about each pixel's own
    channel mean, a black/white point stretch, and a light S-curve. Values
    are clipped to the valid range at every stage that could push outside
    it. Idempotence is not required -- grading an already-graded raster a
    second time is not a supported use.
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"grade_backdrop expects an HxWx3 array, got shape {rgb.shape}")
    a = rgb.astype(np.float32) / 255.0
    a = np.power(a, _MIDTONE_GAMMA)
    mean = a.mean(axis=-1, keepdims=True)
    a = mean + (a - mean) * _SATURATION_GAIN
    a = np.clip((a - _BLACK_POINT) / _WHITE_SCALE, 0.0, 1.0)
    a = _SCURVE_WEIGHT * (a * a * (3.0 - 2.0 * a)) + (1.0 - _SCURVE_WEIGHT) * a
    a = np.clip(a, 0.0, 1.0)
    return np.round(a * 255.0).astype(np.uint8)
