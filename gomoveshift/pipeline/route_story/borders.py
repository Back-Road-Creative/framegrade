"""Compositing helpers for the NASA GIBS transparent reference-features overlay.

Pure numpy/PIL image ops, no I/O — the caller fetches a
:class:`~gomoveshift.pipeline.route_story.sources.GeoRaster` via
``sources.fetch_gibs_reference_features`` and passes its RGBA array in here.
``prepare.py`` overlays the borders on the national opening-scene backdrop,
keys its cache on :data:`BORDERS_RECIPE_VERSION` and refuses an overlay whose
coverage is near zero.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# Warm off-white so the lines read against both ocean blue and land colour,
# without competing with the burned OST text (which is a brighter white).
DEFAULT_BORDER_COLOUR: tuple[int, int, int] = (240, 238, 225)
DEFAULT_BORDER_OPACITY = 0.8
# Fraction the halo ring darkens the backdrop toward black, so a light line
# stays legible over bright desert/snow, not just over dark ocean.
HALO_DARKEN_FRACTION = 0.35
# A pixel counts as "covered" by the overlay above this alpha (of 255) — used
# by `border_coverage` to flag a near-empty/failed fetch.
COVERAGE_ALPHA_THRESHOLD = 32

# Bumped whenever the blend recipe below changes what pixels a cached
# borders-overlaid backdrop contains; ``prepare.py`` folds it into the cache key.
BORDERS_RECIPE_VERSION = "borders-1"


def _validate_rgb(rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be a uint8 HxWx3 array")


def _validate_overlay(overlay_rgba: np.ndarray) -> None:
    if overlay_rgba.dtype != np.uint8 or overlay_rgba.ndim != 3 or overlay_rgba.shape[2] != 4:
        raise ValueError("overlay_rgba must be a uint8 HxWx4 array")


def _resize_overlay(overlay_rgba: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize `overlay_rgba` (H, W, 4) to `size` = (width, height) if it doesn't already match."""
    height, width = size[1], size[0]
    if overlay_rgba.shape[:2] == (height, width):
        return overlay_rgba
    with Image.fromarray(overlay_rgba, mode="RGBA") as image:
        resized = image.resize((width, height), resample=Image.LANCZOS)
        return np.asarray(resized, dtype=np.uint8)


def overlay_borders(
    rgb: np.ndarray,
    overlay_rgba: np.ndarray,
    *,
    colour: tuple[int, int, int] = DEFAULT_BORDER_COLOUR,
    opacity: float = DEFAULT_BORDER_OPACITY,
    halo: bool = True,
) -> np.ndarray:
    """Blend the GIBS reference-features overlay's alpha channel onto `rgb` as coloured lines.

    Only `overlay_rgba`'s alpha is used as a line mask — its own RGB is ignored so the
    line colour is always ours. `opacity=0` or an all-zero alpha returns `rgb` unchanged
    (byte-identical). With `halo=True`, a 1-px-dilated copy of the mask first darkens the
    backdrop under and around each line so light lines stay legible over bright ground.
    """
    _validate_rgb(rgb)
    _validate_overlay(overlay_rgba)

    height, width = rgb.shape[:2]
    overlay_rgba = _resize_overlay(overlay_rgba, (width, height))

    alpha = overlay_rgba[..., 3].astype(np.float64) / 255.0
    if opacity <= 0.0 or not np.any(alpha):
        return rgb.copy()

    base = rgb.astype(np.float64)
    if halo:
        mask = (overlay_rgba[..., 3] > 0).astype(np.uint8) * 255
        with Image.fromarray(mask, mode="L") as mask_image:
            dilated_image = mask_image.filter(ImageFilter.MaxFilter(3))
            dilated = np.asarray(dilated_image, dtype=np.float64) / 255.0
        base = base * (1.0 - HALO_DARKEN_FRACTION * dilated[..., None])

    weight = (alpha * opacity)[..., None]
    colour_arr = np.asarray(colour, dtype=np.float64)
    blended = base * (1.0 - weight) + colour_arr * weight
    return np.clip(blended, 0, 255).astype(np.uint8)


def border_coverage(overlay_rgba: np.ndarray) -> float:
    """Fraction of `overlay_rgba` pixels whose alpha exceeds `COVERAGE_ALPHA_THRESHOLD`.

    Lets a caller detect an empty/failed overlay fetch (``prepare.py`` refuses to ship a
    national-zoom opening scene with ~0 coverage).
    """
    _validate_overlay(overlay_rgba)
    covered = overlay_rgba[..., 3] > COVERAGE_ALPHA_THRESHOLD
    return float(np.count_nonzero(covered)) / covered.size
