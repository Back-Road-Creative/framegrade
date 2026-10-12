"""Star-masked mesh background removal for the deep sky combine.

Why this exists: on a deep stack one edge of the frame averaged down noticeably worse than
the sqrt(N) ideal while the centre and the other edge came close to it.  Noise efficiency is
a smooth field over the frame driven by LOCAL SKY BACKGROUND, not by sensor position: the
edge is not special, and other cells of the grid are as bad or worse.

What sits in that field is sky-fixed structure, not a sensor artefact -- it correlates
between chunks in the sky frame but not in the camera frame -- and it lives in a mid-scale
band of tens of pixels that a low-order surface has no freedom to fit, which is why a
plain gradient test comes back null.

A star-masked 32 px tile-median mesh removes it.  Synthetic controls (white noise and
injected star flux) confirm the mesh leaves real signal alone.

The mesh is deliberately a TILE median sampled on a grid rather than a true k-px median filter --
the same estimator, but seconds per call cheaper on a full frame, and it is what background meshes
actually do.  Star pixels are excluded from every tile's median, because a median that eats a star
halo IS the signal-removal failure the controls exist to catch.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from .astro_noise_gate import detrend

MESH_TILE_PX = 32
STAR_HALO_PX = 15
STAR_SIGMA = 5.0
MAD_TO_SIGMA = 1.4826


def _star_mask(mono: np.ndarray, tile: int) -> np.ndarray:
    """``(H, W)`` bool: True on a star or its halo, thresholded RELATIVE to the local scatter.

    The residual is 2nd-degree DETRENDED, not high-passed.  High-passing at the mesh's own scale
    (this used to blur at ``2 * tile``) strips out the very 8-128 px structure the mesh exists to
    remove; that collapses the MAD sigma, 5 sigma becomes a low bar, star wings clear it, and the
    15 px dilation multiplies each extra seed into a large area.  That inflates the mask's
    coverage severalfold, masking away the very structure the mesh is meant to remove and
    costing much of the efficiency it exists to deliver.
    A low-order detrend leaves the structure in the residual, so sigma stays representative.

    `astro_meteors._star_prominence_mask` is the same idea but cuts at a fixed absolute
    ``STAR_PROMINENCE_ADU`` tuned for developed frames; these are linear sky means sitting near
    0.01, where an absolute cut is meaningless.  So the threshold here is MAD-relative.
    """
    residual = detrend(mono)
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return np.zeros(mono.shape, dtype=bool)
    sigma = float(MAD_TO_SIGMA * np.median(np.abs(finite - np.median(finite))))
    if not np.isfinite(sigma) or sigma <= 0.0:
        return np.zeros(mono.shape, dtype=bool)
    core = (residual > STAR_SIGMA * sigma).astype(np.uint8)
    kernel = np.ones((2 * STAR_HALO_PX + 1, 2 * STAR_HALO_PX + 1), dtype=np.uint8)
    return cv2.dilate(core, kernel) > 0


def mesh_background(mean: np.ndarray, valid: np.ndarray, *, tile: int = MESH_TILE_PX) -> np.ndarray:
    """Background STRUCTURE to subtract from a chunk's per-pixel ``mean``, same shape as it.

    The field is centred to zero median per channel, so subtracting it removes the 8-128 px
    structure and LEAVES the sky pedestal alone.  That distinction is the whole contract: an
    uncentred tile median is the absolute sky level, and subtracting that would hand the
    downstream stretch a frame whose sky sits at zero.

    ``valid`` is the ``(H, W)`` footprint mask -- pixels no frame contributed to.  Those and star
    pixels are held out of the tile medians; a tile that is left with nothing falls back to the
    median of the tiles that did resolve, so an unsampled corner cannot poison the field.

    Each channel gets its own mesh: a colour gradient is per-channel, and one shared field would
    leave two thirds of it in place.  The frame is padded up to a whole number of tiles first --
    production frames are 3728 x 5600 and 3728 % 32 == 16, so truncating would drop a 16 px strip.
    """
    mean = np.asarray(mean, dtype=np.float64)
    if mean.ndim != 3:
        raise ValueError("mesh_background expects an (H, W, C) per-pixel mean")
    height, width, channels = mean.shape
    keep = np.asarray(valid, dtype=bool) & ~_star_mask(mean.mean(axis=2), tile)

    rows = -(-height // tile)
    cols = -(-width // tile)
    field = np.empty(mean.shape, dtype=np.float32)
    for channel in range(channels):
        padded = np.full((rows * tile, cols * tile), np.nan, dtype=np.float64)
        padded[:height, :width] = np.where(keep, mean[..., channel], np.nan)
        tiles = padded.reshape(rows, tile, cols, tile).transpose(0, 2, 1, 3)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            # A tile fully covered by stars or lying outside the footprint is expected, not a
            # problem -- it falls back to the median of the tiles that did resolve, below.
            warnings.simplefilter("ignore", RuntimeWarning)
            coarse = np.nanmedian(tiles.reshape(rows, cols, tile * tile), axis=2)
        resolved = coarse[np.isfinite(coarse)]
        coarse = np.where(
            np.isfinite(coarse), coarse, np.median(resolved) if resolved.size else 0.0
        )
        upsampled = cv2.resize(
            coarse.astype(np.float32), (cols * tile, rows * tile), interpolation=cv2.INTER_CUBIC
        )
        cropped = upsampled[:height, :width]
        # Centre on the median over pixels that carry data -- the mean would be dragged by the
        # padded margin and by any unsampled corner.
        sampled = cropped[keep] if keep.any() else cropped
        field[..., channel] = cropped - np.median(sampled)
    return field
