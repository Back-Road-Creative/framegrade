"""Quality score used to ORDER shots inside one category (never to exclude one).

Scored on the DEVELOPED image (uint8 RGB, HxWx3): technical quality (sharpness,
clipping; plus stars for a night sky), a per-category signal, and an optional
aesthetic score supplied by the caller. Every measure runs on a copy bounded to
``MAX_DIM`` on its long edge; the input is never mutated. Every number returned
is finite and within 0..1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ...analysis.visual_scorer import score_color_vibrancy, score_golden_hour, score_sky_ratio
from . import astro_triage
from .cull import compute_cull_score

CATEGORY_NIGHT_SKY = "astro"
CATEGORY_CLOSE_UP = "close_up"
CATEGORY_LANDSCAPE = "landscape"

#: Long-edge bound for every measure below (px).
MAX_DIM = 1024
#: Weight of the aesthetic score inside ``combine`` (the rest is the category signal).
AESTHETIC_WEIGHT = 0.5
#: Laplacian variance at which the saturating curve ``s / (s + K)`` reads 0.5. Real
#: sharp photos run in the hundreds and the cull floor is 30, so 300 keeps a
#: floor-passing frame low and a crisp one near 0.7-0.9.
SHARPNESS_K = 300.0
#: Clip shares above these floors are penalised (a little clipping is normal).
WHITE_FLOOR = 0.02
BLACK_FLOOR = 0.10
#: Score subtracted per unit of clip share above the floor.
CLIP_PENALTY_WEIGHT = 1.0
#: Night: star count at which the saturating curve ``n / (n + K)`` reads 0.5.
STAR_COUNT_K = 30.0
#: Night: FWHM (px) at/below which a star is "tight", and the span over which it
#: degrades to 0 (smaller fwhm is better).
FWHM_TIGHT_PX = 2.0
FWHM_SPAN_PX = 6.0
#: Night technical weights (sum to 1.0): stars, star sharpness, trailing, frame sharpness.
NIGHT_WEIGHTS = (0.4, 0.2, 0.25, 0.15)
#: Multiplier on a night frame that triage marks ``usable=False`` (kept, ranked low).
UNUSABLE_FACTOR = 0.4
#: Night sky: median luma (0..255) at which the background reads fully bright (0 darkness).
SKY_BRIGHT_LUMA = 80.0
#: Night foreground separation: lower-vs-upper third luma gap that reads 1.0.
FOREGROUND_GAP_LUMA = 64.0
#: Close-up: centre/border sharpness ratio ``r`` maps to ``r / (r + K)``; uniform
#: sharpness (r=1) reads 0.25.
ISOLATION_K = 3.0
#: Close-up: share of the centre crop on each axis (middle half).
CENTRE_FRACTION = 0.5
#: Mid-tone balance: mean luma that reads best, and its half-range.
MIDTONE_LUMA = 118.0


@dataclass(frozen=True)
class QualityScore:
    technical: float
    category: float
    aesthetic: float | None
    total: float
    parts: Mapping[str, float]


def _clamp(x: float) -> float:
    x = float(x)
    return 0.0 if math.isnan(x) else min(1.0, max(0.0, x))


def _bounded(rgb: np.ndarray) -> np.ndarray:
    """A copy of ``rgb`` with its long edge <= MAX_DIM; raises on a bad shape."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an HxWx3 uint8 RGB array, got {rgb.dtype} {rgb.shape}")
    h, w = rgb.shape[:2]
    if max(h, w) > MAX_DIM:
        k = MAX_DIM / float(max(h, w))
        return cv2.resize(
            rgb, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA
        )
    return rgb.copy()


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _night_parts(gray: np.ndarray, path, exposure_s, focal_mm) -> dict[str, float]:
    m = astro_triage.measure_frame(
        path if path is not None else "",
        gray.astype(np.float32),
        exposure_s=exposure_s,
        focal_mm=focal_mm,
        detect_meteors=False,
    )
    stars = m.star_count / (m.star_count + STAR_COUNT_K)
    if m.star_count:
        star_sharp = 1.0 - (m.fwhm_px - FWHM_TIGHT_PX) / FWHM_SPAN_PX
        trailing = 1.0 - m.shape.ellipticity if m.shape is not None else 0.5
    else:
        star_sharp = trailing = 0.0
    return {
        "stars": _clamp(stars),
        "star_sharpness": _clamp(star_sharp),
        "trailing": _clamp(trailing),
        "usable": 1.0 if m.usable else 0.0,
    }


def technical_score(
    rgb, *, night: bool = False, path=None, exposure_s=None, focal_mm=None
) -> tuple[float, dict[str, float]]:
    """(0..1 technical quality, named components). Night adds star measures."""
    small = _bounded(rgb)
    cull = compute_cull_score(small)
    sharp = cull.sharpness / (cull.sharpness + SHARPNESS_K) if cull.sharpness > 0 else 0.0
    penalty = CLIP_PENALTY_WEIGHT * max(0.0, cull.exposure_white_fraction - WHITE_FLOOR)
    if not night:
        penalty += CLIP_PENALTY_WEIGHT * max(0.0, cull.exposure_black_fraction - BLACK_FLOOR)
    parts = {"sharpness": _clamp(sharp), "clip_penalty": _clamp(penalty)}
    if not night:
        return _clamp(parts["sharpness"] - parts["clip_penalty"]), parts
    parts.update(_night_parts(_gray(small), path, exposure_s, focal_mm))
    w = NIGHT_WEIGHTS
    base = (
        w[0] * parts["stars"]
        + w[1] * parts["star_sharpness"]
        + w[2] * parts["trailing"]
        + w[3] * parts["sharpness"]
    )
    if not parts["usable"]:
        base *= UNUSABLE_FACTOR
    return _clamp(base - parts["clip_penalty"]), parts


def _laplacian_var(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    vals = lap[mask] if mask is not None else lap
    return float(vals.var()) if vals.size else 0.0


def category_signals(category: str, rgb) -> tuple[float, dict[str, float]]:
    """(0..1 category signal, named components) for one developed frame."""
    small = _bounded(rgb)
    gray = _gray(small)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)  # visual_scorer is BGR
    if category == CATEGORY_LANDSCAPE:
        golden = score_golden_hour(bgr)
        sky = score_sky_ratio(bgr)
        parts = {"golden_hour": _clamp(golden), "sky_ratio": _clamp(sky)}
        return _clamp((parts["golden_hour"] + parts["sky_ratio"]) / 2.0), parts
    if category == CATEGORY_NIGHT_SKY:
        h = gray.shape[0]
        third = max(1, h // 3)
        stars = astro_triage.detect_stars(gray.astype(np.float32)).shape[0]
        dark = 1.0 - float(np.median(gray)) / SKY_BRIGHT_LUMA
        gap = abs(float(gray[-third:].mean()) - float(gray[:third].mean()))
        parts = {
            "stars": _clamp(stars / (stars + STAR_COUNT_K)),
            "sky_dark": _clamp(dark),
            "foreground_sep": _clamp(gap / FOREGROUND_GAP_LUMA),
        }
        return _clamp(sum(parts.values()) / 3.0), parts
    vibrancy = _clamp(score_color_vibrancy(bgr))
    if category == CATEGORY_CLOSE_UP:
        h, w = gray.shape
        mask = np.zeros(gray.shape, bool)
        y0, x0 = int(h * (1 - CENTRE_FRACTION) / 2), int(w * (1 - CENTRE_FRACTION) / 2)
        mask[y0 : h - y0, x0 : w - x0] = True
        centre, border = _laplacian_var(gray, mask), _laplacian_var(gray, ~mask)
        ratio = centre / border if border > 0 else (ISOLATION_K if centre > 0 else 1.0)
        parts = {"isolation": _clamp(ratio / (ratio + ISOLATION_K)), "vibrancy": vibrancy}
        return _clamp(0.6 * parts["isolation"] + 0.4 * vibrancy), parts
    midtone = 1.0 - abs(float(gray.mean()) - MIDTONE_LUMA) / MIDTONE_LUMA
    parts = {"vibrancy": vibrancy, "midtone": _clamp(midtone)}
    return _clamp((parts["vibrancy"] + parts["midtone"]) / 2.0), parts


def combine(
    technical: float,
    category: float,
    aesthetic: float | None,
    *,
    aesthetic_weight: float = AESTHETIC_WEIGHT,
) -> float:
    """``technical * ((1 - w) * category + w * aesthetic)``; no aesthetic -> ``technical * category``."""
    t, c = _clamp(technical), _clamp(category)
    if aesthetic is None:
        return _clamp(t * c)
    w = _clamp(aesthetic_weight)
    return _clamp(t * ((1.0 - w) * c + w * _clamp(aesthetic)))


def score_developed(
    rgb,
    category: str,
    *,
    aesthetic: float | None = None,
    path=None,
    exposure_s=None,
    focal_mm=None,
) -> QualityScore:
    tech, tparts = technical_score(
        rgb,
        night=category == CATEGORY_NIGHT_SKY,
        path=Path(path) if path is not None else None,
        exposure_s=exposure_s,
        focal_mm=focal_mm,
    )
    cat, cparts = category_signals(category, rgb)
    aes = None if aesthetic is None else _clamp(aesthetic)
    parts = {**tparts, **cparts}
    return QualityScore(tech, cat, aes, combine(tech, cat, aes), parts)


def order_category(scored: Mapping[str, QualityScore]) -> list[str]:
    """Keys by total descending; ties break by key ascending (deterministic)."""
    return sorted(scored, key=lambda k: (-scored[k].total, k))
