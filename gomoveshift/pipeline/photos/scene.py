"""Scene profiling — phase 1 of the photo style variants.

``SceneProfile`` is a pure, deterministic classification of one photo's light,
computed once per photo; later variant phases consume it to pick styles. Every
threshold reuses the grade engine's merged semantics instead of inventing a
second set; metering rides the develop stage's seam — never a second scale.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..color_analysis import SegmentStats
from ..color_correction import (
    CONTRAST_SPREAD_FULL,
    CONTRAST_SPREAD_OFF,
    FLAT_PROFILE_SATURATION,
    PHOTO_SUNSET_WARM_CAST_MIN,
    _haze_veil_factor,
    _is_low_key,
)
from .develop import (
    ANALYSIS_MAX_DIM,
    _analysis_rgb,
    _bounded_resize,
    _frame_stats,
)

# Exposure bands (mid-tone luminance, 0-255). Bright starts where the grade
# engine begins pulling gamma down; overexposed additionally needs highlights
# at the clip point, so a bright-but-intact meadow never reads overexposed.
EXPOSURE_BRIGHT_MID_MIN = 145.0
EXPOSURE_OVEREXPOSED_MID_MIN = 200.0
EXPOSURE_OVEREXPOSED_WHITE_MIN = 250.0

# Warmth floor: the bottom of the golden-hour cast range the photo grade
# already treats as scenic (the sunset guard's onset), mirrored for cool.
WARMTH_CAST_MIN = PHOTO_SUNSET_WARM_CAST_MIN

# Saturation bands at the calibrated analysis scale. Muted is the mist-class
# bound the grade's profile detector uses; rich starts where the boost ramp
# reaches zero (already saturated enough to need no help).
SAT_MUTED_MAX = FLAT_PROFILE_SATURATION
SAT_RICH_MIN = 0.45

# Haze: the merged mist/fog scene class (compressed spread + muted sat), split
# by the dehaze veil factor — how far the blacks are lifted above the clear-day
# floor. Under the hazy floor the blacks sit clear/underexposed, not veiled.
HAZE_HAZY_VEIL_MIN = 0.1
HAZE_FOGGY_VEIL_MIN = 0.75

# Dynamic-range bands on the white-black spread: flat at/under the spread where
# the grade's S-curve ramp reaches full strength.
DYNAMIC_RANGE_FLAT_MAX = CONTRAST_SPREAD_FULL
DYNAMIC_RANGE_WIDE_MIN = 200.0

# Near-mono floor: mean saturation at/under this is effectively a B&W frame.
NEAR_MONO_SAT_MAX = 0.05

# Coarse hue band (degrees, 0-360 circle) for the chroma-weighted color
# census: warm reds-oranges-yellows, wrapping through 0. Hues outside the
# band belong to no family and drop out of the vote.
HUE_WARM_MIN = 305.0  # wraps through 0 down to HUE_WARM_MAX
HUE_WARM_MAX = 60.0

# Headroom reads the already-rich end of the frame: 1 - this percentile of
# per-pixel saturation is how much boost the frame can absorb before clipping.
CHROMA_HEADROOM_PERCENTILE = 95.0


@dataclass(frozen=True)
class PhotoExif:
    """Optional capture metadata, supplied by the caller — never read from disk."""

    iso: float | None = None
    exposure_s: float | None = None
    capture_hour: int | None = None
    capture_utc: datetime.datetime | None = None  # aware UTC capture instant; None = no offset
    solar_elevation_deg: float | None = None  # apparent solar elevation; None = no track fix


@dataclass(frozen=True)
class SceneProfile:
    """One photo's light, classified once; later phases pick styles from it."""

    exposure_class: str  # low_key | normal | bright | overexposed
    warmth: str  # cool | neutral | warm
    sat_profile: str  # muted | normal | rich
    haze: str  # clear | hazy | foggy
    dynamic_range: str  # flat | normal | wide
    is_near_monochrome: bool
    chroma_headroom: float = 1.0  # 1 - p95 per-pixel saturation: push it absorbs
    warm_fraction: float = 0.0  # chroma-weighted share of warm r/o/y hues
    capture_hour: int | None = None  # EXIF hour, camera-local; None = no EXIF
    solar_elevation_deg: float | None = None  # apparent solar elevation; None = no track fix


def _exposure_class(stats: SegmentStats) -> str:
    # The low-key test is the grade engine's own: intact highlights are what
    # separate a silhouette from underexposure, so it never reads dark-normal.
    if _is_low_key(stats):
        return "low_key"
    if (
        stats.mid_tone >= EXPOSURE_OVEREXPOSED_MID_MIN
        and stats.white_point >= EXPOSURE_OVEREXPOSED_WHITE_MIN
    ):
        return "overexposed"
    if stats.mid_tone >= EXPOSURE_BRIGHT_MID_MIN:
        return "bright"
    return "normal"


def _warmth(stats: SegmentStats) -> str:
    r_cast = stats.r_mean - stats.g_mean
    b_cast = stats.b_mean - stats.g_mean
    if r_cast >= WARMTH_CAST_MIN and r_cast >= b_cast:
        return "warm"
    if b_cast >= WARMTH_CAST_MIN:
        return "cool"
    return "neutral"


def _sat_profile(saturation: float) -> str:
    if saturation <= SAT_MUTED_MAX:
        return "muted"
    if saturation >= SAT_RICH_MIN:
        return "rich"
    return "normal"


def _haze(stats: SegmentStats) -> str:
    # Saturated or contrasty scenes are never a veil — the grade's mist gate.
    spread = stats.white_point - stats.black_point
    if spread >= CONTRAST_SPREAD_OFF or stats.saturation > FLAT_PROFILE_SATURATION:
        return "clear"
    veil = _haze_veil_factor(stats)
    if veil >= HAZE_FOGGY_VEIL_MIN:
        return "foggy"
    if veil >= HAZE_HAZY_VEIL_MIN:
        return "hazy"
    return "clear"


def _dynamic_range(stats: SegmentStats) -> str:
    spread = stats.white_point - stats.black_point
    if spread <= DYNAMIC_RANGE_FLAT_MAX:
        return "flat"
    if spread >= DYNAMIC_RANGE_WIDE_MIN:
        return "wide"
    return "normal"


def _color_reading(view: np.ndarray) -> tuple[float, float]:
    """Chroma-weighted hue census on the calibrated uint8 metering view.

    Returns ``(chroma_headroom, warm_fraction)``. Headroom is 1 minus the
    ``CHROMA_HEADROOM_PERCENTILE`` of per-pixel saturation on the grade
    engine's (max-min)/max basis; the warm fraction is the warm hue band's
    share of the frame's total chroma, so gray pixels never vote.
    """
    r = view[:, :, 0].astype(np.float32)
    g = view[:, :, 1].astype(np.float32)
    b = view[:, :, 2].astype(np.float32)
    rgb_max = np.maximum(np.maximum(r, g), b)
    chroma = rgb_max - np.minimum(np.minimum(r, g), b)

    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(rgb_max > 0, chroma / rgb_max, 0.0)
    headroom = 1.0 - float(np.percentile(sat, CHROMA_HEADROOM_PERCENTILE))

    total = float(chroma.sum())
    if total <= 0.0:
        return headroom, 0.0

    # Standard hexagonal hue; zero-chroma pixels get a placeholder divisor and
    # an arbitrary hue, harmless because their vote weight is exactly zero.
    safe_chroma = np.where(chroma > 0, chroma, 1.0)
    hue = 60.0 * np.where(
        rgb_max == r,
        (g - b) / safe_chroma % 6.0,
        np.where(rgb_max == g, (b - r) / safe_chroma + 2.0, (r - g) / safe_chroma + 4.0),
    )
    warm = float(chroma[(hue >= HUE_WARM_MIN) | (hue <= HUE_WARM_MAX)].sum()) / total
    return headroom, warm


def compute_scene_profile(rgb: np.ndarray, exif: PhotoExif | Mapping | None = None) -> SceneProfile:
    """Classify one decoded frame's light. Pure and deterministic.

    ``rgb`` is a full-resolution uint8/uint16 HxWx3 array (a
    ``DecodedPhoto.rgb``); metering rides the develop stage's calibrated
    analysis seam, so uint8 and uint16 decodes of one frame profile
    identically. ``exif`` (dataclass or plain dict) is optional and
    caller-supplied — this function never reads files.
    """
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("compute_scene_profile expects an HxWx3 RGB array")
    if rgb.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"unsupported dtype {rgb.dtype}; expected uint8 or uint16")
    if exif is None:
        exif = PhotoExif()  # an all-None record profiles exactly as no EXIF
    elif isinstance(exif, Mapping):
        exif = PhotoExif(
            iso=exif.get("iso"),
            exposure_s=exif.get("exposure_s"),
            capture_hour=exif.get("capture_hour"),
            capture_utc=exif.get("capture_utc"),
            solar_elevation_deg=exif.get("solar_elevation_deg"),
        )

    # One calibrated metering view shared by the stats and the color census
    # (already uint8 and at scale, so _frame_stats' own view step is identity).
    view = _bounded_resize(_analysis_rgb(rgb), ANALYSIS_MAX_DIM)
    stats = _frame_stats(view)
    headroom, warm = _color_reading(view)
    return SceneProfile(
        exposure_class=_exposure_class(stats),
        warmth=_warmth(stats),
        sat_profile=_sat_profile(stats.saturation),
        haze=_haze(stats),
        dynamic_range=_dynamic_range(stats),
        is_near_monochrome=stats.saturation <= NEAR_MONO_SAT_MAX,
        chroma_headroom=headroom,
        warm_fraction=warm,
        capture_hour=exif.capture_hour,
        solar_elevation_deg=exif.solar_elevation_deg,
    )
