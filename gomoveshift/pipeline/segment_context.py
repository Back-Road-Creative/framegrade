"""
Per-short segment context from GPS data.

Provides terrain classification, nearest POI, named route, and time-of-day
so social captions can reference what's actually in the clip.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class SegmentContext:
    """GPS-derived context for a single short segment."""

    terrain_type: Optional[str] = None  # "coastal", "mountain", "winding", "countryside"
    nearest_poi: Optional[str] = None  # "Caherdaniel", "Moll's Gap"
    named_route: Optional[str] = None  # "Ring of Kerry", "Wild Atlantic Way"
    time_of_day: Optional[str] = None  # "morning", "afternoon", "golden hour"
    scenic_score: Optional[float] = None  # 0-100
    elevation_m: Optional[float] = None  # average elevation in meters

    @property
    def location_detail(self) -> str:
        """Best available location specificity for captions."""
        return self.nearest_poi or self.named_route or ""

    @property
    def has_context(self) -> bool:
        """Whether any meaningful context is available."""
        return bool(self.terrain_type or self.nearest_poi or self.named_route)


# Terrain-classification cuts. NOTE the TRUE units of the inputs — an earlier
# docstring wrongly labelled them per-km. Both production callers feed
# ScenicScorer.get_elevation_gain_rate (metres of climb per MINUTE) and
# get_turn_intensity (an already-normalised 0-100 winding score), so the cuts
# are calibrated to those quantities, not to per-km values.
_MOUNTAIN_GAIN_M_PER_MIN = 20.0  # sustained climb rate (m/min)
_MOUNTAIN_MIN_ELEVATION_M = 200.0  # absolute-height gate (m)
# Net-elevation-range floor for the mountain class. get_elevation_gain_rate SUMS
# every positive consecutive delta, so GPS elevation jitter on a flat plateau
# above 200 m accumulates past the gain cut and ships "#MountainPass". A real
# climb has a large NET range (max-min); flat-with-noise is near-zero. Mirrors
# the chapter-band classifier's fix.
_MOUNTAIN_MIN_NET_RANGE_M = 100.0
_WINDING_TURN_SCORE = 40.0  # get_turn_intensity 0-100 score


def classify_terrain(
    elevation_gain_rate: float,
    elevation_m: float,
    turn_rate: float = 0.0,
    net_elevation_range_m: float = 0.0,
    near_coast: bool = False,
) -> str:
    """Classify terrain from GPS metrics.

    Args:
        elevation_gain_rate: Climb rate in metres per MINUTE (as returned by
            ScenicScorer.get_elevation_gain_rate) — NOT per-km.
        elevation_m: Average elevation in metres over the segment.
        turn_rate: Winding score in [0, 100] (as returned by
            ScenicScorer.get_turn_intensity) — a normalised intensity, NOT
            degrees-per-km.
        net_elevation_range_m: max-min elevation over the segment (metres). The
            mountain class requires this to clear ``_MOUNTAIN_MIN_NET_RANGE_M``
            so summed GPS jitter on a high flat plateau can't fake a climb.
            Defaults to 0.0, which (correctly) blocks "mountain" for any caller
            that cannot supply a real net range.
        near_coast: True only when the segment was verified within ~1 km of an
            OSM ``natural=coastline`` (the ``_coastline_points_for_bbox``
            primitive used by chapter bands). Low elevation is NOT evidence of
            the coast — inland river valleys, fens and floodplains sit below
            30 m — so "coastal" is emitted ONLY on this positive signal, never
            inferred from ``elevation_m``.

    Returns:
        One of: "mountain", "coastal", "winding", "countryside"
    """
    if (
        elevation_gain_rate > _MOUNTAIN_GAIN_M_PER_MIN
        and elevation_m > _MOUNTAIN_MIN_ELEVATION_M
        and net_elevation_range_m >= _MOUNTAIN_MIN_NET_RANGE_M
    ):
        return "mountain"
    if near_coast:
        return "coastal"
    if turn_rate > _WINDING_TURN_SCORE:
        return "winding"
    return "countryside"


def classify_time_of_day(timestamp: float) -> str:
    """Classify time from Unix timestamp to descriptive period.

    Args:
        timestamp: Unix timestamp (UTC)

    Returns:
        One of: "dawn", "morning", "midday", "afternoon", "golden hour", "evening"
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    hour = dt.hour
    if hour < 7:
        return "dawn"
    if hour < 10:
        return "morning"
    if hour < 14:
        return "midday"
    if hour < 17:
        return "afternoon"
    if hour < 20:
        return "golden hour"
    return "evening"


def find_nearest_poi(
    video_time: float,
    pois: list[dict],
    gps_start_time: float,
    time_scale: float,
) -> Optional[str]:
    """Find the nearest POI to a video timestamp.

    Args:
        video_time: Time in video (seconds)
        pois: List of dicts with "gps_time" and "name" keys
        gps_start_time: GPS start timestamp
        time_scale: Ratio of GPS duration to video duration

    Returns:
        Name of nearest POI, or None
    """
    if not pois:
        return None

    gps_time = gps_start_time + video_time * time_scale
    best_name = None
    best_dist = float("inf")

    for poi in pois:
        poi_time = poi.get("gps_time", 0)
        dist = abs(gps_time - poi_time)
        if dist < best_dist:
            best_dist = dist
            best_name = poi.get("name")

    # Only return if within 5 minutes GPS time
    if best_dist < 300:
        return best_name
    return None
