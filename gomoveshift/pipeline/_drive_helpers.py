"""Shared helper functions for the drive pipeline.

Extracted from drive_orchestrator.py so they can be imported by both the
orchestrator shim and the individual phase modules without circular imports.
"""

import logging
import math
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def _bearing_to_cardinal(bearing_degrees: float) -> str:
    """Map a bearing in degrees to an 8-point compass label.

    Bearings follow the compass convention: 0 deg = North, increasing
    clockwise (90 deg = East, 180 deg = South, 270 deg = West). Any numeric
    input is accepted -- values are wrapped into [0, 360).
    """
    # Normalize into [0, 360)
    angle = bearing_degrees % 360
    directions = [
        (22.5, "N"),
        (67.5, "NE"),
        (112.5, "E"),
        (157.5, "SE"),
        (202.5, "S"),
        (247.5, "SW"),
        (292.5, "W"),
        (337.5, "NW"),
    ]
    for cutoff, label in directions:
        if angle < cutoff:
            return label
    return "N"


def _compass_from_waypoints(
    start_lat: float, start_lon: float, end_lat: float, end_lon: float
) -> str:
    """8-point compass for the vector from (start_lat, start_lon) to (end_lat, end_lon)."""
    dlat = end_lat - start_lat
    dlon = end_lon - start_lon
    if dlat == 0 and dlon == 0:
        return "N"
    # Scale the E-W component by cos(mean latitude): a longitude degree spans
    # less ground than a latitude degree away from the equator, so without this
    # the east-west leg is over-weighted and a true-North heading skews toward NE.
    dlon *= math.cos(math.radians((start_lat + end_lat) / 2.0))
    # Convert atan2 result to compass bearing: 0deg = North, clockwise positive
    bearing = math.degrees(math.atan2(dlon, dlat))
    return _bearing_to_cardinal(bearing)


def _extract_episode_ost_meta(gpx_path) -> dict:
    """Read first/last waypoint coordinates and recording date from a gpx.

    Returns a dict with keys 'start', 'end', 'date' suitable for passing to
    differentiate_duplicate_ost. Missing/unreadable files return an empty
    metadata dict with zero-valued defaults -- duplicate detection will then
    fall through to date or index differentiation.
    """
    if not gpx_path:
        return {"start": (0.0, 0.0), "end": (0.0, 0.0), "date": None}

    try:
        from gps_journey.core.gpx_parser import GPXParser

        from .journey_filter import local_day

        parser = GPXParser()
        journey = parser.parse(Path(gpx_path))
        wps = list(journey.waypoints or [])
        # Drop null-island sentinels (a GPS cold-start writes a timestamped
        # (0, 0) fix): a (0, 0) endpoint paired with a real one would seed a
        # genuine-looking compass bearing from the Gulf of Guinea, and a (0, 0)
        # first fix has no usable local day. Keep only real fixes.
        real = [
            wp
            for wp in wps
            if not (abs(wp.latitude or 0.0) < 0.001 and abs(wp.longitude or 0.0) < 0.001)
        ]
        if not real:
            return {"start": (0.0, 0.0), "end": (0.0, 0.0), "date": None}
        first = real[0]
        last = real[-1]
        # Date suffix is the drive's LOCAL calendar day, not the raw UTC first-fix
        # date: a 23:30-local start records after UTC midnight, so the bare UTC
        # date would burn tomorrow's day on an otherwise-correct OST.
        ts = getattr(first, "timestamp", None)
        day = local_day(ts, first.longitude) if ts is not None else None
        return {
            "start": (first.latitude, first.longitude),
            "end": (last.latitude, last.longitude),
            "date": day,
        }
    except Exception as e:
        logger.debug("  _extract_episode_ost_meta failed for %s: %s", gpx_path, e)
        return {"start": (0.0, 0.0), "end": (0.0, 0.0), "date": None}


def differentiate_duplicate_ost(locations: list[str], metadata: list[dict]) -> list[str]:
    """Break ties between episodes with identical OST text.

    Args:
        locations: per-episode OST strings (may contain duplicates or empties)
        metadata: per-episode dicts with at least 'start', 'end' (lat, lon
            tuples) and 'date' (datetime). Must match locations 1:1.

    Returns:
        New list of OST strings where any non-empty duplicates have been
        differentiated via (1) direction suffix or (2) recording date suffix.
        Empty strings are never modified and are not considered duplicates.
    """
    if len(locations) != len(metadata):
        raise ValueError("locations and metadata must have equal length")

    result = list(locations)

    # Group non-empty locations by their current text
    counts = Counter(loc for loc in result if loc)
    dup_keys = {loc for loc, n in counts.items() if n >= 2}
    if not dup_keys:
        return result

    def _is_null_island(meta: dict) -> bool:
        # _extract_episode_ost_meta returns the (0.0, 0.0)/(0.0, 0.0) sentinel on
        # any GPX parse failure. _compass_from_waypoints(0,0,0,0) then returns a
        # real-looking "N", so a fabricated compass direction would burn into the
        # deduped OST pixels (e.g. "Kenmare (North)"). Treat it as "no bearing".
        start = meta.get("start") or (0.0, 0.0)
        end = meta.get("end") or (0.0, 0.0)
        return tuple(start) == (0.0, 0.0) and tuple(end) == (0.0, 0.0)

    for dup_key in dup_keys:
        indices = [i for i, loc in enumerate(result) if loc == dup_key]

        # Step 1: try direction suffix, but only when EVERY episode in the group
        # carries a real GPX vector — a single null-island sentinel would
        # otherwise get a fabricated bearing that reads as genuine. When any
        # member is a sentinel, skip straight to the date/index fallback below.
        directions = {}
        if not any(_is_null_island(metadata[i]) for i in indices):
            for i in indices:
                meta = metadata[i]
                start = meta.get("start") or (0.0, 0.0)
                end = meta.get("end") or (0.0, 0.0)
                directions[i] = _compass_from_waypoints(start[0], start[1], end[0], end[1])

        if directions and len(set(directions.values())) == len(indices):
            # All episodes have distinct directions -- apply suffix
            for i in indices:
                result[i] = f"{dup_key} ({directions[i]})"
            continue

        # Step 2: fall back to a recording-date suffix ("Feb 22"), but only when
        # every episode carries a distinct date. Same-day duplicates (or any
        # missing date) would collide again and re-trip the preflight uniqueness
        # gate, so fall through to a guaranteed-unique index suffix instead.
        date_strs = {}
        for i in indices:
            d = metadata[i].get("date")
            date_strs[i] = d.strftime("%b %d").replace(" 0", " ") if d is not None else None
        values = [date_strs[i] for i in indices]
        if all(v is not None for v in values) and len(set(values)) == len(indices):
            for i in indices:
                result[i] = f"{dup_key} ({date_strs[i]})"
        else:
            for i in indices:
                result[i] = f"{dup_key} #{i + 1}"

    return result
