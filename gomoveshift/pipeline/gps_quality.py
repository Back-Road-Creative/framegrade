"""GPS-quality filter, road-selection, and clip-timeline trim helpers.

Combines the quality filter, significant-road selection and the real
``trim_gps_to_clip_timeline`` implementation.

* ``filter_high_quality_points`` drops noisy GPS (null-island
  cold-start fixes, spatial/temporal teleports, sub-second
  duplicates, high HDOP, stationary jitter) before centroid /
  reverse-geocode use. lat/lon/time are read via tuple-aware
  accessors so the working passes apply to the REAL point types
  the pipeline feeds it — positional ``(t, lat, lon)`` tuples
  (scorer + shorts-OST trace) and ``GPSTrackpoint`` objects — not
  only dicts. HDOP/speed gates stay attribute-only:
  GoPro GPMF tuples + ``GPSTrackpoint`` carry neither, so those
  two passes are honestly inert on those types.
* ``select_significant_roads`` reduces a road-segment timeline to
  the meaningful few (by duration + length), preserving
  chronological order, capped at ``max_count``. Supports the
  multi-road title format.
* ``trim_gps_to_clip_timeline`` restricts a GPS-point sequence to
  the source-timeline ranges that actually survived editing. Used
  by chapters + shorts OST so those surfaces describe what made
  it into the clip — not the raw source footage.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

# GoPro GPS cold-start anomaly: camera may report (0,0) before lock, then briefly
# lock onto a wrong satellite cluster thousands of km from the actual route. Reject
# both: null-island sentinels and spatial outliers beyond this radius from the median
# cluster of surviving points.
_SPATIAL_OUTLIER_KM = 500.0

# Physical speed ceiling for the temporal teleport test. A
# consecutive-fix jump is a teleport ONLY when its IMPLIED SPEED (km / dt)
# exceeds this — never raw km alone. 25 km over a 10-min ferry/tunnel/bridge
# dropout is ~150 km/h (legit); the same 25 km over ~1 s is ~90,000 km/h (a
# cold-start stale lock). 1000 km/h sits well above any ground/ferry consumer
# GPS yet far below an instantaneous teleport, so legit far endpoints over long
# dropouts survive and only physically-impossible seams strip.
_TELEPORT_SPEED_CEILING_KMH = 1000.0

# A seam must clear BOTH a spatial floor and a multiple of the median step
# before the speed test applies — mirrors ``chapters.generator`` so ordinary
# fast-fix noise never trips the teleport strip.
_TELEPORT_JUMP_FLOOR_KM = 25.0
_TELEPORT_K = 20.0


def exceeds_physical_speed(
    km: float, dt_sec: Optional[float], *, ceiling_kmh: float = _TELEPORT_SPEED_CEILING_KMH
) -> bool:
    """Return True iff a ``km`` jump over ``dt_sec`` implies an impossible speed.

    Shared teleport discriminator for both the chapters journey strip and the
    label-path quality filter. ``dt_sec`` of ``None`` or ``<= 0`` means
    the spacing is unknown — return ``False`` (cannot prove a teleport on time
    alone; the caller's spatial guard still applies).
    """
    if dt_sec is None or dt_sec <= 0:
        return False
    return (km / (dt_sec / 3600.0)) > ceiling_kmh


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(min(a, 1.0)))


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _get(point: Any, *names: str, default: Any = None) -> Any:
    """Return the first attribute or item that exists on ``point``.

    GPS points are sometimes dicts (from parsed GPX) and sometimes
    SimpleNamespace / model objects — this helper unifies both.
    """
    for name in names:
        if isinstance(point, dict):
            if name in point and point[name] is not None:
                return point[name]
        else:
            val = getattr(point, name, None)
            if val is not None:
                return val
    return default


def _point_lat(point: Any) -> Any:
    """Latitude of a dict / object / positional-tuple GPS point.

    ``_get`` returns ``None`` on a positional tuple, so the
    null-island/spatial/teleport gates were a silent no-op on the
    ``(t, lat, lon)`` scorer + shorts-OST trace shape. Mirror
    ``title_builder._point_latlon``'s tuple convention here (lat at index 1
    for ``(t, lat, lon)``, index 0 for a bare ``(lat, lon)`` 2-tuple) so the
    working passes apply to the REAL types regardless of caller.
    """
    if isinstance(point, (tuple, list)):
        if len(point) >= 3:
            return point[1]
        if len(point) == 2:
            return point[0]
        return None
    # ``lat`` covers dicts + scorer/short models; ``latitude`` covers the GPX
    # ``Waypoint`` model, whose full-word attrs would otherwise read as None and
    # silently disable the null-island/spatial/teleport gates on that shape.
    return _get(point, "lat", "latitude")


def _point_lon(point: Any) -> Any:
    """Longitude of a dict / object / positional-tuple GPS point (see
    :func:`_point_lat`). Tuple lon is index 2 for ``(t, lat, lon)``, index 1
    for a bare ``(lat, lon)``."""
    if isinstance(point, (tuple, list)):
        if len(point) >= 3:
            return point[2]
        if len(point) == 2:
            return point[1]
        return None
    return _get(point, "lon", "lng", "longitude")


def _point_time(point: Any) -> Any:
    """Source-timeline time of a GPS point. The positional-tuple shape carries
    the source second at index 0 (``(t, lat, lon)``); a bare ``(lat, lon)``
    2-tuple has no time. Dict/object points read ``time``/``timestamp``."""
    if isinstance(point, (tuple, list)):
        return point[0] if len(point) >= 3 else None
    return _get(point, "time", "timestamp")


def filter_high_quality_points(
    gps_points: list,
    *,
    max_hdop: float = 3.0,
    min_speed_mps: float = 1.0,
    min_gap_sec: float = 1.0,
) -> list:
    """Drop GPS points failing quality thresholds.

    Args:
        gps_points: Iterable of GPS point records (dict or object).
        max_hdop: Drop points with HDOP > threshold. Missing HDOP kept.
        min_speed_mps: Drop points with speed < threshold. Missing kept.
        min_gap_sec: Drop the later of any two points within this gap.

    Returns:
        New list of surviving points in original order.
    """
    if not gps_points:
        return []

    survivors: list = []
    last_ts: float | None = None

    for p in gps_points:
        # Reject GPS no-lock sentinel: camera reports (0, 0) before acquiring a fix.
        lat_raw = _point_lat(p)
        lon_raw = _point_lon(p)
        if lat_raw is not None and lon_raw is not None:
            try:
                if float(lat_raw) == 0.0 and float(lon_raw) == 0.0:
                    continue
            except (TypeError, ValueError):
                pass

        hdop = _get(p, "hdop")
        if hdop is not None:
            try:
                if float(hdop) > max_hdop:
                    continue
            except (TypeError, ValueError):
                pass

        speed = _get(p, "speed_mps", "speed")
        if speed is not None:
            try:
                if float(speed) < min_speed_mps:
                    continue
            except (TypeError, ValueError):
                pass

        ts = _point_time(p)
        ts_sec: float | None
        if ts is None:
            ts_sec = None
        elif hasattr(ts, "timestamp"):
            try:
                ts_sec = ts.timestamp()
            except Exception:  # noqa: BLE001
                ts_sec = None
        else:
            try:
                ts_sec = float(ts)
            except (TypeError, ValueError):
                ts_sec = None

        if ts_sec is not None and last_ts is not None:
            if (ts_sec - last_ts) < min_gap_sec:
                continue
        if ts_sec is not None:
            last_ts = ts_sec

        survivors.append(p)

    # Spatial outlier pass: reject points that are geographically distant from the
    # median cluster. Handles GoPro cold-start anomalies where the camera briefly
    # acquires a wrong satellite fix (e.g. Bahamas coords on a NC→PA drive).
    if len(survivors) >= 2:
        lats: list[float] = []
        lons: list[float] = []
        for p in survivors:
            lat_v = _point_lat(p)
            lon_v = _point_lon(p)
            if lat_v is None or lon_v is None:
                continue
            try:
                lats.append(float(lat_v))
                lons.append(float(lon_v))
            except (TypeError, ValueError):
                pass
        if lats:
            med_lat = _median(lats)
            med_lon = _median(lons)
            filtered: list = []
            for p in survivors:
                lat_v = _point_lat(p)
                lon_v = _point_lon(p)
                if lat_v is None or lon_v is None:
                    filtered.append(p)
                    continue
                try:
                    dist = _haversine_km(float(lat_v), float(lon_v), med_lat, med_lon)
                    if dist <= _SPATIAL_OUTLIER_KM:
                        filtered.append(p)
                    else:
                        logger.debug(
                            "GPS spatial outlier dropped: lat=%.4f lon=%.4f dist=%.0f km from median",
                            float(lat_v),
                            float(lon_v),
                            dist,
                        )
                except (TypeError, ValueError):
                    filtered.append(p)
            survivors = filtered

    # Temporal teleport strip: drop a contiguous head/tail run joined
    # to the body by a consecutive-fix jump whose IMPLIED SPEED is physically
    # impossible. This is the same km/SEC discriminator the chapters journey uses
    # (``chapters.generator._strip_teleport_runs``) — ported here so the label
    # path no longer relies solely on the coarse 500 km median test, which both
    # mis-keeps a sub-500 km stale lock and could split a legit cross-state drive.
    survivors = _strip_temporal_teleport_runs(survivors)

    return survivors


def _point_ts_sec(point: Any) -> Optional[float]:
    """Best-effort source-timeline second for a label-path GPS point."""
    ts = _point_time(point)
    if ts is None:
        return None
    if hasattr(ts, "timestamp"):
        try:
            return float(ts.timestamp())
        except Exception:  # noqa: BLE001
            return None
    try:
        return float(ts)
    except (TypeError, ValueError):
        return None


def locked_gps_coverage(
    gps_points: Optional[Sequence],
    window_start_sec: float,
    window_end_sec: float,
    *,
    expected_hz: float = 1.0,
) -> float:
    """Fraction (0.0–1.0) of an LF's footage window carrying a locked GPS fix.

    Every trackpoint in a parsed GPX is a locked fix by construction — the GoPro
    ingest only writes ``gpsfix in GPS_FIXED_VALUES`` into the canonical GPX, and
    the GPX carries no fix channel — so "locked samples" is simply the count of
    supplied points whose timestamp falls inside ``[window_start_sec,
    window_end_sec]``. Coverage is bucketed at ``expected_hz`` (1 Hz = the GoPro
    GPX step) so it is robust to an over-sampled trace (caps at 1.0) and reflects
    dead/sparse stretches (missing buckets lower it). Points/time bases MUST be
    the same clock as the window (unix seconds for GPX ``GPSTrackpoint`` /
    ``{"time": ...}`` dicts; source seconds for ``(t, lat, lon)`` tuples).

    This is the demotion signal the specificity ladder awards tiers on
    (``title_builder.award_tier``) and the stable export other callers import. Returns
    ``0.0`` for an empty trace or a degenerate (non-positive-span) window.
    """
    if expected_hz <= 0:
        raise ValueError("expected_hz must be positive")
    if window_end_sec <= window_start_sec:
        return 0.0
    span = window_end_sec - window_start_sec
    expected = max(1, round(span * expected_hz))
    bucket_sec = 1.0 / expected_hz
    covered: set[int] = set()
    for p in gps_points or []:
        t = _point_ts_sec(p)
        if t is None or t < window_start_sec or t > window_end_sec:
            continue
        idx = int((t - window_start_sec) / bucket_sec)
        covered.add(min(idx, expected - 1))
    return min(1.0, len(covered) / expected)


def locked_gps_duration_coverage(
    gps_points: Optional[Sequence],
    duration_sec: float,
    *,
    expected_hz: float = 1.0,
) -> float:
    """Locked-GPS coverage against an LF's FOOTAGE DURATION.

    :func:`locked_gps_coverage` needs a ``[window_start, window_end]`` in the
    SAME clock as the points — but the evidence site holds epoch-clock fixes
    and episode-media-second segment plans, and a stop-removed LF's footage is
    non-contiguous in either clock (a single hull window would count removed
    stops as dead GPS). Duration sidesteps both: covered buckets are the
    DISTINCT ``expected_hz`` buckets among the supplied points' own timestamps
    (clock-irrelevant), and the denominator is the footage duration — so a
    dead prefix/suffix or interior dropout lowers coverage while removed stops
    (whose fixes were never collected) can't. Callers supply only points that
    fall inside footage segments. Returns ``0.0`` on a non-positive duration.
    """
    if expected_hz <= 0:
        raise ValueError("expected_hz must be positive")
    if duration_sec <= 0:
        return 0.0
    expected = max(1, round(duration_sec * expected_hz))
    covered = {
        int(t * expected_hz) for t in (_point_ts_sec(p) for p in gps_points or []) if t is not None
    }
    return min(1.0, len(covered) / expected)


# ── Specificity-ladder evidence accessors — point-shape handling in one
# home so content_gates reads only these public helpers. ──────────────────
def _in_window_coords(gps_points, w0: float, w1: float) -> list[tuple[float, float, float]]:
    """``[(t, lat, lon), ...]`` for fixes with usable lat/lon/time in ``[w0, w1]``."""
    out: list[tuple[float, float, float]] = []
    for p in gps_points or []:
        t, lat, lon = _point_ts_sec(p), _point_lat(p), _point_lon(p)
        if t is None or lat is None or lon is None or t < w0 or t > w1:
            continue
        try:
            out.append((float(t), float(lat), float(lon)))
        except (TypeError, ValueError):
            continue
    return out


def trace_time_window(gps_points: Optional[Sequence]) -> Optional[tuple[float, float]]:
    """``(min_ts, max_ts)`` over usable timestamps, else ``None`` (<2 fixes) — the
    per-LF window the ladder derives from its own fixes when none is supplied."""
    ts = [t for t in (_point_ts_sec(p) for p in gps_points or []) if t is not None]
    return (min(ts), max(ts)) if len(ts) >= 2 else None


def usable_fix_times(gps_points: Optional[Sequence]) -> list[float]:
    """Sorted timestamps of the fixes with usable lat/lon/time — the fixes a
    short's place vote can count."""
    return sorted(t for t, _, _ in _in_window_coords(gps_points, float("-inf"), float("inf")))


def window_median_speed_mps(gps_points, w0: float, w1: float) -> Optional[float]:
    """Median ground speed (m/s) between consecutive in-window fixes (haversine /
    dt, since GPX has no speed channel), else ``None`` — the stationary guard."""
    pts = sorted(_in_window_coords(gps_points, w0, w1))
    speeds = [
        _haversine_km(la0, lo0, la1, lo1) * 1000.0 / (t1 - t0)
        for (t0, la0, lo0), (t1, la1, lo1) in zip(pts, pts[1:])
        if t1 > t0
    ]
    return _median(speeds) if speeds else None


def window_stationary_fraction(
    gps_points, w0: float, w1: float, *, stop_speed_mps: float
) -> Optional[tuple[float, int]]:
    """``(stationary_fraction, interval_count)`` over a window, else ``None``.

    The per-interval analog of :func:`window_median_speed_mps`: the fraction of
    consecutive in-window fix intervals whose ground speed is below
    ``stop_speed_mps`` (m/s), plus the interval count. ``None`` when fewer than
    two in-window fixes exist (unmeasurable — the caller must NOT auto-pass it).
    Robust to GPS sampling density (a bare median can't express an arbitrary
    majority threshold), so it is the stop-free gate's majority-stationary
    measure."""
    pts = sorted(_in_window_coords(gps_points, w0, w1))
    speeds = [
        _haversine_km(la0, lo0, la1, lo1) * 1000.0 / (t1 - t0)
        for (t0, la0, lo0), (t1, la1, lo1) in zip(pts, pts[1:])
        if t1 > t0
    ]
    if not speeds:
        return None
    stopped = sum(1 for s in speeds if s < stop_speed_mps)
    return stopped / len(speeds), len(speeds)


def nearest_trace_distance_km(lat: float, lon: float, gps_points, w0: float, w1: float):
    """Smallest great-circle distance (km) from ``(lat, lon)`` to any in-window fix,
    else ``None`` — the ladder's endpoint-proximity check reads it."""
    ds = [_haversine_km(lat, lon, la, lo) for _t, la, lo in _in_window_coords(gps_points, w0, w1)]
    return min(ds) if ds else None


def _strip_temporal_teleport_runs(survivors: list) -> list:
    """Strip a leading/trailing run separated by a physically-impossible seam.

    Mirrors ``chapters.generator._strip_teleport_runs`` but for the label path's
    dict/object points: a seam is a teleport only when its implied speed exceeds
    ``_TELEPORT_SPEED_CEILING_KMH`` (via :func:`exceeds_physical_speed`), so a big
    jump over a long GPS dropout is preserved while a sub-second jump is dropped.
    Never strips an interior point and keeps ≥2 points.
    """
    n = len(survivors)
    if n < 3:
        return survivors

    coords: list[Optional[tuple[float, float, Optional[float]]]] = []
    for p in survivors:
        lat_v = _point_lat(p)
        lon_v = _point_lon(p)
        if lat_v is None or lon_v is None:
            coords.append(None)
            continue
        try:
            coords.append((float(lat_v), float(lon_v), _point_ts_sec(p)))
        except (TypeError, ValueError):
            coords.append(None)

    # A seam must ALSO be a large spatial jump relative to the typical step
    # (mirrors the chapters ``20× median`` gate), so ordinary fast-fix noise
    # never trips the speed test.
    step_km: list[Optional[float]] = []
    for i in range(n - 1):
        a, b = coords[i], coords[i + 1]
        step_km.append(None if a is None or b is None else _haversine_km(a[0], a[1], b[0], b[1]))
    known = sorted(s for s in step_km if s is not None)
    if not known:
        return survivors
    median_step = known[len(known) // 2]
    seam_threshold = max(_TELEPORT_JUMP_FLOOR_KM, _TELEPORT_K * median_step)

    def _seam(i: int) -> bool:
        a, b = coords[i], coords[i + 1]
        km = step_km[i]
        if a is None or b is None or km is None or km <= seam_threshold:
            return False
        dt = None if a[2] is None or b[2] is None else (b[2] - a[2])
        return exceeds_physical_speed(km, dt)

    max_strip = max(1, int(n * 0.05))
    start = 0
    for i in range(min(max_strip, n - 1)):
        if _seam(i):
            start = i + 1
            break
    end = n
    for j in range(min(max_strip, n - 1)):
        if _seam(n - 2 - j):
            end = n - 1 - j
            break

    if start == 0 and end == n:
        return survivors
    if end - start < 2:
        return survivors
    if start or end != n:
        logger.debug(
            "GPS temporal teleport strip: dropped %d head + %d tail point(s)",
            start,
            n - end,
        )
    return survivors[start:end]


def select_significant_roads(
    road_segments: list,
    *,
    min_duration_sec: float = 30.0,
    min_length_m: float = 1000.0,
    max_count: int = 3,
) -> list:
    """Reduce a road-segment timeline to the significant few.

    Drops sub-threshold roads, caps to ``max_count`` by duration,
    returns survivors in chronological (input) order.
    """
    if not road_segments:
        return []

    prefiltered: list[tuple[int, Any]] = []
    for i, r in enumerate(road_segments):
        name = _get(r, "name")
        if not name:
            continue
        duration = _get(r, "duration_sec", "duration", default=0.0)
        length = _get(r, "length_m", "length", default=0.0)
        try:
            if float(duration) < min_duration_sec:
                continue
            if float(length) < min_length_m:
                continue
        except (TypeError, ValueError):
            continue
        prefiltered.append((i, r))

    if len(prefiltered) <= max_count:
        return [r for _, r in prefiltered]

    def _dur(entry: tuple[int, Any]) -> float:
        try:
            return float(_get(entry[1], "duration_sec", "duration", default=0.0))
        except (TypeError, ValueError):
            return 0.0

    top = sorted(prefiltered, key=_dur, reverse=True)[:max_count]
    top.sort(key=lambda entry: entry[0])
    return [r for _, r in top]


def _default_time_accessor(point):
    """Extract a source-timeline second from a point.

    Supports ``(t, *rest)`` tuples and objects exposing ``.t`` /
    ``.timestamp_sec`` / ``.time_sec`` / ``.source_sec``. Datetime
    waypoints must be projected via ``time_key`` on the caller side.
    """
    if isinstance(point, (tuple, list)) and point:
        return float(point[0])
    for attr in ("t", "timestamp_sec", "time_sec", "source_sec"):
        if hasattr(point, attr):
            return float(getattr(point, attr))
    raise TypeError(
        f"Cannot extract time from {type(point).__name__}; "
        "pass time_key= to trim_gps_to_clip_timeline."
    )


def trim_gps_to_clip_timeline(
    gps_points: Optional[Iterable],
    included_segments: Optional[Sequence[tuple]],
    *,
    time_key: Optional[Callable] = None,
) -> list:
    """Return points whose timestamp falls in any included range.

    Args:
        gps_points: Iterable of GPS points.
        included_segments: ``(start_sec, end_sec)`` ranges in
            source-timeline seconds. ``None`` is identity.
        time_key: Optional callable projecting a point to seconds.
            For datetime waypoints: ``lambda wp: (wp.timestamp -
            journey_start).total_seconds()``.

    Returns:
        List of points whose time lies in at least one range,
        preserving input order.
    """
    if gps_points is None:
        return []
    if included_segments is None:
        return list(gps_points)
    ranges = [(float(s), float(e)) for s, e in included_segments]
    if not ranges:
        return []
    accessor = time_key or _default_time_accessor
    kept: list = []
    for pt in gps_points:
        t = float(accessor(pt))
        for start, end in ranges:
            if start <= t <= end:
                kept.append(pt)
                break
    return kept
