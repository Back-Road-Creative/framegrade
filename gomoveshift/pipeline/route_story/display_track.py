"""Display-only smoothing of GPS zig-zags for the drawn Route Story line.

Consumer-grade GPS jitters side to side at walking pace, so the raw drawn
line saw-tooths across a path the hiker actually walked smoothly.  This
module produces a *display* track only: distance, pace, elevation gain and
the receipt are always computed from the original recorded points
(``metrics.py``, ``gpx.py``).  No smoothed point is ever written back or
reported as a measurement -- it exists purely to look right on screen.

Every output point keeps its original ``time_s`` untouched: the moving
marker is looked up by time against the same segments used to draw the
line, so a changed clock would reopen the "dot lags the line" bug.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from .metrics import RoutePoint, distance_m

DISPLAY_SMOOTHING_RECIPE_VERSION = "display-smooth-1"

_MIN_POINTS_TO_SMOOTH = 3


@dataclass(frozen=True)
class SmoothedTrack:
    """A display-only smoothed track plus the shift accounting for the receipt."""

    segments: list[list[RoutePoint]]
    max_shift_m_observed: float
    mean_shift_m: float
    points_clamped: int
    recipe: str


def _cumulative_distance_m(segment: Sequence[RoutePoint]) -> list[float]:
    cumulative = [0.0]
    for previous, current in zip(segment, segment[1:]):
        cumulative.append(cumulative[-1] + distance_m(previous, current))
    return cumulative


def _local_plane_xy(
    point: RoutePoint, ref_lat: float, ref_lon: float, meters_per_degree: float
) -> tuple[float, float]:
    x = (point.longitude - ref_lon) * meters_per_degree * math.cos(math.radians(ref_lat))
    y = (point.latitude - ref_lat) * meters_per_degree
    return x, y


def _smooth_segment(
    segment: Sequence[RoutePoint], window_m: float, max_shift_m: float
) -> tuple[list[RoutePoint], list[float], list[bool]]:
    if len(segment) < _MIN_POINTS_TO_SMOOTH:
        return list(segment), [0.0] * len(segment), [False] * len(segment)

    cumdist = _cumulative_distance_m(segment)
    ref_lat = sum(p.latitude for p in segment) / len(segment)
    ref_lon = sum(p.longitude for p in segment) / len(segment)
    meters_per_degree = math.pi / 180.0 * 6_371_008.8
    xs = []
    ys = []
    for point in segment:
        x, y = _local_plane_xy(point, ref_lat, ref_lon, meters_per_degree)
        xs.append(x)
        ys.append(y)

    n = len(segment)
    out: list[RoutePoint] = [segment[0]]
    shifts: list[float] = [0.0]
    clamped: list[bool] = [False]

    lo = 0
    hi = 0
    for i in range(1, n - 1):
        while cumdist[i] - cumdist[lo] > window_m:
            lo += 1
        while hi < n - 1 and cumdist[hi + 1] - cumdist[i] <= window_m:
            hi += 1

        # A distance-windowed (triangular kernel) weighted average of the
        # neighbouring points' positions, in a local metre-scale plane.
        # ``max_shift_m`` is what keeps this from rounding off a genuine
        # direction change (a switchback apex): the window necessarily
        # reaches into the far leg near a sharp turn, but the correction it
        # wants to apply there is large and gets clamped, so the apex still
        # only moves a bounded amount.
        weights = [
            max(0.0, 1.0 - abs(cumdist[j] - cumdist[i]) / window_m) for j in range(lo, hi + 1)
        ]
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            out.append(segment[i])
            shifts.append(0.0)
            clamped.append(False)
            continue

        mean_x = sum(w * xs[lo + k] for k, w in enumerate(weights)) / weight_sum
        mean_y = sum(w * ys[lo + k] for k, w in enumerate(weights)) / weight_sum

        dx = mean_x - xs[i]
        dy = mean_y - ys[i]
        shift = math.hypot(dx, dy)
        was_clamped = shift > max_shift_m
        if was_clamped:
            scale = max_shift_m / shift
            dx *= scale
            dy *= scale
            shift = max_shift_m

        new_lat = segment[i].latitude + dy / meters_per_degree
        new_lon = segment[i].longitude + dx / (meters_per_degree * math.cos(math.radians(ref_lat)))
        out.append(replace(segment[i], latitude=new_lat, longitude=new_lon))
        shifts.append(shift)
        clamped.append(was_clamped)

    out.append(segment[-1])
    shifts.append(0.0)
    clamped.append(False)
    return out, shifts, clamped


def smooth_display_segments(
    segments: Sequence[Sequence[RoutePoint]],
    *,
    window_m: float = 25.0,
    max_shift_m: float = 12.0,
) -> SmoothedTrack:
    """Smooth lat/lon zig-zags for display only, along-track windowed.

    Elevation, ``time_s`` and every other :class:`RoutePoint` field pass
    through untouched.  Segments never blend across their boundary, and the
    first/last point of every segment is pinned exactly (the start/finish
    pins and the video arrival point must not move). No output point moves
    more than ``max_shift_m`` from its recorded position -- points that
    would are clamped onto that radius.
    """
    if window_m <= 0:
        raise ValueError("window_m must be positive")
    if max_shift_m <= 0:
        raise ValueError("max_shift_m must be positive")

    out_segments: list[list[RoutePoint]] = []
    all_shifts: list[float] = []
    points_clamped = 0

    for source in segments:
        segment = list(source)
        smoothed, shifts, clamped = _smooth_segment(segment, window_m, max_shift_m)
        out_segments.append(smoothed)
        all_shifts.extend(shifts)
        points_clamped += sum(clamped)

    max_shift_observed = max(all_shifts) if all_shifts else 0.0
    mean_shift = sum(all_shifts) / len(all_shifts) if all_shifts else 0.0

    return SmoothedTrack(
        segments=out_segments,
        max_shift_m_observed=max_shift_observed,
        mean_shift_m=mean_shift,
        points_clamped=points_clamped,
        recipe=DISPLAY_SMOOTHING_RECIPE_VERSION,
    )
