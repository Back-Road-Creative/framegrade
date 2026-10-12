"""Source-clock telemetry payloads for Route Story preview and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Sequence

from .metrics import RoutePoint
from .timeline import StoryTimeline

_EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class FrameState:
    """Facts shown at one output frame; unknown remains ``None``."""

    recording_time_s: float
    latitude: float | None
    longitude: float | None
    elevation_m: float | None
    speed_mps: float | None
    grade_percent: float | None
    # Cumulative route distance at this frame -- 0 before the first observed
    # time, the route total after the last, and held at the previous
    # segment's end distance through a gap between segments. Matches
    # ``ProfileGeometry.cursor_distance_m`` exactly, so the HUD's profile
    # cursor and this field can never drift apart.
    distance_m: float
    data_available: bool


def frame_state_at(
    timeline: StoryTimeline, frame: int, segments: Sequence[Sequence[RoutePoint]]
) -> FrameState:
    """Resolve a frame through the storyboard onto one observed track edge.

    Segments are searched independently. A time between their endpoints has no
    position, even if a straight interpolation would look visually smooth.
    """
    recording_time_s = timeline.route_time_at_frame(frame)
    ordered = [sorted(segment, key=lambda point: point.time_s) for segment in segments if segment]
    offsets, total_distance_m = _segment_distance_offsets(ordered)
    for points, offset in zip(ordered, offsets):
        state = _state_in_segment(recording_time_s, points, offset)
        if state is not None:
            return state
    distance_m = _gap_distance_m(recording_time_s, ordered, offsets, total_distance_m)
    return FrameState(recording_time_s, None, None, None, None, None, distance_m, False)


def _segment_distance_offsets(
    segments: Sequence[Sequence[RoutePoint]],
) -> tuple[list[float], float]:
    """Cumulative distance travelled before each (already time-sorted) segment.

    Distance accumulates only between consecutive points of the same source
    segment; the gap between two source segments (a recorder pause, a ferry,
    a device swap) adds none of its own -- the same rule
    ``ProfileGeometry.cursor_distance_m`` applies to the HUD's profile cursor.
    """
    offsets: list[float] = []
    cumulative = 0.0
    for points in segments:
        offsets.append(cumulative)
        for start, end in zip(points, points[1:]):
            cumulative += _distance_m(start, end)
    return offsets, cumulative


def _gap_distance_m(
    time_s: float,
    segments: Sequence[Sequence[RoutePoint]],
    offsets: Sequence[float],
    total_distance_m: float,
) -> float:
    """``distance_m`` for a time no segment resolves a position for.

    Before the first observed time this is 0; after the last it is the
    route's total distance; inside a gap between two segments it holds at
    the previous segment's end distance rather than drifting forward
    through the gap or snapping ahead to the next segment.
    """
    if not segments:
        return 0.0
    if time_s <= segments[0][0].time_s:
        return 0.0
    if time_s >= segments[-1][-1].time_s:
        return total_distance_m
    for points, offset in zip(segments, offsets):
        if time_s < points[0].time_s:
            return offset
    return total_distance_m  # pragma: no cover -- ranges above already exhaust time_s


def _state_in_segment(
    time_s: float, points: Sequence[RoutePoint], offset: float
) -> FrameState | None:
    """``points`` must already be sorted by ``time_s``; ``offset`` is the
    cumulative distance travelled before this segment starts."""
    if not points or time_s < points[0].time_s or time_s > points[-1].time_s:
        return None
    if len(points) == 1:
        point = points[0]
        if time_s == point.time_s:
            return FrameState(
                time_s, point.latitude, point.longitude, point.elevation_m, None, None, offset, True
            )
        return None
    cumulative = offset
    for start, end in zip(points, points[1:]):
        duration_s = end.time_s - start.time_s
        if duration_s <= 0:
            raise ValueError("route segment timestamps must strictly increase")
        edge_distance_m = _distance_m(start, end)
        if start.time_s <= time_s <= end.time_s:
            fraction = (time_s - start.time_s) / duration_s
            elevation_m = _interpolate_optional(start.elevation_m, end.elevation_m, fraction)
            grade_percent = None
            if (
                start.elevation_m is not None
                and end.elevation_m is not None
                and edge_distance_m > 0
            ):
                grade_percent = 100 * (end.elevation_m - start.elevation_m) / edge_distance_m
            return FrameState(
                time_s,
                start.latitude + (end.latitude - start.latitude) * fraction,
                start.longitude + (end.longitude - start.longitude) * fraction,
                elevation_m,
                edge_distance_m / duration_s,
                grade_percent,
                cumulative + fraction * edge_distance_m,
                True,
            )
        cumulative += edge_distance_m
    raise AssertionError("time within a valid segment did not resolve to an edge")


def _interpolate_optional(start: float | None, end: float | None, fraction: float) -> float | None:
    if start is None or end is None:
        return None
    return start + (end - start) * fraction


def _distance_m(start: RoutePoint, end: RoutePoint) -> float:
    lat_a, lon_a, lat_b, lon_b = map(
        radians, (start.latitude, start.longitude, end.latitude, end.longitude)
    )
    delta_lat = lat_b - lat_a
    delta_lon = lon_b - lon_a
    h = sin(delta_lat / 2) ** 2 + cos(lat_a) * cos(lat_b) * sin(delta_lon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * asin(sqrt(h))
