"""Route Story telemetry summaries.

The presentation timeline is allowed to compress a journey; these facts are
always calculated from the original GPS clock.  Segments remain separate so a
GPS outage, ferry, or device swap cannot become a fictional travelled edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Iterable, Sequence

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class RoutePoint:
    """One accepted GPS observation in the source recording clock."""

    time_s: float
    latitude: float
    longitude: float
    elevation_m: float | None = None
    heart_rate_bpm: float | None = None
    cadence_rpm: float | None = None
    temperature_c: float | None = None

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be in [-90, 90]")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be in [-180, 180]")


@dataclass(frozen=True)
class RouteSummary:
    """Measurements for one selected route scope, stored in SI units.

    ``elapsed_seconds`` counts only observed source segments.  Time between
    segments is ``unobserved_gap_seconds`` rather than stopped time.  Elevation
    measurements are ``None`` when the source has no usable altitude evidence;
    zero means a measured zero, never a substitute for missing data.
    """

    distance_m: float
    elapsed_seconds: float
    moving_seconds: float
    stopped_seconds: float
    unobserved_gap_seconds: float
    current_elevation_m: float | None
    min_elevation_m: float | None
    max_elevation_m: float | None
    net_elevation_change_m: float | None
    ascent_m: float | None
    descent_m: float | None
    current_speed_mps: float | None
    maximum_speed_mps: float | None
    moving_average_speed_mps: float | None
    elapsed_average_speed_mps: float | None
    current_grade_percent: float | None
    net_average_grade_percent: float | None
    uphill_average_grade_percent: float | None
    downhill_average_grade_percent: float | None
    absolute_average_grade_percent: float | None


@dataclass(frozen=True)
class _Edge:
    start: RoutePoint
    end: RoutePoint
    distance_m: float
    duration_s: float

    @property
    def speed_mps(self) -> float:
        return self.distance_m / self.duration_s

    @property
    def elevation_delta_m(self) -> float | None:
        if self.start.elevation_m is None or self.end.elevation_m is None:
            return None
        return self.end.elevation_m - self.start.elevation_m


def distance_m(a: RoutePoint, b: RoutePoint) -> float:
    """Great-circle distance for two accepted fixes."""

    lat_a, lon_a, lat_b, lon_b = map(radians, (a.latitude, a.longitude, b.latitude, b.longitude))
    d_lat = lat_b - lat_a
    d_lon = lon_b - lon_a
    h = sin(d_lat / 2) ** 2 + cos(lat_a) * cos(lat_b) * sin(d_lon / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(h))


def _edges(segment: Sequence[RoutePoint]) -> list[_Edge]:
    edges: list[_Edge] = []
    for start, end in zip(segment, segment[1:]):
        duration_s = end.time_s - start.time_s
        if duration_s <= 0:
            raise ValueError("route segment timestamps must strictly increase")
        edges.append(_Edge(start, end, distance_m(start, end), duration_s))
    return edges


def _average(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def summarize_track(
    segments: Iterable[Sequence[RoutePoint]], *, moving_speed_mps: float
) -> RouteSummary:
    """Calculate a truthful route summary without joining distinct segments.

    ``moving_speed_mps`` is activity-specific policy supplied by the caller.
    It classifies an observed edge as moving or stopped; it never filters the
    edge out of distance or elapsed time.  Grade is horizontal-distance based,
    signed uphill-positive, and only calculated over edges with both elevations.
    A later elevation-treatment recipe may replace source elevations before this
    function; its algorithm identity belongs in the Route Story receipt.
    """

    if moving_speed_mps < 0:
        raise ValueError("moving_speed_mps must be non-negative")

    normalized = [list(segment) for segment in segments if segment]
    all_edges = [edge for segment in normalized for edge in _edges(segment)]
    distance_m = sum(edge.distance_m for edge in all_edges)
    elapsed_seconds = sum(edge.duration_s for edge in all_edges)
    moving = [edge for edge in all_edges if edge.speed_mps >= moving_speed_mps]
    moving_seconds = sum(edge.duration_s for edge in moving)
    stopped_seconds = elapsed_seconds - moving_seconds

    gap_seconds = 0.0
    for previous, following in zip(normalized, normalized[1:]):
        gap_seconds += max(0.0, following[0].time_s - previous[-1].time_s)

    elevation_edges = [edge for edge in all_edges if edge.elevation_delta_m is not None]
    elevations = [point.elevation_m for segment in normalized for point in segment]
    known_elevations = [elevation for elevation in elevations if elevation is not None]
    elevation_distance_m = sum(edge.distance_m for edge in elevation_edges)
    elevation_delta_m = sum(edge.elevation_delta_m or 0.0 for edge in elevation_edges)
    uphill = [edge for edge in elevation_edges if (edge.elevation_delta_m or 0.0) > 0]
    downhill = [edge for edge in elevation_edges if (edge.elevation_delta_m or 0.0) < 0]
    ascent_m = sum(edge.elevation_delta_m or 0.0 for edge in uphill)
    descent_m = -sum(edge.elevation_delta_m or 0.0 for edge in downhill)
    uphill_distance_m = sum(edge.distance_m for edge in uphill)
    downhill_distance_m = sum(edge.distance_m for edge in downhill)
    absolute_rise_m = sum(abs(edge.elevation_delta_m or 0.0) for edge in elevation_edges)

    current_edge = all_edges[-1] if all_edges else None
    current_grade = None
    if current_edge and current_edge.elevation_delta_m is not None and current_edge.distance_m > 0:
        current_grade = 100.0 * current_edge.elevation_delta_m / current_edge.distance_m

    current_elevation = normalized[-1][-1].elevation_m if normalized else None
    return RouteSummary(
        distance_m=distance_m,
        elapsed_seconds=elapsed_seconds,
        moving_seconds=moving_seconds,
        stopped_seconds=stopped_seconds,
        unobserved_gap_seconds=gap_seconds,
        current_elevation_m=current_elevation,
        min_elevation_m=min(known_elevations) if known_elevations else None,
        max_elevation_m=max(known_elevations) if known_elevations else None,
        net_elevation_change_m=elevation_delta_m if elevation_edges else None,
        ascent_m=ascent_m if elevation_edges else None,
        descent_m=descent_m if elevation_edges else None,
        current_speed_mps=current_edge.speed_mps if current_edge else None,
        maximum_speed_mps=max((edge.speed_mps for edge in all_edges), default=None),
        moving_average_speed_mps=_average(sum(edge.distance_m for edge in moving), moving_seconds),
        elapsed_average_speed_mps=_average(distance_m, elapsed_seconds),
        current_grade_percent=current_grade,
        net_average_grade_percent=(
            100.0 * elevation_delta_m / elevation_distance_m if elevation_distance_m > 0 else None
        ),
        uphill_average_grade_percent=(
            100.0 * ascent_m / uphill_distance_m if uphill_distance_m > 0 else None
        ),
        downhill_average_grade_percent=(
            -100.0 * descent_m / downhill_distance_m if downhill_distance_m > 0 else None
        ),
        absolute_average_grade_percent=(
            100.0 * absolute_rise_m / elevation_distance_m if elevation_distance_m > 0 else None
        ),
    )
