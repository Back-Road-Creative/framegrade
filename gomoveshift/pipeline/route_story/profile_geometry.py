"""Distance-based elevation profile geometry for the Route Story HUD.

The x-axis is cumulative DISTANCE travelled, never time -- a long stop or a
crawl through traffic must never stretch the chart, and the y-axis never
joins two source segments across a recorder gap into a fictional climb. Both
properties come straight from ``metrics.RoutePoint``: a gap between source
segments, or a point with unknown elevation, ends the current
``ProfileSegment`` and starts a new one at the correct cumulative distance --
the gap itself adds no distance of its own.

``cursor_distance_m`` is the single source both the HUD's drawn cursor and
(indirectly, via the shared recording clock) the map marker key off of, so
the two never drift apart through a stop, a hold, or a seek.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .metrics import RoutePoint, distance_m

# A flat route (a beach drive, a dead-flat highway) must still read as a
# profile rather than filling the whole chart height with a few centimetres
# of GPS noise -- the vertical scale never shrinks below this span.
MIN_PROFILE_SPAN_M = 30.0


@dataclass(frozen=True)
class ProfilePoint:
    """One drawable sample: cumulative distance, known elevation, recording time."""

    distance_m: float
    elevation_m: float
    time_s: float


@dataclass(frozen=True)
class ProfileSegment:
    """A run of consecutive drawable points with no gap and no unknown elevation."""

    points: tuple[ProfilePoint, ...]


@dataclass(frozen=True)
class ProfileGeometry:
    """The whole film's profile, built once from its display segments."""

    segments: tuple[ProfileSegment, ...]
    total_distance_m: float
    min_elevation_m: float
    max_elevation_m: float

    @property
    def elevation_span_m(self) -> float:
        return max(self.max_elevation_m - self.min_elevation_m, MIN_PROFILE_SPAN_M)

    def cursor_distance_m(self, at_time_s: float) -> float:
        """Distance along the route at ``at_time_s``.

        Inside a source segment's own time range, this interpolates linearly
        between its neighbouring points. Before the first recorded time it is
        0; after the last, the total distance. Inside a GAP between two
        source segments (a recorder pause, a ferry, a device swap) the cursor
        HOLDS at the previous segment's end distance rather than drifting
        forward through the gap or snapping ahead to the next segment --
        exactly matching how the map marker itself behaves in a gap.
        """
        if not self.segments:
            return 0.0
        if at_time_s <= self.segments[0].points[0].time_s:
            return 0.0
        if at_time_s >= self.segments[-1].points[-1].time_s:
            return self.total_distance_m
        for segment in self.segments:
            points = segment.points
            if at_time_s < points[0].time_s:
                # Between the previous segment's end and this one's start --
                # a gap. Hold at the previous segment's end distance, which
                # is the running "previous" default returned below.
                return _previous_end_distance(self.segments, segment)
            if at_time_s <= points[-1].time_s:
                for start, end in zip(points, points[1:]):
                    if start.time_s <= at_time_s <= end.time_s:
                        span = end.time_s - start.time_s
                        if span <= 0:
                            # Defensive: two adjacent points sharing a
                            # timestamp (a duplicate/malformed source fix)
                            # can't be interpolated -- hold at the first.
                            return start.distance_m
                        fraction = (at_time_s - start.time_s) / span
                        return start.distance_m + fraction * (end.distance_m - start.distance_m)
        # Unreachable for a well-formed ``ProfileGeometry``: the range guards
        # above already confine ``at_time_s`` to within the first and last
        # recorded times, and every segment's points increase monotonically,
        # so the loop above always returns first. Kept as a defensive floor
        # rather than letting a logic error fall through silently.
        return self.total_distance_m  # pragma: no cover


def _previous_end_distance(segments: tuple[ProfileSegment, ...], current: ProfileSegment) -> float:
    previous_end = 0.0
    for segment in segments:
        if segment is current:
            return previous_end
        previous_end = segment.points[-1].distance_m
    return previous_end  # pragma: no cover -- ``current`` is always a member of ``segments``


def build_profile(segments: Sequence[Sequence[RoutePoint]]) -> ProfileGeometry:
    """Build the whole film's distance/elevation geometry from its source segments.

    Cumulative distance runs across ALL source segments in order; the gap
    between two source segments adds no distance of its own, so the second
    segment's first drawable point starts exactly where the first one's last
    drawable point ended. A point with unknown elevation ends the current
    ``ProfileSegment`` (a visible break) without discarding its distance
    contribution -- the next known point still starts at the correct
    cumulative distance. A ``ProfileSegment`` with fewer than 2 points is
    dropped from the drawable set (nothing to draw a line between) but its
    points still advanced the running distance for whatever comes after.
    """
    profile_segments: list[ProfileSegment] = []
    current: list[ProfilePoint] = []
    cumulative = 0.0
    known_elevations: list[float] = []

    def _flush() -> None:
        nonlocal current
        if len(current) >= 2:
            profile_segments.append(ProfileSegment(points=tuple(current)))
        current = []

    for segment in segments:
        # Distance only accumulates BETWEEN consecutive points of the same
        # source segment -- the gap crossing to the next source segment (a
        # recorder pause, a ferry, a device swap) is unmeasured travel and
        # must add nothing, however far apart the two fixes happen to be.
        previous_point_in_segment: RoutePoint | None = None
        for point in segment:
            if previous_point_in_segment is not None:
                cumulative += distance_m(previous_point_in_segment, point)
            previous_point_in_segment = point
            if point.elevation_m is None:
                _flush()
                continue
            known_elevations.append(point.elevation_m)
            current.append(
                ProfilePoint(
                    distance_m=cumulative, elevation_m=point.elevation_m, time_s=point.time_s
                )
            )
        # The end of a source segment is always a break in the drawn line,
        # even when the next source segment continues without a gap in
        # elevation data -- two distinct recordings never join visually.
        _flush()

    min_elevation = min(known_elevations) if known_elevations else 0.0
    max_elevation = max(known_elevations) if known_elevations else 0.0
    return ProfileGeometry(
        segments=tuple(profile_segments),
        total_distance_m=cumulative,
        min_elevation_m=min_elevation,
        max_elevation_m=max_elevation,
    )
