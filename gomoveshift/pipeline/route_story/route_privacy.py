"""Route start/end privacy trim: drop the points nearest a route's endpoints
BEFORE any downstream surface (film, dashboard, metadata) can read them.

Trimming is along-track (cumulative travelled distance in recording order),
never by wall-clock offset or point index -- a paused/slow start otherwise
leaks more of the operator's driveway than intended. Points are only ever
removed, never moved or synthesized: a segment left with fewer than two
points after trimming is dropped whole rather than kept as a single stray
fix. Wired into ``route-story-film`` (``film.py``'s ``--trim-start-m``/
``--trim-end-m``, applied right after ``select_gpx_day`` so every downstream
surface only sees the trimmed route); see
``docs/pipeline/150-route-story-route-privacy.md``.

Interior privacy zones (``PrivacyZone``/``mask_zones``) extend the same
points-only-removed, whole-segment-drop semantics to a mid-route redaction
radius rather than just the two endpoints -- see ``mask_zones`` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .metrics import RoutePoint, distance_m


class RouteTrimError(ValueError):
    """Raised when a requested trim would remove the entire route."""


@dataclass(frozen=True)
class RouteTrim:
    """Requested along-track privacy trim, in meters from each endpoint."""

    start_m: float = 0.0
    end_m: float = 0.0


@dataclass(frozen=True)
class TrimResult:
    """Outcome of applying a ``RouteTrim`` to a route's segments.

    ``removed_start_m``/``removed_end_m`` are the ACTUAL along-track distance
    dropped from each end -- point granularity means this is always >= the
    requested ``trim.start_m``/``trim.end_m``, never less.
    """

    segments: list[list[RoutePoint]]
    removed_start_points: int
    removed_end_points: int
    removed_start_m: float
    removed_end_m: float
    trim: RouteTrim


def trim_route(segments: Sequence[Sequence[RoutePoint]], trim: RouteTrim) -> TrimResult:
    """Drop points within ``trim.start_m``/``trim.end_m`` of the route's ends.

    Distance accumulates only across edges actually measured within a
    segment -- a gap between segments carries the running total forward
    without adding distance for the unobserved crossing, so a short segment
    entirely inside the trim radius is dropped whole rather than anchoring a
    phantom edge.
    """
    if trim.start_m == 0.0 and trim.end_m == 0.0:
        return TrimResult(
            segments=segments,
            removed_start_points=0,
            removed_end_points=0,
            removed_start_m=0.0,
            removed_end_m=0.0,
            trim=trim,
        )

    flat: list[tuple[int, RoutePoint]] = []
    cumulative: list[float] = []
    running = 0.0
    previous_point: RoutePoint | None = None
    previous_segment_index: int | None = None
    for segment_index, segment in enumerate(segments):
        for point in segment:
            if previous_point is not None and previous_segment_index == segment_index:
                running += distance_m(previous_point, point)
            flat.append((segment_index, point))
            cumulative.append(running)
            previous_point = point
            previous_segment_index = segment_index

    total_length_m = cumulative[-1] if cumulative else 0.0

    def _over_trim() -> RouteTrimError:
        return RouteTrimError(
            f"route trim of start_m={trim.start_m} end_m={trim.end_m} would remove "
            f"the entire route; route length is {total_length_m:.1f} m"
        )

    if not flat:
        raise _over_trim()

    keep_start_threshold = trim.start_m
    keep_end_threshold = total_length_m - trim.end_m

    kept_by_segment: dict[int, list[int]] = {}
    for index, (segment_index, _point) in enumerate(flat):
        distance = cumulative[index]
        if distance < keep_start_threshold or distance > keep_end_threshold:
            continue
        kept_by_segment.setdefault(segment_index, []).append(index)

    result_segments: list[list[RoutePoint]] = []
    kept_indices: list[int] = []
    for segment_index in sorted(kept_by_segment):
        indices = kept_by_segment[segment_index]
        if len(indices) < 2:
            continue
        result_segments.append([flat[index][1] for index in indices])
        kept_indices.extend(indices)

    if not result_segments:
        raise _over_trim()

    first_index = kept_indices[0]
    last_index = kept_indices[-1]
    return TrimResult(
        segments=result_segments,
        removed_start_points=first_index,
        removed_end_points=len(flat) - 1 - last_index,
        removed_start_m=cumulative[first_index],
        removed_end_m=total_length_m - cumulative[last_index],
        trim=trim,
    )


def receipt_fragment(result: TrimResult, zone_result: "ZoneMaskResult | None" = None) -> dict:
    """The ``film_receipt`` fragment attached for this trim.

    ``zone_result``, when supplied, adds a ``zones`` block carrying only
    counts and removed points/metres -- never the requested zone centers or
    radii -- so a public receipt (which copies ``route_trim`` verbatim, see
    ``public_receipt.py``) can never leak a zone's location.
    """
    fragment = {
        "route_trim": {
            "start_m": result.trim.start_m,
            "end_m": result.trim.end_m,
            "removed_start_points": result.removed_start_points,
            "removed_end_points": result.removed_end_points,
            "removed_start_m": result.removed_start_m,
            "removed_end_m": result.removed_end_m,
        }
    }
    if zone_result is not None:
        fragment["route_trim"]["zones"] = {
            "count": zone_result.zone_count,
            "removed_points": zone_result.removed_points,
            "removed_m": zone_result.removed_m,
        }
    return fragment


@dataclass(frozen=True)
class PrivacyZone:
    """A requested interior privacy zone, in meters.

    Everything within ``radius_m`` of (``center_lat``, ``center_lon``) is
    masked out of the route by :func:`mask_zones`.
    """

    center_lat: float
    center_lon: float
    radius_m: float

    def __post_init__(self) -> None:
        # A swapped LAT,LON or a zero radius would otherwise mask nothing, silently.
        if not -90.0 <= self.center_lat <= 90.0:
            raise ValueError(f"privacy zone latitude {self.center_lat} is outside -90..90")
        if not -180.0 <= self.center_lon <= 180.0:
            raise ValueError(f"privacy zone longitude {self.center_lon} is outside -180..180")
        if not self.radius_m > 0:
            raise ValueError(f"privacy zone radius must be > 0 m, got {self.radius_m}")


@dataclass(frozen=True)
class ZoneMaskResult:
    """Outcome of applying one or more ``PrivacyZone``s to a route's segments.

    ``removed_points`` is every input point absent from ``segments`` --
    masked-by-radius points and any leftover single-point run too short to
    keep, same whole-segment-drop rule as ``trim_route``. Overlapping zones
    never double-count a point. ``removed_m`` is the along-track span
    actually covered by radius-masked points (the "hole" a zone or union of
    zones cut into the route).
    """

    segments: list[list[RoutePoint]]
    removed_points: int
    removed_m: float
    zone_count: int


def mask_zones(
    segments: Sequence[Sequence[RoutePoint]], zones: Sequence[PrivacyZone]
) -> ZoneMaskResult:
    """Drop every point within any ``PrivacyZone`` in ``zones``.

    Unlike ``trim_route`` (which only ever removes a prefix/suffix), a zone
    sits in the middle of a segment: a segment a zone splits yields two kept
    segments with a real gap between them -- the removed hole is never
    bridged into a single, falsely-continuous segment. A run of kept points
    left with fewer than two points (whether because a zone masked its
    neighbours down to a stray fix, or an entire short segment sits inside a
    zone) is dropped whole, same as ``trim_route``. A route left entirely
    masked (zero kept segments) raises ``RouteTrimError``, with the route's
    actual along-track length in the message -- the same error family
    ``trim_route`` raises on over-trim.
    """
    if not zones:
        return ZoneMaskResult(segments=segments, removed_points=0, removed_m=0.0, zone_count=0)

    def _masked(point: RoutePoint) -> bool:
        return any(
            distance_m(RoutePoint(point.time_s, zone.center_lat, zone.center_lon), point)
            <= zone.radius_m
            for zone in zones
        )

    total_points = 0
    total_length_m = 0.0
    kept_points = 0
    removed_m = 0.0
    result_segments: list[list[RoutePoint]] = []

    for segment in segments:
        if not segment:
            continue
        total_points += len(segment)
        cumulative = [0.0] * len(segment)
        running = 0.0
        for index in range(1, len(segment)):
            running += distance_m(segment[index - 1], segment[index])
            cumulative[index] = running
        total_length_m += running

        masked = [_masked(point) for point in segment]

        run_start: int | None = None
        for index, is_masked in enumerate(masked):
            if is_masked and run_start is None:
                run_start = index
            elif not is_masked and run_start is not None:
                removed_m += cumulative[index - 1] - cumulative[run_start]
                run_start = None
        if run_start is not None:
            removed_m += cumulative[-1] - cumulative[run_start]

        run: list[RoutePoint] = []
        for point, is_masked in zip(segment, masked):
            if is_masked:
                if len(run) >= 2:
                    result_segments.append(run)
                    kept_points += len(run)
                run = []
                continue
            run.append(point)
        if len(run) >= 2:
            result_segments.append(run)
            kept_points += len(run)

    if not result_segments:
        raise RouteTrimError(
            f"privacy zones ({len(zones)}) would remove the entire route; "
            f"route length is {total_length_m:.1f} m"
        )

    return ZoneMaskResult(
        segments=result_segments,
        removed_points=total_points - kept_points,
        removed_m=removed_m,
        zone_count=len(zones),
    )
