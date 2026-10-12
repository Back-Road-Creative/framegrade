"""Evidence-backed links between Route Story media and a recorded track.

Media can carry its own GPS coordinates, or it can be located from its capture
time on a continuous observed track edge.  The latter is an interpolation, not
a claim that the device recorded a fix at that exact instant.  No location is
created across an unobserved gap or outside the recording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Sequence

from .metrics import RoutePoint, distance_m

_PIXEL_FILENAME_START = re.compile(
    r"^PXL_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_"
    r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})(?P<millis>\d{3})",
    re.IGNORECASE,
)


def capture_start_from_filename(name: str) -> datetime | None:
    """Read a phone clip's recording START time out of its filename, in UTC.

    Pixel names it ``PXL_YYYYMMDD_HHMMSSmmm`` in UTC.  A container's
    ``creation_time`` is file-finalisation time, not recording start, so this
    is the stronger evidence for a phone clip and should be preferred over it
    wherever a video's capture time is derived.  Returns ``None`` for any name
    that does not match a recognised phone-clip pattern rather than guessing.
    """
    match = _PIXEL_FILENAME_START.match(name)
    if not match:
        return None
    fields = match.groupdict()
    return datetime(
        int(fields["year"]),
        int(fields["month"]),
        int(fields["day"]),
        int(fields["hour"]),
        int(fields["minute"]),
        int(fields["second"]),
        int(fields["millis"]) * 1000,
        tzinfo=timezone.utc,
    )


AnchorStatus = Literal[
    "embedded_coordinates",
    "coordinate_time_conflict",
    "timestamp_interpolated",
    "unanchored_gap",
    "unanchored_outside_track",
]


@dataclass(frozen=True)
class MediaEvidence:
    """Coordinates and/or capture time read from one original media item.

    ``captured_at_s`` uses the same resolved source clock as the Route Story
    track.  Callers must resolve timezone evidence before constructing this
    value; a naive filesystem timestamp is insufficient evidence for anchoring.
    """

    media_id: str
    captured_at_s: float | None = None
    latitude: float | None = None
    longitude: float | None = None

    def __post_init__(self) -> None:
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("media coordinates must include both latitude and longitude")
        if self.latitude is not None and not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be in [-90, 90]")
        if self.longitude is not None and not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be in [-180, 180]")


InterpolationMethod = Literal["stationary_edge"]


@dataclass(frozen=True)
class MediaAnchor:
    """An auditable media location decision for storyboard planning.

    Coordinate omission supports private export surfaces: the anchoring result
    still records how the event belongs in the timeline without exposing a
    precise location to a preview, project listing, or export manifest.

    ``interpolation_method`` is set to ``"stationary_edge"`` only when a
    ``timestamp_interpolated`` result was allowed by the ground-distance rule
    (a short, GPS-quiet edge such as a rest stop) rather than the time-gap
    rule.  ``rejected_latitude``/``rejected_longitude``/``required_speed_mps``
    are set only when embedded GPS was rejected as physically impossible for
    a certain clock time, so the receipt can show what was discarded and why.
    """

    media_id: str
    status: AnchorStatus
    track_time_s: float | None
    latitude: float | None
    longitude: float | None
    interpolation_method: InterpolationMethod | None = None
    rejected_latitude: float | None = None
    rejected_longitude: float | None = None
    required_speed_mps: float | None = None


def captured_at_route_seconds(captured_at: datetime, source_started_at: datetime) -> float:
    """Translate an explicitly zoned capture instant to the GPX source clock.

    A filesystem mtime or timezone-free camera clock is deliberately not
    accepted here.  Callers must retain the original clock evidence and any
    approved correction in their Route Story receipt.
    """
    if captured_at.tzinfo is None or captured_at.utcoffset() is None:
        raise ValueError("Route Story capture time needs an explicit timezone")
    if source_started_at.tzinfo is None or source_started_at.utcoffset() is None:
        raise ValueError("Route Story source clock needs an explicit timezone")
    return (
        captured_at.astimezone(timezone.utc) - source_started_at.astimezone(timezone.utc)
    ).total_seconds()


def anchor_media(
    media: MediaEvidence,
    track: Sequence[RoutePoint],
    *,
    max_interpolation_gap_s: float = 15.0,
    max_interpolation_gap_m: float = 150.0,
    include_coordinates: bool = True,
) -> MediaAnchor:
    """Anchor media without manufacturing a GPS position.

    Embedded GPS is direct evidence and wins over an unrelated or clock-drifted
    timestamp.  Timestamp anchoring is allowed within one sorted observed edge
    when EITHER the edge is no wider than ``max_interpolation_gap_s`` OR the
    edge covers no more ground than ``max_interpolation_gap_m`` — a recorder
    that samples every 30-60 s still produces a wide-in-time edge at a rest
    stop, and the ground distance there is the honest test, not the clock.
    An edge wide in both time and ground is left unanchored exactly as before.
    """
    if max_interpolation_gap_s <= 0:
        raise ValueError("max_interpolation_gap_s must be positive")
    if max_interpolation_gap_m <= 0:
        raise ValueError("max_interpolation_gap_m must be positive")

    if media.latitude is not None:
        return _result(
            media,
            "embedded_coordinates",
            None,
            media.latitude,
            media.longitude,
            include_coordinates,
        )

    if media.captured_at_s is None or not track:
        return _result(media, "unanchored_outside_track", None, None, None, include_coordinates)

    points = sorted(track, key=lambda point: point.time_s)
    if media.captured_at_s < points[0].time_s or media.captured_at_s > points[-1].time_s:
        return _result(media, "unanchored_outside_track", None, None, None, include_coordinates)

    for start, end in zip(points, points[1:]):
        if end.time_s <= start.time_s:
            raise ValueError("track timestamps must be strictly increasing")
        if start.time_s <= media.captured_at_s <= end.time_s:
            duration_s = end.time_s - start.time_s
            within_time = duration_s <= max_interpolation_gap_s
            if not within_time and distance_m(start, end) > max_interpolation_gap_m:
                return _result(media, "unanchored_gap", None, None, None, include_coordinates)
            fraction = (media.captured_at_s - start.time_s) / duration_s
            return _result(
                media,
                "timestamp_interpolated",
                media.captured_at_s,
                start.latitude + (end.latitude - start.latitude) * fraction,
                start.longitude + (end.longitude - start.longitude) * fraction,
                include_coordinates,
                interpolation_method=None if within_time else "stationary_edge",
            )

    # A one-point track can contain the timestamp but has no observed edge from
    # which to derive a time-position relation.
    return _result(media, "unanchored_outside_track", None, None, None, include_coordinates)


def anchor_media_to_segments(
    media: MediaEvidence,
    segments: Sequence[Sequence[RoutePoint]],
    *,
    max_interpolation_gap_s: float = 15.0,
    max_interpolation_gap_m: float = 150.0,
    coordinate_conflict_distance_m: float = 1_000.0,
    implausible_speed_factor: float = 3.0,
    include_coordinates: bool = True,
) -> MediaAnchor:
    """Anchor media against every accepted segment, preserving conflicts.

    A source may contain both GPS and a capture clock.  GPS remains the display
    location; the clock is used only to place the event on the story timeline.
    When those two evidence streams disagree by more than the explicit safety
    radius, the speed required to travel from the embedded fix to the nearest
    track fix on each side of the clock time decides which evidence wins: a
    speed beyond ``implausible_speed_factor`` times the track's own 95th
    percentile edge speed (floored at 3 m/s) marks the embedded GPS as
    physically impossible and the clock-derived position is used instead,
    with the discarded fix recorded for review.  A merely-implausible-looking
    but physically reachable disagreement keeps today's review-required
    behaviour: embedded GPS stays on screen, flagged for a human to check.
    """
    if coordinate_conflict_distance_m <= 0:
        raise ValueError("coordinate_conflict_distance_m must be positive")
    flattened = [point for segment in segments for point in segment]
    if media.latitude is not None:
        timed = anchor_media(
            MediaEvidence(media.media_id, captured_at_s=media.captured_at_s),
            flattened,
            max_interpolation_gap_s=max_interpolation_gap_s,
            max_interpolation_gap_m=max_interpolation_gap_m,
            include_coordinates=True,
        )
        if timed.track_time_s is None:
            return _result(
                media,
                "embedded_coordinates",
                None,
                media.latitude,
                media.longitude,
                include_coordinates,
            )
        estimated = RoutePoint(timed.track_time_s, timed.latitude or 0.0, timed.longitude or 0.0)
        observed = RoutePoint(timed.track_time_s, media.latitude, media.longitude)
        if distance_m(estimated, observed) <= coordinate_conflict_distance_m:
            return _result(
                media,
                "embedded_coordinates",
                timed.track_time_s,
                media.latitude,
                media.longitude,
                include_coordinates,
            )
        required_speed = _required_speed_mps(observed, flattened, timed.track_time_s)
        if required_speed > _implausible_speed_limit_mps(flattened, implausible_speed_factor):
            return _result(
                media,
                "timestamp_interpolated",
                timed.track_time_s,
                timed.latitude,
                timed.longitude,
                include_coordinates,
                interpolation_method=timed.interpolation_method,
                rejected_latitude=media.latitude,
                rejected_longitude=media.longitude,
                required_speed_mps=required_speed,
            )
        return _result(
            media,
            "coordinate_time_conflict",
            timed.track_time_s,
            media.latitude,
            media.longitude,
            include_coordinates,
        )
    for segment in segments:
        anchored = anchor_media(
            media,
            segment,
            max_interpolation_gap_s=max_interpolation_gap_s,
            max_interpolation_gap_m=max_interpolation_gap_m,
            include_coordinates=include_coordinates,
        )
        if anchored.status in {"timestamp_interpolated", "unanchored_gap"}:
            if anchored.status == "timestamp_interpolated":
                return anchored
            # A timestamp inside a gap in one segment may still belong to a
            # later segment, so keep searching before declaring it unanchored.
            continue
        if anchored.status == "unanchored_outside_track":
            continue
    if media.captured_at_s is not None:
        points = [point for segment in segments for point in segment]
        if points and min(point.time_s for point in points) <= media.captured_at_s <= max(
            point.time_s for point in points
        ):
            return _result(media, "unanchored_gap", None, None, None, include_coordinates)
    return _result(media, "unanchored_outside_track", None, None, None, include_coordinates)


PositionSource = Literal[
    "embedded_gps",
    "clock_interpolated",
    "nearest_observation",
    "none",
]


@dataclass(frozen=True)
class MediaPlacement:
    """An auditable media placement with independent WHEN/WHERE verdicts.

    ``timeline_s`` is the story position derived from the capture clock and is
    set whenever that clock is known and falls inside the recording window; a
    positional uncertainty (a gap, an isolated fix, no fix at all) never
    demotes a certain capture time.  ``position_source`` records how (or
    whether) a marker location was derived, independently of that decision.
    """

    media_id: str
    timeline_s: float | None
    position_source: PositionSource
    latitude: float | None
    longitude: float | None
    radius_s: float | None = None
    conflict: bool = False
    conflict_latitude: float | None = None
    conflict_longitude: float | None = None
    clock_offset_s: float = 0.0
    clock_source: str | None = None
    interpolation_method: InterpolationMethod | None = None
    conflict_rejected: bool = False
    required_speed_mps: float | None = None


def calibrate_clock_offset_s(
    media: MediaEvidence,
    track: Sequence[RoutePoint],
    *,
    max_match_distance_m: float = 200.0,
) -> float:
    """Derive one camera body's clock offset from a corroborated GPS still.

    The still must carry both embedded GPS and a capture-clock reading on the
    route source clock.  The offset is the shift that places the still's
    capture time at the track instant nearest its embedded position.  A still
    whose embedded position is not actually near the track is not
    corroboration and raises rather than manufacturing an offset.
    """
    if media.captured_at_s is None or media.latitude is None or media.longitude is None:
        raise ValueError("clock calibration requires a still with embedded GPS and a capture clock")
    if not track:
        raise ValueError("clock calibration requires a track to corroborate against")
    observed = RoutePoint(media.captured_at_s, media.latitude, media.longitude)
    nearest = min(track, key=lambda point: distance_m(point, observed))
    if distance_m(nearest, observed) > max_match_distance_m:
        raise ValueError("embedded GPS does not corroborate against the track; cannot calibrate")
    return nearest.time_s - media.captured_at_s


def _timeline_s(captured_at_s: float | None, observations: Sequence[RoutePoint]) -> float | None:
    if captured_at_s is None or not observations:
        return None
    lo = min(point.time_s for point in observations)
    hi = max(point.time_s for point in observations)
    if lo <= captured_at_s <= hi:
        return captured_at_s
    return None


def _nearest_observation(
    observations: Sequence[RoutePoint], t: float, radius_s: float
) -> RoutePoint | None:
    candidates = [point for point in observations if abs(point.time_s - t) <= radius_s]
    if not candidates:
        return None
    return min(candidates, key=lambda point: abs(point.time_s - t))


def _clock_derived_position(
    timeline_s: float | None,
    track: Sequence[RoutePoint],
    observations: Sequence[RoutePoint],
    *,
    max_interpolation_gap_s: float,
    max_interpolation_gap_m: float,
    nearest_observation_radius_s: float,
) -> tuple[tuple[float, float] | None, PositionSource, float | None, InterpolationMethod | None]:
    if timeline_s is None:
        return None, "none", None, None
    points = sorted(track, key=lambda point: point.time_s)
    for start, end in zip(points, points[1:]):
        if end.time_s <= start.time_s:
            continue
        if start.time_s <= timeline_s <= end.time_s:
            duration_s = end.time_s - start.time_s
            within_time = duration_s <= max_interpolation_gap_s
            if within_time or distance_m(start, end) <= max_interpolation_gap_m:
                fraction = (timeline_s - start.time_s) / duration_s
                latitude = start.latitude + (end.latitude - start.latitude) * fraction
                longitude = start.longitude + (end.longitude - start.longitude) * fraction
                method: InterpolationMethod | None = None if within_time else "stationary_edge"
                return (latitude, longitude), "clock_interpolated", None, method
            break
    nearest = _nearest_observation(observations, timeline_s, nearest_observation_radius_s)
    if nearest is not None:
        return (
            (nearest.latitude, nearest.longitude),
            "nearest_observation",
            abs(nearest.time_s - timeline_s),
            None,
        )
    return None, "none", None, None


def place_media(
    media: MediaEvidence,
    track: Sequence[RoutePoint],
    observations: Sequence[RoutePoint] | None = None,
    *,
    max_interpolation_gap_s: float = 15.0,
    max_interpolation_gap_m: float = 150.0,
    nearest_observation_radius_s: float = 30.0,
    coordinate_conflict_distance_m: float = 1_000.0,
    implausible_speed_factor: float = 3.0,
    conflict_window_s: float = 600.0,
    clock_offset_s: float = 0.0,
    clock_source: str | None = None,
    include_coordinates: bool = True,
) -> MediaPlacement:
    """Place media on the story timeline, deciding WHEN and WHERE independently.

    ``observations`` is the raw, unfiltered set of GPS fixes (including any
    isolated point ``track`` itself excludes as too short an edge to
    interpolate); it defaults to ``track`` when the caller has nothing wider.
    Embedded GPS that disagrees with a clock-corroborated track position by
    more than ``coordinate_conflict_distance_m`` is a conflict; whether WHERE
    defers to the clock or keeps the embedded fix on screen depends on the
    speed needed to reach the embedded position from the nearest track fix on
    each side of the clock time.  A speed beyond ``implausible_speed_factor``
    times the track's own 95th-percentile edge speed (floored at 3 m/s) marks
    the embedded GPS physically impossible: WHERE defers to the clock and the
    rejected fix is recorded.  A reachable disagreement instead keeps the
    embedded fix on screen, still flagged as a conflict for review.
    """
    if nearest_observation_radius_s <= 0:
        raise ValueError("nearest_observation_radius_s must be positive")
    if coordinate_conflict_distance_m <= 0:
        raise ValueError("coordinate_conflict_distance_m must be positive")

    raw_observations = list(observations) if observations is not None else list(track)
    calibrated_at_s = (
        media.captured_at_s + clock_offset_s if media.captured_at_s is not None else None
    )
    timeline_s = _timeline_s(calibrated_at_s, raw_observations)
    clock_position, clock_source_kind, clock_radius_s, clock_method = _clock_derived_position(
        timeline_s,
        track,
        raw_observations,
        max_interpolation_gap_s=max_interpolation_gap_s,
        max_interpolation_gap_m=max_interpolation_gap_m,
        nearest_observation_radius_s=nearest_observation_radius_s,
    )

    conflict = False
    conflict_rejected = False
    conflict_latitude: float | None = None
    conflict_longitude: float | None = None
    required_speed_mps: float | None = None

    if media.latitude is not None:
        if timeline_s is not None and clock_position is not None:
            nearby = [
                point
                for point in raw_observations
                if abs(point.time_s - timeline_s) <= conflict_window_s
            ]
            observed = RoutePoint(timeline_s, media.latitude, media.longitude)
            if nearby and all(
                distance_m(point, observed) > coordinate_conflict_distance_m for point in nearby
            ):
                conflict = True
                conflict_latitude, conflict_longitude = media.latitude, media.longitude
                required_speed_mps = _required_speed_mps(observed, raw_observations, timeline_s)
                limit = _implausible_speed_limit_mps(raw_observations, implausible_speed_factor)
                conflict_rejected = required_speed_mps > limit
        if conflict and conflict_rejected:
            latitude, longitude = clock_position  # type: ignore[misc]
            position_source: PositionSource = clock_source_kind
            radius_s = clock_radius_s
            interpolation_method = clock_method
        else:
            latitude, longitude = media.latitude, media.longitude
            position_source = "embedded_gps"
            radius_s = None
            interpolation_method = None
    elif clock_position is not None:
        latitude, longitude = clock_position
        position_source = clock_source_kind
        radius_s = clock_radius_s
        interpolation_method = clock_method
    else:
        latitude = longitude = None
        radius_s = None
        position_source = "none"
        interpolation_method = None

    if not include_coordinates:
        latitude = longitude = None
        conflict_latitude = conflict_longitude = None

    return MediaPlacement(
        media_id=media.media_id,
        timeline_s=timeline_s,
        position_source=position_source,
        latitude=latitude,
        longitude=longitude,
        radius_s=radius_s,
        conflict=conflict,
        conflict_latitude=conflict_latitude,
        conflict_longitude=conflict_longitude,
        clock_offset_s=clock_offset_s,
        clock_source=clock_source,
        interpolation_method=interpolation_method,
        conflict_rejected=conflict_rejected,
        required_speed_mps=required_speed_mps,
    )


def _result(
    media: MediaEvidence,
    status: AnchorStatus,
    track_time_s: float | None,
    latitude: float | None,
    longitude: float | None,
    include_coordinates: bool,
    *,
    interpolation_method: InterpolationMethod | None = None,
    rejected_latitude: float | None = None,
    rejected_longitude: float | None = None,
    required_speed_mps: float | None = None,
) -> MediaAnchor:
    if not include_coordinates:
        latitude, longitude = None, None
        rejected_latitude, rejected_longitude = None, None
    return MediaAnchor(
        media.media_id,
        status,
        track_time_s,
        latitude,
        longitude,
        interpolation_method=interpolation_method,
        rejected_latitude=rejected_latitude,
        rejected_longitude=rejected_longitude,
        required_speed_mps=required_speed_mps,
    )


def _p95_edge_speed_mps(points: Sequence[RoutePoint]) -> float:
    """The track's own 95th-percentile edge speed, from positive-duration edges."""
    ordered = sorted(points, key=lambda point: point.time_s)
    speeds = [
        distance_m(a, b) / (b.time_s - a.time_s)
        for a, b in zip(ordered, ordered[1:])
        if b.time_s > a.time_s
    ]
    if not speeds:
        return 0.0
    speeds.sort()
    index = min(len(speeds) - 1, round(0.95 * (len(speeds) - 1)))
    return speeds[index]


def _implausible_speed_limit_mps(points: Sequence[RoutePoint], factor: float) -> float:
    """A speed limit derived from the track itself, floored so a near-stationary
    track cannot make every disagreement look impossible."""
    return max(factor * _p95_edge_speed_mps(points), 3.0)


def _required_speed_mps(observed: RoutePoint, points: Sequence[RoutePoint], t: float) -> float:
    """Speed needed to reach ``observed`` from the nearest track fix on each side of ``t``."""
    ordered = sorted(points, key=lambda point: point.time_s)
    before = next((point for point in reversed(ordered) if point.time_s <= t), None)
    after = next((point for point in ordered if point.time_s >= t), None)
    speeds: list[float] = []
    for flank in (before, after):
        if flank is None:
            continue
        dt = abs(flank.time_s - t)
        distance = distance_m(flank, observed)
        if dt == 0:
            speeds.append(0.0 if distance == 0 else float("inf"))
        else:
            speeds.append(distance / dt)
    return max(speeds) if speeds else float("inf")
