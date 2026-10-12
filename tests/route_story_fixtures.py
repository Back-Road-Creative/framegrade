"""Synthetic Route Story fixtures shared across the route_story test files.

Every ``test_route_story_*`` module hand-built ``RoutePoint`` lists, GPX text,
and ``StoryTimeline``s inline with slightly different idioms. This module is
the single place that does it, using the real production types (``RoutePoint``,
``StoryTimeline``/``Scene``) and the real GPX parser's own extension namespace,
so a fixture can never silently drift from what the pipeline actually accepts.

Plain functions only, no pytest fixtures, so any test module can call them
directly regardless of its own fixture setup.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Callable, Sequence

from gomoveshift.pipeline.route_story.metrics import EARTH_RADIUS_M, RoutePoint
from gomoveshift.pipeline.route_story.timeline import Scene, StoryTimeline

# Editorial speed bands, matching ``HIKING_MAX_SPEED_MPS`` and
# ``RIDING_MAX_SPEED_MPS`` in route_story/metadata.py,
# with headroom so a preset never lands on a threshold.
_HIKING_SPEED_MPS = 1.2
_CYCLING_SPEED_MPS = 6.0
_DRIVING_SPEED_MPS = 20.0

_DEFAULT_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _resolve(value, index: int):
    return value(index) if callable(value) else value


def points(
    n: int,
    *,
    speed_mps: float,
    start_time: float = 0.0,
    dt: float = 1.0,
    elevation_fn: Callable[[int], float | None] | None = None,
    heart_rate: float | Callable[[int], float | None] | None = None,
    cadence: float | Callable[[int], float | None] | None = None,
    temperature: float | Callable[[int], float | None] | None = None,
    start_lat: float = 0.0,
    start_lon: float = 0.0,
) -> list[RoutePoint]:
    """``n`` ``RoutePoint``s advancing due north at exactly ``speed_mps``.

    A pure-meridian advance keeps the great-circle distance between adjacent
    points exactly ``speed_mps * dt`` (no small-angle approximation error),
    so callers can assert on speed without floating-point slack.
    """
    result: list[RoutePoint] = []
    latitude = start_lat
    for index in range(n):
        if index > 0:
            distance_m = speed_mps * dt
            latitude += math.degrees(distance_m / EARTH_RADIUS_M)
        result.append(
            RoutePoint(
                start_time + index * dt,
                latitude,
                start_lon,
                _resolve(elevation_fn, index) if elevation_fn is not None else None,
                _resolve(heart_rate, index),
                _resolve(cadence, index),
                _resolve(temperature, index),
            )
        )
    return result


def segments_with_gap(
    gap_s: float,
    *,
    speed_mps: float,
    n: int = 5,
    dt: float = 1.0,
    start_time: float = 0.0,
    start_lat: float = 0.0,
    start_lon: float = 0.0,
    **point_kwargs,
) -> list[list[RoutePoint]]:
    """Two segments separated by ``gap_s`` seconds of unobserved recording time."""
    first = points(
        n,
        speed_mps=speed_mps,
        start_time=start_time,
        dt=dt,
        start_lat=start_lat,
        start_lon=start_lon,
        **point_kwargs,
    )
    second_start = first[-1].time_s + dt + gap_s
    second = points(
        n,
        speed_mps=speed_mps,
        start_time=second_start,
        dt=dt,
        start_lat=first[-1].latitude,
        start_lon=first[-1].longitude,
        **point_kwargs,
    )
    return [first, second]


def hiking_points(n: int = 5, **kwargs) -> list[RoutePoint]:
    kwargs.setdefault("speed_mps", _HIKING_SPEED_MPS)
    return points(n, **kwargs)


def cycling_points(n: int = 5, **kwargs) -> list[RoutePoint]:
    kwargs.setdefault("speed_mps", _CYCLING_SPEED_MPS)
    return points(n, **kwargs)


def driving_points(n: int = 5, **kwargs) -> list[RoutePoint]:
    kwargs.setdefault("speed_mps", _DRIVING_SPEED_MPS)
    return points(n, **kwargs)


def gpx_text(
    segments: Sequence[RoutePoint] | Sequence[Sequence[RoutePoint]],
    *,
    epoch: datetime = _DEFAULT_EPOCH,
) -> str:
    """Render segments as GPX 1.1 XML that ``gpx.load_gpx_segments`` reads back.

    ``segments`` may be a flat point list (one segment) or a list of segments.
    Extensions use the Garmin ``TrackPointExtension`` shape
    (``_extension_values`` in ``gomoveshift/pipeline/route_story/gpx.py``),
    which only matches on local tag name, so any declared namespace prefix works.
    """
    if segments and isinstance(segments[0], RoutePoint):
        segments = [segments]  # type: ignore[list-item]

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="route_story_fixtures" '
        'xmlns="http://www.topografix.com/GPX/1/1" '
        'xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">',
        "<trk><name>fixture</name>",
    ]
    for segment in segments:
        lines.append("<trkseg>")
        for point in segment:
            timestamp = epoch + timedelta(seconds=point.time_s)
            time_text = timestamp.isoformat().replace("+00:00", "Z")
            lines.append(f'<trkpt lat="{point.latitude!r}" lon="{point.longitude!r}">')
            if point.elevation_m is not None:
                lines.append(f"<ele>{point.elevation_m}</ele>")
            lines.append(f"<time>{time_text}</time>")
            has_extension = (
                point.heart_rate_bpm is not None
                or point.cadence_rpm is not None
                or point.temperature_c is not None
            )
            if has_extension:
                lines.append("<extensions><gpxtpx:TrackPointExtension>")
                if point.heart_rate_bpm is not None:
                    lines.append(f"<gpxtpx:hr>{point.heart_rate_bpm}</gpxtpx:hr>")
                if point.cadence_rpm is not None:
                    lines.append(f"<gpxtpx:cad>{point.cadence_rpm}</gpxtpx:cad>")
                if point.temperature_c is not None:
                    lines.append(f"<gpxtpx:atemp>{point.temperature_c}</gpxtpx:atemp>")
                lines.append("</gpxtpx:TrackPointExtension></extensions>")
            lines.append("</trkpt>")
        lines.append("</trkseg>")
    lines.append("</trk></gpx>")
    return "\n".join(lines)


def timeline_for(segments: Sequence[Sequence[RoutePoint]], fps: int = 30) -> StoryTimeline:
    """A single ``route_follow`` ``StoryTimeline`` spanning every segment's span.

    One scene covering the full recording-time range (including any gap
    between segments) is enough for a fixture: ``dashboard_state_at`` resolves
    each frame's recording time independently against the segments themselves,
    so a frame whose mapped time falls in a gap correctly reports no data.
    """
    all_points = [point for segment in segments for point in segment]
    if not all_points:
        raise ValueError("timeline_for needs at least one point")
    start_s = min(point.time_s for point in all_points)
    end_s = max(point.time_s for point in all_points)
    frame_count = max(1, round((end_s - start_s) * fps))
    scene = Scene(
        kind="route_follow",
        start_frame=0,
        end_frame=frame_count,
        route_start_s=start_s,
        route_end_s=end_s,
    )
    return StoryTimeline(fps=fps, scenes=[scene])
