"""Safe GPX import for the Route Story source clock.

The importer deliberately keeps GPX track segments separate.  It does not sort
points, bridge outages, or manufacture timestamps for geometry-only tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from defusedxml import ElementTree

from .metrics import RoutePoint


@dataclass(frozen=True)
class GpxDaySelection:
    """One audit-friendly calendar-day selection from a GPX source.

    ``segments`` contains only the requested source-date points, with exact
    duplicate segments removed.  The counters belong in the project receipt so
    a map movie cannot quietly turn stale history into the current journey.
    """

    segments: list[list[RoutePoint]]
    source_started_at: datetime
    source_segment_count: int
    excluded_segment_count: int
    excluded_other_date_count: int
    excluded_duplicate_count: int


def _tag(element: ElementTree.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _timestamp(value: str | None) -> datetime:
    if not value:
        raise ValueError("Route Story GPX points need a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Route Story GPX has an invalid timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("Route Story GPX timestamps need an explicit timezone")
    return parsed


_RawPoint = tuple[datetime, float, float, float | None, float | None, float | None, float | None]


def _extension_values(
    point: ElementTree.Element,
) -> tuple[float | None, float | None, float | None]:
    """Read Garmin TrackPointExtension body data; absent fields stay ``None``."""
    hr = cadence = temperature = None
    for child in point:
        if _tag(child) != "extensions":
            continue
        for extension in child.iter():
            if _tag(extension) != "TrackPointExtension":
                continue
            for field in extension:
                tag = _tag(field)
                if field.text is None:
                    continue
                try:
                    if tag == "hr":
                        hr = float(field.text)
                    elif tag == "cad":
                        cadence = float(field.text)
                    elif tag == "atemp":
                        temperature = float(field.text)
                except ValueError as exc:
                    raise ValueError("Route Story GPX point has invalid extension data") from exc
    return hr, cadence, temperature


def _load_raw_segments(source: Path) -> list[list[_RawPoint]]:
    """Parse timed track points without changing source segment order."""
    try:
        root = ElementTree.parse(source).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        raise ValueError("Route Story source is not valid GPX") from exc

    raw_segments: list[list[_RawPoint]] = []
    for segment in root.iter():
        if _tag(segment) != "trkseg":
            continue
        points: list[_RawPoint] = []
        for point in segment:
            if _tag(point) != "trkpt":
                continue
            try:
                latitude, longitude = float(point.attrib["lat"]), float(point.attrib["lon"])
            except (KeyError, ValueError) as exc:
                raise ValueError("Route Story GPX point has invalid coordinates") from exc
            children = {_tag(child): child.text for child in point}
            elevation = children.get("ele")
            try:
                elevation_m = float(elevation) if elevation is not None else None
            except ValueError as exc:
                raise ValueError("Route Story GPX point has invalid elevation") from exc
            heart_rate_bpm, cadence_rpm, temperature_c = _extension_values(point)
            points.append(
                (
                    _timestamp(children.get("time")),
                    latitude,
                    longitude,
                    elevation_m,
                    heart_rate_bpm,
                    cadence_rpm,
                    temperature_c,
                )
            )
        if points:
            raw_segments.append(points)
    if not raw_segments:
        raise ValueError("Route Story GPX has no track points")

    return raw_segments


def _to_route_points(raw_segments: list[list[_RawPoint]]) -> list[list[RoutePoint]]:
    epoch = raw_segments[0][0][0]
    return [
        [
            RoutePoint(
                (timestamp - epoch).total_seconds(),
                latitude,
                longitude,
                elevation_m,
                heart_rate_bpm,
                cadence_rpm,
                temperature_c,
            )
            for timestamp, latitude, longitude, elevation_m, heart_rate_bpm, cadence_rpm, temperature_c in segment
        ]
        for segment in raw_segments
    ]


def load_gpx_segments(source: Path) -> list[list[RoutePoint]]:
    """Load timed GPX track segments without changing their evidence order."""
    return _to_route_points(_load_raw_segments(source))


def select_gpx_day(source: Path, recorded_on: date) -> GpxDaySelection:
    """Select one source calendar day and remove only exact duplicate segments.

    Calendar days use the explicit timezone attached to each GPX timestamp.
    This does not sort, join, smooth, or otherwise manufacture route evidence.
    """
    raw_segments = _load_raw_segments(source)
    selected = [
        segment
        for segment in raw_segments
        if segment and all(timestamp.date() == recorded_on for timestamp, *_ in segment)
    ]
    other_date_count = len(raw_segments) - len(selected)
    unique: list[list[_RawPoint]] = []
    seen: set[tuple[_RawPoint, ...]] = set()
    duplicate_count = 0
    for segment in selected:
        signature = tuple(segment)
        if signature in seen:
            duplicate_count += 1
            continue
        seen.add(signature)
        unique.append(segment)
    if not unique:
        raise ValueError(
            f"Route Story GPX has no complete track segments for {recorded_on.isoformat()}"
        )
    return GpxDaySelection(
        segments=_to_route_points(unique),
        source_started_at=unique[0][0][0],
        source_segment_count=len(raw_segments),
        excluded_segment_count=other_date_count + duplicate_count,
        excluded_other_date_count=other_date_count,
        excluded_duplicate_count=duplicate_count,
    )
