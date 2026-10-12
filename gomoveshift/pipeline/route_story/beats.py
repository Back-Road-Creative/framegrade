"""Story beats and chapter cues derived from Route Story track data.

A beat is one narratively meaningful moment on the recorded route: a stop,
the high point, the steepest sustained climb, a cluster of anchored media, a
GPS signal gap, or the journey's start/finish. Every beat carries only facts
computed from the accepted route evidence -- label/value pairs, never prose
adjectives -- and a shot drawn from the Route Story vocabulary. This module
never assumes a film-time relationship of its own; ``write_chapters_vtt``
accepts a caller-supplied ``route_time -> film_time`` mapping instead of
reaching into the storyboard directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, Sequence

from .metrics import RoutePoint, distance_m, summarize_track

BeatKind = Literal[
    "start",
    "stop",
    "high_point",
    "steepest_climb",
    "media",
    "signal_gap",
    "finish",
]

Shot = Literal["reveal", "arc", "follow", "arrival", "hold", "pullback"]

_M_PER_MILE = 1609.344
_M_PER_FOOT = 0.3048

# Shot vocabulary durations, seconds.
SHOT_DURATIONS_S: dict[Shot, float] = {
    "reveal": 7.0,
    "arc": 7.0,
    "follow": 6.0,
    "arrival": 4.0,
    "hold": 3.0,
    "pullback": 5.0,
}

_KIND_SHOT: dict[BeatKind, Shot] = {
    "start": "reveal",
    "stop": "hold",
    "high_point": "arc",
    "steepest_climb": "follow",
    "media": "arrival",
    "signal_gap": "hold",
    "finish": "pullback",
}

# Stable tie-break order for beats that land on the same route_time_s.
_KIND_ORDER = list(_KIND_SHOT)

_STOP_SPEED_MPS = 0.5

_KIND_TITLE: dict[BeatKind, str] = {
    "start": "Start",
    "stop": "Stop",
    "high_point": "High point",
    "steepest_climb": "Steepest climb",
    "media": "Media",
    "signal_gap": "Signal gap",
    "finish": "Finish",
}


class AnchoredMediaLike(Protocol):
    """Minimal shape a media-cluster beat needs from anchored media."""

    route_time_s: float | None
    lat: float | None
    lon: float | None


@dataclass(frozen=True)
class Beat:
    """One chapter-worthy moment on the route, backed only by observed data."""

    kind: BeatKind
    route_time_s: float
    lat: float
    lon: float
    elevation_m: float | None
    shot: Shot
    facts: tuple[tuple[str, str], ...]
    duration_s: float


def _format_feet(value: float) -> str:
    return f"{value / _M_PER_FOOT:,.0f} ft"


def _format_miles(value: float) -> str:
    return f"{value / _M_PER_MILE:.1f} mi"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    if total_seconds < 60:
        return f"{total_seconds} sec"
    minutes = total_seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _point_beat(kind: BeatKind, point: RoutePoint, *, facts: tuple[tuple[str, str], ...]) -> Beat:
    shot = _KIND_SHOT[kind]
    return Beat(
        kind=kind,
        route_time_s=point.time_s,
        lat=point.latitude,
        lon=point.longitude,
        elevation_m=point.elevation_m,
        shot=shot,
        facts=facts,
        duration_s=SHOT_DURATIONS_S[shot],
    )


def _start_facts(point: RoutePoint) -> tuple[tuple[str, str], ...]:
    if point.elevation_m is None:
        return ()
    return (("Elevation", _format_feet(point.elevation_m)),)


def _finish_facts(distance_m_total: float, elapsed_seconds: float) -> tuple[tuple[str, str], ...]:
    return (
        ("Distance", _format_miles(distance_m_total)),
        ("Duration", _format_duration(elapsed_seconds)),
    )


def _maybe_stop_beat(
    segment: list[RoutePoint], start_idx: int, end_idx: int, min_stop_s: float
) -> list[Beat]:
    start_point = segment[start_idx]
    end_point = segment[end_idx]
    dwell_s = end_point.time_s - start_point.time_s
    if dwell_s < min_stop_s:
        return []
    facts = (("Duration", _format_duration(dwell_s)),)
    return [_point_beat("stop", start_point, facts=facts)]


def _stop_beats(segments: list[list[RoutePoint]], *, min_stop_s: float) -> list[Beat]:
    beats: list[Beat] = []
    for segment in segments:
        if len(segment) < 2:
            continue
        run_start_idx: int | None = None
        for i in range(len(segment) - 1):
            start, end = segment[i], segment[i + 1]
            duration = end.time_s - start.time_s
            speed = distance_m(start, end) / duration if duration > 0 else 0.0
            slow = speed < _STOP_SPEED_MPS
            if slow and run_start_idx is None:
                run_start_idx = i
            elif not slow and run_start_idx is not None:
                beats.extend(_maybe_stop_beat(segment, run_start_idx, i, min_stop_s))
                run_start_idx = None
        if run_start_idx is not None:
            beats.extend(_maybe_stop_beat(segment, run_start_idx, len(segment) - 1, min_stop_s))
    return beats


def _high_point(segments: list[list[RoutePoint]]) -> Beat | None:
    best_point: RoutePoint | None = None
    best_elevation = float("-inf")
    for segment in segments:
        for point in segment:
            if point.elevation_m is None:
                continue
            if point.elevation_m > best_elevation:
                best_elevation = point.elevation_m
                best_point = point
    if best_point is None:
        return None
    facts = (("Elevation", _format_feet(best_elevation)),)
    return _point_beat("high_point", best_point, facts=facts)


def _steepest_climb(segments: list[list[RoutePoint]], *, climb_window_m: float) -> Beat | None:
    best_gain = 0.0
    best_end: RoutePoint | None = None
    best_distance = 0.0

    for segment in segments:
        if len(segment) < 2:
            continue
        elevations = [point.elevation_m for point in segment]
        if all(elevation is None for elevation in elevations):
            continue
        cumulative = [0.0]
        for start, end in zip(segment, segment[1:]):
            cumulative.append(cumulative[-1] + distance_m(start, end))
        for j in range(len(segment)):
            end_elevation = elevations[j]
            if end_elevation is None:
                continue
            for i in range(j):
                start_elevation = elevations[i]
                if start_elevation is None:
                    continue
                window_distance = cumulative[j] - cumulative[i]
                if window_distance <= 0 or window_distance > climb_window_m:
                    continue
                gain = end_elevation - start_elevation
                if gain > best_gain:
                    best_gain = gain
                    best_end = segment[j]
                    best_distance = window_distance

    if best_end is None:
        return None
    facts = (("Climb", f"{_format_feet(best_gain)} over {_format_miles(best_distance)}"),)
    return _point_beat("steepest_climb", best_end, facts=facts)


def _cluster_beat(cluster: list[tuple[float, float, float]]) -> Beat:
    time_s, lat, lon = cluster[0]
    shot = _KIND_SHOT["media"]
    facts = (("Media", str(len(cluster))),)
    return Beat(
        kind="media",
        route_time_s=time_s,
        lat=lat,
        lon=lon,
        elevation_m=None,
        shot=shot,
        facts=facts,
        duration_s=SHOT_DURATIONS_S[shot],
    )


def _media_beats(media: Sequence[AnchoredMediaLike], *, media_cluster_s: float) -> list[Beat]:
    located: list[tuple[float, float, float]] = []
    for item in media:
        if item.route_time_s is None or item.lat is None or item.lon is None:
            continue
        located.append((item.route_time_s, item.lat, item.lon))
    located.sort(key=lambda entry: entry[0])

    beats: list[Beat] = []
    cluster: list[tuple[float, float, float]] = []
    for entry in located:
        if cluster and entry[0] - cluster[-1][0] > media_cluster_s:
            beats.append(_cluster_beat(cluster))
            cluster = []
        cluster.append(entry)
    if cluster:
        beats.append(_cluster_beat(cluster))
    return beats


def _signal_gap_beats(segments: list[list[RoutePoint]]) -> list[Beat]:
    beats: list[Beat] = []
    for previous, following in zip(segments, segments[1:]):
        gap_s = following[0].time_s - previous[-1].time_s
        if gap_s <= 0:
            continue
        facts = (("Gap", _format_duration(gap_s)),)
        beats.append(_point_beat("signal_gap", previous[-1], facts=facts))
    return beats


def extract_beats(
    segments: Sequence[Sequence[RoutePoint]],
    *,
    media: Sequence[AnchoredMediaLike] = (),
    min_stop_s: float = 180.0,
    climb_window_m: float = 800.0,
    media_cluster_s: float = 120.0,
) -> list[Beat]:
    """Derive a deterministic, time-ordered beat list from route evidence.

    Segments stay separate: a segment break is a signal gap, never a joined
    travelled edge. Beats not backed by data (no elevation for the high point
    or steepest climb, no anchored media) are simply omitted rather than
    invented.
    """
    if min_stop_s <= 0:
        raise ValueError("min_stop_s must be positive")
    if climb_window_m <= 0:
        raise ValueError("climb_window_m must be positive")
    if media_cluster_s <= 0:
        raise ValueError("media_cluster_s must be positive")

    normalized = [list(segment) for segment in segments if segment]
    if not normalized:
        raise ValueError("extract_beats needs at least one non-empty segment")

    beats: list[Beat] = [
        _point_beat("start", normalized[0][0], facts=_start_facts(normalized[0][0]))
    ]
    beats.extend(_stop_beats(normalized, min_stop_s=min_stop_s))

    high_point = _high_point(normalized)
    if high_point is not None:
        beats.append(high_point)

    climb = _steepest_climb(normalized, climb_window_m=climb_window_m)
    if climb is not None:
        beats.append(climb)

    beats.extend(_media_beats(media, media_cluster_s=media_cluster_s))
    beats.extend(_signal_gap_beats(normalized))

    summary = summarize_track(normalized, moving_speed_mps=_STOP_SPEED_MPS)
    finish_point = normalized[-1][-1]
    beats.append(
        _point_beat(
            "finish", finish_point, facts=_finish_facts(summary.distance_m, summary.elapsed_seconds)
        )
    )

    beats.sort(key=lambda beat: (beat.route_time_s, _KIND_ORDER.index(beat.kind)))
    return beats


def _vtt_timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs, millis = divmod(remainder_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _cue_title(beat: Beat) -> str:
    label = _KIND_TITLE[beat.kind]
    if not beat.facts:
        return label
    _, value = beat.facts[0]
    return f"{label} · {value}"


def chapter_rows(
    beats: Sequence[Beat],
    film_time: Callable[[float], float],
    *,
    film_duration_s: float,
) -> list[dict]:
    """The film's chapter list: one ``{title, start_s, end_s}`` row per beat.

    The single source for every chapter surface (``chapters.vtt``, the publish
    sidecar, the receipt) so they cannot disagree. A chapter runs from its
    beat's mapped film time to the next beat's, or to ``film_duration_s`` for
    the last one. ``film_time`` is caller-supplied so this module never
    assumes a particular timeline/clock model.
    """
    if not beats:
        raise ValueError("chapter_rows needs at least one beat")
    if film_duration_s <= 0:
        raise ValueError("film_duration_s must be positive")

    ordered = sorted(beats, key=lambda beat: beat.route_time_s)
    starts = [max(0.0, min(film_time(beat.route_time_s), film_duration_s)) for beat in ordered]
    for previous, following in zip(starts, starts[1:]):
        if following < previous:
            raise ValueError("film_time must be non-decreasing across beats")

    ends = starts[1:] + [film_duration_s]
    return [
        {"title": _cue_title(beat), "start_s": start, "end_s": end}
        for beat, start, end in zip(ordered, starts, ends)
    ]


def write_chapters_vtt(
    beats: Sequence[Beat],
    film_time: Callable[[float], float],
    path: Path,
    *,
    film_duration_s: float,
) -> None:
    """Write a valid WebVTT chapter track, one cue per ``chapter_rows`` row."""
    lines = ["WEBVTT", ""]
    for row in chapter_rows(beats, film_time, film_duration_s=film_duration_s):
        lines.append(f"{_vtt_timestamp(row['start_s'])} --> {_vtt_timestamp(row['end_s'])}")
        lines.append(row["title"])
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def beats_receipt(beats: Sequence[Beat]) -> list[dict]:
    """JSON-safe beat rows for the Route Story receipt."""
    return [
        {
            "kind": beat.kind,
            "route_time_s": beat.route_time_s,
            "lat": beat.lat,
            "lon": beat.lon,
            "elevation_m": beat.elevation_m,
            "shot": beat.shot,
            "facts": [[label, value] for label, value in beat.facts],
            "duration_s": beat.duration_s,
        }
        for beat in beats
    ]
