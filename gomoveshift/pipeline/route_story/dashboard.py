"""Shared factual state for a Route Story telemetry dashboard."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Sequence

from .frame_state import frame_state_at
from .metrics import RoutePoint, distance_m, summarize_track
from .timeline import StoryTimeline

# Grade is measured over trailing travel of up to this many metres so one
# noisy short edge cannot dominate the displayed number; a shorter available
# history still reports the best window it has rather than going unknown.
_GRADE_WINDOW_M = 75.0

# The HUD's motion cell (PACE or SPEED, see hud.py) is eased over this many
# trailing recording-clock seconds rather than reading the instantaneous
# edge speed -- a single noisy GPS edge (or a held cadence block jumping to
# a new sample) would otherwise flash a spike value with nothing between it
# and the previous reading.
_PACE_WINDOW_S = 20.0

# Public aliases for the export receipt: ``dashboard_receipt`` (hud.py)
# quotes these two windows so an operator can audit how much smoothing lag
# the on-screen PACE/SPEED and GRADE cells actually carry.
GRADE_WINDOW_M = _GRADE_WINDOW_M
PACE_WINDOW_S = _PACE_WINDOW_S


@dataclass(frozen=True)
class DashboardState:
    """Source-clock facts for one preview or rendered frame.

    Values are in SI units. Unknown telemetry stays ``None`` and a map scene
    cannot create a travelled edge across a source gap.
    """

    data_available: bool
    unobserved_gap: bool
    distance_traveled_m: float
    distance_remaining_m: float
    distance_total_m: float
    completion_fraction: float
    elapsed_seconds: float
    moving_seconds: float
    current_elevation_m: float | None
    current_speed_mps: float | None
    current_grade_percent: float | None
    ascent_m: float | None
    descent_m: float | None
    moving_average_speed_mps: float | None
    current_heart_rate_bpm: float | None = None
    current_cadence_rpm: float | None = None
    current_temperature_c: float | None = None
    # The source recording clock at this frame. Unlike ``elapsed_seconds`` it
    # keeps counting through a recorder pause, so a wall-clock readout built
    # on it never falls behind the real time of day.
    recording_time_s: float | None = None


def _interpolate_optional(start: float | None, end: float | None, fraction: float) -> float | None:
    if start is None or end is None:
        return None
    return start + (end - start) * fraction


@dataclass(frozen=True)
class _SegmentIndex:
    """One segment's per-edge telemetry, precomputed once per (segments,
    moving_speed_mps) pair -- built the first time that combination is seen
    and reused for every subsequent frame, since a film renders through the
    same segments object thousands of times over (once per frame) but the
    per-edge facts never change.

    Every ``cum_*`` array is a prefix sum: index ``i`` is the total
    accumulated through edge ``i - 1`` (index 0 is always 0.0), so a frame's
    position resolves with one ``bisect`` instead of re-walking every edge.
    """

    start_times: list[float]
    end_times: list[float]
    edges: list[tuple[RoutePoint, RoutePoint]]
    cum_distance: list[float]
    cum_duration: list[float]
    cum_moving_duration: list[float]
    cum_moving_distance: list[float]
    cum_ascent: list[float]
    cum_descent: list[float]
    cum_elevation_known: list[int]
    # (cumulative distance before edge, elevation) for edges with a known
    # start elevation only -- exactly what the trailing grade window needs.
    elevation_edge_index: list[int] = field(default_factory=list)
    elevation_track: list[tuple[float, float]] = field(default_factory=list)
    # (cumulative elapsed before edge, cumulative distance before edge) for
    # every edge -- what the trailing pace window bisects into.
    motion_track: list[tuple[float, float]] = field(default_factory=list)


def _build_segment_index(segment: Sequence[RoutePoint], moving_speed_mps: float) -> _SegmentIndex:
    start_times: list[float] = []
    end_times: list[float] = []
    edges: list[tuple[RoutePoint, RoutePoint]] = []
    cum_distance = [0.0]
    cum_duration = [0.0]
    cum_moving_duration = [0.0]
    cum_moving_distance = [0.0]
    cum_ascent = [0.0]
    cum_descent = [0.0]
    cum_elevation_known = [0]
    elevation_edge_index: list[int] = []
    elevation_track: list[tuple[float, float]] = []
    motion_track: list[tuple[float, float]] = []
    for edge_index, (start, end) in enumerate(zip(segment, segment[1:])):
        duration = end.time_s - start.time_s
        if duration <= 0:
            raise ValueError("route segment timestamps must strictly increase")
        edge_distance = distance_m(start, end)
        is_moving = (edge_distance / duration) >= moving_speed_mps
        ascent = descent = 0.0
        both_known = start.elevation_m is not None and end.elevation_m is not None
        if both_known:
            delta = end.elevation_m - start.elevation_m  # type: ignore[operator]
            if delta > 0:
                ascent = delta
            else:
                descent = -delta
        if start.elevation_m is not None:
            elevation_edge_index.append(edge_index)
            elevation_track.append((cum_distance[-1], start.elevation_m))
        motion_track.append((cum_duration[-1], cum_distance[-1]))
        start_times.append(start.time_s)
        end_times.append(end.time_s)
        edges.append((start, end))
        cum_distance.append(cum_distance[-1] + edge_distance)
        cum_duration.append(cum_duration[-1] + duration)
        cum_moving_duration.append(cum_moving_duration[-1] + (duration if is_moving else 0.0))
        cum_moving_distance.append(cum_moving_distance[-1] + (edge_distance if is_moving else 0.0))
        cum_ascent.append(cum_ascent[-1] + ascent)
        cum_descent.append(cum_descent[-1] + descent)
        cum_elevation_known.append(cum_elevation_known[-1] + (1 if both_known else 0))
    return _SegmentIndex(
        start_times=start_times,
        end_times=end_times,
        edges=edges,
        cum_distance=cum_distance,
        cum_duration=cum_duration,
        cum_moving_duration=cum_moving_duration,
        cum_moving_distance=cum_moving_distance,
        cum_ascent=cum_ascent,
        cum_descent=cum_descent,
        cum_elevation_known=cum_elevation_known,
        elevation_edge_index=elevation_edge_index,
        elevation_track=elevation_track,
        motion_track=motion_track,
    )


# A small identity-keyed cache: a ``SceneMapRenderer`` (or any other caller)
# keeps the same segments object for every frame of a leg, so this builds
# each segment's index exactly once per leg rather than once per rendered
# frame. Bounded so a long export cycling through many distinct legs can't
# grow this without limit. Each entry keeps a reference to the segments
# object it was built from and a hit requires ``is`` identity: ``id()``
# alone is not a key, because CPython hands a freed leg's address to the
# next allocation and a bare-``id`` hit would serve the previous leg's
# index to the new one (seen as a missing heart rate on a track that
# carries it). Holding the reference also pins the address while cached.
_INDEX_CACHE: dict[
    tuple[int, float], tuple[Sequence[Sequence[RoutePoint]], list[_SegmentIndex]]
] = {}
_INDEX_CACHE_MAX = 8


def _segment_indexes(
    segments: Sequence[Sequence[RoutePoint]], moving_speed_mps: float
) -> list[_SegmentIndex]:
    key = (id(segments), moving_speed_mps)
    cached = _INDEX_CACHE.get(key)
    if cached is not None and cached[0] is segments:
        return cached[1]
    indexes = [_build_segment_index(segment, moving_speed_mps) for segment in segments]
    if key not in _INDEX_CACHE and len(_INDEX_CACHE) >= _INDEX_CACHE_MAX:
        _INDEX_CACHE.pop(next(iter(_INDEX_CACHE)))
    _INDEX_CACHE[key] = (segments, indexes)
    return indexes


# The whole track's distance, cached under the same identity rule as
# ``_INDEX_CACHE``: it is the same number for every frame of a leg, and
# re-summarizing a 14,001-point track per frame cost ~0.17 s a frame.
_TOTAL_CACHE: dict[int, tuple[Sequence[Sequence[RoutePoint]], float]] = {}


def _track_total_m(segments: Sequence[Sequence[RoutePoint]], moving_speed_mps: float) -> float:
    key = id(segments)
    cached = _TOTAL_CACHE.get(key)
    if cached is not None and cached[0] is segments:
        return cached[1]
    total = summarize_track(segments, moving_speed_mps=moving_speed_mps).distance_m
    if key not in _TOTAL_CACHE and len(_TOTAL_CACHE) >= _INDEX_CACHE_MAX:
        _TOTAL_CACHE.pop(next(iter(_TOTAL_CACHE)))
    _TOTAL_CACHE[key] = (segments, total)
    return total


def _windowed_value(
    xs: list[float], ys: list[float], hi: int, current_x: float, window: float
) -> tuple[float, float] | None:
    """The latest ``(x, y)`` at or before ``current_x - window`` within
    ``xs[:hi]``, or the earliest point in range if the window isn't yet
    full -- the same trailing-window recipe grade and pace both use.
    """
    if hi <= 0:
        return None
    threshold = current_x - window
    idx = bisect_right(xs, threshold, hi=hi) - 1
    if idx < 0:
        idx = 0
    return xs[idx], ys[idx]


def dashboard_state_at(
    timeline: StoryTimeline,
    frame: int,
    segments: Sequence[Sequence[RoutePoint]],
    *,
    moving_speed_mps: float,
    cadence_frames: int = 1,
) -> DashboardState:
    """Return cumulative and current telemetry at a storyboard frame.

    ``cadence_frames`` holds the displayed values steady across a block of
    consecutive output frames (deterministic by frame time) so HUD digits do
    not flicker every render frame; 1 (default) samples every frame.
    """
    if moving_speed_mps < 0:
        raise ValueError("moving_speed_mps must be non-negative")
    if cadence_frames < 1:
        raise ValueError("cadence_frames must be at least 1")
    frame = (frame // cadence_frames) * cadence_frames
    state = frame_state_at(timeline, frame, segments)
    at_s = state.recording_time_s
    total = _track_total_m(segments, moving_speed_mps)

    indexes = _segment_indexes(segments, moving_speed_mps)
    traveled = elapsed = moving = moving_distance = ascent = descent = 0.0
    elevation_seen = False
    current_heart_rate = current_cadence = current_temperature = None
    # Grade and pace windows bisect into only the CURRENT (most recently
    # reached) segment's own points -- reset on every segment the loop
    # visits, so an earlier, fully-covered segment's history never leaks
    # into the trailing window across a recording gap. A window with less
    # history than it wants is truncated at the current segment's own first
    # point rather than reaching backward into a prior segment.
    elevation_xs: list[float] = []
    elevation_ys: list[float] = []
    motion_xs: list[float] = []
    motion_ys: list[float] = []

    for index in indexes:
        if not index.start_times:
            continue
        cut = bisect_left(index.start_times, at_s)
        if cut == 0:
            offset_distance = traveled  # unused placeholder to keep mypy calm below
            continue
        edge_idx = cut - 1
        start, end = index.edges[edge_idx]
        duration = index.end_times[edge_idx] - index.start_times[edge_idx]
        fraction = min(1.0, (at_s - index.start_times[edge_idx]) / duration)
        full_count = edge_idx

        offset_distance = traveled
        offset_elapsed = elapsed
        edge_distance = index.cum_distance[full_count + 1] - index.cum_distance[full_count]
        edge_duration = duration

        traveled += index.cum_distance[full_count] + edge_distance * fraction
        elapsed += index.cum_duration[full_count] + edge_duration * fraction

        speed = edge_distance / duration
        moving += index.cum_moving_duration[full_count]
        moving_distance += index.cum_moving_distance[full_count]
        if speed >= moving_speed_mps:
            moving += edge_duration * fraction
            moving_distance += edge_distance * fraction

        both_known = start.elevation_m is not None and end.elevation_m is not None
        if index.cum_elevation_known[full_count] > 0 or both_known:
            elevation_seen = True
        ascent += index.cum_ascent[full_count]
        descent += index.cum_descent[full_count]
        if both_known:
            delta = (end.elevation_m - start.elevation_m) * fraction  # type: ignore[operator]
            if delta > 0:
                ascent += delta
            else:
                descent -= delta

        current_heart_rate = _interpolate_optional(
            start.heart_rate_bpm, end.heart_rate_bpm, fraction
        )
        current_cadence = _interpolate_optional(start.cadence_rpm, end.cadence_rpm, fraction)
        current_temperature = _interpolate_optional(
            start.temperature_c, end.temperature_c, fraction
        )

        # Reset (not append) each time a segment is reached: only the last
        # segment visited here is "current" for windowing purposes, and its
        # own points must not be mixed with an earlier segment's.
        elevation_xs = []
        elevation_ys = []
        elevation_included = bisect_right(index.elevation_edge_index, edge_idx)
        for local_x, local_y in index.elevation_track[:elevation_included]:
            elevation_xs.append(local_x + offset_distance)
            elevation_ys.append(local_y)
        motion_xs = []
        motion_ys = []
        motion_included = edge_idx + 1
        for local_elapsed, local_distance in index.motion_track[:motion_included]:
            motion_xs.append(local_elapsed + offset_elapsed)
            motion_ys.append(local_distance + offset_distance)

    grade_window = _windowed_value(
        elevation_xs, elevation_ys, len(elevation_xs), traveled, _GRADE_WINDOW_M
    )
    current_grade_percent = None
    if state.elevation_m is not None and grade_window is not None:
        horizontal = traveled - grade_window[0]
        if horizontal > 0:
            current_grade_percent = 100.0 * (state.elevation_m - grade_window[1]) / horizontal

    pace_window = _windowed_value(motion_xs, motion_ys, len(motion_xs), elapsed, _PACE_WINDOW_S)
    current_speed_mps = state.speed_mps
    if state.speed_mps is not None and pace_window is not None:
        duration_span = elapsed - pace_window[0]
        if duration_span > 0:
            current_speed_mps = (traveled - pace_window[1]) / duration_span

    if not state.data_available:
        # Inside a recording gap, current point-in-time readings go
        # unavailable -- only cumulative facts keep their last observed
        # value. current_speed_mps/current_grade_percent are already None
        # here since state.speed_mps/state.elevation_m are None in a gap.
        current_heart_rate = current_cadence = current_temperature = None

    return DashboardState(
        data_available=state.data_available,
        unobserved_gap=not state.data_available,
        distance_traveled_m=traveled,
        distance_remaining_m=max(0.0, total - traveled),
        distance_total_m=total,
        completion_fraction=traveled / total if total > 0 else 0.0,
        elapsed_seconds=elapsed,
        moving_seconds=moving,
        current_elevation_m=state.elevation_m,
        current_speed_mps=current_speed_mps,
        current_grade_percent=current_grade_percent,
        ascent_m=ascent if elevation_seen else None,
        descent_m=descent if elevation_seen else None,
        moving_average_speed_mps=moving_distance / moving if moving > 0 else None,
        current_heart_rate_bpm=current_heart_rate,
        current_cadence_rpm=current_cadence,
        current_temperature_c=current_temperature,
        recording_time_s=at_s,
    )
