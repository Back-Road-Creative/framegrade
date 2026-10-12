"""Source-clock dashboard state for Route Story preview and export."""

import random

import pytest

from gomoveshift.pipeline.route_story.dashboard import dashboard_state_at
from gomoveshift.pipeline.route_story.frame_state import frame_state_at
from gomoveshift.pipeline.route_story.metrics import RoutePoint, distance_m, summarize_track
from gomoveshift.pipeline.route_story.timeline import Scene, StoryTimeline


def _interpolate_optional(start, end, fraction):
    if start is None or end is None:
        return None
    return start + (end - start) * fraction


def _reference_dashboard_state_at(timeline, frame, segments, *, moving_speed_mps):
    """A verbatim copy of the pre-optimization O(n)-per-frame loop.

    Kept ONLY as the equivalence oracle for the bisect/prefix-sum rewrite --
    every field it computes except ``current_speed_mps`` (intentionally
    windowed now, see the pace test above) must match the real function.
    """
    state = frame_state_at(timeline, frame, segments)
    at_s = state.recording_time_s
    total = summarize_track(segments, moving_speed_mps=moving_speed_mps).distance_m
    traveled = elapsed = moving = moving_distance = ascent = descent = 0.0
    elevation_seen = False
    current_heart_rate = current_cadence = current_temperature = None
    elevation_points: list[tuple[float, float]] = []
    for segment in segments:
        if not segment or at_s <= segment[0].time_s:
            continue
        # Reset only for a segment actually reached by this frame: the
        # trailing grade window must never reach back into an earlier
        # segment's points across a recording gap.
        elevation_points = []
        for start, end in zip(segment, segment[1:]):
            duration = end.time_s - start.time_s
            if at_s <= start.time_s:
                break
            fraction = min(1.0, (at_s - start.time_s) / duration)
            if fraction <= 0:
                continue
            if start.elevation_m is not None:
                elevation_points.append((traveled, start.elevation_m))
            edge_distance = distance_m(start, end)
            edge_duration = duration * fraction
            traveled += edge_distance * fraction
            elapsed += edge_duration
            speed = edge_distance / duration
            if speed >= moving_speed_mps:
                moving += edge_duration
                moving_distance += edge_distance * fraction
            if start.elevation_m is not None and end.elevation_m is not None:
                elevation_seen = True
                delta = (end.elevation_m - start.elevation_m) * fraction
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
            if fraction < 1.0:
                break

    grade_window_m = 75.0

    def windowed_grade(elevation_points, current_distance_m, current_elevation_m):
        if current_elevation_m is None or not elevation_points:
            return None
        window_start = elevation_points[0]
        threshold = current_distance_m - grade_window_m
        for point in elevation_points:
            if point[0] <= threshold:
                window_start = point
            else:
                break
        horizontal = current_distance_m - window_start[0]
        if horizontal <= 0:
            return None
        return 100.0 * (current_elevation_m - window_start[1]) / horizontal

    if not state.data_available:
        current_heart_rate = current_cadence = current_temperature = None

    return {
        "data_available": state.data_available,
        "unobserved_gap": not state.data_available,
        "distance_traveled_m": traveled,
        "distance_remaining_m": max(0.0, total - traveled),
        "distance_total_m": total,
        "completion_fraction": traveled / total if total > 0 else 0.0,
        "elapsed_seconds": elapsed,
        "moving_seconds": moving,
        "current_elevation_m": state.elevation_m,
        "current_grade_percent": windowed_grade(elevation_points, traveled, state.elevation_m),
        "ascent_m": ascent if elevation_seen else None,
        "descent_m": descent if elevation_seen else None,
        "moving_average_speed_mps": moving_distance / moving if moving > 0 else None,
        "current_heart_rate_bpm": current_heart_rate,
        "current_cadence_rpm": current_cadence,
        "current_temperature_c": current_temperature,
        "recording_time_s": at_s,
    }


def _noisy_sequential_segments(seed: int) -> list[list[RoutePoint]]:
    """Three sequential legs (a day each), each with jittery edge lengths and
    intermittent missing elevation/heart-rate/cadence -- exercising the
    prefix-sum + bisect rewrite's full-edge, partial-edge and window paths
    across a segment boundary."""
    rng = random.Random(seed)
    segments = []
    t = 0.0
    lon = 0.0
    for leg in range(3):
        points = []
        for i in range(40):
            t += rng.uniform(0.5, 3.0)
            lon += rng.uniform(0.0002, 0.0009)
            elevation = 100.0 + leg * 10 + rng.uniform(-5, 5) if i % 5 != 0 else None
            heart_rate = 120.0 + rng.uniform(-10, 10) if i % 4 != 0 else None
            points.append(
                RoutePoint(
                    time_s=t,
                    latitude=0.0,
                    longitude=lon,
                    elevation_m=elevation,
                    heart_rate_bpm=heart_rate,
                )
            )
        segments.append(points)
        t += rng.uniform(5.0, 20.0)  # a gap between legs
    return segments


def test_dashboard_state_matches_the_reference_loop_across_a_noisy_multi_segment_route():
    segments = _noisy_sequential_segments(seed=7)
    last_time = segments[-1][-1].time_s
    timeline = StoryTimeline(fps=1, scenes=(Scene("route_follow", 0, 1000, 0.0, last_time),))

    for frame in range(0, 1000, 13):
        actual = dashboard_state_at(timeline, frame, segments, moving_speed_mps=0.3)
        expected = _reference_dashboard_state_at(timeline, frame, segments, moving_speed_mps=0.3)
        for field, expected_value in expected.items():
            actual_value = getattr(actual, field)
            if isinstance(expected_value, float) and isinstance(actual_value, float):
                assert actual_value == pytest.approx(expected_value, abs=1e-6), (
                    frame,
                    field,
                )
            else:
                assert actual_value == expected_value, (frame, field)


def test_dashboard_state_is_identical_regardless_of_frame_render_order():
    """``dashboard_state_at`` is a pure function of its arguments: the same

    frame must produce the same ``DashboardState`` no matter what other
    frames were rendered (and thus cached, see ``_segment_indexes``) first.
    """
    segments = _noisy_sequential_segments(seed=11)
    last_time = segments[-1][-1].time_s
    timeline = StoryTimeline(fps=1, scenes=(Scene("route_follow", 0, 1000, 0.0, last_time),))
    frames = list(range(0, 1000, 17))

    forward = {
        frame: dashboard_state_at(timeline, frame, segments, moving_speed_mps=0.3)
        for frame in frames
    }
    shuffled = {
        frame: dashboard_state_at(timeline, frame, segments, moving_speed_mps=0.3)
        for frame in reversed(frames)
    }

    for frame in frames:
        assert shuffled[frame] == forward[frame], frame


def _point(time_s: float, longitude: float, elevation_m: float) -> RoutePoint:
    return RoutePoint(time_s=time_s, latitude=0.0, longitude=longitude, elevation_m=elevation_m)


def test_dashboard_reports_progress_and_current_telemetry_on_the_source_clock():
    timeline = StoryTimeline(fps=10, scenes=(Scene("route_follow", 0, 10, 0.0, 10.0),))
    dashboard = dashboard_state_at(
        timeline,
        5,
        [[_point(0, 0.0, 100.0), _point(10, 0.001, 110.0)]],
        moving_speed_mps=1.0,
    )

    assert dashboard.distance_traveled_m == pytest.approx(55.6, rel=0.01)
    assert dashboard.distance_remaining_m == pytest.approx(55.6, rel=0.01)
    assert dashboard.distance_total_m == pytest.approx(111.2, rel=0.01)
    assert dashboard.completion_fraction == pytest.approx(0.5)
    assert dashboard.elapsed_seconds == 5.0
    assert dashboard.moving_seconds == 5.0
    assert dashboard.current_elevation_m == 105.0
    assert dashboard.current_speed_mps == pytest.approx(11.12, rel=0.01)
    assert dashboard.current_grade_percent == pytest.approx(8.99, rel=0.01)
    assert dashboard.ascent_m == 5.0
    assert dashboard.descent_m == 0.0
    assert dashboard.moving_average_speed_mps == pytest.approx(11.12, rel=0.01)


def test_dashboard_holds_facts_during_a_photo_without_advancing_the_recording():
    timeline = StoryTimeline(
        fps=10,
        scenes=(
            Scene("route_follow", 0, 10, 0.0, 10.0),
            Scene("photo_hold", 10, 30, 10.0, 10.0),
        ),
    )
    dashboard = dashboard_state_at(
        timeline,
        20,
        [[_point(0, 0.0, 100.0), _point(10, 0.001, 110.0)]],
        moving_speed_mps=1.0,
    )

    assert dashboard.distance_traveled_m == dashboard.distance_total_m
    assert dashboard.elapsed_seconds == 10.0
    assert dashboard.current_elevation_m == 110.0


def test_dashboard_marks_a_gap_as_unavailable_without_joining_source_segments():
    timeline = StoryTimeline(fps=10, scenes=(Scene("gap_transition", 0, 10, 30.0, 30.0),))
    dashboard = dashboard_state_at(
        timeline,
        0,
        [
            [_point(0, 0.0, 100.0), _point(10, 0.001, 110.0)],
            [_point(60, 1.0, 120.0), _point(70, 1.001, 130.0)],
        ],
        moving_speed_mps=1.0,
    )

    assert dashboard.data_available is False
    assert dashboard.current_speed_mps is None
    assert dashboard.distance_traveled_m == pytest.approx(111.2, rel=0.01)
    assert dashboard.elapsed_seconds == 10.0
    assert dashboard.unobserved_gap is True


def test_dashboard_grade_is_windowed_across_travel_not_last_edge_noise():
    timeline = StoryTimeline(fps=10, scenes=(Scene("route_follow", 0, 20, 0.0, 20.0),))
    segments = [
        [
            _point(0, 0.0, 100.0),
            _point(10, 0.0007, 100.0),
            _point(20, 0.0008, 110.0),
        ]
    ]

    dashboard = dashboard_state_at(timeline, 20, segments, moving_speed_mps=0.1)

    # The final short edge alone implies a ~90% point-to-point grade; the
    # trailing-window grade over the full climb should smooth that out.
    assert dashboard.current_grade_percent == pytest.approx(11.25, rel=0.05)


def test_dashboard_cadence_holds_values_within_a_frame_bucket_and_updates_across_buckets():
    timeline = StoryTimeline(fps=10, scenes=(Scene("route_follow", 0, 20, 0.0, 2.0),))
    segments = [[_point(0, 0.0, 100.0), _point(2, 0.001, 110.0)]]

    early = dashboard_state_at(timeline, 3, segments, moving_speed_mps=0.1, cadence_frames=10)
    late = dashboard_state_at(timeline, 9, segments, moving_speed_mps=0.1, cadence_frames=10)
    next_bucket = dashboard_state_at(
        timeline, 10, segments, moving_speed_mps=0.1, cadence_frames=10
    )

    assert early == late
    assert early != next_bucket


def test_dashboard_current_speed_is_windowed_not_a_raw_edge_spike():
    # A steady 1 m/s crawl, then one GPS-noise edge that alone implies 100
    # m/s -- the HUD's pace/speed cell must not jump to that spike; it should
    # read close to the trailing-window average instead.
    timeline = StoryTimeline(fps=10, scenes=(Scene("route_follow", 0, 2010, 0.0, 201.0),))
    segments = [
        [
            _point(0, 0.0, 100.0),
            _point(100.0, 0.0009, 100.0),  # 100 m in 100 s -> ~1 m/s
            _point(101.0, 0.0018, 100.0),  # 100 m in 1 s -> ~100 m/s spike
            _point(201.0, 0.0027, 100.0),  # back to ~1 m/s
        ]
    ]

    # Frame 1005 lands mid-way through the spike edge (recording time 100.5s).
    dashboard = dashboard_state_at(timeline, 1005, segments, moving_speed_mps=0.1)

    assert 1.0 < dashboard.current_speed_mps < 10.0


def test_dashboard_windows_never_reach_back_across_a_segment_gap():
    # Segment 1 ends with a huge elevation/distance edge; segment 2 starts a
    # small edge right after a recording gap. If the grade/pace windows
    # reached into segment 1's points, the trailing-window facts at the very
    # start of segment 2 would be dominated by segment 1's numbers instead of
    # being truncated to whatever short history segment 2 itself has.
    timeline = StoryTimeline(fps=1, scenes=(Scene("route_follow", 0, 105, 0.0, 105.0),))
    segments = [
        [_point(0, 0.0, 0.0), _point(10, 0.009, 500.0)],
        [_point(100, 1.0, 200.0), _point(105, 1.00003, 205.0)],
    ]

    dashboard = dashboard_state_at(timeline, 105, segments, moving_speed_mps=0.1)

    assert dashboard.data_available is True
    assert dashboard.current_grade_percent == pytest.approx(151.5, rel=0.05)
    assert dashboard.current_speed_mps == pytest.approx(0.66, rel=0.05)


def test_dashboard_nulls_current_readings_but_not_cumulative_facts_during_a_gap():
    timeline = StoryTimeline(fps=10, scenes=(Scene("gap_transition", 0, 10, 30.0, 30.0),))
    segments = [
        [
            RoutePoint(
                time_s=0,
                latitude=0.0,
                longitude=0.0,
                elevation_m=100.0,
                heart_rate_bpm=120.0,
                cadence_rpm=80.0,
                temperature_c=15.0,
            ),
            RoutePoint(
                time_s=10,
                latitude=0.0,
                longitude=0.001,
                elevation_m=110.0,
                heart_rate_bpm=140.0,
                cadence_rpm=90.0,
                temperature_c=16.0,
            ),
        ],
        [
            RoutePoint(time_s=60, latitude=0.0, longitude=1.0, elevation_m=120.0),
            RoutePoint(time_s=70, latitude=0.0, longitude=1.001, elevation_m=130.0),
        ],
    ]

    dashboard = dashboard_state_at(timeline, 0, segments, moving_speed_mps=1.0)

    assert dashboard.data_available is False
    assert dashboard.current_heart_rate_bpm is None
    assert dashboard.current_cadence_rpm is None
    assert dashboard.current_temperature_c is None
    assert dashboard.current_speed_mps is None
    assert dashboard.current_grade_percent is None
    # Cumulative facts still hold the last observed value through the gap.
    assert dashboard.distance_traveled_m == pytest.approx(111.2, rel=0.01)
    assert dashboard.elapsed_seconds == 10.0


def test_dashboard_reports_heart_rate_and_cadence_when_present_none_when_absent():
    timeline = StoryTimeline(fps=10, scenes=(Scene("route_follow", 0, 10, 0.0, 10.0),))
    segments = [
        [
            RoutePoint(
                time_s=0,
                latitude=0.0,
                longitude=0.0,
                elevation_m=100.0,
                heart_rate_bpm=120.0,
                cadence_rpm=80.0,
            ),
            RoutePoint(
                time_s=10,
                latitude=0.0,
                longitude=0.001,
                elevation_m=110.0,
                heart_rate_bpm=140.0,
                cadence_rpm=90.0,
            ),
        ]
    ]

    dashboard = dashboard_state_at(timeline, 5, segments, moving_speed_mps=1.0)

    assert dashboard.current_heart_rate_bpm == pytest.approx(130.0)
    assert dashboard.current_cadence_rpm == pytest.approx(85.0)
    assert dashboard.current_temperature_c is None
