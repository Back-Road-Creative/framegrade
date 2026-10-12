"""Renderer frame telemetry is derived from source recording time."""

import pytest

from gomoveshift.pipeline.route_story.frame_state import frame_state_at
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.profile_geometry import build_profile
from gomoveshift.pipeline.route_story.timeline import Scene, StoryTimeline


def _point(t, lat, lon, elevation=100.0):
    return RoutePoint(time_s=t, latitude=lat, longitude=lon, elevation_m=elevation)


def _timeline(total_s):
    # fps=1 with a single route_follow scene spanning [0, total_s] makes the
    # output frame number and the recording-clock second the same value, so
    # tests can drive frame_state_at with plain seconds.
    return StoryTimeline(
        fps=1, scenes=(Scene("route_follow", 0, int(total_s), 0.0, float(total_s)),)
    )


def test_frame_state_interpolates_position_and_telemetry_from_recording_time():
    timeline = StoryTimeline(fps=10, scenes=(Scene("route_follow", 0, 10, 0.0, 10.0),))

    state = frame_state_at(
        timeline,
        5,
        [[_point(0, 0.0, 0.0, 100), _point(10, 0.0, 0.01, 110)]],
    )

    assert state.recording_time_s == 5.0
    assert state.latitude == 0.0
    assert state.longitude == 0.005
    assert state.elevation_m == 105.0
    assert state.speed_mps is not None and state.speed_mps > 0
    assert state.grade_percent is not None and state.grade_percent > 0
    assert state.data_available is True


def test_frame_state_at_a_photo_hold_keeps_the_recorded_position_not_animation_speed():
    timeline = StoryTimeline(
        fps=10,
        scenes=(
            Scene("route_follow", 0, 10, 0.0, 10.0),
            Scene("photo_hold", 10, 30, 10.0, 10.0),
        ),
    )

    state = frame_state_at(timeline, 20, [[_point(0, 0.0, 0.0), _point(10, 0.0, 0.01)]])

    assert state.recording_time_s == 10.0
    assert state.longitude == 0.01


def test_frame_state_does_not_invent_a_position_inside_a_segment_gap():
    timeline = StoryTimeline(fps=10, scenes=(Scene("gap_transition", 0, 10, 20.0, 20.0),))

    state = frame_state_at(
        timeline,
        0,
        [[_point(0, 0.0, 0.0)], [_point(60, 0.0, 0.06)]],
    )

    assert state.data_available is False
    assert state.latitude is None
    assert state.speed_mps is None


def test_frame_state_distance_m_is_monotone_non_decreasing_across_a_route():
    timeline = _timeline(40)
    first = [_point(0, 0.0, 0.0), _point(10, 0.0, 0.01)]
    second = [_point(30, 0.0, 0.02), _point(40, 0.0, 0.03)]
    segments = [first, second]

    distances = [frame_state_at(timeline, t, segments).distance_m for t in range(0, 41)]

    assert distances == sorted(distances)
    assert distances[0] == 0.0
    assert distances[-1] == pytest.approx(build_profile(segments).total_distance_m)


def test_frame_state_distance_m_holds_at_previous_segments_end_through_a_gap():
    timeline = _timeline(40)
    first = [_point(0, 0.0, 0.0), _point(10, 0.0, 0.01)]
    second = [_point(30, 0.0, 0.02), _point(40, 0.0, 0.03)]
    segments = [first, second]

    first_segment_end = frame_state_at(timeline, 10, segments).distance_m
    mid_gap = frame_state_at(timeline, 20, segments).distance_m

    assert first_segment_end > 0.0
    assert mid_gap == first_segment_end


def test_frame_state_distance_m_is_zero_before_first_time_and_total_after_last():
    timeline = _timeline(40)
    segments = [[_point(10, 0.0, 0.0), _point(20, 0.0, 0.01)]]

    before = frame_state_at(timeline, 0, segments)
    after = frame_state_at(timeline, 40, segments)

    assert before.distance_m == 0.0
    assert after.distance_m == pytest.approx(build_profile(segments).total_distance_m)
    assert after.distance_m > 0.0


def test_frame_state_distance_m_matches_profile_geometry_cursor_distance_m_on_shared_fixture():
    timeline = _timeline(40)
    first = [_point(0, 35.0, -80.0, 100.0), _point(10, 35.001, -80.0, 110.0)]
    second = [_point(30, 35.002, -80.0, 120.0), _point(40, 35.003, -80.0, 130.0)]
    segments = [first, second]
    geometry = build_profile(segments)

    for t in (0, 3, 10, 20, 30, 33, 40):
        assert frame_state_at(timeline, t, segments).distance_m == pytest.approx(
            geometry.cursor_distance_m(float(t))
        )


def test_frame_state_at_raises_on_duplicate_timestamps_like_the_dashboard_path():
    timeline = _timeline(10)
    segments = [[_point(0, 0.0, 0.0), _point(0, 0.0, 0.01)]]

    with pytest.raises(ValueError, match="strictly increase"):
        frame_state_at(timeline, 0, segments)
