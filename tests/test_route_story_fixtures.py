"""Coverage for the shared route_story synthetic fixtures (route_story_fixtures.py)."""

from gomoveshift.pipeline.route_story import gpx
from gomoveshift.pipeline.route_story.dashboard import dashboard_state_at
from gomoveshift.pipeline.route_story.metadata import HIKING_MAX_SPEED_MPS, RIDING_MAX_SPEED_MPS
from gomoveshift.pipeline.route_story.metrics import summarize_track

from route_story_fixtures import (
    cycling_points,
    driving_points,
    gpx_text,
    hiking_points,
    points,
    segments_with_gap,
    timeline_for,
)


def test_points_round_trip_through_the_real_gpx_parser(tmp_path):
    original = points(
        4,
        speed_mps=2.5,
        elevation_fn=lambda i: 10.0 + i,
        heart_rate=lambda i: 100.0 + i,
        cadence=80.0,
        temperature=15.0,
    )
    gpx_path = tmp_path / "fixture.gpx"
    gpx_path.write_text(gpx_text(original))

    (parsed_segment,) = gpx.load_gpx_segments(gpx_path)

    assert len(parsed_segment) == len(original)
    for expected, actual in zip(original, parsed_segment):
        assert actual.time_s == expected.time_s
        assert actual.latitude == expected.latitude
        assert actual.longitude == expected.longitude
        assert actual.elevation_m == expected.elevation_m
        assert actual.heart_rate_bpm == expected.heart_rate_bpm
        assert actual.cadence_rpm == expected.cadence_rpm
        assert actual.temperature_c == expected.temperature_c


def test_segments_with_gap_round_trip_as_two_segments(tmp_path):
    first, second = segments_with_gap(120.0, speed_mps=1.2, n=3)
    gpx_path = tmp_path / "gap.gpx"
    gpx_path.write_text(gpx_text([first, second]))

    parsed = gpx.load_gpx_segments(gpx_path)

    assert len(parsed) == 2
    assert len(parsed[0]) == len(first)
    assert len(parsed[1]) == len(second)
    assert parsed[1][0].time_s == second[0].time_s


def test_hiking_band_lands_at_or_below_the_hiking_threshold():
    summary = summarize_track([hiking_points(6)], moving_speed_mps=0.1)
    assert summary.moving_average_speed_mps <= HIKING_MAX_SPEED_MPS


def test_cycling_band_lands_between_the_hiking_and_riding_thresholds():
    summary = summarize_track([cycling_points(6)], moving_speed_mps=0.1)
    assert HIKING_MAX_SPEED_MPS < summary.moving_average_speed_mps <= RIDING_MAX_SPEED_MPS


def test_driving_band_lands_above_the_riding_threshold():
    summary = summarize_track([driving_points(6)], moving_speed_mps=0.1)
    assert summary.moving_average_speed_mps > RIDING_MAX_SPEED_MPS


def test_segments_with_gap_reports_no_data_available_inside_the_gap():
    segments = segments_with_gap(120.0, speed_mps=1.2, n=3)
    timeline = timeline_for(segments, fps=1)

    # The gap spans (first segment's last time_s, second segment's first
    # time_s); its midpoint frame has no source data in either segment.
    gap_start = segments[0][-1].time_s
    gap_end = segments[1][0].time_s
    midpoint_frame = round((gap_start + gap_end) / 2)

    state = dashboard_state_at(timeline, midpoint_frame, segments, moving_speed_mps=1.2)

    assert state.data_available is False
