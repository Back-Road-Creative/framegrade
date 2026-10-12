"""Truthful metrics for Route Story GPS overlays and summaries."""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.route_story.metrics import RoutePoint, summarize_track


def point(time_s: float, lon: float, elevation_m: float | None) -> RoutePoint:
    return RoutePoint(time_s=time_s, latitude=0.0, longitude=lon, elevation_m=elevation_m)


def test_summary_reports_current_elevation_speed_and_grade_from_last_valid_edge():
    summary = summarize_track(
        [[point(0, 0.0, 100.0), point(10, 0.001, 110.0)]],
        moving_speed_mps=1.0,
    )

    assert summary.current_elevation_m == 110.0
    assert summary.current_speed_mps == pytest.approx(11.12, rel=0.01)
    assert summary.current_grade_percent == pytest.approx(8.99, rel=0.01)
    assert summary.moving_average_speed_mps == summary.current_speed_mps
    assert summary.elapsed_average_speed_mps == summary.current_speed_mps


def test_out_and_back_keeps_ascent_descent_and_signed_grade_distinct_from_net_grade():
    summary = summarize_track(
        [[point(0, 0.0, 0.0), point(10, 0.001, 100.0), point(20, 0.002, 0.0)]],
        moving_speed_mps=1.0,
    )

    assert summary.ascent_m == 100.0
    assert summary.descent_m == 100.0
    assert summary.net_elevation_change_m == 0.0
    assert summary.net_average_grade_percent == pytest.approx(0.0)
    assert summary.uphill_average_grade_percent == pytest.approx(89.93, rel=0.01)
    assert summary.downhill_average_grade_percent == pytest.approx(-89.93, rel=0.01)
    assert summary.absolute_average_grade_percent == pytest.approx(89.93, rel=0.01)


def test_gap_between_segments_is_not_counted_as_distance_or_stopped_time():
    summary = summarize_track(
        [
            [point(0, 0.0, 10.0), point(10, 0.001, 10.0)],
            [point(110, 1.0, 20.0), point(120, 1.001, 20.0)],
        ],
        moving_speed_mps=1.0,
    )

    assert summary.distance_m == pytest.approx(222.39, rel=0.01)
    assert summary.elapsed_seconds == 20.0
    assert summary.unobserved_gap_seconds == 100.0
    assert summary.stopped_seconds == 0.0


def test_stop_edges_are_separate_from_moving_and_elapsed_averages():
    summary = summarize_track(
        [[point(0, 0.0, 0.0), point(10, 0.0, 0.0), point(20, 0.001, 0.0)]],
        moving_speed_mps=1.0,
    )

    assert summary.moving_seconds == 10.0
    assert summary.stopped_seconds == 10.0
    assert summary.moving_average_speed_mps == pytest.approx(11.12, rel=0.01)
    assert summary.elapsed_average_speed_mps == pytest.approx(5.56, rel=0.01)


def test_missing_elevation_is_unavailable_not_zero_grade_or_false_ascent():
    summary = summarize_track(
        [[point(0, 0.0, None), point(10, 0.001, None)]],
        moving_speed_mps=1.0,
    )

    assert summary.current_elevation_m is None
    assert summary.current_grade_percent is None
    assert summary.ascent_m is None
    assert summary.net_average_grade_percent is None
