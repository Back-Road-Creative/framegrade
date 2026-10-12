"""Segmented distance-elevation profile geometry and its HUD rendering.

The profile's x-axis is cumulative DISTANCE, not time -- a stopped or slow
stretch never stretches the chart, and a recorder gap never draws a synthetic
climb across it. The cursor derives from the same recording clock as the map
marker so the two never drift apart.
"""

from __future__ import annotations

from PIL import Image

from gomoveshift.pipeline.route_story.hud import draw_profile
from gomoveshift.pipeline.route_story.layout import hud_layout
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.profile_geometry import (
    MIN_PROFILE_SPAN_M,
    ProfileGeometry,
    build_profile,
)

WIDTH, HEIGHT = 1920, 1080


def _pt(time_s, lat, lon, elevation_m):
    return RoutePoint(time_s, lat, lon, elevation_m)


def test_a_gap_between_segments_starts_a_new_profile_segment_with_no_added_distance():
    first = [_pt(0.0, 35.0, -80.0, 100.0), _pt(10.0, 35.001, -80.0, 110.0)]
    second = [_pt(100.0, 35.002, -80.0, 120.0), _pt(110.0, 35.003, -80.0, 130.0)]

    geometry = build_profile([first, second])

    assert len(geometry.segments) == 2
    first_seg, second_seg = geometry.segments
    # The gap between the two source segments adds no distance of its own --
    # the second segment starts exactly where the first one ended.
    assert second_seg.points[0].distance_m == first_seg.points[-1].distance_m
    assert geometry.total_distance_m == second_seg.points[-1].distance_m


def test_a_none_elevation_point_splits_the_segment_into_a_break():
    segment = [
        _pt(0.0, 35.0, -80.0, 100.0),
        _pt(10.0, 35.001, -80.0, 110.0),
        _pt(20.0, 35.002, -80.0, None),
        _pt(30.0, 35.003, -80.0, 120.0),
        _pt(40.0, 35.004, -80.0, 130.0),
    ]

    geometry = build_profile([segment])

    assert len(geometry.segments) == 2
    before, after = geometry.segments
    assert [p.elevation_m for p in before.points] == [100.0, 110.0]
    assert [p.elevation_m for p in after.points] == [120.0, 130.0]
    # The unknown-elevation point still advanced cumulative distance.
    assert after.points[0].distance_m > before.points[-1].distance_m


def test_a_single_point_segment_is_dropped_but_still_advances_distance():
    segment = [
        _pt(0.0, 35.0, -80.0, 100.0),
        _pt(10.0, 35.001, -80.0, None),
        _pt(20.0, 35.002, -80.0, 110.0),
        _pt(30.0, 35.003, -80.0, 120.0),
    ]

    geometry = build_profile([segment])

    # Only two points carry known elevation before/after the None split into a
    # lone leading point (dropped, <2 points) and a trailing 2-point segment.
    assert len(geometry.segments) == 1
    assert geometry.segments[0].points[0].elevation_m == 110.0
    # Distance is cumulative across the whole source, including the dropped
    # point -- the surviving segment does not start back at zero.
    assert geometry.segments[0].points[0].distance_m > 0.0
    assert geometry.total_distance_m == geometry.segments[0].points[-1].distance_m


def test_flat_route_clamps_vertical_span_to_the_minimum():
    segment = [_pt(0.0, 35.0, -80.0, 100.0), _pt(10.0, 35.001, -80.0, 100.05)]

    geometry = build_profile([segment])

    assert geometry.min_elevation_m == 100.0
    assert geometry.max_elevation_m == 100.05
    assert geometry.elevation_span_m == MIN_PROFILE_SPAN_M


def test_cursor_inside_a_gap_holds_at_the_previous_segments_end_distance():
    first = [_pt(0.0, 35.0, -80.0, 100.0), _pt(10.0, 35.001, -80.0, 110.0)]
    second = [_pt(100.0, 35.002, -80.0, 120.0), _pt(110.0, 35.003, -80.0, 130.0)]

    geometry = build_profile([first, second])

    held = geometry.cursor_distance_m(50.0)
    end_of_first = geometry.segments[0].points[-1].distance_m
    assert held == end_of_first


def test_cursor_before_and_after_the_recorded_range():
    segment = [_pt(10.0, 35.0, -80.0, 100.0), _pt(20.0, 35.001, -80.0, 110.0)]

    geometry = build_profile([segment])

    assert geometry.cursor_distance_m(0.0) == 0.0
    assert geometry.cursor_distance_m(999.0) == geometry.total_distance_m


def test_cursor_distance_on_an_empty_geometry_is_zero():
    empty = ProfileGeometry(
        segments=(), total_distance_m=0.0, min_elevation_m=0.0, max_elevation_m=0.0
    )
    assert empty.cursor_distance_m(42.0) == 0.0


def test_cursor_holds_at_the_start_point_when_two_adjacent_points_share_a_timestamp():
    from gomoveshift.pipeline.route_story.profile_geometry import ProfilePoint, ProfileSegment

    # A duplicate/malformed source timestamp inside a LATER segment -- the
    # overall-range guards only look at the very first/last recorded times,
    # so this exercises the inner per-pair defensive branch: never divide by
    # a zero time span, just hold at the first point's distance.
    first = ProfileSegment(
        points=(
            ProfilePoint(distance_m=0.0, elevation_m=90.0, time_s=0.0),
            ProfilePoint(distance_m=2.0, elevation_m=95.0, time_s=5.0),
        )
    )
    second = ProfileSegment(
        points=(
            ProfilePoint(distance_m=5.0, elevation_m=100.0, time_s=10.0),
            ProfilePoint(distance_m=8.0, elevation_m=105.0, time_s=10.0),
            ProfilePoint(distance_m=10.0, elevation_m=110.0, time_s=20.0),
        )
    )
    geometry = ProfileGeometry(
        segments=(first, second), total_distance_m=10.0, min_elevation_m=90.0, max_elevation_m=110.0
    )

    assert geometry.cursor_distance_m(10.0) == 5.0


def test_cursor_interpolates_within_a_segments_time_range():
    segment = [_pt(0.0, 35.0, -80.0, 100.0), _pt(10.0, 35.001, -80.0, 110.0)]

    geometry = build_profile([segment])

    midpoint = geometry.cursor_distance_m(5.0)
    assert 0.0 < midpoint < geometry.total_distance_m


def test_draw_profile_on_a_blank_image_only_touches_the_layout_box():
    layout = hud_layout(WIDTH, HEIGHT)
    background = (7, 18, 25)
    image = Image.new("RGB", (WIDTH, HEIGHT), background)

    segment = [_pt(0.0, 35.0, -80.0, 100.0), _pt(10.0, 35.001, -80.0, 200.0)]
    geometry = build_profile([segment])
    draw_profile(image, geometry, 5.0, layout)

    pixels = image.load()
    # Above the profile box stays untouched.
    assert pixels[WIDTH // 2, int(layout.profile_top) - 5] == background
    # Below the profile box (into the dashboard rail) stays untouched by this call.
    below_y = int(layout.profile_top + layout.profile_height) + 5
    assert pixels[WIDTH // 2, below_y] == background


def test_draw_profile_with_empty_geometry_draws_nothing():
    layout = hud_layout(WIDTH, HEIGHT)
    background = (7, 18, 25)
    image = Image.new("RGB", (WIDTH, HEIGHT), background)

    empty = ProfileGeometry(
        segments=(), total_distance_m=0.0, min_elevation_m=0.0, max_elevation_m=0.0
    )
    draw_profile(image, empty, 0.0, layout)

    pixels = image.load()
    for x in range(layout.profile_left, layout.profile_right, 40):
        assert pixels[x, int(layout.profile_top) + int(layout.profile_height) // 2] == background
