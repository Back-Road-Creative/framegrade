"""Tests for display-only GPS zig-zag smoothing.

Pure synthetic geometry only -- no real clock, no I/O.
"""

from __future__ import annotations

import math

import pytest

from gomoveshift.pipeline.route_story.display_track import (
    DISPLAY_SMOOTHING_RECIPE_VERSION,
    smooth_display_segments,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint, distance_m

METERS_PER_DEGREE_LAT = math.pi / 180 * 6_371_008.8


def _offset_point(
    lat0: float, lon0: float, east_m: float, north_m: float, time_s: float, **kwargs
) -> RoutePoint:
    lat = lat0 + north_m / METERS_PER_DEGREE_LAT
    lon = lon0 + east_m / (METERS_PER_DEGREE_LAT * math.cos(math.radians(lat0)))
    return RoutePoint(time_s=time_s, latitude=lat, longitude=lon, **kwargs)


def _straight_segment(n: int, spacing_m: float, lat0: float = 40.0, lon0: float = -74.0):
    """A straight north-heading trail sampled every ``spacing_m`` metres."""
    return [_offset_point(lat0, lon0, 0.0, i * spacing_m, time_s=i * 2.0) for i in range(n)]


def _jittered_straight_segment(
    n: int, spacing_m: float, jitter_m: float, lat0: float = 40.0, lon0: float = -74.0
):
    points = []
    for i in range(n):
        east = jitter_m if i % 2 == 0 else -jitter_m
        points.append(_offset_point(lat0, lon0, east, i * spacing_m, time_s=i * 2.0))
    return points


def _hairpin_segment(
    leg_m: float = 40.0, spacing_m: float = 5.0, lat0: float = 40.0, lon0: float = -74.0
):
    """A synthetic switchback: north leg, sharp apex, then a leg back south-ish east-shifted."""
    n_leg = int(leg_m / spacing_m)
    points = []
    t = 0.0
    for i in range(n_leg + 1):
        points.append(_offset_point(lat0, lon0, 0.0, i * spacing_m, time_s=t))
        t += 2.0
    apex_north = n_leg * spacing_m
    apex_east = 0.0
    for i in range(1, n_leg + 1):
        points.append(
            _offset_point(lat0, lon0, apex_east + i * 2.0, apex_north - i * spacing_m, time_s=t)
        )
        t += 2.0
    return points


def _heading_deg(a: RoutePoint, b: RoutePoint) -> float:
    d_lat = b.latitude - a.latitude
    d_lon = (b.longitude - a.longitude) * math.cos(math.radians((a.latitude + b.latitude) / 2))
    return math.degrees(math.atan2(d_lon, d_lat))


def test_recipe_version_constant():
    assert DISPLAY_SMOOTHING_RECIPE_VERSION == "display-smooth-1"


def test_elevation_and_time_pass_through_untouched():
    segment = [
        _offset_point(
            40.0, -74.0, e, 0.0, time_s=float(i), elevation_m=100.0 + i, heart_rate_bpm=120.0
        )
        for i, e in enumerate([0.0, 4.0, -4.0, 4.0, -4.0, 0.0])
    ]
    result = smooth_display_segments([segment])
    out = result.segments[0]
    assert [p.time_s for p in out] == [p.time_s for p in segment]
    assert [p.elevation_m for p in out] == [p.elevation_m for p in segment]
    assert [p.heart_rate_bpm for p in out] == [p.heart_rate_bpm for p in segment]


def test_segment_and_point_counts_and_time_sequence_preserved():
    seg_a = _straight_segment(6, 5.0)
    seg_b = _jittered_straight_segment(5, 5.0, 3.0, lat0=41.0)
    result = smooth_display_segments([seg_a, seg_b])
    assert len(result.segments) == 2
    assert len(result.segments[0]) == len(seg_a)
    assert len(result.segments[1]) == len(seg_b)
    assert [p.time_s for p in result.segments[0]] == [p.time_s for p in seg_a]
    assert [p.time_s for p in result.segments[1]] == [p.time_s for p in seg_b]


def test_never_smooths_across_segment_boundary():
    # Two segments placed far apart geographically; smoothing must not blend them.
    seg_a = _straight_segment(5, 5.0, lat0=40.0)
    seg_b = _straight_segment(5, 5.0, lat0=50.0)
    result = smooth_display_segments([seg_a, seg_b])
    for original, smoothed in zip(seg_a, result.segments[0]):
        assert distance_m(original, smoothed) < 100.0
    for original, smoothed in zip(seg_b, result.segments[1]):
        assert distance_m(original, smoothed) < 100.0


def test_first_and_last_point_pinned_exactly():
    segment = _jittered_straight_segment(9, 5.0, 4.0)
    result = smooth_display_segments([segment])
    out = result.segments[0]
    assert out[0].latitude == segment[0].latitude
    assert out[0].longitude == segment[0].longitude
    assert out[-1].latitude == segment[-1].latitude
    assert out[-1].longitude == segment[-1].longitude


def _straight_with_outlier(n: int, spacing_m: float, outlier_index: int, outlier_offset_m: float):
    """A regular straight trail with one point pushed far sideways."""
    points = list(_straight_segment(n, spacing_m))
    original = points[outlier_index]
    points[outlier_index] = _offset_point(
        40.0,
        -74.0,
        outlier_offset_m,
        outlier_index * spacing_m,
        time_s=original.time_s,
    )
    return points


def test_no_point_moves_more_than_max_shift_m():
    segment = _straight_with_outlier(15, 3.0, outlier_index=7, outlier_offset_m=10.0)
    result = smooth_display_segments([segment], max_shift_m=5.0)
    out = result.segments[0]
    for original, smoothed in zip(segment, out):
        assert distance_m(original, smoothed) <= 5.0 + 1e-6
    assert result.max_shift_m_observed <= 5.0 + 1e-6


def test_points_clamped_counted_when_shift_would_exceed_limit():
    segment = _straight_with_outlier(15, 3.0, outlier_index=7, outlier_offset_m=10.0)
    result = smooth_display_segments([segment], max_shift_m=5.0)
    assert result.points_clamped > 0


def test_switchback_apex_survives_smoothing():
    segment = _hairpin_segment(leg_m=40.0, spacing_m=5.0)
    result = smooth_display_segments([segment], window_m=25.0, max_shift_m=4.0)
    out = result.segments[0]
    apex_index = len(segment) // 2
    apex_original = segment[apex_index]
    apex_smoothed = out[apex_index]
    assert distance_m(apex_original, apex_smoothed) <= 4.0 + 1e-6

    before = out[apex_index - 2]
    after = out[apex_index + 2]
    apex = out[apex_index]
    heading_in = _heading_deg(before, apex)
    heading_out = _heading_deg(apex, after)
    delta = abs(heading_out - heading_in) % 360
    if delta > 180:
        delta = 360 - delta
    assert delta > 150.0


def test_straight_trail_jitter_reduced_and_length_preserved():
    n = 41
    spacing_m = 5.0
    jitter_m = 4.0
    segment = _jittered_straight_segment(n, spacing_m, jitter_m)
    result = smooth_display_segments([segment], window_m=25.0, max_shift_m=12.0)
    out = result.segments[0]

    def lateral_rms(points):
        offsets = []
        for p in points:
            east = (p.longitude - points[0].longitude) * math.cos(math.radians(points[0].latitude))
            offsets.append(east * METERS_PER_DEGREE_LAT)
        mean = sum(offsets) / len(offsets)
        return math.sqrt(sum((o - mean) ** 2 for o in offsets) / len(offsets))

    raw_rms = lateral_rms(segment)
    smoothed_rms = lateral_rms(out)
    assert smoothed_rms <= raw_rms * 0.4

    def total_length(points):
        return sum(distance_m(a, b) for a, b in zip(points, points[1:]))

    true_length = spacing_m * (n - 1)
    smoothed_length = total_length(out)
    assert smoothed_length <= true_length * 1.03


def test_stationary_cluster_does_not_raise_or_nan():
    points = [
        _offset_point(40.0, -74.0, (i % 3) * 0.3, (i % 2) * 0.2, time_s=i * 30.0) for i in range(20)
    ]
    result = smooth_display_segments([points])
    out = result.segments[0]
    for p in out:
        assert not math.isnan(p.latitude)
        assert not math.isnan(p.longitude)


def test_input_not_mutated():
    segment = _jittered_straight_segment(10, 5.0, 4.0)
    original_copy = list(segment)
    smooth_display_segments([segment])
    assert segment == original_copy


def test_short_segment_returned_unchanged():
    segment = _straight_segment(2, 5.0)
    result = smooth_display_segments([segment])
    assert result.segments[0] == segment


def test_non_positive_window_m_raises():
    segment = _straight_segment(5, 5.0)
    with pytest.raises(ValueError):
        smooth_display_segments([segment], window_m=0.0)


def test_non_positive_max_shift_m_raises():
    segment = _straight_segment(5, 5.0)
    with pytest.raises(ValueError):
        smooth_display_segments([segment], max_shift_m=-1.0)


def test_recipe_field_set():
    segment = _straight_segment(5, 5.0)
    result = smooth_display_segments([segment])
    assert result.recipe == DISPLAY_SMOOTHING_RECIPE_VERSION


def test_mean_shift_m_is_nonnegative_float():
    segment = _jittered_straight_segment(9, 5.0, 3.0)
    result = smooth_display_segments([segment])
    assert result.mean_shift_m >= 0.0
