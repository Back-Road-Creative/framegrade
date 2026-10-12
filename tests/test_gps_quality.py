"""Tests for ``gomoveshift.pipeline.gps_quality``.

Covers:

* High-HDOP drop
* Stationary-jitter drop (low speed)
* Sub-second gap dedup
* Good-driving-point retention
* Road selection: drops tiny, caps count, preserves order
* ``trim_gps_to_clip_timeline`` real implementation
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest  # noqa: F401

from gomoveshift.pipeline.gps_quality import (
    filter_high_quality_points,
    select_significant_roads,
    trim_gps_to_clip_timeline,
)


def _p(**kwargs):
    return SimpleNamespace(**kwargs)


def _ts(seconds_offset: float) -> datetime:
    base = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=seconds_offset)


# filter_high_quality_points


class TestFilterHighQualityPoints:
    def test_filter_drops_high_hdop(self):
        points = [
            _p(hdop=1.2, speed_mps=10.0, time=_ts(0)),
            _p(hdop=8.5, speed_mps=10.0, time=_ts(2)),
            _p(hdop=2.0, speed_mps=10.0, time=_ts(4)),
        ]
        out = filter_high_quality_points(points, max_hdop=3.0)
        assert len(out) == 2
        assert [p.hdop for p in out] == [1.2, 2.0]

    def test_filter_drops_stationary_jitter(self):
        points = [
            _p(hdop=1.0, speed_mps=10.0, time=_ts(0)),
            _p(hdop=1.0, speed_mps=0.2, time=_ts(2)),
            _p(hdop=1.0, speed_mps=12.0, time=_ts(4)),
        ]
        out = filter_high_quality_points(points, min_speed_mps=1.0)
        assert len(out) == 2
        assert [p.speed_mps for p in out] == [10.0, 12.0]

    def test_filter_dedupes_sub_second_points(self):
        points = [
            _p(hdop=1.0, speed_mps=10.0, time=_ts(0.0)),
            _p(hdop=1.0, speed_mps=10.0, time=_ts(0.3)),
            _p(hdop=1.0, speed_mps=10.0, time=_ts(0.7)),
            _p(hdop=1.0, speed_mps=10.0, time=_ts(1.5)),
        ]
        out = filter_high_quality_points(points, min_gap_sec=1.0)
        assert len(out) == 2

    def test_filter_keeps_good_driving_points(self):
        points = [_p(hdop=1.0, speed_mps=15.0, time=_ts(i)) for i in range(0, 20, 2)]
        out = filter_high_quality_points(points)
        assert len(out) == len(points)

    def test_filter_keeps_points_with_missing_fields(self):
        points = [_p(time=_ts(0)), _p(time=_ts(5))]
        out = filter_high_quality_points(points)
        assert len(out) == 2

    def test_filter_empty_input(self):
        assert filter_high_quality_points([]) == []

    def test_filter_drops_null_island_points(self):
        # (0.0, 0.0) is a GPS no-lock sentinel — must be rejected even without HDOP/speed data
        points = [
            _p(lat=35.87, lon=-82.52, time=_ts(0)),  # real NC point
            _p(lat=0.0, lon=0.0, time=_ts(2)),  # GPS no-lock
            _p(lat=35.90, lon=-82.48, time=_ts(4)),  # real NC point
        ]
        out = filter_high_quality_points(points)
        lats = [p.lat for p in out]
        assert 0.0 not in lats, "null-island (0,0) points must be dropped"
        assert len(out) == 2

    def test_filter_drops_spatial_outliers(self):
        # GoPro cold-start anomaly: camera briefly reports Bahamas coords before NC lock.
        # ~1000 km from the true NC cluster — must be filtered as a spatial outlier.
        nc_points = [
            _p(lat=35.87 + i * 0.01, lon=-82.52 + i * 0.01, time=_ts(i * 2)) for i in range(20)
        ]
        bahamas = _p(lat=26.91, lon=-75.42, time=_ts(100))  # Inagua, Bahamas
        out = filter_high_quality_points(nc_points + [bahamas])
        lats = [p.lat for p in out]
        assert 26.91 not in lats, "Bahamas outlier point must be filtered as a spatial outlier"
        assert len(out) == 20

    def test_filter_spatial_outlier_does_not_remove_legitimate_long_drive(self):
        # A NC→PA drive spans ~800 km — should NOT be split as outliers.
        nc_start = [
            _p(lat=35.87 + i * 0.05, lon=-82.52 + i * 0.02, time=_ts(i * 60)) for i in range(10)
        ]
        pa_end = [
            _p(lat=39.90 + i * 0.01, lon=-77.00 + i * 0.01, time=_ts(600 + i * 60))
            for i in range(10)
        ]
        out = filter_high_quality_points(nc_start + pa_end)
        assert len(out) == 20, "legitimate cross-state drive points must all survive"

    def test_filter_all_null_island_returns_empty(self):
        points = [_p(lat=0.0, lon=0.0, time=_ts(i)) for i in range(5)]
        out = filter_high_quality_points(points)
        assert out == [], "all null-island input → empty output"

    def test_filter_strips_sub_second_teleport_lead(self):
        # A cold-start stale lock ~400 km away, then a continuous body
        # one second apart. The implied speed across the seam is physically
        # impossible → strip the lead. The km-only 500 km median test would keep
        # it (< 500 km from the blended median).
        stale = [_p(lat=36.20, lon=-81.70 + i * 0.0001, time=_ts(i)) for i in range(3)]
        body = [
            _p(lat=32.40 + i * 0.001, lon=-80.40 + i * 0.001, time=_ts(3 + i)) for i in range(120)
        ]
        out = filter_high_quality_points(stale + body)
        lats = [round(p.lat, 2) for p in out]
        assert 36.20 not in lats, "sub-second teleport lead must be stripped"
        assert len(out) == len(body)

    def test_filter_keeps_far_endpoint_over_long_dropout(self):
        # A real far endpoint reached after a long GPS dropout
        # (ferry/tunnel) — same spatial jump as a teleport but the implied speed
        # is sane, so it must be preserved. The coarse 500 km median test could
        # split a cross-state drive; the temporal strip never does.
        leg_a = [_p(lat=36.20, lon=-81.70, time=_ts(i * 60)) for i in range(10)]
        # ~25 km jump after a 10-minute dropout, then a continuous body.
        leg_b = [
            _p(lat=36.55 + i * 0.001, lon=-81.30 + i * 0.001, time=_ts(1200 + i * 60))
            for i in range(10)
        ]
        out = filter_high_quality_points(leg_a + leg_b)
        assert len(out) == 20, "legit far endpoint over a long dropout must survive"


# select_significant_roads


class TestSelectSignificantRoads:
    def test_select_significant_roads_drops_tiny(self):
        roads = [
            _p(name="US-17", duration_sec=600.0, length_m=20_000.0),
            _p(name="Side Rd", duration_sec=5.0, length_m=100.0),
            _p(name="SC-170", duration_sec=300.0, length_m=10_000.0),
        ]
        out = select_significant_roads(roads)
        assert [r.name for r in out] == ["US-17", "SC-170"]

    def test_select_significant_roads_caps_count(self):
        roads = [
            _p(name="A", duration_sec=600.0, length_m=5000.0),
            _p(name="B", duration_sec=500.0, length_m=5000.0),
            _p(name="C", duration_sec=400.0, length_m=5000.0),
            _p(name="D", duration_sec=300.0, length_m=5000.0),
            _p(name="E", duration_sec=200.0, length_m=5000.0),
        ]
        out = select_significant_roads(roads, max_count=3)
        assert len(out) == 3

    def test_select_significant_roads_preserves_order(self):
        roads = [
            _p(name="A", duration_sec=200.0, length_m=5000.0),
            _p(name="B", duration_sec=600.0, length_m=5000.0),
            _p(name="C", duration_sec=400.0, length_m=5000.0),
            _p(name="D", duration_sec=500.0, length_m=5000.0),
        ]
        out = select_significant_roads(roads, max_count=3)
        assert [r.name for r in out] == ["B", "C", "D"]

    def test_select_empty_input(self):
        assert select_significant_roads([]) == []

    def test_select_drops_unnamed(self):
        roads = [
            _p(name=None, duration_sec=600.0, length_m=5000.0),
            _p(name="", duration_sec=600.0, length_m=5000.0),
            _p(name="Good", duration_sec=600.0, length_m=5000.0),
        ]
        out = select_significant_roads(roads)
        assert [r.name for r in out] == ["Good"]


# trim_gps_to_clip_timeline


def _make_points(times):
    return [(float(t), 40.0 + i * 0.001, -122.0 + i * 0.001) for i, t in enumerate(times)]


def test_trim_returns_only_points_in_included_ranges():
    points = _make_points([0, 5, 10, 15, 20, 25, 30])
    kept = trim_gps_to_clip_timeline(points, [(4.0, 11.0), (24.0, 31.0)])
    assert [p[0] for p in kept] == [5.0, 10.0, 25.0, 30.0]


def test_trim_handles_multi_segment_lf():
    points = _make_points(range(0, 100, 10))
    included = [(0.0, 25.0), (50.0, 65.0), (85.0, 100.0)]
    kept = trim_gps_to_clip_timeline(points, included)
    assert [p[0] for p in kept] == [0.0, 10.0, 20.0, 50.0, 60.0, 90.0]


def test_trim_single_segment_short_is_identity():
    points = _make_points([12.0, 13.0, 14.0, 15.0, 16.0])
    kept = trim_gps_to_clip_timeline(points, [(12.0, 16.0)])
    assert len(kept) == len(points)


def test_trim_empty_ranges_returns_empty():
    points = _make_points([0, 1, 2, 3])
    assert trim_gps_to_clip_timeline(points, []) == []


def test_trim_accepts_time_key_callable_for_waypoints():
    base = datetime(2026, 4, 22, 10, 0, 0)

    @dataclass
    class WP:
        timestamp: datetime

    wps = [WP(timestamp=base + timedelta(seconds=s)) for s in [0, 30, 60, 90, 120]]
    kept = trim_gps_to_clip_timeline(
        wps,
        [(25.0, 65.0)],
        time_key=lambda w: (w.timestamp - base).total_seconds(),
    )
    assert [w.timestamp for w in kept] == [
        base + timedelta(seconds=30),
        base + timedelta(seconds=60),
    ]


def test_trim_none_segments_is_identity():
    points = _make_points([1, 2, 3])
    assert trim_gps_to_clip_timeline(points, None) == points
