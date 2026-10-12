"""The quality filter applies its working passes (null-island,
spatial outlier, sub-second, temporal teleport) to the REAL point types the
production pipeline feeds it — positional ``(t, lat, lon)`` tuples and
``GPSTrackpoint`` objects — not only dict points.

Previously ``filter_high_quality_points`` read lat/lon/time via ``_get``,
which returns ``None`` on a positional tuple (no ``.lat``/``["lat"]``), so the
null-island/spatial/teleport gates were a silent no-op on the tuple path the
scorer + shorts-OST trace actually use. ``GPSTrackpoint`` (lat/lon/
elevation/timestamp, no hdop/speed) worked for null-island but its honest
no-hdop/no-speed reality is asserted here so the gates that DON'T apply
are documented, not pretended.
"""

from __future__ import annotations

from gomoveshift.analysis.shorts_models import GPSTrackpoint
from gomoveshift.pipeline.gps_quality import filter_high_quality_points


def _tuple(t, lat, lon):
    return (t, lat, lon)


class TestPositionalTupleNullIsland:
    def test_null_island_tuple_dropped(self):
        # A leading (0,0) cold-start fix in (t, lat, lon) shape MUST be dropped.
        pts = [
            _tuple(0.0, 0.0, 0.0),
            _tuple(2.0, 35.5, -82.5),
            _tuple(4.0, 35.6, -82.6),
        ]
        out = filter_high_quality_points(pts)
        assert _tuple(0.0, 0.0, 0.0) not in out
        assert all(not (p[1] == 0.0 and p[2] == 0.0) for p in out)

    def test_bare_two_tuple_null_island_dropped(self):
        # A bare (lat, lon) 2-tuple null-island sentinel is also dropped.
        pts = [(0.0, 0.0), (35.5, -82.5), (35.6, -82.6)]
        out = filter_high_quality_points(pts)
        assert (0.0, 0.0) not in out

    def test_sub_second_tuple_deduped(self):
        # Two tuple fixes < min_gap_sec apart: the later is dropped.
        pts = [
            _tuple(0.0, 35.5, -82.5),
            _tuple(0.4, 35.5, -82.5),  # 0.4s later — dropped
            _tuple(3.0, 35.6, -82.6),
        ]
        out = filter_high_quality_points(pts, min_gap_sec=1.0)
        assert len(out) == 2
        assert out[0][0] == 0.0 and out[1][0] == 3.0

    def test_spatial_outlier_tuple_dropped(self):
        # A tuple fix > 500 km from the median cluster (cold-start wrong-sat lock).
        pts = [
            _tuple(0.0, 35.50, -82.50),
            _tuple(1.0, 35.51, -82.51),
            _tuple(2.0, 35.52, -82.52),
            _tuple(3.0, 25.00, -77.00),  # Bahamas — ~1200 km out
        ]
        out = filter_high_quality_points(pts)
        assert _tuple(3.0, 25.00, -77.00) not in out
        assert len(out) == 3

    def test_temporal_teleport_tuple_head_stripped(self):
        # A leading fix joined to the body by a physically impossible seam
        # (huge km over ~1 s) is stripped on the tuple path too.
        body = [_tuple(float(100 + i), 35.5 + i * 0.001, -82.5) for i in range(8)]
        teleport_head = _tuple(99.5, 36.5, -82.5)  # ~110 km in 0.5 s = impossible
        pts = [teleport_head] + body
        out = filter_high_quality_points(pts)
        assert teleport_head not in out


class TestGPSTrackpointObject:
    def test_null_island_trackpoint_dropped(self):
        pts = [
            GPSTrackpoint(lat=0.0, lon=0.0, elevation=None, timestamp=0.0),
            GPSTrackpoint(lat=35.5, lon=-82.5, elevation=10.0, timestamp=2.0),
            GPSTrackpoint(lat=35.6, lon=-82.6, elevation=11.0, timestamp=4.0),
        ]
        out = filter_high_quality_points(pts)
        assert all(not (p.lat == 0.0 and p.lon == 0.0) for p in out)
        assert len(out) == 2

    def test_spatial_outlier_trackpoint_dropped(self):
        pts = [
            GPSTrackpoint(lat=35.50, lon=-82.50, elevation=None, timestamp=0.0),
            GPSTrackpoint(lat=35.51, lon=-82.51, elevation=None, timestamp=1.0),
            GPSTrackpoint(lat=35.52, lon=-82.52, elevation=None, timestamp=2.0),
            GPSTrackpoint(lat=25.00, lon=-77.00, elevation=None, timestamp=3.0),
        ]
        out = filter_high_quality_points(pts)
        assert len(out) == 3

    def test_sub_second_trackpoint_deduped(self):
        pts = [
            GPSTrackpoint(lat=35.5, lon=-82.5, elevation=None, timestamp=0.0),
            GPSTrackpoint(lat=35.5, lon=-82.5, elevation=None, timestamp=0.3),
            GPSTrackpoint(lat=35.6, lon=-82.6, elevation=None, timestamp=3.0),
        ]
        out = filter_high_quality_points(pts, min_gap_sec=1.0)
        assert len(out) == 2

    def test_no_hdop_no_speed_field_keeps_point(self):
        # GPSTrackpoint carries neither hdop nor speed, so those gates are
        # honestly inert — a valid trackpoint is NOT dropped for missing them.
        pts = [
            GPSTrackpoint(lat=35.5, lon=-82.5, elevation=None, timestamp=0.0),
            GPSTrackpoint(lat=35.6, lon=-82.6, elevation=None, timestamp=5.0),
        ]
        out = filter_high_quality_points(pts, max_hdop=0.1, min_speed_mps=1000.0)
        assert len(out) == 2


class TestDictPathUnchanged:
    def test_dict_null_island_still_dropped(self):
        pts = [
            {"time": 0.0, "lat": 0.0, "lon": 0.0},
            {"time": 2.0, "lat": 35.5, "lon": -82.5},
        ]
        out = filter_high_quality_points(pts)
        assert len(out) == 1

    def test_dict_hdop_gate_still_fires(self):
        pts = [
            {"time": 0.0, "lat": 35.5, "lon": -82.5, "hdop": 9.0},
            {"time": 5.0, "lat": 35.6, "lon": -82.6, "hdop": 1.0},
        ]
        out = filter_high_quality_points(pts, max_hdop=3.0)
        assert len(out) == 1
        assert out[0]["hdop"] == 1.0
