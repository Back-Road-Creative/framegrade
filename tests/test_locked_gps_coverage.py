"""Tests for ``locked_gps_coverage`` — per-LF locked-fix coverage over a window.

Every point in a parsed GPX is a locked fix by construction, so coverage is the
fraction of the LF's footage window carrying a fix. Dead/sparse GPS drives it
below 1.0; a dense (over-sampled) trace saturates at 1.0. This is the signal the
specificity ladder demotes on, and the export imports.
"""

from types import SimpleNamespace

from gomoveshift.analysis.shorts_models import GPSTrackpoint
from gomoveshift.pipeline.gps_quality import locked_gps_coverage


class TestLockedGpsCoverage:
    def test_full_coverage_every_second(self):
        pts = [{"lat": 32.0, "lon": -80.0, "time": float(t)} for t in range(11)]
        assert locked_gps_coverage(pts, 0.0, 10.0) == 1.0

    def test_dead_second_half_is_half(self):
        pts = [{"lat": 32.0, "lon": -80.0, "time": float(t)} for t in range(6)]
        assert 0.45 <= locked_gps_coverage(pts, 0.0, 10.0) <= 0.6

    def test_points_outside_window_ignored(self):
        # 11 fixes at t=0..10, only 2 inside [100, 110] → low coverage there.
        pts = [{"lat": 32.0, "lon": -80.0, "time": float(t)} for t in range(11)]
        pts += [{"lat": 32.0, "lon": -80.0, "time": t} for t in (100.0, 101.0)]
        assert 0.15 <= locked_gps_coverage(pts, 100.0, 110.0) <= 0.25

    def test_oversampled_caps_at_one(self):
        # 5 Hz fixes over a 1 Hz expectation must not exceed 1.0.
        pts = [{"lat": 32.0, "lon": -80.0, "time": t * 0.2} for t in range(51)]
        assert locked_gps_coverage(pts, 0.0, 10.0) == 1.0

    def test_point_shapes_supported(self):
        # GPSTrackpoint objects and the (t, lat, lon) scorer/shorts tuple shape.
        objs = [GPSTrackpoint(32.0, -80.0, None, float(t)) for t in range(11)]
        tuples = [(float(t), 32.0, -80.0) for t in range(11)]
        assert locked_gps_coverage(objs, 0.0, 10.0) == 1.0
        assert locked_gps_coverage(tuples, 0.0, 10.0) == 1.0

    def test_edge_cases_are_zero(self):
        one = [{"lat": 32.0, "lon": -80.0, "time": 0.0}]
        assert locked_gps_coverage([], 0.0, 10.0) == 0.0  # empty
        assert locked_gps_coverage(None, 0.0, 10.0) == 0.0  # None
        assert locked_gps_coverage(one, 5.0, 5.0) == 0.0  # degenerate window
        assert locked_gps_coverage(one, 5.0, 4.0) == 0.0  # inverted window
        assert locked_gps_coverage([SimpleNamespace(lat=32.0, lon=-80.0)], 0.0, 10.0) == 0.0
