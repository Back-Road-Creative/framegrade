"""Tests for ``locked_gps_coverage`` — per-LF locked-fix coverage over a window.

Every point in a parsed GPX is a locked fix by construction, so coverage is the
fraction of the LF's footage window carrying a fix. Dead/sparse GPS drives it
below 1.0; a dense (over-sampled) trace saturates at 1.0. This is the signal the
specificity ladder demotes on, and the export imports.
"""

from types import SimpleNamespace

from gomoveshift.analysis.shorts_models import GPSTrackpoint
from gomoveshift.pipeline.gps_quality import (
    MAX_BRIDGED_GAP_SEC,
    locked_gps_coverage,
    locked_gps_duration_coverage,
)


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


class TestShortGapsBridged:
    """A gap of at most MAX_BRIDGED_GAP_SEC between consecutive fixes is covered.

    A phone or enhanced GPX logs on movement, every 1–9 s. Counting only the
    seconds that hold a fix scored the fully-tracked Ireland day 11 drive at
    41% (2026-11-03), under the tier-1 bar. A known position every few seconds
    is coverage; a dropout longer than the bridge still counts as dead, and a
    1 Hz GoPro trace scores exactly as before.
    """

    def test_continuous_half_hz_trace_is_full(self):
        pts = [{"lat": 52.0, "lon": -9.0, "time": float(t)} for t in range(0, 601, 2)]
        assert locked_gps_coverage(pts, 0.0, 600.0) >= 0.95
        assert locked_gps_duration_coverage(pts, 600.0) >= 0.95

    def test_variable_rate_phone_log_is_full(self):
        steps = [1, 1, 2, 3, 1, 4, 9, 1, 5, 2, 6, 1, 7, 3]
        ts, t = [], 0.0
        while t <= 600:
            ts.append(t)
            t += steps[len(ts) % len(steps)]
        pts = [{"lat": 52.0, "lon": -9.0, "time": x} for x in ts]
        assert locked_gps_coverage(pts, 0.0, 600.0) >= 0.95
        assert locked_gps_duration_coverage(pts, 600.0) >= 0.95

    def test_half_hz_trace_with_dead_half_is_half(self):
        pts = [{"lat": 52.0, "lon": -9.0, "time": float(t)} for t in range(0, 301, 2)]
        assert 0.45 <= locked_gps_coverage(pts, 0.0, 600.0) <= 0.55
        assert 0.45 <= locked_gps_duration_coverage(pts, 600.0) <= 0.55

    def test_one_hz_gopro_trace_unchanged(self):
        full = [{"lat": 32.0, "lon": -80.0, "time": float(t)} for t in range(601)]
        assert locked_gps_coverage(full, 0.0, 600.0) == 1.0
        assert locked_gps_duration_coverage(full, 600.0) == 1.0
        assert 0.49 <= locked_gps_coverage(full[:301], 0.0, 600.0) <= 0.51

    def test_long_dropouts_stay_dead(self):
        # 1 Hz with 150 s and 100 s dropouts: neither is bridged.
        ts = [float(t) for t in range(600) if not (100 <= t < 250 or 400 <= t < 500)]
        pts = [{"lat": 32.0, "lon": -80.0, "time": t} for t in ts]
        assert 0.55 <= locked_gps_coverage(pts, 0.0, 600.0) <= 0.6
        assert 0.55 <= locked_gps_duration_coverage(pts, 600.0) <= 0.6

    def test_thin_trace_stays_thin(self):
        # One fix per 20 s: every gap exceeds the bridge, so ~5% as before.
        pts = [{"lat": 52.0, "lon": -9.0, "time": float(t)} for t in range(0, 601, 20)]
        assert locked_gps_coverage(pts, 0.0, 600.0) <= 0.1
        assert locked_gps_duration_coverage(pts, 600.0) <= 0.1

    def test_bridge_boundary(self):
        at = [{"lat": 52.0, "lon": -9.0, "time": t} for t in (0.0, MAX_BRIDGED_GAP_SEC)]
        over = [{"lat": 52.0, "lon": -9.0, "time": t} for t in (0.0, MAX_BRIDGED_GAP_SEC + 1)]
        assert locked_gps_coverage(at, 0.0, 20.0) > locked_gps_coverage(over, 0.0, 20.0)
