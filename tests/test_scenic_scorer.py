"""Tests for gomoveshift.analysis.scenic_scorer — ScenicScorer."""

import pytest

from gomoveshift.analysis.scenic_scorer import ScenicScorer
from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint


# ── Helpers ──────────────────────────────────────────────────────────


def _make_gps(
    n: int = 20,
    lat_start: float = 51.77,
    lon_start: float = -9.98,
    elev_start: float = 100.0,
    lat_step: float = 0.001,
    lon_step: float = 0.0,
    elev_step: float = 0.0,
    dt: float = 1.0,
    t0: float = 1000.0,
) -> GPSData:
    """Build a simple GPS track with configurable increments."""
    pts = []
    for i in range(n):
        pts.append(
            GPSTrackpoint(
                lat=lat_start + i * lat_step,
                lon=lon_start + i * lon_step,
                elevation=elev_start + i * elev_step,
                timestamp=t0 + i * dt,
            )
        )
    return GPSData(trackpoints=pts)


# ── get_elevation_gain_rate ─────────────────────────────────────────


class TestElevationGainRate:
    def test_flat_route_zero_gain(self):
        gps = _make_gps(n=30, elev_step=0.0)
        scorer = ScenicScorer(gps)
        rate = scorer.get_elevation_gain_rate(1000.0, 1029.0)
        assert rate == pytest.approx(0.0)

    def test_climbing_route_positive_gain(self):
        """Steady climb of 10m per second = 600m/min."""
        gps = _make_gps(n=30, elev_step=10.0)
        scorer = ScenicScorer(gps)
        rate = scorer.get_elevation_gain_rate(1000.0, 1029.0)
        assert rate > 0.0
        # 29 intervals * 10m gain each = 290m over 29/60 min ≈ 600 m/min
        assert rate == pytest.approx(600.0, rel=0.05)

    def test_descending_route_zero_gain(self):
        """Downhill only: elevation gain rate should be 0."""
        gps = _make_gps(n=30, elev_step=-5.0)
        scorer = ScenicScorer(gps)
        rate = scorer.get_elevation_gain_rate(1000.0, 1029.0)
        assert rate == pytest.approx(0.0)

    def test_single_point_returns_zero(self):
        gps = GPSData(trackpoints=[GPSTrackpoint(51.77, -9.98, 100.0, 1000.0)])
        scorer = ScenicScorer(gps)
        assert scorer.get_elevation_gain_rate(999.0, 1001.0) == 0.0

    def test_no_points_in_range(self):
        gps = _make_gps(n=10, t0=1000.0)
        scorer = ScenicScorer(gps)
        # Query a range that has no points
        assert scorer.get_elevation_gain_rate(5000.0, 6000.0) == 0.0


# ── get_turn_intensity ──────────────────────────────────────────────


class TestTurnIntensity:
    def test_straight_road_low_intensity(self):
        """Straight north: no bearing changes."""
        gps = _make_gps(n=20, lat_step=0.001, lon_step=0.0)
        scorer = ScenicScorer(gps)
        intensity = scorer.get_turn_intensity(1000.0, 1019.0)
        assert intensity < 5.0  # near-zero

    def test_winding_road_high_intensity(self):
        """Zig-zag pattern: alternate east/west movement."""
        pts = []
        for i in range(20):
            lon_offset = 0.01 * (1 if i % 2 == 0 else -1)
            pts.append(
                GPSTrackpoint(
                    lat=51.77 + i * 0.001,
                    lon=-9.98 + lon_offset,
                    elevation=100.0,
                    timestamp=1000.0 + i,
                )
            )
        gps = GPSData(trackpoints=pts)
        scorer = ScenicScorer(gps)
        intensity = scorer.get_turn_intensity(1000.0, 1019.0)
        assert intensity > 10.0

    def test_two_points_returns_zero(self):
        """Need >= 3 points for turn calculation."""
        gps = _make_gps(n=2)
        scorer = ScenicScorer(gps)
        assert scorer.get_turn_intensity(1000.0, 1001.0) == 0.0

    def test_stationary_jitter_returns_zero(self):
        """Near-stationary fixes (a parked idle) must not inflate turn intensity.

        Bearings between fixes closer than the GPS noise floor are jitter, not
        real heading changes; before the min-move filter a slow crawl scored as
        a winding road. All points here stay within ~1-2 m of each other, so no
        pair clears _MIN_MOVE_M and the score collapses to 0.
        """
        pts = []
        for i in range(30):
            jitter = 0.00001 * (1 if i % 2 == 0 else -1)  # ~1 m alternating
            pts.append(
                GPSTrackpoint(
                    lat=51.77 + jitter,
                    lon=-9.98 - jitter,
                    elevation=100.0,
                    timestamp=1000.0 + i,
                )
            )
        gps = GPSData(trackpoints=pts)
        scorer = ScenicScorer(gps)
        assert scorer.get_turn_intensity(1000.0, 1029.0) == 0.0

    def test_intensity_capped_at_100(self):
        """Even extreme turns should be capped at 100."""
        # Rapid direction reversals at high frequency
        pts = []
        for i in range(100):
            direction = 1 if i % 2 == 0 else -1
            pts.append(
                GPSTrackpoint(
                    lat=51.77 + i * 0.0001,
                    lon=-9.98 + direction * 0.05,
                    elevation=100.0,
                    timestamp=1000.0 + i * 0.1,  # 10Hz for high per-minute rate
                )
            )
        gps = GPSData(trackpoints=pts)
        scorer = ScenicScorer(gps)
        intensity = scorer.get_turn_intensity(1000.0, 1009.9)
        assert intensity <= 100.0


# ── get_speed_variance ──────────────────────────────────────────────


class TestSpeedVariance:
    def test_constant_speed_low_variance(self):
        """Uniform spacing = constant speed = near-zero std."""
        gps = _make_gps(n=30, lat_step=0.001, dt=1.0)
        scorer = ScenicScorer(gps)
        var = scorer.get_speed_variance(1000.0, 1029.0)
        assert var < 1.0

    def test_varying_speed_high_variance(self):
        """Alternating large/small steps = high speed variance."""
        pts = []
        for i in range(30):
            step = 0.005 if i % 2 == 0 else 0.0005
            pts.append(
                GPSTrackpoint(
                    lat=51.77 + i * step,
                    lon=-9.98,
                    elevation=100.0,
                    timestamp=1000.0 + i,
                )
            )
        gps = GPSData(trackpoints=pts)
        scorer = ScenicScorer(gps)
        var = scorer.get_speed_variance(1000.0, 1029.0)
        assert var > 10.0

    def test_too_few_points_returns_zero(self):
        gps = _make_gps(n=2)
        scorer = ScenicScorer(gps)
        assert scorer.get_speed_variance(1000.0, 1001.0) == 0.0


# ── score_segment ───────────────────────────────────────────────────


class TestScoreSegment:
    def test_score_with_good_data(self):
        """Climbing, winding route should score higher than flat straight."""
        # Good scenic route: climbing + turns
        pts_scenic = []
        for i in range(30):
            pts_scenic.append(
                GPSTrackpoint(
                    lat=51.77 + i * 0.001,
                    lon=-9.98 + 0.005 * (1 if i % 2 == 0 else -1),
                    elevation=100.0 + i * 5.0,
                    timestamp=1000.0 + i,
                )
            )
        gps_scenic = GPSData(trackpoints=pts_scenic)
        scorer_scenic = ScenicScorer(gps_scenic)
        scenic_score = scorer_scenic.score_segment(1000.0, 1029.0)

        # Flat straight route
        gps_flat = _make_gps(n=30, elev_step=0.0, lon_step=0.0)
        scorer_flat = ScenicScorer(gps_flat)
        flat_score = scorer_flat.score_segment(1000.0, 1029.0)

        assert scenic_score > flat_score

    def test_score_no_data_returns_neutral(self):
        """With < 2 points in range, score defaults to 50."""
        gps = GPSData(trackpoints=[GPSTrackpoint(51.77, -9.98, 100.0, 1000.0)])
        scorer = ScenicScorer(gps)
        assert scorer.score_segment(999.0, 1001.0) == pytest.approx(50.0)

    def test_score_range_is_bounded(self):
        """Score should be >= 0 and <= 100."""
        gps = _make_gps(n=50, elev_step=20.0)
        scorer = ScenicScorer(gps)
        score = scorer.score_segment(1000.0, 1049.0)
        assert 0.0 <= score <= 100.0

    def test_poi_bonus_increases_score(self):
        """Setting a POI timestamp in the segment range boosts the score."""
        gps = _make_gps(n=30, elev_step=0.0)
        scorer = ScenicScorer(gps)
        base_score = scorer.score_segment(1000.0, 1029.0)

        scorer.set_poi_timestamps([1015.0])
        poi_score = scorer.score_segment(1000.0, 1029.0)
        assert poi_score > base_score

    def test_poi_outside_range_no_bonus(self):
        """POI outside the segment range should not boost score."""
        gps = _make_gps(n=30, elev_step=0.0)
        scorer = ScenicScorer(gps)
        base_score = scorer.score_segment(1000.0, 1029.0)

        scorer.set_poi_timestamps([9999.0])
        poi_score = scorer.score_segment(1000.0, 1029.0)
        assert poi_score == pytest.approx(base_score)

    def test_none_elevation_handled(self):
        """Points with None elevation should not crash."""
        pts = [GPSTrackpoint(51.77 + i * 0.001, -9.98, None, 1000.0 + i) for i in range(10)]
        gps = GPSData(trackpoints=pts)
        scorer = ScenicScorer(gps)
        score = scorer.score_segment(1000.0, 1009.0)
        assert 0.0 <= score <= 100.0
