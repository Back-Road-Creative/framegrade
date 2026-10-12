"""Tests for the per-day STOP mask.

A published short shipped extended parked/roadside footage. ``build_stop_mask``
produces a per-second boolean veto so downstream planners can STRUCTURALLY
exclude stopped seconds instead of catching them at a late gate.

Scores are 1 Hz; a GPS point is ``(t_master, lat, lon)`` in the same 0-based
day-master seconds as the score rows. Speed is derived from coordinate deltas
(GPS carries no speed channel), so parked spans are built by repeating a
coordinate and moving spans by advancing latitude by a per-second delta.
"""

import numpy as np

from gomoveshift.pipeline.stop_mask import build_stop_mask, mask_summary

COL_MOTION_ENERGY = 2
NUM_COLUMNS = 11

# 1 degree of latitude ≈ 111_320 m. Convert a target ground speed (km/h) into
# the per-second latitude step that produces it.
_M_PER_DEG_LAT = 111_320.0


def _make_scores(n: int, motion=0.5) -> np.ndarray:
    arr = np.zeros((n, NUM_COLUMNS), dtype=np.float64)
    arr[:, 0] = np.arange(n, dtype=np.float64)
    if np.isscalar(motion):
        arr[:, COL_MOTION_ENERGY] = float(motion)
    else:
        arr[:, COL_MOTION_ENERGY] = np.asarray(motion, dtype=np.float64)
    return arr


def _moving_gps(n: int, kmh: float, base_lat: float = 40.0, base_lon: float = -80.0):
    """A GPS trace advancing north at a constant ground speed."""
    step_deg = (kmh / 3.6) / _M_PER_DEG_LAT  # m/s → deg lat per 1 s
    return [(float(t), base_lat + t * step_deg, base_lon) for t in range(n)]


def _parked_gps(n: int, base_lat: float = 40.0, base_lon: float = -80.0):
    """A GPS trace pinned to one coordinate (ground speed 0)."""
    return [(float(t), base_lat, base_lon) for t in range(n)]


class TestGpsAuthoritative:
    def test_parked_day_is_fully_masked(self):
        n = 20
        mask = build_stop_mask(_make_scores(n, motion=0.5), _parked_gps(n))
        assert mask.shape == (n,)
        assert mask.all()

    def test_steady_40kmh_day_is_not_masked(self):
        n = 20
        mask = build_stop_mask(_make_scores(n, motion=0.0), _moving_gps(n, 40.0))
        assert not mask.any()

    def test_sub_10kmh_creep_masked_even_with_high_optical_motion(self):
        # 5 km/h creep (below the 10 km/h stop threshold) with HIGH optical
        # motion — GPS is authoritative and additive, so it masks anyway.
        n = 20
        mask = build_stop_mask(_make_scores(n, motion=0.9), _moving_gps(n, 5.0))
        # Interior seconds (clear of the centered-window edges) must be masked.
        assert mask[5:15].all()


class TestOpticalFallback:
    def test_no_gps_window_falls_back_to_optical_low_motion_masked(self):
        # GPS covers only the first 2 seconds (have_gps True) so every later
        # second's centered window has <2 fixes → speed None → optical fallback.
        n = 20
        motion = np.full(n, 0.9)
        motion[5:20] = 0.001  # low → stopped by optical
        mask = build_stop_mask(_make_scores(n, motion=motion), _moving_gps(2, 40.0))
        assert mask[7:18].all()

    def test_no_gps_window_fallback_high_motion_not_masked(self):
        # Same sparse GPS, but the fallback region has HIGH optical motion —
        # an unmeasurable second must NEVER auto-pass to "stopped".
        n = 20
        motion = np.full(n, 0.9)
        mask = build_stop_mask(_make_scores(n, motion=motion), _moving_gps(2, 40.0))
        assert not mask[5:20].any()

    def test_gps_less_day_masks_purely_on_optical(self):
        n = 20
        motion = np.full(n, 0.9)
        motion[6:16] = 0.001
        mask = build_stop_mask(_make_scores(n, motion=motion), None)
        assert mask[6:16].all()
        assert not mask[0:6].any()
        assert not mask[16:20].any()


class TestSustainedRuns:
    def test_two_second_slowdown_is_not_masked(self):
        # Two low-motion seconds inside an otherwise moving GPS-less day — below
        # the 5 s minimum, so nothing masks.
        n = 20
        motion = np.full(n, 0.9)
        motion[9:11] = 0.001
        mask = build_stop_mask(_make_scores(n, motion=motion), None)
        assert not mask.any()

    def test_exactly_min_duration_run_is_masked(self):
        n = 20
        motion = np.full(n, 0.9)
        motion[8:13] = 0.001  # exactly 5 seconds
        mask = build_stop_mask(_make_scores(n, motion=motion), None, min_stop_duration_s=5.0)
        assert mask[8:13].all()
        assert not mask[0:8].any()
        assert not mask[13:20].any()


class TestShapeAndSummary:
    def test_mask_length_matches_scores(self):
        for n in (0, 1, 7, 50):
            mask = build_stop_mask(_make_scores(n), _parked_gps(n))
            assert mask.shape == (n,)
            assert mask.dtype == bool

    def test_empty_scores_returns_empty_mask(self):
        mask = build_stop_mask(_make_scores(0), None)
        assert mask.shape == (0,)

    def test_mask_summary_reports_fraction(self):
        n = 20
        mask = build_stop_mask(_make_scores(n, motion=0.5), _parked_gps(n))
        s = mask_summary(mask)
        assert s["total_seconds"] == n
        assert s["masked_seconds"] == n
        assert s["masked_fraction"] == 1.0
