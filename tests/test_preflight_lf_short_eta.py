"""Tests for the LF + short encode wall-clock estimators in
``preflight_plan`` (mirrors the master-encode calibration).

Calibration sources:
  * LF: a measured full-day run — lf_06
    9.6 min source → ~10:05 wall-clock (~1.05x realtime); lf_07
    18.1 min source → ~20:27 wall-clock (~1.13x realtime). Pick 1.10.
  * Short: a measured 16:9 trim+OST path produced
    11 shorts in 430s wall-clock for ~30s slices (~1.3x realtime).
    The 9:16 path adds crop+scale+9:16-OST overhead;
    conservatively model as 2.0x realtime with a 30s startup floor.

Each test asserts the calibrated estimate stays within ±25% of the
empirical anchor and that the legacy fallback is preserved when no
per-LF / per-short data is available.
"""

from __future__ import annotations

import logging

from gomoveshift.pipeline import preflight_plan

from _state_helpers import make_state


class TestEstimateLfEncodeSeconds:
    def test_lf_07_matches_empirical_within_25pct(self):
        """lf_07: 18.1 min source → ~20:27 wall-clock."""
        source_s = 18.1 * 60
        empirical_s = 20 * 60 + 27  # ~1227s
        est = preflight_plan.estimate_lf_encode_seconds(source_s)
        lo, hi = empirical_s * 0.75, empirical_s * 1.25
        assert lo <= est <= hi, (
            f"lf_07 estimate {est:.0f}s outside [{lo:.0f}, {hi:.0f}] (empirical ~{empirical_s}s)"
        )

    def test_lf_06_matches_empirical_within_25pct(self):
        """lf_06: 9.6 min source → ~10:05 wall-clock."""
        source_s = 9.6 * 60
        empirical_s = 10 * 60 + 5  # ~605s
        est = preflight_plan.estimate_lf_encode_seconds(source_s)
        lo, hi = empirical_s * 0.75, empirical_s * 1.25
        assert lo <= est <= hi, (
            f"lf_06 estimate {est:.0f}s outside [{lo:.0f}, {hi:.0f}] (empirical ~{empirical_s}s)"
        )

    def test_zero_or_negative_falls_back_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger=preflight_plan.logger.name):
            est = preflight_plan.estimate_lf_encode_seconds(0)
        assert est > 0, "zero source must NOT produce zero ETA (fallback)"
        assert any("invalid duration" in rec.message for rec in caplog.records)
        # Negative
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=preflight_plan.logger.name):
            est_neg = preflight_plan.estimate_lf_encode_seconds(-5.0)
        assert est_neg > 0
        assert any("invalid duration" in rec.message for rec in caplog.records)

    def test_non_numeric_falls_back_without_raising(self, caplog):
        with caplog.at_level(logging.WARNING, logger=preflight_plan.logger.name):
            est = preflight_plan.estimate_lf_encode_seconds("garbage")
        assert est > 0
        assert any("invalid duration" in rec.message for rec in caplog.records)

    def test_floor_keeps_eta_nonzero_for_tiny_inputs(self):
        """A 1-second LF must still account for ffmpeg startup tax."""
        est = preflight_plan.estimate_lf_encode_seconds(1.0)
        assert est >= preflight_plan._LF_FLOOR_SEC


class TestEstimateShortEncodeSeconds:
    def test_30s_short_matches_empirical_within_25pct(self):
        """16:9 path: ~30s slice → ~39s wall-clock.
        New 9:16 path modeled at 2.0x realtime → ~60s. Both lie
        within ±25% of an empirical-anchor band [39, 60]; assert the
        estimate falls in [29, 75]."""
        est = preflight_plan.estimate_short_encode_seconds(30.0)
        # Anchor: empirical 39s for 16:9, conservative 60s for 9:16.
        # ±25% of the 9:16 model anchor (60s) → [45, 75]; the floor
        # might pull below, so allow lower bound at the floor (30s).
        assert preflight_plan._SHORT_FLOOR_SEC <= est <= 75, (
            f"30s short estimate {est:.0f}s outside [floor, 75]; "
            f"empirical 16:9 ~39s, 9:16 conservative ~60s"
        )

    def test_60s_short_costs_more_than_30s_short(self):
        a = preflight_plan.estimate_short_encode_seconds(30.0)
        b = preflight_plan.estimate_short_encode_seconds(60.0)
        assert b > a, f"60s short ({b}) must exceed 30s short ({a})"

    def test_zero_or_negative_falls_back_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger=preflight_plan.logger.name):
            est = preflight_plan.estimate_short_encode_seconds(0)
        assert est > 0
        assert any("invalid duration" in rec.message for rec in caplog.records)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=preflight_plan.logger.name):
            est_neg = preflight_plan.estimate_short_encode_seconds(-1.0)
        assert est_neg > 0
        assert any("invalid duration" in rec.message for rec in caplog.records)

    def test_non_numeric_falls_back_without_raising(self, caplog):
        with caplog.at_level(logging.WARNING, logger=preflight_plan.logger.name):
            est = preflight_plan.estimate_short_encode_seconds(None)
        assert est > 0
        assert any("invalid duration" in rec.message for rec in caplog.records)

    def test_floor_keeps_eta_nonzero(self):
        est = preflight_plan.estimate_short_encode_seconds(1.0)
        assert est >= preflight_plan._SHORT_FLOOR_SEC


class TestWallClockBreakdownIntegration:
    """The ``lfs`` and ``shorts`` lines in the wall-clock breakdown must
    use the calibrated estimators when per-LF data is in state, and must
    fall back to the legacy flat constants when no per-LF data exists."""

    def _build_rows(self, days: list[str]):
        from gomoveshift.pipeline.preflight_plan import PlanRow

        return [PlanRow(label=f"Day {d}", status="RUN", reason="fresh") for d in days]

    def test_calibrated_path_used_when_drive_plan_present(self):
        """With a drive_plan carrying per-LF durations, the lfs breakdown
        line must reflect the sum of the calibrated estimator (NOT the
        legacy ``days * 3 * 5min`` formula)."""

        class _Lf:
            def __init__(self, dur):
                self.duration_s = dur

        class _Plan:
            def __init__(self, lfs):
                self.regular_lfs = lfs

        # Two LFs at 18.1min + 9.6min — the calibrated sum is ~1827s.
        # Legacy fallback would be 1 day * 3 LFs * 5min = 900s.
        state = make_state(
            drive_plan=_Plan([_Lf(18.1 * 60), _Lf(9.6 * 60)]),
            shorts_per_lf={1: ["a", "b", "c"], 2: ["d", "e"]},
        )
        rows = self._build_rows(["2026-02-24"])
        total, breakdown = preflight_plan._estimate_wall_clock(rows, state)
        # Calibrated lfs ≈ (18.1*60)*1.10 + (9.6*60)*1.10 = ~1828s
        # ±25% window around 1828s → [1371, 2285]
        assert 1371 <= breakdown["lfs"] <= 2285, (
            f"lfs={breakdown['lfs']}s outside calibrated window for "
            f"2 LFs (18.1min + 9.6min); empirical sum ~1828s"
        )
        # Shorts: 5 shorts * 60s default = 300s at 2.0x realtime
        assert breakdown["shorts"] >= preflight_plan._SHORT_FLOOR_SEC * 5

    def test_legacy_fallback_when_no_per_lf_data(self):
        """When drive_plan is missing AND shorts_per_lf is missing, the
        breakdown must still be non-zero (legacy flat constants kick in
        scaled by ``days_to_run``). No regression vs the legacy behavior."""
        state = make_state()  # no drive_plan, no shorts_per_lf
        rows = self._build_rows(["2026-02-22", "2026-02-24"])
        total, breakdown = preflight_plan._estimate_wall_clock(rows, state)
        assert breakdown["lfs"] > 0
        assert breakdown["shorts"] > 0
        # Legacy: 2 days * 3 LFs * estimate_lf_encode_seconds(720s)
        expected_lfs = (
            2
            * preflight_plan._DEFAULT_LFS_PER_DAY
            * int(
                preflight_plan.estimate_lf_encode_seconds(preflight_plan._DEFAULT_LF_DURATION_SEC)
            )
        )
        assert breakdown["lfs"] == expected_lfs

    def test_eta_never_zero_for_runnable_day(self):
        """A RUN day with no per-LF state must still yield non-zero
        lfs + shorts breakdown lines (regression guard)."""
        state = make_state()
        rows = self._build_rows(["2026-02-22"])
        _, breakdown = preflight_plan._estimate_wall_clock(rows, state)
        assert breakdown["lfs"] > 0
        assert breakdown["shorts"] > 0
        assert breakdown["master"] > 0
