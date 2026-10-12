"""Tests for the master-encode wall-clock estimator in ``preflight_plan``.

Calibration source: a measured real-world run.
  4K HEVC source @ 30fps, hevc_nvenc preset p7, multipass=2,
  source duration 9815s → ffmpeg's own progress reports 0.7x realtime
  (21fps output for 30fps source) → ETA ~14000s (~3h 53m).

The previous magic constant of ``1.8 * 3600`` per day predicted ~1h 48m
for the same case — ~2.2x too optimistic and a real operator-scheduling
hazard. This test pins the new model to the empirical point and to a
fast-encode lower bound (1080p / p4 / multipass=1) so we don't
over-correct cheaper jobs.
"""

from __future__ import annotations

import logging

from gomoveshift.pipeline import preflight_plan


class TestEstimateMasterEncodeSeconds:
    def test_4k_p7_multipass2_matches_empirical(self):
        """Reference day: 9815s 4K HEVC → ~14000s (~3h 53m)."""
        est = preflight_plan.estimate_master_encode_seconds(
            source_duration_s=9815.0,
            resolution="4K",
            preset="p7",
            multipass=2,
        )
        # Empirical 14021s; require within the [13800, 15000] window.
        assert 13800 <= est <= 15000, (
            f"4K/p7/multipass=2 estimate {est:.0f}s outside [13800, 15000]; "
            f"empirical reference ~14021s"
        )

    def test_1080p_p4_multipass1_stays_fast(self):
        """A cheap 1080p/p4/single-pass job must NOT inherit the 4K cost."""
        est = preflight_plan.estimate_master_encode_seconds(
            source_duration_s=600.0,
            resolution="1080p",
            preset="p4",
            multipass=1,
        )
        assert est <= 600.0, (
            f"1080p/p4/multipass=1 estimate {est:.0f}s exceeded 1x realtime "
            f"(source 600s) — model has over-corrected"
        )
        assert est > 0

    def test_invalid_preset_falls_back_with_warning(self, caplog):
        """Unknown preset names fall back to the documented default
        (p4) with a WARNING log — must not raise."""
        with caplog.at_level(logging.WARNING, logger=preflight_plan.logger.name):
            est = preflight_plan.estimate_master_encode_seconds(
                source_duration_s=600.0,
                resolution="4K",
                preset="ultrafast",  # not an NVENC preset
                multipass=1,
            )
        assert est > 0
        assert any(
            "preset" in rec.message.lower() and "ultrafast" in rec.message for rec in caplog.records
        ), f"expected WARNING mentioning the bad preset; got {[r.message for r in caplog.records]}"

    def test_invalid_resolution_falls_back_with_warning(self, caplog):
        """Unknown resolution falls back to 4K (the GMS default) with a
        WARNING — defensive: we never want a KeyError to crash preflight."""
        with caplog.at_level(logging.WARNING, logger=preflight_plan.logger.name):
            est = preflight_plan.estimate_master_encode_seconds(
                source_duration_s=600.0,
                resolution="720p",
                preset="p7",
                multipass=2,
            )
        assert est > 0
        assert any("resolution" in rec.message.lower() for rec in caplog.records)

    def test_multipass_costs_more_than_single_pass(self):
        """All else equal, multipass=2 must take longer than multipass=1."""
        single = preflight_plan.estimate_master_encode_seconds(
            source_duration_s=3600.0, resolution="4K", preset="p7", multipass=1
        )
        multi = preflight_plan.estimate_master_encode_seconds(
            source_duration_s=3600.0, resolution="4K", preset="p7", multipass=2
        )
        assert multi > single, f"multipass=2 ({multi}) should exceed multipass=1 ({single})"

    def test_higher_preset_costs_more(self):
        """p7 (slowest) must take longer than p1 (fastest)."""
        cheap = preflight_plan.estimate_master_encode_seconds(
            source_duration_s=3600.0, resolution="4K", preset="p1", multipass=1
        )
        slow = preflight_plan.estimate_master_encode_seconds(
            source_duration_s=3600.0, resolution="4K", preset="p7", multipass=1
        )
        assert slow > cheap

    def test_4k_costs_more_than_1080p(self):
        a = preflight_plan.estimate_master_encode_seconds(
            source_duration_s=3600.0, resolution="1080p", preset="p7", multipass=2
        )
        b = preflight_plan.estimate_master_encode_seconds(
            source_duration_s=3600.0, resolution="4K", preset="p7", multipass=2
        )
        assert b > a


class TestPreflightPlanIntegration:
    """The day-master line in the wall-clock breakdown must reflect the
    new calibrated estimate, not the old 1.8h constant."""

    def test_build_plan_master_seconds_uses_new_estimator(self, tmp_path):
        """Force the per-day source duration to the reference day
        and assert the breakdown's master line lands in the calibrated
        window (per-day basis: 13800..15000)."""
        from dataclasses import dataclass

        from gomoveshift.pipeline.drive_state_model import PipelineState

        @dataclass
        class _Ep:
            index: int
            concat_path: object
            total_duration: float
            date: str
            creation_time: None = None
            gpx_path: None = None

        class _Plan:
            def __init__(self, eps):
                self.episodes = list(eps)

        output_base = tmp_path / "out"
        output_base.mkdir()
        stages_dir = tmp_path / "_stages"
        stages_dir.mkdir()
        ep_path = tmp_path / "ep_000.mp4"
        ep_path.write_bytes(b"x")
        eps = [_Ep(index=0, concat_path=ep_path, total_duration=9815.0, date="2026-02-22")]
        state = PipelineState(
            source_dir=tmp_path,
            output_base=output_base,
            region="usa",
            publish_day="wed",
            location="South Carolina",
            fresh=False,
            skip_upload=False,
            skip_csv=False,
            stages_base=stages_dir,
        )
        state["ingest_plan"] = _Plan(eps)
        state["episode_files"] = [ep_path]
        state["episode_dates"] = ["2026-02-22"]

        plan = preflight_plan.build_plan(state, stages_dir)
        master = plan.wall_clock_breakdown.get("master", 0)
        assert 13800 <= master <= 15000, (
            f"plan breakdown master={master}s outside the calibrated "
            f"[13800, 15000] window for 4K/p7/multipass=2/9815s"
        )
