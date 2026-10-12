"""LongFormAssembler ``lf_selection_mode='full_drive'`` wiring.

Verifies the assembler produces exactly one LF per day spanning the day's
full processed timeline when ``config.drive.lf_selection_mode='full_drive'``,
bypasses the adaptive-threshold path entirely, and raises on under/over
duration sources.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.config import DriveConfig
from gomoveshift.pipeline.longform_assembler import LongFormAssembler


def _score_array(duration_sec: int, start_offset: float = 0.0) -> np.ndarray:
    """Dense per-second score array (N, 11) with high composite + motion."""
    n = int(duration_sec)
    arr = np.zeros((n, 11), dtype=np.float64)
    arr[:, 0] = np.arange(n, dtype=np.float64) + start_offset
    # composite — modulated so the p50 peak-pool selection has signal
    arr[:, 1] = 0.5 + 0.3 * np.sin(np.linspace(0, 6.28, n))
    arr[:, 2] = 0.6  # motion energy
    return arr


def _full_drive_cfg(min_lf=480.0, full_drive_max=7200) -> DriveConfig:
    return DriveConfig(
        lf_selection_mode="full_drive",
        full_drive_max_duration_sec=full_drive_max,
        min_lf_duration=min_lf,
        min_segment_duration=30.0,
        min_motion_energy=0.0,
    )


def _keep_all_mask(duration_sec: int) -> np.ndarray:
    """A stop_mask marking nothing as parked — the explicit equivalent of
    the old (now-disallowed) maskless full_drive behavior."""
    return np.zeros(int(duration_sec), dtype=bool)


class TestFullDriveRequiresStopMask:
    """Maskless full_drive planning must fail loud, not ship a
    stops-included video. Covers both the day-first (``build_drive_plan_
    per_day``) and legacy (``build_drive_plan``) entry points."""

    def test_build_drive_plan_per_day_without_mask_raises(self):
        cfg = _full_drive_cfg()
        assembler = LongFormAssembler(cfg)
        scores = _score_array(1800)
        with pytest.raises(ValueError, match="stop_mask"):
            assembler.build_drive_plan_per_day(
                scores=scores,
                processed_files=[Path("/tmp/ep_00.mp4")],
                file_boundaries=[(0.0, 1800.0)],
                episode_dates=["2025-09-14"],
            )

    def test_build_drive_plan_without_mask_raises(self):
        cfg = _full_drive_cfg()
        assembler = LongFormAssembler(cfg)
        scores = _score_array(1800)
        with pytest.raises(ValueError, match="stop_mask"):
            assembler.build_drive_plan(
                scores=scores,
                processed_files=[Path("/tmp/ep_00.mp4")],
                file_boundaries=[(0.0, 1800.0)],
            )


class TestFullDrivePerDay:
    def test_single_day_single_file_yields_one_lf_spanning_timeline(self):
        cfg = _full_drive_cfg()
        assembler = LongFormAssembler(cfg)
        scores = _score_array(1800)  # 30 min
        plan = assembler.build_drive_plan_per_day(
            scores=scores,
            processed_files=[Path("/tmp/ep_00.mp4")],
            file_boundaries=[(0.0, 1800.0)],
            episode_dates=["2025-09-14"],
            stop_mask=_keep_all_mask(1800),
        )
        assert len(plan.regular_lfs) == 1
        lf = plan.regular_lfs[0]
        assert lf.index == 1
        # Spans the entire file (minus 0.5s EOF guard) — one segment per file
        assert len(lf.segments) == 1
        assert lf.total_duration == pytest.approx(1799.5, abs=1.0)
        # trim_ratio = lf.total_duration / file_duration → ≥ 0.5 → "Driving" verb
        assert lf.total_duration / 1800.0 >= 0.5
        # Montage peaks are real sub-windows drawn from the peak_pool, not the giant segment
        assert lf.montage_peaks, "full_drive LF must have montage peaks"
        for peak in lf.montage_peaks:
            assert peak.duration < lf.total_duration

    def test_multi_day_yields_one_lf_per_day(self):
        cfg = _full_drive_cfg()
        assembler = LongFormAssembler(cfg)
        # Two days back-to-back, 20 min each
        day1 = _score_array(1200, start_offset=0.0)
        day2 = _score_array(1200, start_offset=1200.0)
        scores = np.vstack([day1, day2])
        plan = assembler.build_drive_plan_per_day(
            scores=scores,
            processed_files=[Path("/tmp/ep_00.mp4"), Path("/tmp/ep_01.mp4")],
            file_boundaries=[(0.0, 1200.0), (1200.0, 2400.0)],
            episode_dates=["2025-09-14", "2025-09-15"],
            stop_mask=_keep_all_mask(2400),
        )
        assert len(plan.regular_lfs) == 2
        assert [lf.index for lf in plan.regular_lfs] == [1, 2]
        for lf in plan.regular_lfs:
            assert lf.total_duration == pytest.approx(1199.5, abs=1.0)

    def test_skip_file_indices_excluded(self):
        cfg = _full_drive_cfg()
        assembler = LongFormAssembler(cfg)
        day1 = _score_array(1200, start_offset=0.0)
        day2 = _score_array(1200, start_offset=1200.0)
        scores = np.vstack([day1, day2])
        plan = assembler.build_drive_plan_per_day(
            scores=scores,
            processed_files=[Path("/tmp/ep_00.mp4"), Path("/tmp/ep_01.mp4")],
            file_boundaries=[(0.0, 1200.0), (1200.0, 2400.0)],
            episode_dates=["2025-09-14", "2025-09-15"],
            skip_file_indices={0},
            stop_mask=_keep_all_mask(2400),
        )
        # Day 1's only file is skipped → just day 2's LF
        assert len(plan.regular_lfs) == 1
        assert plan.regular_lfs[0].segments[0].source_file_index == 1

    def test_under_min_duration_raises(self):
        # 5 minutes < 8-minute floor → planner raises
        cfg = _full_drive_cfg(min_lf=480.0)
        assembler = LongFormAssembler(cfg)
        scores = _score_array(300)
        with pytest.raises(ValueError, match="shorter than"):
            assembler.build_drive_plan_per_day(
                scores=scores,
                processed_files=[Path("/tmp/ep_00.mp4")],
                file_boundaries=[(0.0, 300.0)],
                episode_dates=["2025-09-14"],
                stop_mask=_keep_all_mask(300),
            )

    def test_over_max_duration_raises(self):
        # 90 min > 60 min ceiling → planner raises (no silent truncation)
        cfg = _full_drive_cfg(min_lf=480.0, full_drive_max=3600)
        assembler = LongFormAssembler(cfg)
        scores = _score_array(5400)
        with pytest.raises(ValueError, match="exceeds the"):
            assembler.build_drive_plan_per_day(
                scores=scores,
                processed_files=[Path("/tmp/ep_00.mp4")],
                file_boundaries=[(0.0, 5400.0)],
                episode_dates=["2025-09-14"],
                stop_mask=_keep_all_mask(5400),
            )

    def test_no_adaptive_threshold_log_lines(self, caplog):
        cfg = _full_drive_cfg()
        assembler = LongFormAssembler(cfg)
        scores = _score_array(1800)
        with caplog.at_level(logging.INFO, logger="gomoveshift.pipeline.longform_assembler"):
            assembler.build_drive_plan_per_day(
                scores=scores,
                processed_files=[Path("/tmp/ep_00.mp4")],
                file_boundaries=[(0.0, 1800.0)],
                episode_dates=["2025-09-14"],
                stop_mask=_keep_all_mask(1800),
            )
        msgs = [r.getMessage() for r in caplog.records]
        assert not any("[adaptive_threshold]" in m for m in msgs), (
            f"full_drive path must NOT call adaptive threshold search: {msgs}"
        )
        assert any("[full_drive]" in m for m in msgs), msgs


class TestFullDriveLegacyPath:
    """Defense-in-depth: build_drive_plan (non-day-first) also branches."""

    def test_legacy_path_yields_single_full_drive_lf(self):
        cfg = _full_drive_cfg()
        assembler = LongFormAssembler(cfg)
        scores = _score_array(1800)
        plan = assembler.build_drive_plan(
            scores=scores,
            processed_files=[Path("/tmp/ep_00.mp4")],
            file_boundaries=[(0.0, 1800.0)],
            stop_mask=_keep_all_mask(1800),
        )
        assert len(plan.regular_lfs) == 1
        lf = plan.regular_lfs[0]
        assert lf.total_duration == pytest.approx(1799.5, abs=1.0)
        assert lf.montage_peaks

    def test_legacy_path_under_min_raises(self):
        cfg = _full_drive_cfg(min_lf=480.0)
        assembler = LongFormAssembler(cfg)
        scores = _score_array(300)
        with pytest.raises(ValueError, match="shorter than"):
            assembler.build_drive_plan(
                scores=scores,
                processed_files=[Path("/tmp/ep_00.mp4")],
                file_boundaries=[(0.0, 300.0)],
                stop_mask=_keep_all_mask(300),
            )


class TestBestMomentsUnchanged:
    """Regression: lf_selection_mode='best_moments' (default) is untouched."""

    def test_best_moments_still_runs_adaptive_threshold(self, caplog):
        cfg = DriveConfig(
            lf_selection_mode="best_moments",
            adaptive_threshold_enabled=True,
            min_lf_duration=480.0,
            min_segment_duration=30.0,
            min_motion_energy=0.0,
        )
        assembler = LongFormAssembler(cfg)
        scores = _score_array(1800)
        with caplog.at_level(logging.INFO, logger="gomoveshift.pipeline.longform_assembler"):
            assembler.build_drive_plan_per_day(
                scores=scores,
                processed_files=[Path("/tmp/ep_00.mp4")],
                file_boundaries=[(0.0, 1800.0)],
                episode_dates=["2025-09-14"],
            )
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[adaptive_threshold]" in m for m in msgs), (
            f"best_moments path must still use adaptive threshold: {msgs}"
        )
        assert not any("[full_drive]" in m for m in msgs)
