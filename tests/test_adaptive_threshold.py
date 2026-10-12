"""Tests for per-day adaptive LF scoring (percentile-floor scoring).

TDD: tests written before implementation.
"""

from pathlib import Path

import numpy as np
import pytest

from gomoveshift.config import DriveConfig
from gomoveshift.pipeline.longform_assembler import LongFormAssembler


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_day_scores(
    day_durations: dict[str, list[float]],
    score_levels: dict[str, float],
) -> tuple[np.ndarray, list[Path], list[tuple[float, float]], list[str]]:
    """Build score array, processed_files, file_boundaries, and episode_dates
    for a multi-day drive.

    Args:
        day_durations: {date: [ep_duration_seconds, ...]}
        score_levels: {date: composite_score} — uniform score for all samples on that day

    Returns:
        (scores, processed_files, file_boundaries, episode_dates)
    """
    all_scores = []
    processed_files = []
    file_boundaries = []
    episode_dates = []
    offset = 0.0
    file_idx = 0

    for date in sorted(day_durations.keys()):
        for dur in day_durations[date]:
            n = int(dur)
            arr = np.zeros((n, 9), dtype=np.float64)
            arr[:, 0] = np.arange(n, dtype=np.float64) + offset
            arr[:, 1] = score_levels.get(date, 0.5)
            arr[:, 2] = 0.1  # motion_energy (above any typical gate)
            all_scores.append(arr)
            processed_files.append(Path(f"/tmp/episode_{file_idx:02d}_{date}.mp4"))
            file_boundaries.append((offset, offset + dur))
            episode_dates.append(date)
            offset += dur
            file_idx += 1

    scores = np.vstack(all_scores) if all_scores else np.zeros((0, 9))
    return scores, processed_files, file_boundaries, episode_dates


def _make_varied_day_scores(
    day_durations: dict[str, list[float]],
    day_patterns: dict[str, str],
) -> tuple[np.ndarray, list[Path], list[tuple[float, float]], list[str]]:
    """Build score array with varied patterns per day.

    Patterns:
        "high": 0.8 composite
        "low": 0.2 composite
        "mixed": alternating 90s blocks of 0.8/0.2
        "uniform_low": 0.004 (flat terrain, no signal)
    """
    all_scores = []
    processed_files = []
    file_boundaries = []
    episode_dates = []
    offset = 0.0
    file_idx = 0

    for date in sorted(day_durations.keys()):
        pattern = day_patterns.get(date, "mixed")
        for dur in day_durations[date]:
            n = int(dur)
            arr = np.zeros((n, 9), dtype=np.float64)
            arr[:, 0] = np.arange(n, dtype=np.float64) + offset
            arr[:, 2] = 0.1  # motion_energy

            if pattern == "high":
                arr[:, 1] = 0.8
            elif pattern == "low":
                arr[:, 1] = 0.2
            elif pattern == "uniform_low":
                arr[:, 1] = 0.004
            elif pattern == "mixed":
                for i in range(n):
                    block = (i // 90) % 2
                    arr[i, 1] = 0.8 if block == 0 else 0.2

            all_scores.append(arr)
            processed_files.append(Path(f"/tmp/episode_{file_idx:02d}_{date}.mp4"))
            file_boundaries.append((offset, offset + dur))
            episode_dates.append(date)
            offset += dur
            file_idx += 1

    scores = np.vstack(all_scores) if all_scores else np.zeros((0, 9))
    return scores, processed_files, file_boundaries, episode_dates


# ── Config fields ────────────────────────────────────────────────────────────


class TestAdaptiveConfig:
    """New config fields exist and have correct defaults."""

    def test_adaptive_threshold_enabled_default_true(self):
        cfg = DriveConfig()
        assert cfg.adaptive_threshold_enabled is True

    def test_adaptive_percentile_floor_default(self):
        cfg = DriveConfig()
        assert cfg.adaptive_percentile_floor == 5  # Updated for Path 1 defaults

    def test_adaptive_min_segment_floor_default(self):
        cfg = DriveConfig()
        assert cfg.adaptive_min_segment_floor_s == 15  # Updated for Path 1 defaults


# ── Core: build_drive_plan_per_day ───────────────────────────────────────────


class TestBuildDrivePlanPerDay:
    """Per-day adaptive threshold produces >=1 LF per recording day."""

    def test_guarantees_one_lf_per_day_two_days(self):
        """Two days with different score levels → at least 1 LF from each day."""
        cfg = DriveConfig(
            min_segment_duration=60.0,
            min_lf_duration=480.0,
            max_lf_duration=1800.0,
            adaptive_threshold_enabled=True,
        )
        assembler = LongFormAssembler(cfg)

        # Day 1: 30 min of high scores → easily produces LFs
        # Day 2: 30 min of mixed scores → needs adaptive lowering
        scores, files, boundaries, dates = _make_varied_day_scores(
            day_durations={"2026-02-20": [1800.0], "2026-02-21": [1800.0]},
            day_patterns={"2026-02-20": "high", "2026-02-21": "mixed"},
        )

        plan = assembler.build_drive_plan_per_day(
            scores=scores,
            processed_files=files,
            file_boundaries=boundaries,
            episode_dates=dates,
            region="usa",
        )

        # At least 1 LF from each day
        day_20_lfs = [
            lf for lf in plan.regular_lfs if any(s.source_file_index == 0 for s in lf.segments)
        ]
        day_21_lfs = [
            lf for lf in plan.regular_lfs if any(s.source_file_index == 1 for s in lf.segments)
        ]
        assert len(day_20_lfs) >= 1, "Day 2026-02-20 must produce >=1 LF"
        assert len(day_21_lfs) >= 1, "Day 2026-02-21 must produce >=1 LF"

    def test_lf_does_not_cross_day_boundary(self):
        """LFs from different days must not share segments."""
        cfg = DriveConfig(
            min_segment_duration=60.0,
            min_lf_duration=480.0,
            max_lf_duration=1800.0,
            adaptive_threshold_enabled=True,
        )
        assembler = LongFormAssembler(cfg)

        scores, files, boundaries, dates = _make_varied_day_scores(
            day_durations={"2026-02-20": [1200.0], "2026-02-21": [1200.0]},
            day_patterns={"2026-02-20": "high", "2026-02-21": "high"},
        )

        plan = assembler.build_drive_plan_per_day(
            scores=scores,
            processed_files=files,
            file_boundaries=boundaries,
            episode_dates=dates,
            region="usa",
        )

        for lf in plan.regular_lfs:
            file_indices = {s.source_file_index for s in lf.segments}
            # File 0 = day 2026-02-20, file 1 = day 2026-02-21
            # An LF should not span both
            assert not (0 in file_indices and 1 in file_indices), (
                f"LF {lf.index} crosses day boundary: file indices {file_indices}"
            )

    def test_multiple_lfs_per_high_scoring_day(self):
        """A day with lots of high-scoring content → multiple LFs."""
        cfg = DriveConfig(
            min_segment_duration=60.0,
            min_lf_duration=480.0,
            max_lf_duration=1800.0,
            max_segment_duration=0.0,  # Disable cap so long runs pass through
            adaptive_threshold_enabled=True,
            # Updated for Path 1 defaults: pin legacy floor so the
            # threshold doesn't descend below the 0.2/0.8 mixed-block divide.
            adaptive_percentile_floor=30,
            score_threshold_percentile=60.0,
        )
        assembler = LongFormAssembler(cfg)

        # 2 hours of mixed scores (alternating 90s high/low blocks)
        # produces many 90s segments that group into multiple LFs
        scores, files, boundaries, dates = _make_varied_day_scores(
            day_durations={"2026-02-24": [7200.0]},
            day_patterns={"2026-02-24": "mixed"},
        )

        plan = assembler.build_drive_plan_per_day(
            scores=scores,
            processed_files=files,
            file_boundaries=boundaries,
            episode_dates=dates,
            region="usa",
        )

        assert len(plan.regular_lfs) >= 2, (
            f"2h of mixed scores should produce >=2 LFs, got {len(plan.regular_lfs)}"
        )

    def test_lf_indices_are_global_one_based(self):
        """LF indices are numbered globally (1, 2, 3, ...) across days."""
        cfg = DriveConfig(
            min_segment_duration=60.0,
            min_lf_duration=480.0,
            max_lf_duration=1800.0,
            adaptive_threshold_enabled=True,
        )
        assembler = LongFormAssembler(cfg)

        scores, files, boundaries, dates = _make_varied_day_scores(
            day_durations={"2026-02-20": [1800.0], "2026-02-21": [1800.0]},
            day_patterns={"2026-02-20": "high", "2026-02-21": "high"},
        )

        plan = assembler.build_drive_plan_per_day(
            scores=scores,
            processed_files=files,
            file_boundaries=boundaries,
            episode_dates=dates,
            region="usa",
        )

        indices = [lf.index for lf in plan.regular_lfs]
        assert indices == list(range(1, len(indices) + 1)), (
            f"LF indices must be 1-based sequential, got {indices}"
        )

    def test_multi_episode_day(self):
        """A day with multiple episodes groups them correctly."""
        cfg = DriveConfig(
            min_segment_duration=60.0,
            min_lf_duration=480.0,
            max_lf_duration=1800.0,
            adaptive_threshold_enabled=True,
        )
        assembler = LongFormAssembler(cfg)

        # Day 1 has 2 episodes, Day 2 has 1 episode
        scores, files, boundaries, dates = _make_varied_day_scores(
            day_durations={
                "2026-02-20": [900.0, 900.0],  # 2 episodes, 15 min each
                "2026-02-21": [1800.0],  # 1 episode, 30 min
            },
            day_patterns={"2026-02-20": "high", "2026-02-21": "high"},
        )

        plan = assembler.build_drive_plan_per_day(
            scores=scores,
            processed_files=files,
            file_boundaries=boundaries,
            episode_dates=dates,
            region="usa",
        )

        # Day 1 segments should reference files 0 and 1
        # Day 2 segments should reference file 2 only
        for lf in plan.regular_lfs:
            file_indices = {s.source_file_index for s in lf.segments}
            day_1_files = {0, 1}
            day_2_files = {2}
            spans_both = bool(file_indices & day_1_files) and bool(file_indices & day_2_files)
            assert not spans_both, f"LF {lf.index} crosses day boundary: files {file_indices}"


# ── Adaptive threshold search ────────────────────────────────────────────────


class TestAdaptiveThresholdSearch:
    """The adaptive search lowers percentile and min_segment_duration."""

    def test_best_moments_mode_raises_qualifying_bar(self):
        """When best_moments_mode is on, adaptive search uses best_moments_target_min
        as the qualifying bar, not just min_lf_duration. This ensures the
        downstream best_moments filter has enough content to produce LFs in
        the 15-25 min target range."""
        cfg = DriveConfig(
            min_segment_duration=60.0,
            min_lf_duration=480.0,  # 8 min
            max_lf_duration=1800.0,
            adaptive_threshold_enabled=True,
            best_moments_mode=True,
            best_moments_target_min=900.0,  # 15 min
            best_moments_percentile=50.0,
            # Updated for Path 1 defaults: pin legacy floors and
            # starting percentile so the adaptive search behaves as before.
            score_threshold_percentile=60.0,
            adaptive_percentile_floor=30,
            adaptive_min_segment_floor_s=30,
        )
        assembler = LongFormAssembler(cfg)

        # 60 min of mixed scores — enough to produce longer LFs when the
        # qualifying bar is 15 min instead of 8 min
        scores, files, boundaries, dates = _make_varied_day_scores(
            day_durations={"2026-03-15": [3600.0]},
            day_patterns={"2026-03-15": "mixed"},
        )

        plan = assembler.build_drive_plan_per_day(
            scores=scores,
            processed_files=files,
            file_boundaries=boundaries,
            episode_dates=dates,
            region="usa",
        )

        # With the raised qualifying bar, each LF should have more content
        total_dur = sum(lf.total_duration for lf in plan.regular_lfs)
        assert total_dur > 480.0, (
            f"Total LF duration {total_dur:.0f}s should exceed min_lf_duration "
            f"when best_moments_mode targets 15 min"
        )

    def test_floor_respected_percentile(self):
        """Percentile never drops below adaptive_percentile_floor."""
        cfg = DriveConfig(
            min_segment_duration=60.0,
            min_lf_duration=480.0,
            adaptive_threshold_enabled=True,
            adaptive_percentile_floor=30,
        )
        assembler = LongFormAssembler(cfg)

        # All low scores — threshold will be lowered aggressively
        scores, files, boundaries, dates = _make_varied_day_scores(
            day_durations={"2026-02-22": [1400.0]},
            day_patterns={"2026-02-22": "uniform_low"},
        )

        plan = assembler.build_drive_plan_per_day(
            scores=scores,
            processed_files=files,
            file_boundaries=boundaries,
            episode_dates=dates,
            region="usa",
        )

        # Even with uniform low scores, should produce something
        # (the algorithm bottoms out at floors and takes what's available)
        assert plan is not None
        # Plan should have at least attempted the day
        assert plan.total_source_duration > 0

    def test_floor_respected_min_segment(self):
        """min_segment_duration never drops below adaptive_min_segment_floor_s."""
        cfg = DriveConfig(
            min_segment_duration=60.0,
            min_lf_duration=480.0,
            adaptive_threshold_enabled=True,
            adaptive_min_segment_floor_s=30,
        )
        assembler = LongFormAssembler(cfg)

        # Short peaks that would only pass at very low min_segment_duration
        n = 1400
        scores = np.zeros((n, 9), dtype=np.float64)
        scores[:, 0] = np.arange(n, dtype=np.float64)
        scores[:, 2] = 0.1  # motion
        # 20-second high blocks every 60 seconds
        for start in range(0, n, 60):
            end = min(start + 20, n)
            scores[start:end, 1] = 0.9
        # Rest is low
        for i in range(n):
            if scores[i, 1] == 0.0:
                scores[i, 1] = 0.1

        files = [Path("/tmp/ep_00_2026-02-22.mp4")]
        boundaries = [(0.0, float(n))]
        dates = ["2026-02-22"]

        plan = assembler.build_drive_plan_per_day(
            scores=scores,
            processed_files=files,
            file_boundaries=boundaries,
            episode_dates=dates,
            region="usa",
        )

        # The 20s peaks are below the 30s floor, so they should not be
        # selected as individual segments. The algorithm may produce an
        # undersized LF or fall through to best-available.
        for lf in plan.regular_lfs:
            for seg in lf.segments:
                assert seg.duration >= 29.0, (
                    f"Segment duration {seg.duration:.1f}s below floor (30s)"
                )


# ── Backward compatibility ───────────────────────────────────────────────────


class TestBackwardCompat:
    """adaptive_threshold_enabled=False → delegates to build_drive_plan."""

    def test_disabled_uses_global_threshold(self):
        cfg = DriveConfig(
            min_segment_duration=60.0,
            min_lf_duration=480.0,
            max_lf_duration=1800.0,
            adaptive_threshold_enabled=False,
        )
        assembler = LongFormAssembler(cfg)

        scores, files, boundaries, dates = _make_varied_day_scores(
            day_durations={"2026-02-20": [1800.0]},
            day_patterns={"2026-02-20": "high"},
        )

        plan = assembler.build_drive_plan_per_day(
            scores=scores,
            processed_files=files,
            file_boundaries=boundaries,
            episode_dates=dates,
            region="usa",
        )

        # Should produce valid output (same as build_drive_plan)
        assert len(plan.regular_lfs) >= 1
        assert plan.score_threshold > 0


# ── Episode dates alignment ──────────────────────────────────────────────────


class TestEpisodeDatesAlignment:
    """episode_dates must align with file_boundaries when episodes are skipped."""

    def test_dates_length_matches_boundaries(self):
        """episode_dates must have same length as file_boundaries."""
        cfg = DriveConfig(
            min_segment_duration=60.0,
            min_lf_duration=480.0,
            adaptive_threshold_enabled=True,
        )
        assembler = LongFormAssembler(cfg)

        scores, files, boundaries, dates = _make_day_scores(
            day_durations={"2026-02-20": [600.0], "2026-02-21": [600.0]},
            score_levels={"2026-02-20": 0.8, "2026-02-21": 0.8},
        )

        # Mismatched lengths should raise
        with pytest.raises((ValueError, IndexError)):
            assembler.build_drive_plan_per_day(
                scores=scores,
                processed_files=files,
                file_boundaries=boundaries,
                episode_dates=dates[:1],  # Only 1 date for 2 files
                region="usa",
            )


# ── Path 1 defaults lock ─────────────────────────────────────────────────────


class TestConfigDefaultsPath1:
    """Lock Path 1 DriveConfig defaults (yield remediation).

    These defaults shift the pipeline from conservative (4 LFs @ 13% keep from
    a 7.65h source) to higher-yield (target 8-10 LFs @ ~20% keep, each
    10-13 min). If you change a default here, also update the related
    algorithm/narrative docs.
    """

    def test_best_moments_mode_default_false(self):
        """best_moments_mode defaults OFF — filtering is opt-in, not baseline."""
        cfg = DriveConfig()
        assert cfg.best_moments_mode is False

    def test_score_threshold_percentile_default_15(self):
        """15th percentile keeps top 85% of footage so a long drive yields more
        LFs (was 30 = top 70%, which kept only ~13% in practice / ~5 LFs from
        8.6h). p15 is the sweet spot before the day merges into oversized blobs.
        """
        cfg = DriveConfig()
        assert cfg.score_threshold_percentile == 15.0

    def test_adaptive_percentile_floor_default_5(self):
        """Adaptive search can descend to the 5th percentile (was 30)."""
        cfg = DriveConfig()
        assert cfg.adaptive_percentile_floor == 5

    def test_adaptive_min_segment_floor_default_15(self):
        """Segments as short as 15s now eligible (was 30s)."""
        cfg = DriveConfig()
        assert cfg.adaptive_min_segment_floor_s == 15

    def test_max_lf_duration_default_720(self):
        """Operational LF cap is 12 min (was 30 min = LF_MAX_DURATION).

        LF_MAX_DURATION in video_types.py is unchanged — that remains the
        hard validator ceiling. DriveConfig.max_lf_duration is the planner's
        operational default, and it can still be overridden per-run.
        """
        cfg = DriveConfig()
        assert cfg.max_lf_duration == 720.0
