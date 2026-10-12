"""Tests for LongFormAssembler — segment selection, grouping, and shorts yield."""

from pathlib import Path

import numpy as np

from gomoveshift.config import DriveConfig
from gomoveshift.models import LFSegment, RegularLFPlan
from gomoveshift.pipeline.longform_assembler import LongFormAssembler


def _make_score_array(duration: float, pattern: str = "mixed") -> np.ndarray:
    """Create a synthetic score array for testing.

    Patterns:
        "mixed": alternating high/low 90-second blocks
        "all_high": everything above 0.7
        "all_low": everything below 0.3
        "peaks": mostly low with periodic peaks
    """
    n = int(duration)
    scores = np.zeros((n, 9), dtype=np.float64)
    scores[:, 0] = np.arange(n)  # timestamps

    if pattern == "mixed":
        for i in range(n):
            block = (i // 90) % 2  # alternating 90-second blocks
            scores[i, 1] = 0.8 if block == 0 else 0.2
    elif pattern == "all_high":
        scores[:, 1] = 0.8
    elif pattern == "all_low":
        scores[:, 1] = 0.2
    elif pattern == "peaks":
        scores[:, 1] = 0.2
        # Add 10-second peaks every 120 seconds
        for peak_start in range(0, n, 120):
            peak_end = min(peak_start + 10, n)
            scores[peak_start:peak_end, 1] = 0.9
    # Real scorer output for moving footage carries motion energy; give every
    # second a realistic driving value so the conservative min_motion_energy
    # floor (default 0.001) doesn't read these composite-only fixtures as dead.
    scores[:, 2] = 0.05
    return scores


class TestFindSegmentsAboveThreshold:
    """Segment detection from scored time series."""

    def test_alternating_pattern_finds_high_blocks(self):
        """Mixed pattern with 90s blocks -> finds the high blocks."""
        cfg = DriveConfig(min_segment_duration=90.0)
        assembler = LongFormAssembler(cfg)

        # 10 minutes of alternating 90s blocks = 3 high + 3 low
        scores = _make_score_array(600.0, "mixed")
        file_boundaries = [(0.0, 600.0)]

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=file_boundaries
        )

        # Should find ~3 segments of 90s each
        assert len(segments) >= 2
        for seg in segments:
            assert seg.duration >= 89.0  # allow 1s tolerance
            assert seg.composite_score > 0.5

    def test_all_low_returns_empty(self):
        """All footage below threshold -> no segments."""
        cfg = DriveConfig(min_segment_duration=90.0)
        assembler = LongFormAssembler(cfg)

        scores = _make_score_array(600.0, "all_low")
        file_boundaries = [(0.0, 600.0)]

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=file_boundaries
        )
        assert len(segments) == 0

    def test_all_high_returns_full_coverage(self):
        """All footage above threshold -> one or more segments covering everything."""
        cfg = DriveConfig(min_segment_duration=90.0, max_segment_duration=0.0)
        assembler = LongFormAssembler(cfg)

        scores = _make_score_array(600.0, "all_high")
        file_boundaries = [(0.0, 600.0)]

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=file_boundaries
        )

        total_duration = sum(s.duration for s in segments)
        assert total_duration >= 580.0  # Nearly all footage

    def test_short_segments_dropped(self):
        """Segments shorter than min_duration are dropped."""
        cfg = DriveConfig(min_segment_duration=90.0)
        assembler = LongFormAssembler(cfg)

        # Peaks pattern: 10-second peaks every 120s (too short for 90s minimum)
        scores = _make_score_array(600.0, "peaks")
        file_boundaries = [(0.0, 600.0)]

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=file_boundaries
        )
        # All peaks are 10s < 90s min, so no segments
        assert len(segments) == 0

    def test_multi_file_boundaries(self):
        """Segments respect file boundaries."""
        cfg = DriveConfig(min_segment_duration=90.0)
        assembler = LongFormAssembler(cfg)

        scores = _make_score_array(600.0, "all_high")
        # Two files of 300s each
        file_boundaries = [(0.0, 300.0), (300.0, 600.0)]

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=file_boundaries
        )
        # Every segment should have a valid source_file_index
        for seg in segments:
            assert seg.source_file_index in (0, 1)


class TestGroupIntoRegularLFs:
    """Bin-packing segments into regular long forms."""

    def _make_segments(self, count: int, duration: float = 120.0) -> list[LFSegment]:
        """Create N segments of given duration."""
        segments = []
        t = 0.0
        for i in range(count):
            segments.append(
                LFSegment(
                    source_file_index=0,
                    source_path=Path("/tmp/file_00.mp4"),
                    start_time=t,
                    end_time=t + duration,
                    original_start_time=t,
                    original_end_time=t + duration,
                    composite_score=0.75,
                    gps_center_lat=51.77,
                    gps_center_lon=-9.98,
                    terrain_type="mountain",
                    contiguous_with_previous=False,
                )
            )
            t += duration + 30.0  # 30s gap between segments
        return segments

    def test_basic_grouping_respects_max_duration(self):
        """LFs don't exceed max_lf_duration."""
        cfg = DriveConfig(max_lf_duration=600.0, min_lf_duration=300.0)
        assembler = LongFormAssembler(cfg)

        # 10 segments of 120s = 1200s total, should split into 2 LFs
        segments = self._make_segments(10, duration=120.0)
        # has_gps=True keeps the operational max_lf_duration target (the fewest-LFs rule only
        # changes GPS-dead packing).
        lfs = assembler.group_into_regular_lfs(segments, has_gps=True)

        for lf in lfs:
            assert lf.total_duration <= 600.0 + cfg.crossfade_duration * 10

    def test_grouping_respects_min_duration(self):
        """Each LF meets min_lf_duration (or uses all remaining segments)."""
        cfg = DriveConfig(max_lf_duration=1800.0, min_lf_duration=480.0)
        assembler = LongFormAssembler(cfg)

        # 8 segments of 120s = 960s total -> 1 LF (all fit in one)
        segments = self._make_segments(8, duration=120.0)
        lfs = assembler.group_into_regular_lfs(segments)

        assert len(lfs) >= 1
        assert lfs[0].total_duration >= 480.0

    def test_empty_segments_no_lfs(self):
        """No segments -> no LFs."""
        cfg = DriveConfig()
        assembler = LongFormAssembler(cfg)
        lfs = assembler.group_into_regular_lfs([])
        assert lfs == []

    def test_segments_maintain_chronological_order(self):
        """Segments within each LF are in chronological order."""
        cfg = DriveConfig(max_lf_duration=600.0, min_lf_duration=120.0)
        assembler = LongFormAssembler(cfg)

        segments = self._make_segments(10, duration=90.0)
        # has_gps=True keeps max_lf_duration packing so segments span multiple LFs.
        lfs = assembler.group_into_regular_lfs(segments, has_gps=True)

        for lf in lfs:
            for i in range(1, len(lf.segments)):
                assert lf.segments[i].original_start_time >= lf.segments[i - 1].original_start_time

    def test_lf_index_is_one_based(self):
        """LF indices start at 1."""
        cfg = DriveConfig(max_lf_duration=600.0, min_lf_duration=120.0)
        assembler = LongFormAssembler(cfg)

        segments = self._make_segments(10, duration=90.0)
        # has_gps=True keeps max_lf_duration packing so segments span multiple LFs.
        lfs = assembler.group_into_regular_lfs(segments, has_gps=True)

        assert lfs[0].index == 1
        if len(lfs) > 1:
            assert lfs[1].index == 2


class TestGpsDeadFewestLFs:
    """A GPS-dead day must pack into the fewest long forms (each up to the
    hard ceiling LF_MAX_DURATION) so one coarse location label covers the merged
    span, instead of many operational-target LFs each demanding its own label.
    A GPS-present day is unchanged — it still packs at max_lf_duration.
    """

    def _make_segments(self, count: int, duration: float = 250.0) -> list[LFSegment]:
        """Create N non-contiguous segments of given duration."""
        segments = []
        t = 0.0
        for _ in range(count):
            segments.append(
                LFSegment(
                    source_file_index=0,
                    source_path=Path("/tmp/file_00.mp4"),
                    start_time=t,
                    end_time=t + duration,
                    original_start_time=t,
                    original_end_time=t + duration,
                    composite_score=0.75,
                    gps_center_lat=51.77,
                    gps_center_lon=-9.98,
                    terrain_type="mountain",
                    contiguous_with_previous=False,
                )
            )
            t += duration + 30.0
        return segments

    def test_gps_dead_day_packs_into_fewer_lfs(self):
        """No-GPS day merges footage into fewer, longer LFs than a GPS day."""
        from gomoveshift.video_types import LF_MAX_DURATION

        # Defaults: max_lf_duration=720, min_lf_duration=480, LF_MAX_DURATION=1800.
        cfg = DriveConfig()
        assembler = LongFormAssembler(cfg)

        # 12 * 250s of footage: at the 720s target this splits into 6 LFs; at the
        # 1800s hard ceiling it collapses into far fewer.
        segments = self._make_segments(12, duration=250.0)

        lfs_gps = assembler.group_into_regular_lfs(segments, has_gps=True)
        lfs_no_gps = assembler.group_into_regular_lfs(segments, has_gps=False)

        assert len(lfs_no_gps) < len(lfs_gps), (
            f"GPS-dead day should pack into fewer LFs: "
            f"no_gps={len(lfs_no_gps)} vs gps={len(lfs_gps)}"
        )
        # The merge must actually use the bigger target — at least one LF grows
        # past the operational ceiling (proving it's not coincidentally fewer).
        assert any(lf.total_duration > cfg.max_lf_duration for lf in lfs_no_gps)
        # ...but never past the hard ceiling.
        for lf in lfs_no_gps:
            assert lf.total_duration <= LF_MAX_DURATION + cfg.crossfade_duration

    def test_gps_present_day_unchanged(self):
        """GPS-present day still packs at max_lf_duration (no ceiling merge)."""
        cfg = DriveConfig()
        assembler = LongFormAssembler(cfg)

        segments = self._make_segments(12, duration=250.0)
        lfs_gps = assembler.group_into_regular_lfs(segments, has_gps=True)

        # Every GPS-present LF stays within the operational target (plus crossfade
        # slack), exactly as before the GPS-dead packing — the hard ceiling is not used.
        for lf in lfs_gps:
            assert lf.total_duration <= cfg.max_lf_duration + cfg.crossfade_duration * len(
                lf.segments
            )


class TestSelectMontagePeaks:
    """Montage peak selection from LF segments."""

    def test_selects_top_peaks(self):
        """Selects the highest-scored segments."""
        cfg = DriveConfig(montage_peak_count=3)
        assembler = LongFormAssembler(cfg)

        segments = []
        for i in range(6):
            segments.append(
                LFSegment(
                    source_file_index=0,
                    source_path=Path("/tmp/file_00.mp4"),
                    start_time=float(i * 120),
                    end_time=float(i * 120 + 90),
                    original_start_time=float(i * 120),
                    original_end_time=float(i * 120 + 90),
                    composite_score=0.5 + i * 0.08,
                    gps_center_lat=51.77,
                    gps_center_lon=-9.98,
                    terrain_type="mountain",
                    contiguous_with_previous=False,
                )
            )

        plan = RegularLFPlan(
            index=1,
            segments=segments,
            montage_peaks=[],
            shorts_target=7,
            region="ireland",
            total_duration=sum(s.duration for s in segments),
        )

        peaks = assembler.select_montage_peaks(plan)
        assert len(peaks) == 3
        # Peaks should be the highest-scored segments
        peak_scores = [p.composite_score for p in peaks]
        assert all(s >= 0.74 for s in peak_scores)  # Top 3 of [0.5, 0.58, 0.66, 0.74, 0.82, 0.90]


class TestBuildDrivePlan:
    """End-to-end drive plan construction."""

    def test_builds_plan_from_scores(self):
        """Full pipeline: scores -> segments -> LFs -> plan."""
        cfg = DriveConfig(
            min_segment_duration=90.0,
            max_lf_duration=600.0,
            min_lf_duration=120.0,
            min_shorts_per_lf=3,
        )
        assembler = LongFormAssembler(cfg)

        # 20 minutes of mixed content
        scores = _make_score_array(1200.0, "mixed")
        processed_files = [Path("/tmp/file_00.mp4")]
        file_boundaries = [(0.0, 1200.0)]

        plan = assembler.build_drive_plan(
            scores=scores,
            processed_files=processed_files,
            file_boundaries=file_boundaries,
            region="ireland",
        )

        # Last timestamp is N-1 for N samples
        assert plan.total_source_duration >= 1199.0
        assert plan.total_highlight_duration > 0
        assert len(plan.regular_lfs) >= 1
        assert plan.score_threshold > 0


class TestDecisionLogging:
    """build_drive_plan must log segment selection statistics."""

    def test_logs_segment_statistics(self, caplog):
        """build_drive_plan must log total segments, above/below threshold, score range."""
        import logging

        cfg = DriveConfig(
            min_segment_duration=10.0,
            max_lf_duration=600.0,
            min_lf_duration=30.0,
        )
        assembler = LongFormAssembler(cfg)

        scores = _make_score_array(300.0, "mixed")
        processed_files = [Path("/tmp/file_00.mp4")]
        file_boundaries = [(0.0, 300.0)]

        with caplog.at_level(logging.INFO, logger="gomoveshift.pipeline.longform_assembler"):
            assembler.build_drive_plan(
                scores=scores,
                processed_files=processed_files,
                file_boundaries=file_boundaries,
                region="ireland",
            )

        log_text = caplog.text
        # Must log score min/max/mean
        assert "score_min" in log_text or "min=" in log_text or "Score range" in log_text, (
            f"Must log score statistics. Got: {log_text}"
        )


class TestMotionThresholdGating:
    """Segments with insufficient motion should be excluded when threshold is set."""

    def _make_scores(self, n, composite, motion_energy):
        """Helper: create (n, 9) score array with given composite and motion."""
        arr = np.zeros((n, 9))
        arr[:, 0] = np.arange(n, dtype=float)  # timestamps
        arr[:, 1] = composite  # composite score
        arr[:, 2] = motion_energy  # motion_energy
        return arr

    def test_parked_segment_excluded_when_threshold_set(self):
        """Segments with low motion are excluded when min_motion_energy > 0."""
        cfg = DriveConfig(min_segment_duration=10.0, min_motion_energy=0.05)
        assembler = LongFormAssembler(cfg)

        # All composite scores high, but motion_energy = 0.01 (parked)
        scores = self._make_scores(120, composite=0.8, motion_energy=0.01)
        boundaries = [(0.0, 120.0)]

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=boundaries
        )

        assert len(segments) == 0, (
            f"Parked segment (motion=0.01) must be excluded when threshold=0.05, "
            f"got {len(segments)} segments"
        )

    def test_driving_segment_included_above_threshold(self):
        """Segments with sufficient motion are included."""
        cfg = DriveConfig(min_segment_duration=10.0, min_motion_energy=0.05)
        assembler = LongFormAssembler(cfg)

        # motion_energy = 0.15 (driving)
        scores = self._make_scores(120, composite=0.8, motion_energy=0.15)
        boundaries = [(0.0, 120.0)]

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=boundaries
        )

        assert len(segments) == 1, (
            f"Driving segment (motion=0.15) must be included, got {len(segments)}"
        )

    def test_zero_threshold_includes_all(self):
        """min_motion_energy=0.0 disables motion gating (backward compat)."""
        cfg = DriveConfig(min_segment_duration=10.0, min_motion_energy=0.0)
        assembler = LongFormAssembler(cfg)

        # motion_energy = 0.0 (completely static)
        scores = self._make_scores(120, composite=0.8, motion_energy=0.0)
        boundaries = [(0.0, 120.0)]

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=boundaries
        )

        assert len(segments) == 1, (
            "With min_motion_energy=0.0 (explicit), static segments must be included"
        )

    def test_default_floor_excludes_truly_dead_footage(self):
        """The default min_motion_energy is a nonzero conservative floor —
        truly-dead (frozen, motion≈0) footage is excluded out of the box, while
        ordinary moving footage well above the floor is kept."""
        cfg = DriveConfig(min_segment_duration=10.0)  # default floor active
        assert cfg.min_motion_energy > 0.0, "default floor must be enabled"

        boundaries = [(0.0, 120.0)]
        dead = self._make_scores(120, composite=0.8, motion_energy=0.0)
        assert (
            assembler_segments := LongFormAssembler(cfg).find_segments_above_threshold(
                dead, threshold=0.5, file_boundaries=boundaries
            )
        ) == [], f"frozen footage must be excluded by the default floor, got {assembler_segments}"

        moving = self._make_scores(120, composite=0.8, motion_energy=0.05)
        kept = LongFormAssembler(cfg).find_segments_above_threshold(
            moving, threshold=0.5, file_boundaries=boundaries
        )
        assert len(kept) == 1, "ordinary moving footage must survive the default floor"

    def test_avg_motion_energy_populated_in_segment(self):
        """LFSegment.avg_motion_energy reflects actual motion in the segment."""
        cfg = DriveConfig(min_segment_duration=10.0)
        assembler = LongFormAssembler(cfg)

        scores = np.zeros((120, 9))
        scores[:, 0] = np.arange(120, dtype=float)
        scores[:, 1] = 0.8  # composite
        scores[:, 2] = 0.3  # motion_energy

        boundaries = [(0.0, 120.0)]
        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=boundaries
        )

        assert len(segments) == 1
        assert abs(segments[0].avg_motion_energy - 0.3) < 0.01, (
            f"Expected avg_motion_energy≈0.3, got {segments[0].avg_motion_energy}"
        )

    def test_mixed_motion_selects_only_driving_portion(self):
        """When first half is parked and second half is driving, only driving selected."""
        cfg = DriveConfig(min_segment_duration=10.0, min_motion_energy=0.05)
        assembler = LongFormAssembler(cfg)

        scores = np.zeros((240, 9))
        scores[:, 0] = np.arange(240, dtype=float)
        # First 120 samples: high composite, low motion (parked)
        scores[:120, 1] = 0.8
        scores[:120, 2] = 0.01
        # Second 120 samples: high composite, high motion (driving)
        scores[120:, 1] = 0.8
        scores[120:, 2] = 0.15

        boundaries = [(0.0, 240.0)]
        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=boundaries
        )

        # Only driving portion should be included
        assert len(segments) == 1
        assert segments[0].original_start_time >= 120.0, (
            f"Expected segment starting at ≥120s, got {segments[0].original_start_time}"
        )


# ── Segment Duration Thresholds ────────────────────────────────────────


def _make_run_scores(run_duration: float, total: float = 300.0) -> np.ndarray:
    """Score array with a single contiguous high-score run of run_duration seconds."""
    n = int(total)
    scores = np.zeros((n, 9), dtype=np.float64)
    scores[:, 0] = np.arange(n, dtype=np.float64)  # timestamps
    scores[:, 1] = 0.1  # all low
    # Single high run starting at t=50
    run_start = 50
    run_end = run_start + int(run_duration)
    scores[run_start:run_end, 1] = 0.8
    # Moving footage carries motion energy — keep the fixture above the
    # conservative min_motion_energy floor (default 0.001).
    scores[:, 2] = 0.05
    return scores


class TestSegmentDurationThresholds:
    """min_segment_duration=60 and max_segment_duration capping."""

    def test_85s_run_selected_at_60s_threshold(self):
        """ep_003's 85s best run must be selected when threshold is 60s."""
        cfg = DriveConfig(min_segment_duration=60.0)
        assembler = LongFormAssembler(cfg)
        scores = _make_run_scores(85.0)
        file_boundaries = [(0.0, 300.0)]

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=file_boundaries
        )
        assert len(segments) == 1, (
            f"85s run should be selected at 60s threshold, got {len(segments)} segments"
        )

    def test_85s_run_excluded_at_90s_threshold(self):
        """Confirm 85s run is excluded at old 90s threshold (regression guard)."""
        cfg = DriveConfig(min_segment_duration=90.0)
        assembler = LongFormAssembler(cfg)
        scores = _make_run_scores(85.0)
        file_boundaries = [(0.0, 300.0)]

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=file_boundaries
        )
        assert len(segments) == 0, (
            f"85s run must be excluded at 90s threshold (regression guard), got {len(segments)}"
        )

    def test_long_segment_capped_at_max_duration(self):
        """338s segment must be capped at max_segment_duration, not skipped."""
        cfg = DriveConfig(min_segment_duration=60.0, max_segment_duration=180.0)
        assembler = LongFormAssembler(cfg)
        scores = _make_run_scores(338.0, total=500.0)
        file_boundaries = [(0.0, 500.0)]

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=file_boundaries
        )
        assert len(segments) == 1, "338s run should produce one segment (capped)"
        assert segments[0].duration <= 180.0 + 1.5, (
            f"Segment duration {segments[0].duration:.1f}s must be capped at <=181.5s"
        )

    def test_segment_exactly_at_min_threshold_included(self):
        """Segment exactly at min_segment_duration boundary must be included."""
        cfg = DriveConfig(min_segment_duration=60.0)
        assembler = LongFormAssembler(cfg)
        scores = _make_run_scores(60.0)
        file_boundaries = [(0.0, 300.0)]

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=file_boundaries
        )
        assert len(segments) == 1, "60s run at exactly min_segment_duration must be included"

    def test_max_segment_duration_zero_disables_cap(self):
        """max_segment_duration=0 disables capping — long segment passes through."""
        cfg = DriveConfig(min_segment_duration=60.0, max_segment_duration=0.0)
        assembler = LongFormAssembler(cfg)
        scores = _make_run_scores(338.0, total=500.0)
        file_boundaries = [(0.0, 500.0)]

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=file_boundaries
        )
        assert len(segments) == 1
        assert segments[0].duration > 180.0, (
            "With max_segment_duration=0, 338s run must NOT be capped"
        )


# ── GPS Metadata Pass-Through ─────────────────────────────────────────


def _make_mock_gps(lat: float = 51.5, lon: float = -8.5, offset: float = 0.0):
    """Create a minimal GPSData-like mock with trackpoints."""
    from unittest.mock import MagicMock

    class _PT:
        def __init__(self, ts):
            self.lat = lat
            self.lon = lon
            self.elevation = 150.0
            self.timestamp = ts

    gps = MagicMock()
    # GPS timestamps span 0+offset to 300+offset
    gps.get_points_in_range.side_effect = lambda s, e: [_PT(ts) for ts in range(int(s), int(e), 5)]
    gps.start_time = offset
    return gps


class TestGPSPassThrough:
    """GPS metadata must flow into LFSegment fields."""

    def test_lf_segment_gps_populated_when_gps_available(self):
        """gps_center_lat/lon must be non-zero when GPS data is provided."""
        cfg = DriveConfig(min_segment_duration=60.0, max_segment_duration=0.0)
        assembler = LongFormAssembler(cfg)

        scores = _make_run_scores(100.0, total=300.0)
        file_boundaries = [(0.0, 300.0)]
        mock_gps = _make_mock_gps(lat=51.5, lon=-8.5, offset=1000.0)
        gps_per_episode = [(mock_gps, 1000.0)]

        segments = assembler.find_segments_above_threshold(
            scores,
            threshold=0.5,
            file_boundaries=file_boundaries,
            gps_per_episode=gps_per_episode,
        )

        assert len(segments) == 1
        assert segments[0].gps_center_lat != 0.0, (
            "gps_center_lat must be populated when GPS data is available"
        )
        assert segments[0].gps_center_lon != 0.0, (
            "gps_center_lon must be populated when GPS data is available"
        )

    def test_lf_segment_gps_defaults_when_no_gps(self):
        """GPS fields remain 0.0/unknown when gps_per_episode is None."""
        cfg = DriveConfig(min_segment_duration=60.0, max_segment_duration=0.0)
        assembler = LongFormAssembler(cfg)

        scores = _make_run_scores(100.0, total=300.0)
        file_boundaries = [(0.0, 300.0)]

        segments = assembler.find_segments_above_threshold(
            scores,
            threshold=0.5,
            file_boundaries=file_boundaries,
        )

        assert len(segments) == 1
        assert segments[0].gps_center_lat == 0.0
        assert segments[0].gps_center_lon == 0.0
        assert segments[0].terrain_type == "unknown"


class TestTimingAlignmentGuard:
    """end_time must be clamped to file duration if scores exceed stop-removed file."""

    def test_segment_end_clamped_when_exceeds_file_boundary(self, caplog):
        """Segment end_time beyond file boundary is clamped with a warning."""
        import logging

        # File boundary ends at 80s, but 90s of scores exist (stop removal shifted things)
        cfg = DriveConfig(min_segment_duration=60.0, max_segment_duration=0.0)
        assembler = LongFormAssembler(cfg)

        # 90 high-score samples starting at t=0, file boundary = 80s
        n = 90
        scores = np.zeros((n, 9), dtype=np.float64)
        scores[:, 0] = np.arange(n, dtype=np.float64)
        scores[:, 1] = 0.9
        scores[:, 2] = 0.05  # moving footage carries motion energy
        file_boundaries = [(0.0, 80.0)]  # boundary ends at 80s

        with caplog.at_level(logging.WARNING):
            segments = assembler.find_segments_above_threshold(
                scores, threshold=0.5, file_boundaries=file_boundaries
            )

        assert len(segments) == 1
        # local_end must be clamped: file is 80s, so local_end <= 79.5
        assert segments[0].end_time <= 79.5, (
            f"Segment end_time {segments[0].end_time:.1f}s must be clamped to <=79.5s"
        )
        assert any(
            "clamp" in r.message.lower() or "exceed" in r.message.lower() for r in caplog.records
        ), "A warning must be logged when end_time is clamped"

    def test_segment_end_not_clamped_when_within_boundary(self):
        """No clamping when end_time is within the file boundary."""
        cfg = DriveConfig(min_segment_duration=60.0, max_segment_duration=0.0)
        assembler = LongFormAssembler(cfg)

        scores = _make_run_scores(70.0, total=200.0)
        file_boundaries = [(0.0, 200.0)]

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=file_boundaries
        )

        assert len(segments) == 1
        # end_time = 50 + 70 + 1 = ~121, well within 200s boundary
        assert segments[0].end_time <= 200.0


class TestCrossfadeAwareGrouping:
    """group_into_regular_lfs must account for crossfade overhead in split decisions."""

    @staticmethod
    def _make_segments(count, duration, contiguous=False, file_index=0):
        """Create segments with controllable contiguity."""
        segments = []
        t = 0.0
        for i in range(count):
            segments.append(
                LFSegment(
                    source_file_index=file_index if contiguous else i,
                    source_path=Path(f"/tmp/file_{i:02d}.mp4"),
                    start_time=t,
                    end_time=t + duration,
                    original_start_time=t,
                    original_end_time=t + duration,
                    composite_score=0.75,
                    gps_center_lat=51.77,
                    gps_center_lon=-9.98,
                    terrain_type="mountain",
                    contiguous_with_previous=(contiguous and i > 0),
                )
            )
            t += duration + (0.0 if contiguous else 30.0)
        return segments

    def test_crossfade_overhead_triggers_split(self):
        """Segments that fit raw but exceed max when crossfade is added must split.

        Setup: max_lf_duration=600s, crossfade=2s.
        3 non-contiguous segments of 198s each:
          raw total = 594s < 600s (fits)
          with crossfade = 594 + 2*2 = 598s (fits)
        4 non-contiguous segments of 198s each:
          raw total = 792s (doesn't fit regardless)

        Better test: 3 segments of 199s each:
          raw total = 597s < 600s
          with 2 crossfades = 597 + 4 = 601s > 600s
          Should split into 2 LFs.
        """
        cfg = DriveConfig(
            max_lf_duration=600.0,
            min_lf_duration=120.0,
            crossfade_duration=2.0,
        )
        assembler = LongFormAssembler(cfg)

        # 3 non-contiguous segments of 199s: raw=597 < 600, with crossfade=601 > 600
        segments = self._make_segments(3, duration=199.0, contiguous=False)
        # has_gps=True: the GPS-dead ceiling doesn't apply, so the 600s target holds.
        lfs = assembler.group_into_regular_lfs(segments, has_gps=True)

        # With crossfade accounting, 3rd segment should be split into LF 2
        assert len(lfs) >= 2, (
            f"Expected split due to crossfade overhead (597s raw + 4s crossfade = 601s > 600s), "
            f"but got {len(lfs)} LF(s) with durations {[lf.total_duration for lf in lfs]}"
        )

    def test_contiguous_segments_no_crossfade_penalty(self):
        """Contiguous segments (same source file, adjacent) get no crossfade penalty."""
        cfg = DriveConfig(
            max_lf_duration=600.0,
            min_lf_duration=120.0,
            crossfade_duration=2.0,
        )
        assembler = LongFormAssembler(cfg)

        # 3 contiguous segments of 199s: raw=597 < 600, no crossfade = 597 < 600
        segments = self._make_segments(3, duration=199.0, contiguous=True)
        # has_gps=True: the GPS-dead ceiling doesn't apply, so the 600s target holds.
        lfs = assembler.group_into_regular_lfs(segments, has_gps=True)

        # Should stay as 1 LF since no crossfade penalty for contiguous
        assert len(lfs) == 1, (
            f"Contiguous segments should not incur crossfade penalty, "
            f"expected 1 LF but got {len(lfs)}"
        )

    def test_exact_boundary_with_crossfade(self):
        """Segments exactly at max when crossfade is included should not split."""
        cfg = DriveConfig(
            max_lf_duration=600.0,
            min_lf_duration=120.0,
            crossfade_duration=2.0,
        )
        assembler = LongFormAssembler(cfg)

        # 3 non-contiguous segments: raw + crossfade = exactly 600
        # 2 crossfades of 2s each = 4s overhead
        # Need raw = 596s → segments of 596/3 ≈ 198.67s each
        seg_dur = (600.0 - 4.0) / 3  # 198.666...
        segments = self._make_segments(3, duration=seg_dur, contiguous=False)
        # has_gps=True: the GPS-dead ceiling doesn't apply, so the 600s target holds.
        lfs = assembler.group_into_regular_lfs(segments, has_gps=True)

        # Exactly at boundary — should fit in 1 LF
        assert len(lfs) == 1, (
            f"Segments exactly at max_lf_duration (with crossfade) should not split, "
            f"got {len(lfs)} LFs"
        )

    def test_no_split_for_lf_with_total_duration_at_max(self):
        """Verify total_duration on resulting LF respects max after crossfade."""
        cfg = DriveConfig(
            max_lf_duration=1800.0,
            min_lf_duration=480.0,
            crossfade_duration=2.0,
        )
        assembler = LongFormAssembler(cfg)

        # 10 non-contiguous segments of 175s each
        # raw=1750, crossfade=9*2=18, total=1768 < 1800 → 1 LF
        segments = self._make_segments(10, duration=175.0, contiguous=False)
        lfs = assembler.group_into_regular_lfs(segments)

        assert len(lfs) >= 1
        for lf in lfs:
            assert lf.total_duration <= 1800.0, (
                f"LF total_duration {lf.total_duration:.1f}s exceeds max 1800s"
            )


class TestBuildDrivePlanSignature:
    """build_drive_plan no longer accepts gps_data/gps_offset params."""

    def test_no_gps_data_param(self):
        """build_drive_plan must not accept gps_data keyword arg."""
        import inspect
        from gomoveshift.pipeline.longform_assembler import LongFormAssembler

        sig = inspect.signature(LongFormAssembler.build_drive_plan)
        assert "gps_data" not in sig.parameters, (
            "build_drive_plan still has dead gps_data parameter"
        )
        assert "gps_offset" not in sig.parameters, (
            "build_drive_plan still has dead gps_offset parameter"
        )

    def test_gps_per_episode_still_works(self):
        """build_drive_plan still accepts gps_per_episode."""
        import inspect
        from gomoveshift.pipeline.longform_assembler import LongFormAssembler

        sig = inspect.signature(LongFormAssembler.build_drive_plan)
        assert "gps_per_episode" in sig.parameters


class TestShortsTargetHonored:
    """Per-LF shorts target must honor shorts_target_count when > min_shorts_per_lf."""

    def _seg(self, start: float, end: float, score: float = 0.8) -> LFSegment:
        return LFSegment(
            source_file_index=0,
            source_path=Path("/tmp/file_00.mp4"),
            start_time=start,
            end_time=end,
            original_start_time=start,
            original_end_time=end,
            composite_score=score,
            gps_center_lat=51.0,
            gps_center_lon=-9.0,
            terrain_type="mountain",
            contiguous_with_previous=False,
        )

    def test_shorts_target_honors_shorts_target_count_when_higher(self):
        """When shorts_target_count > min_shorts_per_lf, per-LF target uses the larger value."""
        cfg = DriveConfig(min_shorts_per_lf=7, shorts_target_count=14)
        assembler = LongFormAssembler(cfg)

        segments = [self._seg(0.0, 90.0), self._seg(100.0, 190.0)]
        lf = assembler._create_lf_plan(segments=segments, index=1)

        assert lf.shorts_target == 14

    def test_shorts_target_falls_back_to_min_when_higher(self):
        """When min_shorts_per_lf > shorts_target_count, use min (defensive floor)."""
        cfg = DriveConfig(min_shorts_per_lf=10, shorts_target_count=5)
        assembler = LongFormAssembler(cfg)

        segments = [self._seg(0.0, 90.0), self._seg(100.0, 190.0)]
        lf = assembler._create_lf_plan(segments=segments, index=1)

        assert lf.shorts_target == 10

    def test_shorts_target_equal_values(self):
        """When min_shorts_per_lf == shorts_target_count, pick that value."""
        cfg = DriveConfig(min_shorts_per_lf=9, shorts_target_count=9)
        assembler = LongFormAssembler(cfg)

        segments = [self._seg(0.0, 90.0), self._seg(100.0, 190.0)]
        lf = assembler._create_lf_plan(segments=segments, index=1)

        assert lf.shorts_target == 9


class TestStopMaskVeto:
    """The per-day stop mask structurally excludes parked seconds from LF
    segment selection, both at the segment level and via
    the percentile-over-unmasked threshold."""

    def _custom(self, n, composite, motion=0.05):
        arr = np.zeros((n, 9), dtype=np.float64)
        arr[:, 0] = np.arange(n)
        arr[:, 1] = composite
        arr[:, 2] = motion
        return arr

    def test_masked_high_composite_span_excluded_from_segments(self):
        cfg = DriveConfig(min_segment_duration=90.0, max_segment_duration=0.0)
        assembler = LongFormAssembler(cfg)
        scores = self._custom(600, composite=0.8)  # all above threshold
        file_boundaries = [(0.0, 600.0)]

        stop_mask = np.zeros(600, dtype=bool)
        stop_mask[100:300] = True  # a high-composite parked span

        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=file_boundaries, stop_mask=stop_mask
        )
        assert segments, "expected surviving segments outside the masked span"
        for seg in segments:
            assert not (seg.original_start_time < 300 and seg.original_end_time > 100), (
                f"segment {seg.original_start_time}-{seg.original_end_time} "
                "overlaps the masked parked span"
            )

    def test_no_mask_keeps_the_span(self):
        cfg = DriveConfig(min_segment_duration=90.0, max_segment_duration=0.0)
        assembler = LongFormAssembler(cfg)
        scores = self._custom(600, composite=0.8)
        file_boundaries = [(0.0, 600.0)]
        segments = assembler.find_segments_above_threshold(
            scores, threshold=0.5, file_boundaries=file_boundaries
        )
        # Without a mask the whole timeline is one contiguous above-threshold run.
        assert any(
            seg.original_start_time <= 100 and seg.original_end_time >= 300 for seg in segments
        )

    def test_mostly_parked_day_percentile_excludes_parked_span(self):
        # 450s of high-composite PARKED footage + 150s of moving footage with a
        # LOWER composite. Without the mask the parked mass dominates the
        # percentile and the parked span is selected; with the mask the
        # threshold is computed over unmasked rows and only the moving span
        # survives.
        cfg = DriveConfig(
            min_segment_duration=90.0,
            min_lf_duration=120.0,
            max_segment_duration=0.0,
            best_moments_mode=False,
        )
        assembler = LongFormAssembler(cfg)
        composite = np.concatenate([np.full(450, 0.9), np.full(150, 0.5)])
        motion = np.concatenate([np.zeros(450), np.full(150, 0.5)])
        scores = self._custom(600, composite=composite, motion=motion)

        stop_mask = np.zeros(600, dtype=bool)
        stop_mask[0:450] = True

        plan = assembler.build_drive_plan_per_day(
            scores=scores,
            processed_files=[Path("f0.mp4")],
            file_boundaries=[(0.0, 600.0)],
            episode_dates=["2026-01-01"],
            stop_mask=stop_mask,
        )
        segs = [s for lf in plan.regular_lfs for s in lf.segments]
        assert segs, "expected the moving span to be selected"
        for seg in segs:
            assert seg.original_start_time >= 450, (
                f"segment {seg.original_start_time}-{seg.original_end_time} "
                "leaked into the masked parked span"
            )
