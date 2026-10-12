"""Tests for Drive Pipeline data models and configuration."""

from pathlib import Path

import pytest

from gomoveshift.models import (
    DrivePlan,
    DrivePipelineResult,
    LFSegment,
    RegularLFPlan,
)
from gomoveshift.config import DriveConfig


# ── LFSegment ────────────────────────────────────────────────────────


class TestLFSegment:
    """LFSegment dataclass behavior."""

    def _make_segment(self, **overrides) -> LFSegment:
        defaults = {
            "source_file_index": 0,
            "source_path": Path("/tmp/file_00.mp4"),
            "start_time": 10.0,
            "end_time": 110.0,
            "original_start_time": 10.0,
            "original_end_time": 110.0,
            "composite_score": 0.75,
            "gps_center_lat": 51.77,
            "gps_center_lon": -9.98,
            "terrain_type": "coastal",
            "contiguous_with_previous": False,
        }
        defaults.update(overrides)
        return LFSegment(**defaults)

    def test_duration_property(self):
        seg = self._make_segment(start_time=10.0, end_time=110.0)
        assert seg.duration == 100.0

    def test_duration_zero_length_segment(self):
        seg = self._make_segment(start_time=50.0, end_time=50.0)
        assert seg.duration == 0.0

    def test_to_dict_roundtrip(self):
        seg = self._make_segment()
        d = seg.to_dict()
        assert d["source_file_index"] == 0
        assert d["composite_score"] == 0.75
        assert d["terrain_type"] == "coastal"
        assert d["duration"] == 100.0
        # Verify all fields present
        assert "source_path" in d
        assert "gps_center_lat" in d

    def test_from_dict_roundtrip(self):
        seg = self._make_segment()
        d = seg.to_dict()
        restored = LFSegment.from_dict(d)
        assert restored.source_file_index == seg.source_file_index
        assert restored.start_time == seg.start_time
        assert restored.end_time == seg.end_time
        assert restored.composite_score == seg.composite_score
        assert restored.terrain_type == seg.terrain_type


# ── RegularLFPlan ────────────────────────────────────────────────────


class TestRegularLFPlan:
    """RegularLFPlan dataclass behavior."""

    def _make_plan(self, num_segments=3, **overrides) -> RegularLFPlan:
        segments = []
        t = 0.0
        for i in range(num_segments):
            segments.append(
                LFSegment(
                    source_file_index=i % 5,
                    source_path=Path(f"/tmp/file_{i:02d}.mp4"),
                    start_time=t,
                    end_time=t + 120.0,
                    original_start_time=t * 2,
                    original_end_time=(t + 120.0) * 2,
                    composite_score=0.7 + i * 0.05,
                    gps_center_lat=51.77 + i * 0.01,
                    gps_center_lon=-9.98 + i * 0.01,
                    terrain_type="mountain",
                    contiguous_with_previous=(i > 0),
                )
            )
            t += 120.0

        defaults = {
            "index": 1,
            "segments": segments,
            "montage_peaks": segments[:2],
            "shorts_target": 7,
            "region": "ireland",
            "total_duration": sum(s.duration for s in segments),
        }
        defaults.update(overrides)
        return RegularLFPlan(**defaults)

    def test_segment_count(self):
        plan = self._make_plan(num_segments=5)
        assert len(plan.segments) == 5

    def test_total_duration_matches_segments(self):
        plan = self._make_plan(num_segments=3)
        expected = sum(s.duration for s in plan.segments)
        assert plan.total_duration == expected

    def test_to_dict_includes_segments(self):
        plan = self._make_plan(num_segments=2)
        d = plan.to_dict()
        assert d["index"] == 1
        assert d["region"] == "ireland"
        assert len(d["segments"]) == 2
        assert len(d["montage_peaks"]) == 2

    def test_from_dict_roundtrip(self):
        plan = self._make_plan(num_segments=2)
        d = plan.to_dict()
        restored = RegularLFPlan.from_dict(d)
        assert restored.index == plan.index
        assert restored.region == plan.region
        assert len(restored.segments) == len(plan.segments)
        assert restored.segments[0].composite_score == plan.segments[0].composite_score


# ── DrivePlan ────────────────────────────────────────────────────────


class TestDrivePlan:
    """DrivePlan dataclass behavior."""

    def test_basic_construction(self):
        plan = DrivePlan(
            source_files=[Path("/tmp/a.mp4"), Path("/tmp/b.mp4")],
            processed_files=[Path("/tmp/a_proc.mp4"), Path("/tmp/b_proc.mp4")],
            regular_lfs=[],
            score_threshold=0.6,
            total_source_duration=7200.0,
            total_highlight_duration=3600.0,
        )
        assert plan.total_source_duration == 7200.0
        assert plan.highlight_ratio == 0.5

    def test_highlight_ratio_zero_source(self):
        plan = DrivePlan(
            source_files=[],
            processed_files=[],
            regular_lfs=[],
            score_threshold=0.6,
            total_source_duration=0.0,
            total_highlight_duration=0.0,
        )
        assert plan.highlight_ratio == 0.0

    def test_to_dict_and_save_load(self, tmp_path):
        plan = DrivePlan(
            source_files=[Path("/tmp/a.mp4")],
            processed_files=[Path("/tmp/a_proc.mp4")],
            regular_lfs=[],
            score_threshold=0.65,
            total_source_duration=3600.0,
            total_highlight_duration=2400.0,
        )
        out = tmp_path / "drive_plan.json"
        plan.save(out)
        loaded = DrivePlan.load(out)
        assert loaded.score_threshold == 0.65
        assert loaded.total_source_duration == 3600.0


# ── DrivePipelineResult ─────────────────────────────────────────────


class TestDrivePipelineResult:
    """DrivePipelineResult dataclass behavior."""

    def test_basic_construction(self):
        result = DrivePipelineResult(
            success=True,
            source_dir=Path("/tmp/source"),
            output_dir=Path("/tmp/output"),
            drive_plan=None,
            regular_lf_paths=[Path("/tmp/lf1.mp4")],
            shorts_per_lf={1: []},
            stages_completed=["ingest", "analyze"],
            errors=[],
            timings={"ingest": 5.0},
        )
        assert result.success is True
        assert len(result.stages_completed) == 2


# ── DriveConfig ──────────────────────────────────────────────────────


class TestDriveConfig:
    """DriveConfig pydantic model behavior."""

    def test_default_values(self):
        cfg = DriveConfig()
        assert cfg.min_segment_duration == 90.0
        assert cfg.crossfade_duration == 2.0
        assert cfg.montage_peak_count == 6
        assert cfg.min_lf_duration == 480.0
        assert cfg.max_lf_duration == 720.0
        assert cfg.min_shorts_per_lf == 7

    def test_default_weights_sum_to_one(self):
        cfg = DriveConfig()
        total = sum(cfg.scoring_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_custom_weights(self):
        cfg = DriveConfig(
            scoring_weights={
                "motion_energy": 0.50,
                "speed_variance": 0.10,
                "color_vibrancy": 0.10,
                "scene_change": 0.05,
                "sky_ratio": 0.05,
                "golden_hour": 0.05,
                "scenic_score": 0.15,
            }
        )
        assert cfg.scoring_weights["motion_energy"] == 0.50
        total = sum(cfg.scoring_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_validate_weights_rejects_bad_sum(self):
        """Bad scoring weights are now rejected at construction time."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            DriveConfig(
                scoring_weights={
                    "motion_energy": 0.50,
                    "speed_variance": 0.50,
                    "color_vibrancy": 0.50,
                }
            )

    def test_validate_weights_accepts_good_sum(self):
        cfg = DriveConfig()
        cfg.validate_weights()  # Should not raise

    def test_bad_weights_rejected_at_construction(self):
        """DriveConfig with bad weight sum must raise at construction, not just on manual call."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            DriveConfig(
                scoring_weights={
                    "motion_energy": 0.50,
                    "speed_variance": 0.50,
                    "color_vibrancy": 0.50,
                }
            )

    def test_invalid_duration_range_rejected(self):
        """min_lf_duration > max_lf_duration must raise at construction."""
        with pytest.raises(ValueError, match="min_lf_duration"):
            DriveConfig(min_lf_duration=2000.0, max_lf_duration=1000.0)

    def test_zero_segment_duration_rejected(self):
        """min_segment_duration <= 0 must raise at construction."""
        with pytest.raises(ValueError, match="min_segment_duration"):
            DriveConfig(min_segment_duration=0.0)

    def test_zero_montage_peak_count_rejected(self):
        """montage_peak_count < 1 must raise at construction."""
        with pytest.raises(ValueError, match="montage_peak_count"):
            DriveConfig(montage_peak_count=0)

    def test_default_superlong_settings(self):
        cfg = DriveConfig()
        assert cfg.superlong_part_max_duration_sec == 14400
        assert cfg.superlong_split_grace == 0.08

    def test_superlong_settings_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="superlong_part_max_duration_sec"):
            DriveConfig(superlong_part_max_duration_sec=0)
        with pytest.raises(ValueError, match="superlong_split_grace"):
            DriveConfig(superlong_split_grace=-0.01)
        with pytest.raises(ValueError, match="superlong_split_grace"):
            DriveConfig(superlong_split_grace=0.51)

    def test_config_loads_drive_section(self):
        """DriveConfig is accessible via the main Config object."""
        from gomoveshift.config import Config

        config = Config()
        assert hasattr(config, "drive")
        assert isinstance(config.drive, DriveConfig)
        assert config.drive.min_segment_duration == 90.0
