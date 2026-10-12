"""Tests for PipelineValidator — final pre-publish quality gate."""

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.pipeline_validator import (
    PipelineValidator,
    ValidationSeverity,
)


@pytest.fixture
def validator():
    return PipelineValidator(
        max_lufs=-24.0,
        max_true_peak=-3.0,
        min_video_bitrate_kbps=5000,
        duration_tolerance=0.10,
    )


class TestContentSafetyGate:
    """Content safety gate must catch unsafe metadata."""

    def test_catches_unsafe_metadata_title(self, validator):
        gate = validator._check_content_safety(
            metadata_json={"title": "You Won't Believe This Road"},
            social_posts=None,
        )
        assert not gate.passed

    def test_catches_unsafe_social_post(self, validator):
        gate = validator._check_content_safety(
            metadata_json=None,
            social_posts=[
                {
                    "youtube_title": "Turn up the volume for this one 🔊",
                    "tiktok_title": "Safe title",
                    "caption": "Safe caption",
                }
            ],
        )
        assert not gate.passed

    def test_passes_safe_content(self, validator):
        gate = validator._check_content_safety(
            metadata_json={
                "title": "POV Drive Through Ireland",
                "description": "A scenic drive through County Tipperary.",
            },
            social_posts=[
                {
                    "youtube_title": "Scenic Drive — Ireland 🚗",
                    "tiktok_title": "Scenic Drive — Ireland",
                    "caption": "Peaceful POV drive through rural Ireland.",
                }
            ],
        )
        assert gate.passed

    def test_catches_unsafe_description(self, validator):
        gate = validator._check_content_safety(
            metadata_json={
                "title": "Safe title",
                "description": "This road is the most beautiful road in the world.",
            },
            social_posts=None,
        )
        assert not gate.passed


class TestFileIntegrityGate:
    """File integrity checks."""

    def test_nonexistent_file_fails(self, validator, tmp_path):
        gate = validator._check_file_integrity(tmp_path / "nonexistent.mp4")
        assert not gate.passed

    def test_tiny_file_fails(self, validator, tmp_path):
        tiny = tmp_path / "tiny.mp4"
        tiny.write_bytes(b"x" * 100)
        gate = validator._check_file_integrity(tiny)
        assert not gate.passed


class TestDurationGate:
    """Duration checks need a real video — skip if ffprobe unavailable."""

    def test_no_expected_duration_passes(self, validator, tmp_path):
        # With expected_duration=0, should pass
        gate = validator._check_duration(tmp_path / "fake.mp4", 0)
        assert gate.passed


# ── Drive-Pipeline Health Checks ───────────────────────────────────────


def _ffprobe_mock(duration: float):
    """Return mock subprocess result with given duration."""
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = json.dumps({"format": {"duration": str(duration)}})
    return mock


def _make_lf_plan(duration: float, index: int = 1) -> RegularLFPlan:
    seg = LFSegment(
        source_file_index=0,
        source_path=Path("/tmp/ep.mp4"),
        start_time=0.0,
        end_time=duration,
        original_start_time=0.0,
        original_end_time=duration,
        composite_score=0.3,
        gps_center_lat=0.0,
        gps_center_lon=0.0,
        terrain_type="unknown",
        contiguous_with_previous=False,
    )
    return RegularLFPlan(
        index=index,
        segments=[seg],
        montage_peaks=[],
        shorts_target=7,
        region="usa",
        total_duration=duration,
    )


def _make_plan(lf_durations: list) -> DrivePlan:
    lfs = [_make_lf_plan(d, i + 1) for i, d in enumerate(lf_durations)]
    return DrivePlan(
        source_files=[Path("/tmp/src.mp4")],
        processed_files=[Path("/tmp/proc.mp4")],
        regular_lfs=lfs,
        score_threshold=0.2,
        total_source_duration=sum(lf_durations),
        total_highlight_duration=sum(lf_durations),
    )


class TestValidateVideoFile:
    def test_missing_file_fails(self, tmp_path):
        v = PipelineValidator()
        result = v.validate_video_file(tmp_path / "nonexistent.mp4")
        assert not result.passed
        assert result.severity == ValidationSeverity.CRITICAL

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_zero_duration_fails(self, mock_run, tmp_path):
        mock_run.return_value = _ffprobe_mock(0.0)
        p = tmp_path / "empty.mp4"
        p.touch()
        result = PipelineValidator().validate_video_file(p)
        assert not result.passed

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_valid_file_passes(self, mock_run, tmp_path):
        mock_run.return_value = _ffprobe_mock(120.0)
        p = tmp_path / "ok.mp4"
        p.touch()
        result = PipelineValidator().validate_video_file(p)
        assert result.passed


class TestValidateDuration:
    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_within_tolerance_passes(self, mock_run, tmp_path):
        mock_run.return_value = _ffprobe_mock(120.0)
        p = tmp_path / "v.mp4"
        p.touch()
        result = PipelineValidator().validate_duration(p, expected_seconds=120.0)
        assert result.passed

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_montage_doubled_duration_fails(self, mock_run, tmp_path):
        """57.8s actual vs 36s expected must fail at 10% tolerance."""
        mock_run.return_value = _ffprobe_mock(57.8)
        p = tmp_path / "montage.mp4"
        p.touch()
        result = PipelineValidator().validate_duration(p, expected_seconds=36.0, tolerance=0.10)
        assert not result.passed

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_pts_inflation_critical(self, mock_run, tmp_path):
        """35177s actual vs 27541s expected (28% over) must fail with CRITICAL."""
        mock_run.return_value = _ffprobe_mock(35177.0)
        p = tmp_path / "lf_01.mp4"
        p.touch()
        result = PipelineValidator().validate_duration(p, expected_seconds=27541.0, tolerance=0.05)
        assert not result.passed
        assert result.severity == ValidationSeverity.CRITICAL


class TestValidateScoreDistribution:
    def test_nan_fails(self):
        scores = np.array([0.1, float("nan"), 0.3])
        result = PipelineValidator().validate_score_distribution(scores)
        assert not result.passed

    def test_inf_fails(self):
        scores = np.array([0.1, float("inf"), 0.3])
        result = PipelineValidator().validate_score_distribution(scores)
        assert not result.passed

    def test_all_identical_fails(self):
        result = PipelineValidator().validate_score_distribution(np.full(100, 0.5))
        assert not result.passed

    def test_normal_distribution_passes(self):
        rng = np.random.default_rng(42)
        scores = rng.uniform(0.1, 0.9, 500)
        result = PipelineValidator().validate_score_distribution(scores)
        assert result.passed


class TestValidateAssemblyPlan:
    def test_empty_plan_fails(self):
        plan = _make_plan([])
        result = PipelineValidator().validate_assembly_plan(plan)
        assert not result.passed

    def test_lf_too_short_fails(self):
        result = PipelineValidator().validate_assembly_plan(_make_plan([300.0]))
        assert not result.passed

    def test_lf_too_long_fails(self):
        result = PipelineValidator().validate_assembly_plan(_make_plan([2000.0]))
        assert not result.passed

    def test_valid_plan_passes(self):
        result = PipelineValidator().validate_assembly_plan(_make_plan([900.0, 1200.0]))
        assert result.passed


class TestValidateMontageOutput:
    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_catches_doubled_duration(self, mock_run, tmp_path):
        """57.8s actual vs 36s target must fail."""
        mock_run.return_value = _ffprobe_mock(57.8)
        p = tmp_path / "montage.mp4"
        p.touch()
        result = PipelineValidator().validate_montage_output(p, target_duration=36.0)
        assert not result.passed

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_within_tolerance_passes(self, mock_run, tmp_path):
        mock_run.return_value = _ffprobe_mock(37.0)
        p = tmp_path / "montage.mp4"
        p.touch()
        result = PipelineValidator().validate_montage_output(p, target_duration=36.0)
        assert result.passed


class TestValidateConcatenatedVideo:
    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_catches_pts_inflation(self, mock_run, tmp_path):
        """35177s vs 27541s expected must fail."""
        mock_run.return_value = _ffprobe_mock(35177.0)
        p = tmp_path / "lf_01.mp4"
        p.touch()
        result = PipelineValidator().validate_concatenated_video(p, expected_duration=27541.0)
        assert not result.passed

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_within_tolerance_passes(self, mock_run, tmp_path):
        mock_run.return_value = _ffprobe_mock(1170.0)
        p = tmp_path / "lf.mp4"
        p.touch()
        result = PipelineValidator().validate_concatenated_video(p, expected_duration=1165.0)
        assert result.passed


class TestMetadataRequiredKeyCheck:
    """validate() must raise ValueError when metadata_json is missing required keys."""

    def _minimal_video(self, tmp_path):
        p = tmp_path / "v.mp4"
        p.write_bytes(b"x" * 1000)
        return p

    def test_raises_on_missing_title(self, validator, tmp_path):
        metadata = {"description": "D", "tags": ["a"], "category": "19"}
        with pytest.raises(ValueError, match="[Tt]itle"):
            validator.validate(self._minimal_video(tmp_path), metadata_json=metadata)

    def test_raises_on_missing_description(self, validator, tmp_path):
        metadata = {"title": "T", "tags": ["a"], "category": "19"}
        with pytest.raises(ValueError, match="[Dd]escription"):
            validator.validate(self._minimal_video(tmp_path), metadata_json=metadata)

    def test_raises_on_missing_tags(self, validator, tmp_path):
        metadata = {"title": "T", "description": "D", "category": "19"}
        with pytest.raises(ValueError, match="[Tt]ags"):
            validator.validate(self._minimal_video(tmp_path), metadata_json=metadata)

    def test_no_raise_on_complete_metadata(self, validator, tmp_path):
        """All required keys present — should not raise on required-key check."""
        metadata = {"title": "T", "description": "D", "tags": ["a"], "category": "19"}
        try:
            validator.validate(self._minimal_video(tmp_path), metadata_json=metadata)
        except ValueError as e:
            msg = str(e).lower()
            for key in ("title", "description", "tags", "category"):
                assert key not in msg, f"Should not raise on key check: {e}"
        except Exception:
            pass  # Other gates may fail on the dummy file


class TestValidationGateSocialPath:
    """Stage 16 must read social posts from shorts/social_posts.json (where stage 12 writes)."""

    def _build_ctx(self, tmp_path):
        """Build a minimal PipelineContext for ValidationGateStage."""
        from gomoveshift.pipeline.context import PipelineContext, PipelinePaths

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        video = project_dir / "final.mp4"
        video.write_bytes(b"x" * (2 * 1024 * 1024))  # >1 MB so should_run passes

        paths = PipelinePaths(
            output_base=tmp_path,
            project_dir=project_dir,
            stages_dir=project_dir / "_stages",
            state_file=project_dir / "_stages" / "pipeline_state.json",
            staging_dir=project_dir / "_staging",
        )
        ctx = MagicMock(spec=PipelineContext)
        ctx.project_name = "test_project"
        ctx.paths = paths
        ctx.final_video = video
        ctx.video_properties = MagicMock(duration=120.0, bitrate_kbps=8000)
        ctx.skip_stages = set()
        ctx.stages_completed = set()
        ctx.config = MagicMock()
        ctx.config.output.audio_max_lufs = -24.0
        ctx.config.output.audio_target_tp = -3.0
        return ctx

    # autospec: the stage's kwargs must bind to the REAL validate() signature,
    # so a call site cannot drift into passing a parameter validate() dropped.
    @patch("gomoveshift.pipeline.orchestrator.PipelineValidator", autospec=True)
    def test_reads_social_posts_from_shorts_dir(self, mock_validator_cls, tmp_path):
        """Social posts at shorts/social_posts.json must be loaded and passed to validate()."""
        from gomoveshift.pipeline.stages.validation_gate import ValidationGateStage

        ctx = self._build_ctx(tmp_path)
        project_dir = ctx.paths.project_dir

        # Write social posts where stage 12 puts them
        shorts_dir = project_dir / "shorts"
        shorts_dir.mkdir()
        posts = [{"youtube_title": "Scenic Drive", "tiktok_title": "Drive", "caption": "POV"}]
        (shorts_dir / "social_posts.json").write_text(json.dumps(posts))

        # Mock validator to capture what social_posts arg is passed
        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.summary.return_value = "all good"
        mock_instance = mock_validator_cls.return_value
        mock_instance.validate.return_value = mock_result

        stage = ValidationGateStage()
        stage.run(ctx)

        # The validate() call must have received the social posts (not None)
        call_kwargs = mock_instance.validate.call_args
        assert call_kwargs is not None, "validate() was never called"
        social_arg = call_kwargs.kwargs.get(
            "social_posts", call_kwargs[1].get("social_posts") if len(call_kwargs) > 1 else None
        )
        assert social_arg is not None, (
            "social_posts was None — validation gate did not read shorts/social_posts.json"
        )
        assert social_arg == posts


class TestValidateSignature:
    """``validate`` must not declare a parameter its body never reads.

    ``source_bitrate_kbps`` was the leftover trigger from when the
    ``bitrate_preservation`` floor was derived from the SOURCE bitrate. That
    floor is absolute and resolution-based now, so the gate runs on every call
    and the value reached nothing — a parameter that only invites a caller to
    believe it measures something.
    """

    def test_validate_does_not_accept_source_bitrate_kbps(self):
        params = inspect.signature(PipelineValidator.validate).parameters
        assert "source_bitrate_kbps" not in params, (
            "validate() still declares source_bitrate_kbps, which its body "
            f"never reads; parameters: {list(params)}"
        )
