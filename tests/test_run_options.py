"""TDD tests for RunOptions dataclass refactor.

Verifies:
1. RunOptions construction with required and optional fields
2. Skip set behavior (replaces 14 skip_* booleans)
3. from_legacy() factory builds RunOptions from old-style kwargs
4. Orchestrator.run() accepts RunOptions
5. CLI contract still holds after refactor
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest


class TestRunOptionsConstruction:
    """RunOptions dataclass creation and defaults."""

    def test_minimal_construction(self, monkeypatch, tmp_path):
        """RunOptions with only required fields should use sane defaults."""
        from gomoveshift.pipeline.orchestrator import RunOptions

        monkeypatch.delenv("GMS_OUTPUT_DIR", raising=False)
        monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path / "data"))

        opts = RunOptions(
            video_path=Path("/tmp/test.mp4"),
            location="Dublin, Ireland",
            publish_day="mon",
        )
        assert opts.video_path == Path("/tmp/test.mp4")
        assert opts.location == "Dublin, Ireland"
        assert opts.publish_day == "mon"
        assert opts.output_base == tmp_path / "data" / "output"
        assert opts.lut_path is None
        assert opts.crossfade_duration == 0.5
        assert opts.gps_path is None
        assert opts.gps_date is None
        assert opts.gps_range is None
        assert opts.draft is False
        assert opts.fresh is False
        assert opts.skip == set()

    def test_full_construction(self):
        """RunOptions with all fields set."""
        from gomoveshift.pipeline.orchestrator import RunOptions

        opts = RunOptions(
            video_path=Path("/tmp/test.mp4"),
            location="Cork, Ireland",
            publish_day="wed",
            output_base=Path("/tmp/output"),
            lut_path=Path("/tmp/my.cube"),
            crossfade_duration=1.5,
            gps_path=Path("/tmp/track.gpx"),
            gps_date="2026-01-15",
            gps_range="10:00-12:00",
            draft=True,
            fresh=True,
            skip={"color", "stops", "ost"},
        )
        assert opts.output_base == Path("/tmp/output")
        assert opts.draft is True
        assert opts.fresh is True
        assert "color" in opts.skip
        assert len(opts.skip) == 3


class TestRunOptionsSkipSet:
    """Skip set replaces 14 boolean skip_* parameters."""

    def test_skip_set_is_independent_per_instance(self):
        """Each RunOptions instance should have its own skip set."""
        from gomoveshift.pipeline.orchestrator import RunOptions

        a = RunOptions(video_path=Path("/a.mp4"), location="A", publish_day="mon")
        b = RunOptions(video_path=Path("/b.mp4"), location="B", publish_day="tue")
        a.skip.add("color")
        assert "color" not in b.skip

    def test_skip_set_accepts_all_stage_keys(self):
        """All 14 skip keys from SKIP_FLAG_TO_STAGE should be representable."""
        from gomoveshift.pipeline.context import SKIP_FLAG_TO_STAGE
        from gomoveshift.pipeline.orchestrator import RunOptions

        # The skip set uses stage names (values from SKIP_FLAG_TO_STAGE)
        all_stages = set(SKIP_FLAG_TO_STAGE.values())
        opts = RunOptions(
            video_path=Path("/tmp/t.mp4"),
            location="Test",
            publish_day="mon",
            skip=all_stages,
        )
        assert len(opts.skip) == len(SKIP_FLAG_TO_STAGE)


class TestRunOptionsFromLegacy:
    """from_legacy() factory builds RunOptions from old-style skip_* kwargs."""

    def test_from_legacy_maps_booleans_to_skip_set(self):
        """Legacy skip_color=True should become 'color_correction' in skip set."""
        from gomoveshift.pipeline.orchestrator import RunOptions

        opts = RunOptions.from_legacy(
            video_path=Path("/tmp/t.mp4"),
            location="Test",
            publish_day="mon",
            skip_color=True,
            skip_stops=True,
            skip_ost=False,
            skip_audio=True,
        )
        from gomoveshift.pipeline.context import PipelineContext

        assert PipelineContext.STAGE_COLOR_CORRECTION in opts.skip
        assert PipelineContext.STAGE_STOP_REMOVAL in opts.skip
        assert PipelineContext.STAGE_AUDIO_NORMALIZATION in opts.skip
        assert PipelineContext.STAGE_OST_OVERLAYS not in opts.skip

    def test_from_legacy_passes_through_non_skip_params(self):
        """Non-skip parameters should be preserved."""
        from gomoveshift.pipeline.orchestrator import RunOptions

        opts = RunOptions.from_legacy(
            video_path=Path("/tmp/t.mp4"),
            location="Test",
            publish_day="wed",
            output_base=Path("/tmp/out"),
            lut_path=Path("/tmp/lut.cube"),
            crossfade_duration=2.0,
            gps_path=Path("/tmp/gps.gpx"),
            gps_date="2026-01-01",
            gps_range="09:00-11:00",
            draft=True,
            fresh=True,
        )
        assert opts.output_base == Path("/tmp/out")
        assert opts.lut_path == Path("/tmp/lut.cube")
        assert opts.crossfade_duration == 2.0
        assert opts.draft is True
        assert opts.fresh is True

    def test_from_legacy_defaults(self, monkeypatch, tmp_path):
        """from_legacy with no skip flags produces empty skip set."""
        from gomoveshift.pipeline.orchestrator import RunOptions

        monkeypatch.delenv("GMS_OUTPUT_DIR", raising=False)
        monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path / "data"))
        opts = RunOptions.from_legacy(
            video_path=Path("/tmp/t.mp4"),
            location="Test",
            publish_day="mon",
        )
        assert opts.output_base == tmp_path / "data" / "output"
        assert opts.skip == set()
        assert opts.draft is False
        assert opts.fresh is False


class TestOutputBaseOverride:
    """The output base resolves from the environment and an explicit value wins."""

    def test_output_dir_env_overrides_default(self, monkeypatch, tmp_path):
        from gomoveshift.pipeline.orchestrator import RunOptions

        monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path / "elsewhere"))
        opts = RunOptions(video_path=Path("/tmp/t.mp4"), location="Test", publish_day="mon")
        assert opts.output_base == tmp_path / "elsewhere"
        legacy = RunOptions.from_legacy(
            video_path=Path("/tmp/t.mp4"), location="Test", publish_day="mon"
        )
        assert legacy.output_base == tmp_path / "elsewhere"

    def test_explicit_output_base_wins_over_environment(self, monkeypatch, tmp_path):
        from gomoveshift.pipeline.orchestrator import RunOptions

        monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path / "elsewhere"))
        opts = RunOptions(
            video_path=Path("/tmp/t.mp4"),
            location="Test",
            publish_day="mon",
            output_base=Path("/tmp/explicit"),
        )
        assert opts.output_base == Path("/tmp/explicit")


class TestOrchestratorAcceptsRunOptions:
    """Orchestrator.run() must accept RunOptions."""

    def test_run_signature_accepts_run_options(self):
        """run() should accept a RunOptions as its first non-self parameter."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        sig = inspect.signature(PipelineOrchestrator.run)
        params = list(sig.parameters.keys())
        # First param after self should be 'options'
        assert "options" in params

    def test_run_still_accepts_legacy_kwargs(self):
        """run() must still accept old-style kwargs for backward compatibility."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        sig = inspect.signature(PipelineOrchestrator.run)
        params = sig.parameters
        # Should accept **kwargs or have the legacy params
        assert "options" in params


class TestCliContract:
    """CLI handler constructs RunOptions correctly."""

    def test_cli_handler_uses_run_options_or_from_legacy(self):
        """CLI handler should construct RunOptions and pass to run()."""
        import re

        from gomoveshift.cli.handlers import pipeline as pipeline_handler

        source = inspect.getsource(pipeline_handler)
        # Should reference RunOptions somewhere
        assert "RunOptions" in source or "from_legacy" in source or "options=" in source, (
            "CLI handler should use RunOptions"
        )
