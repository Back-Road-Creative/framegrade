"""SEO startup-validation wiring in ``cmd_drive`` / ``cmd_pipeline``.

The validator itself is unit-tested in ``test_seo_config_validation.py``. Here
we pin the CLI-handler wiring: both handlers call
``config.seo.validate_paths_exist()`` BEFORE orchestrator construction, and a
``ConfigError`` aborts with exit 2 with no orchestrator side effects.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.cli.handlers import drive as drive_handler
from gomoveshift.cli.handlers import pipeline as pipeline_handler
from gomoveshift.config import SEOConfig

# Opt out of the conftest no-op stub — the whole point is to confirm the gate
# fires end-to-end through the CLI handlers.
pytestmark = pytest.mark.real_seo_validation


def _drive_args(directory, output_dir):
    return SimpleNamespace(
        directory=str(directory),
        region="usa",
        # cmd_drive now refuses when
        # neither --publish nor --lf-day is given, before the SEO gate this
        # file exercises — give it a harmless default publish day.
        publish="mon",
        location=None,
        start_date=None,
        shorts_per_day=None,
        shorts_target=None,
        shorts_gap=None,
        preview=False,
        no_review=True,
        no_audio=False,
        skip_validation=False,
        phase=None,
        fresh=False,
        skip_upload=True,
        skip_csv=True,
        youtube_upload=False,
        no_tmux=True,
        output_dir=str(output_dir),
        per_lf_locations=None,
    )


def _pipeline_args(video, output_dir):
    return SimpleNamespace(
        video=video,
        location=None,
        publish="wed",
        output_dir=output_dir,
        lut=None,
        crossfade=2.0,
        skip_color=True,
        skip_stops=True,
        fresh=False,
        skip_thumbnails=True,
        skip_shorts=True,
        skip_schedule=True,
        skip_social=True,
        skip_upload=True,
        skip_csv=True,
        skip_d_copy=True,
        skip_chapters=True,
        gps=None,
        gps_date=None,
        gps_range=None,
        draft=False,
        no_cleanup=True,
        no_tmux=True,
    )


def _seo_paths(tmp_path, *, all_missing=False):
    """Build a SEOConfig pointed at tmp_path; optionally missing on disk."""
    if all_missing:
        return SEOConfig(
            storage_path=str(tmp_path / "missing-parent" / "seo"),
            credentials_dir=str(tmp_path / "missing-creds"),
            manager_path=str(tmp_path / "missing-manager"),
        )
    (tmp_path / "seo-storage").mkdir()
    (tmp_path / "credentials").mkdir()
    (tmp_path / "manager").mkdir()
    return SEOConfig(
        storage_path=str(tmp_path / "seo-storage" / "seo"),
        credentials_dir=str(tmp_path / "credentials"),
        manager_path=str(tmp_path / "manager"),
    )


@pytest.fixture
def in_tmux(monkeypatch):
    monkeypatch.setattr(drive_handler, "is_in_tmux", lambda: True)
    monkeypatch.setattr(pipeline_handler, "is_in_tmux", lambda: True)


class TestCmdDriveSEOValidation:
    def test_missing_seo_paths_abort_with_exit_2(self, tmp_path, in_tmux, capsys):
        args = _drive_args(tmp_path / "src", tmp_path / "out")
        config = MagicMock()
        config.seo = _seo_paths(tmp_path, all_missing=True)
        orch = MagicMock()
        with (
            patch.object(drive_handler, "configure_logging"),
            patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator", orch),
        ):
            rc = drive_handler.cmd_drive(args, config)
        # exit 2, orchestrator never reached (gate fires before any phase work).
        assert rc == 2
        orch.assert_not_called()
        captured = capsys.readouterr()
        assert "missing-manager" in captured.err

    def test_seo_paths_present_proceeds(self, tmp_path, in_tmux):
        args = _drive_args(tmp_path / "src", tmp_path / "out")
        config = MagicMock()
        config.seo = _seo_paths(tmp_path)
        orch = MagicMock()
        orch.return_value.run.return_value = SimpleNamespace(
            success=True, drive_plan=None, errors=[]
        )
        with (
            patch.object(drive_handler, "configure_logging"),
            patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator", orch),
        ):
            rc = drive_handler.cmd_drive(args, config)
        orch.assert_called_once()  # gate passed; pipeline init reached
        assert rc == 0

    def test_skip_validation_flag_bypasses_gate(self, tmp_path, in_tmux):
        args = _drive_args(tmp_path / "src", tmp_path / "out")
        config = MagicMock()
        seo = _seo_paths(tmp_path, all_missing=True)
        seo.skip_validation = True
        config.seo = seo
        orch = MagicMock()
        orch.return_value.run.return_value = SimpleNamespace(
            success=True, drive_plan=None, errors=[]
        )
        with (
            patch.object(drive_handler, "configure_logging"),
            patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator", orch),
        ):
            rc = drive_handler.cmd_drive(args, config)
        orch.assert_called_once()
        assert rc == 0


class TestCmdPipelineSEOValidation:
    def test_missing_seo_paths_abort_with_exit_2(self, tmp_path, in_tmux, capsys):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"")
        args = _pipeline_args(video, tmp_path / "out")
        config = MagicMock()
        config.seo = _seo_paths(tmp_path, all_missing=True)
        orch = MagicMock()
        with (
            patch.object(pipeline_handler, "configure_logging"),
            patch("gomoveshift.pipeline.PipelineOrchestrator", orch),
        ):
            rc = pipeline_handler.cmd_pipeline(args, config)
        assert rc == 2
        orch.assert_not_called()
        captured = capsys.readouterr()
        assert "missing-manager" in captured.err

    def test_seo_paths_present_proceeds(self, tmp_path, in_tmux):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"")
        args = _pipeline_args(video, tmp_path / "out")
        config = MagicMock()
        config.seo = _seo_paths(tmp_path)
        orch = MagicMock()
        orch.return_value.run.return_value = SimpleNamespace(
            success=True,
            output_dir=tmp_path / "out",
            final_video=None,
            metadata_file=None,
            errors=[],
        )
        with (
            patch.object(pipeline_handler, "configure_logging"),
            patch("gomoveshift.pipeline.PipelineOrchestrator", orch),
        ):
            rc = pipeline_handler.cmd_pipeline(args, config)
        orch.assert_called_once()
        assert rc == 0
