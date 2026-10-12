"""``--publish`` implies ``--youtube-upload``."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest


def _parse_drive(*argv: str):
    from gomoveshift.cli.parser import _build_parser

    return _build_parser().parse_args(["drive", *argv])


class TestDriveRequiresPublishOrLfDay:
    """A real drive run must refuse before any Phase A work when neither
    ``--publish`` nor ``--lf-day`` was given:
    a real run has silently defaulted/skipped upload+journal steps with no
    error, discovered only after the fact. Discovery-only runs
    (``--list-lfs``/``--label-dry-run``) are exempt."""

    def test_no_publish_no_lf_day_refuses_launch(self, tmp_path):
        from gomoveshift.cli.handlers.drive import cmd_drive
        from gomoveshift.config import Config

        args = _parse_drive(str(tmp_path))
        with pytest.raises(SystemExit) as excinfo:
            cmd_drive(args, Config())
        assert "--publish" in str(excinfo.value)
        assert "--lf-day" in str(excinfo.value)

    def test_list_lfs_exempt_from_schedule_guard(self, tmp_path):
        from gomoveshift.cli.handlers.drive import cmd_drive
        from gomoveshift.config import Config

        args = _parse_drive(str(tmp_path), "--list-lfs")
        with (
            patch("gomoveshift.cli.handlers.drive.is_in_tmux", return_value=True),
            patch("gomoveshift.cli.handlers.drive.configure_logging"),
            patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator") as orch_cls,
        ):
            from types import SimpleNamespace

            orch_cls.return_value.run.return_value = SimpleNamespace(
                success=True, stages_completed=[], drive_plan=None
            )
            # Must NOT raise SystemExit from the schedule guard.
            cmd_drive(args, Config())

    def test_label_dry_run_exempt_from_schedule_guard(self, tmp_path):
        from gomoveshift.cli.handlers.drive import cmd_drive
        from gomoveshift.config import Config

        args = _parse_drive(str(tmp_path), "--label-dry-run")
        with (
            patch("gomoveshift.cli.handlers.drive.is_in_tmux", return_value=True),
            patch("gomoveshift.cli.handlers.drive.configure_logging"),
            patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator") as orch_cls,
        ):
            from types import SimpleNamespace

            orch_cls.return_value.run.return_value = SimpleNamespace(
                success=True, stages_completed=[], drive_plan=None
            )
            cmd_drive(args, Config())


class TestPublishImpliesYoutubeUpload:
    def test_publish_alone_implies_upload(self, tmp_path):
        from gomoveshift.cli.handlers.drive import resolve_youtube_upload

        assert resolve_youtube_upload(_parse_drive(str(tmp_path), "--publish", "mon")) is True

    def test_publish_with_opt_out_is_false(self, tmp_path):
        from gomoveshift.cli.handlers.drive import resolve_youtube_upload

        args = _parse_drive(str(tmp_path), "--publish", "mon", "--no-youtube-upload")
        assert resolve_youtube_upload(args) is False

    def test_no_publish_no_flag_default_off(self, tmp_path):
        from gomoveshift.cli.handlers.drive import resolve_youtube_upload

        assert resolve_youtube_upload(_parse_drive(str(tmp_path))) is False


class TestPhaseDBannerLog:
    def test_enabled_banner(self, caplog):
        from gomoveshift.pipeline.drive_phases.phase_d import log_upload_lfs_banner

        with caplog.at_level(logging.INFO):
            log_upload_lfs_banner(
                youtube_upload_enabled=True,
                skip_youtube_upload=False,
                no_youtube_upload=False,
                publish_set=True,
            )
        msgs = [r.getMessage() for r in caplog.records]
        assert any("D-UPLOAD-LFS:" in m and "ENABLED" in m for m in msgs)

    def test_banner_agrees_with_gate_despite_stale_no_youtube_upload(self, caplog):
        """A stale saved no_youtube_upload=True must NOT print DISABLED
        when the actual gate (youtube_upload + skip) will upload."""
        from gomoveshift.pipeline.drive_phases.phase_d import log_upload_lfs_banner

        with caplog.at_level(logging.INFO):
            log_upload_lfs_banner(
                youtube_upload_enabled=True,
                skip_youtube_upload=False,
                no_youtube_upload=True,
                publish_set=True,
            )
        msgs = [r.getMessage() for r in caplog.records]
        assert any("D-UPLOAD-LFS: ENABLED" in m for m in msgs)
        assert not any("DISABLED" in m for m in msgs)

    def test_disabled_banner_mentions_flag(self, caplog):
        from gomoveshift.pipeline.drive_phases.phase_d import log_upload_lfs_banner

        with caplog.at_level(logging.INFO):
            log_upload_lfs_banner(
                youtube_upload_enabled=False,
                skip_youtube_upload=False,
                no_youtube_upload=True,
                publish_set=True,
            )
        msgs = [r.getMessage() for r in caplog.records]
        assert any(
            "D-UPLOAD-LFS:" in m and "DISABLED" in m and "--no-youtube-upload" in m for m in msgs
        )
