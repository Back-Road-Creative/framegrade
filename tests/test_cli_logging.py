"""Tests for CLI logging setup and pipeline handler signal/logging behavior."""

import logging
import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestPipelineConfiguresLogFile:
    """Pipeline handler must pass log_file to configure_logging."""

    def test_pipeline_configures_log_file(self, tmp_path):
        """configure_logging is called with a log_file path inside output_dir."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        args = MagicMock()
        args.video = video_file
        args.no_tmux = True
        args.location = "Test Location"
        args.publish = "mon"
        args.lut = None
        args.draft = False
        args.output_dir = tmp_path / "output"
        args.crossfade = 1.0
        args.skip_color = True
        args.skip_stops = True
        args.fresh = False
        args.skip_thumbnails = True
        args.skip_shorts = True
        args.skip_schedule = True
        args.skip_social = True
        args.skip_upload = True
        args.skip_csv = True
        args.skip_d_copy = True
        args.skip_chapters = True
        args.gps = None
        args.gps_date = None
        args.gps_range = None

        with (
            patch("gomoveshift.cli.handlers.pipeline.is_in_tmux", return_value=True),
            patch("gomoveshift.cli.handlers.pipeline.configure_logging") as mock_log,
            patch("gomoveshift.pipeline.PipelineOrchestrator") as mock_orch,
        ):
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output_dir = tmp_path / "output"
            mock_result.final_video = tmp_path / "output" / "final.mp4"
            mock_result.metadata_file = None
            mock_orch.return_value.run.return_value = mock_result

            from gomoveshift.cli.handlers.pipeline import cmd_pipeline

            from gomoveshift.config import Config

            cmd_pipeline(args, config=Config())

            mock_log.assert_called_once()
            call_kwargs = mock_log.call_args
            # log_file should be a Path in the output directory
            log_file = call_kwargs.kwargs.get("log_file") or (
                call_kwargs[1].get("log_file") if len(call_kwargs) > 1 else None
            )
            assert log_file is not None, "configure_logging must be called with log_file"
            assert isinstance(log_file, Path)
            assert str(args.output_dir) in str(log_file)


class TestSigtermRaisesSystemExit:
    """SIGTERM handler must raise SystemExit(143)."""

    @pytest.mark.skipif(os.name == "nt", reason="SIGTERM not available on Windows")
    def test_sigterm_raises_system_exit(self):
        """Registering the SIGTERM handler then sending SIGTERM raises SystemExit(143)."""
        from gomoveshift.cli.handlers.pipeline import _handle_sigterm

        with pytest.raises(SystemExit) as exc_info:
            # Call the handler directly (simulates OS delivering SIGTERM)
            _handle_sigterm(signal.SIGTERM, None)
        assert exc_info.value.code == 143

    @pytest.mark.skipif(os.name == "nt", reason="SIGTERM not available on Windows")
    def test_sigterm_handler_is_registered_during_pipeline(self, tmp_path):
        """cmd_pipeline registers signal.SIGTERM before running the orchestrator."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        args = MagicMock()
        args.video = video_file
        args.no_tmux = True
        args.location = None
        args.publish = "mon"
        args.lut = None
        args.draft = False
        args.output_dir = tmp_path / "output"
        args.crossfade = 1.0
        args.skip_color = True
        args.skip_stops = True
        args.fresh = False
        args.skip_thumbnails = True
        args.skip_shorts = True
        args.skip_schedule = True
        args.skip_social = True
        args.skip_upload = True
        args.skip_csv = True
        args.skip_d_copy = True
        args.skip_chapters = True
        args.gps = None
        args.gps_date = None
        args.gps_range = None

        registered_handler = None

        def capture_signal(signum, handler):
            nonlocal registered_handler
            if signum == signal.SIGTERM:
                registered_handler = handler

        with (
            patch("gomoveshift.cli.handlers.pipeline.is_in_tmux", return_value=True),
            patch("gomoveshift.cli.handlers.pipeline.configure_logging"),
            patch("gomoveshift.pipeline.PipelineOrchestrator") as mock_orch,
            patch("signal.signal", side_effect=capture_signal),
        ):
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.output_dir = tmp_path / "output"
            mock_result.final_video = tmp_path / "output" / "final.mp4"
            mock_result.metadata_file = None
            mock_orch.return_value.run.return_value = mock_result

            from gomoveshift.cli.handlers.pipeline import cmd_pipeline

            from gomoveshift.config import Config

            cmd_pipeline(args, config=Config())

        assert registered_handler is not None, "SIGTERM handler must be registered"


class TestCmdDriveCreatesDriveLog:
    """cmd_drive must pass log_file so drive runs leave a persistent artifact.

    Drive invocations outside tmux used to leave zero on-disk log. The stream handler's stdout is gone once the tty
    closes; a persistent ``drive.log`` under the project dir is the
    simplest fix. cmd_pipeline already does this.
    """

    def test_cmd_drive_creates_drive_log(self, tmp_path):
        """configure_logging receives a log_file under the output/project dir."""
        args = MagicMock()
        args.directory = tmp_path / "source"
        args.directory.mkdir()
        args.output_dir = tmp_path / "output"
        args.region = "ireland"
        args.publish = None
        args.location = None
        args.start_date = None
        args.shorts_per_day = None
        args.shorts_target = None
        args.shorts_gap = None
        args.day_gpx = None
        args.camera_utc_offset_hours = None
        args.no_tmux = True
        args.no_review = True
        args.preview = False
        args.fresh = False
        args.skip_upload = True
        args.skip_csv = True
        args.skip_validation = True
        args.phase = None
        args.youtube_upload = False
        args.keep_intermediates = False
        args.keep_masters = False
        args.force_sidecar_reset = None
        args.allow_seo_degraded = None
        args.dry_run = False
        args.force = True
        args.skip_invariants = False
        args.no_audio = False

        from gomoveshift.config import Config as _Config

        config = _Config()

        with (
            patch("gomoveshift.cli.handlers.drive.is_in_tmux", return_value=True),
            patch("gomoveshift.cli.handlers.drive.configure_logging") as mock_log,
            patch(
                "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
            ) as mock_orch_cls,
        ):
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.drive_plan = None
            mock_result.errors = []
            mock_orch_cls.return_value.run.return_value = mock_result

            from gomoveshift.cli.handlers.drive import cmd_drive

            cmd_drive(args, config)

            mock_log.assert_called_once()
            call_kwargs = mock_log.call_args.kwargs
            log_file = call_kwargs.get("log_file")
            assert log_file is not None, "cmd_drive must call configure_logging with log_file"
            assert isinstance(log_file, Path)
            assert log_file.name == "drive.log", f"Expected drive.log, got {log_file.name!r}"


class TestStreamHandlerHasTimestamps:
    """Stream handler format must include timestamps."""

    def test_stream_handler_has_timestamps(self):
        """configure_logging sets a stream format that includes asctime."""
        # Reset root logger to a clean state
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        # Clear any prior basicConfig
        root.handlers = []

        try:
            from gomoveshift.cli.logging_setup import configure_logging

            configure_logging()

            # Find the StreamHandler
            stream_handlers = [
                h
                for h in root.handlers
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            ]
            assert stream_handlers, "No StreamHandler found"
            fmt = stream_handlers[0].formatter
            assert fmt is not None, "StreamHandler has no formatter"
            assert "asctime" in fmt._fmt, (
                f"StreamHandler format must include asctime, got: {fmt._fmt!r}"
            )
        finally:
            # Restore original state
            root.handlers = original_handlers
            root.level = original_level

    def test_stream_handler_uses_hms_datefmt(self):
        """Stream handler datefmt uses HH:MM:SS format."""
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        root.handlers = []

        try:
            from gomoveshift.cli.logging_setup import configure_logging

            configure_logging()

            stream_handlers = [
                h
                for h in root.handlers
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            ]
            assert stream_handlers
            fmt = stream_handlers[0].formatter
            assert fmt.datefmt == "%H:%M:%S", f"Expected datefmt '%H:%M:%S', got {fmt.datefmt!r}"
        finally:
            root.handlers = original_handlers
            root.level = original_level

    def test_stream_formatter_includes_level_token(self):
        """Stream formatter must include ``[%(levelname)s]`` + ``%(name)s``.

        Without a level token, operators can't tell WARN from INFO when tailing a 10-hour log.
        """
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        root.handlers = []

        try:
            from gomoveshift.cli.logging_setup import configure_logging

            configure_logging()

            stream_handlers = [
                h
                for h in root.handlers
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            ]
            assert stream_handlers
            fmt = stream_handlers[0].formatter
            assert "%(levelname)s" in fmt._fmt, (
                f"Stream formatter must include %(levelname)s, got: {fmt._fmt!r}"
            )
            assert "%(name)s" in fmt._fmt, (
                f"Stream formatter must include %(name)s, got: {fmt._fmt!r}"
            )
        finally:
            root.handlers = original_handlers
            root.level = original_level

    def test_file_handler_uses_append_mode(self, tmp_path):
        """File handler opens in append mode so resume runs don't overwrite."""
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        root.handlers = []

        try:
            log_file = tmp_path / "test.log"
            from gomoveshift.cli.logging_setup import configure_logging

            configure_logging(log_file=log_file)

            file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
            assert file_handlers, "No FileHandler found"
            assert file_handlers[0].mode == "a", (
                f"FileHandler must use append mode 'a', got {file_handlers[0].mode!r}"
            )
        finally:
            # Close file handlers before restoring
            for h in root.handlers:
                if isinstance(h, logging.FileHandler):
                    h.close()
            root.handlers = original_handlers
            root.level = original_level
