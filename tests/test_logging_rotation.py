"""drive.log is bounded by RotatingFileHandler and can optionally
emit JSON lines for log shippers.

Covers:
  - File handler is a ``RotatingFileHandler`` with 50 MB × 5 backups.
  - ``json_log=True`` writes parseable JSON lines with the required keys.
  - ``json_log=False`` keeps the existing human-readable formatter.
  - The stream handler is untouched by ``json_log`` (tail -f stays
    human-readable even when the on-disk log is JSON).
  - The top-level CLI parser exposes ``--log-json`` and it threads
    through to ``configure_logging``.
"""

import json
import logging
from logging.handlers import RotatingFileHandler
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def clean_root_logger():
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    root.handlers = []
    try:
        yield root
    finally:
        for h in root.handlers:
            if isinstance(h, logging.FileHandler):
                h.close()
        root.handlers = original_handlers
        root.level = original_level


def test_file_handler_is_rotating_with_50mb_and_5_backups(clean_root_logger, tmp_path):
    """drive.log must rotate at 50 MB across 5 backups."""
    from gomoveshift.cli.logging_setup import configure_logging

    log_file = tmp_path / "drive.log"
    configure_logging(log_file=log_file)

    file_handlers = [h for h in clean_root_logger.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers, "No FileHandler attached"
    fh = file_handlers[0]
    assert isinstance(fh, RotatingFileHandler), (
        f"Expected RotatingFileHandler, got {type(fh).__name__}"
    )
    assert fh.maxBytes == 50 * 1024 * 1024, f"Expected maxBytes=50MB, got {fh.maxBytes}"
    assert fh.backupCount == 5, f"Expected backupCount=5, got {fh.backupCount}"


def test_json_log_true_produces_parseable_json_per_line(clean_root_logger, tmp_path):
    """json_log=True writes one JSON object per record with required keys."""
    from gomoveshift.cli.logging_setup import configure_logging

    log_file = tmp_path / "drive.log"
    configure_logging(log_file=log_file, json_log=True)

    logging.getLogger("h1.test").info("hello world")
    logging.getLogger("h1.test").warning("warn line")
    for h in clean_root_logger.handlers:
        h.flush()

    lines = [ln for ln in log_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}: {lines!r}"
    for line in lines:
        payload = json.loads(line)  # raises if not parseable
        assert set(payload.keys()) >= {"timestamp", "level", "logger", "message"}, (
            f"Missing required keys in {payload!r}"
        )
        assert payload["logger"] == "h1.test"
    assert json.loads(lines[0])["level"] == "INFO"
    assert json.loads(lines[0])["message"] == "hello world"
    assert json.loads(lines[1])["level"] == "WARNING"


def test_json_log_true_includes_exception_when_exc_info(clean_root_logger, tmp_path):
    """exc_info=True must surface a formatted traceback under ``exception``."""
    from gomoveshift.cli.logging_setup import configure_logging

    log_file = tmp_path / "drive.log"
    configure_logging(log_file=log_file, json_log=True)

    try:
        raise ValueError("kaboom")
    except ValueError:
        logging.getLogger("h1.exc").error("bad thing", exc_info=True)
    for h in clean_root_logger.handlers:
        h.flush()

    lines = [ln for ln in log_file.read_text().splitlines() if ln.strip()]
    assert lines, "no log lines written"
    payload = json.loads(lines[-1])
    assert "exception" in payload, f"Expected exception key, got {payload!r}"
    assert "ValueError" in payload["exception"]
    assert "kaboom" in payload["exception"]


def test_json_log_false_keeps_human_readable_formatter(clean_root_logger, tmp_path):
    """Default json_log=False keeps the legacy ``asctime [LEVEL] msg`` format."""
    from gomoveshift.cli.logging_setup import configure_logging

    log_file = tmp_path / "drive.log"
    configure_logging(log_file=log_file, json_log=False)

    file_handlers = [h for h in clean_root_logger.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers
    fmt = file_handlers[0].formatter
    assert fmt is not None
    # Legacy human-readable format includes the level token literal.
    assert "%(levelname)s" in fmt._fmt, f"Got unexpected file format: {fmt._fmt!r}"
    assert fmt.datefmt == "%H:%M:%S"

    # And lines on disk are NOT JSON.
    logging.getLogger("h1.text").info("plain")
    for h in clean_root_logger.handlers:
        h.flush()
    line = log_file.read_text().splitlines()[0]
    with pytest.raises(json.JSONDecodeError):
        json.loads(line)


def test_stream_handler_unaffected_by_json_log(clean_root_logger, tmp_path):
    """json_log only swaps the file formatter; the stream handler stays human-readable."""
    from gomoveshift.cli.logging_setup import configure_logging

    log_file = tmp_path / "drive.log"
    configure_logging(log_file=log_file, json_log=True)

    stream_handlers = [
        h
        for h in clean_root_logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert stream_handlers, "No StreamHandler attached"
    fmt = stream_handlers[0].formatter
    assert fmt is not None
    # The human-readable stream format must still carry the asctime + level
    # tokens — operators tailing the run should not see raw JSON on stdout.
    assert "asctime" in fmt._fmt
    assert "%(levelname)s" in fmt._fmt
    assert "%(name)s" in fmt._fmt


def test_cli_parses_log_json_flag():
    """``--log-json`` is a top-level flag and defaults False."""
    from gomoveshift.cli.parser import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["--log-json", "cleanup"])
    assert args.log_json is True

    args_default = parser.parse_args(["cleanup"])
    assert args_default.log_json is False


def test_cmd_drive_threads_log_json_to_configure_logging(tmp_path):
    """cmd_drive forwards args.log_json into configure_logging."""
    from pathlib import Path

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
    args.log_json = True

    from gomoveshift.config import Config as _Config

    config = _Config()

    with (
        patch("gomoveshift.cli.handlers.drive.is_in_tmux", return_value=True),
        patch("gomoveshift.cli.handlers.drive.configure_logging") as mock_log,
        patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator") as mock_orch_cls,
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
        assert call_kwargs.get("json_log") is True
        assert isinstance(call_kwargs.get("log_file"), Path)
