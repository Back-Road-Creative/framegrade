"""Tests for ``gomoveshift/cli/logging_setup.py`` internals.

Complements ``test_cli_logging.py`` (which covers the stream/file-handler
integration) by exercising the parts that suite leaves untested: the
``_JsonLogFormatter`` output shape + exception serialization, and the
``RotatingFileHandler`` size/backup configuration.
"""

import json
import logging
from logging.handlers import RotatingFileHandler

import pytest

from gomoveshift.cli.logging_setup import (
    _LOG_BACKUP_COUNT,
    _LOG_MAX_BYTES,
    _JsonLogFormatter,
    configure_logging,
)


@pytest.fixture
def clean_root():
    """Snapshot/restore the root logger so configure_logging is isolated."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers = []
    yield root
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
    root.handlers, root.level = saved_handlers, saved_level


def _record(msg="hello", level=logging.INFO, name="gms.test", exc_info=None):
    return logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=42, msg=msg, args=(), exc_info=exc_info
    )


class TestJsonLogFormatterShape:
    """``_JsonLogFormatter`` emits one JSON object with the documented keys."""

    def test_emits_valid_json_with_core_keys(self):
        payload = json.loads(_JsonLogFormatter().format(_record(msg="line one")))
        for key in ("timestamp", "level", "logger", "message", "module", "lineno"):
            assert key in payload, f"missing key {key!r} in {payload!r}"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "gms.test"
        assert payload["message"] == "line one"
        assert payload["lineno"] == 42
        assert "exception" not in payload

    def test_timestamp_is_iso8601_utc(self):
        from datetime import datetime

        payload = json.loads(_JsonLogFormatter().format(_record()))
        assert payload["timestamp"].endswith("+00:00")
        datetime.fromisoformat(payload["timestamp"])

    def test_message_interpolates_args(self):
        rec = logging.LogRecord(
            "gms.test", logging.WARNING, __file__, 1, "count=%d name=%s", (3, "kerry"), None
        )
        payload = json.loads(_JsonLogFormatter().format(rec))
        assert payload["message"] == "count=3 name=kerry"
        assert payload["level"] == "WARNING"

    def test_exception_serialized_as_traceback_string(self):
        import sys

        try:
            raise ValueError("explode")
        except ValueError:
            exc_info = sys.exc_info()
        payload = json.loads(_JsonLogFormatter().format(_record(exc_info=exc_info)))
        assert isinstance(payload["exception"], str)
        assert "ValueError: explode" in payload["exception"]
        assert "Traceback" in payload["exception"]


class TestRotatingHandlerConfig:
    """The file handler is a RotatingFileHandler bounded at ~250 MB."""

    def test_file_handler_is_rotating_with_bounded_limits(self, tmp_path, clean_root):
        # 50 MB × 5 backups bounds drive.log at ~250 MB.
        assert _LOG_MAX_BYTES == 50 * 1024 * 1024 and _LOG_BACKUP_COUNT == 5
        configure_logging(log_file=tmp_path / "drive.log")
        rotating = [h for h in clean_root.handlers if isinstance(h, RotatingFileHandler)]
        assert rotating, "configure_logging must add a RotatingFileHandler"
        assert rotating[0].maxBytes == _LOG_MAX_BYTES
        assert rotating[0].backupCount == _LOG_BACKUP_COUNT

    def test_json_log_attaches_json_formatter_only_to_file_handler(self, tmp_path, clean_root):
        configure_logging(log_file=tmp_path / "drive.log", json_log=True)
        rotating = [h for h in clean_root.handlers if isinstance(h, RotatingFileHandler)]
        assert rotating and isinstance(rotating[0].formatter, _JsonLogFormatter)
        # The stream handler stays human-readable (NOT JSON).
        stream = [
            h
            for h in clean_root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert stream and not isinstance(stream[0].formatter, _JsonLogFormatter)

    def test_json_log_writes_parseable_line(self, tmp_path, clean_root):
        log_file = tmp_path / "drive.log"
        configure_logging(log_file=log_file, json_log=True)
        logging.getLogger("gms.e2e").info("json roundtrip")
        for h in clean_root.handlers:
            h.flush()
        lines = [ln for ln in log_file.read_text().splitlines() if ln.strip()]
        payload = json.loads(lines[-1])
        assert payload["message"] == "json roundtrip"
        assert payload["logger"] == "gms.e2e"


class TestVerboseLevel:
    """``verbose`` toggles the root level between DEBUG and INFO."""

    @pytest.mark.parametrize("verbose,expected", [(True, logging.DEBUG), (False, logging.INFO)])
    def test_root_level(self, verbose, expected, clean_root):
        configure_logging(verbose=verbose)
        assert clean_root.level == expected
