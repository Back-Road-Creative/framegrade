"""Unattended entry points must wire ``configure_logging``.

Two unattended entry points historically dropped INFO output because
``configure_logging`` was only called by ``cmd_drive`` / ``cmd_pipeline``:

* ``gomoveshift server`` — uvicorn-only logging, the rest of the GMS
  package logged into a void.
* ``gomoveshift publish-journal`` — cron-friendly replay mode, but no
  handler attached to the module-level logger.

These tests pin the fix: ``--log-file`` plumbs through both entry points,
and ``_JsonLogFormatter`` carries source-location fields for log
collectors that index on ``module`` / ``lineno``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

from gomoveshift.cli import _cmd_server
from gomoveshift.cli.handlers.publish_journal import cmd_publish_journal
from gomoveshift.cli.logging_setup import _JsonLogFormatter
from gomoveshift.config import Config


def test_server_log_file_writes(tmp_path, monkeypatch):
    """``_cmd_server`` configures logging so the startup banner hits the log file."""
    log_file = tmp_path / "server.log"

    # Replace uvicorn.run so the test doesn't actually start a server.
    uvicorn_stub = MagicMock()
    monkeypatch.setattr("uvicorn.run", uvicorn_stub)

    args = MagicMock()
    args.host = "127.0.0.1"
    args.port = 8765
    args.allow_remote = False
    args.log_file = log_file
    args.log_json = False
    args.verbose = False

    config = Config()

    try:
        rc = _cmd_server(args, config)
    finally:
        # Detach the rotating file handler so Windows/WSL doesn't hold the
        # file open past the test's tmp_path cleanup.
        for h in list(logging.getLogger().handlers):
            try:
                h.close()
            except Exception:
                pass

    assert rc == 0
    assert uvicorn_stub.called, "uvicorn.run stub should have been invoked"
    assert log_file.exists(), "configure_logging should have created the log file"
    contents = log_file.read_text()
    assert "Starting GoMoveShift web UI" in contents, (
        f"log file should contain the startup INFO line; got: {contents!r}"
    )


def test_publish_journal_log_file_writes(tmp_path):
    """``cmd_publish_journal`` configures logging before exiting on missing args."""
    log_file = tmp_path / "publish_journal.log"

    args = MagicMock()
    # Force an early-exit branch (no replay, missing required args) so the
    # handler returns 2 without touching the filesystem beyond the log.
    args.replay_pending = False
    args.state_path = None
    args.journey = None
    args.site_path = None
    args.log_file = log_file
    # Avoid getattr default surprises — set every flag the handler consults.
    args.apply = False
    args.allow_day_create = False

    rc = cmd_publish_journal(args, config=None)

    # Close handlers so the rotating file handler releases the file.
    for h in list(logging.getLogger().handlers):
        try:
            h.close()
        except Exception:
            pass

    assert rc == 2, "missing required args should exit 2"
    assert log_file.exists(), "configure_logging should have created the log file"
    # We don't assert specific content here — the handler logs nothing in
    # this early-exit branch. The file's existence proves configure_logging
    # ran with log_file set; the empty-file edge case is what the rotating
    # handler ships by default. Drop one INFO line to make the assertion
    # observable rather than presence-only.
    logging.getLogger("gomoveshift.cli.handlers.publish_journal").info("post-call probe")
    for h in list(logging.getLogger().handlers):
        try:
            h.flush()
        except Exception:
            pass
    assert "post-call probe" in log_file.read_text()


def test_json_log_includes_module_and_lineno():
    """``_JsonLogFormatter`` payload carries ``module`` and ``lineno`` for ingestion."""
    formatter = _JsonLogFormatter()
    record = logging.LogRecord(
        name="gomoveshift.testing",
        level=logging.INFO,
        pathname="/abs/path/to/module_xyz.py",
        lineno=4242,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    payload = json.loads(formatter.format(record))
    assert payload["module"] == "module_xyz", payload
    assert payload["lineno"] == 4242, payload
    # Existing fields preserved.
    assert payload["level"] == "INFO"
    assert payload["logger"] == "gomoveshift.testing"
    assert payload["message"] == "hello world"
