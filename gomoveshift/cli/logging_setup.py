"""Centralized logging configuration for the GoMoveShift CLI.

Replaces the 7 scattered ``logging.basicConfig`` calls that existed in the
monolithic ``cli.py`` with a single ``configure_logging()`` entry point.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 50 MB per file × 5 rotated backups bounds drive.log at ~250 MB
# across a multi-day batch run. Was unbounded FileHandler before.
_LOG_MAX_BYTES = 50 * 1024 * 1024
_LOG_BACKUP_COUNT = 5


class _JsonLogFormatter(logging.Formatter):
    """Minimal JSON line formatter for the file handler.

    Emits one JSON object per record with keys ``timestamp`` (ISO-8601
    UTC), ``level``, ``logger``, ``message``. Includes ``exception`` as
    a formatted traceback string when ``record.exc_info`` is set. Hand-
    rolled to avoid pulling in ``python-json-logger`` (no new install
    surface for an ergonomic feature).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "lineno": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(
    *,
    log_file: Path | None = None,
    verbose: bool = False,
    json_log: bool = False,
) -> None:
    """Set up logging for CLI commands.

    Parameters
    ----------
    log_file:
        Optional path to a log file.  When provided a rotating file
        handler is added alongside the default stream handler.
    verbose:
        When *True* the root logger level is set to ``DEBUG``.  Otherwise
        ``INFO`` is used.
    json_log:
        When *True* the file handler emits one JSON object per line
        (keys: ``timestamp``, ``level``, ``logger``, ``message``, and
        ``exception`` when present). The stream handler is untouched so
        ``tail -f`` stays human-readable.
    """
    level = logging.DEBUG if verbose else logging.INFO

    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler(sys.stdout)
    # Include level + logger name.
    # Without the level token, operators can't distinguish WARN from
    # INFO when tailing a 10-hour Phase 3 log.
    stream_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handlers.append(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # RotatingFileHandler caps drive.log growth. Default
        # ``mode='a'`` (append) preserves resume semantics.
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
        )
        if json_log:
            file_handler.setFormatter(_JsonLogFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
            )
        handlers.append(file_handler)

    # ``force=True`` evicts handlers from prior calls (resume flows,
    # library callers). Without it ``basicConfig`` no-ops on re-invocation
    # and ``drive.log`` stays empty when ``cmd_drive`` is called twice in
    # the same process.
    logging.basicConfig(level=level, handlers=handlers, force=True)
