"""Provenance for a backend run: who started it, recorded where the run starts.

The console's backend-run registry reads ``run_origin.json`` from a run dir. This is the writer the
CLI entry points call, so a CLI or cron drive is recorded as such instead of showing up as
``unknown``. ``GMS_RUN_ORIGIN`` names the launcher (``cron`` for scheduled jobs, ``console`` for the
console's own child processes); absent or unrecognised, the run is ``cli``.

Best effort and write-once: an unwritable dir never breaks a run, and a resumed run keeps the origin
it started with.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SIDECAR = "run_origin.json"
ORIGIN_ENV = "GMS_RUN_ORIGIN"
ORIGINS = ("cli", "cron", "console")
#: Set by ``health backlog`` on the drive it starts; ``drive`` refuses a real render without it.
LAUNCHED_BY_ENV = "GMS_LAUNCHED_BY"
BACKLOG_LAUNCHER = "health-backlog"
#: Every tracked launcher that may start a render: the backlog, a heal resume, the console's job
#: store and the host worker it spools to. A hand-typed CLI run carries none of them.
LAUNCHERS = (BACKLOG_LAUNCHER, "health-heal", "console", "hostworker")


def _command(argv: Optional[list]) -> list:
    """Program basename plus the subcommand, never the operands: host paths and secret-like
    arguments stay out of the run dir."""
    if not argv:
        return []
    kept = [os.path.basename(str(argv[0]))]
    if len(argv) > 1 and not str(argv[1]).startswith("-"):
        kept.append(str(argv[1]))
    return kept


def write_sidecar(directory: Path, argv: Optional[list] = None) -> None:
    target = Path(directory) / SIDECAR
    if target.exists():
        return
    origin = os.environ.get(ORIGIN_ENV, "")
    payload = {
        "origin": origin if origin in ORIGINS else "cli",
        "argv": _command(argv),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        target.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        logger.warning("run origin not recorded in %s: %s", directory, exc)
