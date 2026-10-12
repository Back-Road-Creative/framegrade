"""Read worker outcomes only from this console's configured private project DB."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from . import settings

OUTCOMES = {
    "complete": ("done", 0),
    "failed": ("failed", 1),
    "partial": ("failed", 1),
    "blocked": ("failed", 1),
    "paused": ("paused", 2),
    "waiting": ("waiting", 2),
    "cancelled": ("cancelled", 1),
}


def project_outcome(row: dict) -> dict | None:
    """Return durable outcome evidence, never read a workspace chosen by argv.

    This attests to worker-recorded completion, not that images still exist today.
    Ambiguous commands, foreign brands and missing/malformed state remain unknown.
    """
    try:
        argv = row.get("argv")
        prefix = [sys.executable, "-m", "gomoveshift", "photos", "night"]
        if not isinstance(argv, list) or len(argv) != 11 or argv[:5] != prefix:
            return None
        pairs = list(zip(argv[5::2], argv[6::2]))
        flags = dict(pairs)
        if len(flags) != 3 or set(flags) != {"--project", "--workspace", "--brand"}:
            return None
        brand = settings.current_brand()
        if row.get("brand") != brand or flags["--brand"] != brand:
            return None
        project = flags["--project"]
        if not isinstance(project, str) or not re.fullmatch(r"[0-9a-f]{64}", project):
            return None
        workspace = settings.console_data_dir() / "night-projects"
        if Path(flags["--workspace"]).absolute() != workspace.absolute():
            return None
        database = workspace / "projects.sqlite3"
        if any(path.is_symlink() for path in (database, *database.parents)):
            return None
        with closing(
            sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
        ) as conn:
            record = conn.execute(
                "SELECT data FROM night_projects WHERE id=? AND brand=? AND length(data)<=1048576",
                (project, brand),
            ).fetchone()
        if not record:
            return None
        data = json.loads(record[0])
        outcome = data.get("status")
        if data.get("id") != project or data.get("brand") != brand or outcome not in OUTCOMES:
            return None
        if outcome == "complete":
            units = data.get("tasks") if data.get("kind") == "night_products" else data.get("items")
            if not units or any(item.get("status") != "complete" for item in units):
                return None
        status, rc = OUTCOMES[outcome]
        return {"status": status, "rc": rc, "outcome_evidence": f"night_project:{outcome}"}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, sqlite3.Error):
        return None
