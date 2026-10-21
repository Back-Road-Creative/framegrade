"""Registry of backend runs, including ones the console did not launch.

Review-gate discovery used to read only the console job store's own argvs, so a drive started from
the CLI or by cron (no job row) was invisible: its gate never listed and never alerted. This module
keeps a ``backend_runs`` table (output dir -> provenance) in the job store's database, filled by
:func:`reconcile` from two sources:

* every job-store row, origin ``console`` (argv and start time known);
* a bounded scan of the configured output root for run dirs holding ``_gates``. Provenance comes
  from an optional ``run_origin.json`` sidecar in that dir (``origin`` / ``argv`` / ``started_at``);
  without one the origin is ``unknown`` -- never guessed.

Registration is idempotent (the output dir is the key), concrete provenance is never overwritten or
downgraded, and only ``unknown`` can be upgraded. Read-only against the pipeline: nothing here
writes into a run dir, and the scan never leaves the output root.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from gomoveshift import run_origin
from gomoveshift.console import drives
from gomoveshift.paths import gms_output_dir

ORIGINS = ("console", "cli", "cron", "unknown")
SIDECAR = run_origin.SIDECAR
#: The scan looks at the root, its children and grandchildren (a run dir or its project dirs).
_SCAN_DEPTH = 2

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS backend_runs (output_dir TEXT PRIMARY KEY, origin TEXT NOT NULL, "
    "argv TEXT, started_at TEXT, first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL)"
)


def _connect(store) -> sqlite3.Connection:
    conn = sqlite3.connect(store.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_dirs(root: Path, depth: int = _SCAN_DEPTH):
    yield root
    if depth <= 0:
        return
    try:
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return
    for child in children:
        yield from _candidate_dirs(child, depth - 1)


def _read_sidecar(directory: Path) -> dict:
    try:
        data = json.loads((directory / SIDECAR).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    origin = data.get("origin")
    argv = data.get("argv")
    started = data.get("started_at")
    return {
        "origin": origin if origin in ("cli", "cron", "console") else "unknown",
        "argv": argv if isinstance(argv, list) and all(isinstance(a, str) for a in argv) else None,
        "started_at": started if isinstance(started, str) else None,
    }


def _register(conn, output_dir: str, origin: str, argv, started_at: Optional[str]) -> None:
    now = _now()
    argv_json = json.dumps(argv) if argv is not None else None
    row = conn.execute(
        "SELECT origin FROM backend_runs WHERE output_dir = ?", (output_dir,)
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO backend_runs VALUES (?, ?, ?, ?, ?, ?)",
            (output_dir, origin, argv_json, started_at, now, now),
        )
    elif row["origin"] == "unknown" and origin != "unknown":
        conn.execute(
            "UPDATE backend_runs SET origin = ?, argv = ?, started_at = ?, last_seen_at = ? "
            "WHERE output_dir = ?",
            (origin, argv_json, started_at, now, output_dir),
        )
    else:
        conn.execute(
            "UPDATE backend_runs SET last_seen_at = ? WHERE output_dir = ?", (now, output_dir)
        )


def reconcile(store, roots: "Optional[list[Path]]" = None) -> None:
    """Fold the job store and the output-root scan into the registry (idempotent)."""
    scan_roots = [Path(r) for r in roots] if roots is not None else [gms_output_dir()]
    conn = _connect(store)
    try:
        for job in store.list():
            for output_dir in drives.output_dirs_from_job_argvs([job["argv"]]):
                _register(conn, output_dir, "console", job["argv"], job.get("created_at"))
        for root in scan_roots:
            if not root.is_dir():
                continue
            for directory in _candidate_dirs(root):
                if not (directory / drives.GATE_ANSWERS_SUBDIR).is_dir():
                    continue
                found = _read_sidecar(directory)
                _register(
                    conn,
                    str(directory),
                    found.get("origin", "unknown"),
                    found.get("argv"),
                    found.get("started_at"),
                )
        conn.commit()
    finally:
        conn.close()


def list_runs(store) -> "list[dict]":
    """Registry rows, oldest first, ``argv`` decoded."""
    conn = _connect(store)
    try:
        rows = conn.execute("SELECT * FROM backend_runs ORDER BY first_seen_at, output_dir")
        return [
            {**dict(r), "argv": json.loads(r["argv"]) if r["argv"] is not None else None}
            for r in rows
        ]
    finally:
        conn.close()


def known_output_dirs(store, roots: "Optional[list[Path]]" = None) -> "list[str]":
    """Every output dir the console may look at: job-store dirs first, then registered runs."""
    reconcile(store, roots)
    dirs = drives.output_dirs_from_job_argvs([job["argv"] for job in store.list()])
    for row in list_runs(store):
        if row["output_dir"] not in dirs:
            dirs.append(row["output_dir"])
    return dirs


def origins_by_dir(store) -> "dict[str, str]":
    return {row["output_dir"]: row["origin"] for row in list_runs(store)}
