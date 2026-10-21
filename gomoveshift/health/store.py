"""Incident rows: one JSON file per incident under the incidents dir.

Schema ``health-incident/1`` (``new_incident`` is the single writer of the common fields)."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "health-incident/1"
INCIDENTS_DIR_ENV = "GMS_HEALTH_INCIDENTS_DIR"
DATA_ROOT_ENV = "GMS_DATA_ROOT"


def utc_iso(epoch: Optional[float] = None) -> str:
    moment = (
        datetime.fromtimestamp(epoch, timezone.utc)
        if epoch is not None
        else datetime.now(timezone.utc)
    )
    return moment.isoformat()


def default_data_root() -> Path:
    """``$GMS_DATA_ROOT``, else the main checkout (parent of git's common dir), else cwd."""
    env = os.environ.get(DATA_ROOT_ENV, "").strip()
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    try:
        out = subprocess.run(  # noqa: S603,S607 -- fixed argv
            ["git", "-C", str(here), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        return Path(out).parent
    except (OSError, subprocess.SubprocessError):
        return Path.cwd()


def default_incidents_dir(data_root: Path) -> Path:
    env = os.environ.get(INCIDENTS_DIR_ENV, "").strip()
    return Path(env) if env else data_root / "gms" / "incidents"


def code_sha(code_root: Path) -> Optional[str]:
    try:
        out = subprocess.run(  # noqa: S603,S607 -- fixed argv
            ["git", "-C", str(code_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "unnamed"


def new_incident(
    kind: str, key: str, filename: str, detected_at: str, **fields: Any
) -> Dict[str, Any]:
    """The common envelope. ``filename`` is the row's file name under the incidents dir."""
    return {
        "schema": SCHEMA,
        "kind": kind,
        "key": key,
        "file": filename,
        "detected_at": detected_at,
        "pushed": False,
        "pushed_at": None,
        "resolved_at": None,
        **fields,
    }


def load_all(incidents_dir: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Every readable incident row, plus a count of unreadable files (never raised)."""
    rows: List[Dict[str, Any]] = []
    bad = 0
    if not incidents_dir.is_dir():
        return rows, bad
    for path in sorted(incidents_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(row, dict) or row.get("schema") != SCHEMA or not row.get("file"):
                raise ValueError("not a health incident")
        except (OSError, ValueError):
            bad += 1
            continue
        rows.append(row)
    return rows, bad


def write_row(incidents_dir: Path, row: Dict[str, Any]) -> None:
    incidents_dir.mkdir(parents=True, exist_ok=True)
    path = incidents_dir / row["file"]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
