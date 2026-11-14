"""One read-only snapshot of the live state: jobs, incidents, queue, runway, backlog, machine.

``build_snapshot`` is the single reader behind ``gomoveshift status`` and the console's status
page. Every section carries ``as_of`` and ``source``; a store that cannot be read carries
``unavailable: <why>`` and none of its data keys, so it never reads as empty or healthy.
Nothing here writes, launches or repairs: no ``health check``, the queue is opened read-only,
and a missing file is never created. Docs: ``docs/pipeline/179-status.md``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..health import drives
from ..health.store import default_data_root, default_incidents_dir, load_all, utc_iso

SCHEMA = "gms-status/1"
JOBS_DIR_ENV = "GMS_JOBS_DIR"
CRON_LOGS_DIR_ENV = "GMS_CRON_LOGS_DIR"
#: Where to READ the shared GPU lock from. Separate from ``GMS_GPU_LOCK_PATH`` (the path a
#: process takes the lock at) so a container can be shown the host's lock without holding it.
GPU_LOCK_ENV = "GMS_STATUS_GPU_LOCK_PATH"
#: ``label=path,label=path``: the volumes whose free space is reported.
VOLUMES_ENV = "GMS_STATUS_VOLUMES"

RECENT_JOBS = 10
NEXT_DELIVERIES = 5
BACKLOG_PROJECTS = 20
_CRON_TAIL_BYTES = 200_000
_CRON_LINE_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T[\d:]{8})\]\s?(.*)$")
RUNWAY_SOURCE = "gomoveshift schedule --json"


def _env_path(name: str) -> Optional[Path]:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else None


def _iso(epoch: Any) -> Optional[str]:
    try:
        return utc_iso(float(epoch)) if epoch else None
    except (TypeError, ValueError):
        return None


def _need_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"no directory {path}")


# ── jobs ────────────────────────────────────────────────────────────────


def jobs_dir() -> Path:
    return _env_path(JOBS_DIR_ENV) or default_data_root() / "logs" / "jobs"


def _job_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    result = rec.get("result") or {}
    log = Path(str(rec.get(drives._DIR_KEY))) / "output.log"
    parsed = drives.parse_log(drives._tail(log) or [])
    return {
        "name": rec.get("name"),
        "run_id": rec.get("run_id"),
        "state": result.get("state") or "running",
        "exit_code": result.get("exit_code"),
        "created_at": _iso(rec.get("created_at")),
        "heartbeat_at": _iso(rec.get("heartbeat_at")),
        "ended_at": _iso(result.get("ended_at")),
        "drive": parsed["drive"],
        "phase": parsed["phase"],
        "error": parsed["error"],
        "log": str(log),
    }


def _jobs() -> Dict[str, Any]:
    path = jobs_dir()
    _need_dir(path)
    records, unreadable = drives.load_records(path)
    records.reverse()  # newest first
    running = [r for r in records if not (r.get("result") or {}).get("state")]
    ended = [r for r in records if (r.get("result") or {}).get("state")]
    counts = Counter((r.get("result") or {}).get("state") or "running" for r in records)
    return {
        "source": str(path),
        "running": [_job_row(r) for r in running],
        "recent": [_job_row(r) for r in ended[:RECENT_JOBS]],
        "counts": dict(counts),
        "unreadable": unreadable,
    }


# ── incidents ───────────────────────────────────────────────────────────


def _incidents() -> Dict[str, Any]:
    path = default_incidents_dir(default_data_root())
    _need_dir(path)
    rows, unreadable = load_all(path)
    open_rows = [r for r in rows if not r.get("resolved_at")]
    return {
        "source": str(path),
        "open": open_rows,
        "resolved": len(rows) - len(open_rows),
        "unreadable": unreadable,
    }


# ── publishing queue ────────────────────────────────────────────────────


def _open_read_only(db_path: Path) -> sqlite3.Connection:
    """``mode=ro``; on a location that cannot hold the write-ahead-log index (a read-only
    mount), ``immutable=1``, which reads the file as it stands."""
    last: Optional[sqlite3.Error] = None
    for flags in ("mode=ro", "mode=ro&immutable=1"):
        conn = sqlite3.connect(f"{db_path.as_uri()}?{flags}", uri=True)
        try:
            conn.execute("SELECT 1 FROM deliveries LIMIT 1").fetchall()
        except sqlite3.Error as exc:
            conn.close()
            last = exc
            continue
        conn.row_factory = sqlite3.Row
        return conn
    raise last if last else sqlite3.OperationalError(f"cannot open {db_path}")


def _delivery(row: sqlite3.Row) -> Dict[str, Any]:
    keys = ("id", "platform", "project_name", "short_index", "status", "scheduled_at")
    return {key: row[key] for key in keys}


def _queue(config: Any, now: str) -> Dict[str, Any]:
    from ..cli.handlers.pub_status import _manual_publish_overdue
    from ..publishing.queue import STATUS_AWAITING_MANUAL_PUBLISH, TERMINAL_STATUSES

    db_path = Path(config.seo.storage_path) / "social_queue.sqlite3"
    if not db_path.is_file():
        raise FileNotFoundError(f"no file {db_path}")
    conn = _open_read_only(db_path)
    try:
        counts: Dict[str, Dict[str, int]] = {}
        for row in conn.execute(
            "SELECT platform, status, COUNT(*) AS n FROM deliveries "
            "GROUP BY platform, status ORDER BY platform, status"
        ):
            counts.setdefault(row["platform"], {})[row["status"]] = row["n"]
        parked = conn.execute(
            "SELECT * FROM deliveries WHERE status=? ORDER BY scheduled_at, id",
            (STATUS_AWAITING_MANUAL_PUBLISH,),
        ).fetchall()
        marks = ", ".join("?" for _ in TERMINAL_STATUSES)
        upcoming = conn.execute(
            f"SELECT * FROM deliveries WHERE status NOT IN ({marks}) AND status != ? "  # noqa: S608
            "AND scheduled_at >= ? ORDER BY scheduled_at, id LIMIT ?",
            (*TERMINAL_STATUSES, STATUS_AWAITING_MANUAL_PUBLISH, now, NEXT_DELIVERIES),
        ).fetchall()
    finally:
        conn.close()
    return {
        "source": str(db_path),
        "counts": counts,
        "overdue_manual": [_delivery(r) for r in parked if _manual_publish_overdue(r, now)],
        "next": [_delivery(r) for r in upcoming],
    }


# ── runway ──────────────────────────────────────────────────────────────


def _runway_report(config: Any) -> Dict[str, Any]:
    """The same report ``gomoveshift schedule --json`` prints."""
    from ..pipeline.content_ledger import ContentLedger, _default_ledger_path
    from ..pipeline.schedule_report import build_schedule_report
    from ..publishing.title_kit import build_title_kit

    ledger_path = _default_ledger_path()
    if not ledger_path.is_file():  # ``load`` would hand back an empty ledger: "nothing booked"
        raise FileNotFoundError(f"no file {ledger_path}")
    return build_schedule_report(
        ContentLedger.load(ledger_path),
        config,
        per_location_cap=build_title_kit().per_location_cap(),
    )


def _runway(config: Any) -> Dict[str, Any]:
    report = dict(_runway_report(config))
    report.pop("as_of", None)  # the section's own as_of is the one clock
    return {"source": RUNWAY_SOURCE, **report}


# ── backlog ─────────────────────────────────────────────────────────────


def _backlog() -> Dict[str, Any]:
    import yaml

    from ..paths import drive_backlog_path

    path = drive_backlog_path()
    if not path.is_file():
        raise FileNotFoundError(f"no file {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of rows")
    rows = [r for r in raw if isinstance(r, dict)]
    by_region = Counter(str(r.get("region")) for r in rows)
    return {
        "source": str(path),
        "rows": len(rows),
        "by_region": dict(sorted(by_region.items())),
        "projects": [
            {"project": r.get("project"), "region": r.get("region")}
            for r in rows[:BACKLOG_PROJECTS]
        ],
    }


# ── machine ─────────────────────────────────────────────────────────────


def _volume_specs() -> List[Tuple[str, Path]]:
    raw = os.environ.get(VOLUMES_ENV, "").strip()
    if not raw:
        return [("data", default_data_root())]
    specs = []
    for item in raw.split(","):
        label, sep, path = item.strip().partition("=")
        if sep and label and path:
            specs.append((label, Path(path)))
    return specs


def _volumes() -> List[Dict[str, Any]]:
    out = []
    for label, path in _volume_specs():
        row: Dict[str, Any] = {"label": label, "path": str(path)}
        try:
            usage = shutil.disk_usage(path)
            row.update(free_bytes=usage.free, total_bytes=usage.total)
        except OSError as exc:
            row["unavailable"] = f"{type(exc).__name__}: {exc}"
        out.append(row)
    return out


def _gpu() -> Dict[str, Any]:
    from ..pipeline.orchestrator_lock import gpu_lock_path

    path = _env_path(GPU_LOCK_ENV) or gpu_lock_path()
    if path is None:
        return {"unavailable": "the shared GPU lock is disabled (GMS_GPU_LOCK_PATH is empty)"}
    try:
        holder = json.loads(path.read_text(encoding="utf-8") or "{}")
        if not isinstance(holder, dict):
            raise ValueError("holder is not an object")
    except (OSError, ValueError) as exc:
        return {"source": str(path), "unavailable": f"{type(exc).__name__}: {exc}"}
    return {
        "source": str(path),
        "held": bool(holder.get("owner")),
        "owner": holder.get("owner"),
        "pid": holder.get("pid"),
        "acquired": holder.get("acquired"),
    }


def _cron_row(path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {"name": path.name[: -len("-cron.log")], "log": str(path)}
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - _CRON_TAIL_BYTES))
            lines = fh.read().decode("utf-8", "replace").splitlines()
        row["modified_at"] = utc_iso(path.stat().st_mtime)
    except OSError as exc:
        row["unavailable"] = f"{type(exc).__name__}: {exc}"
        return row
    stamped = [(m.group(1), m.group(2)) for ln in lines if (m := _CRON_LINE_RE.match(ln))]
    failed = [(at, text) for at, text in stamped if "FAILED" in text]
    row["last_at"], row["last_line"] = stamped[-1] if stamped else (None, None)
    row["last_failure"] = {"at": failed[-1][0], "line": failed[-1][1]} if failed else None
    return row


def _cron() -> Any:
    path = _env_path(CRON_LOGS_DIR_ENV) or default_data_root() / "logs"
    if not path.is_dir():
        return {"source": str(path), "unavailable": f"no directory {path}"}
    return [_cron_row(p) for p in sorted(path.glob("gms-*-cron.log"))]


def _machine() -> Dict[str, Any]:
    return {"source": "this host", "volumes": _volumes(), "gpu": _gpu(), "cron": _cron()}


# ── assembly ────────────────────────────────────────────────────────────


def _section(
    now: str, fallback_source: str, reader: Callable[[], Dict[str, Any]]
) -> Dict[str, Any]:
    try:
        body = reader()
    except Exception as exc:  # noqa: BLE001 -- one unreadable store must not blank the page
        body = {"source": fallback_source, "unavailable": f"{type(exc).__name__}: {exc}"}
    return {"as_of": now, **body}


def build_snapshot(config: Any, *, now: Optional[str] = None) -> Dict[str, Any]:
    """The whole snapshot. ``now`` is an ISO timestamp (UTC), fixed by tests."""
    at = now or utc_iso()
    return {
        "schema": SCHEMA,
        "as_of": at,
        "jobs": _section(at, "job-runner records", _jobs),
        "incidents": _section(at, "health incidents", _incidents),
        "queue": _section(at, "publishing queue", lambda: _queue(config, at)),
        "runway": _section(at, RUNWAY_SOURCE, lambda: _runway(config)),
        "backlog": _section(at, "drive backlog", _backlog),
        "machine": _section(at, "this host", _machine),
    }
