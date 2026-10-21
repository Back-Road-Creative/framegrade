"""``run_check``: run the detection rules, reconcile incident rows, push each new one once."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import cadence, drives
from .store import code_sha, load_all, utc_iso, write_row

logger = logging.getLogger(__name__)

_EVENTS = {
    "drive_failed": "health_drive_failed",
    "lf_slot_unbooked": "health_lf_slot_unbooked",
    "lf_slot_uncovered": "health_lf_slot_uncovered",
    "queue_stalled": "health_queue_stalled",
}


def _line(row: Dict[str, Any]) -> str:
    if row["kind"] == "drive_failed":
        return drives.summary(row)
    if row["kind"] == "lf_slot_unbooked":
        return f"LF slot {row['slot']} ({row['region']}) is inside 48 h with nothing booked and a drive ready."
    if row["kind"] == "lf_slot_uncovered":
        nxt = row.get("next_drive")
        todo = (
            f"next drive to audit: {nxt}"
            if nxt
            else f"drive_backlog.yaml has no {row['region']} row left to audit"
        )
        return f"LF slot {row['slot']} ({row['region']}) has no ready drive; {todo}."
    return f"{row['due_deliveries']} publishing deliveries are due and nothing has touched the queue in {row['quiet_hours']} h."


def _next_drives(store: Any, regions: List[str], notes: List[str]) -> Dict[str, Optional[str]]:
    """Region -> the ``drive_backlog.yaml`` project to audit next (rule 2b names it)."""
    from seo_manager.scheduling import backlog

    from ..paths import drive_backlog_path

    path = drive_backlog_path()
    if not path.is_file():
        notes.append(f"{path} absent: rule 2b pages without naming a drive")
        return {}
    rows = backlog.load(path)
    return {
        region: (row.project if (row := backlog.next_row(rows, region, store)) else None)
        for region in regions
    }


def _cadence_rows(
    config: Any, ledger_path: Optional[Path], now: datetime, now_iso: str, notes: List[str]
):
    """Rules 2 and 4. ``found`` maps kind -> rows, only for a rule that actually ran."""
    found: Dict[str, List[Dict[str, Any]]] = {}
    storage = Path(config.seo.storage_path)
    if not storage.is_absolute():
        notes.append(
            f"seo.storage_path is relative ({config.seo.storage_path!r}); rules 2 and 4 not evaluated"
        )
        return found
    try:
        from seo_manager.scheduling.store import ScheduleStore

        from ..pipeline.content_ledger import ContentLedger
        from ..pipeline.schedule_report import build_schedule_report
        from ..pipeline.seo_slots import STORE_FILENAME, waypost_cadence

        store_path = storage / STORE_FILENAME
        if store_path.exists():
            cad = waypost_cadence(config)
            store = ScheduleStore(store_path, cad.tz, cad.regions)
            ready = {r: store.ready_depth(r) for r in sorted(cad.regions)}
            report = build_schedule_report(ContentLedger.load(ledger_path), config, after=now)
            found["lf_slot_unbooked"] = cadence.unbooked_slots(report, ready, now, now_iso)
            found["lf_slot_uncovered"] = cadence.uncovered_slots(
                report, ready, _next_drives(store, sorted(cad.regions), notes), now, now_iso
            )
        else:
            notes.append("schedule.sqlite3 absent: no readiness signal, rule 2 not evaluated")
    except Exception as exc:  # noqa: BLE001 -- a broken rule is reported, never fatal to the others
        notes.append(f"rule 2 not evaluated: {type(exc).__name__}: {exc}")
    db = storage / "social_queue.sqlite3"
    if db.exists():
        try:
            row = cadence.queue_stalled(db, now, now_iso)
            found["queue_stalled"] = [row] if row else []
        except Exception as exc:  # noqa: BLE001
            notes.append(f"rule 4 not evaluated: {type(exc).__name__}: {exc}")
    else:
        notes.append("social_queue.sqlite3 absent: rule 4 not evaluated")
    return found


def run_check(
    config: Any,
    *,
    incidents_dir: Path,
    jobs_dir: Path,
    code_root: Path,
    staging_sha: Optional[str] = None,
    ledger_path: Optional[Path] = None,
    now: Optional[datetime] = None,
    notify_fn: Optional[Callable[[str, Dict[str, Any], str], bool]] = None,
) -> Dict[str, Any]:
    """Returns ``{"incidents": open rows, "new": keys written, "pushed": keys pushed,
    "resolved": keys closed, "unreadable": int, "notes": [str]}``. Never raises on bad input."""
    if notify_fn is None:
        from ..console.notify import notify as notify_fn  # noqa: PLC0415
    now = now or datetime.now(timezone.utc)
    now_iso = utc_iso(now.timestamp())
    notes: List[str] = []
    existing, bad_rows = load_all(incidents_dir)
    by_key = {r["key"]: r for r in existing}
    sha = staging_sha or code_sha(code_root)

    drive_rows, succeeded, bad_jobs = drives.detect(jobs_dir, sha, now_iso)
    found = {"drive_failed": drive_rows, **_cadence_rows(config, ledger_path, now, now_iso, notes)}

    resolved: List[str] = []
    new: List[str] = []
    live_keys = {r["key"] for rows in found.values() for r in rows}
    for row in existing:
        if row.get("resolved_at"):
            continue
        kind = row.get("kind")
        gone = kind in found and row["key"] not in live_keys
        if kind == "drive_failed" and row.get("job_name") in succeeded:
            gone = True
        if gone:
            row["resolved_at"] = now_iso
            write_row(incidents_dir, row)
            resolved.append(row["key"])
    for rows in found.values():
        for row in rows:
            prior = by_key.get(row["key"])
            if prior is not None and not prior.get("resolved_at"):
                continue  # already on file; keeps its pushed state and resume_attempts
            write_row(incidents_dir, row)
            by_key[row["key"]] = row
            new.append(row["key"])

    pushed: List[str] = []
    open_rows = [r for r in by_key.values() if not r.get("resolved_at")]
    for row in open_rows:
        if row.get("pushed"):
            continue
        event = _EVENTS.get(row["kind"], "health_incident")
        if notify_fn(event, {"incident": row["key"]}, _line(row)) is False:
            continue  # attempted and failed: stay unrecorded, the next pass retries
        row["pushed"] = True
        row["pushed_at"] = now_iso
        write_row(incidents_dir, row)
        pushed.append(row["key"])
    return {
        "incidents": sorted(open_rows, key=lambda r: r["key"]),
        "new": new,
        "pushed": pushed,
        "resolved": resolved,
        "unreadable": bad_rows + bad_jobs,
        "notes": notes,
    }
