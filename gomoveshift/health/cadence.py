"""Rule 2 (an LF slot inside 48 h with nothing booked while a ready drive exists), rule 2b (a
region's next free LF slot inside a week with no ready drive at all) and rule 4 (the dispatch
queue has due work and nothing has touched it in 24 h)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..publishing.queue import TERMINAL_STATUSES
from .store import new_incident, safe_name

SLOT_HORIZON_H = 48
# Rule 2b looks further ahead than rule 2: an empty region needs a drive audited, rendered and
# accepted, which can take days, not the hours a booking does.
COVERAGE_HORIZON_H = 7 * 24
DISPATCH_QUIET_H = 24


def _aware(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def unbooked_slots(
    report: Dict[str, Any], ready_by_region: Dict[str, int], now: datetime, now_iso: str
) -> List[Dict[str, Any]]:
    """Rows of ``build_schedule_report``'s ``next_available`` (each is a FREE slot: the booked
    ones are excluded by the scheduler) that start within ``SLOT_HORIZON_H`` of ``now``, for a
    region with at least one ready drive (``ready_drives`` status ``ready``)."""
    out: List[Dict[str, Any]] = []
    horizon = now + timedelta(hours=SLOT_HORIZON_H)
    for row in report.get("next_available", []):
        slot = _aware(row.get("slot"))
        region = row.get("region")
        if slot is None or not (now <= slot <= horizon) or not ready_by_region.get(region, 0):
            continue
        tag = slot.strftime("%Y%m%dT%H%M")
        out.append(
            new_incident(
                "lf_slot_unbooked",
                f"lf_slot_unbooked:{region}:{slot.isoformat()}",
                f"slot_{safe_name(str(region))}_{tag}.json",
                now_iso,
                region=region,
                slot=slot.isoformat(),
                ready_drives=ready_by_region[region],
            )
        )
    return out


def uncovered_slots(
    report: Dict[str, Any],
    ready_by_region: Dict[str, int],
    next_drive_by_region: Dict[str, Optional[str]],
    now: datetime,
    now_iso: str,
) -> List[Dict[str, Any]]:
    """One row per region with NO ready drive whose earliest free slot in ``next_available``
    starts within ``COVERAGE_HORIZON_H``. ``next_drive`` is the ``drive_backlog.yaml`` row to
    audit next (``None`` when the manifest has none left for that region)."""
    earliest: Dict[str, datetime] = {}
    horizon = now + timedelta(hours=COVERAGE_HORIZON_H)
    for row in report.get("next_available", []):
        slot = _aware(row.get("slot"))
        region = row.get("region")
        if slot is None or not (now <= slot <= horizon) or ready_by_region.get(region, 0):
            continue
        if region not in earliest or slot < earliest[region]:
            earliest[region] = slot
    out: List[Dict[str, Any]] = []
    for region, slot in sorted(earliest.items()):
        tag = slot.strftime("%Y%m%dT%H%M")
        out.append(
            new_incident(
                "lf_slot_uncovered",
                f"lf_slot_uncovered:{region}:{slot.isoformat()}",
                f"uncovered_{safe_name(str(region))}_{tag}.json",
                now_iso,
                region=region,
                slot=slot.isoformat(),
                next_drive=next_drive_by_region.get(region),
            )
        )
    return out


def queue_stalled(db_path: Path, now: datetime, now_iso: str) -> Optional[Dict[str, Any]]:
    """An incident when the queue holds DUE non-terminal work (non-shadow, ``scheduled_at`` null
    or past, platform not paused) yet no delivery row was updated in ``DISPATCH_QUIET_H``.
    Every dispatch attempt rewrites ``updated_at`` (lease, transition), so a quiet queue with due
    work means the drain is not running. Opened read-only: detection never creates or migrates."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        marks = ",".join("?" * len(TERMINAL_STATUSES))
        paused = {r[0] for r in conn.execute("SELECT platform FROM platform_pauses")}
        due = [
            r
            for r in conn.execute(
                f"SELECT platform, scheduled_at FROM deliveries WHERE shadow = 0 AND status NOT IN ({marks})",  # noqa: S608
                TERMINAL_STATUSES,
            )
            if r[0] not in paused and (_aware(r[1]) is None or _aware(r[1]) <= now)
        ]
        if not due:
            return None
        last = conn.execute("SELECT MAX(updated_at) FROM deliveries").fetchone()[0]
        touched = _aware(last)
        if touched is not None and touched >= now - timedelta(hours=DISPATCH_QUIET_H):
            return None
    finally:
        conn.close()
    return new_incident(
        "queue_stalled",
        "queue_stalled",
        "queue_stalled.json",
        now_iso,
        due_deliveries=len(due),
        last_updated_at=last,
        quiet_hours=DISPATCH_QUIET_H,
    )
