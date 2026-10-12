"""Next-7-days delivery digest, read straight from the queue DB.

Docs: ``docs/pipeline/149-dynamic-scheduling.md``.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

from .queue import open_queue_db

_HOUR = timedelta(hours=1)


def _slot_label(scheduled_at: datetime) -> str:
    end = scheduled_at + _HOUR
    return f"{scheduled_at:%H:%M}–{end:%H:%M}"


def _parse(scheduled_at: str) -> datetime:
    return datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))


def _reason_for(row: sqlite3.Row, youtube_at_by_key: Dict[Tuple[str, int], str]) -> str:
    """Best-effort label from what the queue schema actually stores.

    ``queue.py`` persists no "why was this the slot"
    column, so this reports the one thing derivable from the stored rows:
    whether a Facebook/Instagram row was moved off its short's YouTube
    slot (a stagger) or shares it. A generic "scheduled slot" covers every
    other case -- including the region-weight/recency ordering itself,
    which affects INSERTION order, not the stored ``scheduled_at``, and so
    leaves no trace in this table to report on.
    """
    key = (row["project_name"], row["short_index"])
    youtube_at = youtube_at_by_key.get(key)
    if (
        youtube_at is not None
        and row["platform"] in ("facebook", "instagram")
        and row["scheduled_at"] != youtube_at
    ):
        return "stagger"
    return "scheduled slot"


def next_week_report(conn: sqlite3.Connection, now: datetime) -> str:
    """One line per ``status='queued'`` delivery scheduled in the next 7
    days, grouped by day. ``'queued'`` is ``queue.enqueue``'s literal
    insert status -- ``TERMINAL_STATUSES = ("published", "failed_permanent",
    "cancelled")`` never reach here, so a delivery already drained never
    double-lists. Empty input -> empty string, so a digest consumer
    can drop the section instead of printing an empty header.
    """
    end = now + timedelta(days=7)
    rows = conn.execute(
        "SELECT project_name, short_index, platform, scheduled_at FROM deliveries "
        "WHERE status='queued' AND scheduled_at IS NOT NULL "
        "AND scheduled_at >= ? AND scheduled_at < ? "
        "ORDER BY scheduled_at, project_name, short_index, platform",
        (now.isoformat(), end.isoformat()),
    ).fetchall()
    if not rows:
        return ""

    youtube_at_by_key: Dict[Tuple[str, int], str] = {
        (row["project_name"], row["short_index"]): row["scheduled_at"]
        for row in rows
        if row["platform"] == "youtube"
    }

    by_day: "Dict[str, List[Tuple[datetime, sqlite3.Row]]]" = {}
    for row in rows:
        when = _parse(row["scheduled_at"])
        by_day.setdefault(when.date().isoformat(), []).append((when, row))

    lines: List[str] = []
    for day in sorted(by_day):
        lines.append(day)
        for when, row in by_day[day]:
            reason = _reason_for(row, youtube_at_by_key)
            lines.append(
                f"  {row['project_name']} short {row['short_index']} "
                f"{row['platform']} {_slot_label(when)} ({reason})"
            )
    return "\n".join(lines)


def main(config) -> str:
    """CLI/cron entrypoint: resolve the queue DB exactly like
    ``cli.handlers.metrics`` does (``config.seo.storage_path`` ->
    ``queue.open_queue_db``) and return the next-week report."""
    conn = open_queue_db(config.seo.storage_path)
    try:
        return next_week_report(conn, datetime.now(timezone.utc))
    finally:
        conn.close()
