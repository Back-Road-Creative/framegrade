"""Console overview: the first screen names what needs an operator
instead of listing where things live.

Read-only, and it adds no new read path -- every source here is an existing
console module's own accessor (``jobs.list_jobs``, ``pubqueue.queue_rows``).
This file only decides which of those reads belong on the first screen, and
what an empty or unreadable one says.

Two rules apply to every section below:

* An empty section says so in plain words (``items == []`` and
  ``unreadable == []`` together mean "genuinely nothing here"). Never a
  silent gap.
* Unknown is not zero. A source that raised is named in ``unreadable``
  rather than folded into an empty ``items`` list -- an empty list there
  would read as "nothing needs you", which is not what "could not read it"
  means.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..publishing.handlers import STATUS_AWAITING_MANUAL_PUBLISH
from ..publishing.queue import STATUS_MISSING_REMOTE
from . import jobs, pubqueue

#: Job statuses worth surfacing on "Needs attention".
_ATTENTION_JOB_STATUSES = ("failed", "interrupted")

#: Job statuses worth surfacing on "Work in progress".
_IN_PROGRESS_JOB_STATUSES = ("queued", "running")

#: Delivery statuses worth surfacing on "Needs attention" -- the same
#: vocabulary ``notify.check_queue_regressions`` watches (its
#: ``_WATCHED_STATUSES``), read directly here rather than through that
#: function: ``notify()`` POSTs a webhook, and a GET must never start work.
_ATTENTION_DELIVERY_STATUSES = (
    "failed_permanent",
    STATUS_AWAITING_MANUAL_PUBLISH,
    STATUS_MISSING_REMOTE,
)

#: Delivery statuses worth surfacing on "Coming up" -- still on their way
#: through the queue, as opposed to ``TERMINAL_STATUSES``.
_UPCOMING_DELIVERY_STATUSES = ("queued", "submitting")

#: Rows shown on "Coming up". Enough to see what's next without turning the
#: overview into a second copy of the Publishing tab.
UPCOMING_LIMIT = 5


def _section(items: List[Dict[str, Any]], unreadable: List[str]) -> Dict[str, Any]:
    return {"rows": items, "unreadable": unreadable}


def _row_label(row: Dict[str, Any]) -> str:
    """``{project_name}#{short_index} ({platform}) — {status}``, falling
    back to ``Delivery #{id}`` only when ``project_name`` is missing."""
    project_name = row.get("project_name")
    subject = (
        f"{project_name}#{row.get('short_index')}" if project_name else f"Delivery #{row['id']}"
    )
    return f"{subject} ({row.get('platform')}) — {row['status']}"


def _parse_instant(value: Optional[str]) -> Optional[datetime]:
    """An aware ``datetime`` from an ISO-8601 string, or ``None`` for
    anything absent, unparsable, or naive -- comparing a naive instant
    against an aware ``now`` would be a guess, not a fact."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _scheduled_instant(row: Dict[str, Any]) -> Optional[datetime]:
    return _parse_instant(row.get("scheduled_at")) or _parse_instant(row.get("scheduled_at_utc"))


def _is_overdue(row: Dict[str, Any], now: datetime) -> bool:
    instant = _scheduled_instant(row)
    return instant is not None and instant < now


def _format_overdue_since(row: Dict[str, Any]) -> str:
    instant = _scheduled_instant(row)
    if instant is None:
        raise ValueError("_format_overdue_since requires a row _is_overdue already confirmed")
    return instant.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _job_rows(store: jobs.JobStore) -> "Optional[List[Dict[str, Any]]]":
    try:
        return jobs.list_jobs(store)
    except sqlite3.Error:
        return None


def _delivery_rows(config: Optional[Any] = None) -> "Optional[List[Dict[str, Any]]]":
    """Delivery rows enriched exactly as the Publishing tab enriches them.

    Not the raw ``queue_rows`` output: ``enrich_rows_with_sidecar`` is what
    resolves a row's EFFECTIVE schedule, preferring the row's own
    ``scheduled_at`` column and falling back to the sidecar's time when that
    column is NULL. Reading the raw column here would drop a delivery that
    Publishing lists as scheduled, and an overview that shows less than the
    page it summarises is worse than no overview."""
    try:
        rows = pubqueue.queue_rows(config)
    except sqlite3.Error:
        return None
    return pubqueue.enrich_rows_with_sidecar(rows)


def needs_attention(
    store: jobs.JobStore, config: Optional[Any] = None, *, now: Optional[datetime] = None
) -> Dict[str, Any]:
    """Jobs that failed or were interrupted, plus deliveries stuck
    ``failed_permanent``/``awaiting_manual_publish`` OR still ``queued``/
    ``submitting`` past their scheduled instant -- each linking to the page
    that resolves it. A queued delivery whose slot has passed and did not go
    out needs an operator; it is not "coming up" any more."""
    if now is None:
        now = datetime.now(timezone.utc)
    items: List[Dict[str, Any]] = []
    unreadable: List[str] = []

    job_rows = _job_rows(store)
    if job_rows is None:
        unreadable.append("jobs")
    else:
        items += [
            {
                "label": f"Job #{row['id']} {row['name']} — {row['status']}",
                "href": f"/jobs/{row['id']}",
            }
            for row in job_rows
            if row.get("status") in _ATTENTION_JOB_STATUSES
        ]

    delivery_rows = _delivery_rows(config)
    if delivery_rows is None:
        unreadable.append("the delivery queue")
    else:
        items += [
            {
                "label": _row_label(row),
                "href": f"/publishing/{row['id']}",
            }
            for row in delivery_rows
            if row.get("status") in _ATTENTION_DELIVERY_STATUSES
        ]
        items += [
            {
                "label": f"{_row_label(row)}, overdue since {_format_overdue_since(row)}",
                "href": f"/publishing/{row['id']}",
            }
            for row in delivery_rows
            if row.get("status") in _UPCOMING_DELIVERY_STATUSES and _is_overdue(row, now)
        ]

    return _section(items, unreadable)


def in_progress(store: jobs.JobStore) -> Dict[str, Any]:
    """Jobs still ``queued``/``running``, with their real status."""
    job_rows = _job_rows(store)
    if job_rows is None:
        return _section([], ["jobs"])
    items = [
        {
            "label": f"Job #{row['id']} {row['name']} — {row['status']}",
            "href": f"/jobs/{row['id']}",
        }
        for row in job_rows
        if row.get("status") in _IN_PROGRESS_JOB_STATUSES
    ]
    return _section(items, [])


def coming_up(config: Optional[Any] = None, *, now: Optional[datetime] = None) -> Dict[str, Any]:
    """The next scheduled deliveries, soonest first -- a row whose slot has
    already passed belongs on "Needs attention" instead (see
    ``needs_attention``), never here."""
    if now is None:
        now = datetime.now(timezone.utc)
    delivery_rows = _delivery_rows(config)
    if delivery_rows is None:
        return _section([], ["the delivery queue"])
    upcoming = [
        row
        for row in delivery_rows
        if row.get("status") in _UPCOMING_DELIVERY_STATUSES
        and row.get("scheduled_at_effective_local")
        and not _is_overdue(row, now)
    ]
    # Ordered by a machine-comparable instant, never by the display string:
    # ``scheduled_at_effective_local`` is formatted for a reader and carries a
    # timezone NAME, so sorting it would order "19:00 America/Chicago" against
    # "19:00 UTC" as if they were the same moment. Same precedence the
    # enricher uses -- the row's own column first, the sidecar's UTC second.
    upcoming.sort(key=lambda row: row.get("scheduled_at") or row.get("scheduled_at_utc") or "")
    items = [
        {
            "label": _row_label(row),
            "href": f"/publishing/{row['id']}",
        }
        for row in upcoming[:UPCOMING_LIMIT]
    ]
    return _section(items, [])


def build_overview(
    store: jobs.JobStore, config: Optional[Any] = None, *, now: Optional[datetime] = None
) -> Dict[str, Any]:
    """The three sections the overview screen renders."""
    if now is None:
        now = datetime.now(timezone.utc)
    return {
        "needs_attention": needs_attention(store, config, now=now),
        "in_progress": in_progress(store),
        "coming_up": coming_up(config, now=now),
    }
