"""Console Publishing tab, calendar reading: the same delivery-queue
rows the list view (``pubqueue.html``) shows, bucketed by day instead of
grouped by status. This module owns the day-bucketing only -- it never
re-derives a schedule, never re-implements the pause/enqueue/drain rules, and
never touches ``publishing/queue.py`` directly. Every row comes back through
``pubqueue.rows_by_state``, so this reading can never disagree with the list
about what a row's effective schedule or publish-confirmation label is.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from . import pubqueue

#: The bucket key for a row with no resolvable schedule. Never a real ISO
#: date string, so it can't collide with one and callers don't need to parse
#: anything to special-case it.
UNSCHEDULED = "unscheduled"

#: Display-only truncation length for a row's title/caption in the calendar
#: grid and the "Not scheduled" table -- long enough to read the gist, short
#: enough that one long TikTok caption can't blow out a table cell. Never
#: applied to the value ``rows_by_state`` returns, only to what the template
#: shows; the full text stays reachable via the row's detail-page link.
TITLE_DISPLAY_MAX_CHARS = 60

#: The three distinct reasons a cell may have no title text, read straight off
#: fields ``pubqueue.rows_by_state`` already computed -- never re-derived here.
#: A title that was never recorded, one the sidecar could not be read for, and
#: one that is genuinely an empty string are different facts about the row and
#: must not collapse into the same dash. See ``pubqueue.enrich_rows_with_sidecar``.
TITLE_NOT_RECORDED = "not_recorded"
TITLE_UNREADABLE = "unreadable"
TITLE_EMPTY = "empty"
TITLE_PRESENT = "present"


def _title_status(row: Dict[str, Any]) -> str:
    """Which of the three "no title text" reasons (or "present") applies to
    ``row``, read from ``title`` and ``sidecar_read_error`` -- both already
    set by ``pubqueue.enrich_rows_with_sidecar`` for every row this module
    ever sees."""
    if row.get("sidecar_read_error"):
        return TITLE_UNREADABLE
    title = row.get("title")
    if title is None:
        return TITLE_NOT_RECORDED
    if title == "":
        return TITLE_EMPTY
    return TITLE_PRESENT


#: What the Format column shows when the sidecar could not be read at all.
#: ``format`` is ``None`` both when a sidecar stated no format and when the
#: read failed, exactly as ``title`` is -- so the Format column needs the same
#: three-way split the Title column has, or an unreadable sidecar silently
#: reads as a deliberate "not stated".
FORMAT_UNREADABLE_LABEL = "could not read sidecar"
FORMAT_NOT_STATED_LABEL = "not stated"


def _format_label(row: Dict[str, Any]) -> str:
    """The Format cell's text: the stated format, or which of the two reasons
    there isn't one. Read off ``format`` and ``sidecar_read_error``, both
    already set by ``pubqueue.enrich_rows_with_sidecar`` -- never re-derived."""
    if row.get("sidecar_read_error"):
        return FORMAT_UNREADABLE_LABEL
    return row.get("format") or FORMAT_NOT_STATED_LABEL


def _all_rows(config: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Every delivery, enriched and labelled exactly like the list view --
    flattened out of ``rows_by_state``'s per-status groups rather than
    re-enriched here, so this reading shares the list's rules by construction
    instead of by two copies staying in sync."""
    grouped = pubqueue.rows_by_state(pubqueue.queue_rows(config))
    return [row for _status, rows in grouped for row in rows]


def _bucket_key(row: Dict[str, Any]) -> str:
    """The row's day bucket, read verbatim from ``scheduled_at_effective_local``
    -- the exact field the list view tests before printing "no schedule
    recorded" (see ``pubqueue.html``). A blank field there means
    ``UNSCHEDULED`` here too: never a default day, never "today", never the
    moment the page happened to render."""
    effective = row.get("scheduled_at_effective_local")
    if not effective:
        return UNSCHEDULED
    return effective[:10]


def week_start(anchor: Optional[date] = None) -> date:
    """The Monday on or before ``anchor`` (default: today)."""
    anchor = anchor or date.today()
    return anchor - timedelta(days=anchor.weekday())


def _instant_key(row: Dict[str, Any]) -> str:
    """The row's machine-comparable instant, same precedence the overview
    tab uses (``overview.py``): the row's own column first, the sidecar's
    UTC second, never the display string."""
    return row.get("scheduled_at") or row.get("scheduled_at_utc") or ""


def _time_label(row: Dict[str, Any]) -> str:
    """``HH:MM`` sliced from ``scheduled_at_effective_local`` -- the same
    field the list view already trusts to say "no schedule recorded".
    Never re-parses ``scheduled_at``: a field too short to hold a time
    yields the empty string rather than raising."""
    effective = row.get("scheduled_at_effective_local") or ""
    if len(effective) < 16:
        return ""
    return effective[11:16]


def _sort_and_label(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = sorted(rows, key=_instant_key)
    for row in rows:
        row["time_label"] = _time_label(row)
        row["title_status"] = _title_status(row)
        row["format_label"] = _format_label(row)
    return rows


def _group_by_project(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``rows`` (already time-ordered) collapsed into one group per
    project, each group ordered by its earliest row -- the order its rows
    already carry, so the first row seen for a project fixes the group's
    position without a second sort."""
    groups: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = row.get("project_name")
        group = groups.get(name)
        if group is None:
            group = {"project_name": name, "rows": []}
            groups[name] = group
        group["rows"].append(row)
    return list(groups.values())


def week_calendar(config: Optional[Any] = None, anchor: Optional[date] = None) -> Dict[str, Any]:
    """One Mon-Sun week of the delivery queue, bucketed by day, plus an
    ``unscheduled`` group for every row with no resolvable schedule.

    The week is always all 7 days, each with its own (possibly empty) row
    list -- an empty day is rendered as an empty day, never omitted, so the
    grid never lies about which days exist. A row whose bucket falls outside
    this week is simply not shown; paging to another week (``prev_week``/
    ``next_week``) is how an operator reaches it."""
    start = week_start(anchor)
    days = [start + timedelta(days=i) for i in range(7)]
    buckets: Dict[str, List[Dict[str, Any]]] = {d.isoformat(): [] for d in days}
    unscheduled: List[Dict[str, Any]] = []
    for row in _all_rows(config):
        key = _bucket_key(row)
        if key in buckets:
            buckets[key].append(row)
        elif key == UNSCHEDULED:
            unscheduled.append(row)
    day_rows = {key: _sort_and_label(rows) for key, rows in buckets.items()}
    for row in unscheduled:
        row["title_status"] = _title_status(row)
        row["format_label"] = _format_label(row)
    week_has_rows = any(day_rows[d.isoformat()] for d in days)
    return {
        "week_start": start,
        "week_end": days[-1],
        "prev_week": (start - timedelta(days=7)).isoformat(),
        "next_week": (start + timedelta(days=7)).isoformat(),
        "week_has_rows": week_has_rows,
        "title_display_max_chars": TITLE_DISPLAY_MAX_CHARS,
        "days": [
            {
                "date": d,
                "rows": day_rows[d.isoformat()],
                "groups": _group_by_project(day_rows[d.isoformat()]),
            }
            for d in days
        ],
        "unscheduled": unscheduled,
    }
