"""Rule 12, ``seo_feedback_stale``: the Shorts feedback loop has stopped feeding itself.

Three conditions, each one grouped incident with a stable key, each resolving when it clears.
Detection only: nothing here edits a plan, a queue row, a tracking record or the decisions file.

- ``seo_feedback_stale:decisions`` -- ``<seo storage>/shorts_arm_decisions.json`` (written by
  ``metrics collect`` once every factor is decided) is missing, unreadable, or its ``decided_at``
  is over ``DECISIONS_STALE_AFTER_H`` old. A missing file is this finding and holds nothing else.
- ``seo_feedback_stale:metrics`` -- ONE row counting the YouTube Shorts whose day-7 window is
  settled (``tracking.day7.window_settled``) and that were posted within ``METRICS_LOOKBACK_DAYS``
  yet have a null ``CORE_METRICS`` field; the detail lists the first ``LISTED`` of them.
- ``seo_feedback_stale:queued`` -- ONE row listing every queued Short whose verdict from
  ``queue_short_text.sync_short_text(conn, apply=False)`` (run on a read-only connection) is
  ``changed``: text the nightly ``health sync-short-text`` would rewrite but has not (the dry run
  gets the brand policy's ``per_location_cap``, as that command does, and has a
  ``DRY_RUN_DEADLINE_S`` deadline). A ``refused`` verdict is that command's own
  ``short_text_sync_refused:<project>`` incident, not this rule's -- except "sidecar unusable" and
  "generator:" refusals, which mean the dry run could not judge the row: the caller notes them
  and holds an open row instead of resolving it.

An open row is refreshed (detail, count, listing) by the caller without a second push.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from ..publishing.experiments import read_arm_decisions
from ..tracking.day7 import posted_date, window_settled
from .store import new_incident, safe_name

KIND = "seo_feedback_stale"
KEY_DECISIONS = f"{KIND}:decisions"
KEY_METRICS = f"{KIND}:metrics"
KEY_QUEUED = f"{KIND}:queued"

#: The nightly ``metrics collect`` refreshes the decisions file; a day plus a night of slack.
DECISIONS_STALE_AFTER_H = 36
#: ``PlatformMetrics`` fields a settled Short must have reported (``tracking/schema.py``).
CORE_METRICS = ("views", "avg_view_percentage")
#: Only Shorts posted this recently are checked: an old Short the API never filled in is history,
#: not a fault the collector can still repair.
METRICS_LOOKBACK_DAYS = 28
#: Shorts named in a grouped row's detail; the rest are counted ("and N more").
LISTED = 10
#: The dry run is bounded like rule 11's listing (``accounting.LIST_DEADLINE_S``): the cron tick
#: holds a lock with no timeout, so a hung queue database must not stall it.
DRY_RUN_DEADLINE_S = 20.0
#: Refusal reasons that mean the dry run could not judge a row (not a text the sync declined).
UNJUDGED_PREFIXES = ("sidecar unusable", "generator:")


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


def _row(key: str, now_iso: str, detail: str, **fields: Any) -> Dict[str, Any]:
    tail = key[len(KIND) + 1 :]
    return new_incident(
        KIND,
        key,
        f"{KIND}_{safe_name(tail)}_{_hash(key)}.json",
        now_iso,
        detail=detail,
        **fields,
    )


def decisions_incidents(storage: Path, now: datetime, now_iso: str) -> List[Dict[str, Any]]:
    """Condition (a). A missing or unreadable file IS the finding."""
    limit = timedelta(hours=DECISIONS_STALE_AFTER_H)
    try:
        record = read_arm_decisions(storage)
    except FileNotFoundError:
        why, stamp = "is missing: `metrics collect` has never recorded the arm decisions", None
    except (OSError, ValueError) as exc:
        why, stamp = f"is unreadable ({type(exc).__name__}: {exc})", None
    else:
        stamp = datetime.fromisoformat(str(record["decided_at"]))
        stamp = stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
        if now - stamp <= limit:
            return []
        why = f"was last written {stamp.isoformat()}, over {DECISIONS_STALE_AFTER_H} h ago"
    return [
        _row(
            KEY_DECISIONS,
            now_iso,
            f"Shorts arm decisions file shorts_arm_decisions.json {why}.",
            decided_at=stamp.isoformat() if stamp else None,
        )
    ]


def _listing(entries: Sequence[str]) -> str:
    shown = ", ".join(entries[:LISTED])
    more = len(entries) - LISTED
    return f"{shown}, and {more} more" if more > 0 else shown


def metrics_incidents(records: Sequence[Any], today: date, now_iso: str) -> List[Dict[str, Any]]:
    """Condition (b), from already-loaded tracking records: ONE grouped row, or none. A Short
    that ages out of the lookback is simply no longer counted."""
    found: List[Dict[str, Any]] = []
    for rec in records:
        for short in rec.shorts:
            posted = posted_date(short.posted_at)
            if posted is None or not short.platform_ids.get("youtube"):
                continue
            if not window_settled(posted, today) or (today - posted).days > METRICS_LOOKBACK_DAYS:
                continue
            yt = short.metrics.get("youtube")
            gaps = [m for m in CORE_METRICS if yt is None or getattr(yt, m, None) is None]
            if gaps:
                found.append({"project": rec.project, "short_index": short.index, "missing": gaps})
    if not found:
        return []
    entries = [f"{f['project']}#{f['short_index']} ({', '.join(f['missing'])})" for f in found]
    return [
        _row(
            KEY_METRICS,
            now_iso,
            f"{len(found)} settled YouTube Short(s) posted in the last {METRICS_LOOKBACK_DAYS} "
            f"days have no recorded core metrics: {_listing(entries)}.",
            count=len(found),
            shorts=found,
        )
    ]


def _dry_run(db_path: Path, now: datetime, cap: Any) -> List[Any]:
    """``sync_short_text(apply=False)`` on a read-only connection, in a daemon thread against
    ``DRY_RUN_DEADLINE_S``; an overrun raises ``TimeoutError`` (the caller notes and holds)."""
    from ..publishing.queue_short_text import sync_short_text

    box: Dict[str, Any] = {}

    def work() -> None:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            try:
                box["verdicts"] = sync_short_text(conn, apply=False, now=now, per_location_cap=cap)
            finally:
                conn.close()
        except BaseException as exc:  # noqa: BLE001 -- re-raised in the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(DRY_RUN_DEADLINE_S)
    if thread.is_alive():
        raise TimeoutError(f"dry run exceeded {DRY_RUN_DEADLINE_S:g} s")
    if "error" in box:
        raise box["error"]
    return box["verdicts"]


def queued_incidents(
    db_path: Path, now: datetime, now_iso: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Condition (c): ``(rows, unjudged)``. ``rows`` is ONE grouped row, or none: the sync's own
    ``changed`` verdicts (dry run, read-only connection, the brand's location cap) name the queued
    Shorts whose rewrite the nightly sync did not apply. ``unjudged`` lists ``project#index:
    reason`` for each "sidecar unusable" / "generator:" refusal: the dry run could not judge
    those, so the caller notes them and holds an open row. A database or sync error (or a
    deadline overrun) propagates: the caller notes it and holds the row."""
    from .short_text_sync import _brand_location_cap

    verdicts = _dry_run(db_path, now, _brand_location_cap())
    unjudged = [
        f"{v.project_name}#{v.short_index}: {v.reason}"
        for v in verdicts
        if v.status == "refused" and v.reason.startswith(UNJUDGED_PREFIXES)
    ]
    changed = [v for v in verdicts if v.status == "changed"]
    if not changed:
        return [], unjudged
    entries = [f"{v.project_name}#{v.short_index}" for v in changed]
    row = _row(
        KEY_QUEUED,
        now_iso,
        f"{len(changed)} queued YouTube Short(s) still carry text `sync-short-text` would "
        f"rewrite: the nightly sync did not apply those rewrites: {_listing(entries)}.",
        count=len(changed),
        shorts=[
            {"delivery_id": v.delivery_id, "project": v.project_name, "short_index": v.short_index}
            for v in changed
        ],
    )
    return [row], unjudged


def aged_out_count(records: Sequence[Any], today: date) -> int:
    """Settled YouTube Shorts with null core metrics that are past the lookback: what a resolved
    ``metrics`` row may have lost to ageing rather than to a fix."""
    n = 0
    for rec in records:
        for short in rec.shorts:
            posted = posted_date(short.posted_at)
            if posted is None or not short.platform_ids.get("youtube"):
                continue
            if (today - posted).days <= METRICS_LOOKBACK_DAYS or not window_settled(posted, today):
                continue
            yt = short.metrics.get("youtube")
            if yt is None or any(getattr(yt, m, None) is None for m in CORE_METRICS):
                n += 1
    return n
