"""Rule 13, ``seo_feedback_stale``: the Shorts feedback loop has stopped feeding itself.

Three conditions, each one grouped incident with a stable key, each resolving when it clears.
Detection only: nothing here edits a plan, a queue row, a tracking record or the decisions file.

- ``seo_feedback_stale:decisions`` -- ``<seo storage>/shorts_arm_decisions.json`` (written by
  ``metrics collect`` once every factor is decided) is missing, unreadable, or its ``decided_at``
  is over ``DECISIONS_STALE_AFTER_H`` old. A missing file is this finding and holds nothing else.
- ``seo_feedback_stale:metrics`` -- ONE row counting the YouTube Shorts whose day-7 window is
  settled (``tracking.day7.window_settled``) and that were posted within ``METRICS_LOOKBACK_DAYS``
  yet have a null ``CORE_METRICS`` field; the detail lists the first ``LISTED`` of them.

- ``seo_feedback_stale:sync`` -- the nightly ``health sync-short-text`` run record
  (``short_text_sync.run_record_path``) is missing, unreadable, or its ``ran_at`` is over
  ``SYNC_STALE_AFTER_H`` old (or over ``SYNC_SKEW_H`` in the future): the job is not running.
  A crashed run still stamps ``ran_at`` (``short_text_sync_crash`` covers it alone). Reads only
  that file.

An open row is refreshed (detail, count, listing) by the caller without a second push.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..publishing.experiments import read_arm_decisions
from ..tracking.day7 import posted_date, window_settled
from .store import new_incident, safe_name

KIND = "seo_feedback_stale"
KEY_DECISIONS = f"{KIND}:decisions"
KEY_METRICS = f"{KIND}:metrics"
KEY_SYNC = f"{KIND}:sync"

#: The nightly ``metrics collect`` refreshes the decisions file; a day plus a night of slack.
DECISIONS_STALE_AFTER_H = 36
#: ``health sync-short-text`` runs nightly, chained after ``metrics collect``, so its start drifts:
#: the same limit as the decisions file.
SYNC_STALE_AFTER_H = 36
#: A ``ran_at`` further ahead of the check than this is clock skew: unreadable, never "fresh".
SYNC_SKEW_H = 1
#: ``PlatformMetrics`` fields a settled Short must have reported (``tracking/schema.py``).
CORE_METRICS = ("views", "avg_view_percentage")
#: Only Shorts posted this recently are checked: an old Short the API never filled in is history,
#: not a fault the collector can still repair.
METRICS_LOOKBACK_DAYS = 28
#: Shorts named in a grouped row's detail; the rest are counted ("and N more").
LISTED = 10


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


def sync_incidents(stamp: Path, now: datetime, now_iso: str) -> List[Dict[str, Any]]:
    """Condition (c): reads only the sync run record. Missing or unreadable IS the finding."""
    stamped = None
    try:
        record = json.loads(stamp.read_text(encoding="utf-8"))
        stamped = datetime.fromisoformat(str(record["ran_at"]))
    except FileNotFoundError:
        why = "is missing: no run of the job is on record"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        why = f"is unreadable ({type(exc).__name__}: {exc})"
    else:
        stamped = stamped if stamped.tzinfo else stamped.replace(tzinfo=timezone.utc)
        age_h = (now - stamped).total_seconds() / 3600
        if age_h < -SYNC_SKEW_H:
            why = f"is unreadable (ran_at {stamped.isoformat()} is in the future)"
        elif age_h <= SYNC_STALE_AFTER_H:
            return []
        else:
            why = (
                f"is stale: the last run was {stamped.isoformat()}, {int(age_h)} h ago "
                f"(limit {SYNC_STALE_AFTER_H} h)"
            )
    return [
        _row(
            KEY_SYNC,
            now_iso,
            f"The nightly Short text sync run record {stamp.name} {why}. "
            "Check that `gomoveshift health sync-short-text` is scheduled and running.",
            ran_at=stamped.isoformat() if stamped else None,
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


def aged_out_count(records: Sequence[Any], today: date, listed: Sequence[Any]) -> int:
    """How many of the Shorts an open ``metrics`` row listed (``{"project", "short_index"}``) are
    still missing core metrics but now past the lookback: what the resolved row lost to ageing
    rather than to a fix. Shorts the row never listed are not counted."""
    wanted = {(e.get("project"), e.get("short_index")) for e in listed}
    n = 0
    for rec in records:
        for short in rec.shorts:
            if (rec.project, short.index) not in wanted:
                continue
            posted = posted_date(short.posted_at)
            if posted is None or (today - posted).days <= METRICS_LOOKBACK_DAYS:
                continue
            yt = short.metrics.get("youtube")
            if yt is None or any(getattr(yt, m, None) is None for m in CORE_METRICS):
                n += 1
    return n
