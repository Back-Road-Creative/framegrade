"""C1: read-only cross-platform reconciliation of the delivery queue.

Reads the same rows as the Publishing tab (``pubqueue``) and sorts them into
states that never blur into one another, lists coverage gaps per platform and
format up to a horizon, and separates recovered failures from unresolved ones.
Nothing here writes, retries, republishes or contacts a platform: a failed
Instagram row is *reported* alongside any earlier successful publication of the
same short, and the decision stays with the operator.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from ..publishing.queue import STATUS_MISSING_REMOTE
from ..publishing.tiktok_pack import SWEEP_HORIZON_DAYS
from . import pubqueue

_ET = ZoneInfo("America/New_York")

#: Result order, and the headings the page prints.
STATE_ORDER = (
    "queued",
    "uploaded",
    "remotely_scheduled",
    "published",
    "awaiting_manual",
    "unverified",
)

#: States that cover a slot: something is on its way or already out. A failed,
#: cancelled or unverified row covers nothing.
_COVERING = ("queued", "uploaded", "remotely_scheduled", "published", "awaiting_manual")


def _state(row: Dict[str, Any]) -> Optional[str]:
    """The reconciliation state of ``row``, or ``None`` for a row that belongs
    to the failure section (or is cancelled / mid-submit)."""
    status = row.get("status")
    if status == "queued":
        return "queued"
    if status == "uploaded":
        return "uploaded"
    if status == "awaiting_manual_publish":
        return "awaiting_manual"
    if status in ("awaiting_ig_reconcile", STATUS_MISSING_REMOTE):
        return "unverified"
    if status == "published":
        return pubqueue.published_kind(row)
    return None


def _day(row: Dict[str, Any]) -> Optional[str]:
    raw = row.get("scheduled_at")
    if not raw:
        return None
    try:
        when = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(_ET).date().isoformat()


def _failures(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_id = {r["id"]: r for r in rows}
    recoverers: Dict[Any, Dict[str, Any]] = {}
    for row in rows:
        target = row.get("recovers_delivery_id")
        if target is not None:
            recoverers[target] = row
    out: Dict[str, List[Dict[str, Any]]] = {
        "recovered": [],
        "recovery_in_flight": [],
        "unresolved": [],
    }
    for row in rows:
        if row.get("status") != "failed_permanent":
            continue
        entry: Dict[str, Any] = {
            "id": row["id"],
            "platform": row.get("platform"),
            "project_name": row.get("project_name"),
            "short_index": row.get("short_index"),
            "error_class": row.get("error_class"),
        }
        recoverer = recoverers.get(row["id"])
        if recoverer is not None and _state(recoverer) in ("published", "remotely_scheduled"):
            entry["recovered_by"] = recoverer["id"]
            out["recovered"].append(entry)
        elif recoverer is not None and recoverer.get("status") not in (
            "failed_permanent",
            "cancelled",
        ):
            entry["recovered_by"] = recoverer["id"]
            out["recovery_in_flight"].append(entry)
        else:
            if row.get("platform") == "instagram":
                prior = next(
                    (
                        r
                        for r in by_id.values()
                        if r["id"] != row["id"]
                        and r.get("platform") == "instagram"
                        and r.get("project_name") == row.get("project_name")
                        and r.get("short_index") == row.get("short_index")
                        and _state(r) == "published"
                    ),
                    None,
                )
                entry["prior_publication"] = (
                    {"id": prior["id"], "permalink": prior.get("permalink")} if prior else None
                )
                entry["action"] = "report only"
            out["unresolved"].append(entry)
    return out


def build(
    rows: List[Dict[str, Any]],
    now: Optional[datetime] = None,
    horizon_days: int = SWEEP_HORIZON_DAYS,
) -> Dict[str, Any]:
    """The reconciliation of ``rows`` (queue rows, enriched with ``format``)."""
    now = now or datetime.now(timezone.utc)
    live = [r for r in rows if not r.get("shadow")]
    states: Dict[str, List[Dict[str, Any]]] = {name: [] for name in STATE_ORDER}
    for row in live:
        name = _state(row)
        if name is not None:
            states[name].append(row)

    start = now.astimezone(_ET).date()
    days = [(start + timedelta(days=i)).isoformat() for i in range(max(horizon_days, 0))]
    covered: Dict[tuple, set] = {}
    for row in live:
        key = (row.get("platform"), row.get("format") or "not stated")
        covered.setdefault(key, set())
        if _state(row) in _COVERING:
            day = _day(row)
            if day:
                covered[key].add(day)
    gaps = {key: [d for d in days if d not in have] for key, have in sorted(covered.items())}

    by_platform: Dict[str, Dict[str, int]] = {}
    for name, members in states.items():
        for row in members:
            counts = by_platform.setdefault(row.get("platform") or "?", {})
            counts[name] = counts.get(name, 0) + 1

    return {
        "as_of": now.isoformat(),
        "horizon_days": horizon_days,
        "days": days,
        "states": states,
        "by_platform": by_platform,
        "gaps": gaps,
        "failures": _failures(live),
        "shadow_ignored": len(rows) - len(live),
    }


def report(config: Optional[Any] = None, now: Optional[datetime] = None) -> Dict[str, Any]:
    """``build`` over the live queue, read-only (a missing DB is an empty queue)."""
    grouped = pubqueue.rows_by_state(pubqueue.queue_rows(config))
    return build([row for _status, members in grouped for row in members], now)
