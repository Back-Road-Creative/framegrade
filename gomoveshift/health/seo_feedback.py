"""Rule 12, ``seo_feedback_stale``: the Shorts feedback loop has stopped feeding itself.

Three conditions, each its own incident key, each resolving when it clears. Detection only:
nothing here edits a plan, a queue row, a tracking record or the decisions file.

- ``seo_feedback_stale:decisions`` -- ``<seo storage>/shorts_arm_decisions.json`` (written by
  ``metrics collect`` once every factor is decided) is missing, unreadable, or its ``decided_at``
  is over ``DECISIONS_STALE_AFTER_H`` old.
- ``seo_feedback_stale:metrics:<project>:<index>`` -- a YouTube Short whose day-7 window is
  settled (``tracking.day7.window_settled``) and posted within ``METRICS_LOOKBACK_DAYS`` has a null
  ``CORE_METRICS`` field.
- ``seo_feedback_stale:queued:<project>:<index>`` -- a queued, unleased, non-shadow YouTube row
  whose slot is at least ``queue_short_text.SYNC_LEAD_TIME`` away (so ``sync-short-text`` would
  rewrite it) whose stored place text still repeats the previous Short's although the clip
  measured another place (C2's ``choose_place_body`` says ``varied``), or whose ``experiment_arms``
  carry an arm the decisions file marks ``DEMOTED``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..publishing.experiments import DEMOTE, PAUSE, read_arm_decisions
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
#: An arm with one of these decisions should no longer ride a queued row's text.
DEMOTED = (DEMOTE, PAUSE)


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


def load_demoted(storage: Path) -> Optional[set]:
    """``{(factor, arm)}`` the decisions file marks ``DEMOTED``, or ``None`` when unreadable."""
    try:
        record = read_arm_decisions(storage)
    except (OSError, ValueError):
        return None
    return {
        (d.get("factor"), d.get("arm"))
        for d in record["decisions"]
        if isinstance(d, dict) and d.get("decision") in DEMOTED
    }


def metrics_incidents(records: Sequence[Any], today: date, now_iso: str) -> List[Dict[str, Any]]:
    """Condition (b), from already-loaded tracking records."""
    out: List[Dict[str, Any]] = []
    for rec in records:
        for short in rec.shorts:
            posted = posted_date(short.posted_at)
            if posted is None or not short.platform_ids.get("youtube"):
                continue
            if not window_settled(posted, today) or (today - posted).days > METRICS_LOOKBACK_DAYS:
                continue
            yt = short.metrics.get("youtube")
            gaps = [m for m in CORE_METRICS if yt is None or getattr(yt, m, None) is None]
            if not gaps:
                continue
            out.append(
                _row(
                    f"{KEY_METRICS}:{rec.project}:{short.index}",
                    now_iso,
                    f"Settled Short {rec.project}#{short.index} (posted {posted.isoformat()}) has "
                    f"no recorded {', '.join(gaps)}.",
                    project=rec.project,
                    short_index=short.index,
                    missing=gaps,
                )
            )
    return out


def _repeat(short: dict, prev: dict, location: str) -> Optional[str]:
    """The surface (``title`` / ``description``) whose stored place still repeats the previous
    Short's although the clip measured another place, by C2's own chooser; else ``None``."""
    from ..pipeline.social_generator import (
        KIND_ROAD,
        REASON_VARIED,
        choose_place_body,
        measured_places,
    )
    from ..publishing.queue_short_text import _stored_places

    mine, before = _stored_places(short, location), _stored_places(prev, location)
    found = measured_places(
        short.get("sampled_regions") or [],
        short.get("sampled_roads") or [],
        short.get("route_record"),
    )
    for surface, kinds_ok in (("title", lambda k: k != KIND_ROAD), ("description", lambda k: True)):
        body, last = mine.get(surface), before.get(surface)
        if not body or not last:
            continue
        cands = [body] + [b for k, b in found if kinds_ok(k)]
        if choose_place_body(cands, [last]).reason == REASON_VARIED:
            return surface
    return None


def queued_incidents(
    db_path: Path,
    demoted: Optional[set],
    now: datetime,
    now_iso: str,
    notes: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Condition (c). Returns ``(incidents, held_keys)``: a row whose sidecar cannot be read
    proves nothing either way, so its key is held (noted), never resolved."""
    from ..publishing.queue_short_text import _YT_SQL, SYNC_LEAD_TIME, _lf_inputs, _slot

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(_YT_SQL, (now.isoformat(),)).fetchall()
    finally:
        conn.close()
    out: List[Dict[str, Any]] = []
    held: List[str] = []
    for row in rows:
        slot = _slot(row["scheduled_at"]) if row["scheduled_at"] else None
        if slot is None or slot - now < SYNC_LEAD_TIME:
            continue  # sync leaves it alone (inside the lead time) or cannot place it
        key = f"{KEY_QUEUED}:{row['project_name']}:{row['short_index']}"
        try:
            sidecar = Path(row["sidecar_path"])
            shorts = json.loads(sidecar.read_text(encoding="utf-8"))["shorts"]
            lf = [s for s in shorts if s["project_name"] == row["project_name"]]
            mine = next(s for s in lf if int(s["short_index"]) == row["short_index"])
            same_lf = sorted(
                (s for s in lf if s["lf_index"] == mine["lf_index"]),
                key=lambda s: s["short_index"],
            )
            at = same_lf.index(mine)
            causes: List[Dict[str, str]] = []
            if at > 0:
                _rec, location, why = _lf_inputs(
                    sidecar.parent, row["project_name"], int(mine["lf_index"])
                )
                if why:
                    raise ValueError(why)
                surface = _repeat(mine, same_lf[at - 1], location)
                if surface:
                    causes.append(
                        {
                            "code": "repeated_place_text",
                            "detail": f"the stored {surface} names the previous Short's place "
                            "although the clip measured another",
                        }
                    )
            if demoted:
                for factor, arm in sorted((mine.get("experiment_arms") or {}).items()):
                    if (factor, arm) in demoted:
                        causes.append(
                            {
                                "code": "demoted_arm",
                                "detail": f"arm {arm} of {factor} is demoted in "
                                "shorts_arm_decisions.json",
                            }
                        )
        except Exception as exc:  # noqa: BLE001 -- unreadable sidecar: noted, row held
            notes.append(
                f"seo_feedback_stale queued row {row['id']} not evaluated: "
                f"{type(exc).__name__}: {exc}"
            )
            held.append(key)
            continue
        if causes:
            out.append(
                _row(
                    key,
                    now_iso,
                    f"Queued YouTube Short {row['id']} ({row['project_name']} "
                    f"#{row['short_index']}, slot {slot.isoformat()}) is outside the sync lead "
                    f"time and still needs `sync-short-text`: "
                    + "; ".join(c["detail"] for c in causes)
                    + ".",
                    delivery_id=row["id"],
                    project=row["project_name"],
                    short_index=row["short_index"],
                    scheduled_at=slot.isoformat(),
                    causes=causes,
                )
            )
    return out, held
