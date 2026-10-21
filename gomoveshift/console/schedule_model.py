"""Read model for the Schedule tab's volume figures (B1).

Four different quantities that used to be easy to conflate, each with its source, a fallback reason
and a policy version, and no new cadence engine behind any of them:

* **selected target** -- Waypost's ``shorts_per_lf`` decision, read from the decision store the
  pipeline reads (``volume_source``);
* **slot capacity** -- the shorts windows the region's slot plan holds per long-form, as
  ``schedule_report`` already derives them (windows x LF interval);
* **feasible output** -- derived here and labelled so: the lower of the two, per region, saying
  which one binds;
* **actually scheduled** -- the shorts the content ledger holds for upcoming dates.

Read-only: nothing is written, and an unreadable store blanks the target rather than guessing.
Times render in America/New_York.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from seo_manager.analytics import cadence_decisions

from ..pipeline import volume_source

logger = logging.getLogger(__name__)

NEW_YORK = ZoneInfo("America/New_York")
_SEED = "seed"
#: ``cadence_decisions``' schema CHECK admits exactly these sources.
_KNOWN_SOURCES = frozenset({"policy", "rollback", "blind_revert"})


def _config() -> Any:
    from ..config import Config

    return Config.load()


def eastern(value: Optional[str]) -> str:
    """An ISO timestamp as ``YYYY-MM-DD HH:MM EST|EDT``; a naive value is read as New York time."""
    if not value:
        return "-"
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    moment = moment.replace(tzinfo=NEW_YORK) if moment.tzinfo is None else moment
    return moment.astimezone(NEW_YORK).strftime("%Y-%m-%d %H:%M %Z")


def _db(config: Any) -> Path:
    return Path(config.seo.storage_path) / volume_source.DB_NAME


def _provenance(stored: Any, what: str) -> dict:
    fallback = None
    if stored.source == _SEED:
        fallback = f"no {what} decision stored; the policy seed is in use"
    elif stored.source not in _KNOWN_SOURCES:
        fallback = (
            f"unknown decision source {stored.source!r}; the {what} figure's origin is unverified"
        )
    return {
        "source": stored.source,
        "reason_code": stored.reason_code,
        "policy_rev": stored.policy_rev,
        "fallback_reason": fallback,
    }


def _selected(db: Path) -> dict:
    try:
        stored = cadence_decisions.current(db, volume_source.KNOB)
    except cadence_decisions.DecisionStoreUnreadable as exc:
        logger.warning("console schedule model: decision store unreadable: %s", exc)
        return {
            "value": None,
            "source": "unreadable",
            "reason_code": None,
            "policy_rev": None,
            "fallback_reason": f"decision store unreadable ({exc}); no target is shown",
        }
    return {"value": int(stored.value), **_provenance(stored, "shorts_per_lf")}


def _capacity(report: dict, db: Path) -> dict:
    by_region: dict = {}
    for row in report.get("next_available") or []:
        region = row["region"]
        if region in by_region and not row.get("earliest_for_region"):
            continue
        entry = {"value": row["shorts_slots"], "windows": list(row.get("shorts_windows") or [])}
        try:
            stored = cadence_decisions.current(db, "shorts_times", region)
            entry.update(_provenance(stored, "shorts_times"))
        except cadence_decisions.DecisionStoreUnreadable as exc:
            entry.update(
                source="unreadable",
                reason_code=None,
                policy_rev=None,
                fallback_reason=f"decision store unreadable ({exc}); windows come from the report",
            )
        by_region[region] = entry
    return {"by_region": by_region}


def _feasible(selected: dict, capacity: dict) -> dict:
    by_region: dict = {}
    for region, cap in capacity["by_region"].items():
        if selected["value"] is None:
            by_region[region] = {"value": None, "binding": None}
            continue
        binding = "capacity" if cap["value"] < selected["value"] else "selected"
        by_region[region] = {"value": min(selected["value"], cap["value"]), "binding": binding}
    return {
        "by_region": by_region,
        "derived": "lower of selected target and slot capacity; not a cadence decision",
    }


def volume_model(report: dict, config: Any) -> dict:
    db = _db(config)
    selected = _selected(db)
    capacity = _capacity(report, db)
    return {
        "selected": selected,
        "capacity": capacity,
        "feasible": _feasible(selected, capacity),
        "scheduled": {
            "total": report.get("scheduled_shorts_count", 0),
            "by_region": dict(report.get("scheduled_shorts_by_region") or {}),
            "source": "content ledger, upcoming shorts entries",
            "policy_rev": None,
            "fallback_reason": "ledger rows are bookings, not policy decisions, so carry no version",
        },
    }
