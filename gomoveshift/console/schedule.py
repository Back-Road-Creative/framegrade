"""Console Schedule tab: read-only view over
``pipeline.schedule_report.build_schedule_report`` -- what is booked, when
the queue drains, and the next free long-form/superlong slots per region.
Mirrors ``cli/handlers/schedule.py::cmd_schedule``'s ledger+config
construction; never re-implements the scheduler.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..config import Config
from ..pipeline.content_ledger import ContentLedger
from ..pipeline.schedule_report import build_schedule_report

logger = logging.getLogger(__name__)

_EMPTY_REPORT: dict = {
    "as_of": None,
    "weekly_longform_capacity": 0,
    "designed_days": [],
    "scheduled_longforms": [],
    "scheduled_shorts_count": 0,
    "scheduled_shorts_by_region": {},
    "queue_ends": None,
    "next_available": [],
    "next_available_superlong": [],
}


def _region_for_journey(config: Any, journey: str) -> "str | None":
    """The region whose CURRENT active journey (``config.drive.
    active_journey_by_region``) matches ``journey`` -- the only region<->journey
    mapping this config carries. The ledger itself has no per-entry journey
    field, only ``region``, so this is the mechanism the Schedule tab filters
    through rather than inventing a second one. Returns ``None`` when no
    region names ``journey`` as its active one -- a real "no match", not a
    reason to fall back to the unfiltered report."""
    mapping = getattr(getattr(config, "drive", None), "active_journey_by_region", None) or {}
    for region, active in mapping.items():
        if active == journey:
            return region
    return None


def _filter_report_by_region(report: dict, region: "str | None") -> dict:
    """Narrow the booked/available rows to ``region``. ``region is None``
    means the journey matched no region at all -- filter to NOTHING, never to
    everything: a schedule that silently ignored an unmatched journey and
    showed every region would look filtered while it was not."""
    longforms = [e for e in report["scheduled_longforms"] if region and e.get("region") == region]
    next_available = [r for r in report["next_available"] if region and r.get("region") == region]
    next_available_superlong = [
        r for r in report["next_available_superlong"] if region and r.get("region") == region
    ]
    report["scheduled_longforms"] = longforms
    report["next_available"] = next_available
    report["next_available_superlong"] = next_available_superlong
    report["queue_ends"] = longforms[-1]["date"] if longforms else None
    # Narrow the shorts figure the same way -- via the per-region breakdown
    # build_schedule_report now reports. region is None on an unmatched
    # journey (filter to nothing); an "unassigned" (region=None) short can
    # never be attributed to a specific region, so it never counts here.
    shorts_by_region = report.get("scheduled_shorts_by_region", {})
    report["scheduled_shorts_count"] = shorts_by_region.get(region, 0) if region else 0
    return report


def schedule_view(config: Optional[Any] = None, *, journey: "str | None" = None) -> dict:
    """The schedule report as a template-ready dict. A missing/unreadable
    ledger or config never 500s -- it returns an empty report plus a
    ``hint`` explaining why, same spirit as ``pubqueue.queue_rows`` treating
    a missing DB file as an empty queue.

    ``journey``, when given, narrows ``scheduled_longforms``/``next_available``/
    ``next_available_superlong``/``scheduled_shorts_count`` to the region
    ``config.drive.active_journey_by_region`` currently maps to it.
    ``report["journey"]``
    and ``report["journey_region"]`` always ride along so the template can show
    the active selection and, on an unmatched journey, say so rather than
    quietly rendering the full queue."""
    config = config if config is not None else Config.load()
    journey = journey or ""
    try:
        ledger = ContentLedger.load()
        report = build_schedule_report(ledger, config)
    except Exception as exc:  # noqa: BLE001 -- any ledger/config failure degrades to empty+hint
        logger.warning("console schedule tab: report unavailable: %s", exc)
        report = dict(_EMPTY_REPORT)
        report["hint"] = f"Schedule report unavailable: {exc}"
        report["journey"] = journey
        report["journey_region"] = None
        return report
    report.setdefault("hint", "")
    if journey:
        region = _region_for_journey(config, journey)
        report = _filter_report_by_region(report, region)
        report["journey_region"] = region
    else:
        report["journey_region"] = None
    report["journey"] = journey
    return report
