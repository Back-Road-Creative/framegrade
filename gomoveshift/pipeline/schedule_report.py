"""Read-only publishing-schedule report — what is booked, when the queue drains,
the next free long-form slot per region and designed day, and its shorts. Also
surfaces the next free superlong-part slot per region (``next_available_superlong``,
resolved via ``resolve_superlong_publish_day`` — never a designed weekly day, so
kept out of the ``next_available`` matrix) and labels each booked long-form's
``kind`` (``superlong_part`` vs ``longform``) by its title's deliberate
``— Part N of M`` suffix.

Composes existing primitives rather than adding a second scheduler, and mutates
nothing: every candidate day goes to ``next_free_slot`` as an EXPLICIT
``publish_day``, so the module-level ``LONG_FORM_DAYS`` mirror — which a drive
run narrows to its own ``--lf-day`` — is never read or written.
"""

from __future__ import annotations

from datetime import datetime
from datetime import time as dt_time
from typing import Optional

from .content_ledger import ContentLedger, next_free_superlong_slot
from .schedule_constants import (
    EST,
    required_shorts_per_lf,
    resolve_lf_time,
    resolve_shorts_times_for_region,
    resolve_superlong_publish_day,
)
from .scheduler import DAY_NAMES
from .seo_slots import build_report
from .title_builder import _SUPERLONG_PART_SUFFIX_RE


def _designed_days(config) -> list[str]:
    """Designed long-form weekdays in Python weekday order, from the
    ``publish_slots`` cadence table — its size IS the weekly capacity."""
    slots = getattr(config.drive, "publish_slots", None) or {}
    known = [d for d in slots if d.lower()[:3] in DAY_NAMES]
    return sorted(known, key=lambda d: DAY_NAMES[d.lower()[:3]])


def _regions(config) -> list[str]:
    """Region roster — the keys of the region-aware shorts map."""
    return sorted(getattr(config.drive, "shorts_publish_times_by_region", None) or {})


def _availability_matrix(
    ledger: ContentLedger, config, days: list[str], now: datetime
) -> tuple[dict, list]:
    """seo_manager's availability report over the ledger, plus the GMS-only shorts columns.

    Each designed day (``publish_slots`` key) is a cadence day at its DESIGNED
    ``lf_time`` (``resolve_lf_time``), so the search validates the real time; every region
    gets every designed day, because no config field binds a region to a weekday.
    """
    day_times = {}
    key_for = {}
    for day in days:
        hour, minute = (int(p) for p in resolve_lf_time(config, day, None).split(":")[:2])
        day_times[DAY_NAMES[day.lower()[:3]]] = dt_time(hour, minute)
        key_for[DAY_NAMES[day.lower()[:3]]] = day
    core = build_report(ledger, config, days, day_times, now)
    regions = set(_regions(config))
    matrix: list[dict] = []
    for row in core["next_available"]:
        if row["region"] not in regions:
            continue  # the synthetic stand-in region of a config with none
        day = key_for[DAY_NAMES[row["day"]]]
        matrix.append(
            {
                "region": row["region"],
                "day": day,
                "slot": row["slot"],
                "shorts_windows": list(resolve_shorts_times_for_region(config, row["region"], day)),
                "shorts_required": required_shorts_per_lf(config, row["region"], day),
                "earliest_for_region": row["earliest_for_region"],
            }
        )
    return core, matrix


def _superlong_availability(ledger: ContentLedger, config, now: datetime) -> list:
    """One row per region: the next free slot on that region's resolved
    superlong publish day (``resolve_superlong_publish_day`` — the same
    resolver ``next_free_superlong_slot`` uses, never a second source of
    truth). Kept OUT of ``next_available``: that matrix is exactly the
    designed weekly days from ``publish_slots``, and a superlong day
    (default Saturday) is deliberately not one of them.
    """
    rows = []
    for region in _regions(config):
        try:
            slot = next_free_superlong_slot(ledger, config, region, after=now)
        except (RuntimeError, ValueError):
            continue  # no free superlong slot for that region inside the horizon
        rows.append(
            {
                "region": region,
                "day": resolve_superlong_publish_day(config, region),
                "slot": slot.isoformat(),
            }
        )
    return rows


def build_schedule_report(
    ledger: ContentLedger,
    config,
    *,
    after: Optional[datetime] = None,
) -> dict:
    """Return the publishing-schedule report as a plain dict (CLI only renders).

    ``queue_ends`` is the last booked LONG-FORM date — a trailing short never
    extends it — and ``None`` when nothing is booked. ``next_available`` is the
    availability MATRIX: one row per (region, designed day), with
    ``earliest_for_region`` marking each region's soonest. It stays a matrix
    because no config field binds a region to a weekday — the "Ireland publishes
    Sunday" pairing lives only in prose and the free-text
    ``cadence_line_by_region`` — so collapsing to one row per region would
    assert a binding the config does not contain. ``next_available_superlong``
    is the sibling one-row-per-region view of the next free superlong-part
    slot; the superlong day is never one of the designed days above, so it is
    never folded into ``next_available``. ``scheduled_shorts_count`` stays the
    whole-queue total (the CLI prints it verbatim); ``scheduled_shorts_by_region``
    is the per-region breakdown beside it, one key per region plus an
    ``"unassigned"`` bucket for entries whose ``region`` is ``None`` -- a
    legacy/anonymous short is never silently attributed to a region, and never
    dropped from the total either.
    """
    now = after or datetime.now(EST)
    ledger.reload()
    days = _designed_days(config)
    core, matrix = _availability_matrix(ledger, config, days, now)

    upcoming = sorted(ledger.get_scheduled(now), key=lambda e: (e.date, e.time))
    longforms = [e for e in upcoming if e.content_type == "longform"]
    shorts = [e for e in upcoming if e.content_type == "shorts"]
    shorts_by_region: dict[str, int] = {}
    for e in shorts:
        key = e.region if e.region is not None else "unassigned"
        shorts_by_region[key] = shorts_by_region.get(key, 0) + 1

    return {
        "as_of": core["as_of"],
        "weekly_longform_capacity": len(days),
        "designed_days": days,
        "scheduled_longforms": [
            {
                "date": e.date,
                "time": e.time,
                "region": e.region,
                "title": e.title,
                "kind": "superlong_part"
                if _SUPERLONG_PART_SUFFIX_RE.search(e.title or "")
                else "longform",
            }
            for e in longforms
        ],
        "scheduled_shorts_count": len(shorts),
        "scheduled_shorts_by_region": shorts_by_region,
        "queue_ends": longforms[-1].date if longforms else None,
        "next_available": matrix,
        "next_available_superlong": _superlong_availability(ledger, config, now),
    }
