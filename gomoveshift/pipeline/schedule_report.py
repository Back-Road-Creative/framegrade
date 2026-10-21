"""Read-only publishing-schedule report — what is booked, when the queue drains,
the next free long-form slot per region and Waypost day, and its shorts. Also
surfaces the next free superlong-part slot per region (``next_available_superlong``,
resolved via ``resolve_superlong_publish_day`` — never a Waypost weekly day, so
kept out of the ``next_available`` matrix) and labels each booked long-form's
``kind`` (``superlong_part`` vs ``longform``) by its title's deliberate
``— Part N of M`` suffix.

Composes existing primitives rather than adding a second scheduler, and mutates
nothing: the long-form days and times come from Waypost's effective cadence
(``seo_slots.build_report``) and each candidate day goes to ``next_free_slot`` as
an EXPLICIT ``publish_day``, so the module-level ``LONG_FORM_DAYS`` mirror — which
a drive run narrows to its own ``--lf-day`` — is never read or written.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from . import volume_source
from .content_ledger import ContentLedger, next_free_superlong_slot
from .schedule_constants import (
    EST,
    required_shorts_per_lf,
    resolve_shorts_times_for_region,
    resolve_superlong_publish_day,
)
from .seo_slots import build_report, waypost_cadence
from .title_builder import _SUPERLONG_PART_SUFFIX_RE


def _regions(config) -> list[str]:
    """Region roster — Waypost's cadence regions."""
    return sorted(waypost_cadence(config).regions)


def _shorts_target(per_location_cap: Optional[int]) -> Optional[int]:
    """The shorts a drive plans per LF: Waypost's decision, trimmed (never raised) by the
    brand's per-location cap -- the same two steps ``day_pipeline`` takes. ``None`` when
    the decision store is unreadable, so the report never prints a guessed count."""
    try:
        decided = volume_source.shorts_per_lf().value
    except volume_source.VolumeSourceUnreadableError:
        return None
    if per_location_cap is not None and per_location_cap < decided:
        return per_location_cap
    return decided


def _availability_matrix(
    ledger: ContentLedger, config, now: datetime, per_location_cap: Optional[int] = None
) -> tuple[dict, list]:
    """seo_manager's availability report over the ledger, plus the GMS-only shorts columns.

    Each region lists ITS OWN Waypost days at Waypost's designed time (the packaged seed
    overlaid with the ``lf_days`` decisions); the shorts windows and count per row are Waypost's
    ``shorts_times`` (dealt by ``drive_share``), looked up by the row's region and day.
    """
    core = build_report(ledger, config, now)
    target = _shorts_target(per_location_cap)
    matrix: list[dict] = []
    for row in core["next_available"]:
        matrix.append(
            {
                "region": row["region"],
                "day": row["day"],
                "slot": row["slot"],
                "shorts_windows": list(
                    resolve_shorts_times_for_region(config, row["region"], row["day"])
                ),
                "shorts_required": target,
                "shorts_slots": required_shorts_per_lf(config, row["region"], row["day"]),
                "earliest_for_region": row["earliest_for_region"],
            }
        )
    return core, matrix


def _superlong_availability(ledger: ContentLedger, config, now: datetime) -> list:
    """One row per region: the next free slot on that region's resolved
    superlong publish day (``resolve_superlong_publish_day`` — the same
    resolver ``next_free_superlong_slot`` uses, never a second source of
    truth). Kept OUT of ``next_available``: that matrix is exactly the
    Waypost's weekly long-form days, and a superlong day
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
    per_location_cap: Optional[int] = None,
) -> dict:
    """Return the publishing-schedule report as a plain dict (CLI only renders).

    ``queue_ends`` is the last booked LONG-FORM date — a trailing short never
    extends it — and ``None`` when nothing is booked. ``designed_days`` is the union
    of Waypost's long-form weekdays (weekday order) and ``lf_days`` says, per region,
    which days and whether they are the packaged seed or a Waypost decision.
    ``next_available`` is the availability MATRIX: one row per (region, ITS OWN
    Waypost day), with ``earliest_for_region`` marking each region's soonest; a day a
    region does not publish has no row. ``next_available_superlong``
    is the sibling one-row-per-region view of the next free superlong-part
    slot; the superlong day is never one of the long-form days above, so it is
    never folded into ``next_available``. ``scheduled_shorts_count`` stays the
    whole-queue total (the CLI prints it verbatim); ``scheduled_shorts_by_region``
    is the per-region breakdown beside it, one key per region plus an
    ``"unassigned"`` bucket for entries whose ``region`` is ``None`` -- a
    legacy/anonymous short is never silently attributed to a region, and never
    dropped from the total either.

    Each ``next_available`` row's ``shorts_required`` is the count a drive plans for that
    LF (Waypost's decision, trimmed by ``per_location_cap``, the brand policy's cap the
    caller passes; ``None`` when the decision store is unreadable); ``shorts_slots`` is the
    slot capacity of its windows over the publish interval, never a target.
    """
    now = after or datetime.now(EST)
    ledger.reload()
    core, matrix = _availability_matrix(ledger, config, now, per_location_cap)
    days = list(core["designed_days"])

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
        "lf_days": core["lf_days"],
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
