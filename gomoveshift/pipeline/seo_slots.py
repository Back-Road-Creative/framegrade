"""GMS side of the single scheduling source: ``seo_manager.scheduling`` decides, this asks.

Every slot decision (next free slot, is this anchor on the cadence, is this date free)
lives in ``seo_manager.scheduling.slots``. GMS keeps what only it knows: the content
ledger (``.content_ledger.json``) and the loaded ``config``. This module translates both
into the two things seo_manager decides over, and makes no slot decision itself:

* :func:`cadence_for` -- config -> ``seo_manager.core.profile.Cadence``;
* :class:`Occupancy` -- ledger long-forms (plus ``schedule.sqlite3`` when it exists) ->
  the ``tz`` + ``bookings()`` pair the slot functions read.

The thin wrappers callers already use (``content_ledger.next_free_slot``,
``schedule_report.build_schedule_report``, Rule 8, Rule 2b, Phase D's LF 2..N guard, the
launch-time anchor gate) call in here. The one GMS slot resolver left outside is the
superlong day (``schedule_constants.resolve_superlong_publish_day``): seo_manager's
``superlong_day`` is a single value, GMS resolves it per region and per related part
(see ``docs/unattended-cadence.md``). It is the ONLY holdout. The drive scheduler's
shorts window end, the metadata next-slot label, Phase D's base date and the single-video
stage / ingest batch default dates all ask here, and so do the dates of LF 2..N:
:func:`lf_series` is ``seo_manager.scheduling.slots.lf_series_dates`` (LF n = LF1 + 7 *
(n - 1), same weekday), the rule the slot claim books too. GMS adds no days of its own.
"""

from __future__ import annotations

import calendar
import logging
from datetime import datetime
from datetime import time as dt_time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

from seo_manager.core.profile import Cadence, RegionCadence, canon_region, parse_weekday
from seo_manager.scheduling import slots
from seo_manager.scheduling.store import Booking, ScheduleStore, SlotTaken

from .schedule_constants import EST, long_form_days, long_form_time

logger = logging.getLogger(__name__)

STORE_FILENAME = "schedule.sqlite3"
_SYNTHETIC_REGION = "default"


def _config_regions(config) -> list[str]:
    drive = getattr(config, "drive", None)
    return sorted(getattr(drive, "shorts_publish_times_by_region", None) or {})


def cadence_for(
    config,
    region: Optional[str] = None,
    *,
    days: Optional[Iterable[int]] = None,
    day_times: Optional[dict] = None,
) -> Cadence:
    """The loaded config as a ``Cadence``.

    GMS config binds no region to its own weekdays, so every region gets the run's
    long-form days (``days`` or ``long_form_days(config)``) and flat time. The requested
    ``region`` is always present (seo_manager refuses an unknown one); with no region at
    all a synthetic one stands in so the all-regions view still has a cadence.
    """
    chosen = sorted(long_form_days(config) if days is None else days)
    names = {canon_region(r) for r in _config_regions(config)}
    if region:
        names.add(canon_region(region))
    if not names:
        names = {_SYNTHETIC_REGION}
    rc = RegionCadence(dow=tuple(chosen), time=long_form_time(config), day_times=day_times or {})
    return Cadence(regions={n: rc for n in sorted(names)}, tz=EST)


class Occupancy:
    """Duck-typed ``ScheduleStore`` for the slot functions: ``tz`` and ``bookings()``.

    Rows are the ledger's long-form entries (any status, as the ledger has always
    blocked) unioned with ``schedule.sqlite3`` beside the ledger when that file exists
    (never created here). A store row for a date+region+project the ledger already holds
    is the same LF written twice and is dropped; another project's row on that date stays. ``aliases`` renames project names, so a launch
    gate can treat both spellings of the run's own project as one.
    """

    tz = EST

    def __init__(
        self, ledger, *, aliases: Optional[dict] = None, cadence: Optional[Cadence] = None
    ):
        self._ledger = ledger
        self._aliases = aliases or {}
        self._regions = sorted(cadence.regions) if cadence is not None else [_SYNTHETIC_REGION]

    def _store_path(self) -> Optional[Path]:
        path = getattr(self._ledger, "_path", None)
        return Path(path).parent / STORE_FILENAME if path else None

    def bookings(self) -> list[Booking]:
        rows: list[Booking] = []
        counters: dict[str, int] = {}
        entries = sorted(
            (e for e in self._ledger.entries if e.content_type == "longform"),
            key=lambda e: (e.date, e.time),
        )
        for e in entries:
            project = self._aliases.get(e.project_name, e.project_name)
            counters[project] = counters.get(project, 0) + 1
            at = e.datetime_est
            rows.append(Booking(project, counters[project], e.region, at, e.video_id, at))
        held = {(b.slot_at.date(), canon_region(b.region), b.project) for b in rows}
        db = self._store_path()
        if db is not None and db.exists():
            for b in ScheduleStore(db, EST, self._regions).bookings():
                project = self._aliases.get(b.project, b.project)
                if (b.slot_at.astimezone(EST).date(), canon_region(b.region), project) in held:
                    continue
                rows.append(
                    Booking(
                        self._aliases.get(b.project, b.project),
                        b.lf_index,
                        b.region,
                        b.slot_at,
                        b.video_id,
                        b.booked_at,
                    )
                )
        return sorted(rows, key=lambda b: b.slot_at)


def next_free_slot(
    ledger,
    config,
    region: Optional[str],
    *,
    publish_day: Optional[str] = None,
    any_day: bool = False,
    slot_time: Optional[dt_time] = None,
    project: Optional[str] = None,
    **kwargs,
) -> datetime:
    """Ask seo_manager for the next free long-form slot over the ledger's occupancy."""
    cadence = cadence_for(config, region)
    return slots.next_free_slot(
        Occupancy(ledger, cadence=cadence),
        cadence,
        canon_region(region),
        publish_day=publish_day,
        any_day=any_day,
        slot_time=slot_time,
        project=project,
        **kwargs,
    )


def default_publish_slot(
    config, publish_day: Optional[str], *, after: Optional[datetime] = None
) -> datetime:
    """The next free long-form slot on ``publish_day`` over the SEO-storage ledger.

    The default publish date of the single-video stage and the ingest batch (never today;
    ``after`` moves the search past an earlier episode's slot). Ledger-booked dates are skipped.
    """
    from .content_ledger import ContentLedger

    ledger = ContentLedger.load(Path(config.seo.storage_path) / ".content_ledger.json")
    return next_free_slot(
        ledger, config, None, publish_day=publish_day, include_today=False, after=after
    )


_NO_LEDGER = SimpleNamespace(entries=[], _path=None)


def next_cadence_day(
    days: Iterable[int], lf_time: dt_time, after: datetime, *, include_today: bool = False
) -> datetime:
    """The first cadence day after ``after`` at ``lf_time`` (today only if ``include_today``
    and its time has not passed).

    Occupancy-blind on purpose: the drive scheduler uses it to end a shorts window at the
    next LF day of the run's own cadence, not to book anything. The weekday search is
    ``slots.next_free_slot`` over an empty occupancy.
    """
    cadence = Cadence(
        regions={_SYNTHETIC_REGION: RegionCadence(dow=tuple(sorted(days)), time=lf_time)},
        tz=EST,
    )
    return slots.next_free_slot(
        Occupancy(_NO_LEDGER), cadence, None, after=after, include_today=include_today
    )


def anchor_errors(dates: Iterable[str], days: Iterable[int]) -> list[str]:
    """Rule 8 / Rule 2b: dates outside ``days`` (``slots.validate_anchor``)."""
    allowed = frozenset(days)
    cadence = Cadence(
        regions={_SYNTHETIC_REGION: RegionCadence(dow=tuple(sorted(allowed)), time=dt_time(0, 0))},
        tz=EST,
    )
    return slots.validate_anchor(cadence, None, [d for d in dates if d], days=allowed)


def anchor_problems(dates: Iterable[str], days: Iterable[int]) -> list[tuple[str, Optional[int]]]:
    """``(date, weekday or None)`` for dates outside ``days`` (``slots.anchor_problems``)."""
    allowed = frozenset(days)
    cadence = Cadence(
        regions={_SYNTHETIC_REGION: RegionCadence(dow=tuple(sorted(allowed)), time=dt_time(0, 0))},
        tz=EST,
    )
    return slots.anchor_problems(cadence, None, [d for d in dates if d], days=allowed)


def lf_series(lf1, count: int) -> list:
    """Dates of a drive's first ``count`` LFs from LF1: seo_manager's series rule."""
    return slots.lf_series_dates(lf1, count)


def assert_dates_free(
    ledger, config, region: Optional[str], dates: Iterable[str], *, project: Optional[str] = None
) -> None:
    """Raise ``SlotTaken`` when any date is held for ``region`` by another project."""
    cadence = cadence_for(config, region)
    slots.assert_free(
        Occupancy(ledger, cadence=cadence), cadence, canon_region(region), dates, project=project
    )


def build_report(ledger, config, days: list[str], day_times: dict, now: datetime) -> dict:
    """The availability core of the schedule report, built by ``seo_manager.scheduling.report``."""
    from seo_manager.scheduling import report as seo_report

    cadence = cadence_for(config, None, days=day_times.keys(), day_times=day_times)
    return seo_report.build(Occupancy(ledger, cadence=cadence), cadence, now=now)


def _label(flag: str) -> str:
    return "lf_publish_date" if flag == "--lf-publish-date" else flag


def anchor_refusal(
    config,
    ledger,
    *,
    anchor: str,
    publish_day: Optional[str],
    region: Optional[str],
    project_names: Iterable[str],
    now: Optional[datetime] = None,
    resumed: bool = False,
    flag: str = "--lf-publish-date",
) -> Optional[str]:
    """Why a run may not launch on ``anchor`` (``--lf-publish-date`` or the saved one).

    Refuses an anchor off the run's LF day, past its slot time, or already booked by
    another project -- before any phase runs, so a bad anchor is a launch error and never
    a Phase D failure two hours into a drive. The message names the next free date.
    ``resumed`` (a saved anchor of a drive that already has a live LF) keeps only the
    weekday check: the anchor is history and moving it would move a live drive.
    """
    from .state_invariants import lf_anchor_problem

    now = now or datetime.now(EST)
    problem = lf_anchor_problem(anchor, publish_day, long_form_days(config), flag=flag)
    if problem:
        return problem + _hint(ledger, config, region, publish_day, now, project_names)
    if resumed:
        return None
    day = datetime.strptime(anchor, "%Y-%m-%d").date()  # the parser validate_anchor used
    if _slot_has_passed(config, region, day, now):
        return f"{_label(flag)} {anchor!r} is in the past" + _hint(
            ledger, config, region, publish_day, now, project_names
        )
    names = list(project_names)
    own = names[0] if names else None
    cadence = cadence_for(config, region)
    occ = Occupancy(ledger, aliases={n: own for n in names}, cadence=cadence)
    try:
        slots.assert_free(occ, cadence, canon_region(region), [anchor], project=own)
    except SlotTaken as exc:
        after = datetime.combine(day, dt_time(0, 0), tzinfo=EST)
        return f"{_label(flag)} {anchor!r}: {exc}" + _hint(
            ledger, config, region, publish_day, max(now, after), project_names
        )
    return None


def _slot_has_passed(config, region, day, now: datetime) -> bool:
    """Whether ``day``'s long-form slot is at or before ``now``. The slot time is
    seo_manager's: the first cadence slot of that day over an empty occupancy."""
    start = datetime.combine(day, dt_time(0, 0), tzinfo=EST)
    slot = slots.next_free_slot(
        Occupancy(_NO_LEDGER),
        cadence_for(config, region),
        canon_region(region),
        after=start,
        include_today=True,
    )
    if slot.date() != day:  # not a cadence day; the weekday check owns that refusal
        return day < now.date()
    return slot <= now


def _hint(ledger, config, region, publish_day, after: datetime, project_names) -> str:
    """The next free date over the SAME allowed set the gate uses (LF days within the
    run's ``--publish``); when that set is empty there is no date to suggest."""
    if publish_day:
        try:
            dow = parse_weekday("publish_day", publish_day)
        except ValueError:
            return ""
        if dow not in long_form_days(config):
            have = [calendar.day_abbr[d] for d in sorted(long_form_days(config))]
            return (
                f"; no LF day: --publish {publish_day} is not one of the configured "
                f"LF days {have} (set --lf-day or --publish to match)"
            )
    names = list(project_names)
    own = names[0] if names else None
    cadence = cadence_for(config, region)
    try:
        nxt = slots.next_free_slot(
            Occupancy(ledger, aliases={n: own for n in names}, cadence=cadence),
            cadence,
            canon_region(region),
            after=after,
            publish_day=(publish_day or None),
            include_today=False,
            project=own,
        )
    except (RuntimeError, ValueError):
        return ""
    return f"; next free slot: --lf-publish-date {nxt.strftime('%Y-%m-%d')}"


def book_in_store(
    ledger_path: Path,
    config,
    *,
    project: str,
    lf_index: int,
    region: Optional[str],
    slot_at: datetime,
    video_id: Optional[str],
) -> None:
    """Mirror a confirmed LF booking into ``schedule.sqlite3`` beside the ledger.

    Best effort: only when the store already exists (the nightly runner creates it), and a
    failure is logged, never raised -- the upload is irreversible, the ledger row is
    already written, and the audit runner reconciles a missing store row.
    """
    db = Path(ledger_path).parent / STORE_FILENAME
    if not db.exists():
        return
    try:
        cadence = cadence_for(config, region)
        ScheduleStore(db, EST, cadence.regions).book(
            project, lf_index, canon_region(region), slot_at, video_id
        )
    except Exception as exc:  # noqa: BLE001 -- never fail an upload over the mirror
        logger.error(
            "SCHEDULE-STORE-UNBOOKED: %s lf %s on %s not written to %s: %s",
            project,
            lf_index,
            slot_at,
            db,
            exc,
        )


__all__ = [
    "Occupancy",
    "anchor_errors",
    "anchor_problems",
    "anchor_refusal",
    "assert_dates_free",
    "book_in_store",
    "build_report",
    "cadence_for",
    "default_publish_slot",
    "next_cadence_day",
    "next_free_slot",
]
