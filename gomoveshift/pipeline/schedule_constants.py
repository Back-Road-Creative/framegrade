"""GoMoveShift publishing schedule constants — single source of truth.

All schedule-aware modules must import from here instead of defining
their own copies. Consumers: longform_metadata, content_ledger, scheduler,
quality_gates, state_invariants.

Configurable cadence
--------------------

Long-form publish days and time are **configurable** via
``config.drive.long_form_publish_days`` / ``long_form_publish_time``.
Historically these were hardwired to Wed @ 11:00 EST; that remains the
default so existing callers and tests stay green, but future cohorts
(e.g. a Friday launch) override via
``~/.config/gomoveshift/config.yaml``.

Two access patterns live side-by-side:

- **Module attributes** (``LONG_FORM_DAYS``, ``LONG_FORM_TIME``) are
  computed once at import from the *default* config. Tests that
  ``monkeypatch.setattr(schedule_constants, "LONG_FORM_DAYS", {...})``
  continue to work — the lazy import sites in ``quality_gates`` and
  ``state_invariants`` still observe the patched value.
- **Accessor functions** (``long_form_days(config)`` /
  ``long_form_time(config)``) are the preferred path for code that
  already has a ``Config`` in hand. Passing ``config=None`` (the
  default) falls back to the module attributes so monkeypatch
  tests remain effective.
"""

import logging
from datetime import datetime as dt_datetime
from datetime import time as dt_time
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Timezone for all scheduling
EST = ZoneInfo("America/New_York")


def _parse_hhmm(value: str) -> dt_time:
    """Parse an ``HH:MM`` 24h string into a ``datetime.time``.

    Validation is already performed by ``DriveConfig``'s pydantic
    validator; this helper just does the conversion.
    """
    hh, mm = value.split(":", 1)
    return dt_time(int(hh), int(mm), 0)


def _default_drive_config():
    """Return a fresh ``DriveConfig`` instance with defaults.

    Imported lazily to avoid a circular import at module load time
    (``gomoveshift.config`` imports from ``gomoveshift.pipeline``).
    """
    from gomoveshift.config import DriveConfig

    return DriveConfig()


# Module-level constants — computed from DriveConfig defaults so there
# is only one source of truth for the default values. Kept as module
# attributes so existing monkeypatch-based tests stay green.
_DEFAULT_DRIVE = _default_drive_config()


def _seed_cadence():
    """Waypost's packaged seed cadence (``brand.yaml``), read with no decision db.

    Import-time only: the mirrors below start from the seed so no hand-set default is
    left; the live answer (seed + ``lf_days`` decisions) is ``seo_slots.waypost_cadence``.
    """
    from pathlib import Path

    from seo_manager.core.profile import load_cadence

    return load_cadence(
        Path(__file__).resolve().parent.parent / "console" / "brands" / "gomoveshift" / "brand.yaml"
    )


def _waypost_time(cadence, days) -> dt_time:
    """Waypost's time for the earliest of ``days`` in the all-regions view; when no region
    publishes that day (a ``--lf-day`` off Waypost's days), the first region's default."""
    if days:
        weekday = min(days)
        if weekday in cadence.days(None):
            return cadence.time_for(None, weekday)
    return cadence.regions[sorted(cadence.regions)[0]].time


_SEED = _seed_cadence()

# Long form publish days (0=Mon .. 6=Sun): the seed's all-regions union, until
# ``apply_config`` replaces it from the live accessors.
LONG_FORM_DAYS = frozenset(_SEED.days(None))

# Long form publish time (EST): the seed's time for its earliest day.
LONG_FORM_TIME = _waypost_time(_SEED, LONG_FORM_DAYS)

# Shorts posting times — computed from DriveConfig defaults, mirroring
# the long-form attrs above. Configurable via
# ``config.drive.shorts_publish_times`` (HH:MM list) or per-invocation
# via the drive CLI's ``--shorts-times`` flag. Module attr remains the
# fallback + monkeypatch target for tests.
SHORTS_TIMES = [_parse_hhmm(t) for t in _DEFAULT_DRIVE.shorts_publish_times]

# Photo post time — the single source of truth for the daily photo slot.
# 09:00 is deliberate: 10:00 is a configured USA shorts window (the Friday
# pair and the second-drive fallback pool), so a photo there double-books a
# video slot. 09:00 is clear of every shorts window, pool entry and the
# long-form time.
PHOTO_POST_TIME = dt_time(9, 0, 0)  # 9:00 AM

# Long-form cadence interval: seo_manager's LF series rule
# (``slots.lf_series_dates``) spaces consecutive LFs exactly this many days apart and
# the drive scheduler fills every day's configured shorts windows in between. The value
# is seo_manager's, shared with ``required_shorts_per_lf`` so the two cannot drift.
from seo_manager.scheduling.slots import LF_PUBLISH_INTERVAL_DAYS  # noqa: E402

# 3-letter weekday abbreviations indexed by Python weekday (0=Mon .. 6=Sun) —
# the canonical ``publish_day`` string form consumed by state-invariant Rule 2
# (``_DAY_NAME_TO_INT`` is its inverse). Used to render a config-derived
# default publish day as a day name.
_DAY_INT_TO_NAME: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def long_form_days(config: Optional[object] = None) -> frozenset:
    """Return the long-form publish-day set.

    The run's ``--lf-day`` transport (``config.drive.long_form_publish_days``) when set,
    otherwise Waypost's days for every region (``seo_slots.waypost_cadence``; raises
    ``CadenceSourceUnreadableError`` when that cannot be read).

    Args:
        config: Optional ``gomoveshift.config.Config`` instance. When
            ``None`` (the default) the module-level ``LONG_FORM_DAYS``
            attribute is returned — this preserves monkeypatch-based
            tests that patch the module attr in-place.

    Returns:
        ``frozenset[int]`` of weekday indices (0=Mon .. 6=Sun).
    """
    if config is None:
        return frozenset(LONG_FORM_DAYS)
    if config.drive.long_form_publish_days:
        return frozenset(config.drive.long_form_publish_days)
    from .seo_slots import waypost_cadence

    return frozenset(waypost_cadence(config).days(None))


def default_publish_day(config: Optional[object] = None) -> str:
    """Return the default ``publish_day`` (3-letter weekday) from config.

    The orchestrator falls back to this when no explicit ``--lf-day`` /
    ``publish_day`` is supplied. It is the weekday name of
    ``min(long_form_days(config))`` — the earliest configured long-form day —
    mirroring ``scheduler.generate_drive_schedule`` (``min(self.long_form_days)``).

    Deriving from config (rather than the old hardwired ``"sat" if region ==
    "usa" else "wed"``) guarantees the result is always in ``LONG_FORM_DAYS``,
    so state-invariant Rule 2 cannot reject the default. The earlier ``"sat"``
    USA literal (dow=5) is NOT in the default ``LONG_FORM_DAYS={2}`` and crashed
    any USA drive started without ``--lf-day``.

    Args:
        config: Optional ``gomoveshift.config.Config``. ``None`` falls back to
            the module-level ``LONG_FORM_DAYS`` (same monkeypatch contract as
            ``long_form_days``).

    Returns:
        3-letter lowercase weekday name (e.g. ``"wed"``).
    """
    days = long_form_days(config)
    return _DAY_INT_TO_NAME[min(days)] if days else "wed"


def long_form_time(config: Optional[object] = None) -> dt_time:
    """Return the long-form publish time-of-day, honoring ``config.drive``.

    Args:
        config: Optional ``gomoveshift.config.Config`` instance. When
            ``None`` the module-level ``LONG_FORM_TIME`` constant is
            returned.

    Returns:
        ``datetime.time`` in EST: the run's ``--lf-time`` transport when set, else
        Waypost's time for the run's day in the all-regions view (the first region's
        default when no region publishes that day).
    """
    if config is None:
        return LONG_FORM_TIME
    if config.drive.long_form_publish_time:
        return _parse_hhmm(config.drive.long_form_publish_time)
    from .seo_slots import waypost_cadence

    return _waypost_time(waypost_cadence(config), long_form_days(config))


def shorts_times(config: Optional[object] = None) -> list:
    """Return the shorts publish times-of-day, honoring ``config.drive``.

    Args:
        config: Optional ``gomoveshift.config.Config`` instance. When
            ``None`` the module-level ``SHORTS_TIMES`` list is returned
            (same monkeypatch contract as ``long_form_days``/``time``).

    Returns:
        ``list[datetime.time]`` in EST.
    """
    if config is None:
        return list(SHORTS_TIMES)
    return [_parse_hhmm(t) for t in config.drive.shorts_publish_times]


def resolve_shorts_times_for_region(
    config, region: Optional[str], lf_day: Optional[str] = None
) -> list:
    """Return the shorts ``HH:MM`` windows of the drive for ``region`` + ``lf_day``.

    Waypost decides them: the region's list in its cadence (``brand.yaml`` seed, replaced
    by the latest ``shorts_times`` decision) is every hour its shorts post at, and each
    long-form day's drive takes its share (``seo_slots.shorts_windows``). A region
    Waypost has no hours for, or an unreadable source, stops planning
    (``CadenceSourceUnreadableError``); nothing falls back to another list. ``region=None``
    keeps the flat ``config.drive.shorts_publish_times``.

    Precedence note: an explicit ``--shorts-times`` is applied UPSTREAM of this (it
    mutates ``shorts_publish_times`` in the drive handler), so the handler only calls this
    when no ``--shorts-times`` was given -- the explicit flag always wins.
    """
    if region is None:
        return list(config.drive.shorts_publish_times)
    from .seo_slots import shorts_windows

    return shorts_windows(config, region, lf_day)


# The DriveConfig field default (config.py). Compared against verbatim to
# detect an "explicit" flat-field value — see rule 1 below. This is the only
# explicitness signal available without a dedicated unset-sentinel on
# DriveConfig itself: a config.yaml/--superlong-day value that happens to
# equal the default is indistinguishable from "never set" and is treated as
# not-explicit, same as every other precedence rung.
_SUPERLONG_DEFAULT_DAY = "sat"


def _related_superlong_weekday(
    ledger, project_name: Optional[str], title: Optional[str]
) -> Optional[str]:
    """Weekday (3-letter) of the most recently booked RELATED superlong part
    in ``ledger``, or ``None`` when nothing related is found.

    "Related" = a booked entry whose title carries the deliberate
    ``— Part N of M`` suffix AND either shares ``project_name`` (preferred —
    the reliable identity) or, when either side's
    ``project_name`` is blank (a few historical rows predate that field), shares
    the base title with the suffix stripped. An empty ``project_name`` never
    matches another empty ``project_name`` — that would treat every
    untagged historical row as one giant series.
    """
    from .title_builder import _SUPERLONG_PART_SUFFIX_RE

    base_title = _SUPERLONG_PART_SUFFIX_RE.sub("", title).strip() if title else ""
    related = []
    for e in ledger.entries:
        if e.content_type != "longform" or not _SUPERLONG_PART_SUFFIX_RE.search(e.title or ""):
            continue
        same_project = (
            bool(project_name) and bool(e.project_name) and e.project_name == project_name
        )
        entry_base = _SUPERLONG_PART_SUFFIX_RE.sub("", e.title or "").strip()
        same_base_title = bool(base_title) and entry_base == base_title
        if same_project or same_base_title:
            related.append(e)
    if not related:
        return None
    latest = max(related, key=lambda e: (e.date, e.time))
    dow = dt_datetime.strptime(latest.date, "%Y-%m-%d").weekday()
    return _DAY_INT_TO_NAME[dow]


def resolve_superlong_publish_day(
    config,
    region: Optional[str],
    *,
    ledger: Optional[object] = None,
    project_name: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """Return the superlong publish weekday for ``region`` — or ``None`` when
    no fixed day applies and the caller must search every weekday for the
    next open slot (see ``content_ledger.next_free_superlong_slot``).

    Precedence (an established cadence outranks the
    region default; supersedes the old flat "Saturday" default):

    1. **Explicit operator instruction** — ``config.drive.superlong_publish_day``
       differs from the class default (``"sat"``), i.e. an operator set it via
       ``--superlong-day`` (mirrored here by ``flag_registry``) or their own
       config file. Always wins, even over ledger history or a per-region map
       entry.
    2. **Previous related videos** — (only when both ``ledger`` and an
       identity, ``project_name`` or ``title``, are supplied) the weekday of
       the most recently booked RELATED superlong part already in the ledger
       (see :func:`_related_superlong_weekday`). A series already publishing
       on a given weekday keeps that weekday even when the region map names
       a different one.
    3. **Location** — ``config.drive.superlong_publish_day_by_region[region]``,
       when there is no related history (or no ``ledger``/identity to check).
    4. **Next open day** — (same ``ledger``+identity gate as (2)) no related
       history AND no region entry found: returns ``None`` so the caller
       searches every weekday instead of defaulting to a hardcoded Saturday.

    Backward-compatible default: a caller that cannot supply ``ledger`` (or
    supplies no identity) gets the OLD flat-field-then-region behavior
    unchanged — this is the safe default for callers (e.g. the read-only
    schedule report) that have no specific video to resolve "related"
    history for.
    """
    flat = str(getattr(config.drive, "superlong_publish_day", None) or _SUPERLONG_DEFAULT_DAY)
    if flat != _SUPERLONG_DEFAULT_DAY:
        return flat
    by_region = getattr(config.drive, "superlong_publish_day_by_region", None) or {}
    has_identity = ledger is not None and (project_name or title)
    if has_identity:
        related = _related_superlong_weekday(ledger, project_name, title)
        if related is not None:
            return related
    if region is not None and region in by_region:
        return str(by_region[region])
    if has_identity:
        return None
    return flat


def required_shorts_per_lf(
    config,
    region: Optional[str] = None,
    lf_day: Optional[str] = None,
    shorts_per_day: Optional[int] = None,
) -> int:
    """Derive the required shorts count per LF from the publish plan.

    ``len(shorts windows per day) × LF_PUBLISH_INTERVAL_DAYS`` — the same
    numbers ``scheduler.generate_drive_schedule`` fills slots from (its
    only structural encoding of "14 = 2/day × 7 days"). The requirement
    used to be hand-copied into three unsynchronized places; this function
    is the single derivation.

    Window resolution reuses ``resolve_shorts_times_for_region``. Pass
    ``region=None`` when the drive handler has ALREADY resolved the
    region/``--shorts-times`` precedence into
    ``config.drive.shorts_publish_times`` (the day pipeline's case) —
    re-resolving by region there would discard an explicit
    ``--shorts-times``.

    ``--shorts-per-day N`` truncates the scheduler to its first N windows
    (Phase D), so the requirement must cap the window count at N — else the
    planner demands full-window shorts the scheduler has no slots for and every
    surplus ships slotless. ``min(windows, N)`` matches the scheduler's own
    ``shorts_times[:N]`` truncation exactly; ``None`` (no cap) preserves the
    full-window requirement.
    """
    if config is None:
        raise ValueError(
            "required_shorts_per_lf: no config — there is no publish plan to derive from"
        )
    n_windows = len(resolve_shorts_times_for_region(config, region, lf_day))
    if shorts_per_day is not None:
        n_windows = min(n_windows, shorts_per_day)
    return n_windows * LF_PUBLISH_INTERVAL_DAYS


def resolve_shorts_pool_for_region(
    config, region: Optional[str], lf_day: Optional[str] = None
) -> list:
    """Return the region's shorts-time POOL (HH:MM) for collision fallback: Waypost's
    whole list for the region (``seo_slots.shorts_pool``), every hour a drive may
    substitute into when its preferred slot is already booked. ``lf_day`` is accepted
    for the callers' symmetry; the pool is the region's, not a drive's share. ``None``
    region: the flat ``shorts_publish_times``."""
    if region is None:
        return list(config.drive.shorts_publish_times)
    from .seo_slots import shorts_pool

    return shorts_pool(config, region)


def resolve_publish_slot(config, lf_day: Optional[str]):
    """Return the ``PublishSlot`` for ``lf_day``, or ``None`` if undesigned.

    ``lf_day`` is a 3-letter weekday key (``"tue"``). Days without a slot
    (e.g. ``"wed"``, the day to avoid) return ``None`` so callers fall back
    to the region/flat defaults.
    """
    if not lf_day:
        return None
    slots = getattr(config.drive, "publish_slots", None) or {}
    return slots.get(lf_day)


def resolve_shorts_times(
    config, lf_day: Optional[str], shorts_times_arg: Optional[str], region: Optional[str]
) -> list:
    """Resolve the effective shorts HH:MM strings — the single precedence point.

    Shorts times are Waypost's (``seo_slots.shorts_windows``). ``lf_day`` selects the
    drive's share of the region's list; it never reaches a separate ``publish_slots``
    shorts source (that override was removed).

    Order:

    1. Explicit ``--shorts-times`` (``shorts_times_arg``, comma-separated).
    2. The region's Waypost windows for ``lf_day``.
    3. The flat ``shorts_publish_times`` when there is no region.
    """
    if isinstance(shorts_times_arg, str) and shorts_times_arg.strip():
        return [t.strip() for t in shorts_times_arg.split(",") if t.strip()]
    return resolve_shorts_times_for_region(config, region, lf_day)


def resolve_lf_time(config, lf_day: Optional[str], lf_time_arg: Optional[str]) -> str:
    """Resolve the effective long-form publish time (HH:MM).

    Order: explicit ``--lf-time`` → the LF day's designed slot →
    Waypost's time (``long_form_time``).
    """
    if isinstance(lf_time_arg, str) and lf_time_arg.strip():
        return lf_time_arg
    slot = resolve_publish_slot(config, lf_day)
    if slot is not None:
        return slot.lf_time
    t = long_form_time(config)
    return f"{t.hour:02d}:{t.minute:02d}"


def apply_config(config) -> None:
    """Push a loaded ``Config``'s cadence into the module-level attrs.

    Call once at CLI entry (right after ``get_config()`` /
    ``Config.load()``) so every lazy-import read downstream — in
    ``state_invariants``, ``quality_gates``, ``longform_metadata``,
    ``content_ledger``, ``scheduler`` — observes the operator-resolved
    values instead of the ``DriveConfig`` defaults that were snapshot
    at import time.

    Without this call, a user override in
    ``~/.config/gomoveshift/config.yaml`` (e.g. a Friday cadence)
    loads into ``Config`` correctly but is invisible to every
    enforcement point that reads ``schedule_constants.LONG_FORM_DAYS``
    directly. That produced a preview crash,
    ``StateInvariantError: publish_day 'fri' ... not in
    LONG_FORM_DAYS=[2]`` even with the override installed.

    Idempotent and safe to call multiple times. Uses ``globals()``
    assignment so ``monkeypatch.setattr(schedule_constants,
    "LONG_FORM_DAYS", ...)`` in tests still wins when applied AFTER
    this call — the monkeypatch targets the same module attrs.
    """
    globals()["LONG_FORM_DAYS"] = long_form_days(config)
    globals()["LONG_FORM_TIME"] = long_form_time(config)
    globals()["SHORTS_TIMES"] = [_parse_hhmm(t) for t in config.drive.shorts_publish_times]
