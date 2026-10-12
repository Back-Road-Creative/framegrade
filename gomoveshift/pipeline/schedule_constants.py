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
import sqlite3
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

# Long form publish days (0=Mon, 2=Wed, 5=Sat). Default: Wednesday only.
LONG_FORM_DAYS = frozenset(_DEFAULT_DRIVE.long_form_publish_days)

# Long form publish time (EST). Default: 11:00 AM.
LONG_FORM_TIME = _parse_hhmm(_DEFAULT_DRIVE.long_form_publish_time)

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
    """Return the long-form publish-day set, honoring ``config.drive``.

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
    return frozenset(config.drive.long_form_publish_days)


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
        ``datetime.time`` in EST.
    """
    if config is None:
        return LONG_FORM_TIME
    return _parse_hhmm(config.drive.long_form_publish_time)


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


def _configured_shorts_times_for_region(
    config, region: Optional[str], lf_day: Optional[str] = None
) -> list:
    """The config-only resolution: config precedence only, no data path."""
    by_region = getattr(config.drive, "shorts_publish_times_by_region", None) or {}
    if region is not None and region in by_region:
        val = by_region[region]
        if isinstance(val, dict):
            return list(val.get(lf_day) or val.get("_default") or config.drive.shorts_publish_times)
        return list(val)
    return list(config.drive.shorts_publish_times)


def best_slot_from_data(
    conn: "sqlite3.Connection", region: str, platform: str, pool: list, min_n: int
) -> "tuple[Optional[str], str]":
    """Resolve Source B (our ranking) gated by Source A (platform-native
    audience-online) where Source A exists for ``platform``.

    Returns ``(best_hhmm, reason)``; ``best_hhmm`` is ``None`` when the
    slot cannot be moved and ``reason`` explains why (logged by the caller).
    Never raises — any DB error is the caller's problem to catch.
    """
    from gomoveshift.tracking.slot_performance import ensure_audience_online_schema, slots_ranked

    ranked = slots_ranked(conn, region=region, platform=platform, min_n=min_n)
    if not ranked:
        return None, (
            f"no ranked slot_performance rows for region={region} platform={platform} "
            f"with n>=min_n={min_n}"
        )
    top = ranked[0]
    candidate = f"{top.hour_local:02d}:00"
    if candidate not in pool:
        return None, (
            f"top slot {candidate} for platform={platform} is not in the region's "
            f"configured shorts_slot_pool_by_region"
        )

    ensure_audience_online_schema(conn)
    online_rows = conn.execute(
        "SELECT hour_local, SUM(n_online) AS total FROM audience_online "
        "WHERE platform = ? GROUP BY hour_local ORDER BY total DESC",
        (platform,),
    ).fetchall()
    if not online_rows:
        # Source A doesn't exist for this platform (e.g. youtube) — B alone
        # decides. Not a fault.
        return (
            candidate,
            f"resolved from slot_performance alone (no audience_online for {platform})",
        )

    top3_hours = {row[0] for row in online_rows[:3]}
    if top.hour_local not in top3_hours:
        return None, (
            f"sources disagree: slot_performance top hour {top.hour_local} for "
            f"platform={platform} is not in audience_online's top-3 {sorted(top3_hours)}"
        )
    return candidate, "resolved: slot_performance and audience_online agree"


def resolve_shorts_times_for_region(
    config,
    region: Optional[str],
    lf_day: Optional[str] = None,
    *,
    conn: "Optional[sqlite3.Connection]" = None,
    platform: Optional[str] = None,
    min_n: int = 5,
) -> list:
    """Return the region-aware shorts HH:MM strings for ``region`` + ``lf_day``.

    Maps ``region`` through ``config.drive.shorts_publish_times_by_region``. A
    region's value is EITHER a flat ``list`` (one window for every drive — e.g.
    Ireland) OR a per-LF-day ``dict`` keyed by 3-letter weekday with a
    ``"_default"`` (the USA Tue+Fri anti-cannibalization case from the
    6-slot map): ``lf_day`` selects the day's window, else
    ``"_default"``. A region absent from the map (or ``None``) falls back to
    the flat ``config.drive.shorts_publish_times``. Returns raw ``HH:MM``
    strings — the only caller assigns the result back to
    ``config.drive.shorts_publish_times``.

    Precedence note: an explicit ``--shorts-times`` is applied UPSTREAM of
    this (it mutates ``shorts_publish_times`` in the drive handler), so the
    handler only calls this when no ``--shorts-times`` was given — the
    explicit flag always wins.

    Data path: when ``conn`` and ``platform`` are both given, the best
    configured slot for ``(region, platform)`` is moved to the front of the
    returned list when our own ranking (``slots_ranked``, Source B) agrees
    with the platform's audience-online curve (Source A) where Source A
    exists for that platform (Meta only today — YouTube has no hour
    dimension by design, TikTok pending). The data only ever picks WHICH
    already-configured slot to prefer; it never invents an hour, and a
    missing/broken/absent database keeps the configured window and logs why
    — it never raises. Every existing caller (``conn=None``) is unaffected.
    """
    configured = _configured_shorts_times_for_region(config, region, lf_day)
    if conn is None or platform is None or region is None:
        return configured

    pool = (getattr(config.drive, "shorts_slot_pool_by_region", None) or {}).get(
        region
    ) or configured

    try:
        best, reason = best_slot_from_data(conn, region, platform, pool, min_n)
    except Exception as exc:  # noqa: BLE001 - any DB failure keeps the configured window
        logger.info(
            "resolve_shorts_times_for_region: data path failed for region=%s "
            "platform=%s (%s) — keeping configured window %s",
            region,
            platform,
            exc,
            configured,
        )
        return configured

    if best is None:
        logger.info(
            "resolve_shorts_times_for_region: keeping configured window %s for "
            "region=%s platform=%s — %s",
            configured,
            region,
            platform,
            reason,
        )
        return configured

    logger.info(
        "resolve_shorts_times_for_region: moving best slot %s to front for "
        "region=%s platform=%s — %s",
        best,
        region,
        platform,
        reason,
    )
    if best in configured:
        return [best] + [t for t in configured if t != best]
    return [best] + configured


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
    """Return the region's candidate shorts-time POOL (HH:MM) for collision fallback.

    Maps ``region`` through ``config.drive.shorts_slot_pool_by_region`` (the full
    set of times the scheduler may substitute into when a preferred slot is
    already booked). A region absent from that map (or ``None``) falls back to
    the region's own default times (``resolve_shorts_times_for_region`` with the
    same ``lf_day``), i.e. pool == default times (no extra fallback room).
    """
    by_region = getattr(config.drive, "shorts_slot_pool_by_region", None) or {}
    if region is not None and region in by_region:
        return list(by_region[region])
    return resolve_shorts_times_for_region(config, region, lf_day)


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

    Shorts times are determined by ``shorts_publish_times_by_region`` (the
    single source of truth). ``lf_day`` selects the
    day's window ONLY within that region map (the USA Tue/Fri per-day
    staggering); it never reaches a separate ``publish_slots`` shorts source
    (that override was removed).

    Order:

    1. Explicit ``--shorts-times`` (``shorts_times_arg``, comma-separated).
    2. ``shorts_publish_times_by_region[region]`` (per-``lf_day`` when that
       region maps to a per-day dict, else the region's flat window).
    3. The flat ``shorts_publish_times`` (via the region resolver's fallback).
    """
    if isinstance(shorts_times_arg, str) and shorts_times_arg.strip():
        return [t.strip() for t in shorts_times_arg.split(",") if t.strip()]
    return resolve_shorts_times_for_region(config, region, lf_day)


def resolve_lf_time(config, lf_day: Optional[str], lf_time_arg: Optional[str]) -> str:
    """Resolve the effective long-form publish time (HH:MM).

    Order: explicit ``--lf-time`` → the LF day's designed slot →
    ``config.drive.long_form_publish_time``.
    """
    if isinstance(lf_time_arg, str) and lf_time_arg.strip():
        return lf_time_arg
    slot = resolve_publish_slot(config, lf_day)
    if slot is not None:
        return slot.lf_time
    return config.drive.long_form_publish_time


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
    globals()["LONG_FORM_DAYS"] = frozenset(config.drive.long_form_publish_days)
    globals()["LONG_FORM_TIME"] = _parse_hhmm(config.drive.long_form_publish_time)
    globals()["SHORTS_TIMES"] = [_parse_hhmm(t) for t in config.drive.shorts_publish_times]
