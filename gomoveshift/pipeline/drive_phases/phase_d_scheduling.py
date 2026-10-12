"""Phase D scheduling helpers.

Extracted from ``phase_d.py`` — the sequence that
builds and persists the drive ``ReleaseSchedule`` and the per-LF
``SocialPost`` list. Pure cut-and-move; no semantic changes.

Logger is pinned to ``gomoveshift.pipeline.drive_phases.phase_d`` so
existing ``caplog``-anchored tests keep matching the records emitted
from these helpers.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from datetime import time as dt_time
from pathlib import Path
from typing import Optional

from ...config import Config
from ...publishing.queue import open_queue_db
from .._atomic_write import atomic_write_json
from ..schedule_constants import (
    EST,
    _parse_hhmm,
    resolve_shorts_pool_for_region,
    resolve_shorts_times_for_region,
)
from ..social_generator import TitleFacts, day_title_ordinals

logger = logging.getLogger("gomoveshift.pipeline.drive_phases.phase_d")

# ``--lf-day`` writes ``config.drive.long_form_publish_days = [<index>]`` (drive
# handler); this is the inverse, the 3-letter key the per-LF-day region map and
# the resolver take.
_LF_DAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _lf_day_key(config) -> Optional[str]:
    days = list(getattr(getattr(config, "drive", None), "long_form_publish_days", None) or [])
    if len(days) == 1 and isinstance(days[0], int) and 0 <= days[0] <= 6:
        return _LF_DAY_KEYS[days[0]]
    return None


def _shorts_times_by_platform(config, region: Optional[str], platforms) -> dict:
    """The shorts window per first-party platform, listing ONLY
    the platforms whose window the data path moved off the slot.

    Opens the spool (``<seo.storage_path>/social_queue.sqlite3`` -- where
    ``metrics ingest`` writes ``slot_performance``) and asks
    ``resolve_shorts_times_for_region`` once per platform. A platform whose
    resolved window equals the configured one is left out, so the sidecar
    never restates the slot. Returns ``{}`` -- every platform keeps the slot
    -- when there is no region, no ``seo`` config, the spool cannot be opened,
    or the flat window no longer equals the region spec (an explicit
    ``--shorts-times`` mutated it upstream: the operator's window is never
    reordered by data, same precedence as the drive handler). Never raises.
    """
    storage_path = getattr(getattr(config, "seo", None), "storage_path", None)
    if region is None or not storage_path or not platforms:
        return {}
    lf_day = _lf_day_key(config)
    configured = list(resolve_shorts_times_for_region(config, region, lf_day))
    flat = list(getattr(config.drive, "shorts_publish_times", None) or [])
    if configured != flat:
        logger.info(
            "  D-publish-plan: per-platform hours skipped -- shorts window %s is not the "
            "region %r spec %s (explicit --shorts-times?); every platform keeps the slot",
            flat,
            region,
            configured,
        )
        return {}
    try:
        conn = open_queue_db(Path(storage_path))
    except Exception as exc:  # noqa: BLE001 - the spool is optional input here
        logger.info(
            "  D-publish-plan: per-platform hours skipped -- spool unavailable (%s); "
            "every platform keeps the slot",
            exc,
        )
        return {}
    try:
        out = {}
        for platform in platforms:
            window = list(
                resolve_shorts_times_for_region(
                    config, region, lf_day, conn=conn, platform=platform
                )
            )
            if window != configured:
                out[platform] = window
        return out
    finally:
        conn.close()


def _build_readonly_youtube_service(credentials_dir=None):
    """YouTube Data API client via the uploader's shared SOPS creds path; raising
    here (creds/libs unavailable) is the gate the reconcile caller traps to WARN.

    ``credentials_dir`` overrides the default brand-credentials directory
    (``None`` uses ``seo.credentials_dir``); the sync-site command threads its
    ``--credentials-dir`` flag through here."""
    from googleapiclient.discovery import build

    from .._yt_creds import load_youtube_credentials
    from .youtube_uploader import _build_credentials_obj

    creds = _build_credentials_obj(load_youtube_credentials(credentials_dir))
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def _maybe_rebase_drive_schedule(
    state: dict,
    output_base: Path,
    *,
    now: Optional[datetime] = None,
    log=logger,
) -> dict:
    """Rebase ``state["start_date"]`` when a resumed run's schedule is stale.

    Three-guard contract (ALL must pass to trigger rebase):
      1. ``drive_schedule.computed_at`` > 12h older than ``now`` (missing = stale),
      2. earliest ``lf_slots`` datetime is in the past relative to ``now``,
      3. ``<output_base>/youtube_uploads.json`` is empty.

    Two backstops checked BEFORE guards:
      * ``state["lf_publish_date"]`` — sets start_date verbatim, returns.
      * ``state["force_no_rebase"]`` — returns immediately, no mutation.

    Splitting publish dates across two reference weeks is worse than a
    stale schedule — that's why guard #3 (uploads_present) blocks.

    Clock injection via ``now=``; tests MUST NOT patch ``datetime``.

    Returns diagnostic ``{"action": <str>, "old_start_date", "new_start_date",
    "old_first_publish", "new_first_publish"}``.
    """
    now = now or datetime.now(timezone.utc)
    info: dict = {
        "action": "skip",
        "old_start_date": state.start_date,
        "new_start_date": state.start_date,
        "old_first_publish": None,
        "new_first_publish": None,
    }
    # Backstop #1: --lf-publish-date overrides every code path.
    explicit = getattr(state, "lf_publish_date", None)
    if explicit:
        state.start_date = explicit
        info["action"] = "explicit"
        info["new_start_date"] = explicit
        log.info("Phase D: --lf-publish-date=%s — start_date set explicitly", explicit)
        return info
    # Backstop #2: --force-no-rebase opts out entirely.
    if getattr(state, "force_no_rebase", None):
        info["action"] = "force_no_rebase"
        log.info("Phase D: --force-no-rebase — schedule rebase disabled")
        return info
    old_schedule = state.drive_schedule
    if not old_schedule:
        info["action"] = "no_old_schedule"
        return info

    # ``_deserialize_dataclass`` guarantees ``state["drive_schedule"]``
    # is a ``DriveReleaseSchedule`` dataclass after any state round-trip,
    # so direct attribute access is the contract here.
    lf_slots = old_schedule.lf_slots or []
    computed_at_str = old_schedule.computed_at
    # Guard #1: computed_at older than 12h (missing stamp = stale).
    stale_age = True
    if computed_at_str:
        try:
            computed_at = datetime.fromisoformat(str(computed_at_str))
            if computed_at.tzinfo is None:
                computed_at = computed_at.replace(tzinfo=timezone.utc)
            stale_age = (now - computed_at) > timedelta(hours=12)
        except (ValueError, TypeError):
            stale_age = True
    if not stale_age:
        info["action"] = "fresh"
        return info
    # Guard #3 (cheaper than slot scan): no uploads recorded.
    from ..youtube_sidecar import load_sidecar

    uploads = load_sidecar(Path(output_base))
    if uploads:
        info["action"] = "uploads_present"
        log.info(
            "Phase D rebase: %d youtube_uploads entries — preserving schedule",
            len(uploads),
        )
        return info
    # Guard #2: earliest publish_at in past.
    earliest_dt: Optional[datetime] = None
    earliest_iso: Optional[str] = None
    for slot in lf_slots:
        iso = slot.datetime_est
        if not iso:
            continue
        try:
            slot_dt = datetime.fromisoformat(str(iso)).astimezone(timezone.utc)
        except (ValueError, TypeError):
            continue
        if earliest_dt is None or slot_dt < earliest_dt:
            earliest_dt = slot_dt
            earliest_iso = str(iso)
    info["old_first_publish"] = earliest_iso
    if earliest_dt is None or earliest_dt >= now:
        info["action"] = "future"
        return info
    # All three guards passed: rebase. Clear start_date so the scheduler
    # picks the next publish-day-of-week from its own ``datetime.now(EST)``.
    info["old_start_date"] = state.start_date
    state.start_date = None
    info["new_start_date"] = None
    info["action"] = "rebase"
    log.info(
        "Phase D: drive_schedule REBASED (computed_at=%s, earliest_publish=%s, "
        "now=%s, uploads=0) — clearing stale start_date=%s",
        computed_at_str,
        earliest_iso,
        now.isoformat(),
        info["old_start_date"],
    )
    if lf_slots:
        log.info(
            "Phase D rebase: OLD lf_slots dates = %s",
            [s.date for s in lf_slots],
        )
    return info


def _assert_lf_dates_free(
    ledger, config, region, base_date: str, n_lfs: int, *, project: Optional[str] = None
) -> None:
    """Refuse a run whose LF 2..N dates are booked by another project.

    LF 1 (the base date) is deliberately not re-checked: the default base
    already comes from ``next_free_slot``, an explicit operator date stays
    authoritative, and the locked-base read-back / earliest-upload anchors
    may legitimately sit on a date this drive already booked itself.

    The scheduler anchors LF 1 at ``base_date`` and seo_manager's series rule dates
    the rest, but only the base date was ever
    ledger-checked. Occupancy is asked of seo_manager (``assert_free`` over the
    ledger's long-forms and the schedule store): a date another project holds for the
    region raises. ``project`` is the run's own project name, so the rows its own claim
    already booked are not mistaken for a rival's. Never shifts.
    """
    from seo_manager.scheduling.store import SlotTaken

    from ..seo_slots import assert_dates_free, lf_series

    ledger.reload()
    series = lf_series(base_date, n_lfs)
    for i in range(1, n_lfs):
        day = series[i]
        try:
            assert_dates_free(ledger, config, region, [day.isoformat()], project=project)
        except SlotTaken as exc:
            raise RuntimeError(
                f"LF {i + 1} of {n_lfs} would publish on {day.isoformat()} "
                f"(LF 1 {base_date} + {(day - series[0]).days}d), a date already "
                f"booked for region {region!r} — refusing to "
                "schedule; pick a different --lf-publish-date"
            ) from exc


def _build_and_persist_drive_schedule(
    *,
    scheduler_cls,
    config: Config,
    drive_plan,
    state,
    region: str,
    stages_dir: Path,
    now: Optional[datetime] = None,
    rebased: bool = False,
    project: Optional[str] = None,
):
    """Build the ``DriveSchedule`` and persist it to ``stages_dir/drive_schedule.json``.

    Returns the schedule. ``custom_shorts_times`` is the first
    ``state["shorts_per_day"]`` entries of
    ``config.drive.shorts_publish_times`` (single source of truth — the
    ``--shorts-times`` flag feeds the same list). Persistence failures
    degrade to WARN (schedule is held in-process regardless of disk
    write success).

    Stamps ``schedule.computed_at`` with the injected ``now`` (or
    ``datetime.now(timezone.utc)`` when omitted) so the rebase-on-stale-
    resume guard can detect drift between Phase D entries.

    ``rebased`` is the caller's ``_maybe_rebase_drive_schedule`` verdict.
    That pass clears an already-``None`` ``start_date``, so the base-date
    read-back below cannot tell "we chose to rebase" from "never resolved";
    ``rebased=True`` says the stale base is meant to be discarded.

    ``project`` is the run's project name (the source stem Phase D books under); the
    LF 2..N guard exempts that project's own claimed rows.
    """
    _shorts_per_day = state.shorts_per_day
    custom_shorts_times = None
    if _shorts_per_day is not None:
        # --shorts-per-day selects the first N configured windows, so
        # config.drive.shorts_publish_times (and the --shorts-times flag
        # that feeds it) is the single source of truth. When N exceeds the
        # configured windows, fall back to all of them rather than
        # fabricating clock times. Replaces the old hardcoded noon /
        # 8am-4pm presets, which silently overrode an explicit
        # --shorts-times and pinned posts to the worst-performing slots.
        _cfg_times = [_parse_hhmm(t) for t in config.drive.shorts_publish_times]
        custom_shorts_times = (
            _cfg_times[:_shorts_per_day] if _shorts_per_day <= len(_cfg_times) else _cfg_times
        )

    scheduler = scheduler_cls.from_config(config)
    if custom_shorts_times is not None:
        scheduler.shorts_times = custom_shorts_times

    # Cross-run collision avoidance: read the shared content ledger so this
    # drive's shorts fill FREE pool slots, interleaving with any other drive
    # already booked at the same (date, time) — regardless of --output-dir. This
    # is the dynamic safety net atop the per-LF-day default map: a drive posts
    # its resolved default times (``scheduler.shorts_times``), and any default
    # already booked is substituted from the region pool. Any ledger failure
    # degrades to the legacy no-collision-check behavior with a WARN.
    #
    # SINGLE SOURCE OF TRUTH: read the SEO-storage
    # ledger — the SAME file confirmed uploads are booked into — NOT the
    # ContentLedger default (``gms_output_dir()/.content_ledger.json``, a stale
    # orphan). Reading the orphan made collision avoidance blind to every real
    # booking, so concurrent drives piled onto identical slots.
    occupied_slots: set = set()
    resolved_start_date = state.start_date
    if resolved_start_date is None:
        # A resumed run's own LFs are already live: their real publish
        # dates are booked into the shared ledger by then, and
        # ``next_free_slot``'s date-granular/region-only occupancy check
        # can't see project_name — it walks straight past the drive's own
        # bookings and anchors the rest of the schedule AFTER them. Anchor
        # to the earliest already-uploaded LF instead, tying the lock to
        # the irreversible act; an explicit operator date (above) still
        # wins, and no valid record falls through to next_free_slot below.
        earliest_upload: Optional[datetime] = None
        uploads = getattr(state, "youtube_uploads", None) or {}
        for record in uploads.values():
            raw = (record or {}).get("publish_at")
            if not raw:
                continue
            try:
                parsed = datetime.fromisoformat(str(raw))
            except (ValueError, TypeError):
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if earliest_upload is None or parsed < earliest_upload:
                earliest_upload = parsed
        if earliest_upload is not None:
            resolved_start_date = earliest_upload.astimezone(EST).strftime("%Y-%m-%d")
            logger.info(
                "Phase D: base LF date %s = earliest already-uploaded LF (of %d)",
                resolved_start_date,
                len(uploads),
            )
    if resolved_start_date is None and not rebased:
        # Read back the run-wide BASE-DATE lock, mirroring the
        # identity-times read-back below. A drive resolves its base date ONCE
        # (its first Phase-D build) and persists it as ``lf_slots[0].date``;
        # every later build reuses it verbatim rather than re-resolving through
        # ``next_free_slot`` against a ledger and a clock that have both moved
        # on — which silently rebuilt a resumed run's schedule on a DIFFERENT
        # base than the one already written to ``drive_schedule.json``. No new
        # state field: the persisted schedule already carries the date. A
        # missing/legacy/malformed schedule falls through to a fresh resolve.
        _prev_lf_slots = getattr(getattr(state, "drive_schedule", None), "lf_slots", None) or []
        _locked_base = getattr(_prev_lf_slots[0], "date", None) if _prev_lf_slots else None
        if _locked_base:
            resolved_start_date = str(_locked_base)
            logger.info(
                "Phase D: reusing locked base LF date %s from persisted schedule",
                resolved_start_date,
            )
    from ..content_ledger import ContentLedger, next_free_slot, reconcile_ledger_with_youtube

    ledger_path = Path(config.seo.storage_path) / ".content_ledger.json"
    try:
        ledger = ContentLedger.load(ledger_path)
        ledger.reload(ledger_path)
    except Exception as e:
        # A ledger that EXISTS but can't be read/parsed must never degrade to
        # an empty in-memory calendar: this schedule is about to be uploaded
        # and exported to Metricool, so a missed booking here is irreversible.
        # ContentLedger.load()/reload() already treat a genuinely missing
        # file as a fresh install (no exception) — this branch only fires on
        # an actual read/parse failure.
        raise RuntimeError(
            f"Content ledger unreadable at {ledger_path}: {e} — refusing to "
            "schedule without collision data"
        ) from e

    # Fail closed: no blanket except. A slot call that raises (corrupt
    # schedule.sqlite3, unreadable lock file) stops Phase D rather than booking the
    # drive by arithmetic with no occupancy check. Only the YouTube reconcile below
    # is best-effort, and it has its own handler.
    # YouTube is truth, the ledger is a cache. Reconcile once, here
    # — the single site turning ledger entries into scheduling decisions —
    # so phantom slots can't bump real slots. Read-only (~3–5 list calls);
    # ANY failure degrades to the unreconciled ledger with one WARN.
    if getattr(state, "no_ledger_reconcile", None):
        logger.info("LEDGER-RECONCILE skipped: --no-ledger-reconcile")
    else:
        try:
            reconcile_ledger_with_youtube(ledger, _build_readonly_youtube_service())
        except Exception as e:  # noqa: BLE001 — reconciliation never blocks a run
            logger.warning("LEDGER-RECONCILE skipped: %s — proceeding with ledger as-is", e)
    occupied_slots = ledger.get_occupied_slots()
    # With no explicit operator date (--start-date/--lf-publish-date),
    # the DEFAULT base date is the next ledger-FREE config-cadence slot for
    # this region — the drive path was ledger-blind for LF dates and could
    # double-book a taken date. Explicit dates stay authoritative.
    if resolved_start_date is None:
        slot_dt = next_free_slot(
            ledger,
            config,
            region,
            publish_day=state.publish_day,
            include_today=False,  # preserve the scheduler's never-today rule
            after=now.astimezone(EST) if now else None,
        )
        resolved_start_date = slot_dt.strftime("%Y-%m-%d")
        logger.info("Phase D: base LF date %s = next free %s slot", resolved_start_date, region)

    # LF 2..N guard. Skipped on a resumed run (any LF uploaded): its own LFs
    # are already booked under this same drive, which date/region-granular
    # occupancy cannot tell apart from a rival. Single-LF runs are unchanged
    # (the base date is already resolved ledger-free above).
    _n_lfs = len(drive_plan.regular_lfs)
    if _n_lfs > 1 and resolved_start_date and not (getattr(state, "youtube_uploads", None) or {}):
        _assert_lf_dates_free(ledger, config, region, resolved_start_date, _n_lfs, project=project)

    slot_pool = [_parse_hhmm(t) for t in resolve_shorts_pool_for_region(config, region)]
    shorts_per_day = len(scheduler.shorts_times)

    # Number slots from the ACTUAL per-LF bucket sizes (restored
    # real, unequal source-window buckets) so an oversized LF's shorts stay
    # within its own window instead of spilling into the next LF's. Falls
    # back to the legacy fixed per-window capacity only when no buckets are
    # known yet (e.g. an early resume before shorts are attributed).
    _buckets = getattr(state, "shorts_per_lf", None) or {}
    shorts_counts = (
        [len(_buckets.get(lf.index, [])) for lf in drive_plan.regular_lfs] if _buckets else None
    )

    # Read back the run-wide posting-time lock. A drive resolves its
    # consistent shorts posting times ONCE (its first Phase-D build) and
    # persists them on the schedule; every later invocation (incl. a resumed
    # run) reuses that pair verbatim rather than re-resolving from the ledger —
    # which by then holds the drive's OWN booked shorts and would flip the
    # identity to a different pair, breaking the one-pair-per-run contract.
    locked_identity_times = None
    _prev_identity = getattr(getattr(state, "drive_schedule", None), "identity_times", None)
    if _prev_identity:
        locked_identity_times = [dt_time.fromisoformat(t) for t in _prev_identity]
        logger.info(
            "Phase D: reusing locked shorts posting times %s from persisted schedule",
            _prev_identity,
        )

    drive_schedule = scheduler.generate_drive_schedule(
        regular_lfs=drive_plan.regular_lfs,
        region=region,
        start_date=resolved_start_date,
        shorts_per_lf=None,
        publish_day=state.publish_day,
        occupied_slots=occupied_slots,
        slot_pool=slot_pool,
        shorts_per_day=shorts_per_day,
        shorts_counts=shorts_counts,
        locked_identity_times=locked_identity_times,
    )
    # Transactional booking: the schedule is NOT recorded in the ledger
    # here (pre-upload → phantom slots). LF entries book after YouTube
    # confirms; shorts at the media-gate pass. The load above is read-only.
    # Stamp the build time. The rebase guard reads this to decide
    # whether a resumed run's schedule is stale enough to recompute.
    drive_schedule.computed_at = (now or datetime.now(timezone.utc)).isoformat()
    schedule_path = stages_dir / "drive_schedule.json"
    try:
        atomic_write_json(schedule_path, drive_schedule.to_dict())
    except Exception as e:
        logger.warning("Failed to write drive_schedule.json: %s", e)
    return drive_schedule


def _per_short_sidecar_labels(lf_shorts: list) -> list:
    """Per-short resolved OST label + clip-GPS regions from the day's sidecar.

    Each rendered short lives at ``<stages>/day_<DATE>/shorts/short_<rank>.mp4``
    and its own resolved+validated burned text AND its clip-GPS vote regions are
    recorded rank-keyed in the day's ``shorts_ost_audit_<date>.json`` sidecar at
    the day dir's PARENT — the same artefact the resume path reconstructs short
    ranges from. Returns a list PARALLEL to ``lf_shorts`` of
    ``(label, regions, place_source)`` tuples: the short's ``ost_text`` (or
    ``None`` when unavailable — unparsable path/filename, absent/unreadable
    sidecar, missing or empty entry), its ``regions`` list (empty for a
    GPS-less/legacy short) and its recorded ``place_source`` (``"clip_gps"``
    when the label is the clip's own GPS vote; ``None`` when the sidecar
    predates the field). ``regions`` is an independent reference to the label;
    the caller owns the fallback / hard-fail policy.
    """
    out: list = []
    cache: dict = {}
    for short in lf_shorts:
        ref = _short_sidecar_ref(short)
        if ref is None:
            out.append((None, [], None))
            continue
        sidecar, _date, rank = ref
        if ("by_rank", sidecar) not in cache:
            by_rank: dict = {}
            for entry in _load_day_sidecar_entries(sidecar, cache):
                try:
                    label = str(entry.get("ost_text") or "")
                    regions = [str(r) for r in (entry.get("regions") or []) if str(r).strip()]
                    source = entry.get("place_source")
                    by_rank[int(entry["rank"])] = (label, regions, source)
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
            cache[("by_rank", sidecar)] = by_rank
        label, regions, source = cache[("by_rank", sidecar)].get(rank, ("", [], None))
        out.append((label or None, regions, source))
    return out


def _short_sidecar_ref(short) -> Optional[tuple]:
    """``(sidecar path, date, rank)`` for ``<stages>/day_<DATE>/shorts/short_<rank>.mp4``,
    or ``None`` when the path or filename does not parse."""
    try:
        path = Path(short)
        day_dir = path.parent.parent
        day_name = day_dir.name
        rank = int(path.stem.split("_")[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not day_name.startswith("day_"):
        return None
    date = day_name[len("day_") :]
    return day_dir.parent / f"shorts_ost_audit_{date}.json", date, rank


def _load_day_sidecar_entries(sidecar: Path, cache: dict) -> list:
    """The day sidecar's ``shorts`` rows (``[]`` when absent/unreadable), cached per path."""
    if ("entries", sidecar) not in cache:
        try:
            payload = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            payload = {}
        entries = payload.get("shorts") if isinstance(payload, dict) else None
        cache[("entries", sidecar)] = list(entries) if isinstance(entries, list) else []
    return cache[("entries", sidecar)]


def _per_short_title_facts(lf_shorts: list) -> list:
    """Per-short :class:`TitleFacts` (or ``None``) PARALLEL to ``lf_shorts``.

    Date comes from the short's own day dir, the ordinal from its rank's position
    in that day's sidecar (start time order, all LFs), ``secs`` from its own row,
    and ``place_ok`` only when the row is clip-GPS verified
    (:func:`_unverified_reasons`). Any unparsable path/date/row gives ``None``,
    which keeps the legacy composition for that short.
    """
    out: list = []
    cache: dict = {}
    for short, (label, regions, source) in zip(lf_shorts, _per_short_sidecar_labels(lf_shorts)):
        out.append(None)
        ref = _short_sidecar_ref(short)
        if ref is None:
            continue
        sidecar, date, rank = ref
        entries = _load_day_sidecar_entries(sidecar, cache)
        try:
            if ("ordinals", sidecar) not in cache:
                cache[("ordinals", sidecar)] = day_title_ordinals(entries)
            row = next(e for e in entries if int(e["rank"]) == rank)
            secs = int(round(float(row["end_sec"]) - float(row["start_sec"])))
            out[-1] = TitleFacts(
                date=date,
                ordinal=cache[("ordinals", sidecar)][rank],
                secs=secs,
                place_ok=not _unverified_reasons(label, regions, source),
            )
        except (KeyError, TypeError, ValueError, StopIteration):
            continue
    return out


def _unverified_reasons(label, regions, source) -> list[str]:
    """Why a short's sidecar entry does NOT prove its label is its own clip's place.

    A short may name only the place its own clip's GPS verified: it needs a
    sidecar label, non-empty clip-GPS ``regions`` and ``place_source ==
    "clip_gps"`` (the label IS the clip's own vote). Empty list = verified.
    """
    reasons = []
    if not label:
        reasons.append("no sidecar label")
    if not regions:
        reasons.append("no clip-GPS regions")
    if source != "clip_gps":
        reasons.append(f"place_source={source!r} (not clip_gps)")
    return reasons


def unverified_place_shorts(shorts_per_lf: dict) -> list[str]:
    """Offender descriptions for every short whose place is unverified.

    Pure (no logging, no raise): ``shorts_per_lf`` is ``{lf_index: [short
    paths]}``; each short is checked against its day's shorts OST sidecar via
    :func:`_unverified_reasons`. A short with a verified place but no
    :class:`TitleFacts` (date/ordinal/duration unreadable from the sidecar) is
    also an offender: its date-stamped title cannot be composed. Empty list =
    every place verified and every title composable.
    """
    problems: list[str] = []
    for lf_index, lf_shorts in shorts_per_lf.items():
        facts = _per_short_title_facts(lf_shorts)
        for pos, (short, entry, tf) in enumerate(
            zip(lf_shorts, _per_short_sidecar_labels(lf_shorts), facts), 1
        ):
            reasons = _unverified_reasons(*entry)
            if not reasons and tf is None:
                reasons = ["no title facts (date/ordinal/duration unreadable from the day sidecar)"]
            if reasons:
                problems.append(
                    f"LF {lf_index} short {pos} ({Path(str(short)).name}): " + ", ".join(reasons)
                )
    return problems


def assert_all_places_verified(shorts_per_lf: dict, *, enforce: bool = True) -> list[str]:
    """Refuse (publish) or warn (audit) on every short whose place is unverified.

    Runs upstream of chapters and scheduling. ``shorts_per_lf`` is
    ``{lf_index: [short paths]}`` — the caller passes the shorts actually being
    posted. Each is checked against its day's shorts OST sidecar via
    :func:`_unverified_reasons`. ``enforce=True`` (a publish run) raises
    :class:`RuntimeError` naming EVERY offender at once; ``enforce=False`` (the
    audit pass) logs one ``UNVERIFIED-PLACES`` WARNING naming them and returns.
    Returns the offender descriptions (empty when every place is verified).
    """
    problems = unverified_place_shorts(shorts_per_lf)
    if not problems:
        return problems
    body = (
        f"{len(problems)} planned short(s) have an unverified place — a short may name "
        "only the place its own clip's GPS verified:\n  " + "\n  ".join(problems)
    )
    if enforce:
        raise RuntimeError(f"{body}\nRe-render those shorts (their clip needs its own GPS place).")
    logger.warning(
        "UNVERIFIED-PLACES: %s\nThe --youtube-upload pass will refuse until each has a "
        "clip-GPS place.",
        body,
    )
    return problems


def _build_social_posts_per_lf(
    *,
    social_gen_cls,
    drive_plan,
    shorts_per_lf: dict,
    per_lf_locations_override: dict,
    project_base: str,
    output_base: Path,
    per_lf_resolved: dict | None = None,
    titles_file: Path | None = None,
    short_global_index: dict | None = None,
    require_verified_places: bool = False,
) -> dict:
    """Generate the per-LF ``SocialPost`` list, write ``social_posts.json``.

    Returns ``{lf.index: [posts]}``. ``num_shorts=0`` LFs contribute
    zero posts (the legacy ``max(num_shorts, 1)``
    floor poisoned the consolidated CSV row-count invariant).

    ``titles_file`` threads straight into every per-LF
    :class:`~gomoveshift.pipeline.social_generator.SocialGenerator` — an
    OPTIONAL hand-title override on every run (one drive-level file, not
    per-LF, because ``generate_all``'s ``index_offset`` makes ``short_index``
    a single GLOBAL sequence across the whole drive). A short it does not
    cover composes its own stamped title. A short with no title facts is
    refused upstream by :func:`assert_all_places_verified`, not here.

    Drive invariant: each short's caption/title/tag surfaces are fed that
    short's OWN resolved location — its burned ``ost_text``, rank-matched from
    the day's shorts OST sidecar — and its ``sampled_regions`` are fed that
    short's OWN clip-GPS vote regions (an independent reference from the label,
    so the per-short CSV region invariant is no longer label-vs-itself). A
    GPS-less/legacy short (no clip regions) whose label is unavailable falls
    back to its parent-LF label with a WARN on an AUDIT pass only; a GPS-BEARING
    short (clip regions present) whose label is missing HARD-FAILS here, upstream
    of the CSV, rather than inherit the parent county past a region gate blind to
    the desync. ``require_verified_places=True`` (a publish run) removes the
    fallback for EVERY short in ``shorts_per_lf`` (each is posted, indexed or
    not): the per-short location list is built ONLY from sidecar labels, and any
    short lacking a label, clip-GPS regions or ``place_source == "clip_gps"``
    raises before a generator is built.
    There is no drive-wide location fallback: Phase D's entry contract
    guarantees ``per_lf_locations_override`` covers every rendered LF, so a
    missing entry raises here (state corruption, never intent to fall back).

    ``short_global_index`` is ``state.short_global_index`` — a
    ``{str(short_path): global_index}`` map assigned once at plan time
    (``social_generator.assign_global_short_indexes``) and persisted, so this
    function CONSUMES a stable index instead of recomputing
    ``global_index_offset`` from ``len(posts)`` (a render failure used to
    silently shift every later short's index). An LF whose shorts are ALL
    covered by the map uses it (``generate_all(global_indexes=...)``); any LF
    with an uncovered short (legacy state predating this field, or state
    genuinely missing an entry) falls back to the legacy ``index_offset``
    recompute for THAT LF, with a WARN naming the fallback — an in-flight
    drive resumed on an old checkpoint is never broken by the upgrade.

    ``per_lf_resolved`` is ``state.per_lf_resolved_location`` — the
    once-resolved canonical :class:`~gomoveshift.models.ResolvedLocation`
    per LF index. Passed into ``SocialGenerator`` so the secondary text
    surfaces (social caption + shorts YouTube description) anchor the
    destination POI (when one was resolved) consistently with the burned
    OST / title. ``None`` / missing entry preserves the legacy label text.
    """
    per_lf_resolved = per_lf_resolved or {}
    social_posts_per_lf: dict[int, list] = {}
    global_index_offset = 0
    for lf in drive_plan.regular_lfs:
        lf_shorts = shorts_per_lf.get(lf.index, [])
        num_shorts = len(lf_shorts)
        project_name = f"{project_base}_lf{lf.index:02d}"
        lf_location = per_lf_locations_override[lf.index]
        sidecar = _per_short_sidecar_labels(lf_shorts)
        # Publish run: EVERY short in lf_shorts is posted (generate_all builds one
        # post per entry, whether or not short_global_index knows it), so each
        # names ONLY its own sidecar label; one without a verified place raises,
        # so there is no LF-label stand-in to borrow by accident.
        posted = [require_verified_places] * len(lf_shorts)
        offenders = [
            f"short {i} ({Path(str(p)).name}): " + ", ".join(reasons)
            for i, (p, entry, live) in enumerate(zip(lf_shorts, sidecar, posted), 1)
            if live and (reasons := _unverified_reasons(*entry))
        ]
        if offenders:
            raise RuntimeError(
                f"LF {lf.index}: {len(offenders)}/{num_shorts} short(s) have no verified "
                "own-clip place — refusing to hand them the LF label "
                f"{lf_location!r}:\n  " + "\n  ".join(offenders)
            )
        # A GPS-bearing short (clip regions present) that lost its label is
        # a desync — its surfaces would fall back to the parent county while its
        # own GPS voted elsewhere, and the per-short CSV region gate can't catch
        # a label checked against itself. Hard-fail upstream of the CSV.
        desynced = [i + 1 for i, (lab, regs, _src) in enumerate(sidecar) if regs and not lab]
        if desynced:
            raise RuntimeError(
                f"LF {lf.index}: per-short OST sidecar label missing for "
                f"{len(desynced)}/{num_shorts} GPS-bearing short(s) (positions "
                f"{desynced}) whose clip GPS voted regions — refusing to inherit the "
                f"parent-LF label {lf_location!r} into their region-checked CSV "
                f"surfaces (far-corner desync)."
            )
        # A GPS-bearing short's label must be its OWN clip-GPS vote. A sidecar
        # that records an inherited parent-LF label, or predates ``place_source``,
        # names a place this clip may never have been at — refuse to plan it.
        unverified = [
            (i + 1, lab)
            for i, (lab, regs, src) in enumerate(sidecar)
            if regs and lab and src != "clip_gps"
        ]
        if unverified:
            raise RuntimeError(
                f"LF {lf.index}: GPS-bearing short(s) {unverified} (position, label) "
                "carry a label that is not their own clip-GPS vote — a short may name "
                "only the place its clip was filmed. Re-render this drive's shorts."
            )
        # AUDIT PASS ONLY (a publish run raised above): a GPS-less/legacy short
        # (no clip regions) with no label keeps the parent-LF fallback so the
        # rendered shorts can still be reviewed — the CSV region gate no-ops on it.
        missing = [i + 1 for i, (lab, regs, _src) in enumerate(sidecar) if not lab and not regs]
        if missing:
            logger.warning(
                "  LF %d: per-short OST sidecar label unavailable for %d/%d "
                "short(s) (positions %s) — falling back to the parent-LF label "
                "%r for those shorts",
                lf.index,
                len(missing),
                num_shorts,
                missing,
                lf_location,
            )
        title_facts = _per_short_title_facts(lf_shorts)
        social_gen = social_gen_cls(
            location=lf_location,
            project_name=project_name,
            series_part=lf.index,
            resolved_location=per_lf_resolved.get(lf.index),
            titles_file=titles_file,
        )
        global_indexes = None
        if short_global_index is not None and lf_shorts:
            resolved = [short_global_index.get(str(p)) for p in lf_shorts]
            if all(idx is not None for idx in resolved):
                global_indexes = resolved
            else:
                logger.warning(
                    "  LF %d: short_global_index missing an entry for %d/%d "
                    "short(s) — falling back to the legacy index_offset "
                    "recompute for this LF (fallback: pre-upgrade state "
                    "or a genuinely uncovered short).",
                    lf.index,
                    sum(1 for idx in resolved if idx is None),
                    num_shorts,
                )
        posts = social_gen.generate_all(
            num_shorts=num_shorts,
            per_short_title_facts=title_facts,
            index_offset=global_index_offset,
            global_indexes=global_indexes,
            per_short_locations=[
                lab if live else (lab or lf_location) for (lab, _, _), live in zip(sidecar, posted)
            ],
            per_short_regions=[regs for _, regs, _ in sidecar],
        )
        social_posts_per_lf[lf.index] = posts
        global_index_offset += len(posts)

        shorts_dir = output_base / "shorts" / f"lf_{lf.index:02d}"
        shorts_dir.mkdir(parents=True, exist_ok=True)
        posts_file = shorts_dir / "social_posts.json"
        try:
            atomic_write_json(posts_file, [p.to_dict() for p in posts])
        except AttributeError:
            # MagicMock posts in tests may lack to_dict; not critical.
            pass
        except (OSError, TypeError) as exc:
            logger.warning(
                "  LF %d: failed to write social_posts.json (%s): %s",
                lf.index,
                posts_file,
                exc,
            )
    return social_posts_per_lf
