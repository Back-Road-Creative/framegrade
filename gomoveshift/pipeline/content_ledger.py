"""
Content Ledger — tracks all scheduled and published GoMoveShift content.

Single source of truth for what's been posted, what's scheduled, and what
slots are available. Reads from Metricool CSVs and pipeline outputs to
stay in sync.

The ledger is stored as a JSON file and updated whenever:
- A new Metricool CSV is generated (shorts scheduled)
- A long form video completes the pipeline
- Content is manually added

This prevents double-booking slots and lets the scheduler know what's
already taken.
"""

import contextlib
import csv
import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import List, Optional

from ..publishing.youtube_uploads_listing import list_uploads
from ._file_lock import make_blocking_file_lock
from .schedule_constants import (
    EST,
    LONG_FORM_DAYS,  # noqa: F401 — re-exported for test_schedule_constants single-source contract
    LONG_FORM_TIME,  # noqa: F401 — re-exported for test_schedule_constants single-source contract
    PHOTO_POST_TIME,
    SHORTS_TIMES,
)

logger = logging.getLogger(__name__)

# Explicit denylist of obvious TEST-origin entry markers that
# must never pollute the persisted real ledger. Tests historically added
# `ContentEntry`s whose title/project_name carried these sentinels and then
# round-tripped them through a saved ledger file. The patterns are kept
# DELIBERATELY narrow — exact-match titles + a `test_`/`test-` project_name
# prefix — so a legitimate real entry (e.g. a video genuinely titled
# "Testing the Brakes on the N86") is NOT dropped. Matching is
# case-insensitive on the stripped value.
_TEST_TITLE_DENYLIST = frozenset({"test location", "test", "test title"})
_TEST_PROJECT_PREFIXES = ("test_", "test-")


def _is_test_entry(entry: "ContentEntry") -> bool:
    """True iff ``entry`` is an obvious test-origin entry (see denylist above)."""
    title = (entry.title or "").strip().lower()
    if title in _TEST_TITLE_DENYLIST:
        return True
    project = (entry.project_name or "").strip().lower()
    return project.startswith(_TEST_PROJECT_PREFIXES)


def _default_ledger_path() -> Path:
    """Return the canonical ledger path: ``<seo.storage_path>/.content_ledger.json``.

    Single-store rule: the ledger lives in the SEO storage
    dir — the same resolution ``phase_d_scheduling`` and ``longform_metadata``
    use — NEVER under ``gms_output_dir()``. The old output-dir default spawned
    orphan ledgers, so collision avoidance read/wrote a different file than
    the one LFs were actually booked into.
    """
    from ..config import get_config

    return Path(get_config().seo.storage_path) / ".content_ledger.json"


# Per-thread reentrancy depth per resolved ledger path. flock associates the
# lock with the open file description, so a second acquire in the same thread
# (record_confirmed_upload's outer lock, then save()'s inner one) would open a
# fresh fd and self-deadlock. We count depth so only the outermost frame takes
# the real flock; nested acquires are no-ops.
_LEDGER_LOCK_STATE = threading.local()

# Two concurrent drives (the documented two-USA-drives mode) would otherwise
# lose-update each other's confirmed booking: both load the same snapshot,
# each saves back only its own single addition, last writer wins and the
# other booking is lost — permanently, since reconcile never re-adds it from
# YouTube. See :func:`._file_lock.make_blocking_file_lock` for the lock's
# mechanism (blocking flock on a ``.lock`` sidecar, per-thread reentrant).
_ledger_lock = make_blocking_file_lock(_LEDGER_LOCK_STATE)


def record_confirmed_upload(
    entries: List["ContentEntry"], ledger_path: Optional[Path] = None
) -> bool:
    """Book confirmed-upload entries (transactional booking): entry
    exists ⟺ upload confirmed — called strictly AFTER confirmation (YouTube
    video id / Metricool media gate), never at plan time or under
    ``--skip-upload``. Best-effort: a confirmed upload must never be failed by
    bookkeeping, so any error logs ``LEDGER-BOOKING-FAILED`` and returns
    ``False`` (raising post-upload would release already-shipped content);
    returns ``True`` when the booking persisted, so a caller can act on a miss.
    Resume-safe: ``video_id`` dups skipped; shorts dedup on ``add()`` slot key.
    """
    try:
        path = ledger_path or _default_ledger_path()
        # Lock the whole load→add→save so a concurrent drive can't lose-update
        # this booking (the two-USA-drives no-double-book contract).
        with _ledger_lock(path):
            ledger = ContentLedger.load(path)
            existing_ids = {e.video_id for e in ledger.entries if e.video_id}
            for entry in entries:
                if not (entry.video_id and entry.video_id in existing_ids):
                    ledger.add(entry)
            ledger.save(path)
        return True
    except Exception:
        msg = "LEDGER-BOOKING-FAILED: %d confirmed entries not recorded — uploads are live"
        logger.error(msg, len(entries), exc_info=True)
        return False


# Bound the uploads-playlist scan, by argument to the shared lister, at
# the 100 newest uploads (~5 read calls/run) — future-scheduled videos are always the most recent uploads.
_CHANNEL_NEWEST_BOUND = 100  # explicit bound passed to the shared lister
_SLOT_MATCH_TOLERANCE_SEC = 3600  # anonymous slot ↔ YouTube publishAt: ±1h
_IN_FLIGHT_GRACE = timedelta(hours=24)  # never drop a slot within the next 24h


def _video_publish_dt(item: dict) -> Optional[datetime]:
    """``status.publishAt`` (scheduled) else ``snippet.publishedAt``, as EST."""
    iso = (item.get("status") or {}).get("publishAt") or (item.get("snippet") or {}).get(
        "publishedAt"
    )
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(EST)
    except (ValueError, TypeError):
        return None


def _list_channel_publish_times(service) -> List[datetime]:
    """Publish times of the channel's most recent uploads (read-only, capped)."""
    times: List[datetime] = []
    for u in list_uploads(service, limit=_CHANNEL_NEWEST_BOUND):
        item = {"status": {"publishAt": u.publish_at}, "snippet": {"publishedAt": u.published_at}}
        if dt := _video_publish_dt(item):
            times.append(dt)
    return times


def reconcile_ledger_with_youtube(ledger: "ContentLedger", service, *, now=None) -> dict:
    """Reconcile ``ledger`` against YouTube — **YouTube is truth, the ledger is
    a cache**. Read-only; all API reads complete before any
    mutation. Policy: ``video_id`` entry kept iff its video exists (slot
    rewritten to the actual ``publishAt`` when it moved), dropped when gone;
    anonymous future ``scheduled`` longform entry with no channel video within
    ±1h of its slot is a phantom → dropped; anonymous PAST and shorts entries
    untouched; nothing whose slot is within the next 24h is ever dropped
    (in-flight grace). Saves only when something changed. Raises on API
    failure — the caller traps to one WARN; never blocks a run.
    """
    now = now or datetime.now(EST)
    grace_cutoff = now + _IN_FLIGHT_GRACE
    # ── API reads first (a failure raises here, before any mutation) ──
    ids = sorted({e.video_id for e in ledger.entries if e.video_id})
    live: dict = {}
    for i in range(0, len(ids), 50):
        resp = (
            service.videos()
            .list(part="status,snippet", id=",".join(ids[i : i + 50]), maxResults=50)
            .execute()
        )
        for item in resp.get("items", []) or []:
            live[item["id"]] = _video_publish_dt(item)
    anon_candidates = [
        e
        for e in ledger.entries
        if not e.video_id
        and e.content_type == "longform"
        and e.status == "scheduled"
        and e.datetime_est >= grace_cutoff
    ]
    channel_times = _list_channel_publish_times(service) if anon_candidates else []
    # ── apply ──
    kept: List[ContentEntry] = []
    dropped: List[ContentEntry] = []
    updated = 0
    for e in ledger.entries:
        if e.video_id:
            if e.video_id not in live:
                (kept if now <= e.datetime_est < grace_cutoff else dropped).append(e)
                continue
            pub = live[e.video_id]
            if pub is not None:
                slot = (pub.strftime("%Y-%m-%d"), pub.strftime("%H:%M:%S"))
                if slot != (e.date, e.time):
                    e.date, e.time = slot
                    updated += 1
            kept.append(e)
        elif e in anon_candidates and not any(
            abs((t - e.datetime_est).total_seconds()) <= _SLOT_MATCH_TOLERANCE_SEC
            for t in channel_times
        ):
            dropped.append(e)
        else:
            kept.append(e)
    if dropped or updated:
        ledger.entries = kept
        for e in dropped:
            logger.warning(
                "LEDGER-RECONCILE: dropped %s %s [%s] video_id=%s — no matching YouTube video",
                *(e.date, e.time, e.content_type, e.video_id or "<anonymous>"),
            )
        ledger.save()
    logger.info("LEDGER-RECONCILE: kept=%d dropped=%d updated=%d", len(kept), len(dropped), updated)
    return {"kept": len(kept), "dropped": len(dropped), "updated": updated}


def next_free_slot(
    ledger: "ContentLedger",
    config,
    region: Optional[str],
    content_type: str = "longform",
    *,
    after: Optional[datetime] = None,
    publish_day: Optional[str] = None,
    include_today: bool = True,
    block_cross_region: bool = False,
    horizon_days: int = 120,
    slot_time: Optional[dt_time] = None,
    min_lead: Optional[timedelta] = None,
    any_day: bool = False,
) -> datetime:
    """Return the next FREE long-form publish slot for ``region`` — the
    operator contract ("publish to the next free Ireland Sunday") as one
    tested function.

    Cadence comes from the LOADED config (``long_form_days``/``long_form_time``
    accessors — the values ``--lf-day``/``--lf-time``/region resolution already
    wrote into ``config.drive``), never the import-time Wed-11:00 module grid;
    ``config=None`` falls back to the module attrs (``apply_config``/monkeypatch
    contract). ``publish_day`` narrows the candidate days; ``include_today=False``
    reproduces the drive scheduler's never-today base-date rule. Occupancy is
    DATE-granular over long-form ledger entries (booked entries carry the REAL
    ``publishAt`` time, so exact (date, time) keys missed them — the phantom-grid
    double-booking class) and region-aware: same region blocks; a DIFFERENT region
    does not (independent audiences on disjoint days) unless
    ``block_cross_region=True``. That "disjoint days" premise is specific to
    REGULAR long-form cadence, where each region's designed weekday differs
    (USA Tue/Fri, Ireland Sun) — it does NOT hold for every content type built
    on this function; see :func:`next_free_superlong_slot`, whose regions
    collide on the SAME day and which defaults ``block_cross_region=True``
    for exactly that reason. A legacy anonymous entry — and a ``region=None``
    caller — gets the conservative all-regions view. ``slot_time`` overrides the
    flat ``long_form_time(config)`` default — a caller with its own designed
    time (e.g. a per-day ``publish_slots`` entry) passes it so the "already
    occurred" check and the returned datetime both use the REAL time, never a
    value patched on after the search already validated a different one.
    ``min_lead`` rejects a candidate closer than that runway to ``now`` — a slot
    resolved right before a long, large upload begins needs enough lead that the
    upload itself can't overrun the slot it was given.
    ``any_day=True`` searches every weekday (candidate days are unnarrowed) —
    used when no fixed publish day applies (see
    ``schedule_constants.resolve_superlong_publish_day``'s rule 4); it takes
    precedence over ``publish_day``/cadence-config narrowing.
    """
    if content_type != "longform":
        raise ValueError(f"next_free_slot implements long-form slots only, got {content_type!r}")
    from .seo_slots import next_free_slot as _ask_seo_manager

    ledger.reload()
    if publish_day:
        from seo_manager.core.profile import DAY_NAMES

        if publish_day.lower()[:3] not in DAY_NAMES:
            raise ValueError(f"Invalid publish_day {publish_day!r}")
        publish_day = publish_day.lower()[:3]
    if slot_time is None:
        from .schedule_constants import long_form_time

        slot_time = long_form_time(config)
    return _ask_seo_manager(
        ledger,
        config,
        region,
        publish_day=publish_day,
        any_day=any_day,
        slot_time=slot_time,
        after=after,
        include_today=include_today,
        block_cross_region=block_cross_region,
        horizon_days=horizon_days,
        min_lead=min_lead,
    )


# A superlong part upload runs to a 4h cap (``superlong_part_max_duration_sec``
# default 14400) and moves tens of GB — a slot resolved with a thin lead can be
# overrun by the upload itself before it ever reaches YouTube. 6h clears the cap with margin.
SUPERLONG_MIN_PUBLISH_LEAD = timedelta(hours=6)


def next_free_superlong_slot(
    ledger: "ContentLedger",
    config,
    region: Optional[str],
    *,
    after: Optional[datetime] = None,
    include_today: bool = True,
    block_cross_region: bool = True,
    horizon_days: int = 120,
    project_name: Optional[str] = None,
    title: Optional[str] = None,
) -> datetime:
    """Return the next FREE superlong-part publish slot for ``region``.

    A superlong part is a flagship-channel upload, so it draws from the
    SAME ledger-checked long-form occupancy machinery as :func:`next_free_slot`
    — no parallel scheduler, no second ledger. The only difference is which
    weekday is a candidate:
    ``schedule_constants.resolve_superlong_publish_day(config, region, ledger=ledger,
    project_name=project_name, title=title)`` — an explicit operator override, else
    the region's own ``config.drive.superlong_publish_day_by_region`` entry, else
    (when ``project_name``/``title`` identify the part) the weekday of the most
    recent RELATED superlong part already booked, else ``None`` — meaning no
    fixed day; every weekday is searched for the next genuinely open slot —
    rather than the region's regular Waypost long-form days. ``project_name``/
    ``title`` default to ``None``, the safe default for a caller with no
    per-video identity to give the resolver (falls back to the flat
    ``superlong_publish_day`` field, same as before this function grew rules
    3/4).

    Unlike :func:`next_free_slot`'s own default, ``block_cross_region``
    defaults to ``True`` here: a regular LF's regions sit on genuinely
    disjoint weekdays, but every region resolves the SAME superlong day
    absent an explicit per-region override, so an Ireland part and a USA
    part are competing for the SAME Saturday, not independent audiences on
    independent days. Blocking makes the superlong slots ONE shared weekly
    queue — whichever region has content ready fills the next free one, in
    processing order, so a region winding down never leaves its day
    permanently reserved and idle for a region still producing. Because occupancy is checked against the SAME ledger under
    the SAME ``content_type="longform"``, a part's slot can also never land
    on a day already booked by a regular LF.

    The slot TIME is the day's own designed ``publish_slots[day].lf_time``
    (falling back to Waypost's ``long_form_time``, via
    ``schedule_constants.resolve_lf_time``) — passed into the search as
    ``slot_time`` rather than stamped on afterward, so the search validates
    the datetime it actually returns. When no fixed day applies (rule 4 — an
    ``any_day`` search), a single ``slot_time`` can't track each candidate
    day's own designed slot, so Waypost's ``long_form_time`` is used
    uniformly (``resolve_lf_time`` with a ``None`` day already falls back to
    it) — a documented simplification for the undesigned-day case.
    """
    from .schedule_constants import resolve_lf_time, resolve_superlong_publish_day

    if config is not None:
        resolved_day = resolve_superlong_publish_day(
            config, region, ledger=ledger, project_name=project_name, title=title
        )
    else:
        resolved_day = "sat"  # no config to resolve from — unchanged legacy fallback
    slot_time = None
    if config is not None:
        hour, minute = (int(p) for p in resolve_lf_time(config, resolved_day, None).split(":")[:2])
        slot_time = dt_time(hour, minute)
    return next_free_slot(
        ledger,
        config,
        region,
        content_type="longform",
        after=after,
        publish_day=resolved_day,
        any_day=resolved_day is None,
        include_today=include_today,
        block_cross_region=block_cross_region,
        slot_time=slot_time,
        horizon_days=horizon_days,
        min_lead=SUPERLONG_MIN_PUBLISH_LEAD,
    )


@dataclass
class ContentEntry:
    """A single piece of scheduled or published content."""

    date: str  # YYYY-MM-DD
    time: str  # HH:MM:SS
    content_type: str  # "longform", "shorts", or "photo"
    title: str = ""
    project_name: str = ""
    platform: str = "all"  # "all", "youtube", "tiktok", etc.
    status: str = "scheduled"  # "scheduled", "published", "draft"
    source: str = ""  # "metricool_csv", "pipeline", "manual"
    source_file: str = ""  # path to the CSV or metadata file
    video_id: Optional[str] = None  # identity fields, defaulted so
    region: Optional[str] = None  # older ledger files load unchanged

    @property
    def datetime_str(self) -> str:
        return f"{self.date} {self.time}"

    @property
    def datetime_est(self) -> datetime:
        return datetime.strptime(f"{self.date} {self.time}", "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=EST
        )


@dataclass
class ContentLedger:
    """Tracks all GoMoveShift content scheduling."""

    entries: List[ContentEntry] = field(default_factory=list)
    last_updated: str = ""
    last_synced_csvs: List[str] = field(default_factory=list)
    # Path this instance was loaded from / last saved to. ``reload()`` and
    # ``save()`` with no argument reuse it, so one instance can never read
    # and write two different files (a divergence regenerator once did this:
    # a no-arg ``reload()`` silently re-merged the orphan default store into
    # an explicitly-loaded ledger). Never serialized.
    _path: Optional[Path] = field(default=None, repr=False, compare=False)

    def add(self, entry: ContentEntry):
        """Add an entry idempotently, keyed on ``(date, time, content_type)``.

        The old body unconditionally overwrote an existing
        entry's ``title``/``project_name`` on every replay, so re-running or
        resuming the pipeline could non-idempotently clobber committed
        metadata with whatever the latest caller happened to pass. Rule now:

        * **Same key, no committed conflict** — a true re-add of identical
          content (or a same-source refresh) is allowed: status/source/
          source_file are refreshed and a blank incoming title/project_name
          inherits the existing value. This keeps legitimate lifecycle
          updates (a producer re-emitting its own slot) working.
        * **Same key, conflicting committed metadata from a DIFFERENT source**
          — keep the existing entry untouched and log a greppable
          ``LEDGER-CONFLICT`` WARN rather than silently clobber. The first
          writer of a slot owns its title/project_name; a different producer
          cannot overwrite them by reusing the slot key.
        """
        key = (entry.date, entry.time, entry.content_type)
        for existing in self.entries:
            if (existing.date, existing.time, existing.content_type) != key:
                continue
            # A committed field (title/project_name) conflicts when the
            # incoming value is non-blank AND differs from what's stored.
            # Blank incoming values never conflict (they inherit).
            title_conflict = bool(entry.title) and entry.title != existing.title
            project_conflict = bool(entry.project_name) and (
                entry.project_name != existing.project_name
            )
            cross_source = (
                bool(entry.source) and bool(existing.source) and entry.source != existing.source
            )
            if (title_conflict or project_conflict) and cross_source:
                logger.warning(
                    "LEDGER-CONFLICT: %s %s [%s] already committed by source=%r "
                    "(title=%r, project=%r); refusing overwrite from source=%r "
                    "(title=%r, project=%r) — keeping existing.",
                    existing.date,
                    existing.time,
                    existing.content_type,
                    existing.source,
                    existing.title,
                    existing.project_name,
                    entry.source,
                    entry.title,
                    entry.project_name,
                )
                return
            # No cross-source committed conflict → idempotent refresh.
            existing.title = entry.title or existing.title
            existing.project_name = entry.project_name or existing.project_name
            existing.status = entry.status
            existing.source = entry.source
            existing.source_file = entry.source_file
            return
        self.entries.append(entry)

    def reload(self, path: Optional[Path] = None) -> None:
        """Re-read ledger from disk and merge entries.

        Entries found on disk that don't already exist in memory are added.
        This allows multiple processes to coordinate via the shared ledger file.

        With no ``path``, re-reads the file this instance was loaded from.
        An in-memory-only ledger (never loaded from disk) is a no-op —
        guessing a default here is exactly how an orphan store got silently
        merged into an explicitly-loaded ledger.
        """
        path = path or self._path
        if path is None:
            return
        self._path = path
        if not path.exists():
            return

        try:
            raw = path.read_text()
        except OSError:
            # Same refusal as ContentLedger.load(): a permission/IO error is
            # operational, not "nothing to merge" — the caller must be able
            # to tell a successful merge from a failed one.
            raise
        try:
            data = json.loads(raw)
            disk_entries = [ContentEntry(**e) for e in data.get("entries", [])]
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            # Do NOT silently keep the in-memory ledger as-is: a caller that
            # can't tell a failed reload from a successful one believes
            # cross-process coordination happened when it didn't. Same
            # refusal as the JSONDecodeError branch in load().
            logger.error(f"  Ledger reload corrupt at {path}: {e} — refusing to merge")
            raise
        existing_keys = {(e.date, e.time, e.content_type) for e in self.entries}
        for entry in disk_entries:
            key = (entry.date, entry.time, entry.content_type)
            if key not in existing_keys:
                self.entries.append(entry)
                existing_keys.add(key)
        self.last_updated = data.get("last_updated", self.last_updated)

    def get_scheduled(self, after: Optional[datetime] = None) -> List[ContentEntry]:
        """Get all scheduled (future) content."""
        now = after or datetime.now(EST)
        return [e for e in self.entries if e.status == "scheduled" and e.datetime_est > now]

    def get_occupied_slots(self, after: Optional[datetime] = None) -> set:
        """Get set of (date, time) tuples that are already taken."""
        scheduled = self.get_scheduled(after)
        return {(e.date, e.time) for e in scheduled}

    def get_next_available_slot(
        self,
        content_type: str = "longform",
        after: Optional[datetime] = None,
        region: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Find the next available publish slot that isn't already taken.

        Long-form slots delegate to :func:`next_free_slot` (config-cadence,
        region-aware, date-granular occupancy) with ``config=None`` so the
        module-attr ``apply_config``/monkeypatch contract is preserved;
        ``region=None`` keeps the legacy conservative any-region view.
        Returns a dict with date, time, day_name, datetime_est — or None.
        """
        self.reload()
        now = after or datetime.now(EST)
        occupied = self.get_occupied_slots(now)

        if content_type == "longform":
            try:
                slot_dt = next_free_slot(self, None, region, after=now)
            except RuntimeError:
                return None
            return {
                "date": slot_dt.strftime("%Y-%m-%d"),
                "time": slot_dt.strftime("%H:%M:%S"),
                "day_name": slot_dt.strftime("%A"),
                "datetime_est": slot_dt.isoformat(),
                "content_type": "longform",
            }
        elif content_type == "photo":
            for days_ahead in range(1, 15):
                candidate_date = (now + timedelta(days=days_ahead)).date()
                slot_dt = datetime.combine(candidate_date, PHOTO_POST_TIME, tzinfo=EST)
                if slot_dt <= now:
                    continue
                slot_key = (slot_dt.strftime("%Y-%m-%d"), slot_dt.strftime("%H:%M:%S"))
                if slot_key not in occupied:
                    return {
                        "date": slot_key[0],
                        "time": slot_key[1],
                        "day_name": slot_dt.strftime("%A"),
                        "datetime_est": slot_dt.isoformat(),
                        "content_type": "photo",
                    }
        else:
            for days_ahead in range(0, 15):
                candidate_date = (now + timedelta(days=days_ahead)).date()
                for slot_time in SHORTS_TIMES:
                    slot_dt = datetime.combine(candidate_date, slot_time, tzinfo=EST)
                    if slot_dt <= now:
                        continue
                    slot_key = (slot_dt.strftime("%Y-%m-%d"), slot_dt.strftime("%H:%M:%S"))
                    if slot_key not in occupied:
                        return {
                            "date": slot_key[0],
                            "time": slot_key[1],
                            "day_name": slot_dt.strftime("%A"),
                            "datetime_est": slot_dt.isoformat(),
                            "content_type": "shorts",
                        }

        return None

    def save(self, path: Optional[Path] = None):
        """Save ledger to JSON under an exclusive inter-process file lock.

        The lock (a ``.lock`` sidecar beside the ledger) is held across the
        write and — reentrantly, via :func:`_ledger_lock` — is reused by
        :func:`record_confirmed_upload` to wrap the whole load→add→save, so two
        concurrent drives can't lose-update each other's booking.

        With no ``path``, writes back to the file this instance was loaded
        from; a never-loaded ledger falls back to the canonical default.
        """
        path = path or self._path or _default_ledger_path()
        self._path = path
        with _ledger_lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            self.last_updated = datetime.now(EST).isoformat()

            # Drop obvious test-origin entries before persisting
            # so a stray fixture entry can't pollute the real shared ledger
            # file. In-memory ledger is untouched (we filter the serialized copy).
            persistable = [e for e in self.entries if not _is_test_entry(e)]
            dropped = len(self.entries) - len(persistable)
            if dropped:
                logger.warning(
                    "LEDGER-TEST-FILTER: dropped %d test-origin entr%s before save → %s",
                    dropped,
                    "y" if dropped == 1 else "ies",
                    path,
                )

            data = {
                "last_updated": self.last_updated,
                "last_synced_csvs": self.last_synced_csvs,
                "entries": [asdict(e) for e in sorted(persistable, key=lambda e: e.datetime_str)],
            }
            tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    f.write(json.dumps(data, indent=2))
                os.replace(tmp_path, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise
            # mkstemp creates 0o600; clobber so group members (users sharing a
            # group via a setgid parent directory) keep write access across saves.
            # Without this, every save flips ownership-effective perms and the
            # other user's next pipeline run hits PermissionError on .lock or
            # the ledger itself.
            with contextlib.suppress(OSError):
                os.chmod(path, 0o664)
            logger.info(f"  Ledger saved: {len(persistable)} entries → {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ContentLedger":
        """Load ledger from JSON, or create empty if not found.

        The resolved path is pinned on the instance so no-arg ``reload()``
        and ``save()`` always operate on the same file.
        """
        path = path or _default_ledger_path()
        if not path.exists():
            return cls(_path=path)

        try:
            raw = path.read_text()
        except OSError:
            # Permission / IO errors are operational — do NOT swallow them as
            # "empty ledger". Silent degradation here once let an unreadable
            # ledger cause the pipeline to run on an empty in-memory ledger while logging "0 entries".
            raise
        try:
            data = json.loads(raw)
            entries = [ContentEntry(**e) for e in data.get("entries", [])]
        except json.JSONDecodeError as e:
            # Do NOT degrade to an empty ledger: an empty in-memory ledger
            # looks like the whole calendar is free, and the very next
            # booking (record_confirmed_upload) then os.replace()s the real
            # file down to just that one entry — destroying every historical
            # and future-scheduled booking. Same refusal as the OSError
            # branch above; log first so the corruption is diagnosable.
            logger.error(f"  Ledger JSON corrupt at {path}: {e} — refusing to load")
            raise
        return cls(
            entries=entries,
            last_updated=data.get("last_updated", ""),
            last_synced_csvs=data.get("last_synced_csvs", []),
            _path=path,
        )

    def ingest_metricool_csv(self, csv_path: Path, project_name: str = ""):
        """
        Import scheduled content from a Metricool CSV.

        Reads the Date, Time, and YouTube Video Title columns to populate
        the ledger with shorts entries.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            logger.warning(f"  CSV not found: {csv_path}")
            return 0

        csv_str = str(csv_path)
        if csv_str in self.last_synced_csvs:
            logger.info(f"  CSV already synced: {csv_path.name}")
            return 0

        count = 0
        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    date = row.get("Date", "").strip()
                    time = row.get("Time", "").strip()
                    title = row.get("YouTube Video Title", "").strip()

                    if not date or not time:
                        continue

                    self.add(
                        ContentEntry(
                            date=date,
                            time=time,
                            content_type="shorts",
                            title=title,
                            project_name=project_name,
                            platform="all",
                            status="scheduled",
                            source="metricool_csv",
                            source_file=csv_str,
                        )
                    )
                    count += 1

            self.last_synced_csvs.append(csv_str)
            logger.info(f"  Ingested {count} entries from {csv_path.name}")

        except Exception as e:
            logger.warning(f"  Failed to read CSV {csv_path}: {e}")

        return count

    def ingest_all_metricool_csvs(self, output_base: Optional[Path] = None):
        """Scan all project output directories for Metricool CSVs and ingest them.

        ``output_base`` defaults to the runtime-resolved ``gms_output_dir()``
        (env-aware, no import-time default).
        """
        if output_base is None:
            from .. import paths as _paths

            output_base = _paths.gms_output_dir()
        total = 0
        for project_dir in sorted(output_base.iterdir()):
            if not project_dir.is_dir():
                continue
            for csv_file in sorted(project_dir.glob("metricool_*RELEASE*.csv")):
                total += self.ingest_metricool_csv(csv_file, project_name=project_dir.name)
            # Also check non-RELEASE CSVs if no RELEASE version exists
            release_exists = any(project_dir.glob("metricool_*RELEASE*.csv"))
            if not release_exists:
                for csv_file in sorted(project_dir.glob("metricool_*FINAL*.csv")):
                    total += self.ingest_metricool_csv(csv_file, project_name=project_dir.name)

        return total

    def summary(self) -> str:
        """Human-readable summary of the ledger."""
        now = datetime.now(EST)
        upcoming = self.get_scheduled(now)
        past = [e for e in self.entries if e.datetime_est <= now]

        lines = [f"Content Ledger: {len(self.entries)} total entries"]
        lines.append(f"  Published/past: {len(past)}")
        lines.append(f"  Upcoming: {len(upcoming)}")

        if upcoming:
            lines.append("  Next scheduled:")
            for e in upcoming[:5]:
                lines.append(f"    {e.date} {e.time} [{e.content_type}] {e.title[:40]}")
            if len(upcoming) > 5:
                lines.append(f"    ... and {len(upcoming) - 5} more")

        return "\n".join(lines)
