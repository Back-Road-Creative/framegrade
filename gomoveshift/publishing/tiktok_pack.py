"""The TikTok Studio pack: TikTok shorts handed to the owner as files.

The TikTok app is unaudited and its review was denied
(``docs/social-publishing-tiktok.md``), so Direct Post is unavailable. Inbox
uploads report ``SEND_TO_USER_INBOX`` yet never reach the owner's inbox.
Instead each TikTok short is written to a folder on the PC,
the owner schedules the week in TikTok Studio (free; it schedules 15 minutes
to 10 days ahead), and the delivery parks ``awaiting_manual_publish`` until the
owner confirms the post in the console (mark-published). The read-only
``reconcile_awaiting_manual`` never closes a pack-parked row: it polls only
rows that carry a ``platform_upload_id``, and a pack-parked row has none.

Honesty rule: packing is NOT publishing. :func:`pack_and_park` never sets
``platform_upload_id`` or ``platform_post_id`` and never transitions to
``published``.

Layout of the pack directory (``$GMS_TIKTOK_PACK_DIR``, default
``tiktok-pack`` under the data home)::

    2026-01-02-1000-ExampleTown-41.mp4
    captions.txt        # UTF-8, Windows line endings, one block per file

Two entry points, one packer. :func:`sweep_queued` runs before the drain
(``cli/handlers/dispatch_queue.py``) and packs everything inside TikTok
Studio's scheduling horizon; ``handlers.make_tiktok_handler`` calls the same
:func:`pack_and_park` for a queued row the sweep missed.

The sweep reuses the drain's gates rather than keeping its own: the row
selection is ``dispatch._select_due`` (pause, shadow, retry backoff,
blackout, quiet window, daily cap) and the privacy attestation is the same
fail-closed check, so a short the drain would refuse is never copied to the
PC either.

A short the owner already copied by hand is not copied twice: a ``.mp4`` in
the pack directory with the source's size and SHA-256 counts as packed under
whatever name it has, and the row is parked without a second file or a
caption block.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional
from zoneinfo import ZoneInfo

from ..paths import gms_data_home
from .dispatch import TransientMediaError, _release_own_lease, _select_due, privacy_hold_reason
from .models import ShortPublishRecord
from .queue import (
    AWAITING_MANUAL_PUBLISH_NEXT_ATTEMPT,
    NEXT_ATTEMPT_AWAITING_MANUAL,
    PUBLISHED_AT_PLATFORM_RECONCILED,
    STATUS_AWAITING_MANUAL_PUBLISH,
    acquire_lease,
    is_platform_paused,
    transition,
)

logger = logging.getLogger(__name__)

ENV_PACK_DIR = "GMS_TIKTOK_PACK_DIR"
CAPTIONS_NAME = "captions.txt"
#: TikTok Studio schedules up to 10 days ahead; one day of margin.
SWEEP_HORIZON_DAYS = 9
_SWEEP_LEASE_SECONDS = 900
#: A slot closer than this to the moment of packing (or already past) gets
#: "Post now" instead of a schedule time. TikTok Studio's own floor is 15
#: minutes; this is a margin so the owner is not racing it, not TikTok's rule.
MIN_SCHEDULE_LEAD = timedelta(minutes=20)
#: ``/mnt/<letter>`` directories exist whether or not the drive is mounted;
#: writing under an unmounted one lands on the Linux root disk.
_DRIVE_MOUNT_ROOT = Path("/mnt")
#: errnos a dead or stale mount returns (EACCES is how a half-dead drvfs
#: mount often reports itself and cannot be told from a real denial here).
_DEAD_MOUNT_ERRNOS = frozenset(
    {errno.EIO, errno.ENODEV, errno.ENXIO, errno.ESTALE, errno.ENOTCONN, errno.EACCES}
)
_SEPARATOR = "-" * 40
#: The place is the text before the first of these.
_PLACE_SPLIT = re.compile(r" — | - |:|,")
#: A name ``pack_file_name`` wrote; the group is its delivery id.
_PACKED_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{4}-[A-Za-z0-9]+-(\d+)\.mp4$")

RecordLoader = Callable[[str, int], ShortPublishRecord]
PathLoader = Callable[[ShortPublishRecord], Path]


@dataclass
class PackSweepResult:
    considered: int = 0
    packed: int = 0
    #: Rows parked because a byte-identical file was already in the pack.
    already_packed: int = 0
    skipped_leased: int = 0
    #: Rows left queued by the privacy gate; the reasons are in ``holds``.
    held_privacy: int = 0
    paused: bool = False
    errors: List[str] = field(default_factory=list)
    holds: List[str] = field(default_factory=list)


class PackOutcome(NamedTuple):
    path: Path
    #: True when the file was already in the pack (not copied, no caption added).
    already_packed: bool


def default_pack_dir() -> Path:
    """The pack directory when ``$GMS_TIKTOK_PACK_DIR`` is unset."""
    return gms_data_home() / "tiktok-pack"


def resolve_pack_dir() -> Path:
    return Path(os.environ.get(ENV_PACK_DIR, "").strip() or default_pack_dir())


def _parse_utc(value: object) -> datetime:
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _slot(scheduled_at: Optional[str], record: ShortPublishRecord) -> datetime:
    """The slot in the display timezone: the delivery row's own time, else the
    record's TikTok-specific time, else its shared time."""
    schedule = record.schedule
    chosen = (
        scheduled_at
        or schedule.scheduled_at_utc_by_platform.get("tiktok")
        or schedule.scheduled_at_utc
    )
    return _parse_utc(chosen).astimezone(ZoneInfo(schedule.display_timezone))


def place_for(record: ShortPublishRecord) -> str:
    """The TikTok title's leading place phrase as ASCII CamelCase
    ("Example Town — roads" -> "ExampleTown"); ``Short<index>`` when empty."""
    payload = record.payloads.tiktok
    lead = _PLACE_SPLIT.split(payload.title if payload else "", maxsplit=1)[0]
    ascii_lead = unicodedata.normalize("NFKD", lead).encode("ascii", "ignore").decode("ascii")
    words = re.findall(r"[A-Za-z0-9]+", ascii_lead)
    return "".join(w[0].upper() + w[1:] for w in words) or f"Short{record.short_index}"


def pack_file_name(
    delivery_id: int, scheduled_at: Optional[str], record: ShortPublishRecord
) -> str:
    slot = _slot(scheduled_at, record)
    return f"{slot:%Y-%m-%d-%H%M}-{place_for(record)}-{delivery_id}.mp4"


def _schedule_line(slot: datetime, now: datetime) -> str:
    hour = slot.hour % 12 or 12
    meridiem = "AM" if slot.hour < 12 else "PM"
    when = f"{slot:%a} {slot:%b} {slot.day}, {hour}:{slot:%M} {meridiem} {slot:%Z}"
    if slot < now + MIN_SCHEDULE_LEAD:
        return f"Schedule: Post now (the slot was {when})"
    return f"Schedule: {when}"


def _caption_block(name: str, slot: datetime, record: ShortPublishRecord, now: datetime) -> str:
    payload = record.payloads.tiktok
    caption = (payload.caption or payload.title).replace("\r\n", "\n")
    lines = [name, _schedule_line(slot, now), "", *caption.split("\n"), "", _SEPARATOR]
    return "\r\n".join(lines) + "\r\n"


def _check_pack_parent(pack_dir: Path) -> None:
    """A missing parent, or a ``/mnt/<drive>`` parent with nothing mounted on
    it, means the drive is offline; writing anything there would land on the
    Linux root disk. The default directory under the data home is ours to
    create, so it is never an offline-drive case."""
    if pack_dir == default_pack_dir():
        return
    parent = pack_dir.parent
    if not parent.is_dir():
        raise TransientMediaError(
            f"TikTok pack directory's parent {parent} does not exist "
            "(drive offline?) -- nothing was written"
        )
    if parent.parent == _DRIVE_MOUNT_ROOT and not os.path.ismount(parent):
        raise TransientMediaError(
            f"{parent} is not mounted (drive offline?) -- nothing was written"
        )


def _ensure_pack_dir(pack_dir: Path) -> None:
    """Create the pack directory itself, and only when its parent is there."""
    _check_pack_parent(pack_dir)
    pack_dir.mkdir(parents=pack_dir == default_pack_dir(), exist_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_identical(pack_dir: Path, source: Path, skip: Path) -> Optional[Path]:
    """An ``.mp4`` in the pack, other than ``skip``, with the source's size and
    SHA-256 (placed by hand, or by an earlier pack under another name). Only a
    file of the source's size is read; the source is hashed once, lazily."""
    size = source.stat().st_size
    source_hash: Optional[str] = None
    for candidate in sorted(pack_dir.iterdir()):
        if candidate == skip or candidate.suffix.lower() != ".mp4" or not candidate.is_file():
            continue
        if candidate.stat().st_size != size:
            continue
        if source_hash is None:
            source_hash = _sha256(source)
        if _sha256(candidate) == source_hash:
            return candidate
    return None


def _copy_into_pack(source: Path, final: Path) -> None:
    if final.is_file() and final.stat().st_size == source.stat().st_size:
        return
    # Its own name per copy: two workers packing the same short must not
    # interleave writes into one shared ``.part`` file.
    part = final.with_name(f"{final.name}.part.{os.getpid()}-{uuid.uuid4().hex}")
    try:
        shutil.copyfile(source, part)
        os.replace(part, final)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def _append_caption(captions: Path, name: str, block: str) -> None:
    if captions.is_file():
        existing = captions.read_text(encoding="utf-8")
        if any(line.strip() == name for line in existing.splitlines()):
            return
    with open(captions, "a", encoding="utf-8", newline="") as handle:
        handle.write(block)


def pack_row(
    conn,
    delivery_id: int,
    record: ShortPublishRecord,
    source: Path,
    *,
    pack_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> PackOutcome:
    """Write ``source`` (the H.264 file) and its caption block into the pack,
    then park the delivery ``awaiting_manual_publish``.

    A byte-identical ``.mp4`` already in the pack under another name (the
    owner copied it by hand) is not copied again and gets no caption block:
    the row is parked and ``already_packed`` is True. When that file is one
    the packer wrote for another delivery of the same short, an ``UPDATED``
    block with this row's slot and caption is added under its name. Safe to
    repeat: an
    identical file at this row's own name is not copied again and a name
    already in ``captions.txt`` gets no second block.

    Raises :class:`TransientMediaError` and writes nothing when the pack
    directory's parent is missing or unmounted, or the drive answers with a
    dead-mount errno. Never sets a platform id and never reaches
    ``published``; parking clears any ``platform_upload_id`` left by an
    earlier inbox upload, so reconcile never polls a stale id."""
    payload = record.payloads.tiktok
    if payload is None:
        raise ValueError(f"short_index={record.short_index} carries no TikTok payload")
    row = conn.execute(
        "SELECT status, scheduled_at FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    if row is None:
        raise KeyError(delivery_id)

    now = now or datetime.now(timezone.utc)
    pack_dir = Path(pack_dir) if pack_dir is not None else resolve_pack_dir()
    name = pack_file_name(delivery_id, row["scheduled_at"], record)
    final = pack_dir / name
    try:
        _ensure_pack_dir(pack_dir)
        existing = _find_identical(pack_dir, Path(source), skip=final)
        if existing is not None:
            final = existing
            packed_for = _PACKED_NAME.match(existing.name)
            if packed_for and int(packed_for.group(1)) != delivery_id:
                # Packed for an earlier delivery of the same short (a plan
                # change cancels and re-enqueues). A second copy could be
                # scheduled twice, so the slot and caption go under the file
                # that is already there.
                header = f"{existing.name} -- UPDATED for delivery {delivery_id}"
                _append_caption(
                    pack_dir / CAPTIONS_NAME,
                    header,
                    _caption_block(header, _slot(row["scheduled_at"], record), record, now),
                )
        else:
            _copy_into_pack(Path(source), final)
            _append_caption(
                pack_dir / CAPTIONS_NAME,
                name,
                _caption_block(name, _slot(row["scheduled_at"], record), record, now),
            )
    except OSError as exc:
        if exc.errno in _DEAD_MOUNT_ERRNOS:
            raise TransientMediaError(
                f"TikTok pack directory {pack_dir} is not answering "
                f"({type(exc).__name__}: {exc}) -- drive offline?"
            ) from exc
        raise

    if not transition(
        conn,
        delivery_id,
        row["status"],
        STATUS_AWAITING_MANUAL_PUBLISH,
        next_attempt_at=AWAITING_MANUAL_PUBLISH_NEXT_ATTEMPT,
        next_attempt_reason=NEXT_ATTEMPT_AWAITING_MANUAL,
        platform_upload_id=None,
        lease_owner=None,
        lease_expires_at=None,
    ):
        raise RuntimeError(
            f"delivery {delivery_id} left {row['status']!r} while it was being packed"
        )
    return PackOutcome(final, existing is not None)


def pack_and_park(
    conn,
    delivery_id: int,
    record: ShortPublishRecord,
    source: Path,
    *,
    pack_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> Path:
    """:func:`pack_row` for a caller that only needs the packed file's path."""
    return pack_row(conn, delivery_id, record, source, pack_dir=pack_dir, now=now).path


def sweep_queued(
    conn,
    record_loader: RecordLoader,
    rendition_path_loader: PathLoader,
    *,
    worker_id: str,
    pack_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
    privacy_attestations: Optional[Callable[[Dict[str, Any]], Optional[Mapping[str, Any]]]] = None,
) -> PackSweepResult:
    """Pack and park every queued TikTok delivery scheduled within TikTok
    Studio's horizon that the drain itself would be allowed to pick up.

    Selection is ``dispatch._select_due`` with the horizon as the schedule
    ceiling, so pause, shadow, retry backoff, brand blackout, the quiet window
    after a suppression alert and live daily caps all apply. The daily cap
    counts today's published rows against rows up to nine days out, which is
    conservative: it can only pack fewer files, never more.

    ``privacy_attestations`` is the drain's resolver (row -> attestation);
    a row without verified required classes, or whose attestation cannot be
    read, is left queued and reported in ``holds``, exactly as the drain holds
    it. A row another worker holds is skipped; a row's failure is recorded in
    the result and never raised."""
    result = PackSweepResult()
    if is_platform_paused(conn, "tiktok"):
        result.paused = True
        return result
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cutoff = now + timedelta(days=SWEEP_HORIZON_DAYS)

    due = _select_due(
        conn,
        now.isoformat(),
        platform="tiktok",
        status="queued",
        scheduled_through=cutoff.isoformat(),
    )
    result.considered = len(due)
    if not due:
        return result

    pack_dir = Path(pack_dir) if pack_dir is not None else resolve_pack_dir()
    try:
        _check_pack_parent(pack_dir)  # once: an offline drive fails every row alike
    except TransientMediaError as exc:
        result.errors.append(str(exc))
        return result

    for row in due:
        delivery_id = row["id"]
        if privacy_attestations is not None:
            try:
                hold = privacy_hold_reason(privacy_attestations(row))
            except Exception as exc:  # noqa: BLE001 -- unreadable == unattested
                hold = f"privacy attestation unreadable: {type(exc).__name__}: {exc}"
            if hold is not None:
                result.held_privacy += 1
                result.holds.append(f"tiktok#{delivery_id}: privacy hold: {hold}")
                logger.warning("tiktok pack: delivery %s held, privacy: %s", delivery_id, hold)
                continue
        expires = (now + timedelta(seconds=_SWEEP_LEASE_SECONDS)).isoformat()
        if not acquire_lease(conn, delivery_id, worker_id, expires, now=now.isoformat()):
            result.skipped_leased += 1
            continue
        try:
            record = record_loader(row["project_name"], row["short_index"])
            source = rendition_path_loader(record)
            outcome = pack_row(conn, delivery_id, record, source, pack_dir=pack_dir, now=now)
            if outcome.already_packed:
                result.already_packed += 1
            else:
                result.packed += 1
        except Exception as exc:  # noqa: BLE001 -- one row never stops the sweep
            result.errors.append(f"delivery {delivery_id}: {type(exc).__name__}: {exc}")
            # Give the lease back so the drain (or the next sweep) can retry,
            # whatever the row became meanwhile; never touch another worker's.
            _release_own_lease(conn, delivery_id, worker_id)
    return result


#: A listed video belongs to a row when its ``create_time`` is within this of
#: the row's slot. Wide on purpose: the owner may schedule at the slot, or
#: "Post now" some hours late. Two rows with the same caption inside one
#: window cannot be told apart and stay parked as ambiguous.
RECONCILE_WINDOW = timedelta(hours=12)
#: Safety stop on paging; 20 videos a page.
_MAX_LIST_PAGES = 10
#: The leading part of the caption's first line used as the match key.
_MATCH_PREFIX_CHARS = 60


@dataclass
class PackReconcileResult:
    considered: int = 0
    published: int = 0
    ambiguous: int = 0
    ambiguous_rows: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _norm(text: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).casefold().split())


def _caption_key(record: ShortPublishRecord) -> str:
    """The match key: the caption's first non-empty line (the title when the
    caption is empty), normalised (NFKC, case-folded, whitespace collapsed),
    cut to ``_MATCH_PREFIX_CHARS``. TikTok may trim or re-wrap what the owner
    pasted, so only the start of the caption is compared."""
    payload = record.payloads.tiktok
    text = (payload.caption or payload.title) if payload else ""
    first = next((line for line in str(text).splitlines() if line.strip()), "")
    return _norm(first)[:_MATCH_PREFIX_CHARS]


def reconcile_packed(
    conn,
    *,
    list_videos: Callable[..., Any],
    permalink_for: Callable[[str], str],
    record_loader: RecordLoader,
    now: datetime,
    window: timedelta = RECONCILE_WINDOW,
) -> PackReconcileResult:
    """Close pack-parked TikTok rows from TikTok's read-only video list.

    Candidates are ``awaiting_manual_publish`` TikTok rows with no
    ``platform_upload_id`` (inbox rows are ``reconcile_awaiting_manual``'s).
    A video matches a row when its ``create_time`` is within ``window`` of
    the row's slot (:func:`_slot`) AND its normalised description starts
    with the row's :func:`_caption_key`. A row publishes only with exactly
    one matching video that matches no other row and whose id is on no other
    delivery; zero matches leave it parked, several leave every row involved
    parked and reported in ``ambiguous_rows``. Never guesses, never
    publishes without a post id; the permalink is built before the transition.

    ``list_videos(cursor)`` returns ``(videos, cursor, has_more)`` newest
    first and is called only as far back as the oldest slot minus ``window``,
    and not at all while every slot's window is still in the future."""
    result = PackReconcileResult()
    rows = conn.execute(
        "SELECT id, project_name, short_index, scheduled_at FROM deliveries "
        "WHERE status=? AND platform='tiktok' AND shadow=0 "
        "AND (platform_upload_id IS NULL OR platform_upload_id='') ORDER BY id",
        (STATUS_AWAITING_MANUAL_PUBLISH,),
    ).fetchall()
    if is_platform_paused(conn, "tiktok"):
        return result
    wanted = []  # (delivery id, slot, key)
    for row in rows:
        try:
            record = record_loader(row["project_name"], row["short_index"])
            slot = _slot(row["scheduled_at"], record).astimezone(timezone.utc)
            key = _caption_key(record)
        except Exception as exc:  # noqa: BLE001 -- one row never stops the rest
            result.errors.append(f"delivery {row['id']}: {type(exc).__name__}: {exc}")
            continue
        if not key or slot - window > now:
            continue  # nothing to match on, or it cannot have posted yet
        wanted.append((row["id"], slot, key))
    result.considered = len(wanted)
    if not wanted:
        return result

    floor = min(slot for _, slot, _ in wanted) - window
    videos: List[Dict[str, Any]] = []
    cursor = None
    for _ in range(_MAX_LIST_PAGES):
        page, cursor, has_more = list_videos(cursor)
        videos.extend(page)
        times = [int(v["create_time"]) for v in page if v.get("create_time") is not None]
        if not has_more or (times and min(times) < floor.timestamp()):
            break

    used = {
        str(r["platform_post_id"])
        for r in conn.execute(
            "SELECT platform_post_id FROM deliveries "
            "WHERE platform='tiktok' AND platform_post_id IS NOT NULL"
        )
    }
    matches: Dict[int, List[str]] = {}
    for delivery_id, slot, key in wanted:
        matches[delivery_id] = [
            str(v["id"])
            for v in videos
            if v.get("id") is not None
            and str(v["id"]) not in used
            and abs(int(v.get("create_time") or 0) - slot.timestamp()) <= window.total_seconds()
            and _norm(v.get("video_description")).startswith(key)
        ]
    claims: Dict[str, int] = {}
    for ids in matches.values():
        for vid in ids:
            claims[vid] = claims.get(vid, 0) + 1

    for delivery_id, ids in matches.items():
        if not ids:
            continue
        if len(ids) > 1 or claims[ids[0]] > 1:
            result.ambiguous += 1
            result.ambiguous_rows.append(
                f"delivery {delivery_id}: ambiguous, candidate videos {sorted(ids)}"
            )
            continue
        video_id = ids[0]
        try:
            # Built BEFORE the transition: a terminal row refuses later writes.
            permalink = permalink_for(video_id)
            if transition(
                conn,
                delivery_id,
                STATUS_AWAITING_MANUAL_PUBLISH,
                "published",
                platform_post_id=video_id,
                permalink=permalink,
                published_at_source=PUBLISHED_AT_PLATFORM_RECONCILED,
            ):
                result.published += 1
                logger.info("tiktok reconcile: delivery %s published as %s", delivery_id, video_id)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"delivery {delivery_id}: {type(exc).__name__}: {exc}")
    return result
