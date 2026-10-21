"""``gomoveshift queue reslot-overdue-manual``: move overdue TikTok slots forward.

A TikTok row parked ``awaiting_manual_publish`` whose slot is more than the grace
period behind (``pub_status._manual_publish_overdue`` -- the digest's own rule) is
given the next Waypost shorts time of its region. Operator-only: the drain never
calls it, it posts nothing, and it never renames or copies a pack file.

Dry run by default. ``--apply`` on a pack-mode row first appends one
``RESLOTTED`` block to the pack's ``captions.txt`` (the owner re-schedules by the
new time), then moves the row; the ``.mp4`` keeps its original name. Any row it
cannot move safely is named ``REFUSED: <id>: <reason>`` and the exit code is 1;
the other rows are still processed. ``mark-published`` stays the only close.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Set
from zoneinfo import ZoneInfo

from ...pipeline import seo_slots
from ...pipeline.playlist_resolver import short_region
from ...publishing.dispatch import TransientMediaError
from ...publishing.handlers import resolve_tiktok_mode
from ...publishing.plan_io import load_publish_plan
from ...publishing.queue import (
    STATUS_AWAITING_MANUAL_PUBLISH,
    TERMINAL_STATUSES,
    _has_live_foreign_lease,
    open_queue_db,
    reschedule_delivery,
)
from ...publishing.tiktok_pack import (
    _DEAD_MOUNT_ERRNOS,
    _PACKED_NAME,
    CAPTIONS_NAME,
    MIN_SCHEDULE_LEAD,
    _append_caption,
    _caption_block,
    _check_pack_parent,
    _slot,
    resolve_pack_dir,
)
from .pub_status import _manual_publish_overdue

#: How many days ahead the next free pool time is searched for.
_SEARCH_DAYS = 21


class _RefusedError(Exception):
    pass


def _shorts_pool(config, region: str) -> List[str]:
    return seo_slots.shorts_pool(config, region)


def _cadence_tz(config) -> str:
    return seo_slots.waypost_cadence(config).tz.key


def _record_for(row):
    """The publish-plan record behind a delivery, from its recorded sidecar."""
    if not row["sidecar_path"]:
        raise _RefusedError("no publish-plan sidecar recorded for this row")
    path = Path(row["sidecar_path"])
    if not path.exists():
        raise _RefusedError(f"publish-plan sidecar not readable: {path}")
    for record in load_publish_plan(path).shorts:
        if record.short_index == row["short_index"]:
            return record
    raise _RefusedError(f"sidecar {path} has no short_index {row['short_index']}")


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _next_slot(pool: List[str], tz: ZoneInfo, now: datetime, taken: Set[datetime]) -> datetime:
    """The first pool time strictly after ``now`` + the pack's minimum lead that
    no other live TikTok row holds, as UTC."""
    floor = now + MIN_SCHEDULE_LEAD
    today = now.astimezone(tz).date()
    for offset in range(_SEARCH_DAYS):
        day = today + timedelta(days=offset)
        for hhmm in sorted(pool):
            hour, minute = (int(p) for p in hhmm.split(":"))
            candidate = datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz)
            candidate = candidate.astimezone(timezone.utc)
            if candidate > floor and candidate not in taken:
                return candidate
    raise _RefusedError(f"no free Waypost shorts time in the next {_SEARCH_DAYS} days")


def _pack_file(pack_dir: Path, delivery_id: int) -> Path:
    try:
        _check_pack_parent(pack_dir)
        names = sorted(p.name for p in pack_dir.iterdir())
    except TransientMediaError as exc:
        raise _RefusedError(str(exc)) from exc
    except OSError as exc:
        if exc.errno in _DEAD_MOUNT_ERRNOS:
            raise _RefusedError(
                f"TikTok pack directory {pack_dir} is not answering: {exc}"
            ) from exc
        raise
    for name in names:
        match = _PACKED_NAME.match(name)
        if match and int(match.group(1)) == delivery_id:
            return pack_dir / name
    raise _RefusedError(f"no pack file for delivery {delivery_id}")


def _reslot_one(conn, row, config, mode: str, now: datetime, taken, apply: bool) -> str:
    record = _record_for(row)
    region = short_region(record.sampled_regions)
    if not region:
        raise _RefusedError(
            f"cannot derive a Waypost region from sampled_regions={record.sampled_regions!r}"
        )
    try:
        pool = _shorts_pool(config, region)
    except Exception as exc:  # noqa: BLE001 -- named per row, the others go on
        raise _RefusedError(f"no shorts times for region {region!r}: {exc}") from exc
    new = _next_slot(pool, ZoneInfo(_cadence_tz(config)), now, taken)
    new_iso = new.isoformat()
    packed = mode == "pack"
    pack_file: Optional[Path] = _pack_file(resolve_pack_dir(), row["id"]) if packed else None
    note = f"pack file {pack_file.name} gets a RESLOTTED caption block" if pack_file else "DB only"
    line = f"delivery {row['id']} {row['platform']} {row['scheduled_at']} -> {new_iso} ({note})"
    if not apply:
        # Reserve it as --apply would, so the preview shows the slots apply books.
        taken.add(new)
        return f"dry run: would move {line}"
    if _has_live_foreign_lease(conn, row["id"], now.isoformat()):
        raise _RefusedError("held by a live lease; try again after it expires")
    if pack_file is not None:
        slot = _slot(new_iso, record)
        # The slot is in the header on purpose: _append_caption dedupes by header
        # line, so a second reslot of the same row on a later day must not be
        # swallowed by the first one's block.
        header = (
            f"{pack_file.name} -- RESLOTTED for delivery {row['id']} ({slot:%Y-%m-%d %H:%M %Z})"
        )
        try:
            _append_caption(
                pack_file.parent / CAPTIONS_NAME, header, _caption_block(header, slot, record, now)
            )
        except OSError as exc:
            raise _RefusedError(f"could not write {CAPTIONS_NAME}: {exc}") from exc
    if not reschedule_delivery(conn, row["id"], new_iso, now=now.isoformat()):
        written = " (its captions.txt block WAS written)" if pack_file else ""
        raise _RefusedError(f"the queue refused the move{written}")
    taken.add(new)
    return f"moved {line}"


def cmd_reslot_overdue_manual(args, config) -> int:
    try:
        mode = resolve_tiktok_mode()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    now = datetime.now(timezone.utc)
    conn = open_queue_db(config.seo.storage_path)
    refused = 0
    try:
        rows = conn.execute(
            "SELECT * FROM deliveries WHERE platform='tiktok' AND status=? ORDER BY id",
            (STATUS_AWAITING_MANUAL_PUBLISH,),
        ).fetchall()
        wanted = getattr(args, "id", None)
        if wanted is not None and not any(r["id"] == wanted for r in rows):
            print(f"REFUSED: {wanted}: not an awaiting_manual_publish TikTok row", file=sys.stderr)
            return 1
        taken = {
            _utc(r["scheduled_at"])
            for r in conn.execute(
                "SELECT scheduled_at, status FROM deliveries "
                "WHERE platform='tiktok' AND scheduled_at IS NOT NULL"
            )
            if r["status"] not in TERMINAL_STATUSES
        }
        overdue = [
            r
            for r in rows
            if (wanted is None or r["id"] == wanted)
            and _manual_publish_overdue(r, now.isoformat()) is not None
        ]
        for row in overdue:
            try:
                print(_reslot_one(conn, row, config, mode, now, taken, bool(args.apply)))
            except _RefusedError as exc:
                refused += 1
                print(f"REFUSED: {row['id']}: {exc}", file=sys.stderr)
    finally:
        conn.close()
    if not overdue:
        print("no overdue awaiting_manual_publish TikTok rows")
    elif not args.apply:
        print(f"dry run: {len(overdue)} overdue row(s); re-run with --apply to move them")
    return 1 if refused else 0
