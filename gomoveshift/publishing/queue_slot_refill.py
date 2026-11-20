"""Give an unrepairable queued Short's slot to the next Short that is ready, and bring back
a failed one that now passes.

``queue_repair`` repairs what it can; a row still refused at dispatch burns its attempts,
ends ``failed_permanent`` and its slot is never refilled. :func:`refill_slots` runs right
after the repair, on the queued, non-shadow YouTube rows:

* **Hand-off.** A refused row whose slot is within ``QUEUED_REFUSED_WITHIN_H`` of now (the
  health rule 7 constant) never reaches publish. It swaps slots with the earliest LATER
  queued row that passes the review, so the ready Short moves up and the refused one takes
  the later slot. Rows are taken in slot order, so a run of refused rows drifts later while
  the ready rows move up. With no passing row left, or a slot too near to publish into
  (``YOUTUBE_MIN_LEAD``), the refused row moves to the first free time of its region's
  daily Waypost grid after the last queued slot. No hour that already holds a YouTube row
  is ever given out.
* **Recover.** A ``failed_permanent`` row the final-text review refused, holding no upload
  id and not already recovered, is re-reviewed; one that now passes is requeued through
  ``queue.recover_terminal_delivery`` into the next free grid slot. One that still cannot
  pass is reported and left. ``JP_KEPT_FAILED_PERMANENT_IDS`` is never touched.

The slot is written with the same writers ``reslot-overdue-youtube`` uses: the sidecar's
``publishAt`` (``plan_title_sync.write_plan_slot``) then ``queue.reschedule_delivery``.
Published rows are never read as candidates. Local writes only; never a YouTube call.
Without ``apply`` the same actions are computed and nothing is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo

from .queue import (
    RecoveryRefusedError,
    _has_live_foreign_lease,
    log_response,
    recover_terminal_delivery,
    reschedule_delivery,
)
from .queue_repair import row_refusal

#: JP's decision (Beaufort Shorts): these five ``failed_permanent`` deliveries stay failed.
#: Never re-reviewed, requeued or reported by this step, whatever the review now says.
JP_KEPT_FAILED_PERMANENT_IDS = frozenset({98, 102, 106, 110, 114})

#: The statuses in which a YouTube row holds no slot (as ``queue_reslot``).
_FREE_STATUSES = ("cancelled", "failed_permanent", "missing_remote")

#: A slot closer than this to now cannot take a new ``publishAt`` (as ``queue_reslot``).
MIN_LEAD = timedelta(hours=1)

_SEARCH_DAYS = 21
AUTHORIZED_BY = "dispatch-queue slot refill: the final-text review now passes"

Grid = Callable[[Any], Optional[Tuple[List[str], ZoneInfo]]]


@dataclass(frozen=True)
class SlotAction:
    kind: str  # "slot" | "recovered" | "unrecoverable" | "skipped"
    delivery_id: int
    old: str = ""
    new: str = ""
    reason: str = ""
    written: bool = False

    def line(self) -> str:
        if self.kind == "slot":
            return f"QUEUE_SLOT {self.delivery_id} {self.old} -> {self.new} {self.reason}"
        if self.kind == "recovered":
            return f"QUEUE_RECOVER {self.delivery_id} -> {self.new} {self.reason}"
        if self.kind == "unrecoverable":
            return f"QUEUE_FAILED {self.delivery_id} still_refused:{self.reason}"
        return f"QUEUE_SLOT_SKIPPED {self.delivery_id} {self.reason}"


def _one_line(text: Any) -> str:
    return " ".join(str(text).split())


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _hour(moment: datetime) -> datetime:
    return moment.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _next_free(
    grid: Tuple[List[str], ZoneInfo], floor: datetime, held: Dict[int, datetime]
) -> Optional[datetime]:
    """The first grid time strictly after ``floor`` whose hour no held row occupies."""
    pool, tz = grid
    busy = {_hour(s) for s in held.values()}
    today = floor.astimezone(tz).date()
    for offset in range(_SEARCH_DAYS):
        day = today + timedelta(days=offset)
        for hhmm in sorted(pool):
            hour, minute = (int(p) for p in hhmm.split(":"))
            cand = datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz)
            cand = cand.astimezone(timezone.utc)
            if cand > floor and _hour(cand) not in busy:
                return cand
    return None


def write_row_slot(row, slot: datetime) -> None:
    """The row's sidecar ``publishAt`` (what ``reslot-overdue-youtube`` writes first)."""
    from .plan_title_sync import write_plan_slot

    path = Path(row["sidecar_path"] or "")
    if not path.is_file():
        raise OSError(f"publish plan sidecar not readable: {path}")
    write_plan_slot(path, row["project_name"], row["short_index"], slot)


def config_grid(config) -> Grid:
    """``row -> (HH:MM pool, tz)`` of the row's region: from its sidecar's sampled regions,
    else the region whose Waypost pool holds the row's own slot time (it was slotted from it)."""
    from ..pipeline import seo_slots
    from ..pipeline.playlist_resolver import short_region
    from .queue_repair import load_row_record

    cadence = seo_slots.waypost_cadence(config)
    tz = ZoneInfo(cadence.tz.key)

    def grid_for(row):
        regions = sorted(cadence.regions)
        try:
            region = short_region(load_row_record(row).sampled_regions)
        except Exception:  # noqa: BLE001 -- a gone sidecar falls back to the slot's own time
            region = ""
        if region not in regions:
            local = _utc(row["scheduled_at"]).astimezone(tz).strftime("%H:%M")
            region = next((r for r in regions if local in seo_slots.shorts_pool(config, r)), "")
        return (seo_slots.shorts_pool(config, region), tz) if region else None

    return grid_for


def _log(conn, action_kind: str, payload: Dict[str, Any], now: datetime) -> None:
    log_response(
        conn,
        "youtube",
        "slot_refill",
        0,
        action_kind,
        "live",
        reason=json.dumps(payload, sort_keys=True),
        now=now.isoformat(),
    )


def refill_slots(
    conn,
    *,
    apply: bool,
    now: datetime,
    refusals: Mapping[int, Optional[str]],
    grid_for: Grid,
    review: Callable[[Any], Optional[str]] = row_refusal,
    write_slot: Callable[[Any, datetime], None] = write_row_slot,
) -> List[SlotAction]:
    """``refusals`` maps each queued YouTube row id to the review's verdict after the
    repair (``None`` = passes); a row missing from it is never moved nor used as a partner."""
    from ..health.check import QUEUED_REFUSED_WITHIN_H

    limit = now + timedelta(hours=QUEUED_REFUSED_WITHIN_H)
    floor = now + MIN_LEAD
    rows = {
        r["id"]: r
        for r in conn.execute(
            "SELECT * FROM deliveries WHERE platform='youtube' AND shadow=0 "
            "AND status='queued' AND scheduled_at IS NOT NULL"
        )
    }
    held: Dict[int, datetime] = {
        r["id"]: _utc(r["scheduled_at"])
        for r in conn.execute(
            "SELECT id, scheduled_at, status FROM deliveries "
            "WHERE platform='youtube' AND scheduled_at IS NOT NULL"
        )
        if r["status"] not in _FREE_STATUSES
    }
    out: List[SlotAction] = []
    order = sorted(rows, key=lambda i: (held[i], i))
    moved: set = set()

    def move(rid: int, new: datetime, reason: str, *, sidecar_required: bool) -> bool:
        row, old = rows[rid], held[rid]
        if apply:
            if _has_live_foreign_lease(conn, rid, now.isoformat()):
                out.append(SlotAction("skipped", rid, reason="held by a live lease"))
                return False
            try:
                write_slot(row, new)
            except (OSError, ValueError, KeyError) as exc:
                if sidecar_required:
                    out.append(SlotAction("skipped", rid, reason=_one_line(f"sidecar: {exc}")))
                    return False
            if not reschedule_delivery(conn, rid, new.isoformat(), now=now.isoformat()):
                out.append(SlotAction("skipped", rid, reason="the queue refused the move"))
                return False
            _log(
                conn,
                "queue_slot",
                {"delivery_id": rid, "from": old.isoformat(), "to": new.isoformat(), "why": reason},
                now,
            )
        held[rid] = new
        moved.add(rid)
        out.append(SlotAction("slot", rid, old.isoformat(), new.isoformat(), reason, apply))
        return True

    for rid in order:
        verdict = refusals.get(rid, "not reviewed")
        if verdict is None or rid in moved or held[rid] > limit:
            continue
        why = _one_line(verdict)
        slot = held[rid]
        partner = None
        if slot > floor:
            partner = next(
                (
                    p
                    for p in order
                    if p not in moved and held[p] > slot and refusals.get(p, "not reviewed") is None
                ),
                None,
            )
        if partner is not None:
            theirs = held[partner]
            if apply and any(
                _has_live_foreign_lease(conn, i, now.isoformat()) for i in (rid, partner)
            ):  # both or neither: a half swap would put two rows in one hour
                out.append(SlotAction("skipped", rid, reason="held by a live lease"))
                continue
            if move(partner, slot, f"moved_up_for:{rid}", sidecar_required=True):
                move(rid, theirs, f"refused:{why}", sidecar_required=False)
            continue
        grid = grid_for(rows[rid])
        last = max(held[i] for i in rows)
        target = _next_free(grid, max(last, floor), held) if grid else None
        if target is None:
            out.append(SlotAction("skipped", rid, reason="no free slot on the daily grid"))
            continue
        move(rid, target, f"refused:{why}", sidecar_required=False)

    for row in conn.execute(
        "SELECT * FROM deliveries WHERE platform='youtube' AND shadow=0 "
        "AND status='failed_permanent' ORDER BY id"
    ).fetchall():
        rid = row["id"]
        if (
            rid in JP_KEPT_FAILED_PERMANENT_IDS
            or "FinalTextReview" not in (row["error_class"] or "")
            or row["platform_upload_id"]
            or conn.execute(
                "SELECT 1 FROM deliveries WHERE recovers_delivery_id=?", (rid,)
            ).fetchone()
        ):
            continue
        try:
            refusal = review(row)
        except Exception as exc:  # noqa: BLE001 -- an unreadable plan is reported, never fatal
            refusal = f"unreadable:{exc}"
        if refusal:
            out.append(SlotAction("unrecoverable", rid, reason=_one_line(refusal)))
            continue
        grid = grid_for(row)
        slot = _next_free(grid, floor, held) if grid else None
        if slot is None:
            out.append(SlotAction("skipped", rid, reason="no free slot on the daily grid"))
            continue
        new_id = 0
        if apply:
            try:
                write_slot(row, slot)
                new_id = recover_terminal_delivery(
                    conn,
                    rid,
                    authorized_by=AUTHORIZED_BY,
                    evidence={"slot": slot.isoformat(), "final_text_review": "agree"},
                    now=now.isoformat(),
                    scheduled_at=slot.isoformat(),
                )
            except (OSError, ValueError, KeyError, RecoveryRefusedError) as exc:
                out.append(SlotAction("skipped", rid, reason=_one_line(exc)))
                continue
        held[-rid] = slot  # a synthetic key: this slot is now taken
        out.append(
            SlotAction(
                "recovered", rid, "", slot.isoformat(), f"queued_as:{new_id or 'dry-run'}", apply
            )
        )
    return out
