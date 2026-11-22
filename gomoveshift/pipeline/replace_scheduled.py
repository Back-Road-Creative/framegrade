"""Re-render a drive and replace its long-form uploads that are scheduled but not yet public.

``health backlog --supersede ... --replace-scheduled`` sets a drive's run aside even though its
``youtube_uploads.json`` holds records, when every record is either public (its LF is kept) or
still scheduled well ahead (its LF is replaced). ``classify_uploads`` decides that, failing closed;
the plan it returns is written beside the superseded run and into the fresh output dir as
``REPLACE_PLAN_FILENAME``, which Phase D reads (``load_plan``).

Two failures must never happen on the live channel: two scheduled videos in one slot, and deleting
a video that is already public. So each replaced LF walks ONE durable sequence, its state kept in
the plan under ``progress`` (written atomically BEFORE each external call and after each step)::

    pending     insert the replacement PRIVATE, no publishAt, new gated title (the title-decision
                gate runs unchanged: a refusal leaves the LF pending and the old video untouched);
                the new id is written to the plan the moment ``videos.insert`` returns (the
                uploader's ``persist`` callback), so a crash cannot orphan a private copy
    uploaded    recheck LIVE: old is private, scheduled, publishAt more than REPLACE_MARGIN_H away.
                Only a ``refuse`` (the old provably exists and is public/too close) deletes the NEW
                video, ``abandoned``. An ``error`` DEFERS (stays uploaded, incident, retried next
                entry); ``gone`` (the old is already deleted) moves on to ``old_deleted``
    old_deleted delete the old (a 404 counts as deleted); set its publishAt (tz-aware) on the NEW
                video unless that slot is within REPLACE_SLOT_PASSED_MARGIN_MIN of now (then it is rebooked at
                the next free slot, else ``slot_passed``: the new stays private, incident), file it in the old record's playlist, swap the
                ledger id, book the store
    scheduled   -> ``done`` once everything is recorded

The one invariant: the NEW video is deleted only by a call that has just proved the OLD one exists.

Every Phase D entry resumes each plan LF from its state before any upload
(``resume_replacements``); a failed delete or swap is retried, and ``slot_passed`` is retried too.
A passed slot is rebooked through ``rebook`` (Waypost's allocator), never published at once, and the
new slot is RESERVED (``reserve``: the ledger entry and ``schedule.sqlite3`` move to it atomically)
before any YouTube call, so no other LF or drive can take it; a taken pick is searched again. Kept (public) LFs get no insert, booking, shorts, Metricool
or superlong emission, and replaced LFs get no new shorts/Metricool/superlong either.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple


def _slot_taken(exc: BaseException) -> bool:
    """True for Waypost's ``SlotTaken`` (matched by name: the media engine imports no Waypost code)."""
    return any(c.__name__ == "SlotTaken" for c in type(exc).__mro__)


PENDING, UPLOADED, OLD_DELETED, SCHEDULED, DONE, ABANDONED, SLOT_PASSED = (
    "pending", "uploaded", "old_deleted", "scheduled", "done", "abandoned", "slot_passed")  # fmt: skip

logger = logging.getLogger(__name__)

#: The ``health`` incident kind every replace refusal or partial failure opens (``gomoveshift
#: status --json`` lists it under ``incidents.open``); ``resolve_incidents`` closes a row when its LF
#: finishes.
INCIDENT_KIND = "replace_scheduled_partial"
REPLACE_PLAN_FILENAME = "replace_plan.json"
REPLACE_PLAN_SCHEMA = "replace-scheduled/1"
#: A scheduled upload is replaced only while its ``publishAt`` is at least this far ahead: the
#: re-render takes hours, and a video that goes public mid-render must never be deleted.
REPLACE_MARGIN_H = 6
#: After the old video is deleted the slot counts as passed only this close to now (minutes): a
#: slot even an hour ahead is scheduled, never rebooked or left private.
REPLACE_SLOT_PASSED_MARGIN_MIN = 5
#: How many free slots ``advance`` tries to reserve before it leaves the LF ``slot_passed``: a
#: ``SlotTaken`` means another drive took the pick between the search and the booking.
REPLACE_RESERVE_TRIES = 5
#: Codes that stop being a problem when the LF finishes; ``abandoned`` (the replace did not
#: happen) and ``new_undeleted`` (a private leftover to delete) stay open on an abandoned LF.
_KEEP_OPEN_ON_ABANDON = ("abandoned", "new_undeleted")
#: ``{video_id: state}`` for the ids asked about; ``"scheduled"`` is a private video with a future
#: ``publishAt`` (``playlist_reconciler.fetch_video_privacy``); an id absent is deleted.
LiveStatus = Callable[[list], Mapping[str, str]]


def record_incident(drive: str, code: str, detail: str, *, lf: Optional[int] = None, incidents_dir: Optional[Path] = None) -> None:  # fmt: skip
    """Open one ``INCIDENT_KIND`` row per drive, ``code`` and LF (an unresolved one is left as it
    is, as ``mask_frame_gate`` does). Never raises: a failed write is logged, the caller goes on."""
    try:
        from ..health.store import (  # fmt: skip
            default_data_root,
            default_incidents_dir,
            new_incident,
            safe_name,
            utc_iso,
            write_row,
        )

        inc = Path(incidents_dir) if incidents_dir else default_incidents_dir(default_data_root())
        tail = "" if lf is None else f"_lf{lf}"
        name = f"{INCIDENT_KIND}_{safe_name(drive)}_{code}{tail}.json"
        try:
            prior = json.loads((inc / name).read_text(encoding="utf-8"))
            if isinstance(prior, dict) and not prior.get("resolved_at"):
                return
        except (OSError, ValueError):
            pass
        write_row(inc, new_incident(INCIDENT_KIND, f"{INCIDENT_KIND}:{safe_name(drive)}:{code}{tail}", name, utc_iso(), drive=drive, code=code, lf=lf, error=detail, detail=detail))  # fmt: skip
    except Exception:  # noqa: BLE001
        logger.exception("REPLACE-INCIDENT-NOT-WRITTEN %s %s: %s", drive, code, detail)


def resolve_incidents(drive: str, *, lf: Optional[int] = None, keep: Tuple[str, ...] = (), drive_level: bool = False, incidents_dir: Optional[Path] = None, codes: Tuple[str, ...] = (), outcome: str = "") -> int:  # fmt: skip
    """Resolve this drive's open ``INCIDENT_KIND`` rows for ``lf`` (all its codes but ``keep``, or
    only ``codes`` when given), or with ``drive_level`` the rows that carry no LF; ``outcome`` is
    stamped on each as the terminal reason. Returns how many it closed; never raises."""
    closed = 0
    try:
        from ..health.store import (  # fmt: skip
            default_data_root,
            default_incidents_dir,
            safe_name,
            utc_iso,
            write_row,
        )

        inc = Path(incidents_dir) if incidents_dir else default_incidents_dir(default_data_root())
        for path in sorted(inc.glob(f"{INCIDENT_KIND}_{safe_name(drive)}_*.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(row, dict) or row.get("resolved_at") or row.get("drive") != drive:
                continue
            if row.get("code") in keep or (codes and row.get("code") not in codes) or (row.get("lf") is None) != drive_level:  # fmt: skip
                continue
            if not drive_level and row.get("lf") != lf:
                continue
            row["resolved_at"] = utc_iso()
            if outcome:
                row["outcome"] = outcome
            write_row(inc, row)
            closed += 1
    except Exception:  # noqa: BLE001
        logger.exception("REPLACE-INCIDENT-NOT-RESOLVED %s lf=%s", drive, lf)
    return closed


def unfinished(plan: Mapping[str, Any]) -> Dict[int, str]:
    """``{lf: state}`` of every replaced LF whose replacement is not ``done`` (an abandoned LF
    kept its old video: the replace did not happen)."""
    prog = progress_map(plan)
    return {lf: prog.get(lf, {}).get("state", PENDING) for lf in replace_map(plan) if prog.get(lf, {}).get("state") != DONE}  # fmt: skip


def _when(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _is_not_found(exc: BaseException) -> bool:
    """A 404 from ``videos.delete``: the video is already gone, which is what was asked for."""
    status = getattr(getattr(exc, "resp", None), "status", None) or getattr(exc, "status_code", None)  # fmt: skip
    if status is not None:  # an explicit HTTP status is the answer: a 403 that mentions 404 is not
        return status == 404
    return re.search(r"\b404\b|notFound", str(exc)) is not None


def unfinished_lfs(plan: Mapping[str, Any]) -> Dict[int, str]:
    """``unfinished`` minus the abandoned LFs: what Phase D must not report as a clean finish."""
    return {lf: st for lf, st in unfinished(plan).items() if st != ABANDONED}


def thumbnail_targets(plan: Mapping[str, Any], uploads: Mapping[Any, Any]) -> Dict[Any, Any]:
    """The upload records a thumbnail may go on under a plan: a replaced LF whose new video is
    scheduled or done and not yet thumbnailed; never an old, abandoned or kept video."""
    prog = progress_map(plan)
    out: Dict[Any, Any] = {}
    for key, rec in uploads.items():
        st = prog.get(key if isinstance(key, int) else _lf(key), {})
        if st.get("state") in (SCHEDULED, DONE) and rec.get("video_id") == st.get("new_id") and not rec.get("thumbnail_path"):  # fmt: skip
            out[key] = rec
    return out


def classify_uploads(
    uploads: Any,
    now: datetime,
    live_status: Optional[LiveStatus],
    margin_h: float = REPLACE_MARGIN_H,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(plan, refusal)`` for a drive's ``youtube_uploads.json`` content.

    Every record must be a dict with a video id and a parseable ``publish_at``: public (past) ->
    kept, scheduled inside ``margin_h`` -> kept too (``margin_kept`` names them), scheduled at
    least ``margin_h`` ahead AND live-confirmed private-scheduled -> replaced. Anything else, including an unreadable record or a failed live lookup, refuses."""
    if not isinstance(uploads, dict) or not uploads:
        return None, "no upload records to classify"
    kept: Dict[str, Any] = {}
    replace: Dict[str, Any] = {}
    margin_kept: list = []
    for lf, rec in uploads.items():
        when = _when(rec.get("publish_at")) if isinstance(rec, dict) else None
        if not (isinstance(rec, dict) and rec.get("video_id") and when):
            return None, f"record {lf!r} is not decidable (no video id or publish_at)"
        if when <= now:
            kept[str(lf)] = rec
        elif when < now + timedelta(hours=margin_h):
            # too close to replace (Waypost refuses a slot inside the margin): the old video is
            # kept and publishes on its own, never blocking the LFs that can still be replaced
            kept[str(lf)] = rec
            margin_kept.append(str(lf))
        else:
            replace[str(lf)] = rec
    if not replace:
        if margin_kept and len(margin_kept) == len(kept):
            return None, f"every record is scheduled inside the {margin_h} h margin (kept by design, each publishes on its own): nothing to replace"  # fmt: skip
        if margin_kept:
            return None, f"every record is public or inside the {margin_h} h margin (kept by design): nothing to replace"  # fmt: skip
        return None, "every record is already public: nothing to replace"
    if live_status is None:
        return None, "no live YouTube check available"
    try:
        live = live_status([r["video_id"] for r in replace.values()])
    except Exception as exc:  # noqa: BLE001 -- fail closed on any lookup failure
        return None, f"live YouTube check failed ({type(exc).__name__}: {exc})"
    for lf, rec in replace.items():
        state = live.get(rec["video_id"])
        if state != "scheduled":
            return None, f"record {lf!r} ({rec['video_id']}) is {state or 'not found'} on YouTube, not scheduled"  # fmt: skip
    plan = {"schema": REPLACE_PLAN_SCHEMA, "kept": kept, "replace": replace}
    if margin_kept:
        plan["margin_kept"] = sorted(margin_kept)  # kept for the margin, each an incident
    return plan, ""


def write_plan(out_dir: Path, plan: Dict[str, Any]) -> Path:
    """Write the plan atomically (temp file, then ``os.replace``): a crash never leaves half."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPLACE_PLAN_FILENAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def load_plan(output_base: Path) -> Dict[str, Any]:
    """The plan under ``output_base``, or ``{}`` (absent or unreadable: a plain Phase D)."""
    try:
        plan = json.loads((Path(output_base) / REPLACE_PLAN_FILENAME).read_text())
    except (OSError, ValueError):
        return {}
    return plan if isinstance(plan, dict) and isinstance(plan.get("replace"), dict) else {}


def _lf(key: Any) -> int:
    return int("".join(ch for ch in str(key) if ch.isdigit()))


def replace_map(plan: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    """``{lf: old record}`` of the LFs to replace, keys normalised to int."""
    return {_lf(k): dict(v) for k, v in (plan.get("replace") or {}).items()}


def kept_map(plan: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {_lf(k): dict(v) for k, v in (plan.get("kept") or {}).items()}


def progress_map(plan: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {_lf(k): dict(v) for k, v in (plan.get("progress") or {}).items()}


def set_progress(plan_dir: Path, lf: int, **fields: Any) -> Dict[str, Any]:
    """Merge ``fields`` into LF ``lf``'s progress entry and write the plan atomically."""
    plan = load_plan(plan_dir)
    if not plan:  # unreadable: writing back would replace the plan with a stub
        raise RuntimeError(f"replace plan under {plan_dir} is unreadable; refusing to rewrite it")
    prog = plan.setdefault("progress", {})
    entry = prog.setdefault(str(lf), {})
    entry.update(fields)
    write_plan(plan_dir, plan)
    return dict(entry)


def init_progress(plan_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Give each replaced LF without one a ``pending`` entry; return the progress map."""
    plan = load_plan(plan_dir)
    changed = False
    prog = plan.setdefault("progress", {})
    for lf, old in replace_map(plan).items():
        if str(lf) not in prog:
            prog[str(lf)] = {"old_id": old["video_id"], "new_id": None, "state": PENDING}
            changed = True
    if changed:
        write_plan(plan_dir, plan)
    return progress_map(plan)


def apply_kept(state: Any, plan: Mapping[str, Any]) -> list:
    """Put each kept LF's record into ``state.youtube_uploads`` unless it holds one; returns the
    LF indices kept. ``upload_if_needed`` then skips them."""
    if state.youtube_uploads is None:
        state.youtube_uploads = {}
    kept = []
    for idx, rec in kept_map(plan).items():
        if idx not in state.youtube_uploads and str(idx) not in state.youtube_uploads:
            state.youtube_uploads[idx] = dict(rec)
        kept.append(idx)
    return kept


def check_old(yt: Any, old: Mapping[str, Any], now: datetime, margin_h: float = REPLACE_MARGIN_H):
    """Live check of the old video: ``(verdict, why)``.

    ``ok`` private, scheduled, live publishAt more than ``margin_h`` ahead; ``gone`` YouTube no
    longer has it; ``refuse`` it is public, unscheduled or too close; ``error`` the lookup failed
    (not decidable: the caller defers and never deletes anything on it)."""
    try:
        live = yt.video_schedule(old["video_id"])
    except Exception as exc:  # noqa: BLE001
        return "error", f"lookup failed ({type(exc).__name__}: {exc})"
    if live is None:
        return "gone", "not found on YouTube"
    when = live.get("publish_at")
    if live.get("privacy") != "private" or when is None:
        return "refuse", f"is {live.get('privacy')} on YouTube, not scheduled"
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    if when - now <= timedelta(hours=margin_h):
        return "refuse", f"publishes {when.isoformat()}, inside the {margin_h} h margin"
    return "ok", ""


def swap_ledger_video_id(ledger_path: Path, old_id: str, new_id: str, *, title: Optional[str] = None, slot: Optional[datetime] = None) -> bool:  # fmt: skip
    """Point the content ledger's entry for ``old_id`` at ``new_id``: the old slot, or ``slot`` when
    the LF was rebooked, and the new title when given (the entry now describes the new video)."""
    from .content_ledger import ContentLedger, _ledger_lock

    with _ledger_lock(ledger_path):
        ledger = ContentLedger.load(ledger_path)
        hits = [e for e in ledger.entries if e.video_id == old_id]
        for entry in hits:
            entry.video_id = new_id
            if title:
                entry.title = title
            if slot is not None:
                from .schedule_constants import EST

                local = slot.astimezone(EST)
                entry.date, entry.time = local.strftime("%Y-%m-%d"), local.strftime("%H:%M:%S")
        if hits:
            ledger.save(ledger_path)
    return bool(hits)


def note_kept_live(plan_dir: Path, lf: int, why: str) -> None:
    """Record why a replaced LF's old video was kept (``kept_live``)."""
    plan = load_plan(plan_dir)
    if not plan:
        raise RuntimeError(f"replace plan under {plan_dir} is unreadable; refusing to rewrite it")
    plan.setdefault("kept_live", {})[str(lf)] = {"why": why, "at": datetime.now(timezone.utc).isoformat()}  # fmt: skip
    write_plan(plan_dir, plan)


def abandon(yt: Any, *, lf: int, old: Mapping[str, Any], plan_dir: Path, state: Any, save_state: Callable, why: str, old_proven: bool = False) -> None:  # fmt: skip
    """The replacement cannot take the slot: delete the NEW video (private, unscheduled, so safe),
    keep the old record on state, never touch the old video. The new is deleted only when the
    caller's own live check just proved the old exists (``old_proven``); otherwise it is left."""
    prog = progress_map(load_plan(plan_dir)).get(lf, {})
    new_id = prog.get("new_id")
    new_gone = new_id is None
    if new_id and not old_proven:
        logger.error("REPLACE-NEW-KEPT lf=%s: old %s not proven to exist; private %s left", lf, old["video_id"], new_id)  # fmt: skip
    elif new_id:
        try:
            yt.delete_video(new_id)
            new_gone = True
        except Exception:  # noqa: BLE001
            logger.error("REPLACE-NEW-UNDELETED: private unscheduled %s remains; delete it", new_id, exc_info=True)  # fmt: skip
            record_incident(Path(plan_dir).name, "new_undeleted", f"LF {lf}: private unscheduled {new_id} remains on YouTube; delete it by hand", lf=lf)  # fmt: skip
    logger.error("REPLACE-ABANDONED lf=%s: old %s kept, new %s removed=%s: %s", lf, old["video_id"], new_id, new_gone, why)  # fmt: skip
    record_incident(Path(plan_dir).name, "abandoned", f"LF {lf} not replaced: old {old['video_id']} kept, new {new_id} removed={new_gone}: {why}", lf=lf)  # fmt: skip
    state.youtube_uploads[lf] = dict(old)
    save_state(state)
    note_kept_live(plan_dir, lf, why)
    set_progress(plan_dir, lf, state=ABANDONED, why=why, new_deleted=new_gone)
    resolve_incidents(Path(plan_dir).name, lf=lf, keep=_KEEP_OPEN_ON_ABANDON)  # fmt: skip


def advance(
    yt: Any, *, lf: int, old: Mapping[str, Any], plan_dir: Path, state: Any, save_state: Callable,
    now: datetime, ledger_path: Path, book: Callable[[dict], None], metadata: Optional[dict] = None,
    config: Any = None, rebook: Optional[Callable[[datetime], Optional[datetime]]] = None,
    reserve: Optional[Callable[[datetime], None]] = None,
) -> str:  # fmt: skip
    """Move one LF from its recorded state as far as it will go; returns the state reached.

    ``rebook(now)`` returns the next free slot (Waypost's allocator) for an LF whose slot passed
    after its old video was deleted; without one the new video stays private (``slot_passed``,
    retried each entry), never published at once. ``reserve(slot)`` books that slot for this LF
    (idempotent) and raises ``SlotTaken`` when another row holds the date; it runs before any
    YouTube call and again on re-entry while the rebooked slot is unsent.

    Safe to call on every entry: each branch checks the durable state first, and a failed delete or
    swap stays where it is (logged loudly) to be retried."""
    st = progress_map(load_plan(plan_dir)).get(lf, {})
    new_id = st.get("new_id")
    if st.get("state") == UPLOADED:
        verdict, why = check_old(yt, old, now)
        if verdict == "refuse":  # the old provably exists and is public or too close
            abandon(yt, lf=lf, old=old, plan_dir=plan_dir, state=state, save_state=save_state, why=f"recheck: {why}", old_proven=True)  # fmt: skip
            return ABANDONED
        if verdict == "error":  # not decidable: the old may already be deleted, so keep the new
            logger.error("REPLACE-RECHECK-DEFERRED lf=%s: %s; new %s kept, retried next entry", lf, why, new_id)  # fmt: skip
            record_incident(Path(plan_dir).name, "recheck_failed", f"LF {lf}: recheck of old {old['video_id']} {why}; the replacement {new_id} stays private and is retried next entry", lf=lf)  # fmt: skip
            return UPLOADED
        if verdict == "ok":
            try:
                yt.delete_video(old["video_id"])
            except Exception as exc:  # noqa: BLE001
                if not _is_not_found(exc):
                    logger.error("REPLACE-OLD-UNDELETED lf=%s: %s still holds the slot; retried next entry", lf, old["video_id"], exc_info=True)  # fmt: skip
                    record_incident(Path(plan_dir).name, "old_undeleted", f"LF {lf}: old {old['video_id']} could not be deleted and still holds the slot; the replacement {new_id} stays private until it is", lf=lf)  # fmt: skip
                    return UPLOADED
        # "gone": an earlier run deleted it and died before writing the state; the slot is free
        set_progress(plan_dir, lf, state=OLD_DELETED)
    st = progress_map(load_plan(plan_dir)).get(lf, {})
    if st.get("state") in (OLD_DELETED, SLOT_PASSED):
        when = _when(st.get("slot") or old.get("publish_at"))
        pid = old.get("playlist_id")
        if when is None or (not pid and metadata is None):
            logger.error("REPLACE-UNSCHEDULED lf=%s: new %s needs a publish_at/playlist by hand", lf, new_id)  # fmt: skip
            record_incident(Path(plan_dir).name, "unscheduled", f"LF {lf}: old deleted, new {new_id} needs a publish_at/playlist by hand", lf=lf)  # fmt: skip
            return OLD_DELETED
        unsent = not st.get("publish_set")
        passed = unsent and when <= now + timedelta(minutes=REPLACE_SLOT_PASSED_MARGIN_MIN)
        if unsent and not passed and st.get("rebooked_from") and reserve is not None:
            try:  # a rebooked slot is only ours while the store still holds it for this LF
                reserve(when)
            except Exception as exc:  # noqa: BLE001
                if _slot_taken(exc):
                    logger.error("REPLACE-RESERVE-LOST lf=%s: slot %s was taken during the backoff; rebooking", lf, when.isoformat())  # fmt: skip
                    passed = True
                else:
                    logger.exception("REPLACE-RESERVE-FAILED lf=%s", lf)
                    record_incident(Path(plan_dir).name, "schedule_failed", f"LF {lf}: the rebooked slot {when.isoformat()} could not be re-reserved; new {new_id} stays private, retried next entry", lf=lf)  # fmt: skip
                    return OLD_DELETED
        if passed:
            # A past publishAt would go public at once (or be rejected): never send one. The old
            # video is gone, so the LF takes the next free slot, RESERVED before any YouTube call.
            nxt = None
            for _ in range(REPLACE_RESERVE_TRIES):
                cand = None
                try:
                    cand = rebook(now) if rebook is not None else None
                except Exception:  # noqa: BLE001
                    logger.exception("REPLACE-REBOOK-FAILED lf=%s", lf)
                if cand is not None and cand.tzinfo is None:
                    cand = cand.replace(tzinfo=timezone.utc)
                if cand is None or cand <= now + timedelta(minutes=REPLACE_SLOT_PASSED_MARGIN_MIN):
                    break
                if reserve is None:
                    nxt = cand
                    break
                try:
                    reserve(cand)
                    nxt = cand
                    break
                except Exception as exc:  # noqa: BLE001 -- never send a slot that is not reserved
                    if _slot_taken(exc):
                        logger.error("REPLACE-RESERVE-TAKEN lf=%s: %s was taken; searching again", lf, cand.isoformat())  # fmt: skip
                        continue
                    logger.exception("REPLACE-RESERVE-FAILED lf=%s", lf)
                    break
            if nxt is None:
                logger.error("REPLACE-SLOT-PASSED lf=%s: slot %s passed, no free slot yet; new %s stays private, retried next entry", lf, when.isoformat(), new_id)  # fmt: skip
                record_incident(Path(plan_dir).name, "slot_passed", f"LF {lf}: slot passed mid-replace ({when.isoformat()}), old deleted, new {new_id} stays private until the next free slot is found; retried next entry", lf=lf)  # fmt: skip
                set_progress(plan_dir, lf, state=SLOT_PASSED)
                return SLOT_PASSED
            record_incident(Path(plan_dir).name, "slot_rebooked", f"LF {lf}: slot {when.isoformat()} passed mid-replace; new {new_id} rebooked at the next free slot {nxt.isoformat()}", lf=lf)  # fmt: skip
            set_progress(plan_dir, lf, slot=nxt.isoformat(), rebooked_from=when.isoformat(), state=OLD_DELETED)  # fmt: skip
            when, st = nxt, {**st, "slot": nxt.isoformat(), "rebooked_from": when.isoformat()}
        try:
            if not st.get("publish_set"):
                yt.update_privacy(new_id, privacy="private", publish_at=when)
                set_progress(plan_dir, lf, publish_set=True)
            if not st.get("playlist_set"):
                if pid:
                    yt.add_to_playlist(new_id, pid)
                else:
                    logger.warning("replace lf=%s: old record has no playlist_id; resolving fresh", lf)  # fmt: skip
                    pid = yt.resolve_playlist_id(metadata=metadata, config=config)
                    yt.add_to_playlist(new_id, pid)  # lists first: a replay never files it twice
                set_progress(plan_dir, lf, playlist_set=True, playlist_id=pid)
            if not st.get("ledger_swapped"):
                swap_ledger_video_id(ledger_path, old["video_id"], new_id, title=st.get("new_title"), slot=when if st.get("slot") else None)  # fmt: skip
                set_progress(plan_dir, lf, ledger_swapped=True)
        except Exception:  # noqa: BLE001
            logger.error("REPLACE-SCHEDULE-FAILED lf=%s: new %s retried next entry", lf, new_id, exc_info=True)  # fmt: skip
            record_incident(Path(plan_dir).name, "schedule_failed", f"LF {lf}: old deleted but new {new_id} is not scheduled yet (publish time, playlist or ledger swap failed); retried next entry", lf=lf)  # fmt: skip
            return OLD_DELETED
        set_progress(plan_dir, lf, state=SCHEDULED)
    cur = progress_map(load_plan(plan_dir)).get(lf, {})
    if cur.get("state") == SCHEDULED:
        rec = dict(state.youtube_uploads.get(lf) or {})
        rec.update(video_id=new_id, url=f"https://youtu.be/{new_id}", title=rec.get("title") or old.get("title"),  # fmt: skip
                   publish_at=_when(cur.get("slot") or old.get("publish_at")).isoformat(),
                   playlist_id=cur.get("playlist_id") or old.get("playlist_id"))  # fmt: skip
        state.youtube_uploads[lf] = rec
        save_state(state)
        try:
            book(rec)
        except Exception as exc:  # noqa: BLE001 -- a SlotTaken here is never swallowed: not DONE
            logger.error("REPLACE-BOOK-FAILED lf=%s: new %s is scheduled but not booked: %s", lf, new_id, exc, exc_info=True)  # fmt: skip
            record_incident(Path(plan_dir).name, "book_failed", f"LF {lf}: new {new_id} is scheduled but its booking failed ({type(exc).__name__}: {exc}); retried next entry", lf=lf)  # fmt: skip
            return SCHEDULED
        set_progress(plan_dir, lf, state=DONE)
        resolve_incidents(Path(plan_dir).name, lf=lf)  # every code this LF opened is over
        return DONE
    return cur.get("state", PENDING)
