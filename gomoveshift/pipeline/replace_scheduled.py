"""Re-render a drive and replace its long-form uploads that are scheduled but not yet public.

``health backlog --supersede ... --replace-scheduled`` sets a drive's run aside even though its
``youtube_uploads.json`` holds records, when every record is either public (its LF is kept) or
still scheduled well ahead (its LF is replaced). ``classify_uploads`` decides that, failing closed;
the plan it returns is written beside the superseded run and into the fresh output dir as
``REPLACE_PLAN_FILENAME``, which Phase D reads (``load_plan``).

Two failures must never happen on the live channel: two scheduled videos in one slot, and deleting
a video that is already public. So each replaced LF walks ONE durable sequence, its state kept in
the plan under ``progress`` (written atomically BEFORE each external call and after each step)::

    pending     insert the replacement PRIVATE, no publishAt, old title (the title-decision gate
                runs unchanged: a refusal leaves the LF pending and the old video untouched)
    uploaded    recheck LIVE: old is private, scheduled, publishAt more than REPLACE_MARGIN_H away.
                Failed or errored -> delete the NEW video, ``abandoned``; the old is never deleted
    old_deleted delete the old; set its publishAt (tz-aware) on the NEW video, file it in the old
                record's playlist, swap the ledger id, book the store
    scheduled   -> ``done`` once everything is recorded

Every Phase D entry resumes each plan LF from its state before any upload
(``resume_replacements``); a failed delete or swap is retried. A passed slot is never rescheduled
under a plan (the old video is kept). Kept (public) LFs get no insert, booking, shorts, Metricool
or superlong emission, and replaced LFs get no new shorts/Metricool/superlong either.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

PENDING, UPLOADED, OLD_DELETED, SCHEDULED, DONE, ABANDONED = (
    "pending", "uploaded", "old_deleted", "scheduled", "done", "abandoned")  # fmt: skip

logger = logging.getLogger(__name__)

#: The ``health`` incident kind every replace refusal or partial failure opens (``gomoveshift
#: status --json`` lists it under ``incidents.open``); nothing here resolves or deletes one.
INCIDENT_KIND = "replace_scheduled_partial"
REPLACE_PLAN_FILENAME = "replace_plan.json"
REPLACE_PLAN_SCHEMA = "replace-scheduled/1"
#: A scheduled upload is replaced only while its ``publishAt`` is at least this far ahead: the
#: re-render takes hours, and a video that goes public mid-render must never be deleted.
REPLACE_MARGIN_H = 6
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


def classify_uploads(
    uploads: Any,
    now: datetime,
    live_status: Optional[LiveStatus],
    margin_h: float = REPLACE_MARGIN_H,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """``(plan, refusal)`` for a drive's ``youtube_uploads.json`` content.

    Every record must be a dict with a video id and a parseable ``publish_at``: public (past) ->
    kept, scheduled (at least ``margin_h`` ahead AND live-confirmed private-scheduled) ->
    replaced. Anything else, including an unreadable record or a failed live lookup, refuses."""
    if not isinstance(uploads, dict) or not uploads:
        return None, "no upload records to classify"
    kept: Dict[str, Any] = {}
    replace: Dict[str, Any] = {}
    for lf, rec in uploads.items():
        when = _when(rec.get("publish_at")) if isinstance(rec, dict) else None
        if not (isinstance(rec, dict) and rec.get("video_id") and when):
            return None, f"record {lf!r} is not decidable (no video id or publish_at)"
        if when <= now:
            kept[str(lf)] = rec
        elif when < now + timedelta(hours=margin_h):
            return None, f"record {lf!r} publishes {rec['publish_at']}, inside the {margin_h} h margin"  # fmt: skip
        else:
            replace[str(lf)] = rec
    if not replace:
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
    return {"schema": REPLACE_PLAN_SCHEMA, "kept": kept, "replace": replace}, ""


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
    (not decidable, so never replaceable; the old video is kept)."""
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


def swap_ledger_video_id(ledger_path: Path, old_id: str, new_id: str) -> bool:
    """Point the content ledger's entry for ``old_id`` at ``new_id`` (same slot)."""
    from .content_ledger import ContentLedger, _ledger_lock

    with _ledger_lock(ledger_path):
        ledger = ContentLedger.load(ledger_path)
        hits = [e for e in ledger.entries if e.video_id == old_id]
        for entry in hits:
            entry.video_id = new_id
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


def abandon(yt: Any, *, lf: int, old: Mapping[str, Any], plan_dir: Path, state: Any, save_state: Callable, why: str) -> None:  # fmt: skip
    """The replacement cannot take the slot: delete the NEW video (private, unscheduled, so safe),
    keep the old record on state, never touch the old video."""
    prog = progress_map(load_plan(plan_dir)).get(lf, {})
    new_id = prog.get("new_id")
    new_gone = new_id is None
    if new_id:
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


def advance(
    yt: Any, *, lf: int, old: Mapping[str, Any], plan_dir: Path, state: Any, save_state: Callable,
    now: datetime, ledger_path: Path, book: Callable[[dict], None], metadata: Optional[dict] = None,
    config: Any = None,
) -> str:  # fmt: skip
    """Move one LF from its recorded state as far as it will go; returns the state reached.

    Safe to call on every entry: each branch checks the durable state first, and a failed delete or
    swap stays where it is (logged loudly) to be retried."""
    st = progress_map(load_plan(plan_dir)).get(lf, {})
    new_id = st.get("new_id")
    if st.get("state") == UPLOADED:
        verdict, why = check_old(yt, old, now)
        if verdict in ("refuse", "error"):
            abandon(yt, lf=lf, old=old, plan_dir=plan_dir, state=state, save_state=save_state, why=f"recheck: {why}")  # fmt: skip
            return ABANDONED
        if verdict == "ok":
            try:
                yt.delete_video(old["video_id"])
            except Exception:  # noqa: BLE001
                logger.error("REPLACE-OLD-UNDELETED lf=%s: %s still holds the slot; retried next entry", lf, old["video_id"], exc_info=True)  # fmt: skip
                record_incident(Path(plan_dir).name, "old_undeleted", f"LF {lf}: old {old['video_id']} could not be deleted and still holds the slot; the replacement {new_id} stays private until it is", lf=lf)  # fmt: skip
                return UPLOADED
        # "gone": an earlier run deleted it and died before writing the state; the slot is free
        set_progress(plan_dir, lf, state=OLD_DELETED)
    if progress_map(load_plan(plan_dir)).get(lf, {}).get("state") == OLD_DELETED:
        when = _when(old.get("publish_at"))
        pid = old.get("playlist_id")
        if when is None or (not pid and metadata is None):
            logger.error("REPLACE-UNSCHEDULED lf=%s: new %s needs a publish_at/playlist by hand", lf, new_id)  # fmt: skip
            record_incident(Path(plan_dir).name, "unscheduled", f"LF {lf}: old deleted, new {new_id} needs a publish_at/playlist by hand", lf=lf)  # fmt: skip
            return OLD_DELETED
        try:
            if not st.get("publish_set"):
                yt.update_privacy(new_id, privacy="private", publish_at=when)
                set_progress(plan_dir, lf, publish_set=True)
            if not st.get("playlist_set"):
                if pid:
                    yt.add_to_playlist(new_id, pid)
                else:
                    logger.warning("replace lf=%s: old record has no playlist_id; resolving fresh", lf)  # fmt: skip
                    pid = yt.assign_video_to_resolved_playlist(new_id, metadata=metadata, config=config)  # fmt: skip
                set_progress(plan_dir, lf, playlist_set=True, playlist_id=pid)
            if not st.get("ledger_swapped"):
                swap_ledger_video_id(ledger_path, old["video_id"], new_id)
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
                   publish_at=_when(old.get("publish_at")).isoformat(),
                   playlist_id=cur.get("playlist_id") or old.get("playlist_id"))  # fmt: skip
        state.youtube_uploads[lf] = rec
        save_state(state)
        book(rec)
        set_progress(plan_dir, lf, state=DONE)
        return DONE
    return cur.get("state", PENDING)
