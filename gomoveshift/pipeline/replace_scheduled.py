"""Re-render a drive and replace its long-form uploads that are scheduled but not yet public.

``health backlog --supersede ... --replace-scheduled`` sets a drive's run aside even though its
``youtube_uploads.json`` holds records, when every record is either public (its LF is kept) or
still scheduled well ahead (its LF is replaced). ``classify_uploads`` decides that, failing closed;
the plan it returns is written beside the superseded run and into the fresh output dir as
``REPLACE_PLAN_FILENAME``, which Phase D reads (``load_plan``):

* a kept LF is put back into ``state.youtube_uploads`` (``apply_kept``), so ``upload_if_needed``
  never uploads it again;
* a replaced LF is re-checked live (``old_still_scheduled``); still scheduled -> the new video is
  uploaded with the SAME ``publishAt`` and title, and only then the old id is deleted
  (``finish_replacement``); gone public meanwhile -> the old video is kept and nothing is uploaded.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

REPLACE_PLAN_FILENAME = "replace_plan.json"
REPLACE_PLAN_SCHEMA = "replace-scheduled/1"
#: A scheduled upload is replaced only while its ``publishAt`` is at least this far ahead: the
#: re-render takes hours, and a video that goes public mid-render must never be deleted.
REPLACE_MARGIN_H = 6
#: ``{video_id: state}`` for the ids asked about; ``"scheduled"`` is a private video with a future
#: ``publishAt`` (``playlist_reconciler.fetch_video_privacy``); an id absent is deleted.
LiveStatus = Callable[[list], Mapping[str, str]]


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
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPLACE_PLAN_FILENAME
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
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


def apply_kept(state: Any, plan: Mapping[str, Any]) -> list:
    """Put each kept LF's record into ``state.youtube_uploads`` unless it holds one; returns the
    LF indices kept. ``upload_if_needed`` then skips them."""
    if state.youtube_uploads is None:
        state.youtube_uploads = {}
    kept = []
    for key, rec in (plan.get("kept") or {}).items():
        idx = _lf(key)
        state.youtube_uploads.setdefault(idx, dict(rec))
        kept.append(idx)
    return kept


def old_still_scheduled(live_status: LiveStatus, old: Mapping[str, Any]) -> bool:
    """True only when YouTube says the old video is private with a future ``publishAt`` now."""
    try:
        return live_status([old["video_id"]]).get(old["video_id"]) == "scheduled"
    except Exception as exc:  # noqa: BLE001 -- not decidable means not replaceable
        logger.warning("replace: live check of %s failed: %s", old.get("video_id"), exc)
        return False


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


def finish_replacement(
    yt: Any, *, old: Mapping[str, Any], new_id: str, ledger_path: Path, plan_dir: Path
) -> None:
    """After the new upload succeeded and was booked: re-point the ledger at it, then delete the
    old video. A failed delete is logged loudly and left in the plan for the next run."""
    old_id = old["video_id"]
    try:
        swap_ledger_video_id(ledger_path, old_id, new_id)
    except Exception:  # noqa: BLE001 -- the live upload is already irreversible
        logger.error("REPLACE-LEDGER-UNSWAPPED: %s -> %s", old_id, new_id, exc_info=True)
    try:
        yt.delete_video(old_id)
    except Exception:  # noqa: BLE001
        logger.error("REPLACE-OLD-UNDELETED: old video %s still exists; delete it", old_id, exc_info=True)  # fmt: skip
        return
    plan = load_plan(plan_dir)
    plan.setdefault("deleted", []).append(old_id)
    write_plan(plan_dir, plan)


def note_kept_live(plan_dir: Path, lf: int, why: str) -> None:
    """Record that a replaced LF's old video was kept (``kept_live``), so the run says why."""
    plan = load_plan(plan_dir)
    plan.setdefault("kept_live", {})[str(lf)] = {"why": why, "at": datetime.now(timezone.utc).isoformat()}  # fmt: skip
    write_plan(plan_dir, plan)
