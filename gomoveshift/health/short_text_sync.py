"""``health sync-short-text``: the unattended nightly Short text sync, with incidents.

Runs ``queue_short_text.sync_short_text(apply=True)`` and turns its outcome into incident
rows under the health incidents dir (``health-incident/1``), pushed once each:

* ``short_text_sync_refused:<project>``: at least one queued Short of that project was
  refused for any reason EXCEPT the few that clear on their own
  (the structured ``ShortTextChange.transient`` flag: a sibling row inside the lead time or
  under the dispatcher's retry backoff, a live lease, the cascade after a failed write that left
  no clash, a write race or a busy database); an unflagged
  reason is actionable, so the check fails closed. The row lists each Short and why; resolved by
  the first run that no longer refuses it. A transient refusal files nothing and does not count
  toward exit 1 UNTIL the same Short has been refused on ``TRANSIENT_ESCALATE_RUNS`` (3)
  consecutive runs (an actionable night in between counts too) and is transient on this one:
  then it is filed like an actionable one, its reason naming the run count. The count lives in one ``short_text_sync_streak:<project>#<n>``
  row per Short in ``<incidents dir>.streaks`` (written with the store's own helpers; a counter,
  not an incident, so ``health check`` and the push never see it), resolved, so the count
  resets, by the first run that no longer refuses the Short. A crash run, and a run that finds
  the store lock held, neither count nor reset it (the count is of runs that completed the
  incident work, not of nights).
* ``health check`` lists these rows (an open one keeps ``health check`` at exit 1, like any
  open incident) but never writes, resolves or pushes them: this command is their only
  writer, and it holds an exclusive non-blocking lock (``<incidents dir>.lock``) around
  the store write. A run that finds the lock held logs it, skips the incident work and exits 0.
* ``short_text_sync_crash``: the sync raised; resolved by the next run that completes.
  A crash proves nothing about the refusals on file, so they stay open until a run completes.

Schedule it AFTER ``metrics collect`` (it writes the nightly title snapshot the sync reads):
``gomoveshift health sync-short-text`` from the service directory (``--data-root`` as the
hourly health cron passes it). Exit 0 clean, 1 when a Short was refused for an actionable reason, 2 on a crash.
A run whose sync returned also stamps ``<incidents dir>.short_text_sync_run.json``
(``run_record_path``; rule 13 flags the job stale when it stops being written).
Never raises: a failure anywhere (sync, store, push) is logged and reported as a crash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from ..pipeline._file_lock import lock_exclusive, unlock
from ..publishing.queue import open_queue_db
from ..publishing.queue_short_text import is_actionable_refusal, sync_short_text
from .store import load_all, new_incident, safe_name, utc_iso, write_row

logger = logging.getLogger(__name__)

REFUSED_KIND = "short_text_sync_refused"
CRASH_KIND = "short_text_sync_crash"
CRASH_KEY = "short_text_sync_crash"
#: A Short refused as transient on this many consecutive runs is filed as an actionable incident:
#: a rewrite that never settles (a writer always holding the lock, a row stuck behind a lease)
#: is no longer clearing on its own.
TRANSIENT_ESCALATE_RUNS = 3
#: A streak row is a counter, not an incident, so it lives in a sibling dir of the incidents dir
#: (like its ``.lock``): the incident rows, ``health check`` and the push never see it.
STREAK_KIND = "short_text_sync_streak"
EVENTS = {
    REFUSED_KIND: "health_short_text_sync_refused",
    CRASH_KIND: "health_short_text_sync_crash",
}


def incident_line(row: Dict[str, Any]) -> str:
    """The one-line push text for a ``short_text_sync_*`` row."""
    if row["kind"] == CRASH_KIND:
        return f"The nightly Short text sync crashed: {row['detail']}. No Short was rewritten."
    shorts = "; ".join(f"#{s['short_index']}: {s['reason']}" for s in row["shorts"])
    return (
        f"The nightly Short text sync refused {len(row['shorts'])} queued Short(s) of "
        f"{row['project']} and left them as stored ({shorts})."
    )


def _streak_key(project: Any, index: Any) -> str:
    return f"{STREAK_KIND}:{project}#{index}"


def _escalated(c: Any, nights: Dict[str, int]) -> bool:
    return (
        c.status == "refused"
        and c.transient
        and nights.get(_streak_key(c.project_name, c.short_index), 0) >= TRANSIENT_ESCALATE_RUNS
    )


def _refused_rows(
    changes: List[Any], now_iso: str, nights: Optional[Dict[str, int]] = None
) -> Dict[str, Dict[str, Any]]:
    """One row per project with an actionable refusal; a transient one that has repeated
    ``TRANSIENT_ESCALATE_RUNS`` runs (``nights``: streak key -> runs) counts as actionable."""
    nights = nights or {}
    by_project: Dict[str, List[Dict[str, Any]]] = {}
    for c in changes:
        long_transient = _escalated(c, nights)
        if is_actionable_refusal(c) or long_transient:
            reason = " ".join(c.reason.split())
            if long_transient:
                n = nights[_streak_key(c.project_name, c.short_index)]
                reason = f"refused as transient on {n} consecutive runs: {reason}"
            by_project.setdefault(c.project_name, []).append(
                {"delivery_id": c.delivery_id, "short_index": c.short_index, "reason": reason}
            )
    out: Dict[str, Dict[str, Any]] = {}
    for project, shorts in sorted(by_project.items()):
        key = f"{REFUSED_KIND}:{project}"
        row = new_incident(
            REFUSED_KIND,
            key,
            f"{REFUSED_KIND}_{safe_name(project)}.json",
            now_iso,
            project=project,
            shorts=sorted(shorts, key=lambda s: s["short_index"]),
        )
        row["detail"] = incident_line(row)
        out[key] = row
    return out


def _brand_location_cap() -> Optional[int]:
    """The brand policy's per-location cap, as ``queue sync-short-text`` loads it."""
    from ..console.settings import current_brand  # noqa: PLC0415
    from ..publishing.brand_policy import load_brand_policy  # noqa: PLC0415

    return load_brand_policy(current_brand()).per_location_cap


@contextmanager
def _store_lock(incidents_dir: Path) -> Iterator[bool]:
    """Exclusive, non-blocking portable file lock on a sibling of the incidents dir; False if held."""
    lock_path = incidents_dir.parent / f"{incidents_dir.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as handle:
        try:
            lock_exclusive(handle, blocking=False)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            unlock(handle)


def _streaks_dir(incidents_dir: Path) -> Path:
    return incidents_dir.parent / f"{incidents_dir.name}.streaks"


def run_record_path(incidents_dir: Path) -> Path:
    """The stamp of the last completed run, next to the incidents dir (read by rule 13)."""
    return incidents_dir.parent / f"{incidents_dir.name}.short_text_sync_run.json"


def _write_run_record(incidents_dir: Path, now_iso: str, changed: int, refused: int) -> None:
    """Atomically stamp a run whose sync returned. Best effort: a failure is logged, never raised."""
    try:
        path = run_record_path(incidents_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            tmp.write_text(
                json.dumps(
                    {"schema": 1, "completed_at": now_iso, "changed": changed, "refused": refused}
                ),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 -- never changes the exit code
        logger.exception("sync-short-text could not write its run record")


def _update_streaks(changes: List[Any], incidents_dir: Path, now_iso: str) -> Dict[str, int]:
    """Count the consecutive runs each Short has been refused as transient: one streak row per
    Short (``short_text_sync_streak:<project>#<index>``, resolved the first run that no longer
    refuses it, like every other row here). -> streak key -> runs, this one included."""
    streaks_dir = _streaks_dir(incidents_dir)
    prior = {r["key"]: r for r in load_all(streaks_dir)[0] if r.get("kind") == STREAK_KIND}
    seen = {
        _streak_key(c.project_name, c.short_index): c
        for c in changes
        if c.status == "refused"  # every refusal counts; only a transient one escalates
    }
    for key, row in prior.items():
        if key not in seen and not row.get("resolved_at"):
            row["resolved_at"] = now_iso
            write_row(streaks_dir, row)
    nights: Dict[str, int] = {}
    for key, c in seen.items():
        row = prior.get(key)
        if row is None or row.get("resolved_at"):
            row = new_incident(
                STREAK_KIND,
                key,
                f"{STREAK_KIND}_{safe_name(c.project_name)}_{c.short_index}_"
                f"{hashlib.sha1(key.encode()).hexdigest()[:8]}.json",  # noqa: S324 -- a filename tag
                now_iso,
                project=c.project_name,
                short_index=c.short_index,
                nights=0,
            )
        row["nights"] = int(row.get("nights") or 0) + 1
        row["last_seen_at"] = now_iso
        write_row(streaks_dir, row)
        nights[key] = row["nights"]
    return nights


def _file_incidents(
    changes: List[Any],
    crash: Optional[str],
    incidents_dir: Path,
    now_iso: str,
    notify_fn: Callable[[str, Dict[str, Any], str], Any],
):
    """File, refresh and resolve the rows, push each new one once. -> (new, resolved, pushed, open, nights)."""
    existing, _bad = load_all(incidents_dir)
    by_key = {r["key"]: r for r in existing}
    nights: Dict[str, int] = {}
    found: Dict[str, Dict[str, Any]] = {}
    if crash is not None:
        found[CRASH_KEY] = new_incident(
            CRASH_KIND, CRASH_KEY, f"{CRASH_KIND}.json", now_iso, detail=crash
        )
    else:
        # a crash proves nothing about the Shorts' refusals: it neither counts nor resets a streak
        nights = _update_streaks(changes, incidents_dir, now_iso)
        found = _refused_rows(changes, now_iso, nights)

    resolved: List[str] = []
    for row in existing:
        if row.get("resolved_at") or row.get("kind") not in EVENTS or row["key"] in found:
            continue
        if crash is not None and row["kind"] == REFUSED_KIND:
            continue  # a crash proves nothing about the refusals on file
        row["resolved_at"] = now_iso
        write_row(incidents_dir, row)
        resolved.append(row["key"])

    new: List[str] = []
    for key, row in found.items():
        prior = by_key.get(key)
        if prior is not None and not prior.get("resolved_at"):
            if prior.get("kind") == REFUSED_KIND and prior.get("shorts") != row["shorts"]:
                prior.update(shorts=row["shorts"], detail=row["detail"])
                write_row(incidents_dir, prior)
            continue
        write_row(incidents_dir, row)
        by_key[key] = row
        new.append(key)

    pushed: List[str] = []
    open_rows = [r for r in by_key.values() if not r.get("resolved_at") and r["kind"] in EVENTS]
    for row in sorted(open_rows, key=lambda r: r["key"]):
        if row.get("pushed"):
            continue
        if notify_fn(EVENTS[row["kind"]], {"incident": row["key"]}, incident_line(row)) is False:
            continue  # attempted and failed: stays unrecorded, the next night retries
        row["pushed"] = True
        row["pushed_at"] = now_iso
        write_row(incidents_dir, row)
        pushed.append(row["key"])
    return new, resolved, pushed, sorted(open_rows, key=lambda r: r["key"]), nights


def run_short_text_sync(
    config: Any,
    *,
    incidents_dir: Path,
    now: Optional[datetime] = None,
    notify_fn: Optional[Callable[[str, Dict[str, Any], str], Any]] = None,
) -> Dict[str, Any]:
    """Sync, file and resolve the incidents, push each new one once.

    Returns ``{"exit", "changed", "refused", "crashed", "skipped", "changes", "new",
    "resolved", "pushed", "incidents"}``. Never raises: a crash of the sync, or of the
    incident work after it, is logged and becomes a crash incident (best effort)."""
    now = now or datetime.now(timezone.utc)
    now_iso = utc_iso(now.timestamp())
    changes: List[Any] = []
    crash: Optional[str] = None
    synced = False
    try:
        if notify_fn is None:
            from ..console.notify import notify as notify_fn  # noqa: PLC0415
        conn = open_queue_db(config.seo.storage_path)
        try:
            changes = sync_short_text(
                conn, apply=True, now=now, per_location_cap=_brand_location_cap()
            )
            synced = True
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 -- the crash is the incident
        logger.exception("sync-short-text crashed")
        crash = f"{type(exc).__name__}: {exc}"

    new: List[str] = []
    resolved: List[str] = []
    pushed: List[str] = []
    open_rows: List[Dict[str, Any]] = []
    nights: Dict[str, int] = {}
    skipped = False
    try:
        with _store_lock(incidents_dir) as got:
            if not got:
                skipped = True
                logger.warning("sync-short-text: the incident store is locked; skipping")
            else:
                notify = notify_fn
                if notify is None:  # the import above failed: file, push nothing
                    notify = lambda *_a: False  # noqa: E731
                new, resolved, pushed, open_rows, nights = _file_incidents(
                    changes, crash, incidents_dir, now_iso, notify
                )
    except Exception as exc:  # noqa: BLE001 -- never raises
        logger.exception("sync-short-text incident work crashed")
        crash = crash or f"{type(exc).__name__}: {exc}"
        try:
            write_row(
                incidents_dir,
                new_incident(CRASH_KIND, CRASH_KEY, f"{CRASH_KIND}.json", now_iso, detail=crash),
            )
        except Exception:  # noqa: BLE001
            logger.exception("sync-short-text could not file its crash incident")

    refused = sum(is_actionable_refusal(c) or _escalated(c, nights) for c in changes)
    changed = sum(c.status in ("changed", "reconciled") for c in changes)
    if synced:  # a crash of the sync stamps nothing: short_text_sync_crash covers it
        _write_run_record(incidents_dir, now_iso, changed, refused)
    if skipped and crash is None:
        code = 0
    else:
        code = 2 if crash is not None else (1 if refused else 0)
    return {
        "exit": code,
        "changed": changed,
        "refused": refused,
        "crashed": crash,
        "skipped": skipped,
        "changes": changes,
        "new": new,
        "resolved": resolved,
        "pushed": pushed,
        "incidents": open_rows,
    }
