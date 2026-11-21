"""``health sync-short-text``: the unattended nightly Short text sync, with incidents.

Runs ``queue_short_text.sync_short_text(apply=True)`` and turns its outcome into incident
rows under the health incidents dir (``health-incident/1``), pushed once each:

* ``short_text_sync_refused:<project>``: at least one queued Short of that project was
  refused for any reason EXCEPT the few that clear on their own
  (the structured ``ShortTextChange.transient`` flag: a sibling row inside the lead time, a live
  lease, the cascade after a failed write, a write race or a busy database); an unflagged
  reason is actionable, so the check fails closed. The row lists each Short and why; resolved by the first run that no
  longer refuses it. A transient refusal files nothing and does not count toward exit 1.
* ``health check`` lists these rows (an open one keeps ``health check`` at exit 1, like any
  open incident) but never writes, resolves or pushes them: this command is their only
  writer, and it holds an exclusive non-blocking lock (``<incidents dir>.lock``) around
  the store write. A run that finds the lock held logs it, skips the incident work and exits 0.
* ``short_text_sync_crash``: the sync raised; resolved by the next run that completes.
  A crash proves nothing about the refusals on file, so they stay open until a run completes.

Schedule it AFTER ``metrics collect`` (it writes the nightly title snapshot the sync reads):
``gomoveshift health sync-short-text`` from the service directory (``--data-root`` as the
hourly health cron passes it). Exit 0 clean, 1 when a Short was refused for an actionable reason, 2 on a crash.
Never raises: a failure anywhere (sync, store, push) is logged and reported as a crash.
"""

from __future__ import annotations

import fcntl
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from ..publishing.queue import open_queue_db
from ..publishing.queue_short_text import is_actionable_refusal, sync_short_text
from .store import load_all, new_incident, safe_name, utc_iso, write_row

logger = logging.getLogger(__name__)

REFUSED_KIND = "short_text_sync_refused"
CRASH_KIND = "short_text_sync_crash"
CRASH_KEY = "short_text_sync_crash"
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


def _refused_rows(changes: List[Any], now_iso: str) -> Dict[str, Dict[str, Any]]:
    by_project: Dict[str, List[Dict[str, Any]]] = {}
    for c in changes:
        if is_actionable_refusal(c):
            by_project.setdefault(c.project_name, []).append(
                {
                    "delivery_id": c.delivery_id,
                    "short_index": c.short_index,
                    "reason": " ".join(c.reason.split()),
                }
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
    """Exclusive, non-blocking ``flock`` on a sibling of the incidents dir; False if held."""
    lock_path = incidents_dir.parent / f"{incidents_dir.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _file_incidents(
    changes: List[Any],
    crash: Optional[str],
    incidents_dir: Path,
    now_iso: str,
    notify_fn: Callable[[str, Dict[str, Any], str], Any],
):
    """File, refresh and resolve the rows, push each new one once. -> (new, resolved, pushed, open)."""
    existing, _bad = load_all(incidents_dir)
    by_key = {r["key"]: r for r in existing}
    found: Dict[str, Dict[str, Any]] = {}
    if crash is not None:
        found[CRASH_KEY] = new_incident(
            CRASH_KIND, CRASH_KEY, f"{CRASH_KIND}.json", now_iso, detail=crash
        )
    else:
        found = _refused_rows(changes, now_iso)

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
    return new, resolved, pushed, sorted(open_rows, key=lambda r: r["key"])


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
    try:
        if notify_fn is None:
            from ..console.notify import notify as notify_fn  # noqa: PLC0415
        conn = open_queue_db(config.seo.storage_path)
        try:
            changes = sync_short_text(
                conn, apply=True, now=now, per_location_cap=_brand_location_cap()
            )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 -- the crash is the incident
        logger.exception("sync-short-text crashed")
        crash = f"{type(exc).__name__}: {exc}"

    new: List[str] = []
    resolved: List[str] = []
    pushed: List[str] = []
    open_rows: List[Dict[str, Any]] = []
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
                new, resolved, pushed, open_rows = _file_incidents(
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

    refused = sum(is_actionable_refusal(c) for c in changes)
    if skipped and crash is None:
        code = 0
    else:
        code = 2 if crash is not None else (1 if refused else 0)
    return {
        "exit": code,
        "changed": sum(c.status in ("changed", "reconciled") for c in changes),
        "refused": refused,
        "crashed": crash,
        "skipped": skipped,
        "changes": changes,
        "new": new,
        "resolved": resolved,
        "pushed": pushed,
        "incidents": open_rows,
    }
