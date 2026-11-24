"""Rebuild a replaced drive's Shorts plan onto the re-rendered media, or hold the Shorts visibly.

A replace supersedes the project dir (the old ``<project>_publish_plan.json`` moves to
``_superseded/<at>/``) and re-renders every Short to the SAME output path, but a replace plan
run emits no sidecar and enqueues nothing. The queued rows' ``sidecar_path`` then names a file
that is gone, so dispatch raises ``TransientMediaError`` forever. :func:`rebuild_replaced_sidecar`
reads the OLD sidecar from the set-aside copy and, per Short, swaps in a fresh
``fingerprint_media`` of the new bytes at the original sidecar path. Every other field is kept,
including ``parent_lf_video_id`` (rebuild, THEN relink: ``relink_replaced_lf`` rewrites it).

A Short is swapped only when ALL hold, else it is REFUSED (rows left as they are, the record
keeps the old fingerprint so old bytes can never upload silently, one
``replace_rebuild_refused`` incident per Short in ``status --json``):

* identity: the re-render at that path is the same clip (same parent LF, and the stages' Shorts
  audit window and the new media agree with the old duration to ``DURATION_TOLERANCE_S``);
* the new run's privacy attestation covers it (:func:`privacy_hold_reason`, never weakened);
* the mask-frame gate passes (:func:`mask_frame_refusal`, never weakened);
* the plan hash is the derived one (``review_hash`` over title, description, tags and media
  sha), so it can be recomputed; a hand-set hash needs a human re-review and is refused;
* no queued row is inside the 24 h lead time (the rule ``sync-short-text`` applies).

Rows (matched by project, short index and this sidecar path): a queued, unleased row with no
upload id moves ``plan_hash`` to the new hash in one transaction with an audit row (the
``_apply_sidecar`` compare-and-set, commit then ``os.replace``; a rerun converges). Leased,
in-flight and uploaded rows are never changed and block that Short's swap; published and
awaiting-manual rows are never changed and are listed in ``RebuildResult.untouched``.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from gomoveshift.publishing.asset_plan import review_hash
from gomoveshift.publishing.dispatch import (
    PRIVACY_ATTESTATION_GLOB,
    merge_privacy_attestations,
    privacy_hold_reason,
)
from gomoveshift.publishing.mask_frame_gate import mask_frame_refusal
from gomoveshift.publishing.media_fingerprint import fingerprint_media
from gomoveshift.publishing.queue import TERMINAL_STATUSES
from gomoveshift.publishing.queue_short_text import (
    SYNC_LEAD_TIME,
    LeaseRaceError,
    _slot,
)
from gomoveshift.publishing.route_remeasure import DURATION_TOLERANCE_S

INCIDENT_KIND = "replace_rebuild_refused"
_RANK = re.compile(r"short_(\d+)\.", re.IGNORECASE)


@dataclass(frozen=True)
class Rebuilt:
    short_index: int
    old_hash: str
    new_hash: str
    delivery_ids: tuple = ()


@dataclass(frozen=True)
class Refused:
    short_index: int
    reason: str
    delivery_ids: tuple = ()


@dataclass(frozen=True)
class Untouched:
    delivery_id: int
    short_index: int
    platform: str
    status: str
    why: str


@dataclass
class RebuildResult:
    rebuilt: List[Rebuilt] = field(default_factory=list)
    refused: List[Refused] = field(default_factory=list)
    untouched: List[Untouched] = field(default_factory=list)


def privacy_refusal(stages_dir: Path) -> Optional[str]:
    """``None`` when the new run's attestations (one per day, merged worst-wins) verify every
    required class; else the hold reason. Absent attestations are a refusal."""
    base = Path(stages_dir)
    paths = sorted(base.glob(PRIVACY_ATTESTATION_GLOB)) + sorted(
        base.glob("day_*/" + PRIVACY_ATTESTATION_GLOB)
    )
    try:
        docs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    except (OSError, ValueError) as exc:
        return f"privacy attestation unreadable: {exc}"
    return privacy_hold_reason(merge_privacy_attestations(docs))


def _audit_entry(stages_dir: Path, date: str, rank: int) -> Optional[dict]:
    try:
        payload = json.loads((stages_dir / f"shorts_ost_audit_{date}.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    for entry in (payload.get("shorts") if isinstance(payload, dict) else None) or []:
        if isinstance(entry, dict) and entry.get("rank") == rank:
            return entry
    return None


def _identity_refusal(short: dict, new_fp, stages_dir: Path) -> Optional[str]:
    media = short["media"]
    found = _RANK.search(Path(str(media["path"])).name)
    if not found:
        return f"cannot read the clip rank from {media['path']!r}"
    entry = _audit_entry(stages_dir, short["recording_date"], int(found.group(1)))
    if entry is None:
        return f"the new run's Shorts audit has no rank {found.group(1)}: identity unconfirmed"
    if entry.get("parent_lf_index") != short["lf_index"]:
        return (
            f"rank {found.group(1)} now belongs to parent LF {entry.get('parent_lf_index')}, "
            f"the Short to LF {short['lf_index']}"
        )
    old = float(media["duration_seconds"])
    window = float(entry["end_sec"]) - float(entry["start_sec"])
    if abs(window - old) > DURATION_TOLERANCE_S or abs(new_fp.duration_seconds - old) > (
        DURATION_TOLERANCE_S
    ):
        return (
            f"duration mismatch: old {old:g} s, new clip window {window:g} s, "
            f"new media {new_fp.duration_seconds:g} s"
        )
    return None


def _rows(conn: sqlite3.Connection, project: str, index: int, sidecar: Path, hashes) -> list:
    marks = ", ".join("?" for _ in hashes)
    return conn.execute(
        "SELECT * FROM deliveries WHERE project_name = ? AND short_index = ? "  # noqa: S608
        f"AND sidecar_path = ? AND plan_hash IN ({marks}) ORDER BY id",
        (project, index, str(sidecar), *hashes),
    ).fetchall()


def _classify(rows, now: datetime):
    """``(movable, blocking, listed, lead_refusal)`` for one Short's rows."""
    movable, blocking, listed, lead = [], [], [], ""
    for r in rows:
        who = f"{r['platform']} row {r['id']}"
        if r["status"] in TERMINAL_STATUSES:
            listed.append((r, f"{r['status']}: never changed"))
        elif r["status"] == "awaiting_manual_publish":
            listed.append((r, "awaiting manual publish: never changed"))
        elif r["platform_upload_id"] or r["status"] != "queued":
            blocking.append((r, f"{r['status']} with an upload id or in flight"))
        elif r["lease_owner"] is not None and (r["lease_expires_at"] or "") > now.isoformat():
            blocking.append((r, "holds a live lease"))
        else:
            at = _slot(r["scheduled_at"]) if r["scheduled_at"] else None
            if at is None or at - now < SYNC_LEAD_TIME:
                lead = f"{who} slot {r['scheduled_at']} is inside the {SYNC_LEAD_TIME} lead time"
            movable.append(r)
    return movable, blocking, listed, lead


def _raise_incident(project: str, index: int, reason: str, now: datetime, directory) -> None:
    from gomoveshift.health.store import (  # lazy: health imports the pipeline
        default_data_root,
        default_incidents_dir,
        new_incident,
        safe_name,
        write_row,
    )

    key = f"{INCIDENT_KIND}:{project}:{index}"
    directory = directory or default_incidents_dir(default_data_root())
    detail = (
        f"{project} short {index}: the replace re-render was NOT swapped into the plan ({reason}); "
        "its rows are held and keep the old media fingerprint."
    )
    write_row(
        directory,
        new_incident(
            INCIDENT_KIND,
            key,
            f"{safe_name(key)}.json",
            now.isoformat(),
            project=project,
            short_index=index,
            reason=reason,
            detail=detail[:500],
        ),
    )


def _resolve_incident(project: str, index: int, now: datetime, directory) -> None:
    from gomoveshift.health.store import (
        default_data_root,
        default_incidents_dir,
        load_all,
        write_row,
    )

    directory = directory or default_incidents_dir(default_data_root())
    rows, _ = load_all(directory)
    for row in rows:
        if row.get("key") == f"{INCIDENT_KIND}:{project}:{index}" and not row.get("resolved_at"):
            row["resolved_at"] = now.isoformat()
            write_row(directory, row)


def rebuild_replaced_sidecar(
    conn: sqlite3.Connection,
    *,
    old_sidecar: Path,
    new_sidecar: Path,
    project: str,
    stages_dir: Path,
    now: datetime,
    incidents_dir: Optional[Path] = None,
    fingerprint_fn: Callable[[Path], Any] = fingerprint_media,
    privacy_fn: Callable[[Path], Optional[str]] = privacy_refusal,
    mask_fn: Callable[..., Optional[str]] = mask_frame_refusal,
) -> RebuildResult:
    """Rebuild ``new_sidecar`` from ``old_sidecar`` onto the media now at each Short's path.

    Never raises for a refused Short (it is in ``RebuildResult.refused`` plus an incident);
    raises :class:`LeaseRaceError` (nothing written) when a row was leased mid-write."""
    old_sidecar, new_sidecar, stages_dir = Path(old_sidecar), Path(new_sidecar), Path(stages_dir)
    raw = json.loads(old_sidecar.read_text(encoding="utf-8"))
    result = RebuildResult()
    gate: Dict[str, Optional[str]] = {}
    moves: List[tuple] = []  # (delivery_id, short_index, old_hash, new_hash)
    for pos, short in enumerate(raw["shorts"]):
        index, media = short["short_index"], short["media"]
        yt = short["payloads"]["youtube"]
        old_hash = short["reviewed_plan_hash"]
        rows = _rows(conn, project, index, new_sidecar, (old_hash,))
        movable, blocking, listed, lead = _classify(rows, now)
        result.untouched += [
            Untouched(r["id"], index, r["platform"], r["status"], why) for r, why in listed
        ]
        if blocking:
            result.untouched += [
                Untouched(r["id"], index, r["platform"], r["status"], why) for r, why in blocking
            ]
            continue
        ids = tuple(r["id"] for r in movable)
        reason = ""
        new_fp = None
        try:
            path = Path(media["path"])
            if not path.is_file():
                reason = f"re-rendered media missing at {path}"
            else:
                new_fp = fingerprint_fn(path)
        except (OSError, ValueError, RuntimeError) as exc:
            reason = f"cannot fingerprint the new media: {exc}"
        if not reason:
            reason = _identity_refusal(short, new_fp, stages_dir) or ""
        if not reason:
            derived = review_hash(yt["title"], yt["description"], yt["tags"], media["sha256"])
            if old_hash != derived:
                reason = "reviewed_plan_hash is not media-derived: a human re-review is needed"
        if not reason:
            for name, fn in (("privacy", privacy_fn), ("mask-frame", mask_fn)):
                if name not in gate:
                    gate[name] = (
                        fn(stages_dir)
                        if name == "privacy"
                        else fn(stages_dir, drive=project, incidents_dir=incidents_dir)
                    )
                if gate[name]:
                    reason = f"{name} gate: {gate[name]}"
                    break
        if not reason and lead:
            reason = lead
        if reason:
            result.refused.append(Refused(index, reason, ids))
            continue
        new_hash = review_hash(yt["title"], yt["description"], yt["tags"], new_fp.sha256)
        short["media"] = {
            "path": new_fp.path,
            "size_bytes": new_fp.size_bytes,
            "duration_seconds": new_fp.duration_seconds,
            "width": new_fp.width,
            "height": new_fp.height,
            "codec": new_fp.codec,
            "sha256": new_fp.sha256,
        }
        short["reviewed_plan_hash"] = new_hash
        if short.get("consent_hash") == old_hash:
            short["consent_hash"] = new_hash
        raw["shorts"][pos] = short
        result.rebuilt.append(Rebuilt(index, old_hash, new_hash, ids))
        moves += [(r["id"], index, old_hash, new_hash) for r in movable]
    refused_idx = {r.short_index for r in result.refused}
    # Always written: a refused Short keeps its old record (old fingerprint), so the rows'
    # sidecar_path resolves and dispatch's media check refuses the stale bytes visibly.
    _write(conn, raw, old_sidecar, new_sidecar, project, moves, now)
    for r in result.refused:
        try:
            _raise_incident(project, r.short_index, r.reason, now, incidents_dir)
        except OSError:
            pass  # never fatal: an unwritable incidents dir must not stop the rebuild
    for r in result.rebuilt:
        if r.short_index not in refused_idx:
            try:
                _resolve_incident(project, r.short_index, now, incidents_dir)
            except OSError:
                pass
    return result


def _write(conn, raw, old_sidecar, new_sidecar, project, moves, now: datetime) -> None:
    """Queue moves + audit rows in one transaction, THEN ``os.replace`` (``_apply_sidecar``'s
    order); a crash between leaves the old file or none, and a rerun converges."""
    new_sidecar.parent.mkdir(parents=True, exist_ok=True)
    tmp = new_sidecar.with_name(new_sidecar.name + ".rebuild-tmp")
    tmp.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    terminal = ", ".join("?" for _ in TERMINAL_STATUSES)
    try:
        from gomoveshift.publishing.plan_io import load_publish_plan

        load_publish_plan(tmp)  # the rebuilt file must still parse before it replaces anything
        with conn:
            for did, index, old, new in moves:
                moved = conn.execute(
                    "UPDATE deliveries SET plan_hash = ? "  # noqa: S608
                    f"WHERE id = ? AND plan_hash = ? AND status NOT IN ({terminal}) "
                    "AND platform_upload_id IS NULL "
                    "AND (lease_owner IS NULL OR lease_expires_at <= ?)",
                    (new, did, old, *TERMINAL_STATUSES, now.isoformat()),
                ).rowcount
                if moved != 1:  # rolls back every move of this sidecar
                    raise LeaseRaceError("a row was leased during the rebuild")
                conn.execute(
                    "INSERT INTO response_log (at, platform, state, cap, action, mode, reason) "
                    "VALUES (?, 'youtube', 'plan_sync', 0, 'replace_rebuild', 'live', ?)",
                    (
                        now.isoformat(),
                        json.dumps(
                            {
                                "delivery_id": did,
                                "project": project,
                                "short_index": index,
                                "old_plan_hash": old,
                                "new_plan_hash": new,
                                "sidecar": str(new_sidecar),
                                "from": str(old_sidecar),
                            },
                            sort_keys=True,
                        ),
                    ),
                )
        os.replace(tmp, new_sidecar)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
