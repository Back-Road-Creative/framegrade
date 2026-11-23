"""Rebuild a replaced drive's Shorts plan onto the re-rendered media, or hold the Shorts visibly.

A replace supersedes the project dir (the old sidecar moves to ``_superseded/<at>/``) and
re-renders every Short to the SAME path but enqueues nothing, so queued rows name a gone file.
:func:`rebuild_replaced_sidecar` reads the OLD sidecar and swaps in a fresh ``fingerprint_media``
at the original path; other fields are kept (rebuild, THEN relink ``parent_lf_video_id``).

Gate C/D is checked first (a refusal refuses every Short, one incident). A Short is swapped only
when it is the same FOOTAGE (old and new audit segments agree within ``DURATION_TOLERANCE_S``),
its OWN day has a verified attestation, the mask gate passes, its hash is the derived one and no
row blocks it; else it is REFUSED (old fingerprint kept, one incident). A Short that is only
waiting (live lease, in-flight row, future slot inside ``SYNC_LEAD_TIME``) is DEFERRED; a past or
missing slot moves. The per-day attestations are placed beside the sidecar in the same write.
Rows move by compare-and-set (commit, then ``os.replace``); a crash leaves rows ahead of the file,
which a refusing rerun restores. See ``docs/social-publishing-queue.md``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from gomoveshift.publishing.asset_plan import review_hash
from gomoveshift.publishing.dispatch import merge_privacy_attestations, privacy_hold_reason
from gomoveshift.publishing.mask_frame_gate import mask_frame_refusal
from gomoveshift.publishing.media_fingerprint import fingerprint_media
from gomoveshift.publishing.queue import TERMINAL_STATUSES
from gomoveshift.publishing.queue_short_text import SYNC_LEAD_TIME, LeaseRaceError, _leased, _slot
from gomoveshift.publishing.route_remeasure import DURATION_TOLERANCE_S

logger = logging.getLogger(__name__)

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
class Deferred:
    """A Short the swap cannot move yet; the next run retries it."""

    short_index: int
    reason: str
    retry_at: Optional[str] = None
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
    deferred: List[Deferred] = field(default_factory=list)
    untouched: List[Untouched] = field(default_factory=list)


def privacy_refusal(stages_dir: Path, date: str) -> Optional[str]:
    """``None`` when the new run's attestation for THIS day verifies every required class; another
    day's never covers this Short and a missing day is a refusal (never merged to a pass)."""
    from gomoveshift.pipeline.day_master_encoder import PRIVACY_ATTESTATION_NAME

    path = Path(stages_dir) / f"day_{date}" / PRIVACY_ATTESTATION_NAME.format(date=date)
    if not path.is_file():
        return f"no privacy attestation for {date} at {path}"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"privacy attestation unreadable: {exc}"
    if not isinstance(doc, dict) or doc.get("date", date) != date:
        return f"the attestation at {path} is not for {date}"
    return privacy_hold_reason(merge_privacy_attestations([doc]))


def _audit_entry(stages_dir: Path, date: str, rank: int) -> Optional[dict]:
    try:
        payload = json.loads((stages_dir / f"shorts_ost_audit_{date}.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    for entry in (payload.get("shorts") if isinstance(payload, dict) else None) or []:
        if isinstance(entry, dict) and entry.get("rank") == rank:
            return entry
    return None


def _identity_refusal(short: dict, new_fp, stages_dir: Path, old_stages_dir: Path) -> Optional[str]:
    """Same footage, not same rank: old and new audit name the same source segment."""
    media = short["media"]
    found = _RANK.search(Path(str(media["path"])).name)
    if not found:
        return f"cannot read the clip rank from {media['path']!r}"
    rank = int(found.group(1))
    new = _audit_entry(stages_dir, short["recording_date"], rank)
    old_entry = _audit_entry(old_stages_dir, short["recording_date"], rank)
    for which, entry in (("new", new), ("old", old_entry)):
        if entry is None:
            return f"the {which} run's Shorts audit has no rank {rank}: identity unconfirmed"
    try:
        for label, entry in (("old", old_entry), ("new", new)):
            if entry.get("parent_lf_index") != short["lf_index"]:
                return (
                    f"rank {rank} in the {label} run belongs to parent LF "
                    f"{entry.get('parent_lf_index')}, the Short to LF {short['lf_index']}"
                )
        o_start, o_end = float(old_entry["start_sec"]), float(old_entry["end_sec"])
        n_start, n_end = float(new["start_sec"]), float(new["end_sec"])
        old = float(media["duration_seconds"])
        new_dur = float(new_fp.duration_seconds)
    except (KeyError, TypeError, ValueError) as exc:
        return f"malformed audit entry for rank {rank}: {exc!r}"
    if abs(n_start - o_start) > DURATION_TOLERANCE_S or abs(n_end - o_end) > DURATION_TOLERANCE_S:
        return (
            f"different source segment: old {o_start:g}-{o_end:g} s, "
            f"new {n_start:g}-{n_end:g} s (rank {rank} no longer names the same footage)"
        )
    window = n_end - n_start
    if abs(window - old) > DURATION_TOLERANCE_S or abs(new_dur - old) > DURATION_TOLERANCE_S:
        return f"duration mismatch: old {old:g} s, window {window:g} s, new media {new_dur:g} s"
    return None


def _same_file(raw, target: Path) -> bool:
    try:
        return bool(raw) and Path(str(raw)).resolve() == target
    except (OSError, RuntimeError):
        return False


def _rows(conn: sqlite3.Connection, project: str, index: int, plan_hash: str) -> list:
    return conn.execute(
        "SELECT * FROM deliveries WHERE project_name = ? AND short_index = ? "
        "AND plan_hash = ? ORDER BY id",
        (project, index, plan_hash),
    ).fetchall()


def _classify(rows, now: datetime, target: Path):
    """``(movable, listed, blocking, refuse, defer, retry_at)``; refuse/defer are reasons (``""`` =
    none) and refuse beats defer."""
    movable, listed, blocking, refuse, defer, retries = [], [], [], "", "", []
    for r in rows:
        who = f"{r['platform']} row {r['id']}"
        done = r["status"] in TERMINAL_STATUSES
        if not _same_file(r["sidecar_path"], target):
            if not done:
                refuse = refuse or f"{who} points at another sidecar file ({r['sidecar_path']})"
        elif r["recovers_delivery_id"] is not None:
            refuse = refuse or f"{who} is a recovery row"
        elif done:
            listed.append((r, f"{r['status']}: never changed"))
        elif r["status"] == "awaiting_manual_publish":
            listed.append((r, "awaiting manual publish: never changed"))
        elif r["platform_upload_id"]:
            blocking.append((r, f"{r['status']} with an upload id"))
            refuse = refuse or f"{who} is already uploaded: one Short cannot be half old, half new"
        elif r["status"] != "queued":
            blocking.append((r, f"{r['status']}: in flight"))
            defer = defer or f"{who} is {r['status']} (in flight)"
        elif _leased(r, now):
            blocking.append((r, "holds a live lease"))
            defer = defer or f"{who} holds a live lease"
        else:
            at = _slot(r["scheduled_at"]) if r["scheduled_at"] else None
            if at is not None and timedelta(0) < at - now < SYNC_LEAD_TIME:
                retries.append(at)
                defer = (
                    defer
                    or f"{who} is queued for {at.isoformat()}, inside the {SYNC_LEAD_TIME} lead time"
                )
            movable.append(r)
    retry_at = min(retries).isoformat() if retries else None
    return movable, listed, blocking, refuse, defer, retry_at


def _ahead(conn: sqlite3.Connection, project: str, index: int, old_hash: str, target: Path):
    """Rows a crashed run moved but whose file never replaced: ``[(id, their_hash)]``."""
    out = set()
    for (reason,) in conn.execute(
        "SELECT reason FROM response_log WHERE action = 'replace_rebuild'"
    ):
        try:
            doc = json.loads(reason)
        except (TypeError, ValueError):
            continue
        if (doc.get("project"), doc.get("short_index"), doc.get("old_plan_hash")) != (
            project,
            index,
            old_hash,
        ):
            continue
        r = conn.execute(
            "SELECT * FROM deliveries WHERE id = ? AND plan_hash = ?",
            (doc.get("delivery_id"), doc.get("new_plan_hash")),
        ).fetchone()
        if r and _same_file(r["sidecar_path"], target) and r["recovers_delivery_id"] is None:
            out.add((r["id"], r["plan_hash"]))
    return sorted(out)


def _incident_dir(directory):
    from gomoveshift.health.store import default_data_root, default_incidents_dir

    return directory or default_incidents_dir(default_data_root())


def _raise_incident(project: str, index, reason: str, now: datetime, directory, *, what="") -> None:
    """Open (or leave alone) a Short's incident (``index=None``: the project's Gate C/D one).
    Never fatal, never silent: a write failure is logged at error level."""
    from gomoveshift.health.store import load_all, new_incident, safe_name, write_row

    key = f"{INCIDENT_KIND}:{project}:{'gate_cd' if index is None else index}"
    detail = f"{project} short {index}: the replace swap was refused ({reason}); rows held."
    if what == "deferred":
        detail = (
            f"{project} short {index}: the replace swap is DEFERRED ({reason}); it will be retried."
        )
    try:
        directory = _incident_dir(directory)
        if any(
            r.get("key") == key and not r.get("resolved_at") and r.get("reason") == reason
            for r in load_all(directory)[0]
        ):
            return  # already open with this reason: not rewritten, not re-pushed
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
    except (OSError, ValueError):
        logger.exception("replace-rebuild: could not write the incident %s", key)


def _resolve_incident(project: str, index, now: datetime, directory) -> None:
    from gomoveshift.health.store import load_all, write_row

    key = f"{INCIDENT_KIND}:{project}:{'gate_cd' if index is None else index}"
    try:
        directory = _incident_dir(directory)
        for row in load_all(directory)[0]:
            if row.get("key") == key and not row.get("resolved_at"):
                row["resolved_at"] = now.isoformat()
                write_row(directory, row)
    except (OSError, ValueError):
        logger.exception("replace-rebuild: could not resolve the incident %s", key)


def _gate_cd_refusal(stages_dir: Path, output_dir: Path) -> Optional[str]:
    """The enqueue-plan Gate C/D check on the replace run's state (``None`` = passes); an unreadable
    state refuses, as ``enqueue-plan`` does."""
    from gomoveshift.cli.handlers import enqueue_plan

    try:
        state = enqueue_plan._load_state(stages_dir / "pipeline_state.json")
        enqueue_plan.assert_gate_cd_approved(state, output_dir)
    except enqueue_plan.GateNotApprovedError as exc:
        return f"Gate C/D must be re-confirmed: {exc}"
    return None


def rebuild_replaced_sidecar(
    conn: sqlite3.Connection,
    *,
    old_sidecar: Path,
    new_sidecar: Path,
    project: str,
    stages_dir: Path,
    old_stages_dir: Path,
    now: datetime,
    incidents_dir: Optional[Path] = None,
    fingerprint_fn: Callable[[Path], Any] = fingerprint_media,
    privacy_fn: Callable[[Path, str], Optional[str]] = privacy_refusal,
    mask_fn: Callable[..., Optional[str]] = mask_frame_refusal,
) -> RebuildResult:
    """Rebuild ``new_sidecar`` from ``old_sidecar`` onto the media now at each Short's path.

    ``stages_dir`` is the replace run's stage dir, ``old_stages_dir`` the set-aside run's.
    Never raises for a refused or deferred Short (it is in ``RebuildResult`` plus an incident);
    raises :class:`LeaseRaceError` (nothing written) when a row changed during the write."""
    old_sidecar, new_sidecar = Path(old_sidecar), Path(new_sidecar)
    stages_dir, old_stages_dir = Path(stages_dir), Path(old_stages_dir)
    raw = json.loads(old_sidecar.read_text(encoding="utf-8"))
    target = new_sidecar.resolve()
    result = RebuildResult()
    gate: Dict[Any, Optional[str]] = {}
    gate_cd = _gate_cd_refusal(stages_dir, new_sidecar.parent)
    moves: List[tuple] = []  # (delivery_id, short_index, from_hash, to_hash, action)
    staged: List[tuple] = []  # (pos, new_short, Rebuilt, moves), applied after placement
    for pos, short in enumerate(raw["shorts"]):
        index, media = short["short_index"], short["media"]
        yt = short["payloads"]["youtube"]
        old_hash = short["reviewed_plan_hash"]
        movable, listed, blocking, refuse, defer, retry_at = _classify(
            _rows(conn, project, index, old_hash), now, target
        )
        result.untouched += [
            Untouched(r["id"], index, r["platform"], r["status"], why)
            for r, why in listed + blocking
        ]
        ids = tuple(r["id"] for r in movable)
        reason = gate_cd or refuse
        new_fp = None
        if not reason and not defer:
            try:
                path = Path(media["path"])
                if not path.is_file():
                    reason = f"re-rendered media missing at {path}"
                else:
                    new_fp = fingerprint_fn(path)
            except (OSError, ValueError, RuntimeError) as exc:
                reason = f"cannot fingerprint the new media: {exc}"
            if not reason:
                reason = _identity_refusal(short, new_fp, stages_dir, old_stages_dir) or ""
            if not reason:
                derived = review_hash(yt["title"], yt["description"], yt["tags"], media["sha256"])
                if old_hash != derived:
                    reason = "reviewed_plan_hash is not media-derived: a human re-review is needed"
            day = short.get("recording_date")
            if not reason and not day:
                reason = "the Short has no recording_date: its privacy day is unknown"
            if not reason and ("privacy", day) not in gate:
                gate["privacy", day] = privacy_fn(stages_dir, day)
            if not reason and gate["privacy", day]:
                reason = f"privacy gate: {gate['privacy', day]}"
            if not reason and "mask" not in gate:
                gate["mask"] = mask_fn(stages_dir, drive=project, incidents_dir=incidents_dir)
            if not reason and gate["mask"]:
                reason = f"mask-frame gate: {gate['mask']}"
        if reason or defer:
            ahead = _ahead(conn, project, index, old_hash, target)
            moves += [
                (did, index, theirs, old_hash, "replace_rebuild_restore") for did, theirs in ahead
            ]
            ids += tuple(did for did, _ in ahead)
            if reason:
                result.refused.append(Refused(index, reason, ids))
            else:
                result.deferred.append(Deferred(index, defer, retry_at, ids))
            continue
        new_hash = review_hash(yt["title"], yt["description"], yt["tags"], new_fp.sha256)
        swapped = dict(short)
        fields = ("path", "size_bytes", "duration_seconds", "width", "height", "codec", "sha256")
        swapped["media"] = {k: getattr(new_fp, k) for k in fields}
        swapped["reviewed_plan_hash"] = new_hash
        if short.get("consent_hash") == old_hash:
            swapped["consent_hash"] = new_hash
        staged.append(
            (
                pos,
                swapped,
                Rebuilt(index, old_hash, new_hash, ids),
                [(r["id"], index, old_hash, new_hash, "replace_rebuild") for r in movable],
            )
        )
    if staged:
        try:
            _place_attestations(new_sidecar, stages_dir, raw)
        except (OSError, ValueError) as exc:
            why = f"cannot place the per-day privacy attestations beside the plan: {exc}"
            result.refused += [Refused(st[2].short_index, why, st[2].delivery_ids) for st in staged]
            staged = []
    for pos, swapped, rebuilt, mv in staged:
        raw["shorts"][pos] = swapped
        result.rebuilt.append(rebuilt)
        moves += mv
    result.refused.sort(key=lambda r: r.short_index)
    # Always written: a refused Short keeps its old record, so dispatch refuses the stale bytes.
    _write(conn, raw, old_sidecar, new_sidecar, project, moves, now)
    if gate_cd:
        _raise_incident(project, None, gate_cd, now, incidents_dir)
    else:
        _resolve_incident(project, None, now, incidents_dir)
    bad = {r.short_index for r in result.refused} | {d.short_index for d in result.deferred}
    for r in result.refused if not gate_cd else []:
        _raise_incident(project, r.short_index, r.reason, now, incidents_dir)
    for d in result.deferred:
        when = f"; earliest retry {d.retry_at}" if d.retry_at else ""
        _raise_incident(
            project, d.short_index, d.reason + when, now, incidents_dir, what="deferred"
        )
    for r in result.rebuilt:
        if r.short_index not in bad:
            _resolve_incident(project, r.short_index, now, incidents_dir)
    return result


def _place_attestations(new_sidecar: Path, stages_dir: Path, raw: dict) -> None:
    """Put the per-day attestations beside the sidecar."""
    from gomoveshift.pipeline.drive_phases import phase_d

    dates = sorted({s["recording_date"] for s in raw["shorts"] if s.get("recording_date")})
    phase_d._sync_privacy_attestations(
        output_base=new_sidecar.parent, stages_base=stages_dir, dates=dates
    )


def _write(conn, raw, old_sidecar, new_sidecar, project, moves, now: datetime) -> None:
    """Queue moves + audit rows in one transaction, THEN ``os.replace``."""
    new_sidecar.parent.mkdir(parents=True, exist_ok=True)
    tmp = new_sidecar.with_name(new_sidecar.name + ".rebuild-tmp")
    tmp.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    try:
        from gomoveshift.publishing.plan_io import load_publish_plan

        load_publish_plan(tmp)  # the rebuilt file must still parse before it replaces anything
        with conn:
            for did, index, old, new, action in moves:
                moved = conn.execute(
                    "UPDATE deliveries SET plan_hash = ? "
                    "WHERE id = ? AND plan_hash = ? AND status = 'queued' "
                    "AND platform_upload_id IS NULL "
                    "AND (lease_owner IS NULL OR lease_expires_at <= ?)",
                    (new, did, old, now.isoformat()),
                ).rowcount
                if moved != 1:  # rolls back every move of this sidecar
                    raise LeaseRaceError("a row changed during the rebuild")
                conn.execute(
                    "INSERT INTO response_log (at, platform, state, cap, action, mode, reason) "
                    "VALUES (?, 'youtube', 'plan_sync', 0, ?, 'live', ?)",
                    (
                        now.isoformat(),
                        action,
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
