"""Rebuild a replaced drive's Shorts plan onto the re-rendered media, or hold the Shorts visibly.

A replace supersedes the project dir (the old sidecar moves to ``_superseded/<at>/``) and
re-renders every Short to the SAME path but enqueues nothing, so queued rows name a gone file.
:func:`rebuild_replaced_sidecar` reads the OLD sidecar and swaps in a fresh ``fingerprint_media``
at the original path; other fields are kept (rebuild, THEN relink ``parent_lf_video_id``).

Gate C/D is checked first (a refusal refuses every Short, one incident; the swap-invariant hash is
accepted only here, and only for a plan that proves to be a swap of the old one). A Short is swapped only
when it is the same FOOTAGE (old and new audit segments agree within ``DURATION_TOLERANCE_S``),
its OWN day has a verified attestation, the mask gate passes, its hash is the derived one and no
row blocks it; else it is REFUSED (old fingerprint kept, one incident). A Short that is only
waiting (live lease, in-flight row, future slot inside ``SYNC_LEAD_TIME``) is DEFERRED; a past or
missing slot moves. The per-day attestations are placed beside the sidecar in the same write.
Rows move by compare-and-set (commit, then ``os.replace``); a crash leaves rows ahead of the file,
which a refusing rerun restores ONLY while the file still carries the old hash and the row is still
queued and unuploaded. A finished swap (its record on disk, no row at the old hash) is carried
through every later run untouched. ``old_stages_dir`` is the set-aside run's stage dir,
``<stages_root>/_superseded/<project>-<stamp>/<project>/_stages`` (``launch.py`` supersede plan).
See ``docs/social-publishing-queue.md``.
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
    #: Days whose verified attestation beside the plan was kept over an unverified stage copy.
    attestations_kept: List[str] = field(default_factory=list)


def privacy_refusal(stages_dir: Path, date: str) -> Optional[str]:
    """``None`` when the new run's attestation for THIS day verifies every required class; another
    day's never covers this Short and a missing day is a refusal (never merged to a pass)."""
    from gomoveshift.pipeline.day_master_encoder import PRIVACY_ATTESTATION_NAME

    return _attestation_refusal(
        Path(stages_dir) / f"day_{date}" / PRIVACY_ATTESTATION_NAME.format(date=date), date
    )


def _attestation_refusal(path: Path, date: str) -> Optional[str]:
    """``None`` when the attestation file at ``path`` names ``date`` and verifies every class."""
    if not path.is_file():
        return f"no privacy attestation for {date} at {path}"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"privacy attestation unreadable: {exc}"
    if not isinstance(doc, dict) or "date" not in doc:
        return f"the attestation at {path} names no date: it cannot be tied to {date}"
    if doc["date"] != date:
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
    date = short["recording_date"]
    new = _audit_entry(stages_dir, date, rank)
    old_entry = _audit_entry(old_stages_dir, date, rank)
    for which, entry, where in (("new", new, stages_dir), ("old", old_entry, old_stages_dir)):
        if entry is None:
            hint = ""
            if which == "old":
                hint = " (the set-aside stage dir is <stages_root>/_superseded/<project>-<stamp>/<project>/_stages)"
            return (
                f"the {which} run's Shorts audit has no rank {rank}: identity unconfirmed; "
                f"expected {where / f'shorts_ost_audit_{date}.json'}{hint}"
            )
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


def _classify(rows, now: datetime, target: Path, null_is_target: bool = False):
    """``(movable, listed, blocking, refuse, defer, retry_at)``; refuse/defer are reasons (``""`` =
    none) and refuse beats defer. ``null_is_target``: a row with no ``sidecar_path`` is served
    from ``target`` (the plan dispatch reads for it)."""
    movable, listed, blocking, refuse, defer, retries = [], [], [], "", "", []
    for r in rows:
        who = f"{r['platform']} row {r['id']}"
        done = r["status"] in TERMINAL_STATUSES
        if not (
            _same_file(r["sidecar_path"], target) or (null_is_target and not r["sidecar_path"])
        ):
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
    if not refuse and any(r["status"] not in TERMINAL_STATUSES for r in rows):
        sent = [r for r in rows if r["status"] == "published" and r["platform_upload_id"]]
        if sent:  # old bytes already went out on a platform: the Short never goes half new
            refuse = (
                f"{sent[0]['platform']} row {sent[0]['id']} already published the old bytes: "
                "one Short cannot be half old, half new"
            )
    retry_at = min(retries).isoformat() if retries else None
    return movable, listed, blocking, refuse, defer, retry_at


def _ahead(conn, project: str, index: int, old_hash: str, target: Path, disk_hash, now):
    """Rows a crashed run moved but whose file never replaced: ``(restorable, left)``.

    Only while the file still carries the old hash (``disk_hash`` is ``None`` or ``old_hash``) and
    only a row that is still queued, unuploaded and unleased goes back; any other such row is
    returned in ``left`` (listed, never changed)."""
    if disk_hash not in (None, old_hash):
        return [], []  # the swap is on disk: finished, nothing to undo
    out, left = {}, {}
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
        if not (r and _same_file(r["sidecar_path"], target) and r["recovers_delivery_id"] is None):
            continue
        if r["status"] == "queued" and not r["platform_upload_id"] and not _leased(r, now):
            out[r["id"]] = r["plan_hash"]
        else:
            left[r["id"]] = r
    return sorted(out.items()), [left[k] for k in sorted(left)]


def _on_disk(new_sidecar: Path) -> dict:
    """``{short_index: record}`` of the sidecar already at ``new_sidecar`` (``{}`` if none)."""
    try:
        shorts = json.loads(new_sidecar.read_text(encoding="utf-8")).get("shorts") or []
        return {s["short_index"]: s for s in shorts if isinstance(s, dict) and "short_index" in s}
    except (OSError, ValueError, AttributeError, KeyError):
        return {}


def _null_path_is_target(conn, project: str, target: Path, sidecar_dir=None) -> bool:
    """Dispatch serves a NULL ``sidecar_path`` row from the project's newest recorded sidecar, else
    ``<sidecar_dir>/<project>_publish_plan.json`` (its ``--sidecar-dir``); is that ``target``? With
    no recorded sidecar and no ``sidecar_dir`` named, nothing proves it, so it is not."""
    row = conn.execute(
        "SELECT sidecar_path FROM deliveries WHERE project_name = ? AND sidecar_path IS NOT NULL "
        "ORDER BY id DESC LIMIT 1",
        (project,),
    ).fetchone()
    if row is not None:
        return _same_file(row["sidecar_path"], target)
    if sidecar_dir is None:
        return False
    return _same_file(Path(sidecar_dir) / f"{project}_publish_plan.json", target)


def _expected_swap(short: dict, media: dict) -> dict:
    """The record the swap makes from ``short`` (the reviewed one) onto ``media``: only the media,
    the media-derived ``reviewed_plan_hash`` and a ``consent_hash`` that followed it change."""
    yt = short["payloads"]["youtube"]
    new_hash = review_hash(yt["title"], yt["description"], yt["tags"], media["sha256"])
    out = dict(short)
    out["media"] = media
    out["reviewed_plan_hash"] = new_hash
    if short.get("consent_hash") == short["reviewed_plan_hash"]:
        out["consent_hash"] = new_hash
    return out


def _carry_refusal(short: dict, disk: dict) -> Optional[str]:
    """``None`` when ``disk`` (the record already in the new plan file) is exactly the swap of
    ``short`` onto its own media: every field but ``media``, ``reviewed_plan_hash`` and
    ``consent_hash`` equals the old record, the media path is the old one, and the hash is the
    review hash of the old text and that media's sha. Else why it must not be carried."""
    try:
        media = disk["media"]
        if media.get("path") != short["media"]["path"]:
            return "its media path is not the Short's own"
        want = _expected_swap(short, media)
    except (KeyError, TypeError, AttributeError) as exc:
        return f"it is malformed ({exc!r})"
    if disk == want:
        return None
    changed = sorted(k for k in set(disk) | set(want) if disk.get(k) != want.get(k))
    return "it differs from the reviewed record in " + ", ".join(changed)


def _changed_again(disk: dict, fingerprint_fn) -> Optional[str]:
    """Why the file at a carried record's path no longer matches the record's own sha (``None``:
    it matches, or the file is gone, as it is once a finished swap's clip is reclaimed)."""
    try:
        path = Path(disk["media"]["path"])
        if not path.is_file():
            return None
        if fingerprint_fn(path).sha256 == disk["media"]["sha256"]:
            return None
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError) as exc:
        return f"cannot fingerprint the media of the finished swap: {exc!r}"
    return "the media changed again since the swap: a human must look"


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


def _swap_proven(old_shorts: list, on_disk: dict) -> bool:
    """True when every record already in the new plan file is the OLD record or exactly its swap
    (:func:`_carry_refusal`) and no Short of it is foreign to the old plan: the plan differs from
    the reviewed one only by a deterministic swap."""
    old = {s["short_index"]: s for s in old_shorts}
    return set(on_disk) <= set(old) and all(
        disk == old[i] or _carry_refusal(old[i], disk) is None for i, disk in on_disk.items()
    )


def _gate_cd_refusal(
    stages_dir: Path, output_dir: Path, own_plan: str, swap_proven: bool = False
) -> Optional[str]:
    """The enqueue-plan Gate C/D check on the replace run's state (``None`` = passes); an unreadable
    state refuses, as ``enqueue-plan`` does.

    The recorded FULL hash is accepted as ``enqueue-plan`` accepts it (own plan included), and with
    ``own_plan`` left out (a confirmation taken before the swap's sidecar existed: a replace run
    emits none). Only here, and only when ``swap_proven`` (every record in the new plan is the old
    one or its proven swap), is the recorded swap-projected hash (media, reviewed_plan_hash and
    consent_hash left out) accepted too, so a confirmation taken after an earlier run wrote the
    sidecar still holds. Any other file in the plan dir still counts whole."""
    from gomoveshift.cli.handlers import enqueue_plan

    try:
        state = enqueue_plan._load_state(stages_dir / "pipeline_state.json")
        try:  # a confirmation taken before the swap covers no own plan, one taken after covers it
            enqueue_plan.assert_gate_cd_approved(state, output_dir, skip_plans=(own_plan,))
        except enqueue_plan.GateNotApprovedError:
            enqueue_plan.assert_gate_cd_approved(state, output_dir, swap_projection=swap_proven)
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
    sidecar_dir: Optional[Path] = None,
) -> RebuildResult:
    """Rebuild ``new_sidecar`` from ``old_sidecar`` onto the media now at each Short's path.

    ``stages_dir`` is the replace run's stage dir, ``old_stages_dir`` the set-aside run's;
    ``sidecar_dir`` is dispatch's ``--sidecar-dir`` (where it reads a row with no ``sidecar_path``).
    Never raises for a refused or deferred Short (it is in ``RebuildResult`` plus an incident);
    raises :class:`LeaseRaceError` (nothing written) when a row changed during the write."""
    old_sidecar, new_sidecar = Path(old_sidecar), Path(new_sidecar)
    stages_dir, old_stages_dir = Path(stages_dir), Path(old_stages_dir)
    raw = json.loads(old_sidecar.read_text(encoding="utf-8"))
    for record in raw["shorts"]:
        # The plan parser needs the key, so a Short whose old record lacks it cannot go back
        # verbatim: it goes back with ``recording_date: null`` (which parses, keeps its short_index,
        # and is what the Short is refused for below), so dispatch holds its stale bytes.
        record.setdefault("recording_date", None)
    target = new_sidecar.resolve()
    result = RebuildResult()
    gate: Dict[Any, Optional[str]] = {}
    settled: set = set()
    settled_pos: set = set()  # positions judged by the copy beside the plan, not a stage copy
    on_disk = _on_disk(new_sidecar)
    gate_cd = _gate_cd_refusal(
        stages_dir, new_sidecar.parent, new_sidecar.name, _swap_proven(raw["shorts"], on_disk)
    )
    null_ok = _null_path_is_target(conn, project, target, sidecar_dir)
    moves: List[tuple] = []  # (delivery_id, short_index, from_hash, to_hash, action)
    staged: List[tuple] = []  # (pos, new_short, Rebuilt, moves), applied after placement
    for pos, short in enumerate(raw["shorts"]):
        index, media = short["short_index"], short["media"]
        yt = short["payloads"]["youtube"]
        old_hash = short["reviewed_plan_hash"]
        old_rows = _rows(conn, project, index, old_hash)
        movable, listed, blocking, refuse, defer, retry_at = _classify(
            old_rows, now, target, null_ok
        )
        result.untouched += [
            Untouched(r["id"], index, r["platform"], r["status"], why)
            for r, why in listed + blocking
        ]
        disk = on_disk.get(index)
        disk_hash = disk.get("reviewed_plan_hash") if disk else None
        carried, carry_bad, changed = False, "", None
        if disk_hash not in (None, old_hash):
            bad = _carry_refusal(short, disk)
            if bad is None:  # a finished swap: judged by ALL the Short's rows, never undone
                carried = True
                raw["shorts"][pos] = disk
                if not (movable or refuse or defer):
                    changed = _changed_again(disk, fingerprint_fn)
                    if changed is None:
                        settled.add(index)  # nothing at the old hash can still move or publish
                        settled_pos.add(pos)
                        continue
            else:
                carry_bad = f"the plan file's record for this Short is not the swap ({bad})"
                disk_hash = None  # it is overwritten with the reviewed record: rows follow it
        elif (
            old_rows
            and not refuse
            and short.get("recording_date")  # a Short refused for its missing day is not finished
            and all(r["status"] in TERMINAL_STATUSES for r in old_rows)
            and not _ahead(conn, project, index, old_hash, target, disk_hash, now)[0]
        ):  # (a row a crashed run moved ahead of the file is not settled: the swap completes)
            settled.add(index)  # nothing of this Short can still publish: no gate, no incident
            settled_pos.add(pos)
            continue
        ids = tuple(r["id"] for r in movable)
        reason = carry_bad or changed or gate_cd or refuse
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
            day = short.get("recording_date")
            if not reason and carried and new_fp.sha256 != disk["media"]["sha256"]:
                reason = "the media changed again since the swap: a human must look"
            if not reason and not day:
                reason = "the Short has no recording_date: its privacy day is unknown"
            if not reason:
                reason = _identity_refusal(short, new_fp, stages_dir, old_stages_dir) or ""
            if not reason:
                derived = review_hash(yt["title"], yt["description"], yt["tags"], media["sha256"])
                if old_hash != derived:
                    reason = "reviewed_plan_hash is not media-derived: a human re-review is needed"
            if not reason and ("privacy", day) not in gate:
                gate["privacy", day] = privacy_fn(stages_dir, day)
            if not reason and gate["privacy", day]:
                reason = f"privacy gate: {gate['privacy', day]}"
            if not reason and "mask" not in gate:
                gate["mask"] = mask_fn(stages_dir, drive=project, incidents_dir=incidents_dir)
            if not reason and gate["mask"]:
                reason = f"mask-frame gate: {gate['mask']}"
        if reason or defer:
            ahead, left = _ahead(conn, project, index, old_hash, target, disk_hash, now)
            moves += [
                (did, index, theirs, old_hash, "replace_rebuild_restore") for did, theirs in ahead
            ]
            ids += tuple(did for did, _ in ahead)
            for r in left:
                result.untouched.append(
                    Untouched(r["id"], index, r["platform"], r["status"], "moved ahead: left as is")
                )
            note = "".join(
                f"; row {r['id']} moved ahead of the plan file but is {r['status']}: left as is"
                for r in left
            )
            reason, defer = (reason + note if reason else ""), (defer + note if defer else "")
            if reason:
                result.refused.append(Refused(index, reason, ids))
            else:
                result.deferred.append(Deferred(index, defer, retry_at, ids))
            continue
        new_hash = review_hash(yt["title"], yt["description"], yt["tags"], new_fp.sha256)
        fields = ("path", "size_bytes", "duration_seconds", "width", "height", "codec", "sha256")
        swapped = _expected_swap(short, {k: getattr(new_fp, k) for k in fields})
        if carried:
            swapped = dict(disk)  # the proven swap already on disk, byte for byte
        staged.append(
            (
                pos,
                swapped,
                Rebuilt(index, old_hash, new_hash, ids),
                [(r["id"], index, old_hash, new_hash, "replace_rebuild") for r in movable],
            )
        )
    if staged:
        live = {
            s.get("recording_date") for i, s in enumerate(raw["shorts"]) if i not in settled_pos
        }
        done = {raw["shorts"][i].get("recording_date") for i in settled_pos} - live
        for day in sorted(d for d in live if d):
            if ("privacy", day) not in gate:
                gate["privacy", day] = privacy_fn(stages_dir, day)
        for day in sorted(d for d in done if d):
            if ("privacy", day) not in gate:  # a finished day: the verified copy beside the plan
                gate["privacy", day] = _finished_day_refusal(
                    new_sidecar.parent, stages_dir, day, privacy_fn
                )
        held = [
            f"{k[1]}: {why}"
            for k, why in sorted(gate.items(), key=str)
            if k[0] == "privacy" and why
        ]
        if held:  # dispatch merges a run's days worst-state-wins: another day holds every upload
            why = (
                "another day's privacy attestation holds the whole run in dispatch ("
                + "; ".join(held)
                + ")"
            )
            result.deferred += [
                Deferred(st[2].short_index, why, None, st[2].delivery_ids) for st in staged
            ]
            staged = []
    if staged:
        try:
            result.attestations_kept = _place_attestations(new_sidecar, stages_dir, raw)
            for day in result.attestations_kept:
                logger.warning(
                    "replace-rebuild: %s kept the verified attestation for %s beside the plan; "
                    "the stage copy is not verified",
                    project,
                    day,
                )
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
    for index in sorted({r.short_index for r in result.rebuilt} | settled):
        if index not in bad:
            _resolve_incident(project, index, now, incidents_dir)
    return result


def _finished_day_refusal(plan_dir: Path, stages_dir: Path, day: str, privacy_fn) -> Optional[str]:
    """A day whose Shorts are all finished is judged by the verified copy beside the plan (what
    dispatch reads); the stage copy, which a reclaim may have removed, only has to agree."""
    from gomoveshift.pipeline.day_master_encoder import PRIVACY_ATTESTATION_NAME

    why = privacy_fn(stages_dir, day)
    if why is None:
        return None
    beside = _attestation_refusal(plan_dir / PRIVACY_ATTESTATION_NAME.format(date=day), day)
    return None if beside is None else why


def _place_attestations(new_sidecar: Path, stages_dir: Path, raw: dict) -> List[str]:
    """Put the per-day attestations beside the sidecar; returns the days whose VERIFIED copy beside
    the plan was kept because the stage copy is present but not verified (an ``inconclusive`` or
    unreadable one would otherwise hold every upload of the run). A copy beside the plan whose
    stage copy is gone is kept too (a reclaim removed it)."""
    from gomoveshift.pipeline.day_master_encoder import PRIVACY_ATTESTATION_NAME
    from gomoveshift.pipeline.drive_phases import phase_d

    dates = sorted({s["recording_date"] for s in raw["shorts"] if s.get("recording_date")})
    kept, notes = {}, []
    for date in dates:
        name = PRIVACY_ATTESTATION_NAME.format(date=date)
        beside, stage = new_sidecar.parent / name, stages_dir / f"day_{date}" / name
        if not beside.is_file():
            continue
        if not stage.is_file():
            kept[name] = beside.read_bytes()
        elif _attestation_refusal(beside, date) is None and _attestation_refusal(stage, date):
            kept[name] = beside.read_bytes()
            notes.append(date)
    phase_d._sync_privacy_attestations(
        output_base=new_sidecar.parent, stages_base=stages_dir, dates=dates
    )
    for name, data in kept.items():
        (new_sidecar.parent / name).write_bytes(data)
    return notes


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
