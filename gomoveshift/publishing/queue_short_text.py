"""Re-write queued YouTube Shorts with the CURRENT text generator.

A Short queued days ago keeps the title, description and tags the generator made then,
so a generator fix (the no-repeated-place title rule, unit C2) never reaches it.
``sync_short_text`` closes that gap for every ``queued`` YouTube row whose slot is more
than :data:`SYNC_LEAD_TIME` away: it regenerates the Short's text through the generator
path Phase D uses (``plan_bracket_sync._generate_lf``, over the parent LF) and writes
title, description and tags back to the publish-plan sidecar.

Kept as it was: the title stamp (``"<date>, Nth stretch"``), recording date, schedule,
media, and ``experiment_arms`` exactly as stored (the sync never picks, swaps or excludes
an arm: arm exclusion waits for a generator that consumes the Shorts arms).

A Short is refused whole (nothing written) when any delivery row of it, on any platform,
is neither ``queued`` nor terminal, holds a live lease, has no ``scheduled_at``, or has a
slot inside the lead time, or when a recovery row exists. A Short with a hand title
(``<plan dir>/social/titles.yaml``, as Phase D reads it) is reported ``unchanged`` /
``hand_title``. The no-repeated-place chain is seeded from the STORED text of the Short
before each rewritten one, and the generator's date is pinned to the LF's publish date,
so a rerun with the same generator changes nothing.

Each rewritten Short must pass ``short_review_refusal``, ``models.validate_payloads`` and
``anti_spam_gate.check_batch`` (the gates enqueue runs, with the queue's recent text);
a refused Short is left unchanged and reported with the reason.

Plan hash (decision D4): a changed Short gets ``reviewed_plan_hash`` recomputed, every
non-terminal, unleased row whose ``plan_hash`` was the old one moves to the new one, and
``response_log`` keeps the old hash. Order: the audit row and queue update commit first,
then the sidecar is replaced, so a failure never leaves them silently apart; every run
also moves a queued row still on a stale hash to the sidecar's when the sidecar already
holds the synced text. No network, no platform call; a dry run writes no file at all.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .plan_bracket_sync import _generate_lf, _ordinal
from .plan_io import load_publish_plan, plan_from_raw
from .queue import TERMINAL_STATUSES

#: A row whose slot is closer than this is left alone: the platform may already be
#: preparing the upload, and a late text change would race the dispatcher.
SYNC_LEAD_TIME = timedelta(hours=24)

_YT_SQL = (
    "SELECT id, project_name, short_index, sidecar_path, scheduled_at, plan_hash "
    "FROM deliveries WHERE platform = 'youtube' AND status = 'queued' AND shadow = 0 "
    "AND sidecar_path IS NOT NULL AND (lease_owner IS NULL OR lease_expires_at <= ?) "
    "ORDER BY id"
)


class SidecarChangedError(ValueError):
    """The sidecar was edited between planning and writing."""


@dataclass(frozen=True)
class ShortTextChange:
    delivery_id: int
    project_name: str
    short_index: int
    # "changed" | "reconciled" | "unchanged" | "inside_lead_time" | "refused"
    status: str
    old_title: str
    new_title: str
    reason: str = ""
    fields: Tuple[str, ...] = ()
    old_hash: str = ""
    new_hash: str = ""
    written: bool = False
    #: queue ``plan_hash`` values that move to ``new_hash`` (``changed``: the old one)
    old_hashes: Tuple[str, ...] = ()
    #: the row's own ``sidecar_path`` text, which the queue update matches verbatim
    sidecar_path: str = ""


def _slot(raw) -> Optional[datetime]:
    try:
        at = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


def _leased(row, now: datetime) -> bool:
    return row["lease_owner"] is not None and (row["lease_expires_at"] or "") > now.isoformat()


def _sibling_refusal(siblings, now: datetime) -> str:
    """Why this Short's delivery rows (any platform) forbid a rewrite ("" = none)."""
    for r in siblings:
        who = f"{r['platform']} row {r['id']}"
        if r["recovers_delivery_id"] is not None:
            return f"{who} is a recovery row"
        if r["status"] in TERMINAL_STATUSES:
            continue
        if r["status"] != "queued":
            return f"{who} is {r['status']}"
        if _leased(r, now):
            return f"{who} holds a live lease"
        at = _slot(r["scheduled_at"]) if r["scheduled_at"] else None
        if at is None:
            return f"{who} has no usable scheduled_at"
        if at - now < SYNC_LEAD_TIME:
            return f"{who} slot {at.isoformat()} is inside the lead time"
    return ""


def _lf_inputs(plan_dir: Path, project: str, lf_index: int) -> Tuple[Optional[dict], str, str]:
    """``(valid bracket record | None, LF location, why not)`` from the LF sidecar."""
    from gomoveshift.pipeline.bracket_route import bracket_record_valid

    path = plan_dir / f"{project}_lf{lf_index:02d}_youtube.json"
    try:
        side = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, "", f"LF sidecar {path.name} unreadable: {exc}"
    location = str(side.get("location") or "")
    if not location:
        return None, "", f"LF sidecar {path.name} has no location"
    record = (side.get("drive_text_inputs") or {}).get("bracket_route")
    return (record if bracket_record_valid(record) else None), location, ""


def _youtube_text(short: dict, post) -> dict:
    """The generator's YouTube title, description and tags for ``short``."""
    from .plan_builder import build_short_payloads

    tiktok = short["payloads"].get("tiktok") or {}
    built = build_short_payloads(
        post,
        recording_date=short["recording_date"],
        parent_id=short["parent_lf_video_id"],
        tiktok_privacy_level=tiktok.get("privacy_level") or "SELF_ONLY",
    ).youtube
    return {"title": built.title, "description": built.description, "tags": list(built.tags)}


def _rewrite(short: dict, text: dict) -> Tuple[dict, Tuple[str, ...]]:
    """A copy of ``short`` carrying ``text`` with the new plan hash, and the names of the
    fields that changed (empty = nothing to write). ``experiment_arms`` is untouched."""
    from .asset_plan import review_hash

    new = copy.deepcopy(short)
    fields = []
    for name, value in text.items():
        if new["payloads"]["youtube"].get(name) != value:
            new["payloads"]["youtube"][name] = value
            fields.append(f"youtube.{name}")
    if fields:
        yt = new["payloads"]["youtube"]
        new["reviewed_plan_hash"] = review_hash(
            yt["title"], yt["description"], yt["tags"], new["media"]["sha256"]
        )
    return new, tuple(fields)


def _stored_places(short: dict, location: str) -> Dict[str, str]:
    """``{"title"|"description": body}``: the place the Short's STORED title / description
    names, found among its measured places and the LF location (omitted = none found)."""
    from gomoveshift.pipeline.social_generator import _norm, measured_places

    found = measured_places(
        short.get("sampled_regions") or [],
        short.get("sampled_roads") or [],
        short.get("route_record"),
    )
    cands = [b for _, b in found] + [location, location.split(",")[0]]
    yt = short["payloads"]["youtube"]
    out: Dict[str, str] = {}
    for key in ("title", "description"):
        text = _norm(yt[key])
        hit = next((c for c in cands if _norm(c) and _norm(c) in text), None)
        if hit:
            out[key] = hit
    return out


def _generate(
    sidecar: Path, shorts: List[dict], active: Set[int], spooled: dict
) -> Tuple[Dict[int, object], Dict[int, str]]:
    """``({position: post}, {position: why refused})`` for the ``active`` positions: one
    generator pass per LF over the whole LF (so each Short keeps its position), with only
    the active Shorts composing text and every other Short's STORED places seeding the
    no-repeated-place chain."""
    posts: Dict[int, object] = {}
    refused: Dict[int, str] = {}
    lfs: Dict[Tuple[str, int], List[int]] = {}
    for pos, s in enumerate(shorts):
        lfs.setdefault((s["project_name"], int(s["lf_index"])), []).append(pos)
    for (project, lf_index), positions in lfs.items():
        mine = [p for p in positions if p in active]
        if not mine:
            continue
        positions.sort(key=lambda p: shorts[p]["short_index"])
        record, location, why = _lf_inputs(sidecar.parent, project, lf_index)
        for pos in mine:
            s = shorts[pos]
            skip = why
            if not skip and not s.get("sampled_regions"):
                skip = "no clip place to compose a title from"
            if not skip and _ordinal(s) is None:
                skip = "old title has no verifiable 'Nth stretch' stamp"
            if skip:
                refused[pos] = skip
        live = [p for p in mine if p not in refused]
        if not live:
            continue
        seeds = {
            i: _stored_places(shorts[positions[i - 2]], location) if i > 1 else {}
            for i in range(1, len(positions) + 1)
            if i == 1 or positions[i - 2] not in live
        }
        dates = [str(shorts[p].get("parent_lf_publish_at") or "")[:10] for p in positions]
        try:
            today = date.fromisoformat(min(d for d in dates if d))
        except ValueError:
            today = date.fromisoformat(shorts[positions[0]]["recording_date"])
        view = [{**shorts[p], "_skip": p not in live} for p in positions]
        try:
            generated = _generate_lf(
                view,
                record,
                location,
                lf_index,
                spooled_texts=spooled,
                experiment_today=today,
                previous_places_at=seeds,
            )
        except Exception as exc:  # the generator's own refusal is the reason
            for pos in live:
                refused[pos] = f"generator: {exc}"
            continue
        posts.update({p: g for p, g in zip(positions, generated) if p in live})
    return posts, refused


def _gate_refusals(
    plan, positions: List[int], spooled: dict, per_location_cap: Optional[int]
) -> Dict[int, str]:
    """Why each rewritten position fails the gates enqueue runs: the Short review,
    ``validate_payloads`` and ``check_batch`` (over the whole trial plan)."""
    from .anti_spam_gate import check_batch
    from .models import PayloadLimitError, validate_payloads
    from .youtube_adapter import short_review_refusal

    out: Dict[int, str] = {}
    batch = check_batch(list(plan.shorts), per_location_cap=per_location_cap, recent_texts=spooled)
    for pos in positions:
        record = plan.shorts[pos]
        refusal = short_review_refusal(record)
        if refusal:
            out[pos] = f"still refused by the Short review: {refusal}"
            continue
        try:
            validate_payloads(record.payloads)
        except PayloadLimitError as exc:
            out[pos] = f"payload limit: {exc}"
            continue
        hit = [
            v
            for v in batch
            if v.index == pos and (v.platform in ("youtube", None) or v.rule == "caption-collision")
        ]
        if hit:
            out[pos] = "anti-spam gate: " + "; ".join(f"{v.rule}: {v.detail}" for v in hit)
    return out


def _sidecar_changes(
    conn: sqlite3.Connection,
    sidecar: Path,
    rows: List[sqlite3.Row],
    now: datetime,
    per_location_cap: Optional[int],
) -> Tuple[Dict[int, ShortTextChange], Dict[int, dict], str]:
    """Per YouTube row id: its verdict; the rewritten Short for each ``changed`` one
    (keyed by position in the plan); the sidecar's sha256. Nothing is written here."""
    from gomoveshift.pipeline.social_generator import load_hand_titles

    body = sidecar.read_bytes()
    raw = json.loads(body)
    load_publish_plan(sidecar)  # the input must already be a valid plan
    sha = hashlib.sha256(body).hexdigest()
    shorts: List[dict] = raw["shorts"]
    by_key = {(s["project_name"], int(s["short_index"])): pos for pos, s in enumerate(shorts)}
    hand = load_hand_titles(sidecar.parent / "social" / "titles.yaml")
    verdict: Dict[int, ShortTextChange] = {}

    def settle(row, pos, status, reason="", **kw):
        title = shorts[pos]["payloads"]["youtube"]["title"] if pos is not None else ""
        verdict[row["id"]] = ShortTextChange(
            row["id"],
            row["project_name"],
            row["short_index"],
            status,
            title,
            kw.pop("new_title", title),
            reason,
            sidecar_path=row["sidecar_path"],
            **kw,
        )

    targets: Dict[int, sqlite3.Row] = {}
    siblings: Dict[int, list] = {}
    for row in rows:
        pos = by_key.get((row["project_name"], int(row["short_index"])))
        if pos is None:
            settle(row, None, "refused", "no such Short in the plan")
            continue
        at = _slot(row["scheduled_at"]) if row["scheduled_at"] else None
        sib = conn.execute(
            "SELECT * FROM deliveries WHERE project_name = ? AND short_index = ? "
            "AND sidecar_path = ?",
            (row["project_name"], row["short_index"], row["sidecar_path"]),
        ).fetchall()
        why = _sibling_refusal(sib, now)
        if at is not None and at - now < SYNC_LEAD_TIME:
            settle(row, pos, "inside_lead_time", f"slot {at.isoformat()} is inside the lead time")
        elif why:
            settle(row, pos, "refused", why)
        elif int(shorts[pos]["short_index"]) in hand:
            settle(row, pos, "unchanged", "hand_title")
        else:
            targets[pos] = row
            siblings[pos] = sib
    if not targets:
        return verdict, {}, sha

    projects = {s["project_name"] for s in shorts}
    from .anti_spam_gate import recent_platform_texts

    spooled = recent_platform_texts(conn, exclude_projects=projects)
    active: Set[int] = set(targets)
    rewritten: Dict[int, dict] = {}
    fields_of: Dict[int, Tuple[str, ...]] = {}
    unchanged: Set[int] = set()
    while True:  # a refusal changes the stored text that seeds its successor: redo
        posts, refused = _generate(sidecar, shorts, active, spooled)
        for pos, why in refused.items():
            settle(targets[pos], pos, "refused", why)
        active -= set(refused)
        rewritten, fields_of, unchanged = {}, {}, set()
        for pos, post in posts.items():
            new, fields = _rewrite(shorts[pos], _youtube_text(shorts[pos], post))
            if fields:
                rewritten[pos], fields_of[pos] = new, fields
            else:
                unchanged.add(pos)
        again = set(refused)
        if rewritten:
            trial = copy.deepcopy(raw)
            for pos, new in rewritten.items():
                trial["shorts"][pos] = new
            plan = plan_from_raw(trial, sidecar)
            bad = _gate_refusals(plan, list(rewritten), spooled, per_location_cap)
            for pos, why in bad.items():
                settle(targets[pos], pos, "refused", why)
                active.discard(pos)
            again |= set(bad)
        if not again:
            break
    for pos in unchanged:
        sidecar_hash = shorts[pos]["reviewed_plan_hash"]
        stale = sorted(
            {
                r["plan_hash"]
                for r in siblings[pos]
                if r["status"] not in TERMINAL_STATUSES and r["plan_hash"] != sidecar_hash
            }
        )
        if stale:
            settle(
                targets[pos],
                pos,
                "reconciled",
                "queue hash was behind the sidecar's synced text",
                old_hash=stale[0],
                new_hash=sidecar_hash,
                old_hashes=tuple(stale),
            )
        else:
            settle(targets[pos], pos, "unchanged", "text already matches the generator")
    for pos, new in list(rewritten.items()):
        old = shorts[pos]["reviewed_plan_hash"]
        new_hash = new["reviewed_plan_hash"]
        odd = [
            r["plan_hash"]
            for r in siblings[pos]
            if r["status"] not in TERMINAL_STATUSES and r["plan_hash"] not in (old, new_hash)
        ]
        if odd:
            del rewritten[pos]
            settle(
                targets[pos], pos, "refused", f"a queue row's plan hash {odd[0][:12]} is unknown"
            )
            continue
        settle(
            targets[pos],
            pos,
            "changed",
            fields=fields_of[pos],
            new_title=new["payloads"]["youtube"]["title"],
            old_hash=old,
            new_hash=new_hash,
            old_hashes=(old,),
        )
    return verdict, rewritten, sha


def _apply_sidecar(
    conn: sqlite3.Connection,
    sidecar: Path,
    verdict: Dict[int, ShortTextChange],
    rewritten: Dict[int, dict],
    sha: str,
    now: datetime,
) -> None:
    """Commit the audit rows and queue hash moves, THEN replace the sidecar, so a failure
    of either never leaves them silently apart (a rerun converges: see the module doc)."""
    if hashlib.sha256(sidecar.read_bytes()).hexdigest() != sha:
        raise SidecarChangedError(f"{sidecar.name} changed while the sync ran")
    tmp = sidecar.with_name(sidecar.name + ".sync-short-tmp")
    if rewritten:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        for pos, new in rewritten.items():
            raw["shorts"][pos] = new
        tmp.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    terminal = ", ".join("?" for _ in TERMINAL_STATUSES)
    try:
        with conn:
            for change in verdict.values():
                if change.status not in ("changed", "reconciled"):
                    continue
                moved = 0
                for old in change.old_hashes:
                    moved += conn.execute(
                        "UPDATE deliveries SET plan_hash = ? "  # noqa: S608
                        "WHERE project_name = ? AND short_index = ? AND sidecar_path = ? "
                        f"AND plan_hash = ? AND status NOT IN ({terminal}) "
                        "AND (lease_owner IS NULL OR lease_expires_at <= ?)",
                        (
                            change.new_hash,
                            change.project_name,
                            change.short_index,
                            change.sidecar_path,
                            old,
                            *TERMINAL_STATUSES,
                            now.isoformat(),
                        ),
                    ).rowcount
                conn.execute(
                    "INSERT INTO response_log (at, platform, state, cap, action, mode, reason) "
                    "VALUES (?, 'youtube', 'plan_sync', 0, 'sync_short_text', 'live', ?)",
                    (
                        now.isoformat(),
                        json.dumps(
                            {
                                "delivery_id": change.delivery_id,
                                "sidecar": str(sidecar),
                                "project": change.project_name,
                                "short_index": change.short_index,
                                "old_plan_hash": change.old_hash,
                                "new_plan_hash": change.new_hash,
                                "rows_moved": moved,
                                "old_title": change.old_title,
                                "new_title": change.new_title,
                                "fields": list(change.fields),
                                "reconcile": change.status == "reconciled",
                            },
                            sort_keys=True,
                        ),
                    ),
                )
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    if rewritten:
        os.replace(tmp, sidecar)


def sync_short_text(
    conn: sqlite3.Connection,
    *,
    apply: bool,
    now: Optional[datetime] = None,
    per_location_cap: Optional[int] = None,
) -> List[ShortTextChange]:
    """Re-write every eligible queued YouTube Short's text (see the module docstring);
    one :class:`ShortTextChange` per queued, unleased YouTube row, in row order.
    ``per_location_cap`` is the brand policy's, as enqueue passes it to ``check_batch``."""
    now = now or datetime.now(timezone.utc)
    rows = conn.execute(_YT_SQL, (now.isoformat(),)).fetchall()
    by_sidecar: Dict[str, List[sqlite3.Row]] = {}
    for row in rows:
        by_sidecar.setdefault(row["sidecar_path"], []).append(row)
    verdicts: Dict[int, ShortTextChange] = {}

    def refuse_all(group, why):
        for row in group:
            verdicts[row["id"]] = ShortTextChange(
                row["id"],
                row["project_name"],
                row["short_index"],
                "refused",
                "",
                "",
                why,
                sidecar_path=row["sidecar_path"],
            )

    for path, group in by_sidecar.items():
        sidecar = Path(path)
        try:
            verdict, rewritten, sha = _sidecar_changes(conn, sidecar, group, now, per_location_cap)
        except (OSError, ValueError, LookupError) as exc:
            refuse_all(group, f"sidecar unusable: {exc}")
            continue
        if apply and any(v.status in ("changed", "reconciled") for v in verdict.values()):
            try:
                _apply_sidecar(conn, sidecar, verdict, rewritten, sha, now)
            except (OSError, ValueError, sqlite3.Error) as exc:
                refuse_all(group, f"sidecar unusable: write failed: {exc}")
                continue
            verdict = {
                k: replace(v, written=True) if v.status in ("changed", "reconciled") else v
                for k, v in verdict.items()
            }
        verdicts.update(verdict)
    return [verdicts[row["id"]] for row in rows]
