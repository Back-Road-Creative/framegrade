"""Re-write queued YouTube Shorts with the CURRENT text generator and arm decisions.

A Short queued days ago keeps the title, description and tags the generator made then,
so a generator fix (the no-repeated-place title rule, unit C2) or an experiment-arm
decision (``publishing/experiments.py``, unit C4) never reaches it. ``sync_short_text``
closes that gap for every ``queued`` YouTube row whose slot is more than
:data:`SYNC_LEAD_TIME` away: it regenerates the Short's text through the generator path
Phase D uses (``plan_bracket_sync._generate_lf``, over the parent LF) and writes title,
description and tags back to the publish-plan sidecar.

Kept as it was: the title stamp (``"<date>, Nth stretch"`` -- the generator composes it
from the same date and ordinal), recording date, schedule, media, and the experiment arm
tags. One exception, forced by the decision: a tagged arm whose latest decision is
``demote`` or ``pause`` is never used for the new text; the Short takes the first other
arm of that factor that is not demoted or paused (its tag moves with it, so outcomes are
credited to the arm that wrote the text), and with none left the Short is left unchanged.
No decision for an arm = the Short keeps it.

Each rewritten Short must pass ``short_review_refusal`` (the review Q1 runs at enqueue
and ``publish`` runs at dispatch); a refused Short is left unchanged and reported with
the reason. Its Facebook / Instagram rows share the one record, so a refusal holds all
of that Short's rows, and the write is all-or-nothing per Short.

Plan hash (decision D4): a changed Short gets ``reviewed_plan_hash`` recomputed
(``asset_plan.review_hash`` of the new text and media, what the plan builder stores when
given none), every non-terminal queue row of the Short whose ``plan_hash`` was the old one
moves to the new one, and ``response_log`` keeps the old hash beside the change
(``sync_short_text``). Atomic and idempotent: the sidecar is replaced via a temp file, the
queue update commits only after it, and a second run with an unchanged generator changes
nothing. No network, no platform call; dry run unless ``apply``.
"""

from __future__ import annotations

import copy
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from .plan_bracket_sync import _generate_lf, _ordinal, _parsed
from .plan_io import load_publish_plan
from .queue import TERMINAL_STATUSES

#: A row whose slot is closer than this is left alone: the platform may already be
#: preparing the upload, and a late text change would race the dispatcher.
SYNC_LEAD_TIME = timedelta(hours=24)

#: Decisions that bar an arm from the re-written text.
_BARRED = frozenset({"demote", "pause"})

#: ``{factor: {arm: decision}}`` -- the latest decision of every arm (empty = none).
ArmDecisions = Mapping[str, Mapping[str, str]]

_YT_SQL = (
    "SELECT id, project_name, short_index, sidecar_path, scheduled_at FROM deliveries "
    "WHERE platform = 'youtube' AND status = 'queued' AND shadow = 0 "
    "AND sidecar_path IS NOT NULL ORDER BY id"
)


@dataclass(frozen=True)
class ShortTextChange:
    delivery_id: int
    project_name: str
    short_index: int
    status: str  # "changed" | "unchanged" | "inside_lead_time" | "refused"
    old_title: str
    new_title: str
    reason: str = ""
    fields: Tuple[str, ...] = ()
    old_hash: str = ""
    new_hash: str = ""
    written: bool = False


def arm_decision_map(decisions_by_factor: Mapping[str, list]) -> Dict[str, Dict[str, str]]:
    """``{factor: {arm: decision}}`` from ``decide_shorts_arms`` results per factor."""
    return {
        factor: {d.arm: d.decision for d in decisions if d.arm != "-"}
        for factor, decisions in decisions_by_factor.items()
    }


def _effective_arms(arms: Mapping[str, str], decisions: ArmDecisions) -> Tuple[dict, str]:
    """``(arms to write the text under, "" | why none)``: a barred arm is swapped for the
    first (sorted) other arm of its factor that is not barred."""
    out = dict(arms)
    for factor, arm in arms.items():
        by_arm = decisions.get(factor) or {}
        if by_arm.get(arm) not in _BARRED:
            continue
        free = sorted(a for a, d in by_arm.items() if d not in _BARRED)
        if not free:
            return out, f"arm {factor}={arm} is {by_arm[arm]} and no other arm is open"
        out[factor] = free[0]
    return out, ""


def _scheduled(short: dict, row) -> Optional[datetime]:
    raw = row["scheduled_at"] or (short.get("schedule") or {}).get("scheduled_at_utc")
    try:
        at = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


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
    return {
        "title": built.title,
        "description": built.description,
        "tags": list(built.tags),
    }


def _rewrite(short: dict, text: dict, arms: dict) -> Tuple[dict, Tuple[str, ...]]:
    """A copy of ``short`` carrying ``text`` (and ``arms``), with the new plan hash, and
    the names of the fields that changed (empty = nothing to write)."""
    from .asset_plan import review_hash

    new = copy.deepcopy(short)
    fields: List[str] = []
    for name, value in text.items():
        if new["payloads"]["youtube"].get(name) != value:
            new["payloads"]["youtube"][name] = value
            fields.append(f"youtube.{name}")
    if new.get("experiment_arms") != arms:
        new["experiment_arms"] = arms
        fields.append("experiment_arms")
    if fields:
        yt = new["payloads"]["youtube"]
        new["reviewed_plan_hash"] = review_hash(
            yt["title"], yt["description"], yt["tags"], new["media"]["sha256"]
        )
    return new, tuple(fields)


def _sidecar_changes(
    sidecar: Path,
    rows: List[sqlite3.Row],
    decisions: ArmDecisions,
    now: datetime,
) -> Tuple[Dict[int, ShortTextChange], Dict[int, dict]]:
    """Per YouTube row id: its verdict, and the rewritten Short for each ``changed`` one
    (keyed by position in the plan). Nothing is written here."""
    from .youtube_adapter import short_review_refusal

    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    load_publish_plan(sidecar)  # the input must already be a valid plan
    shorts: List[dict] = raw["shorts"]
    by_key = {(s["project_name"], int(s["short_index"])): pos for pos, s in enumerate(shorts)}
    verdict: Dict[int, ShortTextChange] = {}
    rewritten: Dict[int, dict] = {}  # position -> new short
    row_of: Dict[int, sqlite3.Row] = {}
    arms_of: Dict[int, dict] = {}
    fields_of: Dict[int, Tuple[str, ...]] = {}

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
            **kw,
        )

    targets: Dict[int, sqlite3.Row] = {}
    for row in rows:
        pos = by_key.get((row["project_name"], int(row["short_index"])))
        if pos is None:
            settle(row, None, "refused", "no such Short in the plan")
            continue
        at = _scheduled(shorts[pos], row)
        if at is None:
            settle(row, pos, "refused", "unreadable slot")
        elif at - now < SYNC_LEAD_TIME:
            settle(row, pos, "inside_lead_time", f"slot {at.isoformat()} is inside the lead time")
        else:
            targets[pos] = row
    if not targets:
        return verdict, rewritten

    lfs: Dict[Tuple[str, int], List[int]] = {}
    for pos, s in enumerate(shorts):
        lfs.setdefault((s["project_name"], int(s["lf_index"])), []).append(pos)
    posts: Dict[int, object] = {}
    for (project, lf_index), positions in lfs.items():
        mine = [p for p in positions if p in targets]
        if not mine:
            continue
        positions.sort(key=lambda p: shorts[p]["short_index"])
        record, location, why = _lf_inputs(sidecar.parent, project, lf_index)
        pins: Dict[int, dict] = {}
        view = []
        for pos in positions:
            s = shorts[pos]
            skip = ""
            if pos in targets:
                pins[s["short_index"]], skip = _effective_arms(
                    s.get("experiment_arms") or {}, decisions
                )
                if not skip and not s.get("sampled_regions"):
                    skip = "no clip place to compose a title from"
                if not skip and _ordinal(s) is None:
                    skip = "old title has no verifiable 'Nth stretch' stamp"
                skip = skip or why
                if skip:
                    settle(targets[pos], pos, "refused", skip)
            view.append({**s, "_skip": bool(skip) or not s.get("sampled_regions")})
        if all(targets[p]["id"] in verdict for p in mine):
            continue
        try:
            generated = _generate_lf(view, record, location, lf_index, pinned_arms=pins)
        except Exception as exc:  # the generator's own refusal is the reason
            for pos in mine:
                if targets[pos]["id"] not in verdict:
                    settle(targets[pos], pos, "refused", f"generator: {exc}")
            continue
        for pos, post in zip(positions, generated):
            if pos in targets and targets[pos]["id"] not in verdict:
                posts[pos] = post
                arms_of[pos] = pins[shorts[pos]["short_index"]]

    for pos, post in posts.items():
        row_of[pos] = targets[pos]
        new, fields = _rewrite(shorts[pos], _youtube_text(shorts[pos], post), arms_of[pos])
        if fields:
            rewritten[pos] = new
            fields_of[pos] = fields
        else:
            settle(targets[pos], pos, "unchanged", "text already matches the generator")

    if rewritten:
        trial = copy.deepcopy(raw)
        for pos, new in rewritten.items():
            trial["shorts"][pos] = new
        plan, tmp = _parsed(trial, sidecar)
        tmp.unlink()
        for pos in list(rewritten):
            refused = short_review_refusal(plan.shorts[pos])
            if refused:
                del rewritten[pos]
                settle(row_of[pos], pos, "refused", f"still refused by the Short review: {refused}")
    for pos, new in rewritten.items():
        old = shorts[pos]
        settle(
            row_of[pos],
            pos,
            "changed",
            fields=fields_of[pos],
            new_title=new["payloads"]["youtube"]["title"],
            old_hash=old["reviewed_plan_hash"],
            new_hash=new["reviewed_plan_hash"],
        )
    return verdict, rewritten


def _apply_sidecar(
    conn: sqlite3.Connection,
    sidecar: Path,
    verdict: Dict[int, ShortTextChange],
    rewritten: Dict[int, dict],
) -> None:
    """Write ``rewritten`` into ``sidecar`` atomically and move the queue rows' hashes."""
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    for pos, new in rewritten.items():
        raw["shorts"][pos] = new
    _, tmp = _parsed(raw, sidecar)
    terminal = ", ".join("?" for _ in TERMINAL_STATUSES)
    try:
        with conn:  # commits only after the sidecar has been replaced
            for change in verdict.values():
                if change.status != "changed":
                    continue
                moved = conn.execute(
                    "UPDATE deliveries SET plan_hash = ?, updated_at = ? "  # noqa: S608
                    "WHERE project_name = ? AND short_index = ? AND sidecar_path = ? "
                    f"AND plan_hash = ? AND status NOT IN ({terminal})",
                    (
                        change.new_hash,
                        datetime.now(timezone.utc).isoformat(),
                        change.project_name,
                        change.short_index,
                        str(sidecar),
                        change.old_hash,
                        *TERMINAL_STATUSES,
                    ),
                ).rowcount
                conn.execute(
                    "INSERT INTO response_log (at, platform, state, cap, action, mode, reason) "
                    "VALUES (?, 'youtube', 'plan_sync', 0, 'sync_short_text', 'live', ?)",
                    (
                        datetime.now(timezone.utc).isoformat(),
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
                            },
                            sort_keys=True,
                        ),
                    ),
                )
            os.replace(tmp, sidecar)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def sync_short_text(
    conn: sqlite3.Connection,
    decisions: ArmDecisions,
    *,
    apply: bool,
    now: Optional[datetime] = None,
) -> List[ShortTextChange]:
    """Re-write every eligible queued YouTube Short's text (see the module docstring);
    one :class:`ShortTextChange` per queued YouTube row, in row order."""
    now = now or datetime.now(timezone.utc)
    rows = conn.execute(_YT_SQL).fetchall()
    by_sidecar: Dict[str, List[sqlite3.Row]] = {}
    for row in rows:
        by_sidecar.setdefault(row["sidecar_path"], []).append(row)
    verdicts: Dict[int, ShortTextChange] = {}
    for path, group in by_sidecar.items():
        sidecar = Path(path)
        try:
            verdict, rewritten = _sidecar_changes(sidecar, group, decisions, now)
        except (OSError, ValueError, LookupError) as exc:
            for row in group:
                verdicts[row["id"]] = ShortTextChange(
                    row["id"],
                    row["project_name"],
                    row["short_index"],
                    "refused",
                    "",
                    "",
                    f"sidecar unusable: {exc}",
                )
            continue
        if apply and rewritten:
            _apply_sidecar(conn, sidecar, verdict, rewritten)
            verdict = {
                k: (
                    ShortTextChange(**{**v.__dict__, "written": True})
                    if v.status == "changed"
                    else v
                )
                for k, v in verdict.items()
            }
        verdicts.update(verdict)
    return [verdicts[row["id"]] for row in rows]
