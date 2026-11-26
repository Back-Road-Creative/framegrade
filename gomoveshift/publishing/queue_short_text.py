"""Re-write queued YouTube Shorts with the CURRENT text generator.

A Short queued days ago keeps the title, description and tags the generator made then,
so a generator fix (the no-repeated-place title rule, unit C2) never reaches it.
``sync_short_text`` closes that gap for every ``queued`` YouTube row whose slot is more
than :data:`SYNC_LEAD_TIME` away: it regenerates the Short's text through the generator
path Phase D uses (``plan_bracket_sync._generate_lf``, over the parent LF) and writes
title, description and tags back to the publish-plan sidecar.

Kept as it was: recording date, schedule,
media, and ``experiment_arms`` exactly as stored (the sync never picks, swaps or excludes
an arm). The generator renders each Short's stored Shorts arms (``SHORTS_FACTORS``)
verbatim; a Short whose known stored arm the regenerated text no longer carries is refused,
so the experiment is never mislabeled (an arm id outside ``SHORTS_ARMS`` is never rendered
and stays as stored).

A Short is refused whole (nothing written) when any delivery row of it, on any platform,
is neither ``queued`` nor terminal, holds a live lease, has no ``scheduled_at``, or has a
slot inside the lead time, or when a recovery row exists. A Short with a hand title
(``<plan dir>/social/titles.yaml``, as Phase D reads it) is reported ``unchanged`` /
``hand_title``. The no-repeated-place chain is seeded from the STORED text of the Short
before each rewritten one, and the generator's date is pinned to the LF's publish date.

Each rewritten Short must pass ``short_review_refusal``, ``models.validate_payloads`` and
``anti_spam_gate.check_batch`` (the gates enqueue runs, with the queue's recent text);
a refused Short is left unchanged and reported with the reason.

Plan hash (decision D4): a changed Short gets ``reviewed_plan_hash`` recomputed, every
non-terminal, unleased row whose ``plan_hash`` was the old one moves to the new one, and
``response_log`` keeps the old hash. Order: the audit row and queue update commit first,
then the sidecar is replaced, so a failure never leaves them silently apart; every run
also moves a queued row still on a stale hash to the sidecar's when the sidecar already
holds the synced text. No network, no platform call; a dry run writes no file at all.

Unattended nightly run (``health sync-short-text``): sidecars are planned in earliest-slot
order (ties by path). A sidecar's "recent text" for the anti-spam gates is the pre-run
snapshot of projects OUTSIDE the sync set plus the FINAL in-memory text of the sidecars
planned before it, never a later one's, so a rerun sees the same inputs and is a no-op.
A Short is also checked, across sidecars, against those earlier records inside the same
refusal loop as the neighbour-place check, so a refusal always re-checks the final text.
A sidecar that cannot be planned (a bad ``titles.yaml``, a trial plan the gates reject) still
contributes its STORED text, as it stays queued unchanged, to the sidecars around it.
The title ledger and the nightly title snapshot are read once per run. The brand's per-location
cap (``per_location_cap``) is counted within ONE sidecar's own untrimmed plan, never across
sidecars (the cross-sidecar passes run without it): an approximation of enqueue, which counts
over every sidecar in the directory (see ``_gate_refusals``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Collection, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from ..pipeline.text_format import split_trailing_hashtags
from .experiments import SHORTS_ARMS
from .plan_bracket_sync import (
    _generate_lf,
    _stored_arms,
    generated_title,
    location_places,
    ordinal_basis,
)
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


class LeaseRaceError(ValueError):
    """A queue row was leased between planning and writing (the write rolled back whole)."""


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
    #: how many non-terminal rows the queue update must move (a short count = a lost race)
    expect_moved: int = 0
    #: a refusal that clears on its own the next run (see ``is_actionable_refusal``); set where
    #: the reason is made, never inferred from its words. False = actionable (fails closed).
    transient: bool = False


#: Refusals that clear on their own the next run carry ``ShortTextChange.transient``, so the
#: nightly entry point neither files an incident for them nor counts them. The flag is set where
#: the reason is created; the reason's words are never matched (a reason can quote a Short's
#: text). EVERY other refusal is actionable, an unclassified or new one included, so the check
#: fails closed. A write-failure reason that carries the clash note is always actionable.
#:
#: * a sibling row's slot inside the lead time (still ahead): the slot passes or is moved by the
#:   queue. A slot already past on a still-queued row is transient only while the row's
#:   ``next_attempt_at`` (the dispatcher's retry backoff) is still ahead, else actionable
#:   ("is past but still queued").
#: * a sibling row holding a live lease: the dispatcher's lease expires or is released.
#: * "retried next run": the cascade after another sidecar's failed write, when the sidecar
#:   skipped by it clashes with nothing that was written (a clash is actionable: the write's
#:   own failure is too, unless it is a race or a busy database).
#: * a write that lost a race (``SidecarChangedError``, ``LeaseRaceError``) or met a busy
#:   database (``sqlite3`` BUSY): it rolled back whole and the next run sees the settled
#:   state.
#:
#: Permanent, needs-a-human refusals (all actionable): "generator: ...", an unreadable LF
#: sidecar or one with no location, a stored arm the generator cannot render, "no such Short in
#: the plan", "already holds the new plan hash", a recovery row, a row in a non-queued state
#: ("is <status>"), "has no usable scheduled_at", "no clip place ...", an old title this
#: plan did not generate (neither its "Nth stretch" stamp nor a measured place), "its ... place
#: repeats ...", a review, payload, plan or anti-spam gate refusal, "sidecar unusable: ..." (any other write failure
#: too), an unknown queue hash.
#:
#: Outcomes that are by design NOT refusals: "already holds the synced plan hash" (``unchanged``,
#: below: the text is already the generator's, the queue's UNIQUE key says another row carries
#: the target hash, and there is nothing for the sync to write) and a place the generator
#: could not vary (``place_variety`` ``no_alternative``: the generator sanctioned the repeat).


def is_actionable_refusal(change: "ShortTextChange") -> bool:
    return change.status == "refused" and not change.transient


def _busy(exc: BaseException) -> bool:
    """A ``sqlite3`` BUSY error (another connection holds the database): the write rolled back.
    LOCKED is not: without shared cache it is a conflict inside this connection, so it repeats."""
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None:
        return (code & 0xFF) == sqlite3.SQLITE_BUSY
    return str(exc) == "database is locked"


def _slot(raw) -> Optional[datetime]:
    try:
        at = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


def _leased(row, now: datetime) -> bool:
    return row["lease_owner"] is not None and (row["lease_expires_at"] or "") > now.isoformat()


def _sibling_refusal(siblings, now: datetime) -> Tuple[str, bool]:
    """``(why, transient)``: why this Short's delivery rows (any platform) forbid a rewrite
    (``""`` = none), and whether that clears on its own (a live lease, a slot in the lead time)."""
    for r in siblings:
        who = f"{r['platform']} row {r['id']}"
        if r["recovers_delivery_id"] is not None:
            return f"{who} is a recovery row", False
        if r["status"] in TERMINAL_STATUSES:
            continue
        if r["status"] != "queued":
            return f"{who} is {r['status']}", False
        if _leased(r, now):
            return f"{who} holds a live lease", True
        at = _slot(r["scheduled_at"]) if r["scheduled_at"] else None
        if at is None:
            return f"{who} has no usable scheduled_at", False
        if at <= now:  # the slot passed and the row is still queued
            # under the dispatcher's retry backoff (``next_attempt_at`` still ahead) it is tried
            # again on its own, and the escalation catches one that never is; else nothing clears it
            retry = _slot(r["next_attempt_at"]) if r["next_attempt_at"] else None
            backoff = retry is not None and retry > now
            return f"{who} slot {at.isoformat()} is past but still queued", backoff
        if at - now < SYNC_LEAD_TIME:
            return f"{who} slot {at.isoformat()} is inside the lead time", True
    return "", False


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
        short_at=(short.get("schedule") or {}).get("scheduled_at_utc"),
        parent_at=short.get("parent_lf_publish_at"),
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
) -> Tuple[Dict[int, object], Dict[int, str], Set[int]]:
    """``({position: post}, {position: why refused}, {positions asked alone})`` for the
    ``active`` positions (the last: Shorts the one-at-a-time fallback generated, seeded with
    their predecessor's STORED places, so no repeat of its NEW places is the generator's
    sanction); one
    generator pass per LF over the whole LF (so each Short keeps its position), with only
    the active Shorts composing text and every other Short's STORED places seeding the
    no-repeated-place chain."""
    posts: Dict[int, object] = {}
    refused: Dict[int, str] = {}
    alone: Set[int] = set()
    lfs: Dict[Tuple[str, int], List[int]] = {}
    for pos, s in enumerate(shorts):
        lfs.setdefault((s["project_name"], int(s["lf_index"])), []).append(pos)
    ordinals = ordinal_basis(shorts)
    # One title ledger across the sidecar's LFs, as Phase D keeps one per drive: a new title an
    # earlier LF took is taken for the later ones (else every LF's Short could take one label).
    taken: Set[str] = set()
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
            if not skip and not generated_title(s, location_places(location, record)):
                skip = "old title is not one this plan generated (no stamp, no measured place)"
            if skip:
                refused[pos] = skip
        live = [p for p in mine if p not in refused]
        if not live:
            continue
        dates = [str(shorts[p].get("parent_lf_publish_at") or "")[:10] for p in positions]
        try:
            today = date.fromisoformat(min(d for d in dates if d))
        except ValueError:
            today = date.fromisoformat(shorts[positions[0]]["recording_date"])

        def run(which: List[int]) -> Dict[int, object]:
            # The titles of the Shorts left as they were are taken: a rewritten Short may not
            # repeat one. A Short asked for composes its own, so its old title is not.
            used = {
                "titles": {
                    split_trailing_hashtags(shorts[p]["payloads"]["youtube"]["title"])[0]
                    .strip()
                    .lower()
                    for p in range(len(shorts))
                    if p not in which
                }
                | taken
            }
            # The generator only looks back, so the place a KEPT next Short names (a hand title,
            # say) is taken too: a rewritten Short varies from it instead of being refused for it.
            used["titles"] |= {
                place.strip().lower()
                for i, p in enumerate(positions[:-1])
                if p in which and positions[i + 1] not in which
                for place in [_stored_places(shorts[positions[i + 1]], location).get("title")]
                if place
            }
            seeds = {
                i: _stored_places(shorts[positions[i - 2]], location) if i > 1 else {}
                for i in range(1, len(positions) + 1)
                if i == 1 or positions[i - 2] not in which
            }
            view = [
                {**shorts[p], "_discard": p not in which, "_ordinal": ordinals[p]}
                for p in positions
            ]
            made = _generate_lf(
                view,
                record,
                location,
                lf_index,
                spooled_texts=spooled,
                experiment_today=today,
                previous_places_at=seeds,
                used_titles=used,
            )
            return {p: g for p, g in zip(positions, made) if p in which}

        try:
            posts.update(run(live))
        except Exception as exc:  # one Short may be what the generator refuses: ask alone
            if len(live) == 1:
                refused[live[0]] = f"generator: {exc}"
                continue
            for pos in live:
                try:
                    posts.update(run([pos]))
                    alone.add(pos)
                    taken.add(posts[pos].youtube_title.strip().lower())
                except Exception as one:  # the generator's own refusal is the reason
                    refused[pos] = f"generator: {one}"
        taken |= {posts[p].youtube_title.strip().lower() for p in live if p in posts}
    for pos in list(posts):
        # A stored arm the text no longer carries would mislabel the experiment.
        carried = getattr(posts[pos], "experiment_arms", None) or {}
        lost = {
            f: a
            for f, a in _stored_arms(shorts[pos]).items()
            if a in SHORTS_ARMS.get(f, ()) and carried.get(f) != a
        }
        if lost:
            del posts[pos]
            refused[pos] = f"stored arm(s) {lost!r} can no longer be rendered by the generator"
    return posts, refused, alone


def _gate_refusals(
    plan,
    positions: List[int],
    posts: Dict[int, object],
    shorts: List[dict],
    spooled: dict,
    earlier: Sequence = (),
    per_location_cap: Optional[int] = None,
) -> Dict[int, str]:
    """Why each rewritten position fails the gates enqueue runs: the Short review,
    ``validate_payloads``, the plan-text gates (``plan_gates``, on the generator's post as
    ``build_publish_plan`` calls them) and ``check_batch`` over THIS sidecar's trial plan
    behind the ``earlier`` sidecars' final records (the cross-sidecar rules; their own
    violations are not this sidecar's). The brand location cap is counted over this sidecar's
    own untrimmed plan only: an approximation of enqueue, which counts over every sidecar in
    the directory with ``select_capped_shorts`` minus the refused keys, so this can refuse a
    Short enqueue would accept (and the reverse)."""
    from . import plan_gates
    from .anti_spam_gate import check_batch
    from .models import PayloadLimitError, validate_payloads
    from .youtube_adapter import short_review_refusal

    out: Dict[int, str] = {}
    offset = len(earlier)
    queued = _QUEUED_YOUTUBE.get()
    batch = check_batch(
        [*earlier, *plan.shorts],
        per_location_cap=None,
        recent_texts=spooled,
        queued_youtube=None if queued is None else queued[1],
        now=None if queued is None else queued[0],
    )
    if per_location_cap is not None:
        own = check_batch(
            list(plan.shorts), per_location_cap=per_location_cap, recent_texts=spooled
        )
        batch = [*batch, *(replace_index(v, offset) for v in own if v.rule == "location-cap")]
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
        by_lf = {int(shorts[pos]["lf_index"]): [posts[pos]]}
        try:
            plan_gates.assert_shorts_metadata_in_region(by_lf)
            plan_gates.assert_plan_text_grammar(by_lf)
        except AssertionError as exc:
            out[pos] = f"plan gate: {exc}"
            continue
        hit = [
            v
            for v in batch
            if v.index == pos + offset
            and (v.platform in ("youtube", None) or v.rule == "caption-collision")
        ]
        if hit:
            out[pos] = "anti-spam gate: " + "; ".join(f"{v.rule}: {v.detail}" for v in hit)
    return out


def _audit_pairs(conn: sqlite3.Connection, row) -> Set[Tuple[str, str]]:
    """``(old, new)`` plan hashes this sync itself moved the Short's rows between (its own
    ``response_log`` rows): the only hashes a later run may treat as its own work."""
    out: Set[Tuple[str, str]] = set()
    for (reason,) in conn.execute(
        "SELECT reason FROM response_log WHERE action = 'sync_short_text'"
    ):
        try:
            note = json.loads(reason)
        except (TypeError, ValueError):
            continue
        if (
            isinstance(note, dict)
            and note.get("project") == row["project_name"]
            and note.get("short_index") == row["short_index"]
        ):
            out.add((str(note.get("old_plan_hash")), str(note.get("new_plan_hash"))))
    return out


def _hash_taken(siblings, conn: sqlite3.Connection, new_hash: str) -> str:
    """Name a row (terminal or not) that already holds ``new_hash`` on a platform whose
    non-terminal row would move to it: the queue's UNIQUE key would refuse that move."""
    for r in siblings:
        if r["status"] in TERMINAL_STATUSES:
            continue
        other = conn.execute(
            "SELECT id, platform FROM deliveries WHERE project_name = ? AND short_index = ? "
            "AND platform = ? AND plan_hash = ? AND id != ?",
            (r["project_name"], r["short_index"], r["platform"], new_hash, r["id"]),
        ).fetchone()
        if other:
            return f"{other['platform']} row {other['id']}"
    return ""


def _successor_clashes(
    sidecar: Path,
    shorts: List[dict],
    rewritten: Dict[int, dict],
    posts: Optional[Mapping[int, object]] = None,
    alone: Collection[int] = (),
) -> Dict[int, str]:
    """Neighbouring Shorts of one LF (any of the pair rewritten) that name the same place
    in their FINAL text (new if rewritten, else stored). The generator only looks backwards
    and a Short asked alone sees its neighbours' stored text, so the pair is checked here;
    the rewritten one is refused, the later of two rewritten ones. A repeat the generator
    itself sanctioned for the later Short (``place_variety`` reason ``no_alternative``: no other
    measured place existed) is not a clash, except, for a Short generated ``alone`` (the
    fallback seeds it with its predecessor's STORED places), where the predecessor's NEW place
    for that key differs from the stored one."""
    from gomoveshift.pipeline.social_generator import _norm

    out: Dict[int, str] = {}
    lfs: Dict[Tuple[str, int], List[int]] = {}
    for pos, s in enumerate(shorts):
        lfs.setdefault((s["project_name"], int(s["lf_index"])), []).append(pos)
    for (project, lf_index), positions in lfs.items():
        positions.sort(key=lambda p: int(shorts[p]["short_index"]))
        _, location, _ = _lf_inputs(sidecar.parent, project, lf_index)
        for a, b in zip(positions, positions[1:]):
            if a not in rewritten and b not in rewritten:
                continue
            mine = _stored_places(rewritten.get(a, shorts[a]), location)
            theirs = _stored_places(rewritten.get(b, shorts[b]), location)
            shared = [k for k in mine if k in theirs and _norm(mine[k]) == _norm(theirs[k])]
            if b in rewritten:
                variety = getattr((posts or {}).get(b), "place_variety", None) or {}
                # asked alone, b was seeded with a's STORED places: its sanction holds for a key
                # only if a's new place there is the stored one (a not rewritten: always)
                seed = (
                    _stored_places(shorts[a], location) if b in alone and a in rewritten else None
                )
                shared = [
                    k
                    for k in shared
                    if (variety.get(k) or {}).get("reason") != "no_alternative"
                    or (seed is not None and _norm(seed.get(k, "")) != _norm(mine[k]))
                ]
            elif shared:
                # b is kept as stored (a hand title, a published Short): the generator was steered
                # off its places (``_generate``), so a repeat left is one a could not avoid: its
                # own ``no_alternative``, or no other measured place. Refusing it would repeat
                # every night, since b never changes.
                variety = getattr((posts or {}).get(a), "place_variety", None) or {}
                i = positions.index(a)  # a's own predecessor's final place is no way out either
                before = (
                    _stored_places(
                        rewritten.get(positions[i - 1], shorts[positions[i - 1]]), location
                    )
                    if i
                    else {}
                )
                shared = [
                    k
                    for k in shared
                    if (variety.get(k) or {}).get("reason") != "no_alternative"
                    and _has_other_place(shorts[a], {_norm(mine[k]), _norm(before.get(k, ""))})
                ]
            if shared:
                later = b in rewritten
                out.setdefault(
                    b if later else a,
                    f"its {shared[0]} place repeats the {'previous' if later else 'next'} Short's",
                )
    return out


def _has_other_place(short: dict, taken: Set[str]) -> bool:
    """Whether the Short's clip measured a title place (a town or county) outside ``taken``."""
    from gomoveshift.pipeline.social_generator import _norm, measured_places

    found = measured_places(
        short.get("sampled_regions") or [],
        short.get("sampled_roads") or [],
        short.get("route_record"),
    )
    return any(_norm(b) not in taken for k, b in found if k != "road")


def _hash_refusal(conn: sqlite3.Connection, siblings, row, short: dict, new: dict) -> str:
    """Why a rewritten Short's queue rows cannot move to its new plan hash ("" = they can)."""
    old = short["reviewed_plan_hash"]
    new_hash = new["reviewed_plan_hash"]
    # a hash this sync wrote from the sidecar's own (crashed before the replace) is ours
    known = {old, new_hash} | {n for o, n in _audit_pairs(conn, row) if o == old}
    odd = [
        r["plan_hash"]
        for r in siblings
        if r["status"] not in TERMINAL_STATUSES and r["plan_hash"] not in known
    ]
    if odd:
        return f"a queue row's plan hash {odd[0][:12]} is unknown"
    taken = _hash_taken(siblings, conn, new_hash)
    return f"{taken} already holds the new plan hash" if taken else ""


def _live_rows(siblings, hashes) -> int:
    """How many non-terminal rows sit on one of ``hashes`` (what the queue update must move)."""
    return sum(
        1 for r in siblings if r["status"] not in TERMINAL_STATUSES and r["plan_hash"] in hashes
    )


def _sidecar_changes(
    conn: sqlite3.Connection,
    sidecar: Path,
    rows: List[sqlite3.Row],
    now: datetime,
    spooled: dict,
    earlier: Sequence = (),
    forced: Optional[Mapping[Tuple[str, int], str]] = None,
    per_location_cap: Optional[int] = None,
) -> Tuple[Dict[int, ShortTextChange], Dict[int, dict], str, list]:
    """Per YouTube row id: its verdict; the rewritten Short for each ``changed`` one
    (keyed by position in the plan); the sidecar's sha256; the sidecar's FINAL records of the
    Shorts it queues (rewritten text where a Short changes, stored text otherwise). ``spooled``
    is this sidecar's recent text (the queue's text outside this run's Shorts plus the final
    text of the sidecars before it), ``earlier`` those sidecars' final records and ``forced``
    the ``(project, short_index)`` Shorts a later sidecar's text obliges us to refuse. A
    plan that cannot be rebuilt raises (never an empty answer). Nothing is written here."""
    from gomoveshift.pipeline.social_generator import load_hand_titles

    body = sidecar.read_bytes()
    raw = json.loads(body)
    load_publish_plan(sidecar)  # the input must already be a valid plan
    sha = hashlib.sha256(body).hexdigest()
    shorts: List[dict] = raw["shorts"]
    by_key = {(s["project_name"], int(s["short_index"])): pos for pos, s in enumerate(shorts)}
    hand = load_hand_titles(sidecar.parent / "social" / "titles.yaml")
    verdict: Dict[int, ShortTextChange] = {}
    forced = forced or {}
    queued = {(r["project_name"], int(r["short_index"])) for r in rows}

    def final_records(rewritten_now: Dict[int, dict], trial_plan=None) -> list:
        if trial_plan is None:
            trial_raw = copy.deepcopy(raw)
            for pos, new in rewritten_now.items():
                trial_raw["shorts"][pos] = new
            trial_plan = plan_from_raw(trial_raw, sidecar)
        return [r for r in trial_plan.shorts if (r.project_name, r.short_index) in queued]

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
        why, why_transient = _sibling_refusal(sib, now)
        if at is not None and at - now < SYNC_LEAD_TIME:
            settle(row, pos, "inside_lead_time", f"slot {at.isoformat()} is inside the lead time")
        elif why:
            settle(row, pos, "refused", why, transient=why_transient)
        elif (row["project_name"], int(row["short_index"])) in forced:
            settle(row, pos, "refused", forced[(row["project_name"], int(row["short_index"]))])
        elif int(shorts[pos]["short_index"]) in hand:
            settle(row, pos, "unchanged", "hand_title")
        else:
            targets[pos] = row
            siblings[pos] = sib
    if not targets:
        return verdict, {}, sha, final_records({})

    active: Set[int] = set(targets)
    rewritten: Dict[int, dict] = {}
    plan = None
    fields_of: Dict[int, Tuple[str, ...]] = {}
    unchanged: Set[int] = set()
    while True:  # a refusal changes the stored text that seeds its successor: redo
        posts, refused, alone = _generate(sidecar, shorts, active, spooled)
        for pos, why in refused.items():
            settle(targets[pos], pos, "refused", why)
        active -= set(refused)
        rewritten, fields_of, unchanged = {}, {}, set()
        for pos, post in posts.items():
            new, fields = _rewrite(shorts[pos], _youtube_text(shorts[pos], post))
            if fields:  # the place the new title names, for the neighbour-place rule
                new.pop("place_variety", None)
                if getattr(post, "place_variety", None):
                    new["place_variety"] = {k: dict(v) for k, v in post.place_variety.items()}
            if fields:
                rewritten[pos], fields_of[pos] = new, fields
            else:
                unchanged.add(pos)
        again = set(refused)
        clash = _successor_clashes(sidecar, shorts, rewritten, posts, alone)
        for pos, why in clash.items():
            settle(targets[pos], pos, "refused", why)
            active.discard(pos)
        again |= set(clash)
        rewritten = {p: n for p, n in rewritten.items() if p not in clash}
        if rewritten:
            trial = copy.deepcopy(raw)
            for pos, new in rewritten.items():
                trial["shorts"][pos] = new
            plan = plan_from_raw(trial, sidecar)
            bad = _gate_refusals(
                plan, list(rewritten), posts, shorts, spooled, earlier, per_location_cap
            )
            for pos, why in bad.items():
                settle(targets[pos], pos, "refused", why)
                active.discard(pos)
            again |= set(bad)
            for pos in set(rewritten) - set(bad):
                why = _hash_refusal(conn, siblings[pos], targets[pos], shorts[pos], rewritten[pos])
                if why:
                    settle(targets[pos], pos, "refused", why)
                    active.discard(pos)
                    again.add(pos)
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
        written = {o for o, n in _audit_pairs(conn, targets[pos]) if n == sidecar_hash}
        foreign = [h for h in stale if h not in written]
        taken = _hash_taken(siblings[pos], conn, sidecar_hash)
        if foreign:
            settle(
                targets[pos],
                pos,
                "refused",
                f"a queue row's plan hash {foreign[0][:12]} is unknown",
            )
        elif stale and taken:
            settle(  # by design not a refusal (see is_actionable_refusal)
                targets[pos], pos, "unchanged", f"{taken} already holds the synced plan hash"
            )
        elif stale:
            settle(
                targets[pos],
                pos,
                "reconciled",
                "queue hash was behind the sidecar's synced text",
                old_hash=stale[0],
                new_hash=sidecar_hash,
                old_hashes=tuple(stale),
                expect_moved=_live_rows(siblings[pos], stale),
            )
        else:
            settle(targets[pos], pos, "unchanged", "text already matches the generator")
    for pos, new in rewritten.items():
        old = shorts[pos]["reviewed_plan_hash"]
        new_hash = new["reviewed_plan_hash"]
        moving = sorted(
            {
                r["plan_hash"]
                for r in siblings[pos]
                if r["status"] not in TERMINAL_STATUSES and r["plan_hash"] != new_hash
            }
        )
        settle(
            targets[pos],
            pos,
            "changed",
            fields=fields_of[pos],
            new_title=new["payloads"]["youtube"]["title"],
            old_hash=old,
            new_hash=new_hash,
            old_hashes=tuple(moving),
            expect_moved=_live_rows(siblings[pos], moving),
        )
    return verdict, rewritten, sha, final_records(rewritten, plan if rewritten else None)


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
                if moved != change.expect_moved:  # rolls back every change of this sidecar
                    raise LeaseRaceError("a row was leased during the write")
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


@contextmanager
def _title_inputs_once() -> Iterator[None]:
    """Read the published-titles ledger and the nightly title snapshot once for the whole run.

    The generator reads both on every title it composes; a per-run cache (a context variable
    the two readers consult, no function swapped) makes every sidecar, dry run or apply, judge
    against the same inputs. Reset on exit, so nothing is cached across runs."""
    from gomoveshift.pipeline.title_decision import title_input_cache

    token = title_input_cache.set({})
    try:
        yield
    finally:
        title_input_cache.reset(token)


#: ``(now, [(slot, record)])``: the queue's active YouTube Shorts as stored, read once per
#: run by ``sync_short_text``, so ``check_batch``'s ``neighbour-place`` rule sees a Short's
#: neighbours by slot from any long-form or project. Unset = the rule is not run.
_QUEUED_YOUTUBE: ContextVar[Optional[Tuple[datetime, list]]] = ContextVar(
    "queue_short_text_queued_youtube", default=None
)


@contextmanager
def _queued_youtube_once(conn: sqlite3.Connection, now: datetime) -> Iterator[None]:
    """Read the queue's active YouTube Shorts (as stored) once for the whole run, before any
    apply, for the ``neighbour-place`` rule (``_QUEUED_YOUTUBE``); reset on exit."""
    from .anti_spam_gate import queued_youtube_records

    token = _QUEUED_YOUTUBE.set((now, queued_youtube_records(conn)))
    try:
        yield
    finally:
        _QUEUED_YOUTUBE.reset(token)


def _with_texts(base: Dict[str, List[str]], records: Sequence) -> Dict[str, List[str]]:
    """``base`` (platform -> texts) plus every text of ``records``, no repeats; a new dict."""
    from .anti_spam_gate import _record_texts

    out = {platform: list(texts) for platform, texts in base.items()}
    for record in records:
        for platform, texts in _record_texts(record).items():
            bucket = out.setdefault(platform, [])
            bucket.extend(t for t in texts if t not in bucket)
    return out


def _slot_order(by_sidecar: Dict[str, List[sqlite3.Row]]) -> List[str]:
    """Sidecar paths by their earliest queued slot, ties (and unusable slots) by path."""
    far = datetime.max.replace(tzinfo=timezone.utc)

    def earliest(path: str) -> datetime:
        slots = [_slot(r["scheduled_at"]) if r["scheduled_at"] else None for r in by_sidecar[path]]
        return min((a for a in slots if a is not None), default=far)

    return sorted(by_sidecar, key=lambda path: (earliest(path), path))


def _new_text_clashes(new_records: Sequence, later: Sequence) -> Dict[Tuple[str, int], str]:
    """``(project, short_index)`` -> why, for each of ``new_records`` (one sidecar's rewritten
    text) whose text collides with the FINAL text of a LATER sidecar, one the later sidecar's
    own gates never saw change (a Short left as stored)."""
    from .anti_spam_gate import check_batch

    if not new_records or not later:
        return {}
    batch = check_batch(
        [*new_records, *later], per_location_cap=None, recent_texts=_with_texts({}, later)
    )
    out: Dict[Tuple[str, int], str] = {}
    for v in batch:
        if v.index < len(new_records) and (
            v.platform in ("youtube", None) or v.rule == "caption-collision"
        ):
            record = new_records[v.index]
            key = (record.project_name, record.short_index)
            why = f"{v.rule}: {v.detail}"
            out[key] = f"{out[key]}; {why}" if key in out else why
    return {k: f"anti-spam gate: {why} (against a later sidecar's text)" for k, why in out.items()}


def _stored_records(path: str) -> list:
    """The records of the sidecar as stored ([] when it cannot be loaded)."""
    try:
        return list(load_publish_plan(Path(path)).shorts)
    except (OSError, ValueError, LookupError):
        return []


def _clash_note(clash: Mapping[Tuple[str, int], str]) -> str:
    return (
        "; text written earlier in this run clashes with this sidecar's stored text: "
        + "; ".join(f"{k[0]} #{k[1]} {v}" for k, v in sorted(clash.items()))
    )


def replace_index(violation, offset: int):
    """``violation`` re-addressed by ``offset`` (a batch run on a plan alone, read in a longer one)."""
    return replace(violation, index=violation.index + offset)


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
    from .anti_spam_gate import recent_text_entries

    def refuse_all(group, why, transient=False):
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
                transient=transient,
            )

    order = _slot_order(by_sidecar)
    sync_keys = {(r["project_name"], int(r["short_index"])) for r in rows}
    with _title_inputs_once(), _queued_youtube_once(conn, now):
        # ONE read of the queue's text per run, taken before any apply. A sidecar's view of it
        # leaves out its own projects and the Shorts this run rewrites for other sidecars (their
        # text comes from `earlier`, final): everything else, a published Short beside a queued
        # one included, reaches its gates. A rerun therefore sees the same inputs.
        entries = recent_text_entries(conn, now=now)

        def spooled_for(path: str, earlier: Sequence) -> Dict[str, List[str]]:
            own = {r["project_name"] for r in by_sidecar[path]}
            base: Dict[str, List[str]] = {}
            for project, index, platform, text in entries:
                if project in own or (project, index) in sync_keys:
                    continue
                bucket = base.setdefault(platform, [])
                if text not in bucket:
                    bucket.append(text)
            return _with_texts(base, earlier)

        forced: Dict[Tuple[str, int], str] = {}
        while True:
            planned: Dict[str, tuple] = {}
            unusable: Dict[str, str] = {}
            finals: Dict[str, list] = {}
            earlier: list = []
            for path in order:
                try:
                    planned[path] = _sidecar_changes(
                        conn,
                        Path(path),
                        by_sidecar[path],
                        now,
                        spooled_for(path, earlier),
                        tuple(earlier),
                        forced,
                        per_location_cap,
                    )
                except (OSError, ValueError, LookupError) as exc:
                    unusable[path] = f"sidecar unusable: {exc}"
                    # It stays queued as stored, so its STORED text still reaches every other
                    # sidecar's gates (as an earlier or a later one); unloadable = nothing to see.
                    finals[path] = _stored_records(path)
                    earlier.extend(finals[path])
                    continue
                finals[path] = planned[path][3]
                earlier.extend(finals[path])
            # An earlier sidecar's NEW text must also clear the later sidecars' final text,
            # including a Short they leave as stored (their own gates only check what changes).
            found: Dict[Tuple[str, int], str] = {}
            for i, path in enumerate(order):
                if path not in planned:
                    continue
                new_keys = {
                    (n["project_name"], int(n["short_index"])) for n in planned[path][1].values()
                }
                later = [r for p in order[i + 1 :] for r in finals.get(p, ())]
                new = [r for r in finals[path] if (r.project_name, r.short_index) in new_keys]
                found.update(_new_text_clashes(new, later))
            fresh = {k: v for k, v in found.items() if k not in forced}
            if not fresh:
                break
            forced.update(fresh)  # grows every round, so the loop ends; re-plan with the refusals
    for path, why in unusable.items():
        refuse_all(by_sidecar[path], why)
    write_failed = False
    written_new: list = []
    for path in order:
        if path not in planned:
            continue
        verdict, rewritten, sha, finals_ = planned[path]
        sidecar, group = Path(path), by_sidecar[path]
        writes = any(v.status in ("changed", "reconciled") for v in verdict.values())
        if apply and writes and write_failed:
            # Skipped, so it stays as STORED: what was written before the failure is checked
            # against that, as for the sidecar whose own write failed (below).
            why = "an earlier sidecar's write failed; retried next run"
            clash = _new_text_clashes(written_new, _stored_records(path))
            if clash:
                why += _clash_note(clash)
            refuse_all(group, why, transient=not clash)
            continue
        if apply and writes:
            try:
                _apply_sidecar(conn, sidecar, verdict, rewritten, sha, now)
            except Exception as exc:  # noqa: BLE001 - a crash after partial writes helps no one
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                known = isinstance(exc, (OSError, ValueError, sqlite3.Error))
                why = f"sidecar unusable: write failed: {exc if known else f'{type(exc).__name__}: {exc}'}"
                # The sidecars written before this one were checked against its PLANNED text;
                # it stays as STORED, so check what they wrote against that.
                clash = _new_text_clashes(written_new, _stored_records(path))
                if clash:
                    why += _clash_note(clash)
                # Transient only when the write lost a race or met a busy database AND nothing
                # earlier in this run clashes with what stays stored: a clash is always actionable.
                raced = isinstance(exc, (SidecarChangedError, LeaseRaceError)) or _busy(exc)
                refuse_all(group, why, transient=raced and not clash)
                write_failed = True
                continue
            new_keys = {(n["project_name"], int(n["short_index"])) for n in rewritten.values()}
            written_new.extend(r for r in finals_ if (r.project_name, r.short_index) in new_keys)
            verdict = {
                k: replace(v, written=True) if v.status in ("changed", "reconciled") else v
                for k, v in verdict.items()
            }
        verdicts.update(verdict)
    return [verdicts[row["id"]] for row in rows]
