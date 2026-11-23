"""A first comment found live with a replaced text: the ``first_comment_stale`` health incident
(shown by ``status --json``) and the automatic in-place fix, run at the start of each drain."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .first_comment import (
    ReplacePostFailedError,
    full_drive_comment_text,
    full_drive_video_id,
    is_full_drive_comment,
    parent_is_public_at,
)
from .queue import (
    FIRST_COMMENT_MAX_ATTEMPTS,
    REOPEN_LAG_WINDOW,
    adopt_capped_first_comment,
    adopt_removed_first_comment,
    capped_first_comment_corrections,
    capped_unposted_lag_rows,
    drop_private_edit,
    drop_reopened_deletes,
    fail_first_comment_correction,
    first_comment_correction_ids,
    first_comment_corrections,
    first_comment_intended_texts,
    first_comment_prior_texts,
    mark_first_comment_removed,
    pending_first_comment_corrections,
    queue_own_extra_deletes,
    registered_platform_id,
    removed_first_comment_ids,
    removed_first_comment_rows,
    reopen_gone_first_comment,
    reopen_lag_expired,
    reopened_delete_rows,
    resolve_first_comment_correction,
    unpost_first_comment_for_repost,
    unpost_removed_first_comment,
    unposted_first_comment_ids,
)

logger = logging.getLogger(__name__)

KIND = "first_comment_stale"
CAPPED_KIND = "first_comment_correction_capped"
REPOST_KIND = "first_comment_repost_failed"
REMOVED_KIND = "first_comment_removed"
OWN_ID_KIND = "first_comment_own_id_mismatch"
LAG_DROP_KIND = "first_comment_lag_delete_dropped"
#: Prefix of an own-id incident cause when the read-back itself failed (nothing was compared).
UNVERIFIED = "unverified: "
#: Prefix of the cause when a read-back CONFIRMED (against the post's listing) that the comment is gone.
GONE = "gone: "
_DETAIL_MAX = 500
_LAG_DROP_DETAIL_MAX = 1500  # a drop names every id: room for a dozen full Meta ids
#: The hand-delete advice of a ``LAG_DROP_KIND`` incident: appended whole, never inside the cap.
_LAG_DROP_ADVICE = (
    ". Delete by hand any comment named here that is live and is a second copy of ours: the "
    "pipeline will not look again, and a copy that appears after the wait needs a hand delete"
)


def _lag_drop_cause(drops: list) -> str:
    """One line for a drop: each reason once, the ids it covers leading it."""
    reasons: dict = {}
    for live_id, reason in drops:
        reasons.setdefault(reason, []).append(live_id)
    return "; ".join(
        f"comments {', '.join(ids)}: {reason}; the deletes are dropped"
        if len(ids) > 1
        else f"comment {ids[0]}: {reason}; the delete is dropped"
        for reason, ids in reasons.items()
    )


def _merge_drops(prior: Any, new: list) -> list:
    """An open incident's drops plus this drain's, one entry per id, in first-seen order (a
    repeated id keeps its place and takes the newer reason)."""
    merged: dict = {}
    for pair in [*(prior if isinstance(prior, list) else []), *new]:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            merged[str(pair[0])] = str(pair[1])
    return [[live_id, reason] for live_id, reason in merged.items()]


class CorrectionUnavailableError(Exception):
    """No client call for this correction, or the live comment id is unknown."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _target(row: Any) -> Optional[str]:
    """The post the row's comment sits on (what the drain posts to), or ``None``."""
    return registered_platform_id(
        row["platform"], row["platform_upload_id"], row["platform_post_id"]
    )


def _key(delivery_id: int, kind: str = KIND) -> str:
    return f"{kind}:{delivery_id}"


def _describe(row: Any) -> str:
    steps = []
    for item in first_comment_corrections(row):
        if item.get("action") == "edit":
            steps.append(f"edit comment {item.get('live_id') or '?'} to {item.get('to')!r}")
        else:
            steps.append(f"delete duplicate comment {item.get('live_id') or '?'}")
    if row["first_comment_correction_attempts"] >= FIRST_COMMENT_MAX_ATTEMPTS:
        # Never promise a fix the drain has stopped making (the capped incident names the hand step).
        tail = f"the drain stopped trying after {FIRST_COMMENT_MAX_ATTEMPTS} refusals; pending: "
    else:
        tail = "the drain will "
    return (
        f"{row['platform']}#{row['id']} {row['project_name']} short {row['short_index']}: "
        f"{row['first_comment_error']}; {tail}{'; '.join(steps)}"
    )[:_DETAIL_MAX]


def _describe_capped(row: Any) -> str:
    return (
        f"{row['platform']}#{row['id']} {row['project_name']} short {row['short_index']}: the "
        f"stale-comment correction was refused {FIRST_COMMENT_MAX_ATTEMPTS} times; the drain "
        "stopped. Fix it by hand: delete or edit the old comment so only the intended one "
        f"({row['first_comment_text']!r}) is live, then clear first_comment_correction. Pending: "
        + "; ".join(
            f"{i.get('action')} comment {i.get('live_id') or '?'}"
            for i in first_comment_corrections(row)
        )
    )[:_DETAIL_MAX]


def raise_stale_incident(
    conn: sqlite3.Connection,
    delivery_id: int,
    incidents_dir: Optional[Path] = None,
    *,
    capped: bool = False,
) -> None:
    """File (or refresh) the one open incident for ``delivery_id`` (``capped``: the hand-step
    one). Never fatal."""
    kind = CAPPED_KIND if capped else KIND
    from ..health.store import (  # lazy: health imports the pipeline
        default_data_root,
        default_incidents_dir,
        new_incident,
        safe_name,
        write_row,
    )

    row = conn.execute(
        "SELECT id, project_name, short_index, platform, first_comment_error, "
        "first_comment_text, first_comment_correction, first_comment_correction_attempts "
        "FROM deliveries WHERE id=?",
        (delivery_id,),
    ).fetchone()
    if row is None or not first_comment_corrections(row):
        return
    directory = incidents_dir or default_incidents_dir(default_data_root())
    filename = f"{safe_name(_key(delivery_id, kind))}.json"
    detail = _describe_capped(row) if capped else _describe(row)
    try:
        path = directory / filename
        prior = json.loads(path.read_text()) if path.is_file() else None
        if isinstance(prior, dict) and not prior.get("resolved_at"):
            prior.update(
                detail=detail,
                error=row["first_comment_error"],
                live_comments=first_comment_corrections(row),
            )
            write_row(directory, prior)
            return
        write_row(
            directory,
            new_incident(
                kind,
                _key(delivery_id, kind),
                filename,
                _now_iso(),
                delivery_id=delivery_id,
                platform=row["platform"],
                project=row["project_name"],
                live_comments=first_comment_corrections(row),
                error=row["first_comment_error"],
                detail=detail,
            ),
        )
    except (OSError, ValueError):
        logger.exception(
            "first comment: could not write the stale-comment incident %s", delivery_id
        )


def _hand_re_add_text(conn: sqlite3.Connection, delivery_id: int) -> str:
    """The exact text a hand re-add must say: the full-drive comment of the row's parent video
    (the plan's sidecar names it), else the shape with the id left to fill in."""
    from . import enqueue_from_plan  # lazy: it imports the queue

    row = conn.execute(
        "SELECT id, project_name, short_index, sidecar_path FROM deliveries WHERE id=?",
        (delivery_id,),
    ).fetchone()
    try:
        video_id = enqueue_from_plan.parent_video_id(row, {}) if row is not None else None
        if video_id:
            return full_drive_comment_text(video_id)
    except Exception:  # noqa: BLE001 -- the shape below is the fallback; never fatal
        logger.exception("first comment: no parent video id for delivery %s", delivery_id)
    return "\U0001f3ac Watch the full drive: https://youtu.be/<the full drive's video id>"


def raise_repost_incident(
    conn: sqlite3.Connection,
    delivery_id: int,
    cause: str,
    incidents_dir: Optional[Path] = None,
    kind: str = REPOST_KIND,
    drops: Optional[list] = None,
) -> None:
    """File (or refresh) the open incident for a row whose stale comment was deleted but whose
    replacement did not post: zero comments are live until the drain reposts. ``kind``
    ``REMOVED_KIND``: the comment was deleted and the row has no text to post (a private relink
    left none); the drain restores it by itself once the row has text. ``kind`` ``LAG_DROP_KIND``
    with ``drops`` (``[live_id, reason]`` pairs): an OPEN incident of the row keeps its earlier
    drops, so one incident names every dropped id, and it is pushed again. Never fatal."""
    from ..health.store import (  # lazy: health imports the pipeline
        default_data_root,
        default_incidents_dir,
        new_incident,
        safe_name,
        write_row,
    )

    row = conn.execute(
        "SELECT project_name, short_index, platform FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    if row is None:
        return
    directory = incidents_dir or default_incidents_dir(default_data_root())
    filename = f"{safe_name(_key(delivery_id, kind))}.json"
    where = f"{row['platform']}#{delivery_id} {row['project_name']} short {row['short_index']}: "
    merged: Optional[list] = None
    if kind == REMOVED_KIND:
        # The exact re-add text leads, so the length cap can never cut it.
        detail = (
            f"{where}a hand re-add (by the channel account) must say exactly "
            f"{_hand_re_add_text(conn, delivery_id)!r} and nothing else. The comment was deleted "
            f"and the row has no text to post ({cause}); no comment is live. The pipeline watches "
            "for our comment and closes this incident when it is live again: it reposts once the "
            "row has text, and a hand re-add is picked up on the next drain"
        )[:_DETAIL_MAX]
    elif kind == REPOST_KIND and cause.startswith(GONE):
        detail = (
            f"{where}the comment we posted was re-read on a later drain and confirmed gone "
            f"({cause[len(GONE) :]}); the row is unposted and the drain reposts it, up to "
            f"{FIRST_COMMENT_MAX_ATTEMPTS} tries; after that it stays unposted and this "
            "incident stays open. The old id is queued as our own delete, held while its read "
            "lags, so a comment that only lagged is deleted, never left as a second copy"
        )[:_DETAIL_MAX]
    elif kind == LAG_DROP_KIND:
        if drops is not None:
            merged = _merge_drops(_open_prior(directory / filename).get("drops"), drops)
            cause = _lag_drop_cause(merged)
        # The ids lead, and only they are capped: the advice is appended whole, never cut.
        detail = (
            f"{where}{cause}"[: max(0, _LAG_DROP_DETAIL_MAX - len(_LAG_DROP_ADVICE))]
            + _LAG_DROP_ADVICE
        )
    elif kind == OWN_ID_KIND and cause.startswith(GONE):
        detail = (
            f"{where}the comment we just posted was read back as gone ({cause[len(GONE) :]}); it "
            "is recorded as posted, and the next drain re-reads it: it closes this incident when "
            "it reads back as ours, and a second confirmed gone routes the row: with text it is "
            f"unposted and reposted, up to {FIRST_COMMENT_MAX_ATTEMPTS} tries (then it stays "
            "unposted with a repost incident open); with no text it is marked removed and a "
            "removed incident opens"
        )[:_DETAIL_MAX]
    elif kind == OWN_ID_KIND:
        who = (
            f"could not be verified as ours ({cause[len(UNVERIFIED) :]})"
            if cause.startswith(UNVERIFIED)
            else f"reads back with author {cause}, not our own id"
        )
        detail = (
            f"{where}the comment we just posted {who}; the configured page_id / ig_user_id is "
            "wrong or Graph's from.id differs for this token (UNVERIFIED which). It is left "
            "alone, never deleted; until the id is right a lost answer can post a second copy. "
            "This incident closes by itself once a read-back shows our own id as its author, "
            "or, when a later drain confirms the comment gone, when that drain routes the row "
            "(reposts it, or marks it removed)"
        )[:_DETAIL_MAX]
    else:
        detail = (
            f"{where}the stale comment was deleted and the replacement did not post ({cause}); "
            "no comment is live until the drain reposts it"
        )[:_DETAIL_MAX]
    try:
        path = directory / filename
        prior = json.loads(path.read_text()) if path.is_file() else None
        if isinstance(prior, dict) and not prior.get("resolved_at"):
            prior.update(detail=detail, error=cause)
            if merged is not None:
                prior.update(drops=merged, pushed=False, pushed_at=None)
            write_row(directory, prior)
            return
        write_row(
            directory,
            new_incident(
                kind,
                _key(delivery_id, kind),
                filename,
                _now_iso(),
                delivery_id=delivery_id,
                platform=row["platform"],
                project=row["project_name"],
                error=cause,
                detail=detail,
                **({"drops": merged} if merged is not None else {}),
            ),
        )
    except (OSError, ValueError):
        logger.exception("first comment: could not write the repost incident %s", delivery_id)


def _open_prior(path: Path) -> dict:
    """The open incident at ``path``, or ``{}`` (none, resolved or unreadable)."""
    try:
        prior = json.loads(path.read_text()) if path.is_file() else None
    except (OSError, ValueError):
        return {}
    return prior if isinstance(prior, dict) and not prior.get("resolved_at") else {}


def _note_removed(
    conn: sqlite3.Connection, delivery_id: int, result: Any, incidents_dir: Optional[Path]
) -> None:
    """A delete landed: when the row now holds no comment text, zero comments are live and
    nothing can be posted, so mark it removed and keep an incident open. It closes only once a
    NEW live id is recorded (``restore_removed`` unposts the row once it has text; the drain then
    reposts or adopts, and the repost incident closes on that)."""
    if not mark_first_comment_removed(conn, delivery_id, _now_iso()):
        return
    cause = "its replacement is private at the short's slot, or the text was dropped"
    result.errors.append(
        f"delivery {delivery_id}: the stale comment was deleted and the row has no text to post; "
        "no comment is live. The pipeline watches for our comment and closes the incident when "
        "it is live again: it reposts once the row has text, and a hand re-add (by the channel "
        "account) is picked up on the next drain"
    )
    raise_repost_incident(conn, delivery_id, cause, incidents_dir, kind=REMOVED_KIND)


def restore_removed(
    conn: sqlite3.Connection,
    platforms: Iterable[str],
    incidents_dir: Optional[Path],
    client_for: Optional[Callable[[str], object]] = None,
    result: Any = None,
) -> None:
    """Put a removed row back on the drain: with its own text, or the one the plan gives it now
    (the backfill's source). The row is unposted, so the drain reposts it or adopts our own copy
    through its scan; a repost incident stays open until a new live id is recorded. No text: the
    post's comments are listed (``client_for(platform).own_comments``) and a top-level comment
    positively ours with the full-drive shape is adopted as the row's live comment (a hand
    re-add), which closes the removed incident. A viewer's or authorless copy is never adopted. A
    listing that fails or is cut short leaves the row removed and adds an error line to
    ``result``."""
    from . import enqueue_from_plan  # lazy: it imports the queue

    cache: dict = {}
    for row in removed_first_comment_rows(conn, platforms):
        text = (row["first_comment_text"] or "").strip() or enqueue_from_plan._backfill_text(
            row, cache
        )
        if text:
            if unpost_removed_first_comment(conn, row["id"], text, _now_iso()):
                raise_repost_incident(
                    conn, row["id"], "the comment text is back; the drain posts it", incidents_dir
                )
        elif client_for is not None:
            _adopt_hand_re_add(conn, row, client_for, result, incidents_dir)


def _adopt_hand_re_add(
    conn: sqlite3.Connection,
    row: Any,
    client_for: Callable[[str], object],
    result: Any,
    incidents_dir: Optional[Path] = None,
) -> None:
    """Adopt our own full-drive comment found live on a removed row's post (see
    ``restore_removed``). Only OUR CURRENT link is adopted: a comment whose link names a video a
    relink replaced (``first_comment_prior_text``, compared by video id) or the id of the comment
    we deleted (``first_comment_live_id``) is skipped, and when the plan's sidecar names the
    current video only a comment linking that video is taken. Comments that all name the same
    video are the same link: the lowest comment id is adopted and the rest are queued as our own
    extra deletes. Comments naming different videos, with the current one unknown, are ambiguous
    and none is adopted (listing order never decides). Never raises; a failed or cut-short
    listing, a refused adopt, an ambiguity or only-skipped candidates is an error line."""
    target = _target(row)
    problem = ""
    try:
        listing = getattr(client_for(row["platform"]), "own_comments", None)
        if listing is None or not target:
            problem = "cannot list the post's comments to look for our comment"
        else:
            comments, complete = listing(target)
            if not complete:
                problem = "too many comments to read them all"
            problem = problem or _adopt_one(conn, row, comments, result, incidents_dir)
    except Exception as exc:  # noqa: BLE001 -- the row stays removed; said, never raised
        problem = (
            f"could not look for our removed comment ({str(exc).strip() or exc.__class__.__name__})"
        )
    if problem:
        logger.warning("first comment: delivery %s: removed-row adopt: %s", row["id"], problem)
        if result is not None:
            result.errors.append(
                f"delivery {row['id']}: {problem}; it stays removed and the incident open until "
                "a later drain can"
            )


def _note(row: Any, result: Any, message: str) -> None:
    """A said-not-silent note on an adopt that went through."""
    logger.warning("first comment: delivery %s: removed-row adopt: %s", row["id"], message)
    if result is not None:
        result.errors.append(f"delivery {row['id']}: {message}")


def _id_order(comment_id: str) -> tuple:
    """Sort key for the copy an adopt keeps: Meta ids are numbers joined by ``_`` (``123_99`` is
    below ``123_100``), so they compare by number; YouTube ids are opaque strings and compare by
    text. Numeric ids sort before the rest."""
    parts = comment_id.split("_")
    if all(p.isdigit() for p in parts):
        return (0, tuple(int(p) for p in parts), "")
    return (1, (), comment_id)


def _adopt_one(
    conn: sqlite3.Connection,
    row: Any,
    comments: Iterable,
    result: Any = None,
    incidents_dir: Optional[Path] = None,
) -> str:
    """Adopt our current full-drive comment of ``comments`` (own comments only) for removed
    ``row``; else why not (``""`` when adopted)."""
    from . import enqueue_from_plan  # lazy: it imports the queue

    prior = {t.strip() for t in first_comment_prior_texts(row)}
    prior_ids = {full_drive_video_id(t) for t in prior} - {None}
    deleted = str(row["first_comment_live_id"] or "")
    try:
        want = enqueue_from_plan.parent_video_id(row, {})
    except Exception:  # noqa: BLE001 -- unknown current link: the other guards still apply
        logger.exception("first comment: no parent video id for delivery %s", row["id"])
        want = None
    groups: dict = {}  # video id -> {comment id: text}
    replaced: dict = {}  # our own copies of a replaced link
    skipped: list = []
    wrong: list = []
    for comment_id, said in comments:
        if not (comment_id and is_full_drive_comment(said)):
            continue
        cid, text = str(comment_id), said.strip()
        vid = full_drive_video_id(text)
        if text in prior or vid in prior_ids or (deleted and cid == deleted):
            skipped.append(cid)
            if cid != deleted:
                replaced[cid] = text
        elif want and vid != want:
            wrong.append(cid)
        else:
            groups.setdefault(vid, {})[cid] = text
    said_wrong = (
        f"our comment {', '.join(wrong)} links a different video than the plan's current "
        f"link ({want}); it is skipped"
        if wrong
        else ""
    )
    if len(groups) > 1:
        ids = sorted(c for g in groups.values() for c in g)
        return (
            f"our full-drive comments {', '.join(ids)} link different videos; none is adopted, "
            "since which is the current link cannot be told"
            + (f"; {said_wrong}" if said_wrong else "")
        )
    if not groups:
        if skipped:
            return (
                f"our comment {', '.join(skipped)} is live but says a replaced link or is the "
                "one we deleted; it is not adopted" + (f"; {said_wrong}" if said_wrong else "")
            )
        return said_wrong
    (same,) = groups.values()
    cid = min(same, key=_id_order)
    if not adopt_removed_first_comment(conn, row["id"], cid, same[cid], _now_iso()):
        return (
            f"our comment {cid} is live but the row cannot take it (a correction is pending or "
            "the row changed meanwhile)"
        )
    logger.info("first comment: delivery %s: our comment %s is live again; adopted", row["id"], cid)
    extras = [(c, t) for c, t in same.items() if c != cid] + list(replaced.items())
    if extras and queue_own_extra_deletes(conn, row["id"], extras):
        _note(
            row,
            result,
            f"our comment {cid} was adopted; our own extra copies "
            f"{', '.join(c for c, _ in extras)} are queued for deletion",
        )
        raise_stale_incident(conn, row["id"], incidents_dir)
    if said_wrong:
        _note(row, result, f"our comment {cid} was adopted; {said_wrong}")
    return ""


def recheck_own_id(
    conn: sqlite3.Connection,
    platforms: Iterable[str],
    client_for: Callable[[str], object],
    incidents_dir: Optional[Path],
    result: Any = None,
) -> None:
    """Close each open own-id incident whose row's recorded comment now reads back with our own
    id as its author (``client.read_comment`` / ``own_id``). A comment the read CONFIRMS gone
    (against the post's listing) is no longer ours to wait on: the incident is resolved with a
    note and the row goes the removed / repost way (``_route_gone``). A read that fails (an error
    line on ``result``) or another author leaves it open. Never raises."""
    from ..health.store import default_data_root, default_incidents_dir, load_all, write_row

    wanted = set(platforms)
    directory = incidents_dir or default_incidents_dir(default_data_root())
    try:
        rows, _ = load_all(directory)
    except OSError:
        logger.exception("first comment: could not read the own-id incidents")
        return
    for incident in rows:
        if (
            incident.get("kind") != OWN_ID_KIND
            or incident.get("resolved_at")
            or incident.get("platform") not in wanted
        ):
            continue
        row = conn.execute(
            "SELECT platform, platform_upload_id, platform_post_id, first_comment_live_id, "
            "first_comment_correction_attempts FROM deliveries WHERE id=?",
            (incident.get("delivery_id"),),
        ).fetchone()
        if row is None or not row["first_comment_live_id"]:
            continue
        try:
            client = client_for(row["platform"])
            said = client.read_comment(row["first_comment_live_id"], object_id=_target(row))
            if said is None:
                note = _route_gone(conn, incident, row, incidents_dir, result)
                if note:
                    incident["resolved_at"] = _now_iso()
                    incident["resolution"] = note
                    write_row(directory, incident)
                else:
                    _say_unroutable(incident, row, result, directory)
            elif getattr(said, "author", None) == client.own_id():
                incident["resolved_at"] = _now_iso()
                write_row(directory, incident)
        except Exception as exc:  # noqa: BLE001 -- it stays open; tried again next drain
            logger.warning(
                "first comment: own-id recheck of delivery %s failed: %s",
                incident.get("delivery_id"), exc,
            )  # fmt: skip
            if result is not None:
                result.errors.append(
                    f"delivery {incident.get('delivery_id')}: the own-id recheck could not read "
                    f"comment {row['first_comment_live_id']} ({str(exc).strip() or exc.__class__.__name__}); "
                    "the incident stays open and the next drain tries again"
                )


def _say_unroutable(incident: dict, row: Any, result: Any, directory: Path) -> None:
    """The comment reads gone but ``_route_gone`` could not move the row (a correction is
    pending, or the row changed meanwhile): say so, and keep the incident open with the reason in
    its detail. The next drain tries again."""
    from ..health.store import write_row

    live_id = row["first_comment_live_id"]
    if row["first_comment_correction_attempts"] >= FIRST_COMMENT_MAX_ATTEMPTS:
        why = (
            f"comment {live_id} reads gone but the row cannot be reposted: its pending "
            f"correction was refused {FIRST_COMMENT_MAX_ATTEMPTS} times and is capped (the "
            "capped incident names the hand step); the repost waits for that fix and the "
            "incident stays open"
        )
    else:
        why = (
            f"comment {live_id} reads gone but the row cannot be reposted yet (a correction is "
            "pending or the row changed meanwhile); the incident stays open and the next drain "
            "tries again"
        )
    logger.warning("first comment: delivery %s: %s", incident.get("delivery_id"), why)
    if result is not None:
        result.errors.append(f"delivery {incident.get('delivery_id')}: {why}")
    incident["detail"] = (
        f"{incident.get('platform')}#{incident.get('delivery_id')} "
        f"{incident.get('project') or ''}: {why}"
    )[:_DETAIL_MAX]
    write_row(directory, incident)


def _route_gone(
    conn: sqlite3.Connection, incident: dict, row: Any, incidents_dir: Optional[Path], result: Any
) -> str:
    """The row's recorded comment is confirmed gone: a row with text is unposted so the drain
    reposts it (``reopen_gone_first_comment`` counts the try, so a comment gone after every
    repost stops at ``FIRST_COMMENT_MAX_ATTEMPTS`` and stays surfaced); a row with no text is
    marked removed. Returns the note to resolve the own-id incident with, ``""`` when the row
    moved (a pending correction) and nothing was done."""
    delivery_id = incident.get("delivery_id")
    live_id = str(row["first_comment_live_id"])
    why = f"comment {live_id} reads gone (confirmed against the post's listing)"
    if reopen_gone_first_comment(conn, delivery_id, live_id, _now_iso(), why):
        raise_repost_incident(conn, delivery_id, GONE + why, incidents_dir)
        note = f"{why}; the row is unposted and the drain reposts it"
    elif mark_first_comment_removed(conn, delivery_id, _now_iso()):
        raise_repost_incident(conn, delivery_id, why, incidents_dir, kind=REMOVED_KIND)
        note = f"{why}; the row has no text, so it is marked removed"
    else:
        return ""
    if result is not None:
        result.errors.append(f"delivery {delivery_id}: {note}")
    return note


def _resolve_incidents(
    platforms: Iterable[str],
    still_pending: set,
    incidents_dir: Optional[Path],
    kind: str = KIND,
) -> None:
    """Resolve open ``kind`` incidents of ``platforms`` whose delivery is not in ``still_pending``."""
    from ..health.store import default_data_root, default_incidents_dir, load_all, write_row

    wanted = set(platforms)
    directory = incidents_dir or default_incidents_dir(default_data_root())
    try:
        rows, _ = load_all(directory)
        for row in rows:
            if (
                row.get("kind") == kind
                and not row.get("resolved_at")
                and row.get("platform") in wanted
                and row.get("delivery_id") not in still_pending
            ):
                row["resolved_at"] = _now_iso()
                write_row(directory, row)
    except OSError:
        logger.exception("first comment: could not resolve stale-comment incidents")


def run_corrections(
    conn: sqlite3.Connection,
    *,
    platforms: Iterable[str],
    client_for: Callable[[str], object],
    refused: Callable[[Exception], bool],
    result: Any,
    incidents_dir: Optional[Path] = None,
    gone: Optional[Callable[[Exception], bool]] = None,
) -> None:
    """Apply every pending correction through ``client_for(platform)`` (built lazily; has
    ``edit_comment(id, text, object_id=None)`` / ``delete_comment(id, object_id=None)``; ``object_id`` is the
    post the comment is on, for a platform that can only replace a comment). ``result`` gains
    ``corrected`` per fix and ``stale`` plus an error line per failure; ``refused(exc)`` counts
    toward the correction cap. After a row's first failure its other items wait for the next
    pass. ``gone(exc)`` is "comment not found": a delete is then done, an edit unposts the row
    (its deletes still run).
    A delete of the row's intended text is never run. A capped row is counted in
    ``result.correction_capped`` and gets the capped incident every pass."""
    platforms = tuple(platforms)
    restore_removed(conn, platforms, incidents_dir, client_for, result)
    settle_capped_reposts(conn, platforms, client_for, result, incidents_dir)
    _resolve_all(conn, platforms, incidents_dir)
    for row, reopened_only in _correction_passes(conn, platforms):
        keep = first_comment_intended_texts(row)
        unposted = False
        dropped: list = []  # reopened deletes dropped past their deadline: ONE incident per row
        for item in first_comment_corrections(row):
            if bool(item.get("reopened")) != reopened_only:
                continue  # a reopened delete runs in its own pass, ahead of and outside the cap
            if unposted and item.get("action") != "delete":
                continue  # the row is reposting: another edit would duplicate it
            if (
                item.get("action") == "delete"
                and not item.get("extra")
                and (item.get("live_text") or "").strip() in keep
            ):
                resolve_first_comment_correction(conn, row["id"], item, _now_iso())
                continue
            if (
                item.get("action") == "edit"
                and item.get("parent_at")
                and not parent_is_public_at(row["scheduled_at"], item["parent_at"])
            ):
                # The replacement is still private at this slot: an edit would link it. Delete.
                drop_private_edit(conn, row["id"], item, _now_iso())
                note = (
                    f"delivery {row['id']}: the replacement is not public by this short's slot; "
                    f"the edit of comment {item.get('live_id') or '?'} became a delete"
                )
                result.stale += 1
                result.errors.append(note)
                raise_stale_incident(conn, row["id"], incidents_dir)
                break
            try:
                client = client_for(row["platform"])
                call = getattr(client, f"{item.get('action')}_comment", None)
                if call is None or not item.get("live_id"):
                    raise CorrectionUnavailableError(
                        f"{row['platform']} cannot {item.get('action')} comment "
                        f"{item.get('live_id') or '(id unknown)'}"
                    )
                new_id = None
                if item["action"] == "edit":
                    new_id = call(item["live_id"], item["to"], object_id=_target(row))
                else:
                    if item.get("extra"):
                        # A second copy: re-read it. Only our own (its author is our account),
                        # still saying the copied text and not the row's recorded comment, is
                        # ever deleted; any other drop that is not a plain "it is gone" is said.
                        left = _extra_check(client, item, row)
                        if left is not True:
                            resolve_first_comment_correction(conn, row["id"], item, _now_iso())
                            if isinstance(left, _LagDropped):
                                dropped.append([item["live_id"], left.reason])
                            if left:
                                result.errors.append(
                                    f"delivery {row['id']}: second comment {item['live_id']} "
                                    f"not deleted: {left}"
                                )
                                logger.warning(
                                    "first comment: delivery %s: extra %s left alone: %s",
                                    row["id"], item["live_id"], left,
                                )  # fmt: skip
                            continue
                    elif _already_intended(client, item, keep):
                        # A stale observation: the comment has since been corrected.
                        resolve_first_comment_correction(conn, row["id"], item, _now_iso())
                        continue
                    call(item["live_id"], object_id=_target(row))
            except Exception as exc:  # noqa: BLE001 -- graded per correction
                if isinstance(exc, ReplacePostFailedError):
                    # The delete is confirmed; the row goes back to unposted with the new text
                    # and the normal posting path (attempts, alerts) owns the post.
                    # Zero comments are live now: the failure counts toward the attempt cap and
                    # an incident stays open until the comment is back.
                    unpost_first_comment_for_repost(
                        conn, row["id"], item, _now_iso(), count_attempt=True
                    )
                    result.errors.append(
                        f"delivery {row['id']}: stale comment deleted, the replacement post "
                        f"failed ({exc.cause}); the row is unposted and the drain posts it again"
                    )
                    raise_repost_incident(conn, row["id"], str(exc.cause), incidents_dir)
                    logger.warning("first comment: delivery %s: %s", row["id"], exc)
                    break
                if gone is not None and gone(exc):
                    if item.get("action") == "edit":
                        unpost_first_comment_for_repost(conn, row["id"], item, _now_iso())
                        unposted = True  # its deletes still run, or the repost scan stalls on them
                        continue
                    resolve_first_comment_correction(conn, row["id"], item, _now_iso())
                    result.corrected += 1
                    if item["action"] == "delete":
                        _note_removed(conn, row["id"], result, incidents_dir)
                    continue
                detail = str(exc).strip() or exc.__class__.__name__
                if isinstance(exc, _LagHeldError):
                    result.errors.append(f"delivery {row['id']}: {detail}")
                    logger.warning("first comment: delivery %s: %s", row["id"], detail)
                    continue  # waiting is not a failure: later items still run
                fail_first_comment_correction(
                    conn,
                    row["id"],
                    detail,
                    refused=refused(exc)
                    and not isinstance(exc, CorrectionUnavailableError)
                    and not reopened_only,  # outside the cap: never counted toward it
                )
                if reopened_only and reopen_lag_expired(item):
                    # Past its 24 hours a delete that still cannot run is dropped, never retried
                    # for ever: the read failed or the platform refuses it.
                    resolve_first_comment_correction(conn, row["id"], item, _now_iso())
                    reason = f"could not be deleted within the wait (last failure: {detail})"
                    dropped.append([item["live_id"], reason])
                    result.errors.append(
                        f"delivery {row['id']}: comment {item['live_id']}, queued for deletion "
                        f"when a gone read reopened the row, {reason}; the delete is dropped"
                    )
                    logger.warning("first comment: delivery %s: %s", row["id"], reason)
                    continue
                result.stale += 1
                result.errors.append(f"delivery {row['id']}: stale comment not corrected: {detail}")
                logger.warning(
                    "first comment: delivery %s correction failed: %s", row["id"], detail
                )
                raise_stale_incident(conn, row["id"], incidents_dir)
                if reopened_only:
                    continue  # a failing reopened delete never blocks another one
                break
            resolve_first_comment_correction(
                conn,
                row["id"],
                item,
                _now_iso(),
                new_live_id=new_id if isinstance(new_id, str) else None,
            )
            result.corrected += 1
            if item["action"] == "delete":
                _note_removed(conn, row["id"], result, incidents_dir)
        if dropped:
            raise_repost_incident(
                conn,
                row["id"],
                _lag_drop_cause(dropped),
                incidents_dir,
                kind=LAG_DROP_KIND,
                drops=dropped,
            )
    capped = capped_first_comment_corrections(conn, platforms)
    result.correction_capped += len(capped)
    for row in capped:
        raise_stale_incident(conn, row["id"], incidents_dir, capped=True)
    _resolve_all(conn, platforms, incidents_dir)


def _correction_passes(conn: sqlite3.Connection, platforms: tuple):
    """``(row, reopened_only)``: first every row holding a delete a gone reopen queued (any
    correction attempt count, only those deletes), then the rows under the correction cap (their
    other items). The second query runs after the first pass, so it sees its writes."""
    for row in reopened_delete_rows(conn, platforms):
        yield row, True
    for row in pending_first_comment_corrections(conn, platforms):
        yield row, False


def _already_intended(client: object, item: dict, keep: set) -> bool:
    """True when the comment a queued delete names now says an intended text (it was corrected
    after the delete was queued): deleting it would remove the row's only comment. Reads that one
    comment; a read that fails raises, so the delete stays pending and is never made blind."""
    read = getattr(client, "read_comment", None)
    if read is None:
        raise CorrectionUnavailableError("cannot confirm what the comment says before deleting it")
    said = read(item["live_id"])
    return said is not None and said.strip() in keep


def _extra_check(client: object, item: dict, row: Any):
    """Whether a queued second-copy delete may run: ``True`` to delete; ``False`` to drop it
    with nothing to say (not marked ours -- a viewer's comment is never queued, and a hand-edited
    or old item is dropped -- the row's own recorded comment, or the comment is confirmed gone);
    else the reason it is dropped with an error line (it now says other text, or its author is
    not positively our account). Reads that one comment through the client: a read that fails,
    a "gone" the post's listing does not confirm, or a client that cannot tell the author or
    our own id raises, so the delete stays pending and is never made blind."""
    if not item.get("ours") or item.get("live_id") == row["first_comment_live_id"]:
        return False
    read = getattr(client, "read_comment", None)
    own = getattr(client, "own_id", None)
    if read is None or own is None:
        raise CorrectionUnavailableError("cannot confirm what the comment says before deleting it")
    said = read(item["live_id"], object_id=_target(row))
    if said is None:
        if item.get("reopened"):
            return _held_gone(item)
        return False
    if said.strip() != (item.get("live_text") or "").strip():
        return "it now says other text, so it is left alone"
    author = getattr(said, "author", None)
    if not author or author != own():
        return "its author is not positively our account, so it is left alone"
    return True


class _LagDropped(str):
    """The reason a reopen-queued delete was dropped after its lag window (an incident is filed);
    ``reason`` is the part that does not name the comment, for the incident's grouped line."""

    reason: str = ""


class _LagHeldError(CorrectionUnavailableError):
    """A reopen-queued delete whose comment reads gone inside ``REOPEN_LAG_WINDOW``: waiting for
    the platform, not a failure (no attempt counted, no stale incident, later items still run)."""


def _held_gone(item: dict) -> str:
    """A delete the gone reopen queued reads gone. The read that said the old id was gone may
    have lagged (a post held back by the platform), so the delete stays pending, said every
    drain, until ``REOPEN_LAG_WINDOW`` after it was queued; past that it is dropped with a reason
    (``_LagDropped``), never silently."""
    hours = int(REOPEN_LAG_WINDOW.total_seconds() // 3600)
    if not reopen_lag_expired(item):
        raise _LagHeldError(
            f"comment {item['live_id']} reads gone, but it may only be lagging (our post held "
            f"back by the platform); the delete is kept and re-read on the next drain for up to "
            f"{hours} hours after the reopen"
        )
    dropped = _LagDropped(
        f"comment {item['live_id']}, queued for deletion when a gone read reopened the row, "
        f"still reads gone {hours} hours later; the delete is dropped"
    )
    dropped.reason = f"still reads gone {hours} hours after the reopen"
    return dropped


def settle_capped_reposts(
    conn: sqlite3.Connection,
    platforms: Iterable[str],
    client_for: Callable[[str], object],
    result: Any,
    incidents_dir: Optional[Path] = None,
) -> None:
    """A row that hit the repost cap on gone reads has deletes queued by the reopens, but the
    drain no longer posts or corrects an unposted row: when the platform's lag clears several
    copies of ours can be live and the row records none. List the post's own comments: the
    lowest id saying the row's text is adopted (the row is posted again, the repost incident
    closes) and every other copy is queued as our own extra delete, so exactly one stays live. No
    copy listed yet: the next drain looks again, until every reopen delete is past
    ``REOPEN_LAG_WINDOW``; then they are dropped with an error line and a ``LAG_DROP_KIND``
    incident, so the listing is never called forever. Never raises."""
    for row in capped_unposted_lag_rows(conn, platforms):
        held = [i for i in first_comment_corrections(row) if i.get("reopened")]
        expired = all(reopen_lag_expired(i) for i in held)
        failed = ""
        comments: list = []
        complete = True
        try:
            listing = getattr(client_for(row["platform"]), "own_comments", None)
            target = _target(row)
            if listing is None or not target:
                raise CorrectionUnavailableError("cannot list the post's comments")
            comments, complete = listing(target)
        except Exception as exc:  # noqa: BLE001 -- tried again next drain, or dropped at the deadline
            failed = str(exc).strip() or type(exc).__name__
            logger.warning("first comment: delivery %s: capped lag check: %s", row["id"], exc)
        if failed:
            if expired:
                _drop_held(
                    conn, row, result, incidents_dir,
                    lambda i: f"the last read of the post's comments failed ({failed}), so it "
                    "could not be told whether it is live",
                )  # fmt: skip
            else:
                result.errors.append(
                    f"delivery {row['id']}: the row is at the repost cap with lagging copies "
                    f"possibly live, and listing them failed ({failed}); the next drain tries again"
                )
            continue
        text = (row["first_comment_text"] or "").strip()
        copies = sorted(
            ((str(c), t) for c, t in comments if c and (t or "").strip() == text),
            key=lambda ct: _id_order(ct[0]),
        )
        if copies:
            keep, *rest = copies
            if adopt_capped_first_comment(
                conn, row["id"], keep[0], row["first_comment_text"], _now_iso()
            ):
                note = f"delivery {row['id']}: our comment {keep[0]} is live; the row records it"
                if rest and queue_own_extra_deletes(conn, row["id"], rest):
                    note += f"; our own extra copies {', '.join(c for c, _ in rest)} are queued for deletion"
                    raise_stale_incident(conn, row["id"], incidents_dir)
                result.errors.append(note)
                logger.warning("first comment: %s", note)
            continue
        # Every listed comment is positively ours (the listing is of our account's comments).
        listed = {str(c) for c, _ in comments if c}

        def why(item: dict, listed: set = listed, complete: bool = complete) -> str:
            if item["live_id"] in listed:
                return (
                    "it is listed but says other text than the row's, so it is left alone "
                    "(never deleted)"
                )
            if not complete:
                return (
                    "the post's listing was cut short, so it could not be told whether it is live"
                )
            return "it is not listed"

        if expired:
            _drop_held(conn, row, result, incidents_dir, why)
        else:
            for item in held:
                if item["live_id"] in listed:
                    result.errors.append(
                        f"delivery {row['id']}: comment {item['live_id']} is listed but says "
                        "other text than the row's; it is left alone and kept pending until the "
                        "wait ends"
                    )


def _drop_held(
    conn: sqlite3.Connection,
    row: Any,
    result: Any,
    incidents_dir: Optional[Path],
    why: Callable[[dict], str],
) -> None:
    """Every reopened delete of a capped row is past its deadline: drop them, an error line each
    and ONE ``LAG_DROP_KIND`` incident naming every dropped id and why."""
    dropped = drop_reopened_deletes(conn, row["id"])
    if not dropped:
        return
    pairs: list = []
    for item in dropped:
        reason = why(item)
        pairs.append([item["live_id"], reason])
        result.errors.append(
            f"delivery {row['id']}: comment {item['live_id']}, queued for deletion when a gone "
            f"read reopened the row, is dropped after the wait: {reason}"
        )
    raise_repost_incident(
        conn,
        row["id"],
        _lag_drop_cause(pairs),
        incidents_dir,
        kind=LAG_DROP_KIND,
        drops=pairs,
    )


def _resolve_all(conn: sqlite3.Connection, platforms: tuple, incidents_dir: Optional[Path]) -> None:
    """Stale incidents close once a row has no correction left or is capped; capped ones close
    once it is not."""
    capped = {r["id"] for r in capped_first_comment_corrections(conn, platforms)}
    pending = first_comment_correction_ids(conn, platforms) - capped
    _resolve_incidents(platforms, pending, incidents_dir)
    _resolve_incidents(platforms, capped, incidents_dir, CAPPED_KIND)
    _resolve_incidents(
        platforms, unposted_first_comment_ids(conn, platforms), incidents_dir, REPOST_KIND
    )
    _resolve_incidents(
        platforms, removed_first_comment_ids(conn, platforms), incidents_dir, REMOVED_KIND
    )
