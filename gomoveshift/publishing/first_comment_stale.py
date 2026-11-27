"""A first comment found live with a replaced text: the ``first_comment_stale`` health incident
(shown by ``status --json``) and the automatic in-place fix, run at the start of each drain."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .first_comment import ReplacePostFailedError, parent_is_public_at
from .queue import (
    FIRST_COMMENT_MAX_ATTEMPTS,
    capped_first_comment_corrections,
    drop_private_edit,
    fail_first_comment_correction,
    first_comment_correction_ids,
    first_comment_corrections,
    first_comment_intended_texts,
    is_auth_blocked,
    is_platform_paused,
    mark_first_comment_removed,
    pause_for_auth,
    pending_first_comment_corrections,
    registered_platform_id,
    removed_first_comment_ids,
    removed_first_comment_rows,
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
_DETAIL_MAX = 500


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


def raise_repost_incident(
    conn: sqlite3.Connection,
    delivery_id: int,
    cause: str,
    incidents_dir: Optional[Path] = None,
    kind: str = REPOST_KIND,
) -> None:
    """File (or refresh) the open incident for a row whose stale comment was deleted but whose
    replacement did not post: zero comments are live until the drain reposts. ``kind``
    ``REMOVED_KIND``: the comment was deleted and the row has no text to post (a private relink
    left none); the drain restores it by itself once the row has text. Never fatal."""
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
    if kind == REMOVED_KIND:
        detail = (
            f"{where}the comment was deleted and the row has no text to post ({cause}); no "
            "comment is live. The drain posts it by itself (or adopts yours) as soon as the row "
            "has text again; add it by hand only if no text will come"
        )[:_DETAIL_MAX]
    elif kind == OWN_ID_KIND:
        detail = (
            f"{where}the comment we just posted reads back with author {cause}, not our own id; "
            "the configured page_id / ig_user_id is wrong or Graph's from.id differs for this "
            "token (UNVERIFIED which). It is left alone, never deleted; until the id is right a "
            "lost answer can post a second copy. Check the platform config by hand"
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
            ),
        )
    except (OSError, ValueError):
        logger.exception("first comment: could not write the repost incident %s", delivery_id)


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
        "no comment is live. The drain restores it by itself once the row has text; add it by "
        "hand only if no text will come"
    )
    raise_repost_incident(conn, delivery_id, cause, incidents_dir, kind=REMOVED_KIND)


def restore_removed(
    conn: sqlite3.Connection, platforms: Iterable[str], incidents_dir: Optional[Path]
) -> None:
    """Put a removed row back on the drain: with its own text, or the one the plan gives it now
    (the backfill's source). The row is unposted, so the drain reposts it or adopts our own copy
    (a hand re-add) through its scan; a repost incident stays open until a new live id is
    recorded. No text: the row stays removed."""
    from . import enqueue_from_plan  # lazy: it imports the queue

    cache: dict = {}
    for row in removed_first_comment_rows(conn, platforms):
        text = (row["first_comment_text"] or "").strip() or enqueue_from_plan._backfill_text(
            row, cache
        )
        if text and unpost_removed_first_comment(conn, row["id"], text, _now_iso()):
            raise_repost_incident(
                conn, row["id"], "the comment text is back; the drain posts it", incidents_dir
            )


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
    auth_refused: Callable[[Exception], bool] = is_auth_blocked,
) -> None:
    """Apply every pending correction through ``client_for(platform)`` (built lazily; has
    ``edit_comment(id, text, object_id=None)`` / ``delete_comment(id, object_id=None)``; ``object_id`` is the
    post the comment is on, for a platform that can only replace a comment). ``result`` gains
    ``corrected`` per fix and ``stale`` plus an error line per failure; ``refused(exc)`` counts
    toward the correction cap. After a row's first failure its other items wait for the next
    pass. ``gone(exc)`` is "comment not found": a delete is then done, an edit unposts the row
    (its deletes still run).
    A delete of the row's intended text is never run. A capped row is counted in
    ``result.correction_capped`` and gets the capped incident every pass.
    ``auth_refused(exc)`` (default ``queue.is_auth_blocked``) is a refused access
    token: the platform is paused (``queue.pause_for_auth``) and its other rows wait."""
    platforms = tuple(platforms)
    restore_removed(conn, platforms, incidents_dir)
    _resolve_all(conn, platforms, incidents_dir)
    for row in pending_first_comment_corrections(conn, platforms):
        if is_platform_paused(conn, row["platform"]):
            continue  # paused mid-pass by a refused token below
        keep = first_comment_intended_texts(row)
        unposted = False
        for item in first_comment_corrections(row):
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
                if auth_refused(exc):
                    detail = pause_for_auth(conn, row["platform"], exc)
                    result.errors.append(
                        f"{row['platform']}: paused, access token refused: {detail}"
                    )
                    break
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
                fail_first_comment_correction(
                    conn,
                    row["id"],
                    detail,
                    refused=refused(exc) and not isinstance(exc, CorrectionUnavailableError),
                )
                result.stale += 1
                result.errors.append(f"delivery {row['id']}: stale comment not corrected: {detail}")
                logger.warning(
                    "first comment: delivery %s correction failed: %s", row["id"], detail
                )
                raise_stale_incident(conn, row["id"], incidents_dir)
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
    capped = capped_first_comment_corrections(conn, platforms)
    result.correction_capped += len(capped)
    for row in capped:
        raise_stale_incident(conn, row["id"], incidents_dir, capped=True)
    _resolve_all(conn, platforms, incidents_dir)


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
        return False
    if said.strip() != (item.get("live_text") or "").strip():
        return "it now says other text, so it is left alone"
    author = getattr(said, "author", None)
    if not author or author != own():
        return "its author is not positively our account, so it is left alone"
    return True


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
