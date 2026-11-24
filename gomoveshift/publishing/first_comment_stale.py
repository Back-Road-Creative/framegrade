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
    pending_first_comment_corrections,
    registered_platform_id,
    resolve_first_comment_correction,
    unpost_first_comment_for_repost,
)

logger = logging.getLogger(__name__)

KIND = "first_comment_stale"
CAPPED_KIND = "first_comment_correction_capped"
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
    ``edit_comment(id, text, object_id=None)`` / ``delete_comment(id)``; ``object_id`` is the
    post the comment is on, for a platform that can only replace a comment). ``result`` gains
    ``corrected`` per fix and ``stale`` plus an error line per failure; ``refused(exc)`` counts
    toward the correction cap. After a row's first failure its other items wait for the next
    pass. ``gone(exc)`` is "comment not found": a delete is then done, an edit unposts the row
    (its deletes still run).
    A delete of the row's intended text is never run. A capped row is counted in
    ``result.correction_capped`` and gets the capped incident every pass."""
    platforms = tuple(platforms)
    _resolve_all(conn, platforms, incidents_dir)
    for row in pending_first_comment_corrections(conn, platforms):
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
                if item["action"] == "edit":
                    call(item["live_id"], item["to"], object_id=_target(row))
                else:
                    if item.get("extra"):
                        # A second copy: re-read it. Only our own, still saying the copied text and
                        # not the row's recorded comment, is ever deleted.
                        if not _extra_is_deletable(client, item, row):
                            resolve_first_comment_correction(conn, row["id"], item, _now_iso())
                            continue
                    elif _already_intended(client, item, keep):
                        # A stale observation: the comment has since been corrected.
                        resolve_first_comment_correction(conn, row["id"], item, _now_iso())
                        continue
                    call(item["live_id"])
            except Exception as exc:  # noqa: BLE001 -- graded per correction
                if isinstance(exc, ReplacePostFailedError):
                    # The delete is confirmed; the row goes back to unposted with the new text
                    # and the normal posting path (attempts, alerts) owns the post.
                    unpost_first_comment_for_repost(conn, row["id"], item, _now_iso())
                    result.errors.append(
                        f"delivery {row['id']}: stale comment deleted, the replacement post "
                        f"failed ({exc.cause}); the row is unposted and the drain posts it again"
                    )
                    logger.warning("first comment: delivery %s: %s", row["id"], exc)
                    break
                if gone is not None and gone(exc):
                    if item.get("action") == "edit":
                        unpost_first_comment_for_repost(conn, row["id"], item, _now_iso())
                        unposted = True  # its deletes still run, or the repost scan stalls on them
                        continue
                    resolve_first_comment_correction(conn, row["id"], item, _now_iso())
                    result.corrected += 1
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
            resolve_first_comment_correction(conn, row["id"], item, _now_iso())
            result.corrected += 1
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


def _extra_is_deletable(client: object, item: dict, row: Any) -> bool:
    """True when a queued second-copy delete may run: the item is marked ours (a viewer's comment
    is never queued, and a hand-edited or old item without the mark is dropped), its id is not the
    comment the row recorded, and a fresh read still shows the text it copied. A read that fails
    raises, so the delete stays pending and is never made blind."""
    if not item.get("ours") or item.get("live_id") == row["first_comment_live_id"]:
        return False
    read = getattr(client, "read_comment", None)
    if read is None:
        raise CorrectionUnavailableError("cannot confirm what the comment says before deleting it")
    said = read(item["live_id"])
    return said is not None and said.strip() == (item.get("live_text") or "").strip()


def _resolve_all(conn: sqlite3.Connection, platforms: tuple, incidents_dir: Optional[Path]) -> None:
    """Stale incidents close once a row has no correction left or is capped; capped ones close
    once it is not."""
    capped = {r["id"] for r in capped_first_comment_corrections(conn, platforms)}
    pending = first_comment_correction_ids(conn, platforms) - capped
    _resolve_incidents(platforms, pending, incidents_dir)
    _resolve_incidents(platforms, capped, incidents_dir, CAPPED_KIND)
