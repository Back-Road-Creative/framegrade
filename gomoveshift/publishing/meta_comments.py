"""Post the scheduled first comment on delivered Instagram / Facebook shorts.

Same contract as ``youtube_comments.drain_first_comments``: rows come from
``queue.select_due_first_comments`` (paused platforms already excluded), a
Graph permission error is recorded on ``first_comment_error``, logged ONCE per
platform per pass naming the permission, never raised, and the row stays
retryable; any other error is recorded and counted ``failed`` per row. A Graph
refusal of the comment itself (``_REFUSED_FOR_THE_POST``) also adds one to
``first_comment_attempts`` (``queue.FIRST_COMMENT_MAX_ATTEMPTS`` ends it); an
outage never does. One bad row never aborts the pass. Adapters are built
lazily, only if a row is due."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Callable, Mapping, Optional

from .facebook_adapter import FacebookPermanentError
from .first_comment import (
    CommentPermissionError,
    ForeignCommentId,
    PostedId,
    PriorCommentLiveError,
)
from .first_comment_stale import (
    OWN_ID_KIND,
    UNVERIFIED,
    raise_repost_incident,
    recheck_own_id,
    run_corrections,
)
from .instagram_adapter import InstagramPermanentError
from .queue import (
    first_comment_prior_texts,
    first_comment_still_current,
    select_due_first_comments,
)
from .youtube_comments import (
    _ERROR_DETAIL_MAX,
    FirstCommentResult,
    record_found_comments,
)

logger = logging.getLogger(__name__)

#: What each adapter's Graph classifier raises when Graph refused the request
#: itself (a deleted reel is Graph code 100) -- the only failure that counts
#: toward ``queue.FIRST_COMMENT_MAX_ATTEMPTS``. A rate limit or 5xx
#: (``*RetryableError``), a lapsed token (``*AuthBlockedError``), a missing
#: permission or a network failure says nothing about the post.
_REFUSED_FOR_THE_POST = (FacebookPermanentError, InstagramPermanentError)


def _author_mismatch(adapter: object, posted_id: object, object_id: str = "") -> Optional[str]:
    """For a comment this drain just posted: ``None`` when it reads back with our own account id
    as its author (or it was not posted by this call), else why it is not confirmed: who Graph
    says wrote it (``"no author"`` when it names none), or ``UNVERIFIED`` plus the reason when the
    read-back failed or answered gone (confirmed against ``object_id``'s listing). The comment
    is live either way and is never touched."""
    if type(posted_id) is not PostedId:
        return None  # found already there, so listed with its author: not a fresh post
    try:
        said = adapter.read_comment(str(posted_id), object_id=object_id or None)
        own = adapter.own_id()
    except Exception as exc:  # noqa: BLE001 -- best effort: the post already landed
        logger.warning("first comment: could not read back comment %s: %s", posted_id, exc)
        return f"{UNVERIFIED}the read-back failed: {str(exc).strip() or exc.__class__.__name__}"
    if said is None:
        return f"{UNVERIFIED}the read-back answered that the comment is gone"
    author = getattr(said, "author", None)
    return None if author == own else (author or "no author")


def drain_meta_first_comments(
    conn: sqlite3.Connection,
    *,
    adapter_factories: Mapping[str, Callable[[], object]],
    now: Optional[str] = None,
    incidents_dir: Optional[Path] = None,
) -> FirstCommentResult:
    """Post every due IG/FB first comment for the platforms in
    ``adapter_factories`` (platform -> zero-arg factory returning an object
    with ``post_comment(object_id, text)``). A replaced text found live is counted
    and filed as a ``first_comment_stale`` incident; the next drain corrects it first
    (``first_comment_stale.run_corrections``: Facebook edits the comment, Instagram deletes it and
    posts the intended text), so the adapters also carry ``edit_comment`` / ``delete_comment``."""
    result = FirstCommentResult()
    adapters: dict = {}
    warned: set = set()

    def adapter_for(platform: str):
        if platform not in adapters:
            adapters[platform] = adapter_factories[platform]()
        return adapters[platform]

    try:
        run_corrections(
            conn,
            platforms=tuple(adapter_factories),
            client_for=adapter_for,
            refused=lambda exc: isinstance(exc, _REFUSED_FOR_THE_POST),
            result=result,
            incidents_dir=incidents_dir,
        )
    except Exception as exc:  # noqa: BLE001 -- a correction pass failing never stops the posting
        detail = str(exc).strip() or exc.__class__.__name__
        logger.exception("drain_meta_first_comments: the correction pass failed")
        result.errors.append(f"stale-comment corrections failed: {detail}")

    try:
        recheck_own_id(conn, tuple(adapter_factories), adapter_for, incidents_dir)
    except Exception:  # noqa: BLE001 -- a recheck failing never stops the posting
        logger.exception("drain_meta_first_comments: the own-id recheck failed")

    due = select_due_first_comments(conn, now, platforms=tuple(adapter_factories))
    for row in due:
        platform = row["platform"]
        # A relink may have rewritten the text after selection: post what the row says NOW.
        fresh = first_comment_still_current(conn, row["id"])
        if fresh is None:
            continue
        text = fresh["first_comment_text"]
        try:
            prior = first_comment_prior_texts(fresh)
            # Only a relinked row has prior texts; every other row keeps the two-argument call.
            extra = {"also_accept": prior} if prior else {}
            posted_id = adapter_for(platform).post_comment(row["comment_target_id"], text, **extra)
        except PriorCommentLiveError as live:
            # A replaced text is already live: never post beside it.
            record_found_comments(
                conn,
                row["id"],
                text,
                live.comment_id,
                live.said,
                live.duplicates,
                result,
                incidents_dir,
            )
            continue
        except Exception as exc:  # noqa: BLE001 -- graded per-row below
            detail = str(exc).strip() or exc.__class__.__name__
            refused = isinstance(exc, _REFUSED_FOR_THE_POST)
            with conn:
                conn.execute(
                    "UPDATE deliveries SET first_comment_error=?, "
                    "first_comment_attempts=first_comment_attempts+? WHERE id=?",
                    (detail[:_ERROR_DETAIL_MAX], int(refused), row["id"]),
                )
            if isinstance(exc, CommentPermissionError):
                result.skipped_scope_error += 1
                if platform not in warned:
                    logger.warning(
                        "drain_meta_first_comments: %s needs %s (%s)",
                        platform,
                        exc.permission,
                        detail,
                    )
                    warned.add(platform)
            else:
                result.failed += 1
                result.errors.append(f"delivery {row['id']}: {detail}")
            continue
        mismatch = _author_mismatch(adapter_for(platform), posted_id, row["comment_target_id"])
        if mismatch:
            result.errors.append(
                f"delivery {row['id']}: comment {posted_id} posted, but "
                + (
                    f"its author could not be verified ({mismatch[len(UNVERIFIED) :]})"
                    if mismatch.startswith(UNVERIFIED)
                    else f"it reads back with author {mismatch}, not our own id"
                )
                + "; check the platform's own id config"
            )
            logger.warning("first comment: delivery %s: own id mismatch: %s", row["id"], mismatch)
            raise_repost_incident(conn, row["id"], str(mismatch), incidents_dir, kind=OWN_ID_KIND)
        record_found_comments(
            conn,
            row["id"],
            text,
            posted_id if isinstance(posted_id, ForeignCommentId) else str(posted_id or ""),
            text,
            (),
            result,
            incidents_dir,
        )
    return result
