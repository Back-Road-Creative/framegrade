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
from typing import Callable, Mapping, Optional

from .facebook_adapter import FacebookPermanentError
from .first_comment import CommentPermissionError
from .instagram_adapter import InstagramPermanentError
from .queue import select_due_first_comments
from .youtube_comments import _ERROR_DETAIL_MAX, FirstCommentResult, _now_iso

logger = logging.getLogger(__name__)

#: What each adapter's Graph classifier raises when Graph refused the request
#: itself (a deleted reel is Graph code 100) -- the only failure that counts
#: toward ``queue.FIRST_COMMENT_MAX_ATTEMPTS``. A rate limit or 5xx
#: (``*RetryableError``), a lapsed token (``*AuthBlockedError``), a missing
#: permission or a network failure says nothing about the post.
_REFUSED_FOR_THE_POST = (FacebookPermanentError, InstagramPermanentError)


def drain_meta_first_comments(
    conn: sqlite3.Connection,
    *,
    adapter_factories: Mapping[str, Callable[[], object]],
    now: Optional[str] = None,
) -> FirstCommentResult:
    """Post every due IG/FB first comment for the platforms in
    ``adapter_factories`` (platform -> zero-arg factory returning an object
    with ``post_comment(object_id, text)``)."""
    result = FirstCommentResult()
    due = select_due_first_comments(conn, now, platforms=tuple(adapter_factories))
    adapters: dict = {}
    warned: set = set()
    for row in due:
        platform = row["platform"]
        try:
            if platform not in adapters:
                adapters[platform] = adapter_factories[platform]()
            adapters[platform].post_comment(row["comment_target_id"], row["first_comment_text"])
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
        result.posted += 1
        with conn:
            conn.execute(
                "UPDATE deliveries SET first_comment_posted_at=?, first_comment_error=NULL "
                "WHERE id=?",
                (_now_iso(), row["id"]),
            )
    return result
