"""Post the scheduled first comment on a delivered YouTube short.

The comment goes on the VIDEO id (``comment_target_id``), never the playlist id
that ``platform_post_id`` holds for YouTube.

``post_first_comment`` makes at most one ``commentThreads().insert`` call,
and none when the video already carries the exact text
(``_existing_first_comment``).
``drain_first_comments`` selects due rows via
``queue.select_due_first_comments`` and posts each -- see its docstring for
how a missing ``youtube.force-ssl`` scope is handled without ever raising
into the caller's drain loop.

``service`` is always injected (see ``service_factory`` below) -- this
module never builds credentials or an API client itself, so tests never
touch the network, mirroring ``cli.handlers.comments.run_comments_pull``'s
injectable-``service`` idiom."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, List, Optional, Tuple

from ..pipeline.drive_phases.youtube_uploader import QUOTA_REASONS, RETRIABLE_STATUS_CODES
from .queue import (
    first_comment_prior_texts,
    first_comment_still_current,
    record_stale_live_comment,
    select_due_first_comments,
)

logger = logging.getLogger(__name__)

#: HTTP 403 ``reason`` strings meaning the OAuth token lacks the
#: ``youtube.force-ssl`` scope (or comment-posting was never granted) --
#: recoverable only by a human re-authorising, never by retrying. Mirrors
#: ``youtube_uploader.QUOTA_REASONS``' shape for a different 403 class.
SCOPE_ERROR_REASONS = frozenset({"insufficientPermissions", "forbidden"})

_ERROR_DETAIL_MAX = 500

#: Pages of top-level comment threads ``post_first_comment`` reads before
#: posting (``_COMMENT_PAGE_SIZE`` each, 1 quota unit a page); past that many
#: it posts without reading the rest.
COMMENT_CHECK_MAX_PAGES = 10
_COMMENT_PAGE_SIZE = 100


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_first_comment(
    service, video_id: str, text: str, also_accept: Iterable[str] = ()
) -> Optional[Tuple[str, str]]:
    """``(thread id, the text it says)`` of a top-level comment thread on ``video_id`` that
    already says exactly ``text`` -- or exactly one of ``also_accept``, the texts a relink
    replaced (``queue.first_comment_prior_texts``) -- (stripped), else None -- so a comment
    YouTube created whose answer was lost (googleapiclient reads with a 60 s timeout) is
    found on the next pass, not posted again. The second item is ``text`` or the prior text
    that matched. Any error raises, so the caller posts nothing."""
    want = {text.strip(): text, **{t.strip(): t for t in also_accept}}
    request = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": _COMMENT_PAGE_SIZE,
        "textFormat": "plainText",
    }
    for _ in range(COMMENT_CHECK_MAX_PAGES):
        response = service.commentThreads().list(**request).execute()
        for thread in response.get("items") or []:
            top = (thread.get("snippet") or {}).get("topLevelComment") or {}
            said = top.get("snippet") or {}
            for field_name in ("textOriginal", "textDisplay"):
                heard = str(said.get(field_name) or "").strip()
                if heard in want:
                    return str(thread.get("id") or ""), want[heard]
        token = response.get("nextPageToken")
        if not token:
            return None
        request = {**request, "pageToken": token}
    logger.warning(
        "first comment: video %s has over %d pages of comments; posting without reading the rest",
        video_id,
        COMMENT_CHECK_MAX_PAGES,
    )
    return None


def _existing_first_comment(
    service, video_id: str, text: str, also_accept: Iterable[str] = ()
) -> Optional[str]:
    """The thread id ``_find_first_comment`` finds, else None."""
    found = _find_first_comment(service, video_id, text, also_accept)
    return found[0] if found else None


def post_first_comment_reporting(
    service, video_id: str, text: str, also_accept: Iterable[str] = ()
) -> Tuple[str, str]:
    """``(thread id, the text that is live)``: POSTs one top-level comment on ``video_id`` -- or,
    when the video already carries exactly ``text`` (``_find_first_comment``) or one of
    ``also_accept``, finds that thread and posts nothing. The live text differs from ``text``
    only when a replaced (prior) text was found live."""
    found = _find_first_comment(service, video_id, text, also_accept)
    if found is not None:
        logger.info(
            "first comment: video %s already carries it (thread %s); not posting it again",
            video_id,
            found[0],
        )
        return found
    body = {
        "snippet": {
            "videoId": video_id,
            "topLevelComment": {"snippet": {"textOriginal": text}},
        }
    }
    response = service.commentThreads().insert(part="snippet", body=body).execute()
    return response["id"], text


def post_first_comment(service, video_id: str, text: str, also_accept: Iterable[str] = ()) -> str:
    """POST one top-level comment on ``video_id``; returns the new comment
    thread id -- or, when the video already carries exactly ``text``
    (``_existing_first_comment``) or one of ``also_accept``, that thread's id, posting
    nothing.

    ``service`` is an already-authenticated ``googleapiclient`` YouTube
    Data API v3 client -- this function performs no auth of its own."""
    return post_first_comment_reporting(service, video_id, text, also_accept)[0]


def _is_scope_error(exc: Exception) -> bool:
    """True iff ``exc`` is an ``HttpError`` 403 caused by the token's scope
    (``SCOPE_ERROR_REASONS``) -- any other status, or a 403 for an
    unrelated reason (e.g. ``commentsDisabled``), is a genuine per-row
    failure, not a re-auth problem."""
    from googleapiclient.errors import HttpError

    if not isinstance(exc, HttpError):
        return False
    if getattr(exc.resp, "status", None) != 403:
        return False
    details = getattr(exc, "error_details", None) or []
    return any(isinstance(d, dict) and d.get("reason") in SCOPE_ERROR_REASONS for d in details)


def _refused_for_the_video(exc: Exception) -> bool:
    """True iff YouTube refused this comment because of its video -- a 4xx
    ``HttpError`` such as 404 ``videoNotFound`` (the short was taken down) or
    403 ``commentsDisabled``. Only these count toward
    ``queue.FIRST_COMMENT_MAX_ATTEMPTS``. A 5xx or 429, spent quota or a
    rate limit (``QUOTA_REASONS``), a lapsed token (401), a missing scope or
    a network failure says nothing about the video."""
    from googleapiclient.errors import HttpError

    if not isinstance(exc, HttpError) or _is_scope_error(exc):
        return False
    status = getattr(exc.resp, "status", None)
    if not isinstance(status, int) or not 400 <= status < 500:
        return False
    if status == 401 or status in RETRIABLE_STATUS_CODES:
        return False
    details = getattr(exc, "error_details", None) or []
    return not any(isinstance(d, dict) and d.get("reason") in QUOTA_REASONS for d in details)


@dataclass
class FirstCommentResult:
    """One drain pass's outcome. Mirrors ``dispatch.DrainResult``'s counter
    shape: a plain summary line, never a per-row breakdown here."""

    posted: int = 0
    failed: int = 0
    #: Rows held for a scope/permission error -- not attempted again until
    #: the operator re-authorises; distinct from ``failed``, which is a per-row
    #: problem a retry might resolve on its own.
    skipped_scope_error: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def needs_reauth(self) -> bool:
        return self.skipped_scope_error > 0


def drain_first_comments(
    conn: sqlite3.Connection,
    *,
    service_factory: Callable[[], object],
    now: Optional[str] = None,
) -> FirstCommentResult:
    """Post every due first comment (``queue.select_due_first_comments``).

    ``service_factory`` is called at most once, lazily -- only once a row
    is actually due -- so an idle pass with nothing due never builds a
    YouTube client or pays the SOPS-decrypt cost.

    A scope/permission 403 (``_is_scope_error``) is recorded on the row's
    ``first_comment_error`` and logged ONCE for the whole pass ("needs
    youtube.force-ssl re-auth"), never raised: it means a human must
    re-authorise the token, not that this delivery failed, and the row is
    left otherwise untouched (``first_comment_posted_at`` stays NULL) so
    the very next pass retries it once that happens. Any other exception
    is recorded and counted as ``failed``, per row; one where YouTube
    refused the comment because of its video (``_refused_for_the_video``)
    also adds one to ``first_comment_attempts``, and after
    ``queue.FIRST_COMMENT_MAX_ATTEMPTS`` of those the row is never tried
    again, while an outage never counts; the pass continues
    either way -- one bad delivery must never abort the rest, same
    contract as ``dispatch.drain_queue``."""
    result = FirstCommentResult()
    due = select_due_first_comments(conn, now, platforms=("youtube",))
    if not due:
        return result
    service = service_factory()
    logged_scope_error = False
    for row in due:
        # A relink may have rewritten the text after selection: post what the row says NOW.
        fresh = first_comment_still_current(conn, row["id"])
        if fresh is None:
            continue
        text = fresh["first_comment_text"]
        try:
            _, live_text = post_first_comment_reporting(
                service,
                row["comment_target_id"],
                text,
                also_accept=first_comment_prior_texts(fresh),
            )
        except Exception as exc:  # noqa: BLE001 -- graded per-row below
            detail = str(exc).strip() or exc.__class__.__name__
            refused = _refused_for_the_video(exc)
            with conn:
                conn.execute(
                    "UPDATE deliveries SET first_comment_error=?, "
                    "first_comment_attempts=first_comment_attempts+? WHERE id=?",
                    (detail[:_ERROR_DETAIL_MAX], int(refused), row["id"]),
                )
            if _is_scope_error(exc):
                result.skipped_scope_error += 1
                if not logged_scope_error:
                    logger.warning(
                        "drain_first_comments: needs youtube.force-ssl re-auth (%s)", detail
                    )
                    logged_scope_error = True
            else:
                result.failed += 1
                result.errors.append(f"delivery {row['id']}: {detail}")
            continue
        if live_text.strip() != text.strip():  # a replaced text was already live: report it
            record_live_comment(conn, row["id"], live_text, result)
        else:
            mark_posted(conn, row["id"], text, result)
    return result


def record_live_comment(
    conn: sqlite3.Connection, delivery_id: int, live_text: str, result: FirstCommentResult
) -> None:
    """A comment saying ``live_text`` is live but the row now holds other text (a relink
    rewrote it): store the live text, mark the row posted, set ``first_comment_error`` and
    add it to ``result.errors`` (``queue.record_stale_live_comment``). Never silent."""
    error = record_stale_live_comment(conn, delivery_id, live_text, _now_iso())
    if error is None:
        logger.warning("first comment: delivery %s changed under the drain; next pass", delivery_id)
        return
    result.posted += 1
    result.errors.append(f"delivery {delivery_id}: {error}")
    logger.warning("first comment: delivery %s: %s", delivery_id, error)


def mark_posted(
    conn: sqlite3.Connection, delivery_id: int, posted_text: str, result: FirstCommentResult
) -> None:
    """Mark ``delivery_id`` posted iff its text is still ``posted_text``. When a relink
    rewrote it during the post, the comment that went live is the OLD text:
    :func:`record_live_comment` records and reports that instead."""
    with conn:
        marked = conn.execute(
            "UPDATE deliveries SET first_comment_posted_at=?, first_comment_error=NULL "
            "WHERE id=? AND first_comment_text=?",
            (_now_iso(), delivery_id, posted_text),
        ).rowcount
    if marked:
        result.posted += 1
    else:
        record_live_comment(conn, delivery_id, posted_text, result)
