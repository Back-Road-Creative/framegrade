"""The first comment every short carries: a link to its full drive.

One text function, one place -- enqueue and backfill both call
``full_drive_comment_text`` so the wording can never diverge. Also home to
the Graph comment request shared by the Instagram and Facebook adapters and
the permission error it maps to. TikTok's Content Posting API has no comment
endpoint, so TikTok rows never get a comment (see
``docs/social-publishing-queue.md``)."""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: Platforms whose shorts get a first comment linking the full drive.
COMMENT_PLATFORMS = ("youtube", "instagram", "facebook")

#: Pages of a post's comments (100 a page) read before posting, looking for
#: the comment already there. A short with more still gets its link: past
#: the cap the post goes ahead.
COMMENT_CHECK_MAX_PAGES = 10
_COMMENT_PAGE_SIZE = 100

#: Graph error codes meaning the token lacks comment permission: 10 is
#: "permission denied", 200-299 is the permissions range. Recoverable only by
#: a human granting the scope, never by retrying.
_GRAPH_PERMISSION_CODES = frozenset({10, *range(200, 300)})


class CommentPermissionError(Exception):
    """The token lacks the permission needed to comment (names it)."""

    def __init__(self, permission: str, detail: str = ""):
        self.permission = permission
        super().__init__(f"needs {permission}" + (f": {detail}" if detail else ""))


class CommentCheckError(RuntimeError):
    """The existing-comment check got a 2xx that is not a comment list, so
    posting could duplicate a comment the check failed to see."""


def full_drive_comment_text(parent_lf_video_id: str) -> str:
    """The first-comment text linking a short to its parent long-form video."""
    parent = (parent_lf_video_id or "").strip()
    if not parent:
        raise ValueError("parent_lf_video_id is empty; cannot link the full drive")
    return f"🎬 Watch the full drive: https://youtu.be/{parent}"


def _graph_body(response, permission: str, raise_for_error: Callable[[int, dict], None]) -> dict:
    """The JSON body of a Graph answer. A permission-class error raises
    ``CommentPermissionError`` naming ``permission``; any other error goes
    through the adapter's own ``raise_for_error`` classifier."""
    try:
        body = response.json()
    except ValueError:
        body = {}
    body = body or {}
    error = body.get("error")
    if response.status_code >= 400 or error:
        if isinstance(error, dict) and error.get("code") in _GRAPH_PERMISSION_CODES:
            raise CommentPermissionError(permission, str(error.get("message") or ""))
        raise_for_error(response.status_code, body)
    return body


def _existing_graph_comment(
    session,
    base: str,
    object_id: str,
    text: str,
    access_token: str,
    *,
    text_field: str,
    permission: str,
    raise_for_error: Callable[[int, dict], None],
) -> Optional[str]:
    """The id of a comment on ``object_id`` whose ``text_field`` is exactly
    ``text`` (whitespace-trimmed), else ``None``. Reads up to
    ``COMMENT_CHECK_MAX_PAGES`` pages. A failed read raises like a failed
    post: the caller must never post on a check that did not finish."""
    want = text.strip()
    # The token rides in a header, never in ``params``: ``requests`` puts
    # params into the URL, and a ConnectionError's message embeds that URL
    # and ends up in the row's ``first_comment_error``.
    headers = {"Authorization": f"OAuth {access_token}"}
    params = {"fields": f"id,{text_field}", "limit": _COMMENT_PAGE_SIZE}
    for _ in range(COMMENT_CHECK_MAX_PAGES):
        response = session.get(
            f"{base}/{object_id}/comments", params=params, headers=headers, timeout=30
        )
        body = _graph_body(response, permission, raise_for_error)
        if not isinstance(body.get("data"), list):
            # A 2xx that is not a comment list (an HTML error page, ``{}``)
            # is a check that did not finish, never "no comments yet".
            raise CommentCheckError(
                f"comment check for {object_id} got no comment list "
                f"(HTTP {response.status_code}); nothing posted"
            )
        for comment in body["data"]:
            if isinstance(comment, dict) and str(comment.get(text_field) or "").strip() == want:
                return str(comment.get("id") or "")
        paging = body.get("paging") or {}
        after = (paging.get("cursors") or {}).get("after")
        if not paging.get("next") or not after:
            return None
        params = {**params, "after": after}
    logger.warning(
        "first comment: %s has over %d pages of comments; posting without reading the rest",
        object_id,
        COMMENT_CHECK_MAX_PAGES,
    )
    return None


def post_graph_comment(
    session,
    base: str,
    object_id: str,
    text: str,
    access_token: str,
    *,
    text_field: str,
    permission: str,
    raise_for_error: Callable[[int, dict], None],
) -> str:
    """POST ``{base}/{object_id}/comments``; returns the new comment id.

    Never a second copy: it first reads the post's comments and, when one
    already carries exactly ``text`` (in ``text_field``: ``text`` on
    Instagram, ``message`` on Facebook), returns that comment's id without
    posting. So a comment Graph created whose answer was lost to a timeout
    is found on the next try, not posted again. A check that fails raises
    and posts nothing.

    A permission-class Graph error raises ``CommentPermissionError`` naming
    ``permission``; any other error goes through the adapter's own
    ``raise_for_error`` classifier."""
    existing = _existing_graph_comment(
        session,
        base,
        object_id,
        text,
        access_token,
        text_field=text_field,
        permission=permission,
        raise_for_error=raise_for_error,
    )
    if existing is not None:
        logger.info(
            "first comment: %s already carries it (comment %s); not posting it again",
            object_id,
            existing,
        )
        return existing
    response = session.post(
        f"{base}/{object_id}/comments",
        data={"message": text, "access_token": access_token},
        timeout=30,
    )
    body = _graph_body(response, permission, raise_for_error)
    return str(body["id"])
