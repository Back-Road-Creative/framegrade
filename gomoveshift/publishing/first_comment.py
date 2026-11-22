"""The first comment every short carries: a link to its full drive.

One text function, one place -- enqueue and backfill both call
``full_drive_comment_text`` so the wording can never diverge. Also home to
the Graph comment request shared by the Instagram and Facebook adapters and
the permission error it maps to. TikTok's Content Posting API has no comment
endpoint, so TikTok rows never get a comment (see
``docs/social-publishing-queue.md``)."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Callable, Iterable, List, Optional, Tuple

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


def _aware(value) -> Optional[datetime]:
    """``value`` (a datetime or ISO string) as a zoned datetime, else ``None``."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime) or value.tzinfo is None:
        return None
    return value


def parent_is_public_at(short_at, parent_at) -> bool:
    """True when the parent drive is public by the time the Short is.

    The one rule for every link from a Short to its full drive: the reviewed
    description block, the plan's description and each first comment. A
    scheduled parent stays private until its publish time, so a Short that goes
    out earlier would link a video nobody can watch. An unknown or unzoned time
    cannot be ordered, so it never links."""
    short, parent = _aware(short_at), _aware(parent_at)
    return short is not None and parent is not None and short >= parent


class PriorCommentLiveError(Exception):
    """A text a relink replaced is already live on the post (``comment_id``, saying ``said``;
    ``said`` is the new text when that one is live too). Nothing was posted: posting the new
    text beside it would leave two comments, one linking a deleted video. ``duplicates`` are the
    further replaced-text comments (id, said) beside it. The drain records ``said`` as the live
    text, queues a delete for each duplicate and reports it all."""

    def __init__(self, comment_id: str, said: str, duplicates: Iterable[Tuple[str, str]] = ()):
        super().__init__(f"a replaced first comment is already live (comment {comment_id})")
        self.comment_id = comment_id
        self.said = said
        self.duplicates = tuple(duplicates)


class ForeignCommentId(str):
    """A comment id that is NOT positively ours (its author is another account, or unknown): a
    viewer's comment saying our text. It may satisfy a "is it already there" scan but is never
    ours: never marked as the row's posted comment, never an edit, delete or adoption target. ``unknown_author`` is set when it had no author."""

    unknown_author = False


class AuthoredText(str):
    """The text a single-comment read answered, with its author's id (``None`` when the read gave
    none): a deletion that must be ours checks ``author`` against our own account id."""

    author: Optional[str] = None


class PostedId(str):
    """The id of a comment this call just posted (not one found already there): the drain reads
    it back once and reports an author other than our own account id."""


class CommentCorrectionError(RuntimeError):
    """Graph answered a comment edit/delete with a 2xx that is not ``{"success": true}``."""


class ReplacePostFailedError(Exception):
    """The stale comment is gone (its DELETE was confirmed, or a read showed it already gone) and
    posting the replacement failed with ``cause``. The caller must record the delete before
    anything else: the row goes back to unposted with the new text, so the normal posting path
    (its attempts, its alerts) owns the post and the row never claims a comment that is not
    there."""

    def __init__(self, cause: BaseException):
        super().__init__(f"stale comment deleted, replacement not posted: {cause}")
        self.cause = cause


def full_drive_comment_text(parent_lf_video_id: str) -> str:
    """The first-comment text linking a short to its parent long-form video."""
    parent = (parent_lf_video_id or "").strip()
    if not parent:
        raise ValueError("parent_lf_video_id is empty; cannot link the full drive")
    return f"🎬 Watch the full drive: https://youtu.be/{parent}"


_FULL_DRIVE_SHAPE = re.compile(r"\A\W*Watch the full drive: https://youtu\.be/[\w-]+\Z")


def is_full_drive_comment(text: str) -> bool:
    """True when ``text`` has the shape ``full_drive_comment_text`` writes (an optional leading
    emoji, then ``Watch the full drive: https://youtu.be/<id>``): the only comment a removed row
    may adopt as its own."""
    return bool(_FULL_DRIVE_SHAPE.match((text or "").strip()))


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


def _author_id(comment: dict) -> str:
    author = comment.get("from")
    return str(author.get("id") or "") if isinstance(author, dict) else ""


def _own_id_unknown(object_id: str) -> CommentCheckError:
    return CommentCheckError(f"comment check for {object_id}: our own account id is unknown")


def _read_graph_comments(
    session,
    base: str,
    object_id: str,
    access_token: str,
    *,
    text_field: str,
    permission: str,
    raise_for_error: Callable[[int, dict], None],
    own_id: str = "",
    stop: Optional[Callable[[List[Tuple[str, str, bool]]], bool]] = None,
) -> Tuple[List[Tuple[str, str, bool]], bool]:
    """``(comments, complete)``: ``(id, text, ours)`` of the comments on ``object_id`` (text
    stripped); ``ours`` only when ``from.id`` is ``own_id``, the Page or IG user. Every other
    comment (someone else's, no ``from``, or no ``own_id``) is a viewer's: its id is a
    ``ForeignCommentId``. Reads up to ``COMMENT_CHECK_MAX_PAGES``
    pages, or fewer once ``stop(comments)`` holds.
    ``complete`` is ``True`` only when the last page was reached. A failed read raises like a
    failed post: the caller must never post on a check that did not finish."""
    # The token rides in a header, never in ``params``: ``requests`` puts
    # params into the URL, and a ConnectionError's message embeds that URL
    # and ends up in the row's ``first_comment_error``.
    headers = {"Authorization": f"OAuth {access_token}"}
    params = {"fields": f"id,{text_field},from", "limit": _COMMENT_PAGE_SIZE}
    found: List[Tuple[str, str, bool]] = []
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
            if isinstance(comment, dict):
                said = str(comment.get(text_field) or "").strip()
                author = _author_id(comment)
                ours = bool(own_id) and author == own_id
                cid = str(comment.get("id") or "")
                if not ours:
                    cid = ForeignCommentId(cid)  # not positively ours: never ours to touch
                    cid.unknown_author = not author
                found.append((cid, said, ours))
        paging = body.get("paging") or {}
        after = (paging.get("cursors") or {}).get("after")
        if not paging.get("next") or not after:
            return found, True
        if stop is not None and stop(found):
            return found, False
        params = {**params, "after": after}
    logger.warning(
        "first comment: %s has over %d pages of comments; not reading the rest",
        object_id,
        COMMENT_CHECK_MAX_PAGES,
    )
    return found, False


def list_own_graph_comments(
    session,
    base: str,
    object_id: str,
    access_token: str,
    *,
    text_field: str,
    permission: str,
    raise_for_error: Callable[[int, dict], None],
    own_id: str = "",
) -> Tuple[List[Tuple[str, str]], bool]:
    """``(comments, complete)``: ``(id, text)`` of each comment on ``object_id`` whose ``from.id``
    is ``own_id``, and whether the listing reached its last page. No ``own_id`` or a failed
    read raises: whose a comment is must be known."""
    if not own_id:
        raise _own_id_unknown(object_id)
    found, complete = _read_graph_comments(
        session, base, object_id, access_token,
        text_field=text_field, permission=permission, raise_for_error=raise_for_error,
        own_id=own_id,
    )  # fmt: skip
    return [(c, s) for c, s, ours in found if ours], complete


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
    also_accept: Iterable[str] = (),
    own_id: str = "",
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """``(current, prior)``: ``(comment id, the text it says)`` of every comment on
    ``object_id`` whose ``text_field`` is exactly ``text`` (``current``, whoever wrote it, ours
    first), or OUR comment saying one of ``also_accept``, the texts a relink replaced (``prior``;
    only these may be edited or deleted) -- whitespace-trimmed. Without ``also_accept`` the scan
    stops at the first match; with it every bounded page is read, and a replaced text with no
    ``own_id`` to attribute it raises ``CommentCheckError``. A failed read raises."""
    wanted = {text.strip(): text}
    prior_texts = {t.strip(): t for t in also_accept}

    def done(found):
        return not prior_texts and any(
            said in wanted and not isinstance(cid, ForeignCommentId) for cid, said, _ in found
        )

    comments, _ = _read_graph_comments(
        session,
        base,
        object_id,
        access_token,
        text_field=text_field,
        permission=permission,
        raise_for_error=raise_for_error,
        own_id=own_id,
        stop=done,
    )
    if not own_id and any(said in prior_texts and said not in wanted for _, said, _ in comments):
        raise _own_id_unknown(object_id)
    hits = [(c, wanted[s], o) for c, s, o in comments if s in wanted]
    if any(getattr(c, "unknown_author", False) for c, _, _ in hits):
        logger.warning(
            "first comment: %s has a comment with no author saying our text; treated as a "
            "viewer's, never edited or deleted (unverified when Graph omits `from`)",
            object_id,
        )
    # Ours first; a stranger's id is flagged and never stands in for ours.
    current = [(c, s) for c, s, o in hits if o] + [(c, s) for c, s, o in hits if not o]
    prior = [
        (cid, prior_texts[said])
        for cid, said, ours in comments
        if ours and said in prior_texts and said not in wanted
    ]
    return current, prior


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
    also_accept: Iterable[str] = (),
    own_id: str = "",
    post_beside_foreign: bool = False,
    report_duplicates: bool = False,
) -> str:
    """POST ``{base}/{object_id}/comments``; returns the new comment id (a :class:`PostedId` when
    this call posted it).

    Never a second copy: it first reads the post's comments and, when one
    already carries exactly ``text`` (in ``text_field``: ``text`` on
    Instagram, ``message`` on Facebook), returns that comment's id without
    posting (unless ``post_beside_foreign``: a stranger's copy then never counts, ours is posted
    beside it; a stranger's match otherwise comes back as a ``ForeignCommentId``). So a comment Graph created whose answer was lost to a timeout
    is found on the next try, not posted again. A check that fails raises
    and posts nothing. When a comment found is one of ``also_accept`` (a text a relink
    replaced) it raises ``PriorCommentLiveError`` (naming the new text's comment when that is
    live too, else the replaced one, plus any further replaced ones) and posts nothing. With
    ``report_duplicates`` (the drain) the same error reports our second and later copies of
    ``text`` as ``duplicates``, so each gets a delete and one stays.

    A permission-class Graph error raises ``CommentPermissionError`` naming
    ``permission``; any other error goes through the adapter's own
    ``raise_for_error`` classifier."""
    current, prior = _existing_graph_comment(
        session,
        base,
        object_id,
        text,
        access_token,
        text_field=text_field,
        permission=permission,
        raise_for_error=raise_for_error,
        also_accept=also_accept,
        own_id=own_id,
    )
    ours = [c for c in current if not isinstance(c[0], ForeignCommentId)]
    if post_beside_foreign:
        current = ours  # a viewer's copy never stands in for ours: post ours beside it
    if prior or (report_duplicates and len(ours) > 1):
        # OUR comment saying the new text is the one to keep; with none, the first replaced one
        # stands in for it. Our further copies and replaced ones are duplicates.
        first = ours[0] if ours else prior[0]
        dupes = [*ours[1:], *prior] if ours else prior[1:]
        raise PriorCommentLiveError(first[0], first[1], dupes)
    if current:
        logger.info(
            "first comment: %s already carries it (comment %s); not posting it again",
            object_id,
            current[0][0],
        )
        return current[0][0]
    response = session.post(
        f"{base}/{object_id}/comments",
        data={"message": text, "access_token": access_token},
        timeout=30,
    )
    body = _graph_body(response, permission, raise_for_error)
    return PostedId(str(body["id"]))


#: Graph's answer to a DELETE of a comment that no longer exists: code 100, subcode 33 ("Object
#: does not exist"). UNVERIFIED against Meta's docs here (no code or doc in this repo names it);
#: it is the pair the reviewer's scenario reports. A wrong pair only means the old behaviour: the
#: 400 is an ordinary refusal, counted and capped.
_GRAPH_GONE = (100, 33)


def _is_gone(response) -> bool:
    try:
        error = (response.json() or {}).get("error")
    except ValueError:
        return False
    return (
        response.status_code >= 400
        and isinstance(error, dict)
        and (error.get("code"), error.get("error_subcode")) == _GRAPH_GONE
    )


def _confirm_gone(
    session, base, comment_id, object_id, access_token, *, text_field, permission, raise_for_error,
    what,
) -> None:  # fmt: skip
    """A "gone" answer (code 100 / subcode 33, whose generic message UNVERIFIED also covers a
    comment that cannot be loaded for missing permissions) is trusted only once the listing of
    ``object_id``'s comments no longer shows ``comment_id``. One still listed, a listing that
    fails or is cut short, or no ``object_id`` to list, raises: nothing is resolved on the
    answer alone."""
    if not object_id:
        raise CommentCheckError(f"{what}: answered gone; no post to list, so not confirmed")
    comments, complete = _read_graph_comments(
        session, base, object_id, access_token,
        text_field=text_field, permission=permission, raise_for_error=raise_for_error,
        stop=lambda found: any(cid == comment_id for cid, _, _ in found),
    )  # fmt: skip
    if any(cid == comment_id for cid, _, _ in comments):
        raise CommentCheckError(f"{what}: answered gone but the comment is still listed")
    if not complete:
        raise CommentCheckError(f"{what}: answered gone; too many comments to confirm it is absent")


def read_graph_comment(
    session, base, comment_id, access_token, *, text_field, permission, raise_for_error,
    object_id=None,
) -> Optional[AuthoredText]:  # fmt: skip
    """The text comment ``comment_id`` says now (``GET /{comment-id}``, stripped; its
    ``.author`` is ``from.id``), or ``None`` when Graph says it is gone (:data:`_GRAPH_GONE`).
    With ``object_id`` a "gone" is confirmed against that post's listing first (see
    :func:`_confirm_gone`): still listed or unconfirmable raises. Any other failure raises: a
    caller never acts on a read that did not answer."""
    response = session.get(
        f"{base}/{comment_id}",
        params={"fields": f"id,{text_field},from"},
        headers={"Authorization": f"OAuth {access_token}"},
        timeout=30,
    )
    if _is_gone(response):
        if object_id:
            _confirm_gone(
                session, base, comment_id, object_id, access_token, text_field=text_field,
                permission=permission, raise_for_error=raise_for_error,
                what=f"read of {comment_id}",
            )  # fmt: skip
        return None
    body = _graph_body(response, permission, raise_for_error)
    if text_field not in body:
        raise CommentCheckError(f"comment {comment_id}: the read answered no {text_field}")
    said = AuthoredText(str(body.get(text_field) or "").strip())
    said.author = _author_id(body) or None
    return said


def _graph_call(
    session, method: str, url: str, access_token: str, permission: str, raise_for_error, **kwargs
) -> dict:
    response = getattr(session, method)(
        url, headers={"Authorization": f"OAuth {access_token}"}, timeout=30, **kwargs
    )
    if method == "delete" and _is_gone(response):
        # Already deleted (an earlier DELETE landed and its answer was lost): the goal holds --
        # but ``gone`` tells the caller to confirm, since the pair also answers other failures.
        return {"success": True, "gone": True}
    body = _graph_body(response, permission, raise_for_error)
    if body.get("success") is not True:
        raise CommentCorrectionError(f"Graph {method.upper()} {url.rsplit('/', 1)[-1]}: not ok")
    return body


# Graph API reference, Comment node: ``POST /{comment-id}`` with ``message`` updates a comment and
# ``DELETE /{comment-id}`` deletes it, each answering ``{"success": bool}``
# (https://developers.facebook.com/docs/graph-api/reference/comment/). Instagram's IG Comment
# supports ``DELETE`` but no text edit -- ``POST`` only hides/unhides
# (https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-comment).
def edit_graph_comment(
    session, base, comment_id, text, access_token, *, permission, raise_for_error
) -> None:
    """Edit a Facebook comment in place (``POST /{comment-id}`` ``message=``)."""
    _graph_call(
        session, "post", f"{base}/{comment_id}", access_token, permission, raise_for_error,
        data={"message": text},
    )  # fmt: skip


def delete_graph_comment(
    session, base, comment_id, access_token, *, text_field, permission, raise_for_error,
    object_id=None,
) -> None:  # fmt: skip
    """Delete one of the account's own comments (``DELETE /{comment-id}``). A "gone" answer
    (code 100 / subcode 33, whose generic message UNVERIFIED also covers a comment that cannot
    be loaded for missing permissions) is trusted only once the listing of ``object_id``'s
    comments no longer shows ``comment_id``: the same answer on ``GET /{comment-id}`` proves
    nothing. One still listed, a listing that fails or is cut short, or no ``object_id`` to list,
    raises: the delete stays pending and nothing is posted or resolved on the answer alone."""
    body = _graph_call(
        session, "delete", f"{base}/{comment_id}", access_token, permission, raise_for_error
    )  # fmt: skip
    if not body.get("gone"):
        return
    _confirm_gone(
        session, base, comment_id, object_id, access_token, text_field=text_field,
        permission=permission, raise_for_error=raise_for_error,
        what=f"Graph DELETE {comment_id}",
    )  # fmt: skip


def replace_graph_comment(
    session,
    base: str,
    object_id: str,
    stale_id: str,
    text: str,
    access_token: str,
    *,
    text_field: str,
    permission: str,
    raise_for_error: Callable[[int, dict], None],
    own_id: str = "",
) -> str:
    """Replace comment ``stale_id`` on ``object_id`` with ``text`` where a comment cannot be
    edited (Instagram): never two comments, never a post before the delete is confirmed.

    Reads every bounded page first. ``stale_id`` listed: delete it (a 2xx without
    ``success`` raises). Not listed: it is gone (an earlier pass deleted it and the post
    failed), but only when the read was complete, else ``CommentCheckError``. Then post ``text``
    unless a comment already says it. Returns the id of the comment that says ``text``. Once the
    stale comment is gone, a failed post raises ``ReplacePostFailedError`` (see there)."""
    if not own_id:
        raise _own_id_unknown(object_id)
    comments, complete = _read_graph_comments(
        session,
        base,
        object_id,
        access_token,
        text_field=text_field,
        permission=permission,
        raise_for_error=raise_for_error,
        own_id=own_id,
    )
    if any(cid == stale_id and ours for cid, _, ours in comments):
        delete_graph_comment(
            session, base, stale_id, access_token,
            text_field=text_field, permission=permission, raise_for_error=raise_for_error,
            object_id=object_id,
        )  # fmt: skip
    elif not complete:
        raise CommentCheckError(
            f"cannot confirm comment {stale_id} on {object_id} is gone (too many comments); "
            "nothing posted"
        )
    try:
        return post_graph_comment(
            session, base, object_id, text, access_token,
            text_field=text_field, permission=permission, raise_for_error=raise_for_error,
            own_id=own_id, post_beside_foreign=True,
        )  # fmt: skip
    except Exception as exc:  # noqa: BLE001 -- the stale comment is gone: the caller must know
        raise ReplacePostFailedError(exc) from exc
