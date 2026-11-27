"""Unwired Facebook Page-photo byte transport; no durable delivery guarantee.

The caller must commit a write-ahead attempt before ``upload_photo``, persist
its receipt before readback, and hold every uncertain write without replay.
See ``docs/facebook-page-photo-client.md`` for the approval/integration boundary.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, TypeGuard

import requests
from requests.adapters import HTTPAdapter

from .facebook_adapter import _AUTH_ERROR_CODES, _AUTH_ERROR_SUBCODES
from .meta_auth import DEFAULT_GRAPH_VERSION
from .meta_graph import graph_base_url

_ID = re.compile(r"[0-9]+")
_VERSION = re.compile(r"v[0-9]+\.[0-9]+")


def _matches(value: object, pattern: re.Pattern[str]) -> TypeGuard[str]:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class PhotoUploadReceipt:
    """Creation response only: neither publication nor ownership confirmation."""

    photo_id: str
    page_id: str
    sha256: str
    published_requested: bool
    requested_version: str
    response_version: str | None


@dataclass(frozen=True, slots=True)
class PhotoStatus:
    """Separate read observations, not an atomic platform snapshot."""

    photo_id: str
    story_id: str
    page_id: str
    is_published: bool


class PhotoUploadUnknownError(Exception):
    """One write was attempted; no trustworthy creation receipt was obtained.

    This does not authorize another POST, even for HTTP errors. The message and
    explicit attributes contain no raw response, request, token or caption;
    the underlying network exception is not chained. Traceback frame locals
    remain sensitive and must never be serialized into logs.
    """

    def __init__(self, page_id: str):
        super().__init__("Facebook photo creation unconfirmed; hold the recorded attempt")
        self.page_id = page_id
        self.photo_id = None


class PhotoUploadAuthBlockedError(PhotoUploadUnknownError):
    """The photo POST was refused for the Page token itself (HTTP 401, or Graph
    code 190 and its subcodes). Still unconfirmed, so it authorizes no other
    POST either; carries nothing from the response."""

    queue_state = "auth_blocked"  # what ``queue.is_auth_blocked`` reads


class PhotoReadbackUnknownError(Exception):
    """Status is unconfirmed; preserve known IDs and repeat only reads."""

    def __init__(self, page_id: str, photo_id: str, story_id: str | None):
        super().__init__("Facebook photo status unconfirmed; retain known IDs")
        self.page_id = page_id
        self.photo_id = photo_id
        self.story_id = story_id


class PhotoReadbackAuthBlockedError(PhotoReadbackUnknownError):
    """The status read was refused for the Page token itself (HTTP 401, or Graph
    code 190 and its subcodes): still unconfirmed, and no read works until
    someone re-authorizes. Carries nothing from the response."""

    queue_state = "auth_blocked"  # what ``queue.is_auth_blocked`` reads


class PhotoCommentUnknownError(Exception):
    """The first comment is unconfirmed; nothing about the photo post changed.

    Raised when the existing-comment read fails (nothing was posted) or when a
    comment write returned no trustworthy ID. Either way the next attempt reads
    first, so a comment Graph created whose answer was lost is found, not
    posted twice. Carries no response, request, token or comment text.
    """

    def __init__(self, page_id: str, story_id: str):
        super().__init__("Facebook first comment unconfirmed; the post itself is unchanged")
        self.page_id = page_id
        self.story_id = story_id


class PhotoCommentAuthBlockedError(PhotoCommentUnknownError):
    """The first comment's read or write was refused for the Page token itself;
    still unconfirmed, and no comment call works until someone re-authorizes."""

    queue_state = "auth_blocked"  # what ``queue.is_auth_blocked`` reads


#: Comment text bound for the approved first comment.
COMMENT_LIMIT = 1000
#: Pages of comments (100 each) read for an existing copy before posting.
COMMENT_CHECK_MAX_PAGES = 10
_COMMENT_ID = re.compile(r"[0-9]+(?:_[0-9]+)*")
_CURSOR = re.compile(r"[A-Za-z0-9_=-]{1,512}")


class _UnconfirmedError(Exception):
    """Internal marker carrying no untrusted response details."""


class _TokenAuthBlockedError(_UnconfirmedError):
    """Internal marker: Graph refused the token (no response details kept)."""


def _token_refused(response: requests.Response) -> bool:
    if response.status_code == 401:
        return True
    try:
        error = response.json().get("error") or {}
    except Exception:  # noqa: BLE001 -- an unreadable body names no refusal
        return False
    return error.get("code") in _AUTH_ERROR_CODES or (
        error.get("error_subcode") in _AUTH_ERROR_SUBCODES
    )


class FacebookPagePhotoClient:
    """One-shot Page.photos upload and read-only status verification.

    Credentials must be supplied by the existing SOPS loader. An injected
    session is a trusted test seam, not an alternate production transport; it
    must honor ``allow_redirects=False`` and perform one request per invocation.
    The default session has retries and environment credentials/proxies disabled.
    A supplied trusted test session must not install more-specific adapters or
    request overrides that add retries. No operation enables scheduling or updates.
    """

    def __init__(
        self,
        credentials: Mapping[str, Any],
        *,
        session: requests.Session | None = None,
    ) -> None:
        page_id = credentials.get("page_id")
        token = credentials.get("page_access_token")
        version = credentials.get("graph_version") or DEFAULT_GRAPH_VERSION
        if not _matches(page_id, _ID):
            raise ValueError("Page ID must be an ASCII numeric string")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("A Page access token is required")
        if not _matches(version, _VERSION):
            raise ValueError("Graph version must have the form vN.N")
        self._page_id: str = page_id
        self._token: str = token
        self._version: str = version
        self._base = graph_base_url(version)
        self._session = session if session is not None else requests.Session()
        self._session.trust_env = False
        self._session.mount("https://", HTTPAdapter(max_retries=0))

    @property
    def account_id(self) -> str:
        return self._page_id

    def _request(self, method: str, path: str, **kwargs: Any) -> tuple[dict[str, Any], str | None]:
        # Never call raise_for_status: its exception retains the prepared
        # authenticated request. Never propagate platform messages or redirects.
        response = self._session.request(
            method,
            f"{self._base}/{path}",
            headers={"Authorization": f"Bearer {self._token}"},
            allow_redirects=False,
            timeout=(10, 60),
            **kwargs,
        )
        if response.status_code != 200:
            raise _TokenAuthBlockedError if _token_refused(response) else _UnconfirmedError
        body = response.json()
        if not isinstance(body, dict) or "error" in body:
            raise _UnconfirmedError
        version = response.headers.get("facebook-api-version")
        return body, version if _matches(version, _VERSION) else None

    def upload_photo(
        self,
        jpeg_bytes: bytes,
        *,
        caption: str,
        expected_sha256: str,
        published: bool,
    ) -> PhotoUploadReceipt:
        """Send one immutable JPEG snapshot with explicit publication intent.

        Validation errors precede network access. Every error after dispatch is
        ambiguous and raises ``PhotoUploadUnknownError``. The caller's durable
        attempt must already exist; even an exception or process crash before
        receiving an ID must never cause an automatic replacement upload.
        """
        if type(published) is not bool:
            raise ValueError("published must be an explicit boolean")
        if not isinstance(caption, str):
            raise ValueError("caption must be text")
        if (
            type(jpeg_bytes) is not bytes
            or not jpeg_bytes.startswith(b"\xff\xd8\xff")
            or not jpeg_bytes.endswith(b"\xff\xd9")
        ):
            raise ValueError("An immutable JPEG byte snapshot is required")
        digest = hashlib.sha256(jpeg_bytes).hexdigest()
        if expected_sha256 != digest:
            raise ValueError("Photo snapshot does not match the approved SHA-256")
        # Raise outside except to omit the sensitive network exception chain.
        # As with any Python exception, traceback locals must not be logged.
        try:
            body, version = self._request(
                "POST",
                f"{self._page_id}/photos",
                data={"caption": caption, "published": "true" if published else "false"},
                files={"source": ("photo.jpg", jpeg_bytes, "image/jpeg")},
            )
            photo_id = body.get("id")
            if not _matches(photo_id, _ID):
                raise _UnconfirmedError
            return PhotoUploadReceipt(
                photo_id, self._page_id, digest, published, self._version, version
            )
        except _TokenAuthBlockedError:
            refused = True
        except Exception:  # noqa: BLE001 -- a write without a receipt must stay unknown
            refused = False
        raise (PhotoUploadAuthBlockedError if refused else PhotoUploadUnknownError)(self._page_id)

    def read_status(self, receipt: PhotoUploadReceipt) -> PhotoStatus:
        """Read the exact Photo's story and verify publication and Page owner.

        No missing, negative or malformed result permits another upload. The
        persisted receipt survives this operation; exceptions additionally carry
        any validated story ID acquired before a later read failed.
        """
        if receipt.page_id != self._page_id or not _matches(receipt.photo_id, _ID):
            raise ValueError("Photo receipt must identify this Page and a numeric Photo ID")
        story_id = None
        try:
            photo, _ = self._request("GET", receipt.photo_id, params={"fields": "id,page_story_id"})
            candidate = photo.get("page_story_id")
            if (
                photo.get("id") != receipt.photo_id
                or not isinstance(candidate, str)
                or not re.fullmatch(rf"{self._page_id}_[0-9]+", candidate)
            ):
                raise _UnconfirmedError
            story_id = candidate
            story, _ = self._request("GET", story_id, params={"fields": "id,is_published"})
            published = story.get("is_published")
            if story.get("id") != story_id or type(published) is not bool:
                raise _UnconfirmedError
            owner, _ = self._request("GET", story_id, params={"fields": "id,from"})
            source = owner.get("from")
            if (
                owner.get("id") != story_id
                or not isinstance(source, dict)
                or source.get("id") != self._page_id
            ):
                raise _UnconfirmedError
            return PhotoStatus(receipt.photo_id, story_id, self._page_id, published)
        except _TokenAuthBlockedError:
            raise PhotoReadbackAuthBlockedError(self._page_id, receipt.photo_id, story_id) from None
        except Exception:  # noqa: BLE001 -- readback cannot authorize repeating a write
            pass
        raise PhotoReadbackUnknownError(self._page_id, receipt.photo_id, story_id)

    def post_comment(self, story_id: str, text: str) -> str:
        """Post ``text`` once as this Page's comment on its own photo post.

        Reads the post's comments first and returns the ID of this Page's
        comment that already says exactly ``text``, without posting. Otherwise
        sends one comment write. Validation errors precede network access; any
        later failure raises ``PhotoCommentUnknownError``. Requires the Page
        token to hold ``pages_manage_engagement``.
        """
        if not isinstance(story_id, str) or not re.fullmatch(rf"{self._page_id}_[0-9]+", story_id):
            raise ValueError("Story ID must belong to this Page")
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > COMMENT_LIMIT
            or any((ord(c) < 32 and c != "\n") or ord(c) == 127 for c in text)
        ):
            raise ValueError("Comment must be 1-1000 characters without control characters")
        try:
            after = None
            for _ in range(COMMENT_CHECK_MAX_PAGES):
                params = {"fields": "id,message,from", "filter": "stream", "limit": "100"}
                if after is not None:
                    params["after"] = after
                page, _ = self._request("GET", f"{story_id}/comments", params=params)
                comments = page.get("data")
                if not isinstance(comments, list):
                    raise _UnconfirmedError
                for comment in comments:
                    if not isinstance(comment, dict):
                        raise _UnconfirmedError
                    source = comment.get("from")
                    if (
                        comment.get("message") == text
                        and isinstance(source, dict)
                        and source.get("id") == self._page_id
                        and _matches(comment.get("id"), _COMMENT_ID)
                    ):
                        return comment["id"]
                paging = page.get("paging")
                cursors = paging.get("cursors") if isinstance(paging, dict) else None
                after = cursors.get("after") if isinstance(cursors, dict) else None
                if not (isinstance(paging, dict) and paging.get("next")) or not _matches(
                    after, _CURSOR
                ):
                    break
            body, _ = self._request("POST", f"{story_id}/comments", data={"message": text})
            comment_id = body.get("id")
            if not _matches(comment_id, _COMMENT_ID):
                raise _UnconfirmedError
            return comment_id
        except _TokenAuthBlockedError:
            refused = True
        except Exception:  # noqa: BLE001 -- an unconfirmed comment must stay unknown
            refused = False
        raise (PhotoCommentAuthBlockedError if refused else PhotoCommentUnknownError)(
            self._page_id, story_id
        )
