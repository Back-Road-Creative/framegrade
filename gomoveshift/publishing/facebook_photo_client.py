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


class PhotoReadbackUnknownError(Exception):
    """Status is unconfirmed; preserve known IDs and repeat only reads."""

    def __init__(self, page_id: str, photo_id: str, story_id: str | None):
        super().__init__("Facebook photo status unconfirmed; retain known IDs")
        self.page_id = page_id
        self.photo_id = photo_id
        self.story_id = story_id


class _UnconfirmedError(Exception):
    """Internal marker carrying no untrusted response details."""


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
            raise _UnconfirmedError
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
        except Exception:  # noqa: BLE001 -- a write without a receipt must stay unknown
            pass
        raise PhotoUploadUnknownError(self._page_id)

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
        except Exception:  # noqa: BLE001 -- readback cannot authorize repeating a write
            pass
        raise PhotoReadbackUnknownError(self._page_id, receipt.photo_id, story_id)
