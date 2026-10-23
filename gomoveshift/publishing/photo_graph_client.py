"""Concrete Meta Graph API implementation of ``photo_adapter.PhotoGraphClient``.
Modeled on ``instagram_adapter.InstagramReelsAdapter`` — same base URL
construction, same ``_post``/``_get`` error-body handling, same
``meta_auth.load_meta_credentials`` credential shape, same host-allowlist
discipline (no upload host is called from this module — see below), and the
same fresh-every-time publishing-limit read (``_check_publishing_limit``,
account-level, never cached).

LIVE EVIDENCE, NAMED EXPLICITLY (see ``docs/social-publishing-meta.md``):
only ``publish_container`` (POST ``.../media_publish`` with a ``creation_id``)
and ``_check_publishing_limit`` (GET ``.../content_publishing_limit``) reuse
a call shape ``InstagramReelsAdapter`` already exercises against a live
account. Everything else in this module is inference from Meta's published
Content Publishing API docs and has **never** been run against a live
account from this codebase:

- ``create_carousel_parent_container`` (POST ``.../media`` with
  ``media_type=CAROUSEL`` and a ``children`` list of item-container ids)
  mirrors the *documented* shape and the same endpoint
  ``InstagramReelsAdapter.create_container`` already calls for a Reel — but
  the carousel-specific fields (``media_type=CAROUSEL``, ``children``) are
  themselves unexercised.
- ``create_single_container``/``create_carousel_item_container`` now take
  an ``image_url`` (see ``photo_image_url.py``) instead of
  raw bytes — Meta's photo-container docs describe an ``image_url`` field
  pointing at a publicly-reachable image; there is no documented raw-bytes
  upload for a photo container analogous to Reels' resumable
  ``rupload.facebook.com`` flow. Both methods verify the URL is actually
  live (``photo_image_url.verify_image_url``) BEFORE creating any
  container — an upstream gate immediately before the irreversible Graph
  call. The ``image_url=``/``is_carousel_item=true`` request shape itself
  is still documentation inference, UNEXERCISED against a live account.

  ``photo_adapter.PhotoGraphClient``'s contract now matches this
  ``image_url: str`` shape end to end:
  ``photo_adapter.PhotoCarouselAdapter.publish`` calls it through a
  ``media_image_url_loader``, and ``cli/handlers/dispatch_queue.py``'s
  ``_build_routed_handlers`` wires one bound to the routed brand's own
  ``config.photos.resolve_site_root()`` -- never
  ``photos.selector.default_site_root()``'s import-time fallback, so a
  second brand's config reaches the deriver/verifier. A
  ``MediaFingerprint.path`` that isn't under the site's served
  ``static/images/journeys/<slug>/<filename>`` layout (e.g. a resized
  social copy, which the site does not serve) is refused as a named,
  permanent failure rather than guessed at -- see
  ``photo_image_url.image_url_for_media_path``. This class still has
  **never** been run against a live account from this codebase for
  anything beyond ``publish_container``/``_check_publishing_limit`` --
  see above.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import requests

from gomoveshift.publishing.meta_auth import graph_error_tag
from gomoveshift.publishing.meta_graph import graph_base_url, graph_get, graph_post
from gomoveshift.publishing.photo_adapter import (
    PhotoAuthBlockedError,
    PhotoPermanentError,
    PhotoPublishError,
    PhotoRetryableError,
)
from gomoveshift.publishing.photo_image_url import verify_image_url

_AUTH_ERROR_CODES = frozenset({190})
_AUTH_ERROR_SUBCODES = frozenset({458, 459, 460, 463, 467})
_RATE_LIMIT_CODES = frozenset({4, 17, 32, 613})


class PhotoPublishingLimitError(PhotoRetryableError):
    """Publishing-limit refusal — mirrors
    ``instagram_adapter.InstagramPublishingLimitError``: the rolling 24h
    window self-heals, so this is retryable, never permanent. Raised both
    when the account is at/over quota and when the quota reading itself is
    malformed/missing (fail closed rather than assume quota is fine)."""


class PhotoImageUrlUnverifiedError(PhotoPermanentError):
    """A derived ``image_url`` failed its pre-publish reachability check —
    refused before any Graph container was created. The message carries
    ``photo_image_url``'s named reason code (``not_found`` /
    ``wrong_content_type`` / ``unreachable``) so the caller never sees a
    bare "invalid URL"."""


def _raise_for_photo_graph_error(status_code: int, body: dict) -> None:
    """Classify a Graph API error onto the ``PhotoPublishError`` hierarchy —
    same code/subcode buckets as ``instagram_adapter._raise_for_graph_error``.
    Never includes the token, only the platform's own error body fields."""
    error = (body or {}).get("error") or {}
    code, subcode = error.get("code"), error.get("error_subcode")
    message = str(error.get("message") or "unknown Graph API error")
    tag = graph_error_tag(status_code, error)
    if code in _AUTH_ERROR_CODES or subcode in _AUTH_ERROR_SUBCODES:
        raise PhotoAuthBlockedError(f"{tag} Meta access token expired or revoked; reauthorize")
    if code in _RATE_LIMIT_CODES or status_code >= 500:
        raise PhotoRetryableError(f"{tag} transient Graph API error: {message}")
    raise PhotoPermanentError(
        f"{tag} Graph API permanently rejected the request: {message} — not "
        "automatically retryable; this can mean a content/account rejection "
        "or a request bug (e.g. an invalid/nonexistent field), verify before "
        "resubmitting"
    )


class MetaPhotoGraphClient:
    """Publishes a single photo or an N-item carousel via the Meta Graph
    API. Satisfies ``photo_adapter.PhotoGraphClient``. See the module
    docstring for exactly which calls below have live evidence."""

    def __init__(self, credentials, session=None, graph_version=None, site_root=None):
        self._account_id = credentials["ig_user_id"]
        self._access_token = credentials["page_access_token"]
        graph_version = graph_version or credentials.get("graph_version")
        self._base = graph_base_url(graph_version)
        self._session = session or requests.Session()
        if site_root is not None:
            self._site_root = site_root
        else:
            from gomoveshift.pipeline.photos.selector import default_site_root

            self._site_root = default_site_root()

    @property
    def account_id(self) -> Optional[str]:
        """The account this client publishes against — recorded on every
        delivery receipt as ``destination_account``, mirroring
        ``InstagramReelsAdapter.account_id``."""
        return self._account_id

    def _post(self, url: str, **kwargs) -> dict:
        return graph_post(self._session, url, _raise_for_photo_graph_error, **kwargs)

    def _get(self, url: str, **kwargs) -> dict:
        return graph_get(self._session, url, _raise_for_photo_graph_error, **kwargs)

    def _check_publishing_limit(self) -> None:
        """Live read of the account's rolling publishing-limit quota
        — called fresh before every
        ``publish_container``, never cached. Same call shape as
        ``InstagramReelsAdapter._check_publishing_limit`` (live evidence)."""
        body = self._get(
            f"{self._base}/{self._account_id}/content_publishing_limit",
            params={"fields": "quota_usage,config", "access_token": self._access_token},
        )
        entries = body.get("data") or []
        if not entries:
            raise PhotoPublishingLimitError(
                "publishing-limit check returned no data — refusing to "
                "publish without a confirmed quota reading"
            )
        entry = entries[0]
        config = entry.get("config") or {}
        quota_usage, quota_total = entry.get("quota_usage"), config.get("quota_total")
        if quota_usage is None or quota_total is None:
            raise PhotoPublishingLimitError(
                "publishing-limit response is missing quota_usage/quota_total "
                "— refusing to publish without a confirmed quota reading"
            )
        if quota_usage >= quota_total:
            window_hours = config.get("quota_duration", 86400) / 3600
            raise PhotoPublishingLimitError(
                f"publishing limit reached: {quota_usage}/{quota_total} used "
                f"in the rolling {window_hours:.0f}h window (0 remaining) — "
                "retry once the window advances"
            )

    def _verify_or_raise(self, image_url: str) -> None:
        result = verify_image_url(image_url, session=self._session, site_root=self._site_root)
        if not result.ok:
            raise PhotoImageUrlUnverifiedError(
                f"refusing to publish {image_url!r} — {result.reason}"
            )

    def create_single_container(self, image_url: str, caption: str) -> str:
        """POST ``.../media`` with ``image_url`` — verifies the URL is
        actually reachable first (see the module docstring: this wire
        shape is documentation inference, unexercised against a live
        account)."""
        self._verify_or_raise(image_url)
        body = self._post(
            f"{self._base}/{self._account_id}/media",
            data={
                "image_url": image_url,
                "caption": caption,
                "access_token": self._access_token,
            },
        )
        return body["id"]

    def create_carousel_item_container(self, image_url: str) -> str:
        """POST ``.../media`` with ``image_url`` + ``is_carousel_item=true``
        — same verify-first, unexercised-shape caveat as
        ``create_single_container``."""
        self._verify_or_raise(image_url)
        body = self._post(
            f"{self._base}/{self._account_id}/media",
            data={
                "image_url": image_url,
                "is_carousel_item": "true",
                "access_token": self._access_token,
            },
        )
        return body["id"]

    def create_carousel_parent_container(self, item_ids: Tuple[str, ...], caption: str) -> str:
        """POST ``.../media`` with ``media_type=CAROUSEL`` and ``children``
        — mirrors the documented shape and the endpoint
        ``InstagramReelsAdapter.create_container`` already calls for a Reel,
        but the carousel-specific fields are unexercised against a live
        account (see the module docstring)."""
        body = self._post(
            f"{self._base}/{self._account_id}/media",
            data={
                "media_type": "CAROUSEL",
                "caption": caption,
                "children": ",".join(item_ids),
                "access_token": self._access_token,
            },
        )
        return body["id"]

    def publish_container(self, container_id: str) -> Dict[str, Any]:
        """POST ``.../media_publish`` — same call shape as
        ``InstagramReelsAdapter.publish_container``'s ``media_publish`` step
        (live evidence). Reads the publishing limit fresh first."""
        self._check_publishing_limit()
        body = self._post(
            f"{self._base}/{self._account_id}/media_publish",
            data={"creation_id": container_id, "access_token": self._access_token},
        )
        post_id = body.get("id")
        permalink = self._fetch_permalink(post_id) if post_id else None
        return {"post_id": post_id, "permalink": permalink}

    def _fetch_permalink(self, media_id: str) -> Optional[str]:
        """Best-effort permalink lookup, mirroring
        ``InstagramReelsAdapter._fetch_permalink`` — a failure here must
        never turn an otherwise-successful publish into a reported failure."""
        try:
            body = self._get(
                f"{self._base}/{media_id}",
                params={"fields": "permalink", "access_token": self._access_token},
            )
        except (PhotoPublishError, requests.RequestException):
            return None
        return body.get("permalink")
