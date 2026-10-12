"""Photo and carousel publishing: a ``format=photo``/``format=carousel``
record gets a real publish path -- an adapter, a delivery row, a receipt --
instead of terminating at a scheduler CSV row and a hope
(``pipeline/photos/orchestrator.py::_publish_batch``).

Narrow interface, not a full Graph API client. This does NOT reimplement
``requests`` plumbing, host-allowlisting, or Meta error classification --
those already exist in ``instagram_adapter.py``/``facebook_adapter.py`` and
this module deliberately does not duplicate them. Instead it defines the
smallest client contract a caller must satisfy (see ``PhotoGraphClient``
below) and leaves constructing a real one -- wiring it to the same
``_post``/``_get``/host-allowlist machinery -- to whoever wires this adapter
up for a live account. The container shapes assumed here (an item container
per carousel item, then a ``CAROUSEL`` parent container listing the item
ids, then one ``media_publish`` call on the parent -- mirroring how
``instagram_adapter.create_container``/``publish_container`` already handle
a single Reel) match Meta's published Content Publishing API docs but have
NOT been exercised against a live account from this codebase -- verify
before depending on this in production.

A carousel publish is atomic FROM THE OPERATOR'S POINT OF VIEW: any failure
partway through item-container creation raises ``PartialCarouselError``
naming exactly which index failed and how many item containers were already
created -- never a blank/partial success. That failure is ALWAYS permanent,
never auto-retried, even when the underlying cause looks transient:
Instagram's item containers have no server-side idempotency key, so a retry
re-creates every item from scratch and would leave BOTH the orphaned
containers from this attempt and a second full set from the retry sitting
unpublished on the account. The orphaned containers from a partial attempt
expire on their own after ``CONTAINER_EXPIRY`` (``instagram_adapter.py``);
a human decides whether to wait that out or re-run once satisfied nothing
was double-posted -- see the class docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, Protocol

from gomoveshift.publishing.models import (
    FORMAT_CAROUSEL,
    FORMAT_PHOTO,
    MediaFingerprint,
    PublishRecord,
)
from gomoveshift.publishing.queue import PUBLISHED_AT_PLATFORM_CONFIRMED as PLATFORM_CONFIRMED
from gomoveshift.publishing.queue import acquire_lease, find_delivery, transition

# Ceiling on a single publish attempt's worker lease -- mirrors the video
# adapters' LEASE_SEC (a worker that dies mid-attempt frees this window for
# another worker, or this one on retry, to reclaim).
LEASE_SEC = 900


class PhotoPublishError(Exception):  # queue_state names the queue.py state this maps to
    queue_state: str = "failed_permanent"


class PhotoAuthBlockedError(PhotoPublishError):
    queue_state = "auth_blocked"


class PhotoRetryableError(PhotoPublishError):
    queue_state = "retry_wait"


class PhotoPermanentError(PhotoPublishError):  # message names the recovery action
    queue_state = "failed_permanent"


class UnsupportedPhotoFormatError(PhotoPermanentError):
    """A record whose ``format`` this adapter does not publish -- named
    explicitly in the message, never silently treated as a short."""


class PartialCarouselError(PhotoPermanentError):
    """A carousel failed partway through item-container creation.

    Never automatically retryable -- see the module docstring. ``succeeded``
    is how many item containers were created before the failure, ``total``
    is the carousel's full item count, and ``failed_index`` is the 0-based
    position that raised.
    """

    def __init__(self, message: str, *, succeeded: int, total: int, failed_index: int):
        super().__init__(message)
        self.succeeded = succeeded
        self.total = total
        self.failed_index = failed_index


class PhotoGraphClient(Protocol):
    """The narrow contract a real Meta Graph API client must satisfy --
    intentionally not ``instagram_adapter.InstagramReelsAdapter`` or
    ``facebook_adapter.FacebookReelsAdapter`` themselves, since a photo/
    carousel container is a different API shape from a Reel's resumable
    upload. See the module docstring for the assumed container flow."""

    #: The account this client publishes against, or ``None`` -- see
    #: ``_account_of`` in the video adapters for why this is read off the
    #: client rather than passed separately.
    account_id: Optional[str]

    def create_single_container(self, image_url: str, caption: str) -> str:
        """Create and return the id of a single-image publish container.
        ``image_url`` is a publicly-reachable URL, not raw bytes -- Meta's
        photo-container docs describe an ``image_url`` field, and there is
        no documented raw-bytes upload for a photo container (see
        ``photo_graph_client.py``'s module docstring)."""

    def create_carousel_item_container(self, image_url: str) -> str:
        """Create and return the id of one ``is_carousel_item`` container.
        ``image_url`` is a publicly-reachable URL, not raw bytes -- see
        ``create_single_container``."""

    def create_carousel_parent_container(self, item_ids: "tuple[str, ...]", caption: str) -> str:
        """Create and return the id of the ``CAROUSEL`` parent container
        listing ``item_ids`` as children."""

    def publish_container(self, container_id: str) -> Dict[str, Any]:
        """Publish a container. Returns a dict with at least ``post_id``
        (the platform's own post id) and, when available, ``permalink``.
        ``post_id`` absent/falsy means the platform did not confirm a post
        went live -- the caller treats that as a failure, never a blank
        -receipt success."""


@dataclass(frozen=True, slots=True)
class PhotoPublishResult:
    post_id: str
    permalink: Optional[str]
    item_count: int


#: Resolves a ``MediaFingerprint`` to the publicly-reachable ``image_url``
#: it should publish under -- NOT raw bytes (see
#: ``photo_graph_client.py``'s module docstring for why the Graph API wants
#: a URL here, unlike the video adapters' byte-loader contract).
MediaImageUrlLoader = Callable[[MediaFingerprint], str]

_SUPPORTED_FORMATS = (FORMAT_PHOTO, FORMAT_CAROUSEL)


class PhotoCarouselAdapter:
    """Publish a single photo or an N-item carousel for one ``PublishRecord``
    to one platform, durably queued through ``queue.py``. Mirrors
    ``InstagramShortsAdapter``/``FacebookShortsAdapter``'s
    persist-before-irreversible-call discipline where the narrow
    ``PhotoGraphClient`` contract allows it -- see the module docstring for
    what this adapter does NOT (yet) checkpoint across a restart.

    ``platform`` is the delivery row's own platform column ("instagram" or
    "facebook") -- used only to read the record's per-platform caption and
    to look up/record the delivery row; it is never sent to the client.
    """

    def __init__(self, client: PhotoGraphClient, platform: str, worker_id: str = "adapter"):
        self._client = client
        self._platform = platform
        self._worker_id = worker_id

    @property
    def account_id(self) -> Optional[str]:
        return getattr(self._client, "account_id", None)

    def publish(
        self,
        record: PublishRecord,
        *,
        conn,
        delivery_id: int,
        media_image_url_loader: MediaImageUrlLoader,
    ) -> PhotoPublishResult:
        if record.format not in _SUPPORTED_FORMATS:
            raise UnsupportedPhotoFormatError(
                f"PhotoCarouselAdapter cannot publish format={record.format!r} for "
                f"short_index={record.short_index}; only {sorted(_SUPPORTED_FORMATS)} "
                "are supported here"
            )

        caption = self._caption(record)

        existing = find_delivery(conn, record.project_name, record.short_index, self._platform)
        current_status = existing["status"] if existing else "queued"

        lease_expires = (datetime.now(timezone.utc) + timedelta(seconds=LEASE_SEC)).isoformat()
        if not acquire_lease(conn, delivery_id, self._worker_id, lease_expires):
            raise RuntimeError(
                f"delivery {delivery_id} is leased by another worker — refusing to "
                "risk a duplicate photo/carousel container"
            )

        if record.format == FORMAT_PHOTO:
            container_id = self._client.create_single_container(
                media_image_url_loader(record.media), caption
            )
            item_count = 1
        else:
            media_items = (record.media, *record.additional_media)
            item_ids: list[str] = []
            for index, media in enumerate(media_items):
                try:
                    item_ids.append(
                        self._client.create_carousel_item_container(media_image_url_loader(media))
                    )
                except Exception as exc:  # noqa: BLE001 -- see class docstring: ALWAYS permanent
                    raise PartialCarouselError(
                        f"carousel item {index} of {len(media_items)} failed to upload after "
                        f"{len(item_ids)} item(s) already succeeded: {exc}",
                        succeeded=len(item_ids),
                        total=len(media_items),
                        failed_index=index,
                    ) from exc
            container_id = self._client.create_carousel_parent_container(tuple(item_ids), caption)
            item_count = len(item_ids)

        result = self._client.publish_container(container_id)
        post_id = result.get("post_id")
        if not post_id:
            raise PhotoPermanentError(
                "platform publish call returned no post id — refusing to record a "
                "receipt the platform never gave"
            )

        transition(
            conn,
            delivery_id,
            current_status,
            "published",
            platform_post_id=post_id,
            permalink=result.get("permalink"),
            published_at_source=PLATFORM_CONFIRMED,
            destination_account=self.account_id,
        )
        return PhotoPublishResult(
            post_id=post_id, permalink=result.get("permalink"), item_count=item_count
        )

    def _caption(self, record: PublishRecord) -> str:
        payload = getattr(record.payloads, self._platform, None)
        if payload is None:
            raise ValueError(
                f"short_index={record.short_index} carries no {self._platform} payload"
            )
        caption = getattr(payload, "caption", None)
        if caption is None:
            caption = getattr(payload, "description", "")
        return caption
