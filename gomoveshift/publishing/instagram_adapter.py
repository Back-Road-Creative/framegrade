"""Instagram Reels publishing adapter (Graph API). Resumable local-file
upload to ``rupload.facebook.com`` — no public Drive URL in the path.
The publishing-limit quota is read live via ``content_publishing_limit``
immediately before every ``media_publish`` call — never cached, see
``docs/social-publishing-meta.md``."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Tuple
from urllib.parse import urlsplit

import requests

from gomoveshift.publishing.first_comment import post_graph_comment
from gomoveshift.publishing.meta_auth import graph_error_tag
from gomoveshift.publishing.meta_graph import graph_base_url, graph_get, graph_post
from gomoveshift.publishing.models import InstagramPayload, ShortPublishRecord
from gomoveshift.publishing.queue import PUBLISHED_AT_PLATFORM_CONFIRMED as PLATFORM_CONFIRMED
from gomoveshift.publishing.queue import acquire_lease, find_delivery, transition

#: Exact-host allowlist for any upload URL, checked before any bytes are sent.
ALLOWED_UPLOAD_HOSTS = frozenset({"rupload.facebook.com", "graph.facebook.com"})
#: Containers expire 24h after creation — never publish one staged earlier.
CONTAINER_EXPIRY = timedelta(hours=24)
_AUTH_ERROR_CODES = frozenset({190})
_AUTH_ERROR_SUBCODES = frozenset({458, 459, 460, 463, 467})
_RATE_LIMIT_CODES = frozenset({4, 17, 32, 613})
#: "The media is not ready for publishing, please wait" (HTTP 400, code 9007,
#: subcode 2207027): the container is still processing, so it is retryable.
_MEDIA_NOT_READY = (9007, 2207027)
#: How often, and for how long at most, to poll a container's ``status_code``
#: for FINISHED before ``media_publish``. The bound sits well inside
#: ``LEASE_SEC``; a container still processing at the bound is retried later.
CONTAINER_READY_POLL_S = 10
CONTAINER_READY_TIMEOUT_S = 300


class InstagramPublishError(Exception):  # queue_state names the queue.py state this maps to
    queue_state: str = "failed_permanent"


class InstagramAuthBlockedError(InstagramPublishError):
    queue_state = "auth_blocked"


class InstagramRetryableError(InstagramPublishError):
    queue_state = "retry_wait"


class InstagramPublishingLimitError(InstagramRetryableError):
    """Publishing-limit refusal — the rolling 24h window self-heals, so this
    is retryable, never permanent. Raised both when the account is at/over
    quota and when the quota reading itself is malformed/missing (fail
    closed rather than assume quota is fine)."""


class InstagramPermanentError(InstagramPublishError):  # message names the recovery action
    queue_state = "failed_permanent"


class UnsafeUploadHostError(InstagramPermanentError):  # upload URL failed the host allowlist
    pass


@dataclass(frozen=True, slots=True)
class ContainerHandle:
    container_id: str
    upload_url: str
    created_at: datetime

    def is_expired(self, now: datetime | None = None) -> bool:
        return (now or datetime.now(timezone.utc)) - self.created_at >= CONTAINER_EXPIRY


def _raise_for_graph_error(status_code: int, body: dict) -> None:
    """Classify a Graph API error. Never includes the token — only the
    platform's own error body fields."""
    error = (body or {}).get("error") or {}
    code, subcode = error.get("code"), error.get("error_subcode")
    message = str(error.get("message") or "unknown Graph API error")
    tag = graph_error_tag(status_code, error)
    if code in _AUTH_ERROR_CODES or subcode in _AUTH_ERROR_SUBCODES:
        raise InstagramAuthBlockedError(f"{tag} Meta access token expired or revoked; reauthorize")
    if (code, subcode) == _MEDIA_NOT_READY:
        raise InstagramRetryableError(f"{tag} media container not ready for publishing: {message}")
    if code in _RATE_LIMIT_CODES or status_code >= 500:
        raise InstagramRetryableError(f"{tag} transient Graph API error: {message}")
    raise InstagramPermanentError(
        f"{tag} Graph API permanently rejected the request: {message} — not "
        "automatically retryable; this can mean a content/account rejection "
        "or a request bug (e.g. an invalid/nonexistent field), verify before "
        "resubmitting"
    )


class InstagramReelsAdapter:
    """Publishes a single Reel via the Instagram Graph API."""

    def __init__(self, credentials, session=None, graph_version=None):
        self._ig_user_id = credentials["ig_user_id"]
        self._access_token = credentials["page_access_token"]
        graph_version = graph_version or credentials.get("graph_version")
        self._base = graph_base_url(graph_version)
        self._session = session or requests.Session()

    @property
    def account_id(self) -> str:
        """The IG user every request from this client is addressed to. Public
        because the delivery receipt records WHERE a post landed and
        this is the only object that knows."""
        return self._ig_user_id

    @property
    def _auth_headers(self) -> dict:
        """The token as a header: ``requests`` puts GET ``params`` into the
        URL, which a ConnectionError message then echoes."""
        return {"Authorization": f"OAuth {self._access_token}"}

    def _post(self, url: str, **kwargs) -> dict:
        return graph_post(self._session, url, _raise_for_graph_error, **kwargs)

    def _get(self, url: str, **kwargs) -> dict:
        return graph_get(self._session, url, _raise_for_graph_error, **kwargs)

    def post_comment(self, object_id: str, text: str) -> str:
        """Post a top-level comment on ``object_id`` (first-comment drain).
        A permission-class error raises ``CommentPermissionError`` naming
        ``instagram_manage_comments``."""
        return post_graph_comment(
            self._session,
            self._base,
            object_id,
            text,
            self._access_token,
            text_field="text",
            permission="instagram_manage_comments",
            raise_for_error=_raise_for_graph_error,
        )

    def _check_publishing_limit(self) -> None:
        """Live read of the account's rolling publishing-limit quota
        — called fresh before every
        ``media_publish``, never cached between publishes."""
        body = self._get(
            f"{self._base}/{self._ig_user_id}/content_publishing_limit",
            params={"fields": "quota_usage,config"},
            headers=self._auth_headers,
        )
        entries = body.get("data") or []
        if not entries:
            raise InstagramPublishingLimitError(
                "Instagram publishing-limit check returned no data — "
                "refusing to publish without a confirmed quota reading"
            )
        entry = entries[0]
        config = entry.get("config") or {}
        quota_usage, quota_total = entry.get("quota_usage"), config.get("quota_total")
        if quota_usage is None or quota_total is None:
            raise InstagramPublishingLimitError(
                "Instagram publishing-limit response is missing quota_usage/quota_total — "
                "refusing to publish without a confirmed quota reading"
            )
        if quota_usage >= quota_total:
            window_hours = config.get("quota_duration", 86400) / 3600
            raise InstagramPublishingLimitError(
                f"Instagram publishing limit reached: {quota_usage}/{quota_total} used "
                f"in the rolling {window_hours:.0f}h window (0 remaining) — "
                "retry once the window advances"
            )

    def create_container(self, payload: InstagramPayload) -> ContainerHandle:
        body = self._post(
            f"{self._base}/{self._ig_user_id}/media",
            data={
                "media_type": "REELS",
                "caption": payload.caption,
                "upload_type": "resumable",
                "access_token": self._access_token,
            },
        )
        return ContainerHandle(body["id"], body["uri"], datetime.now(timezone.utc))

    def upload_video(self, upload_url: str, video_bytes: bytes) -> None:
        parsed = urlsplit(upload_url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_UPLOAD_HOSTS:
            raise UnsafeUploadHostError(
                f"refusing upload to {parsed.scheme}://{parsed.hostname}: not an allowlisted host"
            )
        self._post(
            upload_url,
            headers={
                "Authorization": f"OAuth {self._access_token}",
                "offset": "0",
                "file_size": str(len(video_bytes)),
            },
            data=video_bytes,
        )

    def _fetch_permalink(self, media_id: str) -> Optional[str]:
        """Best-effort permalink lookup on a published media object.

        A *container* has no ``permalink`` field — Graph rejects a request
        for it with (#100) "Tried accessing nonexisting field (permalink)",
        which ``_raise_for_graph_error`` maps to ``InstagramPermanentError``.
        Only the published *media* object supports ``permalink``, so this is
        always a separate GET, never read off a container or off
        ``media_publish``'s own response. Any failure here — including a
        request that itself errors — is swallowed: the permalink is
        presentation-only and must never turn an otherwise-successful
        publish/reconciliation into a reported failure.

        ``requests`` errors are caught alongside ``InstagramPublishError``:
        ``_get`` does NOT wrap the underlying ``self._session.get`` call, so
        a connection reset or read timeout on this presentation-only lookup
        would otherwise propagate and fail a publish that already
        succeeded."""
        try:
            body = self._get(
                f"{self._base}/{media_id}",
                params={"fields": "permalink"},
                headers=self._auth_headers,
            )
        except (InstagramPublishError, requests.RequestException):
            return None
        return body.get("permalink")

    def publish_container(self, container: ContainerHandle) -> dict:
        if container.is_expired():
            raise InstagramPermanentError(
                "media container expired (>24h since creation) — "
                "recreate the container and re-upload; can't be revived"
            )
        self._check_publishing_limit()
        body = self._post(
            f"{self._base}/{self._ig_user_id}/media_publish",
            data={"creation_id": container.container_id, "access_token": self._access_token},
        )
        media_id = body.get("id")
        # NOTE: `media_publish`'s own response does not carry `permalink` —
        # any value read off it here would be silently always None. The
        # permalink is fetched separately, on the published media id.
        permalink = self._fetch_permalink(media_id) if media_id else None
        return {"media_id": media_id, "permalink": permalink}

    def list_recent_media(self, limit: int = 50) -> list:
        """Read-only: the account's most recent media (newest first), for the
        remote duplicate scan in ``queue recover-terminal``. Publishes nothing."""
        body = self._get(
            f"{self._base}/{self._ig_user_id}/media",
            params={"fields": "id,caption,permalink,timestamp,media_product_type", "limit": limit},
            headers=self._auth_headers,
        )
        return list(body.get("data") or [])

    def check_container_status(self, container_id: str) -> dict:
        """Live read of a container's publish state — used to reconcile after
        a ``media_publish`` timeout, which is an unknown outcome, never a
        failure. A container whose ``status_code`` reports ``PUBLISHED``
        means the publish already succeeded server-side even though this
        client never saw the response.

        Only ``status_code``/``id``/``status`` are requested — a container
        has no ``permalink`` field (see ``_fetch_permalink``); when
        ``PUBLISHED``, the permalink is fetched in a separate, best-effort
        call.

        OPEN QUESTION (not settled here — do not assume either way):
        ``media_publish`` returns a *new* media id for the published post,
        which may differ from the container/creation id used to reach this
        method. This method only ever GETs the container's own id and
        returns that id back as ``media_id`` — if Graph in fact mints a
        distinct media id on publish, this hands the caller the wrong id.
        Settling this requires publishing a real Reel end-to-end and
        diffing the container id returned by ``create_container`` against
        the id ``media_publish`` returns for the same publish — out of
        scope here."""
        body = self._get(
            f"{self._base}/{container_id}",
            params={"fields": "status_code,id,status"},
            headers=self._auth_headers,
        )
        status_code = body.get("status_code")
        media_id = body.get("id")
        permalink = (
            self._fetch_permalink(media_id) if status_code == "PUBLISHED" and media_id else None
        )
        return {
            "status_code": status_code,
            "media_id": media_id,
            "permalink": permalink,
            "status": body.get("status"),  # Meta's own text, e.g. why an ERROR
        }


# --- Durable queue wiring (see queue.py; pattern mirrors youtube_adapter.py) ---

PLATFORM = "instagram"
STATUS_QUEUED, STATUS_SUBMITTING, STATUS_UPLOADED = "queued", "submitting", "uploaded"
STATUS_PUBLISHED = "published"

# Ceiling on a single container-create-through-publish attempt's worker
# lease — a worker that dies mid-attempt frees this window for another
# worker (or this one, on retry) to reclaim.
LEASE_SEC = 900


def _account_of(reels) -> Optional[str]:
    """The account the client publishes against, or None.

    Read off the CLIENT because that is the object holding the credential
    that was actually used; the adapter above it never sees one. `None` when
    the client exposes no account identity -- an honest unknown, which is what
    the `destination_account` column stores. `InstagramReelsAdapter.account_id`
    is pinned by a test, so a rename cannot quietly turn every receipt into
    that unknown.
    """
    return getattr(reels, "account_id", None)


@dataclass(frozen=True, slots=True)
class InstagramPublishResult:
    media_id: str
    permalink: Optional[str]


class InstagramShortsAdapter:
    """Publish a single ``ShortPublishRecord`` short to Instagram, durably
    queued through ``queue.py``."""

    def __init__(
        self,
        reels: InstagramReelsAdapter,
        worker_id: str = "adapter",
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self._reels = reels
        self._worker_id = worker_id
        self._sleep = sleep
        self._monotonic = monotonic

    def _await_ready(self, container_id: str) -> dict:
        """Poll the container until Meta has finished with it; returns the
        last status read (``FINISHED``, ``PUBLISHED`` or ``EXPIRED``).
        ``ERROR`` raises permanent, naming Meta's status text. Still
        processing when ``CONTAINER_READY_TIMEOUT_S`` runs out raises
        retryable: the container is kept and the next pass resumes it."""
        deadline = self._monotonic() + CONTAINER_READY_TIMEOUT_S
        while True:
            status = self._reels.check_container_status(container_id)
            code = status.get("status_code")
            if code in ("FINISHED", "PUBLISHED", "EXPIRED"):
                return status
            if code == "ERROR":
                raise InstagramPermanentError(
                    f"Meta could not process media container {container_id}: "
                    f"{status.get('status') or 'no status text'} — re-encode and resubmit"
                )
            if self._monotonic() >= deadline:
                raise InstagramRetryableError(
                    f"media container {container_id} still {code or 'without a status'} after "
                    f"{CONTAINER_READY_TIMEOUT_S}s; the next pass resumes it"
                )
            self._sleep(CONTAINER_READY_POLL_S)

    def _take_lease(self, conn, delivery_id: int) -> None:
        lease_expires = (datetime.now(timezone.utc) + timedelta(seconds=LEASE_SEC)).isoformat()
        if not acquire_lease(conn, delivery_id, self._worker_id, lease_expires):
            raise RuntimeError(
                f"delivery {delivery_id} is leased by another worker — refusing to "
                "risk a duplicate media container"
            )

    def _create_and_upload(
        self, conn, delivery_id: int, from_status: str, payload, video_bytes: bytes
    ) -> str:
        """Lease, create a container, persist it, upload; returns its id."""
        self._take_lease(conn, delivery_id)
        transition(conn, delivery_id, from_status, STATUS_SUBMITTING)
        handle = self._reels.create_container(payload)
        # Persist BEFORE upload/publish — the create call above put a
        # live container on the platform; this write makes it
        # recoverable by reconciliation instead of orphaned.
        transition(
            conn,
            delivery_id,
            STATUS_SUBMITTING,
            STATUS_UPLOADED,
            platform_upload_id=handle.container_id,
        )
        self._reels.upload_video(handle.upload_url, video_bytes)
        return handle.container_id

    def publish(
        self, record: ShortPublishRecord, *, conn, delivery_id: int, video_bytes: bytes
    ) -> InstagramPublishResult:
        payload = record.payloads.instagram
        if payload is None:
            raise ValueError(f"short_index={record.short_index} carries no Instagram payload")

        existing = find_delivery(conn, record.project_name, record.short_index, PLATFORM)
        current_status = existing["status"] if existing else STATUS_QUEUED
        container_id, media_id, permalink = (
            self._reconcile(existing)
            if existing
            else (
                None,
                None,
                None,
            )
        )

        created_here = container_id is None
        if created_here:
            container_id = self._create_and_upload(
                conn, delivery_id, current_status, payload, video_bytes
            )
            current_status = STATUS_UPLOADED
        elif media_id is None:
            # A resumed container is polled for up to CONTAINER_READY_TIMEOUT_S
            # under drain_queue's shorter, pass-start lease: renew it first.
            self._take_lease(conn, delivery_id)

        if media_id is None:
            # Meta processes the upload after it returns: publishing before
            # the container is FINISHED fails (9007/2207027). Wait, on the
            # fresh and the resumed path alike.
            ready = self._await_ready(container_id)
            if ready["status_code"] == "EXPIRED" and not created_here:
                # A resumed container that expired while we waited can never
                # be published and nothing was published from it: start over,
                # as ``_reconcile`` does for one it reads as EXPIRED.
                container_id = self._create_and_upload(
                    conn, delivery_id, current_status, payload, video_bytes
                )
                current_status, created_here = STATUS_UPLOADED, True
                ready = self._await_ready(container_id)
            if ready["status_code"] == "EXPIRED":
                raise InstagramPermanentError(
                    f"container expired right after it was created ({container_id}); "
                    "recreate it and re-upload"
                )
            if ready["status_code"] == "PUBLISHED":  # never publish a second time
                media_id, permalink = ready.get("media_id"), ready.get("permalink")

        if media_id is None:
            # ``ContainerHandle.created_at`` isn't persisted in queue.py's
            # schema; on a resumed container assume "just created" so the
            # local expiry check never false-refuses — an actually-expired
            # container still fails via the Graph API's own error.
            resume_handle = ContainerHandle(container_id, "", datetime.now(timezone.utc))
            result = self._reels.publish_container(resume_handle)
            media_id, permalink = result["media_id"], result["permalink"]

        transition(
            conn,
            delivery_id,
            current_status,
            STATUS_PUBLISHED,
            platform_post_id=media_id,
            permalink=permalink,
            published_at_source=PLATFORM_CONFIRMED,
            # The IG user this adapter published against, as held at
            # construction -- see the note in ``facebook_adapter`` for why it
            # is recorded here and never re-read from credentials later.
            destination_account=_account_of(self._reels),
        )
        return InstagramPublishResult(media_id=media_id, permalink=permalink)

    def _reconcile(self, existing: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Mirrors ``youtube_adapter.YouTubeShortsAdapter._reconcile``: a
        ``media_publish`` timeout is an unknown outcome, never a failure.
        Checks the persisted container's live status before ever creating a
        second container or re-publishing. Returns
        ``(container_id, media_id, permalink)`` — ``media_id`` is non-``None``
        only when the publish is confirmed, so callers never re-create or
        re-publish on a live-outcome retry."""
        container_id = existing.get("platform_upload_id")
        if not container_id:
            return None, None, None
        if existing.get("platform_post_id"):
            return container_id, existing["platform_post_id"], existing.get("permalink")
        status = self._reels.check_container_status(container_id)
        if status.get("status_code") == "PUBLISHED":
            return container_id, status.get("media_id"), status.get("permalink")
        if status.get("status_code") == "EXPIRED":
            # An expired container can never be published; nothing was ever
            # published from it, so a fresh one cannot duplicate. The caller
            # moves the row back to submitting (queue.transition is a plain
            # compare-and-set on the current status) and creates it.
            return None, None, None
        return container_id, None, None
