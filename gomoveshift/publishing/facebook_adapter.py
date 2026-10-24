"""Facebook Reels publishing adapter. Mirrors ``instagram_adapter.py``'s
structure, error classes, and host-allowlist approach."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from urllib.parse import urlsplit

import requests

from gomoveshift.publishing.first_comment import post_graph_comment
from gomoveshift.publishing.meta_auth import graph_error_tag
from gomoveshift.publishing.meta_graph import graph_base_url, graph_get, graph_post
from gomoveshift.publishing.models import FacebookPayload, ShortPublishRecord
from gomoveshift.publishing.queue import PUBLISHED_AT_PLATFORM_CONFIRMED as PLATFORM_CONFIRMED
from gomoveshift.publishing.queue import PUBLISHED_AT_PLATFORM_RECONCILED as PLATFORM_RECONCILED
from gomoveshift.publishing.queue import PUBLISHED_AT_PLATFORM_SCHEDULED as PLATFORM_SCHEDULED
from gomoveshift.publishing.queue import acquire_lease, find_delivery, transition

#: Exact-host allowlist for any upload URL, checked before any bytes are sent.
ALLOWED_UPLOAD_HOSTS = frozenset({"rupload.facebook.com", "graph.facebook.com"})
#: Graph returns ``permalink_url`` RELATIVE ("/reel/<video id>/"); the web
#: origin that makes it clickable is not part of the API response.
WEB_ORIGIN = "https://www.facebook.com"
MIN_SCHEDULE_LEAD = timedelta(minutes=10)  # scheduled_publish_time bounds
MAX_SCHEDULE_LEAD = timedelta(days=29)
VALID_VIDEO_STATES = frozenset({"DRAFT", "SCHEDULED", "PUBLISHED"})
_AUTH_ERROR_CODES = frozenset({190})
_AUTH_ERROR_SUBCODES = frozenset({458, 459, 460, 463, 467})
_RATE_LIMIT_CODES = frozenset({4, 17, 32, 613})


class FacebookPublishError(Exception):  # queue_state names the queue.py state this maps to
    queue_state: str = "failed_permanent"


class FacebookAuthBlockedError(FacebookPublishError):
    queue_state = "auth_blocked"


class FacebookRetryableError(FacebookPublishError):
    queue_state = "retry_wait"


class FacebookPermanentError(FacebookPublishError):  # message names the recovery action
    queue_state = "failed_permanent"


class UnsafeUploadHostError(FacebookPermanentError):  # upload URL failed the host allowlist
    pass


ScheduleWindowError = FacebookPermanentError  # scheduled_publish_time outside the API's bounds


@dataclass(frozen=True, slots=True)
class UploadHandle:
    video_id: str
    upload_url: str


def _raise_for_graph_error(status_code: int, body: dict) -> None:
    """Classify a Graph API error. Never includes the token — only the
    platform's own error body fields."""
    error = (body or {}).get("error") or {}
    code, subcode = error.get("code"), error.get("error_subcode")
    message = str(error.get("message") or "unknown Graph API error")
    tag = graph_error_tag(status_code, error)
    if code in _AUTH_ERROR_CODES or subcode in _AUTH_ERROR_SUBCODES:
        raise FacebookAuthBlockedError(f"{tag} Meta access token expired or revoked; reauthorize")
    if code in _RATE_LIMIT_CODES or status_code >= 500:
        raise FacebookRetryableError(f"{tag} transient Graph API error: {message}")
    raise FacebookPermanentError(
        f"{tag} Graph API rejected the request: {message} — "
        "review the media/description against Meta content policy and resubmit"
    )


def reel_permalink(video_id: str) -> str:
    """The reel's public URL, derived from the video id alone. Graph's own
    ``permalink_url`` for a reel is exactly this path, so the delivery never has to survive a readback to record
    where it published."""
    return f"{WEB_ORIGIN}/reel/{video_id}/"


def absolute_permalink(url: Optional[str]) -> Optional[str]:
    """Graph's relative ``permalink_url`` made clickable; already-absolute
    values and ``None`` pass through untouched."""
    if not url:
        return None
    return url if urlsplit(url).scheme else f"{WEB_ORIGIN}{url}"


def _validate_schedule_window(scheduled_at: datetime, now: datetime | None = None) -> int:
    """Unix timestamp, or raises ``ScheduleWindowError`` naming the bound."""
    lead = scheduled_at - (now or datetime.now(timezone.utc))
    if lead <= MIN_SCHEDULE_LEAD:
        raise ScheduleWindowError(f"scheduled_publish_time must be more than {MIN_SCHEDULE_LEAD}")
    if lead > MAX_SCHEDULE_LEAD:
        raise ScheduleWindowError(f"scheduled_publish_time must be at most {MAX_SCHEDULE_LEAD}")
    return int(scheduled_at.timestamp())


class FacebookReelsAdapter:
    """Publishes a single Reel via the Facebook Graph API `video_reels` endpoint."""

    def __init__(self, credentials, session=None, graph_version=None):
        self._page_id = credentials["page_id"]
        self._access_token = credentials["page_access_token"]
        graph_version = graph_version or credentials.get("graph_version")
        self._base = graph_base_url(graph_version)
        self._session = session or requests.Session()

    @property
    def account_id(self) -> str:
        """The Page every request from this client is addressed to. Public
        because the delivery receipt records WHERE a post landed and
        this is the only object that knows."""
        return self._page_id

    def _post(self, url: str, **kwargs) -> dict:
        return graph_post(self._session, url, _raise_for_graph_error, **kwargs)

    def _get(self, url: str, **kwargs) -> dict:
        return graph_get(self._session, url, _raise_for_graph_error, **kwargs)

    def post_comment(self, object_id: str, text: str) -> str:
        """Post a top-level comment on ``object_id`` (first-comment drain).
        A permission-class error raises ``CommentPermissionError`` naming
        ``pages_manage_engagement``."""
        return post_graph_comment(
            self._session,
            self._base,
            object_id,
            text,
            self._access_token,
            text_field="message",
            permission="pages_manage_engagement",
            raise_for_error=_raise_for_graph_error,
        )

    def start_upload(self) -> UploadHandle:
        body = self._post(
            f"{self._base}/{self._page_id}/video_reels",
            data={"upload_phase": "start", "access_token": self._access_token},
        )
        return UploadHandle(video_id=body["video_id"], upload_url=body["upload_url"])

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

    def finish_upload(
        self,
        upload: UploadHandle,
        payload: FacebookPayload,
        video_state: str,
        scheduled_publish_time: datetime | None = None,
    ) -> dict:
        if video_state not in VALID_VIDEO_STATES:
            raise FacebookPermanentError(
                f"video_state must be one of {sorted(VALID_VIDEO_STATES)}, got {video_state!r}"
            )
        data = {
            "upload_phase": "finish",
            "video_id": upload.video_id,
            "video_state": video_state,
            "description": payload.description,
            "access_token": self._access_token,
        }
        if video_state == "SCHEDULED":
            if scheduled_publish_time is None:
                raise ScheduleWindowError("video_state=SCHEDULED requires scheduled_publish_time")
            data["scheduled_publish_time"] = _validate_schedule_window(scheduled_publish_time)
        body = self._post(f"{self._base}/{self._page_id}/video_reels", data=data)
        return {
            "video_id": upload.video_id,
            "success": body.get("success"),
            "post_id": body.get("post_id"),
        }

    def check_video_status(self, video_id: str) -> dict:
        """Live read of a Reel's publish state — used to reconcile after a
        ``upload_phase=finish`` timeout, an unknown outcome, never a
        failure. A published reel means the finish call already succeeded
        server-side, so the caller must not finish it again.

        The publish outcome is ``status.publishing_phase.publish_status``.
        It is NOT ``status.video_status``, which reads ``ready`` on a live
        published reel and never ``PUBLISHED`` — keying off that field
        (Graph v21.0) makes this check answer
        "not published" for every real reel, which is the one answer that
        lets a retry publish a duplicate."""
        body = self._get(
            f"{self._base}/{video_id}",
            params={"fields": "status,permalink_url", "access_token": self._access_token},
        )
        phase = ((body.get("status") or {}).get("publishing_phase") or {}).get("publish_status")
        return {
            "published": str(phase).lower() == "published",
            "permalink": absolute_permalink(body.get("permalink_url")) or reel_permalink(video_id),
        }


# --- Durable queue wiring (see queue.py; pattern mirrors instagram_adapter.py) ---

PLATFORM = "facebook"
STATUS_QUEUED, STATUS_SUBMITTING, STATUS_UPLOADED = "queued", "submitting", "uploaded"
STATUS_PUBLISHED = "published"

# Ceiling on a single start-through-finish attempt's worker lease — a
# worker that dies mid-attempt frees this window for another worker (or
# this one, on retry) to reclaim.
LEASE_SEC = 900


def due_schedule(record: ShortPublishRecord, now: datetime | None = None) -> "datetime | None":
    """The instant Facebook should publish this short, or ``None`` to publish
    on the call.

    ``make_facebook_handler`` never passed a ``scheduled_publish_time``, so
    ``record.schedule`` -- the slot the scheduler picked for this short -- was
    ignored and every reel published the moment the queue drained it. The
    per-platform slot wins over the shared one: the schedule lists a platform in
    ``scheduled_at_utc_by_platform`` only when it actually resolved a
    different hour for it, so using the shared slot would publish at a time
    the scheduler deliberately moved this platform off.

    A slot less than ``MIN_SCHEDULE_LEAD`` away is ``None``, not an error. The
    queue drains AT or just after the slot, so "under ten minutes away" means
    the moment has arrived and publishing now IS honouring the schedule;
    handing it to the API instead would raise ``ScheduleWindowError``, which
    is permanent, on every normal delivery.
    """
    schedule = record.schedule
    when = schedule.scheduled_at_utc_by_platform.get(PLATFORM) or schedule.scheduled_at_utc
    if when is None:
        return None
    if when - (now or datetime.now(timezone.utc)) <= MIN_SCHEDULE_LEAD:
        return None
    return when


def _account_of(reels) -> "Optional[str]":
    """The account the client publishes against, or None.

    Read off the CLIENT because that is the object holding the credential
    that was actually used; the adapter above it never sees one. `None` when
    the client does not expose an account identity (an injected test double,
    or a future client for a platform with no such notion) -- an honest
    unknown, which is exactly what the `destination_account` column stores.
    `FacebookReelsAdapter.account_id` is pinned by a test, so renaming it
    cannot quietly turn every receipt into that unknown.
    """
    return getattr(reels, "account_id", None)


@dataclass(frozen=True, slots=True)
class FacebookPublishResult:
    video_id: str
    permalink: Optional[str]


class FacebookShortsAdapter:
    """Publish a single ``ShortPublishRecord`` short to Facebook, durably
    queued through ``queue.py``. Mirrors ``InstagramShortsAdapter``, with
    Facebook's three-phase upload (start/upload/finish) standing in for
    Instagram's create/upload/publish.

    ``platform_post_id`` stores the bare video id, never packed JSON, and
    the permalink is derived from it (``reel_permalink``) and written in
    the SAME transition that marks the row published — ``transition``
    refuses every write once a row is terminal, so a permalink fetched
    afterwards could never be stored. A resumed, already-confirmed delivery
    returns ``permalink=None`` with no network call; its row already
    carries one."""

    def __init__(self, reels: FacebookReelsAdapter, worker_id: str = "adapter"):
        self._reels = reels
        self._worker_id = worker_id

    def publish(
        self,
        record: ShortPublishRecord,
        *,
        conn,
        delivery_id: int,
        video_bytes: bytes,
        video_state: str = "PUBLISHED",
        scheduled_publish_time: datetime | None = None,
    ) -> FacebookPublishResult:
        payload = record.payloads.facebook
        if payload is None:
            raise ValueError(f"short_index={record.short_index} carries no Facebook payload")

        # The parameters stay an override for a caller that names both; with
        # them at their defaults the record's own slot decides.
        if scheduled_publish_time is None and video_state == "PUBLISHED":
            scheduled_publish_time = due_schedule(record)
            if scheduled_publish_time is not None:
                video_state = "SCHEDULED"

        existing = find_delivery(conn, record.project_name, record.short_index, PLATFORM)
        current_status = existing["status"] if existing else STATUS_QUEUED
        video_id, post_id, permalink = (
            self._reconcile(existing, conn=conn, delivery_id=delivery_id)
            if existing
            else (None, None, None)
        )

        if video_id is None:
            lease_expires = (datetime.now(timezone.utc) + timedelta(seconds=LEASE_SEC)).isoformat()
            if not acquire_lease(conn, delivery_id, self._worker_id, lease_expires):
                raise RuntimeError(
                    f"delivery {delivery_id} is leased by another worker — refusing to "
                    "risk a duplicate video_reels upload session"
                )
            transition(conn, delivery_id, current_status, STATUS_SUBMITTING)
            handle = self._reels.start_upload()
            video_id = handle.video_id
            # Persist BEFORE upload/finish — the start call above put a
            # live upload session on the platform; this write makes it
            # recoverable by reconciliation instead of orphaned/duplicated.
            transition(
                conn, delivery_id, STATUS_SUBMITTING, STATUS_UPLOADED, platform_upload_id=video_id
            )
            self._reels.upload_video(handle.upload_url, video_bytes)
            current_status = STATUS_UPLOADED
        else:
            current_status = existing["status"]

        if post_id is None:
            upload = UploadHandle(video_id=video_id, upload_url="")
            self._reels.finish_upload(upload, payload, video_state, scheduled_publish_time)
            post_id = video_id  # bare video id — see class docstring
            permalink = reel_permalink(video_id)
            # A scheduled reel is ACCEPTED now and published later, so
            # `platform_confirmed` ("publish call returned success --
            # seconds") would overstate the evidence by the whole length of
            # the schedule. The recorded instant is the same value handed to
            # `finish_upload`, never re-derived, so the receipt and the API
            # request body cannot disagree. `published_at` is untouched: it
            # stays the moment THIS queue handed the delivery over.
            scheduled = video_state == "SCHEDULED" and scheduled_publish_time is not None
            fields = {
                "platform_post_id": post_id,
                "permalink": permalink,
                "published_at_source": PLATFORM_SCHEDULED if scheduled else PLATFORM_CONFIRMED,
                # WHERE it landed. `self._page_id` is the value this
                # adapter published against, taken from the brand's
                # credentials at construction -- recorded here rather than
                # re-read later, because credentials can be rotated or
                # corrected and a derived value would rewrite history to
                # agree with the present configuration.
                "destination_account": _account_of(self._reels),
            }
            if scheduled:
                fields["platform_published_at"] = scheduled_publish_time.isoformat()
            transition(conn, delivery_id, current_status, STATUS_PUBLISHED, **fields)

        return FacebookPublishResult(video_id=post_id, permalink=permalink)

    def _reconcile(
        self, existing: dict, *, conn, delivery_id: int
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Mirrors ``youtube_adapter.YouTubeShortsAdapter._reconcile``: a
        finish call has no create-idempotency key, so a timeout after
        ``upload_phase=finish`` is unknown, never a failure. Checks the
        persisted video id's live status before ever starting a NEW upload
        session or re-finishing. Returns ``(video_id, post_id, permalink)``
        — ``video_id`` set means reuse this session, never call
        ``start_upload`` again; ``post_id`` set means finish already
        succeeded, never call ``finish_upload`` again; ``permalink`` is
        populated only on a live status read."""
        video_id = existing.get("platform_upload_id")
        if not video_id:
            return None, None, None
        if existing.get("platform_post_id"):
            return video_id, existing["platform_post_id"], None
        status = self._reels.check_video_status(video_id)
        if status.get("published"):
            permalink = status.get("permalink")
            transition(
                conn,
                delivery_id,
                existing["status"],
                STATUS_PUBLISHED,
                platform_post_id=video_id,
                permalink=permalink,
                published_at_source=PLATFORM_RECONCILED,
                # The status read that found this post was made against
                # `self._page_id`, so that IS the page it is on -- this is
                # the adapter's own identifier, not a later re-derivation.
                destination_account=_account_of(self._reels),
            )
            return video_id, video_id, permalink
        return video_id, None, None
