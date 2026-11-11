"""Unwired, authorized single-Page-photo delivery with durable uncertain writes.

Only a trusted authorization callback may supply a new immutable snapshot.
Existing attempts reconcile from persisted identity; they never replay a POST.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from .facebook_photo_client import (
    FacebookPagePhotoClient,
    PhotoReadbackUnknownError,
    PhotoUploadReceipt,
)
from .queue import (
    FACEBOOK_PHOTO_KIND,
    STATUS_PHOTO_AWAITING_ID,
    STATUS_PHOTO_RECONCILE,
    acquire_lease,
    begin_facebook_photo_attempt,
    confirm_facebook_photo,
    facebook_photo_attempt,
    retain_facebook_photo_receipt,
    retain_facebook_photo_story,
)


class PhotoAuthorizationError(Exception):
    """No new write is allowed; authority is missing, stale or mismatched."""


class PhotoDeliveryPendingError(Exception):
    """Outcome is uncertain; retained attempts never authorize another POST."""


@dataclass(frozen=True, slots=True)
class AuthorizedPhoto:
    """Result of the trusted port, not itself proof of a signed approval.

    The port must verify the exact current signed plan, consume/recheck the
    decision, check fresh account/privacy/delivery evidence and capture final
    bytes. This type must never be used as a replacement for that verification.
    """

    delivery_id: int
    queue_plan_hash: str
    plan_sha256: str
    receipt_sha256: str
    nonce: str
    item_index: int
    page_id: str
    caption: str
    media_sha256: str
    jpeg_bytes: bytes
    valid_until: datetime
    published: bool
    kind: str
    timing: str
    audience: str
    media_purpose: str


class FacebookPhotoDelivery:
    """Inject explicitly; no production CLI/factory constructs this consumer."""

    def __init__(
        self,
        conn,
        client: FacebookPagePhotoClient,
        *,
        authorize: Callable[[dict], AuthorizedPhoto] | None = None,
        worker_id: str = "adapter",
        clock: Callable[[], datetime] | None = None,
    ):
        self._conn, self._client, self._authorize = conn, client, authorize
        self._worker = worker_id
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _now(self):
        now = self._clock()
        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise PhotoAuthorizationError("Photo clock must be timezone aware")
        return now.astimezone(timezone.utc)

    def _row(self, identifier):
        row = self._conn.execute("SELECT * FROM deliveries WHERE id=?", (identifier,)).fetchone()
        if row is None or row["platform"] != "facebook":
            raise PhotoAuthorizationError("A Facebook delivery is required")
        return dict(row)

    def _authorized(self, row):
        try:
            approved = self._authorize(dict(row)) if self._authorize is not None else None
            now = self._now()
            valid = (
                isinstance(approved, AuthorizedPhoto)
                and type(approved.delivery_id) is int
                and approved.delivery_id == row["id"]
                and approved.queue_plan_hash == row["plan_hash"]
                and approved.page_id == self._client.account_id
                and isinstance(approved.page_id, str)
                and re.fullmatch(r"[0-9]+", approved.page_id)
                and all(
                    isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in (
                        approved.plan_sha256,
                        approved.receipt_sha256,
                        approved.nonce,
                        approved.media_sha256,
                    )
                )
                and type(approved.item_index) is int
                and 0 <= approved.item_index < 20
                and isinstance(approved.caption, str)
                and len(approved.caption) <= 4000
                and type(approved.jpeg_bytes) is bytes
                and approved.jpeg_bytes.startswith(b"\xff\xd8\xff")
                and approved.jpeg_bytes.endswith(b"\xff\xd9")
                and hashlib.sha256(approved.jpeg_bytes).hexdigest() == approved.media_sha256
                and approved.published is True
                and approved.kind == "PHOTO"
                and approved.timing == "immediate"
                and approved.audience == "public"
                and approved.media_purpose == "publication_master"
                and isinstance(approved.valid_until, datetime)
                and approved.valid_until.utcoffset() is not None
                and now < approved.valid_until <= now + timedelta(minutes=15)
            )
            if valid:
                return approved
        except Exception:  # noqa: BLE001 -- authority failures never imply permission
            pass
        raise PhotoAuthorizationError(
            "Fresh trusted authorization for this exact photo is required"
        )

    def deliver(self, supplied_row: dict) -> None:
        row = self._row(supplied_row["id"])
        attempt = facebook_photo_attempt(self._conn, row["id"])
        if attempt is not None or row["upload_kind"] == FACEBOOK_PHOTO_KIND:
            return self.reconcile(row)
        if (
            row["status"] != "queued"
            or row["upload_kind"] is not None
            or row["platform_upload_id"]
            or row["platform_post_id"]
        ):
            raise PhotoAuthorizationError("Delivery cannot start a new Page photo attempt")
        approved = self._authorized(row)
        now = self._now()
        if now >= approved.valid_until:
            raise PhotoAuthorizationError("Photo authorization expired before checkpoint")
        if not acquire_lease(
            self._conn,
            row["id"],
            self._worker,
            (now + timedelta(minutes=15)).isoformat(),
            now=now.isoformat(),
        ):
            raise PhotoDeliveryPendingError("Photo delivery is leased elsewhere")
        snapshot = asdict(approved)
        snapshot.pop("jpeg_bytes")
        snapshot["valid_until"] = approved.valid_until.astimezone(timezone.utc).isoformat()
        if not begin_facebook_photo_attempt(
            self._conn, row["id"], self._worker, snapshot, now=self._now().isoformat()
        ):
            raise PhotoAuthorizationError("Photo attempt checkpoint refused")
        # A lock wait may outlast approval. The already-committed attempt then
        # stays conservatively parked; no new write follows an expired decision.
        if self._now() >= approved.valid_until or self._client.account_id != approved.page_id:
            raise PhotoDeliveryPendingError(
                "Authorization expired or account changed; hold attempt"
            )
        try:
            receipt = self._client.upload_photo(
                approved.jpeg_bytes,
                caption=approved.caption,
                expected_sha256=approved.media_sha256,
                published=True,
            )
            if (
                not isinstance(receipt, PhotoUploadReceipt)
                or receipt.page_id != approved.page_id
                or receipt.sha256 != approved.media_sha256
                or receipt.published_requested is not True
                or not isinstance(receipt.photo_id, str)
                or not re.fullmatch(r"[0-9]+", receipt.photo_id)
            ):
                raise ValueError("Mismatched creation receipt")
            if not retain_facebook_photo_receipt(self._conn, row["id"], asdict(receipt)):
                raise ValueError("Receipt checkpoint refused")
        except Exception:  # noqa: BLE001 -- never expose sensitive transport details or repeat POST
            pass
        else:
            return self.reconcile(row)
        raise PhotoDeliveryPendingError("Photo creation uncertain; hold recorded attempt")

    def reconcile(self, supplied_row: dict) -> None:
        """Read-only remote path: no authority callback, draft or file loading."""
        row = self._row(supplied_row["id"])
        attempt = facebook_photo_attempt(self._conn, row["id"])
        if row["status"] == "published" and attempt and row["upload_kind"] == FACEBOOK_PHOTO_KIND:
            return
        if (
            not attempt
            or row["upload_kind"] != FACEBOOK_PHOTO_KIND
            or row["status"] not in (STATUS_PHOTO_AWAITING_ID, STATUS_PHOTO_RECONCILE)
            or row["upload_account"] != self._client.account_id
            or attempt["request"]["page_id"] != row["upload_account"]
            or not attempt["receipt"]
            or not row["platform_upload_id"]
        ):
            raise PhotoDeliveryPendingError(
                "Photo identity unknown or account changed; hold attempt"
            )
        try:
            receipt = PhotoUploadReceipt(**attempt["receipt"])
            if (
                receipt.photo_id != row["platform_upload_id"]
                or receipt.page_id != row["upload_account"]
                or receipt.sha256 != attempt["request"]["media_sha256"]
                or receipt.published_requested is not True
            ):
                raise ValueError("Retained receipt identity mismatch")
            status = self._client.read_status(receipt)
        except PhotoReadbackUnknownError as exc:
            if (
                exc.page_id == row["upload_account"]
                and exc.photo_id == row["platform_upload_id"]
                and self._valid_story(exc.story_id, row["upload_account"])
            ):
                retain_facebook_photo_story(self._conn, row["id"], exc.story_id)
        except Exception:  # noqa: BLE001 -- negative/unknown read cannot authorize a write
            pass
        else:
            if (
                status.photo_id == row["platform_upload_id"]
                and status.page_id == row["upload_account"]
                and self._valid_story(status.story_id, row["upload_account"])
                and retain_facebook_photo_story(self._conn, row["id"], status.story_id)
                and status.is_published is True
                and confirm_facebook_photo(
                    self._conn, row["id"], status.photo_id, status.story_id, status.page_id
                )
            ):
                return
        raise PhotoDeliveryPendingError(
            "Photo publication unconfirmed; retain identity and read only"
        )

    @staticmethod
    def _valid_story(story_id, account):
        return (
            isinstance(story_id, str)
            and re.fullmatch(rf"{re.escape(account)}_[0-9]+", story_id) is not None
        )
