"""Explicitly injected Reel delivery; uncertain writes are never replayed.

This consumer is not installed by a CLI or queue factory. Its trusted authority
port must verify the actual approved reservation and capture qualified media.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable
from urllib.parse import urlsplit

from .facebook_adapter import FacebookReelsAdapter, UploadHandle, reel_permalink
from .models import FacebookPayload


class ReelAuthorizationError(Exception):
    """No new provider operation has been authorized."""


class ReelDeliveryPendingError(Exception):
    """Retained attempt needs reconciliation, never a blind write retry."""


@dataclass(frozen=True, slots=True)
class AuthorizedReel:
    """Trusted verifier output, not a replacement for verification of approval."""

    delivery_id: int
    plan_sha256: str
    receipt_sha256: str
    nonce: str
    item_index: int
    page_id: str
    caption: str
    media_sha256: str
    video_bytes: bytes
    not_before: datetime
    not_after: datetime
    valid_until: datetime
    kind: str
    audience: str
    media_purpose: str


@dataclass(frozen=True, slots=True)
class ReelDeliveryReceipt:
    page_id: str
    video_id: str
    plan_sha256: str
    receipt_sha256: str
    nonce: str
    item_index: int
    media_sha256: str
    evidence: str
    permalink: str


def init_reel_ledger(conn):
    """Explicit additive initialization; preserve any caller transaction."""
    conn.execute("SAVEPOINT reel_schema")
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS facebook_reel_attempts (
            delivery_id INTEGER PRIMARY KEY, plan_sha256 TEXT NOT NULL,
            item_index INTEGER NOT NULL, page_id TEXT NOT NULL,
            binding TEXT NOT NULL, binding_sha256 TEXT NOT NULL,
            upload_id TEXT, state TEXT NOT NULL, receipt TEXT,
            UNIQUE(plan_sha256,item_index)
        )""")
    except BaseException:
        conn.execute("ROLLBACK TO reel_schema")
        conn.execute("RELEASE reel_schema")
        raise
    conn.execute("RELEASE reel_schema")


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _matches(pattern, value):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def _aware(value):
    return isinstance(value, datetime) and value.utcoffset() is not None


def _binding(proof):
    fields = (
        "delivery_id plan_sha256 receipt_sha256 nonce item_index page_id caption "
        "media_sha256 kind audience media_purpose"
    ).split()
    value = {field: getattr(proof, field) for field in fields}
    for field in ("not_before", "not_after"):
        value[field] = getattr(proof, field).astimezone(timezone.utc).isoformat()
    return _json(value)


def _receipt(row, evidence):
    value = json.loads(row["binding"])
    return ReelDeliveryReceipt(
        page_id=row["page_id"],
        video_id=row["upload_id"],
        plan_sha256=value["plan_sha256"],
        receipt_sha256=value["receipt_sha256"],
        nonce=value["nonce"],
        item_index=value["item_index"],
        media_sha256=value["media_sha256"],
        evidence=evidence,
        permalink=reel_permalink(row["upload_id"]),
    )


class FacebookReelDelivery:
    """A durable consumer for one approved item, with a dedicated local ledger.

    No generic driving record is invented. The caller binds delivery_id to its
    reservation in authorize(), and supplies a qualified, non-retrying client.
    """

    def __init__(
        self,
        conn,
        client: FacebookReelsAdapter,
        *,
        authorize: Callable[[int], AuthorizedReel] | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self._conn, self._client, self._authorize = conn, client, authorize
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _now(self):
        value = self._clock()
        if not _aware(value):
            raise ReelAuthorizationError("A timezone-aware trusted clock is required")
        return value.astimezone(timezone.utc)

    def _current(self, proof):
        now = self._now()
        if (
            self._client.account_id != proof.page_id
            or not proof.not_before <= now < proof.not_after
            or not now < proof.valid_until <= now + timedelta(minutes=15)
        ):
            raise ReelAuthorizationError("Reel account, evidence or execution window changed")

    def _authorized(self, identifier):
        valid, proof = False, None
        try:
            proof = self._authorize(identifier) if self._authorize is not None else None
            valid = (
                not self._conn.in_transaction
                and type(identifier) is int
                and identifier > 0
                and isinstance(proof, AuthorizedReel)
                and type(proof.delivery_id) is int
                and proof.delivery_id == identifier
                and _matches(r"[0-9]{1,200}", proof.page_id)
                and all(
                    _matches(r"[0-9a-f]{64}", getattr(proof, field))
                    for field in ("plan_sha256", "receipt_sha256", "nonce", "media_sha256")
                )
                and type(proof.item_index) is int
                and 0 <= proof.item_index < 20
                and isinstance(proof.caption, str)
                and len(proof.caption) <= 4000
                and type(proof.video_bytes) is bytes
                and 0 < len(proof.video_bytes) <= 512 * 1024 * 1024
                and hashlib.sha256(proof.video_bytes).hexdigest() == proof.media_sha256
                and proof.kind == "VIDEO"
                and proof.audience == "public"
                and proof.media_purpose == "publication_master"
                and all(
                    _aware(getattr(proof, field))
                    for field in ("not_before", "not_after", "valid_until")
                )
            )
            if valid:
                self._current(proof)
        except Exception:
            valid = False
        if not valid:
            raise ReelAuthorizationError(
                "Fresh trusted authorization for this exact Reel is required"
            )
        return proof

    def _row(self, identifier):
        cursor = self._conn.execute(
            "SELECT * FROM facebook_reel_attempts WHERE delivery_id=?", (identifier,)
        )
        value = cursor.fetchone()
        if value is None:
            return None
        row = dict(zip((field[0] for field in cursor.description), value, strict=True))
        if _hash(row["binding"]) != row["binding_sha256"]:
            raise ReelDeliveryPendingError("Retained Reel binding is inconsistent")
        valid = False
        try:
            binding = json.loads(row["binding"])
            valid = all(
                binding[field] == row[field]
                for field in ("delivery_id", "plan_sha256", "item_index", "page_id")
            )
            if row["state"] == "start_unknown":
                valid = valid and row["upload_id"] is None and row["receipt"] is None
            else:
                valid = valid and _matches(r"[0-9]{1,200}", row["upload_id"])
                if row["state"] == "published":
                    receipt = json.loads(row["receipt"])
                    evidence = receipt["evidence"]
                    valid = (
                        valid
                        and evidence in ("finish_confirmed", "status_confirmed")
                        and receipt == asdict(_receipt(row, evidence))
                    )
                else:
                    valid = (
                        valid
                        and row["state"] in ("upload_started", "finish_unknown")
                        and row["receipt"] is None
                    )
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            raise ReelDeliveryPendingError("Retained Reel identity or receipt is inconsistent")
        return row

    def _checkpoint(self, identifier, before, after, upload_id=None):
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE facebook_reel_attempts SET state=?,upload_id=COALESCE(?,upload_id) "
                "WHERE delivery_id=? AND state=?",
                (after, upload_id, identifier, before),
            )
            if cursor.rowcount != 1:
                raise ReelDeliveryPendingError("Reel attempt changed; reconciliation required")

    def deliver(self, identifier: int) -> ReelDeliveryReceipt:
        if self._conn.in_transaction:
            raise ReelDeliveryPendingError("Reel delivery requires its own durable transaction")
        if self._row(identifier) is not None:
            return self.reconcile(identifier)
        proof = self._authorized(identifier)
        binding = _binding(proof)
        conflict = False
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._current(proof)  # Recheck after any database-lock wait.
            self._conn.execute(
                "INSERT INTO facebook_reel_attempts "
                "(delivery_id,plan_sha256,item_index,page_id,binding,binding_sha256,state) "
                "VALUES (?,?,?,?,?,?,'start_unknown')",
                (
                    identifier,
                    proof.plan_sha256,
                    proof.item_index,
                    proof.page_id,
                    binding,
                    _hash(binding),
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            self._conn.rollback()
            conflict = True
        except BaseException:
            self._conn.rollback()
            raise
        if conflict:
            raise ReelDeliveryPendingError("This delivery or signed item already has an attempt")
        failed = False
        try:
            self._current(proof)
            handle = self._client.start_upload()
            if not isinstance(handle, UploadHandle) or not _matches(
                r"[0-9]{1,200}", handle.video_id
            ):
                raise ValueError("start identity unavailable")
            self._checkpoint(identifier, "start_unknown", "upload_started", handle.video_id)
            parsed = urlsplit(handle.upload_url)
            if (
                parsed.scheme != "https"
                or parsed.netloc not in ("rupload.facebook.com", "graph.facebook.com")
                or parsed.fragment
                or not parsed.path.startswith("/")
            ):
                raise ValueError("upload origin refused")
            self._current(proof)
            self._client.upload_video(handle.upload_url, proof.video_bytes)
            refreshed = self._authorized(identifier)
            if _binding(refreshed) != binding:
                raise ValueError("approved identity changed")
            self._checkpoint(identifier, "upload_started", "finish_unknown")
            self._current(refreshed)  # Last boundary, after checkpoint/DB wait.
            response = self._client.finish_upload(
                handle, FacebookPayload(description=proof.caption), "PUBLISHED"
            )
            if (
                not isinstance(response, dict)
                or response.get("success") is not True
                or response.get("video_id") != handle.video_id
            ):
                raise ValueError("publication outcome unknown")
            return self._confirm(identifier, "finish_confirmed")
        except Exception:
            failed = True
        if failed:
            raise ReelDeliveryPendingError("Reel write outcome retained; reconcile without retry")
        raise AssertionError("unreachable")

    def _confirm(self, identifier, evidence):
        row = self._row(identifier)
        if row["page_id"] != self._client.account_id or not row["upload_id"]:
            raise ReelDeliveryPendingError("Reel account or retained identity changed")
        receipt = _receipt(row, evidence)
        with self._conn:
            self._conn.execute(
                "UPDATE facebook_reel_attempts SET state='published',receipt=? "
                "WHERE delivery_id=? AND upload_id=? AND state!='published'",
                (_json(asdict(receipt)), identifier, row["upload_id"]),
            )
        return ReelDeliveryReceipt(**json.loads(self._row(identifier)["receipt"]))

    def reconcile(self, identifier: int) -> ReelDeliveryReceipt:
        """Only read a known provider ID; no approval refresh or provider writes."""
        if self._conn.in_transaction:
            raise ReelDeliveryPendingError("Reconciliation requires its own durable transaction")
        row = self._row(identifier)
        if row is None or row["page_id"] != self._client.account_id or not row["upload_id"]:
            raise ReelDeliveryPendingError(
                "Reel identity unavailable; manual reconciliation required"
            )
        if row["state"] == "published":
            return ReelDeliveryReceipt(**json.loads(row["receipt"]))
        failed = False
        try:
            result = self._client.check_video_status(row["upload_id"])
            if not isinstance(result, dict) or result.get("published") is not True:
                raise ValueError("publication not confirmed")
            return self._confirm(identifier, "status_confirmed")
        except Exception:
            failed = True
        if failed:
            raise ReelDeliveryPendingError("Reel publication remains unconfirmed")
        raise AssertionError("unreachable")
