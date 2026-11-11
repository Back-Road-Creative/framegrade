"""Crash/replay and authority contracts for unwired Page photo delivery."""

import hashlib
import sqlite3
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from gomoveshift.publishing.dispatch import drain_queue
from gomoveshift.publishing.facebook_photo_client import (
    PhotoReadbackUnknownError,
    PhotoStatus,
    PhotoUploadReceipt,
    PhotoUploadUnknownError,
)
from gomoveshift.publishing.facebook_photo_delivery import (
    AuthorizedPhoto,
    FacebookPhotoDelivery,
    PhotoAuthorizationError,
    PhotoDeliveryPendingError,
)
from gomoveshift.publishing.handlers import _dispatch_by_format, make_facebook_page_photo_handler
from gomoveshift.publishing.meta_comments import drain_meta_first_comments
from gomoveshift.publishing.queue import (
    FACEBOOK_PHOTO_KIND,
    STATUS_PHOTO_AWAITING_ID,
    STATUS_PHOTO_RECONCILE,
    acquire_lease,
    enqueue,
    facebook_photo_attempt,
    open_queue_db,
    transition,
)

NOW = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)
JPEG = b"\xff\xd8\xffprivate-approved-snapshot\xff\xd9"
SHA = hashlib.sha256(JPEG).hexdigest()


def row(db, identifier):
    return dict(db.execute("SELECT * FROM deliveries WHERE id=?", (identifier,)).fetchone())


@pytest.fixture
def setup(tmp_path):
    db = open_queue_db(tmp_path)
    identifier = enqueue(db, "job42", 0, "facebook", "queue-plan")
    approval = AuthorizedPhoto(
        delivery_id=identifier,
        queue_plan_hash="queue-plan",
        plan_sha256="a" * 64,
        receipt_sha256="b" * 64,
        nonce="c" * 64,
        item_index=0,
        page_id="123",
        caption="weekend memories\n\nExact caption",
        media_sha256=SHA,
        jpeg_bytes=JPEG,
        valid_until=NOW + timedelta(minutes=5),
        published=True,
        kind="PHOTO",
        timing="immediate",
        audience="public",
        media_purpose="publication_master",
    )
    authorize = Mock(return_value=approval)
    client = Mock(account_id="123")
    client.upload_photo.return_value = PhotoUploadReceipt("456", "123", SHA, True, "v21.0", "v26.0")
    client.read_status.return_value = PhotoStatus("456", "123_789", "123", True)
    delivery = FacebookPhotoDelivery(
        db, client, authorize=authorize, worker_id="worker", clock=lambda: NOW
    )
    yield db, identifier, approval, authorize, client, delivery
    db.close()


def test_success_checkpoints_before_network_and_preserves_caption(setup, tmp_path):
    db, identifier, approval, authorize, client, delivery = setup

    def upload(*args, **kwargs):
        # A different connection sees the committed checkpoint BEFORE POST.
        observer = sqlite3.connect(tmp_path / "social_queue.sqlite3")
        try:
            assert (
                observer.execute(
                    "SELECT status FROM deliveries WHERE id=?", (identifier,)
                ).fetchone()[0]
                == STATUS_PHOTO_AWAITING_ID
            )
            assert (
                observer.execute("SELECT count(*) FROM facebook_photo_attempts").fetchone()[0] == 1
            )
        finally:
            observer.close()
        return PhotoUploadReceipt("456", "123", SHA, True, "v21.0", "v26.0")

    client.upload_photo.side_effect = upload
    delivery.deliver(row(db, identifier))
    client.upload_photo.assert_called_once_with(
        JPEG, caption=approval.caption, expected_sha256=SHA, published=True
    )
    authorize.assert_called_once()
    saved = row(db, identifier)
    assert saved["status"] == "published"
    assert saved["upload_kind"] == FACEBOOK_PHOTO_KIND
    assert saved["upload_account"] == saved["destination_account"] == "123"
    assert (saved["platform_upload_id"], saved["platform_post_id"]) == ("456", "123_789")
    assert saved["published_at_source"] == "platform_reconciled"
    assert facebook_photo_attempt(db, identifier)["request"]["caption"] == approval.caption
    delivery.deliver(row(db, identifier))
    assert client.upload_photo.call_count == 1 and authorize.call_count == 1


@pytest.mark.parametrize(
    "change",
    [
        {"delivery_id": 99},
        {"queue_plan_hash": "edited"},
        {"page_id": "999"},
        {"media_sha256": "0" * 64},
        {"jpeg_bytes": bytearray(JPEG)},
        {"valid_until": NOW},
        {"valid_until": NOW.replace(tzinfo=None)},
        {"caption": None},
        {"item_index": True},
        {"published": False},
        {"kind": "VIDEO"},
        {"timing": "scheduled"},
        {"audience": "private"},
        {"media_purpose": "review_copy"},
        {"receipt_sha256": "unverified"},
    ],
)
def test_invalid_authorization_binding_prevents_checkpoint_and_post(setup, change):
    db, identifier, approved, authorize, client, delivery = setup
    authorize.return_value = replace(approved, **change)
    with pytest.raises(PhotoAuthorizationError):
        delivery.deliver(row(db, identifier))
    assert facebook_photo_attempt(db, identifier) is None
    client.upload_photo.assert_not_called()


@pytest.mark.parametrize(
    "authority", [None, Mock(return_value=None), Mock(side_effect=RuntimeError("private"))]
)
def test_no_trusted_authority_denies_without_checkpoint(setup, authority):
    db, identifier, _, _, client, _ = setup
    delivery = FacebookPhotoDelivery(db, client, authorize=authority, clock=lambda: NOW)
    with pytest.raises(PhotoAuthorizationError) as caught:
        delivery.deliver(row(db, identifier))
    assert "private" not in str(caught.value)
    assert facebook_photo_attempt(db, identifier) is None
    client.upload_photo.assert_not_called()


@pytest.mark.parametrize("failure", [PhotoUploadUnknownError("123"), SystemExit("crash")])
def test_lost_creation_response_parks_and_never_reinvokes_authority(setup, failure):
    db, identifier, _, authorize, client, delivery = setup
    client.upload_photo.side_effect = failure
    with pytest.raises((PhotoDeliveryPendingError, SystemExit)):
        delivery.deliver(row(db, identifier))
    saved = row(db, identifier)
    assert saved["status"] == STATUS_PHOTO_AWAITING_ID
    assert saved["platform_upload_id"] is None
    # Even an explicit call after restart cannot turn this into another POST.
    authorize.side_effect = AssertionError("must not inspect changed approval")
    for _ in range(2):
        with pytest.raises(PhotoDeliveryPendingError):
            delivery.deliver(saved)
    assert client.upload_photo.call_count == 1
    client.read_status.assert_not_called()


def test_crash_after_receipt_checkpoint_reconciles_without_current_draft(setup):
    db, identifier, _, authorize, client, delivery = setup
    client.read_status.side_effect = SystemExit("crash")
    with pytest.raises(SystemExit):
        delivery.deliver(row(db, identifier))
    assert row(db, identifier)["status"] == STATUS_PHOTO_RECONCILE
    authorize.side_effect = AssertionError("draft/source no longer available")
    client.read_status.side_effect = None
    stale = {**row(db, identifier), "format": "reel", "sidecar_path": "missing.json"}
    delivery.deliver(stale)
    assert row(db, identifier)["status"] == "published"
    assert client.upload_photo.call_count == 1


def test_later_read_failure_preserves_story_id(setup):
    db, identifier, _, _, client, delivery = setup
    client.read_status.side_effect = PhotoReadbackUnknownError("123", "456", "123_789")
    with pytest.raises(PhotoDeliveryPendingError):
        delivery.deliver(row(db, identifier))
    assert facebook_photo_attempt(db, identifier)["story_id"] == "123_789"
    assert row(db, identifier)["platform_upload_id"] == "456"


@pytest.mark.parametrize(
    "status",
    [
        PhotoStatus("456", "123_789", "123", False),
        PhotoStatus("456", "123_789", "999", True),
        PhotoStatus("999", "123_789", "123", True),
        PhotoStatus("456", "123_789", "123", 1),
    ],
)
def test_false_or_mismatched_readback_cannot_close_delivery(setup, status):
    db, identifier, _, _, client, delivery = setup
    client.read_status.return_value = status
    with pytest.raises(PhotoDeliveryPendingError):
        delivery.deliver(row(db, identifier))
    assert row(db, identifier)["status"] == STATUS_PHOTO_RECONCILE
    assert row(db, identifier)["destination_account"] is None


def test_account_rotation_stops_reads_and_writes(setup):
    db, identifier, _, _, client, delivery = setup
    client.read_status.side_effect = PhotoReadbackUnknownError("123", "456", None)
    with pytest.raises(PhotoDeliveryPendingError):
        delivery.deliver(row(db, identifier))
    client.account_id = "999"
    client.reset_mock()
    with pytest.raises(PhotoDeliveryPendingError):
        delivery.reconcile(row(db, identifier))
    client.read_status.assert_not_called()
    client.upload_photo.assert_not_called()


def test_dispatch_reconciles_before_draft_privacy_or_validation_loaders(setup):
    db, identifier, _, authorize, client, delivery = setup
    client.read_status.side_effect = PhotoReadbackUnknownError("123", "456", None)
    with pytest.raises(PhotoDeliveryPendingError):
        delivery.deliver(row(db, identifier))
    client.read_status.side_effect = None
    authorize.side_effect = AssertionError("old approval must not gate reads")
    legacy = Mock(side_effect=AssertionError("must not route by current record"))
    privacy = Mock(side_effect=AssertionError("missing source"))
    result = drain_queue(
        db,
        {"facebook": legacy},
        owner="worker",
        now=NOW.isoformat(),
        facebook_photo_reconciler=delivery.reconcile,
        privacy_attestations=privacy,
        held_scopes={("facebook", "job42")},
    )
    assert result.succeeded == 1
    legacy.assert_not_called()
    privacy.assert_not_called()


def test_no_injected_reconciler_never_falls_back_to_legacy_route(setup):
    db, identifier, _, _, client, delivery = setup
    client.read_status.side_effect = PhotoReadbackUnknownError("123", "456", None)
    with pytest.raises(PhotoDeliveryPendingError):
        delivery.deliver(row(db, identifier))
    legacy = Mock(side_effect=AssertionError("wrong photo/Reels route"))
    result = drain_queue(db, {"facebook": legacy}, now=NOW.isoformat())
    assert result.skipped_no_handler == 1
    legacy.assert_not_called()
    assert row(db, identifier)["status"] == STATUS_PHOTO_RECONCILE


def test_current_record_router_refuses_durable_photo_before_loading_record(setup):
    loader = Mock(side_effect=AssertionError("record changed or vanished"))
    legacy = Mock()
    handler = _dispatch_by_format(loader, legacy, legacy)
    with pytest.raises(Exception, match="photo"):
        handler({"platform": "facebook", "upload_kind": FACEBOOK_PHOTO_KIND})
    loader.assert_not_called()
    legacy.assert_not_called()


def test_new_writes_still_obey_dispatch_privacy_hold(setup):
    db, identifier, _, authorize, client, delivery = setup
    result = drain_queue(
        db,
        {"facebook": make_facebook_page_photo_handler(delivery)},
        owner="worker",
        now=NOW.isoformat(),
        privacy_attestations=lambda _: None,
        facebook_photo_reconciler=delivery.reconcile,
    )
    assert result.held_privacy_unverified == 1
    authorize.assert_not_called()
    client.upload_photo.assert_not_called()


def test_foreign_live_lease_prevents_checkpoint_and_write(setup):
    db, identifier, _, _, client, delivery = setup
    assert acquire_lease(
        db, identifier, "other", (NOW + timedelta(hours=1)).isoformat(), now=NOW.isoformat()
    )
    with pytest.raises(PhotoDeliveryPendingError):
        delivery.deliver(row(db, identifier))
    assert facebook_photo_attempt(db, identifier) is None
    client.upload_photo.assert_not_called()


def test_expired_after_database_wait_stays_parked_without_post(setup):
    db, identifier, approved, authorize, client, _ = setup
    # Authority validation, pre-lease check, checkpoint timestamp, pre-POST check.
    clock = Mock(side_effect=[NOW, NOW, NOW, approved.valid_until])
    delivery = FacebookPhotoDelivery(db, client, authorize=authorize, clock=clock)
    with pytest.raises(PhotoDeliveryPendingError):
        delivery.deliver(row(db, identifier))
    assert row(db, identifier)["status"] == STATUS_PHOTO_AWAITING_ID
    client.upload_photo.assert_not_called()


def test_crash_immediately_after_checkpoint_never_posts_on_restart(setup, monkeypatch):
    from gomoveshift.publishing import facebook_photo_delivery as module

    db, identifier, _, authorize, client, delivery = setup
    original = module.begin_facebook_photo_attempt

    def crash(*args, **kwargs):
        assert original(*args, **kwargs)
        raise SystemExit("crash before POST")

    monkeypatch.setattr(module, "begin_facebook_photo_attempt", crash)
    with pytest.raises(SystemExit):
        delivery.deliver(row(db, identifier))
    with pytest.raises(PhotoDeliveryPendingError):
        delivery.deliver(row(db, identifier))
    assert authorize.call_count == 1
    client.upload_photo.assert_not_called()


def test_retained_readback_ignores_expired_approval(setup):
    db, identifier, approved, authorize, client, delivery = setup
    client.read_status.side_effect = PhotoReadbackUnknownError("123", "456", None)
    with pytest.raises(PhotoDeliveryPendingError):
        delivery.deliver(row(db, identifier))
    client.read_status.side_effect = None
    delivery._clock = lambda: approved.valid_until + timedelta(days=1)
    delivery.deliver(row(db, identifier))
    assert row(db, identifier)["status"] == "published"
    assert authorize.call_count == 1 and client.upload_photo.call_count == 1


def test_account_change_after_checkpoint_cannot_send_to_new_page(setup, monkeypatch):
    from gomoveshift.publishing import facebook_photo_delivery as module

    db, identifier, _, _, client, delivery = setup
    original = module.begin_facebook_photo_attempt

    def change_client(*args, **kwargs):
        result = original(*args, **kwargs)
        client.account_id = "999"
        return result

    monkeypatch.setattr(module, "begin_facebook_photo_attempt", change_client)
    with pytest.raises(PhotoDeliveryPendingError):
        delivery.deliver(row(db, identifier))
    client.upload_photo.assert_not_called()


@pytest.mark.parametrize("minutes,posts", [(-1, 1), (30, 0)])
def test_offset_clock_compares_queue_slots_as_utc(setup, minutes, posts):
    db, identifier, _, _, client, delivery = setup
    db.execute(
        "UPDATE deliveries SET scheduled_at=? WHERE id=?",
        ((NOW + timedelta(minutes=minutes)).isoformat(), identifier),
    )
    db.commit()
    delivery._clock = lambda: NOW.astimezone(timezone(timedelta(hours=2)))
    if posts:
        delivery.deliver(row(db, identifier))
    else:
        with pytest.raises(PhotoAuthorizationError):
            delivery.deliver(row(db, identifier))
    assert client.upload_photo.call_count == posts


@pytest.mark.parametrize("missing", ["published", "kind", "timing", "audience", "media_purpose"])
def test_authority_result_cannot_invent_omitted_intent(setup, missing):
    db, identifier, approved, authorize, client, delivery = setup
    values = asdict(approved)
    values.pop(missing)
    with pytest.raises(TypeError):
        AuthorizedPhoto(**values)
    authorize.side_effect = lambda _: AuthorizedPhoto(**values)
    with pytest.raises(PhotoAuthorizationError):
        delivery.deliver(row(db, identifier))
    assert facebook_photo_attempt(db, identifier) is None
    client.upload_photo.assert_not_called()


def test_approved_photo_never_posts_unapproved_first_comment_and_legacy_still_can(setup):
    db, _, approved, authorize, _, delivery = setup
    photo = enqueue(
        db,
        "photo-with-comment",
        0,
        "facebook",
        "queue-plan",
        first_comment_text="This comment was never approved",
    )
    authorize.return_value = replace(approved, delivery_id=photo)
    delivery.deliver(row(db, photo))
    adapter = Mock()
    factory = Mock(return_value=adapter)
    result = drain_meta_first_comments(
        db, adapter_factories={"facebook": factory}, now="2030-01-01T00:00:00+00:00"
    )
    assert result.posted == 0
    factory.assert_not_called()
    adapter.post_comment.assert_not_called()
    assert row(db, photo)["first_comment_posted_at"] is None

    legacy = enqueue(
        db, "legacy", 0, "facebook", "legacy-plan", first_comment_text="Legacy comment"
    )
    assert transition(
        db,
        legacy,
        "queued",
        "published",
        platform_post_id="legacy-story",
        published_at_source="platform_confirmed",
    )
    result = drain_meta_first_comments(
        db, adapter_factories={"facebook": factory}, now="2030-01-01T00:00:00+00:00"
    )
    assert result.posted == 1
    adapter.post_comment.assert_called_once_with("legacy-story", "Legacy comment")
