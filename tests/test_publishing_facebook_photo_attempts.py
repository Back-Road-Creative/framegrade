"""Durable Page-photo attempts cannot be reset into replacement writes."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest

from gomoveshift.publishing.dispatch import drain_queue, reconcile_interrupted_attempts
from gomoveshift.publishing.queue import (
    FACEBOOK_PHOTO_KIND,
    STATUS_PHOTO_AWAITING_ID,
    STATUS_PHOTO_RECONCILE,
    RecoveryRefusedError,
    acquire_lease,
    begin_facebook_photo_attempt,
    cancel_delivery,
    check_recoverable,
    enqueue,
    facebook_photo_attempt,
    open_queue_db,
    reschedule_delivery,
    retry_delivery,
    transition,
)

NOW = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)


def snapshot(identifier):
    return dict(
        delivery_id=identifier,
        queue_plan_hash="queue-plan",
        plan_sha256="a" * 64,
        receipt_sha256="b" * 64,
        nonce="c" * 64,
        item_index=0,
        page_id="123",
        caption="exact",
        media_sha256="d" * 64,
        published=True,
        kind="PHOTO",
        timing="immediate",
        audience="public",
        media_purpose="publication_master",
        valid_until=(NOW + timedelta(minutes=5)).isoformat(),
    )


def prepare(db, name="job"):
    identifier = enqueue(db, name, 0, "facebook", "queue-plan")
    acquire_lease(
        db, identifier, "worker", (NOW + timedelta(minutes=10)).isoformat(), now=NOW.isoformat()
    )
    return identifier


def begin(db, identifier, data=None):
    return begin_facebook_photo_attempt(
        db, identifier, "worker", data or snapshot(identifier), now=NOW.isoformat()
    )


def saved(db, identifier):
    return dict(db.execute("SELECT * FROM deliveries WHERE id=?", (identifier,)).fetchone())


def test_migration_keeps_legacy_kind_unknown(tmp_path):
    db = open_queue_db(tmp_path)
    identifier = enqueue(db, "legacy", 0, "facebook", "hash")
    db.execute("ALTER TABLE deliveries DROP COLUMN upload_kind")
    db.commit()
    db.close()
    for _ in range(2):
        db = open_queue_db(tmp_path)
        assert saved(db, identifier)["upload_kind"] is None
        assert facebook_photo_attempt(db, identifier) is None
        db.close()


def test_checkpoint_is_atomic_and_immutable(tmp_path):
    db = open_queue_db(tmp_path)
    identifier = prepare(db)
    assert begin(db, identifier)
    current = saved(db, identifier)
    assert current["upload_kind"] == FACEBOOK_PHOTO_KIND
    assert current["upload_account"] == "123" and current["destination_account"] is None
    assert current["status"] == STATUS_PHOTO_AWAITING_ID
    assert facebook_photo_attempt(db, identifier)["request"] == snapshot(identifier)
    assert not begin(db, identifier, {**snapshot(identifier), "caption": "changed"})
    assert facebook_photo_attempt(db, identifier)["request"]["caption"] == "exact"


def test_insert_failure_rolls_back_prewrite_delivery_update(tmp_path):
    db = open_queue_db(tmp_path)
    first, second = prepare(db, "one"), prepare(db, "two")
    assert begin(db, first)
    with pytest.raises(sqlite3.IntegrityError):
        begin(db, second)  # same signed plan/item under a different delivery
    assert saved(db, second)["status"] == "queued"
    assert saved(db, second)["upload_kind"] is None
    assert facebook_photo_attempt(db, second) is None


def test_two_connections_cannot_claim_same_approved_item(tmp_path):
    db = open_queue_db(tmp_path)
    identifiers = [prepare(db, "one"), prepare(db, "two")]
    db.close()
    barrier = Barrier(2)

    def compete(identifier):
        conn = open_queue_db(tmp_path)
        try:
            barrier.wait(timeout=5)
            try:
                return begin(conn, identifier)
            except sqlite3.IntegrityError:
                return False
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(compete, identifiers)) == [False, True]


@pytest.mark.parametrize(
    "field,value",
    [("upload_account", "999"), ("platform_upload_id", "456"), ("platform_post_id", "123_789")],
)
def test_generic_transition_cannot_rewrite_attempt_identity(tmp_path, field, value):
    db = open_queue_db(tmp_path)
    identifier = prepare(db)
    assert begin(db, identifier)
    assert not transition(
        db, identifier, STATUS_PHOTO_AWAITING_ID, STATUS_PHOTO_AWAITING_ID, **{field: value}
    )


@pytest.mark.parametrize("state", ["queued", "cancelled", "failed_permanent", "uploaded"])
def test_generic_transition_cannot_release_attempt_hold(tmp_path, state):
    db = open_queue_db(tmp_path)
    identifier = prepare(db)
    assert begin(db, identifier)
    assert not transition(db, identifier, STATUS_PHOTO_AWAITING_ID, state)


def test_no_id_attempt_cannot_be_cancelled_retried_rescheduled_or_replaced(tmp_path):
    db = open_queue_db(tmp_path)
    identifier = prepare(db)
    assert begin(db, identifier)
    later = (NOW + timedelta(hours=2)).isoformat()
    assert not cancel_delivery(db, identifier, now=later)
    assert not retry_delivery(db, identifier, now=later)
    assert not reschedule_delivery(db, identifier, later, now=later)
    with pytest.raises(sqlite3.IntegrityError):
        enqueue(db, "job", 0, "facebook", "replacement-plan")
    # Even a damaged legacy state must not authorize recovery replacement.
    db.execute("UPDATE deliveries SET status='failed_permanent' WHERE id=?", (identifier,))
    db.commit()
    with pytest.raises(RecoveryRefusedError):
        check_recoverable(db, identifier)


def test_no_id_crash_releases_lease_without_retry_exhaustion_or_handler(tmp_path):
    db = open_queue_db(tmp_path)
    identifier = prepare(db)
    assert begin(db, identifier)
    later = (NOW + timedelta(hours=2)).isoformat()
    reconcile_interrupted_attempts(db, now=later, max_attempts=0)
    assert saved(db, identifier)["lease_owner"] is None
    assert saved(db, identifier)["status"] == STATUS_PHOTO_AWAITING_ID
    result = drain_queue(
        db,
        {"facebook": lambda _: pytest.fail("parked attempt selected")},
        now=later,
        max_attempts=0,
    )
    assert result.attempted == 0


def test_unknown_with_id_survives_expired_lease_and_retry_budget(tmp_path):
    db = open_queue_db(tmp_path)
    identifier = prepare(db)
    assert begin(db, identifier)
    # Fixture simulates an already-persisted receipt; no adapter call here.
    db.execute(
        "UPDATE deliveries SET status=?, platform_upload_id='456' WHERE id=?",
        (STATUS_PHOTO_RECONCILE, identifier),
    )
    db.commit()
    later = (NOW + timedelta(hours=2)).isoformat()
    reconcile_interrupted_attempts(db, now=later, max_attempts=0)
    assert saved(db, identifier)["status"] == STATUS_PHOTO_RECONCILE
    assert saved(db, identifier)["platform_upload_id"] == "456"


def test_transition_racing_checkpoint_cannot_cancel_new_attempt(tmp_path):
    db = open_queue_db(tmp_path)
    identifier = prepare(db)
    other = open_queue_db(tmp_path)
    raced = False

    def checkpoint_during_update(sql):
        nonlocal raced
        if not raced and sql.startswith("UPDATE deliveries SET status="):
            raced = True
            assert begin(other, identifier)

    db.set_trace_callback(checkpoint_during_update)
    assert not transition(db, identifier, STATUS_PHOTO_AWAITING_ID, "cancelled")
    db.set_trace_callback(None)
    assert raced
    assert saved(db, identifier)["status"] == STATUS_PHOTO_AWAITING_ID
    other.close()
