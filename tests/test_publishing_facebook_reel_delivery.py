"""Real SQLite checkpoints around mocked Reel I/O; no live provider calls."""

import hashlib
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from gomoveshift.publishing.facebook_adapter import UploadHandle
from gomoveshift.publishing.facebook_reel_delivery import (
    AuthorizedReel,
    FacebookReelDelivery,
    ReelAuthorizationError,
    ReelDeliveryPendingError,
    init_reel_ledger,
)

NOW = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)
VIDEO = b"exact-approved-video-snapshot"
SHA = hashlib.sha256(VIDEO).hexdigest()


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "reels.sqlite3"
    conn = sqlite3.connect(path)
    init_reel_ledger(conn)
    proof = AuthorizedReel(
        delivery_id=42,
        plan_sha256="a" * 64,
        receipt_sha256="b" * 64,
        nonce="c" * 64,
        item_index=0,
        page_id="123",
        caption="Exact caption",
        media_sha256=SHA,
        video_bytes=VIDEO,
        not_before=NOW - timedelta(seconds=1),
        not_after=NOW + timedelta(minutes=10),
        valid_until=NOW + timedelta(minutes=5),
        kind="VIDEO",
        audience="public",
        media_purpose="publication_master",
    )
    authorize = Mock(return_value=proof)
    client = Mock(account_id="123")
    client.start_upload.return_value = UploadHandle("456", "https://rupload.facebook.com/456")
    client.finish_upload.return_value = {"video_id": "456", "success": True}
    client.check_video_status.return_value = {"published": True}
    delivery = FacebookReelDelivery(conn, client, authorize=authorize, clock=lambda: NOW)
    yield conn, path, proof, authorize, client, delivery
    conn.close()


def row(conn):
    conn.row_factory = sqlite3.Row
    value = conn.execute("SELECT * FROM facebook_reel_attempts WHERE delivery_id=42").fetchone()
    return dict(value) if value else None


def test_success_commits_each_checkpoint_before_io_and_never_replays(setup):
    conn, path, proof, authorize, client, delivery = setup

    def state(expected, identifier):
        with sqlite3.connect(path) as observer:
            assert observer.execute(
                "SELECT state,upload_id FROM facebook_reel_attempts"
            ).fetchone() == (expected, identifier)

    def start():
        state("start_unknown", None)
        return UploadHandle("456", "https://rupload.facebook.com/456")

    def upload(url, data):
        state("upload_started", "456")
        assert data == VIDEO

    def finish(handle, payload, video_state):
        state("finish_unknown", "456")
        assert handle.video_id == "456"
        assert payload.description == proof.caption and video_state == "PUBLISHED"
        return {"video_id": "456", "success": True}

    client.start_upload.side_effect = start
    client.upload_video.side_effect = upload
    client.finish_upload.side_effect = finish
    result = delivery.deliver(42)
    assert result.video_id == "456" and result.page_id == "123"
    assert result.plan_sha256 == proof.plan_sha256 and result.item_index == 0
    assert result.evidence == "finish_confirmed"
    assert row(conn)["state"] == "published"
    assert authorize.call_count == 2
    assert delivery.deliver(42) == result
    assert client.start_upload.call_count == client.finish_upload.call_count == 1


def test_absent_authority_has_no_attempt_or_network(setup):
    conn, _, _, _, client, _ = setup
    with pytest.raises(ReelAuthorizationError):
        FacebookReelDelivery(conn, client, clock=lambda: NOW).deliver(42)
    assert row(conn) is None
    client.start_upload.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"delivery_id": 99},
        {"page_id": "999"},
        {"plan_sha256": "bad"},
        {"receipt_sha256": "bad"},
        {"nonce": "bad"},
        {"item_index": True},
        {"item_index": 20},
        {"kind": "PHOTO"},
        {"audience": "private"},
        {"media_purpose": "preview"},
        {"caption": "x" * 4001},
        {"video_bytes": b"different"},
        {"video_bytes": bytearray(VIDEO)},
        {"valid_until": NOW},
        {"valid_until": NOW + timedelta(minutes=16)},
        {"not_before": NOW + timedelta(seconds=1)},
        {"not_after": NOW},
        {"not_after": NOW.replace(tzinfo=None)},
    ],
)
def test_invalid_or_stale_proof_never_starts(setup, changes):
    conn, _, proof, authorize, client, delivery = setup
    authorize.return_value = replace(proof, **changes)
    with pytest.raises(ReelAuthorizationError):
        delivery.deliver(42)
    assert row(conn) is None
    client.start_upload.assert_not_called()


@pytest.mark.parametrize("failure", [RuntimeError("private provider text"), SystemExit("crash")])
def test_unknown_start_parks_even_without_id_and_never_retries(setup, failure):
    conn, _, _, _, client, delivery = setup
    client.start_upload.side_effect = failure
    with pytest.raises((ReelDeliveryPendingError, SystemExit)):
        delivery.deliver(42)
    assert row(conn)["state"] == "start_unknown"
    with pytest.raises(ReelDeliveryPendingError):
        delivery.deliver(42)
    assert client.start_upload.call_count == 1
    client.check_video_status.assert_not_called()


@pytest.mark.parametrize("failure", [RuntimeError("private provider text"), SystemExit("crash")])
def test_lost_upload_or_crash_retains_id_and_only_reads_on_restart(setup, failure):
    conn, _, _, authorize, client, delivery = setup
    client.upload_video.side_effect = failure
    with pytest.raises((ReelDeliveryPendingError, SystemExit)):
        delivery.deliver(42)
    assert row(conn)["upload_id"] == "456"
    authorize.side_effect = AssertionError("readback must not load current authority")
    result = delivery.deliver(42)
    assert result.evidence == "status_confirmed"
    assert client.upload_video.call_count == 1
    client.finish_upload.assert_not_called()


@pytest.mark.parametrize(
    "answer", [None, {}, {"success": False}, {"success": 1}, {"success": True, "video_id": "other"}]
)
def test_uncertain_finish_never_repeats_finish(setup, answer):
    conn, _, _, _, client, delivery = setup
    client.finish_upload.return_value = answer
    with pytest.raises(ReelDeliveryPendingError):
        delivery.deliver(42)
    assert row(conn)["state"] == "finish_unknown"
    assert delivery.deliver(42).evidence == "status_confirmed"
    assert client.finish_upload.call_count == 1


@pytest.mark.parametrize("failure", [RuntimeError("private text"), SystemExit("crash")])
def test_finish_exception_or_crash_is_read_only_after_restart(setup, failure):
    _, _, _, _, client, delivery = setup
    client.finish_upload.side_effect = failure
    with pytest.raises((ReelDeliveryPendingError, SystemExit)):
        delivery.deliver(42)
    assert delivery.deliver(42).evidence == "status_confirmed"
    assert client.finish_upload.call_count == 1


def test_expiry_during_upload_preserves_id_and_never_publishes_late(setup):
    conn, _, _, _, client, delivery = setup
    client.upload_video.side_effect = lambda *args: setattr(
        delivery, "_clock", lambda: NOW + timedelta(minutes=11)
    )
    with pytest.raises(ReelDeliveryPendingError):
        delivery.deliver(42)
    assert row(conn)["upload_id"] == "456"
    client.finish_upload.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"caption": "changed"},
        {"not_after": NOW + timedelta(minutes=12)},
        {"nonce": "d" * 64},
        {"video_bytes": b"different", "media_sha256": hashlib.sha256(b"different").hexdigest()},
    ],
)
def test_second_authorization_cannot_change_signed_binding(setup, changes):
    _, _, proof, authorize, client, delivery = setup
    authorize.side_effect = [proof, replace(proof, **changes)]
    with pytest.raises(ReelDeliveryPendingError):
        delivery.deliver(42)
    client.finish_upload.assert_not_called()


def test_only_fresh_evidence_deadline_may_be_renewed(setup):
    _, _, proof, authorize, _, delivery = setup
    authorize.side_effect = [proof, replace(proof, valid_until=NOW + timedelta(minutes=8))]
    assert delivery.deliver(42).video_id == "456"


def test_rotation_during_upload_blocks_finish_and_readback(setup):
    _, _, _, _, client, delivery = setup
    client.upload_video.side_effect = lambda *args: setattr(client, "account_id", "999")
    with pytest.raises(ReelDeliveryPendingError):
        delivery.deliver(42)
    with pytest.raises(ReelDeliveryPendingError):
        delivery.reconcile(42)
    client.finish_upload.assert_not_called()
    client.check_video_status.assert_not_called()


@pytest.mark.parametrize("answer", [None, {}, {"published": False}, {"published": 1}])
def test_negative_or_malformed_status_is_never_permission_to_write(setup, answer):
    _, _, _, _, client, delivery = setup
    client.finish_upload.side_effect = RuntimeError("unknown")
    with pytest.raises(ReelDeliveryPendingError):
        delivery.deliver(42)
    client.check_video_status.return_value = answer
    with pytest.raises(ReelDeliveryPendingError):
        delivery.reconcile(42)
    assert client.start_upload.call_count == client.finish_upload.call_count == 1


def test_unique_signed_item_cannot_start_under_second_delivery_id(setup):
    _, _, proof, authorize, client, delivery = setup
    delivery.deliver(42)
    authorize.return_value = replace(proof, delivery_id=43)
    with pytest.raises(ReelDeliveryPendingError):
        delivery.deliver(43)
    assert client.start_upload.call_count == 1


def test_nested_transaction_cannot_make_uncommitted_network_attempt(setup):
    conn, _, _, _, client, delivery = setup
    conn.execute("BEGIN")
    with pytest.raises(ReelDeliveryPendingError):
        delivery.deliver(42)
    client.start_upload.assert_not_called()
    conn.rollback()


@pytest.mark.parametrize("second", [False, True])
def test_authority_port_cannot_commit_or_rollback_unrelated_work(setup, second):
    conn, path, proof, authorize, client, delivery = setup
    conn.execute("CREATE TABLE caller_work(value TEXT)")
    calls = []

    def verifier(identifier):
        calls.append(identifier)
        if len(calls) == (2 if second else 1):
            conn.execute("INSERT INTO caller_work VALUES ('pending')")
        return proof

    authorize.side_effect = verifier
    with pytest.raises((ReelAuthorizationError, ReelDeliveryPendingError)):
        delivery.deliver(42)
    assert conn.in_transaction
    assert conn.execute("SELECT value FROM caller_work").fetchall() == [("pending",)]
    with sqlite3.connect(path) as observer:
        assert observer.execute("SELECT count(*) FROM caller_work").fetchone()[0] == 0
    client.finish_upload.assert_not_called()
    if not second:
        client.start_upload.assert_not_called()
    conn.rollback()


def test_expiry_during_finish_checkpoint_prevents_publish(setup, monkeypatch):
    conn, _, _, _, client, delivery = setup
    checkpoint = delivery._checkpoint

    def delayed(*args):
        checkpoint(*args)
        if args[2] == "finish_unknown":
            delivery._clock = lambda: NOW + timedelta(minutes=11)

    monkeypatch.setattr(delivery, "_checkpoint", delayed)
    with pytest.raises(ReelDeliveryPendingError):
        delivery.deliver(42)
    assert row(conn)["state"] == "finish_unknown"
    client.finish_upload.assert_not_called()


def test_database_wait_cannot_start_after_authority_expires(setup):
    conn, _, _, _, client, delivery = setup

    def waited(statement):
        if statement == "BEGIN IMMEDIATE":
            delivery._clock = lambda: NOW + timedelta(minutes=11)

    conn.set_trace_callback(waited)
    with pytest.raises(ReelAuthorizationError):
        delivery.deliver(42)
    assert row(conn) is None
    client.start_upload.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "http://rupload.facebook.com/456",
        "https://elsewhere.invalid/456",
        "https://user@rupload.facebook.com/456",
        "https://rupload.facebook.com:444/456",
        "https://rupload.facebook.com/456#fragment",
    ],
)
def test_unsafe_upload_address_retains_id_without_sending_bytes(setup, url):
    conn, _, _, _, client, delivery = setup
    client.start_upload.return_value = UploadHandle("456", url)
    with pytest.raises(ReelDeliveryPendingError):
        delivery.deliver(42)
    assert row(conn)["upload_id"] == "456"
    client.upload_video.assert_not_called()
    client.finish_upload.assert_not_called()


def test_provider_error_does_not_escape_through_exception_chain(setup):
    _, _, _, _, client, delivery = setup
    client.start_upload.side_effect = RuntimeError("provider private response")
    with pytest.raises(ReelDeliveryPendingError) as error:
        delivery.deliver(42)
    assert error.value.__context__ is None and error.value.__cause__ is None
    assert "provider private response" not in str(error.value)


def test_initialization_preserves_outer_transaction(tmp_path):
    conn = sqlite3.connect(tmp_path / "schema.sqlite")
    conn.execute("BEGIN")
    init_reel_ledger(conn)
    assert conn.in_transaction
    conn.rollback()
    assert not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='facebook_reel_attempts'"
    ).fetchone()
    conn.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("page_id", "999"),
        ("plan_sha256", "f" * 64),
        ("item_index", 8),
        ("state", "made_up"),
        ("receipt", '{"video_id":"777"}'),
    ],
)
def test_corrupt_retained_identity_cannot_be_confirmed(setup, field, value):
    conn, _, _, _, client, delivery = setup
    delivery.deliver(42)
    with conn:
        # Field names come exclusively from the fixed parametrization above.
        conn.execute(f"UPDATE facebook_reel_attempts SET {field}=?", (value,))  # noqa: S608
    if field == "page_id":
        client.account_id = "999"
    with pytest.raises(ReelDeliveryPendingError):
        delivery.reconcile(42)
    client.check_video_status.assert_not_called()
    assert client.start_upload.call_count == client.finish_upload.call_count == 1


def test_concurrent_consumer_cannot_repeat_a_started_attempt(setup):
    _, path, proof, _, client, delivery = setup
    other = sqlite3.connect(path)
    other_client = Mock(account_id="123")
    contender = FacebookReelDelivery(
        other, other_client, authorize=lambda _: proof, clock=lambda: NOW
    )

    def started():
        with pytest.raises(ReelDeliveryPendingError):
            contender.deliver(42)
        other_client.start_upload.assert_not_called()
        return UploadHandle("456", "https://rupload.facebook.com/456")

    client.start_upload.side_effect = started
    assert delivery.deliver(42).video_id == "456"
    other.close()


def test_real_low_level_adapter_contract_uses_exact_approved_description(setup):
    from gomoveshift.publishing.facebook_adapter import FacebookReelsAdapter

    conn, _, _, authorize, _, _ = setup
    session = Mock()
    replies = [
        {"video_id": "456", "upload_url": "https://rupload.facebook.com/456"},
        {"success": True},
        {"success": True},
    ]
    session.post.side_effect = [Mock(status_code=200, json=Mock(return_value=r)) for r in replies]
    client = FacebookReelsAdapter(
        {"page_id": "123", "page_access_token": "fake-token"}, session=session
    )
    result = FacebookReelDelivery(conn, client, authorize=authorize, clock=lambda: NOW).deliver(42)
    assert result.video_id == "456"
    finish = session.post.call_args.kwargs["data"]
    assert finish["description"] == "Exact caption" and finish["video_state"] == "PUBLISHED"
