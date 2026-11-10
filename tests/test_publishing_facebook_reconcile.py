"""A finish request may have succeeded; a retry must only reconcile its ID."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from freezegun import freeze_time
from test_publishing_facebook_adapter import (
    CREDS,
    LIVE_STATUS_BODY,
    REEL_URL,
    _resp,
)
from test_publishing_models import _record

from gomoveshift.publishing import facebook_adapter as facebook
from gomoveshift.publishing.dispatch import PermanentDeliveryError, drain_queue
from gomoveshift.publishing.queue import (
    acquire_lease,
    enqueue,
    find_delivery,
    open_queue_db,
    pause_platform,
    transition,
)

UNKNOWN = "awaiting_facebook_reconcile"
NOW = "2026-11-10T12:00:00+00:00"


@pytest.fixture
def queue_conn(tmp_path):
    conn = open_queue_db(tmp_path / "seo")
    try:
        yield conn
    finally:
        conn.close()


def row(conn, record):
    return dict(find_delivery(conn, record.project_name, record.short_index, "facebook"))


def seed(conn, *, status="uploaded", attempts=0):
    record = _record()
    delivery = enqueue(conn, record.project_name, record.short_index, "facebook", "h1")
    transition(
        conn,
        delivery,
        "queued",
        status,
        platform_upload_id="v-1",
        attempt_count=attempts,
        upload_account="page-1",
    )
    return record, delivery


@pytest.mark.parametrize(
    "finish",
    [
        {},
        {"success": False},
        {"success": None},
        {"success": 1},
        {"success": "true"},
        [],
        ConnectionError("response lost"),
    ],
)
def test_finish_is_checkpointed_before_send_and_unconfirmed_never_published(queue_conn, finish):
    record = _record()
    delivery = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")
    seen = []

    def post(url, **kwargs):
        data = kwargs.get("data", {})
        phase = data.get("upload_phase") if isinstance(data, dict) else None
        seen.append(phase)
        if phase == "start":
            return _resp(200, {"video_id": "v-1", "upload_url": "https://rupload.facebook.com/x"})
        if phase == "finish":
            state = row(queue_conn, record)
            assert state["status"] == UNKNOWN and state["platform_upload_id"] == "v-1"
            if isinstance(finish, Exception):
                raise finish
            return Mock(status_code=200, json=Mock(return_value=finish))
        return _resp(200, {})

    session = Mock(post=Mock(side_effect=post), get=Mock())
    adapter = facebook.FacebookShortsAdapter(facebook.FacebookReelsAdapter(CREDS, session=session))
    with pytest.raises(facebook.FacebookRetryableError, match="unconfirmed"):
        adapter.publish(record, conn=queue_conn, delivery_id=delivery, video_bytes=b"x")
    state = row(queue_conn, record)
    assert state["status"] == UNKNOWN
    assert state["platform_post_id"] is None and state["published_at"] is None
    assert seen == ["start", None, "finish"]
    session.get.assert_not_called()


@pytest.mark.parametrize("status", ["uploaded", UNKNOWN])
@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"status": {"publishing_phase": {"publish_status": "processing"}}},
        {"status": []},
        {"status": {"publishing_phase": {"publish_status": True}}},
        ConnectionError("offline"),
    ],
)
def test_retry_of_retained_id_is_read_only_until_confirmed(queue_conn, status, reply):
    record, delivery = seed(queue_conn, status=status)
    get = (
        Mock(side_effect=reply)
        if isinstance(reply, Exception)
        else Mock(return_value=_resp(200, reply))
    )
    session = Mock(post=Mock(), get=get)
    adapter = facebook.FacebookShortsAdapter(facebook.FacebookReelsAdapter(CREDS, session=session))
    with pytest.raises(facebook.FacebookRetryableError, match="unconfirmed"):
        adapter.publish(record, conn=queue_conn, delivery_id=delivery, video_bytes=b"x")
    state = row(queue_conn, record)
    assert state["status"] == UNKNOWN and state["platform_upload_id"] == "v-1"
    assert state["platform_post_id"] is None
    session.post.assert_not_called()
    assert session.get.call_count == 1


def test_confirmed_reconcile_ignores_changed_draft_intent_and_records_original_id(queue_conn):
    record, delivery = seed(queue_conn, status=UNKNOWN)
    # Reconciliation checks what happened to the old ID, not a new publication request.
    record = replace(record, payloads=replace(record.payloads, facebook=None))
    session = Mock(post=Mock(), get=Mock(return_value=_resp(200, LIVE_STATUS_BODY)))
    result = facebook.FacebookShortsAdapter(
        facebook.FacebookReelsAdapter(CREDS, session=session)
    ).publish(record, conn=queue_conn, delivery_id=delivery, video_bytes=b"", video_state="DRAFT")
    assert result.video_id == "v-1" and result.permalink == REEL_URL
    state = row(queue_conn, record)
    assert state["status"] == "published" and state["published_at_source"] == "platform_reconciled"
    assert state["platform_upload_id"] == state["platform_post_id"] == "v-1"
    session.post.assert_not_called()


def test_checkpoint_compare_and_set_failure_prevents_finish(queue_conn, monkeypatch):
    record = _record()
    delivery = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")
    original = facebook.transition

    def conflict(conn, delivery_id, expected, status, **fields):
        return (
            False if status == UNKNOWN else original(conn, delivery_id, expected, status, **fields)
        )

    monkeypatch.setattr(facebook, "transition", conflict)
    session = Mock(
        post=Mock(
            side_effect=[
                _resp(200, {"video_id": "v-1", "upload_url": "https://rupload.facebook.com/x"}),
                _resp(200, {}),
            ]
        ),
        get=Mock(),
    )
    adapter = facebook.FacebookShortsAdapter(facebook.FacebookReelsAdapter(CREDS, session=session))
    with pytest.raises(facebook.FacebookRetryableError, match="checkpoint"):
        adapter.publish(record, conn=queue_conn, delivery_id=delivery, video_bytes=b"x")
    assert session.post.call_count == 2


def test_unknown_finish_remains_selectable_with_backoff_past_max_attempts(queue_conn):
    record, delivery = seed(queue_conn, status=UNKNOWN, attempts=20)
    session = Mock(post=Mock(), get=Mock(return_value=_resp(200, {})))
    adapter = facebook.FacebookShortsAdapter(facebook.FacebookReelsAdapter(CREDS, session=session))

    def handler(_):
        adapter.publish(record, conn=queue_conn, delivery_id=delivery, video_bytes=b"x")

    outcome = drain_queue(queue_conn, {"facebook": handler}, now=NOW, max_attempts=2)
    state = row(queue_conn, record)
    assert outcome.attempted == 1 and outcome.retried == 1 and outcome.failed_permanent == 0
    assert state["status"] == UNKNOWN and state["attempt_count"] == 21
    assert state["next_attempt_at"] > NOW and state["next_attempt_reason"] == "retry_backoff"
    assert state["lease_owner"] is None and state["platform_upload_id"] == "v-1"
    again = drain_queue(queue_conn, {"facebook": handler}, now=NOW, max_attempts=2)
    assert again.attempted == 0
    session.post.assert_not_called()


@pytest.mark.parametrize("status", ["uploaded", UNKNOWN])
def test_expired_lease_preserves_unknown_facebook_outcome(queue_conn, status):
    record, delivery = seed(queue_conn, status=status, attempts=20)
    acquire_lease(
        queue_conn, delivery, "dead", "2026-11-10T11:59:00+00:00", now="2026-11-10T11:58:00+00:00"
    )
    result = drain_queue(queue_conn, {}, now=NOW, max_attempts=2)
    state = row(queue_conn, record)
    assert state["status"] == UNKNOWN and state["attempt_count"] == 21
    assert state["platform_upload_id"] == "v-1" and state["platform_post_id"] is None
    assert state["next_attempt_at"] > NOW and state["lease_owner"] is None
    assert result.succeeded == result.failed_permanent == 0


def test_unknown_state_return_is_not_a_published_receipt(queue_conn):
    record, delivery = seed(queue_conn, status=UNKNOWN)
    result = drain_queue(queue_conn, {"facebook": lambda _: None}, now=NOW)
    assert result.succeeded == 0 and row(queue_conn, record)["status"] == UNKNOWN


def test_plain_failure_without_upload_id_retains_normal_retry_bound(queue_conn):
    record = _record()
    delivery = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")
    transition(queue_conn, delivery, "queued", "queued", attempt_count=20)

    def fail(_):
        raise facebook.FacebookRetryableError("ordinary failure")

    result = drain_queue(queue_conn, {"facebook": fail}, now=NOW, max_attempts=2)
    assert result.failed_permanent == 1 and row(queue_conn, record)["status"] == "failed_permanent"


def test_paused_legacy_upload_only_releases_expired_lease(queue_conn):
    record, delivery = seed(queue_conn, status="uploaded", attempts=20)
    acquire_lease(
        queue_conn, delivery, "dead", "2026-11-10T11:59:00+00:00", now="2026-11-10T11:58:00+00:00"
    )
    pause_platform(queue_conn, "facebook")
    drain_queue(queue_conn, {}, now=NOW, max_attempts=2)
    state = row(queue_conn, record)
    assert state["status"] == "uploaded" and state["attempt_count"] == 20
    assert state["lease_owner"] is None


def test_legacy_checkpoint_keeps_handler_dict_contract_and_unknown_outcome(queue_conn):
    record, _ = seed(queue_conn, status="uploaded", attempts=20)
    received = []

    def unavailable_media(item):
        received.append(item)
        raise PermanentDeliveryError("record no longer available")

    outcome = drain_queue(queue_conn, {"facebook": unavailable_media}, now=NOW, max_attempts=2)
    assert len(received) == 1 and isinstance(received[0], dict)
    assert outcome.failed_permanent == 0 and outcome.retried == 1
    assert row(queue_conn, record)["status"] == UNKNOWN


@freeze_time(NOW)
def test_drain_records_lost_finish_then_confirms_same_id_without_another_write(queue_conn):
    record = _record()
    delivery = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")
    session = Mock(
        post=Mock(
            side_effect=[
                _resp(200, {"video_id": "v-1", "upload_url": "https://rupload.facebook.com/x"}),
                _resp(200, {}),
                ConnectionError("lost finish response"),
            ]
        ),
        get=Mock(return_value=_resp(200, LIVE_STATUS_BODY)),
    )
    adapter = facebook.FacebookShortsAdapter(
        facebook.FacebookReelsAdapter(CREDS, session=session), worker_id="test"
    )

    def handler(_):
        adapter.publish(record, conn=queue_conn, delivery_id=delivery, video_bytes=b"x")

    first = drain_queue(queue_conn, {"facebook": handler}, owner="test", now=NOW, max_attempts=1)
    state = row(queue_conn, record)
    assert first.retried == 1 and first.failed_permanent == first.succeeded == 0
    assert state["status"] == UNKNOWN and state["platform_upload_id"] == "v-1"
    assert state["next_attempt_at"] > NOW and state["lease_owner"] is None
    session.get.assert_not_called()
    second = drain_queue(
        queue_conn,
        {"facebook": handler},
        owner="test",
        now=state["next_attempt_at"],
        max_attempts=1,
    )
    assert second.succeeded == 1 and row(queue_conn, record)["platform_post_id"] == "v-1"
    assert session.post.call_count == 3 and session.get.call_count == 1


def test_worker_exit_during_finish_leaves_durable_checkpoint(queue_conn):
    record = _record()
    delivery = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")
    reels = Mock()
    reels.account_id = "page-1"
    reels.start_upload.return_value = facebook.UploadHandle("v-1", "https://rupload.facebook.com/x")
    reels.finish_upload.side_effect = KeyboardInterrupt()
    adapter = facebook.FacebookShortsAdapter(reels)
    with pytest.raises(KeyboardInterrupt):
        adapter.publish(record, conn=queue_conn, delivery_id=delivery, video_bytes=b"x")
    assert row(queue_conn, record)["status"] == UNKNOWN
    assert row(queue_conn, record)["platform_upload_id"] == "v-1"
    reels.check_video_status.return_value = {"published": False}
    with pytest.raises(facebook.FacebookRetryableError):
        adapter.publish(record, conn=queue_conn, delivery_id=delivery, video_bytes=b"x")
    assert reels.start_upload.call_count == reels.finish_upload.call_count == 1
    reels.check_video_status.assert_called_once_with("v-1")


@pytest.mark.parametrize("stored_account", [None, "other-page"])
def test_reconcile_holds_missing_or_mismatched_destination_without_relabeling(
    queue_conn, stored_account
):
    record = _record()
    delivery = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")
    fields = {"upload_account": stored_account} if stored_account else {}
    transition(queue_conn, delivery, "queued", UNKNOWN, platform_upload_id="v-1", **fields)
    session = Mock(get=Mock(return_value=_resp(200, LIVE_STATUS_BODY)), post=Mock())
    adapter = facebook.FacebookShortsAdapter(facebook.FacebookReelsAdapter(CREDS, session=session))
    with pytest.raises(facebook.FacebookRetryableError, match="account"):
        adapter.publish(record, conn=queue_conn, delivery_id=delivery, video_bytes=b"x")
    state = row(queue_conn, record)
    assert state["status"] == UNKNOWN and state["upload_account"] == stored_account
    assert state["destination_account"] is None
    assert state["platform_post_id"] is None
    session.get.assert_not_called()
    session.post.assert_not_called()


def test_destination_is_checkpointed_before_start_and_retained_after_lost_finish(queue_conn):
    record = _record()
    delivery = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")
    reels = Mock(account_id="page-1")

    def start():
        assert row(queue_conn, record)["upload_account"] == "page-1"
        assert row(queue_conn, record)["destination_account"] is None
        return facebook.UploadHandle("v-1", "https://rupload.facebook.com/x")

    reels.start_upload.side_effect = start
    reels.finish_upload.side_effect = ConnectionError("lost response")
    with pytest.raises(facebook.FacebookRetryableError, match="unconfirmed"):
        facebook.FacebookShortsAdapter(reels).publish(
            record, conn=queue_conn, delivery_id=delivery, video_bytes=b"x"
        )
    assert row(queue_conn, record)["upload_account"] == "page-1"
    assert row(queue_conn, record)["destination_account"] is None
