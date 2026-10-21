"""Tests for the Instagram Reels adapter: no network (Mock session), no real SOPS call."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
import requests
from test_publishing_models import _record

from gomoveshift.publishing.instagram_adapter import (
    CONTAINER_READY_POLL_S,
    CONTAINER_READY_TIMEOUT_S,
    LEASE_SEC,
    ContainerHandle,
    InstagramAuthBlockedError,
    InstagramPermanentError,
    InstagramPublishingLimitError,
    InstagramReelsAdapter,
    InstagramRetryableError,
    InstagramShortsAdapter,
    UnsafeUploadHostError,
    _raise_for_graph_error,
)
from gomoveshift.publishing.models import InstagramPayload
from gomoveshift.publishing.queue import (
    PUBLISHED_AT_PLATFORM_RECONCILED as PLATFORM,
    acquire_lease,
    enqueue,
    find_delivery,
    open_queue_db,
    transition,
)

FAKE_TOKEN = "EAABsecrettoken12345"  # noqa: S105 — test fixture, not a real credential
CREDS = {"ig_user_id": "ig-1", "page_access_token": FAKE_TOKEN, "graph_version": "v21.0"}


def _resp(status_code=200, body=None):
    return Mock(status_code=status_code, json=Mock(return_value=body or {}))


def _quota_resp(quota_usage, quota_total=100, quota_duration=86400):
    return _resp(
        200,
        {
            "data": [
                {
                    "quota_usage": quota_usage,
                    "config": {"quota_total": quota_total, "quota_duration": quota_duration},
                }
            ]
        },
    )


_FINISHED = _resp(200, {"id": "c-1", "status_code": "FINISHED"})


def _fresh_container():
    return ContainerHandle("c-1", "https://rupload.facebook.com/x", datetime.now(timezone.utc))


def _publish_session(get_response):
    return Mock(
        get=Mock(return_value=get_response),
        post=Mock(return_value=_resp(200, {"id": "m-1", "permalink": "https://x"})),
    )


def test_create_container_sends_caption_and_reels_type():
    session = Mock(post=Mock(return_value=_resp(200, {"id": "c-1", "uri": "https://x"})))
    handle = InstagramReelsAdapter(CREDS, session=session).create_container(
        InstagramPayload(caption="hello world")
    )
    assert handle.container_id == "c-1"
    _, kwargs = session.post.call_args
    assert kwargs["data"]["media_type"] == "REELS"
    assert kwargs["data"]["caption"] == "hello world"
    assert kwargs["data"]["upload_type"] == "resumable"


def test_refusals_happen_before_any_network_call():
    session = Mock()
    adapter = InstagramReelsAdapter(CREDS, session=session)
    for bad_url in ("http://rupload.facebook.com/x", "https://evil.example.com/x"):
        with pytest.raises(UnsafeUploadHostError):
            adapter.upload_video(bad_url, b"bytes")
    stale = ContainerHandle(
        "c-1", "https://rupload.facebook.com/x", datetime.now(timezone.utc) - timedelta(hours=25)
    )
    with pytest.raises(InstagramPermanentError):
        adapter.publish_container(stale)
    session.post.assert_not_called()


@pytest.mark.parametrize(
    "code,expected_exc",
    [
        (190, InstagramAuthBlockedError),
        (4, InstagramRetryableError),
        (9007, InstagramPermanentError),
    ],
)
def test_error_classification_maps_to_expected_exception(code, expected_exc):
    with pytest.raises(expected_exc):
        _raise_for_graph_error(400, {"error": {"code": code, "message": "x"}})


def test_no_token_leaks_into_raised_error_or_logs(caplog):
    caplog.set_level(logging.DEBUG)
    body = {"error": {"code": 190, "message": f"token {FAKE_TOKEN} expired"}}
    with pytest.raises(InstagramAuthBlockedError) as exc_info:
        _raise_for_graph_error(400, body)
    assert FAKE_TOKEN not in str(exc_info.value)

    session = Mock(post=Mock(return_value=_resp(200, {"id": "c-1", "uri": "https://x"})))
    InstagramReelsAdapter(CREDS, session=session).create_container(InstagramPayload("hi"))
    for record in caplog.records:
        assert FAKE_TOKEN not in record.getMessage()


def test_publish_proceeds_when_under_quota():
    session = _publish_session(_quota_resp(quota_usage=4))
    adapter = InstagramReelsAdapter(CREDS, session=session)
    result = adapter.publish_container(_fresh_container())
    assert result["media_id"] == "m-1"
    first_get_kwargs = session.get.call_args_list[0].kwargs
    assert first_get_kwargs["params"]["fields"] == "quota_usage,config"


def test_publish_proceeds_at_99_of_100_no_headroom_reserved():
    # Decision: no headroom is reserved — refusal triggers only at usage >= total,
    # matching the platform's own semantics (a 100th publish is still legal).
    session = _publish_session(_quota_resp(quota_usage=99))
    adapter = InstagramReelsAdapter(CREDS, session=session)
    result = adapter.publish_container(_fresh_container())
    assert result["media_id"] == "m-1"


def test_publish_refused_at_quota_limit():
    session = _publish_session(_quota_resp(quota_usage=100))
    adapter = InstagramReelsAdapter(CREDS, session=session)
    with pytest.raises(InstagramPublishingLimitError) as exc_info:
        adapter.publish_container(_fresh_container())
    message = str(exc_info.value)
    assert "100" in message
    assert "0" in message  # remaining count
    session.post.assert_not_called()
    assert isinstance(exc_info.value, InstagramRetryableError)  # self-heals, retryable


def test_publish_refusal_message_has_no_token():
    session = _publish_session(_quota_resp(quota_usage=100))
    adapter = InstagramReelsAdapter(CREDS, session=session)
    with pytest.raises(InstagramPublishingLimitError) as exc_info:
        adapter.publish_container(_fresh_container())
    assert FAKE_TOKEN not in str(exc_info.value)


def test_quota_is_reread_on_every_publish_never_cached():
    session = _publish_session(_quota_resp(quota_usage=4))
    adapter = InstagramReelsAdapter(CREDS, session=session)
    adapter.publish_container(_fresh_container())
    adapter.publish_container(_fresh_container())
    quota_calls = [
        call
        for call in session.get.call_args_list
        if call.kwargs["params"]["fields"] == "quota_usage,config"
    ]
    assert len(quota_calls) == 2


def test_empty_quota_data_fails_closed():
    session = _publish_session(_resp(200, {"data": []}))
    adapter = InstagramReelsAdapter(CREDS, session=session)
    with pytest.raises(InstagramPublishingLimitError):
        adapter.publish_container(_fresh_container())
    session.post.assert_not_called()


def test_missing_quota_data_key_fails_closed():
    session = _publish_session(_resp(200, {}))
    adapter = InstagramReelsAdapter(CREDS, session=session)
    with pytest.raises(InstagramPublishingLimitError):
        adapter.publish_container(_fresh_container())
    session.post.assert_not_called()


# --- Durable queue wiring ---


@pytest.fixture
def queue_conn(tmp_path):
    conn = open_queue_db(tmp_path / "seo")
    try:
        yield conn
    finally:
        conn.close()


def test_wiring_persists_media_id_and_permalink_and_still_checks_quota(queue_conn):
    session = Mock(
        post=Mock(
            side_effect=[
                _resp(200, {"id": "c-1", "uri": "https://rupload.facebook.com/x"}),
                _resp(200, {}),
                _resp(200, {"id": "m-1"}),
            ]
        ),
        get=Mock(
            side_effect=[
                _FINISHED,
                _quota_resp(quota_usage=4),
                _resp(200, {"id": "m-1", "permalink": "https://instagram.com/reel/m-1"}),
            ]
        ),
    )
    reels = InstagramReelsAdapter(CREDS, session=session)
    adapter = InstagramShortsAdapter(reels, worker_id="w1")
    record = _record()
    delivery_id = enqueue(queue_conn, record.project_name, record.short_index, "instagram", "h1")

    result = adapter.publish(record, conn=queue_conn, delivery_id=delivery_id, video_bytes=b"x")

    assert result.media_id == "m-1"
    assert result.permalink == "https://instagram.com/reel/m-1"
    row = find_delivery(queue_conn, record.project_name, record.short_index, "instagram")
    assert row["status"] == "published"
    assert row["platform_upload_id"] == "c-1"
    assert row["platform_post_id"] == "m-1"
    assert row["permalink"] == "https://instagram.com/reel/m-1"
    status_get, quota_get = session.get.call_args_list[:2]
    assert status_get.kwargs["params"]["fields"] == "status_code,id,status"
    assert quota_get.kwargs["params"]["fields"] == "quota_usage,config"


def test_media_publish_timeout_does_not_create_second_container(queue_conn):
    first_session = Mock(
        post=Mock(
            side_effect=[
                _resp(200, {"id": "c-1", "uri": "https://rupload.facebook.com/x"}),
                _resp(200, {}),
                ConnectionError("lost response after Meta accepted the publish"),
            ]
        ),
        get=Mock(side_effect=[_FINISHED, _quota_resp(quota_usage=4)]),
    )
    reels_first = InstagramReelsAdapter(CREDS, session=first_session)
    adapter_first = InstagramShortsAdapter(reels_first, worker_id="w1")
    record = _record()
    delivery_id = enqueue(queue_conn, record.project_name, record.short_index, "instagram", "h1")

    with pytest.raises(ConnectionError):
        adapter_first.publish(record, conn=queue_conn, delivery_id=delivery_id, video_bytes=b"x")
    assert first_session.post.call_count == 3  # create, upload, failed publish — no retry here

    # Retry: the publish actually succeeded server-side; the container now
    # reports PUBLISHED. No second container may be created.
    second_session = Mock(
        post=Mock(),
        get=Mock(
            return_value=_resp(
                200,
                {
                    "status_code": "PUBLISHED",
                    "id": "m-1",
                    "permalink": "https://instagram.com/reel/m-1",
                },
            )
        ),
    )
    reels_second = InstagramReelsAdapter(CREDS, session=second_session)
    adapter_second = InstagramShortsAdapter(reels_second, worker_id="w1")

    result = adapter_second.publish(
        record, conn=queue_conn, delivery_id=delivery_id, video_bytes=b"x"
    )

    assert result.media_id == "m-1"
    second_session.post.assert_not_called()  # no second /media create_container call
    row = find_delivery(queue_conn, record.project_name, record.short_index, "instagram")
    assert row["status"] == "published"


def test_realistic_full_envelope_drives_publish_and_caller(queue_conn):
    """Full-shape Graph API bodies — including fields the adapter never
    touches (``paging``, ``id`` alongside ``permalink``) — drive both the
    low-level call and its immediate caller, so the ``data`` unwrap on
    ``content_publishing_limit`` is genuinely exercised rather than skipped
    by a hand-built already-unwrapped dict (see docs/social-publishing-meta.md;
    https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user/content_publishing_limit
    for the ``{"data": [...]}`` list-edge shape and
    https://developers.facebook.com/docs/instagram-platform/reference/instagram-media
    for the flat single-object container/publish shape)."""
    quota_body = {
        "data": [
            {
                "quota_usage": 4,
                "config": {"quota_total": 100, "quota_duration": 86400},
            }
        ],
        "paging": {"cursors": {"before": "QVFI", "after": "QVFI"}},
    }
    permalink_body = {
        "id": "m-1",
        "media_type": "REELS",
        "permalink": "https://www.instagram.com/reel/m-1/",
    }
    session = Mock(
        post=Mock(
            side_effect=[
                _resp(200, {"id": "c-1", "uri": "https://rupload.facebook.com/x"}),
                _resp(200, {}),
                _resp(200, {"id": "m-1"}),
            ]
        ),
        get=Mock(side_effect=[_FINISHED, _resp(200, quota_body), _resp(200, permalink_body)]),
    )
    reels = InstagramReelsAdapter(CREDS, session=session)
    adapter = InstagramShortsAdapter(reels, worker_id="w1")
    record = _record()
    delivery_id = enqueue(queue_conn, record.project_name, record.short_index, "instagram", "h1")

    result = adapter.publish(record, conn=queue_conn, delivery_id=delivery_id, video_bytes=b"x")

    assert result.media_id == "m-1"
    assert result.permalink == "https://www.instagram.com/reel/m-1/"


def test_wiring_reconciliation_skips_republish_when_post_id_already_persisted(queue_conn):
    session = Mock(post=Mock(), get=Mock())
    reels = InstagramReelsAdapter(CREDS, session=session)
    adapter = InstagramShortsAdapter(reels, worker_id="w1")
    record = _record()
    delivery_id = enqueue(queue_conn, record.project_name, record.short_index, "instagram", "h1")
    transition(queue_conn, delivery_id, "queued", "submitting")
    transition(queue_conn, delivery_id, "submitting", "uploaded", platform_upload_id="c-1")
    transition(
        queue_conn,
        delivery_id,
        "uploaded",
        "published",
        platform_post_id="m-1",
        permalink="https://instagram.com/reel/m-1",
        published_at_source=PLATFORM,
    )

    result = adapter.publish(record, conn=queue_conn, delivery_id=delivery_id, video_bytes=b"x")

    assert result.media_id == "m-1"
    assert result.permalink == "https://instagram.com/reel/m-1"
    session.post.assert_not_called()
    session.get.assert_not_called()  # already-published delivery: no live call needed at all


def test_http_200_with_error_body_is_classified_not_returned():
    """Graph can answer 200 with an `error` object. Gating only on the status
    code let that through as success — `create_container` then died on a bare
    KeyError('id'), which the dispatcher treats as RETRYABLE, so a permanently
    rejected post would retry forever."""
    session = Mock()
    session.post.return_value = _resp(
        200,
        {
            "error": {
                "message": "Media type not supported",
                "type": "OAuthException",
                "code": 100,
                "error_subcode": 2207026,
            }
        },
    )
    adapter = InstagramReelsAdapter(CREDS, session=session)
    with pytest.raises(InstagramPermanentError):
        adapter.create_container(InstagramPayload(caption="hello"))


# --- Container status readback (no `permalink` field on a container) ---


def test_check_container_status_never_requests_permalink_field():
    """A container object has no ``permalink`` field — Graph rejects a
    request for it with "Tried accessing nonexisting field
    (permalink)", classified as InstagramPermanentError. This is the
    post-timeout reconciliation path: an unrelated (possibly-succeeded)
    publish must never be misreported as failed_permanent because of a
    field the container doesn't support."""
    session = Mock(get=Mock(return_value=_resp(200, {"id": "c-1", "status_code": "IN_PROGRESS"})))
    adapter = InstagramReelsAdapter(CREDS, session=session)
    status = adapter.check_container_status("c-1")
    assert status["status_code"] == "IN_PROGRESS"
    _, kwargs = session.get.call_args
    assert "permalink" not in kwargs["params"]["fields"].split(",")


def test_code_100_permanent_error_message_does_not_assert_content_policy_cause():
    """Code 100 also fires on a plain invalid-field request (this defect:
    requesting the nonexistent container ``permalink`` field) — the message
    must not claim a content-policy cause it cannot know."""
    with pytest.raises(InstagramPermanentError) as exc_info:
        _raise_for_graph_error(
            400,
            {
                "error": {
                    "code": 100,
                    "message": "Tried accessing nonexisting field (permalink)",
                }
            },
        )
    assert "content policy" not in str(exc_info.value).lower()


def test_check_container_status_published_survives_permalink_lookup_failure():
    """A PUBLISHED container's follow-up permalink lookup can itself fail
    (e.g. a transient Graph error) — that must never turn a confirmed
    successful publish into a reported failure. Permalink is
    presentation-only."""
    session = Mock(
        get=Mock(
            side_effect=[
                _resp(200, {"id": "c-1", "status_code": "PUBLISHED"}),
                _resp(400, {"error": {"code": 100, "message": "x"}}),
            ]
        )
    )
    adapter = InstagramReelsAdapter(CREDS, session=session)
    status = adapter.check_container_status("c-1")
    assert status["status_code"] == "PUBLISHED"
    assert status["media_id"] == "c-1"
    assert status["permalink"] is None


def test_check_container_status_published_fetches_permalink_from_media_object():
    session = Mock(
        get=Mock(
            side_effect=[
                _resp(200, {"id": "c-1", "status_code": "PUBLISHED"}),
                _resp(200, {"id": "c-1", "permalink": "https://instagram.com/reel/c-1"}),
            ]
        )
    )
    adapter = InstagramReelsAdapter(CREDS, session=session)
    status = adapter.check_container_status("c-1")
    assert status["permalink"] == "https://instagram.com/reel/c-1"


def test_publish_container_survives_permalink_lookup_failure():
    """The permalink lookup after a successful media_publish can itself fail
    — that must not fail the overall publish; only the permalink is
    unknown."""
    session = Mock(
        post=Mock(return_value=_resp(200, {"id": "m-1"})),
        get=Mock(
            side_effect=[
                _quota_resp(quota_usage=4),
                _resp(400, {"error": {"code": 100, "message": "x"}}),
            ]
        ),
    )
    adapter = InstagramReelsAdapter(CREDS, session=session)
    result = adapter.publish_container(_fresh_container())
    assert result["media_id"] == "m-1"
    assert result["permalink"] is None


def test_publish_container_fetches_permalink_from_media_object_not_publish_response():
    """``media_publish``'s own response is not trusted for ``permalink`` —
    a follow-up GET on the published media id is the source of truth."""
    session = Mock(
        post=Mock(
            return_value=_resp(200, {"id": "m-1", "permalink": "https://bogus-should-be-ignored"})
        ),
        get=Mock(
            side_effect=[
                _quota_resp(quota_usage=4),
                _resp(200, {"id": "m-1", "permalink": "https://instagram.com/reel/m-1"}),
            ]
        ),
    )
    adapter = InstagramReelsAdapter(CREDS, session=session)
    result = adapter.publish_container(_fresh_container())
    assert result["permalink"] == "https://instagram.com/reel/m-1"


def test_permalink_lookup_survives_a_transport_error_not_just_a_graph_error():
    """The other permalink-failure tests feed a Graph ERROR RESPONSE, which
    raises InstagramPublishError. A transport failure is a different
    exception class entirely: ``_get`` does not wrap
    ``self._session.get``, so ``requests.ConnectionError`` propagates
    straight out. Catching only InstagramPublishError would let a connection
    reset on a presentation-only lookup fail a publish that already
    succeeded server-side."""
    session = Mock(
        get=Mock(
            side_effect=[
                _resp(200, {"id": "c-1", "status_code": "PUBLISHED"}),
                requests.ConnectionError("connection reset by peer"),
            ]
        )
    )
    adapter = InstagramReelsAdapter(CREDS, session=session)
    status = adapter.check_container_status("c-1")
    assert status["status_code"] == "PUBLISHED"
    assert status["permalink"] is None


def test_publish_container_permalink_lookup_survives_a_transport_error():
    """Same transport-error case on the publish path — a successful
    media_publish must survive a failed follow-up permalink GET."""
    session = Mock(
        post=Mock(return_value=_resp(200, {"id": "m-9"})),
        get=Mock(side_effect=requests.ReadTimeout("read timed out")),
    )
    adapter = InstagramReelsAdapter(CREDS, session=session)
    adapter._check_publishing_limit = lambda: None
    result = adapter.publish_container(
        ContainerHandle("c-1", "https://rupload.facebook.com/x", datetime.now(timezone.utc))
    )
    assert result["media_id"] == "m-9"
    assert result["permalink"] is None


# --- Publish only once the container is FINISHED --------------------------
#
# An Instagram delivery once called media_publish the instant the upload
# returned. Meta was still processing the video, so the call failed (code 9007,
# subcode 2207027, "media is not ready"), was read as permanent, and the row
# stayed 'uploaded' while its container turned FINISHED ~15 minutes later.


def test_media_not_ready_is_retryable_not_permanent():
    body = {"error": {"code": 9007, "error_subcode": 2207027, "message": "not ready"}}
    with pytest.raises(InstagramRetryableError):
        _raise_for_graph_error(400, body)
    # The same code with another subcode (or none) is still a permanent refusal.
    with pytest.raises(InstagramPermanentError):
        _raise_for_graph_error(400, {"error": {"code": 9007, "error_subcode": 1, "message": "x"}})


def test_every_get_sends_the_token_in_a_header_never_the_url():
    session = Mock(
        get=Mock(
            side_effect=[
                _resp(200, {"id": "c-1", "status_code": "FINISHED"}),
                _quota_resp(quota_usage=4),
                _resp(200, {"id": "m-1", "permalink": "https://x"}),
            ]
        ),
        post=Mock(return_value=_resp(200, {"id": "m-1"})),
    )
    adapter = InstagramReelsAdapter(CREDS, session=session)
    adapter.check_container_status("c-1")
    adapter.publish_container(_fresh_container())
    assert session.get.call_count == 3  # status, quota, permalink
    for call in session.get.call_args_list:
        assert "access_token" not in call.kwargs["params"]
        assert call.kwargs["headers"] == {"Authorization": f"OAuth {FAKE_TOKEN}"}


_T0 = datetime(2026, 10, 2, tzinfo=timezone.utc)  # a fixed moment, not the clock


class _Clock:
    """Injected ``sleep``/``monotonic``: sleeping advances time, never waits."""

    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class _FakeReels:
    """The reels client as the shorts adapter drives it. ``statuses`` are the
    ``status_code`` answers in order; the last one repeats."""

    account_id = "ig-1"

    def __init__(self, *statuses):
        self.statuses = list(statuses)
        self.created, self.checked, self.published = [], [], []

    def create_container(self, payload):
        self.created.append(payload)
        return ContainerHandle("c-new", "https://rupload.facebook.com/x", _T0)

    def upload_video(self, upload_url, video_bytes):
        pass

    def check_container_status(self, container_id):
        self.checked.append(container_id)
        code = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return {
            "status_code": code,
            "media_id": "m-done" if code == "PUBLISHED" else container_id,
            "permalink": "https://instagram.com/reel/m-done" if code == "PUBLISHED" else None,
            "status": "Error: video codec not supported" if code == "ERROR" else None,
        }

    def publish_container(self, container):
        self.published.append(container.container_id)
        return {"media_id": "m-1", "permalink": "https://instagram.com/reel/m-1"}


def _run(queue_conn, reels, *, resume_from=None):
    """Publish one record. ``resume_from`` seeds an 'uploaded' row holding that container."""
    clock = _Clock()
    adapter = InstagramShortsAdapter(
        reels, worker_id="w1", sleep=clock.sleep, monotonic=clock.monotonic
    )
    record = _record()
    delivery_id = enqueue(queue_conn, record.project_name, record.short_index, "instagram", "h1")
    if resume_from:
        transition(queue_conn, delivery_id, "queued", "submitting")
        transition(
            queue_conn, delivery_id, "submitting", "uploaded", platform_upload_id=resume_from
        )

    def publish():
        return adapter.publish(record, conn=queue_conn, delivery_id=delivery_id, video_bytes=b"x")

    def row():
        return find_delivery(queue_conn, record.project_name, record.short_index, "instagram")

    return publish, row, clock


def test_a_fresh_publish_waits_through_in_progress_then_publishes_once(queue_conn):
    reels = _FakeReels("IN_PROGRESS", "IN_PROGRESS", "FINISHED")
    publish, row, clock = _run(queue_conn, reels)
    assert publish().media_id == "m-1"
    assert reels.published == ["c-new"]
    assert clock.sleeps == [CONTAINER_READY_POLL_S] * 2
    assert row()["status"] == "published"


def test_a_resumed_finished_container_is_published(queue_conn):
    reels = _FakeReels("FINISHED")
    publish, row, _ = _run(queue_conn, reels, resume_from="c-old")
    assert publish().media_id == "m-1"
    assert reels.published == ["c-old"] and reels.created == []


def test_a_resumed_container_already_published_is_never_published_again(queue_conn):
    reels = _FakeReels("PUBLISHED")
    publish, row, _ = _run(queue_conn, reels, resume_from="c-old")
    result = publish()
    assert (result.media_id, result.permalink) == ("m-done", "https://instagram.com/reel/m-done")
    assert reels.published == [] and reels.created == []
    assert row()["status"] == "published"


def test_a_container_in_error_is_permanent_and_names_the_status_text(queue_conn):
    reels = _FakeReels("IN_PROGRESS", "ERROR")
    publish, _, _ = _run(queue_conn, reels)
    with pytest.raises(InstagramPermanentError, match="video codec not supported"):
        publish()
    assert reels.published == []


def test_an_expired_resumed_container_starts_over_with_a_fresh_one(queue_conn):
    reels = _FakeReels("EXPIRED", "FINISHED")
    publish, row, _ = _run(queue_conn, reels, resume_from="c-old")
    assert publish().media_id == "m-1"
    assert len(reels.created) == 1 and reels.published == ["c-new"]
    assert row()["status"] == "published"
    assert row()["platform_upload_id"] == "c-new"


def test_a_container_expired_right_after_create_is_permanent(queue_conn):
    reels = _FakeReels("EXPIRED")
    publish, _, _ = _run(queue_conn, reels)
    with pytest.raises(InstagramPermanentError, match="container expired"):
        publish()
    assert len(reels.created) == 1 and reels.published == []


def test_a_container_still_processing_at_the_bound_is_retryable(queue_conn):
    reels = _FakeReels("IN_PROGRESS")
    publish, row, clock = _run(queue_conn, reels)
    with pytest.raises(InstagramRetryableError):
        publish()
    assert reels.published == []
    assert CONTAINER_READY_TIMEOUT_S <= clock.now < CONTAINER_READY_TIMEOUT_S + 2 * 10
    assert CONTAINER_READY_TIMEOUT_S < LEASE_SEC
    # The container is kept, so the next pass resumes it instead of creating another.
    assert row()["platform_upload_id"] == "c-new" and row()["status"] == "uploaded"


def test_a_resumed_container_is_polled_under_a_renewed_lease(queue_conn):
    reels = _FakeReels("FINISHED")
    publish, row, _ = _run(queue_conn, reels, resume_from="c-old")
    assert acquire_lease(queue_conn, row()["id"], "w1", _T0.isoformat())  # long lapsed
    publish()
    assert reels.published == ["c-old"]
    # The renewal runs on the real clock, so it lands well past the lapsed moment
    # plus a full lease; published rows keep their lease columns until dispatch clears them.
    assert datetime.fromisoformat(row()["lease_expires_at"]) > _T0 + timedelta(seconds=LEASE_SEC)


def test_a_resumed_container_under_another_workers_live_lease_is_refused(queue_conn):
    reels = _FakeReels("FINISHED")
    publish, row, _ = _run(queue_conn, reels, resume_from="c-old")
    assert acquire_lease(queue_conn, row()["id"], "other", "2100-01-01T00:00:00+00:00")
    with pytest.raises(RuntimeError, match="leased by another worker"):
        publish()
    assert reels.published == [] and reels.created == []


def test_a_resumed_container_that_expires_while_waiting_starts_over(queue_conn):
    reels = _FakeReels("IN_PROGRESS", "EXPIRED", "FINISHED")
    publish, row, _ = _run(queue_conn, reels, resume_from="c-old")
    assert publish().media_id == "m-1"
    assert len(reels.created) == 1 and reels.published == ["c-new"]
    assert row()["platform_upload_id"] == "c-new" and row()["status"] == "published"


# --- The Graph error's own code/subcode/fbtrace survive into error_detail ----
#
# 2026-10-15: five Instagram rows sat failed_permanent with only "Media ID is
# not available" or "unknown Graph API error" stored. The code and subcode
# that classify the failure were dropped, so the cause could not be read back
# from the queue. The tag leads the message so error_detail's bound never
# cuts it.

_GRAPH_400 = {
    "error": {
        "code": 9007,
        "error_subcode": 2207006,
        "message": "Media ID is not available",
        "fbtrace_id": "SYNTHtrace01",
    }
}


def test_permanent_graph_error_leads_with_code_subcode_and_fbtrace():
    with pytest.raises(InstagramPermanentError) as exc_info:
        _raise_for_graph_error(400, _GRAPH_400)
    assert str(exc_info.value).startswith(
        "[graph http=400 code=9007 subcode=2207006 fbtrace=SYNTHtrace01] "
    )


def test_retryable_graph_error_carries_the_tag_too():
    body = {"error": {"code": 9007, "error_subcode": 2207027, "message": "not ready"}}
    with pytest.raises(InstagramRetryableError) as exc_info:
        _raise_for_graph_error(400, body)
    assert str(exc_info.value).startswith("[graph http=400 code=9007 subcode=2207027 fbtrace=-] ")


def test_stored_error_detail_keeps_the_tag_past_its_bound():
    from gomoveshift.publishing.dispatch import PermanentDeliveryError, _detail

    body = {"error": dict(_GRAPH_400["error"], message="x" * 2000)}
    with pytest.raises(InstagramPermanentError) as exc_info:
        _raise_for_graph_error(400, body)
    stored = _detail(PermanentDeliveryError(str(exc_info.value)))
    assert stored.startswith("[graph http=400 code=9007 subcode=2207006 fbtrace=SYNTHtrace01] ")


def test_fbtrace_from_the_response_is_sanitized_and_bounded():
    body = {"error": {"code": 1, "message": "m", "fbtrace_id": "ab\ncd]" + "z" * 200}}
    with pytest.raises(InstagramPermanentError) as exc_info:
        _raise_for_graph_error(400, body)
    tag = str(exc_info.value).split("] ", 1)[0]
    assert "\n" not in tag
    assert tag == "[graph http=400 code=1 subcode=- fbtrace=" + ("abcd" + "z" * 200)[:64]


def test_an_error_body_meta_left_empty_still_records_the_http_status():
    # Rows 58/66/77 stored only "unknown Graph API error": a 4xx whose body
    # had no error object. The HTTP status is then the only evidence left.
    with pytest.raises(InstagramPermanentError) as exc_info:
        _raise_for_graph_error(403, {})
    assert str(exc_info.value).startswith("[graph http=403 code=- subcode=- fbtrace=-] ")
