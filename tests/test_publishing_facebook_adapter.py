"""Tests for the Facebook Reels adapter: no network (Mock session), no real SOPS call."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from test_publishing_models import _record

from gomoveshift.publishing.facebook_adapter import (
    FacebookAuthBlockedError,
    FacebookPermanentError,
    FacebookReelsAdapter,
    FacebookRetryableError,
    FacebookShortsAdapter,
    ScheduleWindowError,
    UnsafeUploadHostError,
    UploadHandle,
    _raise_for_graph_error,
)
from gomoveshift.publishing.models import FacebookPayload
from gomoveshift.publishing.queue import (
    PUBLISHED_AT_PLATFORM_RECONCILED as PLATFORM,
    enqueue,
    find_delivery,
    open_queue_db,
    transition,
)

FAKE_TOKEN = "EAABsecrettoken12345"  # noqa: S105 — test fixture, not a real credential
CREDS = {"page_id": "page-1", "page_access_token": FAKE_TOKEN, "graph_version": "v21.0"}
HANDLE = UploadHandle(video_id="v-1", upload_url="https://rupload.facebook.com/x")

#: Shape of a Graph v21.0 ``GET /<video-id>`` answer for a published video.
#: Two things a hand-written fixture tends to get wrong: ``video_status`` reads ``ready``
#: and NEVER ``PUBLISHED`` (the publish outcome lives under
#: ``publishing_phase.publish_status``), and ``permalink_url`` comes back
#: RELATIVE. Both are load-bearing — see the adapter's reconcile path.
LIVE_STATUS_BODY = {
    "id": "v-1",
    "status": {
        "video_status": "ready",
        "uploading_phase": {"status": "complete", "bytes_transferred": 59742853},
        "processing_phase": {"status": "complete"},
        "publishing_phase": {"status": "complete", "publish_status": "published"},
        "copyright_check_status": {"status": "complete", "matches_found": False},
    },
    "permalink_url": "/reel/v-1/",
    "post_id": "100000000000002_100000000000003",
    "published": True,
}
REEL_URL = "https://www.facebook.com/reel/v-1/"


def _resp(status_code=200, body=None):
    return Mock(status_code=status_code, json=Mock(return_value=body or {}))


def test_start_upload_finish_sends_description_and_video_state():
    session = Mock(
        post=Mock(
            side_effect=[
                _resp(200, {"video_id": "v-1", "upload_url": "https://rupload.facebook.com/x"}),
                _resp(200, {}),
                _resp(200, {"success": True, "post_id": "p-1"}),
            ]
        )
    )
    adapter = FacebookReelsAdapter(CREDS, session=session)
    start = adapter.start_upload()
    assert start.video_id == "v-1"
    adapter.upload_video(start.upload_url, b"bytes")
    result = adapter.finish_upload(
        start, FacebookPayload(description="hello world"), video_state="DRAFT"
    )
    assert result["video_id"] == "v-1"
    _, kwargs = session.post.call_args
    assert kwargs["data"]["description"] == "hello world"
    assert kwargs["data"]["video_state"] == "DRAFT"


@pytest.mark.parametrize("lead", [timedelta(minutes=5), timedelta(days=30)])
def test_out_of_window_scheduled_time_is_refused(lead):
    session = Mock()
    adapter = FacebookReelsAdapter(CREDS, session=session)
    with pytest.raises(ScheduleWindowError):
        adapter.finish_upload(
            HANDLE,
            FacebookPayload(description="x"),
            video_state="SCHEDULED",
            scheduled_publish_time=datetime.now(timezone.utc) + lead,
        )
    session.post.assert_not_called()


@pytest.mark.parametrize(
    "video_state,scheduled_at,expect_key",
    [
        ("SCHEDULED", lambda: datetime.now(timezone.utc) + timedelta(hours=1), True),
        ("DRAFT", lambda: None, False),
    ],
)
def test_scheduled_key_present_only_when_scheduled(video_state, scheduled_at, expect_key):
    session = Mock(post=Mock(return_value=_resp(200, {"success": True, "post_id": "p-1"})))
    adapter = FacebookReelsAdapter(CREDS, session=session)
    adapter.finish_upload(
        HANDLE,
        FacebookPayload(description="x"),
        video_state=video_state,
        scheduled_publish_time=scheduled_at(),
    )
    _, kwargs = session.post.call_args
    assert kwargs["data"]["video_state"] == video_state
    assert ("scheduled_publish_time" in kwargs["data"]) is expect_key


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://rupload.facebook.com/x",
        "https://evil.example.com/x",
        "https://rupload.facebook.com.evil.com/x",
        "https://user@evil.com/x",
    ],
)
def test_bad_upload_urls_refused_before_any_network_call(bad_url):
    session = Mock()
    adapter = FacebookReelsAdapter(CREDS, session=session)
    with pytest.raises(UnsafeUploadHostError):
        adapter.upload_video(bad_url, b"bytes")
    session.post.assert_not_called()


@pytest.mark.parametrize(
    "code,expected_exc",
    [
        (190, FacebookAuthBlockedError),
        (4, FacebookRetryableError),
        (9007, FacebookPermanentError),
    ],
)
def test_error_classification_maps_to_expected_exception(code, expected_exc):
    with pytest.raises(expected_exc):
        _raise_for_graph_error(400, {"error": {"code": code, "message": "x"}})


def test_no_token_leaks_into_raised_error_or_logs(caplog):
    caplog.set_level(logging.DEBUG)
    body = {"error": {"code": 190, "message": f"token {FAKE_TOKEN} expired"}}
    with pytest.raises(FacebookAuthBlockedError) as exc_info:
        _raise_for_graph_error(400, body)
    assert FAKE_TOKEN not in str(exc_info.value)
    session = Mock(
        post=Mock(return_value=_resp(200, {"video_id": "v-1", "upload_url": "https://x"}))
    )
    FacebookReelsAdapter(CREDS, session=session).start_upload()
    for record in caplog.records:
        assert FAKE_TOKEN not in record.getMessage()


def test_check_video_status_reads_the_publishing_phase_not_video_status():
    """A live reel reports ``video_status: ready`` forever. Keying "published"
    off that field made ``_reconcile`` blind to a reel Meta had already
    published, so a retry after a lost finish response would finish it a
    SECOND time — the duplicate this reconcile path exists to prevent."""
    session = Mock(get=Mock(return_value=_resp(200, LIVE_STATUS_BODY)))
    adapter = FacebookReelsAdapter(CREDS, session=session)
    result = adapter.check_video_status("v-1")
    assert result == {"published": True, "permalink": REEL_URL}
    _, kwargs = session.get.call_args
    assert kwargs["params"]["fields"] == "status,permalink_url"


def test_check_video_status_is_not_published_while_publishing_is_incomplete():
    body = {
        "status": {
            "video_status": "processing",
            "publishing_phase": {"status": "in_progress"},
        }
    }
    session = Mock(get=Mock(return_value=_resp(200, body)))
    adapter = FacebookReelsAdapter(CREDS, session=session)
    assert adapter.check_video_status("v-1") == {"published": False, "permalink": REEL_URL}


# --- Durable queue wiring ---


@pytest.fixture
def queue_conn(tmp_path):
    conn = open_queue_db(tmp_path / "seo")
    try:
        yield conn
    finally:
        conn.close()


def test_wiring_persists_video_id_and_permalink(queue_conn):
    """The permalink must land in the ``permalink`` COLUMN. ``transition``
    refuses every write once a row is terminal (queue.py), so a permalink
    fetched after the published transition can never be stored — a
    delivery once published to the page and the row kept
    ``permalink=NULL`` with no way to fill it. It is now derived from the
    video id and written in the same transition."""
    session = Mock(
        post=Mock(
            side_effect=[
                _resp(200, {"video_id": "v-1", "upload_url": "https://rupload.facebook.com/x"}),
                _resp(200, {}),
                _resp(200, {"success": True, "post_id": "p-1"}),
            ]
        ),
        get=Mock(),
    )
    reels = FacebookReelsAdapter(CREDS, session=session)
    adapter = FacebookShortsAdapter(reels, worker_id="w1")
    record = _record()
    delivery_id = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")

    result = adapter.publish(record, conn=queue_conn, delivery_id=delivery_id, video_bytes=b"x")

    assert result.video_id == "v-1"
    assert result.permalink == REEL_URL
    row = find_delivery(queue_conn, record.project_name, record.short_index, "facebook")
    assert row["status"] == "published"
    assert row["platform_upload_id"] == "v-1"
    assert row["platform_post_id"] == "v-1"  # bare video id, never packed JSON
    assert row["permalink"] == REEL_URL
    # Derived, not fetched: a cosmetic readback must never be able to turn a
    # completed publish into a failure.
    session.get.assert_not_called()


def test_finish_timeout_does_not_start_a_second_upload_session(queue_conn):
    first_session = Mock(
        post=Mock(
            side_effect=[
                _resp(200, {"video_id": "v-1", "upload_url": "https://rupload.facebook.com/x"}),
                _resp(200, {}),
                ConnectionError("lost response after Meta accepted the finish call"),
            ]
        ),
    )
    reels_first = FacebookReelsAdapter(CREDS, session=first_session)
    adapter_first = FacebookShortsAdapter(reels_first, worker_id="w1")
    record = _record()
    delivery_id = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")

    with pytest.raises(ConnectionError):
        adapter_first.publish(record, conn=queue_conn, delivery_id=delivery_id, video_bytes=b"x")
    start_calls = [
        c
        for c in first_session.post.call_args_list
        if isinstance(c.kwargs.get("data"), dict)
        and c.kwargs["data"].get("upload_phase") == "start"
    ]
    assert len(start_calls) == 1

    # Retry: the finish call actually succeeded server-side; the reel now
    # reports a complete publishing phase. No second upload session may be
    # started, and no second finish call.
    second_session = Mock(post=Mock(), get=Mock(return_value=_resp(200, LIVE_STATUS_BODY)))
    reels_second = FacebookReelsAdapter(CREDS, session=second_session)
    adapter_second = FacebookShortsAdapter(reels_second, worker_id="w1")

    result = adapter_second.publish(
        record, conn=queue_conn, delivery_id=delivery_id, video_bytes=b"x"
    )

    assert result.video_id == "v-1"
    assert result.permalink == REEL_URL
    second_session.post.assert_not_called()  # no second /video_reels start OR finish call
    row = find_delivery(queue_conn, record.project_name, record.short_index, "facebook")
    assert row["status"] == "published"
    assert row["permalink"] == REEL_URL


def test_realistic_full_envelope_drives_publish_and_caller(queue_conn):
    """Full-shape ``video_reels`` bodies — including a ``post_id`` field
    alongside ``success`` that this adapter deliberately never surfaces
    (see ``FacebookShortsAdapter`` docstring: the bare video id is the
    durable post id, never packed JSON) — drive both the low-level calls
    and their immediate caller. All three phases return flat single-object
    bodies per
    https://developers.facebook.com/docs/video-api/guides/reels-publishing
    (start/finish) and the standard Graph API single-object GET shape for
    ``check_video_status`` — never a ``data``-wrapped list envelope."""
    session = Mock(
        post=Mock(
            side_effect=[
                _resp(200, {"video_id": "v-1", "upload_url": "https://rupload.facebook.com/x"}),
                _resp(200, {}),
                _resp(200, {"success": True, "post_id": "p-1"}),
            ]
        ),
        get=Mock(return_value=_resp(200, LIVE_STATUS_BODY)),
    )
    reels = FacebookReelsAdapter(CREDS, session=session)
    adapter = FacebookShortsAdapter(reels, worker_id="w1")
    record = _record()
    delivery_id = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")

    result = adapter.publish(record, conn=queue_conn, delivery_id=delivery_id, video_bytes=b"x")

    assert result.video_id == "v-1"
    assert result.permalink == "https://www.facebook.com/reel/v-1/"


def test_wiring_reconciliation_skips_republish_when_post_id_already_persisted(queue_conn):
    session = Mock(post=Mock(), get=Mock())
    reels = FacebookReelsAdapter(CREDS, session=session)
    adapter = FacebookShortsAdapter(reels, worker_id="w1")
    record = _record()
    delivery_id = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")
    transition(queue_conn, delivery_id, "queued", "submitting")
    transition(queue_conn, delivery_id, "submitting", "uploaded", platform_upload_id="v-1")
    transition(
        queue_conn,
        delivery_id,
        "uploaded",
        "published",
        platform_post_id="v-1",
        published_at_source=PLATFORM,
    )

    result = adapter.publish(record, conn=queue_conn, delivery_id=delivery_id, video_bytes=b"x")

    assert result.video_id == "v-1"
    assert result.permalink is None  # not persisted; not re-fetched for a confirmed delivery
    session.post.assert_not_called()
    session.get.assert_not_called()  # already-published delivery: no live call needed at all


def test_http_200_with_error_body_is_classified_not_returned():
    """Same 200-with-error hazard as Instagram: without classifying it,
    start_upload returns a body with no video_id and the caller dies on a
    bare KeyError, which the dispatcher retries forever."""
    session = Mock()
    session.post.return_value = _resp(
        200,
        {
            "error": {
                "message": "Page does not have permission",
                "type": "OAuthException",
                "code": 200,
            }
        },
    )
    adapter = FacebookReelsAdapter(CREDS, session=session)
    with pytest.raises(FacebookPermanentError):
        adapter.start_upload()


def test_permanent_graph_error_leads_with_code_subcode_and_fbtrace():
    body = {"error": {"code": 100, "error_subcode": 33, "message": "m", "fbtrace_id": "SYNTHfb01"}}
    with pytest.raises(FacebookPermanentError) as exc_info:
        _raise_for_graph_error(400, body)
    assert str(exc_info.value).startswith("[graph http=400 code=100 subcode=33 fbtrace=SYNTHfb01] ")
