"""A delivery that starts over (``queue.restart_delivery``) gets a fresh
attempt budget. Before, a Facebook/Instagram row that started over kept the
attempt count its earlier status reads had run up, so the fresh upload failed
permanently on its first ordinary error."""

from datetime import datetime
from unittest.mock import Mock

import requests
from freezegun import freeze_time
import pytest
from test_publishing_facebook_adapter import CREDS, _resp
from test_publishing_facebook_reconcile import UNSENT_STATUS_BODY, _fresh_upload_post
from test_publishing_instagram_adapter import _Clock, _FakeReels
from test_publishing_models import _record

from gomoveshift.publishing import handlers, queue
from gomoveshift.publishing import facebook_adapter as facebook
from gomoveshift.publishing.dispatch import drain_queue
from gomoveshift.publishing.instagram_adapter import InstagramShortsAdapter
from gomoveshift.publishing.queue import enqueue, open_queue_db, transition

T0 = datetime.fromisoformat("2026-11-10T12:00:00+00:00")
FB_UNKNOWN = "awaiting_facebook_reconcile"


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _row(conn, delivery_id):
    return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())


def _seed(conn, platform, *, status="queued", upload_id=None, attempts=0):
    record = _record()
    delivery = enqueue(conn, record.project_name, record.short_index, platform, "h1")
    fields = {"attempt_count": attempts}
    if upload_id:
        fields.update(platform_upload_id=upload_id, upload_account="page-1")
    transition(conn, delivery, "queued", status, **fields)
    return delivery


def _raises(make):
    def handler(_row):
        raise make()

    return handler


def test_a_restarted_delivery_gets_a_fresh_attempt_budget(conn):
    delivery = _seed(conn, "instagram", status="uploaded", upload_id="c-1", attempts=20)
    assert queue.restart_delivery(conn, delivery, "uploaded")
    flaky = _raises(lambda: ConnectionError("blip"))
    result = drain_queue(conn, {"instagram": flaky}, now=T0.isoformat(), max_attempts=8)
    state = _row(conn, delivery)
    assert result.retried == 1 and result.failed_permanent == 0
    assert (state["status"], state["attempt_count"], state["platform_upload_id"]) == (
        "queued",
        1,
        None,
    )


@freeze_time(T0)
def test_facebook_start_over_does_not_inherit_the_unknown_reads_attempts(conn):
    delivery = _seed(conn, "facebook", status=FB_UNKNOWN, upload_id="v-1", attempts=20)
    seen = []
    broke = requests.exceptions.ConnectionError("write timed out")
    session = Mock(
        post=Mock(side_effect=_fresh_upload_post(seen, bytes_reply=broke)),
        get=Mock(return_value=_resp(200, UNSENT_STATUS_BODY)),
    )
    adapter = facebook.FacebookShortsAdapter(
        facebook.FacebookReelsAdapter(CREDS, session=session), worker_id="w1"
    )
    record = _record()

    def handler(_row):
        adapter.publish(record, conn=conn, delivery_id=delivery, video_bytes=b"x")

    result = drain_queue(
        conn, {"facebook": handler}, owner="w1", now=T0.isoformat(), max_attempts=8
    )
    state = _row(conn, delivery)
    assert seen == ["start", None]  # the fresh upload ran and its bytes broke off
    assert result.retried == 1 and result.failed_permanent == 0
    assert (state["status"], state["attempt_count"]) == ("queued", 1)


class _ExpiredThenOffline(_FakeReels):
    def create_container(self, payload):
        raise ConnectionError("graph unreachable")


def _instagram(conn, reels, *, attempts):
    delivery = _seed(conn, "instagram", status="uploaded", upload_id="c-old", attempts=attempts)
    clock = _Clock()
    adapter = InstagramShortsAdapter(
        reels, worker_id="w1", sleep=clock.sleep, monotonic=clock.monotonic
    )
    record = _record()
    handler = handlers.make_instagram_handler(adapter, conn, lambda *_: record, lambda _r: b"x")
    return delivery, handler


@freeze_time(T0)
def test_instagram_expired_start_over_does_not_inherit_old_attempts(conn, monkeypatch):
    monkeypatch.setattr("gomoveshift.publishing.instagram_adapter.make_in_spec_bytes", lambda b: b)
    delivery, handler = _instagram(conn, _ExpiredThenOffline("EXPIRED"), attempts=20)
    result = drain_queue(conn, {"instagram": handler}, owner="w1", now=T0.isoformat())
    state = _row(conn, delivery)
    assert result.retried == 1 and result.failed_permanent == 0
    assert state["attempt_count"] == 1 and state["platform_upload_id"] is None
