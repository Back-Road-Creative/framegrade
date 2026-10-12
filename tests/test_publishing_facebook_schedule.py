"""A Facebook delivery honours its own schedule, and the receipt says so.

Two halves of one gap, mirroring the YouTube fix:

1. The adapter took `scheduled_publish_time` only as an explicit argument and
   `make_facebook_handler` never passes one, so `record.schedule` -- the
   instant the scheduler picked for this short -- was ignored and every
   Facebook reel published the moment the queue drained it.
2. When a caller DID schedule, the row still recorded
   `published_at_source=platform_confirmed`, whose written contract in
   `queue.py` is "publish call returned success -- seconds". Facebook had
   merely accepted a reel it would publish later.

The bound below `MIN_SCHEDULE_LEAD` is deliberate: the queue drains at or
just after the scheduled instant, so "less than ten minutes away" means the
moment has arrived and publishing now IS honouring the schedule. Refusing
those would fail every normal delivery, since `ScheduleWindowError` is
permanent.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from test_publishing_models import _record

from gomoveshift.publishing.facebook_adapter import (
    MIN_SCHEDULE_LEAD,
    FacebookReelsAdapter,
    FacebookShortsAdapter,
    due_schedule,
)
from gomoveshift.publishing.models import ScheduleWindow
from gomoveshift.publishing.queue import (
    PUBLISHED_AT_PLATFORM_CONFIRMED,
    PUBLISHED_AT_PLATFORM_SCHEDULED,
    enqueue,
    find_delivery,
    open_queue_db,
)

CREDS = {"page_id": "page-1", "page_access_token": "EAABsecrettoken12345", "graph_version": "v21.0"}  # noqa: S105,E501


def _resp(status_code=200, body=None):
    return Mock(status_code=status_code, json=Mock(return_value=body or {}))


def _session():
    return Mock(
        post=Mock(
            side_effect=[
                _resp(200, {"video_id": "v-1", "upload_url": "https://rupload.facebook.com/x"}),
                _resp(200, {}),
                _resp(200, {"success": True, "post_id": "p-1"}),
            ]
        ),
        get=Mock(),
    )


@pytest.fixture()
def queue_conn(tmp_path):
    conn = open_queue_db(tmp_path / "seo")
    try:
        yield conn
    finally:
        conn.close()


def _scheduled_record(when: datetime, *, per_platform: datetime | None = None):
    by_platform = {"facebook": per_platform} if per_platform else {}
    return replace(
        _record(),
        schedule=ScheduleWindow(
            scheduled_at_utc=when,
            display_timezone="America/New_York",
            scheduled_at_utc_by_platform=by_platform,
        ),
    )


def _publish(queue_conn, record):
    reels = FacebookReelsAdapter(CREDS, session=_session())
    adapter = FacebookShortsAdapter(reels, worker_id="w1")
    delivery_id = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")
    adapter.publish(record, conn=queue_conn, delivery_id=delivery_id, video_bytes=b"x")
    return find_delivery(queue_conn, record.project_name, record.short_index, "facebook"), reels


# --- half 1: the schedule reaches the platform -----------------------------


def test_a_future_schedule_is_handed_to_facebook(queue_conn):
    when = datetime.now(timezone.utc) + timedelta(hours=6)
    session = _session()
    reels = FacebookReelsAdapter(CREDS, session=session)
    adapter = FacebookShortsAdapter(reels, worker_id="w1")
    record = _scheduled_record(when)
    delivery_id = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")

    adapter.publish(record, conn=queue_conn, delivery_id=delivery_id, video_bytes=b"x")

    _, kwargs = session.post.call_args
    assert kwargs["data"]["video_state"] == "SCHEDULED"
    assert kwargs["data"]["scheduled_publish_time"] == int(when.timestamp())


def test_the_platforms_own_slot_wins_over_the_shared_one(queue_conn):
    """A per-platform hour is resolved; using the shared slot would publish
    at a time the scheduler deliberately moved this platform off."""
    shared = datetime.now(timezone.utc) + timedelta(hours=6)
    facebook = datetime.now(timezone.utc) + timedelta(hours=9)

    assert due_schedule(_scheduled_record(shared, per_platform=facebook)) == facebook


def test_a_schedule_that_has_arrived_publishes_now(queue_conn):
    """The normal case. The queue drains at or just after the slot, so a lead
    under MIN_SCHEDULE_LEAD means the moment is here."""
    when = datetime.now(timezone.utc) + (MIN_SCHEDULE_LEAD / 2)
    session = _session()
    reels = FacebookReelsAdapter(CREDS, session=session)
    adapter = FacebookShortsAdapter(reels, worker_id="w1")
    record = _scheduled_record(when)
    delivery_id = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")

    adapter.publish(record, conn=queue_conn, delivery_id=delivery_id, video_bytes=b"x")

    _, kwargs = session.post.call_args
    assert kwargs["data"]["video_state"] == "PUBLISHED"
    assert "scheduled_publish_time" not in kwargs["data"]


def test_a_schedule_already_past_publishes_now(queue_conn):
    assert due_schedule(_scheduled_record(datetime.now(timezone.utc) - timedelta(days=2))) is None


def test_an_explicit_argument_still_wins(queue_conn):
    """The parameter is an override, not a leftover: a caller that names a
    state and a time keeps them."""
    when = datetime.now(timezone.utc) + timedelta(hours=6)
    session = _session()
    reels = FacebookReelsAdapter(CREDS, session=session)
    adapter = FacebookShortsAdapter(reels, worker_id="w1")
    record = _scheduled_record(datetime.now(timezone.utc) + timedelta(hours=99))
    delivery_id = enqueue(queue_conn, record.project_name, record.short_index, "facebook", "h1")

    adapter.publish(
        record,
        conn=queue_conn,
        delivery_id=delivery_id,
        video_bytes=b"x",
        video_state="SCHEDULED",
        scheduled_publish_time=when,
    )

    _, kwargs = session.post.call_args
    assert kwargs["data"]["scheduled_publish_time"] == int(when.timestamp())


# --- half 2: the receipt says which of the two happened --------------------


def test_a_scheduled_reel_records_the_platforms_own_time(queue_conn):
    when = datetime.now(timezone.utc) + timedelta(hours=6)

    row, _ = _publish(queue_conn, _scheduled_record(when))

    assert row["published_at_source"] == PUBLISHED_AT_PLATFORM_SCHEDULED
    assert row["platform_published_at"] == when.isoformat()


def test_published_at_is_still_when_this_queue_handed_it_over(queue_conn):
    """`published_at` is unchanged by this: it is a fact about this queue, and
    the platform's own instant lives in its own column."""
    when = datetime.now(timezone.utc) + timedelta(hours=6)

    row, _ = _publish(queue_conn, _scheduled_record(when))

    assert row["published_at"] is not None
    assert row["published_at"] != row["platform_published_at"]


def test_an_immediate_publish_keeps_platform_confirmed_and_no_platform_time(queue_conn):
    """The accept-case. A reel that really did publish on the call keeps the
    stronger label, and NULL is an honest unknown -- never `published_at`
    copied across, which would be a fabricated platform receipt."""
    row, _ = _publish(queue_conn, _record())

    assert row["published_at_source"] == PUBLISHED_AT_PLATFORM_CONFIRMED
    assert row["platform_published_at"] is None


def test_a_record_with_no_schedule_publishes_now(queue_conn):
    record = _record()
    assert record.schedule.scheduled_at_utc < datetime.now(timezone.utc)

    assert due_schedule(record) is None
