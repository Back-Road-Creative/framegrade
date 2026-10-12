"""The console notices a delivery whose slot went by.

The product documentation says the failure webhook fires on "a failed
run **or a missed slot**". The second half was not implemented. `notify.py`
watched job failures, `failed_permanent` / `awaiting_manual_publish` rows and
pending review gates -- every one of which is something that *happened*. A
missed slot is the absence of something happening: a drain that stopped, a
ticker that was never started, a platform paused and forgotten. The queue
records it perfectly (`scheduled_at` in the past, status still non-terminal)
and nothing looked.

The pause exclusion is the load-bearing half. A paused platform's rows sit
past their slot ON PURPOSE, so calling those missed would put the webhook in
the one state that trains an operator to ignore it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gomoveshift.console import notify, pubqueue
from gomoveshift.publishing.queue import (
    PUBLISHED_AT_OPERATOR_CONFIRMED,
    enqueue,
    open_queue_db,
    transition,
)


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


@pytest.fixture
def posted(monkeypatch):
    """Every `notify()` call, without a webhook URL or a socket in sight."""
    calls: list[tuple] = []
    monkeypatch.setattr(
        notify, "notify", lambda event, identity, status: calls.append((event, identity, status))
    )
    return calls


def _iso(minutes_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _queue(tmp_path, *, minutes_ago: float = 120.0, platform: str = "instagram"):
    """One delivery scheduled `minutes_ago` minutes in the past."""
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 1, platform, "h1", scheduled_at=_iso(minutes_ago))
    conn.close()
    return delivery_id


def test_a_delivery_whose_slot_went_by_is_notified(tmp_path, posted):
    _queue(tmp_path)

    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))

    assert [event for event, _identity, _status in posted] == ["slot_missed"]


def test_the_notification_names_the_delivery_and_its_platform(tmp_path, posted):
    delivery_id = _queue(tmp_path, platform="tiktok")

    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))

    _event, identity, _status = posted[0]
    assert identity["delivery_id"] == delivery_id
    assert identity["platform"] == "tiktok"


def test_a_row_still_inside_its_grace_window_is_not_missed_yet(tmp_path, posted):
    """A drain runs on an interval, so a row a minute past its slot is simply
    waiting its turn. Notifying there would cry wolf on every healthy queue."""
    _queue(tmp_path, minutes_ago=1.0)

    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))

    assert posted == []


def test_the_grace_window_is_configurable(tmp_path, posted, monkeypatch):
    monkeypatch.setenv("GMS_SLOT_MISSED_MIN", "5")
    _queue(tmp_path, minutes_ago=10.0)

    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))

    assert [event for event, _identity, _status in posted] == ["slot_missed"]


def test_a_published_row_is_never_missed(tmp_path, posted):
    """It went out. That its `scheduled_at` is in the past is the normal state
    of every delivery the console has ever completed."""
    delivery_id = _queue(tmp_path)
    conn = open_queue_db(tmp_path)
    # A receipt must name how it was learned; the queue refuses an
    # unlabelled one, so the test states the same thing the console does.
    transition(
        conn,
        delivery_id,
        "queued",
        "published",
        published_at_source=PUBLISHED_AT_OPERATOR_CONFIRMED,
    )
    conn.close()

    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))

    assert posted == []


def test_a_cancelled_row_is_never_missed(tmp_path, posted):
    delivery_id = _queue(tmp_path)
    conn = open_queue_db(tmp_path)
    transition(conn, delivery_id, "queued", "cancelled")
    conn.close()

    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))

    assert posted == []


def test_a_row_with_no_scheduled_time_is_never_missed(tmp_path, posted):
    """`scheduled_at` is nullable and NULL on legacy rows. A row with no slot
    cannot have missed one -- and treating NULL as the epoch would notify on
    every one of them, forever."""
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "h1")
    conn.close()

    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))

    assert posted == []


def test_a_paused_platforms_overdue_row_is_expected_not_missed(tmp_path, posted):
    """The reason this check had to wait for the pause control. Those rows are
    past their slot deliberately; a webhook that fires on them is a webhook an
    operator learns to ignore."""
    _queue(tmp_path, platform="facebook")
    pubqueue.set_platform_pause(pubqueue.queue_db_path(_config(tmp_path)), "facebook", paused=True)

    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))

    assert posted == []


def test_resuming_a_platform_surfaces_the_slots_it_missed(tmp_path, posted):
    """The other side of that decision, made deliberately rather than left to
    fall out: a pause SUPPRESSES the notification, it does not consume it. The
    backlog is the visible consequence of how long the pause lasted, and an
    operator who resumes needs to be told what is now genuinely late."""
    _queue(tmp_path, platform="facebook")
    db = pubqueue.queue_db_path(_config(tmp_path))
    pubqueue.set_platform_pause(db, "facebook", paused=True)
    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))
    pubqueue.set_platform_pause(db, "facebook", paused=False)

    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))

    assert [event for event, _identity, _status in posted] == ["slot_missed"]


def test_the_same_missed_slot_is_not_notified_twice(tmp_path, posted):
    """The ticker calls this every tick. Without seen-state a single stopped
    drain would post the same delivery every few minutes until someone looked."""
    _queue(tmp_path)

    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))
    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))

    assert len(posted) == 1


def test_the_seen_state_is_its_own_file(tmp_path, posted):
    """Sharing `notify_seen_queue.json` would let a status change clear the
    missed-slot memory, or the reverse -- two independent facts about one row."""
    _queue(tmp_path)

    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))

    written = {p.name for p in tmp_path.iterdir() if p.suffix == ".json"}
    assert "notify_seen_queue.json" not in written
    assert written


def test_a_missing_queue_file_is_an_empty_queue(tmp_path, posted):
    """Same rule the tab already follows: observing must never create the DB."""
    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))

    assert posted == []
    assert not (tmp_path / "social_queue.sqlite3").exists()
