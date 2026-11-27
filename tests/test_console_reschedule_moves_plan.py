"""A console reschedule moves the slot the adapters publish at, not only the row.

The YouTube and Facebook adapters publish at the publish plan's time; the row's
``scheduled_at`` only says when the drain may send it. The console moved the row
alone, so the plan kept the old time. It also stored whatever text was typed:
a time with no zone was kept as typed, compared as text by the drain, and made
``_raise_held_incident`` raise ``TypeError``, which ended the whole drain pass."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from test_cli_enqueue_plan import _record as _plan_record
from test_cli_enqueue_plan import _sidecar

from gomoveshift.console import pubqueue
from gomoveshift.publishing import queue as q
from gomoveshift.publishing.dispatch import drain_queue
from gomoveshift.publishing.models import ScheduleWindow
from gomoveshift.publishing.plan_io import load_publish_plan

UTC = timezone.utc
PLAN_SLOT = datetime(2030, 1, 10, 15, 0, tzinfo=UTC)


@pytest.fixture
def seo(tmp_path, monkeypatch):
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path / "seo")))
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: config))
    return tmp_path / "seo"


def _queued(seo, tmp_path, platform):
    record = dataclasses.replace(
        _plan_record(short_index=4, selected=("youtube", "facebook")),
        schedule=ScheduleWindow(PLAN_SLOT, "America/New_York"),
    )
    sidecar = _sidecar(tmp_path, record)
    conn = q.open_queue_db(seo)
    did = q.enqueue(
        conn,
        "proj",
        4,
        platform,
        "h",
        scheduled_at=PLAN_SLOT.isoformat(),
        sidecar_path=str(sidecar),
    )
    conn.close()
    return did, sidecar


def _slot(seo, did):
    conn = q.open_queue_db(seo)
    try:
        return conn.execute("SELECT scheduled_at FROM deliveries WHERE id=?", (did,)).fetchone()[0]
    finally:
        conn.close()


def _plan(sidecar):
    return load_publish_plan(sidecar).shorts[0].schedule


def test_a_facebook_reschedule_moves_the_plan_slot_facebook_publishes_at(seo, tmp_path):
    did, sidecar = _queued(seo, tmp_path, "facebook")
    pubqueue.reschedule(pubqueue.queue_db_path(), did, "2030-01-11T07:30:00-05:00")
    assert _slot(seo, did) == "2030-01-11T12:30:00+00:00"
    schedule = _plan(sidecar)
    assert schedule.scheduled_at_utc_by_platform["facebook"] == datetime(
        2030, 1, 11, 12, 30, tzinfo=UTC
    )
    assert schedule.scheduled_at_utc == PLAN_SLOT  # YouTube's publishAt is not moved


def test_a_youtube_reschedule_moves_the_plan_publish_at(seo, tmp_path):
    did, sidecar = _queued(seo, tmp_path, "youtube")
    pubqueue.reschedule(pubqueue.queue_db_path(), did, "2030-01-12T16:00:00Z")
    assert _slot(seo, did) == "2030-01-12T16:00:00+00:00"
    assert _plan(sidecar).scheduled_at_utc == datetime(2030, 1, 12, 16, 0, tzinfo=UTC)


@pytest.mark.parametrize("typed", ["2030-01-11T12:30", "next tuesday"])
def test_a_zoneless_or_unreadable_time_is_refused_and_nothing_moves(seo, tmp_path, typed):
    did, sidecar = _queued(seo, tmp_path, "facebook")
    before = sidecar.read_text()
    with pytest.raises(ValueError, match="scheduled_at"):
        pubqueue.reschedule(pubqueue.queue_db_path(), did, typed)
    assert _slot(seo, did) == PLAN_SLOT.isoformat()
    assert sidecar.read_text() == before


def test_a_plan_that_cannot_be_written_leaves_the_row_alone(seo, tmp_path):
    did, sidecar = _queued(seo, tmp_path, "facebook")
    sidecar.unlink()
    with pytest.raises(ValueError, match="publish plan"):
        pubqueue.reschedule(pubqueue.queue_db_path(), did, "2030-01-11T12:30:00+00:00")
    assert _slot(seo, did) == PLAN_SLOT.isoformat()


def test_a_zoneless_slot_already_in_the_queue_does_not_end_the_drain(tmp_path):
    """A row written before the slot was normalised, held by the validate gate:
    the held-incident path must not raise, and later rows still go out."""
    conn = q.open_queue_db(tmp_path / "seo")
    bad = q.enqueue(conn, "bad", 0, "facebook", "h0", scheduled_at="2026-11-20T01:00:00+00:00")
    good = q.enqueue(conn, "good", 1, "instagram", "h1", scheduled_at="2026-11-20T01:00:00+00:00")
    conn.execute("UPDATE deliveries SET scheduled_at='2026-11-20T01:00:00' WHERE id=?", (bad,))
    conn.commit()
    posted = []
    result = drain_queue(
        conn,
        {
            "facebook": lambda r: posted.append(r["id"]),
            "instagram": lambda r: posted.append(r["id"]),
        },
        now="2026-11-20T12:00:00+00:00",
        held_scopes={("facebook", "bad")},
        incidents_dir=tmp_path / "incidents",
    )
    conn.close()
    assert posted == [good]
    assert result.held_validate == 1
