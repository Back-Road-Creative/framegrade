"""One bad ``scheduled_at`` already in the queue must never end a drain pass.

Rows written before ``queue.normalise_slot`` can hold a time with no zone, or
text that is not a time at all. Every place dispatch reads a slot goes through
``dispatch._slot_time``: a zoneless time is read as UTC, and an unreadable one
holds that row (with one incident) while the rest of the pass goes on. Rows are
also taken oldest slot first, so a platform's one turn per pass goes to the row
that has waited longest."""

import json
from types import SimpleNamespace

import pytest

from gomoveshift.publishing import dispatch
from gomoveshift.publishing import queue as q
from gomoveshift.publishing.dispatch import drain_queue

NOW = "2026-11-20T12:00:00+00:00"
ZONELESS = "2026-11-19T09:00:00"
UNREADABLE = "2026-11-15 morning"  # sorts before NOW as text, so it is selected


@pytest.fixture
def conn(tmp_path):
    c = q.open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _row(conn, project, platform, slot):
    did = q.enqueue(conn, project, 0, platform, f"h-{project}", scheduled_at=NOW)
    conn.execute("UPDATE deliveries SET scheduled_at=? WHERE id=?", (slot, did))
    conn.commit()
    return did


def _record(posted):
    return lambda r: posted.append(r["id"])


def test_zoneless_and_unreadable_slots_with_a_blackout_never_end_the_drain(
    conn, tmp_path, monkeypatch
):
    """``_exclude_blackout`` parses every due slot; it used to raise on the first bad one."""
    policy = SimpleNamespace(blackout=["2026-12-25"], quiet_after_alert_hours=None)
    monkeypatch.setattr(dispatch, "load_brand_policy", lambda brand: policy)
    bad = _row(conn, "bad", "facebook", UNREADABLE)
    zoneless = _row(conn, "zoneless", "instagram", ZONELESS)
    good = _row(conn, "good", "tiktok", NOW)
    posted = []
    record = _record(posted)
    result = drain_queue(
        conn,
        {"facebook": record, "instagram": record, "tiktok": record},
        now=NOW,
        incidents_dir=tmp_path / "incidents",
    )
    assert posted == [zoneless, good]  # the zoneless slot is read as UTC and sent
    assert result.held_unreadable_slot == 1
    assert q.find_delivery(conn, "bad", 0, "facebook")["status"] == "queued"
    incident = json.loads(next((tmp_path / "incidents").glob("*.json")).read_text())
    assert incident["kind"] == dispatch.UNREADABLE_SLOT_KIND
    assert incident["delivery_id"] == bad


def test_an_unreadable_youtube_sibling_does_not_end_the_drain(conn, tmp_path):
    """``_defer_same_hour`` reads every other YouTube row's slot to find a free hour."""
    _row(conn, "bad", "youtube", UNREADABLE)
    ok = _row(conn, "ok", "youtube", "2026-11-20T11:00:00+00:00")
    posted = []
    drain_queue(conn, {"youtube": _record(posted)}, now=NOW, incidents_dir=tmp_path / "i")
    assert posted == [ok]


def test_a_platforms_one_turn_goes_to_its_oldest_slot_not_its_lowest_id(conn, tmp_path):
    newer = _row(conn, "newer", "facebook", "2026-11-19T09:00:00+00:00")
    older = _row(conn, "older", "facebook", "2026-11-12T09:00:00-05:00")
    posted = []
    drain_queue(conn, {"facebook": _record(posted)}, now=NOW, incidents_dir=tmp_path / "i")
    assert posted == [older]
    assert newer < older
