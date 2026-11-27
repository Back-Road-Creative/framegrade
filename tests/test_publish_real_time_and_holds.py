"""Q4: the real go-public time, a held row that raises an incident, and one
YouTube Short per clock hour. Recorded API response dicts only, no network."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from gomoveshift.cli.handlers import pub_status
from gomoveshift.publishing import dispatch
from gomoveshift.publishing import queue as q
from gomoveshift.publishing.dispatch import drain_queue
from gomoveshift.publishing.youtube_adapter import live_publish_time
from gomoveshift.tracking.store import TrackingStore

NOW = "2026-11-20T12:00:00+00:00"
SLOT = "2026-11-17T09:00:00+00:00"  # three days before NOW: it went public long ago
REAL = "2026-11-17T09:03:41+00:00"

# videos.list(part="status,snippet") items as YouTube returns them.
PUBLIC = {
    "id": "vid1",
    "status": {"privacyStatus": "public", "publishAt": SLOT.replace("+00:00", "Z")},
    "snippet": {"publishedAt": REAL.replace("+00:00", "Z")},
}
STILL_PRIVATE = {
    "id": "vid1",
    "status": {"privacyStatus": "private", "publishAt": "2026-11-25T09:00:00Z"},
    "snippet": {"publishedAt": "2026-11-18T01:00:00Z"},  # the UPLOAD time, not go-public
}


@pytest.fixture
def conn(tmp_path):
    c = q.open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _scheduled_published(conn, index=0, slot=SLOT):
    did = q.enqueue(conn, "proj", index, "youtube", f"h{index}", scheduled_at=slot)
    conn.execute(
        "UPDATE deliveries SET status='uploaded', upload_kind='short', upload_account='a', "
        "platform_upload_id='vid1' WHERE id=?",
        (did,),
    )
    conn.commit()
    q.transition(
        conn,
        did,
        "uploaded",
        "published",
        platform_post_id="pl",
        published_at_source=q.PUBLISHED_AT_PLATFORM_SCHEDULED,
        platform_published_at=slot,
    )
    return did


# -- real time -------------------------------------------------------------


def test_live_publish_time_reads_only_a_public_video():
    assert live_publish_time(PUBLIC) == REAL
    assert live_publish_time(STILL_PRIVATE) is None
    assert live_publish_time({"id": "x"}) is None


def test_record_live_publish_stores_real_time_with_its_own_source(conn):
    did = _scheduled_published(conn)
    assert q.record_live_publish(conn, did, REAL, now=NOW) is True
    row = dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone())
    assert row["platform_published_at"] == REAL
    assert row["published_at_source"] == q.PUBLISHED_AT_PLATFORM_LIVE
    # idempotent: a second call finds no scheduled row and writes nothing
    assert q.record_live_publish(conn, did, "2026-11-18T00:00:00+00:00", now=NOW) is False


def test_a_scheduled_row_keeps_the_scheduled_source(conn):
    did = _scheduled_published(conn)
    row = dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone())
    assert row["published_at_source"] == q.PUBLISHED_AT_PLATFORM_SCHEDULED
    assert row["platform_published_at"] == SLOT


def test_tracking_posted_at_takes_the_confirmed_time():
    assert q.recorded_publish_time(
        "2026-11-14T02:00:00+00:00", q.PUBLISHED_AT_PLATFORM_LIVE, REAL
    ) == (REAL, q.PUBLISHED_AT_PLATFORM_LIVE)
    # scheduled keeps the row's own pair; live with no time says nothing
    assert q.recorded_publish_time(
        "2026-11-14T02:00:00+00:00", q.PUBLISHED_AT_PLATFORM_SCHEDULED, SLOT
    ) == ("2026-11-14T02:00:00+00:00", q.PUBLISHED_AT_PLATFORM_SCHEDULED)
    assert q.recorded_publish_time("2026-11-14T02:00:00+00:00", "platform_live", None) == (
        None,
        None,
    )


def test_record_live_publish_moves_tracking_posted_at(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path / "out"))
    store = TrackingStore()
    registered = []
    monkeypatch.setattr(
        q,
        "_register_platform_id_best_effort",
        lambda c, d: registered.append(d),
    )
    did = _scheduled_published(conn)
    registered.clear()  # the publish transition registered once already
    q.record_live_publish(conn, did, REAL, now=NOW)
    assert registered == [did]
    assert store is not None


# -- validate hold ---------------------------------------------------------


def _due_youtube(conn, index, slot):
    return q.enqueue(conn, "proj", index, "youtube", f"h{index}", scheduled_at=slot)


def test_hold_is_scoped_to_project_not_to_the_problem_row(conn, monkeypatch):
    """Root cause of the hold: ``gating_problems`` names a delivery, but the drain
    holds the whole (platform, project) pair, so one bad row parks every
    healthy sibling. Characterisation, green before and after."""
    a = _due_youtube(conn, 0, SLOT)
    _due_youtube(conn, 1, SLOT)
    problem = pub_status.RestoreProblem(a, "youtube", "sidecar_unreadable", "bad")
    monkeypatch.setattr(pub_status, "validate_restored_state", lambda *x, **k: [problem])
    held = {(p.platform, p.project) for p in pub_status.gating_problems(conn, None)}
    result = drain_queue(conn, {"youtube": lambda r: None}, now=NOW, held_scopes=held)
    assert result.held_validate == 2 and result.attempted == 0


def _incidents(path):
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]


def test_row_held_past_slot_writes_one_incident(conn, tmp_path):
    did = _due_youtube(conn, 0, "2026-11-20T08:00:00+00:00")  # 4 h before NOW
    inc = tmp_path / "inc"
    for _ in range(2):  # a second pass must not rewrite the open row
        drain_queue(
            conn,
            {"youtube": lambda r: None},
            now=NOW,
            held_scopes={("youtube", "proj")},
            incidents_dir=inc,
        )
    rows = _incidents(inc)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == dispatch.HELD_PAST_SLOT_KIND == "delivery_held_past_slot"
    assert row["key"] == f"delivery_held_past_slot:{did}"
    assert row["pushed"] is False and row["resolved_at"] is None
    assert "validate hold" in row["detail"] and str(did) in row["detail"]


def test_row_held_within_margin_writes_nothing(conn, tmp_path):
    _due_youtube(conn, 0, "2026-11-20T10:30:00+00:00")  # 1.5 h
    inc = tmp_path / "inc"
    drain_queue(
        conn,
        {"youtube": lambda r: None},
        now=NOW,
        held_scopes={("youtube", "proj")},
        incidents_dir=inc,
    )
    assert _incidents(inc) == []


def test_open_incident_is_not_rewritten(conn, tmp_path):
    _due_youtube(conn, 0, "2026-11-20T08:00:00+00:00")
    inc = tmp_path / "inc"
    kw = dict(now=NOW, held_scopes={("youtube", "proj")}, incidents_dir=inc)
    drain_queue(conn, {"youtube": lambda r: None}, **kw)
    path = next(inc.glob("*.json"))
    before = path.read_text()
    drain_queue(conn, {"youtube": lambda r: None}, **{**kw, "now": "2026-11-20T13:00:00+00:00"})
    assert path.read_text() == before


# -- one Short per clock hour ----------------------------------------------


def test_two_shorts_due_in_one_hour_second_moves_to_next_free_slot(conn, monkeypatch, tmp_path):
    slot_a, slot_b = "2026-11-20T11:05:00+00:00", "2026-11-20T11:40:00+00:00"
    a = _due_youtube(conn, 0, slot_a)
    b = _due_youtube(conn, 1, slot_b)
    # 12:xx is taken by a third, already-queued row, so b lands in 13:xx
    _due_youtube(conn, 2, "2026-11-20T12:30:00+00:00")
    written = []
    monkeypatch.setattr(
        dispatch, "write_plan_slot", lambda sc, proj, idx, slot: written.append((idx, slot))
    )
    conn.execute("UPDATE deliveries SET sidecar_path='/x/plan.json'")
    conn.commit()
    ran = []
    result = drain_queue(
        conn, {"youtube": lambda r: ran.append(r["id"])}, now=NOW, incidents_dir=tmp_path / "i"
    )
    assert ran == [a]
    assert result.moved_same_hour == 1
    new = conn.execute("SELECT scheduled_at FROM deliveries WHERE id=?", (b,)).fetchone()[0]
    assert datetime.fromisoformat(new) == datetime(2026, 11, 20, 13, 40, tzinfo=timezone.utc)
    assert written == [(1, datetime(2026, 11, 20, 13, 40, tzinfo=timezone.utc))]


def test_other_platforms_in_the_same_hour_keep_their_slots(conn, tmp_path):
    """Only YouTube rows are moved to a new hour. Another platform's second row
    keeps its slot and simply waits for the next pass (one post per pass)."""
    slots = ["2026-11-20T11:05:00+00:00", "2026-11-20T11:40:00+00:00"]
    ids = [
        q.enqueue(conn, "proj", i, "instagram", f"h{i}", scheduled_at=s)
        for i, s in enumerate(slots)
    ]
    ran = []
    drain_queue(
        conn, {"instagram": lambda r: ran.append(r["id"])}, now=NOW, incidents_dir=tmp_path / "i"
    )
    assert ran == [ids[0]]
    kept = conn.execute("SELECT scheduled_at FROM deliveries WHERE id=?", (ids[1],)).fetchone()[0]
    assert kept == slots[1]


def test_margin_constant_is_two_hours():
    assert dispatch.HELD_PAST_SLOT_MARGIN == timedelta(hours=2)


def test_confirm_scheduled_live_sweep_uses_recorded_responses(conn):
    from gomoveshift.publishing.youtube_adapter import confirm_scheduled_live

    done = _scheduled_published(conn, 0)
    waiting = _scheduled_published(conn, 1)
    conn.execute("UPDATE deliveries SET platform_upload_id='vid2' WHERE id=?", (waiting,))
    conn.commit()
    responses = {"vid1": {"items": [PUBLIC]}, "vid2": {"items": [STILL_PRIVATE]}}

    class _Svc:
        def videos(self):
            return self

        def list(self, part, id):  # noqa: A002
            self._id = id
            return self

        def execute(self):
            return responses[self._id]

    assert confirm_scheduled_live(conn, _Svc(), now=NOW) == 1
    src = lambda i: conn.execute(  # noqa: E731
        "SELECT published_at_source FROM deliveries WHERE id=?", (i,)
    ).fetchone()[0]
    assert src(done) == q.PUBLISHED_AT_PLATFORM_LIVE
    assert src(waiting) == q.PUBLISHED_AT_PLATFORM_SCHEDULED
