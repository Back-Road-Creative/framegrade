"""After an outage every overdue row is due at once. One drain pass must still
post at most one delivery per platform; the rest wait, untouched, for the next
pass. Facebook, Instagram and TikTok post the moment they are called, so without
this the first pass after an outage posted the whole backlog together."""

import pytest

from gomoveshift.publishing import queue as q
from gomoveshift.publishing.dispatch import drain_queue

NOW = "2026-11-20T12:00:00+00:00"
LATER = "2026-11-20T13:00:00+00:00"


@pytest.fixture
def conn(tmp_path):
    c = q.open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _overdue(conn, platform, count):
    return [
        q.enqueue(
            conn,
            f"proj{i}",
            i,
            platform,
            f"h{i}",
            scheduled_at=f"2026-11-{10 + i:02d}T09:00:00+00:00",
        )
        for i in range(count)
    ]


def _status(conn, delivery_id):
    return conn.execute(
        "SELECT status, attempt_count, scheduled_at FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()


def test_ten_overdue_facebook_rows_post_one_per_pass(conn, tmp_path):
    ids = _overdue(conn, "facebook", 10)
    before = {i: tuple(_status(conn, i)) for i in ids}
    posted = []
    result = drain_queue(
        conn, {"facebook": lambda r: posted.append(r["id"])}, now=NOW, incidents_dir=tmp_path
    )
    assert posted == [ids[0]]
    assert result.attempted == 1
    assert result.spaced_one_per_pass == 9
    # The other nine are left exactly as they were: no attempt, no new slot.
    for i in ids[1:]:
        assert tuple(_status(conn, i)) == before[i]
    drain_queue(
        conn, {"facebook": lambda r: posted.append(r["id"])}, now=LATER, incidents_dir=tmp_path
    )
    assert posted == ids[:2]


def test_each_platform_gets_its_own_one_post(conn, tmp_path):
    fb = _overdue(conn, "facebook", 3)
    ig = [q.enqueue(conn, f"proj{i}", i, "instagram", f"h{i}", scheduled_at=NOW) for i in range(3)]
    posted = []
    record = lambda r: posted.append(r["id"])  # noqa: E731
    drain_queue(conn, {"facebook": record, "instagram": record}, now=NOW, incidents_dir=tmp_path)
    assert posted == [fb[0], ig[0]]


def test_a_held_row_does_not_use_up_its_platforms_turn(conn, tmp_path):
    """The validate hold skips the first row; the next row of the same platform
    still goes, so one stuck project cannot stop a platform."""
    ids = _overdue(conn, "facebook", 2)
    posted = []
    drain_queue(
        conn,
        {"facebook": lambda r: posted.append(r["id"])},
        now=NOW,
        held_scopes={("facebook", "proj0")},
        incidents_dir=tmp_path,
    )
    assert posted == [ids[1]]
