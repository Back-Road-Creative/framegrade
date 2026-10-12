"""The Publishing tab reads one page of the queue, not all of it.

``queue_rows`` did ``SELECT * FROM deliveries ORDER BY status, id`` and handed
the whole result to the template -- the same unbounded-read shape already fixed
for the Media library. The queue only grows: every short, every platform, every
week, and nothing ever deletes a ``published`` row.

Two things make this different from the library's fix, and both are pinned here:

* **The bound is in the SQL, not a slice of a materialised list.** Reading
  every row and throwing most away costs exactly what it did before.
* **The lifecycle order goes into the SQL too.** ``ORDER BY status`` is
  alphabetical, so paging on it would put ``awaiting_manual_publish`` -- the one
  state that needs a human -- behind ``cancelled``, ``failed_permanent``,
  ``published`` and ``queued``. It sorts FIRST by design, and page 1 has to be
  the page that shows it.

``queue_rows`` itself stays unbounded on purpose: its other caller is
``notify.check_queue_regressions``, whose whole job is to scan every row. A
bound there would silently stop the observers at row 100.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from gomoveshift.console import pubqueue
from gomoveshift.console.app import create_app
from gomoveshift.publishing.queue import enqueue, open_queue_db, transition


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def _point_at(tmp_path, monkeypatch):
    monkeypatch.setattr(pubqueue.Config, "load", staticmethod(lambda: _config(tmp_path)))


def _fill(tmp_path, count: int, *, status: str = "queued") -> list[int]:
    """``count`` deliveries, all in ``status``."""
    conn = open_queue_db(tmp_path)
    ids = [enqueue(conn, "proj", i, "instagram", f"h{i}") for i in range(count)]
    if status != "queued":
        for delivery_id in ids:
            transition(conn, delivery_id, "queued", status)
    conn.close()
    return ids


def test_a_page_is_bounded(tmp_path):
    _fill(tmp_path, pubqueue.PER_PAGE + 25)

    page = pubqueue.queue_page(_config(tmp_path))

    assert len(page["rows"]) == pubqueue.PER_PAGE
    assert page["total"] == pubqueue.PER_PAGE + 25
    assert page["pages"] == 2


def test_the_bound_is_in_the_sql_not_a_slice(tmp_path, monkeypatch):
    """The point of the change. A read that fetches every row and slices costs
    what it always did -- so the SQL the page runs must say LIMIT itself."""
    _fill(tmp_path, 5)
    seen: list[str] = []
    real_connect = pubqueue.open_queue_db

    def _spy(directory):
        # sqlite's own statement trace, rather than wrapping `execute`: it sees
        # what the driver actually ran, which is the claim being made.
        conn = real_connect(directory)
        conn.set_trace_callback(lambda sql: seen.append(" ".join(sql.split())))
        return conn

    monkeypatch.setattr(pubqueue, "open_queue_db", _spy)
    pubqueue.queue_page(_config(tmp_path))

    selects = [sql for sql in seen if "FROM deliveries" in sql and "COUNT" not in sql.upper()]
    assert selects, f"no delivery SELECT ran at all: {seen}"
    assert all("LIMIT" in sql.upper() for sql in selects), selects


def test_the_second_page_carries_the_rest(tmp_path):
    ids = _fill(tmp_path, pubqueue.PER_PAGE + 3)

    page = pubqueue.queue_page(_config(tmp_path), page=2)

    assert [row["id"] for row in page["rows"]] == ids[pubqueue.PER_PAGE :]
    assert page["page"] == 2
    assert page["has_prev"] is True
    assert page["has_next"] is False


def test_a_state_needing_a_human_is_never_pushed_off_page_one(tmp_path):
    """`ORDER BY status` is alphabetical: `awaiting_manual_publish` would land
    after a page full of `cancelled`. It sorts first by design -- the lifecycle
    order has to be in the SQL, not applied to whatever the page came back
    with."""
    conn = open_queue_db(tmp_path)
    for i in range(pubqueue.PER_PAGE):
        delivery_id = enqueue(conn, "proj", i, "instagram", f"h{i}")
        transition(conn, delivery_id, "queued", "cancelled")
    needs_a_human = enqueue(conn, "proj", 999, "tiktok", "h999")
    transition(conn, needs_a_human, "queued", "awaiting_manual_publish")
    conn.close()

    page = pubqueue.queue_page(_config(tmp_path))

    assert page["rows"][0]["id"] == needs_a_human


def test_a_page_out_of_range_lands_on_a_real_page(tmp_path):
    """Same rule as the library: an operator reads an empty table as an empty
    queue, which is a much worse lie than a nearest-page redirect."""
    _fill(tmp_path, 5)

    assert pubqueue.queue_page(_config(tmp_path), page=99)["page"] == 1
    assert pubqueue.queue_page(_config(tmp_path), page="nonsense")["page"] == 1
    assert pubqueue.queue_page(_config(tmp_path), page=0)["page"] == 1


def test_a_missing_db_file_is_an_empty_page_and_stays_missing(tmp_path):
    """Unchanged rule: viewing the tab must never create the queue."""
    page = pubqueue.queue_page(_config(tmp_path))

    assert page["rows"] == []
    assert page["total"] == 0
    assert page["pages"] == 1
    assert not (tmp_path / "social_queue.sqlite3").exists()


def test_queue_rows_stays_unbounded_for_the_observers(tmp_path):
    """`notify.check_queue_regressions` scans every row. Bounding this would
    quietly stop the failure webhook at row 100."""
    _fill(tmp_path, pubqueue.PER_PAGE + 10)

    assert len(pubqueue.queue_rows(_config(tmp_path))) == pubqueue.PER_PAGE + 10


def test_the_tab_renders_one_page_and_says_so(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    _fill(tmp_path, pubqueue.PER_PAGE + 4)

    body = TestClient(create_app()).get("/publishing").text

    assert f"of {pubqueue.PER_PAGE + 4}" in body
    assert "Page 1 of 2" in body
    assert "/publishing?page=2" in body


def test_the_tab_serves_the_page_that_was_asked_for(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    ids = _fill(tmp_path, pubqueue.PER_PAGE + 2)

    body = TestClient(create_app()).get("/publishing?page=2").text

    # Quoted: `/publishing/1` is a substring of `/publishing/101`, so the bare
    # form would pass on a page that carries neither row.
    assert f'/publishing/{ids[-1]}"' in body
    assert f'/publishing/{ids[0]}"' not in body


def test_a_pause_refusal_still_shows_the_queue(tmp_path, monkeypatch):
    """The refusal path re-renders this tab, so it needs the pager keys too --
    a missing name in the template is a 500 instead of a sentence."""
    _point_at(tmp_path, monkeypatch)
    _fill(tmp_path, 3)

    response = TestClient(create_app()).post(
        "/publishing/pause", data={"platform": "myspace", "action": "pause"}
    )

    assert response.status_code == 400
    assert "Page 1 of 1" in response.text
