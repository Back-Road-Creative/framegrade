"""A reopened delete past its 24 hours is dropped, never retried for ever, and every drop names
every id (relink fixup 16). No network: the repo's own fakes."""

from __future__ import annotations

import json

import pytest

from gomoveshift.publishing import queue
from gomoveshift.publishing.queue import (
    FIRST_COMMENT_MAX_ATTEMPTS,
    first_comment_corrections,
    open_queue_db,
)
from tests.test_first_comment_meta_correction import NEW, FakeGraph
from tests.test_first_comment_meta_correction import _relinked as _meta_relinked
from tests.test_first_comment_meta_correction import _row as _meta_row
from tests.test_first_comment_reopen_cap import (
    HOURS,
    frozen_queue_clock,  # noqa: F401 -- autouse: held deletes are aged against FIXED
    _capped,
    _drain,
    _fail_reads_of,
    _held,
    _lag_drops,
    _MovingRow,
    _posted,
    _removed,
    _settle,
)


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "q")
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def inc(tmp_path):
    return tmp_path / "incidents"


DRAINS = FIRST_COMMENT_MAX_ATTEMPTS * 3


def _ids(conn, did):
    return [i["live_id"] for i in first_comment_corrections(_meta_row(conn, did))]


def test_a_failing_read_past_the_deadline_drops_the_delete_with_an_incident(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_held("c1", HOURS + 1)], live="c2")
    graph = FakeGraph("facebook", [("c1", NEW), ("c2", NEW)])
    _fail_reads_of(graph, "c1")
    res = _drain(conn, graph, inc)
    assert _ids(conn, did) == []
    (drop,) = _lag_drops(inc)
    assert "c1" in drop["detail"] and "reset" in drop["detail"], drop["detail"]
    assert any("c1" in e and "dropped" in e for e in res.errors), res.errors
    assert _meta_row(conn, did)["first_comment_correction_attempts"] == 0
    seen = len(graph.log)
    for _ in range(DRAINS):
        _drain(conn, graph, inc)
    assert len(graph.log) == seen, graph.log  # the drain bounds itself: nothing more is tried
    assert len(_lag_drops(inc)) == 1


def test_a_refused_delete_past_the_deadline_drops_the_delete_with_an_incident(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_held("c1", HOURS + 1)], live="c2")
    graph = FakeGraph("facebook", [("c1", NEW), ("c2", NEW)])
    graph.delete_refusal = True
    res = _drain(conn, graph, inc)
    assert _ids(conn, did) == []
    (drop,) = _lag_drops(inc)
    assert "c1" in drop["detail"] and "refused" in drop["detail"], drop["detail"]
    assert any("c1" in e and "dropped" in e for e in res.errors), res.errors
    # outside the cap: the refusal counted no attempt, and the row is not called capped
    assert _meta_row(conn, did)["first_comment_correction_attempts"] == 0
    assert res.correction_capped == 0
    seen = len(graph.log)
    for _ in range(DRAINS):
        again = _drain(conn, graph, inc)
    assert len(graph.log) == seen and again.correction_capped == 0, graph.log


def test_a_failing_reopened_delete_never_blocks_the_next_one_in_its_row(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_held("c1", HOURS + 1), _held("c3", HOURS + 1)], live="c2")
    graph = FakeGraph("facebook", [("c1", NEW), ("c2", NEW), ("c3", NEW)])
    _fail_reads_of(graph, "c1")
    _drain(conn, graph, inc)
    assert "delete:c3" in graph.log, graph.log  # the second is tried and deleted
    assert _ids(conn, did) == []
    (drop,) = _lag_drops(inc)
    assert "c1" in drop["detail"]


def test_a_failing_reopened_delete_inside_the_window_never_blocks_the_next_one(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_held("c1", 1), _held("c3", 1)], live="c2")
    graph = FakeGraph("facebook", [("c1", NEW), ("c2", NEW), ("c3", NEW)])
    _fail_reads_of(graph, "c1")
    _drain(conn, graph, inc)
    assert "delete:c3" in graph.log, graph.log
    assert _ids(conn, did) == ["c1"]  # still waiting: not dropped inside the window
    assert not _lag_drops(inc)
    assert _meta_row(conn, did)["first_comment_correction_attempts"] == 0


def test_two_expired_held_deletes_on_a_posted_row_raise_one_incident_naming_both(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_held("c1", HOURS + 1), _held("c3", HOURS + 1)], live="c2")
    res = _drain(conn, FakeGraph("facebook", [("c2", NEW)]), inc)  # both read gone
    (drop,) = _lag_drops(inc)
    for field in ("detail", "error"):
        assert "c1" in drop[field] and "c3" in drop[field], drop[field]
    assert sum("dropped" in e for e in res.errors) == 2, res.errors
    assert _ids(conn, did) == []


def test_the_settle_drop_detail_keeps_eight_full_length_meta_ids(conn, inc):
    did = _meta_relinked(conn, "facebook")
    ids = [f"1234567890123456_98765432109876{i}" for i in range(8)]
    _capped(conn, did, [_held(c, HOURS + 1) for c in ids])
    _settle(conn, FakeGraph("facebook", []), inc)
    (drop,) = _lag_drops(inc)
    assert all(c in drop["detail"] for c in ids), drop["detail"]
    assert all(c in drop["error"] for c in ids), drop["error"]
    assert "hand" in drop["detail"]  # the advice is not cut off either


def test_restoring_a_removed_row_refuses_when_the_correction_moved_under_it(conn):
    did = _meta_relinked(conn, "facebook")
    _removed(conn, did, [_held("a")])
    moving = _MovingRow(conn, did, "SELECT first_comment_correction FROM deliveries")
    assert (
        queue.unpost_removed_first_comment(moving, did, NEW, "2026-08-23T00:06:00+00:00") is False
    )
    row = _meta_row(conn, did)
    assert row["first_comment_posted_at"] is not None  # not unposted
    assert row["first_comment_removed_at"] is not None
    assert (row["first_comment_text"] or "") == ""
    assert _ids(conn, did) == ["zz", "yy"]  # only the other writer's change stands


# --- relink fixup 17: one incident per row, and the hand-delete advice is never cut -------------

HAND_DELETE_TAIL = "after the wait needs a hand delete"


def _where(conn, did):
    row = _meta_row(conn, did)
    return f"{row['platform']}#{did} {row['project_name']} short {row['short_index']}: "


def _mark_pushed(inc):
    for path in inc.glob("*.json"):
        row = json.loads(path.read_text())
        row.update(pushed=True, pushed_at="2026-08-23T00:00:00+00:00")
        path.write_text(json.dumps(row))


def test_a_second_drop_on_the_same_row_joins_the_open_incident_and_is_pushed_again(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_held("c1", HOURS + 1)], live="c2")
    graph = FakeGraph("facebook", [("c2", NEW)])
    _drain(conn, graph, inc)
    _mark_pushed(inc)
    with conn:  # a later reopen on the same row queues another delete, which also expires
        conn.execute(
            "UPDATE deliveries SET first_comment_correction=? WHERE id=?",
            (json.dumps([_held("c3", HOURS + 1)]), did),
        )
    _drain(conn, graph, inc)
    (drop,) = _lag_drops(inc)
    for field in ("detail", "error"):
        assert "c1" in drop[field] and "c3" in drop[field], drop[field]
    assert drop["pushed"] is False and drop["pushed_at"] is None
    assert drop["detail"].endswith(HAND_DELETE_TAIL)


def test_the_same_id_dropped_twice_is_named_once(conn, inc):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [("c2", NEW)])
    for _ in range(2):
        _posted(conn, did, [_held("c1", HOURS + 1)], live="c2")
        _drain(conn, graph, inc)
    (drop,) = _lag_drops(inc)
    assert drop["detail"].count("c1") == 1, drop["detail"]


def test_eight_failing_meta_ids_keep_every_id_and_the_hand_delete_advice(conn, inc):
    did = _meta_relinked(conn, "facebook")
    ids = [f"1234567890123456_98765432109876{i}" for i in range(8)]
    _posted(conn, did, [_held(c, HOURS + 1) for c in ids], live="c2")
    graph = FakeGraph("facebook", [("c2", NEW)] + [(c, NEW) for c in ids])
    for c in ids:
        _fail_reads_of(graph, c)
    _drain(conn, graph, inc)
    (drop,) = _lag_drops(inc)
    assert all(c in drop["detail"] for c in ids), drop["detail"]
    assert all(c in drop["error"] for c in ids), drop["error"]
    assert drop["detail"].endswith(HAND_DELETE_TAIL), drop["detail"][-200:]
    assert drop["detail"].startswith(_where(conn, did) + "comments " + ", ".join(ids))
    assert len(drop["detail"]) > 500


def test_two_ids_with_a_long_graph_error_keep_both_ids_and_the_whole_advice(conn, inc):
    did = _meta_relinked(conn, "facebook")
    ids = ["1234567890123456_987654321098760", "1234567890123456_987654321098761"]
    _posted(conn, did, [_held(c, HOURS + 1) for c in ids], live="c2")
    graph = FakeGraph("facebook", [("c2", NEW)] + [(c, NEW) for c in ids])
    real = graph.get
    msg = (
        "(#100) Unsupported get request. The object does not exist, cannot be loaded due to "
        "missing permissions, or does not support this operation. Please read the Graph API "
        "documentation at https://developers.facebook.com/docs/graph-api. "
    ) * 4

    def get(url, params=None, **kw):
        if any(url.endswith("/" + c) for c in ids):
            raise RuntimeError(msg)
        return real(url, params=params, **kw)

    graph.get = get
    _drain(conn, graph, inc)
    (drop,) = _lag_drops(inc)
    assert all(c in drop["detail"] for c in ids), drop["detail"]
    assert drop["detail"].endswith(HAND_DELETE_TAIL), drop["detail"][-200:]
    assert "Delete by hand any comment named here" in drop["detail"]


def test_a_settle_drop_with_two_reasons_names_both_ids_and_both_reasons(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_held("c1", HOURS + 1), _held("c2", HOURS + 1)])
    # c1 is listed with other text (left alone); c2 is not listed at all.
    _settle(conn, FakeGraph("facebook", [("c1", "someone else's words")]), inc)
    (drop,) = _lag_drops(inc)
    for field in ("detail", "error"):
        assert "c1" in drop[field] and "c2" in drop[field], drop[field]
    assert "says other text" in drop["detail"] and "it is not listed" in drop["detail"]
