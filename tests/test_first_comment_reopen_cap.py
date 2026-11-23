"""A delete a gone reopen queued runs ahead of the correction cap, and every bounded wait ends
with an incident (relink fixup 15). No network: the repo's own fakes."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from gomoveshift.publishing import first_comment_stale, meta_comments, queue
from gomoveshift.publishing.first_comment_stale import OWN_ID_KIND, raise_repost_incident
from gomoveshift.publishing.queue import (
    FIRST_COMMENT_MAX_ATTEMPTS,
    REOPEN_LAG_WINDOW,
    first_comment_corrections,
    open_queue_db,
)
from gomoveshift.publishing.youtube_comments import FirstCommentResult
from tests.test_first_comment_meta_correction import NEW, FakeGraph, _adapter
from tests.test_first_comment_meta_correction import _open_incidents
from tests.test_first_comment_meta_correction import _relinked as _meta_relinked
from tests.test_first_comment_meta_correction import _row as _meta_row

LAG_DROP = "first_comment_lag_delete_dropped"
NOW = "2026-08-23T00:06:00+00:00"
FIXED = datetime(2026, 8, 23, 0, 6, tzinfo=timezone.utc)
NEWER = "Watch the full drive: https://youtu.be/newer"
HOURS = REOPEN_LAG_WINDOW.total_seconds() / 3600


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


def _drain(conn, graph, inc):
    adapter = _adapter("facebook", graph)
    return meta_comments.drain_meta_first_comments(
        conn, adapter_factories={"facebook": lambda: adapter}, now=NOW, incidents_dir=inc
    )


class _FrozenDatetime(datetime):
    """``datetime`` whose ``now()`` is ``FIXED``: what the queue's lag-window check reads."""

    @classmethod
    def now(cls, tz=None):
        return FIXED if tz is None else FIXED.astimezone(tz)


@pytest.fixture(autouse=True)
def frozen_queue_clock(monkeypatch):
    monkeypatch.setattr(queue, "datetime", _FrozenDatetime)


def _ago(hours, now=FIXED):
    return (now - timedelta(hours=hours)).isoformat()


def _extra(live_id, text=NEW, **more):
    return {"live_id": live_id, "live_text": text, "action": "delete", "extra": True,
            "ours": True, **more}  # fmt: skip


def _held(live_id, hours=1, text=NEW):
    return _extra(live_id, text, reopened=True, reopened_at=_ago(hours))


def _posted(conn, did, items, live="c1", correction_attempts=0):
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_text=?, first_comment_posted_at=?, "
            "first_comment_live_id=?, first_comment_correction=?, "
            "first_comment_correction_attempts=? WHERE id=?",
            (NEW, NOW, live, json.dumps(items), correction_attempts, did),
        )


def _capped(conn, did, items, text=NEW):
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_text=?, first_comment_posted_at=NULL, "
            "first_comment_live_id=NULL, first_comment_attempts=?, first_comment_correction=? "
            "WHERE id=?",
            (text, FIRST_COMMENT_MAX_ATTEMPTS, json.dumps(items), did),
        )


def _ours(graph, text=NEW):
    return sorted(c for c, t in graph.comments.items() if t == text and graph.authors[c] == "p")


def _settle(conn, graph, inc):
    res = FirstCommentResult()
    first_comment_stale.settle_capped_reposts(
        conn, ("facebook",), lambda p: _adapter("facebook", graph), res, inc
    )
    return res


def _lag_drops(inc):
    return [i for i in _open_incidents(inc) if i["kind"] == LAG_DROP]


def _fail_reads_of(graph, comment_id):
    real = graph.get

    def get(url, params=None, **kw):
        if url.endswith("/" + comment_id):
            raise ConnectionError("reset")
        return real(url, params=params, **kw)

    graph.get = get


def _lagging_graph_then_clear(conn, did, inc, graph, drains=5):
    _drain(conn, graph, inc)
    graph.comments["c1"], graph.authors["c1"] = NEW, "p"  # the platform's lag clears
    for _ in range(drains):
        _drain(conn, graph, inc)


# --- MAJOR A: a reopened delete is never stranded ----------------------------------------------


def test_a_row_at_the_correction_cap_still_deletes_the_lagging_copy_when_it_appears(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_extra("zz")], live="c1", correction_attempts=FIRST_COMMENT_MAX_ATTEMPTS)
    raise_repost_incident(conn, did, "gone: lag", inc, kind=OWN_ID_KIND)
    graph = FakeGraph("facebook", [])  # c1 is lagging: reads gone and is not listed
    _lagging_graph_then_clear(conn, did, inc, graph)
    row = _meta_row(conn, did)
    assert len(_ours(graph)) == 1, (_ours(graph), graph.log)
    assert row["first_comment_live_id"] in _ours(graph)
    assert "c1" not in [i["live_id"] for i in first_comment_corrections(row)]


def test_a_failing_own_extra_ahead_of_it_never_keeps_the_reopen_delete_from_running(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_extra("zz")], live="c1")
    raise_repost_incident(conn, did, "gone: lag", inc, kind=OWN_ID_KIND)
    graph = FakeGraph("facebook", [])
    _fail_reads_of(graph, "zz")
    _lagging_graph_then_clear(conn, did, inc, graph)
    assert len(_ours(graph)) == 1, (_ours(graph), graph.log)
    assert "c1" not in [i["live_id"] for i in first_comment_corrections(_meta_row(conn, did))]


def test_a_failing_sibling_item_never_blocks_a_reopened_delete(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_extra("zz"), _held("c1")], live="c2")
    graph = FakeGraph("facebook", [("c1", NEW), ("c2", NEW)])
    _fail_reads_of(graph, "zz")
    _drain(conn, graph, inc)
    assert _ours(graph) == ["c2"], graph.log


def test_a_reopened_delete_runs_before_the_other_items_of_its_row(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_extra("x9"), _held("c1")], live="c2")
    graph = FakeGraph("facebook", [("c1", NEW), ("c2", NEW), ("x9", NEW)])
    _drain(conn, graph, inc)
    deleted = [e for e in graph.log if e.startswith("delete")]
    assert _ours(graph) == ["c2"] and deleted[0].endswith("c1"), graph.log


def test_a_held_reopened_delete_at_the_cap_counts_no_attempt_and_says_so(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_held("c1")], live="c2", correction_attempts=FIRST_COMMENT_MAX_ATTEMPTS)
    res = _drain(conn, FakeGraph("facebook", [("c2", NEW)]), inc)
    row = _meta_row(conn, did)
    assert row["first_comment_correction_attempts"] == FIRST_COMMENT_MAX_ATTEMPTS
    assert [i["live_id"] for i in first_comment_corrections(row)] == ["c1"]
    assert any("c1" in e and "lagging" in e for e in res.errors), res.errors


def test_a_reopened_delete_at_the_cap_is_dropped_at_its_deadline_with_an_incident(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(
        conn, did, [_held("c1", HOURS + 1)], live="c2",
        correction_attempts=FIRST_COMMENT_MAX_ATTEMPTS,
    )  # fmt: skip
    res = _drain(conn, FakeGraph("facebook", [("c2", NEW)]), inc)
    assert any("c1" in e and "dropped" in e for e in res.errors), res.errors
    assert not first_comment_corrections(_meta_row(conn, did))
    (drop,) = _lag_drops(inc)
    assert "c1" in drop["detail"]


def test_a_failing_reopened_read_counts_no_attempt_toward_the_cap(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_held("c1")], live="c2")
    graph = FakeGraph("facebook", [("c2", NEW)])
    _fail_reads_of(graph, "c1")
    for _ in range(FIRST_COMMENT_MAX_ATTEMPTS + 2):
        res = _drain(conn, graph, inc)
    row = _meta_row(conn, did)
    assert row["first_comment_correction_attempts"] == 0
    assert [i["live_id"] for i in first_comment_corrections(row)] == ["c1"]
    assert any("c1" in e or "not corrected" in e for e in res.errors), res.errors


def test_a_row_at_the_cap_with_only_a_reopened_delete_is_not_called_capped(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_held("c1")], live="c2", correction_attempts=FIRST_COMMENT_MAX_ATTEMPTS)
    res = _drain(conn, FakeGraph("facebook", [("c2", NEW)]), inc)
    assert res.correction_capped == 0
    assert not [i for i in _open_incidents(inc) if i["kind"] == first_comment_stale.CAPPED_KIND]


# --- MAJOR B: a failing listing never makes settle retry forever --------------------------------


def _listing_down(graph):
    real = graph.get

    def get(url, params=None, **kw):
        if url.endswith("/comments"):
            raise ConnectionError("the post is gone")
        return real(url, params=params, **kw)

    graph.get = get


def test_a_failing_listing_past_the_deadline_drops_with_an_incident_naming_the_failed_read(
    conn, inc
):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_held("a", HOURS * 3)])
    graph = FakeGraph("facebook", [])
    _listing_down(graph)
    res = _settle(conn, graph, inc)
    assert not first_comment_corrections(_meta_row(conn, did))
    assert any("a" in e and "dropped" in e for e in res.errors), res.errors
    (drop,) = _lag_drops(inc)
    assert "failed" in drop["detail"] and "by hand" in drop["detail"], drop["detail"]
    assert len([e for e in _settle(conn, graph, inc).errors]) == 0  # nothing left to retry


def test_a_failing_listing_inside_the_window_keeps_the_delete_pending_and_says_so(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_held("a", 1)])
    graph = FakeGraph("facebook", [])
    _listing_down(graph)
    res = _settle(conn, graph, inc)
    assert any("tries again" in e for e in res.errors), res.errors
    assert [i["live_id"] for i in first_comment_corrections(_meta_row(conn, did))] == ["a"]
    assert not _lag_drops(inc)


# --- MINOR C / D / E: what a dropped or kept item says -----------------------------------------


def test_the_lag_drop_incident_names_every_dropped_id(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_held(c, HOURS + 1) for c in ("a1", "b2", "c3")])
    res = _settle(conn, FakeGraph("facebook", []), inc)
    (drop,) = _lag_drops(inc)
    assert all(c in drop["detail"] for c in ("a1", "b2", "c3")), drop["detail"]
    assert sum("dropped" in e for e in res.errors) == 3


def test_an_incomplete_listing_inside_the_window_keeps_the_delete_pending(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_held("zzz_ours", 1)])
    viewers = [(f"v{i:03d}", "nice drive", "viewer") for i in range(40)]
    graph = FakeGraph("facebook", [*viewers, ("zzz_ours", NEW)])
    _settle(conn, graph, inc)
    row = _meta_row(conn, did)
    assert row["first_comment_posted_at"] is None
    assert [i["live_id"] for i in first_comment_corrections(row)] == ["zzz_ours"]
    assert not _lag_drops(inc)


def test_an_incomplete_listing_past_the_window_is_never_called_not_listed(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_held("zzz_ours", HOURS + 1)])
    viewers = [(f"v{i:03d}", "nice drive", "viewer") for i in range(40)]
    graph = FakeGraph("facebook", [*viewers, ("zzz_ours", NEW)])
    res = _settle(conn, graph, inc)
    assert "zzz_ours" in graph.comments
    assert not any("is not listed" in e for e in res.errors), res.errors
    (drop,) = _lag_drops(inc)
    assert "not listed" not in drop["detail"] and "cut short" in drop["detail"], drop["detail"]


def test_a_copy_of_ours_with_other_text_is_kept_and_surfaced_never_called_not_listed(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_held("a1", 1, text=NEW)], text=NEWER)
    graph = FakeGraph("facebook", [("a1", NEW)])
    res = _settle(conn, graph, inc)
    row = _meta_row(conn, did)
    assert "a1" in graph.comments and row["first_comment_posted_at"] is None
    assert [i["live_id"] for i in first_comment_corrections(row)] == ["a1"]
    assert any("a1" in e and "other text" in e for e in res.errors), res.errors
    assert not any("is not listed" in e for e in res.errors)


def test_a_copy_of_ours_with_other_text_past_the_window_is_dropped_saying_so_never_deleted(
    conn, inc
):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_held("a1", HOURS + 1, text=NEW)], text=NEWER)
    graph = FakeGraph("facebook", [("a1", NEW)])
    res = _settle(conn, graph, inc)
    assert "a1" in graph.comments and not first_comment_corrections(_meta_row(conn, did))
    assert any("a1" in e and "other text" in e for e in res.errors), res.errors
    (drop,) = _lag_drops(inc)
    assert "other text" in drop["detail"] and "not listed" not in drop["detail"], drop["detail"]


def test_a_settle_with_one_expired_and_one_live_wait_drops_nothing(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_held("old", HOURS + 1), _held("new", 1)])
    _settle(conn, FakeGraph("facebook", []), inc)
    ids = [i["live_id"] for i in first_comment_corrections(_meta_row(conn, did))]
    assert ids == ["old", "new"] and not _lag_drops(inc)


# --- MINOR F: an adopted capped row is not born capped -----------------------------------------


def test_adopting_a_capped_row_resets_its_correction_attempts(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_held("a", 1)])
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_correction_attempts=? WHERE id=?",
            (FIRST_COMMENT_MAX_ATTEMPTS, did),
        )
    graph = FakeGraph("facebook", [("c1", NEW), ("c2", NEW)])
    _settle(conn, graph, inc)
    row = _meta_row(conn, did)
    assert row["first_comment_live_id"] == "c1"
    assert row["first_comment_correction_attempts"] == 0
    _drain(conn, graph, inc)
    assert _ours(graph) == ["c1"], graph.log


# --- MINOR G: a held reopened delete never delays restoring a removed row ----------------------


def _removed(conn, did, items, text=""):
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_text=?, first_comment_posted_at=?, "
            "first_comment_live_id='gone1', first_comment_removed_at=?, "
            "first_comment_correction=? WHERE id=?",
            (text, NOW, NOW, json.dumps(items), did),
        )


def test_a_held_reopened_delete_does_not_delay_restoring_a_removed_row(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _removed(conn, did, [_held("c1", 1)], text=NEW)
    first_comment_stale.restore_removed(conn, ("facebook",), inc)
    row = _meta_row(conn, did)
    assert row["first_comment_posted_at"] is None and row["first_comment_removed_at"] is None
    assert [i["live_id"] for i in first_comment_corrections(row)] == ["c1"]


def test_a_removed_row_with_another_pending_correction_is_still_not_restored(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _removed(conn, did, [_held("c1", 1), _extra("x9")], text=NEW)
    first_comment_stale.restore_removed(conn, ("facebook",), inc)
    assert _meta_row(conn, did)["first_comment_posted_at"] is not None


# --- kill the survivors of rr32 (A12, A17, A18, A19) --------------------------------------------


def _reopen(conn, did, live="c1"):
    return queue.reopen_gone_first_comment(conn, did, live, NOW, "gone")


def test_a_reopen_beside_an_item_that_is_not_an_own_extra_delete_is_refused(conn):
    did = _meta_relinked(conn, "facebook")
    edit = {"live_id": "e1", "live_text": "x", "action": "edit", "to": NEW}
    _posted(conn, did, [_extra("zz"), edit], live="c1")
    assert _reopen(conn, did) is False
    row = _meta_row(conn, did)
    assert row["first_comment_posted_at"] is not None and len(first_comment_corrections(row)) == 2


def test_a_reopen_never_queues_an_id_twice(conn):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_extra("c1")], live="c1")
    assert _reopen(conn, did) is True
    assert [i["live_id"] for i in first_comment_corrections(_meta_row(conn, did))] == ["c1"]


class _MovingRow:
    """Delegates to a connection; right after the reopen reads the row, another writer changes
    its correction (the compare-and-set must then refuse the write)."""

    def __init__(self, real, did, select="SELECT first_comment_text, first_comment_correction"):
        self._real, self._did, self._select = real, did, select

    def __enter__(self):
        return self._real.__enter__()

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def execute(self, sql, params=()):
        cur = self._real.execute(sql, params)
        if sql.lstrip().startswith(self._select):
            rows = cur.fetchall()
            self._real.execute(
                "UPDATE deliveries SET first_comment_correction=? WHERE id=?",
                (json.dumps([_extra("zz"), _extra("yy")]), self._did),
            )
            return _Rows(rows)
        return cur


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


def test_a_reopen_refuses_when_the_correction_moved_under_it(conn):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_extra("zz")], live="c1")
    assert queue.reopen_gone_first_comment(_MovingRow(conn, did), did, "c1", NOW, "gone") is False
    row = _meta_row(conn, did)
    assert row["first_comment_posted_at"] is not None
    assert [i["live_id"] for i in first_comment_corrections(row)] == ["zz", "yy"]


def test_dropping_reopened_deletes_refuses_when_the_correction_moved_under_it(conn):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_held("a")])
    moving = _MovingRow(conn, did, "SELECT first_comment_correction FROM deliveries")
    assert queue.drop_reopened_deletes(moving, did) == []
    ids = [i["live_id"] for i in first_comment_corrections(_meta_row(conn, did))]
    assert ids == ["zz", "yy"]


def test_dropping_reopened_deletes_leaves_every_other_item(conn):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_held("a"), _extra("x9")])
    assert [i["live_id"] for i in queue.drop_reopened_deletes(conn, did)] == ["a"]
    assert [i["live_id"] for i in first_comment_corrections(_meta_row(conn, did))] == ["x9"]


def test_adopting_a_capped_row_refuses_a_text_the_row_no_longer_has(conn):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_held("a")], text=NEWER)
    assert queue.adopt_capped_first_comment(conn, did, "c9", NEW, NOW) is False
    assert _meta_row(conn, did)["first_comment_posted_at"] is None


def test_adopting_a_capped_row_refuses_a_row_that_is_already_posted(conn):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_held("a")], live="c1")
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_attempts=? WHERE id=?",
            (FIRST_COMMENT_MAX_ATTEMPTS, did),
        )
    assert queue.adopt_capped_first_comment(conn, did, "c9", NEW, NOW) is False
    assert _meta_row(conn, did)["first_comment_live_id"] == "c1"


def test_a_capped_unposted_row_with_no_reopened_delete_is_not_a_lag_row(conn):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_extra("x9")])
    assert queue.capped_unposted_lag_rows(conn, ("facebook",)) == []
    _capped(conn, did, [_extra("x9"), _held("a")])
    assert [r["id"] for r in queue.capped_unposted_lag_rows(conn, ("facebook",))] == [did]
