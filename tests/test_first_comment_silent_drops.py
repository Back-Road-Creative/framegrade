"""An own comment is never dropped silently, a mismatched own id is reported, and an extra delete
re-checks the author. Each test asserts the GOOD outcome. No network: the repo's own fakes."""

from __future__ import annotations

import json
import logging

import pytest

from gomoveshift.publishing import first_comment as fc
from gomoveshift.publishing import meta_comments
from gomoveshift.publishing.facebook_adapter import FacebookReelsAdapter
from gomoveshift.publishing.queue import first_comment_corrections, open_queue_db
from tests.test_first_comment_meta_correction import NEW, OLD, FakeGraph, Resp, _open_incidents
from tests.test_first_comment_meta_correction import _drain as _meta_drain
from tests.test_first_comment_meta_correction import _relinked as _meta_relinked
from tests.test_first_comment_meta_correction import _row as _meta_row
from tests.test_first_comment_owner_match import NoFromGraph, _extra_for, _faithful_service
from tests.test_first_comment_stale import _NOW, NEW_TEXT, _drain, _row

PLATFORMS = ["facebook", "instagram"]


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


class _ReadGone(FakeGraph):
    """``GET /{id}`` answers 100/33 whether or not the comment is there."""

    def get(self, url, params=None, **kw):
        if not url.endswith("/comments"):
            return Resp({"error": {"code": 100, "error_subcode": 33, "message": "no"}}, 400)
        return super().get(url, params=params, **kw)


# --- item 4: an own duplicate is never dropped silently -------------------------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_an_extra_whose_re_read_says_gone_but_is_still_listed_stays_pending(conn, inc, platform):
    did = _meta_relinked(conn, platform)
    graph = _ReadGone(platform, [("o1", NEW), ("o2", NEW)])
    res = [_meta_drain(conn, platform, graph, inc) for _ in range(3)]
    assert list(graph.comments) == ["o1", "o2"] and "delete:o2" not in graph.log
    assert first_comment_corrections(_meta_row(conn, did))  # kept, so it is tried again
    assert res[-1].errors and res[-1].stale  # and said so


@pytest.mark.parametrize("platform", PLATFORMS)
def test_an_extra_really_gone_is_done_without_an_error(conn, inc, platform):
    did = _meta_relinked(conn, platform)
    graph = FakeGraph(platform, [("o1", NEW), ("o2", NEW)])
    _meta_drain(conn, platform, graph, inc)  # queues the delete of o2
    graph.comments.pop("o2")  # a person removed it meanwhile
    res = _meta_drain(conn, platform, graph, inc)
    assert not first_comment_corrections(_meta_row(conn, did)) and not res.errors


@pytest.mark.parametrize("platform", PLATFORMS)
def test_an_extra_whose_text_changed_is_dropped_with_an_error_line(conn, inc, platform):
    did = _meta_relinked(conn, platform)
    graph = FakeGraph(platform, [("o1", NEW), ("o2", NEW)])
    _meta_drain(conn, platform, graph, inc)
    graph.comments["o2"] = "someone changed this"
    res = _meta_drain(conn, platform, graph, inc)
    assert "o2" in graph.comments and "delete:o2" not in graph.log
    assert not first_comment_corrections(_meta_row(conn, did))
    assert any("o2" in e for e in res.errors)


# --- item 5: a mismatched own id is reported ------------------------------------------------


def _fb(graph, page_id):
    return FacebookReelsAdapter({"page_id": page_id, "page_access_token": "t"}, session=graph)


def _drain_as(conn, adapter, inc):
    return meta_comments.drain_meta_first_comments(
        conn, adapter_factories={"facebook": lambda: adapter}, now=_NOW, incidents_dir=inc
    )


def test_a_wrong_own_id_after_a_lost_answer_is_reported_when_the_duplicate_posts(conn, inc):
    _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    graph.lose_post_response = True
    res = [_drain_as(conn, _fb(graph, "not-p"), inc) for _ in range(2)]
    assert any("own id" in e for r in res for e in r.errors)
    assert [i for i in _open_incidents(inc) if i["kind"] == "first_comment_own_id_mismatch"]


def test_a_wrong_own_id_beside_a_stale_old_link_is_reported(conn, inc):
    _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [("c1", OLD)])  # ours (from p), says the replaced text
    res = _drain_as(conn, _fb(graph, "not-p"), inc)
    assert any("own id" in e for e in res.errors)
    assert "delete:c1" not in graph.log and "c1" in graph.comments  # never deleted


def test_a_right_own_id_reports_nothing_and_a_lost_answer_never_duplicates(conn, inc):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    graph.lose_post_response = True
    res = [_drain_as(conn, _fb(graph, "p"), inc) for _ in range(3)]
    assert list(graph.comments.values()) == [NEW]
    assert not [e for r in res for e in r.errors if "own id" in e] and not _open_incidents(inc)
    assert _meta_row(conn, did)["first_comment_live_id"] == "new1"


# --- item 6: the extra re-read checks the author --------------------------------------------


def test_an_extra_whose_author_is_not_ours_is_never_deleted(conn, inc):
    did = _extra_for(conn, inc)  # Y2 is queued as ours
    svc = _faithful_service({"Y": NEW_TEXT, "Y2": NEW_TEXT}, authors={"Y2": "UC-viewer"})
    res = _drain(conn, svc, inc)
    assert svc.comments_api.deletes == [] and not first_comment_corrections(_row(conn, did))
    assert any("Y2" in e for e in res.errors)


def test_an_extra_with_no_author_in_the_read_is_never_deleted(conn, inc):
    did = _extra_for(conn, inc)
    svc = _faithful_service({"Y": NEW_TEXT, "Y2": NEW_TEXT}, authors={"Y2": None})
    res = _drain(conn, svc, inc)
    assert svc.comments_api.deletes == [] and any("Y2" in e for e in res.errors)
    assert not first_comment_corrections(_row(conn, did))


# --- C1 gaps: M08, M15/M17, A19 -------------------------------------------------------------


def test_m08_an_extra_equal_to_the_recorded_live_id_is_never_read_or_deleted(conn, inc):
    did = _extra_for(conn, inc, live_id="Y2")  # a real queued extra, which we then re-point at Y
    item = first_comment_corrections(_row(conn, did))[0]
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_correction=? WHERE id=?",
            (json.dumps([{**item, "live_id": "Y"}]), did),
        )
    svc = _faithful_service({"Y": NEW_TEXT})
    _drain(conn, svc, inc)
    assert svc.comments_api.reads == [] and svc.comments_api.deletes == []
    assert "Y" in svc.comments_api.texts and not first_comment_corrections(_row(conn, did))


@pytest.mark.parametrize("platform", PLATFORMS)
def test_m15_a_no_from_copy_of_our_text_is_reported_once_on_meta(platform, caplog):
    graph = NoFromGraph(platform, [("V", NEW, None)])
    with caplog.at_level(logging.WARNING):
        current, _ = fc._existing_graph_comment(
            graph, "https://g", "post", NEW, "t", text_field=graph.field,
            permission="p", raise_for_error=lambda s, b: None, own_id=graph.own,
        )  # fmt: skip
    assert [str(c[0]) for c in current] == ["V"] and isinstance(current[0][0], fc.ForeignCommentId)
    assert current[0][0].unknown_author
    assert len([r for r in caplog.records if "no author" in r.message]) == 1


def test_a19_an_extra_is_not_queued_when_the_recorded_id_moves_under_the_read(
    tmp_path, conn, inc, monkeypatch
):
    from gomoveshift.publishing import queue

    did = _extra_for(conn, inc, live_id="Y")  # Y is recorded
    other = open_queue_db(tmp_path / "q")
    real = queue.first_comment_corrections

    def moved(row):
        with other:  # another drain re-records the comment between our read and our write
            other.execute("UPDATE deliveries SET first_comment_live_id='Z' WHERE id=?", (did,))
        return real(row)

    monkeypatch.setattr(queue, "first_comment_corrections", moved)
    assert queue.record_stale_live_comment(conn, did, NEW_TEXT, _NOW, "Y3") is None
    monkeypatch.setattr(queue, "first_comment_corrections", real)
    assert not [i for i in real(_row(conn, did)) if i["live_id"] == "Y3"]
    other.close()
