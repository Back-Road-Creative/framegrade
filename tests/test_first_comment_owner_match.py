"""Only a comment positively ours (its author is our channel / Page / IG user) is ever deleted,
edited or adopted; an unknown author is a viewer's. An extra delete re-reads its comment, two
own copies end as one on Meta, and an old row with no recorded comment reports a second copy.
No network: the repo's own fakes."""

from __future__ import annotations

import json
import logging
import sqlite3

import pytest

from gomoveshift.publishing import youtube_comments as yc
from gomoveshift.publishing.first_comment import ForeignCommentId
from gomoveshift.publishing.queue import first_comment_corrections, open_queue_db
from tests.test_first_comment_meta_correction import NEW, OLD, FakeGraph
from tests.test_first_comment_meta_correction import _drain as _meta_drain
from tests.test_first_comment_meta_correction import _relinked as _meta_relinked
from tests.test_first_comment_meta_correction import _row as _meta_row
from tests.test_first_comment_stale import (
    NEW_TEXT,
    OLD_TEXT,
    OWN,
    _drain,
    _published,
    _rewrite,
    _row,
    _Service,
    _thread,
)

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


def _ids(row):
    return [(i["live_id"], i["action"]) for i in first_comment_corrections(row)]


class NoFromGraph(FakeGraph):
    """A comment whose author is None comes back with no ``from`` field at all."""

    def get(self, url, params=None, **kw):
        resp = super().get(url, params=params, **kw)
        if url.endswith("/comments"):
            for c in resp.body["data"]:
                if self.authors.get(c["id"]) is None:
                    c.pop("from", None)
        return resp


class FaithfulComments:
    """A YouTube comments API whose ``list(id=...)`` answers each comment's real text."""

    def __init__(self, texts, authors=None):
        self.texts, self.reads, self.deletes = dict(texts), [], []
        self.authors = authors or {}  # id -> author channel id (None: none given); default ours

    def list(self, part, id, **kw):  # noqa: A002 -- the API's own name
        self.reads.append(id)
        snippet = {"textOriginal": self.texts.get(id)}
        author = self.authors.get(id, OWN)
        if author:
            snippet["authorChannelId"] = {"value": author}
        items = [{"id": id, "snippet": snippet}] if id in self.texts else []
        return type("R", (), {"execute": lambda self_: {"items": items}})()

    def delete(self, id):  # noqa: A002 -- the API's own name
        self.deletes.append(id)
        self.texts.pop(id, None)
        return type("R", (), {"execute": lambda self_: None})()


def _faithful_service(texts, authors=None):
    svc = _Service()
    svc.comments_api = FaithfulComments(texts, authors)
    return svc


# --- a comment with no author is a viewer's: never adopted, never deleted ---------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_x1_meta_a_no_from_copy_beside_ours_is_never_deleted(conn, inc, platform):
    did = _meta_relinked(conn, platform)
    graph = NoFromGraph(platform, [("c1", OLD), ("y1", NEW), ("v1", NEW, None)], page=10)
    _meta_drain(conn, platform, graph, inc)
    assert _ids(_meta_row(conn, did)) == [("c1", "delete")]  # ours, the replaced one
    _meta_drain(conn, platform, graph, inc)
    assert "v1" in graph.comments and "delete:v1" not in graph.log
    assert _meta_row(conn, did)["first_comment_live_id"] == "y1"
    assert list(graph.comments) == ["y1", "v1"]


@pytest.mark.parametrize("platform", PLATFORMS)
def test_x1b_meta_a_no_from_copy_is_never_adopted_as_ours(conn, inc, platform):
    did = _meta_relinked(conn, platform)
    graph = NoFromGraph(platform, [("c1", OLD), ("v1", NEW, None)], page=10)
    _meta_drain(conn, platform, graph, inc)
    assert _meta_row(conn, did)["first_comment_live_id"] == "c1"
    _meta_drain(conn, platform, graph, inc)
    assert "v1" in graph.comments and _meta_row(conn, did)["first_comment_live_id"] != "v1"
    assert [t for i, t in graph.comments.items() if i != "v1"] == [NEW]  # ours says NEW


def test_x2_youtube_a_no_author_copy_beside_ours_is_never_deleted(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    page = {
        "items": [_thread(NEW_TEXT, "Y"), _thread(NEW_TEXT, "V", author=""), _thread(OLD_TEXT, "X")]
    }
    _drain(conn, _Service(pages=[page]), inc)
    assert ("V", "delete") not in _ids(_row(conn, did))
    nxt = _Service()
    _drain(conn, nxt, inc)
    assert "V" not in nxt.comments_api.deletes and _row(conn, did)["first_comment_live_id"] == "Y"


def test_x2b_youtube_a_no_author_copy_listed_first_never_displaces_ours(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    page = {
        "items": [_thread(NEW_TEXT, "V", author=""), _thread(NEW_TEXT, "Y"), _thread(OLD_TEXT, "X")]
    }
    _drain(conn, _Service(pages=[page]), inc)
    row = _row(conn, did)
    assert row["first_comment_live_id"] == "Y" and ("Y", "delete") not in _ids(row)
    nxt = _Service()
    _drain(conn, nxt, inc)
    assert "Y" not in nxt.comments_api.deletes and "V" not in nxt.comments_api.deletes


def test_a_no_author_copy_of_our_text_is_reported_once(conn, caplog):
    page = {"items": [_thread(NEW_TEXT, "V", author="")]}
    with caplog.at_level(logging.WARNING):
        current, _ = yc._scan_first_comments(_Service(pages=[page]), "vid", NEW_TEXT)
    assert [str(c[0]) for c in current] == ["V"]
    assert isinstance(current[0][0], ForeignCommentId)
    assert [r.message for r in caplog.records if "no author" in r.message] != []


# --- a viewer's copy is the only current match: ours is posted (kills the pass-through gaps) --


def test_x9_youtube_a_viewers_only_copy_still_gets_ours_posted(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    svc = _Service(pages=[{"items": [_thread(NEW_TEXT, "V", author="UC-viewer")]}])
    _drain(conn, svc, inc)
    row = _row(conn, did)
    assert svc.inserted == [NEW_TEXT] and row["first_comment_posted_at"]
    assert row["first_comment_live_id"] == "new-thread"


@pytest.mark.parametrize("platform", PLATFORMS)
def test_x9m_meta_a_viewers_only_copy_still_gets_ours_posted(conn, inc, platform):
    did = _meta_relinked(conn, platform)
    graph = FakeGraph(platform, [("v1", NEW, "stranger")])
    _meta_drain(conn, platform, graph, inc)
    row = _meta_row(conn, did)
    assert sorted(graph.comments.values()) == [NEW, NEW] and row["first_comment_posted_at"]
    assert row["first_comment_live_id"] == "new1" and "delete:v1" not in graph.log


# --- two own copies of the current text on Meta end as one ------------------------------------


@pytest.mark.parametrize("platform", PLATFORMS)
def test_x10_meta_two_own_copies_end_with_exactly_one(conn, inc, platform):
    did = _meta_relinked(conn, platform)
    graph = FakeGraph(platform, [("o1", NEW), ("o2", NEW)])
    for _ in range(3):
        _meta_drain(conn, platform, graph, inc)
    row = _meta_row(conn, did)
    assert row["first_comment_posted_at"] and list(graph.comments) == ["o1"]
    assert row["first_comment_live_id"] == "o1" and not first_comment_corrections(row)


# --- an extra delete re-reads its comment and only ever deletes one that is ours --------------


def _extra_for(conn, inc, live_id="Y2", ours=True):
    did = _published(conn)
    _rewrite(conn, did)
    r = yc.FirstCommentResult()
    yc.mark_posted(conn, did, NEW_TEXT, r, "Y", inc)
    yc.record_found_comments(conn, did, NEW_TEXT, live_id, NEW_TEXT, (), r, inc)
    if not ours:
        items = [
            {k: v for k, v in i.items() if k != "ours"}
            for i in first_comment_corrections(_row(conn, did))
        ]
        with conn:
            conn.execute(
                "UPDATE deliveries SET first_comment_correction=? WHERE id=?",
                (json.dumps(items), did),
            )
    return did


def test_an_extra_delete_re_reads_then_deletes_our_own_duplicate(conn, inc):
    did = _extra_for(conn, inc)
    svc = _faithful_service({"Y": NEW_TEXT, "Y2": NEW_TEXT})
    _drain(conn, svc, inc)
    assert svc.comments_api.reads == ["Y2"] and svc.comments_api.deletes == ["Y2"]
    assert not first_comment_corrections(_row(conn, did))


def test_an_extra_delete_never_runs_on_a_comment_not_marked_ours(conn, inc):
    did = _extra_for(conn, inc, ours=False)
    svc = _faithful_service({"Y": NEW_TEXT, "Y2": NEW_TEXT})
    _drain(conn, svc, inc)
    assert svc.comments_api.deletes == [] and "Y2" in svc.comments_api.texts
    assert not first_comment_corrections(_row(conn, did))  # dropped, not retried forever


def test_an_extra_delete_never_runs_on_the_recorded_comment(conn, inc):
    did = _extra_for(conn, inc, live_id="Y")
    svc = _faithful_service({"Y": NEW_TEXT})
    _drain(conn, svc, inc)
    assert svc.comments_api.deletes == [] and "Y" in svc.comments_api.texts
    assert not first_comment_corrections(_row(conn, did))


def test_an_extra_delete_of_a_comment_already_gone_is_done_without_a_delete(conn, inc):
    did = _extra_for(conn, inc)
    svc = _faithful_service({"Y": NEW_TEXT})
    _drain(conn, svc, inc)
    assert svc.comments_api.deletes == [] and not first_comment_corrections(_row(conn, did))


# --- an old posted row with no recorded comment id reports a second copy ----------------------


def test_x7_an_old_row_with_no_recorded_comment_reports_a_second_copy_once(tmp_path, inc):
    path = tmp_path / "old"
    c = open_queue_db(path)
    did = _published(c)
    yc.mark_posted(c, did, OLD_TEXT, yc.FirstCommentResult(), "Y", inc)
    c.close()
    raw = sqlite3.connect(path / "social_queue.sqlite3")
    raw.execute("ALTER TABLE deliveries DROP COLUMN first_comment_live_id")
    raw.commit()
    raw.close()
    c = open_queue_db(path)
    try:
        r = yc.FirstCommentResult()
        yc.mark_posted(c, did, OLD_TEXT, r, "Y2", inc)  # a second copy on a pre-column row
        row = _row(c, did)
        assert _ids(row) == []  # nothing is deleted: which copy is ours to keep is unknown
        assert len(r.errors) == 1 and "Y2" in r.errors[0] and "by hand" in r.errors[0]
        assert "Y2" in row["first_comment_error"]
    finally:
        c.close()
