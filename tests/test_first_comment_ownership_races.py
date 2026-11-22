"""A viewer's comment is never ours (never marked posted, edited or deleted), a stale observation
never deletes a comment that already says an intended text, and a second own copy of the current
text is recorded. Compare-and-set refusals and the Instagram stranger rule are pinned too.
No network: the repo's own fakes."""

from __future__ import annotations

import json

import pytest

from gomoveshift.publishing import first_comment, queue
from gomoveshift.publishing import youtube_comments as yc
from gomoveshift.publishing.queue import (
    first_comment_corrections,
    open_queue_db,
    rewrite_unposted_first_comment,
)
from tests.test_first_comment_correction_safety import _posted_with_edit
from tests.test_first_comment_meta_correction import NEW, OLD, FakeGraph
from tests.test_first_comment_meta_correction import _drain as _meta_drain
from tests.test_first_comment_meta_correction import _relinked as _meta_relinked
from tests.test_first_comment_meta_correction import _row as _meta_row
from tests.test_first_comment_stale import (
    _NOW,
    NEW_TEXT,
    OLD_TEXT,
    _drain,
    _incidents,
    _published,
    _rewrite,
    _row,
    _Service,
    _thread,
)

KW = dict(text_field="message", permission="p", raise_for_error=lambda s, b: None)


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


def _foreign(cid):
    return first_comment.ForeignCommentId(cid)


# --- F1: a viewer's exact-text comment is never ours ------------------------------------------


def test_p1_a_viewers_copy_found_by_a_slow_drain_is_never_a_delete_target_youtube(conn, inc):
    did = _published(conn)
    viewer = _Service(pages=[{"items": [_thread(OLD_TEXT, "V", author="UC-viewer")]}])
    live_id, said, dupes = yc.post_first_comment_reporting(viewer, "vid", OLD_TEXT)
    assert isinstance(live_id, first_comment.ForeignCommentId) and said == OLD_TEXT
    _rewrite(conn, did)  # the relink
    r = yc.FirstCommentResult()
    yc.mark_posted(conn, did, NEW_TEXT, r, "Y", inc)  # drain A posted ours
    yc.record_found_comments(conn, did, OLD_TEXT, live_id, said, dupes, r, inc)  # slow drain B
    assert _ids(_row(conn, did)) == []
    nxt = _Service()
    _drain(conn, nxt, inc)
    assert nxt.comments_api.deletes == [] and nxt.comments_api.updates == []


def test_p1b_a_viewers_copy_never_marks_the_row_posted_or_becomes_an_edit_target(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    with conn:
        conn.execute("UPDATE deliveries SET first_comment_text=? WHERE id=?", (OLD_TEXT, did))
    assert rewrite_unposted_first_comment(conn, did, OLD_TEXT, NEW_TEXT, _NOW)  # relink lands
    r = yc.FirstCommentResult()
    yc.record_found_comments(conn, did, OLD_TEXT, _foreign("V"), OLD_TEXT, (), r, inc)
    row = _row(conn, did)
    assert row["first_comment_posted_at"] is None and _ids(row) == []
    nxt = _Service()
    res = _drain(conn, nxt, inc)  # the next scan posts OUR comment
    assert nxt.inserted == [NEW_TEXT] and nxt.comments_api.updates == [] and res.posted == 1


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_p1m_a_viewers_copy_is_never_a_delete_target_on_meta(conn, inc, platform):
    did = _meta_relinked(conn, platform)  # unposted, NEW (prior OLD)
    with conn:
        conn.execute("UPDATE deliveries SET first_comment_text=? WHERE id=?", (OLD, did))
    graph = FakeGraph(platform, [("v1", OLD, "stranger")])
    kw = dict(text_field=graph.field, permission="p", raise_for_error=lambda s, b: None)
    found = first_comment.post_graph_comment(
        graph, "https://g", "obj", OLD, "t", own_id=graph.own, **kw
    )
    assert found == "v1" and isinstance(found, first_comment.ForeignCommentId)
    assert rewrite_unposted_first_comment(conn, did, OLD, NEW, _NOW)  # the relink
    r = yc.FirstCommentResult()
    yc.mark_posted(conn, did, NEW, r, "c-ours", inc)  # drain A posted ours
    yc.record_found_comments(conn, did, OLD, found, OLD, (), r, inc)  # drain B
    assert _ids(_meta_row(conn, did)) == []
    graph.comments["c-ours"], graph.authors["c-ours"] = NEW, graph.own
    _meta_drain(conn, platform, graph, inc)
    assert "v1" in graph.comments and not [e for e in graph.log if e.startswith("delete")]


def test_p1c_a_viewers_current_copy_never_stands_in_for_ours_youtube(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    page = {"items": [_thread(NEW_TEXT, "V", author="UC-viewer"), _thread(OLD_TEXT, "X")]}
    _drain(conn, _Service(pages=[page]), inc)
    assert _ids(_row(conn, did)) == [("X", "edit")]  # ours is edited, never deleted
    nxt = _Service()
    _drain(conn, nxt, inc)
    assert nxt.comments_api.deletes == [] and nxt.comments_api.updates == [("X", NEW_TEXT)]
    assert nxt.inserted == []


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_p1c_a_viewers_current_copy_never_stands_in_for_ours_meta(conn, inc, platform):
    did = _meta_relinked(conn, platform)
    graph = FakeGraph(platform, [("v1", NEW, "stranger"), ("x1", OLD)])
    _meta_drain(conn, platform, graph, inc)
    _meta_drain(conn, platform, graph, inc)
    assert graph.comments["v1"] == NEW  # the viewer's comment is untouched
    assert sorted(graph.comments.values()) == [NEW, NEW]  # ours now says NEW too
    assert "delete:v1" not in graph.log and not first_comment_corrections(_meta_row(conn, did))


def test_a_viewers_only_copy_still_satisfies_the_scan_but_is_flagged_foreign(conn):
    svc = _Service(pages=[{"items": [_thread(NEW_TEXT, "V", author="UC-viewer")]}])
    current, prior = yc._scan_first_comments(svc, "vid", NEW_TEXT)
    assert current == [("V", NEW_TEXT)] and prior == []
    assert isinstance(current[0][0], first_comment.ForeignCommentId)


def test_the_scan_prefers_our_own_current_match_over_a_viewers_youtube(conn):
    page = {"items": [_thread(NEW_TEXT, "V", author="UC-viewer"), _thread(NEW_TEXT, "mine")]}
    svc = _Service(pages=[page])
    live_id, said, dupes = yc.post_first_comment_reporting(svc, "vid", NEW_TEXT)
    assert (live_id, said) == ("mine", NEW_TEXT)
    assert not isinstance(live_id, first_comment.ForeignCommentId) and dupes == ()


@pytest.mark.parametrize("stranger_first", [True, False])
def test_n33_meta_prefers_our_own_current_match_over_a_strangers(stranger_first):
    rows = [("v1", NEW, "stranger"), ("c1", NEW)]
    graph = FakeGraph("facebook", rows if stranger_first else rows[::-1], page=5)
    got = first_comment.post_graph_comment(graph, "https://g", "obj", NEW, "t", own_id="p", **KW)
    assert got == "c1" and not isinstance(got, first_comment.ForeignCommentId)
    assert not [e for e in graph.log if e.startswith("post:")]


# --- F2: a stale observation never deletes a comment that already says an intended text -------


def test_p2_a_stale_observation_never_deletes_the_corrected_comment(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    r = yc.FirstCommentResult()
    yc.record_found_comments(conn, did, NEW_TEXT, "Y", OLD_TEXT, (), r, inc)  # drain A
    fix = _Service()
    _drain(conn, fix, inc)
    assert fix.comments_api.updates == [("Y", NEW_TEXT)]
    yc.record_found_comments(conn, did, NEW_TEXT, "Y", OLD_TEXT, (), r, inc)  # slow drain B
    nxt = _Service()
    nxt.comments_api.texts["Y"] = NEW_TEXT  # Y says NEW now
    _drain(conn, nxt, inc)
    assert nxt.comments_api.deletes == [] and nxt.comments_api.reads == ["Y"]
    row = _row(conn, did)
    assert not first_comment_corrections(row) and row["first_comment_error"] is None


def test_p2_a_failed_reread_leaves_the_delete_pending_and_visible(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    r = yc.FirstCommentResult()
    yc.record_found_comments(conn, did, NEW_TEXT, "Y", OLD_TEXT, (), r, inc)
    yc.record_found_comments(conn, did, NEW_TEXT, "Z", OLD_TEXT, (), r, inc)
    assert ("Z", "delete") in _ids(_row(conn, did))
    svc = _Service()
    svc.comments_api.read_fail = TimeoutError("read timed out")
    res = _drain(conn, svc, inc)
    assert svc.comments_api.deletes == [] and res.stale >= 1 and res.errors
    assert ("Z", "delete") in _ids(_row(conn, did)) and _incidents(inc)


def test_p2_a_stale_delete_still_runs_when_the_comment_says_a_stale_text(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    r = yc.FirstCommentResult()
    yc.record_found_comments(conn, did, NEW_TEXT, "Y", NEW_TEXT, (), r, inc)
    yc.record_found_comments(conn, did, NEW_TEXT, "Y", NEW_TEXT, (("W", OLD_TEXT),), r, inc)
    svc = _Service()
    svc.comments_api.texts["W"] = OLD_TEXT
    _drain(conn, svc, inc)
    assert svc.comments_api.deletes == ["W"] and svc.comments_api.reads == ["W"]


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_p2_meta_a_stale_delete_never_runs_on_a_comment_that_says_the_new_text(conn, inc, platform):
    did = _meta_relinked(conn, platform)
    graph = FakeGraph(platform, [("n1", NEW), ("o1", OLD)])
    _meta_drain(conn, platform, graph, inc)  # queues delete o1
    graph.comments["o1"] = NEW  # it was edited to the new text before the delete ran
    _meta_drain(conn, platform, graph, inc)
    assert "o1" in graph.comments and not [e for e in graph.log if e.startswith("delete")]
    assert "read:o1" in graph.log and not first_comment_corrections(_meta_row(conn, did))


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_p2_meta_a_failed_reread_never_deletes_blind(conn, inc, platform):
    did = _meta_relinked(conn, platform)
    graph = FakeGraph(platform, [("n1", NEW), ("o1", OLD)])
    _meta_drain(conn, platform, graph, inc)
    graph.read_raises = True
    res = _meta_drain(conn, platform, graph, inc)
    assert "o1" in graph.comments and not [e for e in graph.log if e.startswith("delete")]
    assert res.errors and first_comment_corrections(_meta_row(conn, did))


# --- F3: a second own copy of the current text is recorded, never kept silently ---------------


def test_p3_a_second_copy_of_the_current_text_gets_a_delete_and_the_posted_id_stays(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    r = yc.FirstCommentResult()
    yc.mark_posted(conn, did, NEW_TEXT, r, "Y", inc)
    yc.mark_posted(conn, did, NEW_TEXT, r, "Y2", inc)
    yc.mark_posted(conn, did, NEW_TEXT, r, "Y", inc)  # the posted id again: nothing to delete
    row = _row(conn, did)
    assert _ids(row) == [("Y2", "delete")] and _incidents(inc) and r.errors
    svc = _Service()
    _drain(conn, svc, inc)
    assert svc.comments_api.deletes == ["Y2"]  # Y is never touched
    assert not first_comment_corrections(_row(conn, did))


def test_p3_the_scan_reports_a_second_own_copy_as_a_duplicate(conn):
    page = {"items": [_thread(NEW_TEXT, "Y"), _thread(NEW_TEXT, "Y2")]}
    live_id, said, dupes = yc.post_first_comment_reporting(_Service(pages=[page]), "vid", NEW_TEXT)
    assert live_id == "Y" and dupes == (("Y2", NEW_TEXT),)


# --- the compare-and-set refusals and the Instagram stranger rule ----------------------------


def _interleave(conn, tmp_path, did, monkeypatch):
    """Make the next ``first_comment_corrections`` read be followed by a concurrent JSON write."""
    other = open_queue_db(tmp_path / "q")
    real = queue.first_comment_corrections
    fired = []

    def racing(row):
        out = real(row)
        if not fired:
            fired.append(1)
            items = out + [{"live_id": "Z", "live_text": "dead", "action": "delete"}]
            with other:
                other.execute(
                    "UPDATE deliveries SET first_comment_correction=? WHERE id=?",
                    (json.dumps(items), did),
                )
        return out

    monkeypatch.setattr(queue, "first_comment_corrections", racing)
    return real, other


def test_n03_an_interleaved_write_makes_the_unpost_refuse(conn, inc, tmp_path, monkeypatch):
    did = _posted_with_edit(conn, None, "2026-08-23T00:00:00+00:00")
    item = queue.first_comment_corrections(_row(conn, did))[0]
    real, other = _interleave(conn, tmp_path, did, monkeypatch)
    queue.unpost_first_comment_for_repost(conn, did, item, _NOW)
    monkeypatch.setattr(queue, "first_comment_corrections", real)
    other.close()
    row = _row(conn, did)
    assert row["first_comment_posted_at"] is not None  # still posted: the write refused
    assert ("Z", "delete") in _ids(row)  # and the concurrent item survives


def test_n23_an_interleaved_write_makes_the_resolve_refuse(conn, inc, tmp_path, monkeypatch):
    did = _posted_with_edit(conn, None, "2026-08-23T00:00:00+00:00")
    item = queue.first_comment_corrections(_row(conn, did))[0]
    real, other = _interleave(conn, tmp_path, did, monkeypatch)
    queue.resolve_first_comment_correction(conn, did, item, _NOW)
    monkeypatch.setattr(queue, "first_comment_corrections", real)
    other.close()
    assert ("Z", "delete") in _ids(_row(conn, did)) and ("c1", "edit") in _ids(_row(conn, did))


def test_n04_an_instagram_replace_never_deletes_a_strangers_comment():
    graph = FakeGraph("instagram", [("s1", OLD, "stranger")])
    first_comment.replace_graph_comment(
        graph, "https://g", "obj", "s1", NEW, "t", text_field="text", permission="p",
        raise_for_error=lambda s, b: None, own_id="u",
    )  # fmt: skip
    assert not [e for e in graph.log if e.startswith("delete")] and graph.comments["s1"] == OLD
