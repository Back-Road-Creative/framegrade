"""A removed row adopts only our CURRENT link, never the replaced or deleted one; a comment a
read-back confirms gone is reposted, never left as zero live behind an incident that cannot
close. Each test asserts the GOOD outcome. No network: the repo's own fakes."""

from __future__ import annotations

import json
import logging

import pytest

from gomoveshift.publishing import first_comment, meta_comments
from gomoveshift.publishing.first_comment_stale import (
    OWN_ID_KIND,
    REMOVED_KIND,
    REPOST_KIND,
    raise_repost_incident,
    recheck_own_id,
    restore_removed,
)
from gomoveshift.publishing.queue import (
    FIRST_COMMENT_MAX_ATTEMPTS,
    adopt_removed_first_comment,
    first_comment_prior_texts,
    open_queue_db,
    reopen_gone_first_comment,
)
from gomoveshift.publishing.youtube_comments import FirstCommentResult, YouTubeCommentEditor
from tests.test_first_comment_empty_gone_private import _open, _removed
from tests.test_first_comment_meta_correction import NEW, OLD, FakeGraph, Resp, _adapter
from tests.test_first_comment_meta_correction import _open_incidents
from tests.test_first_comment_meta_correction import _relinked as _meta_relinked
from tests.test_first_comment_meta_correction import _row as _meta_row
from tests.test_first_comment_stale import (
    _NOW,
    NEW_TEXT,
    OLD_TEXT,
    OWN,
    _drain,
    _row,
    _Service,
    _thread,
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


# --- MAJOR-1: adopt takes only the current link -------------------------------------------------


def _still_removed(conn, inc, did):
    assert _row(conn, did)["first_comment_removed_at"]
    assert len(_open(inc, "first_comment_removed")) == 1


def test_a_live_own_copy_of_the_replaced_link_is_never_adopted(conn, inc):
    did = _removed(conn, inc, restore=False)
    assert OLD_TEXT in first_comment_prior_texts(_row(conn, did))
    res = _drain(conn, _Service(pages=[{"items": [_thread(OLD_TEXT, "old-copy")]}]), inc)
    _still_removed(conn, inc, did)
    assert _row(conn, did)["first_comment_live_id"] != "old-copy"
    assert any(f"delivery {did}" in e and "old-copy" in e for e in res.errors), res.errors


def test_the_current_hand_re_add_wins_over_an_older_copy_of_the_replaced_link(conn, inc):
    did = _removed(conn, inc, restore=False)
    page = {"items": [_thread(OLD_TEXT, "old-copy"), _thread(NEW_TEXT, "hand")]}
    _drain(conn, _Service(pages=[page]), inc)
    row = _row(conn, did)
    assert row["first_comment_live_id"] == "hand" and row["first_comment_text"] == NEW_TEXT
    assert row["first_comment_removed_at"] is None and not _open(inc, "first_comment_removed")
    again = _Service(pages=[page])
    _drain(conn, again, inc)
    assert again.comments_api.updates == [] and again.comments_api.deletes == []
    assert _row(conn, did)["first_comment_text"] == NEW_TEXT


def test_the_comment_we_deleted_is_never_adopted_back_from_a_stale_listing(conn, inc):
    did = _removed(conn, inc, restore=False)
    gone_id = _row(conn, did)["first_comment_live_id"]
    assert gone_id
    res = _drain(conn, _Service(pages=[{"items": [_thread(NEW_TEXT, gone_id)]}]), inc)
    _still_removed(conn, inc, did)
    assert any(f"delivery {did}" in e and gone_id in e for e in res.errors), res.errors


def test_two_distinct_current_candidates_adopt_none_and_name_both(conn, inc):
    did = _removed(conn, inc, restore=False)
    other = "Watch the full drive: https://youtu.be/other"
    page = {"items": [_thread(NEW_TEXT, "hand-1"), _thread(other, "hand-2")]}
    res = _drain(conn, _Service(pages=[page]), inc)
    _still_removed(conn, inc, did)
    assert _row(conn, did)["first_comment_live_id"] not in {"hand-1", "hand-2"}
    assert any("hand-1" in e and "hand-2" in e and f"delivery {did}" in e for e in res.errors)


def test_listing_order_never_decides_between_candidates(conn, inc):
    did = _removed(conn, inc, restore=False)
    other = "Watch the full drive: https://youtu.be/other"
    for order in ([("a", NEW_TEXT), ("b", other)], [("b", other), ("a", NEW_TEXT)]):
        page = {"items": [_thread(t, i) for i, t in order]}
        _drain(conn, _Service(pages=[page]), inc)
        assert _row(conn, did)["first_comment_removed_at"]


def test_the_adopt_update_refuses_a_replaced_text_and_the_deleted_id(conn, inc):
    did = _removed(conn, inc, restore=False)
    gone_id = _row(conn, did)["first_comment_live_id"]
    assert not adopt_removed_first_comment(conn, did, "x1", OLD_TEXT, _NOW)
    assert not adopt_removed_first_comment(conn, did, gone_id, NEW_TEXT, _NOW)
    assert _row(conn, did)["first_comment_removed_at"]
    assert adopt_removed_first_comment(conn, did, "hand", NEW_TEXT, _NOW)
    assert _row(conn, did)["first_comment_live_id"] == "hand"


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("first_comment_correction", json.dumps([{"live_id": "zz", "live_text": "x"}])),
        ("first_comment_text", "some text"),
        ("first_comment_removed_at", None),
    ],
)
def test_the_adopt_update_refuses_a_pending_correction_text_or_a_row_not_removed(
    conn, inc, column, value
):
    did = _removed(conn, inc, restore=False)
    with conn:
        conn.execute(f"UPDATE deliveries SET {column}=? WHERE id=?", (value, did))  # noqa: S608
    assert not adopt_removed_first_comment(conn, did, "hand", NEW_TEXT, _NOW)
    assert _row(conn, did)["first_comment_live_id"] != "hand"


def test_a_guarded_adopt_that_writes_nothing_says_why(conn, inc, caplog):
    did = _removed(conn, inc, restore=False)
    item = [{"live_id": "zz", "live_text": "x", "action": "delete", "extra": True}]
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_correction=? WHERE id=?", (json.dumps(item), did)
        )
    res = FirstCommentResult()
    svc = _Service(pages=[{"items": [_thread(NEW_TEXT, "hand")]}])
    with caplog.at_level(logging.WARNING):
        restore_removed(conn, ("youtube",), inc, lambda p: YouTubeCommentEditor(svc), res)
    assert _row(conn, did)["first_comment_removed_at"]
    assert any(f"delivery {did}" in e and "hand" in e for e in res.errors), res.errors
    assert any(r.levelno == logging.WARNING and "hand" in r.getMessage() for r in caplog.records)


def test_a_cut_short_listing_never_adopts_what_it_did_read(conn, inc):
    did = _removed(conn, inc, restore=False)
    pages = [{"items": [_thread(NEW_TEXT, "hand")], "nextPageToken": "t0"}]
    pages += [{"items": [], "nextPageToken": f"t{i}"} for i in range(1, 20)]
    res = _drain(conn, _Service(pages=pages), inc)
    _still_removed(conn, inc, did)
    assert any(f"delivery {did}" in e for e in res.errors), res.errors


def test_a_hand_re_add_on_a_later_page_is_adopted(conn, inc):
    did = _removed(conn, inc, restore=False)
    pages = [{"items": [], "nextPageToken": "t1"}, {"items": [_thread(NEW_TEXT, "hand")]}]
    service = _Service(pages=pages)
    _drain(conn, service, inc)
    assert _row(conn, did)["first_comment_live_id"] == "hand"
    # The second request must carry the page token (the fake serves pages in order either way).
    assert any(kw.get("pageToken") == "t1" for kw in service.lists), service.lists


@pytest.mark.parametrize(
    "text",
    [
        "x Watch the full drive: https://youtu.be/new",
        "Watch the full drive: https://youtu.be/new extra",
        "WATCH THE FULL DRIVE: https://youtu.be/new",
    ],
)
def test_the_shape_rejects_near_misses(text):
    assert not first_comment.is_full_drive_comment(text)


def test_a_graph_listing_with_no_own_id_raises():
    graph = FakeGraph("facebook", [("hand", NEW)])
    with pytest.raises(first_comment.CommentCheckError):
        first_comment.list_own_graph_comments(
            graph, "b", "obj", "t", text_field="message", permission="x",
            raise_for_error=lambda s, b: None, own_id="",
        )  # fmt: skip


def test_the_graph_listing_asks_for_the_author_and_lists_only_our_comments():
    graph = FakeGraph("facebook", [("v", NEW, "someone"), ("n", NEW, ""), ("hand", NEW)])
    found, complete = _adapter("facebook", graph).own_comments("reel1")
    assert found == [("hand", NEW)] and complete


# --- MAJOR-2: a confirmed-gone read-back is reposted --------------------------------------------


def _fb(graph, page_id):
    return _adapter("facebook", graph).__class__(
        {"page_id": page_id, "page_access_token": "t"}, session=graph
    )


def _drain_as(conn, adapter, inc):
    return meta_comments.drain_meta_first_comments(
        conn, adapter_factories={"facebook": lambda: adapter}, now=_NOW, incidents_dir=inc
    )


def _gone_on_read(graph, only=None):
    """A comment the read-back finds gone at once (the listing confirms it)."""
    real = graph.get

    def get(url, params=None, **kw):
        cid = url.rsplit("/", 1)[1]
        if not url.endswith("/comments") and (only is None or cid in only):
            graph.comments.pop(cid, None)
        return real(url, params=params, **kw)

    graph.get = get


def test_a_gone_read_back_is_reposted_and_leaves_one_live_and_no_incident(conn, inc):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    _gone_on_read(graph, only={"new1"})
    res = _drain_as(conn, _fb(graph, "p"), inc)
    row = _meta_row(conn, did)
    assert row["first_comment_posted_at"] is None and row["first_comment_live_id"] is None
    assert row["first_comment_attempts"] == 1
    assert any(f"delivery {did}" in e and "gone" in e for e in res.errors), res.errors
    kinds = {i["kind"] for i in _open_incidents(inc)}
    assert kinds == {REPOST_KIND}  # surfaced, and not blamed on the id config
    for _ in range(3):
        _drain_as(conn, _fb(graph, "p"), inc)
    assert graph.comments == {"new2": NEW}  # exactly one of ours live
    assert _meta_row(conn, did)["first_comment_live_id"] == "new2"
    assert not _open_incidents(inc)


def test_a_comment_that_is_gone_after_every_repost_stops_at_the_attempts_cap(conn, inc):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    _gone_on_read(graph)
    for _ in range(FIRST_COMMENT_MAX_ATTEMPTS + 4):
        _drain_as(conn, _fb(graph, "p"), inc)
    posts = [e for e in graph.log if e.startswith("post:")]
    assert len(posts) == FIRST_COMMENT_MAX_ATTEMPTS  # never reposted forever
    row = _meta_row(conn, did)
    assert row["first_comment_posted_at"] is None
    assert row["first_comment_attempts"] == FIRST_COMMENT_MAX_ATTEMPTS
    assert [i["kind"] for i in _open_incidents(inc)] == [REPOST_KIND]  # and it stays surfaced


def test_a_read_back_by_another_author_is_still_an_own_id_mismatch(conn, inc):
    did = _meta_relinked(conn, "facebook")
    res = _drain_as(conn, _fb(FakeGraph("facebook", []), "not-p"), inc)
    assert _meta_row(conn, did)["first_comment_posted_at"]
    assert [i["kind"] for i in _open_incidents(inc)] == [OWN_ID_KIND]
    assert any("not our own id" in e for e in res.errors), res.errors


def test_the_own_id_incident_says_a_gone_comment_is_reposted(conn, inc):
    did = _meta_relinked(conn, "facebook")
    raise_repost_incident(conn, did, "someone-else", inc, kind=OWN_ID_KIND)
    (incident,) = _open_incidents(inc)
    assert "gone" in incident["detail"] and "repost" in incident["detail"]


def test_a_read_that_answers_gone_but_the_listing_still_shows_it_is_unverified_not_gone(conn, inc):
    """The read-back must ask the post's listing to confirm (``object_id``): a 100/33 for a
    comment the listing shows live is UNVERIFIED, and the live comment is never reposted."""
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    real = graph.get

    def get(url, params=None, **kw):
        if not url.endswith("/comments"):
            return Resp({"error": {"code": 100, "error_subcode": 33, "message": "no"}}, 400)
        return real(url, params=params, **kw)

    graph.get = get
    res = _drain_as(conn, _fb(graph, "p"), inc)
    row = _meta_row(conn, did)
    assert row["first_comment_posted_at"] and row["first_comment_live_id"] == "new1"
    assert row["first_comment_attempts"] == 0
    assert any("could not be verified" in e for e in res.errors), res.errors
    assert not any("reposts it" in e for e in res.errors)  # never taken for a gone comment
    assert graph.comments == {"new1": NEW}
    assert [i["kind"] for i in _open_incidents(inc)] == [OWN_ID_KIND]


def test_an_open_own_id_incident_whose_comment_reads_gone_is_resolved_and_reposted(conn, inc):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    _drain_as(conn, _fb(graph, "not-p"), inc)  # new1 posted, own-id incident open
    assert [i["kind"] for i in _open_incidents(inc)] == [OWN_ID_KIND]
    graph.comments.pop("new1")  # the comment is gone from the platform
    res = _drain_as(conn, _fb(graph, "p"), inc)
    (own,) = [json.loads(p.read_text()) for p in inc.glob("*.json") if OWN_ID_KIND in p.read_text()]
    assert own["resolved_at"] and "gone" in own["resolution"]
    assert graph.comments == {"new2": NEW}
    assert _meta_row(conn, did)["first_comment_live_id"] == "new2"
    assert any(f"delivery {did}" in e and "gone" in e for e in res.errors), res.errors
    _drain_as(conn, _fb(graph, "p"), inc)
    assert not _open_incidents(inc) and graph.comments == {"new2": NEW}


def test_an_own_id_incident_on_a_row_with_no_text_that_reads_gone_routes_to_removed(conn, inc):
    did = _meta_relinked(conn, "facebook")
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_text='', first_comment_posted_at=?, "
            "first_comment_live_id='c-gone' WHERE id=?",
            (_NOW, did),
        )
    raise_repost_incident(conn, did, "someone-else", inc, kind=OWN_ID_KIND)
    graph = FakeGraph("facebook", [])
    _drain_as(conn, _fb(graph, "p"), inc)
    assert _meta_row(conn, did)["first_comment_removed_at"]
    assert REMOVED_KIND in {i["kind"] for i in _open_incidents(inc)}
    assert OWN_ID_KIND not in {i["kind"] for i in _open_incidents(inc)}


def test_the_gone_route_never_touches_a_row_with_a_pending_correction(conn, inc):
    did = _meta_relinked(conn, "facebook")
    item = json.dumps([{"live_id": "zz", "live_text": "x", "action": "delete"}])
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_posted_at=?, first_comment_live_id='c1', "
            "first_comment_correction=? WHERE id=?",
            (_NOW, item, did),
        )
    assert not reopen_gone_first_comment(conn, did, "c1", _NOW, "gone")
    row = _meta_row(conn, did)
    assert row["first_comment_posted_at"] and row["first_comment_live_id"] == "c1"


def test_the_gone_route_never_reposts_a_row_whose_live_id_moved(conn, inc):
    did = _meta_relinked(conn, "facebook")
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_posted_at=?, first_comment_live_id='c2' "
            "WHERE id=?",
            (_NOW, did),
        )
    assert not reopen_gone_first_comment(conn, did, "c1", _NOW, "gone")
    assert _meta_row(conn, did)["first_comment_posted_at"]


# --- MINOR-2 / A20: the recheck's failed read -------------------------------------------------


def test_a_failed_recheck_read_leaves_the_incident_open_and_says_so(conn, inc):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    _drain_as(conn, _fb(graph, "not-p"), inc)
    graph.read_raises = True
    res = FirstCommentResult()
    recheck_own_id(conn, ("facebook",), lambda p: _fb(graph, "p"), inc, res)
    assert [i["kind"] for i in _open_incidents(inc)] == [OWN_ID_KIND]
    assert any(f"delivery {did}" in e and "connection reset" in e for e in res.errors)
    assert _meta_row(conn, did)["first_comment_live_id"] == "new1"
    graph.read_raises = False
    _drain_as(conn, _fb(graph, "p"), inc)
    assert not _open_incidents(inc)
    assert not [e for e in graph.log if e.startswith("delete")]


def test_the_drain_reports_a_failed_recheck_read(conn, inc):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    _drain_as(conn, _fb(graph, "not-p"), inc)
    graph.read_raises = True
    res = _drain_as(conn, _fb(graph, "p"), inc)
    assert any(f"delivery {did}" in e and "connection reset" in e for e in res.errors)


# --- MINOR-4: a hand re-add must be the exact text ----------------------------------------------


def _sidecar(tmp_path, video_id):
    path = tmp_path / "sidecar.json"
    shorts = [{"short_index": 1, "project_name": "proj", "parent_lf_video_id": video_id}]
    path.write_text(json.dumps({"shorts": shorts}))
    return str(path)


def test_the_removed_incident_gives_the_exact_text_with_the_real_id(conn, inc, tmp_path):
    did = _meta_relinked(conn, "facebook")
    with conn:
        conn.execute(
            "UPDATE deliveries SET sidecar_path=? WHERE id=?", (_sidecar(tmp_path, "abc123"), did)
        )
    raise_repost_incident(conn, did, "private", inc, kind=REMOVED_KIND)
    (incident,) = _open_incidents(inc)
    assert first_comment.full_drive_comment_text("abc123") in incident["detail"]
    assert "<id>" not in incident["detail"]


def test_the_removed_incident_with_no_known_id_still_names_the_shape(conn, inc):
    did = _meta_relinked(conn, "facebook")
    raise_repost_incident(conn, did, "private", inc, kind=REMOVED_KIND)
    (incident,) = _open_incidents(inc)
    assert "Watch the full drive: https://youtu.be/<" in incident["detail"]


def test_the_docs_give_the_exact_hand_re_add_text():
    from pathlib import Path

    docs = (Path(__file__).resolve().parents[1] / "docs/social-publishing-queue.md").read_text()
    assert "Watch the full drive: https://youtu.be/<id>" in docs
