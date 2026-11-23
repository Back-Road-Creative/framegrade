"""A comment a read-back says is gone is recorded and re-read before it is ever reposted, so one
lagging read never posts a second live copy and a gone-every-time comment stops at the attempts
cap with the incident open. A removed row adopts only our current link, matched by video id, and
queues our own extra copies for deletion. Each test asserts the GOOD outcome. No network: the
repo's own fakes."""

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
)
from gomoveshift.publishing.queue import (
    FIRST_COMMENT_MAX_ATTEMPTS,
    adopt_removed_first_comment,
    first_comment_corrections,
    open_queue_db,
    reopen_gone_first_comment,
)
from gomoveshift.publishing.youtube_comments import FirstCommentResult
from tests.test_first_comment_empty_gone_private import _open, _removed
from tests.test_first_comment_meta_correction import NEW, OLD, FakeGraph, Resp, _adapter
from tests.test_first_comment_meta_correction import _open_incidents
from tests.test_first_comment_meta_correction import _relinked as _meta_relinked
from tests.test_first_comment_meta_correction import _row as _meta_row
from tests.test_first_comment_stale import (
    _NOW,
    NEW_TEXT,
    OLD_TEXT,
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


def _fb(graph, page_id="p"):
    return _adapter("facebook", graph).__class__(
        {"page_id": page_id, "page_access_token": "t"}, session=graph
    )


def _drain_as(conn, adapter, inc):
    return meta_comments.drain_meta_first_comments(
        conn, adapter_factories={"facebook": lambda: adapter}, now=_NOW, incidents_dir=inc
    )


def _gone_on_read(graph, only=None):
    """A single read of the comment (``GET /{id}``) finds it gone, and the listing agrees."""
    real = graph.get

    def get(url, params=None, **kw):
        cid = url.rsplit("/", 1)[1]
        if not url.endswith("/comments") and (only is None or cid in only):
            graph.comments.pop(cid, None)
        return real(url, params=params, **kw)

    graph.get = get


def _lagging(graph, drains_lagging):
    """Read-after-write lag on the FIRST post only: it answers 100/33 on a read and is missing
    from the listing until ``drains_lagging`` drains have ended; it IS live."""
    real_get, real_post = graph.get, graph.post
    state = {"hidden": set(), "left": {}, "posts": 0}

    def post(url, data=None, **kw):
        r = real_post(url, data=data, **kw)
        if url.endswith("/comments") and isinstance(r.body, dict) and "id" in r.body:
            state["posts"] += 1
            if state["posts"] == 1:
                state["hidden"].add(r.body["id"])
                state["left"][r.body["id"]] = drains_lagging
        return r

    def get(url, params=None, **kw):
        cid = url.rsplit("/", 1)[1]
        if not url.endswith("/comments"):
            if cid in state["hidden"]:
                return Resp({"error": {"code": 100, "error_subcode": 33, "message": "no"}}, 400)
            return real_get(url, params=params, **kw)
        saved = {h: graph.comments.pop(h) for h in list(state["hidden"]) if h in graph.comments}
        try:
            return real_get(url, params=params, **kw)
        finally:
            graph.comments.update(saved)

    graph.get, graph.post = get, post
    return state


def _end_drain(state):
    for cid in list(state["left"]):
        state["left"][cid] -= 1
        if state["left"][cid] <= 0:
            state["hidden"].discard(cid)
            del state["left"][cid]


# --- MAJOR-1 / MAJOR-2: a post-time gone is recorded, never reposted on that one read -----------


def test_a_post_time_gone_is_recorded_and_not_reposted_on_that_one_read(conn, inc):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    _gone_on_read(graph, only={"new1"})
    res = _drain_as(conn, _fb(graph), inc)
    row = _meta_row(conn, did)
    assert row["first_comment_live_id"] == "new1" and row["first_comment_posted_at"]
    assert row["first_comment_attempts"] == 0
    assert [e for e in graph.log if e.startswith("post:")] == ["post:new1"]  # one post, no repost
    assert any(f"delivery {did}" in e and "gone" in e and "re-read" in e for e in res.errors)
    (only,) = _open_incidents(inc)
    assert only["kind"] == OWN_ID_KIND and "re-read" in only["detail"]
    assert "author gone" not in only["detail"]  # never blamed on the id config


def test_a_gone_confirmed_twice_is_reposted_and_the_first_post_is_never_left_live(conn, inc):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    st = _lagging(graph, 2)
    for _ in range(4):
        _drain_as(conn, _fb(graph), inc)
        _end_drain(st)
    ours = [c for c, t in graph.comments.items() if t == NEW]
    assert len(ours) == 1, graph.comments  # exactly one of ours is live once the lag clears
    assert _meta_row(conn, did)["first_comment_live_id"] == ours[0]
    assert not _open_incidents(inc)


def test_lag_within_one_drain_still_posts_exactly_one(conn, inc):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    st = _lagging(graph, 1)
    for _ in range(3):
        _drain_as(conn, _fb(graph), inc)
        _end_drain(st)
    assert list(graph.comments) == ["new1"]
    assert _meta_row(conn, did)["first_comment_live_id"] == "new1"
    assert not _open_incidents(inc)


def test_a_gone_every_time_comment_stops_at_the_cap_with_the_incident_open(conn, inc):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    _gone_on_read(graph)
    for _ in range(FIRST_COMMENT_MAX_ATTEMPTS + 4):
        _drain_as(conn, _fb(graph), inc)
    posts = [e for e in graph.log if e.startswith("post:")]
    assert len(posts) <= FIRST_COMMENT_MAX_ATTEMPTS
    row = _meta_row(conn, did)
    assert row["first_comment_attempts"] == FIRST_COMMENT_MAX_ATTEMPTS
    assert [i["kind"] for i in _open_incidents(inc)] == [REPOST_KIND]


def test_a_post_time_gone_on_a_row_with_a_pending_correction_is_capped(conn, inc):
    did = _meta_relinked(conn, "facebook")
    item = json.dumps([{"live_id": "zz", "live_text": "x", "action": "delete", "extra": True}])
    with conn:
        conn.execute("UPDATE deliveries SET first_comment_correction=? WHERE id=?", (item, did))
    graph = FakeGraph("facebook", [])
    _gone_on_read(graph)
    for _ in range(FIRST_COMMENT_MAX_ATTEMPTS + 3):
        _drain_as(conn, _fb(graph), inc)
    posts = [e for e in graph.log if e.startswith("post:")]
    assert len(posts) <= FIRST_COMMENT_MAX_ATTEMPTS, posts
    assert _meta_row(conn, did)["first_comment_attempts"] == FIRST_COMMENT_MAX_ATTEMPTS
    assert [i["kind"] for i in _open_incidents(inc)] == [REPOST_KIND]


def test_a_gone_route_that_cannot_move_the_row_is_said_and_the_incident_says_why(conn, inc, caplog):
    did = _meta_relinked(conn, "facebook")
    item = json.dumps([{"live_id": "zz", "live_text": "x", "action": "delete"}])
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_posted_at=?, first_comment_live_id='c1', "
            "first_comment_correction=? WHERE id=?",
            (_NOW, item, did),
        )
    raise_repost_incident(conn, did, "someone-else", inc, kind=OWN_ID_KIND)
    graph = FakeGraph("facebook", [])  # c1 is gone and the listing confirms it
    res = FirstCommentResult()
    with caplog.at_level(logging.WARNING):
        recheck_own_id(conn, ("facebook",), lambda p: _fb(graph), inc, res)
    (incident,) = _open_incidents(inc)
    assert incident["kind"] == OWN_ID_KIND and not incident.get("resolved_at")
    assert "correction" in incident["detail"] and "c1" in incident["detail"]
    assert any(f"delivery {did}" in e and "c1" in e and "correction" in e for e in res.errors)
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("c1" in r.getMessage() for r in warned)
    assert _meta_row(conn, did)["first_comment_posted_at"]  # nothing was claimed or moved


def test_reopening_a_gone_comment_queues_our_old_id_for_a_checked_delete(conn):
    did = _meta_relinked(conn, "facebook")
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_text=?, first_comment_posted_at=?, "
            "first_comment_live_id='c1' WHERE id=?",
            (NEW, _NOW, did),
        )
    assert reopen_gone_first_comment(conn, did, "c1", _NOW, "gone")
    (item,) = first_comment_corrections(_meta_row(conn, did))
    assert item["live_id"] == "c1" and item["action"] == "delete"
    assert item["extra"] is True and item["ours"] is True and item["live_text"] == NEW
    assert _meta_row(conn, did)["first_comment_posted_at"] is None


def test_an_earlier_own_id_that_reappears_is_adopted_not_posted_again(conn, inc):
    did = _meta_relinked(conn, "facebook")
    with conn:
        conn.execute("UPDATE deliveries SET first_comment_attempts=1 WHERE id=?", (did,))
    graph = FakeGraph("facebook", [("c1", NEW)])  # the comment we thought gone is listed again
    _drain_as(conn, _fb(graph), inc)
    assert list(graph.comments) == ["c1"]
    assert not [e for e in graph.log if e.startswith("post:")]
    assert _meta_row(conn, did)["first_comment_live_id"] == "c1"


# --- MAJOR-3 / MAJOR-4: adopt by video id -------------------------------------------------------


def _sidecar(tmp_path, did, conn, video_id):
    row = _row(conn, did)
    side = tmp_path / "side.json"
    side.write_text(
        json.dumps(
            {
                "shorts": [
                    {
                        "short_index": row["short_index"],
                        "project_name": row["project_name"],
                        "parent_lf_video_id": video_id,
                    }
                ]
            }
        )
    )  # no parent_lf_publish_at: private at the slot, so no backfill
    with conn:
        conn.execute("UPDATE deliveries SET sidecar_path=? WHERE id=?", (str(side), did))


def _svcs(conn, inc, page, texts, drains=3):
    out = []
    for _ in range(drains):
        svc = _Service(pages=[page, page, page])
        svc.comments_api.texts.update(texts)
        out.append(svc)
        _drain(conn, svc, inc)
    return out


def test_identical_hand_re_adds_adopt_the_lowest_id_and_delete_the_extra(conn, inc):
    did = _removed(conn, inc, restore=False)
    page = {"items": [_thread(NEW_TEXT, "hand-2"), _thread(NEW_TEXT, "hand-1")]}
    svcs = _svcs(conn, inc, page, {"hand-1": NEW_TEXT, "hand-2": NEW_TEXT})
    row = _row(conn, did)
    assert row["first_comment_live_id"] == "hand-1" and row["first_comment_text"] == NEW_TEXT
    assert not _open(inc, "first_comment_removed")
    deleted = [d for s in svcs for d in s.comments_api.deletes]
    assert deleted == ["hand-2"], deleted  # our own extra copy, once; never the adopted one
    assert not first_comment_corrections(row)


def test_the_same_link_with_another_prefix_counts_as_the_same_video(conn, inc):
    did = _removed(conn, inc, restore=False)
    variant = "\U0001f3ac " + NEW_TEXT
    page = {"items": [_thread(NEW_TEXT, "b"), _thread(variant, "a")]}
    _svcs(conn, inc, page, {"a": variant, "b": NEW_TEXT}, drains=1)
    assert _row(conn, did)["first_comment_live_id"] == "a"


def test_two_different_videos_are_still_ambiguous(conn, inc):
    did = _removed(conn, inc, restore=False)
    other = "Watch the full drive: https://youtu.be/other"
    page = {"items": [_thread(NEW_TEXT, "hand-1"), _thread(other, "hand-2")]}
    (svc,) = _svcs(conn, inc, page, {}, drains=1)
    assert _row(conn, did)["first_comment_removed_at"]
    assert svc.comments_api.deletes == []


def test_a_wrong_link_is_not_adopted_when_the_sidecar_names_the_current_one(conn, inc, tmp_path):
    did = _removed(conn, inc, restore=False)
    _sidecar(tmp_path, did, conn, "right")
    typo = "Watch the full drive: https://youtu.be/typo"
    svc = _Service(pages=[{"items": [_thread(typo, "hand")]}])
    res = _drain(conn, svc, inc)
    row = _row(conn, did)
    assert row["first_comment_removed_at"] and row["first_comment_live_id"] != "hand"
    assert len(_open(inc, "first_comment_removed")) == 1
    assert any(f"delivery {did}" in e and "hand" in e and "right" in e for e in res.errors)
    assert svc.comments_api.deletes == []  # a wrong link is skipped, never deleted blind


def test_the_current_link_is_adopted_beside_a_wrong_one_when_the_sidecar_names_it(
    conn, inc, tmp_path
):
    did = _removed(conn, inc, restore=False)
    _sidecar(tmp_path, did, conn, "new")
    typo = "Watch the full drive: https://youtu.be/typo"
    page = {"items": [_thread(typo, "a-typo"), _thread(NEW_TEXT, "z-hand")]}
    _svcs(conn, inc, page, {}, drains=1)
    assert _row(conn, did)["first_comment_live_id"] == "z-hand"


# --- MINOR-5 / MINOR-6: the prior-text guard compares video ids; our replaced copy goes --------


def test_a_replaced_link_with_another_prefix_is_not_adopted(conn, inc):
    did = _removed(conn, inc, restore=False)
    variant = "\U0001f3ac " + OLD_TEXT
    res = _drain(conn, _Service(pages=[{"items": [_thread(variant, "old-hand")]}]), inc)
    row = _row(conn, did)
    assert row["first_comment_live_id"] != "old-hand" and row["first_comment_removed_at"]
    assert any("old-hand" in e for e in res.errors)


def test_the_adopt_update_compares_video_ids_not_strings(conn, inc):
    did = _removed(conn, inc, restore=False)
    variant = "\U0001f3ac " + OLD_TEXT
    assert not adopt_removed_first_comment(conn, did, "x1", variant, _NOW)
    assert _row(conn, did)["first_comment_removed_at"]


def test_our_copy_of_the_replaced_link_is_deleted_after_the_adopt(conn, inc):
    did = _removed(conn, inc, restore=False)
    page = {"items": [_thread(OLD_TEXT, "old-copy"), _thread(NEW_TEXT, "hand")]}
    svcs = _svcs(conn, inc, page, {"old-copy": OLD_TEXT, "hand": NEW_TEXT}, drains=4)
    assert _row(conn, did)["first_comment_live_id"] == "hand"
    deleted = [d for s in svcs for d in s.comments_api.deletes]
    assert deleted == ["old-copy"], deleted
    assert not _open(inc, "first_comment_stale")


def test_a_viewers_copy_of_the_replaced_link_is_never_queued_for_delete(conn, inc):
    did = _removed(conn, inc, restore=False)
    page = {"items": [_thread(OLD_TEXT, "viewer", author="UC-viewer"), _thread(NEW_TEXT, "hand")]}
    svcs = _svcs(conn, inc, page, {"hand": NEW_TEXT}, drains=3)
    assert _row(conn, did)["first_comment_live_id"] == "hand"
    assert [d for s in svcs for d in s.comments_api.deletes] == []


def test_an_adopt_gives_the_row_a_full_attempts_budget(conn, inc):
    did = _removed(conn, inc, restore=False)
    with conn:
        conn.execute("UPDATE deliveries SET first_comment_attempts=5 WHERE id=?", (did,))
    assert adopt_removed_first_comment(conn, did, "hand", NEW_TEXT, _NOW)
    assert _row(conn, did)["first_comment_attempts"] == 0


# --- MINOR-8 / MINOR-9: the incident text -------------------------------------------------------


def test_the_removed_incident_never_cuts_the_exact_re_add_text(conn, inc, tmp_path):
    from gomoveshift.publishing.queue import enqueue

    project = "2026-10-14-wyalusing-state-park-sunrise-loop"
    did = enqueue(conn, project, 12, "facebook", "h9", first_comment_text="x")
    side = tmp_path / "s.json"
    side.write_text(
        json.dumps(
            {
                "shorts": [
                    {
                        "short_index": 12,
                        "project_name": project,
                        "parent_lf_video_id": "dQw4w9WgXcQ",
                    }
                ]
            }
        )
    )
    with conn:
        conn.execute("UPDATE deliveries SET sidecar_path=? WHERE id=?", (str(side), did))
    cause = "comment 122100000000000000_987654321098765 reads gone (confirmed against the post's listing)"
    raise_repost_incident(conn, did, cause, inc, kind=REMOVED_KIND)
    (incident,) = _open_incidents(inc)
    want = first_comment.full_drive_comment_text("dQw4w9WgXcQ")
    assert repr(want) in incident["detail"], incident["detail"]
    assert "nothing else" in incident["detail"]


def test_the_own_id_incident_does_not_promise_a_repost_after_one_gone_read(conn, inc):
    did = _meta_relinked(conn, "facebook")
    raise_repost_incident(
        conn, did, "gone: the read-back answered that the comment is gone", inc, kind=OWN_ID_KIND
    )
    (incident,) = _open_incidents(inc)
    assert "re-read" in incident["detail"] and "second" in incident["detail"]


def test_the_author_mismatch_docstring_describes_the_gone_answer():
    doc = meta_comments._author_mismatch.__doc__
    assert "never touched" not in doc and "re-read" in doc


def test_the_video_id_of_a_full_drive_comment():
    assert first_comment.full_drive_video_id(NEW_TEXT) == "new"
    assert first_comment.full_drive_video_id("\U0001f3ac " + OLD_TEXT) == "old"
    assert first_comment.full_drive_video_id("nice video") is None
    assert OLD != NEW  # the Meta fixtures' own texts stay distinct
