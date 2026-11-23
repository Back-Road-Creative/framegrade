"""A delete the gone reopen queued outlives a lagging read, so no copy of ours is ever left live
twice; and the wording, cap and tie-break edges around it. Each test asserts the GOOD outcome. No
network: the repo's own fakes."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from gomoveshift.publishing import first_comment_stale, queue
from gomoveshift.publishing.first_comment_stale import (
    LAG_DROP_KIND,
    OWN_ID_KIND,
    REMOVED_KIND,
    REPOST_KIND,
    raise_repost_incident,
    recheck_own_id,
)
from gomoveshift.publishing.queue import (
    FIRST_COMMENT_MAX_ATTEMPTS,
    REOPEN_LAG_WINDOW,
    first_comment_corrections,
    open_queue_db,
    queue_own_extra_deletes,
    reopen_gone_first_comment,
)
from gomoveshift.publishing.youtube_comments import FirstCommentResult
from tests.test_first_comment_empty_gone_private import _open, _removed
from tests.test_first_comment_gone_capped import _drain_as, _fb, _sidecar, _svcs
from tests.test_first_comment_meta_correction import NEW, FakeGraph, Resp
from tests.test_first_comment_meta_correction import _open_incidents
from tests.test_first_comment_meta_correction import _relinked as _meta_relinked
from tests.test_first_comment_meta_correction import _row as _meta_row
from tests.test_first_comment_reopen_cap import FIXED  # noqa: F401
from tests.test_first_comment_reopen_cap import frozen_queue_clock  # noqa: F401 -- autouse
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


def _ago(hours, now=FIXED):
    return (now - timedelta(hours=hours)).isoformat()


def _lag(graph, schedule):
    """Read-after-write lag: post number n (1-based) answers 100/33 on a read and is missing
    from the listing for ``schedule(n)`` drain ends; it IS live the whole time."""
    real_get, real_post = graph.get, graph.post
    st = {"hidden": {}, "posts": 0}

    def post(url, data=None, **kw):
        r = real_post(url, data=data, **kw)
        if url.endswith("/comments") and isinstance(r.body, dict) and "id" in r.body:
            st["posts"] += 1
            k = schedule(st["posts"])
            if k > 0:
                st["hidden"][r.body["id"]] = k
        return r

    def get(url, params=None, **kw):
        cid = url.rsplit("/", 1)[1]
        if not url.endswith("/comments"):
            if cid in st["hidden"]:
                return Resp({"error": {"code": 100, "error_subcode": 33, "message": "no"}}, 400)
            return real_get(url, params=params, **kw)
        saved = {h: graph.comments.pop(h) for h in list(st["hidden"]) if h in graph.comments}
        try:
            return real_get(url, params=params, **kw)
        finally:
            graph.comments.update(saved)

    graph.get, graph.post = get, post
    return st


def _end(st):
    for cid in list(st["hidden"]):
        st["hidden"][cid] -= 1
        if st["hidden"][cid] <= 0:
            del st["hidden"][cid]


def _run_lag(conn, inc, schedule, drains=FIRST_COMMENT_MAX_ATTEMPTS + 6, after=4):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    st = _lag(graph, schedule)
    for _ in range(drains):
        _drain_as(conn, _fb(graph), inc)
        _end(st)
    st["hidden"].clear()  # the lag has cleared for good
    for _ in range(after):
        _drain_as(conn, _fb(graph), inc)
    ours = sorted(c for c, t in graph.comments.items() if t == NEW)
    return did, graph, ours


def _posted(conn, did, items, live="c1"):
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_text=?, first_comment_posted_at=?, "
            "first_comment_live_id=?, first_comment_correction=? WHERE id=?",
            (NEW, _NOW, live, json.dumps(items), did),
        )


def _extra(live_id, **more):
    return {
        "live_id": live_id, "live_text": NEW, "action": "delete", "extra": True, "ours": True,
        **more,
    }  # fmt: skip


# --- BLOCKER-1: a delete the reopen queued survives a lagging read -------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_a_lag_on_every_post_ends_with_exactly_one_live_copy_the_row_records(conn, inc, k):
    did, graph, ours = _run_lag(conn, inc, lambda n: k)
    row = _meta_row(conn, did)
    posts = [e for e in graph.log if e.startswith("post:")]
    state = (k, ours, posts, row["first_comment_live_id"], row["first_comment_attempts"],
             [i["kind"] for i in _open_incidents(inc)])  # fmt: skip
    assert len(ours) == 1, state
    assert row["first_comment_live_id"] == ours[0], state
    assert row["first_comment_posted_at"], state
    assert row["first_comment_attempts"] >= len(posts) - 1, state  # every repost counted
    assert not first_comment_corrections(row), state
    assert not _open_incidents(inc), state


def test_a_first_post_that_lags_three_drains_is_deleted_when_it_appears(conn, inc):
    did, graph, ours = _run_lag(conn, inc, lambda n: 3 if n == 1 else 0)
    state = (ours, graph.log, [i["kind"] for i in _open_incidents(inc)])
    assert len(ours) == 1, state
    assert _meta_row(conn, did)["first_comment_live_id"] == ours[0], state
    assert not _open_incidents(inc), state


def test_a_reopen_queued_delete_stays_pending_while_its_read_lags(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_extra("zz", reopened=True, reopened_at=_ago(1))], live="c1")
    graph = FakeGraph("facebook", [("c1", NEW)])  # zz reads gone: it may only be lagging
    res = _drain_as(conn, _fb(graph), inc)
    (item,) = first_comment_corrections(_meta_row(conn, did))
    assert item["live_id"] == "zz" and item["reopened"] is True  # kept, not dropped
    assert any("zz" in e and "lag" in e for e in res.errors), res.errors
    assert all(i["kind"] != LAG_DROP_KIND for i in _open_incidents(inc))


def test_a_reopen_queued_delete_is_deleted_when_the_lagging_copy_appears(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_extra("zz", reopened=True, reopened_at=_ago(1))], live="c1")
    graph = FakeGraph("facebook", [("c1", NEW)])
    _drain_as(conn, _fb(graph), inc)  # zz still gone: held
    graph.comments["zz"], graph.authors["zz"] = NEW, graph.own  # the lag clears
    _drain_as(conn, _fb(graph), inc)
    assert "zz" not in graph.comments and "c1" in graph.comments
    assert not first_comment_corrections(_meta_row(conn, did))


def test_a_reopen_queued_delete_gone_past_the_window_is_dropped_said_and_surfaced(conn, inc):
    did = _meta_relinked(conn, "facebook")
    old = _ago(REOPEN_LAG_WINDOW.total_seconds() / 3600 + 1)
    _posted(conn, did, [_extra("zz", reopened=True, reopened_at=old)], live="c1")
    graph = FakeGraph("facebook", [("c1", NEW)])
    res = _drain_as(conn, _fb(graph), inc)
    assert not first_comment_corrections(_meta_row(conn, did))
    assert any("zz" in e and "dropped" in e for e in res.errors), res.errors
    (incident,) = [i for i in _open_incidents(inc) if i["kind"] == LAG_DROP_KIND]
    assert "zz" in incident["detail"] and "by hand" in incident["detail"]


def test_an_unmarked_extra_delete_still_drops_quietly_on_a_gone_read(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_extra("zz")], live="c1")
    graph = FakeGraph("facebook", [("c1", NEW)])
    res = _drain_as(conn, _fb(graph), inc)
    assert not first_comment_corrections(_meta_row(conn, did))
    assert not [e for e in res.errors if "zz" in e], res.errors
    assert not [i for i in _open_incidents(inc) if i["kind"] == LAG_DROP_KIND]


def test_reopening_marks_the_delete_it_queues_with_its_queue_time(conn):
    did = _meta_relinked(conn, "facebook")
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_posted_at=?, first_comment_live_id='c1' "
            "WHERE id=?",
            (_NOW, did),
        )
    assert reopen_gone_first_comment(conn, did, "c1", "2026-01-02T03:04:05+00:00", "gone")
    (item,) = first_comment_corrections(_meta_row(conn, did))
    assert item["reopened"] is True and item["reopened_at"] == "2026-01-02T03:04:05+00:00"


def test_a_held_delete_never_blocks_the_next_repost_decision(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_extra("zz", reopened=True, reopened_at=_ago(1))], live="c1")
    raise_repost_incident(conn, did, "gone: lag", inc, kind=OWN_ID_KIND)
    graph = FakeGraph("facebook", [])  # c1 reads gone, the listing agrees
    res = FirstCommentResult()
    recheck_own_id(conn, ("facebook",), lambda p: _fb(graph), inc, res)
    row = _meta_row(conn, did)
    assert row["first_comment_posted_at"] is None  # the route moved the row
    assert row["first_comment_attempts"] == 1
    assert [i["live_id"] for i in first_comment_corrections(row)] == ["zz", "c1"]
    assert [i["kind"] for i in _open_incidents(inc) if i["kind"] == REPOST_KIND] == [REPOST_KIND]


def test_a_pending_edit_or_a_viewers_item_still_blocks_the_gone_route(conn, inc):
    did = _meta_relinked(conn, "facebook")
    edit = {"live_id": "zz", "live_text": "x", "action": "edit", "to": NEW}
    _posted(conn, did, [edit], live="c1")
    assert not reopen_gone_first_comment(conn, did, "c1", _NOW, "gone")
    assert _meta_row(conn, did)["first_comment_posted_at"]
    _posted(conn, did, [{"live_id": "zz", "live_text": "x", "action": "delete", "extra": True}])
    assert not reopen_gone_first_comment(conn, did, "c1", _NOW, "gone")


def test_a_capped_correction_does_not_hide_the_repost_incident(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [_extra("zz", reopened=True, reopened_at=_ago(1))], live="c1")
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_correction_attempts=? WHERE id=?",
            (FIRST_COMMENT_MAX_ATTEMPTS, did),
        )
    raise_repost_incident(conn, did, "gone: lag", inc, kind=OWN_ID_KIND)
    recheck_own_id(conn, ("facebook",), lambda p: _fb(FakeGraph("facebook", [])), inc)
    assert [i["kind"] for i in _open_incidents(inc) if i["kind"] == REPOST_KIND] == [REPOST_KIND]


# --- MINOR-2: the incident wording ----------------------------------------------------------------


def test_the_repost_incident_says_the_comment_was_re_read_and_confirmed_gone(conn, inc):
    did = _meta_relinked(conn, "facebook")
    raise_repost_incident(conn, did, "gone: comment c1 reads gone", inc)
    (incident,) = _open_incidents(inc)
    detail = incident["detail"]
    assert "re-read on a later drain and confirmed gone" in detail
    assert "just posted" not in detail and "not recorded as posted" not in detail


def test_the_own_id_incident_says_a_row_with_no_text_is_marked_removed(conn, inc):
    did = _meta_relinked(conn, "facebook")
    raise_repost_incident(conn, did, "gone: lag", inc, kind=OWN_ID_KIND)
    (incident,) = _open_incidents(inc)
    detail = incident["detail"]
    assert "no text" in detail and "marked removed" in detail
    assert "(then it stays unposted with a repost incident open)" in detail  # a row with text


# --- MINOR-3: the unroutable note never promises a retry for a capped correction -------------------


def _unroutable_detail(conn, inc, did):
    raise_repost_incident(conn, did, "someone-else", inc, kind=OWN_ID_KIND)
    res = FirstCommentResult()
    recheck_own_id(conn, ("facebook",), lambda p: _fb(FakeGraph("facebook", [])), inc, res)
    (incident,) = _open_incidents(inc)
    return incident["detail"], res.errors


def test_an_unroutable_note_for_a_pending_correction_says_the_next_drain_tries_again(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [{"live_id": "zz", "live_text": "x", "action": "delete"}])
    detail, errors = _unroutable_detail(conn, inc, did)
    assert "tries again" in detail and any("tries again" in e for e in errors)


def test_an_unroutable_note_for_a_capped_correction_names_the_cap_and_promises_no_retry(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [{"live_id": "zz", "live_text": "x", "action": "delete"}])
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_correction_attempts=? WHERE id=?",
            (FIRST_COMMENT_MAX_ATTEMPTS, did),
        )
    detail, errors = _unroutable_detail(conn, inc, did)
    for text in (detail, *errors):
        if "c1" in text or "comment" in text:
            assert "tries again" not in text and "tries it again" not in text, text
    assert f"{FIRST_COMMENT_MAX_ATTEMPTS} times" in detail and "capped" in detail


# --- MINOR-4: the kept copy is the lowest id by number --------------------------------------------


def test_identical_hand_re_adds_keep_the_numerically_lowest_id(conn, inc):
    did = _removed(conn, inc, restore=False)
    page = {"items": [_thread(NEW_TEXT, "123_100"), _thread(NEW_TEXT, "123_99")]}
    svcs = _svcs(conn, inc, page, {"123_99": NEW_TEXT, "123_100": NEW_TEXT})
    assert _row(conn, did)["first_comment_live_id"] == "123_99"
    assert [d for s in svcs for d in s.comments_api.deletes] == ["123_100"]


def test_ids_that_are_not_numbers_keep_the_lowest_by_text(conn, inc):
    did = _removed(conn, inc, restore=False)
    page = {"items": [_thread(NEW_TEXT, "UgzB"), _thread(NEW_TEXT, "UgzA")]}
    _svcs(conn, inc, page, {"UgzA": NEW_TEXT, "UgzB": NEW_TEXT}, drains=1)
    assert _row(conn, did)["first_comment_live_id"] == "UgzA"


# --- MINOR-5: gaps the mutants found ---------------------------------------------------------------


def test_queueing_extras_never_overwrites_a_correction_written_meanwhile(conn, monkeypatch):
    did = _meta_relinked(conn, "facebook")
    _posted(conn, did, [], live="c1")
    with conn:
        conn.execute("UPDATE deliveries SET first_comment_correction=NULL WHERE id=?", (did,))
    other = json.dumps([_extra("rival")])
    real = queue.first_comment_corrections
    fired = []

    def racing(row):
        got = real(row)
        if not fired:
            fired.append(1)
            conn.execute(
                "UPDATE deliveries SET first_comment_correction=? WHERE id=?", (other, did)
            )
        return got

    monkeypatch.setattr(queue, "first_comment_corrections", racing)
    assert queue_own_extra_deletes(conn, did, [("c2", NEW)]) == 0
    monkeypatch.setattr(queue, "first_comment_corrections", real)
    assert [i["live_id"] for i in first_comment_corrections(_meta_row(conn, did))] == ["rival"]


def test_the_recorded_live_id_is_never_queued_as_an_extra_delete(conn):
    did = _meta_relinked(conn, "facebook")
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_posted_at=?, first_comment_live_id='c1' "
            "WHERE id=?",
            (_NOW, did),
        )
    assert queue_own_extra_deletes(conn, did, [("c1", NEW)]) == 0
    assert not first_comment_corrections(_meta_row(conn, did))


def test_an_adopt_with_extras_says_so_and_opens_the_stale_incident(conn, inc, monkeypatch):
    did = _removed(conn, inc, restore=False)
    opened = []
    real = first_comment_stale.raise_stale_incident
    monkeypatch.setattr(
        first_comment_stale,
        "raise_stale_incident",
        lambda c, d, directory, **kw: (opened.append(d), real(c, d, directory, **kw))[1],
    )
    page = {"items": [_thread(NEW_TEXT, "hand-2"), _thread(NEW_TEXT, "hand-1")]}
    svc = _Service(pages=[page])
    svc.comments_api.texts.update({"hand-1": NEW_TEXT, "hand-2": NEW_TEXT})
    res = _drain(conn, svc, inc)
    assert _row(conn, did)["first_comment_live_id"] == "hand-1"
    assert any("our own extra copies hand-2 are queued for deletion" in e for e in res.errors), (
        res.errors
    )
    assert opened == [did]  # the incident opened when the extras were queued (the delete then ran)
    assert svc.comments_api.deletes == ["hand-2"]


def test_a_replaced_link_with_a_prefix_beside_the_current_hand_re_add_adopts_the_hand(conn, inc):
    did = _removed(conn, inc, restore=False)
    variant = "\U0001f3ac " + OLD_TEXT
    page = {"items": [_thread(variant, "old-var"), _thread(NEW_TEXT, "hand")]}
    _svcs(conn, inc, page, {"old-var": variant, "hand": NEW_TEXT})
    assert _row(conn, did)["first_comment_live_id"] == "hand"
    assert not _open(inc, REMOVED_KIND)


def test_a_viewers_identical_copy_is_never_deleted_and_the_sidecar_keeps_the_lowest(
    conn, inc, tmp_path
):
    did = _removed(conn, inc, restore=False)
    _sidecar(tmp_path, did, conn, "new")
    page = {
        "items": [
            _thread(NEW_TEXT, "a-viewer", author="UC-viewer"),
            _thread(NEW_TEXT, "h2"),
            _thread(NEW_TEXT, "h1"),
        ]
    }
    svcs = _svcs(conn, inc, page, {"a-viewer": NEW_TEXT, "h1": NEW_TEXT, "h2": NEW_TEXT})
    assert _row(conn, did)["first_comment_live_id"] == "h1"
    assert [d for s in svcs for d in s.comments_api.deletes] == ["h2"]


def test_the_stale_module_exports_the_lag_drop_kind():
    assert first_comment_stale.LAG_DROP_KIND == LAG_DROP_KIND == "first_comment_lag_delete_dropped"


# --- a row at the repost cap still settles on one live copy ---------------------------------------


def _capped(conn, did, items):
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_text=?, first_comment_posted_at=NULL, "
            "first_comment_live_id=NULL, first_comment_attempts=?, first_comment_correction=? "
            "WHERE id=?",
            (NEW, FIRST_COMMENT_MAX_ATTEMPTS, json.dumps(items), did),
        )


def test_a_capped_row_adopts_the_lowest_listed_copy_and_queues_the_rest(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_extra("a", reopened=True, reopened_at=_ago(1))])
    graph = FakeGraph("facebook", [("a", NEW), ("b", NEW), ("c", NEW)])
    res = FirstCommentResult()
    first_comment_stale.settle_capped_reposts(conn, ("facebook",), lambda p: _fb(graph), res, inc)
    row = _meta_row(conn, did)
    assert row["first_comment_live_id"] == "a" and row["first_comment_posted_at"]
    assert row["first_comment_attempts"] == FIRST_COMMENT_MAX_ATTEMPTS  # the tries stay counted
    assert sorted(i["live_id"] for i in first_comment_corrections(row)) == ["a", "b", "c"]
    assert any("b, c" in e and "queued for deletion" in e for e in res.errors), res.errors
    assert _open_incidents(inc)


def test_a_capped_row_with_no_copy_listed_yet_waits_inside_the_window(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_extra("a", reopened=True, reopened_at=_ago(1))])
    res = FirstCommentResult()
    first_comment_stale.settle_capped_reposts(
        conn, ("facebook",), lambda p: _fb(FakeGraph("facebook", [])), res, inc
    )
    row = _meta_row(conn, did)
    assert row["first_comment_posted_at"] is None and len(first_comment_corrections(row)) == 1
    assert not _open_incidents(inc)


def test_a_capped_row_with_no_copy_past_the_window_drops_its_deletes_said_and_surfaced(conn, inc):
    did = _meta_relinked(conn, "facebook")
    old = _ago(REOPEN_LAG_WINDOW.total_seconds() / 3600 + 1)
    _capped(conn, did, [_extra("a", reopened=True, reopened_at=old)])
    res = FirstCommentResult()
    first_comment_stale.settle_capped_reposts(
        conn, ("facebook",), lambda p: _fb(FakeGraph("facebook", [])), res, inc
    )
    assert not first_comment_corrections(_meta_row(conn, did))
    assert any("a" in e and "dropped" in e for e in res.errors), res.errors
    assert [i["kind"] for i in _open_incidents(inc)] == [LAG_DROP_KIND]
    again = FirstCommentResult()  # nothing left to list for: the post is not called again
    graph = FakeGraph("facebook", [])
    first_comment_stale.settle_capped_reposts(conn, ("facebook",), lambda p: _fb(graph), again, inc)
    assert graph.log == []


def test_a_capped_row_whose_listing_fails_says_so_and_tries_again(conn, inc):
    did = _meta_relinked(conn, "facebook")
    _capped(conn, did, [_extra("a", reopened=True, reopened_at=_ago(1))])
    graph = FakeGraph("facebook", [("a", NEW)])
    graph.read_raises = True

    def boom(*a, **k):
        raise ConnectionError("reset")

    graph.get = boom
    res = FirstCommentResult()
    first_comment_stale.settle_capped_reposts(conn, ("facebook",), lambda p: _fb(graph), res, inc)
    assert any(f"delivery {did}" in e and "tries again" in e for e in res.errors), res.errors
    assert _meta_row(conn, did)["first_comment_posted_at"] is None
