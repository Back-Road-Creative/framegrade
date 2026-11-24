"""Adoption, a second stale duplicate and a mid-edit relink never lose or delete the intended
comment; a 404 edit never strands a delete; ``forbidden`` is a refusal only for corrections.
No network: fake YouTube services."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gomoveshift.publishing import youtube_comments as yc
from gomoveshift.publishing.queue import (
    first_comment_corrections,
    open_queue_db,
    rewrite_first_comment_corrections,
)
from gomoveshift.publishing.replace_relink import relink_replaced_lf
from tests.test_first_comment_correction_safety import (
    _PARENT_AT,
    _SLOT,
    _posted_stale,
    _relink_text,
)
from tests.test_first_comment_stale import (
    NEW_TEXT,
    OLD_TEXT,
    OWN,
    _NOW,
    _drain,
    _published,
    _Req,
    _rewrite,
    _row,
    _Service,
    _thread,
)
from tests.test_youtube_first_comment import (
    _FakeService,
    _make_http_error,
    _make_published_youtube_row,
)

T_TEXT = "Watch the full drive: https://youtu.be/older"


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


def _comment(vid):
    return f"Watch the full drive: https://youtu.be/{vid}"


def test_r1_a_second_stale_duplicate_gets_a_delete(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    _drain(conn, _Service(pages=[{"items": [_thread(OLD_TEXT, "X"), _thread(OLD_TEXT, "Z")]}]), inc)
    assert _ids(_row(conn, did)) == [("X", "edit"), ("Z", "delete")]
    fix = _Service()
    _drain(conn, fix, inc)
    assert fix.comments_api.updates == [("X", NEW_TEXT)] and fix.comments_api.deletes == ["Z"]
    row = _row(conn, did)
    assert row["first_comment_error"] is None and not first_comment_corrections(row)


def test_r1b_a_second_drain_that_found_the_old_text_tracks_its_comment(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    r = yc.FirstCommentResult()
    yc.record_found_comments(conn, did, OLD_TEXT, "X", OLD_TEXT, (), r, inc)
    yc.record_found_comments(conn, did, OLD_TEXT, "Z", OLD_TEXT, (), r, inc)
    row = _row(conn, did)
    assert _ids(row) == [("X", "edit"), ("Z", "delete")]
    assert row["first_comment_error"] is not None


def test_r2_adopting_the_edited_comment_never_deletes_it(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    r = yc.FirstCommentResult()
    yc.record_found_comments(conn, did, OLD_TEXT, "X", OLD_TEXT, (), r, inc)
    lost = _Service(fail=[TimeoutError("read timed out")])  # the server applied the edit
    _drain(conn, lost, inc)
    assert lost.comments_api.updates == [("X", NEW_TEXT)]
    yc.record_found_comments(conn, did, NEW_TEXT, "X", NEW_TEXT, (), r, inc)
    row = _row(conn, did)
    assert row["first_comment_text"] == NEW_TEXT and not first_comment_corrections(row)
    nxt = _Service()
    _drain(conn, nxt, inc)
    assert nxt.comments_api.deletes == [] and nxt.comments_api.updates == []


def test_r2_adopting_one_comment_deletes_the_others_not_it(conn, inc):
    did = _published(conn)
    items = [
        {"live_id": "X", "live_text": OLD_TEXT, "action": "edit", "to": NEW_TEXT},
        {"live_id": "W", "live_text": OLD_TEXT, "action": "edit", "to": NEW_TEXT},
    ]
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_text=?, first_comment_posted_at=?, "
            "first_comment_error=?, first_comment_correction=? WHERE id=?",
            (OLD_TEXT, _NOW, "stale", json.dumps(items), did),
        )
    r = yc.FirstCommentResult()
    yc.record_found_comments(conn, did, NEW_TEXT, "X", NEW_TEXT, (), r, inc)
    assert _ids(_row(conn, did)) == [("W", "delete")]


def test_r3_a_relink_during_an_in_flight_edit_is_not_lost(conn, inc, tmp_path):
    a, b, c = "aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"
    did = _published(conn)
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_text=?, scheduled_at=? WHERE id=?",
            (_comment(a), _SLOT, did),
        )
    now = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
    relink_replaced_lf("proj", tmp_path, {a: (b, _PARENT_AT)}, conn=conn, now=now)
    _drain(conn, _Service(pages=[{"items": [_thread(_comment(a), "X")]}]), inc)
    assert first_comment_corrections(_row(conn, did))[0]["to"] == _comment(b)

    svc = _Service()
    orig = svc.comments_api.update

    def update(part, body):
        rep = relink_replaced_lf("proj", tmp_path, {b: (c, _PARENT_AT)}, conn=conn, now=now)
        assert [x.written for x in rep.queue_rewritten] == [True]
        return orig(part, body)

    svc.comments_api.update = update
    res = _drain(conn, svc, inc)
    assert svc.comments_api.updates == [("X", _comment(b))] and res.errors == []
    (item,) = first_comment_corrections(_row(conn, did))
    assert item["to"] == _comment(c) and _row(conn, did)["first_comment_error"] is not None
    svc.comments_api.update = orig
    _drain(conn, svc, inc)
    assert svc.comments_api.updates[-1] == ("X", _comment(c))
    row = _row(conn, did)
    assert row["first_comment_text"] == _comment(c) and not first_comment_corrections(row)
    assert row["first_comment_error"] is None


def test_the_rewrite_compare_and_set_refuses_when_the_json_moved(conn):
    did = _published(conn)
    item = {"live_id": "X", "live_text": OLD_TEXT, "action": "edit", "to": NEW_TEXT}
    moved = json.dumps([{**item, "to": "moved"}])
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_posted_at=?, first_comment_correction=? "
            "WHERE id=?",
            (_NOW, json.dumps([item]), did),
        )

    def rewrite(to):
        with conn:  # a concurrent writer lands between the read and the write
            conn.execute(
                "UPDATE deliveries SET first_comment_correction=? WHERE id=?", (moved, did)
            )
        return to + "!"

    assert rewrite_first_comment_corrections(conn, did, rewrite, _NOW) is False
    assert _row(conn, did)["first_comment_correction"] == moved


def test_an_edit_that_404s_after_a_relink_leaves_the_rewritten_item_pending(conn, inc):
    did = _published(conn)
    items = [{"live_id": "X", "live_text": OLD_TEXT, "action": "edit", "to": NEW_TEXT}]
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_posted_at=?, first_comment_text=?, "
            "first_comment_correction=? WHERE id=?",
            (_NOW, OLD_TEXT, json.dumps(items), did),
        )
    svc = _Service(fail=[_make_http_error(404, "commentNotFound")])
    orig = svc.comments_api.update

    def update(part, body):
        with conn:
            conn.execute(
                "UPDATE deliveries SET first_comment_correction=? WHERE id=?",
                (json.dumps([{**items[0], "to": "relinked"}]), did),
            )
        return orig(part, body)

    svc.comments_api.update = update
    _drain(conn, svc, inc)
    row = _row(conn, did)
    assert row["first_comment_posted_at"] is not None and _ids(row) == [("X", "edit")]


def test_r4_a_404_edit_still_runs_the_rows_pending_deletes(conn, inc):
    did = _published(conn)
    items = [
        {"live_id": "X", "live_text": OLD_TEXT, "action": "edit", "to": NEW_TEXT},
        {"live_id": "Z", "live_text": T_TEXT, "action": "delete"},
    ]
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_posted_at=?, first_comment_error=?, "
            "first_comment_correction=?, first_comment_prior_text=? WHERE id=?",
            (_NOW, "stale", json.dumps(items), json.dumps([T_TEXT, OLD_TEXT]), did),
        )
    svc = _Service(pages=[{"items": []}], fail=[_make_http_error(404, "commentNotFound")])
    res = _drain(conn, svc, inc)
    assert svc.comments_api.deletes == ["Z"] and svc.inserted == [NEW_TEXT]
    row = _row(conn, did)
    assert not first_comment_corrections(row) and row["first_comment_posted_at"] is not None
    assert res.errors == [] and res.failed == 0


def test_r5_forbidden_on_posting_is_a_reauth_again(conn):
    did = _make_published_youtube_row(conn, 1)
    service = _FakeService(exc=_make_http_error(403, reason="forbidden"))
    res = yc.drain_first_comments(conn, service_factory=lambda: service)
    row = conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone()
    assert res.needs_reauth and res.failed == 0 and row["first_comment_attempts"] == 0


def test_forbidden_counts_for_a_correction_but_not_for_posting():
    forbidden = _make_http_error(403, "forbidden")
    assert yc._is_scope_error(forbidden) and not yc._refused_for_the_video(forbidden)
    assert yc._refused_for_a_correction(forbidden)
    gap = _make_http_error(403, "insufficientPermissions")
    assert yc._is_scope_error(gap) and not yc._refused_for_a_correction(gap)


def test_a_channels_list_scope_gap_is_a_reauth_not_an_unknown_author(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    svc = _Service(pages=[{"items": [_thread(OLD_TEXT, "X", OWN)]}])
    svc.channels = lambda: SimpleNamespace(
        list=lambda **kw: _Req(exc=_make_http_error(403, "insufficientPermissions"))
    )
    res = _drain(conn, svc, inc)
    assert res.needs_reauth and res.failed == 0 and svc.inserted == []


def test_a_youtube_row_with_a_stale_error_and_no_correction_says_fix_it_by_hand(conn, tmp_path):
    did = _posted_stale(conn, "youtube", [])
    (note,), _ = _relink_text(conn, tmp_path, did)
    assert "fix it by hand" in note and "the drain corrects it" not in note
