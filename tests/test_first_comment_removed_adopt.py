"""A removed first comment (deleted, no text to repost) closes when OUR comment is live again,
whether the pipeline reposts it or the channel account re-adds it by hand; a viewer's comment is
never adopted. Also: a restored row gets a full post budget, a failed own-id read-back is
reported, and the own-id incident closes itself. Each test asserts the GOOD outcome. No network:
the repo's own fakes."""

from __future__ import annotations

import json

import pytest

from gomoveshift.publishing import first_comment, meta_comments
from gomoveshift.publishing.first_comment_stale import (
    OWN_ID_KIND,
    REMOVED_KIND,
    raise_repost_incident,
)
from gomoveshift.publishing.queue import (
    open_queue_db,
    removed_first_comment_rows,
    unpost_removed_first_comment,
)
from tests.test_first_comment_empty_gone_private import _open, _removed
from tests.test_first_comment_meta_correction import (
    NEW,
    FakeGraph,
    Resp,
    _adapter,
    _open_incidents,
)
from tests.test_first_comment_meta_correction import _drain as _meta_drain
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

SLOT = "2026-11-10T00:00:00+00:00"


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


# --- C3-F1: a hand re-add by the channel account closes the removed incident ------------------


def test_shape_check_accepts_only_the_full_drive_comment():
    assert first_comment.is_full_drive_comment(first_comment.full_drive_comment_text("abc123"))
    assert first_comment.is_full_drive_comment("Watch the full drive: https://youtu.be/new")
    assert not first_comment.is_full_drive_comment("great video")
    assert not first_comment.is_full_drive_comment("Watch the full drive: https://evil.example/x")
    assert not first_comment.is_full_drive_comment("")


def test_youtube_our_hand_re_add_is_adopted_and_closes_the_removed_incident(conn, inc):
    did = _removed(conn, inc, restore=False)
    assert len(_open(inc, "first_comment_removed")) == 1
    svc = _Service(pages=[{"items": [_thread(NEW_TEXT, "hand", author=OWN)]}])
    _drain(conn, svc, inc)
    assert svc.inserted == []  # no second copy beside the hand one
    row = _row(conn, did)
    assert row["first_comment_live_id"] == "hand" and row["first_comment_text"] == NEW_TEXT
    assert row["first_comment_posted_at"] and row["first_comment_removed_at"] is None
    assert not _open(inc, "first_comment_removed")  # closed within the one drain
    later = [_Service() for _ in range(2)]
    for s in later:
        _drain(conn, s, inc)
    assert all(s.inserted == [] and s.lists == [] for s in later)


@pytest.mark.parametrize("author", ["UC-viewer", ""])
def test_youtube_a_viewers_or_authorless_copy_is_never_adopted(conn, inc, author):
    did = _removed(conn, inc, restore=False)
    thread = _thread(NEW_TEXT, "viewer", author=author)
    svc = _Service(pages=[{"items": [thread]}])
    _drain(conn, svc, inc)
    row = _row(conn, did)
    assert row["first_comment_live_id"] != "viewer"
    assert row["first_comment_removed_at"] and len(_open(inc, "first_comment_removed")) == 1


def test_youtube_an_own_comment_of_another_shape_is_never_adopted(conn, inc):
    did = _removed(conn, inc, restore=False)
    svc = _Service(pages=[{"items": [_thread("thanks for watching", "chat", author=OWN)]}])
    _drain(conn, svc, inc)
    assert _row(conn, did)["first_comment_live_id"] != "chat"
    assert len(_open(inc, "first_comment_removed")) == 1


def test_youtube_a_cut_short_listing_keeps_it_open_with_an_error_line(conn, inc):
    did = _removed(conn, inc, restore=False)
    pages = [{"items": [], "nextPageToken": f"t{i}"} for i in range(20)]
    res = _drain(conn, _Service(pages=pages), inc)
    row = _row(conn, did)
    assert row["first_comment_removed_at"] and len(_open(inc, "first_comment_removed")) == 1
    assert any(f"delivery {did}" in e for e in res.errors), res.errors


def test_youtube_a_failed_listing_keeps_it_open_with_an_error_line(conn, inc):
    did = _removed(conn, inc, restore=False)

    class _Broken(_Service):
        def list(self, **kw):
            raise ConnectionError("connection reset")

    res = _drain(conn, _Broken(), inc)
    assert _row(conn, did)["first_comment_removed_at"]
    assert len(_open(inc, "first_comment_removed")) == 1
    assert any(f"delivery {did}" in e and "connection reset" in e for e in res.errors)


def test_the_removed_incident_and_error_line_say_the_pipeline_watches_for_our_comment(conn, inc):
    did = _removed(conn, inc, restore=False)
    (removed,) = _open(inc, "first_comment_removed")
    assert "closes" in removed["detail"] and "live again" in removed["detail"]
    assert "hand re-add" in removed["detail"] and f"#{did}" in removed["detail"]
    assert "add it by hand only if" not in removed["detail"]


def _removed_meta(conn, inc, platform):
    did = _meta_relinked(conn, platform)
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_text='', first_comment_posted_at=?, "
            "first_comment_live_id='c-gone', first_comment_removed_at=? WHERE id=?",
            (_NOW, _NOW, did),
        )
    raise_repost_incident(conn, did, "no text", inc, kind=REMOVED_KIND)
    assert len(_open_incidents(inc)) == 1
    return did


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_meta_our_hand_re_add_is_adopted_and_closes_the_removed_incident(conn, inc, platform):
    did = _removed_meta(conn, inc, platform)
    graph = FakeGraph(platform, [("hand", NEW)])  # authored by the Page / IG user
    _meta_drain(conn, platform, graph, inc)
    assert not [e for e in graph.log if e.startswith("post")]
    row = _meta_row(conn, did)
    assert row["first_comment_live_id"] == "hand" and row["first_comment_text"] == NEW
    assert row["first_comment_removed_at"] is None and row["first_comment_posted_at"]
    assert not _open_incidents(inc)


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_meta_a_viewers_copy_is_never_adopted(conn, inc, platform):
    did = _removed_meta(conn, inc, platform)
    graph = FakeGraph(platform, [("v1", NEW, "someone-else"), ("v2", NEW, "")])
    _meta_drain(conn, platform, graph, inc)
    row = _meta_row(conn, did)
    assert row["first_comment_live_id"] == "c-gone" and row["first_comment_removed_at"]
    assert len(_open_incidents(inc)) == 1


def test_meta_a_cut_short_listing_keeps_it_open_with_an_error_line(conn, inc, monkeypatch):
    did = _removed_meta(conn, inc, "facebook")
    graph = FakeGraph("facebook", [(f"x{i}", "chat") for i in range(40)], page=2)
    monkeypatch.setattr(first_comment, "COMMENT_CHECK_MAX_PAGES", 3)
    res = _meta_drain(conn, "facebook", graph, inc)
    assert _meta_row(conn, did)["first_comment_removed_at"] and len(_open_incidents(inc)) == 1
    assert any(f"delivery {did}" in e for e in res.errors), res.errors


# --- C3-F4: a restored row gets a full post budget ---------------------------------------------


def test_a_restored_row_with_spent_attempts_posts(conn, inc):
    did = _removed(conn, inc, restore=False)
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_text=?, first_comment_attempts=8 WHERE id=?",
            (NEW_TEXT, did),
        )
    svc = _Service()
    _drain(conn, svc, inc)
    assert svc.inserted == [NEW_TEXT]
    row = _row(conn, did)
    assert row["first_comment_posted_at"] and row["first_comment_attempts"] == 0
    _drain(conn, _Service(), inc)
    assert not _open(inc, "first_comment_removed") and not _open(inc, "first_comment_repost_failed")


# --- C3-F2 / F5: the test gaps -----------------------------------------------------------------


def test_a_restored_row_whose_repost_insert_fails_keeps_the_repost_incident_open(conn, inc):
    did = _removed(conn, inc, restore=True)  # the row has its text again

    class _FailInsert(_Service):
        def insert(self, part, body):
            raise RuntimeError("insert refused")

    for _ in range(2):
        _drain(conn, _FailInsert(), inc)
    assert _row(conn, did)["first_comment_posted_at"] is None  # zero live
    assert len(_open(inc, "first_comment_repost_failed")) == 1  # something stays open


def test_a_removed_row_with_a_pending_correction_is_not_unposted(conn, inc):
    did = _removed(conn, inc, restore=False)
    item = [{"live_id": "x1", "live_text": "t", "action": "delete"}]
    with conn:
        conn.execute(
            "UPDATE deliveries SET first_comment_correction=? WHERE id=?", (json.dumps(item), did)
        )
    assert unpost_removed_first_comment(conn, did, NEW_TEXT, _NOW) is False
    row = _row(conn, did)
    assert row["first_comment_posted_at"] and row["first_comment_removed_at"]
    assert not row["first_comment_text"]


def test_an_unposted_row_is_not_listed_as_removed(conn, inc):
    did = _removed(conn, inc, restore=False)
    assert [r["id"] for r in removed_first_comment_rows(conn, ("youtube",))] == [did]
    with conn:
        conn.execute("UPDATE deliveries SET first_comment_posted_at=NULL WHERE id=?", (did,))
    assert removed_first_comment_rows(conn, ("youtube",)) == []


# --- C4-F1: a read-back failure right after a post is reported --------------------------------


def _fb(graph, page_id):
    return _adapter("facebook", graph).__class__(
        {"page_id": page_id, "page_access_token": "t"}, session=graph
    )


def _drain_as(conn, adapter, inc):
    return meta_comments.drain_meta_first_comments(
        conn, adapter_factories={"facebook": lambda: adapter}, now=_NOW, incidents_dir=inc
    )


@pytest.mark.parametrize("how", ["raises", "100/33"])
def test_a_wrong_own_id_whose_read_back_fails_is_reported_and_nothing_deleted(conn, inc, how):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    if how == "raises":
        graph.read_raises = True
    else:
        real = graph.get

        def get(url, params=None, **kw):
            if not url.endswith("/comments"):
                return Resp({"error": {"code": 100, "error_subcode": 33, "message": "no"}}, 400)
            return real(url, params=params, **kw)

        graph.get = get
    res = _drain_as(conn, _fb(graph, "not-p"), inc)
    assert _meta_row(conn, did)["first_comment_posted_at"]
    assert not [e for e in graph.log if e.startswith("delete")] and len(graph.comments) == 1
    assert any(f"delivery {did}" in e and "new1" in e for e in res.errors), res.errors
    assert [i for i in _open_incidents(inc) if i["kind"] == OWN_ID_KIND]


def test_a_read_back_that_answers_gone_is_reported(conn, inc):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    real = graph.get

    def get(url, params=None, **kw):
        if url.endswith("/new1"):  # the single read says gone; the listing confirms it
            graph.comments.pop("new1", None)
        return real(url, params=params, **kw)

    graph.get = get
    res = _drain_as(conn, _fb(graph, "p"), inc)
    assert _meta_row(conn, did)["first_comment_posted_at"]
    assert any(f"delivery {did}" in e for e in res.errors), res.errors


# --- C4-F2: the own-id incident closes itself --------------------------------------------------


def test_the_own_id_incident_closes_once_the_id_is_right(conn, inc):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    _drain_as(conn, _fb(graph, "not-p"), inc)
    assert [i for i in _open_incidents(inc) if i["kind"] == OWN_ID_KIND]
    _drain_as(conn, _fb(graph, "not-p"), inc)  # still wrong: stays open
    assert [i for i in _open_incidents(inc) if i["kind"] == OWN_ID_KIND]
    _drain_as(conn, _fb(graph, "p"), inc)
    assert not [i for i in _open_incidents(inc) if i["kind"] == OWN_ID_KIND]
    assert graph.comments == {"new1": NEW} and _meta_row(conn, did)["first_comment_live_id"]


# --- C4-F3: every single-comment read asks Graph for the author --------------------------------


def test_a_correct_own_id_reads_back_clean_only_when_the_read_asks_for_from(conn, inc):
    did = _meta_relinked(conn, "facebook")
    graph = FakeGraph("facebook", [])
    res = _drain_as(conn, _fb(graph, "p"), inc)
    assert _meta_row(conn, did)["first_comment_live_id"] == "new1"
    assert res.errors == [] and not _open_incidents(inc)
