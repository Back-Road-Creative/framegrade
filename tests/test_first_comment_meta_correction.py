"""A replaced first comment found live on Facebook / Instagram is corrected by the next drain.

Meta Graph API (grounded 2026-11-22):
- Comment node: ``POST /{comment-id}`` with ``message`` updates it, ``DELETE /{comment-id}``
  deletes it (https://developers.facebook.com/docs/graph-api/reference/comment/).
- IG Comment: delete only, no way to edit the text (only hide/unhide)
  (https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-comment).
No network: a fake Graph session holds the post's comments."""

from __future__ import annotations

import json

import pytest

from gomoveshift.publishing import meta_comments
from gomoveshift.publishing.facebook_adapter import FacebookReelsAdapter
from gomoveshift.publishing.first_comment import (
    CommentCheckError,
    PriorCommentLiveError,
    post_graph_comment,
)
from gomoveshift.publishing.instagram_adapter import InstagramReelsAdapter
from gomoveshift.publishing.queue import (
    FIRST_COMMENT_MAX_ATTEMPTS,
    PUBLISHED_AT_PLATFORM_CONFIRMED as PLATFORM,
    enqueue,
    first_comment_corrections,
    open_queue_db,
    rewrite_unposted_first_comment,
    transition,
)

OLD = "Watch the full drive: https://youtu.be/old"
NEW = "Watch the full drive: https://youtu.be/new"
_PUB_AT = "2026-08-23T00:00:00+00:00"
_NOW = "2026-08-23T00:06:00+00:00"
FIELD = {"facebook": "message", "instagram": "text"}
OWN = {"facebook": "p", "instagram": "u"}  # the Page / IG user id the test adapters hold


class Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def json(self):
        return self.body


class FakeGraph:
    """One post's comments, newest last; pages of ``page`` comments; every call logged."""

    def __init__(self, platform, comments=(), page=2):
        self.field = FIELD[platform]
        self.own = OWN[platform]
        # (id, text) is ours; (id, text, author_id) is someone else's.
        self.comments = {c[0]: c[1] for c in comments}
        self.authors = {c[0]: (c[2] if len(c) > 2 else self.own) for c in comments}
        self.page, self.log, self.n = page, [], 0
        self.read_raises = False  # a GET of one comment never answers
        self.fail_post = None
        self.fail_edit = None
        self.post_refusal = False  # a permanent refusal (Graph code 100, HTTP 400), not a 500
        self.delete_answer = None  # a 2xx body that is not {"success": true}; the comment stays
        self.delete_raises = False  # the DELETE never lands: a network failure
        self.delete_refusal = False  # Graph code 100 on the DELETE
        self.delete_outage = False  # HTTP 500 on the DELETE
        self.lose_delete_response = False  # the delete lands, its answer is lost once
        self.lose_post_response = False  # the post lands, its answer is lost once

    def get(self, url, params=None, **kw):
        if not url.endswith("/comments"):  # GET /{comment-id}: one comment, or 100/33 if absent
            cid = url.rsplit("/", 1)[1]
            self.log.append(f"read:{cid}")
            if self.read_raises:
                raise ConnectionError("connection reset")
            if cid not in self.comments:
                return Resp({"error": {"code": 100, "error_subcode": 33, "message": "no"}}, 400)
            full = {"id": cid, self.field: self.comments[cid], "from": {"id": self.authors[cid]}}
            asked = ((params or {}).get("fields") or "id,{},from".format(self.field)).split(",")
            return Resp({k: v for k, v in full.items() if k in asked})  # real Graph: fields only
        assert url.endswith("/comments"), url
        after = int((params or {}).get("after") or 0)
        items = list(self.comments.items())
        chunk = items[after : after + self.page]
        asked = ((params or {}).get("fields") or "id,{},from".format(self.field)).split(",")
        body = {
            "data": [
                {
                    k: v
                    for k, v in {
                        "id": i, self.field: t, "from": {"id": self.authors[i]},
                    }.items()
                    if k in asked
                }
                for i, t in chunk
            ]
        }  # fmt: skip
        if after + self.page < len(items):
            body["paging"] = {"cursors": {"after": str(after + self.page)}, "next": "n"}
        self.log.append("list")
        return Resp(body)

    def post(self, url, data=None, **kw):
        if url.endswith("/comments"):
            if self.post_refusal:
                self.log.append("post-refused")
                return Resp({"error": {"code": 100, "message": "refused"}}, 400)
            if self.fail_post:
                self.log.append("post-failed")
                return Resp({"error": {"code": 2, "message": self.fail_post}}, 500)
            self.n += 1
            cid = f"new{self.n}"
            self.comments[cid] = data["message"]
            self.authors[cid] = self.own
            self.log.append(f"post:{cid}")
            if self.lose_post_response:
                self.lose_post_response = False
                raise ConnectionError("read timed out")
            return Resp({"id": cid})
        cid = url.rsplit("/", 1)[1]
        if self.fail_edit:
            self.log.append("edit-refused")
            return Resp({"error": {"code": 100, "message": self.fail_edit}}, 400)
        self.comments[cid] = data["message"]
        self.log.append(f"edit:{cid}")
        return Resp({"success": True})

    def delete(self, url, **kw):
        cid = url.rsplit("/", 1)[1]
        if self.delete_raises:
            self.log.append(f"delete-raised:{cid}")
            raise ConnectionError("connection reset")
        if self.delete_outage:
            self.log.append(f"delete-outage:{cid}")
            return Resp({"error": {"code": 2, "message": "try later"}}, 500)
        if self.delete_refusal:
            self.log.append(f"delete-refused:{cid}")
            return Resp({"error": {"code": 100, "message": "refused"}}, 400)
        if self.delete_answer is not None:
            self.log.append(f"delete-unconfirmed:{cid}")
            return Resp(self.delete_answer)
        if cid not in self.comments:  # Graph: code 100, subcode 33 -- the object does not exist
            self.log.append(f"delete-missing:{cid}")
            return Resp({"error": {"code": 100, "error_subcode": 33, "message": "no such"}}, 400)
        self.comments.pop(cid)
        self.log.append(f"delete:{cid}")
        if self.lose_delete_response:
            self.lose_delete_response = False
            raise ConnectionError("read timed out")
        return Resp({"success": True})


def _adapter(platform, graph):
    if platform == "facebook":
        return FacebookReelsAdapter({"page_id": "p", "page_access_token": "t"}, session=graph)
    return InstagramReelsAdapter({"ig_user_id": "u", "page_access_token": "t"}, session=graph)


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


def _relinked(conn, platform):
    did = enqueue(conn, "proj", 1, platform, "h1", first_comment_text=OLD)
    assert transition(
        conn, did, "queued", "published",
        platform_upload_id="vid", platform_post_id="post", published_at_source=PLATFORM,
    )  # fmt: skip
    conn.execute("UPDATE deliveries SET published_at=? WHERE id=?", (_PUB_AT, did))
    conn.commit()
    assert rewrite_unposted_first_comment(conn, did, OLD, NEW, _NOW)
    return did


def _row(conn, did):
    return conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone()


def _drain(conn, platform, graph, inc):
    adapter = _adapter(platform, graph)
    return meta_comments.drain_meta_first_comments(
        conn, adapter_factories={platform: lambda: adapter}, now=_NOW, incidents_dir=inc
    )


def _open_incidents(inc):
    rows = [json.loads(p.read_text()) for p in sorted(inc.glob("*.json"))] if inc.is_dir() else []
    return [r for r in rows if not r.get("resolved_at")]


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_a_stale_live_comment_is_corrected_to_exactly_one_new_comment(conn, inc, platform):
    did = _relinked(conn, platform)
    graph = FakeGraph(platform, [("c1", OLD)])
    first = _drain(conn, platform, graph, inc)
    assert first.stale == 1 and _open_incidents(inc)
    assert graph.comments == {"c1": OLD}  # the drain that finds it only reports
    second = _drain(conn, platform, graph, inc)
    assert second.corrected == 1 and second.stale == 0
    assert list(graph.comments.values()) == [NEW]
    row = _row(conn, did)
    assert row["first_comment_error"] is None and row["first_comment_text"] == NEW
    assert not first_comment_corrections(row) and not _open_incidents(inc)
    _drain(conn, platform, graph, inc)
    assert list(graph.comments.values()) == [NEW]


def test_facebook_edits_the_comment_in_place(conn, inc):
    _relinked(conn, "facebook")
    graph = FakeGraph("facebook", [("c1", OLD)])
    _drain(conn, "facebook", graph, inc)
    _drain(conn, "facebook", graph, inc)
    assert "edit:c1" in graph.log and not any(e.startswith("post:") for e in graph.log)


def test_instagram_deletes_the_stale_comment_before_it_posts_the_new_one(conn, inc):
    _relinked(conn, "instagram")
    graph = FakeGraph("instagram", [("c1", OLD)])
    _drain(conn, "instagram", graph, inc)
    _drain(conn, "instagram", graph, inc)
    acts = [e for e in graph.log if e != "list"]
    assert acts == ["delete:c1", "post:new1"]


def test_instagram_post_failure_after_the_delete_retries_without_a_second_comment(conn, inc):
    did = _relinked(conn, "instagram")
    graph = FakeGraph("instagram", [("c1", OLD)])
    _drain(conn, "instagram", graph, inc)
    graph.fail_post = "boom"
    failed = _drain(conn, "instagram", graph, inc)
    # never silent: the failed post is reported by the posting path that now owns it
    assert graph.comments == {} and failed.failed == 1 and failed.errors
    (open_one,) = _open_incidents(inc)  # zero comments live: never silent
    assert open_one["delivery_id"] == did and "repost" in open_one["kind"]
    row = _row(conn, did)
    assert row["first_comment_posted_at"] is None and row["first_comment_text"] == NEW
    assert not first_comment_corrections(row) and row["first_comment_error"]
    graph.fail_post = None
    ok = _drain(conn, "instagram", graph, inc)
    assert ok.posted == 1 and list(graph.comments.values()) == [NEW]
    _drain(conn, "instagram", graph, inc)  # the next pass closes the repost incident
    assert not _open_incidents(inc)


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_both_live_new_first_deletes_the_stale_one_and_never_the_new(conn, inc, platform):
    did = _relinked(conn, platform)
    graph = FakeGraph(
        platform, [("n1", NEW), ("x1", "other"), ("x2", "other"), ("x3", "other"), ("o1", OLD)]
    )
    first = _drain(conn, platform, graph, inc)
    assert first.stale == 1 and first.errors
    (item,) = first_comment_corrections(_row(conn, did))
    assert item["action"] == "delete" and item["live_id"] == "o1"
    _drain(conn, platform, graph, inc)
    assert "n1" in graph.comments and "o1" not in graph.comments
    assert not [e for e in graph.log if e.startswith(("post:", "edit:"))]
    assert not _open_incidents(inc)


@pytest.mark.parametrize("platform", ["facebook", "instagram"])
def test_the_scan_reads_every_page_through_the_adapter(platform):
    graph = FakeGraph(platform, [("a", "x"), ("b", "y"), ("c", "z"), ("d", OLD)])
    with pytest.raises(PriorCommentLiveError) as info:
        _adapter(platform, graph).post_comment("obj", NEW, also_accept=(OLD,))
    assert info.value.comment_id == "d" and info.value.said == OLD
    assert not [e for e in graph.log if e.startswith("post:")]


def test_a_new_comment_beside_a_prior_one_raises_with_the_new_id_and_the_duplicate():
    graph = FakeGraph("facebook", [("n1", NEW), ("o1", OLD)], page=1)
    with pytest.raises(PriorCommentLiveError) as info:
        post_graph_comment(
            graph, "https://g", "obj", NEW, "t", text_field="message", permission="p",
            raise_for_error=lambda s, b: None, also_accept=(OLD,), own_id="p",
        )  # fmt: skip
    assert (info.value.comment_id, info.value.said) == ("n1", NEW)
    assert info.value.duplicates == (("o1", OLD),)


def test_facebook_falls_back_to_delete_and_repost_when_graph_refuses_the_edit(conn, inc):
    _relinked(conn, "facebook")
    graph = FakeGraph("facebook", [("c1", OLD)])
    graph.fail_edit = "cannot edit"
    _drain(conn, "facebook", graph, inc)
    _drain(conn, "facebook", graph, inc)
    assert list(graph.comments.values()) == [NEW]


def test_instagram_will_not_post_when_it_cannot_confirm_the_stale_comment_is_gone(conn, inc):
    from gomoveshift.publishing import first_comment

    graph = FakeGraph("instagram", [(f"c{i}", "x") for i in range(30)], page=1)
    with pytest.raises(CommentCheckError):
        first_comment.replace_graph_comment(
            graph, "https://g", "obj", "missing", NEW, "t", text_field="text", permission="p",
            raise_for_error=lambda s, b: None, own_id="u",
        )  # fmt: skip
    assert not [e for e in graph.log if e.startswith("post:")]


# --- a confirmed delete followed by a refused post never ends at zero comments, silently ------


def test_a_refused_post_after_a_confirmed_delete_returns_the_row_to_unposted_with_the_new_text(
    conn, inc
):
    did = _relinked(conn, "instagram")
    graph = FakeGraph("instagram", [("c1", OLD)])
    graph.post_refusal = True
    _drain(conn, "instagram", graph, inc)  # finds the stale comment, queues the replace
    second = _drain(conn, "instagram", graph, inc)  # delete confirmed, post refused (code 100)
    row = _row(conn, did)
    assert graph.comments == {} and "delete:c1" in graph.log
    # never marked posted with the OLD text; the normal posting path owns the NEW one
    assert row["first_comment_posted_at"] is None and row["first_comment_text"] == NEW
    assert not first_comment_corrections(row)
    # the failed repost counts toward the cap (the first pass's attempt plus this one)
    assert second.failed == 1 and second.errors and row["first_comment_attempts"] == 2
    assert row["first_comment_error"]  # visible on the row
    for _ in range(FIRST_COMMENT_MAX_ATTEMPTS + 2):
        _drain(conn, "instagram", graph, inc)
    row = _row(conn, did)
    assert row["first_comment_attempts"] == FIRST_COMMENT_MAX_ATTEMPTS  # ends at the posting cap
    assert row["first_comment_text"] == NEW and row["first_comment_error"]
    graph.post_refusal = False
    row = _row(conn, did)  # the cap is the posting path's: a human reset posts it once
    conn.execute("UPDATE deliveries SET first_comment_attempts=0 WHERE id=?", (did,))
    conn.commit()
    assert _drain(conn, "instagram", graph, inc).posted == 1
    assert list(graph.comments.values()) == [NEW]


def test_a_delete_whose_answer_was_lost_resolves_once_graph_says_it_is_gone(conn, inc):
    did = _relinked(conn, "facebook")
    graph = FakeGraph("facebook", [("n1", NEW), ("o1", OLD)])
    graph.lose_delete_response = True
    _drain(conn, "facebook", graph, inc)  # queues delete o1
    for _ in range(3):
        _drain(conn, "facebook", graph, inc)
    row = _row(conn, did)
    assert graph.comments == {"n1": NEW}
    assert "delete-missing:o1" in graph.log  # Graph: code 100, subcode 33 -> done
    assert not first_comment_corrections(row) and not _open_incidents(inc)
    assert row["first_comment_correction_attempts"] == 0


def test_a_lost_post_answer_after_the_delete_is_found_not_posted_twice(conn, inc):
    did = _relinked(conn, "instagram")
    graph = FakeGraph("instagram", [("c1", OLD)])
    graph.lose_post_response = True
    _drain(conn, "instagram", graph, inc)
    _drain(conn, "instagram", graph, inc)  # delete, post lands, answer lost
    _drain(conn, "instagram", graph, inc)
    assert list(graph.comments.values()) == [NEW]
    assert not first_comment_corrections(_row(conn, did))


# --- a delete that did not succeed never lets the post go ahead ----------------------------


@pytest.mark.parametrize("how", ["success-false", "no-success-key", "raises"], ids=lambda h: h)
def test_instagram_posts_nothing_when_the_delete_is_not_confirmed(conn, inc, how):
    did = _relinked(conn, "instagram")
    graph = FakeGraph("instagram", [("c1", OLD)])
    _drain(conn, "instagram", graph, inc)  # queues the replace
    if how == "raises":
        graph.delete_raises = True
    else:
        graph.delete_answer = {"success": False} if how == "success-false" else {}
    for _ in range(2):
        later = _drain(conn, "instagram", graph, inc)
    assert not [e for e in graph.log if e.startswith("post")]  # nothing posted beside it
    assert graph.comments == {"c1": OLD}
    assert later.stale == 1 and later.errors  # reported, never silent
    assert first_comment_corrections(_row(conn, did)) and _open_incidents(inc)


def test_a_permanent_refusal_of_the_correction_counts_toward_the_cap_and_an_outage_never_does(
    conn, inc
):
    refused = _relinked(conn, "facebook")
    graph = FakeGraph("facebook", [("n1", NEW), ("o1", OLD)])
    _drain(conn, "facebook", graph, inc)  # queues delete o1
    graph.delete_refusal = True
    for _ in range(FIRST_COMMENT_MAX_ATTEMPTS + 2):
        _drain(conn, "facebook", graph, inc)
    row = _row(conn, refused)
    assert row["first_comment_correction_attempts"] == FIRST_COMMENT_MAX_ATTEMPTS
    assert graph.log.count("delete-refused:o1") == FIRST_COMMENT_MAX_ATTEMPTS  # then it stops
    assert {i["kind"] for i in _open_incidents(inc)} == {"first_comment_correction_capped"}


def test_an_outage_on_the_correction_is_never_counted(conn, inc):
    did = _relinked(conn, "facebook")
    graph = FakeGraph("facebook", [("n1", NEW), ("o1", OLD)])
    _drain(conn, "facebook", graph, inc)
    graph.delete_outage = True
    for _ in range(FIRST_COMMENT_MAX_ATTEMPTS + 2):
        _drain(conn, "facebook", graph, inc)
    row = _row(conn, did)
    assert row["first_comment_correction_attempts"] == 0 and first_comment_corrections(row)
    assert graph.log.count("delete-outage:o1") == FIRST_COMMENT_MAX_ATTEMPTS + 2


def test_a_failing_correction_pass_is_recorded_and_meta_posting_continues(conn, inc, monkeypatch):
    did = enqueue(conn, "proj", 1, "facebook", "h1", first_comment_text=NEW)
    assert transition(
        conn, did, "queued", "published",
        platform_upload_id="vid", platform_post_id="post", published_at_source=PLATFORM,
    )  # fmt: skip
    conn.execute("UPDATE deliveries SET published_at=? WHERE id=?", (_PUB_AT, did))
    conn.commit()

    def boom(*a, **k):
        raise RuntimeError("corrections exploded")

    monkeypatch.setattr(meta_comments, "run_corrections", boom)
    graph = FakeGraph("facebook", [])
    result = _drain(conn, "facebook", graph, inc)
    assert any("corrections exploded" in e for e in result.errors)
    assert result.posted == 1 and list(graph.comments.values()) == [NEW]
