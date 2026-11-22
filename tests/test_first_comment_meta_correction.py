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
        self.fail_post = None
        self.fail_edit = None

    def get(self, url, params=None, **kw):
        assert url.endswith("/comments"), url
        after = int((params or {}).get("after") or 0)
        items = list(self.comments.items())
        chunk = items[after : after + self.page]
        body = {
            "data": [{"id": i, self.field: t, "from": {"id": self.authors[i]}} for i, t in chunk]
        }
        if after + self.page < len(items):
            body["paging"] = {"cursors": {"after": str(after + self.page)}, "next": "n"}
        self.log.append("list")
        return Resp(body)

    def post(self, url, data=None, **kw):
        if url.endswith("/comments"):
            if self.fail_post:
                self.log.append("post-failed")
                return Resp({"error": {"code": 2, "message": self.fail_post}}, 500)
            self.n += 1
            cid = f"new{self.n}"
            self.comments[cid] = data["message"]
            self.authors[cid] = self.own
            self.log.append(f"post:{cid}")
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
        self.comments.pop(cid)
        self.log.append(f"delete:{cid}")
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
    assert failed.stale == 1 and graph.comments == {}  # never silent: reported, incident open
    assert _open_incidents(inc) and first_comment_corrections(_row(conn, did))
    graph.fail_post = None
    ok = _drain(conn, "instagram", graph, inc)
    assert ok.corrected == 1 and list(graph.comments.values()) == [NEW]
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
