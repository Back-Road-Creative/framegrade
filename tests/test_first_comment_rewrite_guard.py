"""A first comment rewritten while a drain posts it never marks the wrong text posted."""

import json
from types import SimpleNamespace

import pytest

from gomoveshift.publishing import meta_comments, youtube_comments
from gomoveshift.publishing.first_comment import PriorCommentLiveError, post_graph_comment
from gomoveshift.publishing.queue import (
    PUBLISHED_AT_PLATFORM_CONFIRMED as PLATFORM,
    enqueue,
    open_queue_db,
    rewrite_unposted_first_comment,
    transition,
)
from gomoveshift.publishing.youtube_comments import drain_first_comments, post_first_comment

OLD_TEXT, NEW_TEXT = (
    "Watch the full drive: https://youtu.be/old",
    "Watch the full drive: https://youtu.be/new",
)
_PUB_AT = "2026-08-23T00:00:00+00:00"
_NOW = "2026-08-23T00:06:00+00:00"


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "q")
    try:
        yield c
    finally:
        c.close()


def _published(conn, platform="youtube"):
    did = enqueue(conn, "proj", 1, platform, "h1", first_comment_text=OLD_TEXT)
    assert transition(
        conn,
        did,
        "queued",
        "published",
        platform_upload_id="vid",
        platform_post_id="post",
        published_at_source=PLATFORM,
    )
    conn.execute("UPDATE deliveries SET published_at=? WHERE id=?", (_PUB_AT, did))
    conn.commit()
    return did


def _rewrite(conn, did):
    assert rewrite_unposted_first_comment(conn, did, OLD_TEXT, NEW_TEXT, _NOW)


def _row(conn, did):
    return conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone()


def _rewrite_after_select(monkeypatch, module, conn, did):
    """The drain selects the row, THEN a relink rewrites it, then the drain posts."""
    real = module.select_due_first_comments

    def stale(*a, **k):
        rows = real(*a, **k)
        _rewrite(conn, did)
        return rows

    monkeypatch.setattr(module, "select_due_first_comments", stale)


class _Service:
    """A fake YouTube client: ``list`` answers ``pages`` then empty; ``insert`` records."""

    def __init__(self, pages=(), on_insert=None):
        self.inserted, self.pages, self.on_insert = [], list(pages), on_insert

    def commentThreads(self):
        return self

    def list(self, **_):
        page = self.pages.pop(0) if self.pages else {"items": []}
        return SimpleNamespace(execute=lambda: page)

    def insert(self, part, body):
        self.inserted.append(body["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
        if self.on_insert:
            self.on_insert()
        return SimpleNamespace(execute=lambda: {"id": "new-thread"})


def _thread(text, tid="t-old"):
    return {
        "items": [{"id": tid, "snippet": {"topLevelComment": {"snippet": {"textOriginal": text}}}}]
    }


def test_a_rewrite_during_the_post_records_the_live_old_text_and_reports_it(conn):
    # Stricter than before: the old comment IS live, so the row records the text that is
    # actually live (never the new text) and the failure is visible three ways.
    did = _published(conn)
    service = _Service(on_insert=lambda: _rewrite(conn, did))
    result = drain_first_comments(conn, service_factory=lambda: service, now=_NOW)
    assert service.inserted == [OLD_TEXT]
    row = _row(conn, did)
    assert row["first_comment_posted_at"] is not None and row["first_comment_text"] == OLD_TEXT
    assert row["first_comment_error"] == "stale comment live: links old"
    assert result.posted == 1
    assert result.errors == [f"delivery {did}: stale comment live: links old"]


def test_the_next_pass_recognises_the_old_text_already_live_and_reports_it(conn):
    # Stricter than before: recognised (nothing posted) AND reported, never silently done.
    did = _published(conn)
    _rewrite(conn, did)
    service = _Service(pages=[_thread(OLD_TEXT)])
    result = drain_first_comments(conn, service_factory=lambda: service, now=_NOW)
    assert service.inserted == [] and result.posted == 1
    row = _row(conn, did)
    assert row["first_comment_posted_at"] is not None and row["first_comment_text"] == OLD_TEXT
    assert row["first_comment_error"] == "stale comment live: links old"
    assert result.errors == [f"delivery {did}: stale comment live: links old"]


def test_a_lost_response_then_a_relink_is_reported_not_marked_done(conn):
    did = _published(conn)

    class Lost(_Service):
        def insert(self, part, body):
            self.inserted.append(body["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
            raise TimeoutError("read timed out")  # YouTube did create it

    drain_first_comments(conn, service_factory=lambda: Lost(), now=_NOW)
    _rewrite(conn, did)
    service = _Service(pages=[_thread(OLD_TEXT)])
    result = drain_first_comments(conn, service_factory=lambda: service, now=_NOW)
    row = _row(conn, did)
    assert service.inserted == [] and row["first_comment_text"] == OLD_TEXT
    assert row["first_comment_error"] == "stale comment live: links old"
    assert len(result.errors) == 1


def test_the_new_text_already_live_is_an_ordinary_post(conn):
    did = _published(conn)
    _rewrite(conn, did)
    service = _Service(pages=[_thread(NEW_TEXT)])
    result = drain_first_comments(conn, service_factory=lambda: service, now=_NOW)
    row = _row(conn, did)
    assert result.posted == 1 and result.errors == []
    assert row["first_comment_text"] == NEW_TEXT and row["first_comment_error"] is None


def test_a_stale_report_names_no_id_when_the_text_has_none():
    from gomoveshift.publishing.queue import stale_live_error

    assert stale_live_error("hello", "hello again") == (
        "stale comment live: links an id the relink replaced"
    )
    assert stale_live_error(OLD_TEXT, NEW_TEXT) == "stale comment live: links old"


def test_a_rewrite_between_select_and_post_posts_the_fresh_text(conn, monkeypatch):
    did = _published(conn)
    _rewrite_after_select(monkeypatch, youtube_comments, conn, did)
    service = _Service()
    result = drain_first_comments(conn, service_factory=lambda: service, now=_NOW)
    assert service.inserted == [NEW_TEXT] and result.posted == 1
    assert _row(conn, did)["first_comment_posted_at"] is not None


def test_post_first_comment_accepts_either_text_as_already_there():
    for said in (OLD_TEXT, NEW_TEXT):
        service = _Service(pages=[_thread(said)])
        assert post_first_comment(service, "vid", NEW_TEXT, also_accept=(OLD_TEXT,)) == "t-old"
        assert service.inserted == []


def test_meta_rewrite_during_the_post_records_the_live_old_text_and_reports_it(conn):
    did = _published(conn, platform="instagram")
    seen = []

    class Adapter:
        def post_comment(self, object_id, text, also_accept=()):
            seen.append(text)
            _rewrite(conn, did)

    result = meta_comments.drain_meta_first_comments(
        conn, adapter_factories={"instagram": Adapter}, now=_NOW
    )
    again = meta_comments.drain_meta_first_comments(
        conn, adapter_factories={"instagram": Adapter}, now=_NOW
    )
    assert seen == [OLD_TEXT]  # never a second comment beside the live one
    assert again.posted == 0
    row = _row(conn, did)
    assert row["first_comment_posted_at"] is not None and row["first_comment_text"] == OLD_TEXT
    assert row["first_comment_error"] == "stale comment live: links old"
    assert result.posted == 1 and result.errors == [
        f"delivery {did}: stale comment live: links old"
    ]


def test_meta_a_prior_text_already_live_is_reported_not_posted_over(conn):
    did = _published(conn, platform="facebook")
    _rewrite(conn, did)
    seen = []

    class Adapter:
        def post_comment(self, object_id, text, also_accept=()):
            seen.append((text, tuple(also_accept)))
            raise PriorCommentLiveError("c9", OLD_TEXT)

    result = meta_comments.drain_meta_first_comments(
        conn, adapter_factories={"facebook": Adapter}, now=_NOW
    )
    assert seen == [(NEW_TEXT, (OLD_TEXT,))]
    row = _row(conn, did)
    assert row["first_comment_posted_at"] is not None and row["first_comment_text"] == OLD_TEXT
    assert row["first_comment_error"] == "stale comment live: links old"
    assert result.errors == [f"delivery {did}: stale comment live: links old"]


@pytest.mark.parametrize("field", ["text", "message"])
def test_the_graph_dedup_reports_a_prior_text_and_posts_nothing(field):
    class Resp:
        status_code = 200

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    class Session:
        def __init__(self):
            self.posts = []

        def get(self, url, **kw):
            return Resp({"data": [{"id": "c1", field: OLD_TEXT}]})

        def post(self, url, **kw):
            self.posts.append(url)
            return Resp({"id": "new"})

    session = Session()
    with pytest.raises(PriorCommentLiveError) as info:
        post_graph_comment(
            session, "https://g", "OBJ", NEW_TEXT, "tok", text_field=field, permission="p",
            raise_for_error=lambda s, b: None, also_accept=(OLD_TEXT,),
        )  # fmt: skip
    assert info.value.comment_id == "c1" and info.value.said == OLD_TEXT
    assert session.posts == []


def test_meta_posts_the_fresh_text_when_rewritten_before_the_post(conn, monkeypatch):
    did = _published(conn, platform="instagram")
    _rewrite_after_select(monkeypatch, meta_comments, conn, did)
    seen = []
    adapter = type("A", (), {"post_comment": lambda s, o, t, also_accept=(): seen.append(t)})()
    result = meta_comments.drain_meta_first_comments(
        conn, adapter_factories={"instagram": lambda: adapter}, now=_NOW
    )
    assert seen == [NEW_TEXT] and result.posted == 1
    assert json.loads(_row(conn, did)["first_comment_prior_text"]) == [OLD_TEXT]
