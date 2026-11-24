"""A first comment rewritten while a drain posts it never marks the wrong text posted."""

import json
from types import SimpleNamespace

import pytest

from gomoveshift.publishing import meta_comments, youtube_comments
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


def test_a_rewrite_during_the_post_never_marks_the_new_text_posted(conn):
    did = _published(conn)
    service = _Service(on_insert=lambda: _rewrite(conn, did))
    result = drain_first_comments(conn, service_factory=lambda: service, now=_NOW)
    assert service.inserted == [OLD_TEXT]
    assert result.posted == 0
    row = _row(conn, did)
    assert row["first_comment_posted_at"] is None and row["first_comment_text"] == NEW_TEXT


def test_the_next_pass_recognises_the_old_text_already_live(conn):
    did = _published(conn)
    _rewrite(conn, did)
    service = _Service(pages=[_thread(OLD_TEXT)])
    result = drain_first_comments(conn, service_factory=lambda: service, now=_NOW)
    assert service.inserted == [] and result.posted == 1
    assert _row(conn, did)["first_comment_posted_at"] is not None


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


def test_meta_rewrite_during_the_post_never_marks_the_new_text_posted(conn):
    did = _published(conn, platform="instagram")
    seen = []

    class Adapter:
        def post_comment(self, object_id, text):
            seen.append(text)
            _rewrite(conn, did)

    result = meta_comments.drain_meta_first_comments(
        conn, adapter_factories={"instagram": Adapter}, now=_NOW
    )
    assert seen == [OLD_TEXT] and result.posted == 0
    row = _row(conn, did)
    assert row["first_comment_posted_at"] is None and row["first_comment_text"] == NEW_TEXT


def test_meta_posts_the_fresh_text_when_rewritten_before_the_post(conn, monkeypatch):
    did = _published(conn, platform="instagram")
    _rewrite_after_select(monkeypatch, meta_comments, conn, did)
    seen = []
    adapter = type("A", (), {"post_comment": lambda s, o, t: seen.append(t)})()
    result = meta_comments.drain_meta_first_comments(
        conn, adapter_factories={"instagram": lambda: adapter}, now=_NOW
    )
    assert seen == [NEW_TEXT] and result.posted == 1
    assert json.loads(_row(conn, did)["first_comment_prior_text"]) == [OLD_TEXT]
