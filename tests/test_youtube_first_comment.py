"""Tests that a YouTube first comment posts 24h after publish.

No network: ``post_first_comment``/``drain_first_comments`` take an
injected fake ``service`` object, mirroring
``cli.handlers.comments.run_comments_pull``'s idiom."""

import json

import pytest
from googleapiclient.errors import HttpError

from gomoveshift.publishing import youtube_comments
from gomoveshift.publishing.queue import (
    FIRST_COMMENT_MAX_ATTEMPTS,
    PUBLISHED_AT_PLATFORM_CONFIRMED as PLATFORM,
    enqueue,
    open_queue_db,
    select_due_first_comments,
    transition,
)
from gomoveshift.publishing.youtube_comments import (
    drain_first_comments,
    post_first_comment,
)

_PUB_AT = "2026-08-23T00:00:00+00:00"
_NOW_25H_LATER = "2026-08-23T00:06:00+00:00"


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _make_http_error(status_code: int, reason: str | None = None, message: str = "forbidden"):
    """Mirrors ``tests/test_youtube_uploader.py::_make_http_error``."""

    class _Resp:
        def __init__(self, status):
            self.status = status
            self.reason = "Forbidden" if status == 403 else "Error"

    errors = [{"message": reason or message, "domain": "youtube.commentThreads", "reason": reason}]
    body = json.dumps(
        {"error": {"code": status_code, "message": message, "errors": errors}}
    ).encode("utf-8")
    return HttpError(_Resp(status_code), body, uri="fake://")


class _FakeInsertRequest:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc

    def execute(self):
        if self._exc is not None:
            raise self._exc
        return self._response


class _FakeListRequest:
    def __init__(self, page):
        self._page = page

    def execute(self):
        if isinstance(self._page, Exception):
            raise self._page
        return self._page


class _FakeCommentThreads:
    """``insert`` records to ``calls``; ``list`` records to ``lists`` and
    answers with the next of ``pages`` (an Exception raises), then empty."""

    def __init__(self, response=None, exc=None, calls=None, pages=()):
        self._response = response
        self._exc = exc
        self.calls = calls if calls is not None else []
        self.lists = []
        self._pages = list(pages)

    def insert(self, part, body):
        self.calls.append({"part": part, "body": body})
        return _FakeInsertRequest(self._response, self._exc)

    def list(self, **kwargs):
        self.lists.append(kwargs)
        return _FakeListRequest(self._pages.pop(0) if self._pages else {"items": []})


class _FakeService:
    def __init__(self, response=None, exc=None, pages=()):
        self._threads = _FakeCommentThreads(response=response, exc=exc, pages=pages)

    def commentThreads(self):
        return self._threads


def _make_published_youtube_row(conn, short_index, first_comment_text="great drive!"):
    delivery_id = enqueue(
        conn,
        "proj",
        short_index,
        "youtube",
        f"hash-{short_index}",
        first_comment_text=first_comment_text,
    )
    assert transition(
        conn,
        delivery_id,
        "queued",
        "published",
        platform_upload_id=f"yt-video-{short_index}",
        platform_post_id=f"yt-playlist-{short_index}",
        published_at_source=PLATFORM,
    )
    conn.execute("UPDATE deliveries SET published_at=? WHERE id=?", (_PUB_AT, delivery_id))
    conn.commit()
    return delivery_id


def test_post_first_comment_sends_the_expected_request_body():
    service = _FakeService(response={"id": "comment-abc"})
    comment_id = post_first_comment(service, "video-123", "nice drive!")
    assert comment_id == "comment-abc"
    call = service._threads.calls[0]
    assert call["part"] == "snippet"
    assert call["body"]["snippet"]["videoId"] == "video-123"
    assert call["body"]["snippet"]["topLevelComment"]["snippet"]["textOriginal"] == "nice drive!"


def test_drain_first_comments_posts_and_records_on_the_row(conn):
    delivery_id = _make_published_youtube_row(conn, 60)
    service = _FakeService(response={"id": "comment-xyz"})
    result = drain_first_comments(conn, service_factory=lambda: service, now=_NOW_25H_LATER)
    assert result.posted == 1
    assert result.failed == 0
    assert result.skipped_scope_error == 0
    row = conn.execute(
        "SELECT first_comment_posted_at, first_comment_error FROM deliveries WHERE id=?",
        (delivery_id,),
    ).fetchone()
    assert row["first_comment_posted_at"] is not None
    assert row["first_comment_error"] is None


def test_drain_first_comments_never_builds_a_service_when_nothing_is_due(conn):
    built = []

    def factory():
        built.append(True)
        return _FakeService()

    result = drain_first_comments(conn, service_factory=factory, now=_NOW_25H_LATER)
    assert result.posted == 0
    assert built == []


def test_drain_first_comments_handles_a_scope_error_without_raising(conn):
    delivery_id = _make_published_youtube_row(conn, 61)
    exc = _make_http_error(403, reason="insufficientPermissions")
    service = _FakeService(exc=exc)
    result = drain_first_comments(conn, service_factory=lambda: service, now=_NOW_25H_LATER)
    assert result.posted == 0
    assert result.failed == 0
    assert result.skipped_scope_error == 1
    assert result.needs_reauth is True
    row = conn.execute(
        "SELECT first_comment_posted_at, first_comment_error FROM deliveries WHERE id=?",
        (delivery_id,),
    ).fetchone()
    assert row["first_comment_posted_at"] is None
    assert "insufficientPermissions" in row["first_comment_error"]


def test_drain_first_comments_counts_a_generic_failure_and_continues(conn):
    first_id = _make_published_youtube_row(conn, 62)
    second_id = _make_published_youtube_row(conn, 63)
    service = _FakeService(exc=RuntimeError("boom"))
    result = drain_first_comments(conn, service_factory=lambda: service, now=_NOW_25H_LATER)
    assert result.posted == 0
    assert result.failed == 2
    assert result.skipped_scope_error == 0
    for delivery_id in (first_id, second_id):
        row = conn.execute(
            "SELECT first_comment_error FROM deliveries WHERE id=?", (delivery_id,)
        ).fetchone()
        assert "boom" in row["first_comment_error"]


def test_drain_first_comments_gives_up_on_a_deleted_video(conn):
    # A wrong-place short taken down by hand: its comment must not retry forever.
    delivery_id = _make_published_youtube_row(conn, 64)
    gone = _make_http_error(404, reason="videoNotFound", message="video not found")
    service = _FakeService(exc=gone)
    for _ in range(FIRST_COMMENT_MAX_ATTEMPTS + 2):
        drain_first_comments(conn, service_factory=lambda: service, now=_NOW_25H_LATER)
    assert len(service._threads.calls) == FIRST_COMMENT_MAX_ATTEMPTS
    row = conn.execute(
        "SELECT first_comment_posted_at, first_comment_error, first_comment_attempts "
        "FROM deliveries WHERE id=?",
        (delivery_id,),
    ).fetchone()
    assert row["first_comment_posted_at"] is None
    assert "video not found" in row["first_comment_error"]
    assert row["first_comment_attempts"] == FIRST_COMMENT_MAX_ATTEMPTS
    assert select_due_first_comments(conn, now=_NOW_25H_LATER) == []


def test_drain_first_comments_never_counts_a_scope_error_toward_the_cap(conn):
    # A missing youtube.force-ssl scope waits for a re-auth, not for the video.
    delivery_id = _make_published_youtube_row(conn, 65)
    service = _FakeService(exc=_make_http_error(403, reason="insufficientPermissions"))
    for _ in range(FIRST_COMMENT_MAX_ATTEMPTS + 2):
        drain_first_comments(conn, service_factory=lambda: service, now=_NOW_25H_LATER)
    row = conn.execute(
        "SELECT first_comment_attempts FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row["first_comment_attempts"] == 0
    due = select_due_first_comments(conn, now=_NOW_25H_LATER)
    assert [r["id"] for r in due] == [delivery_id]


def test_drain_first_comments_counts_comments_turned_off(conn):
    delivery_id = _make_published_youtube_row(conn, 66)
    closed = _make_http_error(403, reason="commentsDisabled", message="comments are disabled")
    drain_first_comments(conn, service_factory=lambda: _FakeService(exc=closed), now=_NOW_25H_LATER)
    row = conn.execute(
        "SELECT first_comment_attempts FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row["first_comment_attempts"] == 1


@pytest.mark.parametrize(
    "exc",
    [
        _make_http_error(503, reason="backendError", message="Backend Error"),
        _make_http_error(429, reason="rateLimitExceeded", message="rate limited"),
        _make_http_error(403, reason="quotaExceeded", message="quota exceeded"),
        _make_http_error(401, reason="authError", message="Invalid Credentials"),
        TimeoutError("timed out"),
    ],
    ids=["5xx", "rate-limit", "quota-spent", "lapsed-token", "network-down"],
)
def test_drain_first_comments_never_counts_an_outage_toward_the_cap(conn, exc):
    # An overnight outage or a spent daily quota says nothing about the video:
    # it must not use up a live short's tries, or the short never gets its comment.
    delivery_id = _make_published_youtube_row(conn, 67)
    service = _FakeService(exc=exc)
    for _ in range(FIRST_COMMENT_MAX_ATTEMPTS + 2):
        drain_first_comments(conn, service_factory=lambda: service, now=_NOW_25H_LATER)
    assert len(service._threads.calls) == FIRST_COMMENT_MAX_ATTEMPTS + 2
    row = conn.execute(
        "SELECT first_comment_attempts FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row["first_comment_attempts"] == 0
    due = select_due_first_comments(conn, now=_NOW_25H_LATER)
    assert [r["id"] for r in due] == [delivery_id]


# --- never a second copy ---------------------------------------------------
# Facebook case: Graph created the comment but
# its answer was lost to a read timeout, so the next pass would post it again.
# googleapiclient reads with a 60 s timeout, so YouTube has the same gap:
# before posting, the video's comment threads are read for the exact text.

_TEXT = "great drive!"


def _threads(*texts, next_page=None):
    """One ``commentThreads.list`` page: top-level comments t0, t1, ... saying ``texts``."""
    page = {
        "items": [
            {
                "id": f"t{i}",
                "snippet": {"topLevelComment": {"snippet": {"textOriginal": t, "textDisplay": t}}},
            }
            for i, t in enumerate(texts)
        ]
    }
    if next_page:
        page["nextPageToken"] = next_page
    return page


def test_never_posts_a_comment_that_is_already_there():
    service = _FakeService(response={"id": "new"}, pages=[_threads("nice drive", _TEXT)])
    assert post_first_comment(service, "video-123", _TEXT) == "t1"
    assert service._threads.calls == []
    (listed,) = service._threads.lists
    assert listed["videoId"] == "video-123"
    assert listed["part"] == "snippet"
    assert listed["textFormat"] == "plainText"
    assert listed["maxResults"] == 100
    assert "pageToken" not in listed


def test_only_the_exact_text_counts_as_already_there():
    service = _FakeService(
        response={"id": "new"}, pages=[_threads(f"{_TEXT} 🔥", _TEXT + "X", "nice drive")]
    )
    assert post_first_comment(service, "video-123", _TEXT) == "new"
    assert len(service._threads.calls) == 1


def test_the_check_reads_every_page_of_comments():
    service = _FakeService(
        response={"id": "new"}, pages=[_threads("nice drive", next_page="P2"), _threads(_TEXT)]
    )
    assert post_first_comment(service, "video-123", _TEXT) == "t0"
    assert [call.get("pageToken") for call in service._threads.lists] == [None, "P2"]
    assert service._threads.calls == []


def test_the_check_stops_at_its_page_cap_and_then_posts():
    cap = youtube_comments.COMMENT_CHECK_MAX_PAGES
    pages = [_threads("nice drive", next_page=f"P{i}") for i in range(cap + 2)]
    service = _FakeService(response={"id": "new"}, pages=pages)
    assert post_first_comment(service, "video-123", _TEXT) == "new"
    assert len(service._threads.lists) == cap
    assert len(service._threads.calls) == 1


@pytest.mark.parametrize(
    "exc",
    [
        _make_http_error(503, reason="backendError", message="Backend Error"),
        TimeoutError("timed out"),
    ],
    ids=["5xx", "network-down"],
)
def test_a_failed_check_never_posts(exc):
    service = _FakeService(response={"id": "new"}, pages=[exc])
    with pytest.raises(type(exc)):
        post_first_comment(service, "video-123", _TEXT)
    assert service._threads.calls == []


def test_a_check_refused_for_scope_waits_for_reauth_and_never_posts(conn):
    _make_published_youtube_row(conn, 70, first_comment_text=_TEXT)
    service = _FakeService(
        response={"id": "new"}, pages=[_make_http_error(403, reason="insufficientPermissions")]
    )
    result = drain_first_comments(conn, service_factory=lambda: service, now=_NOW_25H_LATER)
    assert result.skipped_scope_error == 1
    assert service._threads.calls == []


class _SlowYouTube:
    """Every insert creates the comment, then its answer is lost (read timeout)."""

    def __init__(self):
        self.threads = []
        self.inserts = 0

    def commentThreads(self):
        return self

    def list(self, **kwargs):
        return _FakeListRequest(_threads(*(t for v, t in self.threads if v == kwargs["videoId"])))

    def insert(self, part, body):
        self.inserts += 1
        snippet = body["snippet"]
        self.threads.append(
            (snippet["videoId"], snippet["topLevelComment"]["snippet"]["textOriginal"])
        )
        return _FakeInsertRequest(exc=TimeoutError("The read operation timed out"))


def test_a_comment_that_landed_but_timed_out_is_never_posted_twice(conn):
    delivery_id = _make_published_youtube_row(conn, 71, first_comment_text=_TEXT)
    youtube = _SlowYouTube()

    def drain():
        return drain_first_comments(conn, service_factory=lambda: youtube, now=_NOW_25H_LATER)

    assert drain().failed == 1
    assert drain().posted == 1
    assert youtube.inserts == 1
    row = conn.execute(
        "SELECT first_comment_posted_at, first_comment_error FROM deliveries WHERE id=?",
        (delivery_id,),
    ).fetchone()
    assert row["first_comment_posted_at"] is not None
    assert row["first_comment_error"] is None
    assert drain().posted == 0
    assert youtube.inserts == 1
