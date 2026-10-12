"""Every short on YouTube / Instagram / Facebook gets a first comment that
links its full drive (the parent long-form video); TikTok has no comment API.

Tmp sqlite DBs and fake sessions only -- no network, never the live queue."""

from __future__ import annotations

import json
import logging
from unittest import mock

import pytest
import requests

from gomoveshift.publishing import first_comment
from gomoveshift.publishing.enqueue_from_plan import (
    backfill_first_comment_text,
    enqueue_from_plan_sidecar,
)
from gomoveshift.publishing.facebook_adapter import (
    FacebookAuthBlockedError,
    FacebookPermanentError,
    FacebookReelsAdapter,
    FacebookRetryableError,
)
from gomoveshift.publishing.first_comment import CommentPermissionError, full_drive_comment_text
from gomoveshift.publishing.instagram_adapter import InstagramReelsAdapter
from gomoveshift.publishing.meta_comments import drain_meta_first_comments
from gomoveshift.publishing.queue import (
    FIRST_COMMENT_DELAY_S,
    FIRST_COMMENT_MAX_ATTEMPTS,
    enqueue,
    open_queue_db,
    pause_platform,
    select_due_first_comments,
    transition,
)
from gomoveshift.publishing.queue import PUBLISHED_AT_PLATFORM_CONFIRMED as PLATFORM
from gomoveshift.publishing.youtube_comments import FirstCommentResult, drain_first_comments

from test_enqueue_from_plan import _record, _write_sidecar

_PUB_AT = "2026-08-23T00:00:00+00:00"
_NOW = "2026-08-23T00:06:00+00:00"  # 6 minutes after publish: due
_TOO_SOON = "2026-08-23T00:04:00+00:00"  # 4 minutes: not yet
_TEXT = "🎬 Watch the full drive: https://youtu.be/PARENT1"


@pytest.fixture(autouse=True)
def _own_console_data_dir(tmp_path, monkeypatch):
    # Keeps ``dispatch-queue --apply``'s notify seen file out of the live console dir.
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "_console"))


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _published(conn, short_index, platform, text=_TEXT, **ids):
    delivery_id = enqueue(
        conn, "proj", short_index, platform, f"h{short_index}{platform}", first_comment_text=text
    )
    assert transition(conn, delivery_id, "queued", "published", published_at_source=PLATFORM, **ids)
    conn.execute("UPDATE deliveries SET published_at=? WHERE id=?", (_PUB_AT, delivery_id))
    conn.commit()
    return delivery_id


def _row(conn, delivery_id):
    return conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()


# --- text -------------------------------------------------------------------


def test_text_links_the_parent_video():
    assert full_drive_comment_text("PARENT1") == _TEXT


@pytest.mark.parametrize("bad", ["", "   ", "\n"])
def test_text_refuses_a_blank_id(bad):
    with pytest.raises(ValueError):
        full_drive_comment_text(bad)


def test_delay_is_five_minutes():
    assert FIRST_COMMENT_DELAY_S == 300


# --- enqueue ----------------------------------------------------------------


def test_enqueue_sets_text_for_three_platforms_and_none_for_tiktok(conn, tmp_path):
    record = _record(selected=("youtube", "instagram", "facebook", "tiktok"))
    record["parent_lf_video_id"] = "PARENT1"
    path = _write_sidecar(tmp_path, [record])
    enqueue_from_plan_sidecar(conn, path)
    got = dict(conn.execute("SELECT platform, first_comment_text FROM deliveries").fetchall())
    assert got == {
        "youtube": _TEXT,
        "instagram": _TEXT,
        "facebook": _TEXT,
        "tiktok": None,
    }


def test_enqueue_without_parent_id_leaves_text_null(conn, tmp_path):
    path = _write_sidecar(tmp_path, [_record(selected=("youtube",))])
    enqueue_from_plan_sidecar(conn, path)
    assert conn.execute("SELECT first_comment_text FROM deliveries").fetchone()[0] is None


# --- backfill ---------------------------------------------------------------


def _backfill_fixture(conn, tmp_path, platform="youtube", parent="PARENT1"):
    record = _record(short_index=7, selected=(platform,))
    if parent is not None:
        record["parent_lf_video_id"] = parent
    path = _write_sidecar(tmp_path, [record])
    delivery_id = enqueue(
        conn, "p", 7, platform, "hash-a", sidecar_path=str(path), first_comment_text=None
    )
    return delivery_id, path


@pytest.mark.parametrize("platform", ["youtube", "instagram", "facebook"])
def test_backfill_fills_null_and_empty_text(conn, tmp_path, platform):
    delivery_id, _ = _backfill_fixture(conn, tmp_path, platform)
    conn.execute("UPDATE deliveries SET first_comment_text='' WHERE id=?", (delivery_id,))
    conn.commit()
    backfill_first_comment_text(conn)
    assert _row(conn, delivery_id)["first_comment_text"] == _TEXT


def test_backfill_is_idempotent_and_leaves_tiktok_alone(conn, tmp_path):
    delivery_id, _ = _backfill_fixture(conn, tmp_path)
    tiktok_id = enqueue(conn, "p", 7, "tiktok", "hash-a", sidecar_path=None)
    backfill_first_comment_text(conn)
    backfill_first_comment_text(conn)
    assert _row(conn, delivery_id)["first_comment_text"] == _TEXT
    assert _row(conn, tiktok_id)["first_comment_text"] is None


def test_backfill_never_touches_a_posted_row(conn, tmp_path):
    delivery_id, _ = _backfill_fixture(conn, tmp_path)
    conn.execute(
        "UPDATE deliveries SET first_comment_posted_at=?, first_comment_text='' WHERE id=?",
        (_NOW, delivery_id),
    )
    conn.commit()
    backfill_first_comment_text(conn)
    assert _row(conn, delivery_id)["first_comment_text"] == ""


def test_backfill_keeps_existing_text(conn, tmp_path):
    delivery_id, _ = _backfill_fixture(conn, tmp_path)
    conn.execute("UPDATE deliveries SET first_comment_text='custom' WHERE id=?", (delivery_id,))
    conn.commit()
    backfill_first_comment_text(conn)
    assert _row(conn, delivery_id)["first_comment_text"] == "custom"


def test_backfill_skips_unreadable_sidecar_and_missing_parent(conn, tmp_path, caplog):
    missing, _ = _backfill_fixture(conn, tmp_path, parent=None)
    gone = enqueue(conn, "q", 1, "instagram", "hq", sidecar_path=str(tmp_path / "nope.json"))
    nopath = enqueue(conn, "r", 1, "facebook", "hr")
    with caplog.at_level(logging.WARNING):
        backfill_first_comment_text(conn)
    for delivery_id in (missing, gone, nopath):
        assert not _row(conn, delivery_id)["first_comment_text"]
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)


def test_backfill_skips_corrupt_json(conn, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    delivery_id = enqueue(conn, "q", 2, "youtube", "hq", sidecar_path=str(bad))
    backfill_first_comment_text(conn)
    assert not _row(conn, delivery_id)["first_comment_text"]


# --- selection --------------------------------------------------------------


def test_selection_targets_video_id_for_youtube_and_post_id_for_meta(conn):
    yt = _published(conn, 1, "youtube", platform_upload_id="VID", platform_post_id="PLAYLIST")
    ig = _published(conn, 2, "instagram", platform_post_id="IGMEDIA")
    fb = _published(conn, 3, "facebook", platform_post_id="FBREEL")
    due = {r["id"]: r["comment_target_id"] for r in select_due_first_comments(conn, now=_NOW)}
    assert due == {yt: "VID", ig: "IGMEDIA", fb: "FBREEL"}


def test_selection_respects_the_five_minute_delay(conn):
    _published(conn, 1, "instagram", platform_post_id="IGMEDIA")
    assert select_due_first_comments(conn, now=_TOO_SOON) == []


def test_selection_excludes_tiktok_and_missing_target(conn):
    _published(conn, 1, "tiktok", platform_post_id="TT")
    _published(conn, 2, "youtube", platform_post_id="PLAYLIST")  # no video id
    _published(conn, 3, "instagram")  # no media id
    assert select_due_first_comments(conn, now=_NOW) == []


def test_selection_never_trusts_an_unlabelled_published_at(conn):
    # The bare time is old enough, but with no source it is not a measured
    # publish moment: only the row's last write can bound the publish.
    delivery_id = _published(conn, 1, "instagram", platform_post_id="IGMEDIA")
    conn.execute(
        "UPDATE deliveries SET published_at_source=NULL, updated_at=? WHERE id=?",
        (_NOW, delivery_id),
    )
    conn.commit()
    assert select_due_first_comments(conn, now=_NOW) == []


def test_selection_excludes_a_paused_platform(conn):
    _published(conn, 1, "instagram", platform_post_id="IGMEDIA")
    fb = _published(conn, 2, "facebook", platform_post_id="FBREEL")
    assert pause_platform(conn, "instagram", now=_PUB_AT, reason="hold")
    assert [r["id"] for r in select_due_first_comments(conn, now=_NOW)] == [fb]


def test_selection_can_be_limited_to_platforms(conn):
    _published(conn, 1, "instagram", platform_post_id="IGMEDIA")
    fb = _published(conn, 2, "facebook", platform_post_id="FBREEL")
    got = select_due_first_comments(conn, now=_NOW, platforms=("facebook",))
    assert [r["id"] for r in got] == [fb]


# --- selection: shorts published before publish times were recorded -------
# Early shorts published before
# ``published_at_source`` existed, so no labelled moment could ever make them
# due and they never got their comment. A row's last write is never earlier
# than its publish, so ``updated_at`` bounds the publish from above.

_OLD_WRITE = "2026-08-22T23:00:00+00:00"  # an hour before _NOW


def _unlabelled(conn, short_index, platform, *, updated_at=_OLD_WRITE, **ids):
    delivery_id = _published(conn, short_index, platform, **ids)
    conn.execute(
        "UPDATE deliveries SET published_at=NULL, published_at_source=NULL, updated_at=? "
        "WHERE id=?",
        (updated_at, delivery_id),
    )
    conn.commit()
    return delivery_id


def test_selection_reaches_shorts_published_before_publish_times_were_recorded(conn):
    yt = _unlabelled(conn, 1, "youtube", platform_upload_id="VID", platform_post_id="PLAYLIST")
    ig = _unlabelled(conn, 2, "instagram", platform_post_id="IGMEDIA")
    fb = _unlabelled(conn, 3, "facebook", platform_post_id="FBREEL")
    due = {r["id"]: r["comment_target_id"] for r in select_due_first_comments(conn, now=_NOW)}
    assert due == {yt: "VID", ig: "IGMEDIA", fb: "FBREEL"}


def test_an_unlabelled_time_is_bounded_by_the_last_write(conn):
    # A bare published_at, no source.
    delivery_id = _published(conn, 1, "facebook", platform_post_id="FBREEL")
    conn.execute(
        "UPDATE deliveries SET published_at_source=NULL, updated_at=? WHERE id=?",
        (_OLD_WRITE, delivery_id),
    )
    conn.commit()
    assert [r["id"] for r in select_due_first_comments(conn, now=_NOW)] == [delivery_id]


def test_an_unlabelled_row_waits_until_its_last_write_is_past_the_delay(conn):
    delivery_id = _unlabelled(conn, 1, "instagram", updated_at=_PUB_AT, platform_post_id="IGMEDIA")
    assert select_due_first_comments(conn, now=_TOO_SOON) == []
    assert [r["id"] for r in select_due_first_comments(conn, now=_NOW)] == [delivery_id]


def test_an_unlabelled_row_waits_until_its_slot_is_past_the_delay(conn):
    # A scheduled YouTube upload goes public at its slot, not at the upload.
    delivery_id = _unlabelled(conn, 1, "youtube", platform_upload_id="VID")
    conn.execute(
        "UPDATE deliveries SET scheduled_at=? WHERE id=?",
        ("2026-08-23T00:03:00+00:00", delivery_id),
    )
    conn.commit()
    assert select_due_first_comments(conn, now=_NOW) == []
    later = "2026-08-23T00:08:00+00:00"
    assert [r["id"] for r in select_due_first_comments(conn, now=later)] == [delivery_id]


@pytest.mark.parametrize(
    "status", ["queued", "awaiting_manual_publish", "failed_permanent", "cancelled"]
)
def test_an_unlabelled_row_that_never_published_is_never_due(conn, status):
    delivery_id = _unlabelled(conn, 1, "instagram", platform_post_id="IGMEDIA")
    conn.execute("UPDATE deliveries SET status=? WHERE id=?", (status, delivery_id))
    conn.commit()
    assert select_due_first_comments(conn, now=_NOW) == []


def test_an_unlabelled_row_on_a_paused_platform_waits(conn):
    _unlabelled(conn, 1, "instagram", platform_post_id="IGMEDIA")
    fb = _unlabelled(conn, 2, "facebook", platform_post_id="FBREEL")
    assert pause_platform(conn, "instagram", now=_PUB_AT, reason="hold")
    assert [r["id"] for r in select_due_first_comments(conn, now=_NOW)] == [fb]


def test_the_backlog_comments_once_per_platform_per_pass(conn):
    # 20 old posts at once would put the same link under every one of them
    # within a minute; the hourly pass spreads them out instead.
    ig_old = _unlabelled(conn, 1, "instagram", platform_post_id="IG1")
    _unlabelled(conn, 2, "instagram", platform_post_id="IG2")
    fb_old = _unlabelled(conn, 3, "facebook", platform_post_id="FB3")
    _unlabelled(conn, 4, "facebook", platform_post_id="FB4")
    ig_new = _published(conn, 5, "instagram", platform_post_id="IG5")
    got = [r["id"] for r in select_due_first_comments(conn, now=_NOW)]
    assert got == [ig_old, fb_old, ig_new]


def test_a_backlog_row_that_keeps_failing_never_starves_the_rest(conn):
    stuck = _unlabelled(conn, 1, "instagram", platform_post_id="IG1")
    fresh = _unlabelled(conn, 2, "instagram", platform_post_id="IG2")
    conn.execute("UPDATE deliveries SET first_comment_error='gone' WHERE id=?", (stuck,))
    conn.commit()
    assert [r["id"] for r in select_due_first_comments(conn, now=_NOW)] == [fresh]


# --- YouTube drain posts to the VIDEO id ------------------------------------


class _YtService:
    """``insert`` records to ``calls``; ``list`` (the already-there check)
    records to ``lists`` and finds no comments."""

    def __init__(self):
        self.calls = []
        self.lists = []

    def commentThreads(self):
        return self

    def list(self, **kwargs):
        self.lists.append(kwargs)
        return mock.Mock(execute=lambda: {"items": []})

    def insert(self, part, body):
        self.calls.append(body)
        return mock.Mock(execute=lambda: {"id": "c1"})


def test_youtube_drain_posts_to_the_video_id_and_only_youtube(conn):
    yt = _published(conn, 1, "youtube", platform_upload_id="VID", platform_post_id="PLAYLIST")
    ig = _published(conn, 2, "instagram", platform_post_id="IGMEDIA")
    service = _YtService()
    result = drain_first_comments(conn, service_factory=lambda: service, now=_NOW)
    assert result.posted == 1
    assert [c["videoId"] for c in service.lists] == ["VID"]
    assert [c["snippet"]["videoId"] for c in service.calls] == ["VID"]
    assert _row(conn, yt)["first_comment_posted_at"] is not None
    assert _row(conn, ig)["first_comment_posted_at"] is None


def test_youtube_drain_posts_the_first_comment_on_an_unlabelled_short(conn):
    delivery_id = _unlabelled(conn, 1, "youtube", platform_upload_id="VID")
    service = _YtService()
    assert drain_first_comments(conn, service_factory=lambda: service, now=_NOW).posted == 1
    assert [c["snippet"]["videoId"] for c in service.calls] == ["VID"]
    assert _row(conn, delivery_id)["first_comment_posted_at"] is not None


# --- Meta drain -------------------------------------------------------------


class _FakeAdapter:
    def __init__(self, exc=None):
        self.exc = exc
        self.calls = []

    def post_comment(self, object_id, text):
        self.calls.append((object_id, text))
        if self.exc is not None:
            raise self.exc
        return "comment-1"


def test_meta_drain_posts_and_records(conn):
    ig = _published(conn, 1, "instagram", platform_post_id="IGMEDIA")
    fb = _published(conn, 2, "facebook", platform_post_id="FBREEL")
    yt = _published(conn, 3, "youtube", platform_upload_id="VID")
    adapters = {"instagram": _FakeAdapter(), "facebook": _FakeAdapter()}
    result = drain_meta_first_comments(
        conn, adapter_factories={k: (lambda a=a: a) for k, a in adapters.items()}, now=_NOW
    )
    assert result.posted == 2 and result.failed == 0
    assert adapters["instagram"].calls == [("IGMEDIA", _TEXT)]
    assert adapters["facebook"].calls == [("FBREEL", _TEXT)]
    for delivery_id in (ig, fb):
        row = _row(conn, delivery_id)
        assert row["first_comment_posted_at"] is not None
        assert row["first_comment_error"] is None
    assert _row(conn, yt)["first_comment_posted_at"] is None


def test_meta_drain_builds_adapters_lazily(conn):
    _published(conn, 1, "instagram", platform_post_id="IGMEDIA")
    built = []

    def factory(name):
        def make():
            built.append(name)
            return _FakeAdapter()

        return make

    drain_meta_first_comments(
        conn,
        adapter_factories={"instagram": factory("instagram"), "facebook": factory("facebook")},
        now=_NOW,
    )
    assert built == ["instagram"]
    assert (
        drain_meta_first_comments(conn, adapter_factories={"instagram": factory("x")}).posted == 0
    )


def test_meta_drain_skips_a_paused_platform(conn):
    ig = _published(conn, 1, "instagram", platform_post_id="IGMEDIA")
    pause_platform(conn, "instagram", now=_PUB_AT, reason="hold")
    adapter = _FakeAdapter()
    result = drain_meta_first_comments(
        conn, adapter_factories={"instagram": lambda: adapter}, now=_NOW
    )
    assert result.posted == 0 and adapter.calls == []
    assert _row(conn, ig)["first_comment_posted_at"] is None


def test_meta_drain_permission_error_is_logged_once_and_retryable(conn, caplog):
    rows = [_published(conn, i, "instagram", platform_post_id=f"M{i}") for i in (1, 2)]
    adapter = _FakeAdapter(CommentPermissionError("instagram_manage_comments", "denied"))
    with caplog.at_level(logging.WARNING):
        result = drain_meta_first_comments(
            conn, adapter_factories={"instagram": lambda: adapter}, now=_NOW
        )
    assert result.skipped_scope_error == 2 and result.failed == 0 and result.needs_reauth
    warnings = [r for r in caplog.records if "instagram_manage_comments" in r.getMessage()]
    assert len(warnings) == 1
    for delivery_id in rows:
        row = _row(conn, delivery_id)
        assert row["first_comment_posted_at"] is None
        assert "instagram_manage_comments" in row["first_comment_error"]


def test_meta_drain_other_error_is_per_row_and_never_aborts(conn):
    a = _published(conn, 1, "facebook", platform_post_id="A")
    b = _published(conn, 2, "facebook", platform_post_id="B")

    class _Flaky(_FakeAdapter):
        def post_comment(self, object_id, text):
            if object_id == "A":
                raise RuntimeError("boom")
            return super().post_comment(object_id, text)

    result = drain_meta_first_comments(
        conn, adapter_factories={"facebook": lambda: _Flaky()}, now=_NOW
    )
    assert result.failed == 1 and result.posted == 1
    assert "boom" in _row(conn, a)["first_comment_error"]
    assert _row(conn, a)["first_comment_posted_at"] is None
    assert _row(conn, b)["first_comment_posted_at"] is not None


# --- a comment that keeps failing stops ------------------------------------


def test_a_meta_comment_on_a_deleted_reel_stops_after_the_cap(conn):
    # A wrong-place short taken down by hand: its comment must not retry forever.
    delivery_id = _published(conn, 1, "facebook", platform_post_id="GONE")
    adapter = _FakeAdapter(
        FacebookPermanentError(
            "Graph API rejected the request: Object with ID 'GONE' does not exist"
        )
    )
    for _ in range(FIRST_COMMENT_MAX_ATTEMPTS + 2):
        drain_meta_first_comments(conn, adapter_factories={"facebook": lambda: adapter}, now=_NOW)
    assert len(adapter.calls) == FIRST_COMMENT_MAX_ATTEMPTS
    row = _row(conn, delivery_id)
    assert row["first_comment_posted_at"] is None
    assert row["first_comment_attempts"] == FIRST_COMMENT_MAX_ATTEMPTS
    assert "does not exist" in row["first_comment_error"]
    assert select_due_first_comments(conn, now=_NOW) == []


def test_a_meta_permission_error_never_counts_toward_the_cap(conn):
    # A missing permission waits for a re-auth, not for the post.
    delivery_id = _published(conn, 1, "instagram", platform_post_id="M1")
    adapter = _FakeAdapter(CommentPermissionError("instagram_manage_comments", "denied"))
    for _ in range(FIRST_COMMENT_MAX_ATTEMPTS + 2):
        drain_meta_first_comments(conn, adapter_factories={"instagram": lambda: adapter}, now=_NOW)
    assert _row(conn, delivery_id)["first_comment_attempts"] == 0
    assert [r["id"] for r in select_due_first_comments(conn, now=_NOW)] == [delivery_id]


@pytest.mark.parametrize(
    "exc",
    [
        FacebookRetryableError("transient Graph API error: Please retry your request later."),
        FacebookAuthBlockedError("Meta access token expired or revoked; reauthorize"),
        requests.exceptions.ConnectionError("Max retries exceeded with url: /FB1/comments"),
    ],
    ids=["rate-limit-or-5xx", "lapsed-token", "network-down"],
)
def test_a_meta_outage_never_counts_toward_the_cap(conn, exc):
    # An overnight outage says nothing about the post: it must not use up a
    # live short's tries, or the short never gets its comment.
    delivery_id = _published(conn, 1, "facebook", platform_post_id="FB1")
    adapter = _FakeAdapter(exc)
    for _ in range(FIRST_COMMENT_MAX_ATTEMPTS + 2):
        drain_meta_first_comments(conn, adapter_factories={"facebook": lambda: adapter}, now=_NOW)
    assert len(adapter.calls) == FIRST_COMMENT_MAX_ATTEMPTS + 2
    assert _row(conn, delivery_id)["first_comment_attempts"] == 0
    assert [r["id"] for r in select_due_first_comments(conn, now=_NOW)] == [delivery_id]


def test_an_old_post_that_gave_up_is_never_due(conn):
    gave_up = _unlabelled(conn, 1, "instagram", platform_post_id="IG1")
    conn.execute(
        "UPDATE deliveries SET first_comment_error='gone', first_comment_attempts=? WHERE id=?",
        (FIRST_COMMENT_MAX_ATTEMPTS, gave_up),
    )
    conn.commit()
    assert select_due_first_comments(conn, now=_NOW) == []


def test_rows_already_in_the_queue_start_at_zero_attempts(tmp_path):
    # The live queue already holds rows. They must read 0, never NULL: NULL is
    # never under the cap, so they would never get a comment at all.
    c = open_queue_db(tmp_path / "seo")
    c.execute("ALTER TABLE deliveries DROP COLUMN first_comment_attempts")
    c.commit()
    delivery_id = _published(c, 1, "instagram", platform_post_id="M1")
    c.close()
    c = open_queue_db(tmp_path / "seo")
    try:
        assert _row(c, delivery_id)["first_comment_attempts"] == 0
        assert [r["id"] for r in select_due_first_comments(c, now=_NOW)] == [delivery_id]
    finally:
        c.close()


# --- adapters ---------------------------------------------------------------


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class _Session:
    """``post`` answers ``resp``; each ``get`` (the duplicate check) answers
    the next of ``pages`` -- an empty comment list once they run out. An
    exception in either place is raised instead."""

    def __init__(self, resp, *pages):
        self.resp = resp
        self.pages = list(pages)
        self.calls = []
        self.gets = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        page = self.pages.pop(0) if self.pages else _Resp(200, {"data": []})
        if isinstance(page, Exception):
            raise page
        return page


def _comments(field, *texts, after=None):
    """One Graph page of comments carrying ``texts`` in ``field``; ``after``
    adds the cursor and ``next`` link Graph sends when more pages follow."""
    body = {"data": [{"id": f"c{i}", field: text} for i, text in enumerate(texts)]}
    if after:
        body["paging"] = {"cursors": {"after": after}, "next": "https://graph.facebook.com/next"}
    return _Resp(200, body)


def _ig(session):
    return InstagramReelsAdapter({"ig_user_id": "1", "page_access_token": "tok"}, session=session)


def _fb(session):
    return FacebookReelsAdapter({"page_id": "1", "page_access_token": "tok"}, session=session)


@pytest.mark.parametrize("make", [_ig, _fb])
def test_adapter_posts_a_comment_to_the_object(make):
    session = _Session(_Resp(200, {"id": "cmt"}))
    assert make(session).post_comment("OBJ9", "hello") == "cmt"
    url, kwargs = session.calls[0]
    assert url.endswith("/OBJ9/comments")
    assert kwargs["data"]["message"] == "hello"
    assert kwargs["data"]["access_token"] == "tok"  # noqa: S105


@pytest.mark.parametrize(
    "make,perm", [(_ig, "instagram_manage_comments"), (_fb, "pages_manage_engagement")]
)
@pytest.mark.parametrize("code", [10, 200])
def test_adapter_maps_permission_errors(make, perm, code):
    session = _Session(_Resp(403, {"error": {"code": code, "message": "no perm"}}))
    with pytest.raises(CommentPermissionError) as info:
        make(session).post_comment("OBJ9", "hello")
    assert perm in str(info.value)


@pytest.mark.parametrize("make", [_ig, _fb])
def test_adapter_other_graph_errors_are_not_permission_errors(make):
    session = _Session(_Resp(400, {"error": {"code": 100, "message": "bad"}}))
    with pytest.raises(Exception) as info:
        make(session).post_comment("OBJ9", "hello")
    assert not isinstance(info.value, CommentPermissionError)


# --- never a second copy ----------------------------------------------------
#
# Facebook once created a delivery's first comment but its
# answer missed the 30 s read timeout, so the row recorded an error and the
# next pass would have posted the same comment again. Before every post the
# adapter now reads the post's comments and keeps one already carrying the
# text, so a retry can never double-post.

_META = [(_ig, "text"), (_fb, "message")]  # where each platform keeps a comment's text


@pytest.mark.parametrize("make,field", _META)
def test_adapter_never_posts_a_comment_that_is_already_there(make, field):
    session = _Session(_Resp(200, {"id": "new"}), _comments(field, "nice drive", _TEXT))
    assert make(session).post_comment("OBJ9", _TEXT) == "c1"
    assert session.calls == []
    url, kwargs = session.gets[0]
    assert url.endswith("/OBJ9/comments")
    assert field in kwargs["params"]["fields"].split(",")
    assert "access_token" not in kwargs["params"]
    assert kwargs["headers"]["Authorization"] == "OAuth tok"


@pytest.mark.parametrize("make,field", _META)
def test_only_the_exact_text_counts_as_already_there(make, field):
    # A viewer quoting the link, or another drive's link that starts the
    # same, is not the first comment.
    others = _comments(field, f"{_TEXT} 🔥", _TEXT + "X", "nice drive")
    session = _Session(_Resp(200, {"id": "new"}), others)
    assert make(session).post_comment("OBJ9", _TEXT) == "new"
    assert len(session.calls) == 1


@pytest.mark.parametrize("make,field", _META)
def test_the_check_reads_every_page_of_comments(make, field):
    first, second = _comments(field, "nice drive", after="CUR1"), _comments(field, _TEXT)
    session = _Session(_Resp(200, {"id": "new"}), first, second)
    assert make(session).post_comment("OBJ9", _TEXT) == "c0"
    assert session.calls == []
    assert session.gets[1][1]["params"]["after"] == "CUR1"


def test_the_check_stops_at_its_page_cap_and_then_posts():
    # A short with thousands of comments still gets its link: the scan is
    # bounded, and past the cap the post goes ahead.
    cap = first_comment.COMMENT_CHECK_MAX_PAGES
    pages = [_comments("message", f"viewer {i}", after=f"CUR{i}") for i in range(cap + 3)]
    session = _Session(_Resp(200, {"id": "new"}), *pages)
    assert _fb(session).post_comment("OBJ9", _TEXT) == "new"
    assert len(session.gets) == cap
    assert len(session.calls) == 1


@pytest.mark.parametrize("make", [_ig, _fb])
@pytest.mark.parametrize(
    "answer",
    [_Resp(500, {"error": {"code": 2, "message": "down"}}), requests.ConnectionError("reset")],
    ids=["graph-error", "network-down"],
)
def test_a_failed_check_never_posts(make, answer):
    session = _Session(_Resp(200, {"id": "new"}), answer)
    with pytest.raises(Exception) as info:
        make(session).post_comment("OBJ9", _TEXT)
    assert not isinstance(info.value, CommentPermissionError)
    assert session.calls == []


class _NotJson(_Resp):
    def json(self):
        raise ValueError("Expecting value")


@pytest.mark.parametrize("make", [_ig, _fb])
@pytest.mark.parametrize(
    "answer",
    [_NotJson(200, None), _Resp(200, {}), _Resp(200, {"data": None}), _Resp(200, [])],
    ids=["not-json", "empty-object", "data-null", "not-an-object"],
)
def test_a_check_that_reads_no_comment_list_never_posts(make, answer):
    # A 2xx whose body carries no ``data`` list is a check that did not
    # finish, not "no comments yet": posting on it could double-post.
    session = _Session(_Resp(200, {"id": "new"}), answer)
    with pytest.raises(Exception) as info:
        make(session).post_comment("OBJ9", _TEXT)
    assert not isinstance(info.value, CommentPermissionError)
    assert session.calls == []


@pytest.mark.parametrize("make", [_ig, _fb])
def test_an_empty_comment_list_still_posts(make):
    session = _Session(_Resp(200, {"id": "new"}), _Resp(200, {"data": []}))
    assert make(session).post_comment("OBJ9", _TEXT) == "new"
    assert len(session.calls) == 1


@pytest.mark.parametrize("make", [_ig, _fb])
def test_the_check_keeps_the_token_out_of_the_url(make):
    session = _Session(_Resp(200, {"id": "new"}))
    make(session).post_comment("OBJ9", _TEXT)
    _url, kwargs = session.gets[0]
    assert "access_token" not in kwargs["params"]
    assert kwargs["headers"] == {"Authorization": "OAuth tok"}
    assert kwargs["params"]["limit"] == first_comment._COMMENT_PAGE_SIZE


class _LeakySession(_Session):
    """``requests`` puts ``params`` into the URL and a ConnectionError's
    message embeds that URL: so does this fake's."""

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        query = "&".join(f"{k}={v}" for k, v in (kwargs.get("params") or {}).items())
        raise requests.ConnectionError(f"HTTPSConnectionPool: Max retries: {url}?{query}")


@pytest.mark.parametrize("make,platform", [(_ig, "instagram"), (_fb, "facebook")])
def test_a_network_error_on_the_check_leaves_no_token_in_the_stored_error(conn, make, platform):
    delivery_id = _published(conn, 1, platform, platform_post_id="OBJ1")
    graph = _LeakySession(_Resp(200, {"id": "new"}))
    result = drain_meta_first_comments(
        conn, adapter_factories={platform: lambda: make(graph)}, now=_NOW
    )
    assert result.failed == 1
    assert graph.calls == []
    stored = _row(conn, delivery_id)["first_comment_error"]
    assert "HTTPSConnectionPool" in stored  # the leak path was exercised
    assert "tok" not in stored
    assert all("tok" not in line for line in result.errors)


@pytest.mark.parametrize(
    "make,perm", [(_ig, "instagram_manage_comments"), (_fb, "pages_manage_engagement")]
)
def test_a_check_refused_for_permission_names_it_and_never_posts(make, perm):
    session = _Session(
        _Resp(200, {"id": "new"}), _Resp(403, {"error": {"code": 10, "message": "no"}})
    )
    with pytest.raises(CommentPermissionError) as info:
        make(session).post_comment("OBJ9", _TEXT)
    assert perm in str(info.value)
    assert session.calls == []


class _SlowGraph:
    """Facebook as it behaved for delivery 15: every comment it is sent is
    created, but the answer never arrives inside the read timeout."""

    def __init__(self):
        self.posts = 0
        self.comments = []

    def post(self, url, **kwargs):
        self.posts += 1
        self.comments.append({"id": f"C{self.posts}", "message": kwargs["data"]["message"]})
        raise requests.ReadTimeout("Read timed out. (read timeout=30)")

    def get(self, url, **kwargs):
        return _Resp(200, {"data": list(self.comments)})


def test_a_comment_that_landed_but_timed_out_is_never_posted_twice(conn):
    delivery_id = _published(conn, 1, "facebook", platform_post_id="REEL1")
    graph = _SlowGraph()
    factories = {"facebook": lambda: _fb(graph)}
    first = drain_meta_first_comments(conn, adapter_factories=factories, now=_NOW)
    assert first.failed == 1
    assert _row(conn, delivery_id)["first_comment_posted_at"] is None
    second = drain_meta_first_comments(conn, adapter_factories=factories, now=_NOW)
    assert second.posted == 1 and second.failed == 0
    assert graph.posts == 1  # one comment on the reel, ever
    row = _row(conn, delivery_id)
    assert row["first_comment_posted_at"] is not None
    assert row["first_comment_error"] is None
    assert drain_meta_first_comments(conn, adapter_factories=factories, now=_NOW).posted == 0


def test_module_exports():
    assert first_comment.full_drive_comment_text is full_drive_comment_text
    assert json  # sidecars are plain JSON


# --- dispatch-queue wiring --------------------------------------------------


def _dispatch(tmp_path, capsys, *, yt_result=None, meta_result=None, **routes):
    from gomoveshift.cli.handlers.dispatch_queue import cmd_dispatch_queue
    from gomoveshift.publishing.dispatch import DrainResult
    from test_cli_dispatch_queue import _args, _config, _record as _rec, _sidecar

    if yt_result is None:
        yt_result = FirstCommentResult()
    if meta_result is None:
        meta_result = FirstCommentResult(posted=2, failed=1)

    c = open_queue_db(tmp_path)
    enqueue(c, "proj", 3, "instagram", "HASH1")
    c.close()
    _sidecar(tmp_path, _rec())
    base = "gomoveshift.cli.handlers.dispatch_queue."
    with (
        mock.patch(base + "_build_routed_handlers"),
        mock.patch(base + "drain_queue", return_value=DrainResult()),
        mock.patch(base + "backfill_first_comment_text") as backfill,
        mock.patch(base + "drain_first_comments", return_value=yt_result),
        mock.patch(base + "drain_meta_first_comments", return_value=meta_result) as meta,
    ):
        rc = cmd_dispatch_queue(_args(apply=True), _config(tmp_path, **routes))
    assert rc == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    return lines, backfill, meta


def test_dispatch_runs_backfill_and_meta_drain_when_routed(tmp_path, capsys):
    lines, backfill, meta = _dispatch(tmp_path, capsys, instagram="first_party")
    backfill.assert_called_once()
    meta.assert_called_once()
    assert set(meta.call_args.kwargs["adapter_factories"]) == {"instagram"}
    assert "first_comment: posted=2 failed=1" in lines
    assert lines[-1].startswith("attempted=")


def test_dispatch_skips_meta_comments_when_not_routed(tmp_path, capsys):
    lines, backfill, meta = _dispatch(tmp_path, capsys, tiktok="first_party")
    backfill.assert_not_called()
    meta.assert_not_called()
    assert lines[-1].startswith("attempted=")


def test_first_comment_counts_ride_the_summary_line(tmp_path, capsys):
    """The cron wrapper logs only a pass's last line, so the ``first_comment:``
    line above the summary never reaches the log: a pass that posted comments
    and a pass that had none due read the same there. The counts go ON the
    summary line."""
    lines, _, _ = _dispatch(tmp_path, capsys, instagram="first_party")
    assert lines[-1].startswith("attempted=")
    assert "first_comment_posted=2 first_comment_failed=1" in lines[-1]


def test_the_summary_adds_up_youtube_and_meta_first_comments(tmp_path, capsys):
    lines, _, _ = _dispatch(
        tmp_path,
        capsys,
        yt_result=FirstCommentResult(posted=1, failed=2),
        youtube="first_party",
        instagram="first_party",
    )
    assert "first_comment_posted=3 first_comment_failed=3" in lines[-1]


@pytest.mark.parametrize(
    ("yt_result", "meta_result", "who"),
    [
        (FirstCommentResult(skipped_scope_error=1), None, "youtube"),
        (None, FirstCommentResult(skipped_scope_error=2), "meta"),
        (
            FirstCommentResult(skipped_scope_error=1),
            FirstCommentResult(skipped_scope_error=1),
            "youtube,meta",
        ),
    ],
    ids=["youtube", "meta", "both"],
)
def test_the_summary_names_the_app_to_reauthorise(tmp_path, capsys, yt_result, meta_result, who):
    """A missing comment scope holds every first comment on that app until the operator
    re-authorises it, so the one line the log keeps says which app."""
    lines, _, _ = _dispatch(
        tmp_path,
        capsys,
        yt_result=yt_result,
        meta_result=meta_result,
        youtube="first_party",
        instagram="first_party",
    )
    assert f"first_comment_needs_reauth={who}" in lines[-1].split()


def test_the_summary_stays_quiet_when_no_first_comment_was_due(tmp_path, capsys):
    """Only when non-zero, like ``throttled``: a field that reads ``=0`` on
    every quiet pass is a field nobody reads on the pass that matters."""
    lines, _, _ = _dispatch(
        tmp_path, capsys, meta_result=FirstCommentResult(), instagram="first_party"
    )
    assert "first_comment" not in lines[-1]
