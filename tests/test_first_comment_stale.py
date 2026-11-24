"""A first comment found live with a replaced text is counted, filed as a health incident and
corrected in place by the next YouTube drain -- never silent, never needing a human step."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from gomoveshift.publishing import meta_comments, youtube_comments
from gomoveshift.publishing.first_comment import PriorCommentLiveError
from gomoveshift.publishing.queue import (
    PUBLISHED_AT_PLATFORM_CONFIRMED as PLATFORM,
    enqueue,
    open_queue_db,
    record_stale_live_comment,
    rewrite_unposted_first_comment,
    transition,
)
from gomoveshift.publishing.youtube_comments import (
    drain_first_comments,
    post_first_comment,
)

OLD_TEXT = "Watch the full drive: https://youtu.be/old"
NEW_TEXT = "Watch the full drive: https://youtu.be/new"
_PUB_AT = "2026-08-23T00:00:00+00:00"
_NOW = "2026-08-23T00:06:00+00:00"


def first_comment_corrections(row):
    from gomoveshift.publishing import queue  # noqa: PLC0415

    return queue.first_comment_corrections(row)


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


def _published(conn, platform="youtube", idx=1):
    did = enqueue(conn, "proj", idx, platform, f"h{idx}", first_comment_text=OLD_TEXT)
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


def _incidents(inc):
    return [json.loads(p.read_text()) for p in sorted(inc.glob("*.json"))] if inc.is_dir() else []


OWN = "UC-own"  # the authenticated channel's id; a viewer's comment names another


def _thread(text, tid="t-old", author=OWN):
    snippet = {"textOriginal": text, "authorChannelId": {"value": author}}
    return {"id": tid, "snippet": {"topLevelComment": {"snippet": snippet}}}


class _Req:
    def __init__(self, value=None, exc=None):
        self.value, self.exc = value, exc

    def execute(self):
        if self.exc:
            raise self.exc
        return self.value


class _Comments:
    def __init__(self, fail):
        self.updates, self.deletes, self.fail = [], [], list(fail)

    def _go(self):
        return _Req(exc=self.fail.pop(0) if self.fail else None)

    def update(self, part, body):
        self.updates.append((body["id"], body["snippet"]["textOriginal"]))
        return self._go()

    def delete(self, id):  # noqa: A002 -- the API's own name
        self.deletes.append(id)
        return self._go()


class _Service:
    def __init__(self, pages=(), on_insert=None, fail=None):
        self.pages, self.inserted, self.on_insert = list(pages), [], on_insert
        self.comments_api, self.lists = _Comments(fail or []), []

    def commentThreads(self):
        return self

    def comments(self):
        return self.comments_api

    def channels(self):
        return SimpleNamespace(list=lambda **kw: _Req({"items": [{"id": OWN}]}))

    def list(self, **kw):
        self.lists.append(kw)
        return _Req(self.pages.pop(0) if self.pages else {"items": []})

    def insert(self, part, body):
        self.inserted.append(body["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
        if self.on_insert:
            self.on_insert()
        return _Req({"id": "new-thread"})


def _drain(conn, service, inc):
    return drain_first_comments(conn, service_factory=lambda: service, now=_NOW, incidents_dir=inc)


# --- YouTube ---------------------------------------------------------------------------------


def test_a_stale_comment_is_counted_filed_and_then_corrected_in_place(conn, inc):
    did = _published(conn)
    first = _Service(on_insert=lambda: _rewrite(conn, did))
    result = _drain(conn, first, inc)
    assert result.stale == 1 and result.corrected == 0
    (incident,) = _incidents(inc)
    assert incident["kind"] == "first_comment_stale" and incident["delivery_id"] == did
    assert incident["platform"] == "youtube" and not incident.get("resolved_at")
    assert incident["live_comments"][0]["live_id"] == "new-thread"
    assert "links old" in incident["error"]
    assert str(did) in incident["detail"] and "new-thread" in incident["detail"]

    again = _Service()
    fixed = _drain(conn, again, inc)
    assert again.comments_api.updates == [("new-thread", NEW_TEXT)]
    assert again.inserted == [] and fixed.corrected == 1 and fixed.stale == 0
    row = _row(conn, did)
    assert row["first_comment_text"] == NEW_TEXT and row["first_comment_error"] is None
    assert first_comment_corrections(row) == []
    (incident,) = _incidents(inc)
    assert incident["resolved_at"]  # cleared itself


def test_a_failed_correction_keeps_the_incident_and_retries_next_drain(conn, inc):
    did = _published(conn)
    _drain(conn, _Service(on_insert=lambda: _rewrite(conn, did)), inc)
    boom = _Service(fail=[TimeoutError("read timed out")])
    bad = _drain(conn, boom, inc)
    assert bad.stale == 1 and bad.corrected == 0
    assert any("not corrected" in e and "read timed out" in e for e in bad.errors)
    row = _row(conn, did)
    assert row["first_comment_attempts"] == 0  # an outage never counts toward the cap
    assert "correction failed" in row["first_comment_error"]
    (incident,) = _incidents(inc)
    assert not incident.get("resolved_at") and "read timed out" in incident["error"]
    ok = _Service()
    assert _drain(conn, ok, inc).corrected == 1
    assert _incidents(inc)[0]["resolved_at"]


def test_a_platform_refusal_of_the_correction_counts_toward_the_cap(conn, inc):
    from tests.test_youtube_first_comment import _make_http_error  # noqa: PLC0415

    did = _published(conn)
    _drain(conn, _Service(on_insert=lambda: _rewrite(conn, did)), inc)
    refused = _Service(fail=[_make_http_error(400, "invalidValue")])
    _drain(conn, refused, inc)
    row = _row(conn, did)
    assert row["first_comment_correction_attempts"] == 1
    assert row["first_comment_attempts"] == 0  # the posting counter is not shared


def test_both_texts_live_deletes_the_old_one_and_never_the_new_one(conn, inc):
    did = _published(conn)
    _rewrite(conn, did)
    pages = [{"items": [_thread(OLD_TEXT, "t-old"), _thread(NEW_TEXT, "t-new")]}]
    service = _Service(pages=pages)
    result = _drain(conn, service, inc)
    assert service.inserted == [] and result.stale == 1
    row = _row(conn, did)
    assert row["first_comment_text"] == NEW_TEXT and row["first_comment_posted_at"]
    assert [c["action"] for c in first_comment_corrections(row)] == ["delete"]
    assert _incidents(inc) and not _incidents(inc)[0].get("resolved_at")
    nxt = _Service()
    assert _drain(conn, nxt, inc).corrected == 1
    assert nxt.comments_api.deletes == ["t-old"] and nxt.comments_api.updates == []
    assert _incidents(inc)[0]["resolved_at"]


def test_the_scan_reads_every_bounded_page_and_prefers_the_current_text(conn):
    page1 = {"items": [_thread(NEW_TEXT, "t-new")], "nextPageToken": "p2"}
    page2 = {"items": [_thread(OLD_TEXT, "t-old")]}
    service = _Service(pages=[page1, page2])
    current, prior = youtube_comments._scan_first_comments(
        service, "vid", NEW_TEXT, also_accept=(OLD_TEXT,)
    )
    assert current == [("t-new", NEW_TEXT)] and prior == [("t-old", OLD_TEXT)]
    assert len(service.lists) == 2
    # without a prior text the scan still stops at the first match
    quick = _Service(pages=[page1, page2])
    youtube_comments._scan_first_comments(quick, "vid", NEW_TEXT)
    assert len(quick.lists) == 1


def test_a_concurrent_drain_that_posted_other_text_is_still_reported(conn, inc):
    # The row is already posted (another drain finished it) when this drain's comment lands.
    did = _published(conn)
    conn.execute("UPDATE deliveries SET first_comment_posted_at=? WHERE id=?", (_NOW, did))
    conn.commit()
    result = youtube_comments.FirstCommentResult()
    youtube_comments.record_live_comment(conn, did, "someone else's text", "c-late", result, inc)
    assert result.stale == 1 and result.errors
    (item,) = first_comment_corrections(_row(conn, did))
    assert item["action"] == "delete" and item["live_id"] == "c-late"
    assert _incidents(inc)
    # reported once: the same live comment again changes nothing
    assert record_stale_live_comment(conn, did, "someone else's text", _NOW, "c-late") is None
    assert len(first_comment_corrections(_row(conn, did))) == 1


def test_a_posted_row_whose_live_text_is_its_own_is_not_a_stale_report(conn):
    did = _published(conn)
    conn.execute("UPDATE deliveries SET first_comment_posted_at=? WHERE id=?", (_NOW, did))
    conn.commit()
    assert record_stale_live_comment(conn, did, OLD_TEXT, _NOW) is None
    assert record_stale_live_comment(conn, 99999, OLD_TEXT, _NOW, "c") is None  # row gone


def test_an_unposted_row_records_the_live_text_and_keeps_the_intended_one(conn):
    did = _published(conn)
    _rewrite(conn, did)
    err = record_stale_live_comment(conn, did, OLD_TEXT, _NOW, "c1")
    assert err == "stale comment live: links old"
    row = _row(conn, did)
    assert row["first_comment_text"] == OLD_TEXT and row["first_comment_posted_at"] == _NOW
    (item,) = first_comment_corrections(row)
    assert item == {"live_id": "c1", "live_text": OLD_TEXT, "action": "edit", "to": NEW_TEXT}


def test_the_stale_error_names_only_ids_the_current_text_dropped():
    from gomoveshift.publishing.queue import stale_live_error

    live = "a https://youtu.be/old and https://youtu.be/keep"
    current = "a https://youtu.be/new and https://youtu.be/keep"
    assert stale_live_error(live, current) == "stale comment live: links old"


def test_a_paused_platform_is_not_touched(conn, inc, monkeypatch):
    did = _published(conn)
    _drain(conn, _Service(on_insert=lambda: _rewrite(conn, did)), inc)
    monkeypatch.setattr(
        "gomoveshift.publishing.queue.is_platform_paused", lambda conn, platform: True
    )
    service = _Service()
    result = _drain(conn, service, inc)
    assert service.comments_api.updates == [] and result.corrected == 0


def test_an_incident_with_no_correction_left_resolves_itself(conn, inc):
    did = _published(conn)
    _drain(conn, _Service(on_insert=lambda: _rewrite(conn, did)), inc)
    with conn:
        conn.execute("UPDATE deliveries SET first_comment_correction=NULL WHERE id=?", (did,))
    _drain(conn, _Service(), inc)
    assert _incidents(inc)[0]["resolved_at"]


def test_the_incident_shows_in_the_status_snapshot(conn, tmp_path, monkeypatch):
    from gomoveshift.status import snapshot

    monkeypatch.setenv("GMS_HEALTH_INCIDENTS_DIR", str(tmp_path / "snap"))
    did = _published(conn)
    youtube_comments.drain_first_comments(
        conn, service_factory=lambda: _Service(on_insert=lambda: _rewrite(conn, did)), now=_NOW
    )
    rows = snapshot._incidents()["open"]
    assert [r["kind"] for r in rows] == ["first_comment_stale"]


def test_the_summary_last_line_carries_the_stale_and_corrected_counts(tmp_path, capsys):
    from gomoveshift.cli.handlers.dispatch_queue import cmd_dispatch_queue
    from gomoveshift.publishing.dispatch import DrainResult
    from tests.test_cli_dispatch_queue import _args, _config, _record, _sidecar  # noqa: PLC0415

    c = open_queue_db(tmp_path)
    enqueue(c, "proj", 3, "tiktok", "HASH1")
    c.close()
    _sidecar(tmp_path, _record())
    fake = youtube_comments.FirstCommentResult(posted=1, stale=2, corrected=1, errors=["x: y"])
    with (
        mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers"),
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue.drain_queue",
            return_value=DrainResult(attempted=1, succeeded=1),
        ),
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue.drain_first_comments", return_value=fake
        ),
    ):
        rc = cmd_dispatch_queue(_args(apply=True), _config(tmp_path, youtube="first_party"))
    assert rc == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert "first_comment_stale=2" in lines[-1] and "first_comment_corrected=1" in lines[-1]
    assert "  error: x: y" in lines[:-1]


# --- Meta ------------------------------------------------------------------------------------


class _Meta:
    def __init__(self, raise_prior=None):
        self.raise_prior = raise_prior

    def post_comment(self, object_id, text, also_accept=()):
        if self.raise_prior:
            raise self.raise_prior
        return "c-new"


def _meta_drain(conn, platform, adapter, inc):
    return meta_comments.drain_meta_first_comments(
        conn, adapter_factories={platform: lambda: adapter}, now=_NOW, incidents_dir=inc
    )


def test_facebook_a_prior_text_live_is_counted_and_filed(conn, inc):
    did = _published(conn, platform="facebook")
    _rewrite(conn, did)
    adapter = _Meta(raise_prior=PriorCommentLiveError("c9", OLD_TEXT))
    result = _meta_drain(conn, "facebook", adapter, inc)
    assert result.stale == 1 and result.posted == 1
    assert _incidents(inc)[0]["platform"] == "facebook"
    assert not _incidents(inc)[0].get("resolved_at")
    (item,) = first_comment_corrections(_row(conn, did))
    assert item["action"] == "edit" and item["live_id"] == "c9" and item["to"] == NEW_TEXT


def test_meta_a_rewrite_during_the_post_keeps_the_returned_id_for_the_edit(conn, inc):
    did = _published(conn, platform="facebook")
    adapter = _Meta()
    adapter.post_comment = lambda o, t, also_accept=(): (_rewrite(conn, did), "c-live")[1]
    result = _meta_drain(conn, "facebook", adapter, inc)
    assert result.stale == 1
    (item,) = first_comment_corrections(_row(conn, did))
    assert item["live_id"] == "c-live" and item["to"] == NEW_TEXT
