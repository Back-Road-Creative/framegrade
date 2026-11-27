"""A TikTok chunk that fails is retried, never failed for good.

Before, a failed chunk PUT left the row ``submitting`` with its publish id;
the drain's retry then read that as a worker that died mid-upload and marked
the row ``failed_permanent``, so one dropped connection lost the post. Now a
failure that proves the upload incomplete drops the id and puts the row back to
``queued``, and the next try starts over with a fresh init."""

from __future__ import annotations

import dataclasses
from unittest import mock

import pytest
import requests
from test_publishing_models import _record

import gomoveshift.publishing.handlers as handlers_mod
from gomoveshift.publishing.dispatch import drain_queue
from gomoveshift.publishing.handlers import TikTokCredentials, build_handlers
from gomoveshift.publishing.queue import enqueue, open_queue_db
from gomoveshift.publishing.tiktok_adapter import (
    ChunkPlan,
    PublishOutcome,
    TikTokChunkUploadError,
    upload_video_chunks,
)

URL = "https://open-upload.tiktokapis.com/u/1"
NOW = "2026-11-26T12:00:00+00:00"
AFTER_BACKOFF = "2026-11-26T13:00:00+00:00"


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _put(status: int | None):
    if status is None:
        return requests.ConnectionError("reset by peer")
    resp = mock.Mock(status_code=status, content=b"", text="")
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(str(status))
    return resp


@pytest.mark.parametrize(
    "answers, incomplete",
    [
        ([None], True),  # the first of two chunks never answered: the second was never sent
        ([200, 500], True),  # the last chunk refused outright
        ([200, None], False),  # the last chunk unanswered: it may all be there
    ],
)
def test_a_failed_chunk_says_whether_the_upload_is_provably_incomplete(answers, incomplete):
    plan = ChunkPlan(video_size=10, chunk_size=5, total_chunk_count=2)
    with mock.patch(
        "gomoveshift.publishing.tiktok_adapter.requests.put", side_effect=[_put(a) for a in answers]
    ):
        with pytest.raises(TikTokChunkUploadError) as exc_info:
            upload_video_chunks(URL, plan, lambda i: b"x" * 5)
    assert exc_info.value.upload_incomplete is incomplete


def test_a_failed_chunk_is_retried_with_a_fresh_upload_and_publishes(conn, monkeypatch):
    plan = ChunkPlan(video_size=11, chunk_size=11, total_chunk_count=1)
    inits, uploads = [], []

    def init(*a, **k):
        inits.append(1)
        return {"publish_id": f"pub-{len(inits)}", "upload_url": URL, "chunk_plan": plan}

    def upload(*a, **k):
        uploads.append(1)
        if len(uploads) == 1:
            raise TikTokChunkUploadError("chunk upload request failed", upload_incomplete=True)

    monkeypatch.setenv("GMS_TIKTOK_MODE", "direct")
    monkeypatch.setattr(handlers_mod, "query_creator_info", lambda token: {})
    monkeypatch.setattr(handlers_mod, "init_direct_post", init)
    monkeypatch.setattr(handlers_mod, "upload_video_chunks", upload)
    monkeypatch.setattr(
        handlers_mod,
        "confirm_publish",
        lambda token, pid, level: PublishOutcome(publish_id=pid, delivery_status="published"),
    )
    record = _record()
    did = enqueue(conn, record.project_name, record.short_index, "tiktok", "hash-a")
    creds = TikTokCredentials(access_token="act.fake", audit_approved=True)  # noqa: S106
    handlers = build_handlers(
        conn,
        lambda p, i: dataclasses.replace(record, project_name=p, short_index=i),
        tiktok=(creds, lambda r: b"x" * 11),
    )

    first = drain_queue(conn, handlers, now=NOW)

    assert first.failed_permanent == 0 and first.retried == 1
    row = conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone()
    assert row["status"] == "queued"
    assert row["platform_upload_id"] is None  # the broken upload is dropped
    assert row["attempt_count"] == 1  # an ordinary, counted retry

    second = drain_queue(conn, handlers, now=AFTER_BACKOFF)

    assert second.succeeded == 1 and second.failed_permanent == 0
    row = conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone()
    assert row["status"] == "published"
    assert row["platform_post_id"] == "pub-2"  # a fresh init, not the dead upload
    assert len(inits) == 2
