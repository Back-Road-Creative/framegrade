"""Tests for the YouTube Shorts publishing adapter. Filename-guard
coverage lives in ``tests/test_youtube_uploader.py``. Mocks googleapiclient
exactly as that file does — no network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from test_publishing_models import _record as _base_record

from gomoveshift.config import Config, DriveConfig
from gomoveshift.pipeline.drive_phases import youtube_uploader as yu
from gomoveshift.pipeline.exceptions import YouTubeQuotaError
from gomoveshift.publishing.queue import (
    acquire_lease,
    enqueue,
    find_delivery,
    open_queue_db,
    transition,
)
from gomoveshift.publishing.youtube_adapter import (
    AmbiguousUploadMatchError,
    YouTubeShortsAdapter,
)


@pytest.fixture(autouse=True)
def _publication_gate_open(monkeypatch):
    # The publication gate holds every insert until the final text is bound;
    # these upload tests predate the gate, so they open it in-test only.
    monkeypatch.setattr(
        "gomoveshift.pipeline.drive_phases.youtube_uploader.FINAL_TEXT_BIND_ON", True
    )


#: Names the base record's sampled region, so the Short review agrees.
SHORT_TITLE = "County Kerry drive"


@pytest.fixture
def mock_service():
    svc = MagicMock()
    insert_request = MagicMock()
    insert_request.next_chunk.side_effect = [(None, {"id": "SHORTVID123"})]
    svc.videos.return_value.insert.return_value = insert_request
    svc.playlists.return_value.list.return_value.execute.return_value = {"items": []}
    svc.playlists.return_value.insert.return_value.execute.return_value = {"id": "PLAYLIST123"}
    svc.playlistItems.return_value.insert.return_value.execute.return_value = {}
    # Channel-scan fallback (dup-guard retry path) — default to an empty
    # channel so pre-existing tests that never exercise the scan still see
    # no candidates and fall through to a normal insert.
    svc.channels.return_value.list.return_value.execute.return_value = {
        "items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UU_FAKE"}}}]
    }
    svc.playlistItems.return_value.list.return_value.execute.return_value = {"items": []}
    return svc


def _video_item(video_id: str, title: str, *, age: timedelta = timedelta(minutes=5)) -> dict:
    published = (datetime.now(timezone.utc) - age).isoformat().replace("+00:00", "Z")
    return {"id": video_id, "snippet": {"title": title, "publishedAt": published}}


def _seed_channel_scan(mock_service, *video_ids: str) -> None:
    mock_service.playlistItems.return_value.list.return_value.execute.return_value = {
        "items": [{"contentDetails": {"videoId": vid}} for vid in video_ids]
    }


@pytest.fixture
def queue_conn(tmp_path):
    conn = open_queue_db(tmp_path / "seo")
    try:
        yield conn
    finally:
        conn.close()


def _setup(mock_service, queue_conn, tmp_path, *, cfg=None):
    creds = {
        "refresh_token": "1//fake",
        "client_id": "x",
        "client_secret": "y",
    }  # pragma: allowlist secret
    with (
        patch.object(yu, "load_youtube_credentials", return_value=creds),
        patch.object(yu, "_build_credentials_obj", return_value=MagicMock()),
        patch.object(yu, "build", return_value=mock_service),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
    ):
        up = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))
        up._build_service()
    adapter = YouTubeShortsAdapter(up, cfg or Config())
    record = _titled(_base_record(routes=("youtube",)), SHORT_TITLE)
    delivery_id = enqueue(queue_conn, record.project_name, record.short_index, "youtube", "hash")
    video_path = tmp_path / "short_00.mp4"
    video_path.write_bytes(b"\x00\x00\x00\x20ftypiso5")
    return adapter, record, delivery_id, video_path


def _titled(record, title):
    import dataclasses

    yt = dataclasses.replace(record.payloads.youtube, title=title)
    return dataclasses.replace(record, payloads=dataclasses.replace(record.payloads, youtube=yt))


def _publish(adapter, record, queue_conn, delivery_id, video_path):
    with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
        return adapter.publish(
            record, conn=queue_conn, delivery_id=delivery_id, video_path=video_path
        )


def test_publish_sets_publish_at_private_and_orders_id_before_playlist(
    mock_service, queue_conn, tmp_path
):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    order = []

    def _fake_insert(*args, **kwargs):
        row = find_delivery(queue_conn, record.project_name, record.short_index, "youtube")
        order.append(row["platform_upload_id"])
        return MagicMock(execute=MagicMock(return_value={}))

    mock_service.playlistItems.return_value.insert.side_effect = _fake_insert
    _publish(adapter, record, queue_conn, delivery_id, video_path)

    body = mock_service.videos.return_value.insert.call_args.kwargs["body"]
    assert body["status"]["privacyStatus"] == "private"
    assert body["status"]["publishAt"] == "2026-08-24T14:00:00Z"
    assert order == ["SHORTVID123"]  # video_id already persisted when playlist insert ran
    row = find_delivery(queue_conn, record.project_name, record.short_index, "youtube")
    # The insert succeeded synchronously, but it set publishAt + private just
    # above: YouTube ACCEPTED this video and publishes it later. That is
    # "scheduled", not "confirmed within seconds" -- and the receipt records
    # the same instant the body carries, so the two can never disagree.
    assert row["published_at_source"] == "platform_scheduled"
    assert row["platform_published_at"] == "2026-08-24T14:00:00+00:00"


def test_playlist_failure_does_not_lose_video_id(mock_service, queue_conn, tmp_path):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    mock_service.playlists.return_value.list.return_value.execute.side_effect = RuntimeError("boom")

    result = _publish(adapter, record, queue_conn, delivery_id, video_path)

    assert result.video_id == "SHORTVID123"
    assert result.playlist_id is None
    row = find_delivery(queue_conn, record.project_name, record.short_index, "youtube")
    assert row["platform_upload_id"] == "SHORTVID123"


def test_daily_budget_defers_rather_than_fails(mock_service, queue_conn, tmp_path):
    cfg = Config(drive=DriveConfig(youtube_video_insert_daily_uploads=1))
    prior_id = enqueue(queue_conn, "other-proj", 99, "youtube", "hash-prior")
    transition(queue_conn, prior_id, "queued", "uploaded", platform_upload_id="ALREADYUP")

    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path, cfg=cfg)
    result = _publish(adapter, record, queue_conn, delivery_id, video_path)

    assert result.deferred is True
    mock_service.videos.return_value.insert.assert_not_called()
    row = find_delivery(queue_conn, record.project_name, record.short_index, "youtube")
    assert row["status"] == "deferred"
    assert row["next_attempt_at"] is not None
    assert row["next_attempt_reason"] == "platform_deferral"


def test_lost_response_reconciles_and_does_not_double_insert(mock_service, queue_conn, tmp_path):
    """A retry/duplicate-dispatch of an already-completed delivery must not
    re-issue ``videos.insert`` — it reconciles the persisted id via
    ``videos.list`` and adopts it."""
    mock_service.videos.return_value.list.return_value.execute.return_value = {
        "items": [{"id": "SHORTVID123"}]
    }
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)

    first = _publish(adapter, record, queue_conn, delivery_id, video_path)
    assert first.video_id == "SHORTVID123"

    # Simulate a duplicate dispatch (e.g. caller believed the first call
    # timed out and redispatched the same delivery).
    second = _publish(adapter, record, queue_conn, delivery_id, video_path)

    assert second.video_id == "SHORTVID123"
    assert mock_service.videos.return_value.insert.call_count == 1
    mock_service.videos.return_value.list.assert_called_with(
        part="status,snippet", id="SHORTVID123"
    )


def test_realistic_full_envelope_reconcile_drives_reconcile_and_caller(
    mock_service, queue_conn, tmp_path
):
    """A full-shape ``videos.list`` response body (``kind``/``etag``/
    ``pageInfo`` alongside ``items``, and per-item ``kind``/``etag`` beside
    the fields actually read) drives ``_reconcile`` and its immediate
    caller ``publish`` end to end — the googleapiclient ``.execute()``
    result already matches the documented YouTube Data API v3 shape
    (https://developers.google.com/youtube/v3/docs/videos/list), so this
    pins that ``resp.get("items")`` keeps working when the real payload
    carries the extra top-level/per-item fields this adapter never reads."""
    mock_service.videos.return_value.list.return_value.execute.return_value = {
        "kind": "youtube#videoListResponse",
        "etag": "abc123",
        "pageInfo": {"totalResults": 1, "resultsPerPage": 1},
        "items": [
            {
                "kind": "youtube#video",
                "etag": "def456",
                "id": "SHORTVID123",
                "status": {"privacyStatus": "private", "uploadStatus": "processed"},
                "snippet": {"title": SHORT_TITLE, "publishedAt": "2026-08-23T14:00:00Z"},
            }
        ],
    }
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)

    first = _publish(adapter, record, queue_conn, delivery_id, video_path)
    assert first.video_id == "SHORTVID123"

    second = _publish(adapter, record, queue_conn, delivery_id, video_path)

    assert second.video_id == "SHORTVID123"
    assert mock_service.videos.return_value.insert.call_count == 1


def test_reconciliation_of_nonexistent_video_permits_one_retry(mock_service, queue_conn, tmp_path):
    """A persisted id that ``videos.list`` proves does not exist (e.g. a
    corrupted/stale record) is discarded — exactly one fresh insert is
    permitted, never a silent skip."""
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    transition(queue_conn, delivery_id, "queued", "submitting")
    transition(queue_conn, delivery_id, "submitting", "uploaded", platform_upload_id="BOGUS")
    mock_service.videos.return_value.list.return_value.execute.side_effect = [
        {"items": []},
        {"items": [{"id": "SHORTVID123"}]},  # the post-upload verify sees it
    ]

    result = _publish(adapter, record, queue_conn, delivery_id, video_path)

    assert result.video_id == "SHORTVID123"
    assert mock_service.videos.return_value.insert.call_count == 1


def test_killed_worker_lease_is_reclaimable(mock_service, queue_conn, tmp_path):
    mock_service.videos.return_value.list.return_value.execute.return_value = {
        "items": [{"id": "SHORTVID123"}]
    }
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    transition(queue_conn, delivery_id, "queued", "submitting")
    # A dead worker's lease, long expired — never got as far as an id.
    acquire_lease(queue_conn, delivery_id, "dead-worker", "2020-01-01T00:00:00+00:00")

    result = _publish(adapter, record, queue_conn, delivery_id, video_path)

    assert result.video_id == "SHORTVID123"
    assert mock_service.videos.return_value.insert.call_count == 1


def test_live_lease_refuses_a_racing_second_insert(mock_service, queue_conn, tmp_path):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    transition(queue_conn, delivery_id, "queued", "submitting")
    acquire_lease(
        queue_conn,
        delivery_id,
        "live-worker",
        "2999-01-01T00:00:00+00:00",
        now="2020-01-02T00:00:00+00:00",
    )

    with pytest.raises(RuntimeError):
        _publish(adapter, record, queue_conn, delivery_id, video_path)
    mock_service.videos.return_value.insert.assert_not_called()


def test_quota_error_lands_in_retryable_state(mock_service, queue_conn, tmp_path):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    quota_exc = YouTubeQuotaError(reason="quotaExceeded")
    with patch.object(adapter._uploader, "upload_video", side_effect=quota_exc):
        with pytest.raises(YouTubeQuotaError):
            _publish(adapter, record, queue_conn, delivery_id, video_path)

    row = find_delivery(queue_conn, record.project_name, record.short_index, "youtube")
    assert row["status"] == "retry_wait"
    assert row["error_class"] == "quota"


def test_first_clean_publish_never_scans_channel(mock_service, queue_conn, tmp_path):
    """A first, clean attempt (``queued``, nothing ever submitted) must
    never pay for a channel scan — only a retry after a lost response does."""
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)

    _publish(adapter, record, queue_conn, delivery_id, video_path)

    mock_service.playlistItems.return_value.list.assert_not_called()
    mock_service.channels.return_value.list.assert_not_called()
    assert mock_service.videos.return_value.insert.call_count == 1


def test_lost_response_with_nothing_persisted_adopts_via_channel_scan(
    mock_service, queue_conn, tmp_path
):
    """The insert succeeded on YouTube's side but its response was lost
    before anything was persisted (delivery stuck in ``submitting`` with no
    id). The retry must find the matching upload by title+recency and adopt
    it rather than inserting a second time."""
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    transition(queue_conn, delivery_id, "queued", "submitting")
    _seed_channel_scan(mock_service, "ADOPTEDVID")
    mock_service.videos.return_value.list.return_value.execute.return_value = {
        "items": [_video_item("ADOPTEDVID", SHORT_TITLE)]
    }

    result = _publish(adapter, record, queue_conn, delivery_id, video_path)

    assert result.video_id == "ADOPTEDVID"
    mock_service.videos.return_value.insert.assert_not_called()
    row = find_delivery(queue_conn, record.project_name, record.short_index, "youtube")
    assert row["platform_upload_id"] == "ADOPTEDVID"
    assert row["status"] == "published"


def test_lost_response_across_two_attempts_inserts_exactly_once(mock_service, queue_conn, tmp_path):
    """End-to-end: attempt 1's response is lost after the create actually
    happened; attempt 2 (the retry) must adopt it via the scan instead of
    re-inserting — ``videos.insert`` fires exactly once total."""
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    insert_request = mock_service.videos.return_value.insert.return_value
    insert_request.next_chunk.side_effect = TimeoutError("response lost")

    with patch.object(yu.time, "sleep"):
        with pytest.raises(TimeoutError):
            _publish(adapter, record, queue_conn, delivery_id, video_path)
    row = find_delivery(queue_conn, record.project_name, record.short_index, "youtube")
    assert row["status"] == "submitting"
    assert row["platform_upload_id"] is None
    assert mock_service.videos.return_value.insert.call_count == 1

    _seed_channel_scan(mock_service, "ADOPTEDVID")
    mock_service.videos.return_value.list.return_value.execute.return_value = {
        "items": [_video_item("ADOPTEDVID", SHORT_TITLE)]
    }

    result = _publish(adapter, record, queue_conn, delivery_id, video_path)

    assert result.video_id == "ADOPTEDVID"
    assert mock_service.videos.return_value.insert.call_count == 1


def test_channel_scan_finds_no_match_permits_one_fresh_insert(mock_service, queue_conn, tmp_path):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    transition(queue_conn, delivery_id, "queued", "submitting")
    _seed_channel_scan(mock_service, "UNRELATED")
    mock_service.videos.return_value.list.return_value.execute.return_value = {
        "items": [_video_item("UNRELATED", "Some Other Short")]
    }

    result = _publish(adapter, record, queue_conn, delivery_id, video_path)

    assert result.video_id == "SHORTVID123"
    assert mock_service.videos.return_value.insert.call_count == 1


def test_channel_scan_two_matches_raises_rather_than_guessing(mock_service, queue_conn, tmp_path):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    transition(queue_conn, delivery_id, "queued", "submitting")
    _seed_channel_scan(mock_service, "DUP1", "DUP2")
    mock_service.videos.return_value.list.return_value.execute.return_value = {
        "items": [_video_item("DUP1", SHORT_TITLE), _video_item("DUP2", SHORT_TITLE)]
    }

    with pytest.raises(AmbiguousUploadMatchError):
        _publish(adapter, record, queue_conn, delivery_id, video_path)
    mock_service.videos.return_value.insert.assert_not_called()


def test_channel_scan_ignores_same_title_outside_recency_window(mock_service, queue_conn, tmp_path):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    transition(queue_conn, delivery_id, "queued", "submitting")
    _seed_channel_scan(mock_service, "OLDVID")
    mock_service.videos.return_value.list.return_value.execute.return_value = {
        "items": [_video_item("OLDVID", SHORT_TITLE, age=timedelta(days=3))]
    }

    result = _publish(adapter, record, queue_conn, delivery_id, video_path)

    assert result.video_id == "SHORTVID123"
    assert mock_service.videos.return_value.insert.call_count == 1


# ── Shorts playlist routing ────────────────────────────────────
# ``sampled_regions[0]`` is a PLACE label, not a country. It used to be passed
# as ``region`` and matched no rule, so shorts landed in "Unsorted (fallback)".


def _playlist_titles_created(mock_service) -> list[str]:
    return [
        c.kwargs["body"]["snippet"]["title"]
        for c in mock_service.playlists.return_value.insert.call_args_list
    ]


def _with_regions(record, *regions):
    import dataclasses

    record = dataclasses.replace(record, sampled_regions=tuple(regions))
    # The title names the clip's own place, as the Short review requires.
    return _titled(record, f"{regions[0].partition(',')[0]} drive") if regions else record


@pytest.mark.parametrize(
    ("place", "playlist"),
    [
        ("West Cork", "Ireland Shorts"),
        ("County Kerry, Munster", "Ireland Shorts"),
        ("Unicoi, Tennessee", "USA Shorts"),
        ("Example Town, Tennessee", "USA Shorts"),
        ("Unicoi, TN", "USA Shorts"),
    ],
)
def test_short_routes_to_country_shorts_playlist(
    mock_service, queue_conn, tmp_path, place, playlist
):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    record = _with_regions(record, place)

    result = _publish(adapter, record, queue_conn, delivery_id, video_path)

    assert _playlist_titles_created(mock_service) == [playlist]
    assert result.playlist_id == "PLAYLIST123"
    mock_service.playlistItems.return_value.insert.assert_called_once()


def test_short_with_underivable_country_is_not_filed_in_unsorted(
    mock_service, queue_conn, tmp_path, caplog
):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    record = _with_regions(record, "Springfield")

    with caplog.at_level("WARNING", logger="gomoveshift.publishing.youtube_adapter"):
        result = _publish(adapter, record, queue_conn, delivery_id, video_path)

    assert _playlist_titles_created(mock_service) == []
    mock_service.playlistItems.return_value.insert.assert_not_called()
    assert result.video_id == "SHORTVID123"
    assert result.playlist_id is None
    assert any("SHORTVID123" in r.getMessage() for r in caplog.records)
    # Terminal, not parked in ``uploaded``: a resume must not retry forever.
    row = find_delivery(queue_conn, record.project_name, record.short_index, "youtube")
    assert row["status"] == "published"
    assert row["platform_upload_id"] == "SHORTVID123"
    assert row["platform_post_id"] is None


def test_short_without_sampled_regions_is_not_filed_in_unsorted(mock_service, queue_conn, tmp_path):
    adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
    record = _with_regions(record)

    result = _publish(adapter, record, queue_conn, delivery_id, video_path)

    assert result.playlist_id is None
    mock_service.playlistItems.return_value.insert.assert_not_called()
    row = find_delivery(queue_conn, record.project_name, record.short_index, "youtube")
    assert row["status"] == "published"
