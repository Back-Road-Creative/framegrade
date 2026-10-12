"""Shared uncapped uploads lister -- fakes only, no network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from freezegun import freeze_time

from gomoveshift.pipeline.content_ledger import EST, _list_channel_publish_times
from gomoveshift.publishing.youtube_adapter import YouTubeShortsAdapter
from gomoveshift.publishing.youtube_uploads_listing import list_uploads, parse_iso_duration


def fake_channel(n, *, durations=None, privacy=None, published=None):
    """Service with ``n`` uploads (v000 newest), 50 per playlist page."""
    ids = [f"v{i:03d}" for i in range(n)]
    batches: list[int] = []
    pages: list[int] = []

    def req(result):
        return MagicMock(**{"execute.return_value": result})

    def pl_list(**kw):
        start = int(kw.get("pageToken") or 0)
        chunk = ids[start : start + 50]
        pages.append(len(chunk))
        nxt = str(start + 50) if start + 50 < n else None
        body = {"items": [{"contentDetails": {"videoId": v}} for v in chunk]}
        if nxt:
            body["nextPageToken"] = nxt
        return req(body)

    def vids_list(**kw):
        got = kw["id"].split(",")
        batches.append(len(got))
        items = []
        for v in got:
            items.append(
                {
                    "id": v,
                    "snippet": {
                        "title": f"t-{v}",
                        "publishedAt": (published or {}).get(v, "2026-01-01T00:00:00Z"),
                    },
                    "status": {"privacyStatus": (privacy or {}).get(v, "public")},
                    "contentDetails": {"duration": (durations or {}).get(v, "PT30S")},
                }
            )
        return req({"items": items})

    yt = MagicMock()
    yt.channels.return_value.list.return_value = req(
        {"items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UU1"}}}]}
    )
    yt.playlistItems.return_value.list.side_effect = pl_list
    yt.videos.return_value.list.side_effect = vids_list
    yt.batches, yt.pages = batches, pages
    return yt


def test_list_uploads_pages_past_100():
    yt = fake_channel(230, durations={"v005": "PT1M5S"}, privacy={"v006": "unlisted"})
    ups = list_uploads(yt)
    assert [u.video_id for u in ups] == [f"v{i:03d}" for i in range(230)]
    assert yt.batches == [50, 50, 50, 50, 30]
    by = {u.video_id: u for u in ups}
    assert by["v005"].duration_sec == 65.0
    assert by["v006"].privacy == "unlisted"
    assert by["v000"].published_at == "2026-01-01T00:00:00Z"


def test_list_uploads_limit_bounds_the_walk():
    yt = fake_channel(230)
    assert len(list_uploads(yt, limit=60)) == 60
    assert yt.pages == [50, 50]
    assert yt.batches == [50, 10]


def test_parse_iso_duration():
    assert parse_iso_duration("PT1H2M3S") == 3723
    assert parse_iso_duration("PT45S") == 45
    assert parse_iso_duration("garbage") is None
    assert parse_iso_duration(None) is None


def test_adapter_and_ledger_unchanged_through_shared_lister():
    pub = {"v002": "2026-06-09T10:00:00Z"}
    yt = fake_channel(230, published=pub)
    with patch(
        "gomoveshift.pipeline.content_ledger.list_uploads", wraps=list_uploads
    ) as ledger_lister:
        times = _list_channel_publish_times(yt)
    ledger_lister.assert_called_once()
    assert ledger_lister.call_args.kwargs["limit"] == 100
    assert len(times) == 100
    assert datetime(2026, 6, 9, 10, 0, tzinfo=timezone.utc).astimezone(EST) in times

    moment = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)
    recent = moment - timedelta(minutes=5)
    yt2 = fake_channel(230, published={"v001": recent.isoformat().replace("+00:00", "Z")})
    adapter = YouTubeShortsAdapter.__new__(YouTubeShortsAdapter)
    adapter._uploader = MagicMock()
    adapter._uploader._build_service.return_value = yt2
    payload = MagicMock()
    payload.title = "t-v001"
    with (
        freeze_time(moment),
        patch(
            "gomoveshift.publishing.youtube_adapter.list_uploads", wraps=list_uploads
        ) as adapter_lister,
    ):
        assert adapter._find_existing_upload(payload) == "v001"
    assert adapter_lister.call_args.kwargs["limit"] == 100
