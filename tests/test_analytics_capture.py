"""
TDD tests: _batch_sync_stats also pulls YouTube Analytics API v2
metrics (views/CTR/watch-time proxies), not just the Data API statistics.

Before this change the SEO ledger's performance_updated events carried only
views/likes/comments/favorites — grep -c '"ctr"' on the ledger was 0 because
YouTubeAnalyticsSync.fetch_video_analytics was never reached from the GMS
batch sync path.
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator


def _make_yt_api_response(video_ids, published_at=None):
    items = []
    for vid in video_ids:
        items.append(
            {
                "id": vid,
                "snippet": {
                    "title": f"Video {vid}",
                    "publishedAt": published_at or "2026-01-01T00:00:00Z",
                },
                "statistics": {
                    "viewCount": "100",
                    "likeCount": "5",
                    "commentCount": "2",
                    "favoriteCount": "0",
                },
            }
        )
    return {"items": items}


def _make_analytics_response():
    return {
        "columnHeaders": [
            {"name": "views"},
            {"name": "estimatedMinutesWatched"},
            {"name": "averageViewDuration"},
            {"name": "averageViewPercentage"},
            {"name": "cardImpressions"},
            {"name": "cardClickRate"},
            {"name": "subscribersGained"},
        ],
        "rows": [[100, 500, 65.2, 42.5, 20, 0.05, 3]],
    }


@pytest.fixture
def generator_with_store():
    from seo_manager import SEOManager
    from seo_manager.platforms.youtube import YouTubeAdapter

    tmp = tempfile.mkdtemp()
    storage_path = Path(tmp) / "seo"

    with patch.object(LongFormMetadataGenerator, "_init_seo_manager"):
        with patch.object(LongFormMetadataGenerator, "_init_ledger"):
            gen = LongFormMetadataGenerator(seo_storage_path=storage_path)

    gen._manager = SEOManager(
        site_id="gomoveshift",
        platform=YouTubeAdapter(),
        storage_path=storage_path,
    )

    return gen, storage_path


def _events_for(gen):
    from seo_manager.core.events import EventType

    return gen._manager.store.query_by_site(
        "gomoveshift",
        event_type=EventType.PERFORMANCE_UPDATED,
    )


class TestAnalyticsCapture:
    def test_analytics_fields_and_published_at_recorded(self, generator_with_store):
        gen, _ = generator_with_store
        video_mapping = {"vid_A": "content-a"}

        mock_sync = MagicMock()
        mock_sync.youtube_data.videos().list().execute.return_value = _make_yt_api_response(
            ["vid_A"]
        )
        mock_sync.youtube_analytics.reports().query().execute.return_value = (
            _make_analytics_response()
        )

        gen._batch_sync_stats(mock_sync, video_mapping)

        events = _events_for(gen)
        assert len(events) == 1
        data = events[0].data
        assert data["views"] == 100
        assert data["published_at"] == "2026-01-01T00:00:00Z"
        assert data["estimated_minutes_watched"] == 500
        assert data["average_view_duration"] == 65.2
        assert data["average_view_percentage"] == 42.5
        assert data["card_impressions"] == 20
        assert data["ctr"] == 0.05
        assert data["subscribers_gained"] == 3

    def test_analytics_failure_does_not_block_data_api_sync(self, generator_with_store):
        gen, _ = generator_with_store
        video_mapping = {"vid_B": "content-b"}

        mock_sync = MagicMock()
        mock_sync.youtube_data.videos().list().execute.return_value = _make_yt_api_response(
            ["vid_B"]
        )
        mock_sync.youtube_analytics.reports.side_effect = Exception("403 quota exceeded")

        gen._batch_sync_stats(mock_sync, video_mapping)

        events = _events_for(gen)
        assert len(events) == 1
        data = events[0].data
        assert data["views"] == 100
        assert data["likes"] == 5
        assert "estimated_minutes_watched" not in data
        assert "ctr" not in data

    def test_quota_cap_skips_analytics_beyond_newest_200_and_older_than_90_days(
        self, generator_with_store
    ):
        gen, _ = generator_with_store
        old_published = (datetime.now(timezone.utc) - timedelta(days=400)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        with patch.object(LongFormMetadataGenerator, "_ANALYTICS_MAX_VIDEOS", 1):
            video_mapping = {"vid_new": "content-new", "vid_old": "content-old"}

            mock_sync = MagicMock()

            def fake_list(*args, **kwargs):
                ids = kwargs["id"].split(",")
                mock = MagicMock()
                mock.execute.return_value = _make_yt_api_response(ids, published_at=old_published)
                return mock

            mock_sync.youtube_data.videos().list.side_effect = fake_list
            mock_sync.youtube_analytics.reports().query().execute.return_value = (
                _make_analytics_response()
            )

            gen._batch_sync_stats(mock_sync, video_mapping)

            events = {e.data["video_id"]: e.data for e in _events_for(gen)}
            assert "estimated_minutes_watched" in events["vid_new"]
            assert "estimated_minutes_watched" not in events["vid_old"]
