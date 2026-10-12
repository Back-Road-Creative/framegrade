"""
Tests for event store dedup in _batch_sync_stats.

Running _batch_sync_stats twice with the same video_ids should NOT
create duplicate events. The second sync should update existing
events in-place rather than appending new ones.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator


def _make_yt_api_response(video_ids, view_counts=None):
    """Build a mock YouTube API response for a batch of video IDs."""
    if view_counts is None:
        view_counts = {vid: 100 for vid in video_ids}
    items = []
    for vid in video_ids:
        items.append(
            {
                "id": vid,
                "snippet": {"title": f"Video {vid}"},
                "statistics": {
                    "viewCount": str(view_counts.get(vid, 100)),
                    "likeCount": "5",
                    "commentCount": "2",
                    "favoriteCount": "0",
                },
            }
        )
    return {"items": items}


@pytest.fixture
def generator_with_store():
    """Create generator with real SEO event store in a tmp dir."""
    from seo_manager import SEOManager
    from seo_manager.platforms.youtube import YouTubeAdapter

    tmp = tempfile.mkdtemp()
    storage_path = Path(tmp) / "seo"

    with patch.object(LongFormMetadataGenerator, "_init_seo_manager"):
        with patch.object(LongFormMetadataGenerator, "_init_ledger"):
            gen = LongFormMetadataGenerator(seo_storage_path=storage_path)

    # Wire up a real SEO Manager with a real event store
    gen._manager = SEOManager(
        site_id="gomoveshift",
        platform=YouTubeAdapter(),
        storage_path=storage_path,
    )

    return gen, storage_path


class TestEventDedup:
    """_batch_sync_stats must not create duplicate events on re-sync."""

    def test_sync_twice_no_duplicate_events(self, generator_with_store):
        """Syncing the same 2 videos twice must yield 2 events, not 4."""
        gen, storage_path = generator_with_store

        video_mapping = {"vid_A": "content-a", "vid_B": "content-b"}
        response = _make_yt_api_response(["vid_A", "vid_B"])

        mock_sync = MagicMock()
        mock_sync.youtube_data.videos().list().execute.return_value = response

        # First sync
        gen._batch_sync_stats(mock_sync, video_mapping)
        # Second sync (same videos)
        gen._batch_sync_stats(mock_sync, video_mapping)

        # Count events in store for site "gomoveshift"
        from seo_manager.core.events import EventType

        events = gen._manager.store.query_by_site(
            "gomoveshift",
            event_type=EventType.PERFORMANCE_UPDATED,
        )
        event_ids = [e.event_id for e in events]

        assert len(events) == 2, (
            f"Expected 2 events after two syncs, got {len(events)}. IDs: {event_ids}"
        )

    def test_second_sync_updates_view_count(self, generator_with_store):
        """Second sync with higher views should update the event data."""
        gen, storage_path = generator_with_store

        video_mapping = {"vid_X": "content-x"}

        # First sync: 100 views
        resp1 = _make_yt_api_response(["vid_X"], {"vid_X": 100})
        mock_sync1 = MagicMock()
        mock_sync1.youtube_data.videos().list().execute.return_value = resp1
        gen._batch_sync_stats(mock_sync1, video_mapping)

        # Second sync: 200 views
        resp2 = _make_yt_api_response(["vid_X"], {"vid_X": 200})
        mock_sync2 = MagicMock()
        mock_sync2.youtube_data.videos().list().execute.return_value = resp2
        gen._batch_sync_stats(mock_sync2, video_mapping)

        from seo_manager.core.events import EventType

        events = gen._manager.store.query_by_site(
            "gomoveshift",
            event_type=EventType.PERFORMANCE_UPDATED,
        )
        # Should only have ONE event for vid_X
        vid_x_events = [e for e in events if e.data.get("video_id") == "vid_X"]
        assert len(vid_x_events) == 1, f"Expected 1 event for vid_X, got {len(vid_x_events)}"
        # And it should have the updated view count
        assert vid_x_events[0].data["views"] == 200, (
            f"Expected 200 views after update, got {vid_x_events[0].data['views']}"
        )
