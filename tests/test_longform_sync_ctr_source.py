"""The LF analytics sync labels every ``ctr`` it writes with its source.

``_batch_sync_stats`` stored YouTube's ``cardClickRate`` as ``ctr`` with no
``ctr_source``. Waypost's readers normalise an absent label to ``unknown``
(seo_manager.analytics.metric_source), so every GMS row was indistinguishable
from a thumbnail impression CTR of unknown origin. A card
click rate is in-player clicks among viewers already watching; it must say so.
"""

from unittest.mock import MagicMock, patch

from seo_manager.analytics import metric_source

from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator


def _sync_one(tmp_path, analytics):
    with patch("seo_manager.SEOManager") as mock_manager_cls:
        mock_manager_cls.return_value.store.query_by_site.return_value = []
        gen = LongFormMetadataGenerator(
            seo_storage_path=tmp_path / "seo",
            credentials_dir=tmp_path / "creds",
        )
    sync = MagicMock()
    sync.youtube_data.videos.return_value.list.return_value.execute.return_value = {
        "items": [{"id": "vid1", "statistics": {"viewCount": "10"}, "snippet": {"title": "t"}}]
    }
    gen._is_analytics_eligible = MagicMock(return_value=True)
    gen._fetch_video_analytics_metrics = MagicMock(return_value=analytics)
    gen._batch_sync_stats(sync, {"vid1": "content1"})
    return mock_manager_cls.return_value.store.upsert_event.call_args[0][0].data


def test_a_card_click_rate_is_written_as_one(tmp_path):
    data = _sync_one(tmp_path, {"cardClickRate": 0.012, "cardImpressions": 400})
    assert data["ctr"] == 0.012
    assert data["ctr_source"] == metric_source.YOUTUBE_CARD_CLICK_RATE
    assert not metric_source.is_thumbnail_click_through(data["ctr_source"])


def test_no_ctr_row_is_written_unlabelled(tmp_path):
    data = _sync_one(tmp_path, {"cardImpressions": 0})
    assert data.get("ctr") is None
    assert data["ctr_source"] == metric_source.UNKNOWN
