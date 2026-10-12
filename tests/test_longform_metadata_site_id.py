"""LongFormMetadataGenerator must tag SEO events with the configured site id,
not a hardcoded 'gomoveshift' — a non-default brand's events must not be
tagged with another brand's site id (they'd be filtered out of its own
ledger reads).
"""

from unittest.mock import MagicMock, patch

from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator


class TestSiteIdThreading:
    def test_default_site_id_is_gomoveshift(self, tmp_path):
        with patch("seo_manager.SEOManager") as mock_manager_cls:
            mock_manager_cls.return_value.store.query_by_site.return_value = []
            gen = LongFormMetadataGenerator(
                seo_storage_path=tmp_path / "seo",
                credentials_dir=tmp_path / "creds",
            )

        assert gen.site_id == "gomoveshift"
        _, kwargs = mock_manager_cls.call_args
        assert kwargs["site_id"] == "gomoveshift"
        mock_manager_cls.return_value.store.query_by_site.assert_called_with("gomoveshift")

    def test_configured_site_id_is_used_for_manager_init_and_query(self, tmp_path):
        with patch("seo_manager.SEOManager") as mock_manager_cls:
            mock_manager_cls.return_value.store.query_by_site.return_value = []
            gen = LongFormMetadataGenerator(
                seo_storage_path=tmp_path / "seo",
                credentials_dir=tmp_path / "creds",
                site_id="example-site",
            )

        assert gen.site_id == "example-site"
        _, kwargs = mock_manager_cls.call_args
        assert kwargs["site_id"] == "example-site"
        mock_manager_cls.return_value.store.query_by_site.assert_called_with("example-site")

    def test_batch_sync_stats_tags_events_with_configured_site_id(self, tmp_path):
        with patch("seo_manager.SEOManager") as mock_manager_cls:
            mock_manager_cls.return_value.store.query_by_site.return_value = []
            gen = LongFormMetadataGenerator(
                seo_storage_path=tmp_path / "seo",
                credentials_dir=tmp_path / "creds",
                site_id="example-site",
            )

        sync = MagicMock()
        sync.youtube_data.videos.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "vid1",
                    "statistics": {"viewCount": "10"},
                    "snippet": {"title": "t"},
                }
            ]
        }
        gen._is_analytics_eligible = MagicMock(return_value=False)
        gen._batch_sync_stats(sync, {"vid1": "content1"})

        upserted_event = mock_manager_cls.return_value.store.upsert_event.call_args[0][0]
        assert upserted_event.site_id == "example-site"
