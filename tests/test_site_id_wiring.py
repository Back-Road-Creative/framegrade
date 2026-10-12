"""Writers must resolve site_id from config.seo.site_id when the caller
doesn't pass one explicitly -- callers cannot forget to thread it through.
"""

from unittest.mock import MagicMock, patch

from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator
from gomoveshift.pipeline.validation.content_gates import check_youtube_best_practices


def _config_with_site_id(site_id):
    config = MagicMock()
    config.seo.site_id = site_id
    return config


def _metadata_json():
    return {
        "content_id": "abc123",
        "title": "Some Title",
        "description": "Some description",
        "tags": ["tag1"],
    }


class TestLongFormMetadataGeneratorSiteIdFromConfig:
    def test_no_site_id_resolves_from_config(self, tmp_path):
        with (
            patch(
                "gomoveshift.pipeline.longform_metadata._get_config",
                return_value=_config_with_site_id("waypost"),
            ),
            patch("seo_manager.SEOManager") as mock_manager_cls,
        ):
            mock_manager_cls.return_value.store.query_by_site.return_value = []
            gen = LongFormMetadataGenerator(
                seo_storage_path=tmp_path / "seo",
                credentials_dir=tmp_path / "creds",
            )

        assert gen.site_id == "waypost"
        _, kwargs = mock_manager_cls.call_args
        assert kwargs["site_id"] == "waypost"

    def test_explicit_site_id_wins_over_config(self, tmp_path):
        with (
            patch(
                "gomoveshift.pipeline.longform_metadata._get_config",
                return_value=_config_with_site_id("waypost"),
            ),
            patch("seo_manager.SEOManager") as mock_manager_cls,
        ):
            mock_manager_cls.return_value.store.query_by_site.return_value = []
            gen = LongFormMetadataGenerator(
                seo_storage_path=tmp_path / "seo",
                credentials_dir=tmp_path / "creds",
                site_id="other",
            )

        assert gen.site_id == "other"
        _, kwargs = mock_manager_cls.call_args
        assert kwargs["site_id"] == "other"


class TestCheckYoutubeBestPracticesSiteIdFromConfig:
    def test_no_site_id_resolves_from_config(self):
        with (
            patch(
                "gomoveshift.pipeline.validation.content_gates._get_config",
                return_value=_config_with_site_id("waypost"),
            ),
            patch("seo_manager.core.metadata.SEOMetadata") as mock_seo_metadata,
            patch(
                "seo_manager.validation.youtube_best_practices.YouTubeBestPracticesValidator"
            ) as mock_validator_cls,
        ):
            mock_validator_cls.return_value.validate.return_value = []
            check_youtube_best_practices(_metadata_json())

        assert mock_seo_metadata.call_args.kwargs["site_id"] == "waypost"
