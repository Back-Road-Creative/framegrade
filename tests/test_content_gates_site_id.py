"""check_youtube_best_practices must tag SEOMetadata with the configured
site id, not a hardcoded 'gomoveshift' — a non-default brand's metadata
must not be tagged with another brand's site id.
"""

from unittest.mock import patch

from gomoveshift.pipeline.validation.content_gates import check_youtube_best_practices


def _metadata_json():
    return {
        "content_id": "abc123",
        "title": "Some Title",
        "description": "Some description",
        "tags": ["tag1"],
    }


def test_default_site_id_is_gomoveshift():
    with (
        patch("seo_manager.core.metadata.SEOMetadata") as mock_seo_metadata,
        patch(
            "seo_manager.validation.youtube_best_practices.YouTubeBestPracticesValidator"
        ) as mock_validator_cls,
    ):
        mock_validator_cls.return_value.validate.return_value = []
        check_youtube_best_practices(_metadata_json())

    assert mock_seo_metadata.call_args.kwargs["site_id"] == "gomoveshift"


def test_configured_site_id_is_used():
    with (
        patch("seo_manager.core.metadata.SEOMetadata") as mock_seo_metadata,
        patch(
            "seo_manager.validation.youtube_best_practices.YouTubeBestPracticesValidator"
        ) as mock_validator_cls,
    ):
        mock_validator_cls.return_value.validate.return_value = []
        check_youtube_best_practices(_metadata_json(), site_id="examplebrand")

    assert mock_seo_metadata.call_args.kwargs["site_id"] == "examplebrand"
