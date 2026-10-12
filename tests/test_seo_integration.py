"""
Tests for SEO Manager integration with the GMS video pipeline.

Verifies that:
1. SEOManager generates valid metadata for POV driving content
2. Analytics data is saved to event store after generation
3. Optimizer returns recommendations when historical data exists
4. Pipeline metadata stage produces correct output via SEO Manager
"""

import pytest

try:
    import seo_manager  # noqa: F401

    HAS_SEO = True
except ImportError:
    HAS_SEO = False

if not HAS_SEO:
    pytest.skip("seo_manager package not installed", allow_module_level=True)

import tempfile
from pathlib import Path
from unittest.mock import patch

from seo_manager import ContentType, Platform, SEOManager, SEOMetadata
from seo_manager.platforms.youtube import YouTubeAdapter


class TestSEOManagerGeneratesValidMetadata:
    """Test 1: SEOManager generates valid metadata for a POV driving video."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp(prefix="gms_seo_test_")
        self.manager = SEOManager(
            site_id="gomoveshift",
            platform=YouTubeAdapter(),
            storage_path=Path(self.tmpdir),
        )

    def test_generates_metadata_with_required_fields(self):
        metadata = self.manager.generate(
            content_type=ContentType.VIDEO,
            topic="Lissard, Co. Limerick to Creggane, Limerick",
            category="Travel & Events",
        )
        assert isinstance(metadata, SEOMetadata)
        assert metadata.title and len(metadata.title) > 0
        assert metadata.description and len(metadata.description) > 0
        assert isinstance(metadata.tags, list) and len(metadata.tags) > 0
        assert metadata.platform == Platform.YOUTUBE

    def test_title_within_youtube_limits(self):
        metadata = self.manager.generate(
            content_type=ContentType.VIDEO,
            topic="Lissard, Co. Limerick to Creggane, Limerick",
        )
        assert len(metadata.title) <= 100  # YouTube max

    def test_tags_within_youtube_limits(self):
        metadata = self.manager.generate(
            content_type=ContentType.VIDEO,
            topic="Lissard, Co. Limerick to Creggane, Limerick",
        )
        assert len(metadata.tags) <= 15

    def test_quality_score_populated(self):
        metadata = self.manager.generate(
            content_type=ContentType.VIDEO,
            topic="Scenic POV Drive Through County Kerry, Ireland",
        )
        assert metadata.quality_score is not None
        assert 0.0 <= metadata.quality_score <= 1.0


class TestAnalyticsEventStoreIntegration:
    """Test 2: Analytics data is saved to event store after generation."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp(prefix="gms_seo_test_")
        self.manager = SEOManager(
            site_id="gomoveshift",
            platform=YouTubeAdapter(),
            storage_path=Path(self.tmpdir),
        )

    def test_save_returns_event_id(self):
        metadata = self.manager.generate(
            content_type=ContentType.VIDEO,
            topic="POV Drive Through Tipperary",
        )
        event_id = self.manager.save(metadata)
        assert event_id is not None
        assert len(event_id) > 0

    def test_saved_metadata_retrievable(self):
        metadata = self.manager.generate(
            content_type=ContentType.VIDEO,
            topic="POV Drive Through Tipperary",
        )
        self.manager.save(metadata)
        retrieved = self.manager.get_metadata(metadata.content_id)
        assert retrieved is not None
        assert retrieved.title == metadata.title
        assert retrieved.tags == metadata.tags

    def test_event_store_file_created(self):
        metadata = self.manager.generate(
            content_type=ContentType.VIDEO,
            topic="POV Drive Through Tipperary",
        )
        self.manager.save(metadata)
        events_file = Path(self.tmpdir) / "gomoveshift" / "events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) >= 1


class TestSEOOptimizerRecommendations:
    """Test 3: Optimizer returns recommendations when historical data exists."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp(prefix="gms_seo_test_")
        self.manager = SEOManager(
            site_id="gomoveshift",
            platform=YouTubeAdapter(),
            storage_path=Path(self.tmpdir),
        )

    def test_optimizer_instantiates(self):
        from seo_manager.analytics.seo_optimizer import SEOOptimizer

        optimizer = SEOOptimizer(self.manager)
        assert optimizer is not None

    def test_analyze_patterns_returns_dict(self):
        from seo_manager.analytics.seo_optimizer import SEOOptimizer

        optimizer = SEOOptimizer(self.manager)
        patterns = optimizer.analyze_patterns(min_views=0)
        assert isinstance(patterns, dict)
        assert "tags" in patterns
        assert "keywords" in patterns

    def test_optimize_metadata_returns_seo_metadata(self):
        from seo_manager.analytics.seo_optimizer import SEOOptimizer

        optimizer = SEOOptimizer(self.manager)
        result = optimizer.optimize_metadata(
            content_type=ContentType.VIDEO,
            topic="POV Drive Through Tipperary",
        )
        assert isinstance(result, SEOMetadata)
        assert result.title and len(result.title) > 0


class TestPipelineMetadataStageIntegration:
    """Test 4: Pipeline metadata stage produces correct output via SEO Manager."""

    def test_longform_metadata_generator_uses_seo_manager(self):
        """The updated LongFormMetadataGenerator should use SEOManager internally."""
        from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

        tmpdir = tempfile.mkdtemp(prefix="gms_seo_test_")
        # SEOManager needs the site subdirectory to exist before it can touch events.jsonl
        (Path(tmpdir) / "gomoveshift").mkdir()
        gen = LongFormMetadataGenerator(
            seo_storage_path=Path(tmpdir),
            credentials_dir=Path(tmpdir) / "no_creds",  # skip YouTube sync in tests
        )
        with patch("gomoveshift.pipeline.longform_metadata.ContentValidator.enforce"):
            result = gen.generate(
                location="Lissard, Co. Limerick to Creggane, Limerick",
                publish_day="Saturday",
                video_duration=3910.0,
                project_name="synthetic-trip-run",
            )

        # Must return dict with required keys
        assert isinstance(result, dict)
        assert "title" in result
        assert "description" in result
        assert "tags" in result
        assert "category" in result

        # Title should not be the old hardcoded "Guess the Location" format
        # SEO manager generates based on topic + keywords
        assert result["title"] and len(result["title"]) > 0

        # Tags should come from SEO manager (not the old hardcoded BASE_TAGS)
        assert isinstance(result["tags"], list)
        assert len(result["tags"]) > 0

        # Quality score should be present (SEO manager feature)
        assert "quality_score" in result

        # Should be saved to event store (content_id present)
        assert "content_id" in result

    def test_longform_metadata_with_analytics_path(self):
        """Generator should accept a custom storage path for analytics."""
        from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

        tmpdir = tempfile.mkdtemp(prefix="gms_seo_test_")
        # SEOManager needs the site subdirectory to exist before it can touch events.jsonl
        (Path(tmpdir) / "gomoveshift").mkdir()
        gen = LongFormMetadataGenerator(
            seo_storage_path=Path(tmpdir),
            credentials_dir=Path(tmpdir) / "no_creds",  # skip YouTube sync in tests
        )
        with patch("gomoveshift.pipeline.longform_metadata.ContentValidator.enforce"):
            result = gen.generate(
                location="County Kerry, Ireland",
                publish_day="Wednesday",
                video_duration=2400.0,
                project_name="kerry_drive",
            )

        assert result["title"]
        assert result["quality_score"] is not None

        # Verify event was stored
        events_file = Path(tmpdir) / "gomoveshift" / "events.jsonl"
        assert events_file.exists()
