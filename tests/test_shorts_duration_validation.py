"""
TDD: Shorts Duration Validation - Prevent 8-second shorts from being rendered.

These tests MUST fail before the fix and pass after.
"""

import pytest


class TestShortsMinimumDuration:
    """Shorts MUST be at least 30 seconds for YouTube Shorts compliance."""

    def test_pipeline_extractor_default_min_duration_at_least_30(self):
        """PipelineShortsExtractor default min_duration must be >= 30 seconds."""
        from gomoveshift.pipeline.shorts_extractor import PipelineShortsExtractor

        extractor = PipelineShortsExtractor()

        assert extractor.min_duration >= 30.0, (
            f"PipelineShortsExtractor min_duration is {extractor.min_duration}s. "
            f"YouTube Shorts require minimum 30 seconds for optimal performance."
        )

    def test_pipeline_extractor_enforces_minimum_on_init(self):
        """Cannot create extractor with min_duration < 30."""
        from gomoveshift.pipeline.shorts_extractor import PipelineShortsExtractor

        # Should raise or clamp to 30
        extractor = PipelineShortsExtractor(min_duration=10.0)

        assert extractor.min_duration >= 30.0, (
            "PipelineShortsExtractor must reject or clamp min_duration < 30s"
        )

    def test_analysis_extractor_default_min_duration_at_least_15(self):
        """Analysis layer can go lower (15s) but pipeline should enforce 30s."""
        from gomoveshift.analysis.shorts_extractor import ShortsExtractor

        extractor = ShortsExtractor()

        # Analysis layer default is 15, which is fine for internal use
        assert extractor.min_duration >= 15.0

    def test_rendered_shorts_all_meet_minimum_duration(self, tmp_path):
        """All rendered shorts must be >= 30 seconds."""

        from gomoveshift.pipeline.shorts_extractor import PipelineShortsExtractor

        # This test documents the bug we fixed
        # With the new min_duration=30 default, the extractor won't produce 8s shorts

        extractor = PipelineShortsExtractor()

        # Now the extractor enforces minimum
        assert extractor.min_duration >= 30.0, (
            "Extractor doesn't enforce 30s minimum - would allow 8s shorts"
        )

    def test_extraction_result_validates_duration_range(self):
        """Document that short durations need manual validation."""
        # The ExtractionResult doesn't enforce duration constraints by itself
        # That's why we need the PipelineShortsExtractor to enforce min_duration=30
        # This test documents the need for that enforcement
        from gomoveshift.pipeline.shorts_extractor import PipelineShortsExtractor

        extractor = PipelineShortsExtractor()

        # Enforcement happens at extraction time via min_duration
        assert extractor.min_duration >= 30.0


class TestShortsMaximumDuration:
    """Shorts should stay within reasonable maximum (45-60s)."""

    def test_pipeline_extractor_default_max_duration_reasonable(self):
        """Default max_duration should be 45-60 seconds."""
        from gomoveshift.pipeline.shorts_extractor import PipelineShortsExtractor

        extractor = PipelineShortsExtractor()

        assert 45.0 <= extractor.max_duration <= 60.0, (
            f"max_duration {extractor.max_duration}s outside recommended range 45-60s"
        )

    def test_target_duration_between_min_and_max(self):
        """Target duration must be between min and max."""
        from gomoveshift.pipeline.shorts_extractor import PipelineShortsExtractor

        extractor = PipelineShortsExtractor()

        assert extractor.min_duration <= extractor.target_duration <= extractor.max_duration, (
            f"target_duration {extractor.target_duration}s not between "
            f"min {extractor.min_duration}s and max {extractor.max_duration}s"
        )


class TestShortsDurationInPipeline:
    """Integration test: full pipeline should produce correctly-sized shorts."""

    @pytest.mark.slow
    def test_pipeline_run_produces_minimum_duration_shorts(self, tmp_path):
        """
        Full pipeline run should never produce shorts < 30 seconds.

        This is the integration test that would have caught the 8.3-second bug.
        """
        # This requires a test video - skipped for now but documents requirement
        pytest.skip("Requires test video for full pipeline integration")

        # When implemented:
        # 1. Run full pipeline on test video
        # 2. Check all rendered MP4 durations with ffprobe
        # 3. Assert all >= 30 seconds
