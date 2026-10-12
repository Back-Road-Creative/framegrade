"""Tests for recording date extraction and display across the pipeline.

Covers:
- ChapterGenerator.recording_date property
- Orchestrator._extract_recording_date() helper
- Recording date in longform YouTube description
- Recording date in shorts captions (within 280-char limit)
- Weather detection wiring to social generator
- Weather results persistence
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock


from gomoveshift.config import Config
from gomoveshift.pipeline.chapters import ChapterGenerator
from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator
from gomoveshift.pipeline.social_generator import SocialGenerator
from gomoveshift.pipeline.weather import WeatherCondition, WeatherResult


# ── ChapterGenerator.recording_date ──────────────────────


class TestChapterRecordingDate:
    """ChapterGenerator exposes recording_date after generate()."""

    def test_recording_date_none_before_generate(self):
        config = Config()
        gen = ChapterGenerator(config)
        assert gen.recording_date is None

    def test_recording_date_set_after_generate(self):
        """Property is set from journey.start_date after generate()."""
        config = Config()
        gen = ChapterGenerator(config)

        # Build a mock Journey with start/end dates
        mock_journey = MagicMock()
        mock_journey.start_date = datetime(2025, 9, 10, 14, 30, 0)
        mock_journey.end_date = datetime(2025, 9, 10, 16, 0, 0)
        mock_journey.waypoints = []

        chapters = gen.generate(
            gpx_path=Path("/nonexistent.gpx"),
            video_duration=3600.0,
            start_location="Cahir, County Tipperary",
            _mock_journey=mock_journey,
        )

        assert gen.recording_date == datetime(2025, 9, 10, 14, 30, 0)
        assert len(chapters) >= 2  # At least start + end

    def test_recording_date_none_when_no_gpx(self):
        """Property stays None when GPX file doesn't exist."""
        config = Config()
        gen = ChapterGenerator(config)

        gen.generate(
            gpx_path=Path("/nonexistent.gpx"),
            video_duration=3600.0,
            start_location="Test Location",
        )

        assert gen.recording_date is None


# ── _extract_recording_date ──────────────────────


class TestExtractRecordingDate:
    """Orchestrator._extract_recording_date priority chain."""

    def test_recording_date_from_project_name_yyyymmdd(self):
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator()
        dt = orch._extract_recording_date(
            gpx_date=None,
            ffprobe_creation_time=None,
            project_name="20250910example",
        )
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 9
        assert dt.day == 10

    def test_recording_date_from_project_name_yyyy_mm_dd(self):
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator()
        dt = orch._extract_recording_date(
            gpx_date=None,
            ffprobe_creation_time=None,
            project_name="2025-09-10_example",
        )
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 9
        assert dt.day == 10

    def test_gpx_date_takes_priority(self):
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator()
        gpx_date = datetime(2025, 9, 10, 14, 0, 0)
        dt = orch._extract_recording_date(
            gpx_date=gpx_date,
            ffprobe_creation_time="2024-01-01T00:00:00Z",
            project_name="20240101test",
        )
        assert dt == gpx_date

    def test_ffprobe_creation_time_second_priority(self):
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator()
        dt = orch._extract_recording_date(
            gpx_date=None,
            ffprobe_creation_time="2025-09-10T14:30:00.000000Z",
            project_name="exampleroute",
        )
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 9

    def test_returns_none_when_no_date_found(self):
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator()
        dt = orch._extract_recording_date(
            gpx_date=None,
            ffprobe_creation_time=None,
            project_name="exampleroute",
        )
        assert dt is None


# ── Recording date in longform description ──────────────────────


class TestRecordingDateInDescription:
    """Longform YouTube description includes recording date."""

    def test_recording_date_in_description(self):
        gen = LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)
        gen.storage_path = Path("/tmp/test-seo")
        gen.credentials_dir = Path("/tmp/test-creds")
        gen._manager = None
        gen._optimizer = None
        gen._ledger = None

        desc = gen._build_gms_description(
            location="County Kerry, Ireland",
            seo_description="A beautiful drive through Kerry!",
            duration_str="1h 23m",
            publish_day="Saturday",
            publish_date="2026-02-28",
            recording_date="September 10, 2025",
        )

        # Long-form now routes through the shared "Mon YYYY" formatter.
        assert "📅 Filmed: Sep 2025" in desc
        # Publish date was intentionally removed (ages poorly, clutters)
        assert "📅 Published:" not in desc

    def test_description_without_recording_date(self):
        """Description still works when no recording date provided."""
        gen = LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)
        gen.storage_path = Path("/tmp/test-seo")
        gen.credentials_dir = Path("/tmp/test-creds")
        gen._manager = None
        gen._optimizer = None
        gen._ledger = None

        desc = gen._build_gms_description(
            location="County Kerry, Ireland",
            seo_description="A beautiful drive through Kerry!",
            duration_str="1h 23m",
            publish_day="Saturday",
            publish_date="2026-02-28",
        )

        assert "📅 Filmed:" not in desc
        assert "📹 Recorded:" not in desc
        # Publish date was intentionally removed (ages poorly, clutters)
        assert "📅 Published:" not in desc


# ── Recording date in shorts captions ──────────────────────


class TestRecordingDateInCaption:
    """Social generator captions include recording date."""

    def test_recording_date_in_caption(self):
        gen = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="exampleroute",
            recording_date="Sep 2025",
        )
        post = gen.generate_post(1)
        assert "📹 Sep 2025" in post.caption

    def test_caption_under_280_with_date(self):
        gen = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="exampleroute",
            recording_date="Sep 2025",
        )
        for i in range(1, 4):
            post = gen.generate_post(i)
            assert len(post.caption) <= 280, (
                f"Short #{i} caption is {len(post.caption)} chars (max 280)"
            )

    def test_caption_without_recording_date(self):
        """Captions still work without recording_date."""
        gen = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="exampleroute",
        )
        post = gen.generate_post(1)
        assert "📹" not in post.caption
        assert len(post.caption) > 0


# ── Weather detection wiring ──────────────────────


class TestWeatherWiring:
    """Weather detection wired to social generator via orchestrator."""

    def test_social_gen_receives_weather_results(self):
        """When weather_results are passed, captions get weather emoji."""
        gen = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="testproject",
            series_part=1,
        )
        weather = WeatherResult(
            condition=WeatherCondition.SUNNY,
            brightness=620.0,
            confidence=0.8,
            segment_start=0.0,
            segment_end=60.0,
        )
        posts = gen.generate_all(num_shorts=3, weather_results=[weather, weather, weather])
        # At least one caption should have the sunny emoji
        has_sun = any("☀️" in p.caption for p in posts)
        assert has_sun, "No sunny emoji found in any caption"

    def test_social_gen_backward_compatible_without_weather(self):
        """Still works without weather_results (None)."""
        gen = SocialGenerator(
            location="County Kerry, Ireland",
            project_name="testproject",
            series_part=1,
        )
        posts = gen.generate_all(num_shorts=3)
        assert len(posts) == 3
        for p in posts:
            assert len(p.caption) > 0

    def test_weather_results_serialization(self):
        """Weather results can be serialized to/from JSON for resume."""
        weather = WeatherResult(
            condition=WeatherCondition.RAINY,
            brightness=350.0,
            confidence=0.7,
            segment_start=10.0,
            segment_end=70.0,
        )
        d = weather.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["condition"] == "rainy"
        assert loaded["brightness"] == 350.0
        assert loaded["emoji"] == "🌧️"
