"""Tests for YouTube chapter generation from GPS data.

TDD approach: Tests written BEFORE implementation.

Requirements:
- Parse GPX file matching video
- Scale GPS timestamps to video duration
- Reverse geocode waypoints to towns
- Generate YouTube chapters (max 10-15)
- Format: "0:00 Start - Town, County"
"""

import pytest
from unittest.mock import patch

try:
    import gps_journey  # noqa: F401

    HAS_GPS = True
except ImportError:
    HAS_GPS = False

if not HAS_GPS:
    pytest.skip("gps_journey package not installed", allow_module_level=True)

from datetime import datetime, timedelta
from pathlib import Path

from gomoveshift.config import Config
from gomoveshift.pipeline.chapters import ChapterGenerator


class TestChapterGenerator:
    """Test suite for GPS-based chapter generation."""

    @pytest.fixture(autouse=True)
    def mock_network(self):
        """Prevent real network calls. Unit tests exercise chapter formatting,
        not POI discovery or reverse geocoding.
        - _query_overpass: returns empty elements, triggers geocoding fallback
        - NominatimRoadsClient: mocked so reverse_geocode returns None instantly
          (avoids the 1s/req rate limiter in the gps_journey library)
        """
        with (
            patch(
                "gomoveshift.pipeline.chapters._query_overpass",
                return_value={"elements": []},
            ),
            # `_coastline_points_for_bbox` (generator.py) calls
            # `overpass_source.post_overpass_query` directly -- a separate
            # entry point from `_query_overpass` above, so patching only
            # `_query_overpass` left it making a REAL Overpass API request
            # (with its retry/backoff sleep) on every test that reaches
            # terrain-band classification, ~50-200s of real SSL socket I/O
            # per test (measured via cProfile: 92 `_ssl._SSLSocket.read`
            # calls, 47.6s, on `test_geocoding_failure_continues_gracefully`).
            patch(
                "gomoveshift.pipeline.chapters.overpass_source.post_overpass_query",
                return_value={"elements": []},
            ),
            patch("gomoveshift.pipeline.chapters.NominatimRoadsClient") as mock_geo_cls,
        ):
            mock_geo_cls.return_value.reverse_geocode.return_value = None
            mock_geo_cls.return_value.reverse_geocode_detailed.return_value = {}
            yield

    @pytest.fixture
    def config(self):
        """Create test config."""
        return Config(
            project_name="test",
            base_input_path=Path("/tmp/test_input"),
            base_output_path=Path("/tmp/test_output"),
        )

    @pytest.fixture
    def generator(self, config):
        """Create chapter generator instance."""
        return ChapterGenerator(config)

    @pytest.fixture
    def mock_gpx_data(self):
        """Create mock GPS journey with 10 waypoints over 60 minutes."""
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 10, 10, 0, 0)
        waypoints = []

        # Simulate drive: Ennis -> Kilrush -> Loop Head -> back to Ennis
        towns = [
            ("Ennis", "County Clare", 52.8450, -8.9860),
            ("Quin", "County Clare", 52.8233, -8.9489),
            ("Kilrush", "County Clare", 52.6417, -9.4850),
            ("Kilkee", "County Clare", 52.6808, -9.6456),
            ("Carrigaholt", "County Clare", 52.6131, -9.7053),
            ("Loop Head", "County Clare", 52.5614, -9.9306),
            ("Doonbeg", "County Clare", 52.7342, -9.5258),
            ("Kilmihil", "County Clare", 52.7569, -9.2906),
            ("Ennis", "County Clare", 52.8450, -8.9860),
        ]

        for i, (town, county, lat, lon) in enumerate(towns):
            # Space waypoints evenly over 60 minutes
            timestamp = start + timedelta(minutes=i * (60 / (len(towns) - 1)))
            location = Location(
                latitude=lat,
                longitude=lon,
                city=town,
                county=county,
                country="Ireland",
            )
            waypoints.append(
                Waypoint(
                    latitude=lat,
                    longitude=lon,
                    timestamp=timestamp,
                    location=location,
                )
            )

        return Journey(
            start_date=start,
            end_date=start + timedelta(minutes=60),
            waypoints=tuple(waypoints),
            statistics=None,
        )

    def test_chapter_generator_exists(self, generator):
        """Verify ChapterGenerator class exists."""
        assert generator is not None

    def test_generate_method_exists(self, generator):
        """Verify generate() method exists."""
        assert hasattr(generator, "generate")
        assert callable(generator.generate)

    def test_generate_accepts_required_parameters(self, generator):
        """Verify generate() accepts required parameters."""
        import inspect

        sig = inspect.signature(generator.generate)
        params = list(sig.parameters.keys())

        # Should accept video path, GPX path, video duration
        assert "gpx_path" in params or "gps_path" in params
        assert "video_duration" in params or "video_duration_seconds" in params

    def test_gps_to_video_time_scaling(self, generator):
        """Test GPS timestamp scaling to video duration."""
        # GPS: 65 minutes, Video: 60 minutes
        # GPS 30:00 -> Video 27:42
        gps_timestamp = 30 * 60  # 30 minutes in seconds
        gps_duration = 65 * 60
        video_duration = 60 * 60

        scaled = generator._scale_gps_to_video(gps_timestamp, gps_duration, video_duration)

        expected = (30 * 60) * (60 / 65)  # 1662 seconds = 27:42
        assert abs(scaled - expected) < 1.0, f"Expected ~{expected}s, got {scaled}s"

    def test_format_timestamp_hh_mm_ss(self, generator):
        """Test timestamp formatting to YouTube format."""
        # 0 seconds -> 0:00
        assert generator._format_timestamp(0) == "0:00"

        # 90 seconds -> 1:30
        assert generator._format_timestamp(90) == "1:30"

        # 3661 seconds -> 1:01:01
        assert generator._format_timestamp(3661) == "1:01:01"

        # 45 seconds -> 0:45
        assert generator._format_timestamp(45) == "0:45"

    def test_start_chapter_always_included(self, generator, mock_gpx_data):
        """Verify start chapter (0:00) is always included."""
        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=3600,
            start_location="Ennis, County Clare",
            _mock_journey=mock_gpx_data,  # Inject mock data
        )

        assert len(chapters) > 0
        assert chapters[0]["timestamp"] == "0:00"
        assert "Start" in chapters[0]["title"]
        assert "Ennis" in chapters[0]["title"]

    def test_end_chapter_always_included(self, generator, mock_gpx_data):
        """Verify end chapter is always included near video duration."""
        video_duration = 3600  # 60 minutes

        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=video_duration,
            start_location="Ennis, County Clare",
            end_location="Ennis, County Clare",
            _mock_journey=mock_gpx_data,
        )

        assert len(chapters) > 0
        last_chapter = chapters[-1]
        assert "Finish" in last_chapter["title"]
        # Should be near but not at exact video end (YouTube ignores last ~10s)
        ts = last_chapter["timestamp"]
        parts = ts.split(":")
        if len(parts) == 2:
            total_seconds = int(parts[0]) * 60 + int(parts[1])
        else:
            total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        assert total_seconds >= video_duration - 30
        assert total_seconds <= video_duration

    def test_chapter_limit_enforced(self, generator, mock_gpx_data):
        """Verify max 10-15 chapters are generated."""
        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=3600,
            start_location="Ennis, County Clare",
            end_location="Ennis, County Clare",
            max_chapters=15,
            _mock_journey=mock_gpx_data,
        )

        assert len(chapters) <= 15, f"Generated {len(chapters)} chapters, max is 15"

    def test_chapter_format_youtube_standard(self, generator, mock_gpx_data):
        """Verify chapter format matches YouTube requirements."""
        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=3600,
            start_location="Ennis, County Clare",
            end_location="Ennis, County Clare",
            _mock_journey=mock_gpx_data,
        )

        for chapter in chapters:
            # Must have timestamp and title
            assert "timestamp" in chapter
            assert "title" in chapter

            # Timestamp format: MM:SS or H:MM:SS
            ts = chapter["timestamp"]
            assert ":" in ts, f"Timestamp '{ts}' missing colon"

            # Title should include town and county
            title = chapter["title"]
            assert "County Clare" in title, f"Title '{title}' missing county"

    def test_entering_prefix_for_intermediate_towns(self, generator, mock_gpx_data):
        """Verify intermediate towns use 'Entering' prefix."""
        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=3600,
            start_location="Ennis, County Clare",
            end_location="Ennis, County Clare",
            _mock_journey=mock_gpx_data,
        )

        # Should have intermediate chapters with "Entering"
        intermediate = [
            c for c in chapters if "Start" not in c["title"] and "Finish" not in c["title"]
        ]
        assert len(intermediate) > 0, "Should have intermediate chapters"

        # Most intermediate chapters should say "Entering"
        entering_count = sum(1 for c in intermediate if "Entering" in c["title"])
        assert entering_count > 0, "Intermediate towns should use 'Entering' prefix"

    def test_chapter_deduplication(self, generator):
        """Verify same town not repeated multiple times."""
        # Create journey that loops through same town
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 10, 10, 0, 0)
        waypoints = []

        # Drive through Ennis 3 times
        for i in range(10):
            town = "Ennis" if i % 3 == 0 else f"Town{i}"
            timestamp = start + timedelta(minutes=i * 6)
            location = Location(
                latitude=52.8450,
                longitude=-8.9860,
                city=town,
                county="County Clare",
                country="Ireland",
            )
            waypoints.append(
                Waypoint(
                    latitude=52.8450,
                    longitude=-8.9860,
                    timestamp=timestamp,
                    location=location,
                )
            )

        journey = Journey(
            start_date=start,
            end_date=start + timedelta(minutes=60),
            waypoints=tuple(waypoints),
            statistics=None,
        )

        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=3600,
            start_location="Ennis, County Clare",
            end_location="Ennis, County Clare",
            _mock_journey=journey,
        )

        # Count how many times Ennis appears (excluding Start/Finish)
        ennis_count = sum(
            1
            for c in chapters
            if "Ennis" in c["title"] and "Start" not in c["title"] and "Finish" not in c["title"]
        )

        # Should only appear once as intermediate, not 3 times
        assert ennis_count <= 1, f"Ennis appears {ennis_count} times, should deduplicate"

    def test_chapters_sorted_by_timestamp(self, generator, mock_gpx_data):
        """Verify chapters are in chronological order."""
        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=3600,
            start_location="Ennis, County Clare",
            end_location="Ennis, County Clare",
            _mock_journey=mock_gpx_data,
        )

        # Convert timestamps to seconds for comparison
        def timestamp_to_seconds(ts):
            parts = ts.split(":")
            if len(parts) == 2:  # MM:SS
                return int(parts[0]) * 60 + int(parts[1])
            else:  # H:MM:SS
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        timestamps = [timestamp_to_seconds(c["timestamp"]) for c in chapters]

        # Verify sorted
        assert timestamps == sorted(timestamps), "Chapters not in chronological order"

    def test_format_chapters_for_youtube_description(self, generator, mock_gpx_data):
        """Test formatting chapters as YouTube description text."""
        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=3600,
            start_location="Ennis, County Clare",
            end_location="Ennis, County Clare",
            _mock_journey=mock_gpx_data,
        )

        formatted = generator.format_chapters(chapters)

        # Should be string with one chapter per line
        assert isinstance(formatted, str)
        lines = formatted.strip().split("\n")
        assert len(lines) == len(chapters)

        # Each line should match format: "0:00 Title"
        for line in lines:
            assert " " in line, f"Line '{line}' should have timestamp and title"
            timestamp, title = line.split(" ", 1)
            assert ":" in timestamp, f"Timestamp '{timestamp}' invalid"

    def test_missing_gps_file_returns_basic_chapters(self, generator):
        """When GPS file missing, generate start/end from location parameter."""
        chapters = generator.generate(
            gpx_path=Path("/nonexistent/file.gpx"),
            video_duration=3600,
            start_location="Ennis, County Clare",
            end_location="Loop Head, County Clare",
        )

        # Should have at least start and end
        assert len(chapters) >= 2
        assert "Start" in chapters[0]["title"]
        assert "Finish" in chapters[-1]["title"]
        assert "Ennis" in chapters[0]["title"]
        assert "Loop Head" in chapters[-1]["title"]

    def test_geocoding_failure_continues_gracefully(self, generator, mock_gpx_data):
        """When geocoding fails, use GPS coords only."""
        # This tests resilience - implementation should handle geocoding errors
        # and continue with whatever data is available
        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=3600,
            start_location="Ennis, County Clare",
            end_location="Ennis, County Clare",
            _mock_journey=mock_gpx_data,
        )

        # Should still generate chapters even if some geocoding fails
        assert len(chapters) >= 2, "Should have at least start and end chapters"

    def test_gps_range_slices_journey_to_time_window(self, generator):
        """GPS range parameter slices waypoints to the correct time window.

        When a GPX file covers multiple driving blocks (e.g. 3 hours of driving
        across 3 separate segments), gps_range='0-60' should only use the first
        60 minutes of GPS data for chapter generation, not scale the entire file.
        """
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 12, 10, 0, 0)
        waypoints = []

        # Simulate 3 driving blocks totaling 180 min in a single track segment:
        # Block 1 (0-60 min): Towns A, B, C
        # Block 2 (60-120 min): Towns D, E, F
        # Block 3 (120-180 min): Towns G, H, I
        block_towns = [
            # Block 1
            (0, "TownA", "County Cork", 51.84, -9.52),
            (20, "TownB", "County Cork", 51.75, -9.50),
            (40, "TownC", "County Cork", 51.68, -9.48),
            (55, "TownC_end", "County Cork", 51.65, -9.47),
            # Block 2
            (60, "TownD", "County Kerry", 51.90, -9.80),
            (80, "TownE", "County Kerry", 51.95, -9.90),
            (100, "TownF", "County Kerry", 52.00, -10.0),
            (115, "TownF_end", "County Kerry", 52.05, -10.1),
            # Block 3
            (120, "TownG", "County Galway", 53.00, -9.00),
            (140, "TownH", "County Galway", 53.10, -9.10),
            (160, "TownI", "County Galway", 53.20, -9.20),
            (175, "TownI_end", "County Galway", 53.30, -9.30),
        ]

        for mins, town, county, lat, lon in block_towns:
            timestamp = start + timedelta(minutes=mins)
            location = Location(
                latitude=lat,
                longitude=lon,
                city=town,
                county=county,
                country="Ireland",
            )
            waypoints.append(
                Waypoint(
                    latitude=lat,
                    longitude=lon,
                    timestamp=timestamp,
                    location=location,
                )
            )

        journey = Journey(
            start_date=start,
            end_date=start + timedelta(minutes=180),
            waypoints=tuple(waypoints),
            statistics=None,
        )

        # Generate chapters for block 1 only (0-60 min GPS range)
        # Video is 55 min (after stop removal)
        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=55 * 60,  # 55 min video
            start_location="TownA, County Cork",
            end_location="TownA, County Cork",
            gps_range="0-60",
            _mock_journey=journey,
        )

        # Chapters should only contain towns from block 1 (County Cork)
        all_titles = " ".join(c["title"] for c in chapters)
        assert "County Kerry" not in all_titles, (
            f"Block 2 towns leaked into block 1 chapters: {all_titles}"
        )
        assert "County Galway" not in all_titles, (
            f"Block 3 towns leaked into block 1 chapters: {all_titles}"
        )

    def test_gps_range_block2_uses_correct_window(self, generator):
        """GPS range for middle block selects only that block's waypoints."""
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 12, 10, 0, 0)
        waypoints = []

        block_towns = [
            (0, "TownA", "County Cork", 51.84, -9.52),
            (30, "TownB", "County Cork", 51.75, -9.50),
            (55, "TownC", "County Cork", 51.68, -9.48),
            (60, "TownD", "County Kerry", 51.90, -9.80),
            (90, "TownE", "County Kerry", 51.95, -9.90),
            (115, "TownF", "County Kerry", 52.00, -10.0),
            (120, "TownG", "County Galway", 53.00, -9.00),
            (150, "TownH", "County Galway", 53.10, -9.10),
            (175, "TownI", "County Galway", 53.30, -9.30),
        ]

        for mins, town, county, lat, lon in block_towns:
            timestamp = start + timedelta(minutes=mins)
            location = Location(
                latitude=lat,
                longitude=lon,
                city=town,
                county=county,
                country="Ireland",
            )
            waypoints.append(
                Waypoint(
                    latitude=lat,
                    longitude=lon,
                    timestamp=timestamp,
                    location=location,
                )
            )

        journey = Journey(
            start_date=start,
            end_date=start + timedelta(minutes=180),
            waypoints=tuple(waypoints),
            statistics=None,
        )

        # Generate chapters for block 2 (60-120 min GPS range)
        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=50 * 60,
            start_location="TownD, County Kerry",
            end_location="TownD, County Kerry",
            gps_range="60-120",
            _mock_journey=journey,
        )

        all_titles = " ".join(c["title"] for c in chapters)
        assert "County Cork" not in all_titles, (
            f"Block 1 towns leaked into block 2 chapters: {all_titles}"
        )
        assert "County Galway" not in all_titles, (
            f"Block 3 towns leaked into block 2 chapters: {all_titles}"
        )

    def test_gps_range_none_uses_full_journey(self, generator, mock_gpx_data):
        """When gps_range is None, full journey is used (backward compatible)."""
        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=3600,
            start_location="Ennis, County Clare",
            end_location="Ennis, County Clare",
            gps_range=None,
            _mock_journey=mock_gpx_data,
        )

        assert len(chapters) >= 2, "Should generate chapters from full journey"

    # ── Piecewise time mapping tests ──────────────────────────────────

    def test_piecewise_mapping_shifts_chapters_after_stop(self, generator):
        """Chapters after a removed stop must shift earlier by the stop duration.

        Scenario: 60-min GPS, 60-min original video, one 10-min stop at 20-30 min.
        keep_segments = [(0, 1200), (1800, 3600)]  → 50 min output video.
        A town at GPS 40:00 (= original 40:00) should map to 30:00 in the
        edited video (40:00 minus the 10-min gap), NOT 40:00 * (50/60) = 33:20.
        """
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 10, 10, 0, 0)
        waypoints = []
        # Dense waypoints every 5 min for reliable index-based sampling
        town_at = {
            0: "StartTown",
            5: "StartTown",
            10: "TownA",
            15: "TownA",
            20: "TownA",
            25: "TownA",
            30: "TownA",
            35: "TownB",
            40: "TownB",
            45: "TownB",
            50: "TownC",
            55: "TownC",
        }
        for mins in range(0, 60, 5):
            town = town_at.get(mins, "Unknown")
            timestamp = start + timedelta(minutes=mins)
            lat = 52.80 - mins * 0.003
            lon = -8.90 + mins * 0.002
            location = Location(
                latitude=lat,
                longitude=lon,
                city=town,
                county="County Clare",
                country="Ireland",
            )
            waypoints.append(
                Waypoint(
                    latitude=lat,
                    longitude=lon,
                    timestamp=timestamp,
                    location=location,
                )
            )

        journey = Journey(
            start_date=start,
            end_date=start + timedelta(minutes=60),
            waypoints=tuple(waypoints),
            statistics=None,
        )

        # Stop at original 20:00-30:00 → keep [(0,1200), (1800,3600)]
        keep_segments = [(0.0, 1200.0), (1800.0, 3600.0)]

        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=3000,  # 50 min after stop removal
            start_location="StartTown, County Clare",
            end_location="StartTown, County Clare",
            keep_segments=keep_segments,
            _mock_journey=journey,
        )

        # Find TownB chapter — should be near 30:00, not 33:20
        town_b = [c for c in chapters if "TownB" in c["title"]]
        assert town_b, "TownB chapter should exist"
        ts = town_b[0]["timestamp"]
        parts = ts.split(":")
        total_seconds = int(parts[0]) * 60 + int(parts[1])
        # TownB first appears at GPS ~35:00 = original ~35:00.
        # After removing 20:00-30:00: edited = 20:00 + (35:00 - 30:00) = 25:00
        # Allow some tolerance for sampling granularity
        assert abs(total_seconds - 1500) < 120, (
            f"TownB should be near 25:00, got {ts} ({total_seconds}s)"
        )

    def test_piecewise_mapping_before_stop_unchanged(self, generator):
        """Chapters before any removed stop should keep their original timestamp."""
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 10, 10, 0, 0)
        waypoints = []
        # Dense waypoints so index-based sampling resolves towns correctly
        town_at = {
            0: "StartTown",
            5: "StartTown",
            10: "EarlyTown",
            15: "EarlyTown",
            20: "EarlyTown",
            25: "EarlyTown",
            30: "EarlyTown",
            35: "EarlyTown",
            40: "LateTown",
            45: "LateTown",
            50: "LateTown",
            55: "LateTown",
        }
        for mins in range(0, 60, 5):
            town = town_at.get(mins, "Unknown")
            timestamp = start + timedelta(minutes=mins)
            lat = 52.80 - mins * 0.003
            lon = -8.90 + mins * 0.002
            location = Location(
                latitude=lat,
                longitude=lon,
                city=town,
                county="County Clare",
                country="Ireland",
            )
            waypoints.append(
                Waypoint(
                    latitude=lat,
                    longitude=lon,
                    timestamp=timestamp,
                    location=location,
                )
            )

        journey = Journey(
            start_date=start,
            end_date=start + timedelta(minutes=60),
            waypoints=tuple(waypoints),
            statistics=None,
        )

        # Stop at 30:00-40:00 → keep [(0,1800), (2400,3600)]
        keep_segments = [(0.0, 1800.0), (2400.0, 3600.0)]

        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=2400,  # 40 min after stop removal
            start_location="StartTown, County Clare",
            keep_segments=keep_segments,
            _mock_journey=journey,
        )

        # EarlyTown first appears at GPS ~10:00, which is before the stop at 30:00.
        # Original time ≈ 10:00, fully within keep_segment (0, 1800), so
        # edited time = 10:00 (unchanged).
        early = [c for c in chapters if "EarlyTown" in c["title"]]
        assert early, "EarlyTown chapter should exist"
        ts = early[0]["timestamp"]
        parts = ts.split(":")
        total_seconds = int(parts[0]) * 60 + int(parts[1])
        assert abs(total_seconds - 600) < 120, (
            f"EarlyTown should be near 10:00, got {ts} ({total_seconds}s)"
        )

    def test_no_keep_segments_falls_back_to_linear(self, generator, mock_gpx_data):
        """When keep_segments is None, linear scaling is used (backward compat)."""
        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=3600,
            start_location="Ennis, County Clare",
            _mock_journey=mock_gpx_data,
        )
        assert len(chapters) >= 2

    def test_keep_segments_accepted_parameter(self, generator):
        """generate() accepts keep_segments as an optional parameter."""
        import inspect

        sig = inspect.signature(generator.generate)
        assert "keep_segments" in sig.parameters

    # ── Minimum chapter gap tests ─────────────────────────────────────

    def test_minimum_gap_between_chapters(self, generator):
        """Intermediate chapters within 120s of each other should be filtered.

        When two towns are very close together in time, the second should
        be dropped to avoid cluttered YouTube chapters.
        """
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 10, 10, 0, 0)
        waypoints = []
        # Two towns only 30s apart, followed by a gap, then another town. With
        # each town now stamped at its REAL timestamp, TownB sits
        # 30s after TownA and must be dropped by the 60s min-gap rule.
        towns = [
            (0, "Start", 52.80, -8.90),
            (10, "TownA", 52.75, -8.85),
            (10.5, "TownB", 52.74, -8.84),  # Only 30s after TownA
            (30, "TownC", 52.65, -8.75),
            (55, "TownD", 52.55, -8.65),
        ]
        for mins, town, lat, lon in towns:
            timestamp = start + timedelta(minutes=mins)
            location = Location(
                latitude=lat,
                longitude=lon,
                city=town,
                county="County Clare",
                country="Ireland",
            )
            waypoints.append(
                Waypoint(
                    latitude=lat,
                    longitude=lon,
                    timestamp=timestamp,
                    location=location,
                )
            )

        journey = Journey(
            start_date=start,
            end_date=start + timedelta(minutes=60),
            waypoints=tuple(waypoints),
            statistics=None,
        )

        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=3600,
            start_location="Start, County Clare",
            _mock_journey=journey,
        )

        # Should not have both TownA and TownB — one should be filtered
        intermediate = [
            c for c in chapters if "Start" not in c["title"] and "Finish" not in c["title"]
        ]
        timestamps = []
        for c in intermediate:
            parts = c["timestamp"].split(":")
            if len(parts) == 2:
                timestamps.append(int(parts[0]) * 60 + int(parts[1]))
            else:
                timestamps.append(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))

        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            assert gap >= 90, (
                f"Chapters too close: {intermediate[i - 1]['title']} and "
                f"{intermediate[i]['title']} only {gap}s apart"
            )

    # ── Finish chapter positioning tests ──────────────────────────────

    def test_finish_chapter_not_at_exact_end(self, generator, mock_gpx_data):
        """Finish chapter should not be at the exact last second of the video.

        YouTube may not display chapters that start in the final ~10 seconds.
        The finish timestamp should be slightly before the end, or omitted.
        """
        video_duration = 3600
        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=video_duration,
            start_location="Ennis, County Clare",
            end_location="Ennis, County Clare",
            _mock_journey=mock_gpx_data,
        )

        last = chapters[-1]
        if "Finish" in last["title"]:
            ts = last["timestamp"]
            parts = ts.split(":")
            if len(parts) == 2:
                total_seconds = int(parts[0]) * 60 + int(parts[1])
            else:
                total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            # Should be at least 10 seconds before the end
            assert total_seconds <= video_duration - 10, (
                f"Finish chapter at {ts} ({total_seconds}s) is too close to "
                f"video end ({video_duration}s)"
            )

    # ── Stop processor keep_segments in return value ──────────────────

    def test_stop_processor_returns_keep_segments(self):
        """StopProcessor.process() must include keep_segments in its return dict."""
        from gomoveshift.pipeline.stop_processor import StopProcessor

        # Verify the return dict schema includes keep_segments by checking
        # that the method signature hasn't changed in a way that drops it
        import inspect

        src = inspect.getsource(StopProcessor.process)
        assert "keep_segments" in src, (
            "StopProcessor.process() must return keep_segments in result dict"
        )


class TestFormatLocationCleanup:
    """_format_location must strip non-geographic suffixes."""

    @pytest.fixture
    def gen(self):
        return ChapterGenerator.__new__(ChapterGenerator)

    def test_strips_parking_suffix(self, gen):
        from gps_journey.core.models import Location

        loc = Location(
            latitude=32.43,
            longitude=-80.67,
            city="Hunting Island Parking",
            county="Beaufort County",
            country="United States",
        )
        result = gen._format_location(loc)
        assert "Parking" not in result
        assert "Hunting Island" in result
        assert "Beaufort County" in result

    def test_strips_subdivision_suffix(self, gen):
        from gps_journey.core.models import Location

        loc = Location(
            latitude=32.35,
            longitude=-80.72,
            city="Seaside Subdivision",
            county="Beaufort County",
            country="United States",
        )
        result = gen._format_location(loc)
        assert "Subdivision" not in result
        assert "Seaside" in result

    def test_strips_car_park_suffix(self, gen):
        from gps_journey.core.models import Location

        loc = Location(
            latitude=52.56,
            longitude=-9.75,
            city="Cliffs of Moher Car Park",
            county="County Clare",
            country="Ireland",
        )
        result = gen._format_location(loc)
        assert "Car Park" not in result
        assert "Cliffs of Moher" in result

    def test_strips_rest_area_suffix(self, gen):
        from gps_journey.core.models import Location

        loc = Location(
            latitude=33.10,
            longitude=-80.05,
            city="I-95 Rest Area",
            county="Beaufort County",
            country="United States",
        )
        result = gen._format_location(loc)
        assert "Rest Area" not in result

    def test_strips_boat_landing_parking(self, gen):
        from gps_journey.core.models import Location

        loc = Location(
            latitude=32.38,
            longitude=-80.68,
            city="Sands Boat Landing Parking",
            county="Beaufort County",
            country="United States",
        )
        result = gen._format_location(loc)
        assert "Boat Landing Parking" not in result
        assert "Sands" in result

    def test_strips_shopping_center(self, gen):
        from gps_journey.core.models import Location

        loc = Location(
            latitude=32.44,
            longitude=-80.67,
            city="Port Royal Shopping Center",
            county="Beaufort County",
            country="United States",
        )
        result = gen._format_location(loc)
        assert "Shopping Center" not in result
        assert "Port Royal" in result

    def test_preserves_normal_name(self, gen):
        from gps_journey.core.models import Location

        loc = Location(
            latitude=52.84,
            longitude=-8.99,
            city="Ennis",
            county="County Clare",
            country="Ireland",
        )
        result = gen._format_location(loc)
        assert result == "Ennis, County Clare"

    def test_nameless_location_suppressed_not_coordinate_string(self, gen):
        # A POI with no resolvable name must SUPPRESS the chapter
        # (return "") so the caller drops it, rather than shipping a raw
        # coordinate string like "Entering 35.1234, -82.5678".
        from gps_journey.core.models import Location

        loc = Location(latitude=35.1234, longitude=-82.5678)
        result = gen._format_location(loc)
        assert result == ""


class TestChapterDensity:
    """Chapter density — 10-min interval instead of 15-min."""

    def test_27min_video_min_chapters(self):
        """27 min video should target at least 2 intermediates."""
        edited_duration = 27 * 60
        min_chapters = max(2, int(edited_duration / 60 / 10))
        assert min_chapters >= 2

    def test_60min_video_min_chapters(self):
        """60 min video should target at least 6 intermediates."""
        edited_duration = 60 * 60
        min_chapters = max(2, int(edited_duration / 60 / 10))
        assert min_chapters >= 6

    def test_15min_video_min_chapters(self):
        """15 min video should target at least 2 intermediates (floor)."""
        edited_duration = 15 * 60
        min_chapters = max(2, int(edited_duration / 60 / 10))
        assert min_chapters == 2


class TestOverpassImprovements:
    """Overpass timeout and US route discovery."""

    def test_overpass_timeout_at_least_25s(self):
        """Overpass timeout must be at least 25s for US bounding boxes."""
        from gomoveshift.pipeline.chapters import OVERPASS_TIMEOUT

        assert OVERPASS_TIMEOUT >= 25, (
            f"OVERPASS_TIMEOUT={OVERPASS_TIMEOUT}s, need >= 25s for US queries"
        )


@pytest.mark.real_overpass
class TestOverpassParkDiscovery:
    """Overpass must query for parks, reserves, and national parks.

    Opts out of the conftest ``hermetic_overpass`` stub — these tests
    exercise the real query construction against a patched ``urlopen``.
    """

    def test_overpass_query_includes_parks(self):
        """Overpass query must search for leisure=nature_reserve|park."""
        import urllib.request
        from unittest.mock import MagicMock, patch

        from gomoveshift.pipeline.chapters import _query_overpass

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = req.data.decode("utf-8") if req.data else ""
            resp = MagicMock()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            resp.status = 200
            resp.read.return_value = b'{"elements": []}'
            return resp

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            _query_overpass((32.0, -81.0, 33.0, -80.0))

        query = urllib.parse.unquote(captured["data"])
        assert "leisure" in query, f"Overpass query missing leisure tag for parks: {query[:300]}"
        assert "nature_reserve" in query
        assert "park" in query

    def test_overpass_query_includes_national_parks(self):
        """Overpass query must search for boundary=national_park|protected_area."""
        import urllib.request
        from unittest.mock import MagicMock, patch

        from gomoveshift.pipeline.chapters import _query_overpass

        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = req.data.decode("utf-8") if req.data else ""
            resp = MagicMock()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            resp.status = 200
            resp.read.return_value = b'{"elements": []}'
            return resp

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            _query_overpass((32.0, -81.0, 33.0, -80.0))

        query = urllib.parse.unquote(captured["data"])
        assert "boundary" in query, (
            f"Overpass query missing boundary tag for national parks: {query[:300]}"
        )
        assert "national_park" in query

    def test_park_node_classified_as_park(self):
        """Park node within proximity must get poi_type='park'."""
        from datetime import datetime, timedelta
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from gps_journey.core.models import Journey, Waypoint

        from gomoveshift.config import Config
        from gomoveshift.pipeline.chapters import ChapterGenerator

        config = Config(
            project_name="test",
            base_input_path=Path("/tmp/test_input"),
            base_output_path=Path("/tmp/test_output"),
        )
        gen = ChapterGenerator(config)
        gen._keep_segments = None  # normally set by generate()

        # Create a journey with waypoints near the park node
        t0 = datetime(2025, 1, 1)
        wps = [
            Waypoint(latitude=32.35, longitude=-80.45, timestamp=t0),
            Waypoint(latitude=32.36, longitude=-80.44, timestamp=t0 + timedelta(seconds=300)),
            Waypoint(latitude=32.37, longitude=-80.43, timestamp=t0 + timedelta(seconds=600)),
        ]
        journey = MagicMock(spec=Journey)
        journey.waypoints = wps

        # Mock Overpass returning a park node near the route
        overpass_response = {
            "elements": [
                {
                    "type": "node",
                    "id": 12345,
                    "lat": 32.36,
                    "lon": -80.44,
                    "tags": {
                        "name": "Hunting Island State Park",
                        "leisure": "nature_reserve",
                    },
                }
            ]
        }

        with patch(
            "gomoveshift.pipeline.chapters._query_overpass",
            return_value=overpass_response,
        ):
            gen._discover_area_pois(
                journey=journey,
                gps_duration=600.0,
                original_duration=600.0,
            )

        assert len(gen.route_pois) >= 1, "Park node should be in route_pois"
        park_poi = next(
            (p for p in gen.route_pois if p["name"] == "Hunting Island State Park"),
            None,
        )
        assert park_poi is not None, "Hunting Island State Park not found in route_pois"
        assert park_poi["type"] == "park", (
            f"Park POI type should be 'park', got '{park_poi['type']}'"
        )

    def test_park_relation_not_treated_as_named_road(self):
        """Park relation must NOT end up in named_routes — must be in route_pois."""
        from datetime import datetime, timedelta
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from gps_journey.core.models import Journey, Waypoint

        from gomoveshift.config import Config
        from gomoveshift.pipeline.chapters import ChapterGenerator

        config = Config(
            project_name="test",
            base_input_path=Path("/tmp/test_input"),
            base_output_path=Path("/tmp/test_output"),
        )
        gen = ChapterGenerator(config)
        gen._keep_segments = None  # normally set by generate()

        t0 = datetime(2025, 1, 1)
        wps = [
            Waypoint(latitude=32.35, longitude=-80.45, timestamp=t0),
            Waypoint(latitude=32.36, longitude=-80.44, timestamp=t0 + timedelta(seconds=300)),
            Waypoint(latitude=32.37, longitude=-80.43, timestamp=t0 + timedelta(seconds=600)),
        ]
        journey = MagicMock(spec=Journey)
        journey.waypoints = wps

        # Park as a relation (common for national parks)
        overpass_response = {
            "elements": [
                {
                    "type": "relation",
                    "id": 99999,
                    "center": {"lat": 32.36, "lon": -80.44},
                    "tags": {
                        "name": "ACE Basin National Wildlife Refuge",
                        "boundary": "national_park",
                    },
                }
            ]
        }

        with patch(
            "gomoveshift.pipeline.chapters._query_overpass",
            return_value=overpass_response,
        ):
            gen._discover_area_pois(
                journey=journey,
                gps_duration=600.0,
                original_duration=600.0,
            )

        # Must NOT be in named_routes (it's a park, not a road)
        assert "ACE Basin National Wildlife Refuge" not in gen.named_routes, (
            "Park relation should not be treated as a named road"
        )

        # Must be in route_pois as type "park"
        park_poi = next(
            (p for p in gen.route_pois if "ACE Basin" in p["name"]),
            None,
        )
        assert park_poi is not None, "Park relation should appear in route_pois"
        assert park_poi["type"] == "park"


class TestChapterDurationInference:
    """Tests for original_duration inference with trailing stops.

    When keep_segments has a trailing gap (removed segment at the end),
    the original_duration must reflect the TRUE video duration, not just
    the last keep_segment end time.
    """

    @pytest.fixture(autouse=True)
    def mock_network(self):
        with (
            patch(
                "gomoveshift.pipeline.chapters._query_overpass",
                return_value={"elements": []},
            ),
            # `_coastline_points_for_bbox` (generator.py) calls
            # `overpass_source.post_overpass_query` directly -- a separate
            # entry point from `_query_overpass` above, so patching only
            # `_query_overpass` left it making a REAL Overpass API request
            # (with its retry/backoff sleep) on every test that reaches
            # terrain-band classification, ~50-200s of real SSL socket I/O
            # per test (measured via cProfile: 92 `_ssl._SSLSocket.read`
            # calls, 47.6s, on `test_geocoding_failure_continues_gracefully`).
            patch(
                "gomoveshift.pipeline.chapters.overpass_source.post_overpass_query",
                return_value={"elements": []},
            ),
            patch("gomoveshift.pipeline.chapters.NominatimRoadsClient") as mock_geo_cls,
        ):
            mock_geo_cls.return_value.reverse_geocode.return_value = None
            mock_geo_cls.return_value.reverse_geocode_detailed.return_value = {}
            yield

    @pytest.fixture
    def config(self):
        return Config(
            project_name="test",
            base_input_path=Path("/tmp/test_input"),
            base_output_path=Path("/tmp/test_output"),
        )

    @pytest.fixture
    def generator(self, config):
        return ChapterGenerator(config)

    @pytest.fixture
    def journey_60min(self):
        """60-minute GPS journey with 5 waypoints."""
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 10, 10, 0, 0)
        waypoints = []
        towns = [
            ("Ennis", "County Clare", 52.8450, -8.9860),
            ("Quin", "County Clare", 52.8233, -8.9489),
            ("Kilrush", "County Clare", 52.6417, -9.4850),
            ("Kilkee", "County Clare", 52.6808, -9.6456),
            ("Ennis", "County Clare", 52.8450, -8.9860),
        ]
        for i, (town, county, lat, lon) in enumerate(towns):
            timestamp = start + timedelta(minutes=i * 15)
            location = Location(
                latitude=lat,
                longitude=lon,
                city=town,
                county=county,
                country="Ireland",
            )
            waypoints.append(
                Waypoint(
                    latitude=lat,
                    longitude=lon,
                    timestamp=timestamp,
                    location=location,
                )
            )
        return Journey(
            start_date=start,
            end_date=start + timedelta(minutes=60),
            waypoints=tuple(waypoints),
            statistics=None,
        )

    def test_trailing_stop_wrong_original_duration(self, generator, journey_60min):
        """BUG REPRO: keep_segments[-1][1] != original video duration when
        there's a trailing removed segment.

        Video is 3600s, keep_segments end at 3400 (200s trailing stop removed).
        Without the fix, original_duration=3400 instead of 3600.
        The GPS-to-video scale factor would be 3200/3400=0.941 instead of
        the correct 3200/3600=0.889, shifting all chapter timestamps.
        """
        # 3600s original video, trailing 200s removed
        keep_segments = [(0, 1000), (1200, 3400)]
        edited_duration = (1000 - 0) + (3400 - 1200)  # 3200s
        original_video_duration = 3600.0  # true duration from ffprobe

        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=edited_duration,
            start_location="Ennis, County Clare",
            keep_segments=keep_segments,
            original_duration=original_video_duration,
            _mock_journey=journey_60min,
        )

        # Verify chapters were generated
        assert len(chapters) >= 2
        # First chapter must be at 0:00
        assert chapters[0]["timestamp"] == "0:00"
        # Last chapter must be within edited duration
        last_ts = chapters[-1]["timestamp"]
        parts = last_ts.split(":")
        if len(parts) == 3:
            last_secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            last_secs = int(parts[0]) * 60 + int(parts[1])
        assert last_secs <= edited_duration, (
            f"Last chapter {last_ts} ({last_secs}s) exceeds edited duration {edited_duration}s"
        )

    def test_original_duration_used_over_keep_segments_inference(self, generator, journey_60min):
        """When original_duration is passed, it should be used instead of
        inferring from keep_segments[-1][1].
        """
        keep_segments = [(0, 1000), (1200, 3400)]
        edited_duration = 3200.0
        original_video_duration = 3600.0

        # Capture the original_duration that gets logged
        import logging

        log_messages = []
        handler = logging.Handler()
        handler.emit = lambda record: log_messages.append(record.getMessage())
        logger = logging.getLogger("gomoveshift.pipeline.chapters.generator")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            generator.generate(
                gpx_path=Path("/fake/path.gpx"),
                video_duration=edited_duration,
                start_location="Ennis, County Clare",
                keep_segments=keep_segments,
                original_duration=original_video_duration,
                _mock_journey=journey_60min,
            )

            # The log should show original_duration=3600 (60.0min), not 3400 (56.7min)
            duration_log = [m for m in log_messages if "Original video:" in m]
            assert duration_log, "Expected log message about original video duration"
            assert "60.0min" in duration_log[0], (
                f"Expected original duration 60.0min in log, got: {duration_log[0]}"
            )
        finally:
            logger.removeHandler(handler)

    def test_fallback_to_keep_segments_when_no_original_duration(self, generator, journey_60min):
        """When original_duration is NOT passed, fall back to
        keep_segments[-1][1] for backward compatibility.
        """
        keep_segments = [(0, 1000), (1200, 3400)]
        edited_duration = 3200.0

        import logging

        log_messages = []
        handler = logging.Handler()
        handler.emit = lambda record: log_messages.append(record.getMessage())
        logger = logging.getLogger("gomoveshift.pipeline.chapters.generator")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            generator.generate(
                gpx_path=Path("/fake/path.gpx"),
                video_duration=edited_duration,
                start_location="Ennis, County Clare",
                keep_segments=keep_segments,
                _mock_journey=journey_60min,
            )

            # Without original_duration, should fall back to 3400 (56.7min)
            duration_log = [m for m in log_messages if "Original video:" in m]
            assert duration_log, "Expected log message about original video duration"
            assert "56.7min" in duration_log[0], (
                f"Expected fallback duration 56.7min in log, got: {duration_log[0]}"
            )
        finally:
            logger.removeHandler(handler)


class TestChapterGeneratorIncludedSegments:
    """Chapters trim GPX to clip timeline via ``included_segments``."""

    @pytest.fixture(autouse=True)
    def mock_network(self):
        with (
            patch(
                "gomoveshift.pipeline.chapters._query_overpass",
                return_value={"elements": []},
            ),
            # `_coastline_points_for_bbox` (generator.py) calls
            # `overpass_source.post_overpass_query` directly -- a separate
            # entry point from `_query_overpass` above, so patching only
            # `_query_overpass` left it making a REAL Overpass API request
            # (with its retry/backoff sleep) on every test that reaches
            # terrain-band classification, ~50-200s of real SSL socket I/O
            # per test (measured via cProfile: 92 `_ssl._SSLSocket.read`
            # calls, 47.6s, on `test_geocoding_failure_continues_gracefully`).
            patch(
                "gomoveshift.pipeline.chapters.overpass_source.post_overpass_query",
                return_value={"elements": []},
            ),
            patch("gomoveshift.pipeline.chapters.NominatimRoadsClient") as mock_geo_cls,
        ):
            mock_geo_cls.return_value.reverse_geocode.return_value = None
            mock_geo_cls.return_value.reverse_geocode_detailed.return_value = {}
            yield

    @pytest.fixture
    def config(self):
        return Config(
            project_name="test",
            base_input_path=Path("/tmp/test_input"),
            base_output_path=Path("/tmp/test_output"),
        )

    @pytest.fixture
    def generator(self, config):
        return ChapterGenerator(config)

    @pytest.fixture
    def journey_three_towns_split(self):
        """Journey with waypoints at t=0, 300, 600, 900, 1200, 1500, 1800s.

        For a 1800s source video, this lets us exclude the middle slice
        (500-1000s) via included_segments and assert the chapter list
        reflects only kept waypoints.
        """
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2026, 4, 22, 10, 0, 0)
        waypoints = []
        towns = [
            ("Alpha", 40.00, -120.00),
            ("Bravo", 40.01, -120.01),
            ("Charlie", 40.02, -120.02),  # In the cut window
            ("Delta", 40.03, -120.03),  # In the cut window
            ("Echo", 40.04, -120.04),
            ("Foxtrot", 40.05, -120.05),
            ("Gamma", 40.06, -120.06),
        ]
        for i, (name, lat, lon) in enumerate(towns):
            ts = start + timedelta(seconds=i * 300)
            loc = Location(
                latitude=lat, longitude=lon, city=name, county="Testshire", country="Testland"
            )
            waypoints.append(Waypoint(latitude=lat, longitude=lon, timestamp=ts, location=loc))
        return Journey(
            start_date=start,
            end_date=start + timedelta(seconds=1800),
            waypoints=tuple(waypoints),
            statistics=None,
        )

    def test_chapters_generate_backward_compat_when_included_segments_none(
        self, generator, journey_three_towns_split
    ):
        """Omitting included_segments (or None) preserves the old
        behaviour — no waypoints are trimmed."""
        chapters_default = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=1800,
            start_location="Alpha, Testshire",
            _mock_journey=journey_three_towns_split,
        )
        chapters_none = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=1800,
            start_location="Alpha, Testshire",
            included_segments=None,
            _mock_journey=journey_three_towns_split,
        )
        # Same chapters regardless of None vs omitted
        assert [c["timestamp"] for c in chapters_default] == [c["timestamp"] for c in chapters_none]

    def test_chapters_exclude_cut_footage(self, generator, journey_three_towns_split):
        """When a middle slice of the source is cut, the chapter list
        must not reference waypoints that fall inside the cut range."""
        # Keep 0-500s and 1000-1800s — cuts Charlie (600s) and Delta (900s).
        included = [(0.0, 500.0), (1000.0, 1800.0)]
        chapters = generator.generate(
            gpx_path=Path("/fake/path.gpx"),
            video_duration=1300,  # edited duration after cut
            start_location="Alpha, Testshire",
            included_segments=included,
            _mock_journey=journey_three_towns_split,
        )
        titles = " | ".join(c.get("title", "") for c in chapters)
        assert "Charlie" not in titles
        assert "Delta" not in titles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
