"""Superlong chapter profile (load-bearing): check_metadata_quality
requires >= max(1, int(video_duration/60/15)) intermediate chapters for any
GPS-bearing video over 600s -- 16 at the 4h superlong cap, unreachable under
the default max_chapters=15 (13 after Start/Finish). superlong_max_chapters
raises the room to place chapters; the default stays untouched (regression
guard).
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    import gps_journey  # noqa: F401

    HAS_GPS = True
except ImportError:
    HAS_GPS = False

if not HAS_GPS:
    pytest.skip("gps_journey package not installed", allow_module_level=True)

from gomoveshift.config import Config
from gomoveshift.pipeline.chapters.generator import ChapterGenerator, superlong_max_chapters
from gomoveshift.pipeline.validation.content_gates import check_metadata_quality

FOUR_HOURS_SEC = 4 * 3600.0
# 24 distinct towns, 10 min apart -> 240 min of track.
_TOWNS = [(f"Town{i:02d}", "County Test", 35.0 + i * 0.05, -82.0 - i * 0.05) for i in range(24)]


@pytest.fixture(autouse=True)
def mock_network():
    with patch("gomoveshift.pipeline.chapters._query_overpass", return_value={"elements": []}):
        with patch("gomoveshift.pipeline.chapters.NominatimRoadsClient") as mock_geo_cls:
            mock_geo_cls.return_value.reverse_geocode.return_value = None
            mock_geo_cls.return_value.reverse_geocode_detailed.return_value = {}
            yield


@pytest.fixture
def generator():
    config = Config(
        project_name="test",
        base_input_path=Path("/tmp/test_input"),
        base_output_path=Path("/tmp/test_output"),
    )
    return ChapterGenerator(config)


@pytest.fixture
def four_hour_journey():
    from gps_journey.core.models import Journey, Location, Waypoint

    start = datetime(2026, 6, 1, 8, 0, 0)
    waypoints = [
        Waypoint(
            latitude=lat,
            longitude=lon,
            timestamp=start + timedelta(minutes=i * 10),
            location=Location(latitude=lat, longitude=lon, city=t, county=c, country="USA"),
        )
        for i, (t, c, lat, lon) in enumerate(_TOWNS)
    ]
    return Journey(
        start_date=start,
        end_date=start + timedelta(seconds=FOUR_HOURS_SEC),
        waypoints=tuple(waypoints),
        statistics=None,
    )


@pytest.fixture
def stub_full_day_towns():
    """Simulate Overpass's whole-bbox area scan (the real long-drive path)
    instead of the Nominatim time-sampled fallback, which caps at 200 30s
    samples -- 100 min, short of a 4h track."""
    from gps_journey.core.models import Location

    interval = FOUR_HOURS_SEC / (len(_TOWNS) + 1)
    towns = [
        (interval * (i + 1), Location(latitude=lat, longitude=lon, city=t, county=c, country="USA"))
        for i, (t, c, lat, lon) in enumerate(_TOWNS)
    ]
    with patch.object(ChapterGenerator, "_discover_area_pois", return_value=["stub"]):
        with patch(
            "gomoveshift.pipeline.chapters.generator.OverpassChapterSource.pois_to_locations",
            return_value=towns,
        ):
            yield


class TestSuperlongChapterProfile:
    def test_max_chapters_covers_the_gate_requirement(self):
        # Gate needs 16 intermediate at 4h; profile must leave room for
        # that plus the 2 Start/Finish chapters.
        assert superlong_max_chapters(FOUR_HOURS_SEC) >= 18

    def test_medium_lf_default_untouched(self, generator):
        assert generator.max_chapters == 15

    def test_four_hour_part_clears_gate_and_byte_budget(
        self, generator, four_hour_journey, stub_full_day_towns
    ):
        from gomoveshift.pipeline.drive_phases.youtube_uploader import MAX_DESCRIPTION_BYTES

        chapters = generator.generate(
            gpx_path=Path("/fake/superlong.gpx"),
            video_duration=FOUR_HOURS_SEC,
            start_location="Town00, County Test",
            end_location="Town23, County Test",
            max_chapters=superlong_max_chapters(FOUR_HOURS_SEC),
            _mock_journey=four_hour_journey,
        )
        intermediate = [c for c in chapters if not c["title"].startswith(("Start", "Finish"))]
        assert len(intermediate) >= 16, chapters
        result = check_metadata_quality(
            {
                "video_duration": FOUR_HOURS_SEC,
                "has_gps": True,
                "chapters": chapters,
                "description": "",
                "title": "",
                "location": "",
            }
        )
        assert result.passed, result.message
        chapter_bytes = len(generator.format_chapters(chapters).encode("utf-8"))
        assert chapter_bytes < MAX_DESCRIPTION_BYTES


class TestMediumLfChapterRegression:
    """A normal (non-superlong) LF call is untouched by the profile existing."""

    def test_short_lf_unaffected(self, generator):
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2026, 6, 1, 8, 0, 0)
        towns = [
            ("Ennis", "County Clare", 52.8450, -8.9860),
            ("Kilrush", "County Clare", 52.6417, -9.4850),
        ]
        waypoints = [
            Waypoint(
                latitude=lat,
                longitude=lon,
                timestamp=start + timedelta(minutes=i * 15),
                location=Location(latitude=lat, longitude=lon, city=t, county=c, country="Ireland"),
            )
            for i, (t, c, lat, lon) in enumerate(towns)
        ]
        journey = Journey(
            start_date=start,
            end_date=start + timedelta(minutes=30),
            waypoints=tuple(waypoints),
            statistics=None,
        )
        chapters = generator.generate(
            gpx_path=Path("/fake/lf.gpx"),
            video_duration=1800.0,
            start_location="Ennis, County Clare",
            _mock_journey=journey,
        )
        assert len(chapters) >= 2
        assert chapters[0]["timestamp"] == "0:00"
