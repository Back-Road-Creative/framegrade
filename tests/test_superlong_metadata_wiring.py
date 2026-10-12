"""LOAD-BEARING (superlong metadata wiring): a superlong part's REAL
generated metadata -- built by ``day_pipeline._build_superlong_part_metadata``,
the same function ``_produce_superlong_parts`` calls for every part -- must
clear ``validation.content_gates.check_metadata_quality``'s chapter-density
gate at the 4h superlong cap (16 intermediate chapters required), and the
assembled description must stay inside the uploader's 5000-byte budget.

This exercises the WIRING, not just the helpers ``superlong_max_chapters`` /
``build_superlong_part_title`` in isolation (see ``test_superlong_chapters.py``
for those) -- before the wiring landed, ``_build_superlong_part_metadata``
did not exist and no part ever had chapters/description generated at all.
"""

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
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
from gomoveshift.pipeline.day_pipeline import _build_superlong_part_metadata
from gomoveshift.pipeline.drive_phases.youtube_uploader import MAX_DESCRIPTION_BYTES
from gomoveshift.pipeline.validation.content_gates import check_metadata_quality

FOUR_HOURS_SEC = 4 * 3600.0
# 24 distinct towns -- mirrors test_superlong_chapters.py's fixture so the
# Overpass fast path (min_chapters == 24 at this duration) is satisfied.
_TOWNS = [(f"Town{i:02d}", "County Test", 35.0 + i * 0.05, -82.0 - i * 0.05) for i in range(24)]


def _write_gpx(path: Path) -> Path:
    """A real, parseable 4h GPX track (25 points, 10min/0.05deg apart)."""
    start = datetime(2026, 6, 1, 8, 0, 0)
    points = "\n".join(
        f'    <trkpt lat="{35.0 + i * 0.05}" lon="{-82.0 - i * 0.05}">'
        f"<time>{(start + timedelta(minutes=i * 10)).strftime('%Y-%m-%dT%H:%M:%SZ')}</time>"
        f"</trkpt>"
        for i in range(25)
    )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx xmlns="http://www.topografix.com/GPX/1/1" version="1.1">\n'
        f"  <trk><trkseg>\n{points}\n  </trkseg></trk>\n"
        "</gpx>\n"
    )
    return path


@pytest.fixture(autouse=True)
def mock_network():
    with patch("gomoveshift.pipeline.chapters._query_overpass", return_value={"elements": []}):
        with patch("gomoveshift.pipeline.chapters.NominatimRoadsClient") as mock_geo_cls:
            mock_geo_cls.return_value.reverse_geocode.return_value = None
            mock_geo_cls.return_value.reverse_geocode_detailed.return_value = {}
            yield


@pytest.fixture
def stub_full_day_towns():
    """Simulate Overpass's whole-bbox area scan -- the real long-drive path."""
    from gps_journey.core.models import Location

    from gomoveshift.pipeline.chapters.generator import ChapterGenerator

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


class TestSuperlongPartMetadataWiring:
    def test_four_hour_part_metadata_clears_density_gate_and_byte_budget(
        self, tmp_path, stub_full_day_towns
    ):
        config = Config(
            project_name="test",
            base_input_path=tmp_path,
            base_output_path=tmp_path,
        )
        gpx_path = _write_gpx(tmp_path / "day.gpx")
        part = SimpleNamespace(part_index=1, part_count=1, kept_duration_sec=FOUR_HOURS_SEC)

        metadata = _build_superlong_part_metadata(
            part=part,
            date="2026-06-01",
            location="Town00 to Town23, County Test",
            gpx_path=gpx_path,
            resolved_location=None,
            config=config,
            stages_dir=tmp_path,
        )

        assert metadata["has_gps"] is True
        assert metadata["title"].startswith("Driving Town00 to Town23, County Test")
        intermediate = [
            c for c in metadata["chapters"] if not c["title"].startswith(("Start", "Finish"))
        ]
        assert len(intermediate) >= 16, metadata["chapters"]

        result = check_metadata_quality(metadata)
        assert result.passed, result.message

        desc_bytes = len(metadata["description"].encode("utf-8"))
        assert desc_bytes < MAX_DESCRIPTION_BYTES, desc_bytes

    def test_gps_less_day_skips_chapters_and_passes_gate(self, tmp_path):
        config = Config(
            project_name="test",
            base_input_path=tmp_path,
            base_output_path=tmp_path,
        )
        part = SimpleNamespace(part_index=1, part_count=1, kept_duration_sec=FOUR_HOURS_SEC)

        metadata = _build_superlong_part_metadata(
            part=part,
            date="2026-06-01",
            location="Test County, Test State",
            gpx_path=None,
            resolved_location=None,
            config=config,
            stages_dir=tmp_path,
        )

        assert metadata["has_gps"] is False
        assert metadata["chapters"] == []
        result = check_metadata_quality(metadata)
        assert result.passed, result.message
