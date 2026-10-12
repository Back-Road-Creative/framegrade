"""Chapter anchoring defects: (a) head-strip trim offset, (b) non-contiguous
LF town-time mapping.

(a) The ``included_segments`` trim keyed its window on the journey's start
    AFTER the null-island / teleport head strip, so every window shifted forward
    by the stripped head's duration — dropping on-screen towns and keeping cut
    ones. The window must key on the PRE-strip episode-local origin.

(b) The drive path charts each surviving GPS waypoint to a final-video timestamp
    with ONE uniform linear scale and never passes ``keep_segments``, so a town
    just after a cut segment lands minutes late. It must piecewise-map the
    waypoint through the kept (``included_segments``) ranges' cumulative time.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from gps_journey.core.models import Journey, Location, Waypoint

from gomoveshift.config import Config
from gomoveshift.pipeline.chapters.generator import ChapterGenerator

_BASE = datetime(2026, 5, 1, 9, 0, 0)


@pytest.fixture(autouse=True)
def _mock_network():
    from unittest.mock import patch

    with patch(
        "gomoveshift.pipeline.chapters._query_overpass",
        return_value={"elements": []},
    ):
        with patch("gomoveshift.pipeline.chapters.NominatimRoadsClient") as mock_geo_cls:
            mock_geo_cls.return_value.reverse_geocode.return_value = None
            mock_geo_cls.return_value.reverse_geocode_detailed.return_value = {}
            yield


@pytest.fixture
def generator() -> ChapterGenerator:
    return ChapterGenerator(
        Config(
            project_name="test",
            base_input_path=Path("/tmp/anchoring_in"),
            base_output_path=Path("/tmp/anchoring_out"),
        )
    )


def _town_wp(t_local: float, city: str) -> Waypoint:
    lat = 40.0 + t_local * 1e-4
    lon = -120.0 - t_local * 1e-4
    return Waypoint(
        latitude=lat,
        longitude=lon,
        timestamp=_BASE + timedelta(seconds=t_local),
        location=Location(
            latitude=lat, longitude=lon, city=city, county="Testshire", country="Testland"
        ),
    )


def _titles(chapters) -> str:
    return " | ".join(c.get("title", "") for c in chapters)


def test_head_strip_does_not_shift_trim_window(generator):
    """(a) A stripped 90s null-island cold-start head must not shift the
    included_segments window forward by its duration. ``LowEdge`` sits at
    episode-local 90s — inside the real window (60..300) but below a
    head-shifted one (150..390), so a window keyed on the POST-strip origin
    drops it. It must survive."""
    null_a = Waypoint(
        latitude=0.0,
        longitude=0.0,
        timestamp=_BASE,
        location=Location(latitude=0.0, longitude=0.0, country="Testland"),
    )
    null_b = Waypoint(
        latitude=0.0,
        longitude=0.0,
        timestamp=_BASE + timedelta(seconds=30),
        location=Location(latitude=0.0, longitude=0.0, country="Testland"),
    )
    towns = [
        _town_wp(90, "LowEdge"),
        _town_wp(150, "TownM"),
        _town_wp(210, "TownN"),
        _town_wp(270, "TownP"),
        _town_wp(330, "TailTown"),
        _town_wp(390, "TownT2"),
    ]
    journey = Journey(
        start_date=_BASE,
        end_date=_BASE + timedelta(seconds=390),
        waypoints=(null_a, null_b, *towns),
    )
    chapters = generator.generate(
        gpx_path=Path("/fake.gpx"),
        video_duration=240,  # episode-local 60..300 kept
        start_location="Origin, Testshire",
        included_segments=[(60.0, 300.0)],
        _mock_journey=journey,
    )
    titles = _titles(chapters)
    assert "LowEdge" in titles, f"low-edge town dropped by head-shifted window: {titles}"
    assert "TailTown" not in titles, f"cut town kept by head-shifted window: {titles}"


def test_noncontiguous_lf_town_time_is_piecewise(generator):
    """(b) A drive LF assembled from two far-apart clips must stamp a town in
    the second clip at its concatenated position, not the linearly-smeared one.
    Town1620 sits 120s into the second kept clip → edited 300+120 = 420s (7:00),
    not the uniform-scale 540s (9:00)."""
    town_secs = [0, 60, 120, 180, 240, 300, 1500, 1560, 1620, 1680, 1740, 1800]
    journey = Journey(
        start_date=_BASE,
        end_date=_BASE + timedelta(seconds=1800),
        waypoints=tuple(_town_wp(t, f"Town{t}") for t in town_secs),
    )
    chapters = generator.generate(
        gpx_path=Path("/fake.gpx"),
        video_duration=600,  # 300s + 300s kept
        start_location="Town0, Testshire",
        included_segments=[(0.0, 300.0), (1500.0, 1800.0)],
        _mock_journey=journey,
    )
    hit = [c for c in chapters if "Town1620" in c.get("title", "")]
    assert hit, f"Town1620 chapter missing: {_titles(chapters)}"
    parts = hit[0]["timestamp"].split(":")
    secs = int(parts[0]) * 60 + int(parts[1])
    assert 360 <= secs <= 480, f"Town1620 should sit ~7:00 (piecewise), got {hit[0]['timestamp']}"


def test_drive_delegates_piecewise_not_linear(generator):
    """(b) Focused: on the drive path (included_segments set, keep_segments
    unset) the scale→map composition is the piecewise cumulative-kept map, not a
    uniform linear scale."""
    generator._keep_segments = None
    generator._included_segments = [(0.0, 300.0), (1500.0, 1800.0)]
    generator._episode_offset = 0.0
    edited = generator._map_to_edited_time(generator._scale_gps_to_video(1620.0, 1800.0, 600.0))
    assert edited == 420.0
