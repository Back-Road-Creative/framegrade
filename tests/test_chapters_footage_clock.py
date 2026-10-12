"""Long-form chapters key on the footage clock (gms_media_s).

One episode = two recordings. Recording 1 is footage 0-600 s with GPS lock at
footage 90 s; recording 2 is footage 600-1200 s and its wall clock starts 30
min after recording 1 ends. Recording-2 fixes sit ~55 km away, so a chapter
window keyed on wall-clock time (or on the first fix) is visibly wrong.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import gpxpy
import gpxpy.gpx
import pytest
from gps_journey.core.gpx_parser import GPXParser
from gps_journey.core.models import Journey, Location, Waypoint

from gomoveshift.config import Config
from gomoveshift.pipeline.chapters._journey_filter import (
    load_and_filter_journey,
    load_journey_and_origin,
)
from gomoveshift.pipeline.chapters.generator import ChapterGenerator
from gomoveshift.pipeline.gopro_gps import build_stamped_gpx

_REC1_WALL = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
_REC2_WALL = _REC1_WALL + timedelta(seconds=600 + 1800)


def _samples(stamp: bool = True):
    out = []
    for m in range(90, 600, 30):  # recording 1, lock at footage 90 s
        out.append(
            (m if stamp else None, _REC1_WALL + timedelta(seconds=m), 40.0 + m * 1e-4, -120.0, 10.0)
        )
    for m in range(600, 1200, 30):  # recording 2, ~55 km north
        out.append(
            (
                m if stamp else None,
                _REC2_WALL + timedelta(seconds=m - 600),
                40.5 + m * 1e-4,
                -120.0,
                10.0,
            )
        )
    return out


def _write(tmp_path: Path, name: str, gpx: gpxpy.gpx.GPX) -> Path:
    p = tmp_path / name
    p.write_text(gpx.to_xml())
    return p


@pytest.fixture
def stamped(tmp_path) -> Path:
    return _write(tmp_path, "stamped.gpx", build_stamped_gpx(_samples()))


@pytest.fixture(autouse=True)
def _mock_network():
    with patch("gomoveshift.pipeline.chapters._query_overpass", return_value={"elements": []}):
        with patch("gomoveshift.pipeline.chapters.NominatimRoadsClient") as cls:
            cls.return_value.reverse_geocode.return_value = None
            cls.return_value.reverse_geocode_detailed.return_value = {}
            yield


@pytest.fixture
def generator() -> ChapterGenerator:
    return ChapterGenerator(
        Config(
            project_name="test",
            base_input_path=Path("/tmp/chapters_in"),
            base_output_path=Path("/tmp/chapters_out"),
        )
    )


def test_stamped_file_yields_footage_origin(stamped):
    journey, origin = load_journey_and_origin(stamped, None, None, None)
    assert origin == _REC1_WALL
    assert journey is not None
    expected = [s[0] for s in _samples()]
    got = [(wp.timestamp - origin).total_seconds() for wp in journey.waypoints]
    assert got == pytest.approx(expected, abs=0.01)


def test_unstamped_file_unchanged(tmp_path):
    p = _write(tmp_path, "plain.gpx", build_stamped_gpx(_samples(stamp=False)))
    journey, origin = load_journey_and_origin(p, None, None, None)
    ref = GPXParser().parse(p)
    assert origin is None
    assert [(w.latitude, w.longitude, w.timestamp) for w in journey.waypoints] == [
        (w.latitude, w.longitude, w.timestamp) for w in ref.waypoints
    ]


def test_partially_stamped_file_falls_back(tmp_path):
    samples = _samples()
    samples[3] = (None, *samples[3][1:])
    p = _write(tmp_path, "partial.gpx", build_stamped_gpx(samples))
    assert load_journey_and_origin(p, None, None, None) == (None, None)
    assert load_and_filter_journey(p, None, None, None) is None


def test_trim_keys_on_footage_seconds(generator, stamped):
    kept = []
    import gomoveshift.pipeline.chapters.generator as gen_mod

    real = gen_mod._rebuild_journey

    def spy(journey, wps):
        kept[:] = list(wps)
        return real(journey, wps)

    with patch.object(gen_mod, "_rebuild_journey", spy):
        generator.generate(
            gpx_path=stamped,
            video_duration=200,
            start_location="Origin, Testshire",
            included_segments=[(700.0, 900.0)],
        )
    assert kept, "trim dropped every waypoint (wall-clock window)"
    origin = generator._footage_origin
    assert origin == _REC1_WALL
    secs = [(w.timestamp - origin).total_seconds() for w in kept]
    assert all(700 <= s <= 900 for s in secs), secs
    assert all(w.latitude > 40.4 for w in kept)


def test_scale_returns_exact_footage_second(generator, stamped):
    segs = [(0.0, 400.0), (700.0, 1200.0)]
    generator.generate(
        gpx_path=stamped,
        video_duration=900,
        start_location="Origin, Testshire",
        keep_segments=segs,
        included_segments=segs,
    )
    assert generator._footage_origin == _REC1_WALL
    assert generator._scale_gps_to_video(0.0, 1000.0, 900.0) == pytest.approx(90.0)
    assert generator._scale_gps_to_video(60.0, 1000.0, 900.0) == pytest.approx(150.0)


def test_unstamped_anchoring_path_unchanged(generator):
    base = datetime(2026, 5, 1, 9, 0, 0)
    secs = [0, 60, 120, 180, 240, 300, 1500, 1560, 1620, 1680, 1740, 1800]
    wps = tuple(
        Waypoint(
            latitude=40.0 + t * 1e-4,
            longitude=-120.0 - t * 1e-4,
            timestamp=base + timedelta(seconds=t),
            location=Location(
                latitude=40.0, longitude=-120.0, city=f"Town{t}", county="T", country="TL"
            ),
        )
        for t in secs
    )
    journey = Journey(start_date=base, end_date=base + timedelta(seconds=1800), waypoints=wps)
    chapters = generator.generate(
        gpx_path=Path("/fake.gpx"),
        video_duration=600,
        start_location="Town0, Testshire",
        included_segments=[(0.0, 300.0), (1500.0, 1800.0)],
        _mock_journey=journey,
    )
    assert generator._footage_origin is None
    hit = [c for c in chapters if "Town1620" in c.get("title", "")]
    assert hit
    m, s = hit[0]["timestamp"].split(":")[-2:]
    assert 360 <= int(m) * 60 + int(s) <= 480


def test_day_journey_never_sets_origin(generator, stamped):
    base = datetime(2026, 5, 1, 9, 0, 0)
    wps = tuple(
        Waypoint(
            latitude=41.0 + i * 1e-3, longitude=-120.0, timestamp=base + timedelta(seconds=i * 30)
        )
        for i in range(20)
    )
    day = Journey(start_date=wps[0].timestamp, end_date=wps[-1].timestamp, waypoints=wps)
    generator.generate(
        gpx_path=stamped,
        video_duration=600,
        start_location="Origin, Testshire",
        day_journey=day,
    )
    assert generator._footage_origin is None
