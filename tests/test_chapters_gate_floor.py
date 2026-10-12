"""The chapter generator must clear the metadata gate's chapter-density floor.

``content_gates.check_metadata_quality`` hard-fails a GPS-bearing LF over 10 min
with fewer than ``max(1, int(minutes / 15))`` intermediate chapters. The
generator sized its own target differently (start/end towns counted, YouTube's
3-total pad), so a 34-minute long-form could ship Start,
one middle chapter and Finish and fail the gate after the render. Overpass can
return several places that all sit at the same second, which the merger's 60 s
gap collapses to one.

The floor is one shared function; ``generate()`` tops up below it with chapters
named from real reverse-geocoded GPS samples at their own timestamps, and never
pads with a name the GPS doesn't put the car at.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from gps_journey.core.models import Location

from gomoveshift.pipeline.chapters import ChapterGenerator, OverpassChapterSource
from gomoveshift.pipeline.chapters.generator import min_intermediate_chapters
from gomoveshift.pipeline.validation.content_gates import check_metadata_quality

LF_SECONDS = 2049.548  # a 34:10 edited long-form
GPS_SECONDS = 1820  # ~30 min of GPS track (a multiple of the 5 s waypoint step)
START_LAT, END_LAT = 36.2076, 36.5484  # Unicoi -> Kingsport, TN (northbound)
START_LON, END_LON = -82.3549, -82.5618


def _lat_at(gps_s: float) -> float:
    return START_LAT + (END_LAT - START_LAT) * gps_s / GPS_SECONDS


def _town_at(lat: float) -> tuple[str, str]:
    """The place a reverse geocoder reports at ``lat`` on this drive."""
    if lat < 36.25:
        return "Unicoi", "Unicoi County"
    if lat < 36.36:
        return "Johnson City", "Washington County"
    if lat < 36.45:
        return "Gray", "Washington County"
    return "Kingsport", "Sullivan County"


def _fake_reverse(lat, lon):
    city, county = _town_at(lat)
    return Location(latitude=lat, longitude=lon, city=city, county=county, country="United States")


def _journey():
    start = datetime(2026, 3, 4, 14, 0, 0)
    wps = []
    for s in range(0, GPS_SECONDS + 1, 5):
        wp = MagicMock()
        wp.latitude = _lat_at(s)
        wp.longitude = START_LON + (END_LON - START_LON) * s / GPS_SECONDS
        wp.timestamp = start + timedelta(seconds=s)
        wp.elevation = None
        wp.location = None
        wps.append(wp)
    j = MagicMock()
    j.waypoints = tuple(wps)
    j.start_date = wps[0].timestamp
    j.end_date = wps[-1].timestamp
    j.statistics = MagicMock()
    return j


def _clustered_overpass_places():
    """Four places at one second, as Overpass can return them."""
    t = 705.3
    jc = Location(
        latitude=36.3168, longitude=-82.3529, city="Johnson City", county="Washington County"
    )
    parks = [
        Location(latitude=36.3165, longitude=-82.3515, name="Downtown Square Breezeway"),
        Location(latitude=36.3158, longitude=-82.3527, name="Spring Street Park"),
        Location(latitude=36.3172, longitude=-82.3514, name="Majestic Park"),
    ]
    return [(t, jc)] + [(t, p) for p in parks]


def _secs(ts: str) -> int:
    out = 0
    for part in ts.split(":"):
        out = out * 60 + int(part)
    return out


def _intermediate(chapters):
    return [c for c in chapters if not c["title"].startswith(("Start", "Finish"))]


def _generate(tmp_path, reverse=_fake_reverse):
    from gomoveshift.config import Config

    gen = ChapterGenerator(Config(), cache_path=tmp_path / "geocoding_cache.json")
    gen.geocoder.reverse_geocode = MagicMock(side_effect=reverse)
    gen.geocoder.reverse_geocode_detailed = MagicMock(return_value={})
    with (
        patch.object(ChapterGenerator, "_discover_area_pois", return_value=[object()]),
        patch.object(
            OverpassChapterSource,
            "pois_to_locations",
            return_value=_clustered_overpass_places(),
        ),
        patch(
            "gomoveshift.pipeline.chapters.generator._coastline_points_for_bbox",
            return_value=[],
        ),
    ):
        chapters = gen.generate(
            gpx_path=Path("fake.gpx"),
            video_duration=LF_SECONDS,
            start_location="Unicoi to Kingsport, TN",
            end_location="Kingsport",
            _mock_journey=_journey(),
        )
    return chapters


def _gate(chapters, seconds):
    return check_metadata_quality(
        {
            "title": "Unicoi to Kingsport, TN in 4K",
            "location": "Unicoi to Kingsport, TN",
            "description": "POV drive from Unicoi to Kingsport, TN.",
            "has_gps": True,
            "video_duration": seconds,
            "chapters": chapters,
            # The LF's sampled counties, as phase D writes them.
            "resolved_location": {
                "regions": ["Unicoi County", "Washington County", "Sullivan County"]
            },
        }
    )


@pytest.mark.parametrize(
    ("seconds", "floor"),
    [(0, 0), (540, 0), (600, 0), (601, 1), (1799, 1), (1800, 2), (LF_SECONDS, 2), (3600, 4)],
)
def test_floor_is_the_gates_floor(seconds, floor):
    assert min_intermediate_chapters(seconds) == floor


@pytest.mark.parametrize("seconds", [601, 1800, LF_SECONDS, 3600, 4 * 3600])
def test_gate_requires_exactly_the_shared_floor(seconds):
    floor = min_intermediate_chapters(seconds)

    def chapters(n):
        middle = [
            {"timestamp": f"{1 + i}:00", "title": f"Entering Town{i}, Washington County"}
            for i in range(n)
        ]
        return (
            [{"timestamp": "0:00", "title": "Start - Unicoi"}]
            + middle
            + [{"timestamp": "59:00", "title": "Finish - Kingsport"}]
        )

    assert "intermediate chapter" not in _gate(chapters(floor), seconds).message
    assert "intermediate chapter" in _gate(chapters(floor - 1), seconds).message


def test_clustered_overpass_places_still_clear_the_gate(tmp_path):
    chapters = _generate(tmp_path)

    assert len(_intermediate(chapters)) >= min_intermediate_chapters(LF_SECONDS), chapters
    result = _gate(chapters, LF_SECONDS)
    assert result.passed, (result.message, chapters)


def test_every_middle_chapter_names_where_the_gps_puts_the_car(tmp_path):
    chapters = _generate(tmp_path)

    scale = LF_SECONDS / GPS_SECONDS
    for ch in _intermediate(chapters):
        m = re.match(r"^Entering (.+?)(?:,| —|$)", ch["title"])
        assert m, f"middle chapter is not a GPS place: {ch}"
        town, _ = _town_at(_lat_at(_secs(ch["timestamp"]) / scale))
        assert m.group(1) == town, f"{ch} — the GPS puts the car in {town} at that time"


def test_no_geocoder_means_no_invented_chapters(tmp_path, caplog):
    """With nothing to name a middle chapter from, the generator must not pad
    with a guessed town; it reports the shortfall and the gate stays shut."""
    caplog.set_level("WARNING")
    chapters = _generate(tmp_path, reverse=lambda lat, lon: None)

    titles = [re.sub(r" — .*$", "", c["title"]) for c in _intermediate(chapters)]
    assert titles == ["Entering Johnson City, Washington County"], chapters
    assert any("chapter floor" in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]


def test_overpass_down_fallback_spreads_samples_and_leaves_budget_for_the_floor():
    """Overpass down on a 61 min LF: the Nominatim fallback must sample the whole
    drive inside the session budget and leave requests for the floor top-up.

    It used to geocode one sample per 30 s from the start, so 122 samples burned
    all 50 requests in the drive's first 25 minutes: the back half had no names
    and the top-up's road lookups came back empty, leaving 2 of 4 middle chapters.
    """
    from types import SimpleNamespace

    from gomoveshift.pipeline.chapters.nominatim_source import NominatimChapterSource

    gps_seconds = 3660
    start = datetime(2000, 1, 1, 9, 0, 0)
    waypoints = [
        SimpleNamespace(
            latitude=40.1 + s / 100000,
            longitude=-80.2,
            location=None,
            timestamp=start + timedelta(seconds=s),
        )
        for s in range(0, gps_seconds + 1, 5)
    ]
    geocoder = MagicMock()
    geocoder.reverse_geocode.side_effect = lambda lat, lon: Location(
        latitude=lat, longitude=lon, city=f"Townland {lat:.5f}"
    )
    source = NominatimChapterSource(geocoder=geocoder, cache=None)

    geocoded = source.geocode_fallback(
        SimpleNamespace(waypoints=waypoints),
        float(gps_seconds),
        float(gps_seconds),
        lambda g, _gd, _od: g,
        lambda t: t,
        lambda _name: True,
    )

    left = source.MAX_SESSION_REQUESTS - source.session_request_count
    assert left >= source.MAX_SESSION_REQUESTS // 2, (
        f"fallback left {left} Nominatim requests for the floor top-up"
    )
    times = [t for t, _ in geocoded]
    assert times and max(times) >= 0.75 * gps_seconds, (
        f"last fallback sample at {max(times, default=0):.0f}s of a {gps_seconds}s drive"
    )
