"""Chapter GPS-time ↔ waypoint mapping uses REAL waypoint timestamps.

Both chapter sources sampled the trace assuming a uniform GPX sample rate: a
target GPS mark (``i * 30s``) was paired with the waypoint at the *proportional*
index and stamped with the synthetic mark time. On a trace with a variable
sample rate or a dropout the proportional index lands on a waypoint recorded at
a completely different real time, so a town/POI name is placed at the wrong
LF-description timestamp.

The fix maps each sample by the waypoint's ACTUAL timestamp (elapsed from the
journey start): pick the waypoint nearest the mark by real time and carry that
waypoint's real elapsed time. These tests lock that both sources emit a
``gps_time`` that equals the real elapsed time of the waypoint it points at.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from gps_journey.core.models import Journey, Location, Waypoint

from gomoveshift.pipeline.chapters.nominatim_source import NominatimChapterSource
from gomoveshift.pipeline.chapters.overpass_source import OverpassChapterSource

_START = datetime(2025, 1, 1, 12, 0, 0)

# Non-uniform trace: six fixes packed in the first 5s, a 95s dropout, then four
# more. A proportional index picks waypoints whose real time is nowhere near the
# 0/30/60s marks; the real-timestamp map does not.
_REL_SECONDS = [0, 1, 2, 3, 4, 5, 100, 101, 102, 103]


def _journey() -> Journey:
    wps = []
    for i, secs in enumerate(_REL_SECONDS):
        lat = 51.0 + i * 0.01
        lon = -9.0 - i * 0.01
        loc = Location(
            latitude=lat,
            longitude=lon,
            name=f"wp{i}",
            city=f"City{i}",
            county="Kerry",
            country="Ireland",
        )
        wps.append(
            Waypoint(
                latitude=lat,
                longitude=lon,
                timestamp=_START + timedelta(seconds=secs),
                location=loc,
            )
        )
    return Journey(
        start_date=_START,
        end_date=_START + timedelta(seconds=_REL_SECONDS[-1]),
        waypoints=tuple(wps),
    )


def _real_time_by_coord(journey: Journey) -> dict:
    start = journey.waypoints[0].timestamp
    return {
        (round(w.latitude, 6), round(w.longitude, 6)): (w.timestamp - start).total_seconds()
        for w in journey.waypoints
    }


def test_overpass_spatial_index_gps_time_is_the_waypoints_real_time():
    journey = _journey()
    gps_duration = float(_REL_SECONDS[-1])

    sampled = OverpassChapterSource.build_spatial_index(journey, gps_duration)

    real_by_coord = _real_time_by_coord(journey)
    assert sampled
    for gps_time, lat, lon in sampled:
        real = real_by_coord[(round(lat, 6), round(lon, 6))]
        assert abs(gps_time - real) < 1e-6, (
            f"spatial-index sample carries gps_time {gps_time}s but its waypoint "
            f"was recorded at {real}s — assumed-uniform sampling mis-mapped it"
        )


def test_nominatim_fallback_gps_time_is_the_waypoints_real_time():
    journey = _journey()
    gps_duration = float(_REL_SECONDS[-1])
    source = NominatimChapterSource(geocoder=MagicMock())

    def _scale(gps_time, _gps_duration, _original_duration):
        return gps_time

    def _to_edited(original_time):
        return original_time

    geocoded = source.geocode_fallback(
        journey,
        gps_duration,
        gps_duration,
        _scale,
        _to_edited,
        lambda _name: True,
    )

    real_by_coord = _real_time_by_coord(journey)
    assert geocoded
    for video_time, location in geocoded:
        real = real_by_coord[(round(location.latitude, 6), round(location.longitude, 6))]
        assert abs(video_time - real) < 1e-6, (
            f"geocode-fallback sample carries time {video_time}s but its waypoint "
            f"was recorded at {real}s — assumed-uniform sampling mis-mapped it"
        )
