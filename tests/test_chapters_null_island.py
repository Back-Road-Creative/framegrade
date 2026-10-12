"""Null-island (0,0) cold-start guard on the chapter path.

GoPro / Insta360 cameras emit ``(0, 0)`` waypoints before satellite lock.
Reverse-geocoding those produces a bogus mid-ocean chapter and can fabricate
the GPS-detected end location. ``_strip_null_island_waypoints`` removes them at
the single point where the journey is materialised, before both the
geocode-sampling fallback and the end-location detection consume it.

Same root cause as the centroid→dominant-county drive-label fix —
that fix protected per-LF titles/OST/shorts but the chapter
markers burned into the same drive videos (phase_d ChapterGenerator) were not
covered.
"""

from datetime import datetime, timedelta

from gps_journey.core.models import Journey, Waypoint

from gomoveshift.pipeline.chapters.generator import _strip_null_island_waypoints

_START = datetime(2025, 1, 1, 0, 0, 0)


def _wp(lat: float, lon: float, secs: int) -> Waypoint:
    return Waypoint(
        latitude=lat,
        longitude=lon,
        timestamp=_START + timedelta(seconds=secs),
        location=None,
    )


def _journey(waypoints: tuple) -> Journey:
    return Journey(
        start_date=_START,
        end_date=_START + timedelta(seconds=60),
        waypoints=waypoints,
        statistics=None,
    )


class TestStripNullIsland:
    def test_drops_leading_cold_start_fixes(self):
        j = _journey(
            (
                _wp(0.0, 0.0, 0),
                _wp(0.0, 0.0, 1),
                _wp(34.0, -80.0, 30),
                _wp(34.5, -80.5, 60),
            )
        )
        out = _strip_null_island_waypoints(j)
        assert len(out.waypoints) == 2
        assert all(not (w.latitude == 0.0 and w.longitude == 0.0) for w in out.waypoints)

    def test_drops_interior_null_island(self):
        j = _journey(
            (
                _wp(34.0, -80.0, 0),
                _wp(0.0, 0.0, 30),
                _wp(34.5, -80.5, 60),
            )
        )
        out = _strip_null_island_waypoints(j)
        assert len(out.waypoints) == 2

    def test_clean_journey_unchanged(self):
        wps = (_wp(34.0, -80.0, 0), _wp(34.5, -80.5, 60))
        j = _journey(wps)
        out = _strip_null_island_waypoints(j)
        # Nothing to drop → same waypoints returned (identity, not a rebuild).
        assert out.waypoints == wps

    def test_all_null_island_keeps_original(self):
        """If every fix is (0,0) we keep the original so the basic-chapter
        fallback still has data — never return an empty journey."""
        wps = (_wp(0.0, 0.0, 0), _wp(0.0, 0.0, 60))
        j = _journey(wps)
        out = _strip_null_island_waypoints(j)
        assert out.waypoints == wps

    def test_empty_journey_safe(self):
        j = _journey(())
        out = _strip_null_island_waypoints(j)
        assert out.waypoints == ()
