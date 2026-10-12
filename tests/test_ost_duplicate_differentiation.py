"""Tests for OST duplicate differentiation in drive_orchestrator.

The bug: two episodes on different days both render as
"Stocks Creek Road, South Carolina Scenic Drive". Pipeline must detect
duplicates and differentiate via direction or date.
"""

from datetime import datetime


class TestDuplicateDifferentiation:
    """differentiate_duplicate_ost applies direction-then-date waterfall."""

    def test_no_duplicates_noop(self):
        """When all OST strings are unique, return unchanged."""
        from gomoveshift.pipeline._drive_helpers import (
            differentiate_duplicate_ost,
        )

        locations = ["Port Royal to Beaufort", "Seaside Road", "Coastal Hwy"]
        meta = [
            {"start": (32.3, -80.6), "end": (32.4, -80.5), "date": datetime(2026, 2, 22)},
            {"start": (32.5, -80.7), "end": (32.6, -80.8), "date": datetime(2026, 2, 23)},
            {"start": (32.7, -80.9), "end": (32.8, -81.0), "date": datetime(2026, 2, 24)},
        ]
        result = differentiate_duplicate_ost(locations, meta)
        assert result == locations

    def test_two_duplicates_get_direction_appended(self):
        """Two identical OSTs with different bearings → direction suffix."""
        from gomoveshift.pipeline._drive_helpers import (
            differentiate_duplicate_ost,
        )

        locations = [
            "Stocks Creek Road, SC",
            "Stocks Creek Road, SC",
            "Unrelated OST",
        ]
        meta = [
            # Northbound episode
            {"start": (32.0, -80.5), "end": (32.5, -80.5), "date": datetime(2026, 2, 22)},
            # Southbound episode
            {"start": (32.5, -80.5), "end": (32.0, -80.5), "date": datetime(2026, 2, 24)},
            {"start": (33.0, -81.0), "end": (33.1, -81.1), "date": datetime(2026, 2, 26)},
        ]
        result = differentiate_duplicate_ost(locations, meta)
        # Both dupes are now unique
        assert result[0] != result[1]
        # Direction suffix added
        assert " (N)" in result[0] or " (N)" in result[1]
        assert " (S)" in result[0] or " (S)" in result[1]
        # Unrelated episode untouched
        assert result[2] == "Unrelated OST"

    def test_same_direction_dupes_get_date_appended(self):
        """Two identical OSTs going same direction → date suffix fallback."""
        from gomoveshift.pipeline._drive_helpers import (
            differentiate_duplicate_ost,
        )

        locations = [
            "Stocks Creek Road, SC",
            "Stocks Creek Road, SC",
        ]
        meta = [
            # Both northbound
            {"start": (32.0, -80.5), "end": (32.5, -80.5), "date": datetime(2026, 2, 22)},
            {"start": (32.0, -80.5), "end": (32.5, -80.5), "date": datetime(2026, 2, 24)},
        ]
        result = differentiate_duplicate_ost(locations, meta)
        # Both dupes are now unique — date fallback
        assert result[0] != result[1]
        assert "Feb 22" in result[0] or "Feb 24" in result[0]
        assert "Feb 22" in result[1] or "Feb 24" in result[1]

    def test_three_way_duplicate_all_differentiated(self):
        """Three identical OSTs all get differentiation."""
        from gomoveshift.pipeline._drive_helpers import (
            differentiate_duplicate_ost,
        )

        locations = ["Same OST"] * 3
        meta = [
            {"start": (32.0, -80.5), "end": (32.5, -80.5), "date": datetime(2026, 2, 22)},  # N
            {"start": (32.5, -80.5), "end": (32.0, -80.5), "date": datetime(2026, 2, 23)},  # S
            {"start": (32.0, -80.0), "end": (32.0, -80.5), "date": datetime(2026, 2, 24)},  # W
        ]
        result = differentiate_duplicate_ost(locations, meta)
        assert len(set(result)) == 3

    def test_empty_ost_strings_left_alone(self):
        """Empty OST strings are not considered duplicates of each other."""
        from gomoveshift.pipeline._drive_helpers import (
            differentiate_duplicate_ost,
        )

        locations = ["", "", "Real OST"]
        meta = [
            {"start": (32.0, -80.0), "end": (32.1, -80.1), "date": datetime(2026, 2, 22)},
            {"start": (32.2, -80.2), "end": (32.3, -80.3), "date": datetime(2026, 2, 23)},
            {"start": (32.4, -80.4), "end": (32.5, -80.5), "date": datetime(2026, 2, 24)},
        ]
        result = differentiate_duplicate_ost(locations, meta)
        # Empty strings stay empty — dedup only applies to real OSTs
        assert result[0] == ""
        assert result[1] == ""
        assert result[2] == "Real OST"

    def test_bearing_to_cardinal_eight_point_compass(self):
        """_bearing_to_cardinal maps bearings to 8-point compass labels."""
        from gomoveshift.pipeline._drive_helpers import _bearing_to_cardinal

        assert _bearing_to_cardinal(0) == "N"
        assert _bearing_to_cardinal(45) == "NE"
        assert _bearing_to_cardinal(90) == "E"
        assert _bearing_to_cardinal(135) == "SE"
        assert _bearing_to_cardinal(180) == "S"
        assert _bearing_to_cardinal(225) == "SW"
        assert _bearing_to_cardinal(270) == "W"
        assert _bearing_to_cardinal(315) == "NW"
        # Wrap-around
        assert _bearing_to_cardinal(360) == "N"
        assert _bearing_to_cardinal(-45) == "NW"


def _wp(lat, lon, ts):
    from gps_journey.core.models import Location, Waypoint

    return Waypoint(
        latitude=lat,
        longitude=lon,
        timestamp=ts,
        location=Location(latitude=lat, longitude=lon),
    )


def _extract_from_waypoints(wps):
    """Run _extract_episode_ost_meta against a mocked GPX journey."""
    from unittest.mock import MagicMock, patch

    from gps_journey.core.models import Journey

    from gomoveshift.pipeline._drive_helpers import _extract_episode_ost_meta

    journey = Journey(
        start_date=wps[0].timestamp,
        end_date=wps[-1].timestamp,
        waypoints=tuple(wps),
        statistics=None,
    )
    parser = MagicMock()
    parser.parse.return_value = journey
    with patch("gps_journey.core.gpx_parser.GPXParser", return_value=parser):
        return _extract_episode_ost_meta("fake.gpx")


class TestExtractMetaLocalDay:
    """The OST date suffix is the drive's LOCAL calendar day, not the raw
    UTC first-fix date."""

    def test_late_evening_start_uses_local_day_not_utc(self):
        from datetime import date, datetime

        # 23:30 local on Feb 22 (lon -75 ≈ UTC-5) records at 04:30 UTC Feb 23.
        # The raw UTC date would burn "Feb 23" on an otherwise-correct OST.
        ts = datetime(2026, 2, 23, 4, 30)
        meta = _extract_from_waypoints([_wp(34.0, -75.0, ts), _wp(34.5, -75.2, ts)])

        assert meta["date"] == date(2026, 2, 22)
        assert meta["date"].strftime("%b %d").replace(" 0", " ") == "Feb 22"


class TestExtractMetaNullIsland:
    """A timestamped (0, 0) cold-start fix must not seed a fabricated
    compass bearing from the Gulf of Guinea."""

    def test_null_island_first_fix_skipped(self):
        from datetime import datetime

        ts = datetime(2026, 2, 22, 16, 0)
        meta = _extract_from_waypoints(
            [
                _wp(0.0, 0.0, ts),  # GPS cold-start sentinel
                _wp(34.0, -80.0, ts),  # first real fix
                _wp(34.5, -80.5, ts),  # last real fix
            ]
        )

        # start/end come from the real fixes, not the (0, 0) sentinel.
        assert meta["start"] == (34.0, -80.0)
        assert meta["end"] == (34.5, -80.5)

    def test_all_null_island_returns_sentinel(self):
        from datetime import datetime

        ts = datetime(2026, 2, 22, 16, 0)
        meta = _extract_from_waypoints([_wp(0.0, 0.0, ts), _wp(0.0, 0.0, ts)])

        assert meta == {"start": (0.0, 0.0), "end": (0.0, 0.0), "date": None}


class TestSameDayDuplicateFallsThroughToIndex:
    """Same-day duplicates whose date suffix would collide fall through to
    a guaranteed-unique index suffix so preflight uniqueness holds."""

    def test_same_day_same_direction_dupes_get_distinct_suffix(self):
        from datetime import date

        from gomoveshift.pipeline._drive_helpers import differentiate_duplicate_ost
        from gomoveshift.pipeline.preflight import run_preflight_ost_check

        locations = ["Kenmare, County Kerry", "Kenmare, County Kerry"]
        meta = [
            {"start": (51.88, -9.58), "end": (51.90, -9.60), "date": date(2026, 2, 22)},
            {"start": (51.88, -9.58), "end": (51.90, -9.60), "date": date(2026, 2, 22)},
        ]
        result = differentiate_duplicate_ost(locations, meta)

        assert result[0] != result[1]
        # Preflight uniqueness must not re-trip on the differentiated titles.
        run_preflight_ost_check(result)
