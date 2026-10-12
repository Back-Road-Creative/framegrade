"""Tests for drive-level OST span derivation.

Super LF bug: "Longwood to Gneiss via Finch Road" uses the longest road
picked arbitrarily across a multi-region drive. Instead, the drive-level
OST should be 'StartTown to EndTown' derived from the first GPX's first
waypoint and the last GPX's last waypoint (no "via road").
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestDriveLevelSpan:
    """derive_drive_ost_text produces 'StartTown to EndTown' from endpoints."""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_cross_region_uses_start_and_end_towns(self, mock_parser):
        """First-gpx first-waypoint and last-gpx last-waypoint anchor the span."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        first_journey = MagicMock()
        first_journey.waypoints = [
            MagicMock(latitude=32.5, longitude=-80.5),  # Port Royal, SC
            MagicMock(latitude=32.6, longitude=-80.4),
        ]
        last_journey = MagicMock()
        last_journey.waypoints = [
            MagicMock(latitude=35.0, longitude=-83.0),
            MagicMock(latitude=35.1, longitude=-83.2),  # Cashiers, NC
        ]
        # Return different journey per gpx path
        mock_parser.return_value.parse.side_effect = [first_journey, last_journey]

        mock_geocoder = MagicMock()

        def fake_geocode(lat, lon):
            loc = MagicMock()
            if lat < 33:
                loc.town = "Port Royal"
                loc.city = None
                loc.village = None
                loc.state = "South Carolina"
                loc.country = "United States"
            else:
                loc.town = "Cashiers"
                loc.city = None
                loc.village = None
                loc.state = "North Carolina"
                loc.country = "United States"
            return loc

        mock_geocoder.reverse_geocode.side_effect = fake_geocode
        mock_analyzer = MagicMock()  # Not used by new implementation

        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx"), Path("/fake/ep2.gpx")],
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        # No "via road" in drive-level span
        assert " via " not in result
        # Both endpoints appear
        assert "Port Royal" in result
        assert "Cashiers" in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_same_town_fallback_to_single_location(self, mock_parser):
        """When start and end resolve to the same town, report it once."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        journey = MagicMock()
        journey.waypoints = [
            MagicMock(latitude=32.5, longitude=-80.5),
            MagicMock(latitude=32.6, longitude=-80.4),
        ]
        mock_parser.return_value.parse.return_value = journey

        mock_geocoder = MagicMock()
        loc = MagicMock(
            town="Beaufort",
            city=None,
            village=None,
            state="South Carolina",
            country="United States",
        )
        mock_geocoder.reverse_geocode.return_value = loc
        mock_analyzer = MagicMock()

        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx")],
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert result == "Beaufort, SC"
        assert " to " not in result
        assert " via " not in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_missing_geocode_returns_empty(self, mock_parser):
        """If geocoder produces no town, result is empty string."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        journey = MagicMock()
        journey.waypoints = [MagicMock(latitude=0.0, longitude=0.0)]
        mock_parser.return_value.parse.return_value = journey

        mock_geocoder = MagicMock()
        mock_geocoder.reverse_geocode.return_value = None
        mock_analyzer = MagicMock()

        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx")],
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert result == ""

    def test_no_gpx_paths_returns_empty(self):
        """Empty input returns empty string."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        assert derive_drive_ost_text([]) == ""
        assert derive_drive_ost_text([None]) == ""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_leading_stale_run_excluded_from_anchor(self, mock_parser):
        """A leading null-island cold-start run must not win the anchor.

        ``_pick_anchor_waypoint`` only steps ~20 fixes inward, so a long stale
        head run (hundreds of (0,0) cold-start fixes) still owns the anchor and
        burns a wrong drive-wide label. The raw trace must be quality-filtered
        before the anchor pick so the anchor lands on a real drive point.
        """
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        # 30 null-island cold-start fixes, then the real SC drive.
        stale = [SimpleNamespace(latitude=0.0, longitude=0.0) for _ in range(30)]
        real = [SimpleNamespace(latitude=32.40 + i * 0.001, longitude=-80.50) for i in range(30)]
        first_journey = SimpleNamespace(waypoints=stale + real)
        last_journey = SimpleNamespace(waypoints=list(real))
        mock_parser.return_value.parse.side_effect = [first_journey, last_journey]

        seen: list[tuple[float, float]] = []

        def fake_geocode(lat, lon):
            seen.append((lat, lon))
            loc = SimpleNamespace(town=None, city=None, village=None, state=None, country=None)
            if abs(lat) < 0.001 and abs(lon) < 0.001:
                loc.town = "NullIsland"
                loc.state = "Nowhere"
            else:
                loc.town = "Beaufort"
                loc.state = "South Carolina"
            return loc

        mock_geocoder = SimpleNamespace(reverse_geocode=fake_geocode)

        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx"), Path("/fake/ep2.gpx")],
            geocoder=mock_geocoder,
        )
        # The anchor is a real drive point, never a null-island stale fix.
        assert (0.0, 0.0) not in seen
        assert "NullIsland" not in result
        assert "Beaufort" in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_rural_endpoints_fall_back_to_state_span(self, mock_parser):
        """Edge case: endpoints on rural roads have no town but do
        have a state. Fall back to 'StartState to EndState'."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        first_journey = MagicMock()
        first_journey.waypoints = [MagicMock(latitude=32.4, longitude=-80.5)]
        last_journey = MagicMock()
        last_journey.waypoints = [MagicMock(latitude=35.2, longitude=-83.3)]
        mock_parser.return_value.parse.side_effect = [first_journey, last_journey]

        mock_geocoder = MagicMock()

        def fake_geocode(lat, lon):
            loc = MagicMock()
            loc.town = None
            loc.city = None
            loc.village = None
            loc.state = "South Carolina" if lat < 33 else "North Carolina"
            loc.country = "United States"
            return loc

        mock_geocoder.reverse_geocode.side_effect = fake_geocode
        mock_analyzer = MagicMock()

        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx"), Path("/fake/ep2.gpx")],
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert result == "South Carolina to North Carolina"

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_rural_same_state_falls_back_to_state_only(self, mock_parser):
        """Both endpoints rural, same state → 'South Carolina'."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        journey = MagicMock()
        journey.waypoints = [MagicMock(latitude=32.4, longitude=-80.5)]
        mock_parser.return_value.parse.return_value = journey

        mock_geocoder = MagicMock()
        loc = MagicMock(
            town=None,
            city=None,
            village=None,
            state="South Carolina",
            country="United States",
        )
        mock_geocoder.reverse_geocode.return_value = loc
        mock_analyzer = MagicMock()

        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx")],
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert result == "South Carolina"
