"""Tests for OST state-abbreviation append in road+town case.

The bug: `_format_ost_from_route` returns "Seaside Road, Longwood" with no
state context — viewers can't locate the drive. The road+town shape must
append the US state abbreviation when available.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestOSTStateAppend:
    """State abbreviation appended to road+town OST output."""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_road_plus_single_town_appends_state_abbrev(self, mock_parser):
        """Road+town shape: 1 town + top road → '{road}, {town}, {state_abbrev}'."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_journey = MagicMock()
        mock_wp = MagicMock(latitude=32.5, longitude=-80.5)
        mock_journey.waypoints = [mock_wp] * 100
        mock_parser.return_value.parse.return_value = mock_journey

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Seaside Road", "distance_km": 17.7},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {
                "name": "Longwood",
                "duration_minutes": 10,
                "place_class": "village",
                "importance": 0.3,
            },
        ]

        mock_geocoder = MagicMock()
        mock_loc = MagicMock(state="South Carolina", country="United States")
        mock_geocoder.reverse_geocode.return_value = mock_loc

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert result == "Seaside Road, Longwood, SC"

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_non_us_road_plus_town_no_abbrev(self, mock_parser):
        """Non-US drives: no abbreviation (falls back to current behavior)."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_journey = MagicMock()
        mock_wp = MagicMock(latitude=55.2, longitude=-7.8)
        mock_journey.waypoints = [mock_wp] * 100
        mock_parser.return_value.parse.return_value = mock_journey

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "N56", "distance_km": 22.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {
                "name": "Downings",
                "duration_minutes": 8,
                "place_class": "village",
                "importance": 0.3,
            },
        ]

        mock_geocoder = MagicMock()
        mock_loc = MagicMock(state="Ulster", country="Ireland")
        mock_geocoder.reverse_geocode.return_value = mock_loc

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        # No US state abbrev — original "road, town" shape
        assert result == "N56, Downings"

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_two_town_route_format_unchanged(self, mock_parser):
        """Route shape 'A to B via road' is NOT affected by state append."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_journey = MagicMock()
        mock_wp = MagicMock(latitude=32.5, longitude=-80.5)
        mock_journey.waypoints = [mock_wp] * 100
        mock_parser.return_value.parse.return_value = mock_journey

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Trask Parkway", "distance_km": 21.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {"name": "Port Royal", "duration_minutes": 8, "place_class": "town", "importance": 0.5},
            {"name": "Beaufort", "duration_minutes": 15, "place_class": "city", "importance": 0.7},
        ]

        mock_geocoder = MagicMock()
        mock_loc = MagicMock(state="South Carolina", country="United States")
        mock_geocoder.reverse_geocode.return_value = mock_loc

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        # Route format unchanged — state abbrev does NOT apply here
        assert " to " in result
        assert "SC" not in result
        assert result == "Port Royal to Beaufort via Trask Parkway"

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_state_abbrev_all_fifty_us_states(self, mock_parser):
        """_state_abbrev covers all 50 US states + DC."""
        from gomoveshift.pipeline.ingest import _state_abbrev

        expected_samples = {
            "Alabama": "AL",
            "California": "CA",
            "Florida": "FL",
            "New York": "NY",
            "Texas": "TX",
            "South Carolina": "SC",
            "North Carolina": "NC",
            "Wyoming": "WY",
            "District of Columbia": "DC",
        }
        for full_name, abbrev in expected_samples.items():
            assert _state_abbrev(full_name) == abbrev

    def test_state_abbrev_unknown_returns_none(self):
        """Non-US / unknown state name returns None."""
        from gomoveshift.pipeline.ingest import _state_abbrev

        assert _state_abbrev("Ulster") is None
        assert _state_abbrev("") is None
        assert _state_abbrev(None) is None
