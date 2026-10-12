"""Tests for route-based OST derivation using RouteAnalyzer.

OST text must use actual road names and towns from GPS data, not
county-level reverse geocoding. Uses the existing RouteAnalyzer from
gps_journey which extracts road segments and towns from waypoints.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDeriveEpisodeOSTFromRoute:
    """derive_episode_ost_text uses RouteAnalyzer for road names + towns."""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_uses_top_road_and_towns(self, mock_parser):
        """OST should include the longest road and start/end towns."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_journey = MagicMock()
        mock_journey.waypoints = [MagicMock() for _ in range(100)]
        mock_parser.return_value.parse.return_value = mock_journey

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Seaside Road", "distance_km": 17.7},
            {"road": "Janette Drive", "distance_km": 2.4},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {"name": "Longwood", "duration_minutes": 10},
        ]

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
        )
        assert "Seaside Road" in result
        assert "Longwood" in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_two_towns_produces_route_format(self, mock_parser):
        """With two towns, OST should be 'Town A to Town B via Road'."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_journey = MagicMock()
        mock_journey.waypoints = [MagicMock() for _ in range(100)]
        mock_parser.return_value.parse.return_value = mock_journey

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Trask Parkway", "distance_km": 21.0},
            {"road": "Charleston Highway", "distance_km": 9.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {"name": "Port Royal", "duration_minutes": 8},
            {"name": "Beaufort", "duration_minutes": 15},
        ]

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
        )
        assert "Port Royal" in result
        assert "Beaufort" in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_no_towns_uses_road_and_state(self, mock_parser):
        """When no towns found, OST should use road name + state."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_journey = MagicMock()
        mock_journey.waypoints = [MagicMock() for _ in range(100)]
        mock_parser.return_value.parse.return_value = mock_journey

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "US-21", "distance_km": 30.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = []

        # Need geocoder fallback for state
        mock_geocoder = MagicMock()
        mock_loc = MagicMock(state="South Carolina")
        mock_geocoder.reverse_geocode.return_value = mock_loc

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert "US-21" in result

    def test_none_gpx_returns_empty(self):
        """None GPX path returns empty string."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        result = derive_episode_ost_text(None)
        assert result == ""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_no_road_segments_falls_back_to_geocoder(self, mock_parser):
        """When RouteAnalyzer returns no roads, fall back to geocoder."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_journey = MagicMock()
        mock_journey.waypoints = [MagicMock(latitude=32.0, longitude=-80.0)]
        mock_parser.return_value.parse.return_value = mock_journey

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = []
        mock_analyzer.get_towns_passed_through.return_value = []

        mock_geocoder = MagicMock()
        mock_loc = MagicMock(
            town="Beaufort",
            city=None,
            village=None,
            state="South Carolina",
            county="Beaufort County",
            country="United States",
        )
        mock_geocoder.reverse_geocode.return_value = mock_loc

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert "Beaufort" in result
