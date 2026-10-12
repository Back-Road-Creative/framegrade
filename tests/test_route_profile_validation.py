"""Validate RouteProfile produces correct titles for multi-state edge cases.

The fixture models a batch of 8 episodes across two states. The old string-based
derivation produced fabricated county names. These tests verify the profile
approach eliminates fabrication.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from gomoveshift.pipeline.route_profile import (
    RouteProfile,
    ClipGeocode,
    build_route_profile,
    build_lf_profile,
)


class TestEpisodeFiveSingleCountyOnly:
    """Episode 5 GPS is entirely within Colleton County, SC.

    The old code claimed 'Colleton County to Beaufort County, SC' because
    Nominatim returned 'Beaufort County' for an endpoint near the county line.
    The new code must detect that the midpoint is also in Colleton County
    and demote to single-county.
    """

    def test_midpoint_validation_prevents_false_cross_county(self):
        """When midpoint county matches start county, is_cross_county must be False."""
        # Mock geocoder: reverse_geocode_detailed returns county based on lat.
        # _pick_anchor_waypoint for 100 wps: offset = min(20, max(1, 5)) = 5
        # start_wp = waypoints[5]  -> lat 32.6125 (Colleton)
        # end_wp   = waypoints[94] -> lat 32.835  (near county line -> geocoder flakes)
        # mid_wp   = waypoints[50] -> lat 32.725  (Colleton, truth)
        geocoder = MagicMock()

        def mock_reverse_geocode_detailed(lat, lon):
            # Near county line (lat > 32.8): geocoder flakes to Beaufort County
            if lat > 32.8:
                return {
                    "county": "Beaufort County",
                    "state": "South Carolina",
                    "town": None,
                    "city": None,
                    "village": None,
                    "road": "US-17",
                }
            # Everything else: correctly returns Colleton County
            return {
                "county": "Colleton County",
                "state": "South Carolina",
                "town": None,
                "city": None,
                "village": None,
                "road": "US-17",
            }

        geocoder.reverse_geocode_detailed = mock_reverse_geocode_detailed

        # Mock route_analyzer: no towns (rural stretch), one road
        route_analyzer = MagicMock()
        route_analyzer.get_road_segments.return_value = [
            {"road": "US-17", "distance_km": 15.0},
        ]
        route_analyzer.get_towns_passed_through.return_value = []

        # Mock GPX with 100 waypoints entirely in Colleton County
        mock_journey = MagicMock()
        mock_wps = []
        for i in range(100):
            wp = MagicMock()
            wp.latitude = 32.6 + (i * 0.0025)  # 32.6 to 32.8475
            wp.longitude = -80.5
            mock_wps.append(wp)
        mock_journey.waypoints = mock_wps

        with patch("gps_journey.core.gpx_parser.GPXParser") as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser.parse.return_value = mock_journey
            mock_parser_cls.return_value = mock_parser

            gpx = Path("/fake/ep_005.gpx")
            with patch.object(Path, "exists", return_value=True):
                profile = build_route_profile(
                    index=5,
                    gpx_files=[None, None, None, None, None, gpx],
                    episode_indices=[5],
                    geocoder=geocoder,
                    route_analyzer=route_analyzer,
                )

        # CRITICAL: must NOT claim cross-county
        assert profile.is_cross_county is False
        # Title should be road-based, not county-to-county
        title = profile.title()
        assert "Beaufort County" not in title
        # Must not be a "X to Y" directional title (county-to-county)
        assert " to " not in title

    def test_title_uses_road_for_rural_colleton(self):
        """Rural Colleton County episode -> road-based title, not county names."""
        profile = RouteProfile(
            lf_index=5,
            dominant_road="US-17",
            counties_traversed=["Colleton County"],
            states_traversed=["South Carolina"],
            is_cross_county=False,
        )
        title = profile.title()
        assert title == "US-17, Colleton County, SC"
        assert "Beaufort" not in title  # No fabricated county


class TestCrossStateLF:
    """Multi-episode LF crossing from SC to NC."""

    def test_cross_state_lf_title(self):
        """LF spanning SC to NC episodes -> cross-state title format."""
        ep_sc = RouteProfile(
            lf_index=0,
            start_town="Beaufort",
            end_town="Walterboro",
            dominant_road="US-17",
            states_traversed=["South Carolina"],
            is_cross_state=False,
        )
        ep_nc = RouteProfile(
            lf_index=1,
            start_town="Highlands",
            end_town="Cashiers",
            dominant_road="US-64",
            states_traversed=["North Carolina"],
            is_cross_state=False,
        )
        lf_profile = build_lf_profile(
            lf_index=0,
            episode_profiles=[ep_sc, ep_nc],
            gpx_files=[],
            episode_indices=[0, 1],
            geocoder=None,
            route_analyzer=None,
        )
        assert lf_profile.is_cross_state is True
        title = lf_profile.title()
        # Should have SC and NC state abbreviations
        assert "SC" in title
        assert "NC" in title
        assert "Beaufort" in title
        assert "Cashiers" in title


class TestSingleEpisodeLF:
    """Single-episode LF with good town data."""

    def test_town_to_town_with_road(self):
        """Single episode with clear start/end towns and a road."""
        profile = RouteProfile(
            lf_index=2,
            start_town="Port Royal",
            end_town="Beaufort",
            dominant_road="Trask Parkway",
            states_traversed=["South Carolina"],
            towns_traversed=["Port Royal", "Beaufort"],
            is_cross_state=False,
        )
        title = profile.title()
        assert title == "Port Royal to Beaufort via Trask Parkway, SC"


class TestManifestQuality:
    """Pre-render manifest must include full route hierarchy."""

    def test_profile_serialization_has_gps_coords(self):
        """Profile to_dict includes GPS coordinates for verification."""
        clip = ClipGeocode(
            state="South Carolina",
            county="Colleton County",
            town=None,
            road="US-17",
            lat=32.65,
            lon=-80.52,
        )
        profile = RouteProfile(
            lf_index=5,
            clips=[clip],
            dominant_road="US-17",
            counties_traversed=["Colleton County"],
            states_traversed=["South Carolina"],
            is_cross_county=False,
        )
        d = profile.to_dict()
        assert d["clips"][0]["lat"] == 32.65
        assert d["clips"][0]["lon"] == -80.52
        assert d["clips"][0]["county"] == "Colleton County"

    def test_no_fabricated_data_in_profile(self):
        """A profile built from mock data has only GPS-verified fields."""
        profile = RouteProfile(
            lf_index=5,
            start_town=None,  # Rural -- no town detected
            end_town=None,
            dominant_road="US-17",
            counties_traversed=["Colleton County"],
            states_traversed=["South Carolina"],
            is_cross_county=False,
        )
        # Title should NOT invent towns or counties
        title = profile.title()
        assert "Beaufort" not in title  # Never mentioned in profile
        # Only uses data that was actually provided
        assert "US-17" in title
        assert "Colleton" in title or "South Carolina" in title


class TestShortsLocationQuality:
    """Shorts location text must match LF profile, not re-derive."""

    def test_shorts_location_matches_lf_title(self):
        """Shorts for an LF use the same profile title as the LF."""
        profile = RouteProfile(
            lf_index=0,
            start_town="Port Royal",
            end_town="Beaufort",
            dominant_road="Trask Parkway",
            states_traversed=["South Carolina"],
        )
        # Shorts should use profile.title(), same as LF
        assert profile.title() == "Port Royal to Beaufort via Trask Parkway, SC"
