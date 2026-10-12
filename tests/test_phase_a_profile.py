"""Tests for Phase A RouteProfile integration.

Verifies that Phase A builds per-episode RouteProfiles, stores them in state,
and that derive_drive_ost_text uses profiles when available.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.route_profile import (
    ClipGeocode,
    RouteProfile,
    build_route_profile,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_waypoints(n=100, start_lat=35.0, start_lon=-82.0, end_lat=35.5, end_lon=-81.5):
    """Create a list of mock waypoints with interpolated coordinates."""
    wps = []
    for i in range(n):
        frac = i / max(n - 1, 1)
        wp = MagicMock()
        wp.latitude = start_lat + (end_lat - start_lat) * frac
        wp.longitude = start_lon + (end_lon - start_lon) * frac
        wps.append(wp)
    return wps


def _make_geocoder(
    start_town="Springfield",
    end_town="Shelbyville",
    start_county="Sangamon County",
    end_county="Shelby County",
    start_state="Illinois",
    end_state="Illinois",
    mid_county=None,
):
    """Create a mock geocoder that returns different results by position."""
    geocoder = MagicMock()

    def detailed_side_effect(lat, lon):
        if lat < 35.2:
            return {
                "town": start_town,
                "county": start_county,
                "state": start_state,
            }
        elif lat > 35.4:
            return {
                "town": end_town,
                "county": end_county,
                "state": end_state,
            }
        else:
            return {
                "town": None,
                "county": mid_county or start_county,
                "state": start_state,
            }

    geocoder.reverse_geocode_detailed.side_effect = detailed_side_effect
    return geocoder


def _make_route_analyzer(roads=None, towns=None):
    """Create a mock route analyzer."""
    analyzer = MagicMock()
    analyzer.get_road_segments.return_value = (
        roads
        if roads is not None
        else [
            {"road": "US-150", "distance_km": 25.0},
            {"road": "IL-29", "distance_km": 8.0},
        ]
    )
    analyzer.get_towns_passed_through.return_value = (
        towns
        if towns is not None
        else [
            {
                "name": "Springfield",
                "duration_minutes": 12,
                "place_class": "city",
                "importance": 0.7,
                "entry_latitude": 35.05,
                "entry_longitude": -81.95,
            },
            {
                "name": "Shelbyville",
                "duration_minutes": 10,
                "place_class": "town",
                "importance": 0.5,
                "entry_latitude": 35.45,
                "entry_longitude": -81.55,
            },
        ]
    )
    return analyzer


# ============================================================================
# Phase A Episode Profile Tests
# ============================================================================


class TestPhaseAEpisodeProfiles:
    """Phase A builds episode_profiles and stores them in state."""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_builds_episode_profiles(self, mock_parser_cls):
        """Phase A stores episode_profiles in state as list of dicts."""
        from gomoveshift.pipeline.route_profile import build_route_profile

        waypoints = _make_waypoints(100)
        mock_journey = MagicMock()
        mock_journey.waypoints = waypoints
        mock_parser_cls.return_value.parse.return_value = mock_journey

        geocoder = _make_geocoder()
        analyzer = _make_route_analyzer()

        gpx_files = [Path("/fake/ep0.gpx"), Path("/fake/ep1.gpx")]

        with patch.object(Path, "exists", return_value=True):
            profiles = []
            for i in range(len(gpx_files)):
                profile = build_route_profile(
                    index=i,
                    gpx_files=gpx_files,
                    episode_indices=[i],
                    geocoder=geocoder,
                    route_analyzer=analyzer,
                )
                profiles.append(profile.to_dict())

        assert len(profiles) == 2
        for p in profiles:
            assert isinstance(p, dict)
            assert "start_town" in p
            assert "end_town" in p
            assert "dominant_road" in p
            assert "clips" in p
            assert "states_traversed" in p

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_episode_locations_from_profile_title(self, mock_parser_cls):
        """episode_locations derived from profile.title(), not derive_episode_ost_text."""
        waypoints = _make_waypoints(100)
        mock_journey = MagicMock()
        mock_journey.waypoints = waypoints
        mock_parser_cls.return_value.parse.return_value = mock_journey

        geocoder = _make_geocoder()
        analyzer = _make_route_analyzer()

        gpx_files = [Path("/fake/ep0.gpx")]

        with patch.object(Path, "exists", return_value=True):
            profile = build_route_profile(
                index=0,
                gpx_files=gpx_files,
                episode_indices=[0],
                geocoder=geocoder,
                route_analyzer=analyzer,
            )
            ep_ost = profile.title()

        # Profile with Springfield -> Shelbyville via US-150 should produce
        # "Springfield to Shelbyville via US-150, IL"
        assert "Springfield" in ep_ost
        assert "Shelbyville" in ep_ost

    def test_missing_gpx_produces_empty_profile(self):
        """Missing GPX files produce empty profiles and empty strings."""
        profile = build_route_profile(
            index=0,
            gpx_files=[Path("/nonexistent/ep0.gpx")],
            episode_indices=[0],
            geocoder=MagicMock(),
            route_analyzer=MagicMock(),
        )

        assert profile.start_town is None
        assert profile.end_town is None
        assert profile.title() == ""
        d = profile.to_dict()
        assert isinstance(d, dict)
        assert d["start_town"] is None

    def test_none_gpx_produces_empty_profile(self):
        """None GPX path produces empty profile."""
        profile = build_route_profile(
            index=0,
            gpx_files=[None],
            episode_indices=[0],
            geocoder=MagicMock(),
            route_analyzer=MagicMock(),
        )

        assert profile.start_town is None
        assert profile.title() == ""


# ============================================================================
# derive_drive_ost_text with Profiles
# ============================================================================


class TestDriveOstFromProfiles:
    """derive_drive_ost_text uses profiles when available."""

    def test_drive_ost_from_profiles(self):
        """Drive-level OST uses start of first profile, end of last."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        profiles = [
            RouteProfile(
                start_town="Springfield",
                end_town="Decatur",
                dominant_road="US-150",
                states_traversed=["Illinois"],
            ).to_dict(),
            RouteProfile(
                start_town="Decatur",
                end_town="Champaign",
                dominant_road="I-72",
                states_traversed=["Illinois"],
            ).to_dict(),
        ]

        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx"), Path("/fake/ep2.gpx")],
            episode_profiles=profiles,
        )
        assert result == "Springfield to Champaign"

    def test_drive_ost_same_start_end(self):
        """Same start and end town returns single town."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        profiles = [
            RouteProfile(
                start_town="Springfield",
                end_town="Decatur",
            ).to_dict(),
            RouteProfile(
                start_town="Decatur",
                end_town="Springfield",
            ).to_dict(),
        ]

        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx"), Path("/fake/ep2.gpx")],
            episode_profiles=profiles,
        )
        assert result == "Springfield"

    def test_drive_ost_profiles_only_start(self):
        """Profile with only start_town returns that town."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        profiles = [
            RouteProfile(start_town="Springfield").to_dict(),
        ]

        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx")],
            episode_profiles=profiles,
        )
        assert result == "Springfield"

    def test_drive_ost_profiles_only_end(self):
        """Profile with only end_town returns that town."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        profiles = [
            RouteProfile(end_town="Champaign").to_dict(),
        ]

        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx")],
            episode_profiles=profiles,
        )
        assert result == "Champaign"

    def test_drive_ost_fallback_to_strings(self):
        """Without profiles, falls back to string parsing."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        episode_locations = [
            "Fort Fremont to Longwood via Seaside Road",
            "Cashiers to Highlands via Highlands Road",
        ]
        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx"), Path("/fake/ep2.gpx")],
            episode_locations=episode_locations,
        )
        assert result == "Fort Fremont to Highlands"

    def test_drive_ost_empty_profiles_fallback_to_strings(self):
        """Empty profiles list falls through to string parsing."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        episode_locations = [
            "Springfield to Shelbyville via US-150",
        ]
        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx")],
            episode_profiles=[],
            episode_locations=episode_locations,
        )
        assert result == "Springfield to Shelbyville"

    def test_drive_ost_profiles_with_no_towns(self):
        """Profiles with no towns and no other signal return empty string."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        profiles = [
            RouteProfile(dominant_road="US-150").to_dict(),
        ]

        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx")],
            episode_profiles=profiles,
        )
        assert result == ""

    def test_drive_ost_profiles_no_towns_fall_through_to_episode_locations(self):
        """Profiles present but yielding no endpoint towns must NOT short-circuit
        the ladder — fall through to the priority-2 episode_locations span."""
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        profiles = [
            RouteProfile(dominant_road="US-150").to_dict(),
        ]
        episode_locations = [
            "Fort Fremont to Longwood via Seaside Road",
            "Cashiers to Highlands via Highlands Road",
        ]

        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx"), Path("/fake/ep2.gpx")],
            episode_profiles=profiles,
            episode_locations=episode_locations,
        )
        assert result == "Fort Fremont to Highlands"

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_drive_ost_profiles_no_towns_fall_through_to_state_span(self, mock_parser):
        """Profiles present but town-less fall through to priority-3 GPX
        geocoding, which yields a state-level span for rural endpoints."""
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

        profiles = [
            RouteProfile(dominant_road="US-150").to_dict(),
        ]

        result = derive_drive_ost_text(
            [Path("/fake/ep1.gpx"), Path("/fake/ep2.gpx")],
            episode_profiles=profiles,
            geocoder=mock_geocoder,
        )
        assert result == "South Carolina to North Carolina"
