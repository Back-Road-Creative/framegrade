"""Tests for RouteProfile data model, builder, and title generation.

Tests mock external APIs (geocoder, route_analyzer, GPXParser) but NOT
internal functions from ost_formatter.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.route_profile import (
    ClipGeocode,
    RouteProfile,
    _dominant_by_distance,
    build_lf_profile,
    build_route_profile,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_waypoint(lat=35.0, lon=-82.0):
    wp = MagicMock()
    wp.latitude = lat
    wp.longitude = lon
    return wp


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
        # Use latitude to distinguish start/mid/end
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


def _make_route_analyzer(
    roads=None,
    towns=None,
):
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
# Serialization Tests
# ============================================================================


class TestClipGeocodeSerialization:
    def test_round_trip(self):
        clip = ClipGeocode(
            country="US",
            state="Illinois",
            county="Sangamon County",
            town="Springfield",
            road="US-150",
            poi=None,
            lat=39.7817,
            lon=-89.6501,
        )
        from dataclasses import asdict

        d = asdict(clip)
        restored = ClipGeocode(**d)
        assert restored == clip

    def test_defaults(self):
        clip = ClipGeocode()
        assert clip.country is None
        assert clip.lat == 0.0
        assert clip.lon == 0.0


class TestRouteProfileSerialization:
    def test_round_trip(self):
        clip = ClipGeocode(state="Illinois", town="Springfield", lat=39.78, lon=-89.65)
        profile = RouteProfile(
            lf_index=1,
            clips=[clip],
            start_town="Springfield",
            end_town="Shelbyville",
            dominant_road="US-150",
            states_traversed=["Illinois"],
            counties_traversed=["Sangamon County", "Shelby County"],
            towns_traversed=["Springfield", "Shelbyville"],
            roads_traversed=["US-150", "IL-29"],
            features=["Lake Springfield"],
            is_cross_state=False,
            is_cross_county=True,
        )
        d = profile.to_dict()
        restored = RouteProfile.from_dict(d)
        assert restored.lf_index == 1
        assert restored.start_town == "Springfield"
        assert restored.end_town == "Shelbyville"
        assert restored.dominant_road == "US-150"
        assert len(restored.clips) == 1
        assert restored.clips[0].state == "Illinois"
        assert restored.is_cross_county is True
        assert restored.features == ["Lake Springfield"]

    def test_from_dict_empty(self):
        profile = RouteProfile.from_dict({})
        assert profile.lf_index == 0
        assert profile.clips == []
        assert profile.start_town is None
        assert profile.is_cross_state is False


# ============================================================================
# Title Tests — 7-Shape Priority Chain
# ============================================================================


class TestRouteProfileTitle:
    def test_shape1_two_towns_and_road(self):
        """Shape 1: Town A to Town B via Road, ST"""
        p = RouteProfile(
            start_town="Springfield",
            end_town="Shelbyville",
            dominant_road="US-150",
            states_traversed=["Illinois"],
        )
        assert p.title() == "Springfield to Shelbyville via US-150, IL"

    def test_shape2_two_towns_no_road(self):
        """Shape 2: Town A to Town B, ST"""
        p = RouteProfile(
            start_town="Springfield",
            end_town="Shelbyville",
            states_traversed=["Illinois"],
        )
        assert p.title() == "Springfield to Shelbyville, IL"

    def test_shape2_two_towns_no_state(self):
        """Shape 2 without state: Town A to Town B"""
        p = RouteProfile(
            start_town="Springfield",
            end_town="Shelbyville",
        )
        assert p.title() == "Springfield to Shelbyville"

    def test_shape3_feature_and_road(self):
        """Shape 3: Feature via Road, ST"""
        p = RouteProfile(
            features=["Lake Springfield"],
            dominant_road="US-150",
            states_traversed=["Illinois"],
        )
        assert p.title() == "Lake Springfield via US-150, IL"

    def test_shape4_one_town_and_road(self):
        """Shape 4: Road, Town, ST"""
        p = RouteProfile(
            start_town="Springfield",
            dominant_road="US-150",
            states_traversed=["Illinois"],
        )
        assert p.title() == "US-150, Springfield, IL"

    def test_shape4_end_town_only(self):
        """Shape 4 with only end_town: Road, Town, ST"""
        p = RouteProfile(
            end_town="Shelbyville",
            dominant_road="US-150",
            states_traversed=["Illinois"],
        )
        assert p.title() == "US-150, Shelbyville, IL"

    def test_shape5_road_and_county(self):
        """Shape 5: Road, County, ST"""
        p = RouteProfile(
            dominant_road="US-150",
            counties_traversed=["Sangamon County"],
            states_traversed=["Illinois"],
        )
        assert p.title() == "US-150, Sangamon County, IL"

    def test_shape6_cross_county(self):
        """Shape 6: County1 to County2, ST"""
        p = RouteProfile(
            counties_traversed=["Sangamon County", "Shelby County"],
            states_traversed=["Illinois"],
            is_cross_county=True,
        )
        assert p.title() == "Sangamon County to Shelby County, IL"

    def test_shape6_rejected_not_cross_county(self):
        """Shape 6 rejected: is_cross_county=False falls through."""
        p = RouteProfile(
            counties_traversed=["Sangamon County", "Shelby County"],
            states_traversed=["Illinois"],
            is_cross_county=False,
        )
        # No road, no town, no features — empty
        assert p.title() == ""

    def test_shape7_road_and_state(self):
        """Shape 7: Road, StateName"""
        p = RouteProfile(
            dominant_road="US-150",
            states_traversed=["Illinois"],
        )
        assert p.title() == "US-150, Illinois"

    def test_empty_profile(self):
        """Empty profile returns empty string."""
        assert RouteProfile().title() == ""

    def test_cross_state(self):
        """Cross-state: Town, ST1 to Town, ST2"""
        p = RouteProfile(
            start_town="Beaufort",
            end_town="Cashiers",
            states_traversed=["South Carolina", "North Carolina"],
            is_cross_state=True,
        )
        assert p.title() == "Beaufort, SC to Cashiers, NC"

    def test_cross_state_non_us(self):
        """Cross-state with non-US states: no abbreviation."""
        p = RouteProfile(
            start_town="Toronto",
            end_town="Montreal",
            states_traversed=["Ontario", "Quebec"],
            is_cross_state=True,
        )
        assert p.title() == "Toronto, Ontario to Montreal, Quebec"

    def test_roundtrip_same_start_end(self):
        """Roundtrip with road: start==end -> shape 4 'Road, Town, ST'"""
        p = RouteProfile(
            start_town="Springfield",
            end_town="Springfield",
            dominant_road="US-150",
            states_traversed=["Illinois"],
        )
        assert p.title() == "US-150, Springfield, IL"

    def test_roundtrip_no_state(self):
        """Roundtrip without state."""
        p = RouteProfile(
            start_town="Springfield",
            end_town="Springfield",
        )
        assert p.title() == "Springfield"

    def test_single_town_no_road_with_state(self):
        """Fallback: single town with state but no road."""
        p = RouteProfile(
            start_town="Springfield",
            states_traversed=["Illinois"],
        )
        # Shape 4 requires road, so falls through to the end fallback
        assert p.title() == "Springfield, IL"


# ============================================================================
# build_route_profile Tests
# ============================================================================


class TestBuildRouteProfile:
    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_single_episode_two_towns(self, mock_parser_cls):
        """Single episode with two detected towns produces a profile."""
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

        assert profile.start_town == "Springfield"
        assert profile.end_town == "Shelbyville"
        assert profile.dominant_road == "US-150"
        assert "US-150" in profile.roads_traversed
        assert "IL-29" in profile.roads_traversed
        assert "Springfield" in profile.towns_traversed
        assert "Shelbyville" in profile.towns_traversed

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_far_detected_endpoint_town_dropped(self, mock_parser_cls):
        """A detected endpoint town >50 km from its anchor waypoint is dropped.

        RouteAnalyzer sampled "Lake Lure" (NC, ~340 km from the
        coastal-SC start waypoint) as the first town; without reconciliation it
        became ``start_town`` and borrowed the SC start anchor's state. The
        distance guard drops it and falls back to the waypoint anchor.
        """
        waypoints = _make_waypoints(
            n=50, start_lat=32.40, start_lon=-80.45, end_lat=32.55, end_lon=-80.55
        )
        mock_journey = MagicMock()
        mock_journey.waypoints = waypoints
        mock_parser_cls.return_value.parse.return_value = mock_journey

        # The start waypoint geocodes to Beaufort, the end waypoint to Frogmore
        # (the detected end town sits at its anchor and is the same place).
        geocoder = MagicMock()
        geocoder.reverse_geocode_detailed.side_effect = lambda lat, lon: {
            "town": "Beaufort" if lat < 32.5 else "Frogmore",
            "county": "Beaufort County",
            "state": "South Carolina",
        }
        analyzer = _make_route_analyzer(
            roads=[{"road": "Sea Island Parkway", "distance_km": 12.0}],
            towns=[
                {
                    "name": "Lake Lure",
                    "place_class": "town",
                    "importance": 0.6,
                    "duration_minutes": 10,
                    "entry_latitude": 35.43,
                    "entry_longitude": -81.90,
                },
                {
                    "name": "Frogmore",
                    "place_class": "village",
                    "importance": 0.4,
                    "duration_minutes": 8,
                    "entry_latitude": 32.50,
                    "entry_longitude": -80.50,
                },
            ],
        )

        gpx_files = [Path("/fake/ep0.gpx")]
        with patch.object(Path, "exists", return_value=True):
            profile = build_route_profile(
                index=0,
                gpx_files=gpx_files,
                episode_indices=[0],
                geocoder=geocoder,
                route_analyzer=analyzer,
            )

        assert profile.start_town != "Lake Lure"
        assert profile.start_town == "Beaufort"  # fell back to waypoint anchor
        assert profile.end_town == "Frogmore"  # near its anchor → kept

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_single_episode_no_towns_rural(self, mock_parser_cls):
        """Single episode with no towns produces road-only profile."""
        waypoints = _make_waypoints(100)
        mock_journey = MagicMock()
        mock_journey.waypoints = waypoints
        mock_parser_cls.return_value.parse.return_value = mock_journey

        # Geocoder returns no town/city/village — only county and state
        geocoder = MagicMock()
        geocoder.reverse_geocode_detailed.return_value = {
            "county": "Buncombe County",
            "state": "North Carolina",
        }
        analyzer = _make_route_analyzer(
            towns=[],
            roads=[{"road": "Blue Ridge Parkway", "distance_km": 40.0}],
        )

        gpx_files = [Path("/fake/ep0.gpx")]
        with patch.object(Path, "exists", return_value=True):
            profile = build_route_profile(
                index=0,
                gpx_files=gpx_files,
                episode_indices=[0],
                geocoder=geocoder,
                route_analyzer=analyzer,
            )

        assert profile.start_town is None
        assert profile.end_town is None
        assert profile.dominant_road == "Blue Ridge Parkway"

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_cross_state_route(self, mock_parser_cls):
        """Cross-state route sets is_cross_state and both states."""
        waypoints = _make_waypoints(100)
        mock_journey = MagicMock()
        mock_journey.waypoints = waypoints
        mock_parser_cls.return_value.parse.return_value = mock_journey

        geocoder = _make_geocoder(
            start_town="Beaufort",
            end_town="Cashiers",
            start_state="South Carolina",
            end_state="North Carolina",
            start_county="Beaufort County",
            end_county="Jackson County",
        )
        analyzer = _make_route_analyzer(
            towns=[
                {
                    "name": "Beaufort",
                    "duration_minutes": 10,
                    "place_class": "town",
                    "importance": 0.6,
                    "entry_latitude": 35.05,
                    "entry_longitude": -81.95,
                },
                {
                    "name": "Cashiers",
                    "duration_minutes": 8,
                    "place_class": "town",
                    "importance": 0.4,
                    "entry_latitude": 35.45,
                    "entry_longitude": -81.55,
                },
            ],
        )

        gpx_files = [Path("/fake/ep0.gpx")]
        with patch.object(Path, "exists", return_value=True):
            profile = build_route_profile(
                index=0,
                gpx_files=gpx_files,
                episode_indices=[0],
                geocoder=geocoder,
                route_analyzer=analyzer,
            )

        assert profile.is_cross_state is True
        assert "South Carolina" in profile.states_traversed
        assert "North Carolina" in profile.states_traversed

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_midpoint_demotes_false_cross_county(self, mock_parser_cls):
        """Midpoint geocode matching start county demotes cross-county."""
        waypoints = _make_waypoints(100)
        mock_journey = MagicMock()
        mock_journey.waypoints = waypoints
        mock_parser_cls.return_value.parse.return_value = mock_journey

        # Geocoder reports different counties for start/end but midpoint
        # matches start — a flake. mid_county defaults to start_county.
        geocoder = _make_geocoder(
            start_town="Springfield",
            end_town="Shelbyville",
            start_county="Sangamon County",
            end_county="Shelby County",
            mid_county="Sangamon County",  # midpoint matches start
        )
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

        assert profile.is_cross_county is False

    def test_missing_gpx_file_returns_empty(self):
        """Missing GPX file produces an empty profile gracefully."""
        geocoder = MagicMock()
        analyzer = MagicMock()

        profile = build_route_profile(
            index=0,
            gpx_files=[Path("/nonexistent/ep0.gpx")],
            episode_indices=[0],
            geocoder=geocoder,
            route_analyzer=analyzer,
        )

        assert profile.lf_index == 0
        assert profile.start_town is None
        assert profile.clips == []

    def test_episode_index_out_of_range(self):
        """Episode index beyond gpx_files produces an empty profile."""
        profile = build_route_profile(
            index=0,
            gpx_files=[],
            episode_indices=[5],
            geocoder=MagicMock(),
            route_analyzer=MagicMock(),
        )
        assert profile.start_town is None


# ============================================================================
# build_lf_profile Tests
# ============================================================================


class TestBuildLfProfile:
    def test_aggregates_episode_profiles(self):
        """LF profile merges start/end from first/last episodes."""
        ep1 = RouteProfile(
            lf_index=0,
            start_town="Springfield",
            end_town="Decatur",
            dominant_road="US-150",
            states_traversed=["Illinois"],
            counties_traversed=["Sangamon County"],
            towns_traversed=["Springfield", "Decatur"],
            roads_traversed=["US-150", "IL-48"],
        )
        ep2 = RouteProfile(
            lf_index=0,
            start_town="Decatur",
            end_town="Champaign",
            dominant_road="I-72",
            states_traversed=["Illinois"],
            counties_traversed=["Macon County", "Champaign County"],
            towns_traversed=["Decatur", "Champaign"],
            roads_traversed=["I-72", "US-150"],
        )

        lf = build_lf_profile(
            lf_index=0,
            episode_profiles=[ep1, ep2],
            gpx_files=[],
            episode_indices=[0, 1],
            geocoder=MagicMock(),
            route_analyzer=MagicMock(),
        )

        assert lf.start_town == "Springfield"
        assert lf.end_town == "Champaign"
        # Roads merged, deduplicated, order preserved
        assert "US-150" in lf.roads_traversed
        assert "IL-48" in lf.roads_traversed
        assert "I-72" in lf.roads_traversed
        # Towns merged
        assert lf.towns_traversed == ["Springfield", "Decatur", "Champaign"]
        # Counties merged
        assert "Sangamon County" in lf.counties_traversed
        assert "Champaign County" in lf.counties_traversed

    def test_cross_state_across_episodes(self):
        """Cross-state detected when episodes span different states."""
        ep1 = RouteProfile(
            start_town="Beaufort",
            end_town="Greenville",
            states_traversed=["South Carolina"],
            counties_traversed=["Beaufort County"],
        )
        ep2 = RouteProfile(
            start_town="Greenville",
            end_town="Asheville",
            states_traversed=["North Carolina"],
            counties_traversed=["Buncombe County"],
        )

        lf = build_lf_profile(
            lf_index=0,
            episode_profiles=[ep1, ep2],
            gpx_files=[],
            episode_indices=[0, 1],
            geocoder=MagicMock(),
            route_analyzer=MagicMock(),
        )

        assert lf.is_cross_state is True
        assert lf.states_traversed == ["South Carolina", "North Carolina"]

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_fallback_to_build_route_profile(self, mock_parser_cls):
        """Empty episode_profiles falls back to build_route_profile."""
        waypoints = _make_waypoints(100)
        mock_journey = MagicMock()
        mock_journey.waypoints = waypoints
        mock_parser_cls.return_value.parse.return_value = mock_journey

        geocoder = _make_geocoder()
        analyzer = _make_route_analyzer()

        gpx_files = [Path("/fake/ep0.gpx")]
        with patch.object(Path, "exists", return_value=True):
            lf = build_lf_profile(
                lf_index=0,
                episode_profiles=[],
                gpx_files=gpx_files,
                episode_indices=[0],
                geocoder=geocoder,
                route_analyzer=analyzer,
            )

        assert lf.start_town == "Springfield"

    def test_three_episode_road_dominance(self):
        """Dominant road comes from the most-weighted across episodes."""
        ep1 = RouteProfile(
            dominant_road="US-150",
            roads_traversed=["US-150", "IL-48"],
        )
        ep2 = RouteProfile(
            dominant_road="I-72",
            roads_traversed=["I-72"],
        )
        ep3 = RouteProfile(
            dominant_road="I-72",
            roads_traversed=["I-72", "US-45"],
        )

        lf = build_lf_profile(
            lf_index=0,
            episode_profiles=[ep1, ep2, ep3],
            gpx_files=[],
            episode_indices=[0, 1, 2],
            geocoder=MagicMock(),
            route_analyzer=MagicMock(),
        )

        # I-72 appears as dominant_road in 2 episodes (2*10 + count bonus)
        assert lf.dominant_road == "I-72"

    def test_no_dominant_road_below_50_percent_plurality(self):
        """A road that isn't a >50% plurality of the merged weight must
        NOT ship as dominant (mirrors build_route_profile's >50% guard).
        Three equally-weighted roads → top holds 1/3 < 50% → no via-road."""
        ep = RouteProfile(roads_traversed=["Road A", "Road B", "Road C"])
        lf = build_lf_profile(
            lf_index=0,
            episode_profiles=[ep],
            gpx_files=[],
            episode_indices=[0],
            geocoder=MagicMock(),
            route_analyzer=MagicMock(),
        )
        assert lf.dominant_road is None

    def test_dominant_road_declared_above_50_percent(self):
        """A road holding >50% of the merged weight is still declared dominant."""
        ep = RouteProfile(dominant_road="Big Road", roads_traversed=["Big Road", "Small Road"])
        lf = build_lf_profile(
            lf_index=0,
            episode_profiles=[ep],
            gpx_files=[],
            episode_indices=[0],
            geocoder=MagicMock(),
            route_analyzer=MagicMock(),
        )
        assert lf.dominant_road == "Big Road"


class TestDominantByDistanceTieBreak:
    """The dominant-road pick must be deterministic and direction-independent.
    On a distance tie, fall back to alphabetical road name, never dict
    insertion order (= drive direction)."""

    def test_tie_breaks_alphabetically_either_order(self):
        forward = {"Sea Island Parkway": 10.0, "Coastal Highway": 10.0}
        reverse = {"Coastal Highway": 10.0, "Sea Island Parkway": 10.0}
        assert _dominant_by_distance(forward) == "Coastal Highway"
        assert _dominant_by_distance(reverse) == "Coastal Highway"

    def test_higher_distance_wins_over_alpha(self):
        roads = {"Zephyr Road": 30.0, "Alpha Avenue": 10.0}
        assert _dominant_by_distance(roads) == "Zephyr Road"

    def test_empty_returns_none(self):
        assert _dominant_by_distance({}) is None


# ============================================================================
# Endpoint demotions + road member-credit (share build_lf_title's ladder)
# ============================================================================


class TestEndpointDemotions:
    """The episode/drive OST ladder must apply the same township/county/
    region endpoint demotions ``build_lf_title`` does. A rural fallback endpoint
    (township / bare county / broad region the geocoder returns when no
    settlement resolves) must NOT ship as an "A to B" town — it demotes up the
    ladder to a road/county/state shape instead."""

    def test_township_endpoints_demote_up_ladder(self):
        # Once shipped as "Liberty Township to Cumberland Township, PA".
        p = RouteProfile(
            start_town="Liberty Township",
            end_town="Cumberland Township",
            dominant_road="US-30",
            counties_traversed=["Adams County"],
            states_traversed=["Pennsylvania"],
        )
        t = p.title()
        assert "Township" not in t
        assert t == "US-30, Adams County, PA"

    def test_county_shaped_start_demoted(self):
        # A county-shaped endpoint must not win the two-town shape over the
        # real settlement on the other side.
        p = RouteProfile(
            start_town="Beaufort County",
            end_town="Frogmore",
            dominant_road="US-21",
            counties_traversed=["Beaufort County"],
            states_traversed=["South Carolina"],
        )
        t = p.title()
        assert "County" not in t
        assert t == "US-21, Frogmore, SC"

    def test_broad_region_endpoints_demoted(self):
        # Wildlife-area drives once burned the ambient "ACE Basin".
        p = RouteProfile(
            start_town="ACE Basin",
            end_town="ACE Basin",
            dominant_road="US-17",
            counties_traversed=["Colleton County"],
            states_traversed=["South Carolina"],
        )
        t = p.title()
        assert "Basin" not in t
        assert t == "US-17, Colleton County, SC"

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_source_nulls_township_endpoint(self, mock_parser_cls):
        """build_route_profile nulls a township endpoint at the SOURCE so the
        drive-level OST (which reads start_town/end_town directly, bypassing
        title()) never ships it either."""
        waypoints = _make_waypoints(100)
        mock_journey = MagicMock()
        mock_journey.waypoints = waypoints
        mock_parser_cls.return_value.parse.return_value = mock_journey

        geocoder = MagicMock()
        geocoder.reverse_geocode_detailed.return_value = {
            "town": "Cumberland Township",
            "county": "Adams County",
            "state": "Pennsylvania",
        }
        analyzer = _make_route_analyzer(
            towns=[],
            roads=[{"road": "US-30", "distance_km": 20.0}],
        )

        gpx_files = [Path("/fake/ep0.gpx")]
        with patch.object(Path, "exists", return_value=True):
            profile = build_route_profile(
                index=0,
                gpx_files=gpx_files,
                episode_indices=[0],
                geocoder=geocoder,
                route_analyzer=analyzer,
            )

        assert profile.start_town is None
        assert profile.end_town is None


class TestRoadMemberCredit:
    """``roads_by_distance`` must credit EVERY member of an OSM
    concurrency ref so I-81's distance isn't split across "I 81",
    "I 81;US 11", "I 77;I 81;US 11;US 52" variants and lose the >50%
    distance plurality (member credit once never reached this site)."""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_i81_concurrency_variants_win_plurality(self, mock_parser_cls):
        waypoints = _make_waypoints(100)
        mock_journey = MagicMock()
        mock_journey.waypoints = waypoints
        mock_parser_cls.return_value.parse.return_value = mock_journey

        geocoder = _make_geocoder()
        # Every segment carries I-81, joined with different concurrency
        # partners in OSM's ref. Raw-keyed, I-81's top variant holds only
        # 20/72 < 50% so NO road ships; member-credited, I-81 holds 60/72.
        analyzer = _make_route_analyzer(
            roads=[
                {"road": "I 81", "distance_km": 20.0},
                {"road": "I 81;US 11", "distance_km": 15.0},
                {"road": "I 77;I 81", "distance_km": 10.0},
                {"road": "I 77;I 81;US 11;US 52", "distance_km": 8.0},
                {"road": "I 81;US 52", "distance_km": 7.0},
                {"road": "US 11", "distance_km": 12.0},
            ],
        )

        gpx_files = [Path("/fake/ep0.gpx")]
        with patch.object(Path, "exists", return_value=True):
            profile = build_route_profile(
                index=0,
                gpx_files=gpx_files,
                episode_indices=[0],
                geocoder=geocoder,
                route_analyzer=analyzer,
            )

        assert profile.dominant_road == "I-81"
        assert "I-81" in profile.roads_traversed


# ============================================================================
# Road-slot quality pass (junk local names must not win on plurality)
# ============================================================================


def _profile_for_roads(roads):
    """Build an episode profile for ``roads`` segments with a townless
    Seven Mile Ford geocode."""
    waypoints = _make_waypoints(100)
    mock_journey = MagicMock()
    mock_journey.waypoints = waypoints
    geocoder = MagicMock()
    geocoder.reverse_geocode_detailed.return_value = {
        "town": "Seven Mile Ford",
        "county": "Smyth County",
        "state": "Virginia",
    }
    analyzer = _make_route_analyzer(roads=roads, towns=[])
    with patch("gps_journey.core.gpx_parser.GPXParser") as mock_parser_cls:
        mock_parser_cls.return_value.parse.return_value = mock_journey
        with patch.object(Path, "exists", return_value=True):
            return build_route_profile(
                index=0,
                gpx_files=[Path("/fake/ep0.gpx")],
                episode_indices=[0],
                geocoder=geocoder,
                route_analyzer=analyzer,
            )


class TestRoadSlotQuality:
    """RouteAnalyzer segments carry no OSM place_class, so the road slot
    is gated on the signals the video side has: NAME quality (route-ref /
    curated byway / through-road suffix) at the existing >50% bar, and for an
    UNCLASSED name a strong dominance floor (share + absolute km). The Shape-5
    named-road contract holds: a legitimately named road with real dominance
    stays eligible — a hard name-only filter was explicitly rejected."""

    def test_junk_road_bare_plurality_rejected(self):
        """ "Avenue John Brown" at a bare 60% plurality of a short in-town
        drive must NOT take the road slot ("Avenue John Brown, Seven Mile
        Ford, VA" — a junk shape that once shipped)."""
        profile = _profile_for_roads(
            [
                {"road": "Avenue John Brown", "distance_km": 1.8},
                {"road": "Pine Street", "distance_km": 1.2},
            ]
        )
        assert profile.dominant_road is None
        assert "Avenue John Brown" not in profile.title()

    def test_junk_road_supermajority_below_km_floor_rejected(self):
        """An unclassed name owning ~91% of a tiny 2.2 km profile is still a
        street, not a route — the absolute km floor rejects it."""
        profile = _profile_for_roads(
            [
                {"road": "Avenue John Brown", "distance_km": 2.0},
                {"road": "Pine Street", "distance_km": 0.2},
            ]
        )
        assert profile.dominant_road is None

    def test_route_ref_keeps_bare_plurality(self):
        """A route-ref road keeps the existing >50% bar — no new floors."""
        profile = _profile_for_roads(
            [
                {"road": "US 11", "distance_km": 1.8},
                {"road": "Pine Street", "distance_km": 1.2},
            ]
        )
        assert profile.dominant_road == "US-11"

    def test_unclassed_named_road_with_real_dominance_still_wins(self):
        """Shape-5 named-road contract: an unclassed but legitimately named
        road that truly owns the drive (>=75% share, >=5 km) stays eligible."""
        profile = _profile_for_roads(
            [
                {"road": "Military Road", "distance_km": 20.0},
                {"road": "Pine Street", "distance_km": 2.0},
            ]
        )
        assert profile.dominant_road == "Military Road"

    def test_lf_merge_junk_roads_traversed_only_rejected(self):
        """LF merge weights are appearance counts (no distances): an unclassed
        name arriving via roads_traversed alone — never any episode's gated
        dominant_road — must not take the LF road slot at 100% weight."""
        ep = RouteProfile(lf_index=0, roads_traversed=["Avenue John Brown"])
        lf = build_lf_profile(
            lf_index=0,
            episode_profiles=[ep],
            gpx_files=[],
            episode_indices=[0],
            geocoder=MagicMock(),
            route_analyzer=MagicMock(),
        )
        assert lf.dominant_road is None

    def test_lf_merge_unclassed_episode_dominant_survives(self):
        """An unclassed name that cleared an EPISODE's distance-based gate
        (it is that episode's dominant_road) stays eligible at the LF level."""
        ep = RouteProfile(
            lf_index=0,
            dominant_road="Military Road",
            roads_traversed=["Military Road"],
        )
        lf = build_lf_profile(
            lf_index=0,
            episode_profiles=[ep],
            gpx_files=[],
            episode_indices=[0],
            geocoder=MagicMock(),
            route_analyzer=MagicMock(),
        )
        assert lf.dominant_road == "Military Road"


# ============================================================================
# Endpoint names come from the GPS start/end, not a farther detected town
# ============================================================================


def _anchor_geocoder(start, end):
    """Geocoder whose detailed result depends on which half of the track the
    point is in (start/end are dicts of the fields the real geocoder returns).
    """
    geocoder = MagicMock()

    def detailed(lat, lon):
        return dict(start if lat < 36.38 else end)

    geocoder.reverse_geocode_detailed.side_effect = detailed
    return geocoder


def _build_profile(geocoder, towns):
    mock_journey = MagicMock()
    mock_journey.waypoints = _make_waypoints(
        n=100, start_lat=36.21, start_lon=-82.35, end_lat=36.55, end_lon=-82.55
    )
    with (
        patch("gps_journey.core.gpx_parser.GPXParser") as parser_cls,
        patch.object(Path, "exists", return_value=True),
    ):
        parser_cls.return_value.parse.return_value = mock_journey
        return build_route_profile(
            index=0,
            gpx_files=[Path("/fake/ep0.gpx")],
            episode_indices=[0],
            geocoder=geocoder,
            route_analyzer=_make_route_analyzer(
                roads=[{"road": "I-26", "distance_km": 40.0}], towns=towns
            ),
        )


def _town(name, lat=None, lon=None, duration=10):
    town = {"name": name, "place_class": "town", "importance": 0.5, "duration_minutes": duration}
    if lat is not None:
        town["entry_latitude"] = lat
        town["entry_longitude"] = lon
    return town


_UNICOI = {"town": "Unicoi", "county": "Unicoi County", "state": "Tennessee"}
_KINGSPORT = {"city": "Kingsport", "county": "Sullivan County", "state": "Tennessee"}


class TestEndpointNamedWhereGpsStarts:
    def test_start_detected_town_25km_away_does_not_replace_anchor(self):
        """Unicoi shape: the first detected town is Johnson City, ~25 km up the
        road. The LF started in Unicoi, so Unicoi is the start."""
        profile = _build_profile(
            _anchor_geocoder(_UNICOI, _KINGSPORT),
            [_town("Johnson City", 36.435, -82.35), _town("Kingsport", 36.53, -82.54)],
        )
        assert profile.start_town == "Unicoi"
        assert profile.end_town == "Kingsport"

    def test_end_detected_town_25km_before_track_end_does_not_replace_anchor(self):
        profile = _build_profile(
            _anchor_geocoder(_UNICOI, _KINGSPORT),
            [_town("Unicoi", 36.22, -82.35), _town("Bristol", 36.33, -82.45)],
        )
        assert profile.start_town == "Unicoi"
        assert profile.end_town == "Kingsport"

    def test_same_place_detected_spelling_is_kept(self):
        profile = _build_profile(
            _anchor_geocoder(_UNICOI, _KINGSPORT),
            [_town("City of Unicoi", 36.22, -82.35), _town("Kingsport", 36.53, -82.54)],
        )
        assert profile.start_town == "City of Unicoi"

    @pytest.mark.parametrize(
        ("entry", "expected"),
        [
            ((36.25, -82.35), "Frogmore"),  # ~3 km from the anchor: same place
            ((36.30, -82.35), None),  # ~9 km: a different place
        ],
    )
    def test_anchor_without_settlement_trusts_detected_town_only_within_6km(self, entry, expected):
        rural = {"county": "Unicoi County", "state": "Tennessee"}
        town = _town("Frogmore", *(entry or ()))
        profile = _build_profile(_anchor_geocoder(rural, _KINGSPORT), [town, _town("Kingsport")])
        assert profile.start_town == expected


class TestEndpointSamePlaceAndNonSettlementAnchor:
    def test_detected_village_kept_when_anchor_has_village_and_town(self):
        """The geocoder returns both fields (Downings village, Letterkenny town);
        the detected village is the same place, so it keeps its name."""
        both = {"village": "Downings", "town": "Letterkenny", "county": "Donegal"}
        profile = _build_profile(
            _anchor_geocoder(both, _KINGSPORT),
            [_town("Downings", 36.22, -82.35), _town("Kingsport", 36.53, -82.54)],
        )
        assert profile.start_town == "Downings"

    def test_township_anchor_does_not_beat_real_detected_town(self):
        township = {
            "town": "Cumberland Township",
            "county": "Adams County",
            "state": "Pennsylvania",
        }
        profile = _build_profile(
            _anchor_geocoder(township, _KINGSPORT),
            [_town("Real Town", 36.23, -82.35), _town("Kingsport", 36.53, -82.54)],
        )
        assert profile.start_town == "Real Town"

    def test_township_anchor_and_far_detected_town_leave_no_endpoint(self):
        township = {
            "town": "Cumberland Township",
            "county": "Adams County",
            "state": "Pennsylvania",
        }
        profile = _build_profile(
            _anchor_geocoder(township, _KINGSPORT),
            [_town("Real Town", 36.40, -82.35), _town("Kingsport", 36.53, -82.54)],
        )
        assert profile.start_town is None
