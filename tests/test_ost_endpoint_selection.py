"""Tests for chronological-endpoint selection with junk filtering.

Design: the drive's start and end towns (chronologically first and last)
are the correct endpoints for the OST. Importance-based re-ordering was
too aggressive — it let a 4-minute brush past Beaufort replace a 20-minute
stay in Frogmore.

New rule: keep chronological endpoints by default. Only replace an endpoint
if it is "junk" — a pinpoint crossroads that wouldn't describe the drive
to a viewer. Junk = ALL of:
  * duration < 3 minutes, AND
  * place_class is None / "hamlet" / missing, AND
  * importance < 0.3

Junk endpoints are replaced by the next-inward non-junk chronological town.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_journey(lat=32.5, lon=-80.5, wp_count=100):
    mock_journey = MagicMock()
    mock_wp = MagicMock(latitude=lat, longitude=lon)
    mock_journey.waypoints = [mock_wp] * wp_count
    return mock_journey


def _make_us_geocoder(state="South Carolina"):
    mock_geocoder = MagicMock()
    mock_loc = MagicMock(state=state, country="United States")
    mock_geocoder.reverse_geocode.return_value = mock_loc
    return mock_geocoder


class TestEndpointSelection:
    """Chronological first and last towns win unless they are junk."""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_chronological_endpoints_preserved_even_when_middle_more_important(self, mock_parser):
        """Scenario: Frogmore (20min, village) → Beaufort (4min, city) → Port Royal
        (18min, town). Frogmore and Port Royal stay as endpoints — Beaufort's
        higher Nominatim importance does NOT displace them."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey()

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Overland Trail", "distance_km": 15.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {
                "name": "Frogmore",
                "duration_minutes": 20,
                "place_class": "village",
                "importance": 0.15,
            },
            {"name": "Beaufort", "duration_minutes": 4, "place_class": "city", "importance": 0.72},
            {
                "name": "Port Royal",
                "duration_minutes": 18,
                "place_class": "town",
                "importance": 0.45,
            },
        ]

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=_make_us_geocoder(),
        )
        assert result == "Frogmore to Port Royal via Overland Trail"
        assert "Beaufort" not in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_junk_last_endpoint_replaced_by_inward_neighbor(self, mock_parser):
        """Scenario: Cashiers (10min, town) → Highlands (12min, city) →
        Gneiss (2min, hamlet, low-imp). Gneiss is junk — drop it; Highlands
        becomes the new last chronological endpoint."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey(lat=35.1, lon=-83.1)

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Highlands Road", "distance_km": 18.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {"name": "Cashiers", "duration_minutes": 10, "place_class": "town", "importance": 0.55},
            {
                "name": "Highlands",
                "duration_minutes": 12,
                "place_class": "city",
                "importance": 0.62,
            },
            {"name": "Gneiss", "duration_minutes": 2, "place_class": "hamlet", "importance": 0.05},
        ]

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=_make_us_geocoder(state="North Carolina"),
        )
        assert result == "Cashiers to Highlands via Highlands Road"
        assert "Gneiss" not in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_junk_first_endpoint_replaced_by_inward_neighbor(self, mock_parser):
        """A junk first-chronological endpoint is replaced by the next real town."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey()

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Coastal Road", "distance_km": 20.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            # Junk: tiny hamlet, brief pass
            {"name": "Noname", "duration_minutes": 2, "place_class": "hamlet", "importance": 0.02},
            {"name": "RealTown", "duration_minutes": 12, "place_class": "town", "importance": 0.4},
            {"name": "BigCity", "duration_minutes": 15, "place_class": "city", "importance": 0.65},
        ]

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=_make_us_geocoder(),
        )
        assert result == "RealTown to BigCity via Coastal Road"
        assert "Noname" not in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_short_duration_city_is_not_junk(self, mock_parser):
        """A real city or town is never junk even if duration is short.

        Junk requires ALL three criteria — brief + weak-class + low-importance.
        A 2-minute pass through a city stays if the city is well-classified."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey()

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "I-26", "distance_km": 50.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            # Brief but real city
            {"name": "CityA", "duration_minutes": 2, "place_class": "city", "importance": 0.6},
            {
                "name": "MidVillage",
                "duration_minutes": 10,
                "place_class": "village",
                "importance": 0.2,
            },
            {"name": "CityB", "duration_minutes": 2, "place_class": "city", "importance": 0.55},
        ]

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=_make_us_geocoder(),
        )
        # Both cities survive despite short durations
        assert result == "CityA to CityB via I-26"

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_legacy_tests_no_metadata_still_chronological(self, mock_parser):
        """When mock data lacks place_class/importance/duration_minutes (legacy
        test style), treat missing fields safely and fall back to chronological
        first/last."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey()

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Trask Parkway", "distance_km": 21.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {"name": "Port Royal"},
            {"name": "Beaufort"},
        ]

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=_make_us_geocoder(),
        )
        assert result == "Port Royal to Beaufort via Trask Parkway"

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_only_one_non_junk_town_falls_to_single_town_shape(self, mock_parser):
        """If junk filtering leaves only one real town, the result uses the
        single-town shape (road, town, state)."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey()

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Scenic Drive", "distance_km": 10.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {"name": "Junk1", "duration_minutes": 1, "place_class": "hamlet", "importance": 0.01},
            {"name": "RealTown", "duration_minutes": 20, "place_class": "town", "importance": 0.4},
            {"name": "Junk2", "duration_minutes": 2, "place_class": "hamlet", "importance": 0.05},
        ]

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=_make_us_geocoder(),
        )
        assert result == "Scenic Drive, RealTown, SC"

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_all_junk_falls_through_to_road_state(self, mock_parser):
        """If every town is junk, fall through to the road-state shape."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey()

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "US-21", "distance_km": 30.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {"name": "Junk1", "duration_minutes": 1, "place_class": "hamlet", "importance": 0.01},
            {"name": "Junk2", "duration_minutes": 2, "place_class": "hamlet", "importance": 0.05},
        ]

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=_make_us_geocoder(),
        )
        # Both towns filtered → no towns left → road + state shape
        assert result == "US-21, South Carolina"

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_duration_beats_importance_when_both_non_junk(self, mock_parser):
        """(Safety) A long-duration village is not displaced by a high-importance
        brief city — the first-chronological endpoint just stays."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey()

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "River Road", "distance_km": 22.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            # Long stay, obscure name
            {
                "name": "LongStay",
                "duration_minutes": 35,
                "place_class": "village",
                "importance": 0.1,
            },
            # Brief pass, famous name
            {"name": "Famous", "duration_minutes": 3, "place_class": "city", "importance": 0.9},
        ]

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=_make_us_geocoder(),
        )
        # Both survive junk check → chronological order wins
        assert result == "LongStay to Famous via River Road"

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_highway_class_short_duration_is_junk(self, mock_parser):
        """Scenario: Gneiss resolves as class='highway' with
        duration 5 min (OSM has no place feature, just a road running through).
        Real towns that Nominatim matches to a 'place' feature are kept even
        at the same duration."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey(lat=35.1, lon=-83.1)

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Highlands Road", "distance_km": 18.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {"name": "Cashiers", "duration_minutes": 15, "place_class": "shop", "importance": 0.0},
            {
                "name": "Highlands",
                "duration_minutes": 24,
                "place_class": "place",
                "importance": 0.0,
            },
            {
                "name": "Gneiss",
                "duration_minutes": 5,
                "place_class": "highway",
                "importance": 0.053,
            },
        ]

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=_make_us_geocoder(state="North Carolina"),
        )
        assert result == "Cashiers to Highlands via Highlands Road"
        assert "Gneiss" not in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_highway_class_long_duration_is_kept(self, mock_parser):
        """Real towns where the sampled point happened to land on a highway
        feature are kept if you stayed in the area long enough — Rule A only
        fires for short durations."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey()

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "I-26", "distance_km": 40.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            # Real towns where sampling landed on highway features
            {
                "name": "Columbus",
                "duration_minutes": 13,
                "place_class": "highway",
                "importance": 0.053,
            },
            {
                "name": "Mills River",
                "duration_minutes": 14,
                "place_class": "highway",
                "importance": 0.053,
            },
        ]

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=_make_us_geocoder(state="North Carolina"),
        )
        # 13m and 14m are both >= 6 min (highway threshold) → kept
        assert result == "Columbus to Mills River via I-26"
