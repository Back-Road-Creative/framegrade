"""Viewer-facing place naming routes through Waypost's locality rule.

An obscure tract name ("Longwood", on St. Helena Island) must not reach the
description; the island stands in for it, and no place is listed twice in the
"Driving through" sentence.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator
from gomoveshift.pipeline.route_profile import build_route_profile

ISLAND_ADDR = "Seaside Road, Longwood, St. Helena Island, Beaufort County, South Carolina"


def _wp(lat, lon):
    wp = MagicMock()
    wp.latitude = lat
    wp.longitude = lon
    return wp


def _result() -> dict:
    return {
        "title": "Driving Beaufort",
        "description": "POV drive through Beaufort, SC.\n\n📍 Location: Beaufort, SC\n",
        "tags": ["scenic drive"],
        "location": "Beaufort, SC",
    }


def test_driving_through_names_each_place_once_in_first_seen_order():
    profile = {
        "towns_traversed": ["Fort Fremont", "St. Helena Island", "fort fremont", "Fort Fremont"],
        "roads_traversed": [],
        "start_town": None,
        "end_town": None,
    }
    desc = LongFormMetadataGenerator._enrich_with_route_profile(
        _result(), profile, road_coverage_measured=True
    )["description"]
    assert "Driving through Fort Fremont, St. Helena Island." in desc


@patch("gps_journey.core.gpx_parser.GPXParser")
def test_build_route_profile_drops_vicinity_name_for_island(mock_parser_cls):
    journey = MagicMock()
    journey.waypoints = [_wp(32.30, -80.64), _wp(32.42, -80.47)]
    mock_parser_cls.return_value.parse.return_value = journey

    def detailed(lat, lon):
        base = {"county": "Beaufort County", "state": "South Carolina"}
        if lat > 32.4:  # the Longwood sample: a road match, hamlet only in the address
            return {
                **base,
                "hamlet": "Longwood",
                "match_type": "residential",
                "place_class": "highway",
                "full_address": ISLAND_ADDR,
            }
        return {**base, "hamlet": "Fort Fremont", "match_type": "fort", "place_class": "historic"}

    geocoder = MagicMock()
    geocoder.reverse_geocode_detailed.side_effect = detailed
    analyzer = MagicMock()
    analyzer.get_road_segments.return_value = []
    analyzer.get_towns_passed_through.return_value = [
        {
            "name": "Longwood",
            "place_class": "highway",
            "importance": 0.1,
            "entry_latitude": 32.42,
            "entry_longitude": -80.47,
        },
        {
            "name": "Fort Fremont",
            "place_class": "historic",
            "importance": 0.2,
            "entry_latitude": 32.30,
            "entry_longitude": -80.64,
        },
    ]
    with patch.object(Path, "exists", return_value=True):
        profile = build_route_profile(0, [Path("/fake/ep0.gpx")], [0], geocoder, analyzer)

    assert profile.towns_traversed == ["St. Helena Island", "Fort Fremont"]
    assert profile.locality_substitutions == {"Longwood": "St. Helena Island"}
    assert "Longwood" not in profile.towns_traversed


def test_ordered_place_sequence_drops_a_tract_name_without_inventing_one():
    """Real Beaufort cache records (a hamlet of a house/road match, no island in
    the address): the per-LF sequence (Route line, per-LF rescope) drops
    Longwood, and does not substitute the county or an island."""
    from gomoveshift.pipeline.title_builder import dominant_location

    base = {"county": "Beaufort County", "state": "South Carolina"}

    def detailed(lat, lon):
        if lat > 32.33:
            return {
                **base,
                "hamlet": "Longwood",
                "match_type": "residential",
                "place_class": "highway",
                "addresstype": "road",
                "full_address": "Seaside Road, Longwood, Beaufort County, South Carolina",
            }
        return {
            **base,
            "hamlet": "Fort Fremont",
            "match_type": "house",
            "place_class": "place",
            "addresstype": "place",
            "full_address": "27, Finch Road, Fort Fremont, Beaufort County",
        }

    geo = MagicMock()
    geo.reverse_geocode_detailed.side_effect = detailed
    pts = [(32.3048, -80.6413), (32.3551, -80.5753), (32.3600, -80.5700)]
    assert dominant_location(pts, geo).ordered_places == ["Fort Fremont"]


def test_ordered_place_sequence_uses_an_island_named_in_the_record_address():
    from gomoveshift.pipeline.title_builder import dominant_location

    geo = MagicMock()
    geo.reverse_geocode_detailed.return_value = {
        "county": "Z County",
        "state": "Zed",
        "hamlet": "Tractville",
        "match_type": "house",
        "place_class": "place",
        "full_address": "1, Tractville, Big Island, Z County, Zed",
    }
    pts = [(10.0, 10.0), (10.01, 10.01)]
    assert dominant_location(pts, geo).ordered_places == ["Big Island"]
