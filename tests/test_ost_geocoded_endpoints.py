"""Tests for waypoint-geocoded endpoint anchoring.

Three scenarios that the detected-towns approach can't handle:

1. **Long transit** (example): 500+ km drive from SC Lowcountry to NC
   mountains. RouteAnalyzer detects zero towns for the first 400 km
   (interstate), then picks up towns at the end. "First chronological
   detected town" is hundreds of km from the actual drive start. Fix:
   reverse-geocode the first waypoint; if it resolves to a place that's
   far from the first detected town, prefer the waypoint anchor.

2. **Towns-less loop** (example): short drive in a rural county with
   no town/village/city at any sampled point. Geocoder only returns the
   county. Need a county fallback shape: "{road}, {county}, {state}".

3. **Cross-county transit** (example): drive that starts in one
   county and ends in another with no towns detected. Need a cross-county
   shape: "{start_county} to {end_county}, {state}".

All three mock the geocoder's reverse_geocode method with a side_effect
function that returns different data based on the input coordinates.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_journey_with_endpoints(start_coords, end_coords, wp_count=100):
    """Create a mock journey where the first ~25 waypoints are at start_coords
    and the last ~25 waypoints are at end_coords. This mirrors how a real
    drive looks to _pick_anchor_waypoint, which picks slightly inward from
    the ends to dodge stale GPS fixes.
    """
    mock_journey = MagicMock()
    start_block = [
        MagicMock(latitude=start_coords[0], longitude=start_coords[1]) for _ in range(25)
    ]
    middle_wps = [
        MagicMock(
            latitude=(start_coords[0] + end_coords[0]) / 2,
            longitude=(start_coords[1] + end_coords[1]) / 2,
        )
        for _ in range(max(0, wp_count - 50))
    ]
    end_block = [MagicMock(latitude=end_coords[0], longitude=end_coords[1]) for _ in range(25)]
    mock_journey.waypoints = [*start_block, *middle_wps, *end_block]
    return mock_journey


def _make_coord_geocoder(coord_to_location: dict):
    """Build a mock geocoder with both reverse_geocode and
    reverse_geocode_detailed driven by (lat, lon) lookup.

    coord_to_location: dict mapping (lat_rounded, lon_rounded) to the fields
        the real geocoder would return, e.g.
        {(32.36, -80.44): {'village': None, 'town': None, 'city': None,
                           'county': 'Beaufort County',
                           'state': 'South Carolina'}}
    """
    mock_geocoder = MagicMock()

    def detailed_side_effect(lat, lon):
        key = (round(lat, 2), round(lon, 2))
        kwargs = coord_to_location.get(key)
        if kwargs is None:
            return {}
        return {
            "latitude": lat,
            "longitude": lon,
            "town": kwargs.get("town"),
            "city": kwargs.get("city"),
            "village": kwargs.get("village"),
            "county": kwargs.get("county"),
            "state": kwargs.get("state"),
            "country": kwargs.get("country"),
        }

    def reverse_side_effect(lat, lon):
        key = (round(lat, 2), round(lon, 2))
        loc_kwargs = coord_to_location.get(key)
        if loc_kwargs is None:
            return None
        mock_loc = MagicMock()
        for attr in ("town", "city", "village", "county", "state", "country"):
            setattr(mock_loc, attr, loc_kwargs.get(attr))
        return mock_loc

    mock_geocoder.reverse_geocode_detailed.side_effect = detailed_side_effect
    mock_geocoder.reverse_geocode.side_effect = reverse_side_effect
    return mock_geocoder


class TestFarDetectedTownFallback:
    """Long-transit scenario: detected towns cluster at one end of a long drive."""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_start_waypoint_far_from_first_detected_town_uses_waypoint(self, mock_parser):
        """When the first detected town is far from the actual first waypoint,
        use the waypoint's geocoded anchor instead."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        # Drive from SC Lowcountry to NC mountains. 500km span.
        mock_parser.return_value.parse.return_value = _make_journey_with_endpoints(
            start_coords=(32.36, -80.44),  # Beaufort County SC
            end_coords=(35.11, -83.10),  # Cashiers NC
        )

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Hunting Island Drive", "distance_km": 12.0},
        ]
        # Detected towns are clustered in the last 100km of the drive.
        # Both are in NC (near the drive's end), far from the actual SC start.
        mock_analyzer.get_towns_passed_through.return_value = [
            {
                "name": "Lake Lure",
                "duration_minutes": 3,
                "place_class": "place",
                "importance": 0.0,
                "entry_latitude": 35.43,
                "entry_longitude": -82.21,
            },
            {
                "name": "Oakland",
                "duration_minutes": 14,
                "place_class": "place",
                "importance": 0.0,
                "entry_latitude": 35.45,
                "entry_longitude": -82.50,
            },
        ]

        mock_geocoder = _make_coord_geocoder(
            {
                # Start waypoint — no town, only county
                (32.36, -80.44): {
                    "town": None,
                    "city": None,
                    "village": None,
                    "county": "Beaufort County",
                    "state": "South Carolina",
                    "country": "United States",
                },
                # End waypoint — resolves to Cashiers (village)
                (35.11, -83.10): {
                    "town": None,
                    "city": None,
                    "village": "Cashiers",
                    "county": "Jackson County",
                    "state": "North Carolina",
                    "country": "United States",
                },
            }
        )

        result = derive_episode_ost_text(
            Path("/fake/ep7.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        # The drive is SC→NC. Lake Lure (378 km from real start) is dropped
        # and replaced by the waypoint-anchored start (Beaufort County).
        # End upgrades from Oakland (66 km from real end) to Cashiers.
        assert "Lake Lure" not in result
        assert "Oakland" not in result
        assert "Beaufort County" in result or "Beaufort" in result
        assert "Cashiers" in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_local_drive_detected_towns_near_waypoints_kept_unchanged(self, mock_parser):
        """When detected towns are near the actual waypoints (local drive),
        the existing chronological-endpoint behavior is preserved."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey_with_endpoints(
            start_coords=(32.50, -80.60),  # near Frogmore
            end_coords=(32.48, -80.68),  # near Port Royal — same area
        )

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Overland Trail", "distance_km": 15.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {"name": "Frogmore", "duration_minutes": 20, "place_class": "place", "importance": 0.1},
            {
                "name": "Port Royal",
                "duration_minutes": 18,
                "place_class": "place",
                "importance": 0.2,
            },
        ]

        # Waypoint geocodes match the detected towns closely
        mock_geocoder = _make_coord_geocoder(
            {
                (32.50, -80.60): {
                    "town": None,
                    "city": None,
                    "village": "Frogmore",
                    "county": "Beaufort County",
                    "state": "South Carolina",
                },
                (32.48, -80.68): {
                    "town": "Port Royal",
                    "city": None,
                    "village": None,
                    "county": "Beaufort County",
                    "state": "South Carolina",
                },
            }
        )

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert result == "Frogmore to Port Royal via Overland Trail"


class TestCountyFallback:
    """Towns-less loop scenario: no detected towns at all, use county as anchor."""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_towns_less_loop_same_county_gets_county_in_label(self, mock_parser):
        """Short rural loop with zero detected towns — emit
        '{road}, {county}, {state}' using the county from the start waypoint."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey_with_endpoints(
            start_coords=(32.68, -80.56),
            end_coords=(32.70, -80.61),
        )

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Stocks Creek Road", "distance_km": 8.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = []

        mock_geocoder = _make_coord_geocoder(
            {
                (32.68, -80.56): {
                    "town": None,
                    "city": None,
                    "village": None,
                    "county": "Colleton County",
                    "state": "South Carolina",
                },
                (32.70, -80.61): {
                    "town": None,
                    "city": None,
                    "village": None,
                    "county": "Colleton County",
                    "state": "South Carolina",
                },
            }
        )

        result = derive_episode_ost_text(
            Path("/fake/ep5.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert result == "Stocks Creek Road, Colleton County, SC"

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_towns_less_no_county_falls_to_state(self, mock_parser):
        """When the geocoder returns no county either, fall to {road}, {state}."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey_with_endpoints(
            start_coords=(32.68, -80.56),
            end_coords=(32.70, -80.61),
        )

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "US-21", "distance_km": 30.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = []

        mock_geocoder = _make_coord_geocoder(
            {
                (32.68, -80.56): {
                    "town": None,
                    "city": None,
                    "village": None,
                    "county": None,
                    "state": "South Carolina",
                },
                (32.70, -80.61): {
                    "town": None,
                    "city": None,
                    "village": None,
                    "county": None,
                    "state": "South Carolina",
                },
            }
        )

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert result == "US-21, South Carolina"


class TestCrossCountyDrive:
    """Cross-county scenario: drive spans two counties with no detected towns."""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_cross_county_same_state_no_towns_uses_county_to_county_shape(self, mock_parser):
        """Cross-county drive with no detected towns emits
        '{start_county} to {end_county}, {state}' (no via-road)."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey_with_endpoints(
            start_coords=(32.70, -80.62),  # Colleton County
            end_coords=(32.36, -80.44),  # Beaufort County
        )

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Stocks Creek Road", "distance_km": 3.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = []

        mock_geocoder = _make_coord_geocoder(
            {
                (32.70, -80.62): {
                    "town": None,
                    "city": None,
                    "village": None,
                    "county": "Colleton County",
                    "state": "South Carolina",
                },
                (32.36, -80.44): {
                    "town": None,
                    "city": None,
                    "village": None,
                    "county": "Beaufort County",
                    "state": "South Carolina",
                },
            }
        )

        result = derive_episode_ost_text(
            Path("/fake/ep6.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert result == "Colleton County to Beaufort County, SC"
        # Via-road is dropped for cross-county drives — the top road is
        # unreliable as a descriptor for a drive that spans counties.
        assert " via " not in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_cross_county_with_town_on_one_side_uses_town_and_county(self, mock_parser):
        """One endpoint resolves to a town, the other only to a county —
        format as '{town} to {county}, {state}'."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey_with_endpoints(
            start_coords=(32.70, -80.62),
            end_coords=(32.36, -80.44),
        )

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = []
        mock_analyzer.get_towns_passed_through.return_value = []

        mock_geocoder = _make_coord_geocoder(
            {
                (32.70, -80.62): {
                    "town": "Walterboro",
                    "city": None,
                    "village": None,
                    "county": "Colleton County",
                    "state": "South Carolina",
                },
                (32.36, -80.44): {
                    "town": None,
                    "city": None,
                    "village": None,
                    "county": "Beaufort County",
                    "state": "South Carolina",
                },
            }
        )

        result = derive_episode_ost_text(
            Path("/fake/ep.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert result == "Walterboro to Beaufort County, SC"


class TestCrossStateDrive:
    """Long transit crossing a state boundary."""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_cross_state_with_towns_uses_town_state_shape(self, mock_parser):
        """Cross-state drive with towns at both ends: include state abbrevs."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey_with_endpoints(
            start_coords=(32.36, -80.44),
            end_coords=(35.11, -83.10),
        )

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "I-26", "distance_km": 300.0},
        ]
        # Detected towns all clustered at the NC end — entry coords are far
        # from the actual SC start waypoint, so they'll be dropped.
        mock_analyzer.get_towns_passed_through.return_value = [
            {
                "name": "Lake Lure",
                "duration_minutes": 3,
                "place_class": "place",
                "importance": 0.0,
                "entry_latitude": 35.43,
                "entry_longitude": -82.21,
            },
            {
                "name": "Oakland",
                "duration_minutes": 14,
                "place_class": "place",
                "importance": 0.0,
                "entry_latitude": 35.45,
                "entry_longitude": -82.50,
            },
        ]

        mock_geocoder = _make_coord_geocoder(
            {
                (32.36, -80.44): {
                    "town": None,
                    "city": None,
                    "village": None,
                    "county": "Beaufort County",
                    "state": "South Carolina",
                },
                (35.11, -83.10): {
                    "town": None,
                    "city": None,
                    "village": "Cashiers",
                    "county": "Jackson County",
                    "state": "North Carolina",
                },
            }
        )

        result = derive_episode_ost_text(
            Path("/fake/ep7.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        # Cross-state format includes state abbrevs
        assert "SC" in result
        assert "NC" in result
        assert "Cashiers" in result
        # Old detected-town anchors are dropped
        assert "Lake Lure" not in result
        assert "Oakland" not in result


class TestDriveOSTWaypointAnchoring:
    """derive_drive_ost_text's GPX fallback anchors inward from raw endpoints.

    Regression: it geocoded raw first_wps[0]/last_wps[-1] — the exact
    stale-previous-session-fix failure _pick_anchor_waypoint exists to dodge —
    so the burned drive-level OST could name a town hundreds of km away.
    """

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_stale_first_and_last_fixes_do_not_name_drive_endpoints(self, mock_parser):
        from gomoveshift.pipeline.ingest import derive_drive_ost_text

        journey = _make_journey_with_endpoints(
            start_coords=(32.78, -79.93),  # Charleston
            end_coords=(33.69, -78.86),  # Myrtle Beach
        )
        # Corrupt the raw first/last fixes with stale previous-session coords.
        stale = {"latitude": 35.11, "longitude": -83.10}  # Cashiers, ~400 km away
        journey.waypoints[0] = MagicMock(**stale)
        journey.waypoints[-1] = MagicMock(**stale)
        mock_parser.return_value.parse.return_value = journey

        mock_geocoder = _make_coord_geocoder(
            {
                (32.78, -79.93): {"town": "Charleston", "state": "South Carolina"},
                (33.69, -78.86): {"town": "Myrtle Beach", "state": "South Carolina"},
                (35.11, -83.10): {"village": "Cashiers", "state": "North Carolina"},
            }
        )

        result = derive_drive_ost_text([Path("/fake/drive.gpx")], geocoder=mock_geocoder)
        assert result == "Charleston to Myrtle Beach"
        assert "Cashiers" not in result


class TestRoundTripRegression:
    """Regression tests for the Ep 1/2 out-and-back pattern.

    When two consecutive episodes are halves of a single round trip —
    Ep1's end coincides with Ep2's start, and Ep2's end coincides with
    Ep1's start — the OSTs should naturally read as mirror images
    ("A to B" and "B to A"). This is the *correct* behavior, not an
    algorithm artifact to be removed. This test locks the behavior in.
    """

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_outbound_leg_labels_start_to_destination(self, mock_parser):
        """Ep 1 pattern: starts in Longwood, ends at Fort Fremont."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey_with_endpoints(
            start_coords=(32.42, -80.47),  # Longwood
            end_coords=(32.30, -80.64),  # Fort Fremont
        )

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Seaside Road", "distance_km": 18.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {"name": "Longwood", "duration_minutes": 5, "place_class": "place", "importance": 0.05},
            {
                "name": "Fort Fremont",
                "duration_minutes": 3,
                "place_class": "place",
                "importance": 0.0,
            },
        ]

        mock_geocoder = _make_coord_geocoder(
            {
                (32.42, -80.47): {
                    "town": None,
                    "city": None,
                    "village": "Longwood",
                    "county": "Beaufort County",
                    "state": "South Carolina",
                },
                (32.30, -80.64): {
                    "town": None,
                    "city": None,
                    "village": "Fort Fremont",
                    "county": "Beaufort County",
                    "state": "South Carolina",
                },
            }
        )

        result = derive_episode_ost_text(
            Path("/fake/ep1.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert result == "Longwood to Fort Fremont via Seaside Road"

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_return_leg_labels_destination_to_start(self, mock_parser):
        """Ep 2 pattern: starts at Fort Fremont, ends back in Longwood.
        This is the mirror of Ep 1 — intentionally so."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey_with_endpoints(
            start_coords=(32.30, -80.64),  # Fort Fremont
            end_coords=(32.41, -80.48),  # Near Longwood
        )

        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [
            {"road": "Seaside Road", "distance_km": 18.0},
        ]
        mock_analyzer.get_towns_passed_through.return_value = [
            {
                "name": "Fort Fremont",
                "duration_minutes": 5,
                "place_class": "place",
                "importance": 0.0,
            },
            {
                "name": "Longwood",
                "duration_minutes": 43,
                "place_class": "place",
                "importance": 0.05,
            },
        ]

        mock_geocoder = _make_coord_geocoder(
            {
                (32.30, -80.64): {
                    "town": None,
                    "city": None,
                    "village": "Fort Fremont",
                    "county": "Beaufort County",
                    "state": "South Carolina",
                },
                (32.41, -80.48): {
                    "town": None,
                    "city": None,
                    "village": "Longwood",
                    "county": "Beaufort County",
                    "state": "South Carolina",
                },
            }
        )

        result = derive_episode_ost_text(
            Path("/fake/ep2.gpx"),
            route_analyzer=mock_analyzer,
            geocoder=mock_geocoder,
        )
        assert result == "Fort Fremont to Longwood via Seaside Road"


class TestAnchorSettlementBeatsFartherDetectedTown:
    """A detected town up to 50 km from the GPS start/end must not replace the
    settlement the start/end fixes geocode to (the LF starts where it starts)."""

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_unicoi_start_not_named_johnson_city(self, mock_parser):
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey_with_endpoints(
            start_coords=(36.21, -82.35), end_coords=(36.55, -82.55)
        )
        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [{"road": "I-26", "distance_km": 40.0}]
        mock_analyzer.get_towns_passed_through.return_value = [
            {
                "name": "Johnson City",
                "duration_minutes": 12,
                "place_class": "city",
                "importance": 0.6,
                "entry_latitude": 36.435,
                "entry_longitude": -82.35,
            },
            {
                "name": "Kingsport",
                "duration_minutes": 9,
                "place_class": "city",
                "importance": 0.5,
                "entry_latitude": 36.53,
                "entry_longitude": -82.54,
            },
        ]
        geocoder = _make_coord_geocoder(
            {
                (36.21, -82.35): {
                    "town": "Unicoi",
                    "county": "Unicoi County",
                    "state": "Tennessee",
                    "country": "United States",
                },
                (36.55, -82.55): {
                    "city": "Kingsport",
                    "county": "Sullivan County",
                    "state": "Tennessee",
                    "country": "United States",
                },
            }
        )
        result = derive_episode_ost_text(
            Path("/fake/unicoi.gpx"), route_analyzer=mock_analyzer, geocoder=geocoder
        )
        assert "Unicoi" in result
        assert "Kingsport" in result
        assert "Johnson City" not in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_end_detected_town_25km_before_end_not_used(self, mock_parser):
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey_with_endpoints(
            start_coords=(36.21, -82.35), end_coords=(36.55, -82.55)
        )
        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [{"road": "I-26", "distance_km": 40.0}]
        mock_analyzer.get_towns_passed_through.return_value = [
            {
                "name": "Unicoi",
                "duration_minutes": 12,
                "place_class": "town",
                "importance": 0.4,
                "entry_latitude": 36.22,
                "entry_longitude": -82.35,
            },
            {
                "name": "Bristol",
                "duration_minutes": 9,
                "place_class": "city",
                "importance": 0.5,
                "entry_latitude": 36.33,
                "entry_longitude": -82.45,
            },
        ]
        geocoder = _make_coord_geocoder(
            {
                (36.21, -82.35): {
                    "town": "Unicoi",
                    "county": "Unicoi County",
                    "state": "Tennessee",
                    "country": "United States",
                },
                (36.55, -82.55): {
                    "city": "Kingsport",
                    "county": "Sullivan County",
                    "state": "Tennessee",
                    "country": "United States",
                },
            }
        )
        result = derive_episode_ost_text(
            Path("/fake/unicoi.gpx"), route_analyzer=mock_analyzer, geocoder=geocoder
        )
        assert "Kingsport" in result
        assert "Bristol" not in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_ireland_village_anchor_beats_other_village_15km_away(self, mock_parser):
        """Non-US: the anchor is a village in a county with no state; the
        detected village 15 km away is a different place."""
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey_with_endpoints(
            start_coords=(53.80, -9.50), end_coords=(53.80, -9.50)
        )
        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = []
        mock_analyzer.get_towns_passed_through.return_value = [
            {
                "name": "Ballybar",
                "duration_minutes": 10,
                "place_class": "village",
                "importance": 0.3,
                "entry_latitude": 53.935,
                "entry_longitude": -9.50,
            },
        ]
        geocoder = _make_coord_geocoder(
            {
                (53.80, -9.50): {
                    "village": "Ballyfoo",
                    "county": "County Mayo",
                    "country": "Ireland",
                },
            }
        )
        result = derive_episode_ost_text(
            Path("/fake/ie.gpx"), route_analyzer=mock_analyzer, geocoder=geocoder
        )
        assert result == "Ballyfoo"


class TestAnchorSamePlaceAndNonSettlement:
    def _ost(self, mock_parser, anchor_fields, detected_name):
        from gomoveshift.pipeline.ingest import derive_episode_ost_text

        mock_parser.return_value.parse.return_value = _make_journey_with_endpoints(
            start_coords=(36.21, -82.35), end_coords=(36.55, -82.55)
        )
        mock_analyzer = MagicMock()
        mock_analyzer.get_road_segments.return_value = [{"road": "I-26", "distance_km": 40.0}]
        mock_analyzer.get_towns_passed_through.return_value = [
            {
                "name": detected_name,
                "duration_minutes": 12,
                "place_class": "town",
                "importance": 0.4,
                "entry_latitude": 36.22,
                "entry_longitude": -82.35,
            },
            {
                "name": "Kingsport",
                "duration_minutes": 9,
                "place_class": "city",
                "importance": 0.5,
                "entry_latitude": 36.53,
                "entry_longitude": -82.54,
            },
        ]
        geocoder = _make_coord_geocoder(
            {
                (36.21, -82.35): anchor_fields,
                (36.55, -82.55): {
                    "city": "Kingsport",
                    "county": "Sullivan County",
                    "state": "Tennessee",
                    "country": "United States",
                },
            }
        )
        return derive_episode_ost_text(
            Path("/fake/x.gpx"), route_analyzer=mock_analyzer, geocoder=geocoder
        )

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_detected_village_kept_when_anchor_has_village_and_town(self, mock_parser):
        result = self._ost(
            mock_parser,
            {
                "village": "Downings",
                "town": "Letterkenny",
                "county": "Donegal",
                "state": "Tennessee",
            },
            "Downings",
        )
        assert "Downings" in result
        assert "Letterkenny" not in result

    @patch("gps_journey.core.gpx_parser.GPXParser")
    def test_township_anchor_does_not_beat_real_detected_town(self, mock_parser):
        result = self._ost(
            mock_parser,
            {"town": "Cumberland Township", "county": "Adams County", "state": "Tennessee"},
            "Real Town",
        )
        assert "Real Town" in result
        assert "Township" not in result
