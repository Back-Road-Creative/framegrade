"""Tests for GPS validation of cross-county claims in episode OST formatter.

Bug: episode 5 GPS is entirely within Colleton County, but the formatter
produced "Colleton County to Beaufort County, SC" because the end-waypoint
geocode returned "Beaufort County" (geocoder flake). The midpoint check
catches this: if the midpoint is in the same county as the start, the
cross-county claim is demoted and the formatter falls through to a
road-based or single-county shape instead.
"""

from unittest.mock import MagicMock


def _wp(lat, lon):
    wp = MagicMock()
    wp.latitude = lat
    wp.longitude = lon
    return wp


def _make_geocoder(county_map):
    """Create a mock geocoder that returns county based on lat/lon lookup."""
    geo = MagicMock()

    def _detailed(lat, lon):
        for (clat, clon), data in county_map.items():
            if abs(lat - clat) < 0.1 and abs(lon - clon) < 0.1:
                return data
        return {"county": None, "state": None, "name": None}

    geo.reverse_geocode_detailed.side_effect = _detailed
    return geo


class TestCrossCountyValidation:
    """Shape B cross-county claims are validated against GPS midpoint."""

    def test_same_county_midpoint_demotes_cross_county(self):
        """When midpoint is in start county, cross-county claim is rejected."""
        from gomoveshift.pipeline.ingest.ost_formatter import _format_ost_from_route

        # Simulate: start=Colleton, end=Beaufort (geocoder flake), mid=Colleton
        waypoints = [_wp(32.67, -80.56)] * 10 + [_wp(32.69, -80.61)] * 10
        geocoder = _make_geocoder(
            {
                (32.67, -80.56): {
                    "county": "Colleton County",
                    "state": "South Carolina",
                    "name": None,
                },
                (32.69, -80.61): {
                    "county": "Beaufort County",
                    "state": "South Carolina",
                    "name": None,
                },
                # Midpoint — same county as start
                (32.68, -80.58): {
                    "county": "Colleton County",
                    "state": "South Carolina",
                    "name": None,
                },
            }
        )

        road_segments = [{"road": "Stocks Creek Road", "distance_km": 8.0}]
        towns = []  # No towns

        result = _format_ost_from_route(
            road_segments,
            towns,
            geocoder=geocoder,
            waypoints=waypoints,
        )
        # Should NOT produce "Colleton County to Beaufort County, SC"
        assert "Beaufort County" not in result
        # Should fall through to road-based format
        assert "Stocks Creek Road" in result

    def test_genuine_cross_county_preserved(self):
        """When midpoint is in a different county, cross-county is real."""
        from gomoveshift.pipeline.ingest.ost_formatter import _format_ost_from_route

        # Start in Colleton, end in Beaufort, midpoint in Beaufort → real crossing
        waypoints = [_wp(32.50, -80.70)] * 10 + [_wp(32.40, -80.60)] * 10
        geocoder = _make_geocoder(
            {
                (32.50, -80.70): {
                    "county": "Colleton County",
                    "state": "South Carolina",
                    "name": None,
                },
                (32.40, -80.60): {
                    "county": "Beaufort County",
                    "state": "South Carolina",
                    "name": None,
                },
                # Midpoint — matches end county, not start
                (32.45, -80.65): {
                    "county": "Beaufort County",
                    "state": "South Carolina",
                    "name": None,
                },
            }
        )

        road_segments = [{"road": "US-21", "distance_km": 15.0}]
        towns = []

        result = _format_ost_from_route(
            road_segments,
            towns,
            geocoder=geocoder,
            waypoints=waypoints,
        )
        # Genuine cross-county preserved
        assert "Colleton County" in result
        assert "Beaufort County" in result

    def test_no_waypoints_skips_validation(self):
        """Without waypoints, cross-county check can't run — trusts geocoder."""
        from gomoveshift.pipeline.ingest.ost_formatter import _format_ost_from_route

        geocoder = _make_geocoder(
            {
                (32.67, -80.56): {
                    "county": "Colleton County",
                    "state": "South Carolina",
                    "name": None,
                },
                (32.69, -80.61): {
                    "county": "Beaufort County",
                    "state": "South Carolina",
                    "name": None,
                },
            }
        )

        road_segments = [{"road": "Stocks Creek Road", "distance_km": 8.0}]
        towns = []

        result = _format_ost_from_route(
            road_segments,
            towns,
            geocoder=geocoder,
            waypoints=None,
        )
        # No waypoints → can't validate → trusts the geocoder's cross-county claim
        # Falls to Shape E since no towns and start_anchor has no town
        # (geocoder returns name=None for both)
        assert "Stocks Creek Road" in result


class TestCrossStateValidation:
    """Shape A cross-STATE claims are validated against the GPS midpoint too.

    A single flaky end-waypoint geocode reported a neighbouring state on a
    single-state drive, fabricating a "…, SC to …, NC" span. The voted midpoint
    (same modal (county, state) helper the cross-county path uses) catches it —
    when the mid state matches the start state, the cross-state span is demoted
    to the single-state shape.
    """

    def test_same_state_midpoint_demotes_cross_state(self):
        """Mid state == start state → cross-state flake demoted to single-state."""
        from gomoveshift.pipeline.ingest.ost_formatter import _format_ost_from_route

        # SC-only drive; the END waypoint geocode flakes across the NC line.
        # Start region, central window, and end region are geographically
        # distinct so the mock can return NC only for the end anchor.
        waypoints = (
            [_wp(34.00, -81.00)] * 8  # start region (anchor at idx 1)
            + [_wp(34.50, -81.50)] * 6  # central window → votes SC
            + [_wp(35.00, -82.00)] * 6  # end region (anchor at idx 18) → flake
        )
        geocoder = _make_geocoder(
            {
                (34.00, -81.00): {
                    "county": "Richland County",
                    "state": "South Carolina",
                    "name": None,
                },
                # Mid samples resolve to the START state (SC) — the demotion signal.
                (34.50, -81.50): {
                    "county": "Chester County",
                    "state": "South Carolina",
                    "name": None,
                },
                # End-waypoint geocoder flake: claims North Carolina.
                (35.00, -82.00): {
                    "county": "Rutherford County",
                    "state": "North Carolina",
                    "name": None,
                },
            }
        )

        road_segments = [{"road": "Stocks Creek Road", "distance_km": 8.0}]
        towns = []

        result = _format_ost_from_route(
            road_segments,
            towns,
            geocoder=geocoder,
            waypoints=waypoints,
        )
        # No fabricated cross-state span from the end-waypoint flake.
        assert "North Carolina" not in result
        assert ", NC" not in result
        assert "Rutherford County" not in result
        # Demoted to the single-state road+county shape.
        assert "Stocks Creek Road" in result
        assert "Richland County" in result

    def test_genuine_cross_state_preserved(self):
        """Mid state matches the end state (real crossing) → span preserved."""
        from gomoveshift.pipeline.ingest.ost_formatter import _format_ost_from_route

        waypoints = (
            [_wp(34.00, -81.00)] * 8  # SC start (anchor at idx 1)
            + [_wp(35.00, -82.00)] * 6  # central window in NC → real crossing
            + [_wp(35.50, -82.50)] * 6  # NC end (anchor at idx 18)
        )
        geocoder = _make_geocoder(
            {
                (34.00, -81.00): {
                    "county": "Richland County",
                    "state": "South Carolina",
                    "name": "Columbia",
                },
                (35.00, -82.00): {
                    "county": "Rutherford County",
                    "state": "North Carolina",
                    "name": None,
                },
                (35.50, -82.50): {
                    "county": "Buncombe County",
                    "state": "North Carolina",
                    "name": "Asheville",
                },
            }
        )

        road_segments = [{"road": "US-74", "distance_km": 40.0}]
        towns = []

        result = _format_ost_from_route(
            road_segments,
            towns,
            geocoder=geocoder,
            waypoints=waypoints,
        )
        # Genuine SC→NC crossing kept.
        assert "SC" in result
        assert "NC" in result
        assert " to " in result

    def test_tied_mid_vote_keeps_both_states(self):
        """A 2-2 PA/MD mid vote is inconclusive: both states stay in the label
        (alphabetical pick would have demoted a real PA to MD drive to PA-only)."""
        from gomoveshift.pipeline.ingest.ost_formatter import _format_ost_from_route

        waypoints = (
            [_wp(34.00, -81.00)] * 8  # PA start (anchor at idx 1)
            + [_wp(35.00, -82.00)] * 2  # central: PA
            + [_wp(36.50, -83.50)]  # central: no geocode, casts no vote
            + [_wp(35.50, -82.50)] * 2  # central: MD
            + [_wp(35.50, -82.50)] * 7  # MD end (anchor at idx 18)
        )
        geocoder = _make_geocoder(
            {
                (34.00, -81.00): {"county": "Adams County", "state": "Pennsylvania", "name": None},
                (35.00, -82.00): {"county": "Adams County", "state": "Pennsylvania", "name": None},
                (35.50, -82.50): {"county": "Frederick County", "state": "Maryland", "name": None},
            }
        )
        result = _format_ost_from_route(
            [{"road": "US-15", "distance_km": 40.0}],
            [],
            geocoder=geocoder,
            waypoints=waypoints,
        )
        assert "PA" in result
        assert "MD" in result
