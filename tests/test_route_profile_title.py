"""Exhaustive tests for RouteProfile.title() priority chain.

These tests define the contract for the 7-shape title derivation system.
They cover RouteProfile.title(): which shape wins for each combination of
towns, road, feature and counties.

Priority order (first match wins):
  1. Two towns + road
  2. Two towns, no road
  3. Feature + road (towns insufficient)
  4. One town + road
  5. Road + county
  6. County to county (GPS-verified only)
  7. Road + state (full name)
  8. Fallback: empty string
"""

import pytest

try:
    from gomoveshift.pipeline.route_profile import ClipGeocode, RouteProfile
except ImportError:
    pytest.skip("route_profile module not yet available", allow_module_level=True)


def _make_profile(**overrides) -> RouteProfile:
    """Create a RouteProfile with sensible defaults, applying overrides."""
    defaults = {
        "lf_index": 1,
        "clips": [],
        "start_town": None,
        "end_town": None,
        "dominant_road": None,
        "states_traversed": [],
        "counties_traversed": [],
        "towns_traversed": [],
        "roads_traversed": [],
        "features": [],
        "is_cross_state": False,
        "is_cross_county": False,
    }
    defaults.update(overrides)
    return RouteProfile(**defaults)


# ---------------------------------------------------------------------------
# Shape 1: Two towns + road
# ---------------------------------------------------------------------------
class TestTitleShape1TwoTownsWithRoad:
    """start_town != end_town, dominant_road set -> 'Start to End via Road, ST'."""

    def test_basic(self):
        p = _make_profile(
            start_town="Port Royal",
            end_town="Beaufort",
            dominant_road="Trask Parkway",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Port Royal to Beaufort via Trask Parkway, SC"

    def test_via_road_formatting(self):
        p = _make_profile(
            start_town="Frogmore",
            end_town="Port Royal",
            dominant_road="Sea Island Parkway",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Frogmore to Port Royal via Sea Island Parkway, SC"

    def test_long_road_name(self):
        p = _make_profile(
            start_town="Beaufort",
            end_town="Hilton Head Island",
            dominant_road="William Hilton Parkway / US-278 Business",
            states_traversed=["South Carolina"],
        )
        assert (
            p.title()
            == "Beaufort to Hilton Head Island via William Hilton Parkway / US-278 Business, SC"
        )

    def test_no_state_traversed(self):
        """When states_traversed is empty, no trailing state abbreviation."""
        p = _make_profile(
            start_town="Port Royal",
            end_town="Beaufort",
            dominant_road="Trask Parkway",
            states_traversed=[],
        )
        assert p.title() == "Port Royal to Beaufort via Trask Parkway"

    def test_multiple_states_uses_last(self):
        """Multi-state list uses the last state for the trailing abbreviation."""
        p = _make_profile(
            start_town="Port Royal",
            end_town="Beaufort",
            dominant_road="Trask Parkway",
            states_traversed=["Georgia", "South Carolina"],
        )
        assert p.title() == "Port Royal to Beaufort via Trask Parkway, SC"


# ---------------------------------------------------------------------------
# Shape 2: Two towns, no road
# ---------------------------------------------------------------------------
class TestTitleShape2TwoTownsNoRoad:
    """start_town != end_town, no dominant_road -> 'Start to End, ST'."""

    def test_basic(self):
        p = _make_profile(
            start_town="Port Royal",
            end_town="Beaufort",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Port Royal to Beaufort, SC"

    def test_without_state(self):
        """No state -> no trailing comma."""
        p = _make_profile(start_town="Port Royal", end_town="Beaufort")
        assert p.title() == "Port Royal to Beaufort"


# ---------------------------------------------------------------------------
# Shape 3: Feature + road
# ---------------------------------------------------------------------------
class TestTitleShape3FeatureWithRoad:
    """features non-empty, dominant_road set, towns insufficient -> 'Feature via Road, ST'."""

    def test_basic(self):
        p = _make_profile(
            features=["Hunting Island"],
            dominant_road="US-21",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Hunting Island via US-21, SC"

    def test_feature_only_no_towns_needed(self):
        """Feature shape works without any town data."""
        p = _make_profile(
            features=["Blue Ridge Parkway Overlook"],
            dominant_road="Blue Ridge Parkway",
            states_traversed=["North Carolina"],
        )
        assert p.title() == "Blue Ridge Parkway Overlook via Blue Ridge Parkway, NC"

    def test_feature_without_road(self):
        """Feature without road falls through to later shapes (not shape 3)."""
        p = _make_profile(
            features=["Hunting Island"],
            states_traversed=["South Carolina"],
        )
        # Shape 3 requires road — should NOT produce "Hunting Island, SC"
        # Falls through; exact result depends on remaining fields
        result = p.title()
        assert "via" not in result

    def test_feature_uses_first_entry(self):
        """When multiple features, use the first one."""
        p = _make_profile(
            features=["Hunting Island", "Harbor Island"],
            dominant_road="US-21",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Hunting Island via US-21, SC"

    def test_feature_without_state(self):
        p = _make_profile(features=["Hunting Island"], dominant_road="US-21")
        assert p.title() == "Hunting Island via US-21"


# ---------------------------------------------------------------------------
# Shape 4: One town + road
# ---------------------------------------------------------------------------
class TestTitleShape4OneTownWithRoad:
    """Only one town set + dominant_road -> 'Road, Town, ST'."""

    def test_start_town_only(self):
        p = _make_profile(
            start_town="Beaufort",
            dominant_road="US-21",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "US-21, Beaufort, SC"

    def test_end_town_only(self):
        p = _make_profile(
            end_town="Beaufort",
            dominant_road="US-21",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "US-21, Beaufort, SC"

    def test_without_state(self):
        p = _make_profile(start_town="Beaufort", dominant_road="US-21")
        assert p.title() == "US-21, Beaufort"

    def test_same_start_end_with_road(self):
        """Roundtrip (start == end) with road -> shape 4 format."""
        p = _make_profile(
            start_town="Beaufort",
            end_town="Beaufort",
            dominant_road="US-21",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "US-21, Beaufort, SC"


# ---------------------------------------------------------------------------
# Shape 5: Road + county
# ---------------------------------------------------------------------------
class TestTitleShape5RoadWithCounty:
    """No towns, no features, dominant_road + counties -> 'Road, County, ST'."""

    def test_basic(self):
        p = _make_profile(
            dominant_road="Stocks Creek Road",
            counties_traversed=["Colleton County"],
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Stocks Creek Road, Colleton County, SC"

    def test_multiple_counties_uses_first(self):
        p = _make_profile(
            dominant_road="US-17",
            counties_traversed=["Colleton County", "Beaufort County"],
            states_traversed=["South Carolina"],
        )
        assert p.title() == "US-17, Colleton County, SC"

    def test_without_state(self):
        p = _make_profile(
            dominant_road="Stocks Creek Road",
            counties_traversed=["Colleton County"],
        )
        assert p.title() == "Stocks Creek Road, Colleton County"


# ---------------------------------------------------------------------------
# Shape 6: County to county (GPS-verified)
# ---------------------------------------------------------------------------
class TestTitleShape6CrossCounty:
    """is_cross_county=True, 2+ counties, no road/towns/features -> 'C1 to C2, ST'."""

    def test_gps_verified(self):
        p = _make_profile(
            is_cross_county=True,
            counties_traversed=["Colleton County", "Beaufort County"],
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Colleton County to Beaufort County, SC"

    def test_not_verified_falls_through(self):
        """is_cross_county=False -> shape 6 must NOT fire even with 2+ counties."""
        p = _make_profile(
            is_cross_county=False,
            counties_traversed=["Colleton County", "Beaufort County"],
            states_traversed=["South Carolina"],
        )
        # Should NOT produce "Colleton County to Beaufort County, SC"
        result = p.title()
        assert result != "Colleton County to Beaufort County, SC"

    def test_single_county_not_enough(self):
        """Even with is_cross_county=True, need 2+ counties."""
        p = _make_profile(
            is_cross_county=True,
            counties_traversed=["Colleton County"],
            states_traversed=["South Carolina"],
        )
        result = p.title()
        assert "to" not in result

    def test_without_state(self):
        p = _make_profile(
            is_cross_county=True,
            counties_traversed=["Colleton County", "Beaufort County"],
        )
        assert p.title() == "Colleton County to Beaufort County"


# ---------------------------------------------------------------------------
# Shape 7: Road + state
# ---------------------------------------------------------------------------
class TestTitleShape7RoadWithState:
    """dominant_road + states_traversed, nothing else -> 'Road, StateName' (full name)."""

    def test_basic(self):
        p = _make_profile(
            dominant_road="US-17",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "US-17, South Carolina"

    def test_uses_full_state_name_not_abbreviation(self):
        p = _make_profile(
            dominant_road="Blue Ridge Parkway",
            states_traversed=["North Carolina"],
        )
        assert p.title() == "Blue Ridge Parkway, North Carolina"

    def test_multiple_states_uses_last(self):
        p = _make_profile(
            dominant_road="I-95",
            states_traversed=["Georgia", "South Carolina"],
        )
        assert p.title() == "I-95, South Carolina"


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------
class TestTitleFallback:
    """Completely empty profile -> empty string."""

    def test_empty_profile(self):
        p = RouteProfile()
        assert p.title() == ""

    def test_only_lf_index(self):
        p = _make_profile(lf_index=5)
        assert p.title() == ""

    def test_only_clips(self):
        p = _make_profile(clips=[ClipGeocode(lat=32.0, lon=-80.0)])
        assert p.title() == ""

    def test_only_towns_traversed_no_start_end(self):
        """towns_traversed without start/end doesn't trigger any shape."""
        p = _make_profile(towns_traversed=["Beaufort", "Port Royal"])
        assert p.title() == ""


# ---------------------------------------------------------------------------
# Cross-state handling
# ---------------------------------------------------------------------------
class TestTitleCrossState:
    """is_cross_state=True modifies shapes 1 and 2."""

    def test_two_towns_cross_state_no_road(self):
        """'Town, ST1 to Town, ST2' — each endpoint gets its own abbreviation."""
        p = _make_profile(
            start_town="Beaufort",
            end_town="Highlands",
            is_cross_state=True,
            states_traversed=["South Carolina", "North Carolina"],
        )
        assert p.title() == "Beaufort, SC to Highlands, NC"

    def test_two_towns_cross_state_with_road(self):
        """Cross-state with road -> 'Town, ST1 to Town, ST2 via Road'."""
        p = _make_profile(
            start_town="Beaufort",
            end_town="Highlands",
            dominant_road="Blue Ridge Parkway",
            is_cross_state=True,
            states_traversed=["South Carolina", "North Carolina"],
        )
        assert p.title() == "Beaufort, SC to Highlands, NC via Blue Ridge Parkway"

    def test_cross_state_no_trailing_state(self):
        """Cross-state must NOT append a trailing state abbreviation."""
        p = _make_profile(
            start_town="Beaufort",
            end_town="Highlands",
            is_cross_state=True,
            states_traversed=["South Carolina", "North Carolina"],
        )
        result = p.title()
        # Should not end with ", NC" or ", SC" as a trailing suffix
        # The format is "Beaufort, SC to Highlands, NC" — NC is part of
        # the endpoint, not a trailing state
        assert result == "Beaufort, SC to Highlands, NC"
        assert not result.endswith(", NC,")

    def test_cross_state_with_single_town_falls_to_single_town(self):
        """Cross-state with only one town -> standard single-town format."""
        p = _make_profile(
            start_town="Beaufort",
            is_cross_state=True,
            dominant_road="US-21",
            states_traversed=["South Carolina", "North Carolina"],
        )
        # Only one town -> shape 4: "Road, Town, ST"
        assert p.title() == "US-21, Beaufort, SC"

    def test_cross_state_first_state_for_start_last_for_end(self):
        """Start town uses first state, end town uses last state."""
        p = _make_profile(
            start_town="Savannah",
            end_town="Beaufort",
            is_cross_state=True,
            states_traversed=["Georgia", "South Carolina"],
        )
        assert p.title() == "Savannah, GA to Beaufort, SC"

    def test_cross_state_three_states(self):
        """Three states: start=first, end=last."""
        p = _make_profile(
            start_town="Savannah",
            end_town="Asheville",
            is_cross_state=True,
            states_traversed=["Georgia", "South Carolina", "North Carolina"],
        )
        assert p.title() == "Savannah, GA to Asheville, NC"


# ---------------------------------------------------------------------------
# Roundtrip (start == end)
# ---------------------------------------------------------------------------
class TestTitleRoundtrip:
    """start_town == end_town -> treated as single town."""

    def test_same_town_becomes_single_town(self):
        """start == end -> 'Town, ST' (not 'Town to Town')."""
        p = _make_profile(
            start_town="Beaufort",
            end_town="Beaufort",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Beaufort, SC"

    def test_same_town_with_road_shape4(self):
        """start == end + road -> shape 4: 'Road, Town, ST'."""
        p = _make_profile(
            start_town="Beaufort",
            end_town="Beaufort",
            dominant_road="US-21",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "US-21, Beaufort, SC"

    def test_same_town_no_state(self):
        p = _make_profile(start_town="Beaufort", end_town="Beaufort")
        assert p.title() == "Beaufort"


# ---------------------------------------------------------------------------
# Priority order
# ---------------------------------------------------------------------------
class TestTitlePriorityOrder:
    """When multiple shapes could match, higher priority wins."""

    def test_shape1_beats_shape3(self):
        """Two towns + road + feature -> shape 1 wins (towns + road)."""
        p = _make_profile(
            start_town="Port Royal",
            end_town="Beaufort",
            dominant_road="Trask Parkway",
            features=["Hunting Island"],
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Port Royal to Beaufort via Trask Parkway, SC"

    def test_shape1_beats_shape5(self):
        """Two towns + road + county -> shape 1 wins."""
        p = _make_profile(
            start_town="Port Royal",
            end_town="Beaufort",
            dominant_road="Trask Parkway",
            counties_traversed=["Beaufort County"],
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Port Royal to Beaufort via Trask Parkway, SC"

    def test_shape2_beats_shape3(self):
        """Two towns (no road) + feature -> shape 2 wins."""
        p = _make_profile(
            start_town="Port Royal",
            end_town="Beaufort",
            features=["Hunting Island"],
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Port Royal to Beaufort, SC"

    def test_shape3_beats_shape4(self):
        """Feature + road + one town -> shape 3 wins (feature + road)."""
        p = _make_profile(
            start_town="Beaufort",
            features=["Hunting Island"],
            dominant_road="US-21",
            states_traversed=["South Carolina"],
        )
        # Shape 3 should fire: features non-empty + road set + towns insufficient
        # (only one town, so two-town shapes don't apply, but feature+road does)
        assert p.title() == "Hunting Island via US-21, SC"

    def test_shape5_beats_shape6(self):
        """Road + county + is_cross_county -> shape 5 wins (road present)."""
        p = _make_profile(
            dominant_road="US-17",
            counties_traversed=["Colleton County", "Beaufort County"],
            is_cross_county=True,
            states_traversed=["South Carolina"],
        )
        assert p.title() == "US-17, Colleton County, SC"

    def test_shape6_only_fires_when_verified(self):
        """Shape 6 requires is_cross_county=True; False skips to shape 7."""
        p = _make_profile(
            counties_traversed=["Colleton County", "Beaufort County"],
            is_cross_county=False,
            dominant_road="US-17",
            states_traversed=["South Carolina"],
        )
        # Has road -> shape 5 applies (Road, County, ST)
        assert p.title() == "US-17, Colleton County, SC"

    def test_shape6_without_road_skips_to_7_when_unverified(self):
        """No road + is_cross_county=False + states -> falls to shape 7? No road either -> fallback."""
        p = _make_profile(
            counties_traversed=["Colleton County", "Beaufort County"],
            is_cross_county=False,
            states_traversed=["South Carolina"],
        )
        # No road, no towns, no features, cross_county=False
        # Shape 7 requires road -> also fails -> fallback
        assert p.title() == ""


# ---------------------------------------------------------------------------
# Base-title cases: via-road, roundtrip and cross-state
# ---------------------------------------------------------------------------
class TestBaseTitleCases:
    """RouteProfile.title() covers the base-title cases."""

    def test_two_towns_with_road(self):
        """Port Royal to Lady's Island via Sea Island Parkway."""
        p = _make_profile(
            start_town="Port Royal",
            end_town="Lady's Island",
            dominant_road="Sea Island Parkway",
        )
        assert p.title() == "Port Royal to Lady's Island via Sea Island Parkway"

    def test_two_towns_without_road(self):
        p = _make_profile(start_town="Port Royal", end_town="Lady's Island")
        assert p.title() == "Port Royal to Lady's Island"

    def test_two_towns_with_road_and_state(self):
        p = _make_profile(
            start_town="Frogmore",
            end_town="Port Royal",
            dominant_road="Trask Parkway",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Frogmore to Port Royal via Trask Parkway, SC"

    def test_two_towns_with_state_no_road(self):
        p = _make_profile(
            start_town="Frogmore",
            end_town="Port Royal",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Frogmore to Port Royal, SC"

    def test_same_start_end_no_road(self):
        """Roundtrip without turnaround -> single place: 'Beaufort, SC'."""
        p = _make_profile(
            start_town="Beaufort",
            end_town="Beaufort",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Beaufort, SC"

    def test_cross_state_with_road(self):
        """Cross-state: 'Beaufort, SC to Highlands, NC via Blue Ridge Parkway'."""
        p = _make_profile(
            start_town="Beaufort",
            end_town="Highlands",
            dominant_road="Blue Ridge Parkway",
            is_cross_state=True,
            states_traversed=["South Carolina", "North Carolina"],
        )
        assert p.title() == "Beaufort, SC to Highlands, NC via Blue Ridge Parkway"

    def test_cross_state_no_road(self):
        p = _make_profile(
            start_town="Beaufort",
            end_town="Highlands",
            is_cross_state=True,
            states_traversed=["South Carolina", "North Carolina"],
        )
        assert p.title() == "Beaufort, SC to Highlands, NC"


# ---------------------------------------------------------------------------
# State abbreviation edge cases
# ---------------------------------------------------------------------------
class TestTitleStateAbbreviation:
    """US states -> 2-letter abbreviation; non-US stays full name."""

    def test_south_carolina(self):
        p = _make_profile(
            start_town="Beaufort",
            end_town="Port Royal",
            states_traversed=["South Carolina"],
        )
        assert p.title() == "Beaufort to Port Royal, SC"

    def test_non_us_state_stays_full(self):
        """Unrecognized/non-US state -> use full name."""
        p = _make_profile(
            start_town="Quebec City",
            end_town="Montreal",
            states_traversed=["Quebec"],
        )
        assert p.title() == "Quebec City to Montreal, Quebec"

    def test_multiple_us_states(self):
        """Last state in list is used for trailing abbreviation."""
        p = _make_profile(
            start_town="Port Royal",
            end_town="Beaufort",
            dominant_road="US-21",
            states_traversed=["Georgia", "South Carolina"],
        )
        assert p.title() == "Port Royal to Beaufort via US-21, SC"


# ---------------------------------------------------------------------------
# Via-road span guard: drop a far local road on a long crossing
# ---------------------------------------------------------------------------
def _clips_far():
    """Two clip centroids ~370 km apart (coastal SC → NC mountains)."""
    return [ClipGeocode(lat=32.40, lon=-80.40), ClipGeocode(lat=35.60, lon=-82.90)]


def _clips_near():
    """Two clip centroids ~16 km apart (a short same-area drive)."""
    return [ClipGeocode(lat=32.40, lon=-80.40), ClipGeocode(lat=32.50, lon=-80.50)]


class TestViaRoadSpanGuard:
    """A 'via {road}' suffix is suppressed when endpoints are >50 km apart."""

    def test_cross_state_long_span_drops_via_road(self):
        p = _make_profile(
            start_town="Beaufort",
            end_town="Cashiers",
            dominant_road="Hunting Island Drive",
            states_traversed=["South Carolina", "North Carolina"],
            is_cross_state=True,
            clips=_clips_far(),
        )
        assert p.title() == "Beaufort, SC to Cashiers, NC"

    def test_cross_state_short_span_keeps_via_road(self):
        p = _make_profile(
            start_town="Savannah",
            end_town="Hardeeville",
            dominant_road="US-17",
            states_traversed=["Georgia", "South Carolina"],
            is_cross_state=True,
            clips=_clips_near(),
        )
        assert p.title() == "Savannah, GA to Hardeeville, SC via US-17"

    def test_same_state_long_span_drops_via_road(self):
        p = _make_profile(
            start_town="Beaufort",
            end_town="Greenville",
            dominant_road="Hunting Island Drive",
            states_traversed=["South Carolina"],
            clips=_clips_far(),
        )
        assert p.title() == "Beaufort to Greenville, SC"

    def test_single_town_long_span_drops_far_road(self):
        # A bad cross-state town was dropped, leaving a single NC endpoint
        # and the coastal-SC dominant road; the road must NOT stamp next to
        # the NC town ("Hunting Island Drive, Oakland, NC").
        p = _make_profile(
            start_town=None,
            end_town="Oakland",
            dominant_road="Hunting Island Drive",
            states_traversed=["South Carolina", "North Carolina"],
            is_cross_state=True,
            clips=_clips_far(),
        )
        assert p.title() == "Oakland, NC"

    def test_single_town_short_span_keeps_road(self):
        # A genuine short single-town drive keeps its local road (no regression).
        p = _make_profile(
            start_town=None,
            end_town="Beaufort",
            dominant_road="Sea Island Parkway",
            states_traversed=["South Carolina"],
            clips=_clips_near(),
        )
        assert p.title() == "Sea Island Parkway, Beaufort, SC"
