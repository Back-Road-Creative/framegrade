"""Tests for state abbreviation stripping in parse_ost_components.

Bug: "Colleton County to Beaufort County, SC" parses end as
"Beaufort County, SC" instead of "Beaufort County", even though state="SC"
is correctly extracted. This causes double-state in downstream titles:
"Colleton County to Beaufort County, SC, SC".

Fix: when the state abbreviation is extracted from the end of an "X to Y"
format, strip it from the endpoint name too. Same for cross-state: when
both endpoints embed state ("Beaufort County, SC to Cashiers, NC"),
strip the state from both start and end.
"""

from gomoveshift.pipeline.ingest.ost_formatter import parse_ost_components


class TestStateStrippingFromEndpoints:
    """parse_ost_components strips state abbreviation from endpoint names."""

    def test_state_stripped_from_end_town(self):
        """'A to B, SC' → end='B', state='SC'."""
        result = parse_ost_components("Colleton County to Beaufort County, SC")
        assert result["end"] == "Beaufort County"
        assert result["state"] == "SC"
        assert result["start"] == "Colleton County"

    def test_cross_state_both_stripped(self):
        """'A, SC to B, NC' → start='A', end='B', start has no state."""
        result = parse_ost_components("Beaufort County, SC to Cashiers, NC")
        assert result["start"] == "Beaufort County"
        assert result["end"] == "Cashiers"
        assert result["state"] == "NC"  # Last state wins

    def test_cross_state_preserves_both_states(self):
        """Cross-state: need both states accessible."""
        result = parse_ost_components("Beaufort County, SC to Cashiers, NC")
        # start_state and end_state should be available for cross-state formatting
        assert result.get("start_state") == "SC"
        assert result["state"] == "NC"

    def test_no_state_in_endpoint_unchanged(self):
        """'A to B via Road' → no state, endpoints unchanged."""
        result = parse_ost_components("Frogmore to Port Royal via Overland Trail")
        assert result["start"] == "Frogmore"
        assert result["end"] == "Port Royal"
        assert result.get("state") is None

    def test_via_road_with_state(self):
        """'A to B via Road, SC' → road extracted, state from end."""
        result = parse_ost_components("Port Royal to Beaufort via Trask Parkway")
        assert result["start"] == "Port Royal"
        assert result["end"] == "Beaufort"
        assert result["road"] == "Trask Parkway"

    def test_single_comma_state(self):
        """'Place, SC' → start/end='Place', state='SC'."""
        result = parse_ost_components("Beaufort, SC")
        assert result["start"] == "Beaufort"
        assert result["state"] == "SC"

    def test_three_part_road_county_state(self):
        """'Road, County, SC' → road, start/end, state all clean."""
        result = parse_ost_components("Stocks Creek Road, Colleton County, SC")
        assert result["road"] == "Stocks Creek Road"
        assert result["start"] == "Colleton County"
        assert result["state"] == "SC"


# TestFormatBaseTitleNoDoubleState removed: _format_base_title lived in the
# deleted phase_c.py (non-day-first deletion). The endpoint-state
# stripping behavior it relied on is still covered by TestStateStrippingFromEndpoints
# above, which exercises parse_ost_components directly.
