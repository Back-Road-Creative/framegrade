"""Unit tests for ``award_tier`` — the specificity-ladder tier decision.

Pure, deterministic map from (locked-GPS coverage, corroboration count,
bracket-route flag) to a specificity tier (1 = most specific, 5 = no location
claim). This is the demotion decision imported alongside
``gps_quality.locked_gps_coverage``. The decision and the
degraded-label rendering are both exercised here.
"""

from gomoveshift.pipeline.title_builder import (
    LOCKED_COVERAGE_TIER1_MIN,
    award_tier,
    render_degraded_tier_label,
)


class TestAwardTier:
    def test_high_coverage_is_tier1(self):
        assert award_tier(0.95, 0) == 1

    def test_coverage_at_tier1_boundary_is_tier1(self):
        # ≥ LOCKED_COVERAGE_TIER1_MIN (60%) trusts the full endpoints.
        assert LOCKED_COVERAGE_TIER1_MIN == 0.60
        assert award_tier(0.60, 0) == 1

    def test_below_boundary_two_corroborations_is_tier2(self):
        assert award_tier(0.59, 2) == 2

    def test_single_corroboration_is_tier3(self):
        assert award_tier(0.3, 1) == 3

    def test_bracket_route_alone_is_tier4(self):
        assert award_tier(0.1, 0, has_bracket_route=True) == 4

    def test_no_evidence_is_tier5(self):
        assert award_tier(0.0, 0) == 5


class TestRenderDegradedTierLabel:
    """The degraded burned-OST label forms for a below-tier-1 award.

    CONSERVATIVE forms (no named-corridor or genre words the
    pipeline can't mechanically source): tier2 ``{road}, {region}``, tier3
    ``{county}, {region}``, tier4 bare ``{state}``. Tier 1 renders through
    ``build_lf_title``; tier 5 is a genuine no-claim → both render nothing.
    """

    def test_tier2_is_road_and_region(self):
        assert render_degraded_tier_label(2, road="I-81", region="VA") == "I-81, VA"

    def test_tier3_is_county_and_region(self):
        assert (
            render_degraded_tier_label(3, county="Franklin County", region="PA")
            == "Franklin County, PA"
        )

    def test_tier4_is_bare_state(self):
        assert render_degraded_tier_label(4, state="Virginia", region="VA") == "Virginia"

    def test_missing_field_falls_to_broadest_supportable_form(self):
        # No road/county → broadest supportable claim, never a blank/fabrication.
        assert (
            render_degraded_tier_label(2, county="Wythe County", region="VA") == "Wythe County, VA"
        )
        assert render_degraded_tier_label(3, state="Virginia", region="VA") == "Virginia"

    def test_non_degraded_tiers_and_missing_fields_render_nothing(self):
        assert render_degraded_tier_label(1, road="I-81", region="VA") == ""
        assert render_degraded_tier_label(5, state="Virginia") == ""
        assert render_degraded_tier_label(4) == ""
