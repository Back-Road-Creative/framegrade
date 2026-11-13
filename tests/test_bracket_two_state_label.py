"""Bracket fixes in two states of one country name the leg, not just the country.

2026-11-13, nctopa-2026-03-05-rec0063: the fix before the GPS-less footage was in
Roanoke County, Virginia and the fix after in Franklin County, Pennsylvania, 17.5 h
apart. The only region both share is the country, so 2 long-forms and 21 Shorts were
labelled "United States" although the record measured both ends of the leg. The label
now names both states in travel order (level ``states``); it claims the leg between
two measured fixes, never where on it a clip sits.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline import bracket_route as br
from gomoveshift.pipeline import ost_system_review as osr
from gomoveshift.pipeline.validation.content_gates import award_tier_for_evidence

ROANOKE_VA = {"county": "Roanoke County", "state": "Virginia", "country": "United States"}
FRANKLIN_PA = {"county": "Franklin County", "state": "Pennsylvania", "country": "United States"}
KERRY = {"county": "County Kerry", "state": None, "country": "Ireland"}
CORK = {"county": "County Cork", "state": None, "country": "Ireland"}
ONTARIO = {"county": "York", "state": "Ontario", "country": "Canada"}


def _record(gap_h=17.535):
    return {
        "label": "Virginia to Pennsylvania",
        "level": "states",
        "gap_h": gap_h,
        "before": {"time": "2026-03-04T23:28:05Z"},
        "after": {"time": "2026-03-05T17:00:12Z"},
        "before_place": ROANOKE_VA,
        "after_place": FRANKLIN_PA,
    }


def test_two_states_in_one_country_name_both_in_travel_order():
    assert br.common_region(ROANOKE_VA, FRANKLIN_PA) == ("states", "Virginia to Pennsylvania")
    assert br.common_region(FRANKLIN_PA, ROANOKE_VA) == ("states", "Pennsylvania to Virginia")


def test_a_gap_past_the_state_cap_names_only_the_country():
    # 40 h is too long to prove the leg stayed between the two states' fixes.
    assert br.common_region(ROANOKE_VA, FRANKLIN_PA, ("country",)) == ("country", "United States")


def test_stateless_fixes_still_fall_to_the_country():
    assert br.common_region(KERRY, CORK) == ("country", "Ireland")


def test_one_fix_without_a_state_falls_to_the_country():
    no_state = {**FRANKLIN_PA, "state": None}
    assert br.common_region(ROANOKE_VA, no_state) == ("country", "United States")


def test_states_in_different_countries_are_still_refused():
    with pytest.raises(br.BracketRouteRefusedError):
        br.common_region(ROANOKE_VA, ONTARIO)


def test_the_states_level_has_the_state_cap_and_tier():
    assert br.BRACKET_MAX_GAP_H_BY_LEVEL["states"] == br.BRACKET_MAX_GAP_H_BY_LEVEL["state"]
    assert br.BRACKET_LEVEL_TIER["states"] == 4
    assert br.BRACKET_MAX_GAP_H == 48.0
    assert award_tier_for_evidence({"gps_points": None, "bracket": _record()}) == (None, 4)


def test_a_states_record_is_valid_inside_its_cap_only():
    assert br.bracket_record_valid(_record()) is True
    assert br.bracket_record_valid(_record(gap_h=40.0)) is False


def test_the_review_reason_names_both_ends_of_the_leg():
    reason = br.bracket_outcome(_record())["reason"]
    assert "Virginia" in reason and "Pennsylvania" in reason
    assert "both geocode" not in reason


def test_a_short_burning_the_leg_label_earns_the_bracket_source():
    evidence = {"bracket_route": _record(), "placeless_label": None}
    assert osr.unmeasured_place_source(evidence, "Virginia to Pennsylvania") == osr.BRACKET_SOURCE
    assert osr.unmeasured_place_source(evidence, "United States") is None
