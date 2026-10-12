"""Location coverage gates: a label may not omit a county with material footage.

A label must not name one location while the footage shows another: a
drive that split roughly 56/44 between two counties must not ship a label
naming only the first (the old gate only WARNed).
"""

from __future__ import annotations

import pytest

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _assert_pre_encode_location_invariant,
)
from gomoveshift.pipeline.gps_enrichment import build_resolved_location
from gomoveshift.pipeline.validation.content_gates import (
    check_location_invariant,
    check_short_location_invariant,
    resolved_location_gate_dict,
)

MONT = "Montgomery County, Virginia"
ROAN = "Roanoke County, Virginia"


def _meta(title, location=None, *, vote=None, endpoint_regions=None, description=""):
    resolved = {
        "dominant_county": "Montgomery County",
        "dominant_state": "Virginia",
        "regions": [MONT, ROAN],
        "vote": vote or {"Montgomery County|Virginia": 0.562, "Roanoke County|Virginia": 0.437},
    }
    if endpoint_regions is not None:
        resolved["endpoint_regions"] = endpoint_regions
    meta = {"title": title, "location": location or title, "resolved_location": resolved}
    if description:
        meta["description"] = description
    return meta


class TestLfCoverage:
    def test_radford_one_county_label_fails_naming_omitted_county(self):
        res = check_location_invariant(_meta(MONT))
        assert res.passed is False
        assert "HARD FAIL" in res.message
        assert "Roanoke County, Virginia" in res.message
        assert "44%" in res.message

    def test_endpoint_towns_in_both_counties_pass(self):
        res = check_location_invariant(
            _meta(
                "Christiansburg to Salem via US-11, VA",
                endpoint_regions={"start": MONT, "end": ROAN},
            )
        )
        assert res.passed is True, res.message

    def test_both_endpoints_in_one_county_fail(self):
        res = check_location_invariant(
            _meta(
                "Christiansburg to Salem via US-11, VA",
                endpoint_regions={"start": MONT, "end": MONT},
            )
        )
        assert res.passed is False
        assert "Roanoke County, Virginia" in res.message

    def test_endpoint_towns_without_endpoint_regions_fail(self):
        res = check_location_invariant(_meta("Christiansburg to Salem via US-11, VA"))
        assert res.passed is False

    def test_below_material_floor_passes(self):
        res = check_location_invariant(
            _meta(MONT, vote={"Montgomery County|Virginia": 0.85, "Roanoke County|Virginia": 0.15})
        )
        assert res.passed is True, res.message

    def test_unanimous_vote_passes_unchanged(self):
        meta = _meta(MONT, vote={"Montgomery County|Virginia": 1.0})
        meta["resolved_location"]["regions"] = [MONT]
        res = check_location_invariant(meta)
        assert res.passed is True
        assert res.message == "Location invariant checks passed"

    def test_description_naming_the_second_county_passes(self):
        res = check_location_invariant(
            _meta(MONT, description="📍 Location: Montgomery County & Roanoke County, Virginia\n")
        )
        assert res.passed is True, res.message

    def test_comma_form_vote_keys_are_understood(self):
        res = check_location_invariant(_meta(MONT, vote={MONT: 0.56, ROAN: 0.44}))
        assert res.passed is False
        assert "Roanoke" in res.message


class TestShortCoverage:
    def test_two_county_sampled_vote_fails_even_when_burn_matches_one(self):
        res = check_short_location_invariant(
            MONT, [MONT, ROAN], sampled_vote={MONT: 0.97, ROAN: 0.03}
        )
        assert res.passed is False
        assert "clip GPS spans" in res.message
        assert "Roanoke" in res.message and "Montgomery" in res.message

    def test_one_county_vote_passes(self):
        assert check_short_location_invariant(MONT, [MONT], sampled_vote={MONT: 1.0}).passed

    def test_none_vote_keeps_membership_behaviour(self):
        assert check_short_location_invariant(MONT, [MONT, ROAN]).passed is True
        assert check_short_location_invariant(MONT, [MONT, ROAN], sampled_vote=None).passed


class _Geo:
    def reverse_geocode_detailed(self, lat, lon):
        if lat > 32.5:
            return {"county": "Colleton County", "state": "South Carolina", "town": "Edisto"}
        return {"county": "Beaufort County", "state": "South Carolina", "town": "Fripp"}


class TestEndpointRegions:
    def test_builder_populates_endpoint_regions_and_gate_dict_carries_it(self):
        pts = [(32.3095, -80.6332)] * 8 + [(32.6591, -80.6646)] * 2
        rl = build_resolved_location(pts, geocoder=_Geo())
        assert rl.endpoint_regions == {
            "start": "Beaufort County, South Carolina",
            "end": "Colleton County, South Carolina",
        }
        assert resolved_location_gate_dict(rl)["endpoint_regions"] == rl.endpoint_regions

    def test_withheld_endpoint_is_not_geocoded(self):
        pts = [(32.3095, -80.6332)] * 8 + [(32.6591, -80.6646)] * 2
        rl = build_resolved_location(pts, geocoder=_Geo(), withhold_endpoints={"end"})
        assert list(rl.endpoint_regions) == ["start"]

    def test_legacy_model_defaults_empty(self):
        assert ResolvedLocation().endpoint_regions == {}


def test_pre_encode_gate_raises_on_radford_one_county_label():
    rl = ResolvedLocation(
        dominant_county="Montgomery County",
        dominant_state="Virginia",
        regions=[MONT, ROAN],
        vote={MONT: 0.562, ROAN: 0.437},
    )
    with pytest.raises(RuntimeError, match="Roanoke County"):
        _assert_pre_encode_location_invariant({1: MONT}, {1: rl})
