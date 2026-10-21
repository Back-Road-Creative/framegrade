"""U1 (title-accuracy program): location claims fail closed on missing evidence.

Each case is a defect found on the live channel on 2026-10-14 or a gate that
passed a claim it had no evidence for:

* the specificity ladder passed any label handed no evidence record;
* an unresolved county voted into the regions as ``"None, Virginia"``;
* an LF title said "Blountville, TN to McMullin, VA" while its description said
  "Started in Kingsport, ended in Seven Mile Ford" — two endpoint sources;
* a Short in Washington County, Virginia was tagged ``WA``.
"""

from __future__ import annotations

import pytest

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _assert_pre_encode_location_invariant,
)
from gomoveshift.pipeline.drive_phases.phase_d_metadata import _rescope_arc_to_lf
from gomoveshift.pipeline.gps_enrichment import build_resolved_location
from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator
from gomoveshift.pipeline.social_generator import SocialGenerator
from gomoveshift.pipeline.validation.content_gates import _vote_label, check_specificity_ladder

_LABEL = "Wytheville to Fort Chiswell via I-81, VA"


def _moving(n=101):
    return [{"lat": 36.95 + i * 0.0001, "lon": -81.08, "time": float(i)} for i in range(n)]


def _resolved() -> ResolvedLocation:
    rl = ResolvedLocation()
    rl.dominant_county, rl.dominant_state = "Wythe County", "Virginia"
    rl.regions = ["Wythe County, Virginia"]
    rl.vote = {"Wythe County, Virginia": 1.0}
    rl.ordered_place_sequence = ["Wytheville", "Fort Chiswell"]
    return rl


class TestLadderFailsClosed:
    def test_no_evidence_record_fails(self):
        res = check_specificity_ladder({"location": _LABEL}, None)
        assert res.passed is False
        assert "no location evidence" in res.message

    def test_blank_label_fails(self):
        res = check_specificity_ladder({"location": "  "}, {"gps_points": _moving()})
        assert res.passed is False

    def test_record_without_gps_or_corroboration_fails(self):
        # An operator label with nothing behind it: no trace, no voted county.
        res = check_specificity_ladder({"location": _LABEL}, {"gps_points": None})
        assert res.passed is False
        assert "no GPS trace or corroboration" in res.message

    def test_supported_label_still_passes(self):
        ev = {"gps_points": _moving(), "window": (0.0, 100.0)}
        assert check_specificity_ladder({"location": _LABEL}, ev).passed is True


class TestPreEncodeRequiresEvidence:
    def test_lf_with_model_but_no_evidence_raises(self):
        with pytest.raises(RuntimeError, match="no location evidence"):
            _assert_pre_encode_location_invariant({5: _LABEL}, {5: _resolved()}, evidence_by_lf={})

    def test_evidence_map_is_required(self):
        with pytest.raises(TypeError):
            _assert_pre_encode_location_invariant({5: _LABEL}, {5: _resolved()})


class _Geo:
    """Bristol, VA is an independent city: the geocoder returns no county."""

    def reverse_geocode_detailed(self, lat, lon):
        if lat > 36.6:
            return {"county": None, "state": "Virginia", "town": "Bristol"}
        return {"county": "Washington County", "state": "Virginia", "town": "Abingdon"}


class TestUnresolvedCountyNeverLabels:
    def test_vote_label_refuses_none_county(self):
        assert _vote_label("None|Virginia") is None
        assert _vote_label("|Virginia") is None
        assert _vote_label("Washington County|Virginia") == "Washington County, Virginia"

    def test_resolver_drops_unresolved_county_vote(self):
        pts = [{"lat": 36.70, "lon": -81.98}] * 4 + [{"lat": 36.50, "lon": -81.97}] * 8
        rl = build_resolved_location(pts, geocoder=_Geo())
        assert not any(r.startswith("None") for r in rl.regions)
        assert not any(k.startswith("None") for k in rl.vote)
        assert "Washington County, Virginia" in rl.regions


class TestDescriptionEndpointsMatchTitle:
    _SEQ = ["Kingsport", "Bristol", "Meadowview", "Seven Mile Ford"]

    def _profile(self, signature):
        rl = ResolvedLocation(ordered_place_sequence=list(self._SEQ), ost_signature=signature)
        profile = {"towns_traversed": [], "start_town": "X", "end_town": "Y"}
        _rescope_arc_to_lf(profile, rl)
        return profile

    def test_endpoint_label_drives_the_description_arc(self):
        profile = self._profile("Blountville, TN to McMullin, VA via I-81")
        assert (profile["start_town"], profile["end_town"]) == ("Blountville", "McMullin")
        desc = LongFormMetadataGenerator._enrich_with_route_profile(
            {"description": "", "tags": []}, profile
        )["description"]
        assert "Started in Blountville, ended in McMullin" in desc
        assert "Started in Kingsport" not in desc
        assert "ended in Seven Mile Ford" not in desc

    def test_non_endpoint_label_makes_no_endpoint_claim(self):
        profile = self._profile("Washington County, VA")
        assert not profile["start_town"] and not profile["end_town"]
        desc = LongFormMetadataGenerator._enrich_with_route_profile(
            {"description": "", "tags": []}, profile
        )["description"]
        assert "Started in" not in desc
        assert "ended in" not in desc


class TestStateTagFromStateSlot:
    @staticmethod
    def _tags(location):
        gen = SocialGenerator(location=location, project_name="u1-test")
        return [t.strip() for t in gen._build_tags(None).split(",")]

    def test_washington_county_virginia_tags_va_not_wa(self):
        tags = self._tags("Washington County, Virginia")
        assert "VA" in tags
        assert "WA" not in tags

    def test_bare_county_named_after_a_state_tags_no_state(self):
        assert "WA" not in self._tags("Washington County")

    def test_washington_state_still_tags_wa(self):
        assert "WA" in self._tags("Seattle, Washington")
