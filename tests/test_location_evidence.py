"""U1b (title-accuracy program): the location-evidence record.

One record per footage segment, built from that segment's own GPS fixes. Every
place carries a scope (in / through / near), a method, a provenance and a
stable ``evidence_id`` so Waypost's ``Place(name, evidence_id)`` can pass it
straight through. A place claim with no record behind it is refused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gomoveshift.pipeline.location_evidence import (
    LocationEvidence,
    build_location_evidence,
    title_claim_issues,
)
from tools import retitle_apply as ra

# A synthetic north-to-south line: each 0.1° band of latitude is one town.
_TOWNS = {
    0: ("Alder", "Ash County", "Virginia"),
    1: ("Birch", "Ash County", "Virginia"),
    2: ("Cedar", "Elm County", "Virginia"),
    3: ("Dogwood", "Elm County", "Virginia"),
}


class _Geo:
    def reverse_geocode_detailed(self, lat, lon):
        town, county, state = _TOWNS[int(round((37.0 - lat) * 10))]
        return {"town": town, "county": county, "state": state}


def _trace():
    # 4 fixes per town, 10 s apart: t = 0..150.
    return [
        {"lat": 37.0 - band / 10, "lon": -81.0, "time": float(band * 40 + k * 10)}
        for band in range(4)
        for k in range(4)
    ]


def _record(**kw):
    return build_location_evidence(_trace(), geocoder=_Geo(), segment_id="lf01", **kw)


def _forward(coords):
    return lambda name: coords.get(name)


class TestRecord:
    def test_places_are_ordered_with_scope(self):
        rec = _record()
        assert [p.name for p in rec.places] == ["Alder", "Birch", "Cedar", "Dogwood"]
        assert [p.scope for p in rec.places] == ["in", "through", "through", "in"]
        assert {p.method for p in rec.places} == {"reverse_geocode"}
        assert {p.provenance for p in rec.places} == {"gps"}

    def test_evidence_id_is_stable_and_distinct(self):
        a, b = _record(), _record()
        assert [p.evidence_id for p in a.places] == [p.evidence_id for p in b.places]
        assert len({p.evidence_id for p in a.places}) == 4
        assert all(p.evidence_id for p in a.places)

    def test_round_trips_through_json(self):
        rec = _record()
        assert LocationEvidence.from_dict(json.loads(json.dumps(rec.to_dict()))) == rec

    def test_full_journey_gps_on_a_short_clip_uses_only_the_clip_window(self):
        # A short cut from t=80..110 only ever saw Cedar.
        rec = _record(window=(80.0, 110.0))
        assert [p.name for p in rec.places] == ["Cedar"]
        assert title_claim_issues("Alder to Dogwood, VA", rec)

    def test_unresolved_town_is_no_place(self):
        class _Blank(_Geo):
            def reverse_geocode_detailed(self, lat, lon):
                return {"town": None, "county": "None", "state": "Virginia"}

        rec = build_location_evidence(_trace(), geocoder=_Blank(), segment_id="lf01")
        assert rec.places == []


class TestClaims:
    def test_supported_endpoints_pass(self):
        assert title_claim_issues("Alder to Dogwood, VA", _record()) == []

    def test_a_to_b_with_an_unentered_endpoint_fails(self):
        issues = title_claim_issues("Alder to Elkton, VA", _record())
        assert any("Elkton" in i for i in issues)

    def test_reversed_endpoints_fail(self):
        assert title_claim_issues("Dogwood to Alder, VA", _record())

    def test_nearby_not_entered_is_near_not_in(self):
        # Fernwood geocodes 3 km off the trace but no fix ever resolved to it.
        rec = _record()
        assert rec.scope_of("Fernwood", geocode_fn=_forward({"Fernwood": (36.85, -81.03)})) == (
            "near"
        )
        issues = title_claim_issues(
            "Alder to Fernwood, VA", rec, geocode_fn=_forward({"Fernwood": (36.85, -81.03)})
        )
        assert any("Fernwood" in i and "near" in i for i in issues)

    def test_same_name_place_needs_its_state(self):
        rec = _record()
        rec.places.append(
            rec.places[0].__class__(
                name="Birch",
                county="Birch County",
                state="Washington",
                scope="through",
                method="reverse_geocode",
                provenance="gps",
                segment_id="lf01",
                evidence_id="x",
            )
        )
        assert rec.place("Birch") is None
        assert any("Birch" in c for c in rec.conflicts)
        assert rec.place("Birch", state="Virginia").county == "Ash County"

    def test_claim_without_a_record_fails(self):
        assert title_claim_issues("Alder to Dogwood, VA", None) == ["no location evidence record"]


class TestRetitleApplyConsumesTheRecord:
    def _row(self, title, evidence):
        return {
            "video_id": "v",
            "current_title": "old",
            "suggested_title": title,
            "location_evidence": json.dumps(evidence.to_dict()) if evidence else "",
        }

    def test_operator_title_without_evidence_is_refused(self):
        res = ra.dry_run([self._row("Alder to Dogwood, VA", None)], out=open("/dev/null", "w"))
        assert res[0]["verdict"] == ra.REFUSED

    def test_unsupported_endpoint_is_refused(self):
        res = ra.dry_run([self._row("Alder to Elkton, VA", _record())], out=open("/dev/null", "w"))
        assert res[0]["verdict"] == ra.REFUSED

    def test_supported_title_is_approved(self):
        res = ra.dry_run([self._row("Alder to Dogwood, VA", _record())], out=open("/dev/null", "w"))
        assert res[0]["verdict"] == ra.APPROVED
