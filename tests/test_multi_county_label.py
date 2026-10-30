"""A drive through two counties is labelled by both, not by its whole state.

Ace Basin LF 1 (2026-02-21) ran through Beaufort and Colleton counties, SC. No
single county held it, so the label fell back to "South Carolina". A pinned
proof over the union of the two county polygons supports "A & B Counties, ST".
Synthetic county names; the state is real because proofs pin Census FIPS.
"""

from __future__ import annotations

from shapely.geometry import Polygon

from gomoveshift.pipeline.boundary_proof import (
    _SOURCES,
    CensusBoundaryProver,
    pinned_boundary_proof,
)
from gomoveshift.pipeline.final_text_record import route_record
from gomoveshift.pipeline.route_claims import (
    RouteEvidence,
    RouteWindow,
    _Bin,
    _region_candidates,
    _region_ok,
    enforce_route_claims,
    label_issues,
    measure_route,
    place_text_issues,
    retained_windows_key,
    select_supported_label,
)
from gomoveshift.pipeline.title_decision import _retained_title_issues


class Geo:
    def reverse_geocode_detailed(self, lat, lon):
        if lat < 31.1:
            return {"town": "Ash", "county": "Ash County", "state": "South Carolina"}
        return {"town": "Birch", "county": "Pine County", "state": "South Carolina"}


class Proof:
    """Verifies every claim it is asked about; records what it was asked."""

    def __init__(self):
        self.asked = []

    def prove(self, windows, claim):
        self.asked.append(claim["level"])
        source = "county" if claim["level"] == "counties" else claim["level"]
        return {
            "verified": True,
            "basis": "pinned_tiger_polygon_bounded_speed_and_error",
            "source": f"TIGER2026/{source.upper()}/{_SOURCES[source][0]}",
            "sha256": _SOURCES[source][1],
            "region": claim["name"],
            "statefp": "45",
            "max_gap_s": 30.0,
            "max_speed_m_s": 70.0,
            "gps_error_m": 100.0,
            "windows_key": retained_windows_key(windows),
        }


def _windows():
    # 120 s in Ash County, 60 s in Pine County, one 30 s GPS gap inside Ash.
    fixes = [(float(t), 31.0 if t < 120 else 31.2, -80.0) for t in range(180) if not 40 <= t < 70]
    return [RouteWindow("lf1", 0.0, 180.0, fixes)]


def _enforced(prover=None):
    text = {"2026-02-21": {1: "Ash, SC"}}
    measured = enforce_route_claims(
        text,
        {1: _windows()},
        Geo(),
        boundary_prover=prover or Proof(),
        require_review_support=True,
        review_support_available=True,
    )
    return text["2026-02-21"][1], measured[1]


def test_two_county_route_is_labelled_by_both_counties_not_the_state():
    label, ev = _enforced()
    assert label == "Ash & Pine Counties, SC"
    claim = ev.region_claim("counties")
    assert claim["counties"] == ["Ash County", "Pine County"]
    assert claim["state"] == "South Carolina"
    assert claim["boundary_proof"]["source"].startswith("TIGER2026/COUNTY/")
    assert ev.record()["boundary_proofs"]["counties"]["region"] == "Ash & Pine Counties"


def test_counties_label_without_a_proof_falls_back_to_the_state():
    class StateOnly(Proof):
        def prove(self, windows, claim):
            if claim["level"] != "state":
                return {"verified": False, "reason": "near the county line"}
            return super().prove(windows, claim)

    label, _ev = _enforced(StateOnly())
    assert label == "South Carolina"


def test_counties_claim_needs_exactly_two_counties_in_one_state():
    one = RouteEvidence(60.0, [], [], 0.0, bins=[_Bin(60.0, "Ash County", "South Carolina")])
    assert one.region_claim("counties") is None
    three = RouteEvidence(
        90.0,
        [],
        [],
        0.0,
        bins=[_Bin(30.0, f"{n} County", "South Carolina") for n in ("Ash", "Pine", "Oak")],
    )
    assert three.region_claim("counties") is None
    split = RouteEvidence(
        60.0,
        [],
        [],
        0.0,
        bins=[_Bin(30.0, "Ash County", "South Carolina"), _Bin(30.0, "Pine County", "Georgia")],
    )
    assert split.region_claim("counties") is None


def test_unresolved_bin_is_outside_a_counties_claim_without_a_proof():
    ev = RouteEvidence(
        90.0,
        [],
        [],
        0.0,
        bins=[
            _Bin(30.0, "Ash County", "South Carolina"),
            _Bin(30.0, "Pine County", "South Carolina"),
            _Bin(30.0, None, None),
        ],
    )
    claim = ev.region_claim("counties")
    assert claim["name"] == "Ash & Pine Counties"
    assert claim["samples_outside"] == 1
    assert _region_ok(claim, ev, 1.0)


def test_supported_counties_label_passes_every_text_gate():
    _label, ev = _enforced()
    assert label_issues("Ash & Pine Counties, SC", ev) == []
    assert place_text_issues("Ash & Pine Counties, SC", ev) == []
    # The town "Ash" shares the county's name; the region still holds the route.
    title = "Ash & Pine Counties, SC in 4K — Part 1 of 2"
    assert _retained_title_issues(title, ev) == []
    assert select_supported_label("Ash & Pine Counties, SC", ev) == ("Ash & Pine Counties, SC", "")


def test_counties_claim_reaches_the_review_record():
    _label, ev = _enforced()
    claims = {c["name"]: c for c in route_record(ev)["region_claims"]}
    assert claims["Ash & Pine Counties"]["level"] == "counties"
    assert claims["Ash & Pine Counties"]["boundary_proof"]["verified"] is True


def test_pinned_proof_of_a_counties_claim_uses_the_county_archive():
    _label, ev = _enforced()
    claim = ev.region_claim("counties")
    assert pinned_boundary_proof(claim["boundary_proof"], claim)
    forged = dict(claim["boundary_proof"], source="TIGER2026/STATE/tl_2026_us_state.zip")
    assert not pinned_boundary_proof(forged, claim)


def test_prover_proves_a_counties_claim_inside_the_union_of_its_polygons():
    # Ash spans lat 30-31.1, Pine 31.1-32; a fix near the shared line is deep
    # inside the union, and the route leaves through neither county's edge.
    shapes = {
        "Ash County": Polygon([(-81, 30), (-79, 30), (-79, 31.1), (-81, 31.1)]),
        "Pine County": Polygon([(-81, 31.1), (-79, 31.1), (-79, 32), (-81, 32)]),
    }

    class Prover(CensusBoundaryProver):
        def load_region(self, level, name, statefp):
            assert (level, statefp) == ("county", "45")
            return shapes[name]

    windows = [RouteWindow("lf1", 0.0, 4.0, [(0.0, 31.09, -80.0), (2.0, 31.11, -80.0)])]
    claim = {
        "level": "counties",
        "name": "Ash & Pine Counties",
        "counties": ["Ash County", "Pine County"],
        "state": "South Carolina",
    }
    proof = Prover().prove(windows, claim)
    assert proof["verified"] is True
    assert proof["source"] == f"TIGER2026/COUNTY/{_SOURCES['county'][0]}"
    assert proof["region"] == "Ash & Pine Counties"
    # Either county alone has that fix right on its edge.
    alone = dict(claim, level="county", name="Ash County")
    assert Prover().prove(windows, alone)["verified"] is False


def test_measured_two_county_route_offers_the_counties_label_after_the_county():
    ev = measure_route(_windows(), Geo())
    levels = [claim["level"] for claim, _label in _region_candidates(ev)]
    assert levels == ["county", "counties", "state"]
