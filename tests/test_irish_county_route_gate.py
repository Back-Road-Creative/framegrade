"""Route-claim gate for a stateless two-county route (Ireland) and same-county
bridging of GPS gaps and county-less bins.

Synthetic geocoder bands only; the places are real Irish counties because the
label form ("County Cork & County Kerry, Ireland") is what is under test.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.route_claims import (
    RouteClaimRefusedError,
    RouteEvidence,
    RoutePlace,
    RouteWindow,
    _Bin,
    measure_route,
    region_label,
    select_supported_label,
)

CORK = {"town": "Macroom", "county": "County Cork", "country": "Ireland"}
KERRY = {"town": "Kenmare", "county": "County Kerry", "country": "Ireland"}


class _Geo:
    """lat 51.0 -> Cork, 52.0 -> Kerry, 51.5 -> nothing resolves."""

    def reverse_geocode_detailed(self, lat, lon):
        if lat < 51.25:
            return dict(CORK)
        return {} if lat < 51.75 else dict(KERRY)


def _window(end, lat_of, drop=()):
    fixes = [
        (float(t), lat_of(t), 9.0) for t in range(int(end)) if not any(a <= t < b for a, b in drop)
    ]
    return RouteWindow("lf01", 0.0, float(end), fixes)


def _cork_none_cork_kerry(t):
    return 51.0 if t < 300 or 350 <= t < 500 else 51.5 if t < 350 else 52.0


def test_kenmare_shaped_route_labels_itself_two_irish_counties():
    w = _window(600, _cork_none_cork_kerry, drop=[(100, 106)])
    ev = measure_route([w], _Geo(), bin_s=10.0)
    assert ev.gaps and ev.gaps[0].county == "County Cork"
    assert any(b.inferred and b.county == "County Cork" for b in ev.bins)
    label, note = select_supported_label("County Cork, Ireland", ev)
    assert label == "County Cork & County Kerry, Ireland"
    assert note


def test_region_label_appends_country_for_a_stateless_pair():
    ev = RouteEvidence(
        60.0,
        [],
        [],
        0.0,
        [_Bin(40.0, "County Cork", None, country="Ireland"),
         _Bin(20.0, "County Kerry", None, country="Ireland")],
    )  # fmt: skip
    claim = ev.region_claim("counties")
    assert claim["name"] == "County Cork & County Kerry"
    assert claim["country"] == "Ireland"
    assert region_label(claim) == "County Cork & County Kerry, Ireland"


def test_no_county_bin_between_different_counties_is_not_bridged():
    def lat(t):
        return 51.0 if t < 300 else 51.5 if t < 350 else 52.0

    ev = measure_route([_window(600, lat)], _Geo(), bin_s=10.0)
    assert not any(b.inferred for b in ev.bins)
    with pytest.raises(RouteClaimRefusedError):
        select_supported_label("County Cork, Ireland", ev)


def test_gap_beside_an_unresolved_bin_is_not_bridged():
    def lat(t):
        return 51.0 if t < 300 or 400 <= t < 500 else 51.5 if t < 400 else 52.0

    ev = measure_route([_window(600, lat, drop=[(340, 347)])], _Geo(), bin_s=10.0)
    assert ev.gaps and ev.gaps[0].county is None
    with pytest.raises(RouteClaimRefusedError):
        select_supported_label("County Cork, Ireland", ev)


def test_record_round_trips_country_inferred_and_gap_county():
    ev = measure_route([_window(600, _cork_none_cork_kerry, drop=[(100, 106)])], _Geo(), bin_s=10.0)
    back = RouteEvidence.from_record(ev.record())
    assert {b.country for b in back.bins if b.county} == {"Ireland"}
    assert any(b.inferred for b in back.bins)
    assert [g.county for g in back.gaps] == ["County Cork"]
    assert back.region_claim("counties")["name"] == "County Cork & County Kerry"


def test_required_place_inside_a_counties_pair_does_not_raise():
    place = RoutePlace("Macroom", "County Cork", None, 90.0, True, "ev1")
    ev = RouteEvidence(
        90.0,
        [],
        [place],
        0.0,
        [_Bin(60.0, "County Cork", None, "Macroom", country="Ireland"),
         _Bin(30.0, "County Kerry", None, country="Ireland")],
    )  # fmt: skip
    assert ev.region_claim("counties")["required_place_ids"] == ["ev1"]


@pytest.mark.parametrize(
    "states",
    [(None, "Westmark"), ("Westmark", "Eastmark")],
)
def test_mixed_states_give_no_counties_claim(states):
    ev = RouteEvidence(
        60.0,
        [],
        [],
        0.0,
        [_Bin(30.0, "Ashby County", states[0], country="X"),
         _Bin(30.0, "Brell County", states[1], country="X")],
    )  # fmt: skip
    assert ev.region_claim("counties") is None
