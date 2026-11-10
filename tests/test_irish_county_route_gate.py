"""Route-claim gate for a stateless two-county route (Ireland) and same-county
bridging of GPS gaps and county-less bins.

Synthetic geocoder bands only; the places are real Irish counties because the
label form ("County Cork & County Kerry, Ireland") is what is under test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.route_claims import (
    RouteClaimRefusedError,
    RouteEvidence,
    RoutePlace,
    RouteWindow,
    GAP_S,
    _Bin,
    make_short_route_fn,
    measure_route,
    region_label,
    select_supported_label,
    short_route_window,
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


def _cork(t):
    return 51.0


def test_tail_gap_bridges_from_the_next_source_fix_in_the_same_county():
    # The cut ends 10 s after the last retained fix; the next fix of the same
    # file sits just past the cut and is also in County Cork.
    w = _window(600, _cork, drop=[(590, 600)])
    w.next_fix_s, w.next_fix = 605.0, (605.0, 51.0, 9.0)
    ev = measure_route([w], _Geo(), bin_s=10.0)
    assert [g.county for g in ev.gaps] == ["County Cork"]
    assert ev.coverage_excluding({"County Cork"}) == 1.0
    assert any("tail" in n and "County Cork" in n for n in ev.boundary_notes)


def test_head_gap_bridges_from_the_previous_source_fix_in_the_same_county():
    w = _window(600, _cork, drop=[(0, 10)])
    w.prev_fix_s, w.prev_fix = -4.0, (-4.0, 51.0, 9.0)
    ev = measure_route([w], _Geo(), bin_s=10.0)
    assert [g.county for g in ev.gaps] == ["County Cork"]


@pytest.mark.parametrize("outside", [None, (605.0, 52.0, 9.0), (605.0, 51.5, 9.0)])
def test_tail_gap_stays_a_gap_without_a_same_county_fix_past_the_cut(outside):
    w = _window(600, _cork, drop=[(590, 600)])
    if outside:
        w.next_fix_s, w.next_fix = outside[0], outside
    ev = measure_route([w], _Geo(), bin_s=10.0)
    assert [g.county for g in ev.gaps] == [None]
    with pytest.raises(RouteClaimRefusedError):
        select_supported_label("County Cork, Ireland", ev)


def test_region_label_appends_country_for_a_stateless_single_county():
    ev = RouteEvidence(60.0, [], [], 0.0, [_Bin(60.0, "County Cork", None, country="Ireland")])
    claim = ev.region_claim("county")
    assert claim["state"] is None and claim["country"] == "Ireland"
    assert region_label(claim) == "County Cork, Ireland"


def _short_day(lat_after=51.0):
    # A Short [0, 38) whose last fix inside is at 33 s and whose next fix is at
    # 38 s, just past the cut: 33 -> 38 is GAP_S, so no gap, but the last bin
    # [35, 38) holds no fix of its own (Clonakilty 09-17 rank 2).
    return [(float(t), 51.0, 9.0) for t in range(34)] + [
        (float(t), lat_after, 9.0) for t in range(38, 60)
    ]


def test_a_short_window_carries_the_source_fixes_outside_its_cut():
    w = short_route_window(2, 0.0, 38.0, [(-3.0, 51.0, 9.0), *_short_day()])
    assert w.prev_fix == (-3.0, 51.0, 9.0) and w.next_fix == (38.0, 51.0, 9.0)
    assert (w.prev_fix_s, w.next_fix_s) == (-3.0, 38.0)


def test_a_fixless_tail_bin_takes_the_county_of_the_next_source_fix():
    w = short_route_window(2, 0.0, 38.0, _short_day())
    ev = measure_route([w], _Geo(), bin_s=GAP_S, min_place_s=0.0)
    assert not ev.gaps
    assert ev.bins[-1].county == "County Cork" and ev.bins[-1].inferred
    assert ev.region_claim("county")["samples_outside"] == 0


def test_a_countyless_head_bin_takes_the_county_of_the_previous_source_fix():
    # A head bin cannot be fixless without a gap (bins start at the cut), but its
    # lookup can resolve nothing; the fix before the cut stands in.
    pts = [(-1.0, 51.0, 9.0)] + [(float(t), 51.5 if t < 5 else 51.0, 9.0) for t in range(38)]
    w = short_route_window(2, 0.0, 38.0, pts)
    ev = measure_route([w], _Geo(), bin_s=GAP_S, min_place_s=0.0)
    assert ev.bins[0].county == "County Cork" and ev.bins[0].inferred


def test_a_short_ending_on_a_fixless_bin_keeps_its_county_label():
    label, _note = make_short_route_fn(_Geo())(
        _short_day(), SimpleNamespace(rank=2, start_sec=0.0, end_sec=38.0), "County Cork"
    )
    assert label == "County Cork"


@pytest.mark.parametrize("lat_after", [52.0, 51.5])
def test_a_fixless_tail_bin_stays_unresolved_without_a_same_county_fix_past_the_cut(lat_after):
    # The fix past the cut is in another county, or resolves to none: nothing
    # shows where the last seconds were, so the county label is still refused.
    with pytest.raises(RouteClaimRefusedError, match="outside or unresolved"):
        make_short_route_fn(_Geo())(
            _short_day(lat_after),
            SimpleNamespace(rank=2, start_sec=0.0, end_sec=38.0),
            "County Cork",
        )
