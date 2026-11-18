"""Road-only labels broaden to the measured towns before the county (level ``places``)."""

from __future__ import annotations

from gomoveshift.pipeline.route_claims import (
    RouteWindow,
    measure_route,
    route_claim_outcome,
    select_supported_label,
)

_COURT = {"town": "Courtmacsherry", "county": "County Cork", "country": "Ireland"}
_TIMO = {"town": "Timoleague", "county": "County Cork", "country": "Ireland"}
_BARRY = {"town": "Barryroe", "county": "County Cork", "country": "Ireland"}


class _Geo:
    """lat bands: 51.0 Courtmacsherry, 51.1 Timoleague, 51.2 Barryroe."""

    def __init__(self, bands):
        self.bands = bands

    def reverse_geocode_detailed(self, lat, lon):
        for top, geo in self.bands:
            if lat < top:
                return dict(geo)
        return dict(self.bands[-1][1])


def _ev(plan, bands):
    """plan: [(seconds, lat)] consecutive; one fix per second."""
    fixes, t = [], 0
    for secs, lat in plan:
        fixes += [(float(t + i), lat, -8.0) for i in range(int(secs))]
        t += int(secs)
    w = RouteWindow("lf01", 0.0, float(t), fixes)
    return measure_route([w], _Geo(bands), bin_s=10.0)


_BANDS = [(51.05, _COURT), (51.15, _TIMO), (99.0, _BARRY)]


def test_road_only_label_names_the_two_longest_measured_towns():
    ev = _ev([(300, 51.0), (200, 51.1), (30, 51.2)], _BANDS)
    label, note = select_supported_label("L6099, County Cork", ev)
    assert label == "Courtmacsherry & Timoleague, County Cork"
    assert "places" in note


def test_a_place_below_min_place_s_is_never_named():
    ev = _ev([(40, 51.0), (30, 51.1), (20, 51.2)], _BANDS)
    label, _ = select_supported_label("L6099, County Cork", ev)
    assert label == "County Cork, Ireland"


def test_one_qualifying_place_is_named_alone():
    ev = _ev([(300, 51.0), (30, 51.2)], _BANDS)
    label, _ = select_supported_label("R600, County Cork", ev)
    assert label == "Courtmacsherry, County Cork"


def test_three_required_places_fall_through_to_the_county():
    ev = _ev([(300, 51.0), (200, 51.1), (100, 51.2)], _BANDS)
    label, _ = select_supported_label("R600, County Cork", ev)
    assert label == "County Cork, Ireland"


def test_outcome_records_level_places():
    ev = _ev([(300, 51.0), (200, 51.1)], _BANDS)
    chosen, note = select_supported_label("L4021, County Cork", ev)
    out = route_claim_outcome(1, "L4021, County Cork", ev, chosen=chosen, reason=note)
    assert out["level"] == "places"


def test_outcome_records_level_county_on_county_fallback():
    ev = _ev([(40, 51.0), (30, 51.1)], _BANDS)
    chosen, note = select_supported_label("L4021, County Cork", ev)
    out = route_claim_outcome(1, "L4021, County Cork", ev, chosen=chosen, reason=note)
    assert out["level"] == "county"


def test_a_supported_label_is_kept_with_no_level():
    ev = _ev([(300, 51.0)], _BANDS)
    chosen, note = select_supported_label("Courtmacsherry, County Cork", ev)
    out = route_claim_outcome(1, "Courtmacsherry, County Cork", ev, chosen=chosen, reason=note)
    assert chosen == "Courtmacsherry, County Cork"
    assert out["level"] is None
