"""``dominant_location`` must never merge two real counties that share a name.

Washington County exists in TN/VA/MD/PA, Franklin in PA/VA, Cumberland in
PA/NC/VA, Frederick in MD/VA. The collapse (an Irish county tagged once
with a province-as-state and once stateless is ONE county) may only fold a
STATELESS variant into a county's single non-empty state; it must never merge
two distinct non-empty states, and it must not guess a state for stateless
samples of a name that exists in two states.
"""

import logging

from gomoveshift.pipeline.title_builder import dominant_location

_SMYTH = (36.83, -81.50)
_WASH_VA = (36.75, -81.98)
_WASH_TN = (36.35, -82.50)
_WASH_NONE = (36.10, -83.20)
_KERRY_M = (52.14, -10.27)
_KERRY_S = (52.05, -9.50)
_CORK = (51.90, -8.47)

_DETAIL = {
    _SMYTH: {"county": "Smyth County", "state": "Virginia"},
    _WASH_VA: {"county": "Washington County", "state": "Virginia"},
    _WASH_TN: {"county": "Washington County", "state": "Tennessee"},
    _WASH_NONE: {"county": "Washington County", "state": None},
    _KERRY_M: {"county": "County Kerry", "state": "Munster"},
    _KERRY_S: {"county": "County Kerry", "state": None},
    _CORK: {"county": "County Cork", "state": "Munster"},
}


class _Geo:
    def reverse_geocode_detailed(self, lat, lon):
        for (clat, clon), data in _DETAIL.items():
            if abs(lat - clat) + abs(lon - clon) < 0.05:
                return dict(data)
        return {"county": None, "state": None}


def _pts(**counts):
    coords = {
        "smyth": _SMYTH,
        "wva": _WASH_VA,
        "wtn": _WASH_TN,
        "wnone": _WASH_NONE,
        "km": _KERRY_M,
        "ks": _KERRY_S,
        "cork": _CORK,
    }
    out = []
    for name, n in counts.items():
        out += [{"lat": coords[name][0], "lon": coords[name][1]}] * n
    return out


def test_worked_example_never_names_tennessee():
    r = dominant_location(_pts(smyth=3, wva=2, wtn=2), _Geo())
    assert r is not None
    assert (r.county, r.state) == ("Smyth County", "Virginia")


def test_same_name_counties_keep_own_weight():
    r = dominant_location(_pts(wva=4, wtn=3), _Geo())
    assert (r.county, r.state) == ("Washington County", "Virginia")


def test_munster_none_collapse_unchanged():
    r = dominant_location(_pts(km=2, ks=2, cork=3), _Geo())
    assert (r.county, r.state) == ("County Kerry", "Munster")


def test_stateless_sample_of_two_state_name_credited_to_neither(caplog):
    # 3 stateless Washington samples must not be handed to Tennessee (the
    # alphabetical modal); dropped, Smyth's 3 beats VA 2 / TN 2.
    with caplog.at_level(logging.DEBUG):
        r = dominant_location(_pts(smyth=3, wva=2, wtn=2, wnone=3), _Geo())
    assert (r.county, r.state) == ("Smyth County", "Virginia")
    assert any("stateless" in m.lower() for m in caplog.messages)


def test_exact_cross_state_tie_abstains():
    assert dominant_location(_pts(smyth=2, wtn=2), _Geo()) is None


def test_same_state_tie_keeps_alphabetical_winner():
    r = dominant_location(_pts(smyth=2, wva=2), _Geo())
    assert (r.county, r.state) == ("Smyth County", "Virginia")
