"""A route title's state suffix must agree with every observed retained region."""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.route_claims import (
    RouteClaimRefusedError,
    RouteWindow,
    label_issues,
    measure_route,
    select_supported_label,
)


class _Geo:
    def reverse_geocode_detailed(self, lat, lon):
        if lat < 10:
            return {"town": "Alder", "county": "Ash County", "state": "Virginia"}
        return {"town": "Birch", "county": "Elm County", "state": "West Virginia"}


def _evidence(*, cross_state):
    fixes = [(float(t), 9.0 if t < 120 or not cross_state else 11.0, -81.0) for t in range(240)]
    return measure_route([RouteWindow("lf1", 0, 240, fixes)], _Geo())


def test_state_suffix_cannot_hide_an_observed_other_state():
    ev = _evidence(cross_state=True)
    issues = label_issues("Alder and Birch, VA", ev)
    assert any("Virginia" in issue and "outside" in issue for issue in issues)
    with pytest.raises(RouteClaimRefusedError, match="outside claimed state"):
        select_supported_label("Alder and Birch, VA", ev)


def test_matching_state_suffix_stays_eligible():
    ev = _evidence(cross_state=False)
    assert label_issues("Alder, VA", ev) == []


def test_no_state_suffix_does_not_claim_one():
    ev = _evidence(cross_state=True)
    assert label_issues("Alder and Birch", ev) == []
