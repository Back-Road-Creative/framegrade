"""A label names every county with material footage.

The cross-county flake guard drops an endpoint the on-route midpoint does not
corroborate. A real drive that spent 56 % in county A and 44 % in county B has
its midpoint in A, so the guard used to erase the genuine B endpoint. An
endpoint whose county holds >= ``LOCATION_MATERIAL_SHARE`` of the footage vote
is a real location, never a flake.
"""

from types import SimpleNamespace

from gomoveshift.pipeline.title_builder import (
    _validate_route_endpoints,
    build_lf_title,
)
from gomoveshift.pipeline.validation.content_gates import LOCATION_MATERIAL_SHARE

_N = 100
_LAT0 = 37.0
_STEP = 0.002
_LON = -80.4
_LOC = "Montgomery County, Virginia"


def _trace():
    return [{"lat": _LAT0 + _STEP * i, "lon": _LON} for i in range(_N)]


def _seg(lat, lon):
    return SimpleNamespace(gps_center_lat=lat, gps_center_lon=lon)


def _segs():
    return [_seg(_LAT0, _LON), _seg(_LAT0 + _STEP * (_N - 1), _LON)]


class _Geo:
    """Latitude-banded fake. ``band(lat)`` -> (county, town)."""

    def __init__(self, band, towns=True):
        self._band = band
        self._towns = towns

    def reverse_geocode_detailed(self, lat, lon):
        county, town = self._band(lat)
        d = {"county": county, "state": "Virginia"}
        if self._towns:
            d["town"] = town
        return d

    def reverse_geocode(self, lat, lon):
        _, town = self._band(lat)
        return SimpleNamespace(
            city=town if self._towns else None,
            locality=None,
            suburb=None,
            name=None,
            town=None,
        )


def _radford_band(lat):
    if lat < 37.115:  # 58 % of the trace
        return ("Montgomery County", "Christiansburg")
    return ("Roanoke County", "Salem")


def test_material_share_constant_is_a_policy_floor():
    assert LOCATION_MATERIAL_SHARE == 0.20


def test_radford_keeps_both_endpoint_towns():
    title = build_lf_title(_segs(), _Geo(_radford_band), per_lf_location=_LOC, gps_points=_trace())
    assert "Christiansburg" in title and "Salem" in title
    assert "to" in title
    assert not title.startswith("Montgomery County")


def test_true_flake_with_no_footage_is_still_dropped():
    def band(lat):
        if lat >= 37.19:  # last 5 of 100 points: ~6 % weight, below the floor
            return ("Floyd County", "Floyd")
        return ("Montgomery County", "Christiansburg")

    title = build_lf_title(_segs(), _Geo(band), per_lf_location=_LOC, gps_points=_trace())
    assert "Floyd" not in title
    assert "Christiansburg" in title


def test_both_material_end_town_missing_names_both_counties():
    title = build_lf_title(
        _segs(),
        _Geo(_radford_band, towns=False),
        per_lf_location=_LOC,
        gps_points=_trace(),
    )
    assert title == "Montgomery & Roanoke County, VA"


def test_unanimous_vote_unchanged():
    def band(lat):
        return ("Montgomery County", "Christiansburg" if lat < 37.1 else "Blacksburg")

    title = build_lf_title(_segs(), _Geo(band), per_lf_location=_LOC, gps_points=_trace())
    assert title.startswith("Christiansburg to Blacksburg")
    assert "&" not in title


def test_validate_endpoints_shares_protect_material_endpoint():
    geo = _Geo(_radford_band)
    first, last = (_LAT0, _LON), (_LAT0 + _STEP * (_N - 1), _LON)
    mid = [(37.05, _LON)]
    assert _validate_route_endpoints(geo, first, last, mid) == (True, False)
    shares = {("Montgomery County", "Virginia"): 0.58, ("Roanoke County", "Virginia"): 0.42}
    assert _validate_route_endpoints(geo, first, last, mid, county_shares=shares) == (True, True)
    shares_flake = {("Montgomery County", "Virginia"): 0.97, ("Roanoke County", "Virginia"): 0.03}
    assert _validate_route_endpoints(geo, first, last, mid, county_shares=shares_flake) == (
        True,
        False,
    )
