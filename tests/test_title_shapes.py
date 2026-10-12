"""Title shape-ladder collapse and guards.

- Same-endpoint "X to X" is never emitted (incl. the cross-state shape).
- A road below the distance plurality is not declared dominant.
- ost_formatter._format_ost_from_route and RouteProfile.title route
  through one shared ladder (format_location_ladder) so they can't drift.
"""

from gomoveshift.pipeline.ingest.ost_formatter import (
    _format_ost_from_route,
    format_location_ladder,
)
from gomoveshift.pipeline.route_profile import RouteProfile, build_route_profile


class _Wp:
    def __init__(self, lat, lon):
        self.latitude, self.longitude = lat, lon


class _Geo:
    def __init__(self, by_lat):
        self._by_lat = by_lat

    def reverse_geocode_detailed(self, lat, lon):
        return self._by_lat.get(round(lat, 3))


# One shared ladder ----------------------------------------------------


def test_ladder_shapes():
    assert (
        format_location_ladder(
            start="Beaufort",
            end="Cashiers",
            start_state="South Carolina",
            end_state="North Carolina",
            is_cross_state=True,
        )
        == "Beaufort, SC to Cashiers, NC"
    )
    assert (
        format_location_ladder(start="Asheville", end="Boone", road="US-221")
        == "Asheville to Boone via US-221"
    )
    assert (
        format_location_ladder(start="Cashiers", road="NC-107", start_state="North Carolina")
        == "NC-107, Cashiers, NC"
    )
    assert format_location_ladder() == ""


def test_route_profile_delegates_to_ladder():
    """RouteProfile.title equals the shared ladder for the same components."""
    p = RouteProfile(
        start_town="Asheville",
        end_town="Boone",
        dominant_road="US-221",
        states_traversed=["North Carolina"],
    )
    assert p.title() == format_location_ladder(
        start="Asheville",
        end="Boone",
        road="US-221",
        start_state="North Carolina",
        end_state="North Carolina",
    )


# Same-endpoint guard ---------------------------------------------------


def test_cross_state_same_label_collapses():
    out = format_location_ladder(
        start="Clayton",
        end="Clayton",
        start_state="Georgia",
        end_state="North Carolina",
        is_cross_state=True,
    )
    assert " to " not in out and out.startswith("Clayton")


def test_format_ost_cross_state_same_town_no_x_to_x():
    geo = _Geo(
        {
            1.0: {"town": "Clayton", "state": "Georgia", "county": "Rabun County"},
            2.0: {"town": "Clayton", "state": "North Carolina", "county": "Macon County"},
        }
    )
    wps = [_Wp(1.0, 1.0)] * 5 + [_Wp(2.0, 2.0)] * 5
    assert " to " not in _format_ost_from_route([], [], geocoder=geo, waypoints=wps)


# Sub-plurality road is not dominant -----------------------------------


class _Journey:
    waypoints = [_Wp(35.0, -82.0), _Wp(36.0, -81.5)]


class _Parser:
    def parse(self, p):
        return _Journey()


def _build(monkeypatch, tmp_path, segs):
    import gps_journey.core.gpx_parser as gpx_mod

    monkeypatch.setattr(gpx_mod, "GPXParser", _Parser, raising=False)
    gpx = tmp_path / "ep.gpx"
    gpx.write_text("<gpx/>")

    class _RA:
        def get_road_segments(self, wps):
            return segs

        def get_towns_passed_through(self, wps):
            return []

    return build_route_profile(0, [gpx], [0], _Geo({}), _RA())


def test_sub_plurality_road_not_dominant(monkeypatch, tmp_path):
    # Max share 30/79 ≈ 38% — no plurality, so no dominant road.
    segs = [
        {"road": r, "distance_km": d} for r, d in [("US-221", 30), ("NC-105", 25), ("NC-194", 24)]
    ]
    assert _build(monkeypatch, tmp_path, segs).dominant_road is None


def test_majority_road_is_dominant(monkeypatch, tmp_path):
    segs = [{"road": "US-221", "distance_km": 60}, {"road": "NC-105", "distance_km": 20}]
    assert _build(monkeypatch, tmp_path, segs).dominant_road == "US-221"
