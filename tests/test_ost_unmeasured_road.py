"""The burned OST names a road only when that
road's retained-time coverage was measured. Waypost's final-text review accepts
a cited road without checking coverage, so an unmeasured "via {road}" clause is
an unproven claim; the cascade drops it and keeps the place/region text. The
LF is not refused. ``road_coverage_measured=True`` is the path a measured
per-road retained time will use.

Also: a GPS-confirmed loop never carries the "Scenic" genre word in any shape,
since Waypost's name-backing rule has no evidence for it.

Synthetic places only ("Alder", "Dogwood" are invented).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from gomoveshift.pipeline.title_builder import build_lf_title


def _seg(lat, lon):
    return SimpleNamespace(gps_center_lat=lat, gps_center_lon=lon)


def _loc(city):
    return SimpleNamespace(city=city, locality=None, suburb=None, name=None, town=None)


def _geocoder(*cities):
    gc = MagicMock()
    gc.reverse_geocode.side_effect = [_loc(c) for c in cities]
    return gc


_A_TO_B = [_seg(32.30, -80.60), _seg(32.40, -80.70)]
_LOOP = [_seg(32.7, -80.6), _seg(32.7, -80.6)]
# Roads that pass select_significant_roads (>= 30 s and >= 1 km each).
_MULTI_ROADS = [
    {"name": "US 17", "duration_sec": 600.0, "length_m": 9000.0},
    {"name": "SC 21", "duration_sec": 420.0, "length_m": 6000.0},
]


class TestUnmeasuredRoadIsNotBurned:
    def test_a_to_b_drops_the_suffix_road(self):
        title = build_lf_title(
            _A_TO_B,
            _geocoder("Alder", "Dogwood"),
            per_lf_location="Ash County, South Carolina (Sea Fern Parkway)",
        )
        assert title == "Alder to Dogwood, SC"

    def test_a_to_b_drops_the_resolved_road(self):
        title = build_lf_title(
            _A_TO_B,
            _geocoder("Alder", "Dogwood"),
            per_lf_location="Ash County, South Carolina",
            resolved_road="Sea Fern Parkway",
        )
        assert " via " not in title

    def test_a_to_b_multi_road_control_burns_via_when_measured(self):
        # Control: these roads clear select_significant_roads, so the measured
        # path renders the multi-road "via" title. Without it, the default test
        # below would pass even with the road-coverage gate deleted.
        title = build_lf_title(
            _A_TO_B,
            _geocoder("Alder", "Dogwood"),
            per_lf_location="Ash County, South Carolina",
            roads=_MULTI_ROADS,
            road_coverage_measured=True,
        )
        assert title == "Alder to Dogwood via US 17 + SC 21, SC"

    def test_a_to_b_drops_multiple_roads(self):
        title = build_lf_title(
            _A_TO_B,
            _geocoder("Alder", "Dogwood"),
            per_lf_location="Ash County, South Carolina",
            roads=_MULTI_ROADS,
        )
        assert " via " not in title
        assert title == "Alder to Dogwood, SC"

    def test_loop_drops_the_road(self):
        title = build_lf_title(
            _LOOP,
            _geocoder("Alder", "Alder"),
            per_lf_location="Ash County, South Carolina (Sea Fern Parkway)",
        )
        assert title == "Alder Loop, SC"

    def test_county_fallback_drops_the_road(self):
        gc = MagicMock()
        gc.reverse_geocode.return_value = None
        title = build_lf_title(
            [_seg(32.7, -80.6)],
            gc,
            per_lf_location="Ash County, South Carolina (Sea Fern Parkway)",
        )
        assert " via " not in title
        assert "Ash County" in title


class TestMeasuredRoadKeepsTheClause:
    def test_measured_coverage_keeps_via(self):
        title = build_lf_title(
            _A_TO_B,
            _geocoder("Alder", "Dogwood"),
            per_lf_location="Ash County, South Carolina (Sea Fern Parkway)",
            road_coverage_measured=True,
        )
        assert title == "Alder to Dogwood via Sea Fern Parkway, SC"


class TestScenicNeverBurned:
    @staticmethod
    def _circular():
        return [
            SimpleNamespace(lat=35.870, lon=-82.570),
            SimpleNamespace(lat=35.872, lon=-82.572),
            SimpleNamespace(lat=35.871, lon=-82.571),
        ]

    def test_circular_county_fallback_drops_scenic(self):
        gc = MagicMock()
        gc.reverse_geocode.return_value = None
        title = build_lf_title(
            [_seg(35.87, -82.57)],
            gc,
            per_lf_location="Ash County Scenic Loop, NC",
            gps_points=self._circular(),
        )
        assert "scenic" not in title.casefold()
        assert "Ash County" in title
