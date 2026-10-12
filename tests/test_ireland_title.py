"""Tests for the Ireland/non-US GPS-derived title format.

A non-US drive's ``per_lf_location`` carries a trailing region that is NOT
a US state (e.g. ``"..., Kerry"``), so ``normalize_state_abbr`` returns "".
Previously every "A to B" cascade branch required a truthy ``state_abbr``,
so the GPS-derived path fell through to Level 5 (raw ``per_lf_location``).
The fix uses the trailing region verbatim when it isn't a US state, so the
GPS-derived title reads e.g. ``"Killarney to Dingle via N86, Kerry"``.

US behavior (state_abbr present/derivable) is unchanged.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from gomoveshift.pipeline.title_builder import build_lf_title


def _seg(lat: float, lon: float):
    return SimpleNamespace(gps_center_lat=lat, gps_center_lon=lon)


def _loc(**kwargs):
    defaults = dict(city=None, locality=None, suburb=None, name=None, town=None)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestIrelandNonUSTitle:
    def test_via_road_uses_trailing_region_verbatim(self):
        """Level 1: Ireland 'A to B via road' keeps the county region."""
        # Endpoints kept within the via-road span guard (<50 km).
        segs = [_seg(52.06, -9.51), _seg(52.14, -9.70)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Killarney"),
            _loc(city="Dingle"),
        ]

        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Killarney, Kerry (N86)",
        )
        assert title == "Killarney to Dingle via N86, Kerry"

    def test_no_road_uses_trailing_region_verbatim(self):
        """Level 2: 'A to B' (no road) keeps the non-US region."""
        segs = [_seg(52.06, -9.51), _seg(52.14, -10.27)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Killarney"),
            _loc(city="Dingle"),
        ]

        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Killarney, Kerry",
        )
        assert title == "Killarney to Dingle, Kerry"

    def test_county_fallback_uses_trailing_region(self):
        """Level 4: single-town/county fallback keeps the non-US region."""
        segs = [_seg(52.06, -9.51), _seg(52.06, -9.51)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(),  # no town for start
            _loc(),  # no town for end
        ]

        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Kerry, Ireland",
        )
        assert title == "Kerry, Ireland"


class TestUSBehaviorUnchanged:
    def test_us_via_road_still_abbreviates_state(self):
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]

        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina (Sea Island Parkway)",
        )
        assert title == "Port Royal to Lady's Island via Sea Island Parkway, SC"

    def test_us_no_road(self):
        segs = [_seg(32.3, -80.6), _seg(32.4, -80.7)]
        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _loc(city="Port Royal"),
            _loc(city="Lady's Island"),
        ]

        title = build_lf_title(
            segs,
            geocoder,
            per_lf_location="Beaufort County, South Carolina",
        )
        assert title == "Port Royal to Lady's Island, SC"
