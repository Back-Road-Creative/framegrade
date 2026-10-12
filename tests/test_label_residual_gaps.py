"""Residual label gaps that only showed on real data.

Two bugs survived the first hardening pass:

1. A COUNTY name leaking in as an endpoint "town". ``_geocode_town`` returns
   the geocoder's ``.name``/``.locality``, which at a rural centroid with no
   town is the county itself ("Beaufort County"). That non-None value blocked
   the settlement backfill and won the Level-4 anchor — so "Fort Fremont"
   lost to "Beaufort County". Fixed by nulling a county-shaped endpoint town.

2. The ``per_lf_location`` ``(road)`` suffix bypassing the via-road filter.
   ``derive_per_lf_locations`` writes the suffix from an unfiltered centroid
   geocode, so junk roads ("Blocker Run Road", "Savage") reached Level 4. Fixed
   by gating the suffix road through ``_road_passes_quality``.
"""

from types import SimpleNamespace

from gomoveshift.pipeline.title_builder import (
    _geocode_town,
    _looks_like_county,
    build_lf_title,
)

_GPS = [{"lat": 35.10, "lon": -83.10}, {"lat": 35.18, "lon": -83.38}]


class _MissGeo:
    def reverse_geocode(self, lat, lon):
        return SimpleNamespace(
            city=None,
            locality=None,
            suburb=None,
            name=None,
            town=None,
            place=None,
            hamlet=None,
            village=None,
        )

    def reverse_geocode_detailed(self, lat, lon):
        return {}


class _CountyEndpointGeo:
    """``_geocode_town`` resolves to the COUNTY (the rural-centroid case)."""

    def reverse_geocode(self, lat, lon):
        return SimpleNamespace(
            city=None,
            locality="Beaufort County",
            suburb=None,
            name=None,
            town=None,
            place=None,
            hamlet=None,
            village=None,
        )

    def reverse_geocode_detailed(self, lat, lon):
        return {}


class TestLooksLikeCounty:
    def test_county_shaped_strings(self):
        assert _looks_like_county("Beaufort County") is True
        assert _looks_like_county("colleton county") is True
        assert _looks_like_county("County Kerry") is True  # Irish shape

    def test_real_settlement_not_county(self):
        assert _looks_like_county("Fort Fremont") is False
        assert _looks_like_county("Cashiers") is False
        assert _looks_like_county("Mars Hill") is False  # a town used as a location base

    def test_none_and_empty(self):
        assert _looks_like_county(None) is False
        assert _looks_like_county("") is False


class _RuralVillageGeo:
    """A rural endpoint with only a village — no city/town. The county-joined
    `name` is what format_location_string returns for such a point."""

    def reverse_geocode(self, lat, lon):
        return SimpleNamespace(
            city=None,
            locality=None,
            suburb=None,
            name="Rossbeg, County Donegal",
            town=None,
            place=None,
            hamlet=None,
            village="Rossbeg",
        )

    def reverse_geocode_detailed(self, lat, lon):
        return {}


class TestGeocodeTownPrefersSettlement:
    def test_village_beats_county_joined_name(self):
        # Rural endpoint: village="Rossbeg", name="Rossbeg, County Donegal".
        # _geocode_town must return the clean settlement, not the county-joined
        # name (which _looks_like_county would later demote).
        assert _geocode_town(_RuralVillageGeo(), 55.0, -8.3) == "Rossbeg"


class TestCountyEndpointFreesSettlementAnchor:
    def test_county_endpoint_nulled_settlement_wins(self):
        # Endpoints geocode to "Beaufort County"; the sequence has Fort Fremont.
        # The county must NOT anchor the title — the settlement does.
        title = build_lf_title(
            [],
            _CountyEndpointGeo(),
            "Beaufort County, South Carolina",
            gps_points=_GPS,
            resolved_places=["Fort Fremont"],
        )
        assert title == "Fort Fremont, SC"

    def test_county_endpoints_with_two_settlements_make_crossing(self):
        title = build_lf_title(
            [],
            _CountyEndpointGeo(),
            "Transylvania County, North Carolina",
            gps_points=_GPS,
            resolved_places=["Quebec", "Toxaway Falls"],
        )
        assert title == "Quebec to Toxaway Falls, NC"


class TestSuffixRoadQualityFilter:
    def test_junk_suffix_road_dropped(self):
        title = build_lf_title(
            [],
            _MissGeo(),
            "Colleton County, South Carolina (Blocker Run Road)",
            gps_points=_GPS,
        )
        assert "Blocker Run Road" not in title
        assert title == "Colleton County, SC"

    def test_business_suffix_road_dropped(self):
        title = build_lf_title(
            [], _MissGeo(), "Colleton County, South Carolina (Savage)", gps_points=_GPS
        )
        assert "Savage" not in title
        assert title == "Colleton County, SC"

    def test_route_ref_suffix_road_kept(self):
        title = build_lf_title(
            [], _MissGeo(), "Macon County, North Carolina (US 64)", gps_points=_GPS
        )
        assert "US 64" in title
