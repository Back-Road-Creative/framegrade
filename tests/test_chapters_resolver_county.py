"""Chapters consume the ResolvedLocation resolver.

``pois_to_locations`` used to stamp the FIRST geocoded waypoint's county on
EVERY chapter. With a resolver each chapter's county now comes from its OWN
nearest geocoded waypoint, reconciled against the resolved sampled regions
(kept iff the resolver voted that county; else snapped to the voted dominant
county the title/OST also use). The legacy no-resolver path is unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

from gomoveshift.pipeline.chapters.overpass_source import OverpassChapterSource


def _wp(lat, lon, county):
    return SimpleNamespace(
        latitude=lat,
        longitude=lon,
        location=SimpleNamespace(county=county, country="United States") if county else None,
    )


def _poi(t, name, lat, lon, ptype="town"):
    return (t, {"name": name, "type": ptype, "lat": lat, "lon": lon})


def _resolved(regions, dominant):
    return SimpleNamespace(regions=regions, dominant_county=dominant)


class TestLegacyNoResolver:
    def test_first_waypoint_county_stamped_on_all(self):
        # resolved_location=None → preserve the pre-resolver behaviour exactly.
        wps = [_wp(34.0, -80.0, "Clare County"), _wp(35.0, -81.0, "Other County")]
        pois = [_poi(0.0, "A", 34.0, -80.0), _poi(10.0, "B", 35.0, -81.0)]
        out = OverpassChapterSource.pois_to_locations(pois, wps, min_chapters=1)
        assert {loc.county for _, loc in out} == {"Clare County"}


class TestPerPoiReconciliation:
    def test_each_poi_gets_its_own_in_region_county(self):
        regions = ["Beaufort County, South Carolina", "Jasper County, South Carolina"]
        resolved = _resolved(regions, "Beaufort County")
        wps = [_wp(32.4, -80.6, "Beaufort County"), _wp(32.2, -81.0, "Jasper County")]
        pois = [_poi(0.0, "A", 32.4, -80.6), _poi(10.0, "B", 32.2, -81.0)]
        out = dict((loc.name, loc.county) for _, loc in pois_result(pois, wps, resolved))
        assert out["A"] == "Beaufort County"
        assert out["B"] == "Jasper County"

    def test_county_outside_regions_snaps_to_dominant(self):
        # POI's nearest waypoint says Colleton, which the resolver never
        # sampled → snap to the voted dominant (Beaufort).
        regions = ["Beaufort County, South Carolina", "Jasper County, South Carolina"]
        resolved = _resolved(regions, "Beaufort County")
        wps = [_wp(33.0, -80.0, "Colleton County")]
        pois = [_poi(0.0, "A", 33.0, -80.0)]
        out = pois_result(pois, wps, resolved)
        assert out[0][1].county == "Beaufort County"

    def test_no_geocoded_waypoint_omits_county(self):
        # GoPro GPX waypoints carry no county → own=None. The POI may
        # physically sit in a county OTHER than the dominant one, so snapping
        # it to the dominant would stamp a wrong-county chapter (the
        # "Dublin, Wythe County" defect). Omit the county instead of snapping.
        resolved = _resolved(["Beaufort County, South Carolina"], "Beaufort County")
        wps = [_wp(32.4, -80.6, None)]
        pois = [_poi(0.0, "A", 32.4, -80.6)]
        out = pois_result(pois, wps, resolved)
        assert out[0][1].county is None


class TestHelpers:
    def test_region_counties_strips_state(self):
        resolved = _resolved(
            ["Beaufort County, South Carolina", "Kerry, Ireland"], "Beaufort County"
        )
        assert OverpassChapterSource._region_counties(resolved) == {"Beaufort County", "Kerry"}

    def test_region_counties_none_is_empty(self):
        assert OverpassChapterSource._region_counties(None) == set()

    def test_nearest_waypoint_county_picks_closest(self):
        wps = [_wp(0.0, 0.0, "Far County"), _wp(10.0, 10.0, "Near County")]
        assert OverpassChapterSource._nearest_waypoint_county(10.1, 10.1, wps) == "Near County"

    def test_nearest_waypoint_county_none_when_ungeocoded(self):
        wps = [_wp(0.0, 0.0, None)]
        assert OverpassChapterSource._nearest_waypoint_county(0.0, 0.0, wps) is None


def pois_result(pois, wps, resolved):
    return OverpassChapterSource.pois_to_locations(
        pois, wps, min_chapters=1, resolved_location=resolved
    )
