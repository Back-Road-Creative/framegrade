"""build_lf_title cross-county/cross-state flake demotion.

Drive-path equivalent of tests/test_ost_cross_county_validation.py: a
stale-GPS / geocoder flake on one endpoint must not fabricate a bogus
"A to B" route title. The on-route midpoint decides which endpoint is
real; the flake is dropped and the title degrades to the single-location
cascade (whose county base is the dominant county from per_lf_location).
"""

from types import SimpleNamespace

from gomoveshift.pipeline.title_builder import build_lf_title

# Real cached coords. Colleton ~32.66N; Beaufort ~32.30N.
_COLLETON = (32.6591, -80.6646)
_BEAUFORT = (32.3095, -80.6332)
_COLLETON_MID = (32.55, -80.66)  # still Colleton, between the two
_CHARLESTON = (32.78, -79.93)  # genuine third county


def _geo(county_state_by_coord, town_by_coord):
    class _Geo:
        def _nearest(self, lat, lon, table):
            best, best_d = None, 0.4
            for (clat, clon), v in table.items():
                d = abs(lat - clat) + abs(lon - clon)
                if d < best_d:
                    best_d, best = d, v
            return best

        def reverse_geocode(self, lat, lon):
            town = self._nearest(lat, lon, town_by_coord)
            return SimpleNamespace(city=town, locality=None, suburb=None, name=None)

        def reverse_geocode_detailed(self, lat, lon):
            cs = self._nearest(lat, lon, county_state_by_coord)
            return {"county": cs[0], "state": cs[1]} if cs else {}

    return _Geo()


def _seg(lat, lon):
    return SimpleNamespace(gps_center_lat=lat, gps_center_lon=lon)


def _pt(lat, lon):
    return {"lat": lat, "lon": lon}


class TestEndpointDemotion:
    def test_flake_end_endpoint_demoted(self):
        """start=Colleton (real), end=Beaufort (flake), midpoint=Colleton →
        drop the Beaufort endpoint; the bogus 'to Beaufort' route vanishes."""
        county_state = {
            _COLLETON: ("Colleton County", "South Carolina"),
            _COLLETON_MID: ("Colleton County", "South Carolina"),
            _BEAUFORT: ("Beaufort County", "South Carolina"),
        }
        towns = {_COLLETON: "Edisto Beach", _BEAUFORT: "Beaufort", _COLLETON_MID: "Jacksonboro"}
        # Trace stays in Colleton through the midpoint; a single trailing
        # Beaufort flake point (5% of the vote, under LOCATION_MATERIAL_SHARE)
        # drags the end centroid across the line. Midpoint is still Colleton
        # and Beaufort holds no material footage → the endpoint is the flake.
        # (Material Beaufort footage keeps the endpoint instead:
        # tests/test_title_cross_county_endpoints.py.)
        gps_points = [_pt(*_COLLETON)] * 19 + [_pt(*_BEAUFORT)]
        title = build_lf_title(
            segments=[_seg(*_COLLETON), _seg(*_BEAUFORT)],
            geocoder=_geo(county_state, towns),
            per_lf_location="Colleton County, South Carolina",
            gps_points=gps_points,
        )
        assert "Beaufort" not in title
        assert " to " not in title  # no "A to B" route claim survived

    def test_genuine_crossing_keeps_both(self):
        """start=Beaufort, end=Charleston, midpoint=Colleton (third county) →
        genuine multi-county drive; both endpoints kept."""
        county_state = {
            _BEAUFORT: ("Beaufort County", "South Carolina"),
            _COLLETON_MID: ("Colleton County", "South Carolina"),
            _CHARLESTON: ("Charleston County", "South Carolina"),
        }
        towns = {_BEAUFORT: "Beaufort", _CHARLESTON: "Charleston", _COLLETON_MID: "Jacksonboro"}
        gps_points = [_pt(*_BEAUFORT)] * 3 + [_pt(*_COLLETON_MID)] * 3 + [_pt(*_CHARLESTON)] * 3
        title = build_lf_title(
            segments=[_seg(*_BEAUFORT), _seg(*_CHARLESTON)],
            geocoder=_geo(county_state, towns),
            per_lf_location="Beaufort County, South Carolina",
            gps_points=gps_points,
        )
        assert "Beaufort to Charleston" in title

    def test_same_county_unaffected(self):
        """Two towns in the same county render the normal route title."""
        a, b = (32.30, -80.63), (32.33, -80.66)
        county_state = {a: ("Beaufort County", "SC"), b: ("Beaufort County", "SC")}
        towns = {a: "Port Royal", b: "Beaufort"}
        title = build_lf_title(
            segments=[_seg(*a), _seg(*b)],
            geocoder=_geo(county_state, towns),
            per_lf_location="Beaufort County, South Carolina",
            gps_points=[_pt(*a)] * 3 + [_pt(*b)] * 3,
        )
        assert "Port Royal to Beaufort" in title
