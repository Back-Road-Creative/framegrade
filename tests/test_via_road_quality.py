"""Via-road quality filter.

A via-road earns a place in the title / burned OST ONLY when it is a named
through-route a viewer could recognise or search. Without a quality
check the road pick takes the first ``address.road`` in the winning county and
can yield junk via-clauses such as "Colleton County via Dickie's" or
"Beaufort County via Homestead Drive". The filter
drops residential/track/service ways and business names so the via-clause is
OMITTED rather than stamping junk, while keeping real routes (US-64, Sea Island
Parkway).
"""

from gomoveshift.pipeline.title_builder import (
    DominantLocation,
    _road_passes_quality,
    dominant_location,
)


class TestRoadPassesQuality:
    def test_route_ref_accepted(self):
        assert _road_passes_quality("US 64")
        assert _road_passes_quality("US-64")
        assert _road_passes_quality("SC 174")
        assert _road_passes_quality("I-26")
        assert _road_passes_quality("N86")  # Irish national route
        assert _road_passes_quality("Highway 64")

    def test_through_road_name_suffix_accepted_without_osm_class(self):
        # Real named through-roads survive even when OSM class is absent
        # (stale cache / place-node match / suffix path).
        assert _road_passes_quality("Sea Island Parkway")
        assert _road_passes_quality("Blue Ridge Parkway")
        assert _road_passes_quality("Overseas Highway")
        assert _road_passes_quality("Woodland Trail") is False  # "Trail" is NOT a through suffix

    def test_through_road_class_accepted(self):
        assert _road_passes_quality(
            "Andrew Jackson Hwy", place_class="highway", match_type="primary"
        )
        assert _road_passes_quality("Some Secondary", place_class="highway", match_type="secondary")

    def test_known_byway_substring_accepted(self):
        # A named scenic route with a generic suffix survives via the byway
        # SUBSTRING match ("Slea Head Drive" carries the known "slea head").
        assert _road_passes_quality("Slea Head Drive")
        assert _road_passes_quality("Ring of Kerry")
        # ...but a plain junk road carrying no byway name is still dropped.
        assert _road_passes_quality("Blocker Run Road") is False

    def test_local_roads_rejected(self):
        # Junk via-roads, all residential/track/service class.
        assert (
            _road_passes_quality("Homestead Drive", place_class="highway", match_type="residential")
            is False
        )
        assert (
            _road_passes_quality(
                "North 4th Street", place_class="highway", match_type="residential"
            )
            is False
        )
        assert (
            _road_passes_quality(
                "Blocker Run Road", place_class="highway", match_type="unclassified"
            )
            is False
        )
        assert (
            _road_passes_quality("Donnelley Drive", place_class="highway", match_type="track")
            is False
        )
        assert (
            _road_passes_quality("Pineland Road", place_class="highway", match_type="service")
            is False
        )

    def test_business_name_rejected(self):
        # "Dickie's" leaked into OSM's road slot from a business — the
        # possessive/business guard drops it even if tagged highway.
        assert (
            _road_passes_quality("Dickie's", place_class="amenity", match_type="restaurant")
            is False
        )
        assert (
            _road_passes_quality("Dickie's", place_class="highway", match_type="service") is False
        )

    def test_empty_and_none_rejected(self):
        assert _road_passes_quality(None) is False
        assert _road_passes_quality("") is False
        assert _road_passes_quality("   ") is False

    def test_no_class_local_road_rejected(self):
        # Suffix path (no OSM class) — a "... Drive"/"... Road" without a ref
        # or through-suffix is dropped rather than surfaced.
        assert _road_passes_quality("Homestead Drive") is False
        assert _road_passes_quality("Stocks Creek Road") is False


class TestDominantLocationRoadQuality:
    """The filter applies at the robust-by-construction site: the winning-
    county road pick in ``dominant_location``."""

    _P = (32.6591, -80.6646)  # Colleton County (ACE Basin)

    def _geo(self, road, place_class="highway", match_type="residential"):
        detail = {
            "county": "Colleton County",
            "state": "South Carolina",
            "road": road,
            "place_class": place_class,
            "match_type": match_type,
        }

        class _Geo:
            def reverse_geocode_detailed(self, lat, lon):
                return dict(detail)

        return _Geo()

    def _points(self):
        return [{"lat": self._P[0], "lon": self._P[1]}] * 6

    def test_junk_road_dropped(self):
        result = dominant_location(self._points(), self._geo("Donnelley Drive", match_type="track"))
        assert isinstance(result, DominantLocation)
        assert result.county == "Colleton County"
        assert result.road is None  # junk via-road omitted, not stamped

    def test_real_route_kept(self):
        result = dominant_location(
            self._points(), self._geo("US 17", place_class="highway", match_type="primary")
        )
        assert result.road == "US-17"  # route ref display-normalized

    def test_named_parkway_kept_without_class(self):
        result = dominant_location(
            self._points(),
            self._geo("Sea Island Parkway", place_class="place", match_type="hamlet"),
        )
        assert result.road == "Sea Island Parkway"
