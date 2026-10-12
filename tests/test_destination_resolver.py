"""Tests for destination_resolver — most-specific enclosing destination POI.

Coverage: smallest-polygon tie-break (Donnelley WMA beats ACE Basin),
composite-score prominence weighting, reused chapters vocabulary, and the
None paths (no enclosing POI / no segments / source failure). All offline —
the POI source is injected; no live Overpass call.
"""

from types import SimpleNamespace

from gomoveshift.pipeline.destination_resolver import resolve_destination


def _seg(lat, lon, score=1.0):
    return SimpleNamespace(gps_center_lat=lat, gps_center_lon=lon, composite_score=score)


def _node(lat, lon, **tags):
    return {"type": "node", "lat": lat, "lon": lon, "tags": dict(tags)}


def _area(s, w, n, e, **tags):
    """A relation with explicit bounds (a polygon)."""
    bounds = {"minlat": s, "minlon": w, "maxlat": n, "maxlon": e}
    return {"type": "relation", "bounds": bounds, "tags": dict(tags)}


# Featured footage sits inside Donnelley WMA, itself inside the broad ACE Basin.
_PEAK = _seg(32.62, -80.50, score=0.9)
_ACE = _area(32.40, -80.80, 32.90, -80.20, boundary="protected_area", name="ACE Basin")
_WMA = _area(
    32.58,
    -80.56,
    32.66,
    -80.44,
    leisure="nature_reserve",
    name="Donnelley Wildlife Management Area",
)


def test_smallest_polygon_wins_tiebreak():
    """Both enclose the peak with equal prominence; the smaller WMA wins —
    and ordering must not change the winner (determinism)."""
    assert resolve_destination([_PEAK], poi_source=lambda _b: [_ACE, _WMA]) == _WMA["tags"]["name"]
    assert resolve_destination([_PEAK], poi_source=lambda _b: [_WMA, _ACE]) == _WMA["tags"]["name"]


def test_prominence_weighting_decides():
    """A high-score peak in B outvotes a low-score peak in A; both reserves are
    equally specific (same tag/area)."""
    a = _area(40.00, -80.00, 40.10, -79.90, leisure="nature_reserve", name="Reserve A")
    b = _area(41.00, -80.00, 41.10, -79.90, leisure="nature_reserve", name="Reserve B")
    peaks = [_seg(40.05, -79.95, score=0.10), _seg(41.05, -79.95, score=0.95)]
    assert resolve_destination(peaks, poi_source=lambda _b: [a, b]) == "Reserve B"


def test_vocabulary_viewpoint_and_beach():
    vp = _node(32.62, -80.50, tourism="viewpoint", name="Bear Island Overlook")
    assert resolve_destination([_PEAK], poi_source=lambda _b: [vp]) == "Bear Island Overlook"
    beach = _area(33.00, -79.20, 33.05, -79.10, natural="beach", name="Botany Bay Beach")
    peak = _seg(33.02, -79.15, score=0.8)
    assert resolve_destination([peak], poi_source=lambda _b: [beach]) == "Botany Bay Beach"


def test_non_destination_tag_ignored():
    """A town/place node is not a destination POI even if it encloses the peak."""
    town = _node(32.62, -80.50, place="town", name="Green Pond")
    assert resolve_destination([_PEAK], poi_source=lambda _b: [town]) is None


def test_none_paths():
    far = _area(10.0, 10.0, 10.1, 10.1, leisure="nature_reserve", name="Far Reserve")
    unnamed = _area(32.58, -80.56, 32.66, -80.44, leisure="nature_reserve")

    def _boom(_b):
        raise RuntimeError("overpass down")

    # no enclosing POI, no segments, no POIs, source failure, unnamed POI
    assert resolve_destination([_PEAK], poi_source=lambda _b: [far]) is None
    assert resolve_destination([], poi_source=lambda _b: [_WMA]) is None
    assert resolve_destination([_PEAK], poi_source=lambda _b: []) is None
    assert resolve_destination([_PEAK], poi_source=_boom) is None
    assert resolve_destination([_PEAK], poi_source=lambda _b: [unnamed]) is None


def test_out_geom_relation_no_center_resolves():
    """Production ``out geom`` shape: a relation carries ``bounds`` + per-member
    ``geometry`` but NO ``center``. Its bounds must still enclose the peak and
    resolve the destination — this is the exact shape the query fix restores
    (``out center`` returned no ``bounds`` at all, degenerating enclosure)."""
    peak = _seg(52.0, -9.5, score=0.9)
    relation = {
        "type": "relation",
        "bounds": {"minlat": 51.9, "minlon": -9.6, "maxlat": 52.1, "maxlon": -9.4},
        "members": [
            {
                "type": "way",
                "role": "outer",
                "geometry": [{"lat": 51.9, "lon": -9.6}, {"lat": 52.1, "lon": -9.4}],
            }
        ],
        "tags": {"boundary": "national_park", "name": "Killarney National Park"},
    }
    assert (
        resolve_destination([peak], poi_source=lambda _b: [relation]) == "Killarney National Park"
    )


def test_zero_score_peak_does_not_vote():
    """A composite_score-0 peak contributes no weight; with no other voters the
    destination earns no positive vote → None."""
    peak = _seg(32.62, -80.50, score=0.0)
    assert resolve_destination([peak], poi_source=lambda _b: [_WMA]) is None


# ── polygon containment (bbox is only the node/center fallback) ─────────────

# A thin DIAGONAL protected strip: its bounding box covers the whole
# 32.37–32.93 × -80.80–-80.30 span, but the real polygon is a narrow band
# along the diagonal. Way shape: flat ``out geom`` ``geometry``.
_DIAGONAL_STRIP = {
    "type": "way",
    "bounds": {"minlat": 32.37, "minlon": -80.80, "maxlat": 32.93, "maxlon": -80.30},
    "geometry": [
        {"lat": 32.37, "lon": -80.80},
        {"lat": 32.43, "lon": -80.80},
        {"lat": 32.93, "lon": -80.30},
        {"lat": 32.87, "lon": -80.30},
        {"lat": 32.37, "lon": -80.80},
    ],
    "tags": {"boundary": "protected_area", "name": "Diagonal Strip WMA"},
}


def test_diagonal_polygon_bbox_corner_not_enclosed():
    """The key case: footage in the empty bbox corner of a diagonal/thin
    protected area must NOT anchor the title on it — bbox containment said it
    did; polygon containment says it doesn't."""
    corner_peak = _seg(32.90, -80.75, score=0.9)  # inside bbox, far off the strip
    assert resolve_destination([corner_peak], poi_source=lambda _b: [_DIAGONAL_STRIP]) is None


def test_diagonal_polygon_on_strip_encloses():
    """Footage actually ON the diagonal strip still resolves the destination."""
    on_strip = _seg(32.65, -80.55, score=0.9)
    assert (
        resolve_destination([on_strip], poi_source=lambda _b: [_DIAGONAL_STRIP])
        == "Diagonal Strip WMA"
    )


def test_multipolygon_split_outer_and_inner_hole():
    """Relation shape: the outer ring arrives SPLIT across two member ways
    (stitched end-to-end), plus an ``inner`` hole. A peak in the hole is not
    enclosed; a peak inside the outer (outside the hole) is."""
    relation = {
        "type": "relation",
        "bounds": {"minlat": 32.0, "minlon": -81.0, "maxlat": 33.0, "maxlon": -80.0},
        "members": [
            {
                "type": "way",
                "role": "outer",
                "geometry": [
                    {"lat": 32.0, "lon": -81.0},
                    {"lat": 33.0, "lon": -81.0},
                    {"lat": 33.0, "lon": -80.0},
                ],
            },
            {
                "type": "way",
                "role": "outer",
                "geometry": [
                    {"lat": 33.0, "lon": -80.0},
                    {"lat": 32.0, "lon": -80.0},
                    {"lat": 32.0, "lon": -81.0},
                ],
            },
            {
                "type": "way",
                "role": "inner",
                "geometry": [
                    {"lat": 32.4, "lon": -80.6},
                    {"lat": 32.6, "lon": -80.6},
                    {"lat": 32.6, "lon": -80.4},
                    {"lat": 32.4, "lon": -80.4},
                    {"lat": 32.4, "lon": -80.6},
                ],
            },
        ],
        "tags": {"leisure": "nature_reserve", "name": "Holey Reserve"},
    }
    in_hole = _seg(32.5, -80.5, score=0.9)
    assert resolve_destination([in_hole], poi_source=lambda _b: [relation]) is None
    in_outer = _seg(32.2, -80.2, score=0.9)
    assert resolve_destination([in_outer], poi_source=lambda _b: [relation]) == "Holey Reserve"


def test_bounds_only_element_still_uses_bbox_fallback():
    """A bounds-only element (legacy ``out center`` / no geometry) keeps the
    bbox behaviour — the polygon test only arms when geometry is present."""
    assert (
        resolve_destination([_PEAK], poi_source=lambda _b: [_WMA])
        == "Donnelley Wildlife Management Area"
    )


def test_specific_poi_beats_enclosing_superset_region():
    """A broad region encloses EVERY peak, a specific POI encloses only the
    plurality of peak mass. The broad region strictly outvotes the enclosed POI
    (superset), so the pre-fix prominence-first winner anchored the OST on the
    broad region. Once the specific POI clears the peak-mass-share bar it must
    win on SPECIFICITY, regardless of the superset's larger raw vote — and the
    outcome is order-independent (determinism)."""
    # Three peaks inside Donnelley WMA (itself inside ACE Basin), one peak inside
    # ACE Basin but outside the WMA. ACE vote = 4×0.5 = 2.0 > WMA vote = 3×0.5 =
    # 1.5, yet the WMA holds 0.75 of the featured mass — it is the destination.
    inside_wma = [_seg(32.62, -80.50, 0.5), _seg(32.60, -80.52, 0.5), _seg(32.64, -80.48, 0.5)]
    outside_wma_in_ace = _seg(32.45, -80.70, 0.5)
    peaks = inside_wma + [outside_wma_in_ace]
    want = "Donnelley Wildlife Management Area"
    assert resolve_destination(peaks, poi_source=lambda _b: [_ACE, _WMA]) == want
    assert resolve_destination(peaks, poi_source=lambda _b: [_WMA, _ACE]) == want
