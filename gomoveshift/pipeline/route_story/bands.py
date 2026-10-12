"""Per-zoom-band imagery source planning for the Route Story map bundle.

A single NAIP raster over the focal footprint cannot serve zooms 9-15:
`write_tile_pyramid` only writes a tile a source raster *fully* covers
(`bundle.py:_fully_covered`), and a footprint sized for the focal trail can
never fully cover a wide, coarse-zoom tile — so the bundle floor silently
becomes z13 and no zoom-out from the national scene is possible.

This module plans, but does not fetch or write, six imagery zoom bands
that together let a scene descend from the national view to the trail:

- z2-z4  (``"blue_marble"``) — NASA GIBS Blue Marble, a dateless global
  backdrop shared across every bundle (see `prepare.py`'s shared tile cache)
- z5-z8  (``"gibs"``)        — NASA GIBS trip-date true-colour / Blue Marble
  cloud-fraction fallback, over the same national margin as ``blue_marble``
- z9-z10 (``"naip_coarse"``) — a coarse-resolution NAIP pull
- z11-z12 (``"naip_mid"``)    — a mid-resolution NAIP pull
- z13-z14 (``"naip_fine"``)   — a fine-resolution NAIP pull
- z15     (``"naip"``)        — 2 m NAIP over the focal footprint

Splitting the old single z2-8 gibs band in two matters because a national
margin snapped outward to a whole tile **at z2** is a ~90x66 degree tile at
mid-latitudes — writing z2-8 over that one bbox was ~10.9k tiles per bundle.
Snapping the coarsest three zooms at z5
instead keeps that bbox to the national margin's actual size, and z2-4 is
fetched once into a shared cache instead of per bundle.

Each band gets its own bbox, snapped outward to whole tiles **at the band's
own coarsest (minimum) zoom** — not an arbitrary margin. That is the only
choice that makes ``_fully_covered`` true for *every* zoom in the band with
a single rectangular raster: a Web-Mercator tile grid is a strict refinement
across zooms, so a tile boundary aligned at a coarse zoom is automatically a
valid tile boundary at every finer zoom nested under it, but the reverse does
not hold (a box tight-snapped to the finest zoom's tiny tiles is not, in
general, aligned to any coarser tile at all). `prepare_bundle` consumes this plan
to fetch each band and write its tiles; nothing here makes a network call or
touches a bundle directory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from .bundle import Footprint, lonlat_to_tile, tile_bounds
from .sources import DEM_3DEP_MAX_TILE_PX, NAIP_MAX_TILE_PX

BBox = tuple[float, float, float, float]  # (west, south, east, north)

_EARTH_RADIUS_M = 6_371_008.8
_EQUATOR_CIRCUMFERENCE_M = 2 * math.pi * _EARTH_RADIUS_M
_TILE_PX = 256

# exportImage / WMS request-size ceilings this module respects when sizing a
# band's raster fetch. NAIP and 3DEP use the fetchers' own measured tile caps
# (`sources.NAIP_MAX_TILE_PX` / `DEM_3DEP_MAX_TILE_PX`), so the planner and
# the fetch can never disagree. GIBS's WMS endpoint has no such limit coded
# in `sources.py` today; 8192 is the conservative ceiling common to WMS
# deployments and is treated as a planning default, not a measured fact.
NAIP_MAX_REQUEST_PX = NAIP_MAX_TILE_PX
GIBS_MAX_REQUEST_PX = 8192

# The shape of the ladder itself, folded into every bundle's cache address
# (`prepare._band_source_versions` -> `bundle.bundle_dir_for`).
#
# A bundle's other key fields describe WHERE it is and WHICH sources it came
# from. None of them move when this module changes which source covers which
# zoom, how wide a band reaches, or how a band's raster is sized -- so before
# this constant existed, a ladder change could not reach a trip that had
# already been prepared: `prepare` skips its whole fetch on any existing
# bundle directory that verifies clean, and the directory was the same one.
# The only remedy was for an operator to know the hash and delete it by hand.
#
# Bump this whenever a change here would make an already-cached bundle's
# tiles wrong or stale for the same request. "bands-3" is the six-band ladder
# whose finest zooms step at most 2 apart (see _BAND_SPECS); "bands-2" was the
# five-band ladder sized off the opening mosaic cap, whose z8->z11->z14 steps
# left two spans with no sharp raster; "bands-1" was everything before it.
# Bundles built under an older plan keep their own addresses.
BAND_PLAN_VERSION = "bands-3"

# Shared default for both `plan_imagery_bands`' and `plan_terrain_bands`'
# `regional_margin_km` — a single named constant so the two callers (and
# `prepare_bundle`'s own default) can never drift apart and produce
# non-identical regional bboxes by accident.
DEFAULT_REGIONAL_MARGIN_KM = 75.0

_MAX_REQUEST_PX: dict[str, int] = {
    "blue_marble": GIBS_MAX_REQUEST_PX,
    "gibs": GIBS_MAX_REQUEST_PX,
    "naip_coarse": NAIP_MAX_REQUEST_PX,
    "naip_mid": NAIP_MAX_REQUEST_PX,
    "naip_fine": NAIP_MAX_REQUEST_PX,
    "naip": NAIP_MAX_REQUEST_PX,
    # 3DEP's own exportImage max_px (sources.fetch_3dep_dem's default) — the
    # two terrain bands only ever feed `bands_receipt`'s raster_px estimate,
    # never an actual band_raster_size-driven fetch of their own.
    "terrain_regional": DEM_3DEP_MAX_TILE_PX,
    "terrain_focal": DEM_3DEP_MAX_TILE_PX,
}

# (source, zoom range, margin fraction of the corresponding plan_imagery_bands
# keyword) — see plan_imagery_bands for how each margin is applied.
#
# The FINEST zoom of each successive band may be at most
# log2(_OPENING_MOSAIC_MAX_WIDTH_PX / _ASSUMED_OPENING_WIDTH_PX) = 2 levels
# above the previous band's, and that is the whole reason this tuple looks the
# way it does. A cap-widened band's bbox is _OPENING_MOSAIC_MAX_WIDTH_PX over
# its finest zoom's pixel density and it stops resolving at
# _ASSUMED_OPENING_WIDTH_PX over that same density, so one band stretches
# exactly 8192/1920 = 4.27x in span no matter how its margin is set. Step the
# finest zooms 3 levels apart and the camera needs 8x from one band: a gap
# opens that NO margin can close, because the bbox that would cover the wide
# end pushes the finest zoom's mosaic past the cap and
# `satellite._complete_level` drops that zoom entirely.
#
# That is how the ladder shipped z8 -> z11 -> z14 with every band looking
# individually well-sized, and why a render came back "very dark and fuzzy
# when zooming in": measured on a real bundle, 5.63-10.55 deg and
# 0.70-1.32 deg had no sharp raster at all and `_viewport_raster` fell back a
# whole band and upscaled. `naip_fine` exists to make the steps 2 apart.
# `test_consecutive_bands_are_at_most_two_zoom_levels_apart` fails on any
# future edit that widens a step again.
#
# `blue_marble` is exempt: it is the backdrop beneath `gibs`, not a rung. The
# opening camera cannot open wider than the raster it starts on, so its widest
# frame is `gibs`'s own bbox and the step above `gibs` is never traversed.
_BAND_SPECS: tuple[
    tuple[Literal["blue_marble", "gibs", "naip_coarse", "naip_mid", "naip_fine", "naip"], range],
    ...,
] = (
    ("blue_marble", range(2, 5)),
    ("gibs", range(5, 9)),
    ("naip_coarse", range(9, 11)),
    ("naip_mid", range(11, 13)),
    ("naip_fine", range(13, 15)),
    ("naip", range(15, 16)),
)

# Mirrors layers._OPENING_MIN_SOURCE_PX_PER_OUTPUT_PX and film.DELIVERY_WIDTH
# (rendering-side constants) -- duplicated, not imported: bands.py plans a
# bundle before any render code runs and must not depend on it. Used only by
# _sharp_handoff_focal_margin_m to keep the focal band wide enough that the
# opening camera never has to hand off to naip_coarse's own finest zoom past
# where it can resolve sharply (the "dark and fuzzy when zooming in" defect;
# real bundle measurement: naip_coarse's z12 fell as low as 0.17 source px
# per output px at the narrow focal bbox this replaces).
_OPENING_MIN_SOURCE_PX_PER_OUTPUT_PX = 0.5
_ASSUMED_OPENING_WIDTH_PX = 1920
# Mirrors film.DELIVERY_HEIGHT -- used only to keep a cap-widened band's
# bbox tall enough to fully COVER a full-width viewport, not just wide
# enough to resolve one: `_viewport_raster` requires a covering raster's
# bbox to contain the whole viewport rectangle, and the viewport's own
# height is `span_lon * height / width` -- a bbox widened east-west to the
# cap but left at its original (margin-only) height is proportionally too
# short north-south, so a full-width viewport at that band's own aspect
# falls outside it on the north/south edges and the coverage search falls
# back a whole band coarser regardless of how wide the bbox is.
_ASSUMED_OPENING_HEIGHT_PX = 1080

# Mirrors satellite.OPENING_MOSAIC_MAX_WIDTH_PX -- duplicated, not imported
# for the same reason as the constants above. `bundle_opening_rasters` drops
# a whole zoom level of a band whenever that zoom's mosaic would need more
# than this many tile columns (`satellite._complete_level`); before this
# band's own zoom range was sized against that ceiling, a band's bbox --
# snapped at its own coarsest zoom to guarantee coverage -- could still be
# too wide for its own FINEST zoom to fit the cap, silently dropping the
# handoff level the opening camera needed and forcing a fallback two bands
# coarser (measured: 21/101 walked opening fractions below the sharpness
# floor, split between a gibs-z8 window at 0.53-0.58 and a naip_coarse-z12
# window at 0.77-0.91). Sizing each of naip_coarse/naip_mid/naip's bbox to
# fit exactly at its own finest zoom -- never past `regional_margin_km` /
# `focal_margin_m`'s own floor -- closes both windows: see
# `_cap_bbox_width_deg` and `_fit_snapped_bbox_to_cap`.
_OPENING_MOSAIC_MAX_WIDTH_PX = 8192

# Bands whose bbox is additionally floored at `_cap_bbox_width_deg` of their
# own finest zoom -- blue_marble and gibs are untouched by this fix (their
# margins were never the source of the gap; widening them further would only
# grow the shared national fetch for no coverage benefit).
_CAP_WIDENED_SOURCES = frozenset({"naip_coarse", "naip_mid", "naip_fine", "naip"})


def _cap_bbox_width_deg(max_zoom: int) -> float:
    """Widest west-east bbox, in degrees, whose ``max_zoom`` mosaic still
    fits within ``_OPENING_MOSAIC_MAX_WIDTH_PX`` tile columns."""
    return _OPENING_MOSAIC_MAX_WIDTH_PX / _px_per_degree_lon(max_zoom)


def _fit_snapped_bbox_to_cap(bbox: BBox, zooms: range, must_cover: BBox) -> BBox:
    """Resize ``bbox`` (already tile-snapped at ``zooms.start``) to exactly the
    tile-column count ``_cap_bbox_width_deg`` allows at ``zooms.start``.

    The cap is TWO-SIDED, not a floor. Too narrow and the opening camera has
    no raster to hand off to as it descends (the gap this ladder closes); too
    WIDE and ``satellite._complete_level`` drops that zoom level from the
    bundle outright for exceeding ``_OPENING_MOSAIC_MAX_WIDTH_PX``, deleting
    the very handoff level the widening existed to provide. Treating it as a
    floor alone left a real focal band 9 tile columns wide where 8
    fit -- a 9216 px mosaic against an 8192 px ceiling, at every route width
    and focal margin measured.

    Both directions move whole tiles at ``zooms.start`` rather than degrees:
    re-snapping a degree-space edge rounds outward and can silently add a
    column back. Whole tiles keep the finest zoom's column count exactly at
    the cap, because ``target_cols`` scales by the band's zoom span, an exact
    power of two.

    ``must_cover`` (the raw route bbox) is the one thing never traded away:
    a route too wide for this band's finest zoom to hold under the cap keeps
    its own columns and accepts the drop, because no bbox can serve it here
    and the next coarser band is what covers that span.

    Also widens north-south to at least the delivery aspect
    (``_ASSUMED_OPENING_HEIGHT_PX`` / ``_ASSUMED_OPENING_WIDTH_PX``) of the
    resulting width, then re-snaps the whole bbox at ``zooms.start`` -- a
    bbox wide enough east-west but left at its original (margin-only) height
    is too short for a full-width viewport at this aspect to fit inside on
    the north/south edges, and ``_viewport_raster`` requires full coverage
    on every edge (see this constant's own comment).
    """
    z = zooms.start
    n = 2**z
    west, south, east, north = bbox
    mid_lat = (south + north) / 2.0
    x_west_f, _ = lonlat_to_tile(west + _BOUNDARY_NUDGE_DEG, mid_lat, z)
    x_east_f, _ = lonlat_to_tile(east - _BOUNDARY_NUDGE_DEG, mid_lat, z)
    x_min = max(0, min(n - 1, math.floor(x_west_f)))
    x_max = max(0, min(n - 1, math.floor(x_east_f)))
    keep_west_f, _ = lonlat_to_tile(must_cover[0], mid_lat, z)
    keep_east_f, _ = lonlat_to_tile(must_cover[2], mid_lat, z)
    keep_min = max(0, min(n - 1, math.floor(keep_west_f)))
    keep_max = max(0, min(n - 1, math.floor(keep_east_f)))
    zoom_span = (zooms.stop - 1) - z
    target_cols = max(1, (_OPENING_MOSAIC_MAX_WIDTH_PX // _TILE_PX) // 2**zoom_span)
    deficit = target_cols - (x_max - x_min + 1)
    if deficit > 0:
        grow_west = deficit // 2
        grow_east = deficit - grow_west
        x_min = max(0, x_min - grow_west)
        x_max = min(n - 1, x_max + grow_east)
    elif deficit < 0:
        excess = -deficit
        trim_west = excess // 2
        x_min = min(x_min + trim_west, keep_min)
        x_max = max(x_min + target_cols - 1, keep_max)
        x_min = min(x_min, x_max)
    eps = _BOUNDARY_NUDGE_DEG
    widened_west = tile_bounds(z, x_min, 0).west + eps
    widened_east = tile_bounds(z, x_max, 0).east - eps
    widened_width = widened_east - widened_west
    min_height = widened_width * (_ASSUMED_OPENING_HEIGHT_PX / _ASSUMED_OPENING_WIDTH_PX)
    if (north - south) < min_height:
        pad = (min_height - (north - south)) / 2.0
        south, north = max(south - pad, -90.0), min(north + pad, 90.0)
    return _snap_bbox_to_tiles((widened_west, south, widened_east, north), z)


def _px_per_degree_lon(zoom: int) -> float:
    """Web-Mercator east-west pixel density at zoom ``z``.

    Unlike latitude, longitude scales linearly with tile index at every
    zoom, so (unlike ``_resolution_at_zoom``'s m/px) this needs no
    mean-latitude correction.
    """
    return _TILE_PX * 2**zoom / 360.0


def _sharp_handoff_focal_margin_m(route_bbox: BBox, focal_margin_m: float) -> float:
    """Widen ``focal_margin_m`` so the focal (``naip``) band's bbox is at
    least as wide as the narrowest span ``naip_coarse``'s own finest zoom can
    no longer resolve at ``_OPENING_MIN_SOURCE_PX_PER_OUTPUT_PX`` -- otherwise
    the opening camera has nowhere to hand off to between the two bands'
    finest zooms and sits on an upscaled ``naip_coarse`` mosaic for however
    much of the descent falls in that gap. Depends only on ``naip_coarse``'s
    zoom range (fixed by ``_BAND_SPECS``), never its own bbox size, so this
    needs no ``regional_margin_km`` input. Both ``plan_imagery_bands`` and
    ``plan_terrain_bands`` call this with the same inputs, so the imagery
    ``naip`` band and the ``terrain_focal`` band it must sit under stay one
    shared bbox by construction (see ``plan_terrain_bands``'s docstring).
    """
    r_west, r_south, r_east, r_north = route_bbox
    naip_coarse_zooms = next(zooms for source, zooms in _BAND_SPECS if source == "naip_coarse")
    min_focal_span_deg = (
        _OPENING_MIN_SOURCE_PX_PER_OUTPUT_PX
        * _ASSUMED_OPENING_WIDTH_PX
        / _px_per_degree_lon(naip_coarse_zooms.stop - 1)
    )
    route_width_deg = r_east - r_west
    if route_width_deg >= min_focal_span_deg:
        return focal_margin_m
    mean_lat = (r_south + r_north) / 2.0
    cos_lat = max(math.cos(math.radians(mean_lat)), 1e-9)
    lon_margin_deg_required = (min_focal_span_deg - route_width_deg) / 2.0
    lat_margin_deg_required = lon_margin_deg_required * cos_lat
    margin_m_required = lat_margin_deg_required * _EARTH_RADIUS_M * (math.pi / 180.0)
    return max(focal_margin_m, margin_m_required)


@dataclass(frozen=True)
class ZoomBand:
    """One zoom range and the source, bbox, and resolution to fetch it at."""

    kind: Literal["imagery", "terrain"]
    zooms: range
    source: Literal[
        "blue_marble",
        "gibs",
        "naip_coarse",
        "naip_mid",
        "naip_fine",
        "naip",
        "terrain_regional",
        "terrain_focal",
    ]
    bbox: BBox
    target_m_per_px: float


@dataclass(frozen=True)
class RasterRequest:
    """One sub-fetch of a band's raster, within a source's max request size."""

    bbox: BBox
    width_px: int
    height_px: int


def _resolution_at_zoom(z: int, mean_lat_deg: float) -> float:
    """Web-Mercator ground resolution (m/px) of a 256px tile at zoom ``z``."""
    return (_EQUATOR_CIRCUMFERENCE_M / _TILE_PX) * math.cos(math.radians(mean_lat_deg)) / (2**z)


def _expand_bbox(bbox: BBox, margin_m: float) -> BBox:
    """Widen bbox by a metric margin, longitude corrected for latitude."""
    west, south, east, north = bbox
    mean_lat = (south + north) / 2.0
    lat_margin_deg = (margin_m / _EARTH_RADIUS_M) * (180.0 / math.pi)
    cos_lat = max(math.cos(math.radians(mean_lat)), 1e-9)
    lon_margin_deg = lat_margin_deg / cos_lat
    return (
        max(west - lon_margin_deg, -180.0),
        max(south - lat_margin_deg, -90.0),
        min(east + lon_margin_deg, 180.0),
        min(north + lat_margin_deg, 90.0),
    )


# `lonlat_to_tile` and `tile_bounds` are float-arithmetic inverses (one via
# log/tan, the other via atan/sinh) but not bit-exact round trips: a bbox
# edge built exactly on a tile boundary can re-project a few ULPs to the
# wrong side of it, which `tiles_covering`'s unguarded `floor()` on the
# west/north edges (`bundle.py`) then reads as belonging to the *next* tile
# out — silently pulling in a huge neighbouring low-zoom tile. This nudge is
# smaller than any real distance that matters here and well inside
# `_fully_covered`'s own `eps=1e-9` tolerance, so it only absorbs float
# noise, never shrinks real coverage.
_BOUNDARY_NUDGE_DEG = 1e-10


def _snap_bbox_to_tiles(bbox: BBox, z: int) -> BBox:
    """Widen ``bbox`` outward to the smallest whole-tile envelope at zoom ``z``."""
    west, south, east, north = bbox
    n = 2**z
    x_west, y_north = lonlat_to_tile(west, north, z)
    x_east, y_south = lonlat_to_tile(east, south, z)
    x_min = max(0, min(n - 1, math.floor(x_west)))
    x_max = max(0, min(n - 1, math.ceil(x_east) - 1))
    y_min = max(0, min(n - 1, math.floor(y_north)))
    y_max = max(0, min(n - 1, math.ceil(y_south) - 1))
    nw = tile_bounds(z, x_min, y_min)
    se = tile_bounds(z, max(x_max, x_min), max(y_max, y_min))
    eps = _BOUNDARY_NUDGE_DEG
    return (nw.west + eps, se.south + eps, se.east - eps, nw.north - eps)


def _naip_coarse_bbox(route_bbox: BBox, regional_margin_km: float) -> BBox:
    """The ``naip_coarse`` imagery band's bbox -- shared verbatim by
    ``plan_terrain_bands``'s ``terrain_regional`` band, same reasoning as
    ``_naip_mid_bbox`` for ``terrain_focal``/``naip_mid``.
    """
    naip_coarse_zooms = next(zooms for source, zooms in _BAND_SPECS if source == "naip_coarse")
    expanded = _expand_bbox(route_bbox, regional_margin_km * 1000.0)
    snapped = _snap_bbox_to_tiles(expanded, naip_coarse_zooms.start)
    return _fit_snapped_bbox_to_cap(snapped, naip_coarse_zooms, route_bbox)


def _naip_fine_bbox(route_bbox: BBox, focal_margin_m: float) -> BBox:
    """The ``naip_fine`` imagery band's bbox -- shared verbatim by
    ``plan_terrain_bands``'s ``terrain_focal`` band. ``terrain_focal`` (z13)
    sits inside ``naip_fine``'s z13-14 range, so it is ``naip_fine`` it must
    share one bbox with (same pairing guarantee this module's other bands rely
    on: see ``plan_terrain_bands``'s docstring). The pairing follows z13
    wherever it lands: it moved here from ``naip_mid`` when ``naip_fine`` took
    over z13-14 to close the ladder's z12-to-z14 step.
    """
    focal_margin_m = _sharp_handoff_focal_margin_m(route_bbox, focal_margin_m)
    naip_fine_zooms = next(zooms for source, zooms in _BAND_SPECS if source == "naip_fine")
    expanded = _expand_bbox(route_bbox, focal_margin_m)
    snapped = _snap_bbox_to_tiles(expanded, naip_fine_zooms.start)
    return _fit_snapped_bbox_to_cap(snapped, naip_fine_zooms, route_bbox)


def plan_imagery_bands(
    route_bbox: BBox,
    *,
    focal_margin_m: float,
    regional_margin_km: float = DEFAULT_REGIONAL_MARGIN_KM,
    national_margin_km: float = 1000.0,
) -> list[ZoomBand]:
    """Plan the six imagery zoom bands from the national scene down to the trail.

    Each band expands ``route_bbox`` by its own margin; the four finest
    bands (``naip_coarse``, ``naip_mid``, ``naip_fine``, ``naip`` --
    ``_CAP_WIDENED_SOURCES``)
    additionally floor that width at ``_cap_bbox_width_deg`` of their own
    finest zoom, so the opening camera (``satellite.bundle_opening_rasters``)
    never has to drop that zoom's mosaic for being too wide to fit under
    ``_OPENING_MOSAIC_MAX_WIDTH_PX`` (module docstring). The result is then
    snapped outward to whole tiles at the band's own minimum zoom (see module
    docstring for why that guarantees full coverage at every zoom in the
    band). ``target_m_per_px`` is the native Web-Mercator resolution of the
    band's finest (maximum) zoom at the snapped bbox's mean latitude.
    ``blue_marble`` and ``gibs`` share the same national margin — one is the
    shared dateless backdrop, the other the trip-date scene layered over it.
    """
    r_west, r_south, r_east, r_north = route_bbox
    if not (r_west < r_east) or not (r_south < r_north):
        raise ValueError("route_bbox must have positive area: west<east and south<north")
    # `naip`'s own margin is the caller's raw floor -- unlike before `naip_mid`
    # existed, it no longer needs `_sharp_handoff_focal_margin_m`'s widening
    # to reach `naip_coarse`'s own floor span: `naip_mid` (below) now closes
    # that gap, and `naip`'s bbox is separately floored to its own cap-
    # derived width by `_CAP_WIDENED_SOURCES` regardless of this margin.
    margins_m = {
        "blue_marble": national_margin_km * 1000.0,
        "gibs": national_margin_km * 1000.0,
        "naip_coarse": regional_margin_km * 1000.0,
        # `naip_mid` spans the middle of the descent and pairs with no terrain
        # band, so it has no margin of its own to honour: `_CAP_WIDENED_SOURCES`
        # sets its width from its own finest zoom either way, and starting from
        # the focal margin keeps it centred on the route like the bands around
        # it.
        "naip_mid": focal_margin_m,
        "naip": focal_margin_m,
    }
    bands: list[ZoomBand] = []
    for source, zooms in _BAND_SPECS:
        if source == "naip_fine":
            snapped = _naip_fine_bbox(route_bbox, focal_margin_m)
        elif source == "naip_coarse":
            snapped = _naip_coarse_bbox(route_bbox, regional_margin_km)
        else:
            expanded = _expand_bbox(route_bbox, margins_m[source])
            snapped = _snap_bbox_to_tiles(expanded, zooms.start)
            if source in _CAP_WIDENED_SOURCES:
                snapped = _fit_snapped_bbox_to_cap(snapped, zooms, route_bbox)
        west, south, east, north = snapped
        mean_lat = (south + north) / 2.0
        target_m_per_px = _resolution_at_zoom(zooms.stop - 1, mean_lat)
        bands.append(
            ZoomBand(
                kind="imagery",
                zooms=zooms,
                source=source,
                bbox=snapped,
                target_m_per_px=target_m_per_px,
            )
        )
    return bands


_TERRAIN_REGIONAL_ZOOMS = range(9, 13)
_TERRAIN_FOCAL_ZOOMS = range(13, 14)


def plan_terrain_bands(
    route_bbox: BBox,
    *,
    focal_margin_m: float,
    regional_margin_km: float = DEFAULT_REGIONAL_MARGIN_KM,
) -> list[ZoomBand]:
    """Plan the two terrain zoom bands: a regional DEM under the regional
    imagery band and a focal DEM at native 3DEP resolution under the trail.

    ``terrain_regional`` (z9-12) is expanded and snapped with the *same*
    margin and zoom as ``plan_imagery_bands``'s ``naip_coarse`` band, so the
    two share the identical bbox by construction — real relief sits under
    every z9-12 imagery tile instead of the coarse terrain zooms silently
    writing little or nothing over a focal-sized DEM raster (this module's
    docstring; the terrain analogue of the imagery bands). Its resolution is the
    z12 Web-Mercator ground resolution, ~30 m at mid-latitudes — close to
    Copernicus GLO-30's native 30 m, so a regional fallback loses little.

    ``terrain_focal`` (z13) shares ``plan_imagery_bands``'s ``naip_fine`` bbox
    exactly, via ``_naip_fine_bbox`` -- z13 sits inside ``naip_fine``'s z13-14
    range, so pairing with ``naip_fine`` (not ``naip_mid``, which now stops at
    z12) is what keeps real relief under every z13 imagery tile. A Web-Mercator
    tile grid is a strict refinement across zooms, so a bbox snapped at
    ``naip_fine``'s z13 is automatically tile-aligned at z14 too (module
    docstring). An unsnapped focal DEM over a route narrower than one z13
    tile fully covers no z13 tile: it wrote nothing and left the focal
    imagery reaching past the terrain mesh.
    """
    r_west, r_south, r_east, r_north = route_bbox
    if not (r_west < r_east) or not (r_south < r_north):
        raise ValueError("route_bbox must have positive area: west<east and south<north")

    focal_bbox = _naip_fine_bbox(route_bbox, focal_margin_m)
    regional_bbox = _naip_coarse_bbox(route_bbox, regional_margin_km)
    _, regional_south, _, regional_north = regional_bbox
    regional_mean_lat = (regional_south + regional_north) / 2.0
    regional_m_per_px = _resolution_at_zoom(_TERRAIN_REGIONAL_ZOOMS.stop - 1, regional_mean_lat)

    return [
        ZoomBand(
            kind="terrain",
            zooms=_TERRAIN_REGIONAL_ZOOMS,
            source="terrain_regional",
            bbox=regional_bbox,
            target_m_per_px=regional_m_per_px,
        ),
        ZoomBand(
            kind="terrain",
            zooms=_TERRAIN_FOCAL_ZOOMS,
            source="terrain_focal",
            bbox=focal_bbox,
            target_m_per_px=10.0,
        ),
    ]


def _bbox_size_m(bbox: BBox) -> tuple[float, float]:
    """Return (width_m, height_m) of a bbox at its mean latitude."""
    west, south, east, north = bbox
    mean_lat = (south + north) / 2.0
    lat_m_per_deg = (math.pi / 180.0) * _EARTH_RADIUS_M
    lon_m_per_deg = lat_m_per_deg * math.cos(math.radians(mean_lat))
    return (east - west) * lon_m_per_deg, (north - south) * lat_m_per_deg


def _tile_raster_requests(band: ZoomBand, max_px: int) -> list[RasterRequest]:
    west, south, east, north = band.bbox
    width_m, height_m = _bbox_size_m(band.bbox)
    total_w = max(1, round(width_m / band.target_m_per_px))
    total_h = max(1, round(height_m / band.target_m_per_px))
    ncols = max(1, math.ceil(total_w / max_px))
    nrows = max(1, math.ceil(total_h / max_px))
    lon_edges = [west + (east - west) * c / ncols for c in range(ncols + 1)]
    lat_edges = [north - (north - south) * r / nrows for r in range(nrows + 1)]
    requests: list[RasterRequest] = []
    for r in range(nrows):
        tile_h = max(1, total_h // nrows + (1 if r < total_h % nrows else 0))
        for c in range(ncols):
            tile_w = max(1, total_w // ncols + (1 if c < total_w % ncols else 0))
            requests.append(
                RasterRequest(
                    bbox=(lon_edges[c], lat_edges[r + 1], lon_edges[c + 1], lat_edges[r]),
                    width_px=tile_w,
                    height_px=tile_h,
                )
            )
    return requests


def band_pixel_size(band: ZoomBand) -> tuple[int, int]:
    """Raw ``(width_px, height_px)`` for ``band``'s bbox at ``target_m_per_px``.

    Unlike :func:`band_raster_size`, this never tiles into sub-requests — a
    caller that would rather clamp to one request than mosaic several (a
    coarse opening-shot band gains no real detail from the extra
    sub-requests a source's max-request-size ceiling would otherwise force)
    uses this and clamps the result itself.
    """
    width_m, height_m = _bbox_size_m(band.bbox)
    width_px = max(1, round(width_m / band.target_m_per_px))
    height_px = max(1, round(height_m / band.target_m_per_px))
    return width_px, height_px


def band_raster_size(band: ZoomBand) -> tuple[int, int] | list[RasterRequest]:
    """Pixel size to fetch ``band.bbox`` at ``band.target_m_per_px``.

    Returns a plain ``(width_px, height_px)`` when that fits the source's max
    request size, else a list of :class:`RasterRequest` tiling the bbox so no
    sub-request exceeds it.
    """
    width_px, height_px = band_pixel_size(band)
    max_px = _MAX_REQUEST_PX[band.source]
    if width_px <= max_px and height_px <= max_px:
        return width_px, height_px
    return _tile_raster_requests(band, max_px)


def bands_receipt(bands: Sequence[ZoomBand]) -> list[dict[str, Any]]:
    """Manifest-ready summary of a band plan: kind, zooms, source, bbox, m/px, raster px."""
    receipt: list[dict[str, Any]] = []
    for band in bands:
        raster = band_raster_size(band)
        raster_px: Any
        if isinstance(raster, tuple):
            raster_px = {"width": raster[0], "height": raster[1]}
        else:
            raster_px = [
                {"bbox": list(req.bbox), "width": req.width_px, "height": req.height_px}
                for req in raster
            ]
        receipt.append(
            {
                "kind": band.kind,
                "zooms": [band.zooms.start, band.zooms.stop - 1],
                "source": band.source,
                "bbox": list(band.bbox),
                "target_m_per_px": band.target_m_per_px,
                "raster_px": raster_px,
            }
        )
    return receipt


def band_footprint(band: ZoomBand) -> Footprint:
    """Convenience: a band's bbox as a `bundle.Footprint`."""
    west, south, east, north = band.bbox
    return Footprint(west=west, south=south, east=east, north=north)
