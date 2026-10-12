"""Red-first tests for per-zoom-band imagery source planning.

`prepare.py` today asks NAIP for zooms 9-15 over one focal-sized raster, but
`write_tile_pyramid` only writes a tile its source raster *fully* covers, and
a focal-sized raster can never fully cover a wide, coarse-zoom tile — so the
bundle floor silently becomes z13 and no descent from the national scene is
possible. `plan_imagery_bands` fixes this by
giving each zoom range its own bbox, sized and tile-snapped so every zoom in
every band gets a fully-covered raster.
"""

from __future__ import annotations

import math

import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import bands as rsb
from gomoveshift.pipeline.route_story.bundle import Footprint, tile_bounds, tiles_covering
from gomoveshift.pipeline.route_story.layers import check_opening_sharpness
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster

# A real ~6.6 km trail-sized route bbox (Blue Ridge area, ~35 deg N).
FOCAL_ROUTE_BBOX = (-83.55, 35.20, -83.48, 35.26)
# A short route whose focal NAIP band (one z13 tile, 1024 px) fits under the
# 2048 px NAIP tile cap; FOCAL_ROUTE_BBOX's focal band (3072 px) now tiles.
SHORT_ROUTE_BBOX = (-83.52, 35.23, -83.515, 35.235)


def _band_footprint(band: rsb.ZoomBand) -> Footprint:
    west, south, east, north = band.bbox
    return Footprint(west=west, south=south, east=east, north=north)


def _uncovered_tiles(band: rsb.ZoomBand) -> list[tuple[int, int, int]]:
    """Every tile of every zoom in ``band`` that is NOT fully inside its bbox."""
    footprint = _band_footprint(band)
    west, south, east, north = band.bbox
    eps = 1e-9
    problems: list[tuple[int, int, int]] = []
    for z in band.zooms:
        for x, y in tiles_covering(footprint, z):
            bounds = tile_bounds(z, x, y)
            covered = (
                bounds.west >= west - eps
                and bounds.east <= east + eps
                and bounds.south >= south - eps
                and bounds.north <= north + eps
            )
            if not covered:
                problems.append((z, x, y))
    return problems


# --- plan_imagery_bands -----------------------------------------------------


def test_plan_imagery_bands_returns_six_bands_national_to_focal():
    bands = rsb.plan_imagery_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)

    assert [b.source for b in bands] == [
        "blue_marble",
        "gibs",
        "naip_coarse",
        "naip_mid",
        "naip_fine",
        "naip",
    ]
    assert [b.kind for b in bands] == ["imagery"] * 6
    assert list(bands[0].zooms) == list(range(2, 5))
    assert list(bands[1].zooms) == list(range(5, 9))
    # Each successive finest zoom steps at most 2 -- see _BAND_SPECS. z9-11 and
    # z12-14 stepped 3 and left two spans with no sharp raster at all.
    assert list(bands[2].zooms) == list(range(9, 11))
    assert list(bands[3].zooms) == list(range(11, 13))
    assert list(bands[4].zooms) == list(range(13, 15))
    assert list(bands[5].zooms) == list(range(15, 16))


def test_gibs_band_bbox_is_far_smaller_than_a_z2_tile():
    """The bug this fixes: z2-8 in one bbox snapped at z2 forces a ~10.9k-tile
    write per bundle. Splitting blue_marble (z2-4, snapped at z2) from gibs
    (z5-8, snapped at z5) keeps the gibs bbox to roughly the national margin,
    not a whole 90x66 degree z2 tile, for a mid-latitude route."""
    bands = rsb.plan_imagery_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)
    gibs = next(b for b in bands if b.source == "gibs")

    west, south, east, north = gibs.bbox
    # A z2 tile is 90 deg wide, 66.5 deg tall (Web-Mercator's clipped
    # latitude range) — the gibs bbox must be a small fraction of that.
    assert (east - west) < 45.0
    assert (north - south) < 33.0


# --- the opening's sharp handoff: no gap between naip_coarse and naip ------


def test_the_focal_band_is_wide_enough_for_naip_coarses_own_floor(tmp_path):
    """The opening camera hands off from `naip_coarse` to `naip` well before
    `naip_coarse`'s own finest zoom (z12) needs upscaling past the opening's
    sharpness floor -- otherwise it sits on an upscaled `naip_coarse` mosaic
    for however much of the descent falls in the gap between the two bands'
    finest zooms (real bundle measurement: as low as 0.17 source px per
    output px). The focal band must be at least as wide as the widest span
    z12 can no longer resolve at that floor."""
    bands = rsb.plan_imagery_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)
    naip_coarse = next(b for b in bands if b.source == "naip_coarse")
    naip = next(b for b in bands if b.source == "naip")

    z12_px_per_deg = 256 * 2**12 / 360.0
    min_focal_span_deg = 0.5 * 1920 / z12_px_per_deg  # floor * assumed width / resolution

    assert (naip.bbox[2] - naip.bbox[0]) >= min_focal_span_deg
    # And it is unchanged from `naip_coarse`'s own bbox in every other way:
    # still the finest, smallest band, just wide enough to close the gap.
    assert (naip.bbox[2] - naip.bbox[0]) < (naip_coarse.bbox[2] - naip_coarse.bbox[0])


def test_a_cap_widened_bands_width_is_the_cap_not_the_operators_margin():
    """A cap-bound band's width is decided by the mosaic cap, not by
    ``focal_margin_m`` -- in BOTH directions.

    This is deliberately the opposite of what a margin normally does, and it
    replaces an earlier test that asserted "a larger margin always wins over
    the floor -- never a ceiling that clips an operator's own wider margin."
    That premise is physically wrong: a band wider than
    ``_OPENING_MOSAIC_MAX_WIDTH_PX`` at its own finest zoom has that whole
    zoom level DROPPED by ``satellite._complete_level``, so honouring the
    wider margin does not buy the operator more imagery -- it costs them the
    entire level. The band is therefore pinned to the cap regardless of
    margin. The cost is real and bounded: focal imagery past the cap window
    resolves at the next band's zoom instead (z14, ~8.6 m/px, against z15's
    ~2 m/px), and only for routes or margins wide enough to reach it.
    """
    route = (-83.62, 35.18, -83.58, 35.21)  # narrow enough that the cap has room

    tight = rsb.plan_imagery_bands(route, focal_margin_m=0.0)
    generous = rsb.plan_imagery_bands(route, focal_margin_m=50_000.0)

    for source in rsb._CAP_WIDENED_SOURCES:
        tight_band = next(b for b in tight if b.source == source)
        generous_band = next(b for b in generous if b.source == source)
        max_zoom = tight_band.zooms.stop - 1
        cap_width = rsb._OPENING_MOSAIC_MAX_WIDTH_PX / rsb._px_per_degree_lon(max_zoom)

        assert (tight_band.bbox[2] - tight_band.bbox[0]) == pytest.approx(cap_width, rel=1e-9)
        assert (generous_band.bbox[2] - generous_band.bbox[0]) == pytest.approx(cap_width, rel=1e-9)


@pytest.mark.parametrize(
    "route_bbox",
    [
        FOCAL_ROUTE_BBOX,
        (10.0, -0.05, 10.08, 0.02),  # near the equator
        (-120.0, 58.0, -119.9, 58.05),  # high latitude
        (179.9, 10.0, 179.95, 10.05),  # near the antimeridian
    ],
)
def test_every_band_zoom_is_fully_covered_by_its_own_bbox(route_bbox):
    """The bug this fixes: a bbox too small to fully cover a coarse tile makes
    `write_tile_pyramid` silently skip that zoom. Every zoom in every planned
    band must be fully covered by that band's own bbox."""
    bands = rsb.plan_imagery_bands(route_bbox, focal_margin_m=1500.0)

    for band in bands:
        assert _uncovered_tiles(band) == [], (
            f"{band.source} band has uncovered tiles at zooms {list(band.zooms)}"
        )


def test_focal_band_is_not_blown_up_to_the_regional_bbox_size():
    """A 6.6 km route's focal band must stay meaningfully smaller than the
    regional band's tens-of-km size (the original bug's footprint ballooned
    an unsnapped focal raster to the *regional* band's whole size).

    The focal band's floor grew from ~13 km to ~36 km wide when
    `_sharp_handoff_focal_margin_m` started widening it to the span
    `naip_coarse`'s own finest zoom can no longer resolve sharply -- that is
    a real, deliberate cost (more/bigger NAIP fetches) for a real defect (the
    opening camera stuck on an upscaled `naip_coarse` mosaic for the last
    third of its descent); this still bounds it well under the regional
    band's ~150 km (default 75 km margin each side)."""
    bands = rsb.plan_imagery_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)
    focal = next(b for b in bands if b.source == "naip")
    regional = next(b for b in bands if b.source == "naip_coarse")

    focal_west, focal_south, focal_east, focal_north = focal.bbox
    width_km = (focal_east - focal_west) * 111.32 * 0.819  # cos(35 deg)
    height_km = (focal_north - focal_south) * 111.32
    regional_width_km = (regional.bbox[2] - regional.bbox[0]) * 111.32 * 0.819

    assert width_km < 45.0
    assert height_km < 45.0
    assert width_km < regional_width_km / 2.0


def test_bands_are_ordered_national_to_focal_by_decreasing_bbox_area():
    bands = rsb.plan_imagery_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)

    def area(band: rsb.ZoomBand) -> float:
        west, south, east, north = band.bbox
        return (east - west) * (north - south)

    areas = [area(b) for b in bands]
    assert areas[0] > areas[1] > areas[2] > areas[3]


def test_cap_widened_bands_bbox_matches_cap_over_maxzoom_px_per_deg():
    """The core ladder property: each of naip_coarse/naip_mid/naip's bbox
    width is `_OPENING_MOSAIC_MAX_WIDTH_PX / px_per_deg(band's own max zoom)`
    -- derived from the cap, never hardcoded -- within one tile-snap's worth
    of slack at the band's own minimum zoom."""
    bands = rsb.plan_imagery_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)

    for source in ("naip_coarse", "naip_mid", "naip"):
        band = next(b for b in bands if b.source == source)
        expected_width = rsb._OPENING_MOSAIC_MAX_WIDTH_PX / (
            256 * 2 ** (band.zooms.stop - 1) / 360.0
        )
        actual_width = band.bbox[2] - band.bbox[0]
        tile_at_minzoom = 360.0 / 2**band.zooms.start
        assert actual_width == pytest.approx(expected_width, abs=tile_at_minzoom + 1e-6)


@pytest.mark.parametrize("focal_margin_m", [500.0, 1500.0, 4000.0, 12000.0])
@pytest.mark.parametrize("route_width_deg", [0.02, 0.15, 0.4])
def test_no_cap_widened_band_ever_exceeds_the_opening_mosaic_cap(route_width_deg, focal_margin_m):
    """A band whose finest zoom needs MORE tile columns than the opening
    mosaic cap allows has that whole zoom level dropped on disk
    (`satellite._complete_level`) -- so overshooting the cap does not merely
    waste a fetch, it deletes the exact handoff level this ladder exists to
    provide, reopening the "dark and fuzzy when zooming in" window.

    The cap is therefore a two-sided constraint, not a floor. Widening a
    snapped bbox toward the cap and allowing "within one tile-snap's slack"
    permits a 9th column where 8 fit, which at the band's finest zoom is
    9216 px against an 8192 px ceiling -- as seen on a real route
    bbox at the service's own default focal margin.
    """
    west, south = -83.62, 35.18
    route = (west, south, west + route_width_deg, south + route_width_deg * 0.7)
    bands = rsb.plan_imagery_bands(route, focal_margin_m=focal_margin_m)

    for source in rsb._CAP_WIDENED_SOURCES:
        band = next(b for b in bands if b.source == source)
        max_zoom = band.zooms.stop - 1
        mosaic_px = (band.bbox[2] - band.bbox[0]) * rsb._px_per_degree_lon(max_zoom)
        # The one thing the cap never wins against: the route itself. A route
        # too wide for this zoom keeps its own columns and accepts the drop --
        # no bbox can serve it here, and a coarser band is what covers it.
        route_px = route_width_deg * rsb._px_per_degree_lon(max_zoom)
        if route_px > rsb._OPENING_MOSAIC_MAX_WIDTH_PX:
            continue
        assert mosaic_px <= rsb._OPENING_MOSAIC_MAX_WIDTH_PX + 1e-6, (
            f"{source} z{max_zoom} mosaic is {mosaic_px:.0f}px, over the "
            f"{rsb._OPENING_MOSAIC_MAX_WIDTH_PX}px cap -- that zoom level would "
            "be dropped from the bundle entirely"
        )


def test_naip_coarse_and_naip_mid_cover_where_the_finer_bands_own_floor_is_hit():
    """No coverage gap between consecutive cap-widened bands: at the span
    where a band's own finest zoom first drops below the opening's sharpness
    floor, the NEXT (coarser) band's bbox must still fully cover it."""
    bands = rsb.plan_imagery_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)
    by_source = {b.source: b for b in bands}
    floor = 0.5
    width_px = 1920

    for coarser, finer in (("naip_coarse", "naip_mid"), ("naip_mid", "naip")):
        coarser_band = by_source[coarser]
        coarser_max_zoom = coarser_band.zooms.stop - 1
        px_per_deg = 256 * 2**coarser_max_zoom / 360.0
        narrowest_span_coarser_still_resolves = floor * width_px / px_per_deg
        finer_band = by_source[finer]
        finer_width = finer_band.bbox[2] - finer_band.bbox[0]
        assert finer_width >= narrowest_span_coarser_still_resolves


def test_plan_imagery_bands_rejects_degenerate_bbox():
    with pytest.raises(ValueError):
        rsb.plan_imagery_bands((-83.5, 35.2, -83.5, 35.3), focal_margin_m=1500.0)


# --- band_raster_size --------------------------------------------------------


def test_band_request_caps_are_the_fetchers_own_tile_caps():
    """The planner sizes band requests with the same measured per-source
    caps the fetchers tile at, so the two can never disagree."""
    from gomoveshift.pipeline.route_story import sources

    assert rsb.NAIP_MAX_REQUEST_PX == sources.NAIP_MAX_TILE_PX
    assert rsb._MAX_REQUEST_PX["naip"] == sources.NAIP_MAX_TILE_PX
    assert rsb._MAX_REQUEST_PX["naip_coarse"] == sources.NAIP_MAX_TILE_PX
    assert rsb._MAX_REQUEST_PX["terrain_regional"] == sources.DEM_3DEP_MAX_TILE_PX
    assert rsb._MAX_REQUEST_PX["terrain_focal"] == sources.DEM_3DEP_MAX_TILE_PX


def test_band_raster_size_within_cap_returns_a_plain_size():
    # A direct small ZoomBand, not one from plan_imagery_bands: the focal
    # ("naip") band it would return is now always widened past the NAIP
    # cap (see test_focal_band_is_not_blown_up_to_the_regional_bbox_size),
    # so this exercises band_raster_size's plain-tuple branch directly.
    small_band = rsb.ZoomBand(
        kind="imagery",
        zooms=range(13, 16),
        source="naip",
        bbox=(-83.52, 35.23, -83.515, 35.235),
        target_m_per_px=2.0,
    )

    size = rsb.band_raster_size(small_band)

    assert isinstance(size, tuple)
    width_px, height_px = size
    assert 0 < width_px <= rsb.NAIP_MAX_REQUEST_PX
    assert 0 < height_px <= rsb.NAIP_MAX_REQUEST_PX


def test_band_raster_size_over_cap_returns_tiled_requests_covering_the_bbox():
    bands = rsb.plan_imagery_bands(
        FOCAL_ROUTE_BBOX, focal_margin_m=1500.0, national_margin_km=4000.0
    )
    national = next(b for b in bands if b.source == "gibs")

    requests = rsb.band_raster_size(national)

    assert isinstance(requests, list)
    assert len(requests) > 1
    for request in requests:
        assert isinstance(request, rsb.RasterRequest)
        assert request.width_px <= rsb.GIBS_MAX_REQUEST_PX
        assert request.height_px <= rsb.GIBS_MAX_REQUEST_PX

    # The sub-requests' union reaches every corner of the band's bbox.
    west, south, east, north = national.bbox
    assert min(r.bbox[0] for r in requests) == pytest.approx(west)
    assert max(r.bbox[2] for r in requests) == pytest.approx(east)
    assert min(r.bbox[1] for r in requests) == pytest.approx(south)
    assert max(r.bbox[3] for r in requests) == pytest.approx(north)


# --- band_pixel_size ---------------------------------------------------------


def test_band_pixel_size_matches_band_raster_size_within_cap():
    """Under the request-size cap, the raw and cap-aware sizers agree exactly."""
    small_band = rsb.ZoomBand(
        kind="imagery",
        zooms=range(13, 16),
        source="naip",
        bbox=(-83.52, 35.23, -83.515, 35.235),
        target_m_per_px=2.0,
    )

    assert rsb.band_pixel_size(small_band) == rsb.band_raster_size(small_band)


def test_band_pixel_size_never_tiles_even_over_the_cap():
    """Unlike `band_raster_size`, this never mosaics — it always returns one size."""
    bands = rsb.plan_imagery_bands(
        FOCAL_ROUTE_BBOX, focal_margin_m=1500.0, national_margin_km=4000.0
    )
    national = next(b for b in bands if b.source == "gibs")

    width_px, height_px = rsb.band_pixel_size(national)

    assert width_px > rsb.GIBS_MAX_REQUEST_PX or height_px > rsb.GIBS_MAX_REQUEST_PX
    assert isinstance(rsb.band_raster_size(national), list)  # confirms this band tiles


# --- bands_receipt ------------------------------------------------------------


def test_bands_receipt_shape():
    bands = rsb.plan_imagery_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)

    receipt = rsb.bands_receipt(bands)

    assert len(receipt) == 6
    for entry, band in zip(receipt, bands):
        assert entry["kind"] == "imagery"
        assert entry["source"] == band.source
        assert entry["zooms"] == [band.zooms.start, band.zooms.stop - 1]
        assert entry["bbox"] == list(band.bbox)
        assert entry["target_m_per_px"] == band.target_m_per_px
        assert "raster_px" in entry


# --- plan_terrain_bands -------------------------------------------------------


def test_plan_terrain_bands_returns_regional_and_focal():
    bands = rsb.plan_terrain_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)

    assert [b.source for b in bands] == ["terrain_regional", "terrain_focal"]
    assert [b.kind for b in bands] == ["terrain", "terrain"]
    assert list(bands[0].zooms) == list(range(9, 13))
    assert list(bands[1].zooms) == [13]


def test_terrain_regional_bbox_matches_imagery_naip_coarse_bbox():
    """z9-12 imagery has no terrain under it unless the two bands share the
    exact same footprint (coarse terrain zooms otherwise silently write
    little or nothing over a focal-sized DEM raster)."""
    imagery_bands = rsb.plan_imagery_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)
    terrain_bands = rsb.plan_terrain_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)

    naip_coarse = next(b for b in imagery_bands if b.source == "naip_coarse")
    terrain_regional = next(b for b in terrain_bands if b.source == "terrain_regional")

    assert terrain_regional.bbox == naip_coarse.bbox


def test_terrain_focal_bbox_matches_the_imagery_band_holding_z13():
    """z13 (terrain_focal's only zoom) sits inside naip_fine's z13-14 range --
    terrain_focal has no relief under it unless it shares that band's snapped
    footprint. An unsnapped focal DEM over a route narrower than one z13 tile
    fully covers no z13 tile, writes nothing, and fails bundle verification.

    The pairing is asserted by FINDING the band whose range holds z13 rather
    than by naming it: it has already moved once (naip -> naip_mid -> naip_fine)
    as the ladder resized, and each move silently broke this guarantee until
    the test was edited to match. Found, it cannot."""
    imagery_bands = rsb.plan_imagery_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)
    terrain_bands = rsb.plan_terrain_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)

    terrain_focal = next(b for b in terrain_bands if b.source == "terrain_focal")
    z13 = terrain_focal.zooms.start
    holder = next(b for b in imagery_bands if z13 in b.zooms)

    assert holder.source == "naip_fine"
    assert terrain_focal.bbox == holder.bbox


def test_terrain_focal_band_fully_covers_a_z13_tile_for_a_sub_tile_route():
    """A sub-tile route footprint is ~0.0442 deg wide, a hair wider than one
    z13 tile (~0.0439 deg), and straddles two tile columns: unsnapped it
    fully covered none of its 6 candidate tiles."""
    sub_tile_footprint = (
        -83.58,
        35.205,
        -83.5358,
        35.265,
    )
    bands = rsb.plan_terrain_bands(sub_tile_footprint, focal_margin_m=0.0)
    focal = next(b for b in bands if b.source == "terrain_focal")

    assert list(tiles_covering(Footprint(*focal.bbox), 13))
    assert _uncovered_tiles(focal) == []


def test_terrain_regional_target_m_per_px_is_coarser_than_focal():
    bands = rsb.plan_terrain_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)
    regional = next(b for b in bands if b.source == "terrain_regional")
    focal = next(b for b in bands if b.source == "terrain_focal")

    assert focal.target_m_per_px == 10.0
    assert 20.0 < regional.target_m_per_px < 40.0


def test_terrain_regional_band_zooms_are_fully_covered_by_its_own_bbox():
    """The regional band is snapped like an imagery band, so every z9-12 tile
    it touches is fully inside it."""
    bands = rsb.plan_terrain_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)
    regional = next(b for b in bands if b.source == "terrain_regional")

    assert _uncovered_tiles(regional) == []


def test_plan_terrain_bands_rejects_degenerate_bbox():
    with pytest.raises(ValueError):
        rsb.plan_terrain_bands((-83.5, 35.2, -83.5, 35.3), focal_margin_m=1500.0)


def test_terrain_bands_receipt_shape():
    bands = rsb.plan_terrain_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)

    receipt = rsb.bands_receipt(bands)

    assert len(receipt) == 2
    for entry, band in zip(receipt, bands):
        assert entry["kind"] == "terrain"
        assert entry["source"] == band.source
        assert entry["zooms"] == [band.zooms.start, band.zooms.stop - 1]
        assert entry["bbox"] == list(band.bbox)


# --- the opening never hands off to something it must upscale ---------------

_OPENING_TARGET = RoutePoint(time_s=0.0, latitude=35.2287, longitude=-83.5478, elevation_m=900.0)


def _rasters_for_bands(tmp_path, bands: list[rsb.ZoomBand]) -> tuple[SatelliteRaster, ...]:
    """One synthetic raster per zoom in every band, exactly the shape
    `satellite.bundle_opening_rasters` emits from a real bundle -- fully
    resolved at its own band's ``target_m_per_px`` (equivalently, its own
    finest zoom's px/deg), never coarser."""
    rasters = []
    for band in bands:
        west, south, east, north = band.bbox
        for zoom in band.zooms:
            width_px = max(2, round((east - west) * rsb._px_per_degree_lon(zoom)))
            height_px = max(2, round(width_px * (north - south) / (east - west)))
            path = tmp_path / f"{band.source}-z{zoom}.jpg"
            Image.new("RGB", (width_px, height_px), "#3d6b52").save(path, quality=80)
            rasters.append(SatelliteRaster(path, west, south, east, north, band.source))
    return tuple(rasters)


def test_the_fixed_ladder_never_needs_an_upscale_across_the_whole_move(tmp_path):
    """The end-to-end proof this fix exists for: walk the whole opening move,
    fine-grained, against rasters shaped exactly like a real post-fix bundle
    (`plan_imagery_bands`' own widened bands, one raster per zoom) -- never
    hand off to something that needs upscaling past the sharpness floor."""
    bands = rsb.plan_imagery_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)
    rasters = _rasters_for_bands(tmp_path, bands)

    check_opening_sharpness(rasters, _OPENING_TARGET, 1920, 1080)


def test_a_pre_fix_four_band_layout_still_fails_the_walk(tmp_path):
    """Regression guard: a layout shaped like the OLD four-band plan --
    ``naip_coarse`` snapped at its own coarsest zoom with no cap-derived
    floor, and no ``naip_mid`` band at all -- must still trip
    `check_opening_sharpness`'s walk. Proves the walk itself (not just the
    fixed geometry) actually discriminates the two windows the current plan closes."""
    pre_fix_bands = [
        rsb.ZoomBand(
            kind="imagery",
            zooms=range(2, 5),
            source="blue_marble",
            bbox=(-180.0, -85.0, 180.0, 85.0),
            target_m_per_px=0.0,
        ),
        rsb.ZoomBand(
            kind="imagery",
            zooms=range(5, 9),
            source="gibs",
            bbox=(-101.25, 19.475, -65.75, 50.71),  # ~ national margin, snapped at z5
            target_m_per_px=0.0,
        ),
        rsb.ZoomBand(
            kind="imagery",
            zooms=range(9, 16),
            source="naip",
            # A pre-fix, focal-sized bbox at once-shared naip_coarse/naip
            # zooms 9-15 -- far too narrow to cover the gibs-z8 handoff span.
            bbox=(-83.6484375, 34.9580, -83.1445, 35.5322),
            target_m_per_px=0.0,
        ),
    ]
    rasters = _rasters_for_bands(tmp_path, pre_fix_bands)

    with pytest.raises(ValueError, match="upscaled to fill the frame"):
        check_opening_sharpness(rasters, _OPENING_TARGET, 1920, 1080)


def _servable_span_range(band: rsb.ZoomBand) -> tuple[float, float] | None:
    """The span range, in degrees of longitude, ``band`` can fill sharply.

    Ceiling: the bbox itself -- ``_viewport_raster`` needs a raster to CONTAIN
    the whole viewport, so a viewport wider than the bbox falls back a band.
    Floor: the span at which the band's finest usable zoom stops supplying one
    source pixel per output pixel. A zoom whose mosaic would exceed
    ``_OPENING_MOSAIC_MAX_WIDTH_PX`` is not usable -- ``satellite._complete_level``
    drops it from the bundle outright.
    """
    width_deg = band.bbox[2] - band.bbox[0]
    usable = [
        zoom
        for zoom in band.zooms
        if width_deg * rsb._px_per_degree_lon(zoom) <= rsb._OPENING_MOSAIC_MAX_WIDTH_PX
    ]
    if not usable:
        return None
    px_per_deg = rsb._px_per_degree_lon(max(usable))
    return rsb._ASSUMED_OPENING_WIDTH_PX / px_per_deg, width_deg


def test_the_planned_bands_chain_with_no_uncovered_span():
    """Every span the opening camera passes through has a band that resolves it.

    An earlier render was rejected for being "very dark and fuzzy when zooming in". The
    cause was not a bad raster -- it was the absence of one. Each band's bbox
    came from its own margin, never from where the previous band's resolution
    ran out, so two spans had no sharp raster at all and `_viewport_raster`
    fell back a whole band and upscaled. The two uncovered spans were
    5.63-10.55 deg and 0.70-1.32 deg.
    """
    bands = rsb.plan_imagery_bands(FOCAL_ROUTE_BBOX, focal_margin_m=1500.0)
    ranges = [(band, _servable_span_range(band)) for band in bands]
    served = [(band, span) for band, span in ranges if span is not None]
    served.sort(key=lambda pair: pair[1][1], reverse=True)
    # `blue_marble` is the shared global backdrop, not a rung of the descent.
    served = [pair for pair in served if pair[0].source != "blue_marble"]
    gaps = []
    for (coarse, (coarse_floor, _)), (fine, (_, fine_ceiling)) in zip(served, served[1:]):
        if fine_ceiling < coarse_floor:
            gaps.append(
                f"{coarse.source} stops resolving at {coarse_floor:.4f} deg but "
                f"{fine.source} only covers {fine_ceiling:.4f} deg -- "
                f"{fine_ceiling:.4f}-{coarse_floor:.4f} deg has no sharp raster"
            )
    assert not gaps, "; ".join(gaps)


def test_consecutive_bands_are_at_most_two_zoom_levels_apart():
    """The chain rule, checked on the ladder itself rather than on one route.

    A band serves from its bbox down to `_ASSUMED_OPENING_WIDTH_PX` over its
    finest zoom's pixel density, and a cap-widened bbox is
    `_OPENING_MOSAIC_MAX_WIDTH_PX` over that same density -- so one band
    stretches exactly `_OPENING_MOSAIC_MAX_WIDTH_PX / _ASSUMED_OPENING_WIDTH_PX`
    in span, whatever the route. Doubling the pixel density per zoom, that is
    log2 of the ratio zoom levels. Step the finest zooms further apart than
    that and a gap opens no margin can close -- which is how the two gaps
    got in with every individual band looking correctly sized.

    This test is the reason a future `_BAND_SPECS` edit cannot reopen them: it
    needs no bundle, no route and no fetch to fail.

    `blue_marble` is excluded, and measurement is why rather than taste: the
    opening camera's widest rung is `gibs`'s own bbox (33.75 deg on a real
    bundle -- `_viewport_raster` cannot open wider than the raster it
    starts on), while `blue_marble` resolves only 168.75-180 deg. The step
    between them is never traversed; it is a backdrop under `gibs`, not a rung.
    """
    max_step = math.log2(rsb._OPENING_MOSAIC_MAX_WIDTH_PX / rsb._ASSUMED_OPENING_WIDTH_PX)
    rungs = [(source, zooms) for source, zooms in rsb._BAND_SPECS if source != "blue_marble"]
    finest = [zooms.stop - 1 for _, zooms in rungs]
    too_far = [
        f"{rungs[i][0]}(z{finest[i]}) -> "
        f"{rungs[i + 1][0]}(z{finest[i + 1]}) steps {finest[i + 1] - finest[i]}"
        for i in range(len(finest) - 1)
        if finest[i + 1] - finest[i] > max_step
    ]
    assert not too_far, f"max step is {max_step:.2f} zoom levels; " + "; ".join(too_far)
