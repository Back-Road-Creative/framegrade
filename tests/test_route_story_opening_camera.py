"""The opening's country-to-trailhead camera: framing, travel, sharpness and
grade -- the "doesn't look like the USA / dark and fuzzy zoomed in" defect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import layers
from gomoveshift.pipeline.route_story.layers import (
    _opening_camera,
    _render_opening_map,
    check_opening_sharpness,
    opening_credit,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster

TARGET = RoutePoint(time_s=0.0, latitude=35.23, longitude=-83.56, elevation_m=900.0)


def _raster(tmp_path: Path, name: str, bounds, credit: str, width: int, *, texture=False):
    path = tmp_path / f"{name}.jpg"
    height = max(1, width // 2)
    if texture:
        rng = np.random.default_rng(hash(name) % (2**32))
        pixels = rng.integers(40, 220, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels, "RGB").save(path, quality=95)
    else:
        Image.new("RGB", (width, height), "#3d6b52").save(path)
    west, south, east, north = bounds
    return SatelliteRaster(path, west, south, east, north, credit)


WIDTH, HEIGHT = 640, 360


def _bundle_like_rasters(tmp_path: Path) -> tuple[SatelliteRaster, ...]:
    """Mirrors a real bundle's shape: a coarse whole-hemisphere band too
    small to fill the frame sharply, a sharper band covering a useful slice
    of the country, a mid band, and a tight local band -- each resolved
    generously enough that the handoffs between them (unlike a real bundle's
    own, which are not tuned this way) never gap below the sharpness floor."""
    coarse = _raster(tmp_path, "blue_marble", (-180.0, 0.0, -0.0, 66.5), "Coarse world band", 512)
    sharp = _raster(
        tmp_path, "gibs", (-101.25, 21.94, -67.5, 48.92), "Sharp country band", 4000, texture=True
    )
    mid = _raster(
        tmp_path, "naip_coarse", (-85.08, 34.31, -82.27, 36.03), "Mid band", 8000, texture=True
    )
    local = _raster(
        tmp_path, "naip", (-83.63, 35.17, -83.50, 35.28), "Local band", 1000, texture=True
    )
    return coarse, sharp, mid, local


# --- country framing --------------------------------------------------------


def test_the_opening_starts_on_the_sharpest_band_that_can_fill_the_frame(tmp_path):
    rasters = _bundle_like_rasters(tmp_path)

    assert opening_credit(rasters, TARGET, 0.0, WIDTH, HEIGHT) == "Sharp country band"


def test_a_coarse_whole_world_band_alone_is_still_used_as_a_last_resort(tmp_path):
    # Nothing else to fall back to: the country framing still has to pick
    # SOMETHING rather than raise -- check_opening_sharpness is what raises.
    coarse = _raster(tmp_path, "blue_marble", (-180.0, 0.0, -0.0, 66.5), "Only band", 2048)

    assert opening_credit((coarse,), TARGET, 0.0, WIDTH, HEIGHT) == "Only band"


# --- travel: centre moves, then locks --------------------------------------


def test_the_centre_travels_from_the_country_framing_toward_the_target(tmp_path):
    rasters = _bundle_like_rasters(tmp_path)

    start_lat, start_lon, _ = _opening_camera(rasters, TARGET, 0.0, WIDTH)
    mid_lat, mid_lon, _ = _opening_camera(rasters, TARGET, 0.3, WIDTH)

    assert (start_lat, start_lon) != (TARGET.latitude, TARGET.longitude)
    # Monotonic progress toward the target on both axes.
    assert abs(mid_lat - TARGET.latitude) < abs(start_lat - TARGET.latitude)
    assert abs(mid_lon - TARGET.longitude) < abs(start_lon - TARGET.longitude)


def test_the_last_third_of_the_move_has_no_lateral_drift(tmp_path):
    rasters = _bundle_like_rasters(tmp_path)

    for fraction in (2 / 3, 0.8, 0.9, 1.0):
        latitude, longitude, span = _opening_camera(rasters, TARGET, fraction, WIDTH)
        assert (latitude, longitude) == (TARGET.latitude, TARGET.longitude)
        assert span > 0


def test_the_opening_ends_exactly_where_it_ended_before(tmp_path):
    rasters = _bundle_like_rasters(tmp_path)

    _, _, span = _opening_camera(rasters, TARGET, 1.0, WIDTH)

    assert span == pytest.approx(0.085)


# --- never a silent upscale --------------------------------------------------


def test_check_opening_sharpness_passes_a_well_resolved_raster_set(tmp_path):
    rasters = _bundle_like_rasters(tmp_path)

    check_opening_sharpness(rasters, TARGET, WIDTH, HEIGHT)


def test_check_opening_sharpness_raises_naming_the_span_and_raster(tmp_path):
    # A single, far too coarse raster: no span in the move can be resolved
    # without a silent upscale.
    tiny = _raster(tmp_path, "tiny", (-180.0, -90.0, 180.0, 90.0), "Thin Credit", 64)

    with pytest.raises(ValueError, match=r"span of .*°.*Thin Credit"):
        check_opening_sharpness((tiny,), TARGET, WIDTH, HEIGHT)


# --- vibrant: graded once, cached, deterministic ----------------------------


def test_the_opening_composite_is_more_colourful_than_the_ungraded_source(tmp_path):
    # A flat, muted, low-saturation raster -- like a hazy/desaturated
    # basemap crop -- should read more colourful once graded.
    dull = tmp_path / "dull.jpg"
    Image.new("RGB", (256, 128), (140, 150, 160)).save(dull)
    raster = SatelliteRaster(dull, -84.0, 35.0, -83.0, 35.5, "Dull Agency")

    frame = _render_opening_map((raster,), TARGET, 1.0, 200, 112)
    ungraded = layers._raster_image(raster.source)

    from gomoveshift.pipeline.route_story.grade import measure

    assert measure(frame)["colourfulness"] > measure(ungraded)["colourfulness"]


def test_a_raster_is_graded_once_and_the_grade_is_cached(tmp_path, monkeypatch):
    dull = tmp_path / "dull.jpg"
    Image.new("RGB", (256, 128), (140, 150, 160)).save(dull)
    raster = SatelliteRaster(dull, -84.0, 35.0, -83.0, 35.5, "Dull Agency")
    layers._graded_raster_image.cache_clear()

    calls = []
    real_grade_still = layers.grade.grade_still

    def counting_grade_still(image, **kwargs):
        calls.append(1)
        return real_grade_still(image, **kwargs)

    monkeypatch.setattr(layers.grade, "grade_still", counting_grade_still)

    for fraction in (0.2, 0.4, 0.6, 1.0):
        _render_opening_map((raster,), TARGET, fraction, 200, 112)

    assert len(calls) == 1


def test_the_opening_never_needs_an_upscale_across_a_real_band_layout(tmp_path):
    """The end-to-end proof of the fix: walk the whole move against
    rasters sized exactly like a real, post-fix bundle -- `plan_imagery_bands`'
    own widened focal bbox, and one raster per zoom in every band
    (`bundle_opening_rasters`'s coarse-to-fine emission) -- and never need a
    raster upscaled past the floor. `check_opening_sharpness` raising here
    is exactly what would abort a real render at preflight."""
    from gomoveshift.pipeline.route_story.bands import _px_per_degree_lon, plan_imagery_bands

    route_bbox = (-83.55, 35.20, -83.48, 35.26)
    target = RoutePoint(time_s=0.0, latitude=35.23, longitude=-83.515, elevation_m=900.0)
    rasters = []
    for band in plan_imagery_bands(route_bbox, focal_margin_m=1500.0):
        west, south, east, north = band.bbox
        for zoom in band.zooms:
            width_px = max(2, round((east - west) * _px_per_degree_lon(zoom)))
            height_px = max(2, round(width_px * (north - south) / (east - west)))
            path = tmp_path / f"{band.source}-z{zoom}.jpg"
            Image.new("RGB", (width_px, height_px), "#3d6b52").save(path, quality=80)
            rasters.append(SatelliteRaster(path, west, south, east, north, band.source))

    check_opening_sharpness(tuple(rasters), target, 1920, 1080)


def test_the_grade_is_deterministic(tmp_path):
    dull = tmp_path / "dull.jpg"
    Image.new("RGB", (256, 128), (140, 150, 160)).save(dull)
    raster = SatelliteRaster(dull, -84.0, 35.0, -83.0, 35.5, "Dull Agency")

    first = _render_opening_map((raster,), TARGET, 1.0, 200, 112)
    second = _render_opening_map((raster,), TARGET, 1.0, 200, 112)

    assert list(first.getdata()) == list(second.getdata())


def test_the_country_framing_floor_can_never_pick_a_raster_the_sharpness_gate_rejects(tmp_path):
    """`_country_framing` and `check_opening_sharpness` read two different
    constants -- `_OPENING_COUNTRY_FRAMING_MIN_RATIO` (1.25) and
    `_OPENING_MIN_SOURCE_PX_PER_OUTPUT_PX` (0.5) -- against the same imagery.
    Nothing in the code ties them together, so lowering the framing floor
    below the sharpness floor would let the camera OPEN on a raster its own
    first frame then fails on.

    That is not hypothetical slack: on the real five-band ladder the widest
    band (blue_marble, z4 over a 180 deg bbox) resolves 2048px against a
    1920px frame -- a ratio of 1.07. It clears the 0.5 sharpness floor
    comfortably and is excluded only by the 1.25 framing floor. Drop the
    framing floor to 1.0 and the opening starts on a whole-world backdrop
    at a span no finer band covers.

    So the invariant is the ordering itself, asserted directly rather than
    by restating either number.
    """
    assert (
        layers._OPENING_COUNTRY_FRAMING_MIN_RATIO >= layers._OPENING_MIN_SOURCE_PX_PER_OUTPUT_PX
    ), (
        "the country-framing floor must be at least the sharpness floor, or "
        "_country_framing can pick a raster check_opening_sharpness rejects"
    )

    # And it holds end to end: whatever framing the real ladder's shape
    # yields, the gate passes at the very first frame it produces.
    rasters = _bundle_like_rasters(tmp_path)
    check_opening_sharpness(rasters, TARGET, WIDTH, HEIGHT, fractions=(0.0,))
