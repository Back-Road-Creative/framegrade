"""The Pillow map view must show a raster the way it is stored: north up, east right."""

from __future__ import annotations

from PIL import Image

from gomoveshift.pipeline.route_story.layers import _render_geographic_rasters
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster

NW, NE, SW, SE = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)


def _quadrant_raster(tmp_path) -> SatelliteRaster:
    image = Image.new("RGB", (400, 200))
    image.paste(NW, (0, 0, 200, 100))
    image.paste(NE, (200, 0, 400, 100))
    image.paste(SW, (0, 100, 200, 200))
    image.paste(SE, (200, 100, 400, 200))
    source = tmp_path / "quadrants.png"
    image.save(source)
    return SatelliteRaster(source, -100.0, 30.0, -80.0, 40.0, "test quadrants")


def test_the_view_keeps_north_up_and_east_right(tmp_path):
    raster = _quadrant_raster(tmp_path)

    view = _render_geographic_rasters([raster], 35.0, -90.0, 16.0, 320, 160)

    assert view.size == (320, 160)
    assert view.getpixel((20, 20)) == NW
    assert view.getpixel((300, 20)) == NE, "east must be on the right, not below"
    assert view.getpixel((20, 140)) == SW, "south must be below, not on the right"
    assert view.getpixel((300, 140)) == SE


def test_an_off_centre_view_shows_the_ground_under_it(tmp_path):
    raster = _quadrant_raster(tmp_path)

    # Entirely inside the north-east quadrant.
    view = _render_geographic_rasters([raster], 38.0, -84.0, 4.0, 200, 100)

    assert {view.getpixel(xy) for xy in ((5, 5), (195, 5), (5, 95), (195, 95))} == {NE}
