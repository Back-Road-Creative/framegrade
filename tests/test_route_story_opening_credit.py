"""The opening credits the imagery on screen, not always the first raster."""

from pathlib import Path

from PIL import Image

from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story.layers import opening_credit
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster


def _raster(tmp_path: Path, name: str, bounds, credit: str) -> SatelliteRaster:
    path = tmp_path / f"{name}.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(path)
    west, south, east, north = bounds
    return SatelliteRaster(path, west, south, east, north, credit)


def _rasters(tmp_path: Path):
    return (
        _raster(tmp_path, "national", (-130.0, 20.0, -60.0, 55.0), "National agency"),
        _raster(tmp_path, "local", (-83.62, 35.2, -83.37, 35.31), "Local agency"),
    )


_START = RoutePoint(latitude=35.25, longitude=-83.5, elevation_m=900.0, time_s=0.0)


def test_the_wide_opening_credits_the_national_imagery(tmp_path):
    assert opening_credit(_rasters(tmp_path), _START, 0.0, 1920, 1080) == "National agency"


def test_the_arrival_credits_the_local_imagery_it_shows(tmp_path):
    assert opening_credit(_rasters(tmp_path), _START, 1.0, 1920, 1080) == "Local agency"


def test_the_credit_names_the_same_raster_the_frame_is_drawn_from(tmp_path, monkeypatch):
    # Recording ``_raster_image`` calls (as before) no longer isolates the
    # raster actually composited: the new country framing (``_country_framing``)
    # legitimately inspects every raster's own pixel width to choose the
    # opening's establishing shot, not just the one the frame ends up drawn
    # from. ``_viewport_raster`` is the one call both the render and the
    # credit make for their own final, single choice of raster.
    from gomoveshift.pipeline.route_story import layers

    rasters = _rasters(tmp_path)
    drawn = []
    real = layers._viewport_raster

    def recording(rasters_arg, west, east, south, north):
        chosen = real(rasters_arg, west, east, south, north)
        drawn.append(chosen.source)
        return chosen

    monkeypatch.setattr(layers, "_viewport_raster", recording)

    for fraction in (0.0, 0.5, 0.8, 1.0):
        drawn.clear()
        layers._render_opening_map(rasters, _START, fraction, 320, 180)
        shown = next(r for r in rasters if r.source == drawn[-1])
        assert opening_credit(rasters, _START, fraction, 320, 180) == shown.credit


def test_the_film_opening_draws_the_credit_of_the_raster_on_screen(tmp_path, monkeypatch):
    credits = []
    monkeypatch.setattr(
        film_module, "_draw_opening", lambda image, fraction, credit: credits.append(credit)
    )

    film_module._opening_frame(_rasters(tmp_path), _START, 0.0, 320, 180, ())
    film_module._opening_frame(_rasters(tmp_path), _START, 1.0, 320, 180, ())

    assert credits == ["National agency", "Local agency"]
