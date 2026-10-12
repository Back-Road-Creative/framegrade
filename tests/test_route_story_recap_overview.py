"""The recap map shows the whole day on the sharpest imagery, not the last corner of it."""

from __future__ import annotations

import numpy as np
from PIL import Image

from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story.layers import _render_map, overview_viewport
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster

COARSE, FINE = (200, 30, 30), (30, 160, 60)


def _rasters(tmp_path) -> list[SatelliteRaster]:
    coarse, fine = tmp_path / "coarse.png", tmp_path / "fine.png"
    Image.new("RGB", (64, 64), COARSE).save(coarse)
    Image.new("RGB", (64, 64), FINE).save(fine)
    return [
        SatelliteRaster(coarse, -86.0, 33.0, -80.0, 37.5, "coarse"),
        SatelliteRaster(fine, -83.62, 35.20, -83.37, 35.31, "fine"),
    ]


def _day() -> list[list[RoutePoint]]:
    # A route that starts on the east edge and ends low in the south-west.
    return [
        [
            RoutePoint(0, 35.274, -83.3995, 1300),
            RoutePoint(3000, 35.2878, -83.47, 1500),
            RoutePoint(9000, 35.24, -83.597, 1550),
            RoutePoint(13000, 35.2216, -83.5629, 1400),
        ]
    ]


def _view_bounds(center_lat, center_lon, span_lon, width, map_height):
    span_lat = span_lon * map_height / width
    return (
        center_lon - span_lon / 2,
        center_lat - span_lat / 2,
        center_lon + span_lon / 2,
        center_lat + span_lat / 2,
    )


def test_the_overview_holds_every_route_point(tmp_path):
    west, south, east, north = _view_bounds(
        *overview_viewport(_day(), _rasters(tmp_path), 1280, 615), 1280, 615
    )

    for point in _day()[0]:
        assert west < point.longitude < east
        assert south < point.latitude < north


def test_the_overview_stays_inside_the_finest_raster_when_the_route_fits_it(tmp_path):
    rasters = _rasters(tmp_path)
    fine = rasters[-1]

    west, south, east, north = _view_bounds(
        *overview_viewport(_day(), rasters, 1280, 615), 1280, 615
    )

    assert fine.west <= west and east <= fine.east
    assert fine.south <= south and north <= fine.north


def test_a_route_larger_than_the_finest_raster_is_still_shown_whole(tmp_path):
    rasters = _rasters(tmp_path)
    long_day = [[RoutePoint(0, 35.0, -84.5, 300), RoutePoint(9000, 35.4, -83.0, 400)]]

    west, south, east, north = _view_bounds(
        *overview_viewport(long_day, rasters, 1280, 615), 1280, 615
    )

    assert west < -84.5 and east > -83.0 and south < 35.0 and north > 35.4


def test_the_recap_map_is_drawn_from_the_fine_raster(tmp_path):
    image = _render_map(
        _rasters(tmp_path), _day(), 13000, 320, 180, show_dashboard=False, overview=True
    )

    corner = np.asarray(image)[4, 4].astype(int)
    assert abs(corner - np.array(FINE)).sum() < abs(corner - np.array(COARSE)).sum()


def test_a_following_map_leg_keeps_its_close_view(tmp_path):
    rasters = _rasters(tmp_path)

    default = _render_map(rasters, _day(), 13000, 320, 180, show_dashboard=False)
    explicit = _render_map(rasters, _day(), 13000, 320, 180, show_dashboard=False, overview=False)

    assert np.array_equal(np.asarray(default), np.asarray(explicit))


def test_the_film_asks_for_the_overview_on_its_recap(monkeypatch, tmp_path):
    seen = {}

    def fake_render_map(rasters, segments, at_time, width, height, **kwargs):
        seen.update(kwargs, at_time=at_time)
        return Image.new("RGB", (width, height))

    monkeypatch.setattr(film_module, "_render_map", fake_render_map)

    image = film_module._render_recap_map(_rasters(tmp_path), _day(), 13000, 320, 180)

    assert image.size == (320, 180)
    assert seen["overview"] is True
    assert seen["show_dashboard"] is False and seen["show_gap_badge"] is False
    assert seen["at_time"] == 13000
