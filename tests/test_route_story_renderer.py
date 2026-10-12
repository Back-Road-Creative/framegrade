"""The MapRenderer seam: film map legs render through a swappable renderer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import layers as layers_module
from gomoveshift.pipeline.route_story.anchors import MediaAnchor
from gomoveshift.pipeline.route_story.film import (
    PhotoMoment,
    film_receipt,
    render_route_story_film,
)
from gomoveshift.pipeline.route_story.frame_state import FrameState
from gomoveshift.pipeline.route_story.layers import PillowMapRenderer, _render_map
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster


@dataclass
class _RecordingRenderer:
    """A fake MapRenderer that records every call it receives."""

    name: str = "fake-renderer"
    calls: list[float] = field(default_factory=list)

    def render(
        self,
        segments: Sequence[Sequence[RoutePoint]],
        at_time: float,
        width: int,
        height: int,
    ) -> Image.Image:
        self.calls.append(at_time)
        return Image.new("RGB", (width, height), "#123456")

    def measuring(self, segments):
        # The film asks every renderer to measure the recorded points.
        return self

    def render_leg(
        self,
        segments: Sequence[Sequence[RoutePoint]],
        times: Sequence[float],
        width: int,
        height: int,
    ):
        for at_time in times:
            yield self.render(segments, at_time, width, height)


def test_a_supplied_map_renderer_renders_every_map_leg_frame_in_order(tmp_path: Path):
    overview = tmp_path / "overview.jpg"
    local = tmp_path / "local.jpg"
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (720, 360), "#3c6f83").save(overview)
    Image.new("RGB", (640, 360), "#4e7540").save(local)
    Image.new("RGB", (640, 360), "#aa7744").save(photo)
    output = tmp_path / "film.mp4"
    renderer = _RecordingRenderer()

    render_route_story_film(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]],
        national=SatelliteRaster(overview, -180, -90, 180, 90, "Test overview"),
        local=SatelliteRaster(local, -80.1, 34.9, -79.9, 35.1, "Test local"),
        photos=[
            PhotoMoment(
                photo,
                MediaAnchor("photo.jpg", "timestamp_interpolated", 5.0, 35.0001, -79.9999),
            )
        ],
        output=output,
        map_renderer=renderer,
        fps=2,
        width=320,
        height=180,
        intro_seconds=1,
        map_seconds=1,
        photo_seconds=1,
        recap_seconds=1,
    )

    assert output.is_file()
    # Every map-leg frame went through the supplied renderer, and route clock
    # time only ever advances within a leg.
    assert len(renderer.calls) > 0
    for previous, current in zip(renderer.calls, renderer.calls[1:]):
        assert current >= previous


def test_the_receipt_names_the_default_and_a_supplied_renderer(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [[RoutePoint(0, 0, 0), RoutePoint(10, 0, 0.01)]]

    default_receipt = film_receipt(segments=segments, national=raster, local=raster, photos=[])
    assert default_receipt["map_renderer"] == "pillow-2d"

    fake_receipt = film_receipt(
        segments=segments,
        national=raster,
        local=raster,
        photos=[],
        map_renderer=_RecordingRenderer(name="fake-renderer"),
    )
    assert fake_receipt["map_renderer"] == "fake-renderer"


def test_pillow_map_renderer_matches_render_map_pixels(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (64, 32), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -80.1, 34.9, -79.9, 35.1, "test")
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]]

    expected = _render_map([raster], segments, 5.0, 160, 90)
    renderer = PillowMapRenderer(rasters=(raster,))
    actual = renderer.render(segments, 5.0, 160, 90)

    assert renderer.name == "pillow-2d"
    assert list(actual.getdata()) == list(expected.getdata())


def test_render_map_draws_its_marker_from_a_single_frame_state_at_call(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (64, 32), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -80.1, 34.9, -79.9, 35.1, "test")
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]]

    # A frame_state_at result moved far from where the recorded points would
    # otherwise place it must change the rendered marker: the only way the
    # drawn pixel can move to match it is if _render_map's marker literally
    # comes from this call's return value rather than its own interpolation.
    moved = FrameState(5.0, 35.02, -79.98, 105.0, 0.1, 0.0, 5.0, True)

    with patch.object(layers_module, "frame_state_at", return_value=moved) as spy:
        moved_image = _render_map([raster], segments, 5.0, 160, 90)

    assert spy.call_count == 1
    called_timeline, called_frame, called_segments = spy.call_args.args
    assert called_frame == 0
    assert called_segments == segments

    default_image = _render_map([raster], segments, 5.0, 160, 90)
    assert list(moved_image.getdata()) != list(default_image.getdata())


def test_render_map_raises_valueerror_not_zerodivision_on_duplicate_timestamps(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (64, 32), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -80.1, 34.9, -79.9, 35.1, "test")
    # Two points sharing a timestamp: the old private interpolator divided
    # by a zero duration; frame_state_at must raise instead, exactly like
    # the dashboard path already does for the same malformed source data.
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(0, 35.0001, -79.9999, 110)]]

    with pytest.raises(ValueError, match="strictly increase"):
        _render_map([raster], segments, 0.0, 160, 90)
