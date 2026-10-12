from pathlib import Path

from PIL import Image

from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.opening import SatelliteOpening
from gomoveshift.pipeline.route_story.render import render_map_frame, render_video
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster
from gomoveshift.pipeline.route_story.timeline import Scene, StoryTimeline


def test_renderer_writes_a_readable_landscape_frame(tmp_path: Path):
    output = tmp_path / "frame.png"
    timeline = StoryTimeline(fps=10, scenes=(Scene("route_follow", 0, 10, 0.0, 10.0),))
    render_map_frame(
        timeline,
        5,
        [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.01, -79.99, 110)]],
        output,
        width=640,
        height=360,
    )
    image = Image.open(output)
    assert image.size == (640, 360)
    assert image.getbbox() is not None


def test_renderer_writes_a_playable_mp4(tmp_path: Path):
    output = tmp_path / "story.mp4"
    timeline = StoryTimeline(fps=2, scenes=(Scene("route_follow", 0, 2, 0.0, 1.0),))
    render_video(
        timeline,
        [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(1, 35.01, -79.99, 110)]],
        output,
        width=320,
        height=180,
    )
    assert output.is_file()
    assert output.stat().st_size > 0


def test_renderer_uses_a_local_satellite_raster_for_an_opening_frame(tmp_path: Path):
    source = tmp_path / "satellite.png"
    Image.new("RGB", (360, 180), "#4a7f45").save(source)
    output = tmp_path / "opening.png"
    opening = SatelliteOpening.for_route_start(
        latitude=35.0, longitude=-80.0, start_frame=0, end_frame=10, final_zoom=8.0
    )
    timeline = StoryTimeline(fps=10, scenes=(Scene("overview", 0, 10, 0.0, 0.0),))

    render_map_frame(
        timeline,
        0,
        [[RoutePoint(0, 35.0, -80.0, 100)]],
        output,
        width=320,
        height=180,
        opening=opening,
        satellite=SatelliteRaster(source, west=-180, south=-90, east=180, north=90, credit="Test"),
    )

    assert Image.open(output).getpixel((160, 100)) != (16, 33, 42)


def test_renderer_encodes_a_satellite_opening_to_mp4(tmp_path: Path):
    source = tmp_path / "satellite.png"
    Image.new("RGB", (360, 180), "#4a7f45").save(source)
    output = tmp_path / "opening.mp4"
    opening = SatelliteOpening.for_route_start(
        latitude=35.0, longitude=-80.0, start_frame=0, end_frame=2, final_zoom=8.0
    )
    timeline = StoryTimeline(fps=2, scenes=(Scene("overview", 0, 2, 0.0, 0.0),))

    render_video(
        timeline,
        [[RoutePoint(0, 35.0, -80.0, 100)]],
        output,
        width=320,
        height=180,
        opening=opening,
        satellite=SatelliteRaster(source, west=-180, south=-90, east=180, north=90, credit="Test"),
    )

    assert output.is_file()
    assert output.stat().st_size > 0
