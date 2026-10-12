"""The recap map is drawn once; only its fading numbers change per frame.

``_render_recap_map`` was called for every recap frame with identical
arguments -- 180 full map renders for a 6 s recap at 30 fps, each one the
same picture.
"""

from pathlib import Path

from PIL import Image

from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story.film import render_route_story_film
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster


def _rasters(tmp_path: Path):
    overview = tmp_path / "overview.jpg"
    local = tmp_path / "local.jpg"
    Image.new("RGB", (720, 360), "#3c6f83").save(overview)
    Image.new("RGB", (640, 360), "#4e7540").save(local)
    return (
        SatelliteRaster(overview, -180, -90, 180, 90, "Test overview"),
        SatelliteRaster(local, -80.1, 34.9, -79.9, 35.1, "Test local"),
    )


def test_the_recap_map_renders_once_and_every_frame_gets_its_own_fade(tmp_path, monkeypatch):
    national, local = _rasters(tmp_path)
    monkeypatch.setattr(
        film_module, "_encode_frames", lambda frames, output, fps: Path(output).write_bytes(b"f")
    )
    monkeypatch.setattr(
        film_module, "_concat", lambda clips, output: Path(output).write_bytes(b"x")
    )

    renders: list[int] = []
    real_render = film_module._render_recap_map

    def counting_render(*args, **kwargs):
        renders.append(1)
        return real_render(*args, **kwargs)

    fractions: list[float] = []
    bases: list[bytes] = []
    real_draw = film_module._draw_recap

    def spy_draw(image, summary, fraction):
        # The map under the numbers must be the pristine render every time --
        # a fade drawn over the previous frame's fade would compound.
        bases.append(image.tobytes())
        fractions.append(fraction)
        return real_draw(image, summary, fraction)

    monkeypatch.setattr(film_module, "_render_recap_map", counting_render)
    monkeypatch.setattr(film_module, "_draw_recap", spy_draw)

    render_route_story_film(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(20, 35.0002, -79.9998, 110)]],
        national=national,
        local=local,
        photos=[],
        output=tmp_path / "film.mp4",
        fps=4,
        width=320,
        height=180,
        intro_seconds=1,
        map_seconds=1,
        photo_seconds=1,
        recap_seconds=2,
    )

    assert len(renders) == 1
    assert len(fractions) == 8
    assert fractions == sorted(fractions) and fractions[0] == 0.0 and fractions[-1] == 1.0
    assert len(set(bases)) == 1
