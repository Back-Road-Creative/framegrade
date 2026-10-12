"""A photo beat dissolves in from the settled map and back out into the next leg.

A film once cut hard from map to still and back, a jump every few seconds
that read as a glitch. Dissolves live inside the frames each scene
already has, so film length and pacing never move.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from gomoveshift.pipeline.route_story import assembly
from gomoveshift.pipeline.route_story.anchors import MediaAnchor
from gomoveshift.pipeline.route_story.film import PhotoMoment
from gomoveshift.pipeline.route_story.metrics import RoutePoint

RED = (220, 30, 30)
BLUE = (30, 40, 220)


def _mean(frames: Path, number: int) -> np.ndarray:
    return np.asarray(Image.open(frames / f"{number:06d}.jpg").convert("RGB")).mean(axis=(0, 1))


def _photo(tmp_path: Path) -> PhotoMoment:
    source = tmp_path / "blue.jpg"
    Image.new("RGB", (640, 360), BLUE).save(source)
    return PhotoMoment(source, MediaAnchor("blue.jpg", "timestamp_interpolated", 5.0, 35.0, -80.0))


def test_a_photo_dissolves_in_from_the_map_without_adding_frames(tmp_path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    fps = 30

    end = assembly._append_photo(
        frames,
        0,
        _photo(tmp_path),
        fps,
        2.0,
        320,
        180,
        mapped=True,
        dissolve_from=Image.new("RGB", (320, 180), RED),
    )

    assert end == 60
    first, middle, last = _mean(frames, 0), _mean(frames, 7), _mean(frames, 59)
    # Red fades out, the still's blue fades in, and the dissolve is over well
    # before the still's own hold ends.
    assert first[0] > middle[0] > last[0] + 40
    assert first[2] < middle[2] < last[2]
    assert abs(_mean(frames, 30)[0] - last[0]) < 4


class _Solid:
    """A renderer that paints every map frame one flat colour."""

    def render_leg(self, segments, times, width, height):
        for _ in times:
            yield Image.new("RGB", (width, height), BLUE)


def test_a_map_leg_dissolves_in_from_the_still_and_hands_back_its_pure_render(tmp_path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()

    end, last = assembly._append_map_leg(
        frames,
        0,
        [[RoutePoint(0, 35.0, -80.0, 1), RoutePoint(10, 35.001, -80.0, 1)]],
        _Solid(),
        0.0,
        10.0,
        frame_count=30,
        width=320,
        height=180,
        dissolve_from=Image.new("RGB", (320, 180), RED),
        fps=30,
    )

    assert end == 30
    assert _mean(frames, 0)[0] > _mean(frames, 7)[0] > _mean(frames, 29)[0] + 40
    # The settle hold and QA reuse ``last``: it must be the renderer's own
    # pixels, never a blended save.
    assert last is not None and last.getpixel((5, 5)) == BLUE


def test_no_dissolve_source_means_no_blend(tmp_path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    assembly._append_photo(frames, 0, _photo(tmp_path), 30, 1.0, 320, 180, mapped=True)
    assert abs(_mean(frames, 0)[0] - _mean(frames, 29)[0]) < 4
