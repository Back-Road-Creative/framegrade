"""A running film render says how far along it is, so an ETA is read, not guessed."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image

from gomoveshift.pipeline.route_story import assembly
from gomoveshift.pipeline.route_story import film as film_module


class _CountingRenderer:
    name = "counting"

    def render(self, segments, at_time, width, height) -> Image.Image:
        return Image.new("RGB", (width, height), (40, 80, 120))

    def render_leg(
        self, segments, times: Sequence[float], width: int, height: int
    ) -> Iterator[Image.Image]:
        for at_time in times:
            yield self.render(segments, at_time, width, height)


def _append(tmp_path: Path, frame_count: int) -> int:
    frame, _last_image = assembly._append_map_leg(
        tmp_path,
        0,
        [],
        _CountingRenderer(),
        0.0,
        100.0,
        frame_count=frame_count,
        width=32,
        height=18,
    )
    return frame


def test_a_map_leg_reports_progress_with_a_frame_count_and_a_rate(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(assembly, "PROGRESS_EVERY_FRAMES", 4)

    with caplog.at_level(logging.INFO, logger=assembly.__name__):
        assert _append(tmp_path, 10) == 10

    progress = [r.getMessage() for r in caplog.records if "map leg" in r.getMessage()]
    assert any("4/10" in line for line in progress)
    assert any("8/10" in line for line in progress)
    assert any("10/10" in line for line in progress), "the finished leg is reported too"
    assert all("s/frame" in line for line in progress if "/10" in line)


def test_a_short_leg_still_reports_that_it_finished(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=assembly.__name__):
        _append(tmp_path, 3)

    assert any("3/3" in r.getMessage() for r in caplog.records)


def test_progress_logging_does_not_change_the_frames_written(tmp_path):
    assert _append(tmp_path, 5) == 5
    assert sorted(p.name for p in tmp_path.glob("*.jpg")) == [f"{i:06d}.jpg" for i in range(5)]


def test_a_map_leg_returns_the_exact_image_it_last_rendered(tmp_path):
    """A caller holding on the leg's ending view (a photo settle, a video
    arrival) must reuse this image rather than asking the renderer for a
    second frame at the same time -- that second call can disagree with
    whatever rate limiting the renderer applied and snaps the camera."""
    renderer = _CountingRenderer()

    frame, last_image = assembly._append_map_leg(
        tmp_path, 0, [], renderer, 0.0, 100.0, frame_count=4, width=32, height=18
    )

    assert frame == 4
    assert last_image is not None
    assert last_image.size == (32, 18)
    assert list(last_image.getdata()) == list(Image.new("RGB", (32, 18), (40, 80, 120)).getdata())


def test_the_film_cli_turns_progress_logging_on(monkeypatch):
    seen: dict[str, object] = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: seen.update(kwargs))

    film_module.configure_progress_logging()

    assert seen["level"] == logging.INFO
