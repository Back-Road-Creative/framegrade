"""A still the GPS track cannot place is captioned as film, not as a console."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from gomoveshift.pipeline.route_story import assembly as assembly_module
from gomoveshift.pipeline.route_story import hud as hud_module
from gomoveshift.pipeline.route_story.anchors import MediaAnchor
from gomoveshift.pipeline.route_story.film import PhotoMoment


def _photo(tmp_path: Path) -> PhotoMoment:
    source = tmp_path / "vista.jpg"
    Image.new("RGB", (640, 480), "#5f8a4a").save(source)
    anchor = MediaAnchor(
        media_id="vista.jpg",
        status="unanchored_gap",
        track_time_s=None,
        latitude=None,
        longitude=None,
    )
    return PhotoMoment(source=source, anchor=anchor)


def _captions(monkeypatch) -> list[str]:
    drawn: list[str] = []
    monkeypatch.setattr(
        assembly_module, "_draw_photo_label", lambda image, label: drawn.append(label)
    )
    return drawn


class TestUnplacedPhotoCaption:
    def test_the_caption_reads_as_a_sentence_a_viewer_understands(self):
        caption = hud_module.UNPLACED_PHOTO_CAPTION

        assert "UNLOCATED" not in caption.upper()
        assert caption != caption.upper()
        assert "not placed on the route map" in caption

    def test_an_unplaced_still_carries_that_caption_on_every_frame(self, tmp_path, monkeypatch):
        drawn = _captions(monkeypatch)
        frames = tmp_path / "frames"
        frames.mkdir()

        end = assembly_module._append_photo(
            frames, 0, _photo(tmp_path), 2, 1.0, 320, 180, mapped=False
        )

        assert end == 2
        assert drawn == [hud_module.UNPLACED_PHOTO_CAPTION] * 2

    def test_a_placed_still_carries_no_caption(self, tmp_path, monkeypatch):
        drawn = _captions(monkeypatch)
        frames = tmp_path / "frames"
        frames.mkdir()

        assembly_module._append_photo(frames, 0, _photo(tmp_path), 2, 1.0, 320, 180, mapped=True)

        assert drawn == []
