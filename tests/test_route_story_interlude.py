"""The video interlude is finished film, not an operator console.

An earlier film cut to a near-black card reading "LOCATION CONFLICT • REVIEW GPS /
CLOCK" and then to a portrait clip between two flat dark bars. The review
instruction is for whoever reads the log and the receipt; the viewer is told
only what is true for them -- the clip is not placed on the map.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from gomoveshift.pipeline.route_story import assembly as assembly_module
from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story.anchors import MediaAnchor
from gomoveshift.pipeline.route_story.hud import _unlocated_video_card

_OPERATOR_WORDS = ("REVIEW", "CONFLICT", "GPS", "CLOCK")


def _clip(path: Path, size: str, source: str = "testsrc2") -> None:
    joiner = ":" if "=" in source else "="
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"{source}{joiner}s={size}:d=0.5:r=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip


def _first_frame(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "image2pipe",
         "-vcodec", "png", "-"],
        check=True,
        capture_output=True,
    ).stdout  # fmt: skip
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.int16)


class TestWhatTheViewerIsTold:
    def test_a_location_conflict_is_not_put_on_screen(self):
        anchor = MediaAnchor("clip.mp4", "coordinate_time_conflict", 5.0, 40.0, 40.0)

        subtitle = assembly_module._interlude_subtitle(anchor)

        assert not any(word in subtitle.upper() for word in _OPERATOR_WORDS)
        assert "not placed on the route map" in subtitle

    def test_a_conflict_reads_the_same_on_screen_as_a_clip_with_no_anchor(self):
        anchor = MediaAnchor("clip.mp4", "coordinate_time_conflict", 5.0, 40.0, 40.0)

        assert assembly_module._interlude_subtitle(anchor) == assembly_module._interlude_subtitle(
            None
        )

    def test_an_anchored_clip_still_says_it_is_on_the_route(self):
        anchor = MediaAnchor("clip.mp4", "embedded_coordinates", 5.0, 40.0, 40.0)

        assert "ROUTE ANCHORED" in assembly_module._interlude_subtitle(anchor)

    def test_the_conflict_is_logged_for_whoever_runs_the_render(self, tmp_path, caplog):
        clip = tmp_path / "clip.mp4"
        _clip(clip, "64x64")
        anchor = MediaAnchor("clip.mp4", "coordinate_time_conflict", 5.0, 40.0, 40.0)

        with caplog.at_level("WARNING"):
            film_module._interlude_card(clip, anchor, tmp_path, 320, 180)

        assert "clip.mp4" in caplog.text
        assert "review" in caplog.text.lower()


class TestTheCardIsAPictureNotABlackFrame:
    def test_the_card_sits_on_the_clips_own_first_frame(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        _clip(clip, "64x64", source="color=c=0x7090b0")

        card = film_module._interlude_card(clip, None, tmp_path, 320, 180)

        assert card.size == (320, 180)
        luma = np.asarray(card.convert("L"))
        # ffmpeg's blackdetect calls a pixel black under 10% luminance and a
        # frame black at 98% such pixels; the old flat card was one.
        assert (luma > 0.10 * 255).mean() > 0.5
        blue = np.asarray(card)[:, :, 2].mean()
        red = np.asarray(card)[:, :, 0].mean()
        assert blue > red, "the backdrop should carry the clip's own colour"

    def test_a_clip_that_cannot_be_read_falls_back_to_the_flat_card(self, tmp_path, caplog):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"not a video")

        with caplog.at_level("WARNING"):
            card = film_module._interlude_card(broken, None, tmp_path, 320, 180)

        assert card.size == (320, 180)
        assert "backdrop" in caplog.text

    def test_the_flat_card_still_draws_its_disclosure(self):
        card = _unlocated_video_card(320, 180)

        assert len(card.convert("L").getcolors(maxcolors=65536) or []) > 1


class TestAPortraitClipFillsTheFrame:
    def test_the_bars_beside_a_portrait_clip_carry_the_clips_own_picture(self, tmp_path):
        source = tmp_path / "portrait.mp4"
        _clip(source, "72x128", source="color=c=0x7090b0")
        interlude = tmp_path / "interlude.mp4"

        assembly_module._encode_source_video(source, interlude, 10, 320, 180)

        frame = _first_frame(interlude)
        assert frame.shape[:2] == (180, 320)
        bar = frame[:, :60].reshape(-1, 3).mean(axis=0)
        flat_pad = np.array([0x09, 0x14, 0x1A])
        assert np.abs(bar - flat_pad).max() > 25, "the side bar is still the flat pad colour"
        assert bar[2] > bar[0], "the side bar should carry the clip's colour"

    def test_the_clip_itself_is_shown_whole_and_undimmed_in_the_centre(self, tmp_path):
        source = tmp_path / "portrait.mp4"
        _clip(source, "72x128", source="color=c=0x7090b0")
        interlude = tmp_path / "interlude.mp4"

        assembly_module._encode_source_video(source, interlude, 10, 320, 180)

        frame = _first_frame(interlude)
        centre = frame[60:120, 150:170].reshape(-1, 3).mean(axis=0)
        bar = frame[60:120, 10:40].reshape(-1, 3).mean(axis=0)
        assert centre.sum() > bar.sum() * 1.3, "the backdrop must sit behind the clip, dimmed"

    def test_a_clip_that_already_fills_the_frame_is_unchanged_in_shape(self, tmp_path):
        source = tmp_path / "landscape.mp4"
        _clip(source, "320x180")
        interlude = tmp_path / "interlude.mp4"

        assembly_module._encode_source_video(source, interlude, 10, 320, 180)

        assert _first_frame(interlude).shape[:2] == (180, 320)
