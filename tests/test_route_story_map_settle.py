"""The map settles on the spot before a still plays.

Before this, a map leg ran right up to a photo's route moment and the film cut
straight to the still on the next frame -- the camera was still moving when the
picture arrived, so the still read as an interruption rather than as the thing
the ride had just arrived at. A short hold on the settled map fixes that, and
it is the same hold shape the placed-video branch already used, so there is one
implementation of "stop here for a beat" rather than two.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from gomoveshift.pipeline.route_story import assembly

WIDTH, HEIGHT = 64, 36


def _frame(colour: tuple[int, int, int] = (40, 80, 120)) -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), colour)


class TestAppendHold:
    def test_a_hold_writes_one_frame_per_frame_of_its_duration(self, tmp_path: Path) -> None:
        frames = tmp_path / "frames"
        frames.mkdir()

        end = assembly._append_hold(frames, 0, _frame(), fps=30, seconds=0.6)

        assert end == 18
        assert len(sorted(frames.glob("*.jpg"))) == 18

    def test_a_hold_continues_the_caller_s_frame_numbering(self, tmp_path: Path) -> None:
        """Holds land mid-sequence, between a map leg and a still, so they
        start where the caller left off and hand back where they stopped."""
        frames = tmp_path / "frames"
        frames.mkdir()

        end = assembly._append_hold(frames, 100, _frame(), fps=30, seconds=0.2)

        assert end == 106
        assert (frames / "000100.jpg").exists()
        assert (frames / "000105.jpg").exists()

    def test_every_frame_of_a_hold_is_the_same_image(self, tmp_path: Path) -> None:
        """One render call, reused. Re-rendering per frame would cost real
        minutes under the scene renderer for pixels that never change."""
        frames = tmp_path / "frames"
        frames.mkdir()

        assembly._append_hold(frames, 0, _frame(), fps=30, seconds=0.4)

        written = [Image.open(path).convert("RGB") for path in sorted(frames.glob("*.jpg"))]
        first = list(written[0].getdata())
        assert all(list(image.getdata()) == first for image in written[1:])

    def test_a_duration_shorter_than_one_frame_still_holds_one_frame(self, tmp_path: Path) -> None:
        """A settle rounded away to nothing would put the cut back exactly
        where it was, silently, at a low enough fps."""
        frames = tmp_path / "frames"
        frames.mkdir()

        assert assembly._append_hold(frames, 0, _frame(), fps=2, seconds=0.1) == 1


class TestSettleDurations:
    def test_the_still_settle_is_shorter_than_the_video_arrival_hold(self) -> None:
        """A placed video announces itself with a badge and needs a beat to be
        read; a still needs only enough stillness to land on. Equal lengths
        would make every photo feel like a video that failed to play.
        """
        assert 0 < assembly.MAP_SETTLE_SECONDS < assembly.VIDEO_ARRIVAL_HOLD_SECONDS
