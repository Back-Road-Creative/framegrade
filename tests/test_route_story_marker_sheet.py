"""Marker close-ups in the review pack: is the dot still on its own line?

The defect this sheet exists for was found by eye, not by a gate: partway
through a finished film the marker began trailing behind the bright travelled
line, and nobody saw it until the whole multi-hour render was watched. The
measurement already exists (``frame_qa.line_end_to_marker_px``) -- what did not
exist was a place to SEE it before committing to the render. A wide map frame
at 640px across hides a four-pixel gap completely; a close-up does not.

Each tile is the crop AND the measured gap, so the operator is reading the same
number the render's own gate will later enforce -- never a second opinion kept
in step by hand.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from gomoveshift.pipeline.route_story import review_pack
from gomoveshift.pipeline.route_story.frame_qa import FrameQaThresholds

FRAME_W, FRAME_H = 640, 360


def _frame(colour: tuple[int, int, int] = (60, 70, 80)) -> Image.Image:
    return Image.new("RGB", (FRAME_W, FRAME_H), colour)


def _colour_count(image: Image.Image, colour: tuple[int, int, int], tolerance: int = 20) -> int:
    return sum(
        1
        for pixel in image.convert("RGB").getdata()
        if all(abs(a - b) <= tolerance for a, b in zip(pixel, colour))
    )


class TestMarkerCloseup:
    def test_a_closeup_is_exactly_the_requested_size(self) -> None:
        crop = review_pack.marker_closeup(_frame(), (320.0, 180.0), size=200)

        assert crop.size == (200, 200)

    def test_the_marker_lands_in_the_middle_of_its_own_closeup(self) -> None:
        """The whole point of the crop is that the dot is at its centre; an
        off-centre crop would put the gap being judged near an edge."""
        image = _frame()
        image.paste(Image.new("RGB", (6, 6), (255, 0, 0)), (297, 177))

        crop = review_pack.marker_closeup(image, (300.0, 180.0), size=100)

        middle = crop.crop((40, 40, 60, 60))
        assert _colour_count(middle, (255, 0, 0)) >= 30

    def test_a_marker_against_the_frame_edge_still_gets_a_full_size_crop(self) -> None:
        """Clamped, not shrunk. A ragged tile would make the sheet's rows
        stop lining up exactly where the map is most cramped."""
        crop = review_pack.marker_closeup(_frame(), (2.0, 2.0), size=200)

        assert crop.size == (200, 200)

    def test_a_frame_smaller_than_the_crop_is_returned_whole(self) -> None:
        small = Image.new("RGB", (80, 50), (10, 20, 30))

        assert review_pack.marker_closeup(small, (40.0, 25.0), size=200).size == (80, 50)


class TestMarkerSheet:
    def _tiles(self, gaps: list[float | None]) -> list[tuple[str, Image.Image, float | None]]:
        return [(f"{i * 10}s", _frame(), gap) for i, gap in enumerate(gaps)]

    def test_the_sheet_is_written_with_one_tile_per_sample(self, tmp_path: Path) -> None:
        output = review_pack.marker_sheet(self._tiles([0.0, 1.0, 2.0]), tmp_path / "marker.jpg")

        assert output.exists()
        with Image.open(output) as sheet:
            assert sheet.width >= review_pack.MARKER_CROP_PX * 3

    def test_a_route_with_nothing_to_show_gets_a_sheet_that_says_so(self, tmp_path: Path) -> None:
        """A missing file reads as a pack that failed to build, which is the
        one thing the pack must never be ambiguous about."""
        output = review_pack.marker_sheet([], tmp_path / "marker.jpg")

        assert output.exists()

    def test_a_gap_over_the_ceiling_is_flagged_in_the_warning_colour(self, tmp_path: Path) -> None:
        ceiling = FrameQaThresholds().max_line_end_to_marker_px
        output = review_pack.marker_sheet(self._tiles([ceiling + 5.0]), tmp_path / "marker.jpg")

        with Image.open(output) as sheet:
            assert _colour_count(sheet, review_pack.WARNING_RGB) > 100

    def test_a_gap_within_the_ceiling_is_not_flagged(self, tmp_path: Path) -> None:
        ceiling = FrameQaThresholds().max_line_end_to_marker_px
        output = review_pack.marker_sheet(
            self._tiles([max(0.0, ceiling - 1.0)]), tmp_path / "marker.jpg"
        )

        with Image.open(output) as sheet:
            assert _colour_count(sheet, review_pack.WARNING_RGB) <= 100

    def test_an_unmeasurable_gap_is_shown_as_unknown_not_as_zero(self, tmp_path: Path) -> None:
        """``line_end_to_marker_px`` returns None when no line pixel is in
        range -- the film's very first frame, before any line is drawn. Zero
        would read as a perfect result on a frame nothing was measured on.
        """
        output = review_pack.marker_sheet(self._tiles([None]), tmp_path / "marker.jpg")

        with Image.open(output) as sheet:
            assert _colour_count(sheet, review_pack.WARNING_RGB) <= 100

    def test_the_ceiling_comes_from_the_render_s_own_gate(self) -> None:
        """One number, not two. A sheet that flagged at its own threshold
        would show green on a film the render then refuses, or the reverse.
        """
        assert review_pack.MARKER_GAP_CEILING_PX == FrameQaThresholds().max_line_end_to_marker_px
