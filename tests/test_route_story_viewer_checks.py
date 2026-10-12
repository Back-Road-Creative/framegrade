"""Viewer-level Route Story checks: the marker/line-end gap and the
opening-shot legibility floors -- the two defects an operator rejected a
film for that the pixel-fraction detectors in frame_qa.py never catch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story.frame_qa import (
    TRAVELLED_LINE_RGB,
    FrameQaThresholds,
    check_frame,
    check_opening_frame,
    line_end_to_marker_px,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.preflight import RoutePreflightError, run_preflight

MARKER_XY = (100.0, 100.0)
SEGMENTS = [[RoutePoint(0.0, 35.0, -80.0, 100.0), RoutePoint(10.0, 35.001, -79.999, 110.0)]]


def _canvas(size: int = 200, base: tuple[int, int, int] = (30, 40, 50)) -> np.ndarray:
    frame = np.empty((size, size, 3), dtype=np.uint8)
    frame[:, :] = base
    return frame


def _draw_vertical_line(
    frame: np.ndarray, x: int, colour: tuple[int, int, int] = TRAVELLED_LINE_RGB
) -> np.ndarray:
    # 6px wide so travelled_line_min_width's own floor never trips here --
    # these tests are isolating the line-END-distance check, not width.
    frame[:, x : x + 6] = colour
    return frame


# --- line_end_to_marker_px ------------------------------------------------


def test_line_touching_the_marker_disc_reads_close_to_zero() -> None:
    frame = _canvas()
    # Line drawn right up to the marker exclusion disc edge (12px default).
    _draw_vertical_line(frame, x=112)
    gap = line_end_to_marker_px(frame, MARKER_XY)
    assert gap is not None
    assert gap == pytest.approx(0.0, abs=1.0)


def test_line_lagging_behind_the_marker_reads_a_large_gap() -> None:
    frame = _canvas()
    # Line stops 40px short of the marker -- the "dot lags the line" defect.
    _draw_vertical_line(frame, x=152)
    gap = line_end_to_marker_px(frame, MARKER_XY)
    assert gap is not None
    assert gap > 30.0


def test_no_line_in_range_returns_none() -> None:
    frame = _canvas()
    gap = line_end_to_marker_px(frame, MARKER_XY)
    assert gap is None


def test_check_frame_flags_a_line_that_lags_the_marker() -> None:
    frame = _canvas()
    _draw_vertical_line(frame, x=152)
    result = check_frame(0, frame, marker_xy=MARKER_XY)
    assert not result.ok
    assert result.line_end_to_marker_px is not None
    assert result.line_end_to_marker_px > 30.0
    assert any("travelled line ends" in p for p in result.problems)


def test_check_frame_passes_a_line_that_reaches_the_marker() -> None:
    frame = _canvas()
    _draw_vertical_line(frame, x=112)
    result = check_frame(0, frame, marker_xy=MARKER_XY)
    assert result.ok
    assert result.line_end_to_marker_px is not None
    assert result.line_end_to_marker_px <= FrameQaThresholds().max_line_end_to_marker_px


def test_check_frame_skips_line_end_check_with_no_marker() -> None:
    frame = _canvas()
    _draw_vertical_line(frame, x=152)  # would fail, if checked
    result = check_frame(0, frame)
    assert result.ok
    assert result.line_end_to_marker_px is None


# --- check_opening_frame ---------------------------------------------------


def _dark_blurred_flat_frame(size: int = 200) -> np.ndarray:
    """Measured mean luma ~19, colourfulness ~0.8, sharpness ~0.003 -- must
    fail all three floors."""
    import cv2

    rng = np.random.default_rng(0)
    frame = np.full((size, size, 3), (18, 19, 21), dtype=np.uint8)
    noise = rng.integers(-3, 4, size=(size, size, 3))
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(frame, (15, 15), 8)


def _bright_saturated_sharp_frame(size: int = 200) -> np.ndarray:
    """Measured mean luma ~130, colourfulness ~201, sharpness ~1320 -- must
    pass all three floors."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    colours = [(230, 40, 30), (30, 200, 60), (40, 90, 230), (240, 220, 30)]
    block = 20
    for y in range(size):
        for x in range(size):
            frame[y, x] = colours[((x // block) + (y // block)) % len(colours)]
    return frame


def test_dark_blurred_flat_frame_fails_all_three_floors() -> None:
    problems = check_opening_frame(_dark_blurred_flat_frame())
    assert len(problems) == 3
    assert any("mean luma" in p for p in problems)
    assert any("colourfulness" in p for p in problems)
    assert any("sharpness" in p for p in problems)


def test_bright_saturated_sharp_frame_passes() -> None:
    problems = check_opening_frame(_bright_saturated_sharp_frame())
    assert problems == ()


def test_check_opening_frame_bad_dtype_raises() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        check_opening_frame(frame)


# --- run_preflight: opening_frames -----------------------------------------


class _FakeRenderer:
    name = "fake"

    def __init__(self, ok_color=(80, 90, 100)):
        self._ok_color = ok_color
        self.render_calls: list[float] = []
        self.leg_calls = 0

    def render(self, segments, at_time, width, height):
        self.render_calls.append(at_time)
        return Image.new("RGB", (width, height), self._ok_color)

    def render_leg(self, segments, times, width, height):
        self.leg_calls += 1
        for at_time in times:
            yield self.render(segments, at_time, width, height)


def test_run_preflight_passes_with_a_good_opening_frame(tmp_path: Path) -> None:
    good = Image.fromarray(_bright_saturated_sharp_frame(), mode="RGB")
    result = run_preflight(
        _FakeRenderer(),
        SEGMENTS,
        start_time=0.0,
        end_time=10.0,
        width=64,
        height=36,
        count=2,
        output=tmp_path / "out-preflight.jpg",
        opening_frames=[good],
    )
    assert result.passed


def test_run_preflight_fails_on_a_dark_blurred_opening_frame(tmp_path: Path) -> None:
    bad = Image.fromarray(_dark_blurred_flat_frame(), mode="RGB")
    output = tmp_path / "out-preflight.jpg"
    with pytest.raises(RoutePreflightError, match="mean luma"):
        run_preflight(
            _FakeRenderer(),
            SEGMENTS,
            start_time=0.0,
            end_time=10.0,
            width=64,
            height=36,
            count=2,
            output=output,
            opening_frames=[bad],
        )
    # The contact sheet is written before the opening-frame check raises.
    assert output.is_file()


# --- run_preflight: marker-geometry checks on a checks_raw renderer --------


class _MarkerLaggingRawRenderer(_FakeRenderer):
    """Like SceneMapRenderer (checks_raw_frames=True): QA's nothing here
    (the real renderer QA's its own raw capture before composing), but
    exposes last_marker_xy and draws a travelled line that lags the marker
    -- the defect the checks_raw branch must still catch via the
    marker-geometry-only path."""

    checks_raw_frames = True

    def __init__(self, gap_px: float):
        super().__init__(ok_color=(30, 40, 50))
        self._gap_px = gap_px
        self.last_marker_xy = None

    def render(self, segments, at_time, width, height):
        image = super().render(segments, at_time, width, height)
        pixels = image.load()
        marker_x, marker_y = 50, 20
        line_x = int(marker_x + 12 + self._gap_px)
        for y in range(height):
            for dx in range(6):  # a wide stroke so line-width never trips here
                x = line_x + dx
                if 0 <= x < width:
                    pixels[x, y] = TRAVELLED_LINE_RGB
        self.last_marker_xy = (float(marker_x), float(marker_y))
        return image


def test_checks_raw_renderer_still_fails_a_lagging_marker(tmp_path: Path) -> None:
    with pytest.raises(RoutePreflightError, match="travelled line ends"):
        run_preflight(
            _MarkerLaggingRawRenderer(gap_px=20.0),
            SEGMENTS,
            start_time=0.0,
            end_time=10.0,
            width=100,
            height=40,
            count=2,
            output=tmp_path / "out-preflight.jpg",
        )


def test_checks_raw_renderer_passes_a_marker_reaching_the_line(tmp_path: Path) -> None:
    result = run_preflight(
        _MarkerLaggingRawRenderer(gap_px=0.0),
        SEGMENTS,
        start_time=0.0,
        end_time=10.0,
        width=100,
        height=40,
        count=2,
        output=tmp_path / "out-preflight.jpg",
    )
    assert result.passed
