"""Tests for the Route Story rendered-frame QA detectors (void/unloaded/black pixels)."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from gomoveshift.pipeline.route_story.frame_qa import (
    FLAT_BLACK_MAX,
    PAGE_VOID_RGB,
    UNLOADED_RGB,
    FrameQaError,
    FrameQaResult,
    FrameQaThresholds,
    assert_frame_ok,
    check_frame,
    summarize,
    travelled_line_min_width,
)

# 100x10x3 = 1000 pixels: a round total makes exact-fraction boundaries easy.
HEIGHT, WIDTH = 100, 10
TOTAL = HEIGHT * WIDTH
BASE_RGB = (100, 110, 120)


def _frame(base: tuple[int, int, int] = BASE_RGB) -> np.ndarray:
    frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :] = base
    return frame


def _paint(frame: np.ndarray, count: int, color: tuple[int, int, int]) -> np.ndarray:
    """Paint the first ``count`` pixels (row-major) a solid colour."""
    flat = frame.reshape(-1, 3)
    flat[:count] = color
    return frame


def test_clean_frame_passes() -> None:
    result = check_frame(0, _frame())
    assert result.ok
    assert result.problems == ()
    assert result.void_fraction == 0.0
    assert result.unloaded_fraction == 0.0
    assert result.black_fraction == 0.0


def test_one_percent_white_fails_naming_frame_and_metric() -> None:
    frame = _paint(_frame(), 10, PAGE_VOID_RGB)  # 10/1000 = 1%
    result = check_frame(42, frame)
    assert not result.ok
    assert result.void_fraction == pytest.approx(0.01)
    assert len(result.problems) == 1
    assert "page-white" in result.problems[0]
    assert "42" in result.problems[0]


def test_exactly_at_threshold_passes() -> None:
    # 5/1000 = 0.5% == the default max_void_fraction; boundary must pass.
    frame = _paint(_frame(), 5, PAGE_VOID_RGB)
    result = check_frame(1, frame)
    assert result.void_fraction == pytest.approx(0.005)
    assert result.ok


def test_unloaded_tile_block_fails_only_that_metric() -> None:
    frame = _paint(_frame(), 20, UNLOADED_RGB)  # 2%
    result = check_frame(2, frame)
    assert not result.ok
    assert result.unloaded_fraction == pytest.approx(0.02)
    assert result.void_fraction == 0.0
    assert result.black_fraction == 0.0
    assert len(result.problems) == 1
    assert "unloaded-tile" in result.problems[0]


def test_near_black_block_fails_only_that_metric() -> None:
    frame = _paint(_frame(), 20, (0, 0, 0))  # 2%, well under FLAT_BLACK_MAX
    result = check_frame(3, frame)
    assert not result.ok
    assert result.black_fraction == pytest.approx(0.02)
    assert result.void_fraction == 0.0
    assert result.unloaded_fraction == 0.0
    assert len(result.problems) == 1
    assert "flat-black" in result.problems[0]
    assert FLAT_BLACK_MAX == 8


def test_tolerance_includes_near_white_but_not_further_off_white() -> None:
    frame = _frame()
    flat = frame.reshape(-1, 3)
    flat[:10] = (254, 254, 254)  # #fefefe: diff 1, within default tolerance 6
    flat[10:20] = (240, 240, 240)  # #f0f0f0: diff 15, outside tolerance
    result = check_frame(4, frame)
    assert result.void_fraction == pytest.approx(10 / TOTAL)


def test_hud_mask_excludes_white_bar() -> None:
    frame = _frame()
    flat = frame.reshape(-1, 3)
    flat[:100] = PAGE_VOID_RGB  # a "HUD bar": 100/1000 = 10% white
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask.reshape(-1)[:100] = True
    result = check_frame(5, frame, hud_mask=mask)
    assert result.ok
    assert result.void_fraction == 0.0


def test_bad_dtype_raises() -> None:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        check_frame(0, frame)


def test_hud_mask_shape_mismatch_raises() -> None:
    frame = _frame()
    mask = np.zeros((HEIGHT + 1, WIDTH), dtype=bool)
    with pytest.raises(ValueError):
        check_frame(0, frame, hud_mask=mask)


def test_bad_shape_raises() -> None:
    frame = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        check_frame(0, frame)

    frame_2d = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    with pytest.raises(ValueError):
        check_frame(0, frame_2d)


def test_assert_frame_ok_raises_with_frame_and_pose() -> None:
    frame = _paint(_frame(), 10, PAGE_VOID_RGB)
    result = check_frame(42, frame)
    pose = {"longitude": -122.4, "latitude": 37.8, "zoom": 14}
    with pytest.raises(FrameQaError) as excinfo:
        assert_frame_ok(result, pose)
    message = str(excinfo.value)
    assert "42" in message
    assert "page-white" in message
    assert "-122.4" in message
    assert excinfo.value.result is result
    assert excinfo.value.pose == pose


def test_assert_frame_ok_passes_for_clean_result() -> None:
    result = check_frame(0, _frame())
    assert_frame_ok(result, {"longitude": 0, "latitude": 0})


def test_summarize_picks_worst_frame_per_metric() -> None:
    results = [
        FrameQaResult(
            frame=0, void_fraction=0.01, unloaded_fraction=0.0, black_fraction=0.0, problems=("p",)
        ),
        FrameQaResult(
            frame=1, void_fraction=0.02, unloaded_fraction=0.03, black_fraction=0.0, problems=("p",)
        ),
        FrameQaResult(
            frame=2, void_fraction=0.0, unloaded_fraction=0.0, black_fraction=0.04, problems=("p",)
        ),
    ]
    summary = summarize(results)
    assert summary["frames_checked"] == 3
    assert summary["frames_failed"] == 3
    assert summary["void"] == {"worst_fraction": pytest.approx(0.02), "frame": 1}
    assert summary["unloaded"] == {"worst_fraction": pytest.approx(0.03), "frame": 1}
    assert summary["black"] == {"worst_fraction": pytest.approx(0.04), "frame": 2}


def test_summarize_empty() -> None:
    summary = summarize([])
    assert summary["frames_checked"] == 0
    assert summary["frames_failed"] == 0


def test_thresholds_are_configurable() -> None:
    frame = _paint(_frame(), 10, PAGE_VOID_RGB)  # 1%
    lenient = FrameQaThresholds(max_void_fraction=0.5)
    result = check_frame(0, frame, lenient)
    assert result.ok


def test_check_frame_is_fast_and_vectorised() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    started = time.monotonic()
    check_frame(0, frame)
    elapsed = time.monotonic() - started
    assert elapsed < 0.5


# --- travelled_line_min_width -------------------------------------------

TRAVELLED_RGB = (0xFF, 0xCF, 0x3F)
MARKER_XY = (100.0, 100.0)


def _canvas(size: int = 200, base: tuple[int, int, int] = (30, 40, 50)) -> np.ndarray:
    frame = np.empty((size, size, 3), dtype=np.uint8)
    frame[:, :] = base
    return frame


def _draw_vertical_line(
    frame: np.ndarray, x: int, half_width: int, colour: tuple[int, int, int] = TRAVELLED_RGB
) -> np.ndarray:
    frame[:, x - half_width : x + half_width + 1] = colour
    return frame


def test_min_width_of_a_uniform_3px_line() -> None:
    frame = _canvas()
    _draw_vertical_line(frame, x=100, half_width=1)  # ~3px wide
    width = travelled_line_min_width(frame, MARKER_XY, radius_px=80)
    assert width is not None
    assert 2.5 <= width <= 3.5


def test_min_width_of_a_1px_hairline() -> None:
    frame = _canvas()
    _draw_vertical_line(frame, x=100, half_width=0)  # 1px wide
    width = travelled_line_min_width(frame, MARKER_XY, radius_px=80)
    assert width is not None
    assert width < 2.0


def test_min_width_tracks_the_narrowest_segment() -> None:
    frame = _canvas()
    # Wide (4px) above the marker row, hairline (1px) below it.
    for y in range(0, 100):
        frame[y, 96:104] = TRAVELLED_RGB
    for y in range(100, 200):
        frame[y, 99:100] = TRAVELLED_RGB
    width = travelled_line_min_width(frame, MARKER_XY, radius_px=90)
    assert width is not None
    assert width < 2.0


def test_min_width_of_a_sharp_switchback_beside_the_marker() -> None:
    # Pins the "corner artifact" suspicion from the real-capture probe: a
    # uniformly 5px-wide stroke with several sharp switchback bends close to
    # the marker (as a real chase-cam frame over tight terrain produces)
    # must still measure close to its true width, not the couple-of-px
    # underestimate a strict single-pixel ridge min reads at a bend's own
    # join. If this ever regresses below 4px, the 5th-percentile mitigation
    # in travelled_line_min_width needs revisiting -- it is not just theory.
    frame = _canvas(size=300, base=(30, 40, 50))
    marker = (150.0, 150.0)
    points = np.array(
        [
            [155, 152],
            [175, 140],
            [160, 125],
            [185, 110],
            [165, 95],
            [190, 75],
            [170, 55],
            [195, 35],
        ],
        dtype=np.int32,
    )
    cv2.polylines(
        frame,
        [points],
        isClosed=False,
        color=TRAVELLED_RGB,
        thickness=5,
        lineType=cv2.LINE_8,
    )
    width = travelled_line_min_width(frame, marker, radius_px=140)
    assert width is not None
    assert width >= 4.0


def test_no_line_near_marker_returns_none() -> None:
    frame = _canvas()
    width = travelled_line_min_width(frame, MARKER_XY, radius_px=80)
    assert width is None


def test_line_fully_hidden_under_marker_disc_returns_none() -> None:
    frame = _canvas()
    # A travelled-coloured line that only exists inside the marker's own
    # footprint (never reaches the surrounding ring) must not be measured --
    # there is no real line pixel visible there to report a width for.
    frame[98:103, 98:103] = TRAVELLED_RGB
    width = travelled_line_min_width(frame, MARKER_XY, radius_px=80)
    assert width is None


def test_a_sliver_poking_out_from_under_the_marker_is_not_a_line_to_measure() -> None:
    """The frame a full render was refused for, 30 minutes in.

    Eight minutes into the hike the walker was still ~50 m from the start, so
    the travelled line was a few pixels long and sat almost entirely under the
    marker disc and the start pin. What showed outside the disc was SIX pixels
    of line colour, the clipped edge of a correctly drawn stroke -- and the
    detector reported them as a 1.0 px hairline. Real frames from the same film
    show 61-141 line pixels in range. A width needs a length to be measured
    along; six pixels is not evidence of a draped-line hairline, it is the
    absence of a line, and the honest answer is the same ``None`` as the
    fully-hidden case above.
    """
    frame = _canvas()
    frame[100, 113:119] = TRAVELLED_RGB  # six px, just outside the 12px marker disc
    assert travelled_line_min_width(frame, MARKER_XY, radius_px=80) is None


def test_a_short_but_real_hairline_is_still_caught() -> None:
    """The floor on evidence must not blind the detector it protects."""
    frame = _canvas()
    frame[100, 113:143] = TRAVELLED_RGB  # 1px wide, 30px long
    width = travelled_line_min_width(frame, MARKER_XY, radius_px=80)
    assert width is not None
    assert width < 2.0


def test_a_line_clipped_by_the_sampling_ring_is_not_a_hairline() -> None:
    """The second render this detector refused, 1841 frames in.

    The measurement used to crop the colour mask to the sampling ring BEFORE
    the distance transform, so the ring's own cut read as background: the
    stroke's clipped end sat at a fake edge and measured one pixel across
    while the same stroke measured 4.6 px everywhere else. Measured on the
    refused frame: 975 line px in range, median ridge width 4.6 px, reported
    width 1.0 px.

    The transform now runs on the whole colour mask and only the RIDGE is
    sampled inside the ring, so a cut edge can no longer invent thinness.
    """
    frame = _canvas(size=400)
    # A 5px-wide stroke running past the sampling ring at both ends, so the
    # ring cuts it where it is at full width.
    frame[:, 198:203] = TRAVELLED_RGB
    width = travelled_line_min_width(frame, (200.0, 200.0), radius_px=80)
    assert width is not None
    assert width >= 4.0


def test_check_frame_flags_a_hairline_travelled_line() -> None:
    frame = _canvas(size=200, base=BASE_RGB)
    _draw_vertical_line(frame, x=100, half_width=0)  # 1px hairline
    result = check_frame(0, frame, marker_xy=MARKER_XY)
    assert not result.ok
    assert result.travelled_line_width_px is not None
    assert result.travelled_line_width_px < 2.0
    assert any("travelled line" in p for p in result.problems)


def test_check_frame_passes_a_3px_travelled_line() -> None:
    frame = _canvas(size=200, base=BASE_RGB)
    _draw_vertical_line(frame, x=100, half_width=1)  # ~3px
    result = check_frame(0, frame, marker_xy=MARKER_XY)
    assert result.ok
    assert result.travelled_line_width_px is not None
    assert result.travelled_line_width_px >= 2.0


def test_check_frame_skips_line_width_check_with_no_marker() -> None:
    frame = _canvas(size=200, base=BASE_RGB)
    _draw_vertical_line(frame, x=100, half_width=0)  # would fail, if checked
    result = check_frame(0, frame)
    assert result.ok
    assert result.travelled_line_width_px is None


def test_summarize_reports_travelled_line_width() -> None:
    results = [
        FrameQaResult(
            frame=0,
            void_fraction=0.0,
            unloaded_fraction=0.0,
            black_fraction=0.0,
            travelled_line_width_px=3.0,
        ),
        FrameQaResult(
            frame=1,
            void_fraction=0.0,
            unloaded_fraction=0.0,
            black_fraction=0.0,
            travelled_line_width_px=1.5,
            problems=("thin",),
        ),
        FrameQaResult(
            frame=2,
            void_fraction=0.0,
            unloaded_fraction=0.0,
            black_fraction=0.0,
            travelled_line_width_px=None,
        ),
    ]
    summary = summarize(results)
    assert summary["travelled_line_width"] == {"worst_px": 1.5, "frame": 1}


def test_summarize_travelled_line_width_none_when_never_measured() -> None:
    results = [FrameQaResult(frame=0, void_fraction=0.0, unloaded_fraction=0.0, black_fraction=0.0)]
    summary = summarize(results)
    assert summary["travelled_line_width"] == {"worst_px": None, "frame": None}
