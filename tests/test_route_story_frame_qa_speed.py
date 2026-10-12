"""Frame QA does its per-frame work once, on the pixels it asks about -- and
still returns exactly what the full-frame implementation returned.

A render profile once put ``check_frame`` at 453 ms of a ~1 s film
frame: the travelled-line colour mask and a full-frame distance grid were built
twice per frame (once per line check) over all 1920x923 pixels, to answer a
question about the 200 px around the marker. The reference functions below are
that implementation, frozen, so the speedup can never change a verdict.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import maximum_filter

import cv2
from gomoveshift.pipeline.route_story import frame_qa
from gomoveshift.pipeline.route_story.frame_qa import (
    FLAT_BLACK_MAX,
    PAGE_VOID_RGB,
    TRAVELLED_LINE_RGB,
    UNLOADED_RGB,
    FrameQaThresholds,
    check_frame,
    line_end_to_marker_px,
    travelled_line_min_width,
)

FILM_HEIGHT, FILM_WIDTH = 923, 1920  # the scene map area of a 1080p film frame


# --- the full-frame implementation, frozen as the reference -------------------


def _ref_mask(rgb, marker_xy, colour, tolerance):
    height, width = rgb.shape[0], rgb.shape[1]
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.sqrt((xx - marker_xy[0]) ** 2 + (yy - marker_xy[1]) ** 2)
    diff = np.abs(rgb.astype(np.int16) - np.array(colour, dtype=np.int16))
    return (diff <= tolerance).all(axis=-1), dist


def _ref_line_end(rgb, marker_xy, radius_px=200.0, tolerance=40, exclude=12.0):
    mask, dist = _ref_mask(rgb, marker_xy, TRAVELLED_LINE_RGB, tolerance)
    mask = mask & (dist >= exclude) & (dist <= radius_px)
    if not mask.any():
        return None
    return float(dist[mask].min()) - exclude


def _ref_min_width(rgb, marker_xy, radius_px=200.0, tolerance=40, exclude=12.0, end=6.0):
    mask, dist = _ref_mask(rgb, marker_xy, TRAVELLED_LINE_RGB, tolerance)
    in_ring = mask & (dist >= exclude) & (dist <= radius_px - end)
    if not in_ring.any():
        return None
    transform = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    ridge = (transform == maximum_filter(transform, size=5)) & in_ring & (transform > 0)
    if int(ridge.sum()) < 8:
        return None
    return float(np.percentile(transform[ridge], 5) * 2.0 - 1.0)


def _ref_fractions(rgb, tolerance, hud_mask):
    valid = ~hud_mask if hud_mask is not None else np.ones(rgb.shape[:2], dtype=bool)
    total = int(valid.sum())

    def near(target):
        diff = np.abs(rgb.astype(np.int16) - np.array(target, dtype=np.int16))
        return float(((diff <= tolerance).all(axis=-1) & valid).sum()) / total

    black = float(((rgb.astype(np.int16) <= FLAT_BLACK_MAX).all(axis=-1) & valid).sum()) / total
    return near(PAGE_VOID_RGB), near(UNLOADED_RGB), black


# --- frames that look like the film ----------------------------------------------


def _scene_frame(rng: np.random.Generator) -> tuple[np.ndarray, tuple[float, float]]:
    frame = rng.integers(0, 256, (FILM_HEIGHT, FILM_WIDTH, 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (0, 0), 3)
    marker = (float(rng.uniform(-30, FILM_WIDTH + 30)), float(rng.uniform(-30, FILM_HEIGHT + 30)))
    # The travelled line: a wandering stroke that ends near the marker.
    points = [np.array(marker) + rng.normal(0, 8, 2)]
    for _ in range(int(rng.integers(3, 12))):
        points.append(points[-1] + rng.normal(0, 90, 2))
    cv2.polylines(
        frame,
        [np.round(np.array(points)).astype(np.int32)],
        False,
        TRAVELLED_LINE_RGB,
        int(rng.integers(1, 14)),
        cv2.LINE_AA if rng.random() < 0.5 else cv2.LINE_8,
    )
    # Line-coloured clutter: foliage, a far stretch of line, a big bright patch.
    for _ in range(int(rng.integers(0, 4))):
        centre = (int(rng.uniform(0, FILM_WIDTH)), int(rng.uniform(0, FILM_HEIGHT)))
        tint = tuple(int(c + rng.integers(-35, 36)) for c in TRAVELLED_LINE_RGB)
        tint = tuple(max(0, min(255, c)) for c in tint)
        cv2.circle(frame, centre, int(rng.integers(3, 60)), tint, -1)
    # The detectors' own colours, so the fractions are not all zero.
    for colour in (PAGE_VOID_RGB, UNLOADED_RGB, (4, 5, 3)):
        if rng.random() < 0.6:
            y, x = int(rng.uniform(0, FILM_HEIGHT - 40)), int(rng.uniform(0, FILM_WIDTH - 40))
            frame[y : y + int(rng.integers(1, 40)), x : x + int(rng.integers(1, 40))] = colour
    return frame, marker


@pytest.mark.parametrize("seed", range(40))
def test_check_frame_matches_the_full_frame_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    frame, marker = _scene_frame(rng)
    hud_mask = None
    if seed % 3 == 0:
        hud_mask = np.zeros(frame.shape[:2], dtype=bool)
        hud_mask[-120:, :] = True

    result = check_frame(seed, frame, FrameQaThresholds(), hud_mask=hud_mask, marker_xy=marker)

    void, unloaded, black = _ref_fractions(frame, FrameQaThresholds().tolerance, hud_mask)
    assert result.void_fraction == void
    assert result.unloaded_fraction == unloaded
    assert result.black_fraction == black
    assert result.line_end_to_marker_px == _ref_line_end(frame, marker)
    assert result.travelled_line_width_px == _ref_min_width(frame, marker)


@pytest.mark.parametrize("seed", range(10))
def test_public_line_checks_match_the_reference_at_other_settings(seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    frame, marker = _scene_frame(rng)
    radius, tolerance = float(rng.uniform(20, 400)), int(rng.integers(5, 60))
    assert line_end_to_marker_px(
        frame, marker, radius_px=radius, tolerance=tolerance
    ) == _ref_line_end(frame, marker, radius_px=radius, tolerance=tolerance)
    assert travelled_line_min_width(
        frame, marker, radius_px=radius, tolerance=tolerance
    ) == _ref_min_width(frame, marker, radius_px=radius, tolerance=tolerance)


def test_a_marker_off_the_frame_measures_nothing() -> None:
    frame = np.zeros((FILM_HEIGHT, FILM_WIDTH, 3), dtype=np.uint8)
    frame[:, :40] = TRAVELLED_LINE_RGB
    far = (-5000.0, 400.0)
    assert line_end_to_marker_px(frame, far) is None is _ref_line_end(frame, far)
    assert travelled_line_min_width(frame, far) is None is _ref_min_width(frame, far)


# --- the work is done once, on the window it is about --------------------------


def _film_frame_with_line() -> tuple[np.ndarray, tuple[float, float]]:
    frame = np.full((FILM_HEIGHT, FILM_WIDTH, 3), (60, 90, 50), dtype=np.uint8)
    marker = (960.0, 460.0)
    cv2.line(frame, (960, 460), (700, 300), TRAVELLED_LINE_RGB, 7)
    return frame, marker


def test_check_frame_builds_the_line_colour_mask_once(monkeypatch) -> None:
    calls = []
    original = frame_qa._travelled_line_colour_mask

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(frame_qa, "_travelled_line_colour_mask", spy)
    frame, marker = _film_frame_with_line()
    check_frame(0, frame, marker_xy=marker)
    assert len(calls) == 1


def test_the_line_distance_field_covers_the_marker_window_not_the_frame(monkeypatch) -> None:
    shapes = []
    original = cv2.distanceTransform

    def spy(src, *args, **kwargs):
        shapes.append(src.shape)
        return original(src, *args, **kwargs)

    monkeypatch.setattr(frame_qa.cv2, "distanceTransform", spy)
    frame, marker = _film_frame_with_line()
    check_frame(0, frame, marker_xy=marker)
    assert shapes, "the width check never ran"
    for height, width in shapes:
        assert height * width < 0.25 * FILM_HEIGHT * FILM_WIDTH
