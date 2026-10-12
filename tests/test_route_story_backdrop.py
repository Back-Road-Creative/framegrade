"""Tests for the Blue Marble backdrop grading recipe."""

from __future__ import annotations

import numpy as np
import pytest

from gomoveshift.pipeline.route_story.backdrop import grade_backdrop

# A flat raster at the measured dark GIBS Blue Marble mean (37-47 of 255).
_DARK_LAND_OCEAN_RGB = np.array(
    [
        [[37, 47, 60], [30, 60, 90]],
        [[60, 50, 40], [20, 30, 50]],
    ],
    dtype=np.uint8,
)


def test_grade_backdrop_preserves_dtype_and_shape() -> None:
    out = grade_backdrop(_DARK_LAND_OCEAN_RGB)
    assert out.dtype == np.uint8
    assert out.shape == _DARK_LAND_OCEAN_RGB.shape


def test_grade_backdrop_lifts_mean_luma_into_target_band() -> None:
    before = float(_DARK_LAND_OCEAN_RGB.mean())
    out = grade_backdrop(_DARK_LAND_OCEAN_RGB)
    after = float(out.mean())
    assert before < 60  # sanity: this fixture really is dark
    # A vivid opening backdrop needs real lift, but must not blow out to
    # near-white either -- the measured recipe lands comfortably mid-tone.
    assert 1.4 * before < after < 220


def test_grade_backdrop_preserves_hue_order() -> None:
    """A pixel's channel rank order (which of r/g/b is brightest) survives
    grading -- the recipe reshapes contrast, it does not shift hue."""
    out = grade_backdrop(_DARK_LAND_OCEAN_RGB)
    before_order = np.argsort(_DARK_LAND_OCEAN_RGB.astype(np.int16), axis=-1)
    after_order = np.argsort(out.astype(np.int16), axis=-1)
    assert np.array_equal(before_order, after_order)


def test_grade_backdrop_pure_black_stays_near_black() -> None:
    black = np.zeros((2, 2, 3), dtype=np.uint8)
    out = grade_backdrop(black)
    assert int(out.max()) <= 5


def test_grade_backdrop_never_overflows_a_channel() -> None:
    near_white = np.full((2, 2, 3), 250, dtype=np.uint8)
    out = grade_backdrop(near_white)
    assert out.min() >= 0
    assert out.max() <= 255


def test_grade_backdrop_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        grade_backdrop(np.zeros((4, 4), dtype=np.uint8))
