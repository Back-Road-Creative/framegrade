"""Red-first tests for the Route Story border-overlay compositor (pure numpy/PIL, no I/O)."""

from __future__ import annotations

import numpy as np
import pytest

from gomoveshift.pipeline.route_story import borders as brd


def _white(h=20, w=20):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _line_overlay(h=20, w=20, col=10, alpha=255):
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[:, col, 3] = alpha
    return overlay


# --- overlay_borders --------------------------------------------------------


def test_overlay_borders_opaque_line_becomes_colour_at_full_opacity_no_halo():
    rgb = _white()
    overlay = _line_overlay()
    colour = (240, 238, 225)

    result = brd.overlay_borders(rgb, overlay, colour=colour, opacity=1.0, halo=False)

    assert result.dtype == np.uint8
    assert result.shape == rgb.shape
    assert tuple(result[0, 10]) == colour
    assert tuple(result[5, 10]) == colour
    # untouched columns stay white
    assert tuple(result[0, 0]) == (255, 255, 255)


def test_overlay_borders_opacity_zero_is_byte_identical():
    rgb = _white()
    overlay = _line_overlay()

    result = brd.overlay_borders(rgb, overlay, opacity=0.0)

    assert np.array_equal(result, rgb)


def test_overlay_borders_all_zero_alpha_is_byte_identical():
    rgb = _white()
    overlay = np.zeros((20, 20, 4), dtype=np.uint8)

    result = brd.overlay_borders(rgb, overlay, opacity=0.8)

    assert np.array_equal(result, rgb)


def test_overlay_borders_halo_darkens_neighbours_but_not_far_pixels():
    rgb = _white()
    overlay = _line_overlay(col=10)

    result = brd.overlay_borders(rgb, overlay, opacity=1.0, halo=True)

    # 1px neighbour of the line is darkened (halo), not full white, not the line colour
    neighbour = result[5, 9]
    assert neighbour[0] < 255
    assert tuple(neighbour) != (240, 238, 225)

    # 3+ px away is untouched
    far = result[5, 7]
    assert tuple(far) == (255, 255, 255)


def test_overlay_borders_resizes_mismatched_overlay_instead_of_rejecting():
    rgb = _white(h=20, w=20)
    overlay = _line_overlay(h=10, w=10, col=5)

    result = brd.overlay_borders(rgb, overlay, opacity=1.0, halo=False)

    assert result.shape == rgb.shape


def test_overlay_borders_rejects_wrong_rgb_dtype():
    rgb = _white().astype(np.float32)
    overlay = _line_overlay()
    with pytest.raises(ValueError):
        brd.overlay_borders(rgb, overlay)


def test_overlay_borders_rejects_wrong_overlay_shape():
    rgb = _white()
    overlay = np.zeros((20, 20, 3), dtype=np.uint8)  # missing alpha channel
    with pytest.raises(ValueError):
        brd.overlay_borders(rgb, overlay)


# --- border_coverage ---------------------------------------------------------


def test_border_coverage_empty_overlay_is_zero():
    overlay = np.zeros((10, 10, 4), dtype=np.uint8)
    assert brd.border_coverage(overlay) == 0.0


def test_border_coverage_half_covered_overlay_is_half():
    overlay = np.zeros((10, 10, 4), dtype=np.uint8)
    overlay[:5, :, 3] = 255
    assert brd.border_coverage(overlay) == pytest.approx(0.5)
