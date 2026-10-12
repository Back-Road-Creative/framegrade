"""The opening title card must never render wider than the frame.

A render opened on "Appalachian Trail — Burningtown Gap to Rocky
Bald" and burned it as "achian Trail — Burningtown Gap to Rock": the title font
is sized off the frame width alone (``width // 22``) and drawn centred with no
measurement, so a long trip name is clipped at BOTH edges. Eight hours of render
produced an unusable opening, and nothing in the pipeline noticed -- the frame QA
detectors look for void/unloaded/black pixels, not for text running off the edge.

These tests assert the viewer-level property (no title ink on the frame's edge
columns) rather than a font size, so they keep holding if the fitting strategy
changes from shrink to wrap.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story.hud import draw_opening_title

WIDTH = 1920
HEIGHT = 1080
# The title is fully opaque across the middle of the opening; fade maths is
# tested elsewhere, so every case here renders at full strength.
FULLY_VISIBLE = 0.5
REAL_LONG_TITLE = "Appalachian Trail — Burningtown Gap to Rocky Bald"
REAL_SUBTITLE = "16 September 2026"


def _blank() -> Image.Image:
    # Mid-grey, so both the white glyphs and their near-black stroke differ from
    # the background: a clipped title is visible whichever part lands on the edge.
    return Image.new("RGB", (WIDTH, HEIGHT), (128, 128, 128))


def _edge_columns(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(image)
    return pixels[:, 0, :], pixels[:, -1, :]


def _painted(column: np.ndarray) -> int:
    """Pixels in a column that the card drew over the flat background."""
    return int((np.abs(column.astype(np.int16) - 128) > 8).any(axis=-1).sum())


@pytest.mark.parametrize(
    "title",
    [
        REAL_LONG_TITLE,
        # Longer than any real trip name, to prove the fit is computed and not
        # a constant tuned to the one title that exposed the bug.
        "Appalachian Trail — Burningtown Gap to Rocky Bald and Back Again Twice",
    ],
)
def test_long_title_never_reaches_the_frame_edges(title: str) -> None:
    image = _blank()
    draw_opening_title(image, FULLY_VISIBLE, title, REAL_SUBTITLE)
    left, right = _edge_columns(image)
    assert _painted(left) == 0, "title ink reached the left edge -- it is clipped"
    assert _painted(right) == 0, "title ink reached the right edge -- it is clipped"


def test_a_short_title_is_not_shrunk() -> None:
    """Fitting must only ever shrink; an ordinary title keeps its full size."""
    short = _blank()
    draw_opening_title(short, FULLY_VISIBLE, "Rocky Bald", REAL_SUBTITLE)
    long_title = _blank()
    draw_opening_title(long_title, FULLY_VISIBLE, REAL_LONG_TITLE, REAL_SUBTITLE)

    def ink_height(image: Image.Image) -> int:
        pixels = np.asarray(image).astype(np.int16)
        rows = (np.abs(pixels - 128) > 8).any(axis=-1).any(axis=-1)
        return int(rows.sum())

    # The long title, having been shrunk to fit, cannot occupy more vertical
    # band than the short one drawn at full size.
    assert ink_height(short) >= ink_height(long_title)


def test_the_subtitle_also_stays_inside_the_frame() -> None:
    image = _blank()
    draw_opening_title(image, FULLY_VISIBLE, "", "A Subtitle So Long It Would Overrun " * 4)
    left, right = _edge_columns(image)
    assert _painted(left) == 0
    assert _painted(right) == 0


def test_nothing_is_drawn_before_the_fade_starts() -> None:
    image = _blank()
    draw_opening_title(image, 0.0, REAL_LONG_TITLE, REAL_SUBTITLE)
    assert _painted(np.asarray(image)[:, WIDTH // 2, :]) == 0
