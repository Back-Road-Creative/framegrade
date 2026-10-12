"""Regression guard for the IMMERSIVE thumbnail style's legibility.

The immersive style drew white 56px title text with only a 2px black drop-shadow offset — invisible over a bright frame/sky. The fix
gives the glyphs a real black stroke (outline). This test renders the style over
a solid-white background and asserts the title band carries dark stroke pixels;
a regression to white-on-white (no stroke) drops the count to ~0 and fails.
"""

from PIL import Image

from gomoveshift.pipeline.thumbnail_text_overlay import _draw_immersive_style


def _dark_pixel_count(img: Image.Image, box: tuple[int, int, int, int]) -> int:
    px = img.load()
    x0, y0, x1, y1 = box
    n = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            r, g, b = px[x, y][:3]
            if r < 80 and g < 80 and b < 80:
                n += 1
    return n


def test_immersive_title_has_dark_stroke_over_white_bg():
    bg = Image.new("RGB", (1280, 720), (255, 255, 255))
    out = _draw_immersive_style(bg, "Cloonlee to Portumna", "R352 POV")
    # Title is drawn at (title_x=50, title_y=50) with a 56px font.
    dark = _dark_pixel_count(out, (50, 50, 900, 140))
    assert dark > 100, f"immersive title lacks a dark contrast stroke (dark px={dark})"
