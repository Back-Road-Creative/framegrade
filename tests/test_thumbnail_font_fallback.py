"""Regression guard for the thumbnail font fallback losing the requested size.

When no TrueType font on the box could be loaded, the
overlay fell back to an unsized ``ImageFont.load_default()``. On Pillow 12 that
returns a *FreeTypeFont at size 10* — so the thumbnail silently shipped ~10px
title text instead of the requested 72/85px, and an ``isinstance(..., FreeTypeFont)``
guard would happily pass it. These tests therefore assert the **rendered size**,
never the font type.
"""

import pytest
from PIL import ImageFont

from gomoveshift.pipeline import thumbnail_text_overlay as tto


def _rendered_height(font) -> int:
    """Actual drawn height of a tall+descending sample, in pixels."""
    bbox = font.getbbox("Ag")
    return bbox[3] - bbox[1]


@pytest.fixture
def no_system_fonts(monkeypatch):
    """Force _get_font down its ultimate-fallback path."""
    monkeypatch.setattr(tto, "WINDOWS_FONTS", [])
    monkeypatch.setattr(tto, "LINUX_FONTS", [])
    # The canonical burn font is searched ahead of both platform lists.
    monkeypatch.setattr(tto, "FONT_PATH", "/nonexistent/canonical.ttf")


@pytest.mark.parametrize("size", [40, 56, 72, 85])
def test_fallback_font_keeps_requested_size(no_system_fonts, size):
    """The fallback must render near the requested size, not ~10px."""
    font = tto._get_font(size, bold=True)
    height = _rendered_height(font)
    assert height >= size * 0.5, (
        f"fallback font requested {size}px but renders {height}px tall — unreadable thumbnail text"
    )


def test_fallback_raises_when_size_is_dropped(no_system_fonts, monkeypatch):
    """A future Pillow that ignores size= must hard-fail, not ship tiny text.

    Simulates that by making load_default() return the unsized (10px) default
    regardless of the kwarg — exactly the shape of the original bug.
    """
    # Capture the real 10px default BEFORE patching: tto.ImageFont *is* the
    # global PIL.ImageFont module, so patching it would recurse otherwise.
    tiny = ImageFont.load_default()
    monkeypatch.setattr(tto.ImageFont, "load_default", lambda *a, **kw: tiny)
    with pytest.raises(RuntimeError, match="size"):
        tto._get_font(72, bold=True)
