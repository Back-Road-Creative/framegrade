"""HUD font loading is cached — the TrueType loader runs at most once per (size, bold)."""

from unittest.mock import patch

from PIL import ImageFont

from gomoveshift.pipeline.route_story import hud as hud_module


def setup_function(_function) -> None:
    hud_module._font.cache_clear()


def test_font_loader_called_once_for_repeated_args() -> None:
    with patch.object(ImageFont, "truetype", wraps=ImageFont.truetype) as spy:
        first = hud_module._font(24, bold=False)
        second = hud_module._font(24, bold=False)

    assert spy.call_count == 1
    assert first is second


def test_font_cache_is_keyed_on_size_and_bold() -> None:
    with patch.object(ImageFont, "truetype", wraps=ImageFont.truetype) as spy:
        regular = hud_module._font(24, bold=False)
        bold = hud_module._font(24, bold=True)
        other_size = hud_module._font(30, bold=False)

    assert spy.call_count == 3
    assert regular is not bold
    assert regular is not other_size
