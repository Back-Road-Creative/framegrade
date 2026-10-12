"""Tests for resolution-aware font scaling in SinglePassProcessor.

The GoMoveShift brand (64px), location (48px) and website CTA (48px)
text are sized for the 1920px reference resolution. When rendering at
4K (3840px), those sizes appear tiny in playback. The _build_video_filter
method must scale fonts proportionally to video_width, with:
  * floor of 1.0x  (no shrinking below reference)
  * ceiling of 1.5x (cap to avoid overwhelming text)

Targets:
  * 1920 wide → base (64 / 48 / 48)
  * 2560 wide → ~1.33x (≈85 / 64 / 64)
  * 3840 wide → 1.5x   (96 / 72 / 72)
  * 1080 wide → floored at 1.0x (64 / 48 / 48)
"""

from unittest.mock import patch

from gomoveshift.config import Config
from gomoveshift.pipeline.single_pass import SinglePassProcessor


def _build(processor: SinglePassProcessor, width: int) -> str:
    with patch("os.path.exists", return_value=True):
        return processor._build_video_filter(
            color_filter="null",
            # Short location that never wraps at any test resolution
            location="Beaufort, SC",
            font_path="/fake/font.ttf",
            duration=600.0,
            video_width=width,
        )


def _get_drawtext_part(filter_str: str, contains: str) -> str:
    parts = filter_str.split(",drawtext=")
    match = next((p for p in parts if contains in p), None)
    assert match is not None, f"{contains!r} drawtext not found in filter"
    return match


class TestFontScaling:
    def test_base_1920_width(self):
        proc = SinglePassProcessor(Config())
        f = _build(proc, 1920)
        gms = _get_drawtext_part(f, "GoMoveShift")
        loc = _get_drawtext_part(f, "Beaufort")
        assert "fontsize=64" in gms
        assert "fontsize=48" in loc

    def test_4k_3840_width_hits_cap(self):
        proc = SinglePassProcessor(Config())
        f = _build(proc, 3840)
        gms = _get_drawtext_part(f, "GoMoveShift")
        loc = _get_drawtext_part(f, "Beaufort")
        # 1.5x cap: 64*1.5=96, 48*1.5=72
        assert "fontsize=96" in gms
        assert "fontsize=72" in loc

    def test_2560_width_scales_proportionally(self):
        proc = SinglePassProcessor(Config())
        f = _build(proc, 2560)
        gms = _get_drawtext_part(f, "GoMoveShift")
        loc = _get_drawtext_part(f, "Beaufort")
        # 2560/1920 ≈ 1.333 → 64*1.333≈85, 48*1.333≈64
        assert "fontsize=85" in gms
        assert "fontsize=64" in loc

    def test_narrow_width_floored_to_reference(self):
        """Below reference width, sizes are NOT shrunk (floor = 1.0x)."""
        proc = SinglePassProcessor(Config())
        f = _build(proc, 1080)
        gms = _get_drawtext_part(f, "GoMoveShift")
        loc = _get_drawtext_part(f, "Beaufort")
        assert "fontsize=64" in gms
        assert "fontsize=48" in loc

    def test_website_cta_also_scales(self):
        proc = SinglePassProcessor(Config())
        f = _build(proc, 3840)
        ws = _get_drawtext_part(f, "gomoveshift.com")
        assert "fontsize=72" in ws
