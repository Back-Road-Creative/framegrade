"""Pixel-width OST budgets + font pinning.

``validate_ost`` historically gated burned-OST text on a *codepoint* count
(``len(text)``). Codepoints are only a fair proxy for narrow Latin glyphs:
a string of wide CJK / fullwidth characters can blow past the frame width
while staying under the codepoint cap, and a string of narrow characters
(``"iiii..."``) can fail the cap while fitting comfortably. The pixel budget replaces the
proxy with a real measurement — wrap the text with PIL ``textbbox`` against
the overlay's text-area width and reject only when it cannot fit even after
wrapping (line count over the format's max, or an unspaceable word wider than
the frame). A run of wide glyphs (e.g. capital ``W``) can stay under the
codepoint cap yet render ~2x wider and overflow the frame.

The measurement font MUST be the EXACT ffmpeg ``FONT_PATH`` (DejaVuSans-Bold),
NOT the ``_find_font`` Liberation/Ubuntu fallback chain — otherwise PIL
measures one font while ffmpeg renders another. Both production OST
surfaces (long-form ``ost_applier`` and shorts ``ffmpeg_filters``) hardcode the
same ``FONT_PATH``; this test pins to that constant.
"""

from gomoveshift.pipeline import ffmpeg_filters, ost_applier
from gomoveshift.pipeline.ost_validator import (
    _FORMAT_PIXEL_SPEC,
    measure_wrapped_lines,
    text_fits_overlay,
    validate_ost,
)


class TestFontPinning:
    """PIL must measure against the exact ffmpeg FONT_PATH, not a fallback."""

    def test_validator_font_matches_both_render_surfaces(self):
        # Long-form and shorts render through the same hardcoded DejaVuSans-Bold.
        assert ffmpeg_filters.FONT_PATH == ost_applier.FONT_PATH
        # Every format spec measures against that same font.
        for spec in _FORMAT_PIXEL_SPEC.values():
            assert spec.font_path == ffmpeg_filters.FONT_PATH
            assert "DejaVuSans-Bold" in spec.font_path
            # NOT the Liberation/Ubuntu _find_font fallback chain.
            assert "Liberation" not in spec.font_path
            assert "Ubuntu" not in spec.font_path


class TestMeasureWrappedLines:
    def test_short_latin_label_one_line(self):
        assert measure_wrapped_lines("Asheville, NC", "lf") == 1

    def test_long_latin_label_wraps_within_max(self):
        text = "Cataloochee Valley to Maggie Valley via Blue Ridge Parkway, NC"
        n = measure_wrapped_lines(text, "lf")
        assert 1 < n <= _FORMAT_PIXEL_SPEC["lf"].max_lines


class TestTextFitsOverlay:
    def test_normal_label_fits(self):
        assert text_fits_overlay("Killarney to Dingle via N86, Kerry", "lf") is True

    def test_short_label_fits_shorts(self):
        assert text_fits_overlay("Dingle, Kerry", "shorts") is True

    def test_wide_glyphs_overflow_within_codepoint_budget(self):
        # 50 wide capitals: under the 60-codepoint ``lf`` cap, but each glyph
        # is ~2x the width of an average char, so the rendered run overflows
        # the frame. This is the exact case codepoint budgets miss:
        # measured against the real ffmpeg font, ~50 'W's exceed the text area.
        text = "W" * 50  # unspaceable wide single "word"
        assert len(text) < 60  # passes the old codepoint gate
        assert text_fits_overlay(text, "lf") is False

    def test_extremely_long_label_overflows(self):
        text = "A to B " + ("Verylongplacename " * 8)
        assert text_fits_overlay(text, "lf") is False


class TestValidateOstPixelGate:
    def test_pixel_overflow_rejected_under_codepoint_cap(self):
        text = "W" * 50
        assert len(text) <= 60  # passes the codepoint cap; must fail on pixels
        ok, reason = validate_ost(text, "lf")
        assert ok is False
        assert (
            "pixel" in reason.lower() or "frame" in reason.lower() or "overflow" in reason.lower()
        )

    def test_normal_label_still_passes(self):
        ok, _ = validate_ost("Lake Lure to Oakland, NC", "lf_title")
        assert ok is True

    def test_shorts_label_pixel_gated(self):
        ok, _ = validate_ost("Dingle, Kerry", "shorts")
        assert ok is True

    def test_codepoint_garbage_checks_still_run(self):
        # Pixel gate must not bypass the existing garbage-pattern checks.
        ok, reason = validate_ost("None, None", "lf")
        assert ok is False
        assert "None" in reason or "none" in reason
