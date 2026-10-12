"""Tests for SinglePassProcessor — location overlay branding."""

import os
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.single_pass import SinglePassProcessor, _clean_location


class TestCleanLocation:
    """The unconditional ' Scenic Drive' suffix is not added —
    'scenic' is an unverifiable visual claim; the overlay burns the factual
    location only (country suffix still stripped)."""

    def test_clean_location_strips_united_states(self):
        assert _clean_location("South Carolina, United States") == "South Carolina"

    def test_clean_location_strips_ireland(self):
        assert _clean_location("County Kerry, Ireland") == "County Kerry"

    def test_clean_location_strips_uk(self):
        assert _clean_location("Snowdonia, UK") == "Snowdonia"

    def test_clean_location_strips_united_kingdom(self):
        assert _clean_location("Yorkshire Dales, United Kingdom") == "Yorkshire Dales"

    def test_clean_location_strips_canada(self):
        assert _clean_location("Banff, Canada") == "Banff"

    def test_clean_location_strips_australia(self):
        assert _clean_location("Blue Mountains, Australia") == "Blue Mountains"

    def test_clean_location_strips_new_zealand(self):
        assert _clean_location("Fiordland, New Zealand") == "Fiordland"

    def test_clean_location_no_suffix(self):
        # No country suffix and no genre word appended — location shown as-is.
        assert _clean_location("Western North Carolina") == "Western North Carolina"

    def test_clean_location_partial_match_not_stripped(self):
        """Only strip if the location ends with the suffix — not a partial match."""
        result = _clean_location("United States Route 1")
        assert result == "United States Route 1"

    # ── Route-based OST text ─────────────────────────────

    def test_route_via_no_scenic_drive(self):
        """Route descriptions with 'to...via' are already self-descriptive."""
        result = _clean_location("Port Royal to Gardens Corner via Trask Parkway")
        assert result == "Port Royal to Gardens Corner via Trask Parkway"
        assert "Scenic Drive" not in result

    def test_route_two_towns_no_scenic_drive(self):
        """'Town A to Town B' format should not get Scenic Drive appended."""
        result = _clean_location("Frogmore to Port Royal")
        assert result == "Frogmore to Port Royal"

    def test_route_with_country_strips_country_no_scenic_drive(self):
        """Route with country suffix: strip country but don't add Scenic Drive."""
        result = _clean_location("Cork to Killarney via N22, Ireland")
        assert result == "Cork to Killarney via N22"

    def test_route_containing_drive_in_road_name(self):
        """Road names containing 'Drive' should not cause false positives."""
        result = _clean_location("Sunny View to Oakland via Hunting Island Drive")
        assert result == "Sunny View to Oakland via Hunting Island Drive"

    def test_simple_road_town_no_scenic_drive(self):
        """Simple 'Road, Town' format is shown as-is — no genre word appended."""
        result = _clean_location("Seaside Road, Longwood")
        assert result == "Seaside Road, Longwood"
        assert "Scenic Drive" not in result

    def test_road_state_no_scenic_drive(self):
        """'Road, State' format is shown as-is — no genre word appended."""
        result = _clean_location("US-21, South Carolina")
        assert result == "US-21, South Carolina"
        assert "Scenic Drive" not in result


class TestLocationOverlayBranding:
    def test_overlay_contains_gomoveshift_brand(self):
        """The filter chain must include GoMoveShift brand text."""
        from gomoveshift.config import Config

        config = Config()
        proc = SinglePassProcessor(config)

        with patch("os.path.exists", return_value=True):
            result = proc._build_video_filter(
                color_filter="null",
                location="South Carolina, United States",
                font_path="/fake/font.ttf",
                duration=600.0,
                video_width=3840,
            )

        assert "GoMoveShift" in result

    def test_overlay_contains_location_no_scenic_drive(self):
        """The filter chain must include the location line, with no genre word."""
        from gomoveshift.config import Config

        config = Config()
        proc = SinglePassProcessor(config)

        with patch("os.path.exists", return_value=True):
            result = proc._build_video_filter(
                color_filter="null",
                location="County Kerry, Ireland",
                font_path="/fake/font.ttf",
                duration=600.0,
                video_width=3840,
            )

        assert "County Kerry" in result
        assert "Scenic Drive" not in result

    def test_overlay_gomoveshift_fontsize_scales_at_4k(self):
        """GoMoveShift brand line scales from 64 to 96 at 4K (1.5x cap)."""
        from gomoveshift.config import Config

        config = Config()
        proc = SinglePassProcessor(config)

        with patch("os.path.exists", return_value=True):
            result = proc._build_video_filter(
                color_filter="null",
                location="South Carolina, United States",
                font_path="/fake/font.ttf",
                duration=600.0,
                video_width=3840,
            )

        # GoMoveShift drawtext entry at 3840 should be 64*1.5=96
        parts = result.split(",drawtext=")
        gms_part = next((p for p in parts if "GoMoveShift" in p), None)
        assert gms_part is not None, "GoMoveShift drawtext not found"
        assert "fontsize=96" in gms_part

    def test_overlay_location_fontsize_scales_at_4k(self):
        """Location line scales from 48 to 72 at 4K (1.5x cap)."""
        from gomoveshift.config import Config

        config = Config()
        proc = SinglePassProcessor(config)

        with patch("os.path.exists", return_value=True):
            result = proc._build_video_filter(
                color_filter="null",
                location="South Carolina, United States",
                font_path="/fake/font.ttf",
                duration=600.0,
                video_width=3840,
            )

        parts = result.split(",drawtext=")
        loc_part = next((p for p in parts if "South Carolina" in p), None)
        assert loc_part is not None, "location drawtext not found"
        assert "fontsize=72" in loc_part

    def test_overlay_route_text_no_scenic_drive(self):
        """Route-based OST should NOT have 'Scenic Drive' appended in overlay."""
        from gomoveshift.config import Config

        config = Config()
        proc = SinglePassProcessor(config)

        with patch("os.path.exists", return_value=True):
            result = proc._build_video_filter(
                color_filter="null",
                location="Port Royal to Gardens Corner via Trask Parkway",
                font_path="/fake/font.ttf",
                duration=600.0,
                video_width=3840,
            )

        assert "Scenic Drive" not in result
        assert "Trask Parkway" in result

    def test_overlay_wraps_long_location(self):
        """Very long location text should produce multiple drawtext lines."""
        from gomoveshift.config import Config

        config = Config()
        proc = SinglePassProcessor(config)

        # Force narrow width to trigger wrapping
        with patch("os.path.exists", return_value=True):
            result = proc._build_video_filter(
                color_filter="null",
                location="Fort Fremont to Oceanmarsh Subdivision via Seaside Road",
                font_path="/fake/font.ttf",
                duration=600.0,
                video_width=800,  # narrow to force wrap
            )

        # With 800px width, the ~55-char location should wrap to 2 lines
        parts = result.split(",drawtext=")
        # Find all drawtext parts that are NOT GoMoveShift and NOT Subscribe
        location_parts = [
            p
            for p in parts
            if "GoMoveShift" not in p and "Subscribe" not in p and "fontsize=48" in p
        ]
        assert len(location_parts) >= 2, (
            f"Expected wrapped location (2+ drawtext entries), got {len(location_parts)}"
        )


class TestWrapTextDuplicateWords:
    """Regression: _wrap_text used words.index(word) which finds the FIRST
    occurrence, garbling output when words repeat (e.g. 'Cork, Cork')."""

    def _make_processor(self):
        from gomoveshift.config import Config

        return SinglePassProcessor(Config())

    def test_duplicate_city_name_preserved(self):
        """'New York, New York, United States' must not reorder/garble words (the
        words.index bug). Words stay in input order; a trailing ellipsis only
        marks width-truncation of the final line, never reordering."""
        proc = self._make_processor()
        # Use a narrow width so wrapping is forced (fontsize=48, width=300).
        lines = proc._wrap_text("New York, New York, United States", fontsize=48, max_width_px=300)
        kept = " ".join(lines).rstrip("…").split()
        original = "New York, New York, United States".split()
        # Whatever survives is an in-order prefix — both "New York," occurrences
        # preserved, never garbled to e.g. "New New York,".
        assert kept == original[: len(kept)], f"wrap_text garbled duplicate words: {lines!r}"
        assert kept[:4] == ["New", "York,", "New", "York,"]

    def test_cork_cork_preserved(self):
        """'Cork, Cork, Ireland' must keep both occurrences of Cork."""
        proc = self._make_processor()
        lines = proc._wrap_text("Cork, Cork, Ireland", fontsize=48, max_width_px=300)
        joined = " ".join(lines)
        assert joined == "Cork, Cork, Ireland", f"wrap_text garbled duplicate words: {lines!r}"

    def test_no_wrap_needed(self):
        """Short text that fits on one line should return a single element."""
        proc = self._make_processor()
        lines = proc._wrap_text("Hello World", fontsize=48, max_width_px=2000)
        assert lines == ["Hello World"]

    def test_wrap_respects_max_lines(self):
        """Output must never exceed max_lines. When the text can't fit, the final
        line is truncated with an ellipsis rather than cramming every remaining
        word onto one off-frame line."""
        proc = self._make_processor()
        text = "one two three four five six seven eight nine ten eleven twelve"
        lines = proc._wrap_text(text, fontsize=48, max_width_px=200, max_lines=2)
        assert len(lines) <= 2
        joined = " ".join(lines)
        # Overflow is dropped + ellipsis-marked; surviving words are an in-order
        # prefix of the input (never reordered or off-frame).
        assert joined.endswith("…")
        kept = joined.rstrip("…").split()
        assert kept == text.split()[: len(kept)]


class TestWrapTextPILFontMetrics:
    """PIL-backed pixel-accurate text measurement for proportional fonts.

    The old approximation (avg_char_width = fontsize * 0.55) overflows on
    variable-width glyphs. The PIL path uses ImageDraw.textbbox() so every
    line's actual pixel width stays at or under max_width_px.
    """

    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    def _make_processor(self):
        from gomoveshift.config import Config

        return SinglePassProcessor(Config())

    @pytest.mark.skipif(
        not os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        reason="DejaVu Sans Bold not installed",
    )
    def test_long_location_no_pixel_overflow(self):
        """Each wrapped line must measure <= max_width_px under the actual font."""
        pytest.importorskip("PIL")
        from PIL import Image, ImageDraw, ImageFont

        proc = self._make_processor()
        text = "Longwood to Highlands, North Carolina"
        fontsize = 48
        max_width_px = 800
        lines = proc._wrap_text(text, fontsize=fontsize, max_width_px=max_width_px, max_lines=2)

        # All words preserved (no dropping/reordering)
        assert " ".join(lines) == text, f"words mangled: {lines!r}"
        # Must fit within max_lines
        assert 1 <= len(lines) <= 2

        # Verify each line's measured pixel width fits the budget.
        font = ImageFont.truetype(self.FONT_PATH, fontsize)
        img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(img)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            width = bbox[2] - bbox[0]
            assert width <= max_width_px, (
                f"line {line!r} measured {width}px, exceeds max {max_width_px}px"
            )

    @pytest.mark.skipif(
        not os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        reason="DejaVu Sans Bold not installed",
    )
    def test_wide_glyph_text_does_not_overflow(self):
        """Text dominated by wide glyphs ('W', 'M') must still fit after wrap."""
        pytest.importorskip("PIL")
        from PIL import Image, ImageDraw, ImageFont

        proc = self._make_processor()
        # Wide glyphs: char-average approximation typically underestimates width.
        text = "WWW MMM WWW MMM WWW MMM WWW MMM"
        fontsize = 48
        max_width_px = 800
        lines = proc._wrap_text(text, fontsize=fontsize, max_width_px=max_width_px, max_lines=3)

        font = ImageFont.truetype(self.FONT_PATH, fontsize)
        img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(img)
        # Only assert lines that fit naturally (not the forced last-line dump).
        for line in lines[:-1]:
            bbox = draw.textbbox((0, 0), line, font=font)
            width = bbox[2] - bbox[0]
            assert width <= max_width_px, (
                f"line {line!r} measured {width}px, exceeds max {max_width_px}px"
            )

    def test_fallback_when_font_missing(self):
        """When the font can't be resolved, falls back to char-width approximation."""
        proc = self._make_processor()
        # Point _find_font at a nonexistent path by monkey-patching the instance.
        proc._find_font = lambda _=None: None  # type: ignore[assignment]
        text = "Longwood to Highlands, North Carolina"
        lines = proc._wrap_text(text, fontsize=48, max_width_px=800, max_lines=2)
        assert " ".join(lines) == text
        assert 1 <= len(lines) <= 2
