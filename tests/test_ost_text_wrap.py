"""Tests for OST text wrapping in shorts_extractor.

Validates that long location text wraps to multiple lines
rather than overflowing the 1080px portrait frame, and that
CTA filters (YouTube cross-promo, website) are generated correctly.
"""

from gomoveshift.pipeline.ffmpeg_filters import (
    FONT_PATH,
    LOC_LINE_HEIGHT,
    build_ost_filter,
    shorten_ost_for_shorts,
    wrap_text_for_ffmpeg,
)


class TestWrapText:
    """Test wrap_text_for_ffmpeg greedy word-wrap algorithm."""

    def test_short_text_single_line(self):
        """Short text that fits should return a single line."""
        lines = wrap_text_for_ffmpeg("Ireland", 48, 1000, FONT_PATH)
        assert len(lines) == 1
        assert lines[0] == "Ireland"

    def test_long_text_wraps_to_two_lines(self):
        """Kerry location text should wrap to 2 lines."""
        text = "Killorglin to Kenmare, County Kerry, Ireland"
        lines = wrap_text_for_ffmpeg(text, 48, 1000, FONT_PATH)
        assert len(lines) == 2
        # All original words should be present
        rejoined = " ".join(lines)
        assert rejoined == text

    def test_very_long_text_wraps_to_three_plus_lines(self):
        """Very long text should wrap to 3+ lines."""
        text = "From the Beautiful Town of Killorglin to the Scenic Village of Kenmare, County Kerry, Republic of Ireland"
        lines = wrap_text_for_ffmpeg(text, 48, 1000, FONT_PATH)
        assert len(lines) >= 3
        # All words preserved
        assert " ".join(lines) == text

    def test_single_long_word_no_mid_word_break(self):
        """A single word exceeding max_width stays on one line (no mid-word break)."""
        text = "Superlongplacenamethatexceedsanyreasonablewidth"
        lines = wrap_text_for_ffmpeg(text, 48, 200, FONT_PATH)
        assert len(lines) == 1
        assert lines[0] == text

    def test_empty_text(self):
        """Empty text should return single empty line."""
        lines = wrap_text_for_ffmpeg("", 48, 1000, FONT_PATH)
        assert len(lines) == 1

    def test_exact_fit_no_wrap(self):
        """Text that exactly fits should not wrap."""
        text = "Short"
        lines = wrap_text_for_ffmpeg(text, 48, 1000, FONT_PATH)
        assert len(lines) == 1
        assert lines[0] == text


class TestBuildOstFilter:
    """Test build_ost_filter generates correct drawtext filter chains."""

    def test_short_text_three_drawtext_filters(self):
        """Short location + YouTube CTA + website -> 3 drawtext filters."""
        result = build_ost_filter("Ireland", "", 30.0)
        filters = result.split(",drawtext=")
        # 1 loc + 1 youtube CTA + 1 website
        assert len(filters) == 3

    def test_long_text_more_drawtext_filters(self):
        """Long location text should produce additional drawtext filters."""
        result = build_ost_filter(
            "Killorglin to Kenmare, County Kerry, Ireland",
            "",
            30.0,
        )
        filters = result.split(",drawtext=")
        # Should be 2 loc lines + 2 CTA filters = 4 filters
        assert len(filters) >= 4

    def test_all_location_lines_share_alpha(self):
        """All location drawtext filters should have the same alpha expression."""
        result = build_ost_filter(
            "Killorglin to Kenmare, County Kerry, Ireland",
            "",
            30.0,
        )
        # Parse individual drawtext blocks
        parts = result.split(",drawtext=")
        # Prefix first one back
        parts[0] = parts[0].replace("drawtext=", "", 1)

        # Extract alpha from location lines (all except last 2 which are CTAs)
        loc_alphas = []
        for part in parts[:-2]:  # exclude youtube CTA and website
            alpha_start = part.find("alpha='") + 7
            alpha_end = part.find("'", alpha_start)
            loc_alphas.append(part[alpha_start:alpha_end])

        # All location lines must share the same alpha
        assert len(set(loc_alphas)) == 1

    def test_y_positions_sequential(self):
        """Location lines should have sequential y positions."""
        result = build_ost_filter(
            "Killorglin to Kenmare, County Kerry, Ireland",
            "",
            30.0,
        )
        parts = result.split(",drawtext=")
        parts[0] = parts[0].replace("drawtext=", "", 1)

        # Extract y values from location lines (all except last 2 CTAs)
        y_values = []
        for part in parts[:-2]:
            y_start = part.find(":y=") + 3
            y_end = part.find(":", y_start)
            y_values.append(int(part[y_start:y_end]))

        # Should start at 280 and increment by LOC_LINE_HEIGHT
        assert y_values[0] == 280
        for i in range(1, len(y_values)):
            assert y_values[i] == 280 + i * LOC_LINE_HEIGHT

    def test_youtube_cta_present(self):
        """YouTube CTA should be present in filter chain for shorts."""
        result = build_ost_filter("Ireland", "", 30.0)
        assert "Full drive on YouTube" in result

    def test_website_cta_present(self):
        """Website CTA should be present in filter chain."""
        result = build_ost_filter("Ireland", "", 30.0)
        assert "gomoveshift.com" in result

    def test_returns_comma_joined_string(self):
        """Result should be a valid comma-joined filter chain."""
        result = build_ost_filter("Ireland", "", 30.0)
        assert isinstance(result, str)
        assert "drawtext=" in result
        # No empty segments
        for segment in result.split(","):
            assert segment.strip()


class TestShortenOstForShorts:
    """Test shorts-specific text shortening for mobile readability."""

    def test_strips_via_clause(self):
        """Route with 'via' should drop the via clause for shorts."""
        result = shorten_ost_for_shorts("Port Royal to Gardens Corner via Trask Parkway")
        assert result == "Port Royal to Gardens Corner"

    def test_preserves_two_town_route(self):
        """'Town A to Town B' without via should be kept as-is."""
        result = shorten_ost_for_shorts("Frogmore to Port Royal")
        assert result == "Frogmore to Port Royal"

    def test_preserves_simple_location(self):
        """Simple 'Road, Town' text should pass through unchanged."""
        result = shorten_ost_for_shorts("Seaside Road, Longwood")
        assert result == "Seaside Road, Longwood"

    def test_preserves_short_text(self):
        """Already-short text should not be modified."""
        result = shorten_ost_for_shorts("Ireland")
        assert result == "Ireland"

    def test_truncates_very_long_text(self):
        """Text over 40 chars (after via strip) should be truncated."""
        result = shorten_ost_for_shorts("Fort Fremont to Oceanmarsh Subdivision and Beyond")
        assert len(result) <= 40

    def test_empty_string(self):
        result = shorten_ost_for_shorts("")
        assert result == ""

    def test_scenic_drive_suffix_stripped(self):
        """'Scenic Drive' suffix should be stripped for shorts brevity."""
        result = shorten_ost_for_shorts("South Carolina Scenic Drive")
        assert result == "South Carolina"

    def test_drops_county_clause_before_charchop(self):
        """Over-budget "Town, County X" drops the whole region clause first —
        keeping the distinguishing town — instead of a dangling comma / a
        mangled "Count" fragment (region-preserving truncation)."""
        result = shorten_ost_for_shorts("Dingle Peninsula toward Gortmore, County Kerry")
        assert result == "Dingle Peninsula toward Gortmore"
        assert not result.endswith(",")
        assert "County" not in result

    def test_via_then_region_drop(self):
        """A "…, County X via Long Road" string strips the via clause AND then
        drops the county clause — never a mid-word char-chop."""
        result = shorten_ost_for_shorts(
            "Dingle Peninsula toward Gortmore, County Kerry via Slea Head Drive"
        )
        assert result == "Dingle Peninsula toward Gortmore"
        assert len(result) <= 40
