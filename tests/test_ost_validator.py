"""Tests for OST validator — catches malformed text before it hits rendering."""


class TestOSTValidator:
    """validate_ost rejects empty, overly long, or malformed OST text."""

    def test_normal_lf_text_is_valid(self):
        from gomoveshift.pipeline.ost_validator import validate_ost

        ok, _ = validate_ost("Port Royal to Beaufort via Trask Parkway", "lf")
        assert ok is True

    def test_normal_shorts_text_is_valid(self):
        from gomoveshift.pipeline.ost_validator import validate_ost

        ok, _ = validate_ost("Seaside Road", "shorts")
        assert ok is True

    def test_empty_text_is_invalid(self):
        from gomoveshift.pipeline.ost_validator import validate_ost

        ok, reason = validate_ost("", "lf")
        assert ok is False
        assert "empty" in reason.lower()

    def test_whitespace_only_is_invalid(self):
        from gomoveshift.pipeline.ost_validator import validate_ost

        ok, reason = validate_ost("   \t  ", "lf")
        assert ok is False
        assert "empty" in reason.lower()

    def test_lf_length_limit_is_60(self):
        from gomoveshift.pipeline.ost_validator import validate_ost

        too_long = "a" * 61
        ok, reason = validate_ost(too_long, "lf")
        assert ok is False
        assert "length" in reason.lower() or "long" in reason.lower()

        # A realistic spaced 60-char label (wraps within the pixel budget) is
        # valid at the codepoint cap. A 60-char unspaceable word now overflows
        # the frame pixel-wise — that gate is tested separately in
        # test_pixel_budgets.py; here we exercise the codepoint cap only.
        exactly_60 = "Cataloochee Valley to Maggie Valley via Parkway Road, NC AB"
        assert len(exactly_60) <= 60
        ok, _ = validate_ost(exactly_60, "lf")
        assert ok is True

    def test_shorts_length_limit_is_40(self):
        from gomoveshift.pipeline.ost_validator import validate_ost

        too_long = "a" * 41
        ok, _ = validate_ost(too_long, "shorts")
        assert ok is False

        exactly_40 = "Killarney to Dingle via the N86 road now"  # 40, spaced
        assert len(exactly_40) == 40
        ok, _ = validate_ost(exactly_40, "shorts")
        assert ok is True

    def test_literal_none_string_is_invalid(self):
        from gomoveshift.pipeline.ost_validator import validate_ost

        for val in ("None", "null", "NULL", "None, None"):
            ok, reason = validate_ost(val, "lf")
            assert ok is False, f"{val!r} should be invalid"
            assert "broken" in reason.lower() or "none" in reason.lower()

    def test_pure_digits_is_invalid(self):
        from gomoveshift.pipeline.ost_validator import validate_ost

        ok, reason = validate_ost("12345", "lf")
        assert ok is False
        assert "digit" in reason.lower() or "broken" in reason.lower()

    def test_coordinate_strings_are_invalid(self):
        from gomoveshift.pipeline.ost_validator import validate_ost

        for coord in ("32.5, -80.5", "32.5°, -80.5°", "-55.23 7.81"):
            ok, reason = validate_ost(coord, "lf")
            assert ok is False, f"{coord!r} should be invalid"

    def test_hoa_words_still_blocked(self):
        from gomoveshift.pipeline.ost_validator import validate_ost

        for hoa in (
            "Oceanmarsh Subdivision",
            "Lakeside Estates Scenic Drive",
            "Seaside Road, Pine Community",
        ):
            ok, reason = validate_ost(hoa, "lf")
            assert ok is False, f"{hoa!r} should be HOA-rejected"
            assert "hoa" in reason.lower() or "subdivision" in reason.lower()

    def test_unknown_format_defaults_to_lf_limits(self):
        from gomoveshift.pipeline.ost_validator import validate_ost

        # Spaced realistic label under the 60-char default cap (fits pixel-wise).
        ok, _ = validate_ost("Lake Lure to Oakland via the Parkway, NC", "unknown_format")
        assert ok is True

        ok, _ = validate_ost("a" * 75, "unknown_format")
        assert ok is False

    def test_literal_unknown_string_is_invalid(self):
        """Reject literal 'UNKNOWN' — a sentinel from broken
        geocoder or title fallback paths, not a real location."""
        from gomoveshift.pipeline.ost_validator import validate_ost

        for val in ("UNKNOWN", "unknown", "Unknown", "UNKNOWN, UNKNOWN"):
            ok, reason = validate_ost(val, "lf")
            assert ok is False, f"{val!r} should be invalid"
            assert "placeholder" in reason.lower() or "unknown" in reason.lower()

    def test_literal_placeholder_string_is_invalid(self):
        """Reject literal placeholder markers the pipeline
        must never render."""
        from gomoveshift.pipeline.ost_validator import validate_ost

        for val in (
            "LOCATION_PLACEHOLDER",
            "location_placeholder",
            "TODO",
            "TBD",
            "PLACEHOLDER",
        ):
            ok, reason = validate_ost(val, "lf")
            assert ok is False, f"{val!r} should be invalid"
            assert "placeholder" in reason.lower()

    def test_real_location_containing_unknown_substring_still_valid(self):
        """Substring check must not match words that merely CONTAIN
        'unknown' etc. — whole-text rejection only."""
        from gomoveshift.pipeline.ost_validator import validate_ost

        # Contrived but illustrative — any legitimate place name that
        # happens to include a blocked substring should pass.
        ok, _ = validate_ost("Unknowngate, SC", "lf")
        assert ok is True
