"""Burned-OST-text validation gate.

The per-LF OST text produced by ``build_lf_title`` is the label that burns
into pixels AND becomes the YouTube title, but it was the one surface with
no validator between geocoder output and the irreversible encode —
``validate_ost`` ran only on the episode-OST log line and the drive
fallback in ``phase_a``. ``validate_burned_ost_text`` closes that gap.
"""

import logging

from gomoveshift.pipeline.gps_enrichment import validate_burned_ost_text
from gomoveshift.pipeline.ost_validator import validate_ost


class TestLfTitleLengthFormat:
    """The burned title uses the 100-char ``lf_title`` budget, not the
    60-char OST-overlay ``lf`` limit, so legitimate long titles survive."""

    def test_70_char_title_valid_under_lf_title(self):
        text = "Cataloochee Valley to Maggie Valley via Blue Ridge Parkway, NC"  # 61 chars
        assert len(text) > 60
        ok, _ = validate_ost(text, "lf_title")
        assert ok is True
        # Same string would fail the 60-char OST-overlay limit:
        ok_lf, _ = validate_ost(text, "lf")
        assert ok_lf is False

    def test_over_100_chars_rejected(self):
        text = "A" * 101
        ok, reason = validate_ost(text, "lf_title")
        assert ok is False
        assert "exceeds" in reason


class TestValidateBurnedOstText:
    def test_valid_title_passes_through(self):
        out = validate_burned_ost_text(
            "Lake Lure to Oakland, NC", "Rutherford County, North Carolina", 3
        )
        assert out == "Lake Lure to Oakland, NC"

    def test_coordinate_garbage_falls_back(self, caplog):
        with caplog.at_level(logging.WARNING):
            out = validate_burned_ost_text("32.41, -80.40", "Beaufort County, South Carolina", 1)
        assert out == "Beaufort County, South Carolina"
        assert "OST-GATE" in caplog.text

    def test_literal_none_falls_back(self, caplog):
        with caplog.at_level(logging.WARNING):
            out = validate_burned_ost_text("None, None", "Macon County, North Carolina", 2)
        assert out == "Macon County, North Carolina"
        assert "OST-GATE" in caplog.text

    def test_none_canonical_uses_location(self):
        out = validate_burned_ost_text(None, "Colleton County, South Carolina", 0)
        assert out == "Colleton County, South Carolina"

    def test_empty_canonical_uses_location(self):
        out = validate_burned_ost_text("", "Colleton County, South Carolina", 0)
        assert out == "Colleton County, South Carolina"

    def test_valid_long_title_not_truncated(self):
        # 61-char legit title must survive (would fail the old 60-char "lf" gate).
        text = "Cataloochee Valley to Maggie Valley via Blue Ridge Parkway, NC"
        out = validate_burned_ost_text(text, "Haywood County, North Carolina", 4)
        assert out == text
