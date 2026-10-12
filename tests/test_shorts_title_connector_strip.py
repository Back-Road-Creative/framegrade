"""A shorts-title chop must never burn a dangling connector.

Title assembly joins endpoints/roads with ``"to"`` / ``"via"`` / ``"+"``
(``"{start} to {end} via {A} + {B}"``). A truncation that cuts right after
one of those tokens burns an orphaned arc ("Killarney to") — the chop sites
must strip trailing connectors by construction, and ``validate_ost`` is the
upstream backstop before the irreversible burn.
"""

from gomoveshift.pipeline.day_pipeline import _fit_shorts_title, _gate_short_ost_text
from gomoveshift.pipeline.ffmpeg_filters import (
    shorten_ost_for_shorts,
    strip_trailing_connectors,
)
from gomoveshift.pipeline.ost_validator import validate_ost


class TestChopNeverDanglesConnector:
    def test_fit_shorts_title_strips_dangling_to(self):
        # Word-chop lands right after "to": the overlong endpoint is dropped,
        # the orphaned connector must go with it.
        result = _fit_shorts_title("Killarney to " + "X" * 39)
        assert result == "Killarney"

    def test_fit_shorts_title_strips_dangling_via(self):
        result = _fit_shorts_title("Killarney via " + "X" * 39)
        assert result == "Killarney"

    def test_shorten_ost_word_boundary_cut_strips_dangling_to(self):
        # 46-char arc: the 40-char word-boundary cut lands mid-endpoint,
        # leaving "… to" without the strip.
        result = shorten_ost_for_shorts("Ballylongfordbridgetown to Kilmacthomasbeggars")
        assert result == "Ballylongfordbridgetown"

    def test_gate_burns_no_dangling_connector(self):
        burned = _gate_short_ost_text(
            "Ballylongfordbridgetown to Kilmacthomasbeggars", None, 0, "2026-01-01"
        )
        assert burned == "Ballylongfordbridgetown"

    def test_strip_helper_never_empties(self):
        # A text that is ONLY a connector is left for the validator to reject.
        assert strip_trailing_connectors("to") == "to"

    def test_strip_helper_drops_stacked_connectors(self):
        assert strip_trailing_connectors("Kenmare via +") == "Kenmare"


class TestValidatorBackstop:
    def test_rejects_trailing_to(self):
        ok, reason = validate_ost("Killarney to", "shorts")
        assert not ok
        assert "connector" in reason

    def test_rejects_trailing_via(self):
        ok, reason = validate_ost("Ring of Kerry via", "shorts")
        assert not ok
        assert "connector" in reason

    def test_rejects_trailing_road_joiner(self):
        ok, _ = validate_ost("Asheville via US-19 +", "shorts")
        assert not ok

    def test_reason_never_triggers_the_fit_path(self):
        # _gate_short_ost_text only hard-fits on "exceeds"/"overflows" reasons;
        # a dangling connector is garbage, not an overflow.
        _, reason = validate_ost("Killarney to", "shorts")
        assert "exceeds" not in reason and "overflows" not in reason

    def test_accepts_complete_arc(self):
        ok, reason = validate_ost("Bantry to Kenmare", "shorts")
        assert ok, reason

    def test_accepts_names_merely_ending_in_to(self):
        ok, reason = validate_ost("Toronto", "shorts")
        assert ok, reason
