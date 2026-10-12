"""``pipeline/text_format.py`` text-shaping helpers moved out of the
retired Metricool CSV writer -- pure formatting, no CSV coupling.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from gomoveshift.pipeline import text_format
from gomoveshift.pipeline.text_format import (
    NON_TWITTER_CHAR_LIMIT,
    TWITTER_CHAR_LIMIT,
    close_sentence,
    format_recording_date_pretty,
    hashtags_in,
    split_trailing_hashtags,
    truncate_for_platforms,
)


class TestFormatRecordingDatePretty:
    @pytest.mark.parametrize(
        "raw, pretty",
        [
            ("2025-09-14", "September 2025"),
            ("2026-01-01", "January 2026"),
            ("2025-12-31", "December 2025"),
        ],
    )
    def test_formats_month_year(self, raw, pretty):
        assert format_recording_date_pretty(raw) == pretty

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            format_recording_date_pretty("")

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            format_recording_date_pretty("not-a-date")


class TestTruncateForPlatforms:
    def test_twitter_disabled_never_truncates(self):
        text = "x" * (TWITTER_CHAR_LIMIT + 50)
        assert truncate_for_platforms(text, twitter_enabled=False) == text

    def test_short_text_unchanged(self):
        assert truncate_for_platforms("short text", twitter_enabled=True) == "short text"

    def test_twitter_enabled_truncates_at_word_boundary(self):
        text = "word " * 100  # far over the limit, all whitespace-separated
        out = truncate_for_platforms(text, twitter_enabled=True)
        assert len(out) <= TWITTER_CHAR_LIMIT
        assert out.endswith("…")
        assert not out[:-1].endswith(" ")

    def test_custom_limit(self):
        text = "x" * 300
        out = truncate_for_platforms(text, twitter_enabled=True, limit=150)
        assert len(out) <= 150

    def test_empty_text_passes_through(self):
        assert truncate_for_platforms("", twitter_enabled=True) == ""

    def test_non_twitter_char_limit_exported(self):
        assert NON_TWITTER_CHAR_LIMIT == 5000


class TestStructuralInvariants:
    def test_no_lazy_relative_imports(self):
        source = Path(inspect.getfile(text_format)).read_text()
        for line in source.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("from ."), (
                f"text_format.py must use absolute imports, found: {stripped!r}"
            )

    def test_never_mentions_metricool(self):
        source = Path(inspect.getfile(text_format)).read_text()
        assert "metricool" not in source.lower()


class TestHashtags:
    def test_splits_the_closing_hashtag_run_off_the_sentence(self):
        assert split_trailing_hashtags("Worth it?\n\n#POVDriving  #PA \n") == (
            "Worth it?",
            "#POVDriving #PA",
        )

    def test_a_caption_of_only_hashtags_has_no_body(self):
        assert split_trailing_hashtags("#PA #RoadTrip") == ("", "#PA #RoadTrip")

    def test_hashtags_mid_sentence_or_a_number_sign_are_not_a_closing_run(self):
        assert split_trailing_hashtags("Drive #1 of #PA today") == ("Drive #1 of #PA today", "")
        assert split_trailing_hashtags("Stop #2") == ("Stop #2", "")

    def test_hashtags_in_keeps_order_and_drops_case_insensitive_repeats(self):
        assert hashtags_in("a #Shorts b #PA #shorts c#x #2") == ["#Shorts", "#PA"]


class TestCloseSentence:
    """A caption with no closing punctuation ("Just driving through Unicoi")
    followed by the filmed date was built "Just driving through Unicoi Filmed
    March 2026."."""

    @pytest.mark.parametrize(
        "text, closed",
        [
            ("Just driving through Unicoi", "Just driving through Unicoi."),
            ("Stop #2", "Stop #2."),
            ("Mile marker (17)", "Mile marker (17)."),
            ('Up "the Gap"', 'Up "the Gap".'),
            ("Bettws-y-Coed", "Bettws-y-Coed."),
        ],
    )
    def test_an_open_sentence_gets_its_full_stop(self, text, closed):
        assert close_sentence(text) == closed

    @pytest.mark.parametrize(
        "text",
        [
            "Made it.",
            "Worth the detour?",
            "Which way!",
            "Wait for it…",
            "From behind the wheel 🚗",
            "",
        ],
    )
    def test_a_sentence_that_already_ended_is_left_alone(self, text):
        assert close_sentence(text) == text

    def test_a_closing_hashtag_run_is_not_a_sentence(self):
        assert close_sentence("Unicoi roads #POVDriving #TN") == "Unicoi roads #POVDriving #TN"
