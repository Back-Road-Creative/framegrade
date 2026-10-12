"""Tests for Metricool CSV `Text` field truncation policy.

Bug: csv_generator.generate_lf_row hard-truncated `Text` to 280 chars
(Twitter limit) even when Twitter was disabled — chopping mid-emoji and
losing the trailing hashtag block. Fix: per-platform-aware truncation.

TDD: tests first, fix follows.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gomoveshift.pipeline.csv_generator import CSVGenerator


@pytest.fixture
def gen() -> CSVGenerator:
    return CSVGenerator()


def _read_text_col(csv_path: Path) -> str:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    return rows[0]["Text"]


def _read_twitter_col(csv_path: Path) -> str:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    return rows[0]["Twitter"]


# Realistic 700-char description: prose + emoji + trailing hashtag block.
LONG_DESC = (
    "Drive through the rolling Tipperary countryside on a crisp October morning. "
    "We pass ancient hedgerows, stone walls, and quiet farm lanes — the kind of "
    "scenic drive that reminds you why Ireland's back roads are unmatched. "
    "Listen for the soft hum of the engine and the gentle wind as we wind through "
    "the Galtee foothills past sleepy villages and centuries-old churches. "
    "If you enjoyed this drive, please like and subscribe for more. \U0001f1ee\U0001f1ea\u2728\n\n"
    "#Ireland #ScenicDrive #Tipperary #IrishCountryside #DrivingTour #RoadTrip "
    "#WildAtlanticWay #VisitIreland #IrishRoads #CountryDrive"
)


class TestTwitterDisabledNoTruncation:
    """When Twitter is disabled (LF rows), full description must be preserved."""

    def test_lf_row_preserves_full_700_char_description(self, gen, tmp_path):
        assert len(LONG_DESC) > 280, "fixture must exceed legacy 280 cap"
        out = tmp_path / "lf.csv"
        gen.generate_lf_row(
            title="Tipperary Scenic Drive",
            tags="ireland,driving",
            category="19",
            drive_link="https://drive.google.com/abc",
            schedule_date="2026-04-29",
            schedule_time="11:00:00",
            output_path=out,
            description=LONG_DESC,
        )
        text = _read_text_col(out)
        assert text == LONG_DESC, "LF rows have Twitter disabled — no truncation expected"

    def test_lf_row_preserves_trailing_hashtag_block(self, gen, tmp_path):
        out = tmp_path / "lf.csv"
        gen.generate_lf_row(
            title="t",
            tags="t",
            category="19",
            drive_link="x",
            schedule_date="2026-04-29",
            schedule_time="11:00:00",
            output_path=out,
            description=LONG_DESC,
        )
        text = _read_text_col(out)
        assert "#Ireland" in text
        assert "#CountryDrive" in text  # final hashtag survives

    def test_lf_row_preserves_trailing_emoji(self, gen, tmp_path):
        out = tmp_path / "lf.csv"
        gen.generate_lf_row(
            title="t",
            tags="t",
            category="19",
            drive_link="x",
            schedule_date="2026-04-29",
            schedule_time="11:00:00",
            output_path=out,
            description=LONG_DESC,
        )
        text = _read_text_col(out)
        # Irish flag = two regional-indicator codepoints; sparkle = one.
        assert "\U0001f1ee\U0001f1ea" in text
        assert "\u2728" in text

    def test_lf_row_twitter_column_is_empty(self, gen, tmp_path):
        """Sanity: confirm the LF row indeed leaves Twitter disabled."""
        out = tmp_path / "lf.csv"
        gen.generate_lf_row(
            title="t",
            tags="t",
            category="19",
            drive_link="x",
            schedule_date="2026-04-29",
            schedule_time="11:00:00",
            output_path=out,
            description=LONG_DESC,
        )
        twitter = _read_twitter_col(out)
        assert twitter in ("", "FALSE"), f"Twitter should be disabled on LF rows, got {twitter!r}"


class TestEmptyDescriptionFallback:
    """Existing behaviour: when description is empty, fall back to title."""

    def test_empty_description_falls_back_to_title(self, gen, tmp_path):
        out = tmp_path / "lf.csv"
        gen.generate_lf_row(
            title="My Title",
            tags="t",
            category="19",
            drive_link="x",
            schedule_date="2026-04-29",
            schedule_time="11:00:00",
            output_path=out,
            description="",
        )
        text = _read_text_col(out)
        assert text == "My Title"


# --- Twitter-enabled path: covered via the truncation helper ---------------

from gomoveshift.pipeline.csv_generator import truncate_for_platforms  # noqa: E402


class TestTwitterEnabledWordBoundary:
    """When Twitter IS enabled, truncation kicks in but at a word boundary."""

    def test_twitter_enabled_truncates_to_280(self):
        result = truncate_for_platforms(LONG_DESC, twitter_enabled=True)
        assert len(result) <= 280

    def test_twitter_enabled_truncates_at_word_boundary(self):
        result = truncate_for_platforms(LONG_DESC, twitter_enabled=True)
        # Result should not end mid-word: either ends at a word char then ellipsis,
        # or ends at a hashtag boundary. Crucially: no partial word like "count"
        # from "countryside" cut off.
        # Heuristic: the char preceding the trailing ellipsis (if any) is the
        # last char of a complete token.
        stripped = result.rstrip("\u2026").rstrip()
        # The next char in LONG_DESC after `stripped` must be whitespace or end
        # of string — i.e. we cut at a token boundary.
        if stripped and len(stripped) < len(LONG_DESC):
            next_char = LONG_DESC[len(stripped) : len(stripped) + 1]
            assert next_char in (" ", "\n", "", "\t"), (
                f"Truncation cut mid-word: stripped ends {stripped[-15:]!r}, "
                f"next char in source is {next_char!r}"
            )

    def test_twitter_enabled_no_mid_emoji_split(self):
        """The Irish flag is two codepoints (regional indicators). A naive
        slice at byte ~280 would split them. Word-boundary truncation must
        not produce a lone regional-indicator codepoint."""
        result = truncate_for_platforms(LONG_DESC, twitter_enabled=True)
        # No lone regional indicator (U+1F1E6..U+1F1FF) at the tail.
        for ch in result[-2:]:
            cp = ord(ch)
            if 0x1F1E6 <= cp <= 0x1F1FF:
                # If a regional indicator appears, its pair must be present.
                assert "\U0001f1ee\U0001f1ea" in result, (
                    "lone regional-indicator codepoint at tail — emoji was split"
                )

    def test_twitter_disabled_no_truncation(self):
        result = truncate_for_platforms(LONG_DESC, twitter_enabled=False)
        assert result == LONG_DESC
