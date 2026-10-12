"""Tests for VideoType enum and classify_video() — TDD (red phase).

Covers incompatible is_short, the 61-479s dead zone, conflicting ranges,
and naming scheme mismatches.
"""

import logging

import pytest

from gomoveshift.video_types import (
    LF_MAX_DURATION,
    LF_MIN_DURATION,
    SHORTS_MAX_DURATION,
    VideoType,
    classify_video,
    suggest_action,
)


# --- Constants ---


class TestDurationConstants:
    def test_shorts_max_is_60(self):
        assert SHORTS_MAX_DURATION == 60

    def test_lf_min_is_480(self):
        assert LF_MIN_DURATION == 480

    def test_lf_max_is_1800(self):
        assert LF_MAX_DURATION == 1800

    def test_dead_zone_exists(self):
        """The gap between shorts max and LF min is the dead zone."""
        assert LF_MIN_DURATION - SHORTS_MAX_DURATION > 0


# --- VideoType enum ---


class TestVideoTypeEnum:
    def test_has_short(self):
        assert VideoType.SHORT.value == "short"

    def test_has_regular_lf(self):
        assert VideoType.REGULAR_LF.value == "regular_lf"

    def test_has_unclassified(self):
        assert VideoType.UNCLASSIFIED.value == "unclassified"


# --- classify_video() ---


class TestClassifyVideoShorts:
    """Vertical (height > width) + duration <= 60s → SHORT."""

    def test_15s_vertical(self):
        assert classify_video(15, 1080, 1920) == VideoType.SHORT

    def test_60s_vertical(self):
        assert classify_video(60, 1080, 1920) == VideoType.SHORT

    def test_1s_vertical(self):
        assert classify_video(1, 1080, 1920) == VideoType.SHORT

    def test_exact_boundary(self):
        assert classify_video(60.0, 1080, 1920) == VideoType.SHORT


class TestClassifyVideoRegularLF:
    """Duration 480-1800s → REGULAR_LF (orientation doesn't matter for LF)."""

    def test_480s_horizontal(self):
        assert classify_video(480, 1920, 1080) == VideoType.REGULAR_LF

    def test_1800s_horizontal(self):
        assert classify_video(1800, 1920, 1080) == VideoType.REGULAR_LF

    def test_900s_horizontal(self):
        assert classify_video(900, 1920, 1080) == VideoType.REGULAR_LF

    def test_480s_vertical(self):
        """Vertical long-form is still a regular LF by duration."""
        assert classify_video(480, 1080, 1920) == VideoType.REGULAR_LF


class TestClassifyVideoUnclassified:
    """Dead zone (61-479s) and other edge cases → UNCLASSIFIED."""

    def test_61s_vertical(self):
        result = classify_video(61, 1080, 1920)
        assert result == VideoType.UNCLASSIFIED

    def test_180s_vertical(self):
        assert classify_video(180, 1080, 1920) == VideoType.UNCLASSIFIED

    def test_479s_horizontal(self):
        assert classify_video(479, 1920, 1080) == VideoType.UNCLASSIFIED

    def test_300s_horizontal(self):
        assert classify_video(300, 1920, 1080) == VideoType.UNCLASSIFIED

    def test_30s_horizontal(self):
        """Landscape short: not vertical, not long enough for LF → unclassified."""
        assert classify_video(30, 1920, 1080) == VideoType.UNCLASSIFIED

    def test_61s_horizontal(self):
        assert classify_video(61, 1920, 1080) == VideoType.UNCLASSIFIED

    def test_1801s_horizontal(self):
        """Over max LF duration → unclassified."""
        assert classify_video(1801, 1920, 1080) == VideoType.UNCLASSIFIED

    def test_logs_warning(self, caplog):
        """Dead zone classification should log a warning."""
        with caplog.at_level(logging.WARNING):
            classify_video(200, 1080, 1920)
        assert any("unclassified" in r.message.lower() for r in caplog.records)


# --- Mapping helpers ---


class TestYouTubeTypeMapping:
    """VideoType → YouTube API video type string."""

    def test_short_maps_to_SHORT(self):
        assert VideoType.SHORT.to_youtube_type() == "SHORT"

    def test_regular_lf_maps_to_VIDEO(self):
        assert VideoType.REGULAR_LF.to_youtube_type() == "VIDEO"

    def test_unclassified_maps_to_VIDEO(self):
        assert VideoType.UNCLASSIFIED.to_youtube_type() == "VIDEO"


class TestLedgerTypeMapping:
    """VideoType → content ledger type string."""

    def test_short_maps_to_shorts(self):
        assert VideoType.SHORT.to_ledger_type() == "shorts"

    def test_regular_lf_maps_to_longform(self):
        assert VideoType.REGULAR_LF.to_ledger_type() == "longform"


class TestEventTypeMapping:
    """VideoType → content events storage type string."""

    def test_short_maps_to_short(self):
        assert VideoType.SHORT.to_event_type() == "short"

    def test_regular_lf_maps_to_long(self):
        assert VideoType.REGULAR_LF.to_event_type() == "long"


# --- suggest_action() ---


class TestSuggestAction:
    """Dead-zone videos (UNCLASSIFIED) get an action recommendation."""

    def test_61s_vertical_trim_to_short(self):
        """61s vertical, close to 60s boundary → trim_to_short."""
        assert suggest_action(61, 1080, 1920) == "trim_to_short"

    def test_120s_vertical_split(self):
        """120s vertical, too long for short, too short for LF → split."""
        assert suggest_action(120, 1080, 1920) == "trim_to_short"

    def test_200s_vertical_split(self):
        """200s vertical, middle of dead zone → split."""
        assert suggest_action(200, 1080, 1920) == "split"

    def test_420s_horizontal_extend_to_lf(self):
        """420s horizontal, close to LF minimum → extend_to_lf."""
        assert suggest_action(420, 1920, 1080) == "extend_to_lf"

    def test_470s_vertical_pad_to_lf(self):
        """470s vertical, close to 480s boundary → pad_to_lf."""
        assert suggest_action(470, 1080, 1920) == "pad_to_lf"

    def test_61s_horizontal_skip(self):
        """61s horizontal → skip (not vertical)."""
        assert suggest_action(61, 1920, 1080) == "skip"

    def test_boundary_120s_is_trim(self):
        """Exactly 120s is the boundary for trim_to_short."""
        assert suggest_action(120, 1080, 1920) == "trim_to_short"

    def test_121s_vertical_split(self):
        """121s vertical → split (past trim threshold)."""
        assert suggest_action(121, 1080, 1920) == "split"

    def test_419s_vertical_split(self):
        """419s vertical → split (not yet at pad threshold)."""
        assert suggest_action(419, 1080, 1920) == "split"

    def test_420s_vertical_pad_to_lf(self):
        """Exactly 420s vertical → pad_to_lf."""
        assert suggest_action(420, 1080, 1920) == "pad_to_lf"
