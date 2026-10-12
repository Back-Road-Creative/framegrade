"""YouTube chapter-format gate: a description chapter block only renders on
YouTube when it has >=3 timestamps, the first at 0:00, each >=10s apart.

The sparse-GPS basic fallback emits only 2 chapters (Start/Finish), which
YouTube silently refuses to render as chapters. The gate pads such a block to
the minimum renderable shape (and warns) so an LF never ships an
unrenderable chapter block. See docs/pipeline/04-chapters.md (YouTube renderability gate).
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.chapters import ChapterGenerator
from gomoveshift.pipeline.chapters.generator import (
    YOUTUBE_MIN_CHAPTER_GAP_SEC,
    YOUTUBE_MIN_CHAPTERS,
)


@pytest.fixture
def config():
    from gomoveshift.config import Config

    return Config()


def _parse_ts(ts: str) -> int:
    """Parse a YouTube 'M:SS' or 'H:MM:SS' timestamp into seconds."""
    parts = [int(p) for p in ts.split(":")]
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return secs


def _zero_duration_journey():
    """A journey whose start == end (gps_duration <= 0) forces the basic
    2-chapter fallback inside ``generate``."""
    wp = MagicMock()
    wp.latitude = 32.43
    wp.longitude = -80.67
    wp.timestamp = datetime(2026, 2, 20, 9, 0, 0)
    wp.location = None
    j = MagicMock()
    j.waypoints = (wp,)
    j.start_date = wp.timestamp
    j.end_date = wp.timestamp
    return j


def _assert_youtube_renderable(chapters):
    assert len(chapters) >= YOUTUBE_MIN_CHAPTERS, (
        f"YouTube needs >= {YOUTUBE_MIN_CHAPTERS} chapters; got {len(chapters)}: {chapters}"
    )
    assert chapters[0]["timestamp"] == "0:00", "first chapter must be 0:00"
    secs = [_parse_ts(c["timestamp"]) for c in chapters]
    assert secs == sorted(secs), f"chapters must be ascending: {secs}"
    for a, b in zip(secs, secs[1:], strict=False):
        assert b - a >= YOUTUBE_MIN_CHAPTER_GAP_SEC, (
            f"chapters must be >= {YOUTUBE_MIN_CHAPTER_GAP_SEC}s apart: {secs}"
        )


class TestChapterGate:
    def test_basic_fallback_padded_to_renderable(self, config):
        """A sparse-GPS LF (basic 2-chapter fallback) is padded to a
        YouTube-renderable >=3-chapter block."""
        gen = ChapterGenerator(config)
        chapters = gen.generate(
            gpx_path=Path("fake.gpx"),
            video_duration=900.0,
            start_location="Beaufort, South Carolina",
            _mock_journey=_zero_duration_journey(),
        )
        _assert_youtube_renderable(chapters)

    def test_padding_warns(self, config, caplog):
        """The gate logs a greppable warning when it pads."""
        gen = ChapterGenerator(config)
        with caplog.at_level("WARNING"):
            gen.generate(
                gpx_path=Path("fake.gpx"),
                video_duration=900.0,
                start_location="Beaufort, South Carolina",
                _mock_journey=_zero_duration_journey(),
            )
        assert any("chapter gate" in r.message for r in caplog.records), (
            f"expected a 'chapter gate' warning; got {[r.message for r in caplog.records]}"
        )

    def test_pad_uses_location_label(self, config):
        """Padded midpoint chapters carry the LF location, not a bare
        placeholder."""
        gen = ChapterGenerator(config)
        chapters = gen.generate(
            gpx_path=Path("fake.gpx"),
            video_duration=900.0,
            start_location="Beaufort, South Carolina",
            _mock_journey=_zero_duration_journey(),
        )
        interior = chapters[1:-1]
        assert interior, "expected at least one padded interior chapter"
        assert any("Beaufort" in c["title"] for c in interior), (
            f"padded chapter should reference the LF location: {interior}"
        )

    def test_already_renderable_not_padded(self, config):
        """When >=3 real chapters already exist, the gate is a no-op (no
        synthetic chapters added)."""
        gen = ChapterGenerator(config)
        real = [
            {"timestamp": "0:00", "title": "Start - A"},
            {"timestamp": "2:00", "title": "Entering B"},
            {"timestamp": "4:00", "title": "Entering C"},
            {"timestamp": "14:45", "title": "Finish - D"},
        ]
        out = gen._pad_to_min_chapters(list(real), 900.0, "A")
        assert out == real

    def test_short_lf_not_over_padded(self, config):
        """An LF too short to host 3 chapters >=10s apart is left as-is
        (with a warning) rather than crammed."""
        gen = ChapterGenerator(config)
        two = [
            {"timestamp": "0:00", "title": "Start - A"},
            {"timestamp": "0:05", "title": "Finish - A"},
        ]
        out = gen._pad_to_min_chapters(list(two), 20.0, "A")
        # 20s can't host 3 chapters >=10s apart → unchanged
        assert len(out) == 2


class TestChapterGateConstants:
    def test_constants_match_youtube_policy(self):
        assert YOUTUBE_MIN_CHAPTERS == 3
        assert YOUTUBE_MIN_CHAPTER_GAP_SEC == 10
