"""Tests for undersized LF handling and dead zone awareness.

TDD red phase: tests written before implementation changes.
"""

import logging
from pathlib import Path

from gomoveshift.config import DriveConfig
from gomoveshift.models import LFSegment
from gomoveshift.pipeline.longform_assembler import LongFormAssembler


def _make_segment(
    start: float,
    end: float,
    score: float = 0.7,
    file_idx: int = 0,
) -> LFSegment:
    """Helper to create an LFSegment for testing."""
    return LFSegment(
        source_file_index=file_idx,
        source_path=Path(f"file_{file_idx:02d}.mp4"),
        start_time=start,
        end_time=end,
        original_start_time=start,
        original_end_time=end,
        composite_score=score,
        gps_center_lat=0.0,
        gps_center_lon=0.0,
        terrain_type="unknown",
        contiguous_with_previous=False,
    )


class TestUndersizedFinalLF:
    """Final group must not silently produce a 3-min 'long form'."""

    def test_undersized_group_merged_with_previous(self):
        """If final group < min_lf, merge with previous LF if under max."""
        cfg = DriveConfig(min_lf_duration=480.0, max_lf_duration=1800.0)
        assembler = LongFormAssembler(cfg)

        # 5 segments: first 4 = 600s (valid LF), last = 200s (undersized alone)
        segments = [
            _make_segment(0, 150),
            _make_segment(150, 300),
            _make_segment(300, 450),
            _make_segment(450, 600),
            _make_segment(600, 800),  # 200s remainder
        ]

        lfs = assembler.group_into_regular_lfs(segments)

        # Should produce 1 LF (600+200=800s, under max)
        assert len(lfs) == 1
        assert lfs[0].total_duration >= 480

    def test_undersized_group_redistributes_when_merge_exceeds_max(self):
        """If direct merge exceeds max but segments can be redistributed into
        two valid LFs, rebalance instead of publishing an undersized LF."""
        cfg = DriveConfig(min_lf_duration=480.0, max_lf_duration=1800.0, best_moments_mode=False)
        assembler = LongFormAssembler(cfg)

        # First LF fills to ~1750s, remainder is 300s (total 2050s).
        # Direct merge would be 2050 > 1800 max, so rebalance kicks in.
        # Valid splits exist (e.g. 1500 + 550), so we redistribute rather than tag.
        segments = [
            _make_segment(0, 500),
            _make_segment(500, 1000),
            _make_segment(1000, 1500),
            _make_segment(1500, 1750),  # First LF ~1750s
            _make_segment(1750, 2050),  # 300s remainder
        ]

        lfs = assembler.group_into_regular_lfs(segments)

        # Both LFs should be valid-sized (no undersized tag)
        assert len(lfs) == 2
        assert all(lf.total_duration >= cfg.min_lf_duration for lf in lfs), (
            f"Both LFs should be >= min_lf_duration after redistribute; "
            f"got {[lf.total_duration for lf in lfs]}"
        )
        assert all(lf.total_duration <= cfg.max_lf_duration for lf in lfs)
        assert not any(getattr(lf, "undersized", False) for lf in lfs)

    def test_normal_case_not_tagged(self):
        """Normal LFs should NOT be tagged as undersized."""
        cfg = DriveConfig(min_lf_duration=480.0, max_lf_duration=1800.0)
        assembler = LongFormAssembler(cfg)

        segments = [
            _make_segment(0, 250),
            _make_segment(250, 500),
            _make_segment(500, 750),
            _make_segment(750, 1000),
        ]

        lfs = assembler.group_into_regular_lfs(segments)
        assert len(lfs) == 1
        assert lfs[0].undersized is False

    def test_exact_min_not_undersized(self):
        """LF at exactly min_lf_duration is NOT undersized."""
        cfg = DriveConfig(min_lf_duration=480.0, max_lf_duration=1800.0)
        assembler = LongFormAssembler(cfg)

        segments = [_make_segment(0, 480)]
        lfs = assembler.group_into_regular_lfs(segments)
        assert len(lfs) == 1
        assert lfs[0].undersized is False

    def test_oversized_merge_when_redistribute_impossible(self, caplog):
        """If direct merge exceeds max AND redistribute fails, merge anyway
        up to LF_MAX_DURATION hard ceiling. Better one too-long LF than one
        too-short LF that can't stand alone."""
        cfg = DriveConfig(min_lf_duration=480.0, max_lf_duration=600.0)
        assembler = LongFormAssembler(cfg)

        # First LF: 500s (valid, just over min=480). Remainder: 200s (below min=480).
        # Direct merge = 700s > max=600 so option 1 fails.
        # Redistribute: need both halves >= 480 AND <= 600. Total = 700, impossible.
        # Oversized merge: 700s < LF_MAX_DURATION=1800 so accept.
        segments = [
            _make_segment(0, 500),
            _make_segment(500, 700),
        ]

        # has_gps=True keeps the 600s target so the tail-rebalance path is exercised
        # (the fewest-LFs rule only changes GPS-dead packing).
        with caplog.at_level(logging.WARNING):
            lfs = assembler.group_into_regular_lfs(segments, has_gps=True)

        assert len(lfs) == 1, "Should merge into one LF rather than publish an undersized tail"
        assert lfs[0].total_duration >= cfg.min_lf_duration
        assert any(
            "oversized_merge" in r.message.lower() or "above max_lf_duration" in r.message.lower()
            for r in caplog.records
        ), "Expected warning about oversized merge"

    def test_undersized_only_as_last_resort(self, caplog):
        """If all three strategies fail (combined > LF_MAX_DURATION and no
        valid split), THEN publish undersized as last resort."""
        # Use tiny min/max so we can construct an actual impossible case.
        # Single 200s segment followed by a 100s remainder; max=150, min=180.
        # Direct merge 300 > 150 max. Redistribute: need both >= 180, impossible.
        # But total 300 <= 1800 hard ceiling → oversized_merge wins.
        # To force last-resort, combined must exceed LF_MAX_DURATION (1800s).
        # Build: prev LF at 1700s, tail at 200s → combined 1900 > 1800.
        # max_lf_duration=1700 (so prev LF of 1700 is allowed and the xfade check
        # doesn't trigger a split within prev).
        # Redistribute: min=1700 means both halves need 1700+. Impossible.
        # oversized_merge: 1900 > 1800 hard ceiling. Fails.
        # Last resort: publish undersized.
        cfg = DriveConfig(min_lf_duration=1700.0, max_lf_duration=1700.0, best_moments_mode=False)
        assembler = LongFormAssembler(cfg)

        segments = [
            _make_segment(0, 1700),  # prev LF = 1700s
            _make_segment(1700, 1900),  # tail = 200s
        ]

        with caplog.at_level(logging.WARNING):
            lfs = assembler.group_into_regular_lfs(segments)

        assert len(lfs) == 2, "Last-resort path keeps tail as separate undersized LF"
        assert lfs[1].undersized is True
        assert any("undersized" in r.message.lower() for r in caplog.records)


class TestOversizedSegment:
    """Single segment exceeding max_lf_duration should warn."""

    def test_oversized_segment_produces_lf_with_warning(self, caplog):
        """A single segment >max should still be included, with a warning."""
        cfg = DriveConfig(min_lf_duration=480.0, max_lf_duration=1800.0)
        assembler = LongFormAssembler(cfg)

        segments = [_make_segment(0, 2000)]  # 2000s > 1800s max

        with caplog.at_level(logging.WARNING):
            lfs = assembler.group_into_regular_lfs(segments)

        assert len(lfs) == 1
        assert any("exceeds" in r.message.lower() for r in caplog.records)
