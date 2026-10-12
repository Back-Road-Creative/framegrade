"""Tests for GPS-to-video time mapping (shorts_time_mapping)."""

import pytest

from gomoveshift.analysis.shorts_time_mapping import (
    VideoToGpsTimeMapper,
    map_from_edited_time,
)


class TestMapFromEditedTime:
    """Tests for the standalone map_from_edited_time function."""

    def test_no_keep_segments_identity(self):
        """Without segments, edited time passes through unchanged."""
        assert map_from_edited_time(42.0, None) == 42.0
        assert map_from_edited_time(42.0, []) == 42.0

    def test_single_segment(self):
        """Single keep segment: maps within that segment's range."""
        # Keep 10-20s of original video
        segs = [(10.0, 20.0)]
        # edited_time 0 -> original 10
        assert map_from_edited_time(0.0, segs) == 10.0
        # edited_time 5 -> original 15
        assert map_from_edited_time(5.0, segs) == 15.0

    def test_multiple_segments(self):
        """Time maps across segment boundaries correctly."""
        # Keep 10-20 (10s) and 30-50 (20s)
        segs = [(10.0, 20.0), (30.0, 50.0)]
        # edited_time 0 -> original 10 (start of first seg)
        assert map_from_edited_time(0.0, segs) == 10.0
        # edited_time 10 -> original 30 (start of second seg, first seg used 10s)
        assert map_from_edited_time(10.0, segs) == 30.0
        # edited_time 15 -> original 35 (5s into second seg)
        assert map_from_edited_time(15.0, segs) == 35.0

    def test_past_all_segments_clamps(self):
        """Edited time past total duration clamps to end of last segment."""
        segs = [(10.0, 20.0)]
        # 10s of content, asking for t=15 -> clamp to 20.0
        assert map_from_edited_time(15.0, segs) == 20.0

    def test_zero_duration_segment(self):
        """Zero-length segment is skipped (seg_duration == 0)."""
        segs = [(5.0, 5.0), (10.0, 20.0)]
        # First segment has 0 duration, so edited_time 0 starts in second segment
        assert map_from_edited_time(0.0, segs) == 10.0

    def test_exact_boundary(self):
        """Edited time exactly at segment boundary goes to next segment."""
        segs = [(0.0, 10.0), (20.0, 30.0)]
        # Exactly at 10s (end of first seg's duration) -> start of second seg
        assert map_from_edited_time(10.0, segs) == 20.0


class TestVideoToGpsTimeMapper:
    """Tests for VideoToGpsTimeMapper.video_to_gps_time."""

    def test_simple_offset_fallback(self):
        """Without segments/durations, uses simple gps_offset addition."""
        mapper = VideoToGpsTimeMapper(gps_offset=100.0)
        assert mapper.video_to_gps_time(10.0) == 110.0

    def test_zero_offset(self):
        """Zero offset passes through."""
        mapper = VideoToGpsTimeMapper(gps_offset=0.0)
        assert mapper.video_to_gps_time(25.0) == 25.0

    def test_full_mapping_with_segments(self):
        """Full chain: edited -> original -> GPS with scaling and offset."""
        mapper = VideoToGpsTimeMapper(
            gps_offset=1000.0,
            keep_segments=[(10.0, 20.0)],  # 10s of original kept
            gps_duration=600.0,  # GPS track is 600s
            original_duration=60.0,  # Original video is 60s
        )
        # edited_time=5 -> original=15 -> gps_time=15*(600/60)=150 -> +1000=1150
        assert mapper.video_to_gps_time(5.0) == pytest.approx(1150.0)

    def test_fallback_when_gps_duration_zero(self):
        """Falls back to simple offset when gps_duration is 0."""
        mapper = VideoToGpsTimeMapper(
            gps_offset=50.0,
            keep_segments=[(10.0, 20.0)],
            gps_duration=0.0,
            original_duration=60.0,
        )
        assert mapper.video_to_gps_time(5.0) == 55.0

    def test_fallback_when_original_duration_zero(self):
        """Falls back to simple offset when original_duration is 0."""
        mapper = VideoToGpsTimeMapper(
            gps_offset=50.0,
            keep_segments=[(10.0, 20.0)],
            gps_duration=600.0,
            original_duration=0.0,
        )
        assert mapper.video_to_gps_time(5.0) == 55.0

    def test_negative_offset(self):
        """Negative GPS offset works (GPS started before video)."""
        mapper = VideoToGpsTimeMapper(gps_offset=-30.0)
        assert mapper.video_to_gps_time(50.0) == 20.0


class TestCrossfadeCorrection:
    """A crossfade-stitched edited timeline overlaps each seam by `crossfade`,
    so non-final segments advance the edited clock by (duration − crossfade)."""

    SEGS = [(10.0, 20.0), (30.0, 50.0)]  # 10s + 20s of original

    def test_crossfade_zero_matches_hard_cut(self):
        for t in (0.0, 4.0, 10.0, 15.0):
            assert map_from_edited_time(t, self.SEGS, 0.0) == map_from_edited_time(t, self.SEGS)

    def test_crossfade_pulls_each_seam_earlier(self):
        # With a 2s dissolve, the seam lands at edited 8 (10 − 2), not 10.
        assert map_from_edited_time(0.0, self.SEGS, 2.0) == 10.0
        assert map_from_edited_time(8.0, self.SEGS, 2.0) == 30.0  # start of 2nd seg
        assert map_from_edited_time(10.0, self.SEGS, 2.0) == 32.0  # 2s into 2nd seg
        # Total edited length is Σ − (N−1)·crossfade = 30 − 2 = 28; past it clamps.
        assert map_from_edited_time(28.0, self.SEGS, 2.0) == 50.0

    def test_mapper_applies_crossfade_duration(self):
        mapper = VideoToGpsTimeMapper(
            gps_offset=0.0,
            keep_segments=self.SEGS,
            gps_duration=60.0,
            original_duration=60.0,  # 1:1 scale isolates the stop-removal inversion
            crossfade_duration=2.0,
        )
        # edited 8 → original 30 (seam pulled earlier) → ×1 → 30, vs 18 without.
        assert mapper.video_to_gps_time(8.0) == pytest.approx(30.0)
