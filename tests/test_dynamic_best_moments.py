"""Tests for _dynamic_best_moments_filter — duration-aware quality scaling."""

from pathlib import Path

from gomoveshift.config import DriveConfig
from gomoveshift.models import LFSegment
from gomoveshift.pipeline.longform_assembler import _dynamic_best_moments_filter


def _make_segment(
    index: int,
    start: float,
    duration: float,
    score: float,
) -> LFSegment:
    """Create a minimal LFSegment for testing."""
    return LFSegment(
        source_file_index=0,
        source_path=Path("test.mp4"),
        start_time=start,
        end_time=start + duration,
        original_start_time=start,
        original_end_time=start + duration,
        composite_score=score,
        gps_center_lat=0.0,
        gps_center_lon=0.0,
        terrain_type="unknown",
        contiguous_with_previous=False,
    )


class TestDynamicBestMomentsFilter:
    """Tests for _dynamic_best_moments_filter."""

    def test_dynamic_filter_respects_target_min(self):
        """Even low-scoring footage reaches target_min duration."""
        config = DriveConfig(
            best_moments_mode=True,
            best_moments_target_min=300.0,  # 5 min
            best_moments_target_max=600.0,  # 10 min
        )
        # 10 segments of 60s each, all low scores (below any quality floor)
        segments = [_make_segment(i, i * 60.0, 60.0, 0.1 + i * 0.01) for i in range(10)]
        result = _dynamic_best_moments_filter(segments, config)
        total_dur = sum(s.duration for s in result)
        # Must reach at least target_min (300s = 5 segments of 60s)
        assert total_dur >= config.best_moments_target_min

    def test_dynamic_filter_target_max_is_soft(self):
        """target_max is soft — all high-scoring segments kept regardless."""
        config = DriveConfig(
            best_moments_mode=True,
            best_moments_percentile=50.0,
            best_moments_target_min=300.0,  # 5 min
            best_moments_target_max=600.0,  # 10 min
        )
        # 20 segments of 60s each, all score=0.9 (identical)
        # With percentile=50, floor is at the 50th-percentile score = 0.9
        # All segments are AT the quality floor, so none dropped
        segments = [_make_segment(i, i * 60.0, 60.0, 0.9) for i in range(20)]
        result = _dynamic_best_moments_filter(segments, config)
        total_dur = sum(s.duration for s in result)
        # target_max is soft — quality floor not breached, so all kept
        assert len(result) == len(segments)
        assert total_dur >= config.best_moments_target_min

    def test_dynamic_filter_stops_at_quality_floor(self):
        """Stops when the next segment drops below quality floor."""
        config = DriveConfig(
            best_moments_mode=True,
            best_moments_percentile=50.0,  # keep top 50%
            best_moments_target_min=120.0,  # 2 min — low so floor triggers first
            best_moments_target_max=3600.0,  # 60 min — high so max doesn't trigger
        )
        # 5 high-scoring segments (0.9) + 5 low ones (0.1)
        # Sorted scores: [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9]
        # 50th percentile = scores[5] = 0.9
        # So floor = 0.9, and all 0.1-scored segments are below floor.
        # After adding the 5 high segments (300s >= 120s target_min), the next
        # segment has score 0.1 < 0.9 floor → stop.
        segments = [_make_segment(i, i * 60.0, 60.0, 0.9) for i in range(5)] + [
            _make_segment(i + 5, (i + 5) * 60.0, 60.0, 0.1) for i in range(5)
        ]
        result = _dynamic_best_moments_filter(segments, config)
        total_dur = sum(s.duration for s in result)
        # Should include only the 5 high segments (300s)
        assert len(result) == 5
        assert total_dur == 300.0
        assert all(s.composite_score == 0.9 for s in result)

    def test_lower_percentile_keeps_more_footage(self):
        """Lower percentile (e.g. 25) keeps more footage than higher (e.g. 75)."""
        # 10 segments with linearly increasing scores 0.1 to 1.0
        segments = [_make_segment(i, i * 60.0, 60.0, 0.1 * (i + 1)) for i in range(10)]

        config_aggressive = DriveConfig(
            best_moments_mode=True,
            best_moments_percentile=75.0,  # keep top 25%
            best_moments_target_min=60.0,
            best_moments_target_max=3600.0,
        )
        config_lenient = DriveConfig(
            best_moments_mode=True,
            best_moments_percentile=25.0,  # keep top 75%
            best_moments_target_min=60.0,
            best_moments_target_max=3600.0,
        )

        result_aggressive = _dynamic_best_moments_filter(segments, config_aggressive)
        result_lenient = _dynamic_best_moments_filter(segments, config_lenient)

        # Lenient filter should keep more segments
        assert len(result_lenient) >= len(result_aggressive)

    def test_dynamic_filter_preserves_timeline_order(self):
        """Output is sorted by original_start_time regardless of score order."""
        config = DriveConfig(
            best_moments_mode=True,
            best_moments_target_min=120.0,
            best_moments_target_max=600.0,
        )
        # Segments with scores in non-chronological order
        segments = [
            _make_segment(0, 0.0, 60.0, 0.5),
            _make_segment(1, 60.0, 60.0, 0.9),  # highest score
            _make_segment(2, 120.0, 60.0, 0.7),
            _make_segment(3, 180.0, 60.0, 0.3),
        ]
        result = _dynamic_best_moments_filter(segments, config)
        # Verify timeline order
        for i in range(1, len(result)):
            assert result[i].original_start_time > result[i - 1].original_start_time

    def test_dynamic_filter_handles_empty_segments(self):
        """Returns empty list for empty input."""
        config = DriveConfig(
            best_moments_mode=True,
            best_moments_target_min=300.0,
            best_moments_target_max=600.0,
        )
        result = _dynamic_best_moments_filter([], config)
        assert result == []


class TestAbsoluteFloor:
    """Tests for the opt-in absolute composite-score floor (visual-only)."""

    def _cfg(self, **kw) -> DriveConfig:
        base = dict(
            best_moments_mode=True,
            best_moments_absolute_floor_enabled=True,
            best_moments_absolute_floor=0.26,
            # Make the percentile path obviously different so a fallback is detectable:
            # target_min forces percentile mode to pad, absolute mode must not.
            best_moments_target_min=900.0,
            best_moments_target_max=1500.0,
        )
        base.update(kw)
        return DriveConfig(**base)

    def test_keeps_only_segments_at_or_above_floor(self):
        """Absolute mode keeps exactly the segments whose score >= floor."""
        segs = [
            _make_segment(0, 0.0, 90.0, 0.20),
            _make_segment(1, 90.0, 90.0, 0.24),
            _make_segment(2, 180.0, 90.0, 0.27),
            _make_segment(3, 270.0, 90.0, 0.30),
        ]
        result = _dynamic_best_moments_filter(segs, self._cfg(), has_gps=False)
        assert [round(s.composite_score, 2) for s in result] == [0.27, 0.30]

    def test_no_padding_when_sparse(self):
        """Sparse footage is NOT padded up to target_min (the percentile path would)."""
        # Only 180s clears the floor; target_min is 900s. Absolute mode must
        # keep just the 2 qualifying segments, never reach for target_min.
        segs = [_make_segment(i, i * 90.0, 90.0, 0.10 + i * 0.02) for i in range(10)]
        # scores: 0.10..0.28; floor 0.26 → only the last 2 (0.26, 0.28) qualify
        result = _dynamic_best_moments_filter(segs, self._cfg(), has_gps=False)
        total = sum(s.duration for s in result)
        assert total < 900.0  # not padded to target_min
        assert all(s.composite_score >= 0.26 for s in result)

    def test_scales_with_quality(self):
        """A high-quality drive keeps more than a sparse one at the same floor."""
        good = [_make_segment(i, i * 90.0, 90.0, 0.30) for i in range(8)]
        sparse = [_make_segment(i, i * 90.0, 90.0, 0.22) for i in range(8)] + [
            _make_segment(8, 720.0, 90.0, 0.31)
        ]
        kept_good = _dynamic_best_moments_filter(good, self._cfg(), has_gps=False)
        kept_sparse = _dynamic_best_moments_filter(sparse, self._cfg(), has_gps=False)
        assert len(kept_good) == 8
        assert len(kept_sparse) == 1  # only the single 0.31 segment clears 0.26

    def test_floors_to_one_lf_never_empty(self):
        """When nothing clears the floor, keep the single best segment (never empty)."""
        segs = [_make_segment(i, i * 90.0, 90.0, 0.10 + i * 0.01) for i in range(5)]
        result = _dynamic_best_moments_filter(segs, self._cfg(), has_gps=False)
        assert len(result) == 1
        assert result[0].composite_score == max(s.composite_score for s in segs)

    def test_gps_present_falls_back_to_percentile(self):
        """Mode-aware: with GPS present the absolute floor is NOT applied."""
        # All 5 segments below the 0.26 absolute floor. Absolute mode would
        # floor to 1; the percentile fallback (target_min=900) keeps all 5.
        segs = [_make_segment(i, i * 90.0, 90.0, 0.20 + i * 0.005) for i in range(5)]
        result = _dynamic_best_moments_filter(segs, self._cfg(), has_gps=True)
        assert len(result) == len(segs)  # percentile path padded to target_min

    def test_disabled_uses_percentile(self):
        """Flag off → unchanged percentile behavior (regression guard)."""
        segs = [_make_segment(i, i * 90.0, 90.0, 0.20 + i * 0.005) for i in range(5)]
        cfg = self._cfg(best_moments_absolute_floor_enabled=False)
        result = _dynamic_best_moments_filter(segs, cfg, has_gps=False)
        assert len(result) == len(segs)

    def test_preserves_timeline_order(self):
        """Output stays in timeline order regardless of score order."""
        segs = [
            _make_segment(0, 0.0, 90.0, 0.27),
            _make_segment(1, 90.0, 90.0, 0.31),
            _make_segment(2, 180.0, 90.0, 0.28),
        ]
        result = _dynamic_best_moments_filter(segs, self._cfg(), has_gps=False)
        for i in range(1, len(result)):
            assert result[i].original_start_time > result[i - 1].original_start_time
