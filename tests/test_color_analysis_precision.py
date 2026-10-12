"""Adaptive-grading precision: local span coarsening and short-event survival.

Two ways the analyser threw away precision it had already measured:

* Span overflow coarsened the WHOLE day (one oscillating stretch degraded a
  calm morning that was merging perfectly well).
* A lighting event shorter than the sample interval appeared in a minority of
  a segment's samples and was voted away by the median.
"""

import pytest

from gomoveshift.pipeline.color_analysis import (
    FrameStats,
    SegmentStats,
    aggregate_segment_stats,
    merge_adjacent_segments,
)


def _seg(black=16, white=235, mid=128, sat=0.3) -> SegmentStats:
    return SegmentStats(black_point=black, white_point=white, mid_tone=mid, saturation=sat)


class TestOverflowCoarseningIsLocal:
    """A busy afternoon must not coarsen a calm morning."""

    # Morning: three well-separated blocks (mid_tone 110 apart -> feature
    # distance 0.43). Afternoon: fast oscillation with a weaker boundary
    # (mid_tone 80 apart -> 0.31), which is what must give way first.
    def _day(self) -> list[SegmentStats]:
        morning = [_seg(mid=40)] * 4 + [_seg(mid=150)] * 4 + [_seg(mid=250)] * 4
        afternoon = [_seg(mid=100 if i % 2 == 0 else 180) for i in range(20)]
        return morning + afternoon

    def test_calm_region_keeps_its_span_boundaries(self):
        spans, durations = merge_adjacent_segments(self._day(), segment_duration=30.0, max_spans=8)
        # The morning's three blocks survive intact, in order, at full duration.
        assert [s.mid_tone for s in spans[:3]] == [40.0, 150.0, 250.0]
        assert durations[:3] == [120.0, 120.0, 120.0]

    def test_busy_region_absorbs_the_whole_coarsening(self):
        spans, durations = merge_adjacent_segments(self._day(), segment_duration=30.0, max_spans=8)
        assert len(spans) == 8
        # 12 calm segments in 3 spans; the 20 busy ones fold into the rest.
        assert sum(durations[3:]) == pytest.approx(20 * 30.0)
        assert sum(durations) == pytest.approx(32 * 30.0)

    def test_coarsening_never_exceeds_the_span_budget(self):
        """Encode cost is bounded by max_spans exactly as before."""
        segs = [
            _seg(black=5 + (i % 7) * 30, white=90 + (i % 5) * 30, mid=40 + (i % 9) * 20)
            for i in range(100)
        ]
        spans, durations = merge_adjacent_segments(segs, segment_duration=30.0, max_spans=12)
        assert len(spans) <= 12
        assert sum(durations) == pytest.approx(100 * 30.0)

    def test_day_that_does_not_overflow_is_untouched(self):
        day = _seg(black=20, white=220, mid=140, sat=0.35)
        tunnel = _seg(black=2, white=80, mid=30, sat=0.15)
        segs = [day] * 4 + [tunnel] * 3 + [day] * 5
        spans, durations = merge_adjacent_segments(segs, segment_duration=30.0, max_spans=12)
        assert durations == [120.0, 90.0, 150.0]
        assert [s.mid_tone for s in spans] == [140.0, 30.0, 140.0]

    def test_run_count_exactly_at_cap_is_not_coarsened(self):
        segs: list[SegmentStats] = []
        for i in range(6):
            segs += [_seg(mid=40 if i % 2 == 0 else 200)] * 2
        spans, durations = merge_adjacent_segments(segs, segment_duration=30.0, max_spans=6)
        assert len(spans) == 6
        assert durations == [60.0] * 6


class TestShortEventSurvivesAggregation:
    """A sub-sample-interval excursion must not be median-voted away."""

    BRIGHT = FrameStats(black_point=20, white_point=220, mid_tone=140, saturation=0.35)
    TUNNEL = FrameStats(black_point=2, white_point=80, mid_tone=30, saturation=0.15)

    def test_short_tunnel_moves_the_segment_stats(self):
        seg = aggregate_segment_stats([self.BRIGHT, self.BRIGHT, self.TUNNEL])
        # Median would have returned the bright frame verbatim, discarding the
        # event entirely. The aggregate must move toward it, in proportion.
        assert seg.mid_tone == pytest.approx((140 + 140 + 30) / 3)
        assert seg.white_point == pytest.approx((220 + 220 + 80) / 3)
        assert seg.black_point == pytest.approx((20 + 20 + 2) / 3)
        assert seg.saturation == pytest.approx((0.35 + 0.35 + 0.15) / 3)

    def test_excursion_shifts_only_by_its_time_share(self):
        """Self-limiting: a rare event on a long aggregate barely moves it."""
        seg = aggregate_segment_stats([self.BRIGHT] * 99 + [self.TUNNEL])
        assert seg.mid_tone == pytest.approx(140 - 110 / 100, abs=0.01)

    def test_small_fluctuations_are_still_rejected(self):
        """No false positives: sensor/scene noise keeps the robust median."""
        frames = [
            FrameStats(black_point=20, white_point=220, mid_tone=140, saturation=0.35),
            FrameStats(black_point=21, white_point=222, mid_tone=142, saturation=0.36),
            FrameStats(black_point=19, white_point=218, mid_tone=138, saturation=0.34),
        ]
        seg = aggregate_segment_stats(frames)
        assert seg.mid_tone == 140.0
        assert seg.white_point == 220.0
        assert seg.black_point == 20.0
        assert seg.saturation == pytest.approx(0.35)

    def test_moderate_scene_drift_is_still_rejected(self):
        """Foliage-vs-open-road drift under one illuminant is not an event."""
        frames = [
            FrameStats(black_point=18, white_point=215, mid_tone=130, saturation=0.33),
            FrameStats(black_point=22, white_point=225, mid_tone=145, saturation=0.37),
            FrameStats(black_point=20, white_point=220, mid_tone=140, saturation=0.35),
        ]
        seg = aggregate_segment_stats(frames)
        assert seg.mid_tone == 140.0
        assert seg.white_point == 220.0

    def test_single_frame_and_empty_unchanged(self):
        one = aggregate_segment_stats([self.TUNNEL])
        assert one.mid_tone == 30.0
        empty = aggregate_segment_stats([])
        assert empty.mid_tone == 128.0
