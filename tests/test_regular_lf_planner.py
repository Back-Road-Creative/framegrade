"""Tests for regular_lf_planner, part of the day-first design.

Regular LFs are the best continuous ranges of the day's super LF, in
temporal order. Not highlight compilations, not crosscut reels. The
planner consumes the merged per-second score array from
UnifiedSegmentScorer and produces a list of RegularLFRange objects.

Locked behavior:
  - Each range is contiguous (no cuts within a range)
  - Duration is in [min_duration_sec, max_duration_sec] (default 8-30 min)
  - Ranges never overlap
  - Ranges are returned sorted by start_sec (temporal order)
  - At most `max_count` ranges returned; fewer if the day is too short
    or not enough distinct high-interest zones exist
"""

import numpy as np


# Score-array column indices (mirror unified_scorer constants)
COL_TIMESTAMP = 0
COL_COMPOSITE = 1
NUM_COLUMNS = 9


def _make_scores(duration_sec: int, composite_fn=None) -> np.ndarray:
    """Build a (duration_sec, NUM_COLUMNS) score array with 1-second resolution.

    composite_fn: callable(t_sec) -> score in [0, 1]. Defaults to flat 0.5.
    """
    arr = np.zeros((duration_sec, NUM_COLUMNS), dtype=np.float64)
    arr[:, COL_TIMESTAMP] = np.arange(duration_sec, dtype=np.float64)
    if composite_fn is None:
        arr[:, COL_COMPOSITE] = 0.5
    else:
        arr[:, COL_COMPOSITE] = np.array(
            [composite_fn(t) for t in range(duration_sec)], dtype=np.float64
        )
    return arr


def _peak(center: int, width: int, height: float = 0.9, base: float = 0.2):
    """Return a callable that produces a Gaussian-ish bump at `center`."""

    def fn(t):
        dist = abs(t - center)
        if dist > width:
            return base
        # Linear ramp down from height at center to base at width
        return base + (height - base) * (1 - dist / width)

    return fn


class TestRegularLFRangeBasics:
    def test_range_has_duration_properties(self):
        from gomoveshift.pipeline.regular_lf_planner import RegularLFRange

        r = RegularLFRange(start_sec=600, end_sec=1800, integrated_score=500.0)
        assert r.duration_sec == 1200
        assert r.duration_min == 20.0
        assert r.integrated_score == 500.0


class TestPlanRegularLFsEdgeCases:
    def test_empty_scores_returns_empty(self):
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        scores = np.zeros((0, NUM_COLUMNS), dtype=np.float64)
        result = plan_regular_lfs(scores, mode="best_moments")
        assert result == []

    def test_scores_shorter_than_min_duration_returns_empty(self):
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        # 5 minutes of scores, min_duration = 8 minutes → no valid LF possible
        scores = _make_scores(300)
        result = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=480,
            max_duration_sec=1800,
            target_window_sec=1200,
        )
        assert result == []

    def test_scores_exactly_min_duration_returns_one_lf(self):
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        # Exactly 8 minutes of scores. Target window 8 min. Should produce
        # exactly one LF covering the entire range.
        scores = _make_scores(480)
        result = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=480,
            max_duration_sec=1800,
            target_window_sec=480,
        )
        assert len(result) == 1
        assert result[0].start_sec == 0
        assert result[0].end_sec == 480

    def test_max_count_caps_output(self):
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        # 2 hours of uniform scores, target 20 min windows, max_count=2
        scores = _make_scores(7200)
        result = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=480,
            max_duration_sec=1800,
            target_window_sec=1200,
            max_count=2,
        )
        assert len(result) <= 2


class TestPlanRegularLFsOrdering:
    def test_results_sorted_by_start_sec(self):
        """Even when best-scoring windows are discovered out of order,
        results must come back sorted chronologically."""
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        # 90-minute day with three peaks: at t=5400 (late), t=900 (early), t=3000 (mid)
        def fn(t):
            score = 0.2
            for center in (5400, 900, 3000):
                if abs(t - center) < 600:
                    score = max(score, 0.9 - abs(t - center) / 600 * 0.5)
            return score

        scores = _make_scores(5400, composite_fn=fn)
        result = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=480,
            max_duration_sec=1800,
            target_window_sec=1200,
            max_count=3,
        )
        assert len(result) >= 2
        # Each window's start time must be strictly greater than the previous
        starts = [r.start_sec for r in result]
        assert starts == sorted(starts)

    def test_results_never_overlap(self):
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        # 2-hour day, request up to 4 LFs
        scores = _make_scores(7200)
        result = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=480,
            max_duration_sec=1800,
            target_window_sec=1200,
            max_count=4,
        )
        # Sort by start time (should already be sorted, but defensive)
        sorted_result = sorted(result, key=lambda r: r.start_sec)
        for i in range(1, len(sorted_result)):
            prev = sorted_result[i - 1]
            curr = sorted_result[i]
            assert curr.start_sec >= prev.end_sec, (
                f"LF {i} overlaps previous: prev.end={prev.end_sec}, curr.start={curr.start_sec}"
            )


class TestPlanRegularLFsScoring:
    def test_highest_scoring_window_picked_first(self):
        """When peaks are clearly separated, the highest-integrated-score
        window gets picked first (from the internal sorted list). After
        returning we see chronological order but the chosen ranges MUST
        include the highest-score region."""
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        # 3-hour day with one obvious peak at t=5400 (90 min in)
        def fn(t):
            if 4800 < t < 6000:
                return 0.95  # strong peak
            return 0.1

        scores = _make_scores(10800, composite_fn=fn)
        result = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=480,
            max_duration_sec=1800,
            target_window_sec=1200,
            max_count=1,  # force single LF — must pick the peak region
        )
        assert len(result) == 1
        # The LF must cover the peak region
        lf = result[0]
        assert lf.start_sec <= 5400 <= lf.end_sec

    def test_integrated_score_matches_sum_of_composite_in_range(self):
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        # All 0.5 scores — easy arithmetic
        scores = _make_scores(1800)
        result = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=480,
            max_duration_sec=1800,
            target_window_sec=1200,
            max_count=1,
        )
        assert len(result) == 1
        # 1200 seconds of 0.5 = 600.0
        assert abs(result[0].integrated_score - 600.0) < 0.01

    def test_multiple_non_overlapping_peaks_each_selected(self):
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        # 2-hour day with 3 well-separated peaks
        peak_centers = [1200, 3600, 6000]  # 20min, 60min, 100min

        def fn(t):
            for c in peak_centers:
                if abs(t - c) < 600:
                    return 0.9
            return 0.1

        scores = _make_scores(7200, composite_fn=fn)
        result = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=480,
            max_duration_sec=1800,
            target_window_sec=1200,
            max_count=3,
        )
        assert len(result) == 3
        # Each LF's center should be near one of the peaks
        lf_centers = [(r.start_sec + r.end_sec) / 2 for r in result]
        for center in lf_centers:
            assert any(abs(center - pc) < 700 for pc in peak_centers), (
                f"LF centered at {center} is not near any peak: {peak_centers}"
            )


class TestPlanRegularLFsDurationBounds:
    def test_chosen_windows_respect_min_duration(self):
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        scores = _make_scores(7200)
        result = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=600,  # 10 min min
            max_duration_sec=1800,
            target_window_sec=1200,
        )
        for r in result:
            assert r.duration_sec >= 600

    def test_chosen_windows_respect_max_duration(self):
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        scores = _make_scores(7200)
        result = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=480,
            max_duration_sec=900,  # 15 min max
            target_window_sec=900,
        )
        for r in result:
            assert r.duration_sec <= 900

    def test_target_window_used_when_within_bounds(self):
        """When target_window_sec is within [min, max], the planner produces
        windows of exactly that duration (modulo edge-of-track clipping)."""
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        scores = _make_scores(7200)
        result = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=480,
            max_duration_sec=1800,
            target_window_sec=1200,  # 20 min target
            max_count=2,
        )
        for r in result:
            # Allow small clipping at the very start/end; core windows
            # should hit 1200 exactly
            assert 1100 <= r.duration_sec <= 1200


class TestPlanRegularLFsLongDayFixture:
    """Realistic fixture: a 3-hour day with 2 high-interest
    zones and one low-interest transit section. The planner should pick
    both high-interest zones, not the transit."""

    def test_long_day_picks_both_high_interest_zones(self):
        from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs

        # 3-hour day
        # 0-45 min: interesting coastal drive (high score)
        # 45-105 min: transit on interstate (low score)
        # 105-180 min: interesting mountain drive (high score)
        def fn(t):
            if t < 45 * 60:
                return 0.8
            if t < 105 * 60:
                return 0.15
            return 0.85

        scores = _make_scores(180 * 60, composite_fn=fn)
        result = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=480,
            max_duration_sec=1800,
            target_window_sec=1200,
            max_count=3,
        )

        # At least 2 LFs should be produced
        assert len(result) >= 2

        # The two highest-scored LFs should be in the high-interest zones,
        # not the transit
        # Zone 1: 0-2700, Zone 2: 6300-10800
        for r in result:
            center = (r.start_sec + r.end_sec) / 2
            in_zone_1 = 0 <= center <= 2700
            in_zone_2 = 6300 <= center <= 10800
            assert in_zone_1 or in_zone_2, f"LF at {center}s fell in the transit dead zone"

        # Results in temporal order
        starts = [r.start_sec for r in result]
        assert starts == sorted(starts)
