"""Tests for short_planner: picking shorts from the scored day timeline.

Shorts are picked by the same scored-timeline
input as regular LFs, but with different constraints: 15-55 second clips,
target ~9 per day, spaced out across the day so the published shorts
cover the drive's full arc rather than clustering on one peak.

Unlike regular LFs (which return chronologically), shorts return in
RANK order (best score first) so the pipeline can use rank as the
publish sequence — short_01 is the day's best content, short_02 second
best, etc.
"""

import logging

import numpy as np
import pytest


# Score-array column indices (mirror unified_scorer)
COL_TIMESTAMP = 0
COL_COMPOSITE = 1
NUM_COLUMNS = 9


def _make_scores(duration_sec: int, composite_fn=None) -> np.ndarray:
    arr = np.zeros((duration_sec, NUM_COLUMNS), dtype=np.float64)
    arr[:, COL_TIMESTAMP] = np.arange(duration_sec, dtype=np.float64)
    if composite_fn is None:
        arr[:, COL_COMPOSITE] = 0.5
    else:
        arr[:, COL_COMPOSITE] = np.array(
            [composite_fn(t) for t in range(duration_sec)], dtype=np.float64
        )
    return arr


class TestShortRangeBasics:
    def test_range_has_duration_and_rank(self):
        from gomoveshift.pipeline.short_planner import ShortRange

        r = ShortRange(start_sec=120, end_sec=158, integrated_score=30.0, rank=1)
        assert r.duration_sec == 38
        assert r.rank == 1
        assert r.integrated_score == 30.0


class TestPlanShortsEdgeCases:
    def test_empty_scores_returns_empty(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        result = plan_shorts(np.zeros((0, NUM_COLUMNS), dtype=np.float64))
        assert result == []

    def test_scores_shorter_than_min_duration_returns_empty(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        # 10 sec of scores, default min_duration_sec=15 → not enough for
        # any short.
        result = plan_shorts(_make_scores(10))
        assert result == []

    def test_scores_exactly_target_duration_has_no_window_above_its_own_mean(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        # 38 sec → the only window IS the whole timeline, so its mean equals the
        # timeline mean and cannot be strictly above it: no short.
        assert plan_shorts(_make_scores(38)) == []


class TestPlanShortsRanking:
    def test_results_returned_in_rank_order_best_first(self):
        """Shorts return sorted by integrated_score desc, NOT chronologically."""
        from gomoveshift.pipeline.short_planner import plan_shorts

        # 10 minutes with peaks at t=60 (medium), t=300 (high), t=540 (low)
        def fn(t):
            if 40 < t < 90:
                return 0.5
            if 280 < t < 340:
                return 0.95
            if 520 < t < 560:
                return 0.25
            return 0.05

        scores = _make_scores(600, composite_fn=fn)
        result = plan_shorts(
            scores,
            min_duration_sec=30,
            max_duration_sec=45,
            target_duration_sec=38,
            target_count=3,
            min_gap_sec=30,
        )

        assert len(result) == 3
        # Ranks assigned 1, 2, 3 in that order
        assert [r.rank for r in result] == [1, 2, 3]
        # Integrated scores are monotonically decreasing
        scores_list = [r.integrated_score for r in result]
        assert scores_list == sorted(scores_list, reverse=True)

    def test_rank_1_is_the_best_scoring_window(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        # Single clear peak at t=300
        def fn(t):
            if 280 < t < 340:
                return 0.95
            return 0.1

        scores = _make_scores(600, composite_fn=fn)
        result = plan_shorts(
            scores,
            target_count=3,
            min_gap_sec=30,
        )
        assert len(result) >= 1
        rank_1 = next(r for r in result if r.rank == 1)
        # Rank 1's center is in the peak region
        center = (rank_1.start_sec + rank_1.end_sec) / 2
        assert 280 <= center <= 340


class TestPlanShortsSpacing:
    def test_shorts_non_overlapping(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = _make_scores(3600)  # 1 hour flat
        result = plan_shorts(
            scores,
            target_count=9,
            min_gap_sec=30,
        )
        sorted_by_time = sorted(result, key=lambda r: r.start_sec)
        for i in range(1, len(sorted_by_time)):
            prev = sorted_by_time[i - 1]
            curr = sorted_by_time[i]
            assert curr.start_sec >= prev.end_sec, f"Short {i} overlaps previous: {prev}, {curr}"

    def test_shorts_respect_min_gap(self):
        """Consecutive shorts must be at least min_gap_sec apart (end-to-start)."""
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = _make_scores(3600)
        result = plan_shorts(
            scores,
            target_count=9,
            min_gap_sec=60,  # 1 minute min gap
        )
        sorted_by_time = sorted(result, key=lambda r: r.start_sec)
        for i in range(1, len(sorted_by_time)):
            prev_end = sorted_by_time[i - 1].end_sec
            curr_start = sorted_by_time[i].start_sec
            gap = curr_start - prev_end
            assert gap >= 60, f"Gap {gap}s < min_gap_sec=60 between shorts"

    def test_target_count_cap(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = _make_scores(3600)
        result = plan_shorts(scores, target_count=5, min_gap_sec=30)
        assert len(result) <= 5

    def test_long_day_with_many_peaks_returns_target_count(self):
        """A 2-hour day with 9 well-separated peaks should yield 9 shorts."""
        from gomoveshift.pipeline.short_planner import plan_shorts

        peak_centers = [300, 600, 900, 1200, 2400, 2700, 3000, 3300, 4200]

        def fn(t):
            for c in peak_centers:
                if abs(t - c) < 25:
                    return 0.9
            return 0.1

        scores = _make_scores(5400, composite_fn=fn)
        result = plan_shorts(
            scores,
            target_count=9,
            min_gap_sec=60,
        )
        assert len(result) == 9


class TestPlanShortsDurationBounds:
    def test_duration_within_min_max(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = _make_scores(3600)
        result = plan_shorts(
            scores,
            min_duration_sec=30,
            max_duration_sec=45,
            target_duration_sec=38,
        )
        for r in result:
            assert 30 <= r.duration_sec <= 45

    def test_target_duration_used_when_within_bounds(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = _make_scores(3600)
        result = plan_shorts(
            scores,
            min_duration_sec=30,
            max_duration_sec=45,
            target_duration_sec=38,
        )
        # All shorts should be exactly 38s (except edge clipping)
        for r in result:
            assert 37 <= r.duration_sec <= 38


class TestPlanShortsRealisticFixture:
    def test_realistic_day_picks_distributed_shorts(self):
        """3-hour day with continuous moderate scoring and two clear peaks.
        Planner should pick the two peaks (the only windows above the
        timeline mean), ranked 1-2, never padding to the target of 9."""
        from gomoveshift.pipeline.short_planner import plan_shorts

        def fn(t):
            # Two peaks at t=1800 (30 min) and t=7200 (2 hr)
            if 1780 < t < 1820:
                return 0.95
            if 7180 < t < 7220:
                return 0.95
            # Baseline moderate interest
            return 0.4

        scores = _make_scores(10800, composite_fn=fn)
        result = plan_shorts(
            scores,
            target_count=9,
            min_gap_sec=120,
        )

        # Only the two peaks clear the timeline-mean bar; the moderate baseline
        # does not, so the day yields 2 shorts, not the target of 9.
        assert len(result) == 2

        # Rank 1 and rank 2 should be at or near the peaks
        top_two = sorted(result, key=lambda r: r.rank)[:2]
        top_centers = [(r.start_sec + r.end_sec) / 2 for r in top_two]
        for center in top_centers:
            near_peak_1 = abs(center - 1800) < 60
            near_peak_2 = abs(center - 7200) < 60
            assert near_peak_1 or near_peak_2, (
                f"Top-ranked short at center {center} not near either peak"
            )

        # All shorts non-overlapping and respecting min gap
        sorted_by_time = sorted(result, key=lambda r: r.start_sec)
        for i in range(1, len(sorted_by_time)):
            gap = sorted_by_time[i].start_sec - sorted_by_time[i - 1].end_sec
            assert gap >= 120, f"Gap {gap}s < min_gap_sec=120"


class TestConfigFlowToShorts:
    """Verify that DriveConfig.shorts_target_count and shorts_min_gap_sec
    are correctly wired through day_pipeline to plan_shorts."""

    def test_config_fields_have_backward_compatible_defaults(self):
        """New config fields default to the existing hardcoded values."""
        from gomoveshift.config import DriveConfig

        dc = DriveConfig()
        assert dc.shorts_target_count == 9
        assert dc.shorts_min_gap_sec == 60.0

    def test_target_count_14_yields_up_to_14_shorts(self):
        """With target_count=14 and min_gap_sec=45, a sufficiently long
        timeline should yield up to 14 shorts."""
        from gomoveshift.pipeline.short_planner import plan_shorts

        # 14 well-separated peaks across a 2-hour timeline
        peak_centers = [
            120,
            300,
            480,
            660,
            840,
            1200,
            1500,
            1800,
            2100,
            2700,
            3000,
            3600,
            4200,
            5400,
        ]

        def fn(t):
            for c in peak_centers:
                if abs(t - c) < 25:
                    return 0.9
            return 0.1

        scores = _make_scores(7200, composite_fn=fn)
        result = plan_shorts(
            scores,
            target_count=14,
            min_gap_sec=45,
        )
        assert len(result) == 14

    def test_config_shorts_target_count_overrides_default(self):
        """DriveConfig.shorts_target_count can be set to a non-default value."""
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_target_count=14, shorts_min_gap_sec=45.0)
        assert dc.shorts_target_count == 14
        assert dc.shorts_min_gap_sec == 45.0

    def test_consistency_warning_logged_when_target_below_min(self, caplog):
        """When shorts_target_count < min_shorts_per_lf, a warning is logged
        during config consistency check."""
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_target_count=5, min_shorts_per_lf=7)
        with caplog.at_level(logging.WARNING):
            dc.check_shorts_consistency()
        msgs = [m for m in caplog.messages if "shorts_target_count" in m]
        assert msgs
        # The count is a maximum: no validation gate requires min_shorts_per_lf.
        assert all("validation gate" not in m and "requires" not in m for m in msgs)
        assert any("will cap at 5" in m for m in msgs)

    def test_no_warning_when_target_gte_min(self, caplog):
        """No warning when shorts_target_count >= min_shorts_per_lf."""
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_target_count=14, min_shorts_per_lf=7)
        with caplog.at_level(logging.WARNING):
            dc.check_shorts_consistency()
        assert not any("shorts_target_count" in msg for msg in caplog.messages)


class TestQualityBar:
    """No count-filling: a window qualifies only when its mean composite is
    strictly above the eligible timeline's mean, the gap is never relaxed, and
    target_count is a maximum. Fewer shorts (even zero) is a valid plan."""

    @staticmethod
    def _peaks(centers, n=3600, hi=0.9, lo=0.1, half=25):
        return _make_scores(
            n, composite_fn=lambda t: hi if any(abs(t - c) < half for c in centers) else lo
        )

    def test_flat_timeline_yields_fewer_than_target(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        # Every window's mean equals the timeline mean: none is strictly above.
        assert plan_shorts(_make_scores(3600), target_count=9, min_gap_sec=60) == []

    def test_nothing_above_mean_returns_zero_without_raising(self, caplog):
        from gomoveshift.pipeline.short_planner import plan_shorts

        with caplog.at_level(logging.INFO):
            result = plan_shorts(_make_scores(600), target_count=7, min_gap_sec=60)
        assert result == []
        assert any("SHORTS-PLAN: 0 of target 7 clear the quality bar" in m for m in caplog.messages)

    def test_only_windows_above_the_timeline_mean_are_picked(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = self._peaks([600, 1800, 3000])
        mean = float(scores[:, COL_COMPOSITE].mean())
        result = plan_shorts(scores, target_count=9, min_gap_sec=60)
        assert 1 <= len(result) < 9
        for r in result:
            window = scores[
                (scores[:, COL_TIMESTAMP] >= r.start_sec) & (scores[:, COL_TIMESTAMP] < r.end_sec)
            ]
            assert window[:, COL_COMPOSITE].mean() > mean

    def test_target_count_is_still_a_maximum(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = self._peaks([300, 900, 1500, 2100, 2700])
        assert len(plan_shorts(scores, target_count=2, min_gap_sec=60)) == 2

    def test_gap_is_never_relaxed_to_reach_the_target(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        # Three peaks 60s apart centre to centre: the old ladder would fill the
        # extras at half gap / gap 0. Now only the gap-respecting ones remain.
        scores = self._peaks([300, 360, 420], n=900, half=20)
        result = plan_shorts(scores, target_count=3, min_gap_sec=60)
        assert len(result) < 3
        ordered = sorted(result, key=lambda r: r.start_sec)
        for prev, nxt in zip(ordered, ordered[1:]):
            assert nxt.start_sec - prev.end_sec >= 60

    def test_gap_loss_is_logged(self, caplog):
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = self._peaks([300, 360, 420], n=900, half=20)
        with caplog.at_level(logging.INFO):
            result = plan_shorts(scores, target_count=3, min_gap_sec=60)
        line = next(
            m
            for m in caplog.messages
            if m.startswith("SHORTS-PLAN: ") and "clear the quality bar" in m
        )
        assert line.startswith(f"SHORTS-PLAN: {len(result)} of target 3 clear the quality bar (")
        assert "candidates below the timeline mean" in line
        assert "lost to the 60s gap)" in line
        lost = int(line.split("mean, ")[1].split(" lost")[0])
        assert lost >= 1

    def test_stride_is_never_halved(self, monkeypatch):
        from gomoveshift.pipeline import short_planner

        strides = []
        real = short_planner._build_candidate_windows

        def spy(timestamps, composites, window_sec, stride_sec, stop_mask=None):
            strides.append(stride_sec)
            return real(timestamps, composites, window_sec, stride_sec, stop_mask=stop_mask)

        monkeypatch.setattr(short_planner, "_build_candidate_windows", spy)
        short_planner.plan_shorts(
            self._peaks([300], n=900), target_count=5, min_gap_sec=60, stride_sec=5.0
        )
        assert strides == [5.0]

    def test_one_log_line_per_chosen_short_with_its_mean(self, caplog):
        from gomoveshift.pipeline.short_planner import plan_shorts

        with caplog.at_level(logging.INFO):
            result = plan_shorts(self._peaks([600, 1800]), target_count=9, min_gap_sec=60)
        per_short = [m for m in caplog.messages if m.startswith("SHORTS-PLAN: short ")]
        assert len(per_short) == len(result) == 2
        assert all("mean composite" in m for m in per_short)

    def test_explicit_quality_bar_overrides_the_timeline_mean(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = self._peaks([600, 1800])
        assert plan_shorts(scores, target_count=9, min_gap_sec=60, quality_bar=0.95) == []
        assert len(plan_shorts(scores, target_count=9, min_gap_sec=60, quality_bar=0.0)) == 9

    def test_bar_excludes_stopped_seconds(self):
        from gomoveshift.pipeline.short_planner import plan_shorts, timeline_quality_bar

        scores = self._peaks([600])
        mask = np.zeros(len(scores), dtype=bool)
        mask[560:640] = True  # the only peak is stopped
        assert timeline_quality_bar(scores, mask) == pytest.approx(0.1)
        assert plan_shorts(scores, target_count=9, min_gap_sec=60, stop_mask=mask) == []

    def test_no_eligible_second_has_no_bar(self):
        from gomoveshift.pipeline.short_planner import timeline_quality_bar

        scores = _make_scores(100)
        assert timeline_quality_bar(scores, np.ones(100, dtype=bool)) is None
        assert timeline_quality_bar(np.zeros((0, NUM_COLUMNS))) is None


class TestPlannerDeterminism:
    """Same scores in → same picks out, regardless of input ordering."""

    def _varied_scores(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        scores = _make_scores(900)
        scores[:, COL_COMPOSITE] = rng.random(900)
        return scores

    def test_plan_shorts_twice_identical(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = self._varied_scores()
        a = plan_shorts(scores, target_count=9, min_gap_sec=60)
        b = plan_shorts(scores, target_count=9, min_gap_sec=60)
        assert [(r.start_sec, r.end_sec, r.integrated_score, r.rank) for r in a] == [
            (r.start_sec, r.end_sec, r.integrated_score, r.rank) for r in b
        ]

    def test_greedy_pick_invariant_under_candidate_order(self):
        """Permuting the candidate-window list does not change the picks —
        tie-breaks are explicit (score desc, then start-time asc)."""
        import random

        from gomoveshift.pipeline.short_planner import _greedy_pick_with_gap

        windows = [(float(t), float(t + 38), 1.0 + (t % 3)) for t in range(0, 500, 20)]
        baseline = _greedy_pick_with_gap(windows, target_count=5, min_gap_sec=30)
        shuffled = list(windows)
        random.Random(7).shuffle(shuffled)
        assert _greedy_pick_with_gap(shuffled, target_count=5, min_gap_sec=30) == baseline


class TestPlanShortsStopMask:
    """The stop mask STRUCTURALLY vetoes candidate windows overlapping a
    stopped second — nothing relaxes the
    mask, so a masked (parked) region can never surface a short."""

    def _peaked_scores(self, n: int = 900) -> np.ndarray:
        # One dominant high-score plateau in [400, 460); baseline low elsewhere.
        scores = _make_scores(n, composite_fn=lambda t: 0.9 if 400 <= t < 460 else 0.05)
        return scores

    def test_mask_over_only_high_score_region_yields_no_window_there(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = self._peaked_scores()
        stop_mask = np.zeros(len(scores), dtype=bool)
        stop_mask[395:465] = True  # covers the plateau + its window edges

        # Even with a large target
        # no chosen short may overlap the masked span.
        result = plan_shorts(scores, target_count=20, min_gap_sec=60, stop_mask=stop_mask)
        for r in result:
            assert not (r.start_sec < 465 and r.end_sec > 395), (
                f"short {r.start_sec}-{r.end_sec} overlaps the masked region"
            )

    def test_no_mask_is_unchanged_behavior(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = self._peaked_scores()
        baseline = plan_shorts(scores, target_count=5, min_gap_sec=60)
        with_none = plan_shorts(scores, target_count=5, min_gap_sec=60, stop_mask=None)
        assert [(r.start_sec, r.end_sec, r.rank) for r in baseline] == [
            (r.start_sec, r.end_sec, r.rank) for r in with_none
        ]

    def test_masking_the_peak_removes_the_rank1_short(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = self._peaked_scores()
        unmasked = plan_shorts(scores, target_count=3, min_gap_sec=60)
        # Without a mask the top-ranked short sits on the plateau.
        assert any(r.start_sec >= 395 and r.end_sec <= 465 for r in unmasked)

        stop_mask = np.zeros(len(scores), dtype=bool)
        stop_mask[395:465] = True
        masked = plan_shorts(scores, target_count=3, min_gap_sec=60, stop_mask=stop_mask)
        assert all(not (r.start_sec < 465 and r.end_sec > 395) for r in masked)


class TestMomentLengthVariesPerMoment:
    """Clip length follows the moment (15-55s) rather than a fixed
    target -- a still-rising moment grows past the target width; a flat
    or already-declining one (the common case) is unaffected."""

    def test_flat_moment_stays_at_target_duration(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        # One flat plateau above a low baseline: no growth signal inside it.
        scores = _make_scores(600, composite_fn=lambda t: 0.5 if 200 <= t < 300 else 0.25)
        result = plan_shorts(scores, target_count=1, min_gap_sec=60)
        assert len(result) == 1
        assert 37 <= result[0].duration_sec <= 38
        assert result[0].moment_kind == "peak", (
            "a flat moment that never grew must be classified 'peak', not 'sustained'"
        )

    def test_sustained_rising_moment_grows_past_target_duration(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        def fn(t):
            if t < 300:
                return 0.1
            if t < 396:
                return 0.5 + (t - 300) * (0.4 / 96.0)  # continuous rise 0.5 -> 0.9
            return 0.05  # sharp decline -- the moment's boundary

        scores = _make_scores(700, composite_fn=fn)
        result = plan_shorts(scores, target_count=1, min_gap_sec=60)
        assert len(result) == 1
        assert result[0].duration_sec > 38, (
            f"a still-rising moment should grow past the 38s target, got {result[0].duration_sec}"
        )
        assert result[0].duration_sec <= 55
        assert result[0].moment_kind == "sustained", (
            "a window that actually grew past its search width must be classified 'sustained'"
        )

    def test_growth_never_exceeds_max_duration_sec(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        def fn(t):
            return 0.1 + t / 2000.0  # monotonically increasing, no ceiling of its own

        scores = _make_scores(2000, composite_fn=fn)
        result = plan_shorts(scores, target_count=1, min_gap_sec=60, max_duration_sec=55)
        assert len(result) == 1
        assert result[0].duration_sec <= 55

    def test_growth_respects_min_gap_to_the_next_short(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        def fn(t):
            if 100 <= t < 400:
                return 0.2 + (t - 100) / 1000.0  # monotonically increasing
            if 450 <= t < 500:
                return 0.6
            return 0.05

        scores = _make_scores(700, composite_fn=fn)
        result = plan_shorts(scores, target_count=2, min_gap_sec=30)
        assert len(result) == 2
        by_time = sorted(result, key=lambda r: r.start_sec)
        gap = by_time[1].start_sec - by_time[0].end_sec
        assert gap >= 30

    def test_growth_never_invades_a_stop_masked_second(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        def fn(t):
            if 300 <= t < 400:
                return 0.2 + (t - 300) / 200.0  # monotonically increasing
            return 0.05

        scores = _make_scores(700, composite_fn=fn)
        stop_mask = np.zeros(len(scores), dtype=bool)
        stop_mask[380:] = True
        result = plan_shorts(scores, target_count=1, min_gap_sec=60, stop_mask=stop_mask)
        assert len(result) == 1
        assert result[0].end_sec <= 380
