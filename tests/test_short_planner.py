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

    def test_scores_exactly_target_duration_returns_one_short(self):
        from gomoveshift.pipeline.short_planner import plan_shorts

        # 38 sec → one target-sized short covering everything
        result = plan_shorts(_make_scores(38))
        assert len(result) == 1
        assert result[0].start_sec == 0
        assert result[0].end_sec == 38


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
        Planner should pick 9 shorts spread across the day, with rank 1-2
        landing in the peak regions."""
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

        assert len(result) == 9

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
        assert any("shorts_target_count" in msg for msg in caplog.messages)

    def test_no_warning_when_target_gte_min(self, caplog):
        """No warning when shorts_target_count >= min_shorts_per_lf."""
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_target_count=14, min_shorts_per_lf=7)
        with caplog.at_level(logging.WARNING):
            dc.check_shorts_consistency()
        assert not any("shorts_target_count" in msg for msg in caplog.messages)


class TestRelaxationLadder:
    """Deterministic relaxation ladder: plan_shorts may under-deliver ONLY when the
    timeline physically can't hold target_count windows — never because of
    tunable gap spacing."""

    def test_rung_1_sufficient_stays_silent_and_unchanged(self, caplog):
        """A timeline where rung 1 reaches target emits no SHORTS-PLAN log
        and all picks respect the full min_gap_sec (historical behavior)."""
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = _make_scores(3600)
        with caplog.at_level(logging.INFO):
            result = plan_shorts(scores, target_count=9, min_gap_sec=60)
        assert len(result) == 9
        assert not any("SHORTS-PLAN" in msg for msg in caplog.messages)
        sorted_by_time = sorted(result, key=lambda r: r.start_sec)
        for i in range(1, len(sorted_by_time)):
            assert sorted_by_time[i].start_sec - sorted_by_time[i - 1].end_sec >= 60

    def test_shortfall_fills_at_relaxed_gap(self, caplog):
        """250s flat timeline, gap 60: rung 1 holds only 3 windows of 38s;
        target 5 is reached by gap-0 fills (rung 3) without raising."""
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = _make_scores(250)
        with caplog.at_level(logging.INFO):
            result = plan_shorts(scores, target_count=5, min_gap_sec=60)
        assert len(result) == 5
        assert any("SHORTS-PLAN: target 5 reached at rung 3" in m for m in caplog.messages)
        # Fills are non-overlapping even at gap 0
        sorted_by_time = sorted(result, key=lambda r: r.start_sec)
        for i in range(1, len(sorted_by_time)):
            assert sorted_by_time[i].start_sec >= sorted_by_time[i - 1].end_sec

    def test_prior_rung_picks_immutable_across_rungs(self):
        """The relaxed-target run's first picks are exactly the picks a
        rung-1-sufficient (smaller-target) run makes — later rungs only
        ADD, never drop or reorder."""
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = _make_scores(250)
        rung1_only = plan_shorts(scores, target_count=3, min_gap_sec=60)
        ladder = plan_shorts(scores, target_count=5, min_gap_sec=60)
        assert len(rung1_only) == 3
        assert [(r.start_sec, r.end_sec) for r in ladder[:3]] == [
            (r.start_sec, r.end_sec) for r in rung1_only
        ]

    def test_rung_4_denser_stride_fills(self, caplog):
        """A coarse stride (76s) leaves a 38s hole only a half-stride
        candidate can fill — rung 4 reaches the target."""
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = _make_scores(190)
        with caplog.at_level(logging.INFO):
            result = plan_shorts(scores, target_count=4, min_gap_sec=60, stride_sec=76.0)
        assert len(result) == 4
        assert any("SHORTS-PLAN: target 4 reached at rung 4" in m for m in caplog.messages)
        assert any("rung1=2 rung2=1 rung3=0 rung4=1" in m for m in caplog.messages)

    def test_physical_impossibility_returns_max_without_raising(self, caplog):
        """A 60s timeline physically holds one 38s window: target 5 returns
        1 and logs ladder exhaustion — the SHORTS-CONTRACT gate (not the
        planner) owns the contract failure."""
        from gomoveshift.pipeline.short_planner import plan_shorts

        scores = _make_scores(60)
        with caplog.at_level(logging.INFO):
            result = plan_shorts(scores, target_count=5, min_gap_sec=60)
        assert len(result) == 1
        assert any("ladder exhausted" in m for m in caplog.messages)


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
    stopped second — the relaxation ladder can relax gap/stride, never the
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

        # Even at the fullest relaxation (large target forces the whole ladder)
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

        scores = _make_scores(600)  # flat 0.5 everywhere -- no growth signal
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
