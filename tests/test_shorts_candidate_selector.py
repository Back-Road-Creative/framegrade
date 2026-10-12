"""Tests for gomoveshift.analysis.shorts_candidate_selector — ShortsCandidateSelector."""

import logging
from unittest.mock import MagicMock

import pytest

from gomoveshift.analysis.shorts_candidate_selector import ShortsCandidateSelector
from gomoveshift.analysis.shorts_models import ScoringWeights
from gomoveshift.models import DeadSpot, HighMotionSegment


# ── Helpers ──────────────────────────────────────────────────────────


def _seg(
    start: float,
    end: float,
    avg: float = 0.05,
    peak: float = 0.10,
) -> HighMotionSegment:
    return HighMotionSegment(
        start_time=start,
        end_time=end,
        avg_motion_energy=avg,
        peak_motion_energy=peak,
    )


def _dead(start: float, end: float) -> DeadSpot:
    return DeadSpot(start_time=start, end_time=end, motion_energy=0.001)


def _selector(**kw) -> ShortsCandidateSelector:
    defaults = dict(
        scoring_weights=ScoringWeights(),
        min_gap=10.0,
        min_duration=15.0,
        max_duration=60.0,
        target_duration=30.0,
    )
    defaults.update(kw)
    return ShortsCandidateSelector(**defaults)


# ── select_best_shorts ──────────────────────────────────────────────


class TestSelectBestShorts:
    def test_empty_candidates_returns_empty(self):
        sel = _selector()
        result = sel.select_best_shorts([], [], video_duration=300.0)
        assert result == []

    def test_single_candidate_selected(self):
        sel = _selector()
        candidates = [_seg(10, 40)]
        result = sel.select_best_shorts(candidates, [], video_duration=300.0, num_shorts=3)
        assert len(result) == 1
        assert result[0].start_time == 10

    def test_respects_num_shorts(self):
        sel = _selector(min_gap=5.0)
        candidates = [_seg(i * 100, i * 100 + 30, avg=0.05 + i * 0.01) for i in range(10)]
        result = sel.select_best_shorts(candidates, [], video_duration=1000.0, num_shorts=3)
        assert len(result) <= 3

    def test_avoids_dead_spots(self):
        """Candidates overlapping dead spots are filtered out."""
        sel = _selector()
        candidates = [
            _seg(10, 40, avg=0.10),  # overlaps dead spot
            _seg(200, 230, avg=0.05),  # clean
        ]
        dead_spots = [_dead(20, 50)]
        result = sel.select_best_shorts(
            candidates,
            dead_spots,
            video_duration=300.0,
            num_shorts=5,
        )
        # If only one passes the filter, that is the one selected
        # (unless fallback kicks in — in which case both are available)
        assert len(result) >= 1

    def test_non_overlapping_selection(self):
        """Selected segments should not be too close together."""
        sel = _selector(min_gap=50.0)
        # 5 candidates all bunched within 20s of each other
        candidates = [_seg(i * 5, i * 5 + 30, avg=0.05 + i * 0.01) for i in range(5)]
        result = sel.select_best_shorts(candidates, [], video_duration=300.0, num_shorts=5)
        # Only 1 can be selected since they're all within 50s min_gap
        assert len(result) == 1

    def test_selected_sorted_by_time(self):
        """Returned segments are ordered by start_time."""
        sel = _selector(min_gap=5.0)
        candidates = [
            _seg(200, 230, avg=0.10),
            _seg(50, 80, avg=0.08),
            _seg(400, 430, avg=0.06),
        ]
        result = sel.select_best_shorts(candidates, [], video_duration=500.0, num_shorts=3)
        for i in range(len(result) - 1):
            assert result[i].start_time <= result[i + 1].start_time

    def test_ranks_assigned_by_energy(self):
        """Ranks should be 1=highest energy, 2=next, etc."""
        sel = _selector(min_gap=5.0)
        candidates = [
            _seg(0, 30, avg=0.03),
            _seg(100, 130, avg=0.10),
            _seg(200, 230, avg=0.06),
        ]
        result = sel.select_best_shorts(candidates, [], video_duration=300.0, num_shorts=3)
        # Rank 1 should have the highest avg_motion_energy
        rank1 = [s for s in result if s.rank == 1]
        if rank1:
            for s in result:
                assert rank1[0].avg_motion_energy >= s.avg_motion_energy

    def test_buffer_only_overlap_keeps_mostly_moving_candidates(self):
        """Crossfade-style dead spots (buffer-only overlap) keep candidates via fallback."""
        sel = _selector()
        # Both segments only overlap the stop via the 5s proximity buffer:
        # their actual footage spends 0% of its duration inside the stop.
        candidates = [_seg(10, 40, avg=0.08), _seg(50, 80, avg=0.06)]
        dead_spots = [_dead(42, 48)]  # short transition between the two
        result = sel.select_best_shorts(
            candidates,
            dead_spots,
            video_duration=300.0,
            num_shorts=5,
        )
        # Fallback: mostly-moving candidates are still selectable
        assert len(result) >= 1

    def test_majority_stopped_candidate_dropped_in_fallback(self):
        """Fallback keeps mostly-moving candidates and drops majority-stopped ones.

        Tightened to the 10% tolerance: A spends ~3% inside the stop (kept),
        B spends ~67% (dropped)."""
        sel = _selector()
        candidates = [_seg(10, 40, avg=0.08), _seg(40, 70, avg=0.10)]
        dead_spots = [_dead(39, 60)]
        result = sel.select_best_shorts(
            candidates,
            dead_spots,
            video_duration=300.0,
            num_shorts=2,
        )
        assert len(result) == 1
        assert result[0].start_time == 10

    def test_half_stopped_candidate_dropped_under_tightened_tolerance(self):
        """In the fallback, a clip 50% inside a stop is no longer shipped
        (it survived at the old 0.5 tolerance)."""
        sel = _selector()
        # Both candidates buffer-overlap the single stop (so the fallback fires),
        # but A is 50% inside it (dropped now) while B is 0% inside (kept).
        candidates = [_seg(10, 40, avg=0.08), _seg(50, 80, avg=0.06)]
        dead_spots = [_dead(25, 46)]
        result = sel.select_best_shorts(
            candidates,
            dead_spots,
            video_duration=300.0,
            num_shorts=5,
        )
        assert [seg.start_time for seg in result] == [50]

    def test_fully_stopped_candidates_are_not_shipped(self):
        """When every candidate is majority inside dead spots, select none."""
        sel = _selector()
        candidates = [_seg(10, 40, avg=0.08), _seg(50, 80, avg=0.06)]
        dead_spots = [_dead(0, 100)]  # covers everything
        result = sel.select_best_shorts(
            candidates,
            dead_spots,
            video_duration=300.0,
            num_shorts=5,
        )
        assert result == []


# ── _segment_overlaps_stop ──────────────────────────────────────────


class TestSegmentOverlapsStop:
    def test_overlapping(self):
        sel = _selector()
        seg = _seg(10, 40)
        stops = [_dead(30, 60)]
        assert sel._segment_overlaps_stop(seg, stops)

    def test_non_overlapping(self):
        sel = _selector()
        seg = _seg(10, 40)
        stops = [_dead(100, 120)]
        assert not sel._segment_overlaps_stop(seg, stops)

    def test_buffer_extends_overlap(self):
        """Buffer makes close-but-non-overlapping segments count as overlapping."""
        sel = _selector()
        seg = _seg(10, 40)
        stops = [_dead(42, 60)]  # gap of 2s < buffer of 5
        assert sel._segment_overlaps_stop(seg, stops, buffer=5.0)

    def test_no_stops_no_overlap(self):
        sel = _selector()
        assert not sel._segment_overlaps_stop(_seg(10, 40), [])


# ── _dead_spot_overlap_fraction ─────────────────────────────────────


class TestDeadSpotOverlapFraction:
    def test_disjoint_is_zero(self):
        sel = _selector()
        assert sel._dead_spot_overlap_fraction(_seg(10, 40), [_dead(100, 120)]) == 0.0

    def test_near_stop_uses_no_buffer(self):
        """Unlike _segment_overlaps_stop, proximity alone contributes nothing."""
        sel = _selector()
        assert sel._dead_spot_overlap_fraction(_seg(10, 40), [_dead(42, 60)]) == 0.0

    def test_fully_covered_is_one(self):
        sel = _selector()
        assert sel._dead_spot_overlap_fraction(_seg(10, 40), [_dead(0, 100)]) == 1.0

    def test_partial_overlap_fraction(self):
        sel = _selector()
        # 10s of the 30s segment inside the stop
        frac = sel._dead_spot_overlap_fraction(_seg(10, 40), [_dead(30, 60)])
        assert abs(frac - 10 / 30) < 1e-9

    def test_multiple_stops_sum_capped_at_one(self):
        sel = _selector()
        frac = sel._dead_spot_overlap_fraction(
            _seg(10, 40),
            [_dead(0, 30), _dead(20, 50)],
        )
        assert frac == 1.0


# ── _segments_too_close ─────────────────────────────────────────────


class TestSegmentsTooClose:
    def test_close_segments(self):
        sel = _selector(min_gap=20.0)
        assert sel._segments_too_close(_seg(10, 40), _seg(45, 75))

    def test_far_segments(self):
        sel = _selector(min_gap=5.0)
        assert not sel._segments_too_close(_seg(10, 40), _seg(100, 130))


# ── _calculate_spread_score ─────────────────────────────────────────


class TestSpreadScore:
    def test_first_segment_gets_full_score(self):
        sel = _selector()
        score = sel._calculate_spread_score(_seg(100, 130), [], video_duration=300.0)
        assert score == 100.0

    def test_far_segment_high_score(self):
        sel = _selector()
        selected = [_seg(10, 40)]
        score = sel._calculate_spread_score(
            _seg(250, 280),
            selected,
            video_duration=300.0,
        )
        assert score > 50.0

    def test_near_segment_low_score(self):
        sel = _selector()
        selected = [_seg(10, 40)]
        score = sel._calculate_spread_score(
            _seg(15, 45),
            selected,
            video_duration=300.0,
        )
        assert score < 50.0

    def test_zero_duration_returns_full(self):
        sel = _selector()
        score = sel._calculate_spread_score(_seg(0, 10), [], video_duration=0.0)
        assert score == 100.0


# ── _score_segment ──────────────────────────────────────────────────


class TestScoreSegment:
    def test_higher_motion_scores_higher(self):
        sel = _selector()
        low = sel._score_segment(_seg(100, 130, avg=0.01, peak=0.02), video_duration=300.0)
        high = sel._score_segment(_seg(100, 130, avg=0.10, peak=0.15), video_duration=300.0)
        assert high > low

    def test_target_duration_match_boosts_score(self):
        """A segment matching target_duration should score higher on duration factor."""
        sel = _selector(target_duration=30.0)
        perfect = sel._score_segment(_seg(100, 130, avg=0.05), video_duration=300.0)
        off = sel._score_segment(_seg(100, 155, avg=0.05), video_duration=300.0)
        # The perfect-duration one should score higher (all else equal)
        assert perfect > off


# ── Shortfall warning + configurable geo dedup ──────────────────────


class TestShortfallWarning:
    """Silent shortfalls must surface a WARNING with rejection breakdown."""

    def test_warning_emitted_when_selection_short(self, caplog):
        """Request more shorts than gap spacing allows -> WARNING logged."""
        sel = _selector(min_gap=50.0)
        # 3 candidates clustered within 10s of each other; min_gap=50s means
        # only 1 can be selected. We ask for 3 -> shortfall of 2.
        candidates = [
            _seg(100, 130, avg=0.10),
            _seg(105, 135, avg=0.09),
            _seg(110, 140, avg=0.08),
        ]
        with caplog.at_level(
            logging.WARNING, logger="gomoveshift.analysis.shorts_candidate_selector"
        ):
            selected = sel.select_best_shorts(candidates, [], video_duration=300.0, num_shorts=3)

        assert len(selected) < 3
        shortfall_records = [r for r in caplog.records if "Shorts shortfall" in r.getMessage()]
        assert shortfall_records, "Expected a 'Shorts shortfall' WARNING log"
        msg = shortfall_records[0].getMessage()
        assert "requested=3" in msg
        assert "gap=" in msg
        assert "geo_dedup=" in msg
        assert "dead_spot_filter=" in msg

    def test_no_warning_when_fully_satisfied(self, caplog):
        """When all requested shorts are selected, no shortfall warning."""
        sel = _selector(min_gap=5.0)
        candidates = [_seg(i * 100, i * 100 + 30, avg=0.05) for i in range(5)]
        with caplog.at_level(
            logging.WARNING, logger="gomoveshift.analysis.shorts_candidate_selector"
        ):
            selected = sel.select_best_shorts(candidates, [], video_duration=1000.0, num_shorts=3)

        assert len(selected) == 3
        shortfall_records = [r for r in caplog.records if "Shorts shortfall" in r.getMessage()]
        assert not shortfall_records


class TestConfigurableGeoDedupThreshold:
    """Geographic dedup distance must be configurable, not hardcoded."""

    def _stubbed_selector(
        self,
        geo_dedup_meters: float,
        min_gap: float = 5.0,
    ) -> ShortsCandidateSelector:
        """Selector wired with a stub scenic_scorer + time_mapper so geo-dedup runs."""
        time_mapper = MagicMock()
        time_mapper.video_to_gps_time.side_effect = lambda t: t

        # Fake point keyed by segment start time. Each 1s of separation
        # corresponds to 0.00001° lat (~1.1m). This lets us test the
        # geo-dedup threshold independent of time-based min_gap.
        def _points_in_range(start, end):
            pt = MagicMock()
            pt.lat = 50.0 + start * 0.00001  # ~1.1m per second of start-time delta
            pt.lon = -9.0
            return [pt]

        scorer = MagicMock()
        scorer.gps_data.get_points_in_range.side_effect = _points_in_range
        scorer.score_segment.return_value = 50.0

        return ShortsCandidateSelector(
            scoring_weights=ScoringWeights(),
            min_gap=min_gap,
            min_duration=15.0,
            max_duration=60.0,
            target_duration=30.0,
            time_mapper=time_mapper,
            scenic_scorer=scorer,
            geo_dedup_meters=geo_dedup_meters,
        )

    def test_default_geo_dedup_is_500m(self):
        """Default geo_dedup_meters should stay 500 for backward compat."""
        sel = ShortsCandidateSelector(
            scoring_weights=ScoringWeights(),
            min_gap=5.0,
            min_duration=15.0,
            max_duration=60.0,
            target_duration=30.0,
        )
        assert sel.geo_dedup_meters == 500.0

    def test_tight_threshold_allows_neighbors(self):
        """With a small geo-dedup radius, geographically-near segments are NOT deduplicated."""
        # Segments 100s apart -> GPS ~110m apart. min_gap=10 to let both through time gate.
        sel = self._stubbed_selector(geo_dedup_meters=10.0, min_gap=10.0)
        segs = [_seg(100, 130, avg=0.10), _seg(200, 230, avg=0.09)]
        selected = sel.select_best_shorts(segs, [], video_duration=500.0, num_shorts=2)
        assert len(selected) == 2

    def test_wide_threshold_rejects_neighbors(self):
        """With a large geo-dedup radius, geographically-near segments ARE deduplicated."""
        # Same 100s / ~110m separation but with a 5km threshold -> dedup triggers.
        sel = self._stubbed_selector(geo_dedup_meters=5000.0, min_gap=10.0)
        segs = [_seg(100, 130, avg=0.10), _seg(200, 230, avg=0.09)]
        selected = sel.select_best_shorts(segs, [], video_duration=500.0, num_shorts=2)
        assert len(selected) == 1
