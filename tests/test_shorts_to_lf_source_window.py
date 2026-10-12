"""Source-window shorts↔LF mapping.

Replaces the legacy equal-split assignment with a semantic mapping:
each short attaches to the LF whose source segments contain its
midpoint. Falls back to nearest-LF when a short's source footage is
not part of any LF body (percentile cut / inter-LF gap).

The legacy behaviour produced the correct *count* per LF for CSV
purposes but assigned shorts purely by rank-position — so a short
whose source footage was inside LF #7's body could be attributed to
LF #1 because it had rank #1. This module locks the new behaviour.
"""

from types import SimpleNamespace

from gomoveshift.pipeline.day_pipeline import (
    assign_shorts_to_lfs_by_source_window,
    compute_day_file_boundaries,
)


def _seg(source_file_index: int, start_time: float, end_time: float):
    return SimpleNamespace(
        source_file_index=source_file_index,
        start_time=start_time,
        end_time=end_time,
    )


def _lf(index: int, segments: list):
    return SimpleNamespace(index=index, segments=segments)


def _short(start: float, end: float):
    return SimpleNamespace(start_sec=start, end_sec=end)


def _batch(durations: list[float]):
    """Synthesize a day batch with N episodes of given durations.

    ``compute_day_file_boundaries`` only reads ``episodes[*].total_duration``.
    """
    episodes = [SimpleNamespace(total_duration=d) for d in durations]
    return SimpleNamespace(episodes=episodes)


class TestSourceWindowAssignment:
    def test_short_inside_lf_segments_attaches_to_that_lf(self):
        # Day master: ep0 [0, 600), ep1 [600, 1500).
        boundaries = compute_day_file_boundaries(_batch([600.0, 900.0]))
        # LF #1 covers ep0 [60, 360); LF #2 covers ep1 [60, 360) (master 660-960).
        lf1 = _lf(1, [_seg(0, 60.0, 360.0)])
        lf2 = _lf(2, [_seg(1, 60.0, 360.0)])
        # Short A midpoint=200 (in LF1); Short B midpoint=800 (in LF2).
        shorts = [_short(180.0, 220.0), _short(780.0, 820.0)]
        out = assign_shorts_to_lfs_by_source_window(shorts, [lf1, lf2], boundaries)
        assert out == {1: [0], 2: [1]}

    def test_legacy_equal_split_would_have_misattributed(self):
        """Concrete demonstration that the new behaviour differs from
        the equal-split. Three rank-ordered shorts, two LFs: equal-split
        gives [rank0, rank1] → LF1, [rank2] → LF2 regardless of source.
        Source-window assigns by where each short's midpoint actually
        falls."""
        boundaries = compute_day_file_boundaries(_batch([1200.0]))
        # LF1 owns the first half of the master, LF2 the second half.
        lf1 = _lf(1, [_seg(0, 0.0, 600.0)])
        lf2 = _lf(2, [_seg(0, 600.0, 1200.0)])
        # Rank ordering: best=in LF2, 2nd-best=in LF2, 3rd-best=in LF1.
        # Equal-split would force [rank0, rank1]=LF1 (wrong source) and
        # [rank2]=LF2 (wrong source). Source-window flips it.
        shorts = [
            _short(900.0, 940.0),  # rank 0 — sits in LF2
            _short(750.0, 790.0),  # rank 1 — sits in LF2
            _short(120.0, 160.0),  # rank 2 — sits in LF1
        ]
        out = assign_shorts_to_lfs_by_source_window(shorts, [lf1, lf2], boundaries)
        assert out == {1: [2], 2: [0, 1]}, out

    def test_orphan_short_attaches_to_nearest_lf(self):
        """A short whose midpoint lies in footage culled from every LF
        (percentile cut, inter-LF gap) attaches to the nearest LF
        rather than being dropped."""
        boundaries = compute_day_file_boundaries(_batch([1800.0]))
        lf1 = _lf(1, [_seg(0, 0.0, 300.0)])
        lf2 = _lf(2, [_seg(0, 1500.0, 1800.0)])
        # Short midpoint=750 — closer to LF1's [0,300] (gap 450)
        # than LF2's [1500,1800] (gap 750). Attaches to LF1.
        near_lf1 = _short(730.0, 770.0)
        # Short midpoint=1200 — closer to LF2 (gap 300) than LF1 (gap 900).
        near_lf2 = _short(1180.0, 1220.0)
        out = assign_shorts_to_lfs_by_source_window([near_lf1, near_lf2], [lf1, lf2], boundaries)
        assert out == {1: [0], 2: [1]}

    def test_count_preservation(self):
        """Total assigned shorts must equal total input — no drops."""
        boundaries = compute_day_file_boundaries(_batch([1000.0, 1000.0]))
        lf1 = _lf(1, [_seg(0, 100.0, 400.0)])
        lf2 = _lf(2, [_seg(1, 100.0, 400.0)])  # master 1100-1400
        shorts = [_short(i * 100.0, i * 100.0 + 30.0) for i in range(20)]
        out = assign_shorts_to_lfs_by_source_window(shorts, [lf1, lf2], boundaries)
        total = sum(len(v) for v in out.values())
        assert total == len(shorts), out

    def test_multi_segment_lf_with_gap(self):
        """An LF with two non-contiguous segments owns shorts inside
        either segment AND the bridge between them via nearest-fallback
        if the bridge is closer than any other LF."""
        boundaries = compute_day_file_boundaries(_batch([2000.0]))
        # LF1: [0, 300) ∪ [800, 1100)
        # LF2: [1500, 1800)
        lf1 = _lf(1, [_seg(0, 0.0, 300.0), _seg(0, 800.0, 1100.0)])
        lf2 = _lf(2, [_seg(0, 1500.0, 1800.0)])
        # Short at midpoint=550 — inside the LF1 gap, but LF1's nearest edge
        # is gap 250 (from 300 to 550 or from 550 to 800), LF2's is 950. → LF1.
        bridge = _short(530.0, 570.0)
        out = assign_shorts_to_lfs_by_source_window([bridge], [lf1, lf2], boundaries)
        assert out == {1: [0], 2: []}

    def test_empty_lfs_returns_empty(self):
        out = assign_shorts_to_lfs_by_source_window([_short(0, 30)], [], [(0, 1000)])
        assert out == {}

    def test_empty_shorts_yields_empty_buckets_per_lf(self):
        boundaries = compute_day_file_boundaries(_batch([500.0]))
        lf1 = _lf(1, [_seg(0, 0.0, 250.0)])
        lf2 = _lf(2, [_seg(0, 250.0, 500.0)])
        out = assign_shorts_to_lfs_by_source_window([], [lf1, lf2], boundaries)
        assert out == {1: [], 2: []}

    def test_preserves_emission_order_within_bucket(self):
        """Order within each LF's bucket follows the input order
        (which is rank order from short_planner)."""
        boundaries = compute_day_file_boundaries(_batch([1000.0]))
        lf1 = _lf(1, [_seg(0, 0.0, 1000.0)])
        shorts = [_short(800, 830), _short(100, 130), _short(500, 530)]
        out = assign_shorts_to_lfs_by_source_window(shorts, [lf1], boundaries)
        # All three fall inside LF1 — bucket order = input order = rank order.
        assert out == {1: [0, 1, 2]}


class TestDayArtifactsCarriesShortRanges:
    """run_day_pipeline must populate DayArtifacts.short_ranges parallel
    to short_paths so phase_bc_day_first has the data it needs to do
    the source-window mapping. The two lists must be the same length
    after degenerate-skip filtering."""

    def test_dayartifacts_has_short_ranges_field(self):
        from gomoveshift.pipeline.day_pipeline import DayArtifacts

        a = DayArtifacts(date="2026-03-04", master_path=None)
        assert hasattr(a, "short_ranges")
        assert a.short_ranges == []
