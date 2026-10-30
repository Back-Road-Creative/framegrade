"""Per-LF shorts capacity cap.

The shorts planner is DAY-WIDE: it picks the N best-scoring windows across
the whole day and attribution to LFs happens afterwards, by source window.
The scheduler is PER-LF: an LF's shorts window is ``lf_date → next_lf_date``
× the region's publish times, a physical ceiling of
``required_shorts_per_lf``.

Without a cap, a multi-LF day can produce 56 shorts distributed 11 / 21 / 10 / 14
over four LFs. The drive-wide count gate sees 56 == 4 × 14 and passes, but LF 2's
last 7 shorts have no slot to land in, so ``csv_generator`` refuses to write
the Metricool CSV at all (``METRICOOL-CSV-NO-SCHEDULE``) — after every one
of the 56 has been rendered.

The fix is upstream, at attribution: never render a short that cannot be
placed. This module locks that cap.
"""

from types import SimpleNamespace

from gomoveshift.pipeline.day_pipeline import (
    cap_shorts_to_per_lf_capacity,
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


def _short(rank: int, start: float, end: float):
    return SimpleNamespace(rank=rank, start_sec=start, end_sec=end)


def _batch(durations: list[float]):
    episodes = [SimpleNamespace(total_duration=d) for d in durations]
    return SimpleNamespace(episodes=episodes)


class TestPerLfCapacityCap:
    def test_surplus_beyond_capacity_is_dropped_best_ranked_kept(self):
        """One LF holding five shorts with room for three keeps ranks 1-3."""
        boundaries = compute_day_file_boundaries(_batch([1000.0]))
        lf1 = _lf(1, [_seg(0, 0.0, 1000.0)])
        shorts = [_short(i + 1, i * 100.0, i * 100.0 + 30.0) for i in range(5)]
        kept, dropped = cap_shorts_to_per_lf_capacity(shorts, [lf1], boundaries, 3)
        assert [s.rank for s in kept] == [1, 2, 3]
        assert [s.rank for s in dropped] == [4, 5]

    def test_only_the_over_full_lf_is_trimmed(self):
        """An under-full LF keeps everything it has.

        LF 1 owns two shorts, LF 2 owns five, capacity is three. LF 1 is
        left alone (trimming to a uniform count would throw away placeable
        content); LF 2 sheds its two worst-ranked.
        """
        boundaries = compute_day_file_boundaries(_batch([1200.0]))
        lf1 = _lf(1, [_seg(0, 0.0, 600.0)])
        lf2 = _lf(2, [_seg(0, 600.0, 1200.0)])
        shorts = [
            _short(1, 700.0, 730.0),  # LF 2
            _short(2, 100.0, 130.0),  # LF 1
            _short(3, 800.0, 830.0),  # LF 2
            _short(4, 900.0, 930.0),  # LF 2
            _short(5, 200.0, 230.0),  # LF 1
            _short(6, 1000.0, 1030.0),  # LF 2
            _short(7, 1100.0, 1130.0),  # LF 2
        ]
        kept, dropped = cap_shorts_to_per_lf_capacity(shorts, [lf1, lf2], boundaries, 3)
        assert [s.rank for s in kept] == [1, 2, 3, 4, 5]
        assert [s.rank for s in dropped] == [6, 7]

    def test_every_kept_lf_bucket_is_within_capacity(self):
        """The invariant the Metricool CSV gate depends on."""
        from gomoveshift.pipeline.day_pipeline import assign_shorts_to_lfs_by_source_window

        boundaries = compute_day_file_boundaries(_batch([1200.0]))
        lf1 = _lf(1, [_seg(0, 0.0, 600.0)])
        lf2 = _lf(2, [_seg(0, 600.0, 1200.0)])
        shorts = [_short(i + 1, 600.0 + i * 20.0, 600.0 + i * 20.0 + 15.0) for i in range(9)]
        kept, _ = cap_shorts_to_per_lf_capacity(shorts, [lf1, lf2], boundaries, 4)
        buckets = assign_shorts_to_lfs_by_source_window(kept, [lf1, lf2], boundaries)
        assert all(len(v) <= 4 for v in buckets.values()), buckets

    def test_kept_preserves_input_order(self):
        """Downstream pairs ranges to paths positionally — order must hold."""
        boundaries = compute_day_file_boundaries(_batch([1000.0]))
        lf1 = _lf(1, [_seg(0, 0.0, 1000.0)])
        shorts = [_short(1, 800.0, 830.0), _short(2, 100.0, 130.0), _short(3, 500.0, 530.0)]
        kept, _ = cap_shorts_to_per_lf_capacity(shorts, [lf1], boundaries, 2)
        assert [s.start_sec for s in kept] == [800.0, 100.0]

    def test_no_capacity_is_a_noop(self):
        """Negative control: no derived per-LF target (legacy/test config)."""
        boundaries = compute_day_file_boundaries(_batch([1000.0]))
        lf1 = _lf(1, [_seg(0, 0.0, 1000.0)])
        shorts = [_short(i + 1, i * 100.0, i * 100.0 + 30.0) for i in range(5)]
        kept, dropped = cap_shorts_to_per_lf_capacity(shorts, [lf1], boundaries, None)
        assert kept == shorts
        assert dropped == []

    def test_under_capacity_is_a_noop(self):
        """Negative control: nothing is dropped when every LF fits."""
        boundaries = compute_day_file_boundaries(_batch([1200.0]))
        lf1 = _lf(1, [_seg(0, 0.0, 600.0)])
        lf2 = _lf(2, [_seg(0, 600.0, 1200.0)])
        shorts = [_short(1, 100.0, 130.0), _short(2, 700.0, 730.0)]
        kept, dropped = cap_shorts_to_per_lf_capacity(shorts, [lf1, lf2], boundaries, 14)
        assert kept == shorts
        assert dropped == []

    def test_no_lf_plans_is_a_noop(self):
        shorts = [_short(1, 0.0, 30.0)]
        kept, dropped = cap_shorts_to_per_lf_capacity(shorts, [], [(0.0, 1000.0)], 1)
        assert kept == shorts
        assert dropped == []

    def test_unmappable_attribution_keeps_everything(self, monkeypatch):
        """Fail-safe: a mapping failure must not silently delete content.

        ``_render_day_shorts`` re-raises an ambiguous mapping on a multi-LF
        day, so no-op here leaves that guard as the single place that decides.
        """
        from gomoveshift.pipeline import day_pipeline

        def _boom(**_kwargs):
            raise IndexError("segment translation failed")

        monkeypatch.setattr(day_pipeline, "assign_shorts_to_lfs_by_source_window", _boom)
        boundaries = compute_day_file_boundaries(_batch([1000.0]))
        lf1 = _lf(1, [_seg(0, 0.0, 1000.0)])
        shorts = [_short(i + 1, i * 100.0, i * 100.0 + 30.0) for i in range(5)]
        kept, dropped = cap_shorts_to_per_lf_capacity(shorts, [lf1], boundaries, 2)
        assert kept == shorts
        assert dropped == []


class TestPerLfFill:
    """An under-capacity LF is refilled from its OWN source window."""

    CAP = 3
    GAP = 60.0

    def _day(self):
        """1200 s master: LF 1 = [0, 600), LF 2 = [600, 1200). LF 2 scores far higher."""
        import numpy as np

        ts = np.arange(1200, dtype=float)
        scores = np.zeros((1200, 9))
        scores[:, 0] = ts
        scores[:, 1] = np.where(ts >= 600, 0.9, 0.3)
        # a little texture so windows differ
        scores[:, 1] += 0.01 * np.sin(ts / 7.0)
        boundaries = compute_day_file_boundaries(_batch([1200.0]))
        lf1 = _lf(1, [_seg(0, 0.0, 600.0)])
        lf2 = _lf(2, [_seg(0, 600.0, 1200.0)])
        return scores, boundaries, [lf1, lf2]

    def _planned_and_capped(self, scores, boundaries, lfs, target):
        from gomoveshift.pipeline.short_planner import plan_shorts

        planned = plan_shorts(scores, target_count=target, min_gap_sec=self.GAP)
        kept, _ = cap_shorts_to_per_lf_capacity(planned, lfs, boundaries, self.CAP)
        return kept

    def _fill(self, kept, scores, boundaries, lfs, mask=None):
        from gomoveshift.pipeline.day_pipeline import fill_shorts_to_per_lf_capacity

        return fill_shorts_to_per_lf_capacity(
            kept, lfs, boundaries, self.CAP, scores, mask, {"min_gap_sec": self.GAP}
        )

    def test_top_windows_in_one_lf_still_fill_every_lf(self):
        import numpy as np

        from gomoveshift.pipeline.day_pipeline import assign_shorts_to_lfs_by_source_window

        scores, boundaries, lfs = self._day()
        mask = np.zeros(1200, dtype=bool)
        mask[100:140] = True  # a parked stretch inside LF 1
        kept = self._planned_and_capped(scores, boundaries, lfs, self.CAP + 2)
        assert {int(s.start_sec // 600) for s in kept} == {1}  # all LF 2 before refill
        result, added = self._fill(kept, scores, boundaries, lfs, mask)
        buckets = assign_shorts_to_lfs_by_source_window(result, lfs, boundaries)
        assert {k: len(v) for k, v in buckets.items()} == {1: self.CAP, 2: self.CAP}
        assert len(added) == self.CAP
        for s in added:
            assert 0.0 <= s.start_sec and s.end_sec <= 600.0  # LF 1's own window
            assert not mask[int(s.start_sec) : int(s.end_sec)].any()
        ordered = sorted(result, key=lambda s: s.start_sec)
        for a, b in zip(ordered, ordered[1:]):
            assert a.end_sec <= b.start_sec  # no overlap
        assert [s.rank for s in result[: len(kept)]] == [s.rank for s in kept]
        assert [s.rank for s in added] == sorted(s.rank for s in added)
        assert min(s.rank for s in added) > max(s.rank for s in kept)

    def test_fully_masked_lf_gets_no_refill_and_reaches_the_contract_gate(self):
        import numpy as np
        import pytest

        from gomoveshift.pipeline.day_pipeline import _enforce_shorts_contract

        scores, boundaries, lfs = self._day()
        mask = np.zeros(1200, dtype=bool)
        mask[:600] = True  # LF 1's whole window is vetoed
        kept = self._planned_and_capped(scores, boundaries, lfs, self.CAP + 2)
        result, added = self._fill(kept, scores, boundaries, lfs, mask)
        assert added == []
        assert len(result) == self.CAP
        with pytest.raises(RuntimeError, match="SHORTS-CONTRACT"):
            _enforce_shorts_contract(len(result), 2 * self.CAP, "2026-10-28", False)
        _enforce_shorts_contract(len(result), 2 * self.CAP, "2026-10-28", True)

    def test_balanced_day_is_unchanged(self):
        scores, boundaries, lfs = self._day()
        kept = [_short(1, 100.0, 138.0), _short(2, 700.0, 738.0), _short(3, 300.0, 338.0)] + [
            _short(4, 800.0, 838.0),
            _short(5, 900.0, 938.0),
            _short(6, 400.0, 438.0),
        ]
        result, added = self._fill(kept, scores, boundaries, lfs)
        assert result == kept
        assert added == []

    def test_no_capacity_is_a_noop(self):
        from gomoveshift.pipeline.day_pipeline import fill_shorts_to_per_lf_capacity

        scores, boundaries, lfs = self._day()
        kept = [_short(1, 700.0, 738.0)]
        result, added = fill_shorts_to_per_lf_capacity(
            kept, lfs, boundaries, None, scores, None, {"min_gap_sec": self.GAP}
        )
        assert result == kept and added == []
