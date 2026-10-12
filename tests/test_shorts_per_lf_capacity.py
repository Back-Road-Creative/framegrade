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
