"""Balanced full_drive day split (``split_ranges_balanced`` + assembler wiring)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.config import DriveConfig
from gomoveshift.pipeline.longform_assembler import LongFormAssembler
from gomoveshift.pipeline.regular_lf_planner import (
    RegularLFRange,
    plan_full_drive_parts,
    split_ranges_balanced,
)

MIN, MAX = 3600, 5400


def _rngs(*spans):
    # integrated_score is a SUM of per-second composites (see
    # _full_drive_unmasked_ranges), so give each range 1.0/sec.
    return [RegularLFRange(float(a), float(b), float(b - a)) for a, b in spans]


def _dur(part):
    return sum(r.duration_sec for r in part)


def _assert_union(parts, ranges):
    flat = [r for p in parts for r in p]
    # coverage identical: merge touching pieces and compare to the input
    merged = []
    for r in flat:
        if merged and abs(merged[-1][1] - r.start_sec) < 1e-9:
            merged[-1][1] = r.end_sec
        else:
            merged.append([r.start_sec, r.end_sec])
    assert [tuple(m) for m in merged] == [(r.start_sec, r.end_sec) for r in ranges]
    assert sum(r.integrated_score for r in flat) == pytest.approx(
        sum(r.integrated_score for r in ranges)
    )


# Synthetic day: 7968 s kept (132.8 min), with a 37 s and an 11-min gap.
def _day_132():
    # Ranges 0-2000, 2037-4500 (2463s), 5160-8665 (3505s); gaps of 37 s and 660 s.
    return _rngs((0, 2000), (2037, 4500), (5160, 8665))


def test_two_parts_each_in_window():
    rs = _day_132()
    assert _dur(rs) == 7968
    parts = split_ranges_balanced(rs, min_sec=MIN, max_sec=MAX)
    assert len(parts) == 2
    for p in parts:
        assert 3600 <= _dur(p) <= 4368
    _assert_union(parts, rs)


def test_snaps_to_in_window_range_boundary():
    # cumulative edges at 2000, 4463; target 3984 -> nearest in [3600,4368]
    # is none of those; use a layout with an edge at 4000.
    rs = _rngs((0, 4000), (4100, 8068))  # 4000 + 3968 = 7968
    parts = split_ranges_balanced(rs, min_sec=MIN, max_sec=MAX)
    assert len(parts) == 2
    assert parts[0] == [rs[0]]
    assert parts[1] == [rs[1]]


def test_mid_span_cut_when_no_boundary_in_window():
    rs = _rngs((0, 7968))
    parts = split_ranges_balanced(rs, min_sec=MIN, max_sec=MAX)
    assert len(parts) == 2
    assert _dur(parts[0]) == pytest.approx(3984)
    assert parts[0][0].end_sec == parts[1][0].start_sec
    _assert_union(parts, rs)


def test_bisected_range_splits_sum_score_proportionally():
    rs = [RegularLFRange(0.0, 7968.0, 1000.0)]
    a, b = split_ranges_balanced(rs, min_sec=MIN, max_sec=MAX)
    assert a[0].integrated_score == pytest.approx(1000.0 * 3984 / 7968)
    assert b[0].integrated_score == pytest.approx(1000.0 * 3984 / 7968)


def test_100_min_is_one_part():
    rs = _rngs((0, 6000))
    parts = split_ranges_balanced(rs, min_sec=MIN, max_sec=MAX)
    assert parts == [rs]


def test_30_min_unchanged():
    rs = _rngs((0, 900), (1000, 1900))
    assert split_ranges_balanced(rs, min_sec=MIN, max_sec=MAX) == [rs]


def test_three_parts_bounds_and_union():
    rs = _rngs((0, 3000), (3050, 9000), (9500, 14000))  # 3000+5950+4500
    parts = split_ranges_balanced(rs, min_sec=MIN, max_sec=MAX)
    assert len(parts) == math.ceil(13450 / MAX)
    for p in parts:
        assert MIN <= _dur(p) <= max(MAX, math.ceil(13450 / len(parts)))
    _assert_union(parts, rs)


def test_empty_ranges():
    assert split_ranges_balanced([], min_sec=MIN, max_sec=MAX) == [[]]


def test_invalid_bounds_raise():
    with pytest.raises(ValueError):
        split_ranges_balanced(_rngs((0, 100)), min_sec=5400, max_sec=3600)


def _scores(n):
    arr = np.zeros((n, 11))
    arr[:, 0] = np.arange(n)
    arr[:, 1] = 0.5
    arr[:, 2] = 0.6
    return arr


def test_plan_full_drive_parts_none_config_is_single_part():
    parts = plan_full_drive_parts(
        _scores(1800),
        min_duration_sec=480,
        full_drive_max_duration_sec=7200,
        stop_mask=np.zeros(1800, dtype=bool),
        part_min_sec=None,
        part_max_sec=None,
    )
    assert len(parts) == 1 and parts[0][0].duration_sec == 1800


def test_per_part_ceiling_raises_never_truncates():
    with pytest.raises(ValueError, match="ceiling"):
        plan_full_drive_parts(
            _scores(8000),
            min_duration_sec=480,
            full_drive_max_duration_sec=3000,
            stop_mask=np.zeros(8000, dtype=bool),
            part_min_sec=MIN,
            part_max_sec=MAX,
        )


def test_per_part_ceiling_allows_total_over_ceiling_when_parts_fit():
    parts = plan_full_drive_parts(
        _scores(8000),
        min_duration_sec=480,
        full_drive_max_duration_sec=7200,
        stop_mask=np.zeros(8000, dtype=bool),
        part_min_sec=MIN,
        part_max_sec=MAX,
    )
    assert len(parts) == 2


def _cfg(**kw):
    return DriveConfig(
        lf_selection_mode="full_drive",
        full_drive_max_duration_sec=7200,
        min_lf_duration=480.0,
        min_segment_duration=30.0,
        min_motion_energy=0.0,
        **kw,
    )


def _build(cfg, n=8000):
    return LongFormAssembler(cfg).build_drive_plan_per_day(
        scores=_scores(n),
        processed_files=[Path("/tmp/ep_00.mp4")],
        file_boundaries=[(0.0, float(n))],
        episode_dates=["2025-09-14"],
        stop_mask=np.zeros(n, dtype=bool),
    )


def test_assembler_emits_indexed_parts():
    plan = _build(_cfg(full_drive_part_min_sec=MIN, full_drive_part_max_sec=MAX))
    lfs = plan.regular_lfs
    assert [lf.index for lf in lfs] == [1, 2]
    assert [(lf.part_index, lf.part_count) for lf in lfs] == [(1, 2), (2, 2)]
    for lf in lfs:
        assert 3600 <= lf.total_duration <= 4400
    # segments partitioned: part 2 starts where part 1 ends
    assert lfs[0].segments[-1].original_end_time <= lfs[1].segments[0].original_start_time + 1


def test_assembler_none_config_unchanged():
    plan = _build(_cfg(), n=7000)
    assert len(plan.regular_lfs) == 1
    assert plan.regular_lfs[0].part_index is None
    assert plan.regular_lfs[0].part_count is None


def test_assembler_legacy_path_raises_on_multiple_parts():
    cfg = _cfg(full_drive_part_min_sec=MIN, full_drive_part_max_sec=MAX)
    with pytest.raises(ValueError, match="day-first"):
        LongFormAssembler(cfg).build_drive_plan(
            scores=_scores(8000),
            processed_files=[Path("/tmp/ep_00.mp4")],
            file_boundaries=[(0.0, 8000.0)],
            stop_mask=np.zeros(8000, dtype=bool),
        )


@pytest.mark.parametrize(
    "kw",
    [
        {"full_drive_part_min_sec": MIN},
        {"full_drive_part_max_sec": MAX},
        {"full_drive_part_min_sec": 0, "full_drive_part_max_sec": MAX},
        {"full_drive_part_min_sec": -5, "full_drive_part_max_sec": MAX},
        {"full_drive_part_min_sec": MAX + 1, "full_drive_part_max_sec": MAX},
    ],
)
def test_driveconfig_rejects_bad_part_bounds(kw):
    with pytest.raises(ValueError, match="full_drive_part"):
        DriveConfig(**kw)


def test_driveconfig_accepts_valid_or_absent_part_bounds():
    assert DriveConfig().full_drive_part_min_sec is None
    cfg = DriveConfig(full_drive_part_min_sec=MAX, full_drive_part_max_sec=MAX)
    assert (cfg.full_drive_part_min_sec, cfg.full_drive_part_max_sec) == (MAX, MAX)


def test_assembler_empty_part_raises(monkeypatch):
    calls = {"n": 0}
    orig = LongFormAssembler._build_full_drive_lf_segments

    def flaky(self, *a, **kw):
        calls["n"] += 1
        return [] if calls["n"] == 2 else orig(self, *a, **kw)

    monkeypatch.setattr(LongFormAssembler, "_build_full_drive_lf_segments", flaky)
    with pytest.raises(ValueError, match="empty"):
        _build(_cfg(full_drive_part_min_sec=MIN, full_drive_part_max_sec=MAX))


def test_three_parts_single_range():
    # 12000 s -> K=ceil(12000/5400)=3, 4000 s each, cuts mid-range.
    rs = _rngs((0, 12000))
    parts = split_ranges_balanced(rs, min_sec=MIN, max_sec=MAX)
    assert len(parts) == 3
    assert [_dur(p) for p in parts] == [4000.0, 4000.0, 4000.0]
    for p in parts:
        assert len(p) == 1
    _assert_union(parts, rs)


def test_fractional_timestamps_on_cut():
    rs = [RegularLFRange(100.25, 7400.75, 7300.5)]
    parts = split_ranges_balanced(rs, min_sec=MIN, max_sec=MAX)
    assert len(parts) == 2
    a, b = parts[0][-1], parts[1][0]
    assert a.end_sec == b.start_sec  # exact tiling, no float gap
    assert a.start_sec == 100.25 and b.end_sec == 7400.75
    assert a.end_sec == pytest.approx(100.25 + 3650.25)
    _assert_union(parts, rs)


def test_shipped_config_splits_full_drive_days_into_60_to_90_min_parts():
    """Long-forms run 60-90 minutes. The shipped sample config turns the split on."""
    from pathlib import Path

    from gomoveshift.config import Config

    cfg = Config.load(Path(__file__).resolve().parents[1] / "config.yaml")
    assert cfg.drive.full_drive_part_min_sec == 3600
    assert cfg.drive.full_drive_part_max_sec == 5400
