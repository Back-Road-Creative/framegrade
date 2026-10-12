"""Tests for superlong_planner. Contract mirrors
``plan_regular_lfs(mode="full_drive")``: ``stop_mask`` is
REQUIRED (``None`` raises ``ValueError``)."""

import numpy as np
import pytest

from gomoveshift.pipeline.superlong_planner import plan_superlong_parts

CAP = 14400  # 4h, mirrors config.DriveConfig.superlong_part_max_duration_sec
GRACE = 0.08
GRACE_CAP = CAP * (1 + GRACE)


def _mask(total_len: int, unmasked_spans: list[tuple[int, int]]) -> np.ndarray:
    """stop_mask (True == masked/STOP) keeping exactly the given spans."""
    mask = np.ones(total_len, dtype=bool)
    for start, end in unmasked_spans:
        mask[start:end] = False
    return mask


def test_none_or_all_masked_stop_mask_raises():
    with pytest.raises(ValueError, match="stop_mask"):
        plan_superlong_parts(None, part_max_duration_sec=CAP, split_grace=GRACE)
    with pytest.raises(ValueError):
        all_masked = np.ones(3600, dtype=bool)
        plan_superlong_parts(all_masked, part_max_duration_sec=CAP, split_grace=GRACE)


def test_short_day_is_one_part():
    """2h kept, well under the 4h cap -> exactly one part."""
    mask = _mask(7200, [(0, 7200)])
    parts = plan_superlong_parts(mask, part_max_duration_sec=CAP, split_grace=GRACE)
    assert len(parts) == 1
    assert (parts[0].part_index, parts[0].part_count) == (1, 1)
    assert parts[0].ranges == [(0.0, 7200.0)]
    assert parts[0].kept_duration_sec == 7200.0


def test_grace_window_is_still_one_part():
    """cap < kept <= cap*(1+grace) -> exactly one part, not two."""
    kept = int(CAP * 1.05)
    mask = _mask(kept, [(0, kept)])
    parts = plan_superlong_parts(mask, part_max_duration_sec=CAP, split_grace=GRACE)
    assert len(parts) == 1
    assert parts[0].kept_duration_sec == float(kept)


def test_6h_kept_splits_near_midpoint_boundary():
    """6h kept, 4h cap -> 2 parts, split at the ONE boundary near the mark."""
    span1, span2 = (0, 11000), (11500, 21600)  # 500s stop gap between them
    mask = _mask(21600, [span1, span2])
    parts = plan_superlong_parts(mask, part_max_duration_sec=CAP, split_grace=GRACE)
    assert len(parts) == 2
    assert [(p.part_index, p.part_count) for p in parts] == [(1, 2), (2, 2)]
    assert parts[0].ranges == [(0.0, 11000.0)]
    assert parts[1].ranges == [(11500.0, 21600.0)]
    assert (parts[0].kept_duration_sec, parts[1].kept_duration_sec) == (11000.0, 10100.0)
    for p in parts:
        assert p.kept_duration_sec <= GRACE_CAP
        for start, end in p.ranges:
            assert not (start < 11500.0 and end > 11000.0)


def test_10h_kept_splits_into_three_balanced_parts():
    """10h kept, 4h cap -> 3 parts, splits land exactly on the even marks."""
    spans = [(0, 12000), (12100, 24100), (24200, 36200)]
    mask = _mask(36200, spans)
    parts = plan_superlong_parts(mask, part_max_duration_sec=CAP, split_grace=GRACE)
    assert len(parts) == 3
    assert [p.ranges for p in parts] == [[s] for s in spans]
    assert sum(p.kept_duration_sec for p in parts) == 36000.0
    for p in parts:
        assert p.kept_duration_sec <= GRACE_CAP


def test_single_long_span_with_no_interior_boundary_bisects():
    """A single 5h span has NO interior boundary -> exact mid-span cut."""
    kept = 5 * 3600
    mask = _mask(kept, [(0, kept)])
    parts = plan_superlong_parts(mask, part_max_duration_sec=CAP, split_grace=GRACE)
    assert len(parts) == 2
    for p in parts:
        assert p.kept_duration_sec <= GRACE_CAP
    assert parts[0].ranges[-1][1] == parts[1].ranges[0][0]  # contiguous bisection
    assert sum(p.kept_duration_sec for p in parts) == float(kept)


def test_boundary_preferred_over_mid_span_when_within_reach():
    """A boundary near (not at) the mark wins over a mid-span cut when in reach."""
    cap, grace = 1000, 0.15  # grace_cap = 1150
    span1, span2 = (0, 900), (905, 1905)  # boundary at kept=900, mark=950
    mask = _mask(1905, [span1, span2])
    parts = plan_superlong_parts(mask, part_max_duration_sec=cap, split_grace=grace)
    assert len(parts) == 2
    assert parts[0].ranges == [(0.0, 900.0)]
    assert parts[1].ranges == [(905.0, 1905.0)]
    assert (parts[0].kept_duration_sec, parts[1].kept_duration_sec) == (900.0, 1000.0)
