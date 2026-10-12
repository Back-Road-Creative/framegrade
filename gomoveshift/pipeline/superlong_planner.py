"""Superlong part planner — the whole drive day's moving footage in 1-3 parts.

Distinct from `regular_lf_planner` (RegularLF pack/merge/classify must never
see `SuperlongPartPlan`). `stop_mask` is REQUIRED, mirroring
`plan_regular_lfs(mode="full_drive")`. Reuses
`_full_drive_unmasked_ranges` instead of re-deriving span boundaries.

Split algorithm: `K = ceil(kept / part_max_duration_sec)` parts; each split
lands at the stop-mask boundary nearest its even `k * kept / K` mark, chosen
only if it keeps the part it closes AND everything left for later parts
within `part_max_duration_sec * (1 + split_grace)` (checked fresh per
split — cap-safe by construction). No boundary in reach -> exact mid-span
cut at the clamped mark.
"""

import itertools
import math
from typing import Optional

import numpy as np

from gomoveshift.models import SuperlongPartPlan
from gomoveshift.pipeline.regular_lf_planner import _full_drive_unmasked_ranges


def _unmasked_ranges(stop_mask: np.ndarray) -> list[tuple[float, float]]:
    """Maximal contiguous unmasked spans, via `_full_drive_unmasked_ranges`
    (synthetic 1 Hz index; this planner has no score array)."""
    mask = np.asarray(stop_mask, dtype=bool)
    n = mask.shape[0]
    timestamps = np.arange(n, dtype=np.float64)
    composites = np.zeros(n, dtype=np.float64)
    spans = _full_drive_unmasked_ranges(timestamps, composites, mask)
    return [(r.start_sec, r.end_sec) for r in spans]


def _choose_split_positions(
    ranges: list[tuple[float, float]], kept: float, part_count: int, grace_cap: float
) -> list[float]:
    """Choose `part_count - 1` kept-time split positions in [0, kept]."""
    durations = [e - s for s, e in ranges]
    cum_end = list(itertools.accumulate(durations))
    remaining_boundaries = cum_end[:-1]  # positions between consecutive spans

    positions: list[float] = []
    prev = 0.0
    for k in range(1, part_count):
        target = k * kept / part_count
        remaining_after = part_count - k
        # Fresh window each split -> cap-safe by construction, not drift-checked after.
        low = max(prev, kept - remaining_after * grace_cap)
        high = min(kept, prev + grace_cap)
        candidates = [b for b in remaining_boundaries if low <= b <= high]
        split_pos = min(candidates, key=lambda b: abs(b - target)) if candidates else target
        split_pos = min(max(split_pos, low), high)  # fallback-mark clamp
        positions.append(split_pos)
        prev = split_pos
        remaining_boundaries = [b for b in remaining_boundaries if b > prev]
    return positions


def _slice_ranges_at_positions(
    ranges: list[tuple[float, float]], split_positions: list[float], kept: float
) -> list[list[tuple[float, float]]]:
    """Partition `ranges` at kept-time split positions; a split inside a
    span bisects it (kept-time maps linearly to calendar time per span)."""
    durations = [e - s for s, e in ranges]
    cum_end = list(itertools.accumulate(durations))
    cum_start = [ce - d for ce, d in zip(cum_end, durations)]

    bounds = [0.0, *split_positions, kept]
    parts: list[list[tuple[float, float]]] = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        part_ranges = []
        for (s, _e), cs, ce in zip(ranges, cum_start, cum_end):
            overlap_lo, overlap_hi = max(cs, lo), min(ce, hi)
            if overlap_hi > overlap_lo:
                part_ranges.append((s + (overlap_lo - cs), s + (overlap_hi - cs)))
        parts.append(part_ranges)
    return parts


def plan_superlong_parts(
    stop_mask: Optional[np.ndarray],
    *,
    part_max_duration_sec: int,
    split_grace: float,
) -> list[SuperlongPartPlan]:
    """Plan the whole drive day's moving footage as 1-3 superlong parts.

    ``stop_mask`` is REQUIRED (``None`` raises ``ValueError``). Mirrors
    ``DriveConfig.superlong_part_max_duration_sec``/``_split_grace``; every
    returned part's ``kept_duration_sec`` <= ``part_max_duration_sec * (1 +
    split_grace)``.
    """
    if stop_mask is None:
        raise ValueError(
            "plan_superlong_parts requires a stop_mask — maskless planning "
            "would silently include parked stops and measure the cap against "
            "wall-time instead of moving-time. Pass stop_mask=<per-day mask>."
        )

    ranges = _unmasked_ranges(stop_mask)
    kept = sum(e - s for s, e in ranges)
    if kept <= 0:
        raise ValueError("plan_superlong_parts: stop_mask leaves nothing unmasked to plan.")

    cap = float(part_max_duration_sec)
    grace_cap = cap * (1.0 + split_grace)

    if kept <= grace_cap:
        groups, part_count = [ranges], 1
    else:
        part_count = math.ceil(kept / cap)
        split_positions = _choose_split_positions(ranges, kept, part_count, grace_cap)
        groups = _slice_ranges_at_positions(ranges, split_positions, kept)

    result = [
        SuperlongPartPlan(
            part_index=idx,
            part_count=part_count,
            ranges=group,
            kept_duration_sec=sum(e - s for s, e in group),
        )
        for idx, group in enumerate(groups, start=1)
    ]

    for part in result:
        if part.kept_duration_sec > grace_cap + 1e-6:
            raise RuntimeError(
                f"superlong part {part.part_index}/{part.part_count} kept "
                f"{part.kept_duration_sec:.1f}s exceeds cap*(1+grace)={grace_cap:.1f}s"
            )

    return result
