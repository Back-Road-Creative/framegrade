"""Shorts scheduler slots must be numbered by the ACTUAL per-LF bucket
sizes, not a fixed per-LF-window capacity.

Posts are numbered globally by real (unequal) bucket sizes
(``global_index_offset += len(posts)`` in ``_build_social_posts_per_lf``),
but the slot generator historically filled a fixed capacity per window
(``windows/day × LF_PUBLISH_INTERVAL_DAYS``) and numbered slots
continuously across windows. When an LF's bucket differs from that fixed
capacity, the two numberings drift: a post whose global index belongs to
LF *i* gets matched (by ``short_index``) to a slot that physically lands
in a DIFFERENT LF's window — scheduling a short against the wrong parent
LF's publish window (e.g. before its own LF has even published).

These tests pin the invariant: for a multi-LF day with unequal buckets,
every post index belonging to LF *i* must map to a slot inside LF *i*'s
own window (on/after that LF's publish, before the next LF's).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from _state_helpers import make_state

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_phases.phase_d import _build_and_persist_drive_schedule
from gomoveshift.pipeline.scheduler import PipelineScheduler


def _lf(idx: int) -> SimpleNamespace:
    return SimpleNamespace(index=idx, total_duration=1800.0)


def _post_index_ranges(buckets: list[int]) -> list[range]:
    """Global 1-based post-index range per LF, mirroring the offset advance
    in ``_build_social_posts_per_lf`` (offset += bucket size per LF)."""
    ranges: list[range] = []
    offset = 0
    for size in buckets:
        ranges.append(range(offset + 1, offset + size + 1))
        offset += size
    return ranges


def test_unequal_buckets_keep_each_lfs_shorts_in_its_own_window(tmp_path):
    """A multi-LF day with unequal buckets: LF1's shorts must not land in
    LF0's window (the spill). LF0 bucket (10) < fixed window capacity
    (2/day × 7 = 14), so LF1's early posts previously took slots that
    physically sit before LF1 publishes."""
    buckets = [10, 8]
    state = make_state(
        shorts_per_day=2,
        shorts_per_lf={0: ["a"] * buckets[0], 1: ["b"] * buckets[1]},
        start_date="2026-03-25",  # Wednesday, US EDT (no DST edge in window)
        publish_day="wed",
        region="usa",
        no_ledger_reconcile=True,
    )
    schedule = _build_and_persist_drive_schedule(
        scheduler_cls=PipelineScheduler,
        config=Config(),
        drive_plan=SimpleNamespace(regular_lfs=[_lf(0), _lf(1)]),
        state=state,
        region="usa",
        stages_dir=tmp_path,
    )

    lf_dates = [datetime.strptime(s.date, "%Y-%m-%d").date() for s in schedule.lf_slots]
    slot_date_by_index = {
        s.short_index: datetime.strptime(s.date, "%Y-%m-%d").date() for s in schedule.shorts_slots
    }
    ranges = _post_index_ranges(buckets)

    for lf_i, post_range in enumerate(ranges):
        lower = lf_dates[lf_i]
        upper = lf_dates[lf_i + 1] if lf_i + 1 < len(lf_dates) else None
        for post_index in post_range:
            slot_date = slot_date_by_index.get(post_index)
            assert slot_date is not None, f"post #{post_index} (LF {lf_i}) has no schedule slot"
            assert slot_date >= lower, (
                f"post #{post_index} (LF {lf_i}) scheduled {slot_date} "
                f"before its own LF publishes {lower}"
            )
            if upper is not None:
                assert slot_date < upper, (
                    f"post #{post_index} (LF {lf_i}) scheduled {slot_date} "
                    f"in the NEXT LF's window (>= {upper}) — spill"
                )


def test_scheduler_shorts_counts_numbers_slots_per_bucket(tmp_path):
    """Direct scheduler check: ``shorts_counts`` caps and renumbers each
    window so slot ``short_index`` values are contiguous, non-overlapping,
    and aligned with the per-LF post offsets."""
    scheduler = PipelineScheduler()
    lfs = [_lf(0), _lf(1), _lf(2)]
    buckets = [3, 5, 4]

    schedule = scheduler.generate_drive_schedule(
        regular_lfs=lfs,
        region="usa",
        start_date="2026-03-25",
        publish_day="wed",
        shorts_per_day=2,
        shorts_counts=buckets,
    )

    lf_dates = [datetime.strptime(s.date, "%Y-%m-%d").date() for s in schedule.lf_slots]
    slot_date_by_index = {
        s.short_index: datetime.strptime(s.date, "%Y-%m-%d").date() for s in schedule.shorts_slots
    }

    # Contiguous, non-overlapping numbering: 1..sum(buckets), each once.
    assert sorted(slot_date_by_index) == list(range(1, sum(buckets) + 1))

    for lf_i, post_range in enumerate(_post_index_ranges(buckets)):
        lower = lf_dates[lf_i]
        upper = lf_dates[lf_i + 1] if lf_i + 1 < len(lf_dates) else None
        for post_index in post_range:
            slot_date = slot_date_by_index[post_index]
            assert slot_date >= lower
            if upper is not None:
                assert slot_date < upper
