"""A partial bucket (smaller than a window's full per-day capacity) must
be spread across the whole window, not front-loaded into the earliest days.

Regression for a cadence gap: a shorts bucket smaller than
the window's full per-day capacity previously placed the identity pair
2/day until the bucket ran out, leaving the tail of the window (e.g. the
Thursday before the next long-form) with no shorts at all. See
docs/pipeline-internals.md for the distribution rule.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from gomoveshift.pipeline.scheduler import PipelineScheduler, _spread_bucket_across_window

_IDENTITY = [dt.time(10, 0), dt.time(16, 0)]
# An early long-form time keeps both identity times eligible on every
# interior day of the window (no LF-day boundary exclusion muddying the
# eligible-count math) while the far boundary (next LF's own drop) still
# excludes the window's last calendar day, as it does in production.
_EARLY_LF_TIME = dt.time(6, 0)


def _lf(idx: int) -> SimpleNamespace:
    return SimpleNamespace(index=idx)


def _scheduler() -> PipelineScheduler:
    return PipelineScheduler(long_form_time=_EARLY_LF_TIME, shorts_times=list(_IDENTITY))


def _window_by_date(schedule, buckets: list, lf_i: int) -> dict:
    """Slots belonging to LF ``lf_i``'s own bucket, identified by
    ``short_index`` range (the per-LF post-numbering contract) rather
    than a date range — a window's shorts can legitimately land on the
    date shared with the next LF's own window (the boundary day)."""
    offset = sum(buckets[:lf_i])
    idx_range = range(offset + 1, offset + buckets[lf_i] + 1)
    by_date: dict = {}
    for s in schedule.shorts_slots:
        if s.short_index in idx_range:
            d = dt.datetime.strptime(s.date, "%Y-%m-%d").date()
            by_date.setdefault(d, []).append(s)
    return by_date


def _generate(buckets):
    scheduler = _scheduler()
    return scheduler.generate_drive_schedule(
        regular_lfs=[_lf(0), _lf(1)],
        region="usa",
        start_date="2026-08-21",  # Friday
        publish_day="fri",
        shorts_per_day=2,
        shorts_counts=buckets,
    )


def test_partial_bucket_leaves_no_interior_day_empty():
    """8 shorts in a 7-day window (2/day capacity, all 7 days eligible)
    must reach every day the window can reach rather than filling the
    earliest days to capacity and leaving the tail at zero. This is RED
    on the pre-fix code: it places 2,2,2,2,0,0,0 (days 25-27 empty)."""
    schedule = _generate([8, 14])
    by_date = _window_by_date(schedule, [8, 14], 0)

    assert sum(len(v) for v in by_date.values()) == 8
    empty_days = [d for d in by_date if len(by_date[d]) == 0]
    assert not empty_days

    window_dates = [
        dt.date(2026, 8, 21) + dt.timedelta(days=n) for n in range(7)
    ]  # Fri..Thu — all eligible under this identity/LF-time combo
    covered = [d for d in window_dates if len(by_date.get(d, [])) > 0]
    assert len(covered) == 7, (
        f"only {len(covered)}/7 window days received a short — the bucket "
        f"was front-loaded instead of spread. Placement: "
        f"{ {str(d): len(v) for d, v in sorted(by_date.items())} }"
    )


def test_full_bucket_still_fills_window():
    """An exactly-full bucket (matching total eligible capacity, 14 = 2/day
    x 7 eligible days) still places every eligible slot at 2/day — no
    regression for the already-correct case."""
    schedule = _generate([14, 14])
    by_date = _window_by_date(schedule, [14, 14], 0)
    assert sum(len(v) for v in by_date.values()) == 14
    window_dates = [dt.date(2026, 8, 21) + dt.timedelta(days=n) for n in range(7)]
    assert all(len(by_date.get(d, [])) == 2 for d in window_dates)


def test_legacy_none_path_unchanged():
    """``shorts_counts=None`` (legacy path) is untouched by the
    distribution logic — every window date still gets its full identity
    allocation, exactly as before."""
    scheduler = _scheduler()
    schedule = scheduler.generate_drive_schedule(
        regular_lfs=[_lf(0)],
        region="usa",
        start_date="2026-08-21",
        publish_day="fri",
        shorts_per_day=2,
        shorts_counts=None,
    )
    assert len(schedule.shorts_slots) == 14


def test_spread_bucket_across_window_pure_helper():
    """Pure helper: never exceeds a day's eligible cap, never leaves a day
    at zero while capacity remains elsewhere, and totals exactly
    min(bucket, sum(eligible))."""
    eligible = [1, 2, 2, 2, 2, 2, 1]
    allowance = _spread_bucket_across_window(eligible, 12)
    assert sum(allowance) == 12
    assert all(a <= e for a, e in zip(allowance, eligible))
    assert all(a >= 1 for a in allowance), allowance

    # Bucket smaller than day count: some days legitimately still get 0.
    allowance2 = _spread_bucket_across_window(eligible, 3)
    assert sum(allowance2) == 3
    assert all(a <= e for a, e in zip(allowance2, eligible))

    # Bucket larger than total eligible: capped, not over-allocated.
    allowance3 = _spread_bucket_across_window(eligible, 999)
    assert sum(allowance3) == sum(eligible)
    assert allowance3 == eligible
