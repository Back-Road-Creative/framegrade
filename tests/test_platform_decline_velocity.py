"""A velocity nobody measured is ``unknown``, never ``falling``.

An unmeasured half is absence, not a measured zero: folding it into ``0.0`` reports a
collapse that was never observed. A genuinely measured zero still reads ``falling``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

from gomoveshift.tracking.decline import detect_platform_decline
from gomoveshift.tracking.schema import (
    DailyMetricPoint,
    PlatformMetrics,
    ShortRecord,
    TrackingRecord,
)

PLATFORM = "facebook"
HALVED = [1400, 1470, 1330, 1400, 1470, 1330, 1400] * 2 + [700, 770, 630, 700, 770, 630, 700] * 2
WINDOW = 14


def _short(index: int, per_day: List[int], total: Optional[int] = None) -> ShortRecord:
    daily = [DailyMetricPoint(day=f"2026-01-{d:02d}", views=v) for d, v in enumerate(per_day)]
    return ShortRecord(
        index=index,
        posted_at=f"2026-01-{index + 1:02d}T12:00:00+00:00",
        # The pair is inseparable by construction, and the reason bites
        # HERE: velocity is measured off the first 48 hours from ``posted_at``,
        # so a label promising anything looser than the exact instant would make
        # a synthetic series start at a time it did not. Same label, same
        # reasoning, as the sibling suite tests/test_platform_decline.py.
        posted_at_source="platform_confirmed",
        platform_ids={PLATFORM: f"vid{index}"},
        metrics={
            PLATFORM: PlatformMetrics(views=sum(per_day) if total is None else total, daily=daily)
        },
    )


def _series_records() -> List[TrackingRecord]:
    """The day-7 series the decline itself is computed from: a halved, significant drop."""
    return [
        TrackingRecord(
            project="proj", shorts=[_short(i, [v // 7] * 10) for i, v in enumerate(HALVED)]
        )
    ]


def _velocity_store(shorts: List[ShortRecord]):
    """Stands in for TrackingStore: only these shorts have a first-48h measurement."""
    recs = [TrackingRecord(project="proj", shorts=shorts)]
    return SimpleNamespace(load_all_records=lambda: recs)


def _measured(index: int) -> ShortRecord:  # first 2 of 10 days carry views -> velocity 0.2
    return _short(index, [10] * 10)


def _measured_zero(index: int) -> ShortRecord:  # measured, and the first 48h were genuinely 0
    return _short(index, [0, 0] + [10] * 8)


def test_an_unmeasured_recent_half_is_unknown_not_falling():
    records = _series_records()
    prior_only = [_measured(i) for i in range(WINDOW)]  # recent half: no velocity at all
    finding = detect_platform_decline(records, PLATFORM, store=_velocity_store(prior_only))
    assert finding is not None
    assert finding.velocity == "unknown"
    assert "velocity=unknown" in finding.log_line()


def test_a_measured_zero_recent_half_still_reads_falling():
    records = _series_records()
    both = [_measured(i) for i in range(WINDOW)] + [
        _measured_zero(i) for i in range(WINDOW, 2 * WINDOW)
    ]
    finding = detect_platform_decline(records, PLATFORM, store=_velocity_store(both))
    assert finding is not None
    assert finding.velocity == "falling"
