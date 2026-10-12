"""Shadow post-anomaly policy (dip/spike over already-collected day-7 views).

Every candidate is gated by the kernel's own refusals before a policy threshold is even asked:
infeasible policy, too few references, and a flat (no-spread) reference all print a typed refusal
line instead of a fabricated number; a real dip/spike prints once, and a small effect that clears
significance but not the practical-effect gate prints nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gomoveshift.tracking import anomaly
from gomoveshift.tracking.schema import (
    DailyMetricPoint,
    PlatformMetrics,
    ShortRecord,
    TrackingRecord,
)

PLATFORM = "facebook"
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _short(index: int, day7: int, posted: datetime) -> ShortRecord:
    n = day7 // 7
    daily = [DailyMetricPoint(day=f"2026-01-{d:02d}", views=n) for d in range(1, 11)]
    return ShortRecord(
        index=index,
        posted_at=posted.isoformat(),
        # A posting time must say which moment it is: ShortRecord refuses a
        # posted_at with no provenance on construction, so the fixture states
        # one rather than leaving the comparison undefined.
        posted_at_source="platform_confirmed",
        platform_ids={PLATFORM: f"vid{index}"},
        metrics={PLATFORM: PlatformMetrics(views=n * 10, daily=daily)},
    )


def _records(values_with_offsets):
    shorts = [
        _short(i, v, BASE + timedelta(days=offset))
        for i, (v, offset) in enumerate(values_with_offsets)
    ]
    return [TrackingRecord(project="proj", shorts=shorts)]


def test_policy_infeasible_prints_one_line(monkeypatch):
    monkeypatch.setattr(anomaly, "TOTAL_ALPHA", 0.05)
    monkeypatch.setattr(anomaly, "REGISTERED_TESTS", 10)
    monkeypatch.setattr(anomaly, "REFERENCE_CAP", 200)
    result = anomaly.check_feasibility()
    from seo_manager.analytics.anomaly_stats import POLICY_INFEASIBLE, Refusal

    assert isinstance(result, Refusal) and result.reason == POLICY_INFEASIBLE


def test_fewer_than_required_references_is_insufficient_resolution():
    # 78 flat-ish-but-varied references (< 79 required at allocated_alpha=0.025), then a target.
    offsets = list(range(78))
    values = [(1000 + (i % 3) * 10, offset) for i, offset in enumerate(offsets)]
    values.append((100, 78))
    records = _records(values)
    result = anomaly.detect_post_anomaly(records, PLATFORM, allocated_alpha=0.025)
    assert isinstance(result, anomaly.AnomalyRefusal)
    assert result.reason == "insufficient_resolution"
    assert result.required == 79


def test_flat_references_are_no_baseline_variation():
    values = [(1000, offset) for offset in range(100)]
    values.append((100, 100))
    records = _records(values)
    result = anomaly.detect_post_anomaly(records, PLATFORM, allocated_alpha=0.025)
    assert isinstance(result, anomaly.AnomalyRefusal)
    assert result.reason == "no_baseline_variation"


def _varied_references(n=100, base=1400, spread=70):
    # Multiples of 7: day7 recomputes as (value // 7) * 7, and a non-multiple base/spread
    # collapses the intended spread once every reference round-trips through that division.
    return [(base + (i % 5) * spread, i) for i in range(n)]


def test_a_deep_dip_is_reported():
    values = _varied_references()
    values.append((100, 100))
    records = _records(values)
    result = anomaly.detect_post_anomaly(records, PLATFORM, allocated_alpha=0.025)
    assert isinstance(result, anomaly.AnomalyFinding)
    assert result.kind == "dip"
    assert result.p_value <= 0.025
    assert result.deviation < 0


def test_a_high_spike_is_reported():
    values = _varied_references()
    values.append((20000, 100))
    records = _records(values)
    result = anomaly.detect_post_anomaly(records, PLATFORM, allocated_alpha=0.025)
    assert isinstance(result, anomaly.AnomalyFinding)
    assert result.kind == "spike"
    assert result.p_value <= 0.025
    assert result.deviation > 0


def test_a_small_dip_clears_significance_but_not_the_practical_effect_gate():
    values = _varied_references(spread=7)
    target = int(1414 * 0.9)  # ~10% below the reference median: below the 25% relative floor
    values.append((target, 100))
    records = _records(values)
    result = anomaly.detect_post_anomaly(records, PLATFORM, allocated_alpha=0.025)
    assert result is None


def test_references_outside_the_lookback_window_are_excluded():
    # One ancient reference (200 days back) that would blow up the spread if included.
    values = [(9_999_999, -200)]
    values += _varied_references()
    values.append((100, 100))
    records = _records(values)
    result = anomaly.detect_post_anomaly(records, PLATFORM, allocated_alpha=0.025)
    assert isinstance(result, anomaly.AnomalyFinding)
    assert result.n == 100  # the ancient reference didn't make it in


def test_the_newest_200_references_are_kept_when_more_are_available():
    # 250 references packed inside the 180-day lookback (half-day spacing) — the reference
    # cap, not the lookback, must be what trims them.
    values = [(1400 + (i % 5) * 70, i * 0.5) for i in range(250)]
    values.append((100, 130))
    records = _records(values)
    result = anomaly.detect_post_anomaly(records, PLATFORM, allocated_alpha=0.025)
    assert isinstance(result, anomaly.AnomalyFinding)
    assert result.n == anomaly.REFERENCE_CAP


@pytest.mark.parametrize("case", ["dip", "spike"])
def test_log_line_shape(case):
    values = _varied_references()
    values.append((100 if case == "dip" else 20000, 100))
    records = _records(values)
    result = anomaly.detect_post_anomaly(records, PLATFORM, allocated_alpha=0.025)
    line = result.log_line()
    assert line.startswith("POST_ANOMALY platform=facebook video=vid100 kind=" + case)
    assert "shadow=true" in line and "\n" not in line
    assert "allocated_alpha=0.025" in line
