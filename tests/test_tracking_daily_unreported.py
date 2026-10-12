"""A day's metrics can say nothing was reported.

Regression coverage for the DailyMetricPoint half of the fabricated-zero bug
(see test_tracking_platform_metrics_missing.py for the PlatformMetrics half,
fixed): every per-day field defaulted to numeric 0, so a day the
platform never reported was indistinguishable from a day it reported a real
zero. Readers (velocity, volume) then summed those fabricated zeros as if
they were measured.
"""

from __future__ import annotations

import json

from gomoveshift.tracking.schema import (
    DailyMetricPoint,
    PlatformMetrics,
    ShortRecord,
    TrackingRecord,
)
from gomoveshift.tracking.velocity import first_hours_velocity
from datetime import date

from gomoveshift.tracking.day7 import settled_day7_views


def test_unreported_day_round_trips_as_missing_not_zero():
    point = DailyMetricPoint(day="2026-08-01")  # nothing reported that day
    restored = DailyMetricPoint.from_dict(json.loads(json.dumps(point.__dict__)))

    assert restored.views is None
    assert point.__dict__["views"] is None


def test_measured_zero_day_survives_as_zero_not_missing():
    point = DailyMetricPoint(day="2026-08-01", views=0, likes=0, comments=0, shares=0)
    restored = DailyMetricPoint.from_dict(json.loads(json.dumps(point.__dict__)))

    assert restored.views == 0
    assert restored.views is not None


def test_legacy_zero_payload_still_loads():
    """A day persisted before this fix, carrying plain zeros, must still
    load without raising — we cannot retroactively know whether a legacy
    zero was measured or missing (PlatformMetrics.from_dict follows the
    same rule for its own legacy payloads), so it loads as the literal
    value on disk."""
    legacy_payload = {
        "day": "2026-08-01",
        "views": 0,
        "likes": 0,
        "comments": 0,
        "shares": 0,
        "watch_time_seconds": 0.0,
    }
    restored = DailyMetricPoint.from_dict(legacy_payload)

    assert restored.views == 0
    assert restored.day == "2026-08-01"


def test_day_identity_field_is_never_optional():
    point = DailyMetricPoint(day="2026-08-01")
    assert point.day == "2026-08-01"


def test_velocity_is_unknown_when_a_day_in_the_window_is_unreported():
    """early_views/total_views is a ratio, same as engagement_rate — a
    partial sum over the early window would silently answer a different,
    smaller question than "what fraction of views came early", so an
    unreported day inside the window makes the whole ratio unknown rather
    than treating the gap as zero views."""
    metrics = PlatformMetrics(
        views=100,
        daily=[
            DailyMetricPoint(day="2026-08-01", views=40),
            DailyMetricPoint(day="2026-08-02"),  # unreported, not zero
        ],
    )
    short = ShortRecord(index=1, platform_ids={"youtube": "vidX"}, metrics={"youtube": metrics})

    class _FakeStore:
        def load_all_records(self):
            return [TrackingRecord(project="proj", shorts=[short])]

    assert first_hours_velocity("vidX", hours=48, store=_FakeStore()) is None


def test_velocity_computes_normally_when_the_window_is_fully_reported():
    metrics = PlatformMetrics(
        views=100,
        daily=[
            DailyMetricPoint(day="2026-08-01", views=40),
            DailyMetricPoint(day="2026-08-02", views=20),
        ],
    )
    short = ShortRecord(index=1, platform_ids={"youtube": "vidX"}, metrics={"youtube": metrics})

    class _FakeStore:
        def load_all_records(self):
            return [TrackingRecord(project="proj", shorts=[short])]

    assert first_hours_velocity("vidX", hours=48, store=_FakeStore()) == 0.6


def test_volume_day7_skips_a_platform_whose_window_has_an_unreported_day():
    """Mirrors the existing under-7-points skip: an unreported day inside
    the first 7 is data the platform couldn't provide, not a real zero, so
    the platform is skipped for this short rather than under-scored."""
    daily = [DailyMetricPoint(day=f"2026-01-{d:02d}", views=10) for d in range(1, 8)]
    daily[3] = DailyMetricPoint(day="2026-01-04")  # unreported mid-window

    assert settled_day7_views("2026-01-01T00:00:00", daily, date(2026, 6, 1)) is None
