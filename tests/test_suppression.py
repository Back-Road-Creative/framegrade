"""Tests for `gomoveshift.tracking.suppression`; every case is a synthetic series."""

from __future__ import annotations

import datetime

from gomoveshift.tracking.channel_daily import TRAFFIC_SOURCE_DIMENSION
from gomoveshift.tracking.suppression import (
    classify_facebook_reel_reach,
    classify_youtube_shorts_feed,
)


def test_zero_total_views_day_is_suppressed_immediately():
    rows = [
        {
            "day": "2000-01-02",
            "dimension": TRAFFIC_SOURCE_DIMENSION,
            "value": "SHORTS",
            "views": 300,
        },
        {
            "day": "2000-01-02",
            "dimension": TRAFFIC_SOURCE_DIMENSION,
            "value": "OTHER",
            "views": 100,
        },
        {"day": "2000-01-03", "dimension": TRAFFIC_SOURCE_DIMENSION, "value": "OTHER", "views": 0},
    ]
    status = classify_youtube_shorts_feed(rows)
    assert status["2000-01-03"] == "SUPPRESSED"


def test_healthy_start_before_any_dip():
    rows = [
        {
            "day": "2000-01-01",
            "dimension": TRAFFIC_SOURCE_DIMENSION,
            "value": "SHORTS",
            "views": 800,
        },
        {
            "day": "2000-01-01",
            "dimension": TRAFFIC_SOURCE_DIMENSION,
            "value": "OTHER",
            "views": 200,
        },
    ]
    status = classify_youtube_shorts_feed(rows)
    assert status["2000-01-01"] == "HEALTHY"


def test_facebook_suppressed_when_seven_day_median_drops_below_threshold():
    daily = {}
    day0 = datetime.date(2000, 1, 1)
    # 28 days of a stable baseline (1000 reel views/day).
    for i in range(28):
        daily[(day0 + datetime.timedelta(days=i)).isoformat()] = 1000.0
    # 7 more days collapsed well under 0.3x the 28-day median.
    for i in range(28, 35):
        daily[(day0 + datetime.timedelta(days=i)).isoformat()] = 100.0

    status = classify_facebook_reel_reach(daily)
    last_day = (day0 + datetime.timedelta(days=34)).isoformat()
    assert status[last_day] == "SUPPRESSED"


def test_facebook_healthy_when_no_collapse():
    daily = {}
    day0 = datetime.date(2000, 1, 1)
    for i in range(40):
        daily[(day0 + datetime.timedelta(days=i)).isoformat()] = 1000.0

    status = classify_facebook_reel_reach(daily)
    assert all(v == "HEALTHY" for v in status.values())


def _shorts_rows(shares_pct):
    """Constructed series: 100 invented views a day, ``shares_pct[i]`` of them SHORTS."""
    first = datetime.date(2000, 1, 1)
    rows, days = [], []
    for offset, pct in enumerate(shares_pct):
        day = (first + datetime.timedelta(days=offset)).isoformat()
        days.append(day)
        for value, views in (("SHORTS", pct), ("OTHER", 100 - pct)):
            rows.append(
                {"day": day, "dimension": TRAFFIC_SOURCE_DIMENSION, "value": value, "views": views}
            )
    return rows, days


def test_every_separated_collapse_is_flagged_within_two_days():
    # Three separated dips, each followed by a full healthy streak.
    series = [80] * 30 + [20] + [80] * 10 + [30] + [80] * 10 + [10, 15] + [80] * 10
    rows, days = _shorts_rows(series)
    dips = [30, 41, 52]

    status = classify_youtube_shorts_feed(rows)

    for dip in dips:
        window = days[dip : dip + 3]
        assert any(status[d] == "SUPPRESSED" for d in window), (dip, window)
    # The first dip day itself is the flagged one (no two-day lag needed).
    assert status[days[30]] == "SUPPRESSED"


def test_recovery_after_a_collapse_is_reported_as_recovering():
    series = [80] * 30 + [20] + [80] * 10
    rows, days = _shorts_rows(series)

    status = classify_youtube_shorts_feed(rows)

    assert status[days[30]] == "SUPPRESSED"
    # The share is back above the 50% floor and 70% bar from the next day on.
    assert all(status[d] == "RECOVERING" for d in days[31:35])
    assert all(status[d] != "SUPPRESSED" for d in days[31:])


def test_first_non_low_day_after_suppression_is_recovering_and_stays_so():
    series = [80] * 30 + [20] + [80] * 10
    rows, days = _shorts_rows(series)

    status = classify_youtube_shorts_feed(rows)

    assert status[days[31]] == "RECOVERING"
    # Even past a full 7-day healthy streak (days 31..37) the series stays
    # RECOVERING; the downstream ladder, not this classifier, steps it up.
    assert status[days[37]] == "RECOVERING"
    assert status[days[40]] == "RECOVERING"


def test_a_share_between_floor_and_healthy_bar_stays_recovering_not_healthy():
    # 60% clears the 50% floor but never reaches the 70% healthy bar, so the
    # streak never builds and the state never returns to HEALTHY.
    series = [80] * 30 + [20] + [60] * 12
    rows, days = _shorts_rows(series)

    status = classify_youtube_shorts_feed(rows)

    assert status[days[30]] == "SUPPRESSED"
    assert all(status[d] == "RECOVERING" for d in days[31:])


def test_a_relapse_during_recovery_is_flagged_again():
    series = [80] * 30 + [20] + [80] * 4 + [20] + [80] * 3
    rows, days = _shorts_rows(series)

    status = classify_youtube_shorts_feed(rows)

    assert status[days[30]] == "SUPPRESSED"
    assert [status[d] for d in days[31:35]] == ["RECOVERING"] * 4
    assert status[days[35]] == "SUPPRESSED"
    assert [status[d] for d in days[36:]] == ["RECOVERING"] * 3


def test_quiet_synthetic_window_raises_no_alarm_after_warmup():
    series = [75, 80, 85, 78, 82] * 20
    rows, days = _shorts_rows(series)

    status = classify_youtube_shorts_feed(rows)

    assert set(status) == set(days)
    assert all(value == "HEALTHY" for value in status.values())
