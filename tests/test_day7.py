"""The one settled day-7 function: date-keyed, window = posted day .. +6."""

from __future__ import annotations

from datetime import date

from gomoveshift.tracking.day7 import window_settled, settled_day7_views
from gomoveshift.tracking.schema import DailyMetricPoint

POSTED = "2026-09-20T14:00:00"
LATE = date(2026, 10, 9)


def _pts(days, views=10):
    return [DailyMetricPoint(day=f"2026-09-{d:02d}", views=views) for d in days]


def test_day7_is_date_keyed_sum_over_posted_day_plus_6():
    # 09-22 is a zero-view day the API omitted; 09-27 is outside the window.
    daily = _pts([20, 21, 23, 24, 25, 26, 27])
    assert settled_day7_views(POSTED, daily, LATE) == 60
    # Fewer than 7 points no longer drops the short, and points before the
    # posted day are not counted.
    assert settled_day7_views(POSTED, _pts([19]) + _pts([20, 26]), LATE) == 20


def test_window_unsettled_until_posted_plus_7d_le_today_minus_2d():
    posted = date(2026, 9, 20)
    assert not window_settled(posted, date(2026, 9, 28))
    assert window_settled(posted, date(2026, 9, 29))
    daily = _pts(range(20, 27))
    assert settled_day7_views(POSTED, daily, date(2026, 9, 28)) is None
    assert settled_day7_views(POSTED, daily, date(2026, 9, 29)) == 70


def test_unusable_series_is_none_never_zero():
    assert settled_day7_views(POSTED, [], LATE) is None
    assert settled_day7_views(None, _pts(range(20, 27)), LATE) is None
    unreported = _pts(range(20, 27))
    unreported[2] = DailyMetricPoint(day="2026-09-22")
    assert settled_day7_views(POSTED, unreported, LATE) is None
    # A series that stops before the window's last day is incomplete.
    assert settled_day7_views(POSTED, _pts(range(20, 25)), LATE) is None
