"""The one settled day-7 function.

Day-7 views are the sum of the Analytics daily rows whose ``day`` falls in
``posted day .. posted day + 6`` -- keyed by DATE, not by position, because the
API omits a zero-view day and a positional "first 7 points" then reads past the
window. A window is *settled* once ``posted + 7d <= today - 2d``: Analytics
reports lag about two days, so a series fetched earlier is still moving.

``None`` means "no usable settled series" (never a zero): no posted day, an
unsettled window, no point in the window, a point whose views were never
reported, or a series that stops before the window's last day.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Iterable, List, Optional, Tuple

DAY7_DAYS = 7  #: the window spans the posted day and the 6 after it
SETTLE_LAG_DAYS = 2  #: Analytics reporting lag


def posted_date(posted_at: Optional[str]) -> Optional[date]:
    """The calendar date of an ISO timestamp / ``YYYY-MM-DD`` string, or None."""
    try:
        return date.fromisoformat(str(posted_at)[:10]) if posted_at else None
    except ValueError:
        return None


def window_settled(posted: date, today: date) -> bool:
    """True once ``posted + 7d <= today - 2d``."""
    return posted + timedelta(days=DAY7_DAYS) <= today - timedelta(days=SETTLE_LAG_DAYS)


def settled_day7_views(
    posted_at: Optional[str], daily: Iterable[Any], today: date
) -> Optional[int]:
    """Day-7 views from ``daily`` (``DailyMetricPoint``s), or None (see module doc)."""
    posted = posted_date(posted_at)
    if posted is None or not window_settled(posted, today):
        return None
    last = posted + timedelta(days=DAY7_DAYS - 1)
    total = 0
    reached = False
    in_window = False
    for point in daily or []:
        try:
            day = date.fromisoformat(str(point.day)[:10])
        except ValueError:
            continue
        if day == last:
            reached = True
        if posted <= day <= last:
            if point.views is None:
                return None
            total += point.views
            in_window = True
        elif day > last:
            reached = True
    return total if in_window and reached else None


def short_day7_views(short, today: date) -> Optional[int]:
    """A short's settled day-7 views summed over the platforms that have a
    settled series; ``None`` if none does."""
    views = [settled_day7_views(short.posted_at, m.daily, today) for m in short.metrics.values()]
    measured = [v for v in views if v is not None]
    return sum(measured) if measured else None


def rank_shorts(record, today: date) -> List[Tuple[int, int]]:
    """``(short_index, day7_views)`` for shorts with a settled series,
    sorted by day7_views descending."""
    ranked = []
    for short in record.shorts:
        views = short_day7_views(short, today)
        if views is not None:
            ranked.append((short.index, views))
    return sorted(ranked, key=lambda pair: pair[1], reverse=True)


def has_settled_series(short, today: date, platform: str = "youtube") -> bool:
    """True when ``short`` already holds a settled series for ``platform``."""
    m = short.metrics.get(platform)
    return m is not None and settled_day7_views(short.posted_at, m.daily, today) is not None
