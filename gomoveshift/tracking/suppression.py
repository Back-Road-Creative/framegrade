"""
Per-platform suppression detector.

A platform's daily reach can be throttled by the platform itself (a
feed-share collapse, an account-level reach cap) well before any downstream
metric (views, engagement) makes the throttle obvious. This module
classifies each platform's day as ``HEALTHY``, ``SUPPRESSED`` or
``RECOVERING`` from the metric series the ``channel_daily`` pull already
stores -- report-only, no callers act on the classification.

YouTube reuses ``channel_daily.shorts_feed_share`` -- this module does not
re-derive a share from raw rows, only classifies the series that function
already produces.

Thresholds:
  * YouTube low-share trigger: share <50% OR share <0.5x its own 28-day
    trailing median.
  * YouTube healthy recovery: share >=70% sustained for 7 days. The first
    day after a SUPPRESSED day that is not itself low is RECOVERING, and the
    series stays RECOVERING until the next low (or zero-total) day: the
    7-day streak is counted, but the HEALTHY branch is only reachable while
    the state is still SUPPRESSED, which a non-low day normally ends. The
    exception is a suppression that starts inside the 28-day warm-up: with
    no trailing median yet the state stays SUPPRESSED while the streak is
    counted. On the first day the median exists the series goes straight to
    HEALTHY only if that day is not low and at least 7 consecutive days at
    or above 70% have accumulated by then; otherwise it becomes RECOVERING
    (or stays SUPPRESSED if that day is low), like any other suppression.
    Downstream
    ``response_policy.daily_cap_for`` climbs its ladder one rung per 7
    healthy days while RECOVERING.
  * Facebook low-reach trigger: 7-day median reel views <0.3x the 28-day
    median. The 28-day median that triggered it is frozen as the baseline;
    recovery is judged against that baseline (a live median would decay
    toward the suppressed level and fake a recovery): 7-day median >=0.5x
    -> RECOVERING, <0.3x -> back to SUPPRESSED, 7 consecutive days >=0.7x
    -> HEALTHY (baseline cleared).

The YouTube trigger fires on the first day whose share breaches either
threshold, not after two consecutive low days: a single-day collapse that
recovers the next day would otherwise never be flagged. It fires only once
28 days of share history exist to compute the relative threshold against
(``_MEDIAN_WINDOW_DAYS``). That warm-up gate is required, not optional:
without it, an early day with no trailing median yet would breach the 50%
absolute threshold on its own and raise a false alarm.

A day with zero total channel views (a reporting-outage day, not a graceful
share of 0%) is flagged SUPPRESSED directly: ``shorts_feed_share`` correctly
omits a zero-total day rather than fabricating a 0.0 share, but this
detector must not treat "no share could be computed" as "nothing is wrong".
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Sequence

from .channel_daily import TRAFFIC_SOURCE_DIMENSION, shorts_feed_share

Status = Literal["HEALTHY", "SUPPRESSED", "RECOVERING"]

#: YouTube Shorts-feed share thresholds (see the module docstring).
YOUTUBE_LOW_SHARE_ABS = 0.50
YOUTUBE_LOW_SHARE_REL_OF_MEDIAN = 0.5
YOUTUBE_HEALTHY_SHARE_ABS = 0.70
YOUTUBE_HEALTHY_STREAK_DAYS = 7
_MEDIAN_WINDOW_DAYS = 28

#: Facebook reel-views thresholds (see the module docstring).
FACEBOOK_LOW_REACH_REL_OF_MEDIAN = 0.3
FACEBOOK_ROLLING_WINDOW_DAYS = 7
_FACEBOOK_MEDIAN_WINDOW_DAYS = 28
#: SUPPRESSED -> RECOVERING once the 7-day median reaches this share of the frozen baseline.
FACEBOOK_RECOVERING_REL_OF_BASELINE = 0.5
#: RECOVERING -> SUPPRESSED again if the 7-day median falls under this share of the frozen baseline.
FACEBOOK_RELAPSE_REL_OF_BASELINE = 0.3
#: A day counts toward the HEALTHY streak at or above this share of the frozen baseline.
FACEBOOK_HEALTHY_REL_OF_BASELINE = 0.7
#: Consecutive days at the healthy share before RECOVERING becomes HEALTHY.
FACEBOOK_HEALTHY_STREAK_DAYS = 7


def _daily_traffic_totals(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Per-day total traffic-source views, summed directly over the
    `channel_daily` row shape.

    `shorts_feed_share` deliberately omits a zero-total day rather than
    reporting a fabricated 0.0 share -- correct for a share consumer, but
    this detector needs to see the zero itself: it is a suppression signal
    in its own right (see module docstring).
    """
    totals: Dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("dimension") != TRAFFIC_SOURCE_DIMENSION:
            continue
        day = row.get("day")
        views = row.get("views")
        if not day or views is None:
            continue
        totals[day] += views
    return dict(totals)


def classify_youtube_shorts_feed(rows: Sequence[Dict[str, Any]]) -> Dict[str, Status]:
    """Classify every day present in `rows` (`channel_daily.load_rows`
    shape) as ``HEALTHY``, ``SUPPRESSED`` or ``RECOVERING``.

    Returns a ``{day: status}`` map covering every day that has at least
    one traffic-source row (including a zero-views day) -- never a day
    absent from `rows` entirely, which is "no data pulled", not a status.
    """
    totals = _daily_traffic_totals(rows)
    shares = shorts_feed_share(list(rows))

    status: Dict[str, Status] = {}
    state: Status = "HEALTHY"
    healthy_streak = 0
    share_history: List[float] = []

    for day in sorted(totals):
        if totals[day] == 0:
            state = "SUPPRESSED"
            healthy_streak = 0
            status[day] = state
            continue

        share = shares.get(day)
        if share is None:
            status[day] = state
            continue

        median28: Optional[float] = (
            statistics.median(share_history[-_MEDIAN_WINDOW_DAYS:])
            if len(share_history) >= _MEDIAN_WINDOW_DAYS
            else None
        )
        is_healthy_day = share >= YOUTUBE_HEALTHY_SHARE_ABS
        healthy_streak = healthy_streak + 1 if is_healthy_day else 0

        if median28 is not None:
            is_low = share < YOUTUBE_LOW_SHARE_ABS or (
                share < YOUTUBE_LOW_SHARE_REL_OF_MEDIAN * median28
            )
            if is_low:
                state = "SUPPRESSED"
            elif state == "SUPPRESSED" and healthy_streak >= YOUTUBE_HEALTHY_STREAK_DAYS:
                state = "HEALTHY"
            elif state == "SUPPRESSED":
                state = "RECOVERING"

        status[day] = state
        share_history.append(share)

    return status


def classify_facebook_reel_reach(daily_reel_views: Dict[str, float]) -> Dict[str, Status]:
    """Classify each day of a Facebook reel-views series.

    ``daily_reel_views`` is ``{day (YYYY-MM-DD): reel views that day}``,
    already ordered by the caller's pull -- this module only sorts by the
    day string, so callers must use a sortable (ISO) day format.

    SUPPRESSED triggers when the trailing 7-day median of reel views falls
    below 0.3x the trailing 28-day median -- both windows
    computed over days strictly before the day being classified, so a
    day's own value never inflates the baseline it is compared against.

    The 28-day median at the moment of the SUPPRESSED flip is frozen as the
    baseline. While SUPPRESSED/RECOVERING the 7-day median is compared with
    that frozen baseline, never the live 28-day median: 7-day median >=0.5x
    -> RECOVERING; <0.3x -> SUPPRESSED; ``FACEBOOK_HEALTHY_STREAK_DAYS``
    consecutive days >=0.7x -> HEALTHY, which clears the baseline.
    """
    days = sorted(daily_reel_views)
    values = [daily_reel_views[d] for d in days]
    status: Dict[str, Status] = {}
    state: Status = "HEALTHY"
    baseline: Optional[float] = None
    healthy_streak = 0

    for i, day in enumerate(days):
        window7 = values[max(0, i - FACEBOOK_ROLLING_WINDOW_DAYS) : i]
        window28 = values[max(0, i - _FACEBOOK_MEDIAN_WINDOW_DAYS) : i]
        if (
            len(window7) >= FACEBOOK_ROLLING_WINDOW_DAYS
            and len(window28) >= _FACEBOOK_MEDIAN_WINDOW_DAYS
        ):
            median7 = statistics.median(window7)
            median28 = statistics.median(window28)
            if baseline is None:
                if median28 and median7 < FACEBOOK_LOW_REACH_REL_OF_MEDIAN * median28:
                    state = "SUPPRESSED"
                    baseline = median28
                    healthy_streak = 0
            else:
                if median7 >= FACEBOOK_HEALTHY_REL_OF_BASELINE * baseline:
                    healthy_streak += 1
                else:
                    healthy_streak = 0
                if median7 < FACEBOOK_RELAPSE_REL_OF_BASELINE * baseline:
                    state = "SUPPRESSED"
                elif healthy_streak >= FACEBOOK_HEALTHY_STREAK_DAYS:
                    state = "HEALTHY"
                    baseline = None
                elif median7 >= FACEBOOK_RECOVERING_REL_OF_BASELINE * baseline:
                    state = "RECOVERING"
        status[day] = state

    return status
