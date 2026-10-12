"""Constructed suppression scenarios; these are not measured channel analytics."""

from datetime import date, timedelta

import pytest
from gomoveshift.tracking.channel_daily import TRAFFIC_SOURCE_DIMENSION
from gomoveshift.tracking.suppression import classify_youtube_shorts_feed


def traffic_rows(shorts_percentages):
    """Use 100 invented views per day so each input is the Shorts percentage."""
    first_day = date(2000, 1, 1)
    rows = []
    days = []
    for offset, percentage in enumerate(shorts_percentages):
        day = (first_day + timedelta(days=offset)).isoformat()
        days.append(day)
        for category, count in (("SHORTS", percentage), ("OTHER", 100 - percentage)):
            rows.append(
                {
                    "day": day,
                    "dimension": TRAFFIC_SOURCE_DIMENSION,
                    "value": category,
                    "views": count,
                }
            )
    return rows, days


@pytest.mark.parametrize("prior_days", [0, 1, 26, 27])
def test_low_positive_share_during_warmup_does_not_alarm(prior_days):
    rows, days = traffic_rows([80] * prior_days + [20])

    status = classify_youtube_shorts_feed(rows)

    assert set(status) == set(days)
    assert all(value == "HEALTHY" for value in status.values())


def test_stable_synthetic_history_has_no_false_alarm():
    rows, days = traffic_rows([70, 80, 90] * 30)

    status = classify_youtube_shorts_feed(rows)

    assert set(status) == set(days)
    assert all(value == "HEALTHY" for value in status.values())


@pytest.mark.parametrize("prior_days", [28, 35])
def test_single_day_collapse_after_warmup_is_flagged_on_that_day(prior_days):
    # A later healthy observation makes this a one-day collapse, not a streak.
    rows, days = traffic_rows([80] * prior_days + [20, 80])

    status = classify_youtube_shorts_feed(rows)

    assert set(status) == set(days)
    assert all(status[day] == "HEALTHY" for day in days[:prior_days])
    assert status[days[prior_days]] == "SUPPRESSED"
