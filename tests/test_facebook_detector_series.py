"""`facebook_detector_series`: the detector reads the first preferred Page metric
whose stored series has more than ``min_days`` days; none qualifies -> empty."""

from __future__ import annotations

from gomoveshift.tracking.channel_daily import (
    FACEBOOK_DETECTOR_PREFERENCE,
    FACEBOOK_PAGE_METRICS,
    facebook_detector_series,
)


def _rows(metric, n):
    return [
        {"day": f"2026-08-{d:02d}", "dimension": f"facebook:{metric}", "value": "page", "views": d}
        for d in range(1, n + 1)
    ]


def test_preference_covers_exactly_the_pulled_metrics_unique_first():
    assert set(FACEBOOK_DETECTOR_PREFERENCE) == set(FACEBOOK_PAGE_METRICS)
    assert FACEBOOK_DETECTOR_PREFERENCE[0] == "page_total_media_view_unique"


def test_prefers_unique_viewers_when_long_enough():
    rows = _rows("page_media_view", 30) + _rows("page_total_media_view_unique", 29)
    name, series = facebook_detector_series(rows, 28)
    assert name == "page_total_media_view_unique"
    assert len(series) == 29


def test_skips_a_metric_with_too_few_days():
    rows = _rows("page_total_media_view_unique", 28) + _rows("page_media_view", 29)
    name, series = facebook_detector_series(rows, 28)
    assert name == "page_media_view"
    assert len(series) == 29


def test_empty_when_no_metric_qualifies():
    rows = _rows("page_media_view", 28) + _rows("page_video_views", 5)
    assert facebook_detector_series(rows, 28) == ("", {})
    assert facebook_detector_series([], 28) == ("", {})
