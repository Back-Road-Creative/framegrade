"""Channel-level daily views by traffic source / content type.

Mocks the YouTube Analytics client entirely -- no network.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from gomoveshift.tracking import channel_daily


def _fake_collector(rows_by_dimension: dict) -> MagicMock:
    """A stand-in for `YouTubeCollector` exposing only
    `fetch_channel_daily_dimension`, keyed by dimension name."""
    collector = MagicMock()
    collector.fetch_channel_daily_dimension.side_effect = lambda dimension, start_date, end_date: (
        rows_by_dimension.get(dimension, [])
    )
    return collector


def test_pull_channel_daily_stores_both_dimensions(tmp_path):
    collector = _fake_collector(
        {
            channel_daily.TRAFFIC_SOURCE_DIMENSION: [
                ("2026-09-20", "SHORTS", 100),
                ("2026-09-20", "YT_SEARCH", 50),
            ],
            channel_daily.CONTENT_TYPE_DIMENSION: [
                ("2026-09-20", "SHORTS", 90),
                ("2026-09-20", "VIDEO_ON_DEMAND", 60),
            ],
        }
    )

    pulled = channel_daily.pull_channel_daily(
        collector, "gomoveshift", "2026-09-20", "2026-09-20", root=tmp_path
    )

    assert pulled == 4
    rows = channel_daily.load_rows("gomoveshift", root=tmp_path)
    assert len(rows) == 4
    assert {(r["dimension"], r["value"], r["views"]) for r in rows} == {
        (channel_daily.TRAFFIC_SOURCE_DIMENSION, "SHORTS", 100),
        (channel_daily.TRAFFIC_SOURCE_DIMENSION, "YT_SEARCH", 50),
        (channel_daily.CONTENT_TYPE_DIMENSION, "SHORTS", 90),
        (channel_daily.CONTENT_TYPE_DIMENSION, "VIDEO_ON_DEMAND", 60),
    }


def test_pull_channel_daily_is_idempotent_on_replay(tmp_path):
    collector = _fake_collector(
        {channel_daily.TRAFFIC_SOURCE_DIMENSION: [("2026-09-20", "SHORTS", 100)]}
    )

    channel_daily.pull_channel_daily(
        collector, "gomoveshift", "2026-09-20", "2026-09-20", root=tmp_path
    )
    channel_daily.pull_channel_daily(
        collector, "gomoveshift", "2026-09-20", "2026-09-20", root=tmp_path
    )

    rows = channel_daily.load_rows("gomoveshift", root=tmp_path)
    assert len(rows) == 1
    assert rows[0]["views"] == 100


def test_pull_channel_daily_replaces_a_repulled_day_without_dropping_others(tmp_path):
    collector = _fake_collector(
        {channel_daily.TRAFFIC_SOURCE_DIMENSION: [("2026-09-20", "SHORTS", 100)]}
    )
    channel_daily.pull_channel_daily(
        collector, "gomoveshift", "2026-09-20", "2026-09-20", root=tmp_path
    )

    corrected = _fake_collector(
        {channel_daily.TRAFFIC_SOURCE_DIMENSION: [("2026-09-20", "SHORTS", 120)]}
    )
    channel_daily.pull_channel_daily(
        corrected, "gomoveshift", "2026-09-20", "2026-09-20", root=tmp_path
    )

    rows = channel_daily.load_rows("gomoveshift", root=tmp_path)
    assert len(rows) == 1
    assert rows[0]["views"] == 120


def test_pull_channel_daily_is_scoped_per_brand(tmp_path):
    collector = _fake_collector(
        {channel_daily.TRAFFIC_SOURCE_DIMENSION: [("2026-09-20", "SHORTS", 100)]}
    )
    channel_daily.pull_channel_daily(
        collector, "brand-a", "2026-09-20", "2026-09-20", root=tmp_path
    )

    assert channel_daily.load_rows("brand-a", root=tmp_path)
    assert channel_daily.load_rows("brand-b", root=tmp_path) == []


def test_load_rows_missing_store_returns_empty(tmp_path):
    assert channel_daily.load_rows("nobody", root=tmp_path) == []


def test_shorts_feed_share_divides_shorts_views_by_daily_total():
    rows = [
        {
            "day": "2026-09-20",
            "dimension": channel_daily.TRAFFIC_SOURCE_DIMENSION,
            "value": "SHORTS",
            "views": 75,
        },
        {
            "day": "2026-09-20",
            "dimension": channel_daily.TRAFFIC_SOURCE_DIMENSION,
            "value": "YT_SEARCH",
            "views": 25,
        },
        {
            "day": "2026-09-21",
            "dimension": channel_daily.TRAFFIC_SOURCE_DIMENSION,
            "value": "SHORTS",
            "views": 10,
        },
        # A content-type row for the same day must never leak into the
        # traffic-source total.
        {
            "day": "2026-09-20",
            "dimension": channel_daily.CONTENT_TYPE_DIMENSION,
            "value": "SHORTS",
            "views": 9999,
        },
    ]

    share = channel_daily.shorts_feed_share(rows)

    assert share["2026-09-20"] == 0.75
    assert share["2026-09-21"] == 1.0


def test_shorts_feed_share_omits_a_day_with_no_traffic_source_rows():
    rows = [
        {
            "day": "2026-09-20",
            "dimension": channel_daily.CONTENT_TYPE_DIMENSION,
            "value": "SHORTS",
            "views": 50,
        },
    ]

    assert channel_daily.shorts_feed_share(rows) == {}


def test_shorts_feed_share_skips_unreported_views():
    rows = [
        {
            "day": "2026-09-20",
            "dimension": channel_daily.TRAFFIC_SOURCE_DIMENSION,
            "value": "SHORTS",
            "views": None,
        },
    ]

    assert channel_daily.shorts_feed_share(rows) == {}
