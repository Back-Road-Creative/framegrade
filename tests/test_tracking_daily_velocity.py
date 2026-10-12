"""Daily time-series analytics collection + first-48h velocity."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from gomoveshift.tracking.collector import YouTubeCollector
from gomoveshift.tracking.schema import (
    DailyMetricPoint,
    PlatformMetrics,
    ShortRecord,
    TrackingRecord,
)
from gomoveshift.tracking.store import TrackingStore
from gomoveshift.tracking.velocity import first_hours_velocity


def _write_token(tmp_path):
    token_path = tmp_path / "youtube_credentials.json"
    token_path.write_text(
        json.dumps({"token": "t", "refresh_token": "r", "scopes": YouTubeCollector.SCOPES})
    )
    return token_path


def test_platform_metrics_daily_round_trips_through_json():
    metrics = PlatformMetrics(views=100, daily=[DailyMetricPoint(day="2026-08-01", views=40)])
    restored = PlatformMetrics.from_dict(json.loads(json.dumps(metrics.to_dict())))
    assert restored.daily == [DailyMetricPoint(day="2026-08-01", views=40)]
    assert PlatformMetrics().daily == []


def test_no_token_or_api_failure_returns_empty_daily(tmp_path):
    no_token = YouTubeCollector(token_path=tmp_path / "missing.json")
    assert no_token.fetch_analytics_daily_metrics("vid1", "2026-08-01") == []

    with_token = YouTubeCollector(token_path=_write_token(tmp_path))
    with patch("googleapiclient.discovery.build", side_effect=Exception("quota")):
        assert with_token.fetch_analytics_daily_metrics("vid1", "2026-08-01") == []


def test_daily_rows_capped_to_max_days(tmp_path):
    collector = YouTubeCollector(token_path=_write_token(tmp_path))
    rows = [[f"2026-08-{d:02d}", 10 * d, 5, 30.0, 1, 0] for d in range(1, 20)]
    analytics = MagicMock()
    analytics.reports().query().execute.return_value = {"rows": rows}

    with patch("googleapiclient.discovery.build", return_value=analytics):
        with patch("google.oauth2.credentials.Credentials"):
            daily = collector.fetch_analytics_daily_metrics("vid1", "2026-08-01", max_days=14)

    assert len(daily) == 14
    assert daily[0] == dict(
        zip(
            ["day", "views", "watch_time_minutes", "avg_view_duration_seconds", "likes", "shares"],
            ["2026-08-01", 10, 5, 30.0, 1, 0],
        )
    )


def _fake_youtube(view_count):
    fake = MagicMock()
    fake.videos().list().execute.return_value = {
        "items": [
            {"statistics": {"viewCount": str(view_count), "likeCount": "0", "commentCount": "0"}}
        ]
    }
    return fake


def test_daily_attached_when_aggregate_succeeds_and_skipped_when_it_fails(tmp_path):
    collector = YouTubeCollector(token_path=_write_token(tmp_path))
    analytics = MagicMock()
    analytics.reports().query().execute.side_effect = [
        {"rows": [[500, 120, 45.0, 10, 3]]},
        {"rows": [["2026-08-01", 200, 40, 45.0, 5, 1]]},
    ]

    with patch.object(collector, "_get_service", return_value=_fake_youtube(500)):
        with patch("googleapiclient.discovery.build", return_value=analytics):
            with patch("google.oauth2.credentials.Credentials"):
                metrics = collector.collect_short_metrics(
                    "vid1", published_at="2026-08-01T00:00:00"
                )
    assert metrics.views == 500
    assert len(metrics.daily) == 1
    assert metrics.daily[0].views == 200

    no_token = YouTubeCollector(token_path=tmp_path / "missing.json")
    with patch.object(no_token, "_get_service", return_value=_fake_youtube(1)):
        with patch.object(no_token, "fetch_analytics_daily_metrics") as fetch_daily:
            metrics = no_token.collect_short_metrics("vid1", published_at="2026-08-01T00:00:00")
    fetch_daily.assert_not_called()
    assert metrics.daily == []


def test_first_hours_velocity_ratio(tmp_path):
    store = TrackingStore(root=tmp_path)
    metrics = PlatformMetrics(
        views=100,
        daily=[
            DailyMetricPoint(day="2026-08-01", views=40),
            DailyMetricPoint(day="2026-08-02", views=20),
            DailyMetricPoint(day="2026-08-03", views=40),
        ],
    )
    short = ShortRecord(index=1, platform_ids={"youtube": "vidX"}, metrics={"youtube": metrics})
    store.save(TrackingRecord(project="proj", shorts=[short]))

    assert first_hours_velocity("vidX", hours=48, store=store) == 0.6


def test_first_hours_velocity_none_without_daily_data_or_unknown_video(tmp_path):
    store = TrackingStore(root=tmp_path)
    short = ShortRecord(
        index=1, platform_ids={"youtube": "vidY"}, metrics={"youtube": PlatformMetrics(views=10)}
    )
    store.save(TrackingRecord(project="proj", shorts=[short]))

    assert first_hours_velocity("vidY", store=store) is None
    assert first_hours_velocity("nope", store=store) is None
