"""A published Short's observation carries watch time and view percentage.

``collect_observation`` used to read only the Data API (views/likes/comments),
so every Short had null watch time and ``averageViewPercentage``. It now also
asks the Analytics client the collector already builds for long-form, filtered
to the one video since publish. No network: the clients are faked with
responses shaped like ``reports.query`` output (columnHeaders + rows).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gomoveshift.tracking.collector import MetricsCollector, YouTubeCollector
from gomoveshift.tracking.observation import MetricObservation, ObservationStatus
from gomoveshift.tracking.schema import PlatformMetrics, ShortRecord

_HEADERS = [
    {"name": n, "columnType": "METRIC", "dataType": "INTEGER"}
    for n in (
        "views",
        "engagedViews",
        "estimatedMinutesWatched",
        "averageViewDuration",
        "averageViewPercentage",
    )
]


def _report(row):
    return {"columnHeaders": _HEADERS, "rows": [row] if row else []}


def _data_api():
    fake = MagicMock()
    fake.videos().list().execute.return_value = {
        "items": [{"statistics": {"viewCount": "500", "likeCount": "3", "commentCount": "1"}}]
    }
    return fake


def _analytics(report=None, exc=None):
    fake = MagicMock()
    q = fake.reports().query().execute
    if exc is not None:
        q.side_effect = exc
    else:
        q.return_value = report
    return fake


def _observe(analytics, published_at="2026-11-01T12:00:00+00:00", now="2026-11-02T13:00:00+00:00"):
    c = YouTubeCollector()
    with (
        patch.object(c, "is_available", return_value=True),
        patch.object(c, "_get_service", return_value=_data_api()),
        patch.object(c, "_analytics_client", return_value=analytics),
        patch("gomoveshift.tracking.collector.datetime") as dt,
    ):
        from datetime import datetime as real

        dt.now.return_value = real.fromisoformat(now)
        dt.fromisoformat = real.fromisoformat
        obs = c.collect_observation("vid1", published_at=published_at)
    return obs, analytics


def test_per_video_analytics_stored_on_the_observation():
    obs, analytics = _observe(_analytics(_report([480, 400, 20, 15, 62.5])))

    assert obs.status == ObservationStatus.MEASURED
    m = obs.cumulative
    assert m.views == 500  # Data API views are kept
    assert m.engaged_views == 400
    assert m.watch_time_seconds == 1200  # 20 min
    assert m.avg_view_duration_seconds == 15
    assert m.avg_view_percentage == 62.5
    assert obs.analytics_unavailable is None
    kwargs = analytics.reports().query.call_args.kwargs
    assert kwargs["filters"] == "video==vid1"
    assert kwargs["startDate"] == "2026-11-01"
    assert kwargs["metrics"] == (
        "views,engagedViews,estimatedMinutesWatched,averageViewDuration,averageViewPercentage"
    )


def test_age_bucket_is_the_latest_fixed_age_reached():
    cases = {
        "2026-11-02T13:00:00+00:00": None,  # not yet 24 h after 11-01T12
        "2026-11-02T12:30:00+00:00": "24h",
        "2026-11-04T13:00:00+00:00": "72h",
        "2026-11-08T13:00:00+00:00": "7d",
    }
    for now, bucket in cases.items():
        if bucket is None:
            now = "2026-11-02T11:00:00+00:00"
        obs, _ = _observe(_analytics(_report([1, 1, 1, 1, 1])), now=now)
        assert obs.age_bucket == bucket, now


def test_rejected_metric_combination_keeps_views_and_records_reason(caplog):
    caplog.set_level("WARNING", logger="gomoveshift.tracking.collector")
    obs, _ = _observe(_analytics(exc=RuntimeError("400 The query is not supported")))

    assert obs.status == ObservationStatus.MEASURED
    assert obs.cumulative.views == 500
    assert obs.cumulative.watch_time_seconds is None
    assert "not supported" in obs.analytics_unavailable
    lines = [r for r in caplog.records if "analytics_unavailable" in r.getMessage()]
    assert len(lines) == 1


def test_no_analytics_client_records_reason_and_keeps_views():
    obs, _ = _observe(None)
    assert obs.cumulative.views == 500
    assert obs.analytics_unavailable


def test_empty_report_is_unavailable_not_zero():
    obs, _ = _observe(_analytics(_report(None)))
    assert obs.cumulative.watch_time_seconds is None
    assert obs.analytics_unavailable


def test_new_fields_round_trip_through_to_dict():
    obs = MetricObservation(
        platform="youtube",
        status=ObservationStatus.MEASURED,
        post_id="vid1",
        cumulative=PlatformMetrics(views=1, engaged_views=2),
        age_bucket="72h",
        analytics_unavailable="x",
    )
    back = MetricObservation.from_dict(obs.to_dict())
    assert back.cumulative.engaged_views == 2
    assert back.age_bucket == "72h"
    assert back.analytics_unavailable == "x"


class _Scripted:
    def __init__(self, metrics):
        self._metrics = metrics

    def is_available(self):
        return True

    def collect_observation(self, video_id, published_at=None):
        return MetricObservation(
            platform="youtube",
            status=ObservationStatus.MEASURED,
            post_id=video_id,
            post_format="short",
            fetched_at="2026-11-03T00:00:00+00:00",
            cumulative=self._metrics,
            analytics_unavailable=None if self._metrics.watch_time_seconds else "boom",
        )


def test_second_pass_without_analytics_does_not_null_prior_fields():
    mc = MetricsCollector()
    short = ShortRecord(index=1, platform_ids={"youtube": "vid1"})
    from gomoveshift.tracking.schema import TrackingRecord

    record = TrackingRecord(project="p", shorts=[short])
    full = PlatformMetrics(
        views=10,
        engaged_views=8,
        watch_time_seconds=600.0,
        avg_view_duration_seconds=12.0,
        avg_view_percentage=55.0,
    )
    mc._collectors["youtube"] = _Scripted(full)
    mc.collect_all(record)
    mc._collectors["youtube"] = _Scripted(PlatformMetrics(views=20))
    mc.collect_all(record)

    kept = short.metrics["youtube"]
    assert kept.views == 20
    assert kept.engaged_views == 8
    assert kept.watch_time_seconds == 600.0
    assert kept.avg_view_duration_seconds == 12.0
    assert kept.avg_view_percentage == 55.0
