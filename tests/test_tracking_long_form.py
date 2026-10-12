"""Waypost tracks long-forms: record round-trip, duration provenance, and
unreported-is-None for the per-day / lifetime metrics a length analysis needs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gomoveshift.pipeline.content_ledger import ContentEntry
from gomoveshift.tracking.collector import MetricsCollector, YouTubeCollector
from gomoveshift.tracking.schema import (
    DailyMetricPoint,
    LongFormRecord,
    PlatformMetrics,
    TrackingRecord,
)
from gomoveshift.tracking.store import TrackingStore


def _lf_entry(date: str, vid: str | None, project: str = "ProjA", ctype: str = "longform"):
    return ContentEntry(
        date=date,
        time="10:00:00",
        content_type=ctype,
        title=f"t {date}",
        project_name=project,
        video_id=vid,
    )


def test_long_form_round_trips_through_store(tmp_path):
    store = TrackingStore(root=tmp_path)
    store.register_long_form("ProjA", "abc123", title="Drive", published_date="2026-10-11", index=2)
    rec = store.load("ProjA")
    assert rec.shorts == []
    [lf] = rec.long_forms
    assert (lf.index, lf.video_id, lf.title, lf.published_date) == (
        2,
        "abc123",
        "Drive",
        "2026-10-11",
    )
    assert lf.duration_sec is None and lf.duration_source is None
    assert lf.metrics == {}


def test_register_long_form_updates_not_duplicates(tmp_path):
    store = TrackingStore(root=tmp_path)
    store.register_long_form("ProjA", "abc123")
    store.register_long_form("ProjA", "abc123", title="later")
    [lf] = store.load("ProjA").long_forms
    assert lf.title == "later"


def test_daily_avg_view_duration_survives_round_trip():
    lf = LongFormRecord(index=1, video_id="v")
    lf.metrics["youtube"] = PlatformMetrics(
        views=10,
        avg_view_percentage=41.5,
        daily=[DailyMetricPoint(day="2026-10-11", views=4, avg_view_duration_seconds=900.0)],
    )
    rec = TrackingRecord(project="P", long_forms=[lf])
    back = TrackingRecord.from_json(rec.to_json()).long_forms[0].metrics["youtube"]
    assert back.daily[0].avg_view_duration_seconds == 900.0
    assert back.avg_view_percentage == 41.5
    # an unreported per-day metric stays None, never 0
    assert DailyMetricPoint(day="d").avg_view_duration_seconds is None
    assert PlatformMetrics().avg_view_percentage is None


def test_legacy_record_without_long_forms_loads():
    rec = TrackingRecord.from_dict({"project": "P", "shorts": []})
    assert rec.long_forms == []


def test_sync_from_ledger_registers_only_longform_with_video_id(tmp_path):
    store = TrackingStore(root=tmp_path)
    entries = [
        _lf_entry("2026-10-18", "idB"),
        _lf_entry("2026-10-11", "idA"),
        _lf_entry("2026-10-11", None),  # no confirmed id
        _lf_entry("2026-10-12", "sh1", ctype="shorts"),
        _lf_entry("2026-10-11", "idC", project="Other"),
    ]
    assert store.sync_long_forms_from_ledger(entries) == 3
    assert store.sync_long_forms_from_ledger(entries) == 0  # idempotent
    a = store.load("ProjA")
    assert [(lf.index, lf.video_id) for lf in a.long_forms] == [(1, "idA"), (2, "idB")]
    assert a.shorts == []
    assert [lf.video_id for lf in store.load("Other").long_forms] == ["idC"]


def test_backfilled_earlier_long_form_does_not_overwrite_known_records(tmp_path):
    store = TrackingStore(root=tmp_path)
    store.sync_long_forms_from_ledger(
        [_lf_entry("2026-10-11", "idA"), _lf_entry("2026-10-18", "idB")]
    )
    rec = store.load("ProjA")
    for lf in rec.long_forms:
        lf.duration_sec = {"idA": 100.0, "idB": 200.0}[lf.video_id]
        lf.metrics["youtube"] = PlatformMetrics(views=1 if lf.video_id == "idA" else 2)
    store.save(rec)
    entries = [
        _lf_entry("2026-10-11", "idA"),
        _lf_entry("2026-10-18", "idB"),
        _lf_entry("2026-10-04", "idC"),  # backfilled, sorts first
    ]
    assert store.sync_long_forms_from_ledger(entries) == 1
    assert store.sync_long_forms_from_ledger(entries) == 0  # stable, no swapping
    got = {lf.video_id: lf for lf in store.load("ProjA").long_forms}
    assert set(got) == {"idA", "idB", "idC"}
    assert len({lf.index for lf in got.values()}) == 3
    assert (got["idA"].duration_sec, got["idA"].metrics["youtube"].views) == (100.0, 1)
    assert (got["idB"].duration_sec, got["idB"].metrics["youtube"].views) == (200.0, 2)
    assert got["idC"].duration_sec is None and got["idC"].metrics == {}


def test_removed_ledger_entry_does_not_let_new_id_take_another_slot(tmp_path):
    store = TrackingStore(root=tmp_path)
    store.sync_long_forms_from_ledger(
        [_lf_entry("2026-10-11", "idA"), _lf_entry("2026-10-18", "idB")]
    )
    store.sync_long_forms_from_ledger(
        [_lf_entry("2026-10-18", "idB"), _lf_entry("2026-10-25", "idD")]
    )
    got = {lf.video_id: lf.index for lf in store.load("ProjA").long_forms}
    assert set(got) == {"idA", "idB", "idD"} and len(set(got.values())) == 3


def test_register_long_form_requires_video_id(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        TrackingStore(root=tmp_path).register_long_form("ProjA", "")


def test_zero_duration_is_none():
    c = YouTubeCollector()
    for dur in ("PT0S", "P0D", "garbage"):
        svc = MagicMock()
        svc.videos().list().execute.return_value = {
            "items": [{"statistics": {"viewCount": "1"}, "contentDetails": {"duration": dur}}]
        }
        with patch.object(c, "_get_service", return_value=svc):
            _, duration = c.fetch_video_metrics_and_duration("v")
        assert duration is None


def test_avg_view_percentage_requested_only_for_long_form():
    c = YouTubeCollector()
    shorts_analytics = MagicMock()
    shorts_analytics.reports().query().execute.return_value = {"rows": [[1, 2.0, 3.0, 0, 0]]}
    with patch.object(c, "_analytics_client", return_value=shorts_analytics):
        out = c.fetch_analytics_metrics("v")
    kw = shorts_analytics.reports().query.call_args.kwargs
    assert kw["metrics"] == "views,estimatedMinutesWatched,averageViewDuration,likes,shares"
    assert out["avg_view_percentage"] is None

    lf_analytics = MagicMock()
    lf_analytics.reports().query().execute.return_value = {"rows": [[1, 2.0, 3.0, 0, 0, 55.0]]}
    with patch.object(c, "_analytics_client", return_value=lf_analytics):
        out = c.fetch_analytics_metrics("v", include_avg_view_percentage=True)
    assert (
        lf_analytics.reports().query.call_args.kwargs["metrics"].endswith(",averageViewPercentage")
    )
    assert out["avg_view_percentage"] == 55.0


def test_collect_short_metrics_does_not_request_avg_view_percentage():
    c = YouTubeCollector()
    with (
        patch.object(c, "fetch_video_metrics", return_value=PlatformMetrics(views=1)),
        patch.object(c, "fetch_analytics_metrics", return_value={}) as fam,
    ):
        c.collect_short_metrics("v")
    fam.assert_called_once_with("v", include_avg_view_percentage=False)


def test_lone_avg_view_percentage_counts_as_measured():
    mc = MetricsCollector()
    lf = LongFormRecord(index=1, video_id="vid")
    record = TrackingRecord(project="P", long_forms=[lf])
    with (
        patch.object(mc.youtube, "is_available", return_value=True),
        patch.object(
            mc.youtube,
            "collect_long_form_metrics",
            return_value=(PlatformMetrics(avg_view_percentage=40.0), None),
        ),
    ):
        assert mc.collect_long_forms(record) == 1
    assert lf.metrics["youtube"].avg_view_percentage == 40.0


def _yt(items, rows=None, daily=None):
    svc = MagicMock()
    svc.videos().list().execute.return_value = {"items": items}
    analytics = MagicMock()
    q = analytics.reports().query
    q().execute.side_effect = [{"rows": rows or []}, {"rows": daily or []}]
    return svc, analytics


def test_collect_long_form_persists_duration_with_provenance():
    c = YouTubeCollector()
    svc, analytics = _yt(
        [{"statistics": {"viewCount": "7"}, "contentDetails": {"duration": "PT1H5M30S"}}],
        rows=[[7, 70.0, 1200.0, 1, 0, 33.3]],
        daily=[["2026-10-11", 7, 70.0, 1200.0, 1, 0]],
    )
    with (
        patch.object(c, "_get_service", return_value=svc),
        patch.object(c, "_analytics_client", return_value=analytics),
    ):
        metrics, duration = c.collect_long_form_metrics("vid", published_at="2025-10-11T10:00:00")
    assert duration == 3930.0
    assert metrics.avg_view_percentage == 33.3
    assert metrics.avg_view_duration_seconds == 1200.0
    assert metrics.daily[0].avg_view_duration_seconds == 1200.0
    assert metrics.daily[0].views == 7


def test_missing_duration_and_metrics_are_none_not_zero():
    c = YouTubeCollector()
    svc, analytics = _yt(
        [{"statistics": {"viewCount": "7"}, "contentDetails": {}}],
        rows=[[7, 70.0, 1200.0, 1, 0]],  # no averageViewPercentage column
        daily=[["2026-10-11", 7]],
    )
    keys = ["day", "views"]
    with (
        patch.object(c, "_get_service", return_value=svc),
        patch.object(c, "_analytics_client", return_value=analytics),
        patch.object(
            c,
            "fetch_analytics_daily_metrics",
            return_value=[dict(zip(keys, daily_row)) for daily_row in [["2026-10-11", 7]]],
        ),
    ):
        metrics, duration = c.collect_long_form_metrics("vid", published_at="2025-10-11T10:00:00")
    assert duration is None
    assert metrics.avg_view_percentage is None
    assert metrics.daily[0].avg_view_duration_seconds is None


def test_metrics_collector_fills_long_form_record(tmp_path):
    mc = MetricsCollector()
    lf = LongFormRecord(index=1, video_id="vid", published_date="2026-10-11")
    record = TrackingRecord(project="P", long_forms=[lf])
    fake = PlatformMetrics(views=5)
    with (
        patch.object(mc.youtube, "is_available", return_value=True),
        patch.object(mc.youtube, "collect_long_form_metrics", return_value=(fake, 3600.0)),
    ):
        assert mc.collect_long_forms(record) == 1
    assert lf.metrics["youtube"].views == 5
    assert lf.duration_sec == 3600.0
    assert lf.duration_source == "youtube_data_api"
    assert lf.collected_at is not None


def test_empty_collection_does_not_overwrite_prior_measurement():
    mc = MetricsCollector()
    lf = LongFormRecord(
        index=1, video_id="vid", duration_sec=3600.0, duration_source="youtube_data_api"
    )
    lf.metrics["youtube"] = PlatformMetrics(views=5)
    record = TrackingRecord(project="P", long_forms=[lf])
    with (
        patch.object(mc.youtube, "is_available", return_value=True),
        patch.object(
            mc.youtube, "collect_long_form_metrics", return_value=(PlatformMetrics(), None)
        ),
    ):
        assert mc.collect_long_forms(record) == 0
    assert lf.metrics["youtube"].views == 5
    assert lf.duration_sec == 3600.0


def test_shorts_path_unchanged(tmp_path):
    store = TrackingStore(root=tmp_path)
    store.register_short("ProjA", 1, file="s1.mp4", duration_sec=30.0)
    d = store.load("ProjA").to_dict()
    sh = d["shorts"][0]
    assert sh["index"] == 1 and sh["file"] == "s1.mp4"
    assert sh["content"]["duration_sec"] == 30.0
    assert sh["metrics"] == {} and sh["platform_ids"] == {}
