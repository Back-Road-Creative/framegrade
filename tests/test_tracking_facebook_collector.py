"""Facebook video metrics collector. Mirrors the Instagram collector:
missing creds never raise, a Graph error is caught+logged, video metric names stay on the video edge across
Graph versions, and the live token's missing
``read_insights`` logs the scope NAME once, requests nothing, returns empty.
No network call ever leaves this test — the HTTP layer is a fake session."""

from __future__ import annotations

import pytest

from gomoveshift.tracking.collector import FacebookCollector, MetricsCollector

TOKEN = "page-token-secret"  # noqa: S105 — fixture value, asserted never logged
SCOPES = ["pages_read_engagement", "read_insights"]
#: Video and Post insight names are not API-version aliases. Retired
#: impressions must not poison the whole lookup (Meta's June 2026 removal).
VIDEO_METRICS = "total_video_views,total_video_view_total_time,total_video_avg_time_watched"
METRICS_BY_VERSION = dict.fromkeys(("v20.0", "v21.0", "v25.0", "v26.0"), VIDEO_METRICS)


class _FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self, *responses: _FakeResponse):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


def _creds(graph_version="v20.0", scopes=None):
    return {
        "page_id": "page_1",
        "page_access_token": TOKEN,
        "graph_version": graph_version,
        "scopes": SCOPES if scopes is None else scopes,
    }


def _insights_body(names: dict) -> dict:
    return {"data": [{"name": k, "values": [{"value": v}]} for k, v in names.items()]}


def test_missing_credentials_degrades_gracefully(monkeypatch):
    # credentials=None takes the real lazy-load path — patched out so this test
    # never touches SOPS/disk.
    monkeypatch.setattr(
        "gomoveshift.tracking.collector.load_meta_credentials",
        lambda **kw: (_ for _ in ()).throw(FileNotFoundError("no creds in test")),
    )
    collector = FacebookCollector(credentials=None)
    assert collector.is_available() is False
    # Nothing was collected, so nothing was reported: None, not a measured 0.
    # A 0 here would claim the video had no views.
    assert collector.collect_short_metrics("video_1").views is None
    assert collector.fetch_audience_online() == {}


@pytest.mark.parametrize("graph_version", list(METRICS_BY_VERSION))
def test_collect_short_metrics_uses_video_edge_contract_across_versions(graph_version):
    names = METRICS_BY_VERSION[graph_version].split(",")
    session = _FakeSession(_FakeResponse(200, _insights_body(dict(zip(names, (900, 45000, 5000))))))
    collector = FacebookCollector(credentials=_creds(graph_version), session=session)

    result = collector.collect_short_metrics("video_1")

    assert (result.views, result.impressions) == (900, None)
    # Graph reports video watch times in milliseconds.
    assert (result.watch_time_seconds, result.avg_view_duration_seconds) == (45.0, 5.0)
    url, params = session.calls[0]
    assert "video_1/video_insights" in url and graph_version in url
    assert params["metric"] == ",".join(names)


def test_missing_read_insights_scope_logs_once_and_returns_empty(caplog):
    session = _FakeSession(_FakeResponse(200, _insights_body({"total_video_views": 5})))
    creds = _creds(scopes=["pages_read_engagement", "pages_show_list"])
    collector = FacebookCollector(credentials=creds, session=session)
    with caplog.at_level("WARNING"):
        assert collector.collect_short_metrics("video_1").views is None
        assert collector.fetch_audience_online() == {}
        assert collector.collect_short_metrics("video_2").views is None

    assert session.calls == []  # no request attempted, no retry loop
    scope_logs = [r for r in caplog.records if "read_insights" in r.message]
    assert len(scope_logs) == 1
    assert TOKEN not in caplog.text


def test_graph_error_is_caught_logged_and_returns_empty(caplog):
    error_body = {"error": {"message": "(#10) permission denied, missing scope", "code": 10}}
    collector = FacebookCollector(
        credentials=_creds(), session=_FakeSession(_FakeResponse(403, error_body))
    )
    with caplog.at_level("ERROR"):
        result = collector.collect_short_metrics("video_1")

    # The Graph error means the platform told us nothing: unreported, not a
    # measured 0 (same contract as the other collectors and the Instagram
    # collector).
    assert result.views is None
    assert any("Facebook" in rec.message for rec in caplog.records)
    assert TOKEN not in caplog.text


def test_absent_metric_key_is_none_not_zero():
    # Graph named no metrics for this video: unreported, not a measured zero.
    session = _FakeSession(_FakeResponse(200, _insights_body({})))
    collector = FacebookCollector(credentials=_creds(), session=session)

    result = collector.collect_short_metrics("video_1")

    assert (
        result.views,
        result.impressions,
        result.watch_time_seconds,
        result.avg_view_duration_seconds,
    ) == (None, None, None, None)


def test_reported_zero_metric_stays_zero():
    names = METRICS_BY_VERSION["v20.0"].split(",")
    session = _FakeSession(_FakeResponse(200, _insights_body(dict(zip(names, (0, 0, 0))))))
    collector = FacebookCollector(credentials=_creds(), session=session)

    result = collector.collect_short_metrics("video_1")

    assert (
        result.views,
        result.impressions,
        result.watch_time_seconds,
        result.avg_view_duration_seconds,
    ) == (0, None, 0.0, 0.0)


def test_fetch_audience_online_follows_paging_and_sums_hours():
    page_one = {
        "data": [{"name": "page_fans_online", "values": [{"value": {"0": 10, "9": 100}}]}],
        "paging": {"next": "https://graph.facebook.com/v20.0/page_1/insights/page_fans_online?p=2"},
    }
    page_two = {"data": [{"name": "page_fans_online", "values": [{"value": {"9": 250, "20": 40}}]}]}
    session = _FakeSession(_FakeResponse(200, page_one), _FakeResponse(200, page_two))
    collector = FacebookCollector(credentials=_creds(), session=session)

    histogram = collector.fetch_audience_online()
    assert histogram == {0: 10, 9: 250, 20: 40}  # latest per hour wins
    assert len(session.calls) == 2
    assert "page_1/insights/page_fans_online" in session.calls[0][0]


def test_facebook_registered_in_metrics_collector():
    mc = MetricsCollector()
    assert "facebook" in mc._collectors
    assert isinstance(mc.facebook, FacebookCollector)


def test_mixed_platform_run_reports_facebook_failure_separately(caplog, monkeypatch):
    """A successful sibling must not make a failed Facebook read look healthy."""
    from gomoveshift.tracking.observation import MetricObservation, ObservationStatus
    from gomoveshift.tracking.schema import PlatformMetrics, ShortRecord, TrackingRecord

    short = ShortRecord(index=1, platform_ids={"youtube": "yt_1", "facebook": "video_1"})
    mc = MetricsCollector()
    monkeypatch.setattr(mc, "_collect_settled_series", lambda *args: None)
    monkeypatch.setattr(
        mc,
        "collect_observations_for_short",
        lambda short: {
            "youtube": MetricObservation(
                platform="youtube",
                post_id="yt_1",
                status=ObservationStatus.MEASURED,
                cumulative=PlatformMetrics(views=0),
            ),
            "facebook": MetricObservation(
                platform="facebook",
                post_id="video_1",
                status=ObservationStatus.PROVIDER_ERROR,
                detail="invalid insights metric",
            ),
        },
    )
    with caplog.at_level("INFO"):
        count = mc.collect_all(TrackingRecord(project="fixture", shorts=[short]))
    assert count == 1  # A real zero is a measurement, but only on YouTube.
    assert "metrics health platform=facebook measured=0 attempted=1 unanswered=1" in caplog.text
    assert "metrics health platform=youtube measured=1 attempted=1 unanswered=0" in caplog.text
    assert "facebook" not in short.metrics


def test_missing_observation_is_counted_as_unanswered(caplog, monkeypatch):
    from gomoveshift.tracking.schema import ShortRecord, TrackingRecord

    mc = MetricsCollector()
    monkeypatch.setattr(mc, "_collect_settled_series", lambda *args: None)
    monkeypatch.setattr(mc, "collect_observations_for_short", lambda short: {})
    record = TrackingRecord(
        project="fixture", shorts=[ShortRecord(index=1, platform_ids={"facebook": "video_1"})]
    )
    with caplog.at_level("INFO"):
        assert mc.collect_all(record) == 0
    assert "metrics health platform=facebook measured=0 attempted=1 unanswered=1" in caplog.text


REEL_METRICS = (
    "blue_reels_play_count,fb_reels_replay_count,fb_reels_total_plays,"
    "post_video_view_time,post_video_avg_time_watched"
)


@pytest.mark.parametrize("graph_version", ["v21.0", "v26.0"])
def test_reel_metrics_are_recovered_without_relabeling_plays_as_views(graph_version):
    from gomoveshift.tracking.observation import ObservationStatus, compare_views

    session = _FakeSession(
        _FakeResponse(200, {"data": []}),
        _FakeResponse(
            200, _insights_body(dict(zip(REEL_METRICS.split(","), (100, 25, 125, 45000, 5000))))
        ),
    )
    obs = FacebookCollector(credentials=_creds(graph_version), session=session).collect_observation(
        "video_1"
    )
    assert obs.status == ObservationStatus.MEASURED
    assert obs.post_format == "reel"
    metrics = obs.cumulative
    assert (metrics.reel_initial_plays, metrics.reel_replays, metrics.reel_total_plays) == (
        100,
        25,
        125,
    )
    assert (metrics.views, metrics.impressions) == (None, None)
    assert (metrics.watch_time_seconds, metrics.avg_view_duration_seconds) == (45.0, 5.0)
    assert compare_views([obs]).blended_views is None
    assert len(session.calls) == 2
    assert all(url.endswith("/video_1/video_insights") for url, _ in session.calls)
    assert [params["metric"] for _, params in session.calls] == [VIDEO_METRICS, REEL_METRICS]
    assert all(params["period"] == "lifetime" for _, params in session.calls)


def test_reel_only_zero_is_measured_and_survives_record_round_trip():
    from gomoveshift.tracking.observation import MetricObservation, ObservationStatus
    from gomoveshift.tracking.schema import ShortRecord, TrackingRecord
    from gomoveshift.tracking.store import append_observation

    session = _FakeSession(
        _FakeResponse(200, {"data": []}),
        _FakeResponse(200, _insights_body({"fb_reels_total_plays": 0})),
    )
    obs = FacebookCollector(credentials=_creds(), session=session).collect_observation("video_1")
    assert obs.status == ObservationStatus.MEASURED
    assert obs.cumulative.reel_total_plays == 0
    assert obs.cumulative.reel_initial_plays is None
    assert obs.cumulative.reel_replays is None
    assert obs.cumulative.views is None
    record = TrackingRecord(project="fixture", shorts=[ShortRecord(index=1)])
    append_observation(record, 1, obs)
    restored = TrackingRecord.from_dict(record.to_dict())
    round_trip = MetricObservation.from_dict(restored.shorts[0].observation_log["facebook"][0])
    assert round_trip.post_format == "reel"
    assert round_trip.cumulative.reel_total_plays == 0
    assert round_trip.cumulative.reel_initial_plays is None
    assert round_trip.cumulative.views is None


def test_generic_video_measurement_does_not_probe_reel_metrics():
    session = _FakeSession(_FakeResponse(200, _insights_body({"total_video_views": 0})))
    obs = FacebookCollector(credentials=_creds(), session=session).collect_observation("video_1")
    assert obs.post_format == "video"
    assert obs.cumulative.views == 0
    assert obs.cumulative.reel_total_plays is None
    assert len(session.calls) == 1


@pytest.mark.parametrize("http_status,code", [(400, 100), (403, 200), (429, 4), (500, 1)])
def test_failed_video_request_never_fans_out_to_reel_request(http_status, code):
    session = _FakeSession(
        _FakeResponse(http_status, {"error": {"code": code, "message": "provider refused"}})
    )
    obs = FacebookCollector(credentials=_creds(), session=session).collect_observation("video_1")
    assert not obs.is_measurement
    assert len(session.calls) == 1


def test_empty_video_and_reel_responses_are_unknown_and_bounded():
    from gomoveshift.tracking.observation import ObservationStatus

    session = _FakeSession(_FakeResponse(200, {"data": []}))
    obs = FacebookCollector(credentials=_creds(), session=session).collect_observation("video_1")
    assert obs.status == ObservationStatus.NOT_RETURNED
    assert obs.post_format == ""
    assert obs.cumulative.reel_total_plays is None
    assert len(session.calls) == 2


def test_failed_reel_lookup_records_failure_and_next_video_resets_format():
    from gomoveshift.tracking.observation import ObservationStatus

    session = _FakeSession(
        _FakeResponse(200, {"data": []}),
        _FakeResponse(403, {"error": {"code": 200, "message": "permission denied"}}),
        _FakeResponse(200, _insights_body({"total_video_views": 5})),
    )
    collector = FacebookCollector(credentials=_creds(), session=session)
    first = collector.collect_observation("video_1")
    second = collector.collect_observation("video_2")
    assert first.status == ObservationStatus.PERMISSION_DENIED
    assert second.status == ObservationStatus.MEASURED
    assert second.post_format == "video"
    assert second.cumulative.views == 5
    assert len(session.calls) == 3


def test_collect_all_persists_play_only_measurement(tmp_path, monkeypatch):
    from gomoveshift.tracking.schema import PlatformMetrics, ShortRecord, TrackingRecord
    from gomoveshift.tracking.store import TrackingStore

    session = _FakeSession(
        _FakeResponse(200, {"data": []}),
        _FakeResponse(200, _insights_body({"fb_reels_total_plays": 0})),
    )
    mc = MetricsCollector()
    mc._collectors["facebook"] = FacebookCollector(credentials=_creds(), session=session)
    monkeypatch.setattr(mc, "_collect_settled_series", lambda *args: None)
    record = TrackingRecord(
        project="fixture", shorts=[ShortRecord(index=1, platform_ids={"facebook": "video_1"})]
    )
    assert mc.collect_all(record) == 1
    store = TrackingStore(root=tmp_path)
    store.save(record)
    restored = store.load("fixture")
    assert restored.shorts[0].metrics["facebook"].reel_total_plays == 0
    assert restored.shorts[0].metrics["facebook"].views is None
    assert restored.shorts[0].observation_log["facebook"][0]["post_format"] == "reel"
    legacy = PlatformMetrics.from_dict({"views": 10})
    assert legacy.views == 10
    assert legacy.reel_initial_plays is None
    assert legacy.reel_replays is None
    assert legacy.reel_total_plays is None
