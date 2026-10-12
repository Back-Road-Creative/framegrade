"""Facebook video metrics collector. Mirrors the Instagram collector:
missing creds never raise, a Graph error is caught+logged, metric names follow
the credential's own ``graph_version``, and the live token's missing
``read_insights`` logs the scope NAME once, requests nothing, returns empty.
No network call ever leaves this test — the HTTP layer is a fake session."""

from __future__ import annotations

import pytest

from gomoveshift.tracking.collector import FacebookCollector, MetricsCollector

TOKEN = "page-token-secret"  # noqa: S105 — fixture value, asserted never logged
SCOPES = ["pages_read_engagement", "read_insights"]
#: Canonical metric order: views, impressions, total watch time, avg watch time.
METRICS_BY_VERSION = {
    "v20.0": "total_video_views,total_video_impressions,"
    "total_video_view_total_time,total_video_avg_time_watched",
    "v21.0": "post_video_views,post_impressions,post_video_view_time,post_video_avg_time_watched",
}


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
def test_collect_short_metrics_uses_version_correct_metric_names(graph_version):
    names = METRICS_BY_VERSION[graph_version].split(",")
    session = _FakeSession(
        _FakeResponse(200, _insights_body(dict(zip(names, (900, 1200, 45000, 5000)))))
    )
    collector = FacebookCollector(credentials=_creds(graph_version), session=session)

    result = collector.collect_short_metrics("video_1")

    assert (result.views, result.impressions) == (900, 1200)
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
    session = _FakeSession(_FakeResponse(200, _insights_body(dict(zip(names, (0, 0, 0, 0))))))
    collector = FacebookCollector(credentials=_creds(), session=session)

    result = collector.collect_short_metrics("video_1")

    assert (
        result.views,
        result.impressions,
        result.watch_time_seconds,
        result.avg_view_duration_seconds,
    ) == (0, 0, 0.0, 0.0)


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
