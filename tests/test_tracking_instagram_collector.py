"""Instagram Reels metrics collector. Missing credentials never raise,
a scope error is caught+logged, and the metric name follows the credential's
own graph_version (IG renamed plays->views at v20.0). No network call ever
leaves this test — the Graph HTTP layer is a fake session object."""

from __future__ import annotations

import pytest

from gomoveshift.tracking.collector import InstagramCollector, MetricsCollector


class _FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self._response


def _insights_body(play_metric: str) -> dict:
    values = {play_metric: 1000, "reach": 800, "likes": 50, "comments": 5, "shares": 3}
    return {"data": [{"name": k, "values": [{"value": v}]} for k, v in values.items()]}


def test_missing_credentials_degrades_gracefully(monkeypatch):
    # credentials=None triggers the real lazy-load path — patch it out so
    # this test never touches SOPS/disk, only exercises the None-result branch.
    monkeypatch.setattr(
        "gomoveshift.tracking.collector.load_meta_credentials",
        lambda **kw: (_ for _ in ()).throw(FileNotFoundError("no creds in test")),
    )
    collector = InstagramCollector(credentials=None)
    assert collector.is_available() is False
    # Missing credentials means "unreported", not "measured 0 views".
    assert collector.collect_short_metrics("media_123").views is None
    assert collector.fetch_audience_online() == {}


@pytest.mark.parametrize("graph_version,play_metric", [("v21.0", "views"), ("v17.0", "plays")])
def test_collect_short_metrics_uses_version_correct_play_metric(graph_version, play_metric):
    creds = {"ig_user_id": "ig_1", "page_access_token": "tok", "graph_version": graph_version}
    session = _FakeSession(_FakeResponse(200, _insights_body(play_metric)))
    collector = InstagramCollector(credentials=creds, session=session)

    metrics = collector.collect_short_metrics("media_123")

    got = (metrics.views, metrics.impressions, metrics.likes, metrics.comments, metrics.shares)
    assert got == (1000, 800, 50, 5, 3)
    url, params = session.calls[0]
    assert play_metric in params["metric"] and "media_123/insights" in url


def test_scope_error_is_caught_logged_and_returns_empty(caplog):
    creds = {"ig_user_id": "ig_1", "page_access_token": "tok", "graph_version": "v21.0"}
    error_body = {"error": {"message": "(#10) permission denied, missing scope", "code": 10}}
    session = _FakeSession(_FakeResponse(403, error_body))
    collector = InstagramCollector(credentials=creds, session=session)

    with caplog.at_level("ERROR"):
        metrics = collector.collect_short_metrics("media_123")

    # Was `== 0` when this collector landed. A 403 scope denial is
    # Graph refusing to answer, and the test's own name calls that "empty" --
    # which `PlatformMetrics` spells `None`, per the Optional contract in
    # schema.py. Recording 0 made a permissions problem read downstream as a
    # Reel nobody watched; `decline.py` and `report.py` consume these as
    # measurements.
    assert metrics.views is None
    assert metrics.likes is None
    assert metrics.impressions is None
    assert any("Instagram" in rec.message for rec in caplog.records)


def test_fetch_audience_online_returns_hour_histogram():
    creds = {"ig_user_id": "ig_1", "page_access_token": "tok", "graph_version": "v21.0"}
    body = {"data": [{"name": "online_followers", "values": [{"value": {"0": 12, "9": 340}}]}]}
    session = _FakeSession(_FakeResponse(200, body))
    collector = InstagramCollector(credentials=creds, session=session)

    histogram = collector.fetch_audience_online()

    assert histogram == {0: 12, 9: 340}
    url, params = session.calls[0]
    assert "ig_1/insights" in url and params["metric"] == "online_followers"


def test_instagram_registered_in_metrics_collector():
    mc = MetricsCollector()
    assert "instagram" in mc._collectors
    assert isinstance(mc.instagram, InstagramCollector)
