"""TikTok metrics collector: batches ids into one request, pages
`cursor`/`has_more`, skips upload handles, degrades gracefully. HTTP layer
is a fake session — no real network call ever leaves this test."""

from __future__ import annotations

from types import SimpleNamespace

from gomoveshift.tracking.collector import MetricsCollector, TikTokCollector


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, params=None, json=None, headers=None, timeout=None):
        self.calls.append((url, params, json, headers))
        body = self._responses.pop(0)
        return SimpleNamespace(json=lambda: body)


def _video(video_id, views=1, likes=0, comments=0, shares=0):
    return {
        "id": video_id,
        "view_count": views,
        "like_count": likes,
        "comment_count": comments,
        "share_count": shares,
    }


def _ok(videos, cursor=0, has_more=False):
    return {
        "data": {"videos": videos, "cursor": cursor, "has_more": has_more},
        "error": {"code": "ok"},
    }


# Stats shaped like a live POST response from video/query/?fields=...
_SAMPLE_PAYLOAD = _ok([_video("7000000000002", views=4210, likes=37)])


def test_missing_credentials_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(
        "gomoveshift.tracking.collector.get_valid_tiktok_credentials",
        lambda **kw: (_ for _ in ()).throw(FileNotFoundError("no creds in test")),
    )
    collector = TikTokCollector(access_token=None)
    assert collector.is_available() is False
    # Unreported, not zero: no credentials means TikTok told us nothing
    # (same contract as the other collectors).
    assert collector.collect_short_metrics("7000000000002").views is None


def test_live_payload_single_request_and_parses_stats():
    session = _FakeSession([_SAMPLE_PAYLOAD])
    collector = TikTokCollector(access_token="tok", session=session)  # noqa: S106
    metrics = collector.collect_short_metrics("7000000000002")
    assert (metrics.views, metrics.likes, metrics.comments, metrics.shares) == (4210, 37, 0, 0)
    assert session.calls[0][3]["Authorization"] == "Bearer tok"


def test_batch_sends_one_request_for_many_ids_and_skips_upload_handles():
    session = _FakeSession([_ok([_video("111", views=10), _video("222", views=20)])])
    collector = TikTokCollector(access_token="tok", session=session)  # noqa: S106
    results = collector.collect_metrics_batch(["111", "222", "v_inbox_file~v2.7000000000003"])
    assert len(session.calls) == 1
    assert sorted(session.calls[0][2]["filters"]["video_ids"]) == ["111", "222"]
    assert results["111"].views == 10 and results["222"].views == 20
    assert "v_inbox_file~v2.7000000000003" not in results


def test_collect_metrics_batch_pages_through_cursor():
    page1 = _ok([_video("111", views=1)], cursor=5, has_more=True)
    page2 = _ok([_video("222", views=2)])
    collector = TikTokCollector(access_token="tok", session=_FakeSession([page1, page2]))  # noqa: S106
    results = collector.collect_metrics_batch(["111", "222"])
    assert collector._session.calls[1][2]["cursor"] == 5
    assert set(results) == {"111", "222"}


def test_scope_error_is_caught_logged_and_returns_empty(caplog):
    error_body = {"data": {}, "error": {"code": "scope_not_authorized", "message": "nope"}}
    collector = TikTokCollector(access_token="tok", session=_FakeSession([error_body]))  # noqa: S106
    with caplog.at_level("ERROR"):
        metrics = collector.collect_short_metrics("7000000000002")
    assert metrics.views is None
    assert any("TikTok" in rec.message for rec in caplog.records)


def test_absent_metric_key_is_none_not_zero():
    # TikTok didn't return any count field for this video: unreported, not
    # a measured zero (same contract as _reported_metric and the
    # other collectors).
    session = _FakeSession([_ok([{"id": "999"}])])
    collector = TikTokCollector(access_token="tok", session=session)  # noqa: S106
    metrics = collector.collect_short_metrics("999")
    assert (metrics.views, metrics.likes, metrics.comments, metrics.shares) == (
        None,
        None,
        None,
        None,
    )


def test_reported_zero_metric_stays_zero():
    session = _FakeSession([_ok([_video("999", views=0, likes=0, comments=0, shares=0)])])
    collector = TikTokCollector(access_token="tok", session=session)  # noqa: S106
    metrics = collector.collect_short_metrics("999")
    assert (metrics.views, metrics.likes, metrics.comments, metrics.shares) == (0, 0, 0, 0)


def test_tiktok_registered_in_metrics_collector():
    mc = MetricsCollector()
    assert "tiktok" in mc._collectors
    assert isinstance(mc.tiktok, TikTokCollector)
