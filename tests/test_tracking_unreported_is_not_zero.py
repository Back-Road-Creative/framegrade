"""An absent metric key from a platform API must stay ``None`` ("unreported")
through collector.py, never get coerced to a fabricated measured 0 — the
policy `PlatformMetrics` documents on itself in schema.py.

YouTube's Data API omits `likeCount` when a creator hides likes, and
omits `commentCount` when comments are disabled — these are real, routine
response shapes, not hypotheticals.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gomoveshift.tracking.collector import InstagramCollector, YouTubeCollector


class _FakeResponse:
    """Local fakes rather than an import from the Instagram collector's own
    test module: a test file that imports another test file's helpers breaks
    the moment either is reorganised, and these are four lines."""

    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self, response: "_FakeResponse"):
        self._response = response
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self._response


def _fake_youtube(statistics: dict):
    fake = MagicMock()
    fake.videos().list().execute.return_value = {"items": [{"statistics": statistics}]}
    return fake


def test_hidden_likes_stay_none_not_zero():
    """likeCount absent (creator hid likes) must not become a measured 0."""
    collector = YouTubeCollector()
    with patch.object(collector, "_get_service", return_value=_fake_youtube({"viewCount": "500"})):
        metrics = collector.fetch_video_metrics("vid1")

    assert metrics.likes is None


def test_disabled_comments_stay_none_not_zero():
    """commentCount absent (comments disabled) must not become a measured 0."""
    collector = YouTubeCollector()
    with patch.object(collector, "_get_service", return_value=_fake_youtube({"viewCount": "500"})):
        metrics = collector.fetch_video_metrics("vid1")

    assert metrics.comments is None


def test_genuine_zero_view_count_is_preserved_as_zero():
    """A platform-reported "0" must stay 0, not be flattened into None."""
    collector = YouTubeCollector()
    with patch.object(
        collector,
        "_get_service",
        return_value=_fake_youtube({"viewCount": "0", "likeCount": "0", "commentCount": "0"}),
    ):
        metrics = collector.fetch_video_metrics("vid1")

    assert metrics.views == 0
    assert metrics.likes == 0
    assert metrics.comments == 0


def test_analytics_row_missing_watch_time_does_not_overwrite_with_zero(tmp_path):
    """An analytics dict lacking `watch_time_minutes` must leave the
    unreported (None) default alone rather than fabricating 0."""
    token_path = tmp_path / "youtube_credentials.json"
    token_path.write_text(
        '{"token": "t", "refresh_token": "r", "scopes": ["x"]}',
    )
    collector = YouTubeCollector(token_path=token_path)

    with patch.object(collector, "_get_service", return_value=_fake_youtube({"viewCount": "500"})):
        with patch.object(
            collector,
            "fetch_analytics_metrics",
            return_value={"avg_view_duration_seconds": 30.0},  # no watch_time_minutes key
        ):
            metrics = collector.collect_short_metrics("vid1")

    assert metrics.watch_time_seconds is None
    assert metrics.avg_view_duration_seconds == 30.0


def test_an_instagram_scope_denial_is_unknown_not_a_flop():
    """Graph returning "(#10) permission denied, missing scope" means it
    would not say. Every metric therefore stays None: a 0 here is the same
    shape a genuinely unwatched Reel produces, so an ops problem would be
    indistinguishable from a content problem on every surface downstream."""
    creds = {"ig_user_id": "ig_1", "page_access_token": "tok", "graph_version": "v21.0"}
    body = {"error": {"message": "(#10) permission denied, missing scope", "code": 10}}
    collector = InstagramCollector(
        credentials=creds, session=_FakeSession(_FakeResponse(403, body))
    )

    metrics = collector.collect_short_metrics("media_123")

    assert (metrics.views, metrics.likes, metrics.comments) == (None, None, None)
    assert (metrics.shares, metrics.impressions) == (None, None)


def test_an_instagram_reel_with_real_zeros_keeps_them():
    """The other half of the contract: a metric Graph DOES report as 0 is a
    measurement and survives as 0, not folded into the unknown."""
    creds = {"ig_user_id": "ig_1", "page_access_token": "tok", "graph_version": "v21.0"}
    body = {
        "data": [
            {"name": "views", "values": [{"value": 0}]},
            {"name": "likes", "values": [{"value": 0}]},
        ]
    }
    collector = InstagramCollector(
        credentials=creds, session=_FakeSession(_FakeResponse(200, body))
    )

    metrics = collector.collect_short_metrics("media_123")

    assert metrics.views == 0
    assert metrics.likes == 0
    assert metrics.shares is None  # never reported at all — still unknown
