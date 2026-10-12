"""`YouTubeCollector.fetch_channel_daily_dimension`: the Analytics v2
query the channel-daily pull reuses. No network -- the Analytics client is
mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from gomoveshift.tracking.collector import YouTubeCollector


def _write_token(tmp_path):
    token_path = tmp_path / "youtube_credentials.json"
    token_path.write_text(
        json.dumps({"token": "t", "refresh_token": "r", "scopes": YouTubeCollector.SCOPES})
    )
    return token_path


def test_fetch_channel_daily_dimension_returns_day_value_views_tuples(tmp_path):
    collector = YouTubeCollector(token_path=_write_token(tmp_path))
    analytics = MagicMock()
    analytics.reports().query().execute.return_value = {
        "rows": [
            ["2026-09-20", "SHORTS", 100],
            ["2026-09-20", "YT_SEARCH", 50],
        ]
    }

    with patch("googleapiclient.discovery.build", return_value=analytics):
        with patch("google.oauth2.credentials.Credentials"):
            rows = collector.fetch_channel_daily_dimension(
                "insightTrafficSourceType", "2026-09-20", "2026-09-20"
            )

    assert rows == [("2026-09-20", "SHORTS", 100), ("2026-09-20", "YT_SEARCH", 50)]

    call_kwargs = analytics.reports().query.call_args.kwargs
    assert call_kwargs["ids"] == "channel==MINE"
    assert call_kwargs["dimensions"] == "day,insightTrafficSourceType"
    assert call_kwargs["metrics"] == "views"


def test_fetch_channel_daily_dimension_returns_empty_without_a_token(tmp_path):
    no_token = YouTubeCollector(token_path=tmp_path / "missing.json")
    assert (
        no_token.fetch_channel_daily_dimension("creatorContentType", "2026-09-20", "2026-09-20")
        == []
    )


def test_fetch_channel_daily_dimension_returns_empty_on_api_failure(tmp_path):
    collector = YouTubeCollector(token_path=_write_token(tmp_path))
    with patch("googleapiclient.discovery.build", side_effect=Exception("quota")):
        assert (
            collector.fetch_channel_daily_dimension(
                "creatorContentType", "2026-09-20", "2026-09-20"
            )
            == []
        )
