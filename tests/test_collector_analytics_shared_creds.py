"""The Analytics client resolves credentials like the Data API client: an
explicit token file wins, else the shared SOPS store, else None."""

from __future__ import annotations

import json
import logging
from googleapiclient.errors import HttpError
from unittest.mock import MagicMock, patch

from gomoveshift.tracking import collector as col
from gomoveshift.tracking.collector import YouTubeCollector

_SHARED = {
    "token": "t",
    "refresh_token": "r",
    "token_uri": "https://oauth2.googleapis.com/token",
    "client_id": "shared-client",
    "client_secret": "s",
    "scopes": ["https://www.googleapis.com/auth/yt-analytics.readonly"],
}


def test_shared_creds_used_when_token_file_absent(tmp_path):
    c = YouTubeCollector(token_path=tmp_path / "missing.json")
    with (
        patch.object(col, "_load_shared_youtube_credentials", return_value=_SHARED) as load,
        patch("googleapiclient.discovery.build") as build,
    ):
        assert c._analytics_client() is build.return_value
        c._analytics_client()
    assert load.call_count == 1
    args, kwargs = build.call_args_list[0]
    assert args == ("youtubeAnalytics", "v2")
    creds = kwargs["credentials"]
    assert creds.client_id == "shared-client"
    assert list(creds.scopes) == _SHARED["scopes"]


def test_no_creds_returns_none(tmp_path, caplog):
    c = YouTubeCollector(token_path=tmp_path / "missing.json")
    with (
        patch.object(col, "_load_shared_youtube_credentials", return_value={}),
        patch("googleapiclient.discovery.build") as build,
        caplog.at_level(logging.WARNING, logger=col.logger.name),
    ):
        assert c._analytics_client() is None
    build.assert_not_called()
    assert [r.getMessage() for r in caplog.records].count("No token for Analytics API") == 0
    assert c.analytics_status().status == "no_token"


def test_shared_loader_raising_returns_none(tmp_path):
    c = YouTubeCollector(token_path=tmp_path / "missing.json")
    with patch.object(col, "_load_shared_youtube_credentials", side_effect=RuntimeError("x")):
        assert c._analytics_client() is None


def test_token_file_wins_over_shared(tmp_path):
    f = tmp_path / "tok.json"
    f.write_text(json.dumps({**_SHARED, "client_id": "file-client"}))
    c = YouTubeCollector(token_path=f)
    with (
        patch.object(col, "_load_shared_youtube_credentials", return_value=_SHARED) as load,
        patch("googleapiclient.discovery.build", MagicMock()) as build,
    ):
        c._analytics_client()
    load.assert_not_called()
    assert build.call_args.kwargs["credentials"].client_id == "file-client"


def _lines(caplog, prefix):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


def test_scope_missing_from_scopes_list_one_line_no_network(tmp_path, caplog):
    creds = {**_SHARED, "scopes": ["https://www.googleapis.com/auth/youtube.readonly"]}
    c = YouTubeCollector(token_path=tmp_path / "missing.json")
    with (
        patch.object(col, "_load_shared_youtube_credentials", return_value=creds),
        patch("googleapiclient.discovery.build") as build,
        caplog.at_level(logging.WARNING, logger=col.logger.name),
    ):
        assert c.analytics_status().status == "scope_missing"
        assert c.fetch_analytics_metrics("v1") == {}
        assert c.fetch_analytics_metrics("v2") == {}
    build.assert_not_called()
    assert len(_lines(caplog, "ANALYTICS_SCOPE_MISSING scope=yt-analytics.readonly")) == 1


def test_no_creds_status_and_single_line(tmp_path, caplog):
    c = YouTubeCollector(token_path=tmp_path / "missing.json")
    with (
        patch.object(col, "_load_shared_youtube_credentials", return_value={}),
        caplog.at_level(logging.WARNING, logger=col.logger.name),
    ):
        assert c.fetch_analytics_daily_metrics("v", "2026-01-01") == []
        assert c.fetch_channel_daily_dimension("d", "2026-01-01", "2026-01-02") == []
    assert c.analytics_status().status == "no_token"
    assert len(_lines(caplog, "ANALYTICS_NO_TOKEN")) == 1


def _http403():
    resp = MagicMock(status=403, reason="Forbidden")
    return HttpError(resp, b'{"error":{"errors":[{"reason":"insufficientPermissions"}]}}')


def test_fetch_403_insufficient_permissions_caches_scope_missing(tmp_path, caplog):
    c = YouTubeCollector(token_path=tmp_path / "missing.json")
    with (
        patch.object(col, "_load_shared_youtube_credentials", return_value=_SHARED),
        patch("googleapiclient.discovery.build") as build,
        caplog.at_level(logging.WARNING, logger=col.logger.name),
    ):
        build.return_value.reports.return_value.query.return_value.execute.side_effect = _http403()
        assert c.analytics_status().status == "ok"
        assert c.fetch_analytics_metrics("v1") == {}
        assert c.fetch_analytics_metrics("v2") == {}
    assert c.analytics_status().status == "scope_missing"
    assert len(_lines(caplog, "ANALYTICS_SCOPE_MISSING")) == 1


def test_ok_status(tmp_path):
    c = YouTubeCollector(token_path=tmp_path / "missing.json")
    with patch.object(col, "_load_shared_youtube_credentials", return_value=_SHARED):
        st = c.analytics_status()
    assert st.status == "ok"
    assert st.line() == "ANALYTICS_OK"


def test_probe_other_failure_is_error(tmp_path):
    c = YouTubeCollector(token_path=tmp_path / "missing.json")
    with (
        patch.object(col, "_load_shared_youtube_credentials", return_value=_SHARED),
        patch("googleapiclient.discovery.build") as build,
    ):
        build.return_value.reports.return_value.query.return_value.execute.side_effect = (
            RuntimeError("boom")
        )
        st = c.probe_analytics()
    assert st.status == "error"
    assert st.line() == "ANALYTICS_UNAVAILABLE reason=boom"
