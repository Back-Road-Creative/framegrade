"""Facebook Page-level daily series (plan U1): one Graph request per metric
per window, the Meta (Pacific) day taken from ``end_time``, rows upserted into
the brand's ``channel_daily`` store under a ``facebook:`` dimension, and a
self-backfill from 2026-06-01 that resumes from the last stored day.

A metric Graph refuses is reported unavailable -- never stored as zero -- and
never stops the other metrics. No network: the HTTP layer is a fake session."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.cli.handlers import metrics as metrics_handler
from gomoveshift.tracking import channel_daily
from gomoveshift.tracking.channel_daily import (
    FACEBOOK_BACKFILL_START,
    FACEBOOK_PAGE_METRICS,
    facebook_page_series,
    load_rows,
    pull_facebook_page_daily,
    upsert_rows,
)
from gomoveshift.tracking.collector import FacebookCollector
from gomoveshift.tracking.suppression import classify_youtube_shorts_feed

TOKEN = "page-token-secret"  # noqa: S105 - fixture value, asserted never logged


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


def _creds(scopes=("pages_read_engagement", "read_insights")):
    return {
        "page_id": "page_1",
        "page_access_token": TOKEN,
        "graph_version": "v21.0",
        "scopes": list(scopes),
    }


def _day_body(metric: str, points: dict, paging: dict | None = None) -> dict:
    body = {
        "data": [
            {
                "name": metric,
                "period": "day",
                "values": [{"value": v, "end_time": t} for t, v in points.items()],
            }
        ]
    }
    if paging:
        body["paging"] = paging
    return body


# -- collector: one window ---------------------------------------------------


def test_fetch_page_daily_labels_each_value_with_its_pacific_day():
    session = _FakeSession(
        _FakeResponse(
            200,
            _day_body(
                "page_media_view",
                {
                    # Midnight Pacific in daylight time (UTC-7) closes Oct 20.
                    "2026-10-21T07:00:00+0000": 410,
                    # Midnight Pacific in standard time (UTC-8) closes Nov 2.
                    "2026-11-03T08:00:00+0000": 37,
                },
            ),
        )
    )
    collector = FacebookCollector(credentials=_creds(), session=session)

    points = collector.fetch_page_daily("page_media_view", "2026-10-20", "2026-11-03")

    assert points == [("2026-10-20", 410), ("2026-11-02", 37)]
    url, params = session.calls[0]
    assert url.endswith("/v21.0/page_1/insights")
    assert (params["metric"], params["period"]) == ("page_media_view", "day")
    assert (params["since"], params["until"]) == ("2026-10-20", "2026-11-03")


def test_fetch_page_daily_never_follows_paging_into_another_window():
    """A time-ranged insights answer pages to the NEXT date window; following
    it would read days nobody asked for."""
    nxt = {"next": "https://graph.facebook.com/v21.0/page_1/insights?since=later"}
    session = _FakeSession(
        _FakeResponse(200, _day_body("page_media_view", {"2026-10-21T07:00:00+0000": 5}, nxt))
    )
    collector = FacebookCollector(credentials=_creds(), session=session)

    assert collector.fetch_page_daily("page_media_view", "2026-10-20", "2026-10-21") == [
        ("2026-10-20", 5)
    ]
    assert len(session.calls) == 1


def test_fetch_page_daily_skips_non_numeric_values_and_other_metrics():
    body = _day_body("page_media_view", {"2026-10-21T07:00:00+0000": {"true": 3}})
    body["data"].append(
        {
            "name": "page_media_view",
            "period": "week",
            "values": [{"value": 99, "end_time": "2026-10-21T07:00:00+0000"}],
        }
    )
    collector = FacebookCollector(
        credentials=_creds(), session=_FakeSession(_FakeResponse(200, body))
    )

    assert collector.fetch_page_daily("page_media_view", "2026-10-20", "2026-10-21") == []


def test_fetch_page_daily_is_none_on_refusal_or_missing_scope(caplog):
    refused = _FakeSession(
        _FakeResponse(400, {"error": {"code": 100, "message": "(#100) invalid metric"}})
    )
    collector = FacebookCollector(credentials=_creds(), session=refused)
    assert collector.fetch_page_daily("page_video_views", "2026-10-20", "2026-10-21") is None
    assert "invalid metric" in collector.last_failure_detail

    no_scope = _FakeSession(_FakeResponse(200, {"data": []}))
    unscoped = FacebookCollector(credentials=_creds(scopes=["pages_show_list"]), session=no_scope)
    with caplog.at_level("WARNING"):
        assert unscoped.fetch_page_daily("page_media_view", "2026-10-20", "2026-10-21") is None
    assert no_scope.calls == []
    assert TOKEN not in caplog.text


# -- channel_daily: windows, storage, resume ----------------------------------


class _ScriptedCollector:
    """Answers `fetch_page_daily` from a per-metric script; records calls."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.calls = []
        self.last_failure_detail = "(#100) The value must be a valid insights metric"

    def fetch_page_daily(self, metric, since, until):
        self.calls.append((metric, since, until))
        answer = self.answers.get(metric)
        if answer is None:
            return None
        return [(d, v) for d, v in answer if since <= d < until]


def test_first_pull_backfills_from_june_in_bounded_windows(tmp_path):
    days = [(f"2026-{m:02d}-{d:02d}", 7) for m, d in ((6, 1), (8, 29), (8, 30), (10, 29))]
    collector = _ScriptedCollector({"page_media_view": days})

    status = pull_facebook_page_daily(collector, "gomoveshift", date(2026, 10, 30), root=tmp_path)

    windows = [(s, u) for m, s, u in collector.calls if m == "page_media_view"]
    assert windows[0][0] == FACEBOOK_BACKFILL_START == "2026-06-01"
    assert windows[-1][1] == "2026-10-30"  # until is exclusive: yesterday is the last day
    for since, until in windows:
        span = date.fromisoformat(until) - date.fromisoformat(since)
        assert span.days <= 90
    # Consecutive windows tile the range with no gap and no overlap.
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))

    assert status["page_media_view"] == 4
    rows = load_rows("gomoveshift", root=tmp_path)
    assert {r["dimension"] for r in rows} == {"facebook:page_media_view"}
    assert all(r["value"] == "page" for r in rows)
    assert facebook_page_series(rows, "page_media_view") == dict(days)


def test_a_refused_metric_is_unavailable_and_never_blocks_the_others(tmp_path, caplog):
    collector = _ScriptedCollector({"page_media_view": [("2026-10-29", 12)]})

    with caplog.at_level("WARNING"):
        status = pull_facebook_page_daily(
            collector, "gomoveshift", date(2026, 10, 30), root=tmp_path
        )

    assert status["page_media_view"] == 1
    refused = [m for m in FACEBOOK_PAGE_METRICS if m != "page_media_view"]
    assert all(status[m] is None for m in refused)
    # A refused metric stops after its first window instead of retrying each one.
    for metric in refused:
        assert [c for c in collector.calls if c[0] == metric] == [
            (metric, "2026-06-01", "2026-08-30")
        ]
    assert "valid insights metric" in caplog.text
    rows = load_rows("gomoveshift", root=tmp_path)
    assert {r["dimension"] for r in rows} == {"facebook:page_media_view"}


def test_next_pull_resumes_three_days_before_the_last_stored_day(tmp_path):
    channel_daily._save_rows(
        "gomoveshift",
        [
            {
                "day": "2026-10-20",
                "dimension": "facebook:page_media_view",
                "value": "page",
                "views": 9,
            }
        ],
        root=tmp_path,
    )
    collector = _ScriptedCollector({"page_media_view": [("2026-10-20", 11), ("2026-10-21", 4)]})

    pull_facebook_page_daily(collector, "gomoveshift", date(2026, 10, 22), root=tmp_path)

    assert ("page_media_view", "2026-10-17", "2026-10-22") in collector.calls
    # The other metrics have nothing stored yet, so they still backfill from June.
    assert ("page_video_views", "2026-06-01", "2026-08-30") in collector.calls
    series = facebook_page_series(load_rows("gomoveshift", root=tmp_path), "page_media_view")
    assert series == {"2026-10-20": 11, "2026-10-21": 4}  # re-pulled day replaced, not summed


def test_up_to_date_store_repulls_only_the_settle_window(tmp_path):
    channel_daily._save_rows(
        "gomoveshift",
        [
            {"day": "2026-10-29", "dimension": f"facebook:{m}", "value": "page", "views": 1}
            for m in FACEBOOK_PAGE_METRICS
        ],
        root=tmp_path,
    )
    collector = _ScriptedCollector({})
    pull_facebook_page_daily(collector, "gomoveshift", date(2026, 10, 30), root=tmp_path)
    # Each metric re-pulls only its settle window, one request apiece.
    assert sorted(collector.calls) == sorted(
        (m, "2026-10-26", "2026-10-30") for m in FACEBOOK_PAGE_METRICS
    )


def test_facebook_rows_never_reach_the_youtube_classifier():
    fb_rows = [
        {
            "day": f"2026-10-{d:02d}",
            "dimension": "facebook:page_media_view",
            "value": "page",
            "views": 0,
        }
        for d in range(1, 30)
    ]
    assert classify_youtube_shorts_feed(fb_rows) == {}
    merged = upsert_rows(
        fb_rows,
        [
            {
                "day": "2026-10-01",
                "dimension": "insightTrafficSourceType",
                "value": "SHORTS",
                "views": 8,
            }
        ],
    )
    assert len(merged) == len(fb_rows) + 1


# -- nightly handler ----------------------------------------------------------


def test_nightly_handler_prints_one_status_line_per_metric(capsys):
    status = {m: None for m in FACEBOOK_PAGE_METRICS}
    status["page_media_view"] = 152
    collector = SimpleNamespace(facebook=object())
    with patch.object(metrics_handler, "pull_facebook_page_daily", return_value=status) as pull:
        metrics_handler._pull_facebook_page_daily(collector, "gomoveshift")

    assert pull.call_args.args[:2] == (collector.facebook, "gomoveshift")
    line = capsys.readouterr().out.strip()
    assert line.startswith("FB_PAGE_DAILY page_media_view=152 ")
    assert "page_video_views=unavailable" in line


def test_nightly_handler_swallows_a_pull_failure(capsys, caplog):
    collector = SimpleNamespace(facebook=object())
    with patch.object(
        metrics_handler, "pull_facebook_page_daily", side_effect=RuntimeError("boom")
    ):
        metrics_handler._pull_facebook_page_daily(collector, "gomoveshift")
    assert "FB_PAGE_DAILY" not in capsys.readouterr().out
    assert "facebook page-daily pull skipped" in caplog.text
