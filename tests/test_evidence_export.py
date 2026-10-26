"""Settled day-7 evidence: collection budget, snapshot fallback, golden export."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from googleapiclient.errors import HttpError

from gomoveshift.cli.handlers.metrics import cmd_metrics
from gomoveshift.tracking import MetricsCollector, TrackingStore
from gomoveshift.tracking.collector import AnalyticsStatus, YouTubeCollector
from gomoveshift.tracking.evidence_export import build_evidence_bundle
from gomoveshift.tracking.observation import MetricObservation, ObservationStatus
from gomoveshift.tracking.schema import (
    DailyMetricPoint,
    LongFormRecord,
    PlatformMetrics,
    ShortRecord,
    TrackingRecord,
)

FIXTURE = Path(__file__).parent / "fixtures" / "evidence_bundle_v2.json"
TODAY = date(2026, 10, 9)


def _vid(item):
    return getattr(item, "video_id", None) or next(iter(item.platform_ids.values()))


def _short(index, vid, posted="2026-09-20T14:00:00", daily=None, obs=None):
    s = ShortRecord(
        index=index,
        posted_at=posted,
        posted_at_source="platform_confirmed",
        platform_ids={"youtube": vid},
        metrics={"youtube": PlatformMetrics(views=1, daily=daily or [])},
    )
    if obs:
        s.observation_log = {"youtube": obs}
    return s


def _series(posted_day, views):
    d0 = date.fromisoformat(posted_day)
    return [
        DailyMetricPoint(day=(d0.replace(day=d0.day + i)).isoformat(), views=v)
        for i, v in enumerate(views)
    ]


class FakeYouTube:
    def __init__(self, exhaust_after=None):
        self.calls = []
        self.exhaust_after = exhaust_after
        self.quota_exhausted = False

    def is_available(self):
        return True

    def collect_observation(self, video_id, published_at=None, account_id=None):
        return MetricObservation(
            platform="youtube",
            status=ObservationStatus.MEASURED,
            post_id=video_id,
            cumulative=PlatformMetrics(views=5),
            event_at=published_at,
            fetched_at="2026-10-09T06:00:00+00:00",
        )

    def fetch_analytics_daily_metrics(self, video_id, start_date, max_days=14):
        self.calls.append(video_id)
        if self.exhaust_after is not None and len(self.calls) > self.exhaust_after:
            self.quota_exhausted = True
            return []
        return [{"day": f"2026-09-{20 + i}", "views": 3} for i in range(7)]


def _collector(fake):
    mc = MetricsCollector()
    mc.youtube = fake
    mc._collectors = {"youtube": fake}
    return mc


def _plain_short(index, vid):
    s = _short(index, vid)
    s.metrics = {}
    return s


def test_settled_series_not_refetched():
    settled = _short(1, "vid-a", daily=_series("2026-09-20", [4] * 7))
    unsettled = _plain_short(2, "vid-b")
    young = _plain_short(3, "vid-c")
    young.posted_at = "2026-10-08T10:00:00"
    record = TrackingRecord(project="p", shorts=[settled, unsettled, young])
    fake = FakeYouTube()
    _collector(fake).collect_all(record, today=TODAY)
    # Only the settled-window short WITHOUT a settled series is fetched.
    assert fake.calls == ["vid-b"]
    assert settled.metrics["youtube"].daily[0].views == 4
    assert len(unsettled.metrics["youtube"].daily) == 7


def test_quota_403_429_stops_with_ANALYTICS_BUDGET_EXHAUSTED_remaining_N():
    shorts = [_plain_short(i, f"v{i}") for i in range(1, 6)]
    fake = FakeYouTube(exhaust_after=2)
    mc = _collector(fake)
    mc.collect_all(TrackingRecord(project="p", shorts=shorts), today=TODAY)
    assert fake.calls == ["v1", "v2", "v3"]  # v3 hit the quota; v4/v5 never asked
    assert mc.budget_exhausted_line == "ANALYTICS_BUDGET_EXHAUSTED remaining=3 reason=quota"
    assert shorts[2].metrics["youtube"].daily == []


def test_quota_http_errors_flag_the_collector(tmp_path):
    for status, reason in ((429, "rateLimitExceeded"), (403, "quotaExceeded")):
        c = YouTubeCollector(token_path=tmp_path / "missing.json")
        body = json.dumps({"error": {"errors": [{"reason": reason}]}}).encode()
        c._note_analytics_exception(HttpError(MagicMock(status=status, reason="x"), body))
        assert c.quota_exhausted
    c = YouTubeCollector(token_path=tmp_path / "missing.json")
    c._note_analytics_exception(
        HttpError(MagicMock(status=403), b'{"error":{"errors":[{"reason":"forbidden"}]}}')
    )
    assert not c.quota_exhausted


def _obs(fetched, views, event="2026-09-20T00:00:00+00:00", status="measured"):
    return {
        "platform": "youtube",
        "status": status,
        "post_id": "v",
        "event_at": event,
        "fetched_at": fetched,
        "cumulative": {"views": views},
    }


def _fb_short(index, vid, posted, obs):
    s = ShortRecord(
        index=index,
        posted_at=posted,
        posted_at_source="platform_confirmed",
        platform_ids={"facebook": vid},
    )
    s.observation_log = {"facebook": obs}
    return s


def _fb_obs(
    total=None,
    initial=None,
    watch=None,
    views=None,
    post_format="reel",
    status="measured",
    age=timedelta(days=7),
    event="2026-09-22T15:00:00+00:00",
):
    return {
        "platform": "facebook",
        "status": status,
        "post_format": post_format,
        "event_at": event,
        "fetched_at": (datetime.fromisoformat(event) + age).isoformat(),
        "cumulative": {
            "views": views,
            "reel_total_plays": total,
            "reel_initial_plays": initial,
            "watch_time_seconds": watch,
        },
    }


def _bundle(shorts):
    rec = TrackingRecord(project="p", shorts=shorts)
    return build_evidence_bundle([rec], AnalyticsStatus("ok"), datetime(2026, 10, 9, 6, 0, 0))


def test_snapshot_fallback_nearest_to_age_7_within_12h_measured_only_else_excluded_and_counted():
    near = _short(
        1,
        "near",
        obs=[
            _obs("2026-09-26T20:00:00+00:00", 70),  # age 6d20h (4h off)
            _obs("2026-09-27T06:00:00+00:00", 90),  # age 7d6h (6h off)
            _obs("2026-09-27T01:00:00+00:00", 80, status="not_returned"),  # refusal: ignored
        ],
    )
    far = _short(2, "far", obs=[_obs("2026-09-28T00:00:00+00:00", 99)])  # 8d off 12h
    none = _short(3, "none")
    bundle = _bundle([near, far, none])
    assert [(r["video_id"], r["day7"], r["source"]) for r in bundle["shorts"]] == [
        ("near", {"views": 70}, "snapshot")
    ]
    health = bundle["health"]
    assert (health["n_registered"], health["n_with_series"]) == (3, 1)
    assert health["excluded"] == [
        {"video_id": "far", "reason": "no_series"},
        {"video_id": "none", "reason": "no_series"},
    ]


def test_export_matches_golden_bundle():
    a = _short(1, "vid-a", daily=_series("2026-09-20", [412 // 7] * 6 + [412 - 6 * (412 // 7)]))
    b = _short(
        2,
        "vid-b",
        posted="2026-09-21T18:30:00",
        obs=[_obs("2026-09-28T18:30:00+00:00", 87.5, event="2026-09-21T18:30:00+00:00")],
    )
    c = _short(3, "vid-c")
    reel = _fb_short(4, "fb-reel", "2026-09-22T15:00:00", [_fb_obs(1300, 1000, 9100)])
    video = _fb_short(
        5, "fb-video", "2026-09-23T15:00:00", [_fb_obs(post_format="video", views=640)]
    )
    lost = _fb_short(6, "fb-lost", "2026-09-24T15:00:00", [])
    lf = LongFormRecord(
        index=1,
        video_id="vid-lf",
        published_date="2026-09-19T15:00:00",
        duration_sec=912,
        metrics={
            "youtube": PlatformMetrics(
                views=9000,
                avg_view_percentage=41.2,
                daily=_series("2026-09-19", [5310] + [0] * 6),
            )
        },
    )
    record = TrackingRecord(
        project="gomoveshift", shorts=[a, b, c, reel, video, lost], long_forms=[lf]
    )
    regions = {"vid-a": "us", "vid-lf": "us"}
    bundle = build_evidence_bundle(
        [record],
        AnalyticsStatus("ok"),
        datetime(2026, 10, 9, 6, 0, 0),
        slot_for=lambda project, item: {"region": regions.get(_vid(item))},
    )
    assert bundle == json.loads(FIXTURE.read_text())


def test_cli_prints_bundle_and_excluded_line(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path))
    store = TrackingStore(tmp_path)
    store.save(TrackingRecord(project="p", shorts=[_short(1, "v1")]))
    fake = SimpleNamespace(analytics_status=lambda: AnalyticsStatus("scope_missing", "x"))
    with patch("gomoveshift.cli.handlers.metrics.YouTubeCollector", return_value=fake):
        rc = cmd_metrics(SimpleNamespace(metrics_command="export-evidence"), SimpleNamespace())
    out = capsys.readouterr()
    assert rc == 0
    bundle = json.loads(out.out)
    assert bundle["health"]["analytics_status"] == "scope_missing"
    assert bundle["health"]["detail"] == "x"
    assert "EVIDENCE_EXCLUDED n=1 reason=no_settled_series" in out.err


def test_youtube_row_carries_short_format_and_day7_views_only():
    row = _bundle([_short(1, "yt", obs=[_obs("2026-09-27T00:00:00+00:00", 50)])])["shorts"][0]
    assert row["post_format"] == "short"
    assert row["day7"] == {"views": 50}
    assert "day7_views" not in row


def test_facebook_reel_keeps_plays_in_day7_and_views_null():
    reel = _fb_short(1, "fb", "2026-09-22T15:00:00", [_fb_obs(1300, 1000, 9100)])
    bundle = _bundle([reel])
    assert bundle["schema_version"] == 2
    assert bundle["shorts"] == [
        {
            "video_id": "fb",
            "project": "p",
            "region": None,
            "platform": "facebook",
            "post_format": "reel",
            "posted_at": "2026-09-22T15:00:00",
            "weekday": 1,
            "hour_local": 11,
            "day7": {
                "views": None,
                "reel_total_plays": 1300,
                "reel_initial_plays": 1000,
                "watch_time_seconds": 9100,
            },
            "source": "snapshot",
        }
    ]


def test_facebook_video_row_carries_views():
    vid = _fb_short(1, "fbv", "2026-09-22T15:00:00", [_fb_obs(post_format="video", views=640)])
    row = _bundle([vid])["shorts"][0]
    assert row["post_format"] == "video"
    assert row["day7"]["views"] == 640
    assert row["day7"]["reel_total_plays"] is None


def test_facebook_short_without_observation_in_window_is_excluded():
    late = _fb_short(1, "late", "2026-09-22T15:00:00", [_fb_obs(5, age=timedelta(days=9))])
    bare = _fb_short(2, "bare", "2026-09-22T15:00:00", [])
    bundle = _bundle([late, bare])
    assert bundle["shorts"] == []
    assert bundle["health"]["excluded"] == [
        {"video_id": "late", "reason": "no_series", "platform": "facebook"},
        {"video_id": "bare", "reason": "no_series", "platform": "facebook"},
    ]
    assert (bundle["health"]["n_registered"], bundle["health"]["n_with_series"]) == (2, 0)


def test_unmeasured_facebook_observation_is_not_chosen():
    empty = _fb_obs()  # measured status but every counter unreported
    refused = _fb_obs(99, status="not_returned")
    good = _fb_obs(7, age=timedelta(days=7, hours=8))
    short = _fb_short(1, "fb", "2026-09-22T15:00:00", [empty, refused, good])
    row = _bundle([short])["shorts"][0]
    assert row["day7"]["reel_total_plays"] == 7
    only_empty = _fb_short(2, "fb2", "2026-09-22T15:00:00", [empty, refused])
    assert _bundle([only_empty])["shorts"] == []


def test_unknown_facebook_format_is_null():
    short = _fb_short(1, "fb", "2026-09-22T15:00:00", [_fb_obs(3, post_format="")])
    assert _bundle([short])["shorts"][0]["post_format"] is None
