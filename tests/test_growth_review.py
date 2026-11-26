"""Weekly growth review: format scorecard over settled day-7 views, and queue post problems."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from gomoveshift.tracking import growth_review as gr
from gomoveshift.tracking.schema import DailyMetricPoint

TODAY = date(2026, 11, 25)
NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


def _daily(posted: date, per_day: int):
    return [
        DailyMetricPoint(day=(posted + timedelta(days=i)).isoformat(), views=per_day)
        for i in range(7)
    ]


def _short(posted: date, per_day: int, seconds: float, platform="facebook"):
    return SimpleNamespace(
        posted_at=posted.isoformat() + "T10:00:00",
        content=SimpleNamespace(duration_sec=seconds),
        metrics={platform: SimpleNamespace(daily=_daily(posted, per_day))},
    )


def test_length_bands():
    assert gr.length_band(None) == "length_unknown"
    assert gr.length_band(12) == "under15s"
    assert gr.length_band(29.9) == "15to30s"
    assert gr.length_band(45) == "30to60s"
    assert gr.length_band(90) == "over60s"


def test_underperforming_group_is_flagged_only_with_enough_posts():
    posted = TODAY - timedelta(days=12)
    shorts = [_short(posted, 100, 45) for _ in range(6)] + [
        _short(posted, 10, 12) for _ in range(5)
    ]
    record = SimpleNamespace(location="Donegal", shorts=shorts)
    scores = gr.format_scorecard(gr.samples_from_records([record], TODAY))
    by_group = {s.group: s for s in scores}
    short_clips = by_group["length:under15s"]
    assert short_clips.n == 5 and short_clips.median == 70
    assert short_clips.underperforming
    assert "GROWTH_UNDERPERFORMER platform=facebook group=length:under15s n=5" in short_clips.line()
    assert not by_group["length:30to60s"].underperforming
    assert by_group["place:Donegal"].n == 11


def test_thin_group_is_reported_but_never_flagged():
    posted = TODAY - timedelta(days=12)
    shorts = [_short(posted, 100, 45) for _ in range(6)] + [_short(posted, 1, 12) for _ in range(2)]
    scores = gr.format_scorecard(
        gr.samples_from_records([SimpleNamespace(location="", shorts=shorts)], TODAY)
    )
    thin = next(s for s in scores if s.group == "length:under15s")
    assert thin.n == 2 and not thin.underperforming


def test_unsettled_or_old_posts_are_left_out_never_zero():
    fresh = _short(TODAY - timedelta(days=3), 100, 45)
    old = _short(TODAY - timedelta(days=60), 100, 45)
    samples = gr.samples_from_records([SimpleNamespace(location="", shorts=[fresh, old])], TODAY)
    assert samples == []


def _queue(tmp_path, rows):
    db = tmp_path / "social_queue.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE deliveries (id INTEGER PRIMARY KEY, project_name TEXT, short_index INTEGER, "
        "platform TEXT, plan_hash TEXT, status TEXT, error_class TEXT, scheduled_at TEXT, "
        "published_at TEXT, platform_post_id TEXT, platform_upload_id TEXT, updated_at TEXT, "
        "shadow INTEGER DEFAULT 0)"
    )
    for r in rows:
        conn.execute(
            "INSERT INTO deliveries (id, project_name, short_index, platform, plan_hash, status, "
            "error_class, scheduled_at, published_at, updated_at, shadow) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            r,
        )
    conn.commit()
    conn.close()
    return db


def test_post_problems_finds_failed_stuck_and_duplicate(tmp_path):
    t = lambda h: (NOW - timedelta(hours=h)).isoformat()  # noqa: E731
    db = _queue(
        tmp_path,
        [
            (1, "day-10", 0, "instagram", "a", "failed_permanent", "auth", t(50), None, t(40), 0),
            (2, "day-11", 0, "facebook", "a", "queued", None, t(30), None, t(30), 0),
            (3, "day-12", 0, "facebook", "a", "published", None, t(20), t(20), t(20), 0),
            (4, "day-12", 0, "facebook", "b", "published", None, t(10), t(10), t(10), 0),
            (5, "day-13", 0, "facebook", "a", "queued", None, t(2), None, t(2), 0),
            (6, "day-14", 0, "facebook", "a", "failed_permanent", "x", t(50), None, t(40), 1),
        ],
    )
    problems = gr.post_problems(db, NOW)
    kinds = {(p.kind, p.project) for p in problems}
    assert kinds == {("failed", "day-10"), ("stuck", "day-11"), ("duplicate", "day-12")}
    dup = next(p for p in problems if p.kind == "duplicate")
    assert dup.delivery_ids == (3, 4)
    assert dup.line().startswith("POST_DUPLICATE platform=facebook project=day-12")


def test_post_problems_ignores_anything_older_than_the_window(tmp_path):
    old = (NOW - timedelta(days=40)).isoformat()
    db = _queue(
        tmp_path, [(1, "old", 0, "facebook", "a", "failed_permanent", "x", old, None, old, 0)]
    )
    assert gr.post_problems(db, NOW) == []


def test_summary_line_counts():
    line = gr.summary_line([], [gr.PostProblem("stuck", "facebook", "p", 0, (1,), "")])
    assert line == (
        "GROWTH_REVIEW window_days=28 groups=0 underperforming=0 failed=0 stuck=1 duplicate=0"
    )
