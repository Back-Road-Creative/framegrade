"""Rule 9, the fast guard: N consecutive settled Shorts under a fraction of the channel's
trailing median raise ONE ``short_bad_run`` incident carrying a recorded-data cause per Short."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from gomoveshift.health import bad_run, run_check
from gomoveshift.health.check import _line
from gomoveshift.health.store import load_all
from gomoveshift.publishing.queue import open_queue_db
from gomoveshift.tracking.schema import (
    ContentMeta,
    DailyMetricPoint,
    PlatformMetrics,
    ShortRecord,
    TrackingRecord,
)
from gomoveshift.tracking.store import TrackingStore

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
TODAY = NOW.date()
NOW_ISO = NOW.isoformat()


def _short(
    project: str,
    index: int,
    posted: datetime,
    views,
    *,
    hook: str = "",
    metrics: bool = True,
) -> ShortRecord:
    """A settled Short: a day-7 series summing to ``views`` (None = no series at all)."""
    daily = []
    if views is not None:
        day = posted.date()
        daily = [
            DailyMetricPoint(day=(day + timedelta(days=i)).isoformat(), views=0) for i in range(7)
        ]
        daily[0].views = views
    pm = PlatformMetrics(
        views=views, watch_time_seconds=10.0, avg_view_duration_seconds=5.0, daily=daily
    )
    return ShortRecord(
        index=index,
        posted_at=posted.isoformat(),
        posted_at_source="platform_confirmed",
        platform_ids={"youtube": f"{project}-{index}"},
        content=ContentMeta(hook=hook or f"{project} place {index}"),
        metrics={"youtube": pm} if metrics else {},
    )


def _channel(views_oldest_first, **kw):
    """One project; Shorts posted every 12 h, ending 20 days ago (all settled)."""
    start = NOW - timedelta(days=20 + len(views_oldest_first) // 2)
    shorts = [
        _short("p", i + 1, start + timedelta(hours=12 * i), v, **kw)
        for i, v in enumerate(views_oldest_first)
    ]
    return [TrackingRecord(project="p", shorts=shorts)]


HEALTHY = [40, 35, 50, 42, 38, 45, 41, 39, 44, 37]
#: The measured base rate: 4 of the last 10 Shorts under 5 views.
BASE_RATE = [40, 35, 50, 42, 38, 45, 3, 1, 4, 2]


def _run(records, queue_rows=None, regions=None):
    return bad_run.bad_run_incidents(
        records, queue_rows or {}, regions or (lambda p, i: []), TODAY, NOW_ISO
    )


def test_base_rate_pattern_trips_one_incident():
    rows = _run(_channel(BASE_RATE + []))
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "short_bad_run" and row["key"] == "short_bad_run"
    assert row["n"] == bad_run.GUARD_N
    assert [s["index"] for s in row["shorts"]] == [10, 9, 8]  # newest first
    assert row["median_views"] == 40.0 and row["threshold_views"] == 40.0 * bad_run.LOW_FRACTION


def test_constants_make_the_base_rate_trip():
    # 4 consecutive Shorts under 5 views against a median of a few dozen: 5 < fraction * median
    assert bad_run.GUARD_N <= 4
    assert 5 < bad_run.LOW_FRACTION * 40


def test_healthy_channel_is_quiet():
    assert _run(_channel(HEALTHY + [41, 43, 40])) == []


def test_one_good_short_in_the_window_breaks_the_run():
    assert _run(_channel(HEALTHY + [2, 60, 1])) == []


def test_not_enough_settled_baseline_is_not_evaluated():
    assert _run(_channel([40, 38, 2, 1, 3])) == []


def test_unsettled_shorts_are_ignored():
    records = _channel(HEALTHY)
    fresh = [_short("p", 20 + i, NOW - timedelta(days=2, hours=i), 0) for i in range(3)]
    records[0].shorts.extend(fresh)
    assert _run(records) == []


def test_short_with_no_series_counts_as_low_and_names_missing_metrics():
    rows = _run(_channel(HEALTHY + [None, 1, 2]))
    causes = {c["code"] for s in rows[0]["shorts"] if s["index"] == 11 for c in s["causes"]}
    assert "missing_core_metrics" in causes
    assert rows[0]["shorts"][0]["views"] == 2


def test_duplicate_place_text_is_c5_fixable():
    recs = _channel(HEALTHY + [2, 1, 3])
    for s in recs[0].shorts[-3:]:
        s.content.hook = "  Colleton  County DRIVE "
    recs[0].shorts[0].content.hook = "colleton county drive"  # an earlier Short with the same text
    rows = _run(recs)
    for short in rows[0]["shorts"]:
        dup = [c for c in short["causes"] if c["code"] == "duplicate_place_text"]
        assert dup and dup[0]["c5_fixable"] is True
    others = [
        c for s in rows[0]["shorts"] for c in s["causes"] if c["code"] != "duplicate_place_text"
    ]
    assert all(c["c5_fixable"] is False for c in others)


def test_no_duplicate_when_text_differs():
    rows = _run(_channel(HEALTHY + [2, 1, 3]))
    assert not [
        c for s in rows[0]["shorts"] for c in s["causes"] if c["code"] == "duplicate_place_text"
    ]


def test_queue_causes_status_late_and_bunched():
    recs = _channel(HEALTHY + [2, 1, 3])
    s13 = recs[0].shorts[-1]
    # a second Short posted 20 minutes after the newest one
    recs[0].shorts[-2].posted_at = (
        datetime.fromisoformat(s13.posted_at) - timedelta(minutes=20)
    ).isoformat()
    sched = datetime.fromisoformat(s13.posted_at)
    queue = {
        ("p", 13): {
            "status": "failed_permanent",
            "error_class": "quota",
            "scheduled_at": sched.isoformat(),
            "published_at": (sched + timedelta(hours=9)).isoformat(),
        }
    }
    rows = _run(recs, queue)
    newest = next(s for s in rows[0]["shorts"] if s["index"] == 13)
    codes = {c["code"] for c in newest["causes"]}
    assert {"delivery_status", "late_publish", "bunched_publish"} <= codes
    prev = next(s for s in rows[0]["shorts"] if s["index"] == 12)
    assert "bunched_publish" in {c["code"] for c in prev["causes"]}


def test_published_on_time_is_not_a_cause():
    recs = _channel(HEALTHY + [2, 1, 3])
    sched = datetime.fromisoformat(recs[0].shorts[-1].posted_at)
    queue = {
        ("p", 13): {
            "status": "published",
            "error_class": None,
            "scheduled_at": sched.isoformat(),
            "published_at": (sched + timedelta(minutes=5)).isoformat(),
        }
    }
    rows = _run(recs, queue)
    newest = next(s for s in rows[0]["shorts"] if s["index"] == 13)
    assert not {"delivery_status", "late_publish"} & {c["code"] for c in newest["causes"]}


def test_place_label_vs_measured_county():
    recs = _channel(HEALTHY + [2, 1, 3])
    recs[0].shorts[-1].content.hook = "Driving through Berkeley County, South Carolina"
    rows = _run(recs, regions=lambda p, i: ["Colleton County, South Carolina"] if i == 13 else [])
    newest = next(s for s in rows[0]["shorts"] if s["index"] == 13)
    cause = [c for c in newest["causes"] if c["code"] == "place_county_mismatch"]
    assert cause and "berkeley" in cause[0]["detail"]
    other = next(s for s in rows[0]["shorts"] if s["index"] == 12)
    assert "place_county_mismatch" not in {c["code"] for c in other["causes"]}


def test_short_without_any_cause_lists_none():
    rows = _run(_channel(HEALTHY + [2, 1, 3]))
    assert all(s["causes"] == [] for s in rows[0]["shorts"])


def test_line_prints_without_keyerror():
    row = _run(_channel(BASE_RATE))[0]
    line = _line(row)
    assert "3" in line and "#10" in line


def test_sidecar_regions_reads_the_short_entry(tmp_path):
    side = tmp_path / "plan.json"
    side.write_text(
        json.dumps({"shorts": [{"short_index": 4, "sampled_regions": ["A County, X"]}]})
    )
    assert bad_run.sidecar_regions(str(side), 4) == ["A County, X"]
    assert bad_run.sidecar_regions(str(side), 5) == []
    assert bad_run.sidecar_regions(str(tmp_path / "gone.json"), 4) == []
    side.write_text("not json")
    assert bad_run.sidecar_regions(str(side), 4) == []


def _storage(tmp_path, views):
    root = tmp_path / "out"
    store = TrackingStore(root)
    for rec in _channel(views):
        store.save(rec)
    seo = tmp_path / "seo"
    conn = open_queue_db(seo)
    side = tmp_path / "plan.json"
    side.write_text(
        json.dumps({"shorts": [{"short_index": 12, "sampled_regions": ["C County, S"]}]})
    )
    conn.execute(
        "INSERT INTO deliveries (project_name, short_index, platform, plan_hash, status, shadow,"
        " sidecar_path, scheduled_at, published_at, created_at, updated_at)"
        " VALUES ('p', 12, 'youtube', 'h', 'failed_permanent', 0, ?, '2026-02-01T10:00:00+00:00',"
        " NULL, 'c', 'u')",
        (str(side),),
    )
    conn.commit()
    conn.close()
    return root, seo


def test_load_queue_rows_is_read_only_and_keyed(tmp_path):
    _root, seo = _storage(tmp_path, HEALTHY)
    rows = bad_run.load_queue_rows(seo / "social_queue.sqlite3")
    assert rows[("p", 12)]["status"] == "failed_permanent"
    assert rows[("p", 12)]["sidecar_path"]
    assert bad_run.load_queue_rows(tmp_path / "missing.sqlite3") == {}


def test_run_check_raises_files_and_resolves(tmp_path):
    root, seo = _storage(tmp_path, HEALTHY + [2, 1, 3])
    cfg = SimpleNamespace(seo=SimpleNamespace(storage_path=str(seo)))
    pushed = []

    def notify(event, payload, text):
        if event != "health_seo_feedback_stale":  # rule 12 is not this test's subject
            pushed.append((event, text))
        return True

    kw = dict(
        incidents_dir=tmp_path / "inc",
        jobs_dir=tmp_path / "jobs",
        code_root=tmp_path,
        staging_sha="abc",
        now=NOW,
        notify_fn=notify,
        tracking_root=root,
    )
    res = run_check(cfg, **kw)
    assert [k for k in res["new"] if not k.startswith("seo_feedback_stale")] == ["short_bad_run"]
    assert [e for e, _ in pushed] == ["health_short_bad_run"]
    rows, _bad = load_all(tmp_path / "inc")
    row = next(r for r in rows if r["kind"] == "short_bad_run")
    assert any(
        c["code"] == "delivery_status"
        for s in row["shorts"]
        for c in s["causes"]
        if s["index"] == 12
    )
    # idempotent: same condition, nothing new, nothing re-pushed
    again = run_check(cfg, **kw)
    assert again["new"] == [] and len(pushed) == 1
    # a recovery (three good Shorts) closes it
    store = TrackingStore(root)
    rec = store.load("p")
    start = NOW - timedelta(days=15)
    rec.shorts.extend(_short("p", 40 + i, start + timedelta(hours=12 * i), 45) for i in range(3))
    store.save(rec)
    done = run_check(cfg, **kw)
    assert done["resolved"] == ["short_bad_run"]


def test_run_check_skips_with_relative_storage(tmp_path):
    cfg = SimpleNamespace(seo=SimpleNamespace(storage_path="relative/seo"))
    res = run_check(
        cfg,
        incidents_dir=tmp_path / "inc",
        jobs_dir=tmp_path / "jobs",
        code_root=tmp_path,
        staging_sha="abc",
        now=NOW,
        notify_fn=lambda *a: True,
        tracking_root=tmp_path / "nothing",
    )
    assert not [r for r in res["incidents"] if r["kind"] == "short_bad_run"]
    assert any("short_bad_run" in n for n in res["notes"])
