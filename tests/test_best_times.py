"""``gomoveshift metrics best-times`` (CLI): the region x platform x top-3
report, with the platform's own audience-online hours beside it. Report-only, always exits 0."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from gomoveshift.cli.handlers.metrics import cmd_metrics
from gomoveshift.tracking.schema import ensure_slot_performance_schema
from gomoveshift.tracking.slot_performance import (
    ensure_audience_online_schema,
    ensure_slot_performance_history_schema,
)


def _conn():
    conn = sqlite3.connect(":memory:")
    ensure_slot_performance_schema(conn)
    ensure_audience_online_schema(conn)
    ensure_slot_performance_history_schema(conn)
    return conn


def _insert_history(conn, folded_at, region, platform, weekday, hour_local, n, mean_views):
    conn.execute(
        "INSERT INTO slot_performance_history "
        "(folded_at, region, platform, weekday, hour_local, n, mean_views, mean_engagement) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (folded_at, region, platform, weekday, hour_local, n, mean_views, 0.0),
    )
    conn.commit()


def _insert_slot(conn, region, platform, weekday, hour_local, n, mean_views):
    conn.execute(
        "INSERT INTO slot_performance "
        "(region, platform, weekday, hour_local, n, mean_views, mean_engagement) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (region, platform, weekday, hour_local, n, mean_views, 0.0),
    )
    conn.commit()


def _insert_online(conn, platform, weekday, hour_local, n_online):
    conn.execute(
        "INSERT INTO audience_online (platform, weekday, hour_local, n_online, fetched_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (platform, weekday, hour_local, n_online, "2026-09-10T00:00:00"),
    )
    conn.commit()


def _run(monkeypatch, conn, capsys, min_n=5, trend=False):
    monkeypatch.setattr("gomoveshift.cli.handlers.metrics.open_queue_db", lambda *a: conn)
    config = SimpleNamespace(
        seo=SimpleNamespace(storage_path="ignored"),
    )
    args = SimpleNamespace(metrics_command="best-times", min_n=min_n, trend=trend)
    rc = cmd_metrics(args, config)
    return rc, capsys.readouterr().out


def test_empty_store_exits_zero_saying_so(monkeypatch, capsys):
    rc, out = _run(monkeypatch, _conn(), capsys)
    assert rc == 0
    assert "insufficient data" in out.lower()


def test_top3_and_the_audience_hours_printed_for_a_ranked_platform(monkeypatch, capsys):
    conn = _conn()
    _insert_slot(conn, "ireland", "instagram", weekday=1, hour_local=12, n=8, mean_views=500)
    _insert_slot(conn, "ireland", "instagram", weekday=2, hour_local=8, n=8, mean_views=100)
    _insert_online(conn, "instagram", weekday=1, hour_local=12, n_online=900)
    _insert_online(conn, "instagram", weekday=2, hour_local=9, n_online=50)
    _insert_online(conn, "instagram", weekday=3, hour_local=20, n_online=10)

    rc, out = _run(monkeypatch, conn, capsys)
    assert rc == 0
    assert "ireland / instagram" in out
    assert "12:00" in out and "n=8" in out
    assert "audience_online top-3: 12:00, 09:00, 20:00" in out
    assert "agreement:" not in out  # GMS judges no slot from the ranking (that path is retired)


def test_a_platform_without_audience_rows_prints_no_audience_line(monkeypatch, capsys):
    conn = _conn()
    _insert_slot(conn, "usa", "youtube", weekday=1, hour_local=14, n=10, mean_views=1000)

    rc, out = _run(monkeypatch, conn, capsys)
    assert rc == 0
    assert "usa / youtube" in out
    assert "audience_online" not in out


def test_trend_prints_last_4_week_history_for_top_slot(monkeypatch, capsys):
    conn = _conn()
    _insert_slot(conn, "ireland", "instagram", weekday=1, hour_local=12, n=8, mean_views=500)
    _insert_history(conn, "2026-08-18T00:00:00+00:00", "ireland", "instagram", 1, 12, 4, 200)
    _insert_history(conn, "2026-08-25T00:00:00+00:00", "ireland", "instagram", 1, 12, 6, 350)
    _insert_history(conn, "2026-09-01T00:00:00+00:00", "ireland", "instagram", 1, 12, 8, 500)

    rc, out = _run(monkeypatch, conn, capsys, trend=True)
    assert rc == 0
    assert "trend" in out.lower()
    assert "2026-08-18T00:00:00+00:00" in out
    assert "2026-09-01T00:00:00+00:00" in out


def test_trend_omitted_by_default(monkeypatch, capsys):
    conn = _conn()
    _insert_slot(conn, "ireland", "instagram", weekday=1, hour_local=12, n=8, mean_views=500)
    _insert_history(conn, "2026-08-18T00:00:00+00:00", "ireland", "instagram", 1, 12, 4, 200)

    rc, out = _run(monkeypatch, conn, capsys)
    assert rc == 0
    assert "trend" not in out.lower()


def test_multiple_groups_each_get_their_own_section(monkeypatch, capsys):
    conn = _conn()
    _insert_slot(conn, "ireland", "instagram", weekday=1, hour_local=12, n=8, mean_views=500)
    _insert_online(conn, "instagram", weekday=1, hour_local=12, n_online=900)
    _insert_slot(conn, "usa", "youtube", weekday=1, hour_local=14, n=10, mean_views=1000)

    rc, out = _run(monkeypatch, conn, capsys)
    assert rc == 0
    assert "ireland / instagram" in out
    assert "usa / youtube" in out
