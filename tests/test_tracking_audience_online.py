"""audience_online table — hour-of-day follower/fan-online snapshots,
keyed by platform/weekday/hour, beside slot_performance."""

from __future__ import annotations

import sqlite3

from gomoveshift.tracking.slot_performance import (
    ensure_audience_online_schema,
    upsert_audience_online,
)


def test_ensure_schema_creates_table():
    conn = sqlite3.connect(":memory:")
    ensure_audience_online_schema(conn)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "audience_online" in tables


def test_upsert_writes_one_row_per_hour_and_is_readable():
    conn = sqlite3.connect(":memory:")
    written = upsert_audience_online(
        conn, "instagram", {0: 12, 9: 340}, fetched_at="2026-09-08T00:00:00", weekday=1
    )
    assert written == 2
    rows = conn.execute(
        "SELECT platform, weekday, hour_local, n_online, fetched_at "
        "FROM audience_online ORDER BY hour_local"
    ).fetchall()
    assert rows == [
        ("instagram", 1, 0, 12, "2026-09-08T00:00:00"),
        ("instagram", 1, 9, 340, "2026-09-08T00:00:00"),
    ]


def test_upsert_replace_on_same_key_updates_value():
    conn = sqlite3.connect(":memory:")
    upsert_audience_online(conn, "instagram", {9: 100}, fetched_at="t1", weekday=2)
    upsert_audience_online(conn, "instagram", {9: 250}, fetched_at="t1", weekday=2)
    assert conn.execute("SELECT n_online FROM audience_online").fetchall() == [(250,)]
