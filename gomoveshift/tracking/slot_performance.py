"""Slot-keyed performance aggregate store.

Aggregates collected short metrics into ``(region, platform, weekday,
hour_local)`` buckets, derived from the plan sidecar
(``<project>_publish_plan.json``) — an unresolvable sidecar/short is
skipped rather than guessed.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .schema import ensure_slot_performance_history_schema, ensure_slot_performance_schema

#: Per-short observation backing ``slot_performance`` — one row per
#: collected short, keyed by content identity (project + short + platform).
#: In this fold the source and metric set are fixed (one collector, two
#: lifetime metrics per platform), so that key IS this fold's slice of the
#: plan's canonical `(content id, platform, source, metric, observation
#: period)` composite key. Recollecting a short is an INSERT OR REPLACE of
#: its row here, so a replay can never inflate `slot_performance.n` — the
#: aggregate is a full rebuild over this table on every fold, never an
#: incremental running mean.
SLOT_OBSERVATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS slot_observations (
    project_name TEXT NOT NULL,
    short_index INTEGER NOT NULL,
    platform TEXT NOT NULL,
    region TEXT NOT NULL,
    weekday INTEGER NOT NULL,
    hour_local INTEGER NOT NULL,
    views INTEGER NOT NULL,
    engagement_rate REAL NOT NULL,
    PRIMARY KEY (project_name, short_index, platform)
);
"""


def ensure_slot_observations_schema(conn: sqlite3.Connection) -> None:
    """Create the ``slot_observations`` table if it doesn't exist yet."""
    conn.executescript(SLOT_OBSERVATIONS_SCHEMA)


#: Hour-of-day audience-online snapshots (raw platform-reported presence,
#: not our own post performance) — separate from slot_performance below.
AUDIENCE_ONLINE_SCHEMA = """
CREATE TABLE IF NOT EXISTS audience_online (
    platform TEXT NOT NULL,
    weekday INTEGER NOT NULL,
    hour_local INTEGER NOT NULL,
    n_online INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (platform, weekday, hour_local, fetched_at)
);
"""


def ensure_audience_online_schema(conn: sqlite3.Connection) -> None:
    """Create the ``audience_online`` table if it doesn't exist yet."""
    conn.executescript(AUDIENCE_ONLINE_SCHEMA)


def upsert_audience_online(
    conn: sqlite3.Connection,
    platform: str,
    hour_counts: dict,
    fetched_at: Optional[str] = None,
    weekday: Optional[int] = None,
) -> int:
    """Persist ``{hour_local: n_online}``; ``weekday``/``fetched_at``
    default to now. Same key replaces rather than duplicates. Returns the
    row count written."""
    ensure_audience_online_schema(conn)
    now = datetime.now()
    fetched_at = fetched_at if fetched_at is not None else now.isoformat()
    weekday = now.weekday() if weekday is None else weekday
    written = 0
    with conn:
        for hour_local, n_online in hour_counts.items():
            conn.execute(
                "INSERT OR REPLACE INTO audience_online "
                "(platform, weekday, hour_local, n_online, fetched_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (platform, weekday, int(hour_local), int(n_online), fetched_at),
            )
            written += 1
    return written


@dataclass(frozen=True)
class MetricRow:
    """One collected metric sample for a published short."""

    project_name: str
    short_index: int
    platform: str
    #: ``None`` when the platform did not report the measurement — e.g. a
    #: short with no shares has no derivable engagement rate.
    views: Optional[int]
    engagement_rate: Optional[float]


@dataclass(frozen=True)
class SlotRow:
    """One ranked slot aggregate."""

    region: str
    platform: str
    weekday: int  # datetime.weekday(): 0=Monday .. 6=Sunday
    hour_local: int
    n: int
    mean_views: float
    mean_engagement: float


def read_sidecar_slot(sidecar_path: Path, short_index: int) -> Optional[dict]:
    """``{"region", "weekday", "hour_local"}`` for ``short_index`` from a
    publish-plan sidecar, or ``None`` if unresolvable — never fabricated."""
    try:
        raw = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    shorts = raw.get("shorts") if isinstance(raw, dict) else None
    if not isinstance(shorts, list):
        return None

    for entry in shorts:
        if not isinstance(entry, dict) or entry.get("short_index") != short_index:
            continue
        schedule = entry.get("schedule") or {}
        scheduled_at_utc = schedule.get("scheduled_at_utc")
        display_timezone = schedule.get("display_timezone")
        sampled_regions = entry.get("sampled_regions") or []
        if not scheduled_at_utc or not display_timezone or not sampled_regions:
            return None
        try:
            dt_utc = datetime.fromisoformat(str(scheduled_at_utc).replace("Z", "+00:00"))
            local_dt = dt_utc.astimezone(ZoneInfo(display_timezone))
        except (ValueError, ZoneInfoNotFoundError):
            return None
        return {
            "region": sampled_regions[0],
            "weekday": local_dt.weekday(),
            "hour_local": local_dt.hour,
        }
    return None


def upsert_from_metrics(
    conn: sqlite3.Connection,
    rows: Iterable[MetricRow],
    sidecar_lookup: Callable[[str], Optional[Path]],
    folded_at: Optional[str] = None,
) -> int:
    """Fold ``rows`` into the ``slot_performance`` table. ``sidecar_lookup``
    resolves a project's publish-plan sidecar path (or ``None``); a row
    whose sidecar/slot can't be resolved, or whose ``views``/
    ``engagement_rate`` the platform never reported, is skipped — an
    unknown measurement is never folded in as a zero, and never costs the
    other rows their fold. Returns the count folded in.

    ``folded_at`` (ISO-UTC) stamps the ``slot_performance_history`` snapshot
    this fold appends; defaults to now — a test-only seam so a
    multi-fold trend test doesn't depend on wall-clock timing.
    """
    ensure_slot_performance_schema(conn)
    ensure_slot_observations_schema(conn)
    folded = 0
    with conn:
        for row in rows:
            sidecar_path = sidecar_lookup(row.project_name)
            if sidecar_path is None:
                continue
            slot = read_sidecar_slot(Path(sidecar_path), row.short_index)
            if slot is None:
                continue
            if row.views is None or row.engagement_rate is None:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO slot_observations "
                "(project_name, short_index, platform, region, weekday, hour_local, "
                "views, engagement_rate) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row.project_name,
                    row.short_index,
                    row.platform,
                    slot["region"],
                    slot["weekday"],
                    slot["hour_local"],
                    row.views,
                    row.engagement_rate,
                ),
            )
            folded += 1
        if folded:
            _rebuild_slot_performance(conn)
            _append_slot_performance_history(conn, folded_at)
    return folded


def _rebuild_slot_performance(conn: sqlite3.Connection) -> None:
    """Recompute every ``slot_performance`` row from ``slot_observations``.

    A full rebuild, not an incremental update — a replayed collection
    (``INSERT OR REPLACE`` above) is already reflected in the observation
    table, so recomputing here can never double-count it. A
    ``slot_performance`` row with no backing observation was produced by
    the retired incremental fold and is dropped.
    """
    conn.execute("DELETE FROM slot_performance")
    conn.execute(
        "INSERT INTO slot_performance "
        "(region, platform, weekday, hour_local, n, mean_views, mean_engagement) "
        "SELECT region, platform, weekday, hour_local, COUNT(*), "
        "AVG(views), AVG(engagement_rate) "
        "FROM slot_observations "
        "GROUP BY region, platform, weekday, hour_local"
    )


def _append_slot_performance_history(conn: sqlite3.Connection, folded_at: Optional[str]) -> None:
    """Append a snapshot of the freshly rebuilt ``slot_performance`` to
    ``slot_performance_history``. Never DELETEs — unlike
    ``_rebuild_slot_performance`` above, which overwrites the aggregate on
    every fold, this table is append-only so a trend across folds survives.
    """
    ensure_slot_performance_history_schema(conn)
    stamp = folded_at if folded_at is not None else datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO slot_performance_history "
        "(folded_at, region, platform, weekday, hour_local, n, mean_views, mean_engagement) "
        "SELECT ?, region, platform, weekday, hour_local, n, mean_views, mean_engagement "
        "FROM slot_performance",
        (stamp,),
    )


def slot_performance_trend(
    conn: sqlite3.Connection,
    region: str,
    platform: str,
    weekday: int,
    hour_local: int,
    weeks: int = 4,
) -> List[dict]:
    """The last ``weeks`` fold snapshots for one slot, oldest first — the
    trend ``metrics best-times --trend`` prints per top slot."""
    ensure_slot_performance_history_schema(conn)
    cur = conn.execute(
        "SELECT folded_at, n, mean_views, mean_engagement FROM slot_performance_history "
        "WHERE region = ? AND platform = ? AND weekday = ? AND hour_local = ? "
        "ORDER BY folded_at DESC LIMIT ?",
        (region, platform, weekday, hour_local, weeks),
    )
    return [
        {"folded_at": r[0], "n": r[1], "mean_views": r[2], "mean_engagement": r[3]}
        for r in reversed(cur.fetchall())
    ]


def slots_ranked(
    conn: sqlite3.Connection,
    region: Optional[str] = None,
    platform: Optional[str] = None,
    min_n: int = 3,
) -> List[SlotRow]:
    """Slot aggregates with ``n >= min_n``, ranked by mean views (then mean
    engagement) descending; ``region``/``platform`` narrow the result."""
    ensure_slot_performance_schema(conn)
    clauses = ["n >= ?"]
    params: list = [min_n]
    if region is not None:
        clauses.append("region = ?")
        params.append(region)
    if platform is not None:
        clauses.append("platform = ?")
        params.append(platform)
    query = (
        "SELECT region, platform, weekday, hour_local, n, mean_views, mean_engagement "
        "FROM slot_performance WHERE " + " AND ".join(clauses) + " "
        "ORDER BY mean_views DESC, mean_engagement DESC"
    )
    cur = conn.execute(query, params)
    return [SlotRow(*record) for record in cur.fetchall()]
