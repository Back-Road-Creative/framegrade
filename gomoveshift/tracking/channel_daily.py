"""Channel-level daily views by traffic source / content type.

Every collector in this package (`collector.py`) answers "how did THIS
short do" -- nothing pulls channel-wide views by how the viewer arrived
(``insightTrafficSourceType``) or what content type YouTube attributes the
watch to (``creatorContentType``). This module fills that gap: a daily pull
of both dimensions from the YouTube Analytics v2 ``reports.query`` (the same
``channel==MINE`` call `YouTubeCollector.fetch_analytics_metrics` already
makes), and a pure function over the stored rows that derives the
Shorts-feed share of channel views a day carried -- the number the
social-reach surfaces read.

Storage follows `tracking.store.TrackingStore`'s own root and per-project
layout (``<root>/<project>/tracking/<project>.json``): one JSON file under
``<root>/<brand>/tracking/channel_daily.json``, keyed by brand instead of
project since this pull is channel-wide, not scoped to one drive. No new
database engine and no second storage root.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..paths import gms_output_dir

logger = logging.getLogger(__name__)

#: YouTube Analytics v2 dimension names this pull persists one row set per.
TRAFFIC_SOURCE_DIMENSION = "insightTrafficSourceType"
CONTENT_TYPE_DIMENSION = "creatorContentType"

#: The Shorts-feed traffic source value ``insightTrafficSourceType`` reports.
SHORTS_TRAFFIC_SOURCE = "SHORTS"

_DIMENSIONS = (TRAFFIC_SOURCE_DIMENSION, CONTENT_TYPE_DIMENSION)


def _store_path(brand: str, root: Optional[Path] = None) -> Path:
    """``<root>/<brand>/tracking/channel_daily.json``."""
    base = Path(root) if root else gms_output_dir()
    return base / brand / "tracking" / "channel_daily.json"


def load_rows(brand: str, root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every stored row for ``brand``: one per ``(day, dimension, value)``.

    Returns ``[]`` when nothing has been pulled yet, or the store is
    unreadable -- a missing/corrupt channel-daily store must never abort a
    caller that only wants what IS there.
    """
    path = _store_path(brand, root=root)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load channel_daily store at %s: %s", path, exc)
        return []
    rows = data.get("rows") if isinstance(data, dict) else None
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _save_rows(brand: str, rows: List[Dict[str, Any]], root: Optional[Path] = None) -> Path:
    path = _store_path(brand, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"brand": brand, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _row_key(row: Dict[str, Any]) -> tuple:
    return (row.get("day"), row.get("dimension"), row.get("value"))


def upsert_rows(
    existing: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge ``new_rows`` into ``existing``, replacing a stored row sharing
    the same ``(day, dimension, value)`` key rather than duplicating it --
    re-pulling a day (a partial-day re-pull, a backfill re-run) is
    idempotent and can never double-count."""
    by_key = {_row_key(r): r for r in existing}
    for row in new_rows:
        by_key[_row_key(row)] = row
    return sorted(
        by_key.values(),
        key=lambda r: (r.get("day") or "", r.get("dimension") or "", r.get("value") or ""),
    )


def pull_channel_daily(
    youtube_collector,
    brand: str,
    start_date: str,
    end_date: str,
    root: Optional[Path] = None,
) -> int:
    """Pull daily channel-level views by traffic source and content type
    for ``brand`` between ``start_date``/``end_date`` (``YYYY-MM-DD``,
    inclusive), and upsert them into the brand's channel-daily store.

    ``youtube_collector`` is a `collector.YouTubeCollector` (or anything
    exposing its ``fetch_channel_daily_dimension`` method) -- this reuses
    that collector's already-authenticated Analytics v2 client rather than
    building a second one: there is exactly one YouTube auth path in this
    service.

    Returns the number of rows pulled from the API this call (not
    necessarily the number newly stored -- a re-pulled day replaces its
    existing row rather than adding one).
    """
    new_rows: List[Dict[str, Any]] = []
    for dimension in _DIMENSIONS:
        for day, value, views in youtube_collector.fetch_channel_daily_dimension(
            dimension, start_date, end_date
        ):
            new_rows.append({"day": day, "dimension": dimension, "value": value, "views": views})
    if not new_rows:
        return 0
    existing = load_rows(brand, root=root)
    merged = upsert_rows(existing, new_rows)
    _save_rows(brand, merged, root=root)
    return len(new_rows)


def shorts_feed_share(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Per-day Shorts-feed share of channel views: SHORTS-traffic-source
    views divided by that day's total traffic-source views.

    A pure function over `load_rows`'s own row shape -- no I/O, no API
    access -- so a consumer (the social-reach surfaces) can derive the
    share from whatever is already stored without depending on when a pull
    last ran. A day with zero total traffic-source views is omitted, never
    reported as a fabricated ``0.0`` share.
    """
    totals: Dict[str, int] = defaultdict(int)
    shorts_views: Dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("dimension") != TRAFFIC_SOURCE_DIMENSION:
            continue
        views = row.get("views")
        day = row.get("day")
        if views is None or not day:
            continue
        totals[day] += views
        if row.get("value") == SHORTS_TRAFFIC_SOURCE:
            shorts_views[day] += views
    return {day: shorts_views.get(day, 0) / total for day, total in totals.items() if total}
