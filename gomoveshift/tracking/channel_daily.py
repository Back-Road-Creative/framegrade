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

Facebook Page-level daily insights share the store under a
``facebook:<metric>`` dimension (value ``"page"``), labelled by the Meta
(Pacific) day: `pull_facebook_page_daily` self-backfills each metric from
`FACEBOOK_BACKFILL_START` and then re-pulls a short settle window nightly.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..paths import gms_output_dir

logger = logging.getLogger(__name__)

#: YouTube Analytics v2 dimension names this pull persists one row set per.
TRAFFIC_SOURCE_DIMENSION = "insightTrafficSourceType"
CONTENT_TYPE_DIMENSION = "creatorContentType"

#: The Shorts-feed traffic source value ``insightTrafficSourceType`` reports.
SHORTS_TRAFFIC_SOURCE = "SHORTS"

_DIMENSIONS = (TRAFFIC_SOURCE_DIMENSION, CONTENT_TYPE_DIMENSION)

#: Page-level daily metrics tried, each on its own request: Meta retires
#: insight names without notice, so one refused name must not sink the rest.
#: Which ones answer is printed nightly (``FB_PAGE_DAILY``).
FACEBOOK_PAGE_METRICS = (
    "page_media_view",
    "page_total_media_view_unique",
    "page_video_views",
    "page_impressions_unique",
)
#: Order the suppression detector tries the Page metrics in: unique viewers is
#: closest to reach, so it leads; the rest follow by how directly they count views.
FACEBOOK_DETECTOR_PREFERENCE = (
    "page_total_media_view_unique",
    "page_media_view",
    "page_video_views",
    "page_impressions_unique",
)
#: First day an empty store backfills from.
FACEBOOK_BACKFILL_START = "2026-06-01"
#: Days before the last stored day that are re-pulled while Meta settles them.
_FACEBOOK_SETTLE_DAYS = 3
#: Longest ``since``..``until`` span one Page-insights request is sent.
_FACEBOOK_WINDOW_DAYS = 90


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


def _facebook_dimension(metric: str) -> str:
    return f"facebook:{metric}"


def _facebook_windows(start: date, until: date) -> List[tuple[str, str]]:
    """``[since, until)`` spans tiling ``start``..``until``, none longer
    than `_FACEBOOK_WINDOW_DAYS`."""
    windows: List[tuple[str, str]] = []
    while start < until:
        end = min(start + timedelta(days=_FACEBOOK_WINDOW_DAYS), until)
        windows.append((start.isoformat(), end.isoformat()))
        start = end
    return windows


def pull_facebook_page_daily(
    facebook_collector,
    brand: str,
    today: date,
    root: Optional[Path] = None,
) -> Dict[str, Optional[int]]:
    """Pull each of `FACEBOOK_PAGE_METRICS` through yesterday and upsert the
    days into ``brand``'s channel-daily store.

    A metric with nothing stored backfills from `FACEBOOK_BACKFILL_START`;
    otherwise it re-pulls from `_FACEBOOK_SETTLE_DAYS` before its last stored
    day. ``facebook_collector`` is a `collector.FacebookCollector` (anything
    with ``fetch_page_daily`` and ``last_failure_detail``).

    Returns rows pulled per metric, or ``None`` for a metric Graph refused:
    that metric stops at its first refused window and is never stored as 0.
    """
    existing = load_rows(brand, root=root)
    backfill = date.fromisoformat(FACEBOOK_BACKFILL_START)
    status: Dict[str, Optional[int]] = {}
    new_rows: List[Dict[str, Any]] = []
    for metric in FACEBOOK_PAGE_METRICS:
        days = list(facebook_page_series(existing, metric))
        start = backfill
        if days:
            start = max(
                backfill, date.fromisoformat(max(days)) - timedelta(days=_FACEBOOK_SETTLE_DAYS)
            )
        pulled: Optional[int] = 0
        for since, until in _facebook_windows(start, today):
            points = facebook_collector.fetch_page_daily(metric, since, until)
            if points is None:
                logger.warning(
                    "Facebook page metric %s unavailable: %s",
                    metric,
                    facebook_collector.last_failure_detail or "no answer",
                )
                pulled = None
                break
            pulled = (pulled or 0) + len(points)
            new_rows.extend(
                {"day": day, "dimension": _facebook_dimension(metric), "value": "page", "views": v}
                for day, v in points
            )
        status[metric] = pulled
    if new_rows:
        _save_rows(brand, upsert_rows(existing, new_rows), root=root)
    return status


def facebook_page_series(rows: List[Dict[str, Any]], metric: str) -> Dict[str, int]:
    """``{day: value}`` for one Page metric from `load_rows`'s rows."""
    dimension = _facebook_dimension(metric)
    return {
        r["day"]: r["views"]
        for r in rows
        if r.get("dimension") == dimension and r.get("day") and isinstance(r.get("views"), int)
    }


def facebook_detector_series(
    rows: List[Dict[str, Any]], min_days: int
) -> Tuple[str, Dict[str, int]]:
    """``(metric, series)`` for the first `FACEBOOK_DETECTOR_PREFERENCE` metric
    whose stored series has MORE than ``min_days`` days, else ``("", {})``."""
    for metric in FACEBOOK_DETECTOR_PREFERENCE:
        series = facebook_page_series(rows, metric)
        if len(series) > min_days:
            return metric, series
    return "", {}
