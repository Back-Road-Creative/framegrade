"""EvidenceBundle v2 export: settled day-7 facts, no thresholds, no recommendation.

Every YouTube short with a settled Analytics series is a row with
``source=analytics_daily``; one without falls back to the measured snapshot
nearest age 7 days (within 12h) in ``observation_log`` with ``source=snapshot``;
one with neither is listed under ``health.excluded`` and counted.

Every Facebook short is a row from the MEASURED Facebook observation nearest
age 7 days (same 12h tolerance; an observation reporting no counter at all is
not a measurement) with ``source=snapshot``, or is listed under
``health.excluded`` with ``platform: facebook``. A row's ``day7`` maps counter
name to its day-7 value, null when the platform did not report it: Facebook
Reel plays go in ``reel_total_plays`` / ``reel_initial_plays`` and are never
written into ``views``. ``post_format`` is ``short`` for YouTube, the measured
observation's ``reel`` / ``video`` for Facebook, null when unknown.

A consumer decides what the numbers mean (which counter is reach) -- nothing
here ranks, filters or recommends. ``n_registered`` counts every short with an
id on either platform; ``n_with_series`` is the number of short rows.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from .day7 import posted_date, settled_day7_views
from .observation import ObservationStatus

SCHEMA_VERSION = 2
PLATFORM = "youtube"
FACEBOOK = "facebook"
_FACEBOOK_COUNTERS = (
    "views",
    "reel_total_plays",
    "reel_initial_plays",
    "watch_time_seconds",
)
SNAPSHOT_TARGET = timedelta(days=7)
SNAPSHOT_TOLERANCE = timedelta(hours=12)
_LOCAL_TZ = ZoneInfo("America/New_York")  # the plan builder's display timezone

SlotFor = Callable[[str, Any], Optional[Dict[str, Any]]]


def _utc(ts: Optional[str]) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(ts)) if ts else None
    except ValueError:
        return None
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def snapshot_day7_views(short) -> Optional[Any]:
    """Views of the MEASURED youtube observation nearest age 7d, or None when
    none lies within 12h of it."""
    best = None
    for entry in short.observation_log.get(PLATFORM) or []:
        if entry.get("status") != ObservationStatus.MEASURED:
            continue
        views = (entry.get("cumulative") or {}).get("views")
        event, fetched = _utc(entry.get("event_at")), _utc(entry.get("fetched_at"))
        if views is None or event is None or fetched is None:
            continue
        miss = abs(fetched - event - SNAPSHOT_TARGET)
        if miss <= SNAPSHOT_TOLERANCE and (best is None or miss < best[0]):
            best = (miss, views)
    return best[1] if best else None


def snapshot_day7_facebook(short) -> Optional[Dict[str, Any]]:
    """``{"post_format", "day7"}`` of the MEASURED facebook observation nearest
    age 7d, or None when none within 12h of it reports a counter."""
    best = None
    for entry in short.observation_log.get(FACEBOOK) or []:
        if entry.get("status") != ObservationStatus.MEASURED:
            continue
        cumulative = entry.get("cumulative") or {}
        day7 = {name: cumulative.get(name) for name in _FACEBOOK_COUNTERS}
        event, fetched = _utc(entry.get("event_at")), _utc(entry.get("fetched_at"))
        if all(v is None for v in day7.values()) or event is None or fetched is None:
            continue
        miss = abs(fetched - event - SNAPSHOT_TARGET)
        if miss <= SNAPSHOT_TOLERANCE and (best is None or miss < best[0]):
            best = (miss, {"post_format": entry.get("post_format") or None, "day7": day7})
    return best[1] if best else None


def _slot(posted_at: str, slot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """region / weekday / hour_local: the plan sidecar's slot when known, else
    the posting time in the display timezone with no region."""
    slot = slot or {}
    local = None
    parsed = _utc(posted_at)
    if parsed is not None:
        local = parsed.astimezone(_LOCAL_TZ)
    return {
        "region": slot.get("region"),
        "weekday": slot.get("weekday", local.weekday() if local else None),
        "hour_local": slot.get("hour_local", local.hour if local else None),
    }


def _row(vid, record, short, platform, post_format, day7, source, slot) -> Dict[str, Any]:
    return {
        "video_id": vid,
        "project": record.project,
        "region": slot["region"],
        "platform": platform,
        "post_format": post_format,
        "posted_at": short.posted_at,
        "weekday": slot["weekday"],
        "hour_local": slot["hour_local"],
        "day7": day7,
        "source": source,
    }


def build_evidence_bundle(
    records,
    status,
    now: datetime,
    slot_for: Optional[SlotFor] = None,
) -> Dict[str, Any]:
    """The EvidenceBundle v2 dict. ``now`` supplies ``as_of`` and today."""
    today = now.date()
    lookup = slot_for or (lambda project, item: None)
    shorts: List[Dict[str, Any]] = []
    longforms: List[Dict[str, Any]] = []
    excluded: List[Dict[str, str]] = []
    registered = 0
    for record in records:
        for short in record.shorts:
            slot = _slot(short.posted_at, lookup(record.project, short))
            vid = short.platform_ids.get(PLATFORM)
            if vid:
                registered += 1
                metrics = short.metrics.get(PLATFORM)
                views = settled_day7_views(short.posted_at, metrics.daily if metrics else [], today)
                source = "analytics_daily"
                if views is None:
                    views, source = snapshot_day7_views(short), "snapshot"
                if views is None:
                    excluded.append({"video_id": vid, "reason": "no_series"})
                else:
                    shorts.append(
                        _row(vid, record, short, PLATFORM, "short", {"views": views}, source, slot)
                    )
            fb_id = short.platform_ids.get(FACEBOOK)
            if fb_id:
                registered += 1
                fb = snapshot_day7_facebook(short)
                if fb is None:
                    excluded.append(
                        {"video_id": fb_id, "reason": "no_series", "platform": FACEBOOK}
                    )
                else:
                    shorts.append(
                        _row(
                            fb_id,
                            record,
                            short,
                            FACEBOOK,
                            fb["post_format"],
                            fb["day7"],
                            "snapshot",
                            slot,
                        )
                    )
        for lf in record.long_forms:
            m = lf.metrics.get(PLATFORM)
            posted = lf.published_date
            if not lf.video_id or not posted or m is None or lf.duration_sec is None:
                continue
            views = settled_day7_views(posted, m.daily, today)
            if views is None or posted_date(posted) is None:
                continue
            longforms.append(
                {
                    "video_id": lf.video_id,
                    "project": record.project,
                    "region": (lookup(record.project, lf) or {}).get("region"),
                    "posted_at": posted if len(posted) > 10 else f"{posted}T00:00:00",
                    "duration_sec": lf.duration_sec,
                    "day7_views": views,
                    "avg_view_percentage": m.avg_view_percentage,
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": now.replace(microsecond=0).isoformat(),
        "health": {
            "analytics_status": status.status,
            "n_registered": registered,
            "n_with_series": len(shorts),
            "excluded": excluded,
            "detail": status.detail or None,
        },
        "shorts": shorts,
        "longforms": longforms,
    }
