"""How far the channel is from each YouTube Partner Program tier.

Nothing else in GMS answers the one question monetization hangs on: how many
subscribers, public long-form watch hours and Shorts views the channel has
against what YouTube asks for. This module holds the thresholds and a pure
comparison; :func:`fetch_channel_counts` is the only live part and reuses
``YouTubeCollector``'s own clients (no second auth path).

Thresholds are YouTube's published numbers, not tuned here:

* support.google.com/youtube/answer/72851 (read 2026-11-25): ad revenue needs
  1,000 subscribers and 4,000 public long-form watch hours in 12 months, or
  10 million Shorts views in 90 days; the page says its terms change on
  2027-02-01.
* newscaststudio.com, 2026-08-10, reporting YouTube's announcement: from
  2027-02-01 a NEW applicant needs 8,000 hours or 20 million Shorts views;
  the fan-funding tier stays at 500 subscribers, 3 public uploads in 90 days,
  and 3,000 hours or 3 million Shorts views.

The live counts are APPROXIMATE, and every line says so. The Analytics API
counts minutes on private and unlisted views and cannot exclude ad-campaign
views, which YouTube's qualifying hours leave out; Shorts views here are all
views YouTube attributes to Shorts, not only Shorts-feed views. Studio's Earn
tab is the authoritative number. A count nobody measured is ``None`` and prints
``unmeasured`` -- never 0, the same rule the rest of ``tracking`` follows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: The day YouTube's raised entry thresholds apply to new applicants.
NEW_TERMS_DATE = date(2027, 2, 1)

#: Days the watch-hour and Shorts-view windows span.
WATCH_HOURS_WINDOW_DAYS = 365
SHORTS_VIEWS_WINDOW_DAYS = 90
UPLOADS_WINDOW_DAYS = 90

#: ``creatorContentType`` values whose minutes count as long-form watch time.
#: Spelled as developers.google.com/youtube/analytics/dimensions lists them
#: (read 2026-11-27); every comparison goes through
#: :func:`normalize_content_type`, so ``videoOnDemand`` or ``video_on_demand``
#: match too.
LONG_FORM_CONTENT_TYPES = frozenset({"VIDEO_ON_DEMAND", "LIVE_STREAM"})
SHORTS_CONTENT_TYPE = "SHORTS"
#: Every value the dimensions page lists. A row outside this set is logged.
KNOWN_CONTENT_TYPES = LONG_FORM_CONTENT_TYPES | {SHORTS_CONTENT_TYPE, "STORY", "UNSPECIFIED"}


def normalize_content_type(value: Any) -> str:
    """One spelling for a ``creatorContentType`` value: lower case, no
    underscores, hyphens or spaces (``VIDEO_ON_DEMAND`` and ``videoOnDemand``
    both become ``videoondemand``)."""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


_LONG_FORM_KEYS = frozenset(normalize_content_type(v) for v in LONG_FORM_CONTENT_TYPES)
_SHORTS_KEY = normalize_content_type(SHORTS_CONTENT_TYPE)
_KNOWN_KEYS = frozenset(normalize_content_type(v) for v in KNOWN_CONTENT_TYPES)


@dataclass(frozen=True)
class Tier:
    name: str
    subscribers: int
    watch_hours: int
    shorts_views: int
    uploads_90d: Optional[int] = None  # None: the tier asks for no upload count


def tiers_for(today: date) -> List[Tier]:
    """The tiers a NEW applicant faces on ``today``."""
    raised = today >= NEW_TERMS_DATE
    return [
        Tier("fan_funding", 500, 3_000, 3_000_000, uploads_90d=3),
        Tier(
            "ad_revenue",
            1_000,
            8_000 if raised else 4_000,
            20_000_000 if raised else 10_000_000,
        ),
    ]


@dataclass(frozen=True)
class ChannelCounts:
    subscribers: Optional[int] = None
    long_form_watch_hours_365d: Optional[float] = None
    shorts_views_90d: Optional[int] = None
    public_uploads_90d: Optional[int] = None


@dataclass(frozen=True)
class TierProgress:
    tier: Tier
    met: Optional[bool]  # None: a count it needs was not measured
    gaps: Dict[str, Optional[float]]  # what is still missing per requirement; 0 = met

    def line(self, counts: ChannelCounts) -> str:
        def show(value: Any) -> str:
            if value is None:
                return "unmeasured"
            return f"{value:.0f}" if isinstance(value, float) else str(value)

        t = self.tier
        verdict = "unknown" if self.met is None else ("yes" if self.met else "no")
        parts = [
            f"PARTNER_PROGRESS tier={t.name} met={verdict}",
            f"subs={show(counts.subscribers)}/{t.subscribers}",
            f"watch_hours_365d={show(counts.long_form_watch_hours_365d)}/{t.watch_hours}",
            f"shorts_views_90d={show(counts.shorts_views_90d)}/{t.shorts_views}",
        ]
        if t.uploads_90d is not None:
            parts.append(f"uploads_90d={show(counts.public_uploads_90d)}/{t.uploads_90d}")
        parts.append("approximate=yes")
        return " ".join(parts)


def _gap(have: Optional[float], need: int) -> Optional[float]:
    return None if have is None else max(0.0, float(need) - float(have))


def tier_progress(tier: Tier, counts: ChannelCounts) -> TierProgress:
    """Compare ``counts`` with one tier. Either watch hours OR Shorts views
    satisfies the viewing requirement, as YouTube states it."""
    gaps: Dict[str, Optional[float]] = {
        "subscribers": _gap(counts.subscribers, tier.subscribers),
        "watch_hours": _gap(counts.long_form_watch_hours_365d, tier.watch_hours),
        "shorts_views": _gap(counts.shorts_views_90d, tier.shorts_views),
    }
    if tier.uploads_90d is not None:
        gaps["uploads"] = _gap(counts.public_uploads_90d, tier.uploads_90d)

    viewing = [gaps["watch_hours"], gaps["shorts_views"]]
    if any(g == 0 for g in viewing if g is not None):
        viewing_met: Optional[bool] = True
    elif all(g is not None for g in viewing):
        viewing_met = False
    else:
        viewing_met = None

    others = [g for k, g in gaps.items() if k not in ("watch_hours", "shorts_views")]
    if any(g is not None and g > 0 for g in others) or viewing_met is False:
        met: Optional[bool] = False
    elif any(g is None for g in others) or viewing_met is None:
        met = None
    else:
        met = True
    return TierProgress(tier, met, gaps)


def deadline_line(today: date) -> Optional[str]:
    """One line while the old thresholds still apply; None once they don't."""
    if today >= NEW_TERMS_DATE:
        return None
    days = (NEW_TERMS_DATE - today).days
    return (
        f"PARTNER_DEADLINE new_terms={NEW_TERMS_DATE.isoformat()} days_left={days} "
        "ad_revenue_after=8000h_or_20000000_shorts_views (new applicants only)"
    )


def report_lines(counts: ChannelCounts, today: date) -> List[str]:
    lines = [tier_progress(t, counts).line(counts) for t in tiers_for(today)]
    deadline = deadline_line(today)
    if deadline:
        lines.append(deadline)
    return lines


def count_public_uploads(records: List[Any], today: date) -> Optional[int]:
    """Uploads in the last 90 days from the tracking store: shorts with a
    YouTube id and a posted day, plus long-forms with a published day. ``None``
    when the store holds no dated YouTube upload at all (unknown, not zero)."""
    since = today - timedelta(days=UPLOADS_WINDOW_DAYS)
    seen_any = False
    count = 0
    for record in records:
        for short in getattr(record, "shorts", []) or []:
            if not (short.platform_ids or {}).get("youtube") or not short.posted_at:
                continue
            seen_any = True
            try:
                if date.fromisoformat(str(short.posted_at)[:10]) >= since:
                    count += 1
            except ValueError:
                continue
        for lf in getattr(record, "long_forms", []) or []:
            if not lf.video_id or not lf.published_date:
                continue
            seen_any = True
            try:
                if date.fromisoformat(str(lf.published_date)[:10]) >= since:
                    count += 1
            except ValueError:
                continue
    return count if seen_any else None


def counts_from_rows(rows: List[List[Any]]) -> Dict[str, Optional[float]]:
    """``(creatorContentType, views, estimatedMinutesWatched)`` rows to totals.
    An empty answer is unmeasured (None), never zero.

    Content types are compared through :func:`normalize_content_type`, so the
    case and underscores the API sends do not matter. Rows whose content type
    is none the dimensions page lists are logged; when no row at all is a
    known type, the answer is logged as an error and reported unmeasured
    rather than as 0 hours and 0 views."""
    if not rows:
        return {"long_form_minutes": None, "shorts_views": None}
    minutes = 0.0
    shorts = 0
    unknown: List[str] = []
    for row in rows:
        raw, views, mins = row[0], row[1], row[2]
        kind = normalize_content_type(raw)
        if kind not in _KNOWN_KEYS:
            unknown.append(str(raw))
            continue
        if kind in _LONG_FORM_KEYS and mins is not None:
            minutes += float(mins)
        if kind == _SHORTS_KEY and views is not None:
            shorts += int(views)
    if len(unknown) == len(rows):
        logger.error(
            "PARTNER_PROGRESS_UNMAPPED rows=%d values=%s: no creatorContentType value "
            "matched a known type; reporting watch hours and Shorts views as unmeasured",
            len(rows),
            sorted(set(unknown)),
        )
        return {"long_form_minutes": None, "shorts_views": None}
    if unknown:
        logger.warning(
            "partner-progress: ignored rows with unknown creatorContentType values %s",
            sorted(set(unknown)),
        )
    return {"long_form_minutes": minutes, "shorts_views": shorts}


def fetch_channel_counts(collector: Any, records: List[Any], today: date) -> ChannelCounts:
    """Live counts through ``collector`` (a ``YouTubeCollector``). Every
    failure degrades that one count to ``None`` and is logged; this never raises."""
    subscribers: Optional[int] = None
    try:
        service = collector._get_service()
        resp = service.channels().list(part="statistics", mine=True).execute()
        items = resp.get("items") or []
        if items and not items[0].get("statistics", {}).get("hiddenSubscriberCount"):
            subscribers = int(items[0]["statistics"]["subscriberCount"])
    except Exception as exc:  # noqa: BLE001 - one count failing never sinks the rest
        logger.warning("partner-progress: subscriber count unavailable: %s", exc)

    def window(days: int) -> Dict[str, Optional[float]]:
        try:
            analytics = collector._analytics_client()
            if analytics is None:
                return {"long_form_minutes": None, "shorts_views": None}
            resp = (
                analytics.reports()
                .query(
                    ids="channel==MINE",
                    startDate=(today - timedelta(days=days)).isoformat(),
                    endDate=today.isoformat(),
                    dimensions="creatorContentType",
                    metrics="views,estimatedMinutesWatched",
                )
                .execute()
            )
            return counts_from_rows(resp.get("rows") or [])
        except Exception as exc:  # noqa: BLE001
            collector._note_analytics_exception(exc)
            logger.warning("partner-progress: %s-day window unavailable: %s", days, exc)
            return {"long_form_minutes": None, "shorts_views": None}

    year = window(WATCH_HOURS_WINDOW_DAYS)
    quarter = window(SHORTS_VIEWS_WINDOW_DAYS)
    minutes = year["long_form_minutes"]
    shorts = quarter["shorts_views"]
    return ChannelCounts(
        subscribers=subscribers,
        long_form_watch_hours_365d=None if minutes is None else minutes / 60.0,
        shorts_views_90d=None if shorts is None else int(shorts),
        public_uploads_90d=count_public_uploads(records, today),
    )
