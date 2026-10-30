"""Interim account-level daily Facebook reel-plays series.

No per-day Facebook series is stored anywhere (``PlatformMetrics.daily`` is
empty for facebook); what exists is the nightly cumulative observation log,
``short.observation_log["facebook"]``. This derives a daily series from it
for ``suppression.classify_facebook_reel_reach``:

  * per post, the delta of ``cumulative.reel_total_plays`` between
    consecutive MEASURED entries, attributed to the New York calendar day of
    the later entry's ``fetched_at``; a counter that moved backwards skips
    that interval (as a counter reset);
  * a post's first observation counts its full cumulative on its fetch day
    only when ``event_at`` (post time) is within 1 day before ``fetched_at``,
    otherwise it is a baseline only;
  * summed across all shorts. Days with no observation are absent (never a
    fabricated 0), so gaps shrink the detector's trailing windows.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

from .observation import ObservationStatus

_FACEBOOK = "facebook"
_COUNTER = "reel_total_plays"
_FRESH_POST_WINDOW = timedelta(days=1)
_DEFAULT_TZ = ZoneInfo("America/New_York")


def _parse(ts: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(ts)) if ts else None
    except ValueError:
        return None
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def facebook_reel_plays_series(
    records: Iterable[Any], *, tz: tzinfo = _DEFAULT_TZ
) -> Dict[str, float]:
    """``{YYYY-MM-DD: reel plays that day}`` from every record's facebook log."""
    series: Dict[str, float] = defaultdict(float)
    for record in records:
        for short in getattr(record, "shorts", None) or []:
            by_post: Dict[str, List[Tuple[datetime, Optional[datetime], float]]] = defaultdict(list)
            for entry in (short.observation_log or {}).get(_FACEBOOK) or []:
                if entry.get("status") != ObservationStatus.MEASURED:
                    continue
                total = (entry.get("cumulative") or {}).get(_COUNTER)
                fetched = _parse(entry.get("fetched_at"))
                if total is None or fetched is None:
                    continue
                by_post[str(entry.get("post_id"))].append(
                    (fetched, _parse(entry.get("event_at")), float(total))
                )
            for obs in by_post.values():
                obs.sort(key=lambda o: o[0])
                first_fetched, event, first_total = obs[0]
                if (
                    event is not None
                    and timedelta(0) <= first_fetched - event <= _FRESH_POST_WINDOW
                ):
                    series[first_fetched.astimezone(tz).date().isoformat()] += first_total
                for (_, _, prev), (fetched, _, cur) in zip(obs, obs[1:]):
                    if cur < prev:
                        continue
                    series[fetched.astimezone(tz).date().isoformat()] += cur - prev
    return dict(series)
