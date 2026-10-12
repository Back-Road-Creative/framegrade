"""Day-based grouping for the day-first pipeline architecture.

First step of the day-first design. The grouper takes an `IngestPlan`
produced by Phase A and splits its episodes into per-day batches so
downstream phases (super LF encoding, regular LF extraction, shorts,
metadata) can process each day independently.

Locked behavior:
  * A "day" is the calendar date of the episode's GPX first fix, converted
    to longitude-local time — the same satellite-UTC clock every LOCATION
    surface (county vote, chapters, title) already keys off. The naive
    container `creation_time` (camera wall clock — GoPro writes local,
    generic-mode sources write UTC) is only a fallback when no GPX is
    available. Episodes that start on Feb 20 at 23:45 and run 30 minutes
    into Feb 21 stay with Feb 20 — they are not split.
  * Day batches are returned in chronological order (oldest day first).
  * Episodes within a batch preserve their original order from the ingest
    plan (which is chronological within a day).
  * Episodes without a resolvable date are skipped with a debug log, not
    raised — the grouper must be resilient to partial data from upstream.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from gps_journey.core.models import to_local

logger = logging.getLogger(__name__)


@dataclass
class DayBatch:
    """All episodes from a single calendar date, in chronological order.

    Downstream phases consume this: concatenate `concat_paths` into a
    raw super LF, merge `gpx_paths` for the day's GPS track, derive OST
    from the full day's waypoints, etc.
    """

    date: str  # YYYY-MM-DD
    episodes: list = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        """Total seconds across all episodes in the day."""
        return sum(float(getattr(ep, "total_duration", 0.0) or 0.0) for ep in self.episodes)

    @property
    def total_duration_min(self) -> float:
        return self.total_duration / 60.0

    @property
    def start_time(self) -> Optional[datetime]:
        """Creation time of the first episode in the day (chronologically)."""
        if not self.episodes:
            return None
        return getattr(self.episodes[0], "creation_time", None)

    @property
    def end_time(self) -> Optional[datetime]:
        """Estimated end time of the last episode (start + duration)."""
        if not self.episodes:
            return None
        last = self.episodes[-1]
        start = getattr(last, "creation_time", None)
        if start is None:
            return None
        duration = float(getattr(last, "total_duration", 0.0) or 0.0)
        return start + timedelta(seconds=duration)

    @property
    def gpx_paths(self) -> list:
        """Ordered list of GPX file paths, skipping any None entries."""
        return [ep.gpx_path for ep in self.episodes if getattr(ep, "gpx_path", None) is not None]

    @property
    def concat_paths(self) -> list:
        """Ordered list of concatenated episode file paths, skipping None."""
        return [
            ep.concat_path for ep in self.episodes if getattr(ep, "concat_path", None) is not None
        ]

    def summary(self) -> str:
        return (
            f"Day {self.date}: {len(self.episodes)} episodes, "
            f"{self.total_duration_min:.0f} min total"
        )


# Shifting the GPX first fix's satellite-UTC timestamp to longitude-local time
# (``to_local``, one hour per 15° of longitude) yields the drive's local
# wall-clock date — the calendar day an operator (and the journal / description
# / social date) would call it. This is robust against camera-clock timezone
# quirks the naive container creation_time can't distinguish (GoPro writes local
# wall-clock; generic-mode Insta360/DJI/phone sources write true UTC), which
# otherwise deterministically misgroup a US-East evening drive into the next
# calendar day. The whole-hour shift is a DST-blind, political-boundary-blind
# approximation of the real UTC offset, though: within ~2 h of local midnight it
# can still tip the day, so ``_gpx_local_date_key`` warns there below.

# Warn when the first fix lands this close to local midnight: the ~1 h slack in
# the longitude→offset approximation can flip the calendar day.
_NEAR_MIDNIGHT_MINUTES = 120


def _gpx_first_fix(gpx_path: Any) -> Optional[tuple[datetime, float]]:
    """Return ``(utc_timestamp, longitude)`` of the GPX's first valid fix.

    Parses the GPX lazily via ``gpxpy``; returns None on any failure or
    when no fix carries both a timestamp and a plausible (non-null-island)
    longitude. Never raises — a missing/garbage GPX must degrade to the
    creation_time fallback, not crash the grouper.
    """
    if not gpx_path:
        return None
    try:
        import gpxpy

        with open(gpx_path, encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
        for track in gpx.tracks:
            for segment in track.segments:
                for pt in segment.points:
                    if pt.time is None or pt.longitude is None:
                        continue
                    # Skip null-island sentinels written on GPS lock loss.
                    if abs(pt.latitude or 0.0) < 0.001 and abs(pt.longitude) < 0.001:
                        continue
                    return pt.time, pt.longitude
    except Exception as e:
        logger.debug("GPX first-fix parse failed for %s: %s", gpx_path, e)
    return None


def _gpx_local_date_key(episode: Any) -> Optional[str]:
    """Day key from the episode's GPX first fix, shifted to longitude-local time.

    Warns when the first fix lands within ``_NEAR_MIDNIGHT_MINUTES`` of local
    midnight: the whole-hour longitude→offset approximation can tip the day
    there, so the operator should confirm the day's LF / county / journal date.
    """
    fix = _gpx_first_fix(getattr(episode, "gpx_path", None))
    if fix is None:
        return None
    ts, lon = fix
    local = to_local(ts, lon)
    minutes_into_day = local.hour * 60 + local.minute
    if min(minutes_into_day, 1440 - minutes_into_day) < _NEAR_MIDNIGHT_MINUTES:
        logger.warning(
            "NEAR-MIDNIGHT day key: episode %r first fix maps to %s %02d:%02d "
            "local, within %d min of midnight — the whole-hour longitude→"
            "timezone approximation may tip the calendar day. Verify the day's "
            "LF / county / journal date.",
            getattr(episode, "index", "?"),
            local.strftime("%Y-%m-%d"),
            local.hour,
            local.minute,
            _NEAR_MIDNIGHT_MINUTES,
        )
    return local.strftime("%Y-%m-%d")


def _episode_date_key(episode: Any) -> Optional[str]:
    """Extract a 'YYYY-MM-DD' date key for an episode.

    Prefers the GPX first-fix timestamp converted to longitude-local time
    (satellite-UTC → the drive's local calendar day). Falls back to the
    container `creation_time` (naive camera wall clock) and finally the
    `.date` string when no GPX is available. Returns None when no date
    information is available at all.

    When both a GPX-local date and a creation_time date exist and they
    disagree, logs a loud warning — the camera clock's timezone is skewing
    the naive key (a US-East evening drive groups into the next calendar
    day otherwise). The GPX-local date wins.
    """
    ct = getattr(episode, "creation_time", None)
    ct_key = ct.strftime("%Y-%m-%d") if isinstance(ct, datetime) else None

    gpx_key = _gpx_local_date_key(episode)
    if gpx_key is not None:
        if ct_key is not None and ct_key != gpx_key:
            logger.warning(
                "CLOCK-DISAGREEMENT: episode %r GPX-local day %s != container "
                "creation_time day %s — using GPX-local (camera-clock timezone "
                "skew). Verify the day's LF / county / journal date.",
                getattr(episode, "index", "?"),
                gpx_key,
                ct_key,
            )
        return gpx_key

    if ct_key is not None:
        return ct_key

    # Fall back to the .date string (already populated by Phase A ingest)
    date_str = getattr(episode, "date", None)
    if isinstance(date_str, str) and date_str.strip():
        return date_str.strip()

    return None


def group_episodes_by_day(ingest_plan: Any) -> list[DayBatch]:
    """Split an IngestPlan's episodes into per-day batches.

    Args:
        ingest_plan: an object with an `episodes` attribute that is an
            iterable of Episode-like objects. Each episode should have
            `creation_time` (datetime) or `date` (YYYY-MM-DD string),
            plus `total_duration`, `gpx_path`, and `concat_path` fields
            for downstream consumption.

    Returns:
        A list of `DayBatch` objects, one per calendar date, sorted
        chronologically. Each batch's `episodes` list preserves the
        original chronological order from the ingest plan.
    """
    episodes = getattr(ingest_plan, "episodes", None) or []
    if not episodes:
        return []

    # Bucket episodes by date key, preserving input order within each bucket.
    by_date: dict[str, list] = {}
    for episode in episodes:
        date_key = _episode_date_key(episode)
        if date_key is None:
            logger.debug(
                "Skipping episode %r — no creation_time or date available",
                getattr(episode, "index", "?"),
            )
            continue
        by_date.setdefault(date_key, []).append(episode)

    # Sort each bucket's episodes by creation_time to guarantee chronological
    # order even if the input wasn't sorted. Episodes without creation_time
    # sort by their original input position (stable sort fallback).
    def _sort_key(ep):
        ct = getattr(ep, "creation_time", None)
        if isinstance(ct, datetime):
            return (0, ct)
        return (1, 0)  # unordered — stable-sort keeps input order

    for date_key in by_date:
        by_date[date_key].sort(key=_sort_key)

    # Return batches sorted by date (chronologically, lexicographic works for
    # YYYY-MM-DD).
    return [DayBatch(date=date_key, episodes=eps) for date_key, eps in sorted(by_date.items())]
