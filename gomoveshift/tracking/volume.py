"""Library hygiene: the weekly, suggest-only list of low-viewed public shorts.

Nothing in GMS recommends a shorts volume: the SEO service decides it from the
evidence export (``gomoveshift metrics export-evidence``). What remains is the
selector below, which reads ``TrackingStore`` records and never changes a privacy setting itself.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import List, Optional

from .store import TrackingStore  # module scope, not lazy, so tests can patch it

#: Library hygiene: a weekly, suggest-only digest list of public shorts old enough
#: and low-viewed enough,
#: in a series large enough, to be worth setting private (``gomoveshift
#: library-hygiene --apply``, ``cli/handlers/library_hygiene.py``, holds the adapter).
#: Never a privacy CHANGE here — this module holds no publishing/adapter import and
#: makes no network call; only ``select_library_hygiene_candidates`` below
#: decides what to print, and it only reads its ``shorts`` argument.
DEFAULT_MIN_AGE_DAYS = 30
DEFAULT_MAX_VIEWS = 100
DEFAULT_MIN_SERIES_SIZE = 5
DEFAULT_HYGIENE_BATCH_SIZE = 50


@dataclass(frozen=True)
class TrackedShort:
    """One short as the hygiene selector sees it. ``series`` is the grouping
    batches are built by (e.g. "County Road 12") — here, the tracked
    location string (``TrackingRecord.location``), the only place-grouping a
    ``TrackingRecord`` carries today (see ``tracking/schema.py``). ``privacy``
    is a plain string the caller supplies; this module never inspects a
    platform API to get one, and a caller with no known privacy status
    should pass ``None`` — never guess ``"public"``."""

    id: str
    published_at: date
    privacy: Optional[str]
    views: int
    series: str


@dataclass(frozen=True)
class HygieneCandidate:
    id: str
    series: str
    views: int
    age_days: int

    def log_line(self) -> str:
        return (
            f"LIBRARY_HYGIENE_CANDIDATE id={self.id} series={self.series!r} "
            f"views={self.views} age_days={self.age_days}"
        )


def select_library_hygiene_candidates(
    shorts: List[TrackedShort],
    today: date,
    *,
    min_age_days: int = DEFAULT_MIN_AGE_DAYS,
    max_views: int = DEFAULT_MAX_VIEWS,
    min_series_size: int = DEFAULT_MIN_SERIES_SIZE,
    batch_size: int = DEFAULT_HYGIENE_BATCH_SIZE,
) -> List[List[HygieneCandidate]]:
    """Public, low-view, old shorts in a heavy series — batched, heaviest
    series first. Pure and read-only: never calls any
    privacy-changing API; ``library-hygiene --apply`` acts on the result.

    A short qualifies only when ``privacy == "public"`` exactly (``None`` or
    any other value is excluded, never assumed public), it is strictly older
    than ``min_age_days``, its views are strictly under ``max_views``, and its
    series has strictly more than ``min_series_size`` qualifying members.
    Series size is counted only over shorts that already pass the other
    filters -- a series is heavy because of how many LOW-VIEW old shorts it holds,
    not its total size."""
    eligible = [
        s
        for s in shorts
        if s.privacy == "public"
        and (today - s.published_at).days > min_age_days
        and s.views < max_views
    ]
    series_sizes = Counter(s.series for s in eligible)
    qualifying = [s for s in eligible if series_sizes[s.series] > min_series_size]
    qualifying.sort(key=lambda s: (-series_sizes[s.series], s.series, s.views, s.id))
    candidates = [
        HygieneCandidate(
            id=s.id,
            series=s.series,
            views=s.views,
            age_days=(today - s.published_at).days,
        )
        for s in qualifying
    ]
    if not candidates:
        return []
    return [candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)]


def _tracked_shorts_from_records(records) -> List[TrackedShort]:
    """Build ``TrackedShort`` rows from the tracking store's own records.

    ``ShortRecord`` (``tracking/schema.py``) carries no privacy field today --
    ``getattr(short, "privacy_status", None)`` reads it only once the schema
    grows one. Until then this always yields ``None``, which
    ``select_library_hygiene_candidates``'s own ``privacy == "public"`` check
    excludes -- never guessed as public from silence."""
    shorts: List[TrackedShort] = []
    for record in records:
        series = record.location or record.project
        for short in record.shorts:
            if not short.posted_at:
                continue
            try:
                published = date.fromisoformat(short.posted_at[:10])
            except ValueError:
                continue
            shorts.append(
                TrackedShort(
                    id=f"{record.project}#{short.index}",
                    published_at=published,
                    privacy=getattr(short, "privacy_status", None),
                    views=short.total_views,
                    series=series,
                )
            )
    return shorts


def library_hygiene_report(today: Optional[date] = None) -> str:
    """The digest-facing entry point (``python3 -m gomoveshift.tracking.volume
    library-hygiene``): read the tracking store, run the pure selector, and
    render plain lines -- or say plainly why there is nothing to show.
    An external digest script only omits a section that returns empty, so a real
    gap (no records / no privacy data tracked yet) must say so in words rather
    than read as a clean week."""
    today = today or date.today()
    store = TrackingStore()
    records = store.load_all_records()
    if not records:
        return "LIBRARY_HYGIENE: no tracking records yet"
    shorts = _tracked_shorts_from_records(records)
    if not any(s.privacy == "public" for s in shorts):
        return (
            "LIBRARY_HYGIENE: no candidates -- no short in the tracking store carries a "
            "privacy status yet, so none can be confirmed public (never assumed)"
        )
    batches = select_library_hygiene_candidates(shorts, today)
    if not batches:
        return "LIBRARY_HYGIENE: no candidates"
    return "\n".join(c.log_line() for batch in batches for c in batch)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry an external digest script can call directly (``python3 -m
    gomoveshift.tracking.volume library-hygiene``) -- there is no
    ``cli/handlers/`` file for this module."""
    argv = list(argv if argv is not None else sys.argv[1:])
    if argv[:1] != ["library-hygiene"]:
        print("usage: python3 -m gomoveshift.tracking.volume library-hygiene", file=sys.stderr)
        return 2
    print(library_hygiene_report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
