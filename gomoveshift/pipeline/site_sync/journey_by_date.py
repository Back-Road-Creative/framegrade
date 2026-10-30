"""Date-based journey resolution.

Resolves the journey slug for a drive run from the run's recording date by
scanning ``content/journeys/*.md`` for a trip whose ``trip.start_date`` ..
``trip.end_date`` range (inclusive) contains that date — no static
per-region config to go stale on every new trip. Cross-trip overlap, the
condition the old static map assumed away, is DETECTED here (refused) rather
than guessed at.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from .._atomic_write import atomic_write_text
from ..journal_publisher import (
    _day_number_for_new_date,
    journey_lock,
    load_journey,
    serialize_journey,
)


class AmbiguousJourneyError(ValueError):
    """Two or more journeys' trip ranges contain the run date.

    The date-resolution equivalent of the cross-trip overlap the static
    region map used to assume away by construction — now detected and
    refused instead of guessed at.
    """


class JourneyAlreadyExistsError(FileExistsError):
    """The slug derived from region+date already has a journey file.

    A deterministic slug colliding with an on-disk file that
    ``resolve_journey_by_date`` did NOT already match is anomalous — refuse
    rather than overwrite (or silently reuse) someone else's content.
    """


def _parse_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def resolve_journey_by_date(site_path: Path, run_date: str) -> str | None:
    """Return the single journey slug whose trip range contains ``run_date``.

    ``run_date`` is ``YYYY-MM-DD`` (the run's recording day, never today's
    wall clock). Returns ``None`` on zero matches — the caller keeps its
    existing loud warning. Raises :class:`AmbiguousJourneyError`, naming
    every matching slug, on two or more matches; never guesses between them.
    A missing journeys directory or an unparsable journey file is skipped,
    not fatal.
    """
    day = _parse_date(run_date)
    if day is None:
        return None
    journeys_dir = Path(site_path) / "content" / "journeys"
    if not journeys_dir.is_dir():
        return None

    matches: list[str] = []
    for path in sorted(journeys_dir.glob("*.md")):
        if path.stem == "_index":
            continue
        try:
            doc, _body = load_journey(path)
        except (OSError, ValueError):
            continue
        trip = doc.get("trip") or {}
        start = _parse_date(trip.get("start_date"))
        end = _parse_date(trip.get("end_date"))
        if start and end and start <= day <= end:
            matches.append(path.stem)

    if not matches:
        return None
    if len(matches) > 1:
        raise AmbiguousJourneyError(
            f"run date {run_date} falls within {len(matches)} journeys' trip "
            f"ranges ({', '.join(sorted(matches))}); refusing to guess. Set "
            "config.drive.active_journey_by_region or pass "
            "--auto-publish-journal SLUG for this run."
        )
    return matches[0]


def extend_adjacent_journey(site_path: Path, run_date: str) -> str | None:
    """Extend the one journey whose trip ended the day before ``run_date``.

    A multi-day trip is driven and processed one day at a time, so the second
    day's run finds a journey whose ``trip.end_date`` is the day before. That
    day belongs in the same journey, not in a new draft. The trip's
    ``end_date`` (and ``duration_days`` when present) move to ``run_date`` and
    an empty day stub is added in date order, as ``apply_merge`` would; the
    merge then lands the videos in it. Returns the extended slug, ``None``
    when no trip ends the day before. Only forward: a trip starting the day
    after is left alone, since prepending a day renumbers a published day 1.
    Raises :class:`AmbiguousJourneyError` when two or more trips end the day
    before.
    """
    day = _parse_date(run_date)
    if day is None:
        return None
    journeys_dir = Path(site_path) / "content" / "journeys"
    if not journeys_dir.is_dir():
        return None

    eve = day - timedelta(days=1)
    matches: list[Path] = []
    for path in sorted(journeys_dir.glob("*.md")):
        if path.stem == "_index":
            continue
        try:
            doc, _body = load_journey(path)
        except (OSError, ValueError):
            continue
        trip = doc.get("trip") or {}
        if _parse_date(trip.get("end_date")) == eve and _parse_date(trip.get("start_date")):
            matches.append(path)

    if not matches:
        return None
    if len(matches) > 1:
        raise AmbiguousJourneyError(
            f"run date {run_date} is the day after {len(matches)} journeys' trips "
            f"end ({', '.join(p.stem for p in matches)}); refusing to guess. Set "
            "config.drive.active_journey_by_region or pass "
            "--auto-publish-journal SLUG for this run."
        )

    path = matches[0]
    with journey_lock(path):
        doc, body = load_journey(path)
        trip = doc["trip"]
        trip["end_date"] = run_date
        if "duration_days" in trip:
            start = _parse_date(trip.get("start_date"))
            trip["duration_days"] = (day - start).days + 1
        days = doc.setdefault("days", [])
        if run_date not in {str(d.get("date")) for d in days}:
            days.append(
                {
                    "day_number": _day_number_for_new_date(days, run_date),
                    "date": run_date,
                    "day_of_week": day.strftime("%A"),
                    "heading": "",
                    "narrative": "",
                    "videos": [],
                }
            )
        atomic_write_text(path, serialize_journey(doc, body))
    return path.stem


def derive_journey_slug(region: str, run_date: str) -> str:
    """Deterministic slug from facts the run actually has — region + the
    recording date. Never derived from anything invented."""
    normalized_region = "-".join(str(region).strip().lower().split())
    return f"{normalized_region}-{run_date}"


def create_draft_journey(site_path: Path, region: str, run_date: str) -> str:
    """Create a minimal ``draft: true`` journey stub for a run whose date
    matched no existing journey.

    Only run-derived facts (region, recording date) populate the frontmatter;
    narrative/heading/locations/title/description are left empty for the
    operator to fill in — never fabricated. ``draft: true`` keeps the stub out
    of any build that doesn't pass ``--buildDrafts`` (a site checkout normally
    does that only for its local dev server), so it never reaches production
    unreviewed. Returns the created slug. Raises :class:`JourneyAlreadyExistsError`
    rather than overwrite a file at the derived path.
    """
    day = _parse_date(run_date)
    if day is None:
        raise ValueError(f"unparsable run_date: {run_date!r}")
    slug = derive_journey_slug(region, run_date)
    journeys_dir = Path(site_path) / "content" / "journeys"
    journeys_dir.mkdir(parents=True, exist_ok=True)
    journey_path = journeys_dir / f"{slug}.md"
    if journey_path.exists():
        raise JourneyAlreadyExistsError(
            f"{journey_path} already exists; refusing to overwrite. If this "
            "run's date belongs in an existing trip, fix its trip.start_date/"
            "end_date instead of creating a new journey."
        )
    doc = {
        "title": "",
        "date": run_date,
        "author": "Go Move Shift",
        "draft": True,
        "categories": ["Journeys"],
        "tags": [],
        "description": "",
        "summary": "",
        "trip": {"start_date": run_date, "end_date": run_date},
        "days": [
            {
                "day_number": 1,
                "date": run_date,
                "day_of_week": day.strftime("%A"),
                "heading": "",
                "narrative": "",
                "locations": [],
                "videos": [],
            }
        ],
        "video_count": 0,
    }
    journey_path.write_text(serialize_journey(doc, ""))
    return slug
