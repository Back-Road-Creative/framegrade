"""Read-only media-catalog reader — backend contract only, no
templates/routes. Photos: journey markdown under
``content/journeys/*.md``, identity from ``photo_hashes`` (legacy = no
hash = unrecorded, never dropped/faked). Videos: ``data/video_routing.yaml``
via the pipeline's own ``load_routing`` — its ``date`` is a travel date,
never a publication date. Detail: ``docs/pipeline-internals.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ruamel.yaml import YAMLError as RuamelYAMLError
from yaml import YAMLError as PyYAMLError

from ..pipeline.journal_publisher import load_journey
from ..pipeline.photos.ingest import BLUR_KEY, PROVENANCE_KEY
from ..pipeline.site_sync.journey_reconcile import load_routing

_JOURNEYS_SUBDIR = ("content", "journeys")
_ROUTING_PATH = ("data", "video_routing.yaml")

# Both sources fail the same two ways — read the file, then parse it — but
# through DIFFERENT yaml libraries, and neither library's YAMLError is a
# subclass of the other's, so one merged tuple would silently stop catching
# half of what it names. load_journey parses Hugo frontmatter with ruamel
# (ValueError for a missing/malformed delimiter); load_routing uses PyYAML.
# What matters either way: a source the reader cannot use must never take the
# OTHER source down with it.
_LOAD_JOURNEY_ERRORS = (OSError, ValueError, RuamelYAMLError)
_LOAD_ROUTING_ERRORS = (OSError, ValueError, PyYAMLError)


class CatalogSourceUnavailableError(Exception):
    """The site checkout itself is missing/unreadable — distinct from a real,
    empty catalog. A missing ``content/journeys`` dir or ``video_routing.yaml``
    is a legitimate "nothing here yet" and is NOT this error."""


@dataclass(frozen=True)
class DamagedSource:
    """A file the catalog reads from that could not be read or parsed — a
    journey markdown, or ``video_routing.yaml``. Distinct from a source that
    parsed cleanly and simply holds nothing — that is a real, empty subset,
    never reported here."""

    path: str
    error: str


@dataclass(frozen=True)
class AbsentSource:
    """A source that is not there at all. The read treats it as that source's
    own empty subset -- a site with no videos yet must not 500 -- but "not
    there" and "there and holding nothing" are different facts about the
    operator's configuration, and only one of them asks for action. Exclusive
    with :class:`DamagedSource`: a file that exists and will not parse is
    damaged, never absent."""

    path: str


@dataclass(frozen=True)
class CatalogResult:
    """``read_catalog``'s return: the entries it could read, plus every
    journey file it could not, so a caller can tell "nothing to show" from
    "something broke" without comparing counts against the filesystem.

    ``absent`` splits "nothing to show" itself in two, for the same reason:
    an empty page over a journeys directory that is not there usually means
    ``GMS_SITE_PATH`` points at the wrong tree, and read like "nothing yet"
    it sends nobody to look."""

    entries: list[CatalogEntry]
    damaged: list[DamagedSource] = field(default_factory=list)
    absent: list[AbsentSource] = field(default_factory=list)


@dataclass(frozen=True)
class CatalogEntry:
    """One catalog row. Photo and video rows share the same shape; the
    fields the other media type doesn't use stay at their default."""

    catalog_id: str
    media_type: str  # "photo" | "video"
    journey: str
    date: str  # day date (photo) / travel date (video) — never a publish date
    content_hash: str | None = None
    # A hash string was present in the journey's own recorded photo_hashes —
    # never means anything hashed the media file and compared it. This module
    # is read-only/offline (docs/pipeline-internals.md) and never opens media.
    hash_recorded: bool = False
    decoder: str | None = None
    routing_id: str | None = None
    # The recorded blur outcome for a photo export ("blurred_regions_found",
    # "blurred_no_regions", "skipped") — never inferred or defaulted. ``None``
    # means the journey's ``photo_blur`` map has no entry for this filename:
    # a distinct "not recorded" state, not "unblurred" or any other guess.
    blur_state: str | None = None


def _iter_journey_paths(site_path: Path):
    journeys_dir = site_path.joinpath(*_JOURNEYS_SUBDIR)
    if not journeys_dir.is_dir():
        return
    for path in sorted(journeys_dir.glob("*.md")):
        if path.stem == "_index":
            continue
        yield path


def _read_photo_entries(
    site_path: Path,
) -> tuple[list[CatalogEntry], list[DamagedSource], list[AbsentSource]]:
    entries: list[CatalogEntry] = []
    damaged: list[DamagedSource] = []
    journeys_dir = site_path.joinpath(*_JOURNEYS_SUBDIR)
    # The directory, not the files in it: a journeys dir holding no markdown
    # is a site with no journeys published yet, which is a real answer.
    absent = [] if journeys_dir.is_dir() else [AbsentSource(path=str(journeys_dir))]
    for path in _iter_journey_paths(site_path):
        try:
            doc, _body = load_journey(path)
        except _LOAD_JOURNEY_ERRORS as exc:
            damaged.append(DamagedSource(path=str(path), error=str(exc)))
            continue
        journey = path.stem
        provenance = doc.get(PROVENANCE_KEY) or {}
        for day in doc.get("days", []) or []:
            day_date = str(day.get("date"))
            for group in day.get("photos", []) or []:
                hashes = group.get("photo_hashes") or {}
                blur_states = group.get(BLUR_KEY) or {}
                for name in group.get("photos", []) or []:
                    filename = str(name)
                    content_hash = hashes.get(filename)
                    entries.append(
                        CatalogEntry(
                            catalog_id=f"photo:{journey}:{day_date}:{filename}",
                            media_type="photo",
                            journey=journey,
                            date=day_date,
                            content_hash=str(content_hash) if content_hash else None,
                            hash_recorded=bool(content_hash),
                            decoder=(str(provenance[filename]) if filename in provenance else None),
                            blur_state=(
                                str(blur_states[filename]) if filename in blur_states else None
                            ),
                        )
                    )
    return entries, damaged, absent


def _read_video_entries(
    site_path: Path,
) -> tuple[list[CatalogEntry], list[DamagedSource], list[AbsentSource]]:
    routing_path = site_path.joinpath(*_ROUTING_PATH)
    if not routing_path.exists():
        # Checked before the load rather than inferred from its error: a
        # missing file and a truncated one both raise OSError/ValueError out
        # of `load_routing`, and telling an operator to go find a file they
        # already have is worse than saying nothing.
        return [], [], [AbsentSource(path=str(routing_path))]
    try:
        routes = load_routing(routing_path)
    except _LOAD_ROUTING_ERRORS as exc:
        # One file, and every photo the reader had already gathered used to go
        # with it: this call had no error handling, so a truncated routing file
        # emptied the whole library and surfaced as a 500 rather than as the
        # visible partial state the console promises.
        return [], [DamagedSource(path=str(routing_path), error=str(exc))], []
    return (
        [
            CatalogEntry(
                catalog_id=f"video:{video_id}",
                media_type="video",
                journey=str(route["journey"]),
                date=str(route["date"]),
                routing_id=str(video_id),
            )
            for video_id, route in routes.items()
        ],
        [],
        [],
    )


def read_catalog(
    site_path: str | Path,
    *,
    media_type: str | None = None,
    journey: str | None = None,
    date: str | None = None,
    catalog_id: str | None = None,
) -> CatalogResult:
    """Catalog entries for photos/videos already on disk. Raises
    :class:`CatalogSourceUnavailableError` only when ``site_path`` isn't a
    directory; a missing journeys dir or routing file is each source's own
    empty subset, not an error. Filters are AND-combined, by identity only
    (id/type/date/journey) — never an operator-entered path. A journey file
    that could not be read/parsed is reported in the result's ``damaged``
    list rather than silently shrinking ``entries``; a source that is not
    there at all is reported in ``absent``, which a ``media_type`` filter
    narrows with everything else -- a photo-only read never opens the routing
    file, so it says nothing about whether that file exists."""
    path = Path(site_path)
    if not path.is_dir():
        raise CatalogSourceUnavailableError(f"{path}: site path does not exist")

    entries: list[CatalogEntry] = []
    damaged: list[DamagedSource] = []
    absent: list[AbsentSource] = []
    if media_type in (None, "photo"):
        photo_entries, photo_damaged, photo_absent = _read_photo_entries(path)
        entries.extend(photo_entries)
        damaged.extend(photo_damaged)
        absent.extend(photo_absent)
    if media_type in (None, "video"):
        video_entries, video_damaged, video_absent = _read_video_entries(path)
        entries.extend(video_entries)
        damaged.extend(video_damaged)
        absent.extend(video_absent)

    if journey is not None:
        entries = [e for e in entries if e.journey == journey]
    if date is not None:
        entries = [e for e in entries if e.date == date]
    if catalog_id is not None:
        entries = [e for e in entries if e.catalog_id == catalog_id]
    return CatalogResult(entries=entries, damaged=damaged, absent=absent)
