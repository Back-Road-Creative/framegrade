"""Console Media library: the collection view plus the asset page
and photo byte-serving route, all over ``catalog.read_catalog``.

A brand is a tenant boundary: ``site_path`` is resolved from the instance's
own config (``GMS_SITE_PATH``) and NOTHING ELSE -- never a query parameter,
a form field, or any operator-supplied string.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2

from ..catalog import CatalogEntry, CatalogResult, CatalogSourceUnavailableError, read_catalog
from ..catalog import tags as library_tags
from ..pipeline.photos import cull as cull_scoring
from ..pipeline.photos.ingest import BLUR_NO_REGIONS, BLUR_REGIONS_FOUND, BLUR_SKIPPED
from ..pipeline.photos.selector import IMAGES_BASE
from ..publishing.models import FORMAT_CAROUSEL, FORMAT_PHOTO
from .settings import console_data_dir, current_brand, load_brand_settings

logger = logging.getLogger(__name__)

#: Case-insensitive; anything else is a 404, never a download.
_PHOTO_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

#: Derived, never a second hand-kept list: a suffix that passes the allowlist
#: but has no content type would be a KeyError (a 500) on a legitimate photo.
ALLOWED_PHOTO_SUFFIXES = frozenset(_PHOTO_CONTENT_TYPES)


def resolve_site_path(brand: "str | None" = None) -> "str | None":
    """``GMS_SITE_PATH``: brand override, else env, else ``None``. Takes no
    path argument -- config is the only source."""
    slug = brand if brand is not None else current_brand()
    overrides = load_brand_settings(slug)
    raw = overrides.get("GMS_SITE_PATH") or os.environ.get("GMS_SITE_PATH")
    return raw.strip() if raw and str(raw).strip() else None


#: The catalog reads two kinds of source, and only one of them is a journey.
_ROUTING_STEM = "video_routing"


def _source_name(path: str) -> str:
    """A name the reader recognises -- never the raw path, which belongs in
    the server log, not the page. The routing file is named for what it is:
    calling it the journey "video_routing" would send an operator looking for
    a journey that does not exist."""
    stem = Path(path).stem
    return "the video routing file" if stem == _ROUTING_STEM else stem


def _damage_summary(damaged) -> list[dict]:
    """A name + a fixed reason; the error text goes to the log."""
    summaries = []
    for item in damaged:
        logger.warning("catalog: could not read source at %s: %s", item.path, item.error)
        summaries.append({"source": _source_name(item.path), "reason": "could not be read"})
    return summaries


def _absent_summary(absent) -> list[dict]:
    """A source that is not there at all, named so the page can say
    which kind of empty it is showing. "Nothing here yet" over a journeys
    directory that does not exist usually means the site path is wrong, and
    reading like the ordinary empty state sent nobody to look."""
    summaries = []
    for item in absent:
        logger.info("catalog: source not present at %s", item.path)
        summaries.append({"source": _source_name(item.path)})
    return summaries


def _display_name(entry: CatalogEntry) -> str:
    """A name an operator can recognise their own asset by -- never the
    catalog id or a hash. Photos: the export filename the catalog id already
    encodes (``CatalogEntry`` carries no separate filename field). Videos:
    ``routing_id``, the only name-like field ``video_routing.yaml`` records --
    nothing here invents a title the catalog does not carry."""
    if entry.media_type == "photo":
        filename = _photo_filename(entry.catalog_id)
        if filename:
            return filename
    return entry.routing_id or entry.catalog_id


#: `CatalogEntry.blur_state` (`catalog/reader.py`) now carries the recorded
#: per-photo blur outcome for photo entries, populated from the journey's
#: `photo_blur` frontmatter map (`pipeline/photos/ingest.py`). Video entries
#: never get one -- `catalog/reader.py`'s video path leaves it `None`. `None`
#: means "not recorded" (predates the field, or a video), a distinct fourth
#: state, never "unblurred" or any other guess. This reader only repeats what
#: was written elsewhere; it never inspects pixels, so the strings below say
#: what was RECORDED, never an assurance -- no "verified", "clean", or "safe".
_PRIVACY_STATUS_UNRECORDED = "privacy status not recorded in this catalog"
_PRIVACY_STATUS_REGIONS_FOUND = "blurred: privacy regions were found and obscured"
_PRIVACY_STATUS_NO_REGIONS = "blurred: no privacy regions were found"
#: The operator's `--skip-blur` escape shipped this export unblurred by
#: request -- phrased as a risk an operator must notice, not a routine state.
_PRIVACY_STATUS_SKIPPED = "NOT blurred: operator skipped blurring for this export"


def _privacy_status(entry: CatalogEntry) -> str:
    """What the catalog recorded for this entry's blur outcome, never an
    inference. An unrecognised recorded value gets its own honest reading
    (including the raw value) instead of being silently folded into one of
    the known states."""
    state = entry.blur_state
    if state is None:
        return _PRIVACY_STATUS_UNRECORDED
    if state == BLUR_REGIONS_FOUND:
        return _PRIVACY_STATUS_REGIONS_FOUND
    if state == BLUR_NO_REGIONS:
        return _PRIVACY_STATUS_NO_REGIONS
    if state == BLUR_SKIPPED:
        return _PRIVACY_STATUS_SKIPPED
    return f"privacy status recorded as an unrecognised value: {state!r}"


#: The drive pipeline (``plan_builder.py``) only ever emits
#: FORMAT_SHORT/FORMAT_LONG_FORM records, and ``CatalogEntry.blur_state`` is
#: architecturally ``None`` for every video entry (``catalog/reader.py``,
#: above) -- there is no per-delivery fact to read. What IS true and provable
#: is that a blur-enabled drive run cannot ship a FINAL that isn't provably
#: blurred: ``pipeline/blur_gate.py`` hard-fails the day master upstream of
#: every short/long-form cut from it, unless the run passed ``--skip-blur``
#: (a run-level flag this reader has no way to see from one delivery). This
#: names that gate rather than asserting an outcome for THIS item -- the
#: honesty rule above forbids the stronger claim.
_PRIVACY_STATUS_VIDEO_NOT_TRACKED = (
    "not recorded per item -- video face/plate blur is checked on the drive "
    "day master by pipeline/blur_gate.py unless the run used --skip-blur, "
    "and neither outcome is recorded per delivery"
)

#: Statuses that must refuse a publish rather than ship a post whose
#: privacy outcome nobody can name. Never includes ``_PRIVACY_STATUS_SKIPPED``
#: -- an operator's recorded ``--skip-blur`` choice is not a missing fact.
_PRIVACY_STATUS_UNRECOGNISED_PREFIX = "privacy status recorded as an unrecognised value"


def privacy_status_blocks_publish(status: str) -> bool:
    """Whether ``status`` (from ``privacy_status_for_media``) must
    refuse a publish -- unrecorded or unrecognised only, never a recorded
    choice (``_PRIVACY_STATUS_SKIPPED``) and never the video "not tracked"
    reading, which names a gate that already ran, not a missing fact."""
    return status == _PRIVACY_STATUS_UNRECORDED or status.startswith(
        _PRIVACY_STATUS_UNRECOGNISED_PREFIX
    )


def privacy_status_for_media(
    media_path: str, format_: "str | None", *, site_path: "str | None" = None
) -> str:
    """The privacy report line for one publish record's media item,
    reused by both the console publish-queue detail page and
    ``enqueue-plan``'s publish gate so neither re-derives ``_privacy_status``'s
    strings. Only FORMAT_PHOTO/FORMAT_CAROUSEL items have a catalog entry to
    look up (photos ingest records ``blur_state`` there); every other format
    -- including ``None``, what every record the drive pipeline builds today
    carries -- reads as ``_PRIVACY_STATUS_VIDEO_NOT_TRACKED``. A photo item
    with no resolvable site, or no catalog entry matching its exported
    filename, reads as unrecorded: an unreachable fact is not a safe one to
    report as present."""
    if format_ not in (FORMAT_PHOTO, FORMAT_CAROUSEL):
        return _PRIVACY_STATUS_VIDEO_NOT_TRACKED
    resolved_site = site_path if site_path is not None else resolve_site_path()
    if not resolved_site:
        return _PRIVACY_STATUS_UNRECORDED
    try:
        result: CatalogResult = read_catalog(Path(resolved_site))
    except CatalogSourceUnavailableError:
        return _PRIVACY_STATUS_UNRECORDED
    filename = Path(media_path).name
    for entry in result.entries:
        if entry.media_type == "photo" and _photo_filename(entry.catalog_id) == filename:
            return _privacy_status(entry)
    return _PRIVACY_STATUS_UNRECORDED


def _display_row(entry: CatalogEntry, tags: "list[str] | None" = None) -> dict:
    """Collection-table presentation for one entry, beside (never instead of)
    the existing catalog id/hash columns some operators use. No thumbnail
    field: see ``_PRIVACY_STATUS_UNRECORDED``."""
    return {
        "name": _display_name(entry),
        "kind": entry.media_type.capitalize(),
        "treatment": entry.decoder or "not recorded",
        "privacy_status": _privacy_status(entry),
        "tags": tags or [],
    }


def _search_haystack(entry: CatalogEntry, row: dict, tags: "list[str]") -> str:
    """Everything a free-text search may match, lowercased: title, journey
    (the closest thing this catalog records to a location label -- it has no
    separate location field), and this brand's own tags for the item."""
    parts = [row["name"], entry.journey, *tags]
    return " ".join(parts).lower()


def library_tags_for(catalog_id: str) -> list[str]:
    """This brand's own tags for ``catalog_id`` -- never another brand's."""
    return library_tags.load_tags(console_data_dir(), current_brand()).get(catalog_id, [])


def add_library_tag(catalog_id: str, tag: str) -> list[str]:
    """Add ``tag`` to ``catalog_id`` in this brand's own tag store."""
    return library_tags.add_tag(console_data_dir(), current_brand(), catalog_id, tag)


def remove_library_tag(catalog_id: str, tag: str) -> list[str]:
    """Remove ``tag`` from ``catalog_id`` in this brand's own tag store."""
    return library_tags.remove_tag(console_data_dir(), current_brand(), catalog_id, tag)


#: Rows per page. A site accumulates photos across many journeys, and every
#: ingested card adds more, so handing the whole read to the template built a
#: table that grows without bound. This
#: bounds the PAGE. The read underneath still walks every journey file --
#: that is a different change (a rebuildable index) and one is worth introducing
#: only when the scan itself is the bottleneck.
PER_PAGE = 100


def _pager(total: int, page) -> dict:
    """Where in ``total`` rows ``page`` lands, for a ``page`` that came off a
    query string and may be anything at all. Out of range lands on the nearest
    real page rather than an empty table, which an operator reads as an empty
    catalog."""
    pages = max(1, -(-total // PER_PAGE))
    try:
        wanted = int(page)
    except (TypeError, ValueError):
        wanted = 1
    wanted = min(max(wanted, 1), pages)
    return {
        "total": total,
        "per_page": PER_PAGE,
        "page": wanted,
        "pages": pages,
        "has_prev": wanted > 1,
        "has_next": wanted < pages,
    }


def build_context(
    *,
    media_type: "str | None" = None,
    journey: "str | None" = None,
    date: "str | None" = None,
    catalog_id: "str | None" = None,
    q: "str | None" = None,
    page=1,
) -> dict:
    """Template context for ``GET /library``, one bounded page of it. Never
    raises -- degrades to a visible state rather than a 500, and the degraded
    states carry the pager keys too so the template renders instead of
    raising on a missing name."""
    site_path = resolve_site_path()
    if site_path is None:
        return {
            "entries": [],
            "display": {},
            "damaged": [],
            # Nothing was read, so nothing can be said about what is there --
            # and the template needs the key regardless.
            "absent": [],
            "site_configured": False,
            "site_error": "",
            **_pager(0, page),
        }
    try:
        result = read_catalog(
            site_path,
            media_type=media_type,
            journey=journey,
            date=date,
            catalog_id=catalog_id,
        )
    except CatalogSourceUnavailableError as exc:
        logger.warning("catalog: configured site path unavailable: %s", exc)
        return {
            "entries": [],
            "display": {},
            "damaged": [],
            "absent": [],
            "site_configured": True,
            "site_error": "The configured site path could not be read.",
            **_pager(0, page),
        }
    # Free-text search combines with the structural filters above rather
    # than replacing them, and runs BEFORE the pager so a filtered match on
    # row 400 still counts toward the right page.
    all_tags = library_tags.load_tags(console_data_dir(), current_brand())
    entries = result.entries
    needle = q.strip().lower() if q else ""
    if needle:
        entries = [
            e
            for e in entries
            if needle in _search_haystack(e, _display_row(e), all_tags.get(e.catalog_id, []))
        ]
    # Slice AFTER the filters, never before: a filter whose only match is row
    # 400 must still find it.
    pager = _pager(len(entries), page)
    start = (pager["page"] - 1) * PER_PAGE
    page_entries = entries[start : start + PER_PAGE]
    return {
        "entries": page_entries,
        "display": {
            e.catalog_id: _display_row(e, all_tags.get(e.catalog_id, [])) for e in page_entries
        },
        "damaged": _damage_summary(result.damaged),
        "absent": _absent_summary(result.absent),
        "site_configured": True,
        "site_error": "",
        **pager,
    }


def _photo_filename(catalog_id: str) -> "str | None":
    """``photo:{journey}:{date}:{filename}`` -> ``filename``. Journey and
    date are catalog identity, never colon-bearing; only ``maxsplit`` keeps a
    filename that itself contains ``:`` intact."""
    parts = catalog_id.split(":", 3)
    return parts[3] if len(parts) == 4 and parts[0] == "photo" else None


def resolve_photo_path(site_path: "str | Path", entry: CatalogEntry) -> "Path | None":
    """The on-disk path for a photo entry, or ``None`` if it cannot be
    served. Resolve-then-compare only -- never a stripped/sanitised
    filename, which only looks safe."""
    filename = _photo_filename(entry.catalog_id)
    if not filename:
        return None
    if Path(filename).suffix.lower() not in ALLOWED_PHOTO_SUFFIXES:
        return None
    root = (Path(site_path) / IMAGES_BASE / entry.journey).resolve()
    candidate = (root / filename).resolve()
    if not candidate.is_relative_to(root):
        return None
    return candidate


def _provenance(site_path: str, entry: CatalogEntry) -> tuple[dict, dict]:
    """``(original, derivative)`` for a photo row: which recorded fact is
    about which file.

    ``photo_hashes[filename]`` is the SHA-256 of the ORIGINAL card file --
    ``_source_hash`` is explicit that it is "never derived from the developed
    output" (see ``_source_hash`` in ``pipeline/photos/ingest.py``). What this console serves
    is the developed export under ``static/images/journeys``. Showing the
    image beside a bare "hash recorded" line invited an operator to hash the
    bytes in front of them, get a different number, and conclude the file was
    damaged.

    ``present_here`` is always ``False``: the original is on the card the
    photos were ingested from, and the site checkout never holds it. It is a
    field rather than a constant sentence so the template states a fact it was
    given, and so a future import that DOES keep originals has somewhere to
    say so.
    """
    filename = _photo_filename(entry.catalog_id) or ""
    relative = f"{IMAGES_BASE}/{entry.journey}/{filename}"
    path = resolve_photo_path(site_path, entry)
    present = path is not None and path.is_file()
    original = {
        "content_hash": entry.content_hash,
        "decoder": entry.decoder,
        "present_here": False,
    }
    derivative = {
        "relative_path": relative,
        "present": present,
        # stat only -- this module never opens media, and a hash of the export
        # would have nothing to compare against: the recorded one is the
        # original's.
        "size_bytes": path.stat().st_size if present else None,
    }
    return original, derivative


def _cull_score_for_path(path: Path) -> "cull_scoring.CullScore | None":
    """Read-only culling score for one on-disk photo, or ``None`` when it
    can't be decoded -- a corrupt/placeholder file must never fail the asset
    page, only leave it without a score."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        return cull_scoring.compute_cull_score(rgb)
    except ValueError:
        return None


def _sibling_photo_hashes(site_path: str, entry: CatalogEntry) -> dict:
    """``{catalog_id: phash}`` for every decodable photo in *entry*'s own
    journey+day -- near-duplicate grouping is scoped to one day's shots
    (typically a burst), never the whole journey/library, so this stays
    cheap on a page view."""
    try:
        siblings = read_catalog(
            site_path, journey=entry.journey, date=entry.date, media_type="photo"
        ).entries
    except CatalogSourceUnavailableError:
        return {}
    hashes = {}
    for sibling in siblings:
        sib_path = resolve_photo_path(site_path, sibling)
        if sib_path is None or not sib_path.is_file():
            continue
        score = _cull_score_for_path(sib_path)
        if score is not None:
            hashes[sibling.catalog_id] = score.phash
    return hashes


def _cull_info(site_path: str, entry: CatalogEntry, path: Path) -> "dict | None":
    """Read-only cull measurement for the console asset page: sharpness,
    exposure clipping, and near-duplicate filenames within the same day.
    Never a verdict -- nothing here deletes, rejects, or ranks a photo."""
    score = _cull_score_for_path(path)
    if score is None:
        return None
    hashes = _sibling_photo_hashes(site_path, entry)
    hashes[entry.catalog_id] = score.phash
    groups = cull_scoring.group_near_duplicates(hashes)
    near_duplicates = [
        _photo_filename(catalog_id) or catalog_id for catalog_id in groups.get(entry.catalog_id, [])
    ]
    return {
        "sharpness": score.sharpness,
        "exposure_black_fraction": score.exposure_black_fraction,
        "exposure_white_fraction": score.exposure_white_fraction,
        "near_duplicates": near_duplicates,
    }


def build_asset_context(catalog_id: str) -> "dict | None":
    """Template context for ``GET /library/{catalog_id}``, or ``None`` for a
    404 -- unknown id or no site configured. A video row carries no
    ``original``/``derivative``: its bytes live on the destination platform,
    and borrowing the photo shape would name a file that does not exist."""
    site_path = resolve_site_path()
    if site_path is None:
        return None
    try:
        result = read_catalog(site_path, catalog_id=catalog_id)
    except CatalogSourceUnavailableError:
        return None
    if not result.entries:
        return None
    entry = result.entries[0]
    original: "dict | None" = None
    derivative: "dict | None" = None
    cull: "dict | None" = None
    if entry.media_type == "photo":
        original, derivative = _provenance(site_path, entry)
        photo_path = resolve_photo_path(site_path, entry)
        if photo_path is not None and photo_path.is_file():
            cull = _cull_info(site_path, entry, photo_path)
    return {
        "entry": entry,
        "media_available": bool(derivative and derivative["present"]),
        "original": original,
        "derivative": derivative,
        "cull": cull,
    }


def resolve_photo_media(catalog_id: str) -> "tuple[Path, str] | None":
    """``(path, content_type)`` for ``GET /library/media/{catalog_id}``, or
    ``None`` for a 404. Never called for a video id."""
    site_path = resolve_site_path()
    if site_path is None:
        return None
    try:
        result = read_catalog(site_path, catalog_id=catalog_id)
    except CatalogSourceUnavailableError:
        return None
    if not result.entries or result.entries[0].media_type != "photo":
        return None
    entry = result.entries[0]
    path = resolve_photo_path(site_path, entry)
    if path is None or not path.is_file():
        return None
    content_type = _PHOTO_CONTENT_TYPES[path.suffix.lower()]
    return path, content_type
