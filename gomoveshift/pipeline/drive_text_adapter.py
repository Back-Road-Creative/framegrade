"""GMS drive facts -> Waypost's typed ``DriveTextData``.

Waypost owns the description and tag text (``build_drive_text``); GMS supplies
facts. Every place a line prints is a ``PlaceRef`` that cites route-record or
region evidence, so a name the retained route never backed is refused here,
before anything renders or uploads (``DriveTextRefusedError``).

The caps follow GMS's own: roads, features and passing places keep the first
five, the route eight and the geo hashtags four; more than twelve towns or
states refuse (Waypost's registered line has no slot for them).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from seo_manager.analytics.drive_templates import HEADINGS, REGIONS_DISPLAY, WEEKDAYS
from seo_manager.analytics.final_text_params import Slot, valid_value
from seo_manager.analytics.final_text_surfaces import (
    build_visible,
    compose_description,
    compose_tags,
)
from seo_manager.generation.drive_text import (
    MAX_FEATURES,
    MAX_GEO,
    MAX_PASSING,
    MAX_ROADS,
    MAX_ROUTE,
    MAX_STATES,
    MAX_TOWNS,
    ChapterData,
    DriveTextData,
    PlaceRef,
    SeoArc,
    build_drive_text,
)

from .geo_constants import US_STATE_ABBREVS

logger = logging.getLogger(__name__)

#: GMS discloses an edit only when the cut is at least this much shorter, and
#: frames a trim below this ratio as highlights (longform_metadata).
_EDIT_MIN_DELTA_SEC = 30.0
_HIGHLIGHTS_BELOW = 0.5

#: ``{casefolded name: (evidence_id, display text)}``.
EvidenceIndex = Dict[str, Tuple[str, str]]


class DriveTextRefusedError(ValueError):
    """The facts cannot be printed as backed text; ``reasons`` name why."""

    def __init__(self, reasons: Sequence[str], detail: str = ""):
        self.reasons = list(reasons)
        super().__init__(
            f"drive text refused ({', '.join(self.reasons)})" + (f": {detail}" if detail else "")
        )


@dataclass(frozen=True)
class DriveFacts:
    """What a GMS description states, by place name (not yet evidence-bound)."""

    place: str
    final_seconds: float
    source_seconds: Optional[float] = None
    highlights: bool = False
    edited: bool = False
    seo: Optional[Tuple[str, Optional[str], Optional[str], Optional[str]]] = None
    terrain: Optional[str] = None
    body: bool = False
    road: Optional[str] = None
    distance_km: Optional[float] = None
    us: bool = False
    heading: Optional[str] = None
    through: Sequence[str] = ()
    route: Sequence[str] = ()
    insert_count: int = 0
    filmed: Optional[Tuple[int, int]] = None
    site: bool = True
    geo_hashtags: Sequence[str] = ()
    chapters: Sequence[Tuple[int, str, Sequence[str]]] = ()
    place_tags: Sequence[str] = ()
    place_drive_tags: Sequence[str] = ()
    year: Optional[int] = None
    month: Optional[int] = None
    journey_start: Optional[str] = None
    journey_end: Optional[str] = None
    passing: Sequence[str] = ()
    towns: Sequence[str] = ()
    roads: Sequence[str] = ()
    states: Sequence[str] = ()
    features: Sequence[str] = ()
    playlist: Optional[Tuple[str, str]] = None
    cadence: Optional[Tuple[str, str]] = None
    extra: dict = field(default_factory=dict)


def _key(name: str) -> str:
    return " ".join(str(name or "").casefold().split())


def evidence_index(route_entry: Optional[dict]) -> EvidenceIndex:
    """The citable names of one LF: its retained-route places and region claims
    (a US state also by its abbreviation, as the review does)."""
    from .final_text_record import _region_id

    entry = route_entry or {}
    index: EvidenceIndex = {}
    for p in (entry.get("record") or {}).get("places") or []:
        index.setdefault(_key(p["name"]), (p["evidence_id"], p["name"]))
    for claim in entry.get("region_claims") or []:
        for name in filter(None, [claim["name"], US_STATE_ABBREVS.get(claim["name"])]):
            index.setdefault(_key(name), (_region_id(claim["level"], name), name))
    return index


def _ref(index: EvidenceIndex, field_name: str, name: Optional[str]) -> PlaceRef:
    hit = index.get(_key(name or ""))
    if hit is None:
        raise DriveTextRefusedError([f"name_unbacked:{field_name}:{_key(name or '')}"])
    return PlaceRef(*hit)


def _refs(index, field_name, names: Sequence[str], cap: Optional[int] = None) -> List[PlaceRef]:
    return [_ref(index, field_name, n) for n in (names[:cap] if cap else names)]


def _bounded(field_name: str, names: Sequence[str], limit: int) -> None:
    if len(names) > limit:
        raise DriveTextRefusedError([f"too_many:{field_name}"])


def cadence_pair(prose: str) -> Tuple[str, str]:
    """``(region, weekday)`` from GMS's "New <Region> drive every <Weekday>"."""
    m = re.fullmatch(r"\s*new\s+(.+?)\s+drive\s+every\s+(\w+)\s*", prose or "", re.IGNORECASE)
    region = next((r for r in REGIONS_DISPLAY if m and r.casefold() == m.group(1).casefold()), "")
    day = next((d for d in WEEKDAYS if m and d.casefold() == m.group(2).casefold()), "")
    if not (region and day):
        raise DriveTextRefusedError(["cadence_unmappable"], repr(prose))
    return region, day


def build_drive_data(facts: DriveFacts, index: EvidenceIndex) -> DriveTextData:
    """``facts`` as Waypost data, every printed place bound to evidence."""
    if (facts.playlist or facts.cadence) and not facts.site:
        raise DriveTextRefusedError(["site_line_required"])
    if facts.playlist and not valid_value(Slot("url", "playlist_url"), facts.playlist[1]):
        raise DriveTextRefusedError(["playlist_url_invalid"])
    _bounded("towns", facts.towns, MAX_TOWNS)
    _bounded("states", facts.states, MAX_STATES)
    towns = list(facts.towns)
    start, end, passing = facts.journey_start, facts.journey_end, list(facts.passing)
    if start is not None and end is not None and _key(start) == _key(end):
        # An arc that ends where it starts says nothing; the towns line stands in.
        start = end = None
        passing = []
    if (start is None) != (end is None):
        start = end = None
        passing = []
    if start is not None:
        towns = []
    seo = None
    if facts.seo is not None:
        region, a, b, road = facts.seo
        seo = SeoArc(
            _ref(index, "seo.region", region),
            _ref(index, "seo.a", a) if a and b else None,
            _ref(index, "seo.b", b) if a and b else None,
            _ref(index, "seo.road", road) if road and a and b else None,
        )
    ref = lambda f, n: _ref(index, f, n)  # noqa: E731
    return DriveTextData(
        place=ref("place", facts.place),
        final_seconds=facts.final_seconds,
        source_seconds=facts.source_seconds,
        highlights=facts.highlights,
        edited=facts.edited,
        seo=seo,
        terrain=facts.terrain,
        body=facts.body,
        road=ref("road", facts.road) if facts.road else None,
        distance_km=facts.distance_km,
        us=facts.us,
        heading=facts.heading,
        through=_refs(index, "through", facts.through),
        route=_refs(index, "route", facts.route, MAX_ROUTE),
        insert_count=facts.insert_count,
        filmed=facts.filmed,
        site=facts.site,
        geo_hashtags=_refs(index, "geo", facts.geo_hashtags, MAX_GEO),
        chapters=[
            ChapterData(
                int(s),
                ref("chapter", p),
                tuple(ref("chapter.window", w).evidence_id for w in window),
            )
            for s, p, window in facts.chapters
        ],
        place_tags=_refs(index, "tag", facts.place_tags),
        place_drive_tags=_refs(index, "tag.drive", facts.place_drive_tags),
        year=facts.year,
        month=facts.month,
        journey_start=ref("journey.start", start) if start else None,
        journey_end=ref("journey.end", end) if end else None,
        passing=_refs(index, "passing", passing, MAX_PASSING),
        towns=_refs(index, "towns", towns),
        roads=_refs(index, "roads", facts.roads, MAX_ROADS),
        states=_refs(index, "states", facts.states),
        features=_refs(index, "features", facts.features, MAX_FEATURES),
        playlist=facts.playlist,
        cadence=facts.cadence,
    )


def drive_entries(data: DriveTextData) -> list:
    """Waypost's visible entries for ``data`` (what the review binds)."""
    return build_visible(build_drive_text(data))


def compose(data: DriveTextData) -> Tuple[str, List[str]]:
    """``(description, tags)`` exactly as the byte-bound check composes them."""
    entries = drive_entries(data)
    return compose_description(entries), compose_tags(entries)


def _named_in(label: str, index: EvidenceIndex) -> List[str]:
    """Evidence names the label says as whole words, in label order, places
    before regions. A name said only inside a longer evidenced name is not
    said: "Beaufort County" names the county, not the town of Beaufort."""
    spans = []
    for key, (eid, text) in index.items():
        for m in re.finditer(rf"(?<!\w){re.escape(text)}(?!\w)", label or "", re.IGNORECASE):
            spans.append((m.start(), m.end(), eid.startswith("region-"), text))
    found = {
        (region, start, text)
        for start, end, region, text in spans
        if not any(s <= start and end <= e and e - s > end - start for s, e, _, _ in spans)
    }
    return list(dict.fromkeys(text for _, _, text in sorted(found)))


def _seconds(stamp: str) -> int:
    parts = [int(x) for x in str(stamp).split(":")]
    return sum(v * 60**i for i, v in enumerate(reversed(parts)))


_MIN_CHAPTERS = 3  # YouTube only renders a chapter list of three or more


def _bound_chapters(chapters: Sequence[dict], index: EvidenceIndex) -> list:
    """``(start_s, place, window)`` per chapter, bound to evidenced names.

    A generated title is decorated ("Start - X, Y", "Entering Z"), so its place
    is the first route place it says and its window every route place it says;
    a region ("Beaufort County") is no chapter place, and Waypost refuses one in
    a window (``route_place_unknown``) and a chapter with an empty window
    (``chapter_without_window``). A title naming no route place is dropped;
    consecutive chapters on one place collapse to the earliest. A list that no
    longer opens at 0:00 or is under three chapters is sent as none: chapter
    blocks are optional, a refusal is not."""
    bound: list = []
    for c in chapters:
        named = [
            n for n in _named_in(c["title"], index) if not index[_key(n)][0].startswith("region-")
        ]
        if not named:
            continue
        if bound and bound[-1][1] == named[0]:
            continue
        bound.append((_seconds(c["timestamp"]), named[0], named))
    if len(bound) < _MIN_CHAPTERS or bound[0][0] != 0:
        return []
    return bound


def lf_drive_facts(
    label: str,
    index: EvidenceIndex,
    *,
    final_seconds: float,
    source_seconds: Optional[float] = None,
    trim_ratio: Optional[float] = None,
    distance_km: Optional[float] = None,
    bearing: Optional[str] = None,
    recording_date: Optional[str] = None,
    insert_count: int = 0,
    us: bool = False,
    chapters: Sequence[dict] = (),
) -> DriveFacts:
    """The facts a drive description states for one LF or superlong part.

    The hook and location name the first evidenced place in ``label``; a label
    that names none is refused (``name_unbacked:place``)."""
    named = _named_in(label, index)
    if not named:
        raise DriveTextRefusedError([f"name_unbacked:place:{_key(label)}"])
    year = month = None
    if recording_date and re.fullmatch(r"\d{4}-\d{2}-\d{2}", recording_date):
        year, month = int(recording_date[:4]), int(recording_date[5:7])
    measured = source_seconds is not None and source_seconds > 0
    highlights = measured and trim_ratio is not None and trim_ratio < _HIGHLIGHTS_BELOW
    edited = measured and not highlights and source_seconds - final_seconds >= _EDIT_MIN_DELTA_SEC
    return DriveFacts(
        place=named[0],
        final_seconds=final_seconds,
        source_seconds=source_seconds if (highlights or edited) else None,
        highlights=highlights,
        edited=edited,
        distance_km=distance_km,
        us=us,
        heading=bearing if bearing in HEADINGS else None,
        insert_count=insert_count,
        filmed=(year, month) if year else None,
        place_tags=named,
        place_drive_tags=named[:1],
        year=year,
        month=month,
        chapters=_bound_chapters(chapters, index),
    )


def attach_drive_text(route_entry: Optional[dict], metadata: dict, label: str, **facts_kw) -> dict:
    """Set ``metadata["final_text_description"]`` to Waypost's entries for the
    LF, or to ``[]`` with ``metadata["drive_text_refusal"]`` naming why. A
    refusal leaves the review without its layout blocks, so it refuses and the
    upload is held -- the pipeline run itself keeps going."""
    metadata.pop("drive_text_refusal", None)
    metadata["drive_text_inputs"] = {"label": label, **facts_kw}
    try:
        index = evidence_index(route_entry)
        data = build_drive_data(lf_drive_facts(label, index, **facts_kw), index)
        metadata["final_text_description"] = drive_entries(data)
    except DriveTextRefusedError as exc:
        logger.error("DRIVE-TEXT: %s", exc)
        metadata["final_text_description"] = []
        metadata["drive_text_refusal"] = exc.reasons
    return metadata


def reattach_with_chapters(route_entry: Optional[dict], metadata: dict, chapters) -> dict:
    """Re-run :func:`attach_drive_text` with the LF's chapters, from the inputs
    the first attach kept (chapters from an earlier reattach are replaced); a
    no-op for an LF that never attached."""
    inputs = dict(metadata.get("drive_text_inputs") or {})
    if not inputs:
        return metadata
    label = inputs.pop("label")
    inputs.pop("chapters", None)
    return attach_drive_text(route_entry, metadata, label, **inputs, chapters=list(chapters))
