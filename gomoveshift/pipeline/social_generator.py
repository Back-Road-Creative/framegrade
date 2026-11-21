"""
Pipeline Social Generator — unique social post text per short.

Generates varied, non-generic social captions for each short using
location, weather, segment position, and time-of-day context.


Usage:
    from gomoveshift.pipeline.social_generator import SocialGenerator

    generator = SocialGenerator(location="County Tipperary, Ireland")
    posts = generator.generate_all(
        num_shorts=9,
        weather_results=[...],  # from WeatherDetector
    )
    # posts[0].caption = "☀️ Sunny drive through Tipperary — can you guess..."
"""

import hashlib
import json
import logging
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from string import Formatter
from types import SimpleNamespace
from typing import Iterable, List, Mapping, Optional, Sequence

import yaml

from ..models import ResolvedLocation
from ..publishing.experiments import (
    SHORTS_ARMS,
    SHORTS_HASHTAG_SET_ARMS,
    SHORTS_TITLE_BODY_ARMS,
    Experiment,
    assign_arms,
    assign_shorts_arms,
    hashtag_mode_for,
    record_shorts_arms,
)
from ._series_utils import _strip_series_suffix, parse_lf_label
from .content_validator import ContentValidationError, ContentValidator
from .final_text_record import reviewable_short_tags
from .geo_constants import US_STATE_ABBREVS
from .title_decision import decide_title, shipped_title_snapshot, short_place_facts
from .weather import WeatherCondition, WeatherResult

logger = logging.getLogger(__name__)


class TitleCapacityExceeded(ValueError):  # noqa: N818 — name is part of the public API
    """Raised when a batch asks composition for more distinct titles than the
    grounded facts can honestly support (never pad with a near-duplicate)."""


class InteractiveShortHasBurnedLabel(ValueError):  # noqa: N818 — name is part of the public API
    """Raised when an ``interactive`` short's burned on-screen label names a
    place: "where is this?" is a lie the moment the OST already answers
    it. Fix: render without the OST for that index (not yet
    implemented — today interactive shorts are the hand-made / label-free
    lane only) or don't flag the index ``interactive``."""


@dataclass
class HandTitle:
    """One hand-authored ``titles.yaml`` entry."""

    title: str
    caption: Optional[str] = None
    interactive: bool = False


def load_hand_titles(path: Path) -> "dict[int, HandTitle]":
    """Load a drive-level ``<project_dir>/social/titles.yaml`` hand-title file.

    Returns ``{}`` when ``path`` does not exist (absence is not malformed —
    a short with no hand title simply composes normally).
    A present-but-malformed file (not a mapping, a non-int key, a missing or
    non-string ``title``, or a non-bool ``interactive``) is a hard
    ``ValueError`` naming the path and the offending key.
    """
    if not path.exists():
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"titles file {path} must be a mapping of short_index to title info, "
            f"got {type(raw).__name__}"
        )
    result: dict[int, HandTitle] = {}
    for key, value in raw.items():
        try:
            idx = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"titles file {path} has a non-integer short_index key: {key!r}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"titles file {path} entry for short_index {idx} must be a "
                f"mapping, got {type(value).__name__}"
            )
        title = value.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(
                f"titles file {path} entry for short_index {idx} is missing a non-empty 'title'"
            )
        caption = value.get("caption")
        if caption is not None and not isinstance(caption, str):
            raise ValueError(
                f"titles file {path} entry for short_index {idx} has a non-string 'caption'"
            )
        interactive = value.get("interactive", False)
        if not isinstance(interactive, bool):
            raise ValueError(
                f"titles file {path} entry for short_index {idx} has a non-bool 'interactive'"
            )
        result[idx] = HandTitle(title=title, caption=caption, interactive=interactive)
    return result


def derive_interactive_indexes(hand_titles: "dict[int, HandTitle]") -> "set[int]":
    """The set of GLOBAL short indexes flagged ``interactive`` in
    a loaded ``titles.yaml`` (``load_hand_titles`` — never re-parse the file
    separately). Callers persist this once at plan time (e.g. onto pipeline
    state) instead of re-deriving it at render time."""
    return {idx for idx, hand in hand_titles.items() if hand.interactive}


def assign_global_short_indexes(
    shorts_per_lf: "dict[int, list]",
    existing: "Optional[dict[str, int]]" = None,
) -> "dict[str, int]":
    """Assign each planned short a STABLE global index, keyed by
    its path (as ``str``), in plan order — ascending LF index, then list
    order within each LF (the order the shorts ladder is already in).

    ``existing`` (a previously-persisted ``{path_str: global_index}`` map,
    e.g. ``state.short_global_index``) is honoured verbatim for any path it
    already covers — an index, once assigned, never changes and is never
    reused. A short whose path drops out of ``shorts_per_lf`` on a later
    call (culled / failed to render) simply leaves its index unassigned to
    anything else: the next NEW path takes ``max(existing values) + 1``, not
    the freed slot, so a render failure leaves a hole rather than shifting
    every later short's index.

    Returns a NEW dict — never mutates ``existing`` in place.
    """
    assigned: dict[str, int] = dict(existing or {})
    next_index = (max(assigned.values()) + 1) if assigned else 1
    for lf_index in sorted(shorts_per_lf.keys()):
        for path in shorts_per_lf.get(lf_index) or []:
            key = str(path)
            if key in assigned:
                continue
            assigned[key] = next_index
            next_index += 1
    return assigned


def _full_month_year(formatted: str) -> str:
    """``"Sep 2025"`` (the generator's own date form) as ``"September 2025"``, the
    form of the ``Filmed <Month YYYY>.`` sentence; an unparseable value is kept."""
    try:
        return datetime.strptime(formatted, "%b %Y").strftime("%B %Y")
    except ValueError:
        return formatted


def format_recording_date(raw: "str | datetime | None") -> str:
    """Normalize a recording date string to 'Mon YYYY' format.

    Accepts:
        - datetime(2024, 9, 15)  -> "Sep 2024"
        - "2024-09-15"           -> "Sep 2024"
        - "September 10, 2024"   -> "Sep 2024"
        - "Sep 2024"             -> "Sep 2024" (passthrough)
        - ""                     -> ""
        - None                   -> ""

    The SHARED formatter for the canonical ``📅 Filmed: <date>`` line on
    BOTH long-form descriptions (``longform_metadata._build_gms_description``)
    and child shorts (``SocialGenerator``) — single source of truth so the
    two surfaces can never disagree on date shape.

    REJECTS (does NOT pass through) an unrecognized non-empty value: it
    logs a greppable ``FILMED-DATE-REJECT`` WARN and returns ``""`` so a
    malformed string can never burn into a published surface. Upstream the
    per-LF recording date is a validated ``day_<YYYY-MM-DD>`` token, so this
    is defense-in-depth, not the common path.
    """
    if not raw:
        return ""

    # Handle datetime objects directly
    if isinstance(raw, datetime):
        return raw.strftime("%b %Y")

    # Already in "Mon YYYY" format (e.g. "Sep 2024")
    try:
        parsed = datetime.strptime(raw, "%b %Y")
        return parsed.strftime("%b %Y")
    except ValueError:
        pass

    # ISO format: YYYY-MM-DD
    try:
        parsed = datetime.strptime(raw, "%Y-%m-%d")
        return parsed.strftime("%b %Y")
    except ValueError:
        pass

    # Full date: "September 10, 2024"
    try:
        parsed = datetime.strptime(raw, "%B %d, %Y")
        return parsed.strftime("%b %Y")
    except ValueError:
        pass

    # Unrecognized — REJECT rather than burn garbage. The old
    # pass-through let an unparseable string reach the published surface.
    logger.warning("FILMED-DATE-REJECT: unparseable recording date %r — dropping", raw)
    return ""


# ─── series suffix stripping ──────────────────────────────────────
# Robust-by-construction: applied at the title→tag/hashtag boundary so the
# series suffix " — Part N" can never bleed into discovery surfaces.
# The regex + helper now live in ``_series_utils`` (single source of truth,
# imported above) — shared with ``longform_metadata`` and ``region_resolver``.


# ─── drive-arc → single caption-safe anchor ───────────────────────


def _arc_safe_anchor(component: str) -> str:
    """Reduce a GPS-less drive-arc component to a single caption-safe place.

    A GPS-less LF whose title is an ``"A to B via {road}"`` / ``"A to B"`` arc
    leaks the WHOLE arc into the first comma-component (the state is the second),
    so every casual social surface interpolating ``{location}`` — captions,
    titles, hooks, the smashed location hashtag — read "Rolling Through Fancy Gap
    to Hillsville via I-77 🚗". Collapse the arc to its DESTINATION endpoint (what
    the highlights are ABOUT — mirrors the destination-anchored
    ``social_location``): drop a trailing ``" via {road}"`` clause, then keep the
    text after the final ``" to "``. A normal single-place component
    ("County Kerry", "Carroll County") has neither marker and passes through
    unchanged.
    """
    text = re.split(r"\s+via\s+", component.strip(), maxsplit=1)[0].strip()
    if " to " in text:
        text = text.rsplit(" to ", 1)[-1].strip()
    # A leading operator title word ("To Clonakilty") is not a place
    # noun — drop it here so it can never leak into a shorts title, caption,
    # ``social_location``, or the hashtag this same anchor feeds.
    text = parse_lf_label(text).place
    return text or component.strip()


def backed_bracket_label(text: str, bracket_route: "Optional[dict]") -> str:
    """``text`` when it is exactly the label of a valid ``--bracket-gpx`` record, else ``""``.

    A GPS-less drive's place comes from the fixes before and after the footage
    (``bracket_route``): its label ("Virginia to Pennsylvania") names the leg
    between two measured fixes. Only that exact label may keep its "A to B" shape
    in Short text; any other arc still collapses and any other direction framing
    is still refused.
    """
    from .bracket_route import bracket_record_valid

    label = str((bracket_route or {}).get("label") or "").strip()
    if label and bracket_record_valid(bracket_route) and text.strip() == label:
        return label
    return ""


# ─── location hashtags ────────────────────────────────────────────


def _location_hashtags(location: str, bracket_route: "Optional[dict]" = None) -> list[str]:
    """Emit one hashtag per comma-delimited location component.

    Splits on commas, strips a trailing series suffix, and removes any
    non-alphanumeric character inside each component (YouTube/IG reject
    punctuation in hashtags). Empty components are dropped, so a stray
    ``"Foo, , Bar"`` never produces a bare ``"#"``.

    The smashed concatenation produced by the legacy
    ``"#" + loc.replace(" ", "").replace(",", "")`` path is unrepresentable.
    """
    base = _strip_series_suffix(location)
    # A backed bracket-route label ("Virginia to Pennsylvania") names a leg, not a
    # place, so it gets no hashtag (JP 2026-11-16: "#VirginiatoPennsylvania" reads
    # badly); collapsing it to the destination would name an unmeasured endpoint.
    parts = [
        _arc_safe_anchor(p.strip())
        for p in base.split(",")
        if not backed_bracket_label(p, bracket_route)
    ]
    out: list[str] = []
    for part in parts:
        # NFKD-fold accented letters to their base ASCII form (ú→u) BEFORE
        # stripping punctuation. The bare [^A-Za-z0-9] strip deleted the fada
        # outright, turning "Dún Laoghaire" into "#DnLaoghaire"; folding first
        # keeps the letter → "#DunLaoghaire". YouTube/IG hashtags are ASCII.
        folded = unicodedata.normalize("NFKD", part).encode("ascii", "ignore").decode("ascii")
        cleaned = re.sub(r"[^A-Za-z0-9]", "", folded)
        if cleaned:
            out.append("#" + cleaned)
    return out


# ─── region tags ──────────────────────────────────────────────────


def _region_tags(
    location: str,
    resolved: "ResolvedLocation | None" = None,
    own_regions: "list[str] | None" = None,
) -> list[str]:
    """Emit region-derived tags from the resolved model, else the label string.

    When a per-LF :class:`~gomoveshift.models.ResolvedLocation` is
    available, the US-ness verdict comes from ``resolved.is_us`` — the SAME
    signal that drives the title verb, miles/feet units, and playlist routing
    — so the region tags agree with those surfaces *by
    construction* instead of independently re-parsing the display label (which
    could disagree if the trailing component isn't a recognised state). A US
    drive emits ``"USA"`` tags; a non-US drive uses the resolved model's
    geocoded ``country_code`` as the country noun (e.g. "Ireland") — the SAME
    country that routes the playlist — falling back to the label's last
    comma-component only when the model carries no country. An empty/degraded
    resolver (no geocoder or no usable points) carries no verdict and falls
    through to the string-parse path below.

    Back-compat string-parse fallback (no resolved location, or a degraded
    one): the country is the last comma-delimited component, with any series
    suffix stripped. A trailing US **state** (full name or 2-letter abbr) maps
    to ``"USA"`` — dominant-county labels are bare ``"County, State"`` strings
    with no country component now, and a state name must never land in
    the country slot. The literal string ``"road trip USA"`` is only ever
    emitted when the location is actually in the US — wrong-region output is
    unrepresentable on either path.

    ``own_regions`` (a per-short variant's OWN clip-GPS ``"County, State"``
    regions) replaces the parent's ``resolved`` verdict entirely: a short cut on
    a cross-border drive must not inherit the drive's country. A US state among
    them gives ``"USA"``; the regions carry no country, so anything else has no
    derivable country and yields no region tags.
    """
    from .title_builder import is_us_state

    if own_regions:
        if not any(is_us_state(r.rpartition(",")[2]) for r in own_regions if "," in r):
            return []
        return [f"USA {kind}" for kind in ("road trip", "drive", "POV drive")]
    base = _strip_series_suffix(location)
    parts = [p.strip() for p in base.split(",") if p.strip()]
    if not parts:
        return []

    # Resolver-driven verdict. is_us is meaningful only when the
    # builder actually resolved a country — a degraded ResolvedLocation()
    # leaves is_us=False AND country_code=None, which we must NOT read as
    # "non-US"; fall through to the string-parse in that case.
    if resolved is not None and (resolved.is_us or resolved.country_code):
        country = "USA" if resolved.is_us else (resolved.country_code or parts[-1])
    else:
        country = "USA" if is_us_state(parts[-1]) else parts[-1]
    # A real region noun is short ("USA", "Ireland", "County Kerry"). On a
    # GPS-less arc label with no country component ("X to Y via Z"), parts[-1]
    # is the WHOLE arc — emitting "{arc} road trip"/"{arc} drive" produces
    # 50+ char tags that blow YouTube's 500-char total (Metricool rejects the
    # import). Skip the region-variant tags when the noun isn't region-shaped.
    if len(country) > 20 or " to " in country or " via " in country:
        return []
    return [
        f"{country} road trip",
        f"{country} drive",
        f"{country} POV drive",
    ]


# ─── Terrain-aware descriptors ──────────────────────────────────────────
# classify_terrain() returns one of these keys; templates use {terrain_desc}
TERRAIN_DESCRIPTORS: dict[str | None, str] = {
    "coastal": "coastal roads",
    "mountain": "mountain roads",
    "winding": "winding roads",
    "countryside": "roads",  # GPS can't prove road type — keep neutral
    None: "roads",
}

TERRAIN_TAGS: dict[str | None, list[str]] = {
    "coastal": ["coastal drive", "ocean road", "seaside drive"],
    "mountain": ["mountain roads", "mountain pass", "alpine drive"],
    "winding": ["winding roads", "hairpin turns", "curvy roads"],
    "countryside": [],  # GPS can't determine road type — no specific tags
    None: [],
}
TERRAIN_HASHTAGS: dict[str | None, list[str]] = {
    "coastal": ["#CoastalDrive", "#OceanRoad", "#SeasideDrive"],
    "mountain": ["#MountainRoads", "#MountainPass", "#MountainDrive"],
    "winding": ["#WindingRoads", "#HairpinTurns", "#CurvyRoads"],
    "countryside": [],  # GPS can't determine road type — no specific hashtags
    None: [],
}


@dataclass
class SocialPost:
    """Generated social post for a single short."""

    short_index: int
    caption: str  # Main caption text (all platforms)
    youtube_title: str
    youtube_tags: str  # Comma-separated
    youtube_category: str
    tiktok_title: str
    pinterest_board: str
    first_comment: str  # Instagram first comment (extra hashtags)
    youtube_description: str = ""  # Short YT description (max 100 chars, for API upload)
    # This short's OWN resolved region label(s) — the county the clip's footage
    # is in (its burned OST label, region-verified upstream). The CSV region
    # gate checks each shipped surface's county against this set; empty for a
    # GPS-less / legacy short (no per-short region → gate no-ops).
    sampled_regions: list[str] = field(default_factory=list)
    #: The clip's OWN GPS road names; empty for an LF or a road-less clip.
    sampled_roads: list[str] = field(default_factory=list)
    #: The clip's measured route record (``RouteEvidence.record()``) for the place gate.
    route_record: Optional[dict] = None
    #: ``state.bracket_route`` when this GPS-less short's only place is the backed
    #: bracket-route label (its sole ``sampled_regions`` entry); the place gate then
    #: judges the label against this record instead of a route record it has none of.
    bracket_route: Optional[dict] = None
    # Whether this short is flagged for the interactive family — the
    # generator withholds the place from ``youtube_title``, ``tiktok_title``,
    # every ``captions[...]``/``caption`` and ``youtube_tags`` by
    # construction, and instead names it in ``youtube_description``'s last
    # line and ``first_comment``. Included in ``to_dict()`` so
    # downstream sidecars carry it.
    interactive: bool = False
    # One caption per platform ("youtube" | "facebook" | "instagram" |
    # "tiktok"). ``caption`` above stays the shared/legacy column and equals
    # the Instagram text — the other readers of ``.caption``
    # (content_gates.py, csv_generator.py's Metricool column, pubqueue.py,
    # instagram_adapter.py) keep reading it unchanged. Empty dict for legacy
    # construction (e.g. an older sidecar round-trip).
    captions: "dict[str, str]" = field(default_factory=dict)
    # Waypost's title decision for ``youtube_title`` (format "short").
    # The publish plan refuses a YouTube title that is not its
    # ``chosen_title``. None for an older sidecar round-trip.
    title_decision: Optional[dict] = None
    #: ``{experiment name: arm id}`` for the caption experiments active when
    #: this post was generated (``publishing/experiments.py``); ``{}`` = none.
    experiment_arms: "dict[str, str]" = field(default_factory=dict)
    #: ``{platform: caption_template_id}`` of the opener each composed caption
    #: was built from, the key Meta caption learning credits reach and plays to.
    #: A hand caption has no template, so its platforms are absent; ``{}`` for a
    #: hand-captioned Short or an older sidecar round-trip.
    caption_templates: "dict[str, str]" = field(default_factory=dict)
    #: ``{platform: Waypost CaptionRanking record}`` (JSON form) for each Meta
    #: caption whose opener order was re-ranked from ``caption_evidence.json``;
    #: absent when there was no evidence file or Waypost refused it.
    caption_rankings: "dict[str, dict]" = field(default_factory=dict)
    #: ``{"title"|"description": {"body", "reason", "kind"}}`` -- the place each
    #: surface names and why (``place_variety.PlaceChoice``): ``distinct``, ``varied``
    #: (another measured place replaced a repeat) or ``no_alternative`` (a repeat
    #: with no other measured place, today's text kept). ``{}`` for a hand title, a
    #: legacy title or an older sidecar round-trip.
    place_variety: "dict[str, dict]" = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "short_index": self.short_index,
            "caption": self.caption,
            "captions": dict(self.captions),
            "youtube_title": self.youtube_title,
            "youtube_tags": self.youtube_tags,
            "youtube_category": self.youtube_category,
            "youtube_description": self.youtube_description,
            "tiktok_title": self.tiktok_title,
            "pinterest_board": self.pinterest_board,
            "first_comment": self.first_comment,
            "interactive": self.interactive,
            "title_decision": self.title_decision,
            "experiment_arms": dict(self.experiment_arms),
            "caption_templates": dict(self.caption_templates),
            "caption_rankings": dict(self.caption_rankings),
            "place_variety": {k: dict(v) for k, v in self.place_variety.items()},
        }


# ─── Title composition ─────────────────────────────────────────
# The fixed-string title pools that used to live here are gone — a title
# is now COMPOSED from grounded fact slots on the generator
# (location, terrain, road, county, moment kind) via a small set of grammar
# frames. A frame is sentence SHAPE only — it never stands in as a complete
# title without slot data, and every frame here requires >=2 distinct slots
# so a composed title always carries >=2 grounded facts. ``location`` and
# ``terrain`` are always available (``location_short`` / ``terrain_desc``
# default to "roads" when unknown); ``road``, ``county`` and ``moment`` are
# optional and only widen the applicable-frame set when present — never
# required. At most 6 frames (spec cap). Auto-discovered and content-
# validated by test_social_generator_safety.py like the pools it replaces.
TITLE_FRAME_TEMPLATES = [
    "{location}, by {terrain_desc}",
    "A drive along the {terrain_desc} of {location}",
    "{terrain_desc} in {location}",
    "{road} cuts through {location}",
    "{county}'s {terrain_desc}, featuring {location}",
    "{moment}: the {terrain_desc} of {location}",
]

# ─── Place variety (unit C2) ───────────────────────────────────────
# Near-zero-view Shorts line up with the same place named Short after Short.
# Within a project, and from one publish day to the next, a Short's place body
# must differ from the previous Short's whenever its own clip measured another
# place. Robust by construction: candidates are only places the Short's
# final-text review already accepts (``publishing.clip_place_evidence.clip_towns`` /
# ``clip_places``: a town held for every retained second, a county the clip's GPS
# voted) plus the clip's measured road, offered to the DESCRIPTION line only (the
# review refuses a road named in a title: it cites the title's one place, and a
# road is evidence for a tag only). Nothing is ever guessed.
# ``choose_place_body`` is the pure chooser; later units (queued-row repair) import it.
#: ``PlaceChoice.reason`` values. Recorded on the Short as ``place_variety``.
REASON_DISTINCT = "distinct"  # today's body is not a repeat; kept
REASON_VARIED = "varied"  # today's body repeated; another measured place used
REASON_NO_ALTERNATIVE = "no_alternative"  # repeated, no other measured place; kept

KIND_TOWN = "town"
KIND_ROAD = "road"
KIND_COUNTY = "county"
#: Most specific first.
KIND_ORDER = (KIND_TOWN, KIND_ROAD, KIND_COUNTY)


@dataclass(frozen=True, slots=True)
class PlaceChoice:
    """The chosen place ``body``, why (``reason``), and its ``kind`` (``"today"``
    when today's text was kept, else town/road/county)."""

    body: str
    reason: str
    kind: str = "today"

    def to_dict(self) -> dict:
        return {"body": self.body, "reason": self.reason, "kind": self.kind}


def _norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def choose_place_body(
    candidates: Sequence[str],
    previous: Sequence[str] = (),
    *,
    kinds: Optional[Sequence[str]] = None,
) -> PlaceChoice:
    """Pick the Short's place body: ``candidates[0]`` (today's text) unless it
    repeats one of ``previous`` (the previous Short's body), then the first later
    candidate that does not. Pure: no I/O, deterministic.

    ``candidates`` is today's body first, then the clip's other measured places in
    :data:`KIND_ORDER`; ``kinds`` (parallel, optional) labels them for the result.
    No candidate left -> today's body with ``no_alternative``. Comparison ignores
    case and spacing."""
    if not candidates:
        raise ValueError("choose_place_body needs today's body as the first candidate")
    seen = {_norm(p) for p in previous if _norm(p)}
    today = candidates[0]
    if _norm(today) not in seen:
        return PlaceChoice(today, REASON_DISTINCT)
    for i, cand in enumerate(candidates[1:], start=1):
        if _norm(cand) and _norm(cand) not in seen:
            kind = kinds[i] if kinds and i < len(kinds) else "other"
            return PlaceChoice(cand, REASON_VARIED, kind)
    return PlaceChoice(today, REASON_NO_ALTERNATIVE)


def measured_places(
    regions: Sequence[str] = (),
    roads: Sequence[str] = (),
    route_record: Optional[Mapping] = None,
) -> "list[tuple[str, str]]":
    """The clip's measured ``(kind, body)`` places, town, road, county order, each
    once. Towns and counties come from the same functions the Short's review
    reads, so every town/county here is one the review accepts."""
    from ..publishing.clip_place_evidence import clip_places, clip_towns

    rec = SimpleNamespace(sampled_regions=tuple(regions), route_record=dict(route_record or {}))
    found = (
        [(KIND_TOWN, t["value"]) for t in clip_towns(rec)]
        + [(KIND_ROAD, r.strip()) for r in roads if r and r.strip()]
        + [(KIND_COUNTY, c["value"]) for c in clip_places(rec, "")]
    )
    out: "list[tuple[str, str]]" = []
    seen: set = set()
    for kind, body in found:
        if _norm(body) not in seen:
            seen.add(_norm(body))
            out.append((kind, body))
    return out


def _title_place_bodies(
    measured: "Sequence[tuple[str, str]]", today: str
) -> "list[tuple[str, str]]":
    """``(kind, body)`` title candidates, most specific first: the measured towns,
    then the counties, then ``today`` (the Short's own label) when it is neither."""
    found = [m for m in measured if m[0] != KIND_ROAD]
    found.sort(key=lambda m: KIND_ORDER.index(m[0]))
    if today and _norm(today) not in {_norm(b) for _, b in found}:
        found.append(("today", today))
    return found


def town_evidence(route_record: Optional[Mapping]) -> "dict[str, str]":
    """Town name -> evidence id for the towns the review accepts (``towns=`` of
    ``title_decision.short_place_facts``)."""
    from ..publishing.clip_place_evidence import clip_towns

    rec = SimpleNamespace(route_record=dict(route_record or {}))
    return {t["value"]: t["evidence_id"] for t in clip_towns(rec)}


# ─── Fact titles and stamped titles (drive path) ─────────────────────
# A drive short's title is the place its own clip measured and nothing else: the
# most specific place the final-text review accepts (a town held for every retained
# second, else a county the clip voted -- ``_title_place_bodies``). The review backs
# ONE cited place per title and no road, so no second place, road, date or count is
# added. Titles stay unique against the published ledger and within a batch
# (``_fact_title``); a Short with no unused measured place left, and a Short whose
# place is not verified, take the STAMPED title instead: ``"{stamp}: {body}"``. The
# stamp -- ``"Sep 14, 3rd stretch"`` -- is built from the short's own day key and its
# position in that day's sidecar, and IS the whole first-3-word prefix, so those titles
# are unique by construction (see docs/pipeline/135-shorts-titles-and-interactive-family.md).
# Only the body rotates. The body may name only the short's own ``location_short``, and
# only when its place is clip-GPS verified; no preposition precedes ``{location}`` (a town
# label is the nearest settlement, not proof of containment). ``TITLE_FRAME_TEMPLATES``
# above still serves callers with no facts (the single-video stage).
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

STAMPED_BODY_TEMPLATES = [  # place verified
    "{location}",
    "{location}, on a {weekday}",
    "{secs} seconds, {location}",
]
STAMPED_PLACE_FREE_TEMPLATES = [  # place not verified (dry-run only; publish refuses)
    "{secs} seconds on a {weekday}",
]


def _parse_iso_date(date_iso: str) -> datetime:
    if not isinstance(date_iso, str) or not _ISO_DATE_RE.fullmatch(date_iso):
        raise ValueError(f"title date must be YYYY-MM-DD, got {date_iso!r}")
    return datetime.strptime(date_iso, "%Y-%m-%d")


def weekday_name(date_iso: str) -> str:
    """The weekday word a Short's title says for ``date_iso`` ("Tuesday")."""
    return _WEEKDAYS[_parse_iso_date(date_iso).weekday()]


@dataclass(frozen=True)
class TitleFacts:
    """Per-short facts a stamped title is composed from (all read from the day
    sidecar): ``date`` (YYYY-MM-DD day key), ``ordinal`` (1-based position in
    that day, by start time), ``secs`` (clip length) and ``place_ok`` (the
    short's own place is clip-GPS verified, so its label may be named)."""

    date: str
    ordinal: int
    secs: int
    place_ok: bool

    def __post_init__(self) -> None:
        _parse_iso_date(self.date)
        if self.ordinal < 1 or self.secs < 1:
            raise ValueError(f"TitleFacts needs ordinal >= 1 and secs >= 1: {self!r}")


def ordinal_suffix(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def title_stamp(date_iso: str, ordinal: int) -> str:
    """``"Sep 14, 3rd stretch"``. Fixed month names (never locale ``%b``)."""
    day = _parse_iso_date(date_iso)
    return f"{_MONTHS[day.month - 1]} {day.day}, {ordinal}{ordinal_suffix(ordinal)} stretch"


def day_title_ordinals(entries: Iterable[Mapping]) -> dict:
    """``{rank: k}``: each sidecar row's 1-based position by ``(start_sec, rank)``.

    A failed render keeps its row, so later shorts are never renumbered.
    Raises ``ValueError`` on a missing/non-numeric ``start_sec``/``rank`` or a
    duplicate rank."""
    keyed = []
    for entry in entries:
        try:
            keyed.append((float(entry["start_sec"]), int(entry["rank"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"sidecar entry needs numeric start_sec and rank: {entry!r}") from exc
    ranks = [rank for _, rank in keyed]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"duplicate short rank in day sidecar: {sorted(ranks)}")
    return {rank: k for k, (_, rank) in enumerate(sorted(keyed), 1)}


def assert_title_prefixes_unique(titles_by_key: Mapping, *, where: str) -> None:
    """Refuse titles sharing a first-3-word prefix, naming every key."""
    from ..publishing.anti_spam_gate import shared_prefix_groups

    keys = list(titles_by_key)
    groups = shared_prefix_groups([titles_by_key[k] for k in keys])
    if groups:
        detail = "; ".join(
            f"{' '.join(prefix)!r}: {[keys[i] for i in indices]}"
            for prefix, indices in groups.items()
        )
        raise RuntimeError(
            f"{where}: titles share a first-3-word prefix (the anti-spam gate refuses these at enqueue): {detail}"
        )


def assert_day_title_stamps(date: str, entries: Sequence[Mapping]) -> None:
    """Plan-time check, before any burn: every entry gets a stamp and no two
    stamps share a first-3-word prefix. Raises ``ValueError``/``RuntimeError``."""
    ordinals = day_title_ordinals(entries)
    if len(ordinals) != len(entries):
        raise ValueError(f"day {date}: {len(entries)} sidecar entries but {len(ordinals)} stamps")
    assert_title_prefixes_unique(
        {rank: title_stamp(date, k) + ": x" for rank, k in ordinals.items()},
        where=f"day {date} pre-burn",
    )


# A short is ~30 s of a drive: its title/caption may name ONLY the place its own
# clip was filmed. "Drive to X" framing names a destination the clip is not at.
_DIRECTION_FRAMING_RE = re.compile(
    r"\b(?i:to|into|toward|towards|approaching|entering|heading\s+(?:to|for)|"
    r"bound\s+for|on\s+the\s+way\s+to|en\s+route\s+to)\s+(?:the\s+)?[A-Z]"
)
_PLACE_TOKEN_RE = re.compile(r"[A-Z][A-Za-z'-]{3,}")
_PLACE_TOKEN_STOPLIST = frozenset(
    {
        "County", "Road", "Roads", "Drive", "Route", "Highway", "Street", "Lane",
        "Park", "State", "National", "North", "South", "East", "West", "Through",
    }
)  # fmt: skip

# Word/phrase a moment kind (short_planner.ShortRange.moment_kind, threaded
# in via SocialGenerator(moment_kinds=...)) renders as in the {moment} slot.
_MOMENT_PHRASES: "dict[str, str]" = {
    "peak": "The peak moment",
    "sustained": "A long stretch",
}

# Weather-specific title variants REMOVED. Weather here
# is a brightness heuristic (weather.py) that cannot verify rain/sun/overcast as
# fact — "Rainy Day Drive", "Sunny Day Drive", "Overcast Vibes" etc. asserted a
# publicly-checkable weather claim the pipeline can't stand behind. Weather now
# surfaces ONLY as the soft caption emoji ({weather_emoji}); no weather-claim
# title template exists.


def _frame_fields(template: str) -> "set[str]":
    """Placeholder names a title frame references (e.g. {"location", "road"})."""
    return {name for _, name, _, _ in Formatter().parse(template) if name}


# ─── Per-platform caption parts ────────────────────────────────────
# A caption is COMPOSED, never drawn whole from a fixed pool: an opener
# fragment (this platform's own pool, so no phrase is shared across
# platforms), then either a data-only detail clause or, for an ``interactive``
# short, an engagement prompt. Each part rotates through its own pool in a
# per-project shuffled order (``_rotating_pick``), independently of the others,
# so a batch of up to ``len(opener pool)`` shorts never repeats an opener and
# two projects at the same place open differently. The pools are sized to the
# 12-short batch the anti-spam gate's 6-word-prefix rule is held to; a
# longer batch wraps, and ``check_batch`` refuses a repeat across batches.
#
# Every opener requires {location} -- the mandatory place fact slot; when a
# short is flagged ``interactive``, ``_build_platform_captions`` fills
# {location} with LOCATION_WITHHELD instead so the place is never named,
# never by omitting the slot from some openers and not others. A caption states
# the measured place and, for classified terrain only, one terrain clause:
# facts the record holds, no filler. Shapes are few and disjoint per platform. No shape
# claims a camera, a vehicle, a viewpoint or the weather: the pipeline measures none of
# them ("Dashcam footage from ...", "POV", "from the driver's seat", "by car" are gone).
# "Passing through" stays on Facebook only because ``facebook_caption_edit.MEASURED_TEMPLATE``
# is pinned to it.
YOUTUBE_CAPTION_TEMPLATES = [
    "{location}.",
    "Location: {location}.",
    "Clip location: {location}.",
]

FACEBOOK_CAPTION_TEMPLATES = [
    "Passing through {location}.",
    "In {location}.",
    "This is {location}.",
]

INSTAGRAM_CAPTION_TEMPLATES = [
    "📍 {location}",
    "Here: {location}",
    "At {location}",
]

TIKTOK_CAPTION_TEMPLATES = [
    "{location} clip",
    "Clip from {location}",
    "Footage from {location}",
]

#: The one terrain clause a caption may carry, and only for a terrain the
#: classifier named. Unclassified terrain (``countryside``, ``None``) states
#: nothing: a caption never says "Mostly roads".
CAPTION_DETAIL_TEMPLATES = [
    "{terrain_desc}.",
]

#: Terrain classes whose clause a caption may state.
_CAPTION_TERRAINS = ("coastal", "mountain", "winding")

#: An ``interactive`` short withholds its place, so it is the guess prompt
#: alone: no opener, no terrain clause (either would hint at the answer).
INTERACTIVE_PROMPT_TEMPLATES = [
    "Where do you think this is?",
    "Can you guess the route?",
    "Any guesses where this is?",
    "Could you place this drive?",
    "Where would you say this was filmed?",
    "Think you know where this is?",
]

_PLATFORM_CAPTION_TEMPLATES = {
    "youtube": YOUTUBE_CAPTION_TEMPLATES,
    "facebook": FACEBOOK_CAPTION_TEMPLATES,
    "instagram": INSTAGRAM_CAPTION_TEMPLATES,
    "tiktok": TIKTOK_CAPTION_TEMPLATES,
}


def caption_template_id(template: str) -> str:
    """The stable id of one caption opener template: the first 12 hex digits of the
    SHA-256 of its text. Reordering a pool keeps every id; rewording a template gives
    it a new one, so its performance is never credited to the old wording."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:12]


#: Generic hashtags rotated per short (``_platform_hashtags`` takes
#: ``GENERIC_HASHTAGS_PER_POST`` of them), so no two posts need carry the same
#: fixed generic pair.
GENERIC_HASHTAGS = [
    "#POVDriving",
    "#RoadTrip",
    "#DriveWithMe",
    "#POVDrive",
    "#OpenRoad",
    "#DrivingVideo",
]
GENERIC_HASHTAGS_PER_POST = 2

#: The lead of a short's YouTube description ("{lead} driving through
#: {location} Pt.N"), rotated per project like the caption openers. The gate's
#: ``cross-batch-repeat`` key is the first 6 words, and the part number is
#: word 7, so one fixed lead made every drive at a place open alike. No lead
#: may open with a bare preposition ("Through ..."): the plan grammar gate
#: (``find_grammar_issues``) refuses that shape and fails the whole plan.
YOUTUBE_DESCRIPTION_LEADS = [
    "POV",
    "Dashcam POV",
    "Driver's-seat view",
    "Front-seat view",
    "Windshield view",
    "Ride along",
    "Eyes forward",
    "Hands on the wheel",
    "Dashboard view",
    "Over the hood",
    "Seat belt on",
    "Road ahead",
    "Wheel's-eye view",
    "Cab view",
]

#: The ``june`` experiment hashtag mode (``publishing/experiments.py``): the
#: June 2026 Facebook posts carried these three generics, then the place and
#: region tags (upstream plan Appendix A, e.g. "#POVDriving #ScenicDrive
#: #RoadTrip #StradballytoTralee #CountyKerry").
JUNE_STYLE_HASHTAGS = ["#POVDriving", "#ScenicDrive", "#RoadTrip"]

# The place-withheld filler for an ``interactive`` short's captions —
# never a real place, so the interactive family can't leak one through a
# platform caption while the title/OST correctly hide it.
LOCATION_WITHHELD = "this route"

# Base tags that go on every video (kept small so rotating tags fit in 15 cap).
# Each is a Waypost tag template, so a Short's final-text review can cover it.
# "driving" and "travel" have no template and were dropped.
BASE_TAGS = [
    "POV driving",
    "drive",
    "road trip",
    "drive with me",
]


def _opener_keys(texts: "Mapping[str, Sequence[str]]") -> "dict[str, list[tuple[str, ...]]]":
    """Platform -> the gate's 6-word opener key of each text."""
    from ..publishing.anti_spam_gate import text_opener_key

    return {
        platform: [k for k in (text_opener_key(t) for t in platform_texts) if k]
        for platform, platform_texts in texts.items()
    }


def _opens_like(text: str, keys: "Sequence[tuple[str, ...]]") -> bool:
    """Could text starting with ``text`` share a 6-word opener key with one of
    ``keys``? A shorter fragment matches on the words it has, since what
    follows it is not chosen here."""
    from ..publishing.anti_spam_gate import CROSS_BATCH_PREFIX_WORDS, _normalize_words

    words = tuple(_normalize_words(text)[:CROSS_BATCH_PREFIX_WORDS])
    return any(key[: len(words)] == words for key in keys) if words else False


def _read_spooled_texts(project_name: str) -> "dict[str, list[str]]":
    """Text other projects spooled in the gate's window, read-only from the
    queue beside ``DEFAULT_SEO_STORAGE``; no queue (or unreadable) is ``{}``
    and creates nothing."""
    import sqlite3

    from ..publishing import anti_spam_gate
    from . import longform_metadata

    db_path = Path(longform_metadata.DEFAULT_SEO_STORAGE) / "social_queue.sqlite3"
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        logger.warning("spooled-openers: cannot open %s: %s", db_path, exc)
        return {}
    conn.row_factory = sqlite3.Row
    try:
        return anti_spam_gate.recent_platform_texts(conn, exclude_projects={project_name})
    except sqlite3.Error as exc:
        logger.warning("spooled-openers: cannot read %s: %s", db_path, exc)
        return {}
    finally:
        conn.close()


class SocialGenerator:
    """
    Generates unique social media posts for each short.

    Uses a deterministic seed based on project name + short index so
    the same inputs always produce the same outputs (reproducible).
    Varies titles, captions, hooks, and hashtags per short.
    """

    def __init__(
        self,
        location: str,
        project_name: str = "",
        pinterest_board: str = "POV Driving | Road Trips",
        youtube_category: str = "19",  # YouTube numeric category ID (19 = Travel & Events)
        recording_date: "str | datetime | None" = None,
        series_part: Optional[int] = None,
        terrain_type: Optional[str] = None,
        road_name: Optional[str] = None,
        county: Optional[str] = None,
        nearest_poi: Optional[str] = None,
        roads: Optional[list[str]] = None,
        resolved_location: Optional[ResolvedLocation] = None,
        titles_file: Optional[Path] = None,
        moment_kinds: "Optional[Mapping[int, str]]" = None,
        experiments: "Optional[Sequence[Experiment]]" = None,
        experiment_today: Optional[date] = None,
        spooled_texts: "Optional[Mapping[str, Sequence[str]]]" = None,
        bracket_route: "Optional[dict]" = None,
    ):
        """
        Args:
            location: Location name (e.g., "County Tipperary, Ireland")
            bracket_route: ``state.bracket_route`` of a GPS-less drive. A text
                that is exactly its valid label keeps its "A to B" shape in the
                Short's title/location; nothing else does.
            project_name: Project name for deterministic seeding
            pinterest_board: Pinterest board name
            youtube_category: YouTube numeric category ID (default "19" = Travel & Events)
            recording_date: Formatted recording date (e.g., "Sep 2024")
            series_part: Part number within a multi-video series (changes seed)
            terrain_type: GPS-derived terrain ("coastal", "mountain", "winding",
                "countryside", or None for unknown)
            road_name: Notable road name for extended title templates (e.g., "US-64")
            county: County name for extended title templates (e.g., "Macon County")
            nearest_poi: Nearest GPS-verifiable point of interest (e.g., "Hunting Island State Park")
            roads: List of road names traversed (e.g., ["US-21", "Sea Island Parkway"])
            resolved_location: Canonical per-LF ``ResolvedLocation``.
                When supplied, region tags derive their US-ness from
                ``resolved_location.is_us`` so they agree with the title/units/
                playlist by construction; absent, the legacy string-parse of
                ``location`` is used.
            titles_file: Hand-title file (``<project_dir>/social/titles.yaml``),
                a mapping of ``short_index`` (the same 1-based index
                ``generate_all`` uses after ``index_offset``) to
                ``{title, caption?, interactive?}``. When it covers an index,
                ``generate_post`` uses the hand title/caption instead of the
                template pool. ``None`` (default) means no hand-title lane.
            moment_kinds: Optional ``{short_index: "peak"|"sustained"}``
                map (``short_planner.ShortRange.moment_kind``) composition may
                use as an extra title fact slot. Absent index or absent
                mapping simply leaves that slot unavailable — never raises.
            experiments: Caption experiments (``publishing/experiments.py``).
                ``None`` (default) reads the current brand's
                ``policy.experiments``; ``()`` runs none.
            experiment_today: The date experiment windows are judged against
                (default: today); fixed for tests and reruns.
            spooled_texts: Platform -> text other projects already spooled
                (``anti_spam_gate.recent_platform_texts``). Caption openers
                and description leads skip any that would open the same 6
                words, so the gate's ``cross-batch-repeat`` rule never fires
                on generated text. ``None`` (default) reads the queue beside
                ``DEFAULT_SEO_STORAGE`` on first use; ``{}`` avoids nothing.
        """
        self._spooled_keys: "Optional[dict[str, list[tuple[str, ...]]]]" = (
            None if spooled_texts is None else _opener_keys(spooled_texts)
        )
        #: Meta caption evidence, loaded on first use; ``False`` = none usable.
        self._caption_evidence = None
        self._title_snapshot = None
        self._caption_rankings: "dict[str, dict]" = {}
        if experiments is None:
            from ..console.settings import current_brand
            from ..publishing.brand_policy import load_brand_policy

            experiments = load_brand_policy(current_brand()).experiments
        self.experiments = tuple(experiments)
        self.experiment_today = experiment_today or date.today()
        self.location = location
        self._parent_place_tokens: "list[str]" = []
        self.titles_file = titles_file
        self.moment_kinds = moment_kinds
        self._hand_titles = load_hand_titles(titles_file) if titles_file else {}
        self._pool_fallback_logged = False
        self.resolved_location = resolved_location
        # Set only on a per-short variant: the short's OWN clip-GPS regions.
        self._own_regions: Optional[list[str]] = None
        self._own_roads: list[str] = []
        self.project_name = project_name
        self.pinterest_board = pinterest_board
        self.youtube_category = youtube_category
        self.recording_date = format_recording_date(recording_date)
        self.series_part = series_part
        self.terrain_type = terrain_type
        self.terrain_desc = TERRAIN_DESCRIPTORS.get(terrain_type, "roads")
        self.nearest_poi = nearest_poi
        self.roads = roads
        # Fallback: if roads provided but road_name is None, use first road
        if road_name is None and roads:
            self.road_name = roads[0]
        else:
            self.road_name = road_name
        self.county = county

        # Short location for casual social content (region only, no country)
        # Strip any trailing " — Part N" before splitting so the first
        # component is the actual region, not a region with series suffix.
        location_no_suffix = _strip_series_suffix(location)
        parts = [p.strip() for p in location_no_suffix.split(",")]
        # ``location_short_full`` keeps the whole first component — including any
        # "A to B [via {road}]" drive-arc / origin — for the surfaces that name
        # the WHOLE route: the long-form ``youtube_description`` and the phrase/
        # compound YouTube tags. ``location_short`` (the BURNED-caption /
        # title / hook / hashtag anchor) instead collapses that arc to its single
        # destination, so those casual surfaces never read "Rolling Through Fancy
        # Gap to Hillsville via I-77 🚗". A normal single-place label leaves the
        # two identical.
        # A leading operator title word ("To Clonakilty") is not a
        # place noun — strip it here too, so the shorts ``youtube_description``
        # (via ``social_location_full``) never reads "through To Clonakilty".
        # Anchored at the string start, so a mid-string arc ("Fancy Gap to
        # Hillsville") — kept whole here on purpose — is untouched.
        self.location_short_full = (
            parse_lf_label(parts[0]).place if parts else parse_lf_label(location_no_suffix).place
        )
        self._bracket_route = bracket_route
        self.location_short = backed_bracket_label(
            self.location_short_full, bracket_route
        ) or _arc_safe_anchor(self.location_short_full)

        # Destination-anchored location text for the secondary text surfaces
        # (social CAPTION + shorts YouTube DESCRIPTION). When the canonical
        # per-LF ``ResolvedLocation`` named a destination POI (a WMA / reserve /
        # park / named viewpoint enclosing the LF's montage-peak footage), these
        # surfaces should name what the highlights are ABOUT — consistent with
        # the burned OST + long-form/shorts title which already anchor it via
        # ``build_lf_title``. Shape mirrors the OST builder: ``"{dest}, {region}"``
        # (the destination carries no region of its own), with the same
        # double-region guard so an already-qualified destination isn't
        # re-suffixed. ``None`` destination / no resolved model preserves the
        # legacy ``location_short`` text exactly (no-destination case unchanged).
        _destination = (
            (resolved_location.destination or "").strip() if resolved_location is not None else ""
        )

        def _anchor_with_dest(region: str) -> str:
            if not _destination:
                return region
            if region and not _destination.endswith(f", {region}"):
                return f"{_destination}, {region}"
            return _destination

        # CAPTION anchors on the arc-free ``location_short``; the DESCRIPTION
        # (``social_location_full``) keeps the whole route text. Identical when
        # the label carries no arc.
        self.social_location = _anchor_with_dest(self.location_short)
        self.social_location_full = _anchor_with_dest(self.location_short_full)

        # Per-component hashtags. Replaces the legacy single smashed
        # hashtag (``"".join(c for c in location if c.isalnum())``) which
        # produced unreadable output like ``#DinglePeninsulaCountyKerryIrelandPart1``.
        self.location_hashtags = _location_hashtags(location, bracket_route)
        # Backwards-compat alias for any external caller that read .location_tag.
        # Now points to the first (most-specific) component instead of the
        # smashed concatenation. Discovery surfaces should prefer
        # .location_hashtags.
        if self.location_hashtags:
            self.location_tag = self.location_hashtags[0][1:]
        elif backed_bracket_label(self.location_short_full, bracket_route):
            self.location_tag = ""  # a backed bracket leg gets no hashtag (above)
        else:
            self.location_tag = "".join(c for c in self.location_short if c.isalnum())

        # Track POI usage within this instance for description deduplication
        self._used_pois: dict[str, int] = {}
        # Titles this batch already used, by title kind ("titles", "tiktok"), shared
        # with every per-short variant: a fact title never repeats within a batch.
        self._used_titles: "dict[str, set[str]]" = {}

    def _project_seed(self, suffix: str = "") -> int:
        """Get a deterministic seed for this project + series part."""
        base = f"{self.project_name}:{self.series_part or 0}"
        if suffix:
            base += f":{suffix}"
        return int(hashlib.md5(base.encode()).hexdigest()[:8], 16)

    def _get_rng(self, short_index: int) -> random.Random:
        """Get a deterministic RNG for this short."""
        seed = f"{self.project_name}:{short_index}:{self.series_part or 0}"
        seed_hash = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
        return random.Random(seed_hash)

    def _place_strings(self) -> "list[str]":
        """Distinct, non-empty strings that name THIS generator's place.

        Used to guarantee an ``interactive`` short's hand title/caption never
        leaks the answer — the same set of surfaces
        :func:`_build_platform_captions` and :func:`_build_tags` withhold.
        """
        candidates = [
            self.location_short,
            self.location_short_full,
            self.social_location,
            self.social_location_full,
            self.county,
        ]
        seen: set[str] = set()
        result = []
        for c in candidates:
            if c and c.lower() not in seen:
                seen.add(c.lower())
                result.append(c)
        return result

    def _check_hand_text_place(self, hand_idx: int, hand: "HandTitle") -> None:
        """Refuse hand text naming a place this clip was not filmed at.

        Two refusals: direction framing before a capitalised word ("into
        Gettysburg"), and — on a per-short variant — a token of the PARENT LF's
        place that the short's own label does not carry.
        """
        for kind, text in (("title", hand.title), ("caption", hand.caption)):
            if not text:
                continue
            # The one lifted case: the backed bracket-route label itself.
            label = backed_bracket_label(self.location_short_full, self._bracket_route)
            m = _DIRECTION_FRAMING_RE.search(text.replace(label, "") if label else text)
            if m:
                raise ValueError(
                    f"short_index {hand_idx}: hand {kind} uses direction framing "
                    f"{m.group(0)!r}: {text!r} — the {kind} may name only the "
                    f"place this clip was filmed ({self.location!r}), not a "
                    f"destination"
                )
            for token in self._parent_place_tokens:
                if re.search(rf"\b{re.escape(token)}\b", text):
                    raise ValueError(
                        f"short_index {hand_idx}: hand {kind} names {token!r}, a "
                        f"place of the parent long-form, not this clip's own "
                        f"place ({self.location!r}): {text!r} — the {kind} may "
                        f"name only the place this clip was filmed"
                    )

    def _leaks_place(self, text: "Optional[str]") -> "Optional[str]":
        """Return the first place-string found in ``text`` (case-insensitive), else None."""
        if not text:
            return None
        text_lower = text.lower()
        for place in self._place_strings():
            if place.lower() in text_lower:
                return place
        return None

    def _title_slots(self, short_index: int) -> "defaultdict[str, str]":
        """Grounded fact slots available for title composition."""
        moment = ""
        if self.moment_kinds is not None:
            moment = _MOMENT_PHRASES.get(self.moment_kinds.get(short_index) or "", "")
        return defaultdict(
            str,
            location=self.location_short,
            terrain_desc=self.terrain_desc.title(),
            road=self.road_name or "",
            county=self.county or "",
            moment=moment,
        )

    def _compose_stamped_title(
        self,
        facts: TitleFacts,
        seed_suffix: str,
        max_len: int,
        place: Optional[str] = None,
        title_body_arm: Optional[str] = None,
        candidates: "Sequence[str]" = (),
    ) -> str:
        """The Short's title: its measured place (:meth:`_fact_title`), else
        ``"{stamp}: {body}"``. The ledger walk rotates the BODY only; the
        stamp (<=22 chars, the whole first-3-word prefix) never changes or truncates.

        ``candidates`` are the measured place bodies to try, most specific first;
        a verified place with none left unused takes the stamped title (logged).
        ``title_body_arm`` (a ``yt_title_body`` arm id) narrows a place-named
        stamped title's bodies to that arm's; ``None`` or an unknown id keeps them all."""
        stamp = title_stamp(facts.date, facts.ordinal)
        place = place or self.location_short
        named = facts.place_ok and bool(place)
        if named:
            fact_title = self._fact_title(seed_suffix, max_len, [place, *candidates])
            if fact_title:
                return fact_title
            logger.info(
                "no unused measured place for the %s title of %s #%s; stamped title",
                seed_suffix,
                facts.date,
                facts.ordinal,
            )
        slots = defaultdict(
            str,
            location=place if named else "",
            secs=str(facts.secs),
            weekday=weekday_name(facts.date),
        )
        bodies = STAMPED_BODY_TEMPLATES if named else STAMPED_PLACE_FREE_TEMPLATES
        arm_bodies = self._arm_title_bodies(facts, place, title_body_arm, max_len)
        if arm_bodies:
            bodies = arm_bodies
        start = (facts.ordinal + self._project_seed(seed_suffix)) % len(bodies)

        from .longform_metadata import load_published_titles

        published = {t.strip().lower() for t in load_published_titles()}
        for offset in range(len(bodies)):
            body = bodies[(start + offset) % len(bodies)]
            candidate = f"{stamp}: {body.format_map(slots)}"[:max_len]
            if candidate.strip().lower() not in published:
                return candidate
        return f"{stamp}: {bodies[start].format_map(slots)}"[:max_len]

    def _fact_title(self, seed_suffix: str, max_len: int, bodies: "Sequence[str]") -> str:
        """The first of ``bodies`` (measured places, most specific first) that fits
        ``max_len`` whole and is not a published title, not already used by this
        batch's ``seed_suffix`` titles and shares no first-3-word prefix with one;
        ``""`` when none is left."""
        from ..publishing.anti_spam_gate import text_opener_key, title_prefix_key
        from .longform_metadata import load_published_titles

        published = {t.strip().lower() for t in load_published_titles()}
        used = self._used_titles.get(seed_suffix, set())
        used_prefixes = {k for k in map(title_prefix_key, used) if k}
        # The queue gate refuses a title whose whole opener key a spooled text already holds.
        spooled = set(self._spooled_opener_keys().get("youtube", []))
        for body in dict.fromkeys(b.strip() for b in bodies if b and b.strip()):
            key = body.lower()
            if len(body) > max_len or key in published or key in used:
                continue
            if text_opener_key(body) in spooled:
                continue
            if title_prefix_key(body) in used_prefixes:
                continue
            return body
        return ""

    def _arm_title_bodies(
        self, facts: TitleFacts, place: Optional[str], arm: Optional[str], max_len: int
    ) -> "list[str]":
        """The ``arm``'s body templates that make a usable title for ``facts``: the
        title names a verified place, the whole title fits ``max_len`` (never cut
        mid-word) and is not already published. ``[]`` means the arm cannot be
        carried, so the caller drops it and walks the full template set."""
        place = place or self.location_short
        if arm not in SHORTS_TITLE_BODY_ARMS or not (facts.place_ok and place):
            return []
        from .longform_metadata import load_published_titles

        stamp = title_stamp(facts.date, facts.ordinal)
        slots = defaultdict(
            str, location=place, secs=str(facts.secs), weekday=weekday_name(facts.date)
        )
        published = {t.strip().lower() for t in load_published_titles()}
        usable = []
        for body in SHORTS_TITLE_BODY_ARMS[arm]:
            candidate = f"{stamp}: {body.format_map(slots)}"
            if len(candidate) <= max_len and candidate.strip().lower() not in published:
                usable.append(body)
        return usable

    def _compose_title(
        self,
        short_index: int,
        seed_suffix: str,
        max_len: int,
        facts: Optional[TitleFacts] = None,
        place: Optional[str] = None,
        title_body_arm: Optional[str] = None,
        candidates: "Sequence[str]" = (),
    ) -> str:
        """Compose a title from grounded fact slots.

        Selects the applicable subset of ``TITLE_FRAME_TEMPLATES`` — those
        whose placeholders are ALL backed by a non-empty fact slot — then
        shuffles that subset deterministically per-project+series (reusing
        ``_project_seed``, keyed by ``seed_suffix`` so YouTube and TikTok
        composition don't collide on the same frame order) and picks by
        ``short_index``. There is no decorator/cosmetic-suffix layer: the
        candidate set is exactly the applicable frames, so a batch asking
        for more distinct titles than there are applicable frames gets a
        loud :class:`TitleCapacityExceeded` instead of a near-duplicate
        (same frame, different trailing emoji) — that shared-prefix shape
        is the templated-spam signal the anti-spam gate refuses at enqueue.

        Before returning, checks the candidate against the channel's
        published-titles ledger (:func:`longform_metadata.load_published_titles`).
        A collision walks FORWARD through this same deterministic shuffled
        frame list — never a random re-pick — until an unpublished title is
        found, so the same inputs always resolve to the same alternative.

        With ``facts`` (the drive path) the title is the measured place --
        :meth:`_compose_stamped_title` -- or its stamp, and never raises
        :class:`TitleCapacityExceeded`.

        Raises:
            ValueError: fewer than 2 distinct fact slots are available (a
                regression guard — ``location`` and ``terrain`` are always
                populated in normal operation, so this never fires today).
            TitleCapacityExceeded: ``short_index`` exceeds the number of
                distinct applicable frames. Fix: add a ``titles.yaml`` hand
                title for that index, or lower ``--shorts-target``.
        """
        if facts is not None:
            return self._compose_stamped_title(
                facts, seed_suffix, max_len, place, title_body_arm, candidates
            )
        slots = self._title_slots(short_index)
        available = {name for name, value in slots.items() if value}
        applicable = [t for t in TITLE_FRAME_TEMPLATES if _frame_fields(t) <= available]
        if not applicable:
            raise ValueError(
                f"short_index {short_index}: cannot compose a title — fewer than "
                f"2 distinct grounded fact slots available (have: {sorted(available)})"
            )
        if short_index > len(applicable):
            raise TitleCapacityExceeded(
                f"short_index {short_index}: only {len(applicable)} distinct composed "
                f"title(s) available from the grounded facts on hand — cannot compose "
                f"a {short_index}th title without repeating a frame (which would "
                f"produce a near-duplicate). Add a titles.yaml entry for short "
                f"{short_index}, or lower --shorts-target."
            )

        def _render(frame: str) -> str:
            text = frame[0].upper() + frame[1:] if frame else frame
            text = text.format_map(slots)
            return text[:max_len]

        project_rng = random.Random(self._project_seed(seed_suffix))
        project_rng.shuffle(applicable)
        idx = (short_index - 1) % len(applicable)

        from .longform_metadata import load_published_titles

        published = {t.strip().lower() for t in load_published_titles()}
        for offset in range(len(applicable)):
            frame = applicable[(idx + offset) % len(applicable)]
            candidate = _render(frame)
            if candidate.strip().lower() not in published:
                return candidate
        # Every applicable frame already collides with a published title
        # (exhausted, unlikely) — fall back to the primary deterministic pick
        # rather than raising; the channel-level dedup gate (title_builder's
        # long-form path) is the hard backstop for genuine duplicates.
        return _render(applicable[idx])

    def _pick_title(
        self,
        rng: random.Random,
        weather: Optional[WeatherResult],
        short_index: int,
        facts: Optional[TitleFacts] = None,
        place: Optional[str] = None,
        title_body_arm: Optional[str] = None,
        candidates: "Sequence[str]" = (),
    ) -> str:
        """Pick a unique composed title for this short.

        Weather is NOT interpolated into titles:
        brightness-derived weather can't verify rain/sun as fact. `weather`
        is retained in the signature for caller stability; it no longer
        selects a title.
        """
        return self._compose_title(
            short_index,
            "titles",
            max_len=100,
            facts=facts,
            place=place,
            title_body_arm=title_body_arm,
            candidates=candidates,
        )

    def _spooled_opener_keys(self) -> "dict[str, list[tuple[str, ...]]]":
        """Platform -> 6-word opener keys of text other projects spooled."""
        if self._spooled_keys is None:
            self._spooled_keys = _opener_keys(_read_spooled_texts(self.project_name))
        return self._spooled_keys

    def _fresh_rotating_pick(
        self, pool: "list[str]", platform: str, part: str, short_index: int, render
    ) -> str:
        """:meth:`_rotating_pick` over the pool's entries whose ``render``-ed
        text cannot open like spooled text, then the rest.

        A Meta opener pool is first re-ranked from Meta evidence
        (:meth:`_meta_ranked`), so the freshness split below still keeps a
        proven opener out of the anti-spam gate's window.

        The fresh entries keep the project's shuffled order, so a batch still
        never repeats an entry until they run out; past that the stale ones
        follow (and the gate refuses them) rather than raising here.
        """
        order = list(pool)
        random.Random(self._project_seed(f"captions:{platform}:{part}")).shuffle(order)
        if part == "opener":
            order = self._meta_ranked(platform, order)
        taken = self._spooled_opener_keys().get(platform, [])
        fresh = [e for e in order if not _opens_like(render(e), taken)]
        if len(fresh) < len(order):
            order = fresh + [e for e in order if e not in fresh]
        return order[(short_index - 1) % len(order)]

    def _meta_ranked(self, platform: str, order: "list[str]") -> "list[str]":
        """``order`` re-ranked by Waypost's ``rank_captions`` from the nightly
        ``caption_evidence.json`` (``tracking/caption_evidence.py``) for
        Facebook and Instagram; unchanged for other platforms, with no evidence
        file, or when Waypost refuses the evidence. The record is kept per
        platform for :attr:`SocialPost.caption_rankings`."""
        from seo_manager.analytics.caption_ranking import (
            DEFAULT_POLICY,
            PLATFORMS,
            CaptionRankingRefused,
            rank_captions,
            record_to_dict,
        )

        from ..tracking.caption_evidence import load_caption_evidence
        from . import longform_metadata

        if platform not in PLATFORMS:
            return order
        if self._caption_evidence is None:
            self._caption_evidence = (
                load_caption_evidence(Path(longform_metadata.DEFAULT_SEO_STORAGE)) or False
            )
        if self._caption_evidence is False:
            return order
        by_id = {caption_template_id(t): t for t in order}
        try:
            ranking = rank_captions(platform, list(by_id), self._caption_evidence, DEFAULT_POLICY)
        except CaptionRankingRefused as exc:
            logger.warning("caption ranking refused for %s: %s", platform, exc)
            return order
        self._caption_rankings[platform] = json.loads(json.dumps(record_to_dict(ranking)))
        return [by_id[t] for t in ranking.order]

    def _rotating_pick(self, pool: "list[str]", platform: str, part: str, short_index: int) -> str:
        """The ``short_index``-th entry of ``pool`` in this project's shuffled order.

        One shuffle per (project, platform, part): each part of a caption walks
        its own pool, so openers, details and prompts never rotate in lockstep,
        and no two shorts of a batch share an entry until the pool wraps.
        """
        order = list(pool)
        random.Random(self._project_seed(f"captions:{platform}:{part}")).shuffle(order)
        return order[(short_index - 1) % len(order)]

    def _compose_caption(
        self,
        platform: str,
        short_index: int,
        fmap: "dict[str, str]",
        interactive: bool,
        picked: Optional["dict[str, str]"] = None,
    ) -> str:
        """One platform caption: the place, plus a terrain clause only for classified
        terrain; an interactive short is its guess prompt alone.

        ``picked``, when given, receives ``{platform: caption_template_id}`` of the
        opener this caption opens with."""

        def render(entry: str) -> str:
            text = entry.format_map(fmap).strip()
            return text[:1].upper() + text[1:]

        def fill(pool: "list[str]", part: str) -> str:
            return render(self._rotating_pick(pool, platform, part, short_index))

        if interactive:
            prompts = list(INTERACTIVE_PROMPT_TEMPLATES)
            random.Random(self._project_seed("captions:interactive:prompt")).shuffle(prompts)
            slot = list(_PLATFORM_CAPTION_TEMPLATES).index(platform)
            prompt = prompts[(short_index - 1 + slot) % len(prompts)]
            if picked is not None:
                picked[platform] = caption_template_id(prompt)
            return prompt
        template = self._fresh_rotating_pick(
            _PLATFORM_CAPTION_TEMPLATES[platform], platform, "opener", short_index, render
        )
        if picked is not None:
            picked[platform] = caption_template_id(template)
        opener = render(template)
        if opener[-1] not in ".?!":
            opener += "."
        if self.terrain_type not in _CAPTION_TERRAINS:
            return opener
        return f"{opener} {fill(CAPTION_DETAIL_TEMPLATES, 'tail')}"

    def _platform_hashtags(
        self,
        platform: str,
        short_index: int,
        interactive: bool,
        limit: int,
        hashtag_mode: Optional[str] = None,
    ) -> "list[str]":
        """This platform's hashtags for ``short_index``, capped to ``limit``.

        Location hashtags are per-component (e.g. #DinglePeninsula
        #CountyKerry #Ireland). Withheld entirely when ``interactive`` — a
        location hashtag would leak the place a withheld caption just hid.

        The generic tags are ``GENERIC_HASHTAGS_PER_POST`` drawn from
        ``GENERIC_HASHTAGS`` with a seed keyed on ``short_index``, so no two
        posts need carry the same fixed pair; order is shuffled by the same
        per-short seed. The place and region tags are never part of that draw.

        ``hashtag_mode`` (a caption-experiment arm, ``publishing/experiments.py``)
        replaces all of that: ``none`` is no hashtags, ``location`` the first two
        location tags (none when ``interactive`` withholds the place), ``june``
        ``JUNE_STYLE_HASHTAGS`` then those location tags. ``limit`` does not
        apply to an arm.
        """
        if hashtag_mode == "none":
            return []
        if hashtag_mode in ("location", "june"):
            generics = list(JUNE_STYLE_HASHTAGS) if hashtag_mode == "june" else []
            if interactive:
                return generics
            ordered = self._mandatory_location_tags() + self.location_hashtags
            return generics + list(dict.fromkeys(ordered))[:2]
        rng = random.Random(self._project_seed(f"captions:{platform}:tags:{short_index}"))
        tags = rng.sample(GENERIC_HASHTAGS, GENERIC_HASHTAGS_PER_POST)
        terrain_ht = TERRAIN_HASHTAGS.get(self.terrain_type, [])
        if terrain_ht:
            tags.append(terrain_ht[0])
        mandatory = [] if interactive else self._mandatory_location_tags()
        if not interactive:
            seen_lower = {t.lower() for t in tags} | {t.lower() for t in mandatory}
            for loc_ht in self.location_hashtags:
                if loc_ht.lower() not in seen_lower:
                    tags.append(loc_ht)
                    seen_lower.add(loc_ht.lower())
        rng.shuffle(tags)
        if not mandatory:
            return tags[:limit]
        # The place and region tags are never cut by the platform limit: they
        # take their slots first, generic tags fill the rest, and the final
        # order is shuffled again so captions still vary between shorts.
        chosen = mandatory + tags[: max(limit - len(mandatory), 0)]
        rng.shuffle(chosen)
        return chosen

    def _mandatory_location_tags(self) -> "list[str]":
        """The place tag and the region (state/county) tag, in that order.

        Empty unless the location has a region component (2+ distinct
        components); a single-component location has no region to name.
        """
        tags = self.location_hashtags
        if len(tags) < 2 or tags[0].lower() == tags[-1].lower():
            return []
        return [tags[0], tags[-1]]

    def _assert_region_present(self, short_index: int, label: str, text: str) -> None:
        """Upstream gate: a region-bearing short's surface must name its region."""
        mandatory = self._mandatory_location_tags()
        if not mandatory:
            return
        region = mandatory[-1]
        if region.lower() not in text.lower():
            raise ValueError(
                f"short_index {short_index}: {label} missing the region tag {region!r}: {text!r}"
            )

    def _fit_hashtag_caption(
        self, base: str, hashtags: "list[str]", mandatory: "list[str]", limit: int
    ) -> str:
        """Compose ``base`` + hashtags within ``limit`` chars.

        Over the limit: drop generic (non-mandatory) tags from the end first,
        then shorten ``base`` on a word boundary (never inside the place).
        Mandatory tags are never cut.
        """
        tags = list(hashtags)
        caption = self._append_hashtags(base, tags)
        mand = {m.lower() for m in mandatory}
        while len(caption) > limit:
            generic = [t for t in tags if t.lower() not in mand]
            if not generic:
                break
            tags.remove(generic[-1])
            caption = self._append_hashtags(base, tags)
        if len(caption) <= limit:
            return caption
        block = " ".join(tags)
        budget = limit - len(block) - 2
        floor = 0
        place = self.social_location
        if place and place in base:
            floor = base.index(place) + len(place)
        if budget - 1 > floor:
            head = base[: budget - 1]
            cut = head.rsplit(" ", 1)[0] if " " in head else head
            if len(cut) >= floor:
                base = cut.rstrip(" ,;:-\u2014") + "\u2026"
        return self._append_hashtags(base, tags)

    @staticmethod
    def _append_hashtags(base: str, hashtags: "list[str]") -> str:
        if not hashtags:
            return base
        return f"{base}\n\n{' '.join(hashtags)}"

    def _build_platform_captions(
        self,
        weather: Optional[WeatherResult],
        short_index: int,
        interactive: bool = False,
        hand_caption: Optional[str] = None,
        experiment_arms: Optional["dict[str, str]"] = None,
        picked: Optional["dict[str, str]"] = None,
    ) -> "dict[str, str]":
        """Compose one distinct caption per platform.

        ``picked``, when given, receives each composed caption's opener id
        (:func:`caption_template_id`); a hand caption composes none.

        ``experiment_arms`` (``{experiment: arm}``) swaps the Facebook hashtag
        treatment for the arm's ``hashtag_mode``; the ``none`` arm drops the
        location tags, so it is exempt from the Facebook region-tag check
        below (the caption text still names the place).

        Each composed (non-hand) caption carries the mandatory place fact
        slot (``self.social_location``) — except when ``interactive`` is
        True, where ``LOCATION_WITHHELD`` fills {location} on every platform
        instead, so the place can never leak through a caption.

        A hand caption (``HandTitle.caption``) becomes the YouTube caption
        verbatim and seeds the other three platforms as their opening
        sentence — each still made distinct from the others by that
        platform's own hashtag count/length shape.

        Raises:
            ValueError: two platforms resolved to an identical caption, or
                (non-hand, non-interactive) a composed caption is missing
                the place, or (interactive) ``hand_caption`` leaks the place
                (an interactive hand caption must be written place-free,
                same as the hand title).
        """
        if interactive and hand_caption:
            leak = self._leaks_place(hand_caption)
            if leak:
                raise ValueError(
                    f"short_index {short_index}: interactive short's hand "
                    f"caption leaks the place {leak!r}: {hand_caption!r}"
                )
        weather_emoji = weather.emoji if weather else "🚗"
        location_text = LOCATION_WITHHELD if interactive else self.social_location
        fmap = defaultdict(
            str,
            location=location_text,
            terrain_desc=self.terrain_desc,
            weather_emoji=weather_emoji,
        )
        first_sentence = f"{hand_caption.strip().rstrip('.')}." if hand_caption else None

        captions: "dict[str, str]" = {}

        # YouTube — description-style, no hashtag wall (0 hashtags).
        yt = (
            hand_caption.strip()
            if hand_caption
            else self._compose_caption("youtube", short_index, fmap, interactive, picked)
        )
        if self.recording_date:
            yt += f"\n\nFilmed {_full_month_year(self.recording_date)}."
        captions["youtube"] = yt

        # Facebook — one line, <=2 hashtags.
        fb_base = first_sentence or self._compose_caption(
            "facebook", short_index, fmap, interactive, picked
        )
        mandatory = [] if interactive else self._mandatory_location_tags()
        fb_mode = hashtag_mode_for(self.experiments, experiment_arms or {}, "facebook")
        captions["facebook"] = self._append_hashtags(
            fb_base,
            self._platform_hashtags(
                "facebook", short_index, interactive, limit=2, hashtag_mode=fb_mode
            ),
        )

        # Instagram — short line + recording date + up to 8 tags. This is
        # the legacy shared-column shape (``post.caption`` below).
        # No weather or vehicle emoji: the weather is a brightness guess, not a fact.
        ig_base = first_sentence or self._compose_caption(
            "instagram", short_index, fmap, interactive, picked
        )
        if self.recording_date:
            ig_base += f"\n\n📹 {self.recording_date}"
        # Enforce Twitter's 280 char limit for Metricool compatibility: drop
        # generic tags, then shorten the sentence on a word boundary -- the
        # place and region tags are never cut.
        ig_caption = self._fit_hashtag_caption(
            ig_base,
            self._platform_hashtags("instagram", short_index, interactive, limit=8),
            mandatory,
            limit=280,
        )
        captions["instagram"] = ig_caption

        # TikTok — <=150 chars, <=4 tags.
        tt_base = first_sentence or self._compose_caption(
            "tiktok", short_index, fmap, interactive, picked
        )
        tt_caption = self._fit_hashtag_caption(
            tt_base,
            self._platform_hashtags("tiktok", short_index, interactive, limit=4),
            mandatory,
            limit=150,
        )
        captions["tiktok"] = tt_caption

        if not interactive and not hand_caption and self.social_location:
            for platform, text in captions.items():
                if self.social_location not in text:
                    raise ValueError(
                        f"short_index {short_index}: {platform} caption missing "
                        f"the place {self.social_location!r}: {text!r}"
                    )

        if not interactive:
            for platform in ("facebook", "instagram", "tiktok"):
                if platform == "facebook" and fb_mode == "none":
                    continue
                self._assert_region_present(short_index, f"{platform} caption", captions[platform])

        seen: "dict[str, str]" = {}
        for platform, text in captions.items():
            for other_platform, other_text in seen.items():
                if text == other_text:
                    raise ValueError(
                        f"short_index {short_index}: {platform} and "
                        f"{other_platform} captions are identical: {text!r}"
                    )
            seen[platform] = text

        return captions

    def _build_youtube_description(
        self,
        short_index: int = 1,
        interactive: bool = False,
        place: "Optional[PlaceChoice]" = None,
        place_kind: str = "",
        hashtag_arm: Optional[str] = None,
    ) -> str:
        """Build a short YouTube description for Shorts (max 200 chars).

        ``hashtag_arm`` (a ``yt_hashtag_set`` arm id) swaps the optional hashtags
        of a non-interactive description; ``None`` or an unknown id keeps today's.

        Per-short differentiation uses most specific data available (priority):
          1. POI name  — "POV driving near {poi}, {location}"
          2. Road name — "POV drive along {road}, {location}"
          3. Part num  — "POV driving through {location} Pt.{short_index}"

        #Shorts is fixed; remaining hashtags are droppable. No "4K" tag: shorts
        are hard-gated to 1080x1920 (encoding_constants SHORTS_WIDTH/HEIGHT),
        never 4K, so a 4K claim on any shorts surface would be false.

        When ``interactive``, the base text and hashtags withhold the
        place (no POI/road/location-tag branch) and the LAST line becomes
        ``"📍 {social_location_full}"`` instead — so search can still index
        the place while every other surface stays place-free.
        """

        def lead_for(phrase: str) -> str:
            return self._fresh_rotating_pick(
                YOUTUBE_DESCRIPTION_LEADS,
                "youtube",
                "description-lead",
                short_index,
                lambda lead: f"{lead} {phrase}",
            )

        if interactive:
            phrase = f"driving through {LOCATION_WITHHELD} Pt.{short_index}"
            base = f"{lead_for(phrase)} {phrase}"
            fixed_hashtags = ["#Shorts"]
            optional_hashtags = ["#POVDriving"]
            terrain_ht = TERRAIN_HASHTAGS.get(self.terrain_type, [])
            if terrain_ht:
                optional_hashtags.append(terrain_ht[0])
            all_hashtags = fixed_hashtags + optional_hashtags
            place_line = f"📍 {self.social_location_full}"
            description = f"{base} {' '.join(all_hashtags)}\n\n{place_line}"
            while optional_hashtags and len(description) > 200:
                optional_hashtags.pop()
                all_hashtags = fixed_hashtags + optional_hashtags
                description = f"{base} {' '.join(all_hashtags)}\n\n{place_line}"
            return description

        # ``social_location_full`` prefers the destination POI ("Donnelley
        # Wildlife Management Area, Beaufort County") when the resolved model
        # named one, else the whole route text (``location_short_full`` — the
        # long-form description names the whole origin→dest drive, NOT the
        # arc-free burned-caption anchor). The explicit ``nearest_poi`` still
        # wins its own branch (it's the most-specific GPS-verified anchor).
        if self.nearest_poi:
            self._used_pois[self.nearest_poi] = self._used_pois.get(self.nearest_poi, 0) + 1
            phrase = f"driving near {self.nearest_poi}, {self.social_location_full}"
            if self._used_pois[self.nearest_poi] > 1:
                phrase += f" Pt.{short_index}"
        elif place is not None and place.reason == REASON_VARIED:
            # A repeated place body gave way to another the clip measured.
            if place_kind == KIND_ROAD:
                phrase = f"driving along {place.body}, {self.social_location_full}"
            else:
                phrase = f"driving through {place.body}"
            phrase += f" Pt.{short_index}"
        elif self.road_name:
            phrase = f"driving along {self.road_name}, {self.social_location_full} Pt.{short_index}"
        else:
            phrase = f"driving through {self.social_location_full} Pt.{short_index}"
        base = f"{lead_for(phrase)} {phrase}"

        # #Shorts is fixed; remaining hashtags are droppable. #4K removed:
        # shorts are 1080x1920 (never 4K), so the claim would be false.
        # #ScenicDrive removed: unverifiable genre claim (scenic-purge).
        fixed_hashtags = ["#Shorts"]
        arm_tags = list(SHORTS_HASHTAG_SET_ARMS.get(hashtag_arm or "", ()))
        optional_hashtags = arm_tags or ["#POVDriving"]
        terrain_ht = TERRAIN_HASHTAGS.get(self.terrain_type, [])
        if terrain_ht:
            optional_hashtags.append(terrain_ht[0])
        # Place + region are pinned (never dropped), generic-to-specific so the
        # plan builder's title hashtags (taken from the end) lead with the
        # place, then the region.
        mandatory = self._mandatory_location_tags()
        if mandatory:
            pinned = [mandatory[1], mandatory[0]]
            for mid in self.location_hashtags[1:-1]:
                if mid.lower() not in {t.lower() for t in pinned + optional_hashtags}:
                    optional_hashtags.append(mid)
        else:
            pinned = []
            if self.location_tag:
                optional_hashtags.append(f"#{self.location_tag}")

        all_hashtags = fixed_hashtags + optional_hashtags + pinned
        description = f"{base} {' '.join(all_hashtags)}"

        if len(description) <= 200:
            self._assert_region_present(short_index, "youtube description", description)
            return description

        # Drop optional hashtags from the end until it fits; an assigned arm's
        # tags are never dropped (the stored arm must match the rendered text).
        protected = {t.lower() for t in arm_tags}
        while len(description) > 200:
            droppable = [i for i, t in enumerate(optional_hashtags) if t.lower() not in protected]
            if not droppable:
                break
            optional_hashtags.pop(droppable[-1])
            all_hashtags = fixed_hashtags + optional_hashtags + pinned
            description = f"{base} {' '.join(all_hashtags)}"

        # Final safety: the anchor alone still overruns 200 chars with every
        # optional hashtag dropped. Cut on a word boundary via the shared
        # truncator so a placename is never sliced mid-word (e.g. "…, Beaufort
        # Cou"); the ellipsis signals the cut. Falls back to a hard slice only
        # when the first token itself overruns (no whitespace to break on).
        from gomoveshift.pipeline.text_format import truncate_for_platforms

        if (pinned or arm_tags) and len(description) > 200:
            tag_block = " ".join(all_hashtags)
            base = truncate_for_platforms(
                base, twitter_enabled=True, limit=max(200 - len(tag_block) - 1, 1)
            )
            description = f"{base} {tag_block}"
        else:
            description = truncate_for_platforms(description, twitter_enabled=True, limit=200)
        self._assert_region_present(short_index, "youtube description", description)
        return description

    @staticmethod
    def _dedup_add(tags: list[str], candidates: list[str]) -> None:
        """Add candidates to tags with case-insensitive deduplication."""
        lower_set = {t.lower() for t in tags}
        for c in candidates:
            if c and c.lower() not in lower_set:
                tags.append(c)
                lower_set.add(c.lower())

    def _build_tags(
        self,
        weather: Optional[WeatherResult],
        short_index: int = 1,
        interactive: bool = False,
    ) -> str:
        """Build YouTube tags string (comma-separated for Metricool).

        Rotates supplementary tags per short so each has a slightly
        different tag set while keeping core tags consistent.

        When ``interactive``, the place (``location_short``,
        ``location_short_full``, ``social_location``, ``county``) is withheld
        by construction — no location-phrase tags, no county tag, no
        road+city compound tag — so "where is this?" can't be answered from
        the tag list. Road-only and POI-only tags are still added: they are
        not "the place" (a road/POI alone rarely identifies the destination),
        and a construction check below guarantees no place-string ends up in
        the result regardless.
        """
        tags = BASE_TAGS.copy()

        # Strip any trailing " — Part N" series suffix before deriving
        # tags from the location. Ensures fragments like "Ireland — Part 1"
        # never end up in the tag list.
        location_for_tags = _strip_series_suffix(self.location)

        # Strip a leading operator title word ("To"/"Through"/...) from EACH
        # comma component (``parse_lf_label``, as ``_arc_safe_anchor`` does), so
        # "To Goleen, County Cork" cannot leak "To Goleen" into the phrase tags
        # or into ``_region_tags``' string-parse fallback. No-op without one.
        location_stripped = ", ".join(
            place
            for place in (parse_lf_label(p).place for p in location_for_tags.split(","))
            if place.strip()
        )

        if not interactive:
            # Add location parts as phrase tags (split by comma, not whitespace)
            self._dedup_add(
                tags,
                [p.strip() for p in location_stripped.split(",") if len(p.strip()) > 2],
            )

        # Region-derived tags ("Ireland road trip", etc.) replace the
        # hardcoded "road trip USA" that used to appear on every video.
        self._dedup_add(
            tags, _region_tags(location_stripped, self.resolved_location, self._own_regions)
        )

        # Add road names from roads list
        if self.roads:
            self._dedup_add(tags, [r.strip() for r in self.roads if r.strip()])

        # Add nearest POI as a tag
        if self.nearest_poi:
            self._dedup_add(tags, [self.nearest_poi])

        # Add county as a tag
        if self.county and not interactive:
            self._dedup_add(tags, [self.county])

        # Add state abbreviation from the location's STATE SLOT only — the
        # trailing comma component — longest name first, so "West Virginia" is
        # tested before "Virginia". A state name inside the county ("Washington
        # County, Virginia") is not a state claim; tagging it WA was a false
        # location claim No comma → no state slot → no tag.
        _head, comma, state_slot = (self.location or "").rpartition(",")
        for state_name, abbrev in sorted(
            US_STATE_ABBREVS.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            if comma and re.match(rf"\s*{re.escape(state_name)}\b", state_slot):
                self._dedup_add(tags, [abbrev])
                break

        # Compound road+city tags: "{road} {city}" for each road. Uses the FULL
        # route text (not the arc-free caption anchor) — tags name the whole
        # origin→dest drive, and the density budget counts on it. Withheld for
        # an interactive short — the compound tag names the destination.
        city = self.location_short_full
        if self.roads and not interactive:
            self._dedup_add(tags, [f"{r.strip()} {city}" for r in self.roads])

        # Add terrain-specific tags early so they aren't bumped by filler
        self._dedup_add(tags, TERRAIN_TAGS.get(self.terrain_type, [])[:2])

        # Rotating supplementary tags — each short picks a different subset.
        # Inserted BEFORE the static road_context fillers so the per-short
        # variance (relied on by test_tags_vary_across_shorts) survives the
        # final 20-tag cap. Without this, region tags + base tags + location
        # tags fully consume the budget and every short gets identical tags.
        supplementary = [
            "nature",
            "explore",
            "dashcam",
            "road trip vibes",
            "travel vlog",
            "driving",
            "asmr driving",
            "virtual travel",
            "dashboard camera",
            "first person",
            "car ride",
            "no talking",
            "ambient",
            "calming drive",
            "slow travel",
            "road therapy",
            "window view",
            "drive life",
            "pov car",
            "open road",
            "weekend drive",
            "long drive",
            "relaxing drive",
            "on the road",
            "driving route",
            "road trip life",
        ]
        # Pick 2-3 supplementary tags based on short_index
        start = ((short_index - 1) * 2) % len(supplementary)
        extras = supplementary[start : start + 3]
        self._dedup_add(tags, extras)

        # Static road-context filler tags — appended AFTER the rotating
        # supplementary set so the cap-at-20 doesn't squeeze out per-short
        # variance. Hardcoded "road trip USA" was removed; region
        # phrasing is contributed by _region_tags().
        road_context_tags = [
            "dashcam footage",
            "highway",
            "road trip",
            "pov road trip",
            "driving tour",
            "road adventure",
            "highway drive",
            "travel by car",
            "drive footage",
        ]
        self._dedup_add(tags, road_context_tags)

        # Weather-specific tags
        if weather:
            weather_tags = [f"{weather.adjective} drive"]
            if weather.condition == WeatherCondition.RAINY:
                weather_tags.extend(["rain driving", "rainy day"])
            elif weather.condition == WeatherCondition.SUNNY:
                # "beautiful day" dropped: "beautiful"
                # is a subjective claim on a ContentValidator-unseen tag surface.
                weather_tags.extend(["sunny drive"])
            self._dedup_add(tags, weather_tags)

        # A Short (own clip-GPS regions) keeps only tags Waypost can review:
        # its tag templates, the clip's county names and the clip's own road names.
        if self._own_regions:
            tags = reviewable_short_tags(tags, self._own_regions, self._own_roads)

        # Case-insensitive dedup (final pass to catch duplicates from extend)
        seen_lower: set[str] = set()
        deduped: list[str] = []
        for tag in tags:
            if tag.lower() not in seen_lower:
                seen_lower.add(tag.lower())
                deduped.append(tag)
        deduped = deduped[:20]

        # Enforce YouTube's 500-char tag total. YouTube (and Metricool's import
        # validator) count a tag CONTAINING A SPACE with surrounding quotes
        # (+2), so the raw ",".join length UNDER-counts and the cap silently
        # never fired — multi-word-heavy tag sets imported over-limit. Count the
        # quoted width and keep a safety margin below 500 for Metricool's own
        # slightly-higher count. Drop lowest-priority tags until under.
        def _yt_tag_len(ts: list[str]) -> int:
            return sum(len(t) + (2 if " " in t else 0) for t in ts) + max(0, len(ts) - 1)

        while deduped and _yt_tag_len(deduped) > 480:
            deduped.pop()
        result = ",".join(deduped)

        if interactive:
            leak = self._leaks_place(result)
            if leak:
                raise ValueError(
                    f"short_index {short_index}: interactive short's youtube_tags "
                    f"leak the place {leak!r}: {result!r}"
                )

        return result

    def _build_first_comment(
        self,
        weather: Optional[WeatherResult],
        short_index: int = 1,
        interactive: bool = False,
    ) -> str:
        """Build Instagram first comment with extra hashtags.

        Rotates hashtag pools per short so each post has a different
        set while keeping core hashtags (#POVDriving, location) constant.

        When ``interactive``, the pinned comment is where the place
        finally gets named — its FIRST line becomes
        ``"📍 {social_location_full}"``, then the existing hashtags follow
        unchanged.
        """
        # Core hashtags — always included.
        # Per-component location hashtags (e.g. #DinglePeninsula
        # #CountyKerry #Ireland) replace the legacy single smashed
        # hashtag (#DinglePeninsulaCountyKerryIrelandPart1).
        core = ["#POVDriving", *self.location_hashtags]

        # Rotating pools — each short picks from a different starting point
        # (terrain-specific hashtags removed; added dynamically from TERRAIN_HASHTAGS)
        pool = [
            "#RoadTrip",
            "#DashCam",
            "#Travel",
            "#Driving",
            "#POV",
            "#DriveWithMe",
            "#Explore",
            "#Nature",
            "#RoadTripVibes",
            "#TravelGram",
            "#DashcamFootage",
            "#WanderLust",
            "#TravelMore",
            "#OnTheRoad",
            "#OpenRoad",
            "#DrivingPOV",
            "#RoadTripLife",
            "#ExploreMore",
            "#DiscoverPlaces",
            "#AdventureAwaits",
            "#SlowTravel",
            "#DrivingASMR",
            "#RoadTherapy",
            "#POVCAR",
            "#DriveLife",
            "#WindowView",
            "#WeekendDrive",
            "#NoTalking",
            "#RelaxingDrive",
            "#VirtualTravel",
            "#LongDrive",
            "#CarRide",
        ]

        # Rotate the pool per short so each gets different hashtags
        offset = ((short_index - 1) * 3) % len(pool)
        rotated = pool[offset:] + pool[:offset]
        extras = rotated[:11]  # 11 from pool + 2 core + 2 terrain = 15

        hashtags = core + extras
        # Add terrain-specific hashtags
        hashtags.extend(TERRAIN_HASHTAGS.get(self.terrain_type, [])[:2])

        if weather:
            if weather.condition == WeatherCondition.RAINY:
                hashtags.extend(["#RainyDay", "#RainDriving"])
            elif weather.condition == WeatherCondition.SUNNY:
                # "#PerfectWeather" dropped: subjective
                # weather claim on a ContentValidator-unseen hashtag surface.
                hashtags.extend(["#SunnyDay"])

        hashtag_line = " ".join(hashtags[:15])  # Instagram best practice: 15 max
        if interactive:
            return f"📍 {self.social_location_full}\n\n{hashtag_line}"
        return hashtag_line

    def _pick_tiktok_title(
        self,
        rng: random.Random,
        short_index: int,
        facts: Optional[TitleFacts] = None,
        place: Optional[str] = None,
        candidates: "Sequence[str]" = (),
    ) -> str:
        """Pick a TikTok-native title (folded into the same composition
        as ``_pick_title`` — no separate pool — with a distinct seed suffix
        so frame order differs from the YouTube title, truncated to
        TikTok's 90-char max).
        """
        return self._compose_title(
            short_index, "tiktok", max_len=90, facts=facts, place=place, candidates=candidates
        )

    def _choose_places(self, previous: "Mapping[str, str]", route_record: Optional[dict]):
        """The place bodies this Short's title and description name, varied from the
        previous Short's when the clip measured another place (``place_variety``).

        Title candidates are the clip's towns, then its counties (the places its
        final-text review accepts), then today's label; a road is a description-only
        candidate (the review refuses one in a title). Returns ``(title_choice,
        title_bodies, description_choice, description_kind, record)`` (``title_bodies``:
        every title candidate, the chosen one first); the description choice that
        keeps today's text is returned as ``None`` so its text is byte-identical."""
        measured = measured_places(self._own_regions or [], self._own_roads, route_record)
        title = _title_place_bodies(measured, self.location_short)
        desc = [("today", self.social_location_full)] + measured
        t_choice = (
            choose_place_body(
                [b for _, b in title], [previous.get("title", "")], kinds=[k for k, _ in title]
            )
            if title
            else None
        )
        d_choice = choose_place_body(
            [b for _, b in desc], [previous.get("description", "")], kinds=[k for k, _ in desc]
        )
        record = {"description": d_choice.to_dict()}
        title_bodies: "list[str]" = []
        if t_choice is not None:
            record = {"title": t_choice.to_dict(), **record}
            title_bodies = [t_choice.body] + [b for _, b in title if b != t_choice.body]
        return (
            t_choice,
            title_bodies,
            d_choice if d_choice.reason == REASON_VARIED else None,
            d_choice.kind,
            record,
        )

    def _shorts_arm_factors(self, hand_idx: int, title_facts: Optional[TitleFacts]) -> "list[str]":
        """The Shorts factors this Short's text can carry: the hashtag set only off
        an interactive Short (whose description withholds the place). The title body
        is never carried: a title is the measured place alone, with no body variants
        (``SHORTS_TITLE_BODY_ARMS`` shapes only the stamped fallback, never assigned)."""
        hand = self._hand_titles.get(hand_idx)
        factors = []
        if not (hand.interactive if hand is not None else False):
            factors.append("yt_hashtag_set")
        return factors

    def generate_post(
        self,
        short_index: int,
        weather: Optional[WeatherResult] = None,
        global_index: Optional[int] = None,
        title_facts: Optional[TitleFacts] = None,
        previous_places: "Optional[Mapping[str, str]]" = None,
        route_record: Optional[dict] = None,
        shorts_arms: "Optional[Mapping[str, str]]" = None,
    ) -> SocialPost:
        """
        Generate a unique social post for one short.

        ``shorts_arms`` (``{factor: arm}``, ``publishing.experiments.SHORTS_ARMS``)
        renders the assigned ``yt_title_body`` / ``yt_hashtag_set`` variant and is
        stored in ``experiment_arms``. A factor the Short's text cannot carry
        (hand title, no verified place, interactive) or an unknown arm is dropped,
        so a Short with none renders exactly as before.

        Args:
            short_index: 1-based index of the short (local to this generator's
                RNG seeding — unaffected by the hand-title lane).
            weather: Optional weather result for this segment
            global_index: The ``titles.yaml`` lookup key (the same
                post-``index_offset`` index ``generate_all`` assigns to
                ``post.short_index``). Defaults to ``short_index`` when the
                caller has no offset to apply.
            title_facts: Drive path only -- composes a stamped title (see
                :class:`TitleFacts`). ``None`` keeps the legacy composition.
            previous_places: ``{"title": body, "description": body}`` -- the place
                bodies the previous Short named (``SocialPost.place_variety``).
                A stamped title or description naming the same body takes another
                place the clip measured (``place_variety``); ``None`` = no previous.
            route_record: This clip's measured route record (towns held for every
                retained second are title candidates).

        Returns:
            SocialPost with all platform-specific content

        Raises:
            ValueError: a hand title collides with the published-titles
                ledger, or an ``interactive`` hand title names the
                place it's supposed to withhold.
        """
        max_attempts = 3
        validator = ContentValidator()

        hand_idx = global_index if global_index is not None else short_index
        hand = self._hand_titles.get(hand_idx)
        if hand is not None:
            from .longform_metadata import load_published_titles

            published = {t.strip().lower() for t in load_published_titles()}
            if hand.title.strip().lower() in published:
                raise ValueError(
                    f"hand title for short_index {hand_idx} collides with a "
                    f"published title: {hand.title!r}"
                )
            self._check_hand_text_place(hand_idx, hand)
            if hand.interactive:
                leak = self._leaks_place(hand.title)
                if leak:
                    raise ValueError(
                        f"short_index {hand_idx}: interactive short's hand title "
                        f"leaks the place {leak!r}: {hand.title!r} — write it "
                        f"place-free (the place only appears in the "
                        f"description tail and first comment)"
                    )
        elif not self._pool_fallback_logged:
            logger.info("pool titles: %s absent/incomplete", self.titles_file)
            self._pool_fallback_logged = True

        arms = assign_arms(self.experiments, self.project_name, hand_idx, self.experiment_today)
        eligible = self._shorts_arm_factors(hand_idx, title_facts)
        shorts_arms = {
            f: a
            for f, a in (shorts_arms or {}).items()
            if f in eligible and a in SHORTS_ARMS.get(f, ())
        }
        arms.update(shorts_arms)
        title_place, title_bodies, desc_place, desc_kind, variety = None, [], None, "", {}
        if hand is None and title_facts is not None and title_facts.place_ok:
            title_place, title_bodies, desc_place, desc_kind, variety = self._choose_places(
                previous_places or {}, route_record
            )
        for attempt in range(max_attempts):
            # Vary the seed on retries so we pick different hooks/titles
            rng = self._get_rng(short_index + attempt * 1000)

            interactive = hand.interactive if hand is not None else False
            if hand is not None:
                # Hand title/caption bypass the template pool entirely.
                title = hand.title
            else:
                title = self._pick_title(
                    rng,
                    weather,
                    short_index,
                    title_facts,
                    place=title_place.body if title_place else None,
                    title_body_arm=shorts_arms.get("yt_title_body"),
                    candidates=title_bodies,
                )
            # One caption per platform — a hand caption becomes the
            # YouTube caption verbatim and seeds the other three.
            caption_templates: "dict[str, str]" = {}
            captions = self._build_platform_captions(
                weather,
                short_index,
                interactive=interactive,
                hand_caption=(hand.caption if hand is not None else None),
                experiment_arms=arms,
                picked=caption_templates,
            )
            tags = self._build_tags(weather, short_index, interactive=interactive)
            first_comment = self._build_first_comment(weather, short_index, interactive=interactive)

            # A hand title bypasses composition entirely, TikTok included
            # — composing anyway could raise TitleCapacityExceeded for an
            # otherwise fully-covered index.
            tiktok_title = (
                hand.title[:90]
                if hand is not None
                else self._pick_tiktok_title(
                    rng,
                    short_index,
                    title_facts,
                    place=title_place.body if title_place else None,
                    candidates=title_bodies,
                )
            )
            pois_before = dict(self._used_pois)
            youtube_description = self._build_youtube_description(
                short_index,
                interactive=interactive,
                place=desc_place,
                place_kind=desc_kind,
                hashtag_arm=shorts_arms.get("yt_hashtag_set"),
            )
            hashtag_arm = shorts_arms.get("yt_hashtag_set")
            if hashtag_arm:
                pois_arm = self._used_pois
                self._used_pois = dict(pois_before)
                legacy = self._build_youtube_description(
                    short_index, interactive=interactive, place=desc_place, place_kind=desc_kind
                )

                def _body(text: str) -> str:
                    return text.split(" #Shorts", 1)[0]

                carried = set(SHORTS_HASHTAG_SET_ARMS[hashtag_arm]) <= set(
                    re.findall(r"#\w+", youtube_description)
                )
                if carried and _body(youtube_description) == _body(legacy):
                    self._used_pois = pois_arm
                else:
                    # The limit cut an arm tag, or kept the tags only by cutting more of
                    # the body than the legacy render: never store an arm whose text is
                    # not whole. Render the legacy set (its POI state is already set).
                    del shorts_arms["yt_hashtag_set"], arms["yt_hashtag_set"]
                    youtube_description = legacy

            baseline = title[:100]  # YT max 100
            place_facts = None
            if hand is None and title_facts is not None and title_facts.place_ok:
                place_facts = short_place_facts(
                    baseline,
                    title_place.body if title_place else self.location_short,
                    self._own_regions,
                    segment_id=f"lf{self.series_part or 0:02d}-short{short_index:03d}",
                    towns=town_evidence(route_record),
                )
            if self._title_snapshot is None:
                self._title_snapshot = shipped_title_snapshot() or False
            decision = decide_title(
                "short",
                baseline,
                None,
                suffix="",
                snapshot=self._title_snapshot or None,
                facts=place_facts,
            )
            post = SocialPost(
                experiment_arms=arms,
                caption_templates=caption_templates,
                caption_rankings={
                    p: self._caption_rankings[p]
                    for p in caption_templates
                    if p in self._caption_rankings
                },
                short_index=short_index,
                caption=captions["instagram"],
                captions=captions,
                youtube_title=decision["chosen_title"],
                title_decision=decision,
                youtube_tags=tags,
                youtube_category=self.youtube_category,
                tiktok_title=tiktok_title,
                pinterest_board=self.pinterest_board,
                first_comment=first_comment,
                youtube_description=youtube_description,
                # This generator is anchored on the short's OWN resolved region
                # (the per-short OST label, via ``_per_short_variant``), so its
                # display label is the region reference the CSV gate checks the
                # shipped surfaces against. Series suffix stripped so it reads as
                # a bare "County, State".
                sampled_regions=([_strip_series_suffix(self.location)] if self.location else []),
                interactive=interactive,
                place_variety=variety,
            )

            try:
                # Upstream gate: first_comment and the
                # shorts youtube_description are shipped surfaces too — validate
                # them here (first_comment is caption-like free text; the YT
                # description is a description surface) so a blocklisted phrase
                # can't ride an unchecked field onto YouTube/Metricool.
                validator.enforce(
                    titles=[post.youtube_title, post.tiktok_title],
                    captions=[c for c in (post.caption, post.first_comment) if c],
                    descriptions=[d for d in (post.youtube_description,) if d],
                    interactive=interactive,
                )
                logger.debug(f"Generated post for short #{short_index}: title='{title[:50]}...'")
                return post
            except ContentValidationError:
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"Content validation failed for short #{short_index} "
                        f"(attempt {attempt + 1}/{max_attempts}), retrying"
                    )
                else:
                    raise

    def _per_short_variant(
        self,
        location: str,
        regions: "list[str] | None" = None,
        roads: "list[str] | None" = None,
    ) -> "SocialGenerator":
        """Clone this generator anchored on a short's OWN resolved location.

        The short's burned ``ost_text`` is already the fully-resolved surface
        label, so the clone constructs with ``resolved_location=None`` (no
        destination re-anchoring on top of it) and re-attaches
        ``resolved_location`` afterwards so region tags keep the resolved
        US-ness verdict. Seeds derive from ``project_name`` / ``series_part``
        / the local short index — all shared — so title/hook/caption rotation
        is byte-identical to the parent's; ``_used_pois`` is shared so
        description POI dedup still spans the whole LF batch.

        The LF-specific place fields — ``terrain_type``, ``road_name``,
        ``county``, ``nearest_poi`` and ``roads`` — are deliberately NOT
        inherited. This variant is built ONLY when the short's location DIFFERS
        from the parent LF (a short cut elsewhere on the drive), so every one of
        those parent values is exactly the datum least likely to hold for this
        clip: inheriting them shipped the parent's terrain (``#CoastalDrive`` on
        an inland clip), road in ``{road}`` titles / "4K POV drive along
        {road}" descriptions, POI in "POV driving near {poi}" descriptions, and
        county/road tags — while the CSV cross-region gate checks COUNTY tokens
        only, so the road/POI leaks passed unseen. Every consumer of these
        fields is ``None``-guarded (``if self.<field>`` / ``self.<field> or
        ""``), so the clone omits them and each surface falls back to the
        short's OWN resolved label — robust by construction, no leak possible.
        ``roads`` is nulled too: with a non-empty ``roads`` the constructor
        re-derives ``road_name = roads[0]`` and re-adds every road as a tag.
        The re-attached ``resolved_location`` (below) is the parent's, and it
        supplies only the drive-wide country/US-ness verdict for a variant with
        no clip-GPS ``regions`` (dry-run legacy). When ``regions`` (the short's
        OWN ``"County, State"`` GPS votes) is given, the region tags come from
        those alone — a short on a cross-border drive never inherits the
        drive's country or state.
        """
        clone = SocialGenerator(
            location=location,
            project_name=self.project_name,
            pinterest_board=self.pinterest_board,
            youtube_category=self.youtube_category,
            series_part=self.series_part,
            terrain_type=None,
            road_name=None,
            county=None,
            nearest_poi=None,
            roads=None,
            titles_file=self.titles_file,
            moment_kinds=self.moment_kinds,
            experiments=self.experiments,
            experiment_today=self.experiment_today,
            bracket_route=self._bracket_route,
        )
        clone.recording_date = self.recording_date
        clone.resolved_location = self.resolved_location
        clone._spooled_keys = self._spooled_opener_keys()
        clone._own_regions = list(regions) if regions else None
        # The clip's own GPS road names, never the parent LF's ``roads``.
        clone._own_roads = list(roads) if roads else []
        clone._used_pois = self._used_pois
        clone._used_titles = self._used_titles
        parent_places = list(self._place_strings())
        dest = getattr(self.resolved_location, "destination", None)
        if dest:
            parent_places.append(dest)
        own = location
        clone._parent_place_tokens = sorted(
            {
                tok
                for place in parent_places
                for tok in _PLACE_TOKEN_RE.findall(place)
                if tok not in _PLACE_TOKEN_STOPLIST
                and not re.search(rf"\b{re.escape(tok)}\b", own, re.I)
            }
        )
        return clone

    def generate_all(
        self,
        num_shorts: int,
        weather_results: Optional[List[WeatherResult]] = None,
        index_offset: int = 0,
        per_short_locations: Optional[List[str]] = None,
        per_short_regions: Optional[List[List[str]]] = None,
        per_short_roads: Optional[List[List[str]]] = None,
        per_short_route_records: Optional[List[Optional[dict]]] = None,
        global_indexes: Optional[Sequence[int]] = None,
        per_short_title_facts: Optional[Sequence[Optional[TitleFacts]]] = None,
        previous_places: "Optional[Mapping[str, str]]" = None,
        shorts_arms_enabled: bool = False,
        shorts_arm_history: "Optional[dict[tuple[str, str, str], list[str]]]" = None,
        used_titles: "Optional[dict[str, set[str]]]" = None,
        per_short_shorts_arms: "Optional[Sequence[Optional[Mapping[str, str]]]]" = None,
        previous_places_at: "Optional[Mapping[int, Mapping[str, str]]]" = None,
    ) -> List[SocialPost]:
        """
        Generate unique social posts for all shorts.

        Args:
            num_shorts: Number of shorts
            weather_results: Optional weather results, one per short
            index_offset: DEPRECATED — offset added to each short_index
                for global remapping in multi-LF drives, only used when
                ``global_indexes`` is not supplied. Kept working unchanged for
                existing callers; a caller with a stable, plan-time-assigned
                per-short global index (``social_generator.assign_global_short_indexes``)
                should pass ``global_indexes`` instead — it wins.
            global_indexes: Optional list parallel to the shorts — each
                short's own STABLE global index, assigned once at plan
                time and persisted so a render failure leaves a hole instead
                of shifting every later short's index. When given, it is used
                verbatim for ``SocialPost.short_index``/``generate_post``'s
                ``global_index`` in place of ``i + index_offset``. Must have
                length ``num_shorts`` when supplied.
            per_short_locations: Optional list parallel to the shorts — each
                short's OWN resolved location (its burned OST text). An entry
                that is falsy or equal to this generator's location keeps the
                parent-LF surfaces; ``None`` list preserves legacy behaviour.
            per_short_roads: Optional list parallel to ``per_short_regions`` — each
                short's OWN clip-GPS road names; they, with the clip's
                county names, are the only non-template Short tags.
            per_short_regions: Optional list parallel to the shorts — each
                short's OWN clip-GPS vote regions. A non-empty entry becomes that
                post's ``sampled_regions`` (the independent reference the CSV
                region gate checks the shipped surfaces against, so the gate is
                no longer label-vs-itself). A falsy entry / ``None`` list
                keeps the resolved-label default.
            per_short_title_facts: Optional list parallel to the shorts -- each
                short's :class:`TitleFacts`. A non-``None`` entry composes a
                stamped title for that short; ``None`` keeps the legacy
                composition. Must have length ``num_shorts`` when supplied.

            shorts_arms_enabled: Give each Short its ``yt_title_body`` and
                ``yt_hashtag_set`` arms (``assign_shorts_arms``, stratified within
                this project and the Short's position in its day) and render them.
                Deterministic for the same inputs; ``False`` keeps today's text.

            shorts_arm_history: The running ``assign_shorts_arms`` record. Pass ONE
                dict to every per-long-form call of a drive so each stratum
                (the base project, with its ``_lfNN`` suffix stripped, and the
                Short's position in its day) is balanced across the whole drive;
                ``None`` starts a fresh one for this call only.

            used_titles: The running record of the fact titles this batch used, by
                kind (``"titles"``, ``"tiktok"``). Pass ONE dict to every
                per-long-form call of a drive so a place is never the title of two
                Shorts of the drive; ``None`` starts a fresh one for this call only.

            per_short_shorts_arms: Explicit ``{factor: arm}`` per Short (a stored
                ``experiment_arms`` filtered to ``SHORTS_FACTORS``), rendered
                verbatim; no arm is ever assigned. An arm the text can no longer
                carry is absent from the post's ``experiment_arms``, which the
                caller compares to refuse the Short. Length ``num_shorts``; not
                combinable with ``shorts_arms_enabled``.

            previous_places: ``{"title": body, "description": body}`` of the Short
                published just before the first of these (the previous publish day's
                last Short), from its ``SocialPost.place_variety``. Each Short after
                the first is compared with the one before it in this batch.
            previous_places_at: ``{1-based position: {"title": body, "description":
                body}}`` that replaces the chained previous Short before that
                position (a queued-row rewrite seeds it from a Short's STORED text).

        Returns:
            List of SocialPost objects

        Raises:
            ValueError: If series_part is None — required for title uniqueness
                across multiple invocations with the same project_name.
            InteractiveShortHasBurnedLabel: An index flagged
                ``interactive`` has a non-empty ``per_short_locations`` entry
                (the burned on-screen OST label names a place). Today there
                is no per-short OST skip (not yet built), so an
                interactive short can only ship truthfully when the caller
                passes an empty/falsy label for that index.
        """
        if self.series_part is None:
            raise ValueError(
                "series_part is required for generate_all() to guarantee title "
                "uniqueness across invocations. Set series_part to an integer "
                "(e.g., 1 for first video in series, 0 for standalone)."
            )
        if global_indexes is not None and len(global_indexes) != num_shorts:
            raise ValueError(
                f"global_indexes has {len(global_indexes)} entries, expected "
                f"num_shorts={num_shorts}"
            )
        if per_short_title_facts is not None and len(per_short_title_facts) != num_shorts:
            raise ValueError(
                f"per_short_title_facts has {len(per_short_title_facts)} entries, expected "
                f"num_shorts={num_shorts}"
            )

        posts = []
        previous = dict(previous_places or {})
        if per_short_shorts_arms is not None:
            if shorts_arms_enabled:
                raise ValueError("per_short_shorts_arms and shorts_arms_enabled are exclusive")
            if len(per_short_shorts_arms) != num_shorts:
                raise ValueError(
                    f"per_short_shorts_arms has {len(per_short_shorts_arms)} entries, "
                    f"expected num_shorts={num_shorts}"
                )
        arm_history = {} if shorts_arm_history is None else shorts_arm_history
        if used_titles is not None:
            self._used_titles = used_titles
        # One stratum per base project: Phase D's title ordinals are unique across the
        # whole day, so a per-position stratum would hold one Short and balance nothing.
        arm_stratum = (re.sub(r"_lf\d+$", "", self.project_name), "")
        for i in range(1, num_shorts + 1):
            if previous_places_at and i in previous_places_at:
                previous = dict(previous_places_at[i])
            weather = None
            if weather_results and i - 1 < len(weather_results):
                weather = weather_results[i - 1]

            loc = None
            if per_short_locations and i - 1 < len(per_short_locations):
                loc = per_short_locations[i - 1]
            # A GPS-bearing short (own clip-GPS regions) ALWAYS gets the variant,
            # even when its label equals the parent's: the parent's destination
            # POI / road / county / terrain describe the LF, not this clip.
            has_regions = bool(
                per_short_regions and i - 1 < len(per_short_regions) and per_short_regions[i - 1]
            )
            roads = (
                per_short_roads[i - 1] if per_short_roads and i - 1 < len(per_short_roads) else None
            )
            gen = (
                self
                if not loc or (loc == self.location and not has_regions)
                else self._per_short_variant(
                    loc, per_short_regions[i - 1] if has_regions else None, roads
                )
            )
            # An explicit stable global index (assigned at plan time) wins
            # over the legacy len(posts)-derived index_offset. titles.yaml is
            # keyed by this same GLOBAL index — the one assigned to
            # post.short_index below.
            global_idx = global_indexes[i - 1] if global_indexes is not None else i + index_offset
            facts = per_short_title_facts[i - 1] if per_short_title_facts is not None else None
            route_rec = (
                per_short_route_records[i - 1]
                if per_short_route_records and i - 1 < len(per_short_route_records)
                else None
            )
            if per_short_shorts_arms is not None:
                shorts_arms = dict(per_short_shorts_arms[i - 1] or {})
            elif shorts_arms_enabled:
                shorts_arms = assign_shorts_arms(
                    arm_stratum,
                    global_idx,
                    arm_history,
                    gen._shorts_arm_factors(global_idx, facts),
                )
            else:
                shorts_arms = None
            post = gen.generate_post(
                i,
                weather,
                global_index=global_idx,
                title_facts=facts,
                previous_places=previous,
                route_record=route_rec,
                shorts_arms=shorts_arms,
            )
            if shorts_arms is not None and per_short_shorts_arms is None:
                record_shorts_arms(
                    arm_stratum,
                    arm_history,
                    {f: a for f, a in post.experiment_arms.items() if shorts_arms.get(f) == a},
                )
            self._used_titles.setdefault("titles", set()).add(post.youtube_title.strip().lower())
            self._used_titles.setdefault("tiktok", set()).add(post.tiktok_title.strip().lower())
            if post.place_variety:
                previous = {k: v["body"] for k, v in post.place_variety.items()}
            # An interactive short whose burned OST label is non-empty
            # makes "where is this?" a lie — the answer is already on screen.
            # No per-short OST skip exists yet, so this fires for every
            # interactive index unless the caller passes an empty label.
            if post.interactive and loc:
                raise InteractiveShortHasBurnedLabel(
                    f"short_index {global_idx}: flagged interactive but "
                    f"its burned on-screen label is {loc!r} — the OST already "
                    f"answers 'where is this?'"
                )
            # the short's OWN clip-GPS regions are the CSV gate's reference,
            # not the label the surfaces derive from — override when supplied so
            # the per-short region invariant stops checking a label against
            # itself. A falsy entry keeps the resolved-label default (GPS-less).
            if per_short_regions and i - 1 < len(per_short_regions):
                regions = per_short_regions[i - 1]
                if regions:
                    post.sampled_regions = list(regions)
                    post.sampled_roads = list(roads or [])
                    if per_short_route_records and i - 1 < len(per_short_route_records):
                        post.route_record = per_short_route_records[i - 1] or None
                    if (
                        not post.route_record
                        and len(regions) == 1
                        and backed_bracket_label(regions[0], self._bracket_route)
                    ):
                        post.bracket_route = dict(self._bracket_route)
            # Apply the resolved global index (explicit list wins,
            # else the legacy offset) for multi-LF drives.
            if global_indexes is not None or index_offset:
                post.short_index = global_idx
            posts.append(post)

        logger.info(f"Generated {len(posts)} unique social posts")

        # Verify uniqueness
        titles = [p.youtube_title for p in posts]
        captions = [p.caption for p in posts]
        unique_titles = len(set(titles))
        unique_captions = len(set(captions))

        if unique_titles < len(titles):
            logger.warning(f"Title uniqueness: {unique_titles}/{len(titles)} unique")
        if unique_captions < len(captions):
            logger.warning(f"Caption uniqueness: {unique_captions}/{len(captions)} unique")

        return posts
