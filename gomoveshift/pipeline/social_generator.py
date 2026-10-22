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
import logging
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from string import Formatter
from typing import Iterable, List, Mapping, Optional, Sequence

import yaml

from ..models import ResolvedLocation
from ._series_utils import _strip_series_suffix, parse_lf_label
from .content_validator import ContentValidationError, ContentValidator
from .final_text_record import reviewable_short_tags
from .geo_constants import US_STATE_ABBREVS
from .title_decision import decide_title, short_place_facts
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


# ─── location hashtags ────────────────────────────────────────────


def _location_hashtags(location: str) -> list[str]:
    """Emit one hashtag per comma-delimited location component.

    Splits on commas, strips a trailing series suffix, and removes any
    non-alphanumeric character inside each component (YouTube/IG reject
    punctuation in hashtags). Empty components are dropped, so a stray
    ``"Foo, , Bar"`` never produces a bare ``"#"``.

    The smashed concatenation produced by the legacy
    ``"#" + loc.replace(" ", "").replace(",", "")`` path is unrepresentable.
    """
    base = _strip_series_suffix(location)
    parts = [_arc_safe_anchor(p.strip()) for p in base.split(",")]
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

# ─── Stamped titles (drive path) ─────────────────────────────────────
# A drive short's title is ``"{stamp}: {body}"``. The stamp -- ``"Sep 14, 3rd
# stretch"`` -- is built from the short's own day key and its position in that
# day's sidecar, and IS the whole first-3-word prefix, so titles are unique
# by construction (see docs/pipeline/135-shorts-titles-and-interactive-family.md). Only the body rotates. The body may
# name only the short's own ``location_short``, and only when its place is
# clip-GPS verified; no preposition precedes ``{location}`` (a town label is the
# nearest settlement, not proof of containment). ``TITLE_FRAME_TEMPLATES`` above
# still serves callers with no facts (the single-video stage).
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


# ─── Per-platform caption templates ────────────────────────────────
# One SHORT template pool per platform — no shared "all platforms" pool and
# no hook-pool prefix (a caption that opens with the same rotating hook line
# regardless of platform is the templated-spam shape the anti-spam gate refuses at enqueue).
# Every template requires {location} — the mandatory place fact slot; when a
# short is flagged ``interactive``, ``_build_platform_captions`` fills
# {location} with LOCATION_WITHHELD instead so the place is never named,
# never by omitting the slot from some templates and not others.
YOUTUBE_CAPTION_TEMPLATES = [
    "A drive along {terrain_desc} through {location}.",
    "POV driving through {location}, following {terrain_desc}.",
    "This short traces {terrain_desc} in and around {location}.",
    "Behind the wheel through {location} — {terrain_desc} the whole way.",
]

FACEBOOK_CAPTION_TEMPLATES = [
    "Would you drive through {location}?",
    "Ever been through {location}?",
    "Just {terrain_desc} through {location}.",
    "{location} — worth the detour?",
]

INSTAGRAM_CAPTION_TEMPLATES = [
    "POV driving through {location} {weather_emoji}",
    "Rolling through {location} {weather_emoji}",
    "{location}, from behind the wheel {weather_emoji}",
    "Just {location} and the open road {weather_emoji}",
]

TIKTOK_CAPTION_TEMPLATES = [
    "POV: driving through {location}",
    "Somewhere in {location}",
    "{location} — {terrain_desc}",
    "Just driving through {location}",
]

_PLATFORM_CAPTION_TEMPLATES = {
    "youtube": YOUTUBE_CAPTION_TEMPLATES,
    "facebook": FACEBOOK_CAPTION_TEMPLATES,
    "instagram": INSTAGRAM_CAPTION_TEMPLATES,
    "tiktok": TIKTOK_CAPTION_TEMPLATES,
}

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
    ):
        """
        Args:
            location: Location name (e.g., "County Tipperary, Ireland")
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
        """
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
        self.location_short = _arc_safe_anchor(self.location_short_full)

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
        self.location_hashtags = _location_hashtags(location)
        # Backwards-compat alias for any external caller that read .location_tag.
        # Now points to the first (most-specific) component instead of the
        # smashed concatenation. Discovery surfaces should prefer
        # .location_hashtags.
        self.location_tag = (
            self.location_hashtags[0][1:]
            if self.location_hashtags
            else "".join(c for c in self.location_short if c.isalnum())
        )

        # Track POI usage within this instance for description deduplication
        self._used_pois: dict[str, int] = {}

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
            m = _DIRECTION_FRAMING_RE.search(text)
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

    def _compose_stamped_title(self, facts: TitleFacts, seed_suffix: str, max_len: int) -> str:
        """``"{stamp}: {body}"``. The ledger walk rotates the BODY only; the
        stamp (<=22 chars, the whole first-3-word prefix) never changes or truncates."""
        stamp = title_stamp(facts.date, facts.ordinal)
        named = facts.place_ok and bool(self.location_short)
        slots = defaultdict(
            str,
            location=self.location_short if named else "",
            secs=str(facts.secs),
            weekday=_WEEKDAYS[_parse_iso_date(facts.date).weekday()],
        )
        bodies = STAMPED_BODY_TEMPLATES if named else STAMPED_PLACE_FREE_TEMPLATES
        start = (facts.ordinal + self._project_seed(seed_suffix)) % len(bodies)

        from .longform_metadata import load_published_titles

        published = {t.strip().lower() for t in load_published_titles()}
        for offset in range(len(bodies)):
            body = bodies[(start + offset) % len(bodies)]
            candidate = f"{stamp}: {body.format_map(slots)}"[:max_len]
            if candidate.strip().lower() not in published:
                return candidate
        return f"{stamp}: {bodies[start].format_map(slots)}"[:max_len]

    def _compose_title(
        self,
        short_index: int,
        seed_suffix: str,
        max_len: int,
        facts: Optional[TitleFacts] = None,
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

        With ``facts`` (the drive path) the title is stamped instead --
        :meth:`_compose_stamped_title` -- and never raises
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
            return self._compose_stamped_title(facts, seed_suffix, max_len)
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
    ) -> str:
        """Pick a unique composed title for this short.

        Weather is NOT interpolated into titles:
        brightness-derived weather can't verify rain/sun as fact. `weather`
        is retained in the signature for caller stability; it no longer
        selects a title.
        """
        return self._compose_title(short_index, "titles", max_len=100, facts=facts)

    def _pick_platform_template(self, platform: str, short_index: int) -> str:
        """Pick this platform's caption template for ``short_index``.

        Project+series-level shuffle keyed by platform (not shared with any
        other platform's order) so two platforms never happen to walk their
        pools in lockstep.
        """
        templates = _PLATFORM_CAPTION_TEMPLATES[platform]
        project_rng = random.Random(self._project_seed(f"captions:{platform}"))
        shuffled = list(templates)
        project_rng.shuffle(shuffled)
        return shuffled[(short_index - 1) % len(shuffled)]

    def _platform_hashtags(
        self, platform: str, short_index: int, interactive: bool, limit: int
    ) -> "list[str]":
        """This platform's hashtags for ``short_index``, capped to ``limit``.

        Location hashtags are per-component (e.g. #DinglePeninsula
        #CountyKerry #Ireland). Withheld entirely when ``interactive`` — a
        location hashtag would leak the place a withheld caption just hid.

        Order is shuffled with a seed keyed on ``short_index`` (not just the
        template pick's ``(short_index - 1) % len(pool)`` cycle) — with only
        4 templates per platform, a batch of 5+ shorts is guaranteed by the
        pigeonhole principle to repeat a template; the independent per-short
        hashtag order is what keeps those two captions from being byte-
        identical.
        """
        tags = ["#POVDriving", "#RoadTrip"]
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
        rng = random.Random(self._project_seed(f"captions:{platform}:tags:{short_index}"))
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
    ) -> "dict[str, str]":
        """Compose one distinct caption per platform.

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
            else (self._pick_platform_template("youtube", short_index).format_map(fmap))
        )
        if self.recording_date:
            yt += f"\n\nFilmed {self.recording_date}."
        captions["youtube"] = yt

        # Facebook — one line, <=2 hashtags.
        fb_base = first_sentence or self._pick_platform_template(
            "facebook", short_index
        ).format_map(fmap)
        mandatory = [] if interactive else self._mandatory_location_tags()
        captions["facebook"] = self._append_hashtags(
            fb_base, self._platform_hashtags("facebook", short_index, interactive, limit=2)
        )

        # Instagram — short line + recording date + up to 8 tags. This is
        # the legacy shared-column shape (``post.caption`` below).
        ig_base = first_sentence or self._pick_platform_template(
            "instagram", short_index
        ).format_map(fmap)
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
        tt_base = first_sentence or self._pick_platform_template("tiktok", short_index).format_map(
            fmap
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

    def _build_youtube_description(self, short_index: int = 1, interactive: bool = False) -> str:
        """Build a short YouTube description for Shorts (max 200 chars).

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
        if interactive:
            base = f"POV driving through {LOCATION_WITHHELD} Pt.{short_index}"
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
            base = f"POV driving near {self.nearest_poi}, {self.social_location_full}"
            if self._used_pois[self.nearest_poi] > 1:
                base += f" Pt.{short_index}"
        elif self.road_name:
            base = f"POV drive along {self.road_name}, {self.social_location_full} Pt.{short_index}"
        else:
            base = f"POV driving through {self.social_location_full} Pt.{short_index}"

        # #Shorts is fixed; remaining hashtags are droppable. #4K removed:
        # shorts are 1080x1920 (never 4K), so the claim would be false.
        # #ScenicDrive removed: unverifiable genre claim (scenic-purge).
        fixed_hashtags = ["#Shorts"]
        optional_hashtags = ["#POVDriving"]
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
            optional_hashtags.append(f"#{self.location_tag}")

        all_hashtags = fixed_hashtags + optional_hashtags + pinned
        description = f"{base} {' '.join(all_hashtags)}"

        if len(description) <= 200:
            self._assert_region_present(short_index, "youtube description", description)
            return description

        # Drop optional hashtags from the end until it fits
        while optional_hashtags and len(description) > 200:
            optional_hashtags.pop()
            all_hashtags = fixed_hashtags + optional_hashtags + pinned
            description = f"{base} {' '.join(all_hashtags)}"

        # Final safety: the anchor alone still overruns 200 chars with every
        # optional hashtag dropped. Cut on a word boundary via the shared
        # truncator so a placename is never sliced mid-word (e.g. "…, Beaufort
        # Cou"); the ellipsis signals the cut. Falls back to a hard slice only
        # when the first token itself overruns (no whitespace to break on).
        from gomoveshift.pipeline.text_format import truncate_for_platforms

        if pinned and len(description) > 200:
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

        if not interactive:
            # Add location parts as phrase tags (split by comma, not whitespace)
            self._dedup_add(
                tags,
                [p.strip() for p in location_for_tags.split(",") if len(p.strip()) > 2],
            )

        # Region-derived tags ("Ireland road trip", etc.) replace the
        # hardcoded "road trip USA" that used to appear on every video.
        self._dedup_add(
            tags, _region_tags(self.location, self.resolved_location, self._own_regions)
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
    ) -> str:
        """Pick a TikTok-native title (folded into the same composition
        as ``_pick_title`` — no separate pool — with a distinct seed suffix
        so frame order differs from the YouTube title, truncated to
        TikTok's 90-char max).
        """
        return self._compose_title(short_index, "tiktok", max_len=90, facts=facts)

    def generate_post(
        self,
        short_index: int,
        weather: Optional[WeatherResult] = None,
        global_index: Optional[int] = None,
        title_facts: Optional[TitleFacts] = None,
    ) -> SocialPost:
        """
        Generate a unique social post for one short.

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

        for attempt in range(max_attempts):
            # Vary the seed on retries so we pick different hooks/titles
            rng = self._get_rng(short_index + attempt * 1000)

            interactive = hand.interactive if hand is not None else False
            if hand is not None:
                # Hand title/caption bypass the template pool entirely.
                title = hand.title
            else:
                title = self._pick_title(rng, weather, short_index, title_facts)
            # One caption per platform — a hand caption becomes the
            # YouTube caption verbatim and seeds the other three.
            captions = self._build_platform_captions(
                weather,
                short_index,
                interactive=interactive,
                hand_caption=(hand.caption if hand is not None else None),
            )
            tags = self._build_tags(weather, short_index, interactive=interactive)
            first_comment = self._build_first_comment(weather, short_index, interactive=interactive)

            # A hand title bypasses composition entirely, TikTok included
            # — composing anyway could raise TitleCapacityExceeded for an
            # otherwise fully-covered index.
            tiktok_title = (
                hand.title[:90]
                if hand is not None
                else self._pick_tiktok_title(rng, short_index, title_facts)
            )
            youtube_description = self._build_youtube_description(
                short_index, interactive=interactive
            )

            baseline = title[:100]  # YT max 100
            place_facts = None
            if hand is None and title_facts is not None and title_facts.place_ok:
                place_facts = short_place_facts(
                    baseline,
                    self.location_short,
                    self._own_regions,
                    segment_id=f"lf{self.series_part or 0:02d}-short{short_index:03d}",
                )
            decision = decide_title("short", baseline, None, suffix="", facts=place_facts)
            post = SocialPost(
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
        )
        clone.recording_date = self.recording_date
        clone.resolved_location = self.resolved_location
        clone._own_regions = list(regions) if regions else None
        # The clip's own GPS road names, never the parent LF's ``roads``.
        clone._own_roads = list(roads) if roads else []
        clone._used_pois = self._used_pois
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
        for i in range(1, num_shorts + 1):
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
            post = gen.generate_post(i, weather, global_index=global_idx, title_facts=facts)
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
