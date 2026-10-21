"""Per-LF title builder — produces the canonical endpoint-places + via-road
format used for YouTube LF titles.

Target format:
    "Port Royal to Lady's Island via Sea Island Parkway, SC"
     └────┬────┘    └─────┬─────┘     └────────┬────────┘  └┬┘
      start_town      end_town                road          state

Fallback cascade (each level degrades gracefully on missing data):
    1.  {start} to {end} via {road}, {state}         ← canonical
    2.  {start} to {end}, {state}                    ← no road
    3.  {town} Loop via {road}, {state}              ← start == end
    4.  {town}, {state}                              ← no start/end distinction
    5.  {per_lf_location}                            ← total geocode failure
                                                       (current behavior pre-fix)

The road name is harvested from the parenthetical suffix already produced
by ``derive_per_lf_locations`` in phase_bc_day_first.py — that function
reverse-geocodes the LF's GPS centroid and adds a ``(road)`` suffix when
an LF collides with sibling LFs in the same base location.

This module is deliberately standalone. It takes a geocoder interface and
a per-LF location string; it does not import Overpass or the SEO manager
directly, which keeps it unit-testable with mocks.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .validation.place_corrections import PlaceCorrections

from .place_shape import (  # noqa: F401 — re-exported for callers/tests of the old home
    _MID_VOTE_SAMPLES,
    _REGION_TOKENS_RE,
    _US_STATE_ABBR,
    _US_STATE_ABBR_VALUES,
    _central_window,
    _endpoint_county_state,
    _evenly_sample,
    _hyphenate_route_ref,
    _looks_like_county,
    _looks_like_region,
    _looks_like_township,
    _road_ref_members,
    _vote_mid_region,
    normalize_state_abbr,
    render_degraded_tier_label,
)

logger = logging.getLogger(__name__)

# ── YouTube title length policy ────────────────────────────────────────
# YouTube hard-limits titles at 100 characters. Search-result truncation
# typically hits around 70; we aim for ≤80 chars to stay readable in
# mobile list views while leaving headroom for "(Part N)" disambiguation.
MAX_TITLE_LEN = 100
SOFT_TITLE_LEN = 80

# A "via {local road}" suffix only describes the drive when both endpoints sit
# near that road. On a long crossing (e.g. a 250-mi coast-to-mountains
# drive) the dominant local road is ~250 mi from one endpoint and
# the suffix is actively wrong. When the start/end centroids are farther apart
# than this, every via-road title shape is skipped and the cascade falls through
# to the no-road "A to B, ST" form. Matches ost_formatter's endpoint sanity.
_VIA_ROAD_MAX_SPAN_KM = 50.0

# Deliberate superlong-part disambiguator (``build_superlong_part_title``) —
# an em-dash suffix, distinct BY CONSTRUCTION from ``ensure_unique_titles``'s
# parenthetical "(Part N of M)" collision-dedup suffix, so a superlong title
# that already carries this can't be mistaken for one needing dedup.
_SUPERLONG_PART_SUFFIX_RE = re.compile(r"\s+—\s+Part\s+\d+\s+of\s+\d+\s*$")


def is_us_state(token: Optional[str]) -> bool:
    """True when ``token`` is a US state — full name or 2-letter abbreviation.

    Used by the country-derivation helpers (longform_metadata,
    playlist_resolver) to recognise a US drive whose location label is a
    bare ``"County, State"`` / ``"... , ST"`` string with NO country
    component — the shape produced by ``derive_per_lf_locations`` /
    ``build_lf_title`` now that the drive-wide country-bearing
    fallback is gone. Strict membership (full-name keys OR known 2-letter values)
    so a non-US 2-letter token like a country code (``"IE"``) is NOT
    mistaken for a state.
    """
    if not token:
        return False
    s = token.strip()
    if not s:
        return False
    if s.lower() in _US_STATE_ABBR:
        return True
    return len(s) == 2 and s.upper() in _US_STATE_ABBR_VALUES


# Reverse of ``_US_STATE_ABBR`` (abbr → title-cased full name) for the
# front-loaded title, which leads with the full state name as a search term.
_US_STATE_FULL: dict[str, str] = {abbr: name.title() for name, abbr in _US_STATE_ABBR.items()}


def state_full_name(token: Optional[str]) -> str:
    """Full US state name for a 2-letter abbr or full name (front-load).

    ``"NC"`` → ``"North Carolina"``; a non-US/unknown token (e.g. ``"Kerry"``)
    is returned stripped-unchanged so it front-loads verbatim. Inverse of
    ``normalize_state_abbr``.
    """
    if not token:
        return ""
    s = token.strip()
    if not s:
        return ""
    if s.lower() in _US_STATE_ABBR:  # already a full state name
        return s.title()
    if len(s) == 2 and s.upper() in _US_STATE_ABBR_VALUES:
        return _US_STATE_FULL[s.upper()]
    return s


def _extract_road_from_suffix(per_lf_location: str) -> Optional[str]:
    """Pull ``road`` out of a per-LF location string's ``(road)`` suffix.

    ``derive_per_lf_locations`` writes strings like
    ``"Colleton County, South Carolina (Blocker Run Road)"`` when
    road-name disambiguation fires. This helper extracts the
    parenthetical content.

    Returns None when the string has no ``(...)`` suffix or the
    suffix looks like a ``(Part N)`` numeric disambiguator (the
    no-geocoder fallback).
    """
    if not per_lf_location:
        return None
    m = re.search(r"\(([^)]+)\)\s*$", per_lf_location)
    if not m:
        return None
    content = m.group(1).strip()
    # Skip the numeric "Part N" disambiguator — that's an index, not a road
    if re.match(r"^Part\s+\d+$", content, re.IGNORECASE):
        return None
    return content


# ── Via-road quality filter ──────────────────────────────────────
# A via-road earns a place in the title / burned OST ONLY when it is a
# *named through-route* a viewer could recognise or search — never a
# residential side street, a track/service way, or a business name that
# leaked into OSM's ``road`` slot (the "Smith County via Dickie's" /
# "Macon County via North 4th Street" class). The signals come free from ``reverse_geocode_detailed``:
# ``place_class`` ("highway" when the matched OSM object is a road) and
# ``match_type`` (the highway tag). Below the through-road tier we OMIT the
# via-clause rather than stamp junk (robust-by-construction: the generator
# simply cannot surface a junk road).

# Highway tags that describe a road a drive video would name — the
# trunk/primary/secondary/tertiary "through" tier plus motorways and their
# link ramps. Everything else (residential, unclassified, living_street,
# track, service, footway, path, cycleway, pedestrian) is local or
# non-drivable and is dropped.
_THROUGH_ROAD_TYPES = frozenset(
    {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
    }
)

# Route-ref shapes: US-64, SC 174, I-26, NC-281, N86, R563, M7, A87, or a
# spelled "Highway 64" / "Route 28". A road carrying a route number is a
# named route regardless of the matched OSM class.
_ROUTE_REF_RE = re.compile(
    r"\b[A-Z]{1,2}[-\s]?\d{1,4}\b|\b(?:Highway|Hwy|Route|Rte|Interstate)\s+\d{1,4}\b",
    re.IGNORECASE,
)

# Through-road NAME suffixes — a road named "... Parkway/Highway/Turnpike"
# is a named through-route even when OSM ``place_class``/``match_type`` is
# absent (stale cache, place-node match, the suffix/no-detail path). This is
# what keeps a real "Sea Island Parkway" while still dropping the local
# "Homestead Drive" / "North 4th Street" / "Blocker Run Road" class
# (Drive/Street/Road/Lane/Court are NOT through suffixes). "Pass"/scenic
# names are covered by ``_KNOWN_DISPLAY_NAMES`` instead, to avoid the many
# residential "... Trail" false positives.
_THROUGH_ROAD_NAME_RE = re.compile(
    r"\b(?:Parkway|Highway|Freeway|Expressway|Turnpike|Byway|Pike|Beltway|Causeway|Motorway)\b",
    re.IGNORECASE,
)

# A hyphenated interstate ref as produced by ``_hyphenate_route_ref`` ("I-81").
# Used by the class-aware via-road tie policy: an interstate member is the only
# class trusted to win a vote tie outright.
_INTERSTATE_REF_RE = re.compile(r"^I-\d+$")


def _road_passes_quality(
    road: Optional[str],
    *,
    place_class: Optional[str] = None,
    match_type: Optional[str] = None,
) -> bool:
    """True when ``road`` may be surfaced as a via-road.

    Accepts, in descending confidence:
      1. a road carrying a route ref (US-64, SC-174, N86) — a named route
         regardless of the matched OSM class;
      2. a well-known scenic byway (``chapters._shared._KNOWN_DISPLAY_NAMES``);
      3. a road whose matched OSM feature is a *through-road* highway
         (``place_class == "highway"`` AND ``match_type`` in the
         trunk/primary/secondary/tertiary/motorway tier).

    Rejects everything else — empty roads, residential/track/service ways,
    and business/POI names that landed in OSM's ``road`` slot (the
    possessive/tourism guard in ``is_valid_chapter_placename`` catches the
    "Dickie's" class). The caller omits the via-clause rather than stamping
    a junk road. ``place_class``/``match_type`` default None (the suffix /
    no-detail path), where only the ref + byway accepts can fire.
    """
    if not road or not road.strip():
        return False
    name = road.strip()
    # 1. Route ref → a named route, accept regardless of OSM class.
    if _ROUTE_REF_RE.search(name):
        return True
    from .chapters._shared import _KNOWN_DISPLAY_NAMES, is_valid_chapter_placename

    # 2. Known scenic byway (curated display-name map) — SUBSTRING match so a
    #    named scenic route carrying a generic suffix ("Slea Head Drive" holds
    #    "slea head") still survives even though "Drive" is not a through-suffix.
    #    The curated list is specific enough that a junk road won't contain one.
    low = name.lower()
    if any(byway in low for byway in _KNOWN_DISPLAY_NAMES):
        return True
    # 3. Through-road NAME suffix (Parkway/Highway/Turnpike/…) — survives a
    #    missing OSM class. Still drop a possessive/business string.
    if _THROUGH_ROAD_NAME_RE.search(name):
        return is_valid_chapter_placename(name)
    # 4. Through-road highway class — reliable only when the matched OSM
    #    object IS the highway. Still reject a possessive/business string in
    #    the road slot even if OSM tagged it highway (defensive).
    if place_class == "highway" and (match_type or "").lower() in _THROUGH_ROAD_TYPES:
        return is_valid_chapter_placename(name)
    return False


def _recognizable_settlements(places: Any, county_part: Optional[str]) -> list[str]:
    """Travel-ordered, consecutively-deduped list of RECOGNIZABLE settlements
    from a place sequence.

    Filters the canonical ``ordered_place_sequence`` / ``DominantLocation.
    ordered_places`` down to names a viewer would recognise and search: drops
    bare county-shaped entries (the sequence falls back to the county when a
    sample has no town — that's not a settlement) and business/POI/possessive
    strings (``is_valid_chapter_placename``'s "Dickie's" guard). The first
    element anchors a single-settlement LF; first+last frame a crossing.
    """
    from .chapters._shared import is_valid_chapter_placename

    county_base = (county_part or "").strip().lower()
    out: list[str] = []
    for p in places or []:
        if not isinstance(p, str):
            continue
        s = p.strip()
        if not s:
            continue
        low = s.lower()
        # Skip the county itself and any bare "... County"/"... Township"/broad
        # REGION ("ACE Basin") entry — those are the rural fallback, not a
        # recognizable settlement. The region check is the sequence-ladder
        # counterpart to the endpoint demotion in ``build_lf_title``.
        if "county" in low or "township" in low or low == county_base:
            continue
        if _looks_like_region(s):
            continue
        # Drop a bare single-token possessive ("Dickie's") — a business name,
        # not a place. ``is_valid_chapter_placename`` only catches the
        # multi-word "Sam's Diner" shape (it keys on ``"'s "`` with a space).
        if re.fullmatch(r"[\w.]+'s", s):
            continue
        if not is_valid_chapter_placename(s):
            continue
        if out and out[-1] == s:
            continue
        out.append(s)
    return out


def _strip_paren_suffix(location: str) -> str:
    """Drop a trailing ``(road)`` / ``(Part N)`` disambiguator suffix.

    The one home for the paren-suffix regex — shared by
    ``_split_location_parts`` and ``region_resolver._region_from_label``
    (the suffix must never poison the region/state token).
    """
    if not location:
        return location
    return re.sub(r"\s*\([^)]+\)\s*$", "", location).strip()


def _split_location_parts(per_lf_location: str) -> tuple[str, str]:
    """Split ``"County, State (...)"`` into ``(county, state)``.

    Drops any trailing ``(...)`` suffix. Returns ``("", "")`` when the
    string doesn't match the expected comma pattern.
    """
    if not per_lf_location:
        return ("", "")
    base = _strip_paren_suffix(per_lf_location)
    parts = [p.strip() for p in base.split(",", 1)]
    if len(parts) == 2:
        return (parts[0], parts[1])
    if len(parts) == 1:
        return (parts[0], "")
    return ("", "")


def _geocode_town(geocoder, lat: float, lon: float) -> Optional[str]:
    """Return a short town/city/locality name for a coordinate.

    Uses the geocoder's ``reverse_geocode`` method (which returns a
    location-like object with attrs like ``city``, ``locality``,
    ``suburb``). Returns the first non-empty preferred field.

    None on failure OR when lat/lon is (0, 0) — no call made.
    """
    if lat == 0.0 and lon == 0.0:
        return None
    if geocoder is None:
        return None
    try:
        loc = geocoder.reverse_geocode(lat, lon)
    except Exception as exc:  # noqa: BLE001 — log + fall back
        logger.debug("title_builder._geocode_town reverse_geocode failed: %s", exc)
        return None
    if loc is None:
        return None
    # most significant settlement class first — city/town before
    # village/hamlet (the old hamlet-before-city order let hamlet "Cline" beat
    # town "Wytheville"). ``name`` is deliberately NOT read: it comes from
    # ``format_location_string``, which prefers ``poi_name`` — a business/POI
    # the vehicle merely passed near — so accepting it can burn a pub or shop
    # as an "A to B" endpoint. A geocode with no settlement field returns None
    # and the sequence backfill / county cascade takes over instead.
    for attr in ("city", "town", "village", "hamlet", "locality", "suburb", "place"):
        val = getattr(loc, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


# ── Dominant-county reducer (cross-county-line robustness) ─────────────
# Replaces single-centroid reverse-geocoding on the drive day-first label
# path. A drive's arithmetic-mean centroid lands on the wrong side of a
# county/state line when the route crosses one (e.g. SC: Beaufort/Fripp
# ~32.30N vs Colleton/Edisto ~32.66N). Voting the modal county across
# sampled points keeps the label on the county that actually holds the
# plurality of footage. Same idiom as the Counter votes already used in
# ``_drive_helpers.py`` and ``ensure_unique_titles`` below.
_NAME_FIELDS = ("city", "town", "village", "hamlet", "locality", "suburb", "name")

# settlement class per name field — lower is more significant. The
# ``_NAME_FIELDS`` iteration order above is best-class-first, so the first
# non-empty field IS a sample's best candidate; this map scores it so the
# cross-sample pick can rank a town ("Wytheville") over a hamlet ("Cline")
# regardless of travel/sample order. The generic ``name`` slot is POI-prone,
# hence least significant until refined by the rank fields below.
_NAME_FIELD_CLASS_RANK = {
    "city": 0,
    "town": 0,
    "village": 1,
    "hamlet": 2,
    "locality": 3,
    "suburb": 3,
    "name": 4,
}
# Nominatim ``addresstype`` (the address bucket the match fell into — exposed
# by the gps service's detailed dict) on the same class scale.
_ADDRESSTYPE_CLASS_RANK = {
    "city": 0,
    "town": 0,
    "village": 1,
    "hamlet": 2,
    "locality": 3,
    "suburb": 3,
    "neighbourhood": 3,
}


def _sample_name_candidate(detailed: dict) -> Optional[tuple[str, int]]:
    """Best ``(name, class_rank)`` for one geocoded sample, or None.

    A pick from the generic ``name`` slot is refined by the settlement-rank
    fields when present: ``addresstype`` names the matched object's class
    directly; else ``place_rank`` (Nominatim tier — city=16, town=18,
    village=19, hamlet=20; lower = more significant) maps onto the class
    scale. Absent/None rank fields (legacy cached dicts) are neutral — the
    field-derived class stands unchanged.
    """
    for fld in _NAME_FIELDS:
        val = detailed.get(fld)
        if not (isinstance(val, str) and val.strip()):
            continue
        rank = _NAME_FIELD_CLASS_RANK.get(fld, 4)
        if fld == "name":
            atype = detailed.get("addresstype")
            prank = detailed.get("place_rank")
            if isinstance(atype, str) and atype.strip().lower() in _ADDRESSTYPE_CLASS_RANK:
                rank = _ADDRESSTYPE_CLASS_RANK[atype.strip().lower()]
            elif isinstance(prank, int) and 16 <= prank <= 20:
                rank = {19: 1, 20: 2}.get(prank, 0)
        return (val.strip(), rank)
    return None


def _pick_ranked_name(cands: list[tuple[str, int, int]]) -> Optional[str]:
    """Winning-county name from ``(name, class_rank, weight)`` candidates in
    travel order: best (lowest) class rank, then highest accumulated footage
    weight, then earliest appearance. Deterministic and direction-independent
    for distinct classes; a full tie keeps travel order (old behaviour)."""
    if not cands:
        return None
    best_rank: dict[str, int] = {}
    weight_sum: dict[str, int] = {}
    first_idx: dict[str, int] = {}
    for idx, (nm, rank, wt) in enumerate(cands):
        first_idx.setdefault(nm, idx)
        best_rank[nm] = min(rank, best_rank.get(nm, rank))
        weight_sum[nm] = weight_sum.get(nm, 0) + wt
    return min(best_rank, key=lambda nm: (best_rank[nm], -weight_sum[nm], first_idx[nm]))


@dataclass(frozen=True)
class DominantLocation:
    """Modal ``(county, state)`` across a sampled GPS trace.

    ``name`` is a representative place inside the winning county, picked
    rank-aware ACROSS its samples: best settlement class first —
    city/town beat village/hamlet regardless of sample order — while a
    hamlet-only county still surfaces its hamlet ("Fort Fremont") instead of
    the bare county. ``votes`` maps ``"County|State"`` → sample count,
    surfaced for the ``--label-dry-run`` breakdown.
    """

    county: Optional[str]
    state: Optional[str]
    name: Optional[str] = None
    votes: dict[str, int] = field(default_factory=dict)
    # A road sampled INSIDE the winning county, so the via-road
    # disambiguator follows the vote, not the (possibly wrong-county) centroid.
    road: Optional[str] = None
    # route-ordered representative place labels (town/village, then
    # county), consecutively deduped — feeds ResolvedLocation.ordered_place_
    # sequence (the description ``🛤️ Route:`` line + chapter names). Default
    # empty so existing constructions / tests are unaffected.
    ordered_places: list[str] = field(default_factory=list)
    # COUNT of sampled points that reverse-geocoded to ANY (county,
    # state) — the true quorum unit. Distinct from ``votes`` mass, which is
    # weighted by footage-span catchment (a single sample can carry a mass of
    # several), so callers must gate the quorum on THIS, not the vote total.
    # Default 0 keeps existing constructions / tests valid.
    geocoded_sample_count: int = 0
    # The geocoded country of the winning county (e.g. "Ireland"), so a
    # non-US ``ResolvedLocation`` carries a real country instead of None — the
    # single signal every region surface (playlist/tags) routes off. Default
    # None keeps existing constructions / tests valid.
    country: Optional[str] = None
    # RAW quality-passing road strings from the winning county's samples
    # (one entry per sample, OSM shape — space-separated, ";"-joined
    # concurrencies). Carried onto ResolvedLocation so the specificity ladder's
    # via-road support check has real refs; normalized at the evidence site.
    road_refs: list[str] = field(default_factory=list)
    # the FULL winning-county road tally (``{hyphenated_member: count}``)
    # the via-road winner was chosen from — carried onto ResolvedLocation so the
    # provenance sidecar can persist the vote, not just the winner. Default empty
    # keeps existing constructions / tests valid.
    road_votes: dict[str, int] = field(default_factory=dict)


def _point_latlon(point: Any) -> Optional[tuple[float, float]]:
    """Extract ``(lat, lon)`` from a dict, model object, or positional
    tuple/list. Tuples are the shorts ``(t, lat, lon)`` shape (lat/lon at
    indices 1/2); a bare 2-tuple is ``(lat, lon)``. Returns None when no
    usable pair is present."""
    if isinstance(point, (tuple, list)):
        if len(point) >= 3:
            lat, lon = point[1], point[2]
        elif len(point) == 2:
            lat, lon = point[0], point[1]
        else:
            return None
    elif isinstance(point, dict):
        lat = point.get("lat")
        lon = point.get("lon", point.get("lng"))
    else:
        lat = getattr(point, "lat", None)
        lon = getattr(point, "lon", None)
        if lon is None:
            lon = getattr(point, "lng", None)
    if lat is None or lon is None:
        return None
    try:
        return (float(lat), float(lon))
    except (TypeError, ValueError):
        return None


def _evenly_sample_weighted(seq: list, k: int) -> list:
    """Like :func:`_evenly_sample`, but pairs each sampled item with the
    count of original points it stands for (its catchment span). Used to
    weight the county vote by *footage duration* (≈ point density), not just
    sample count, so a long stretch in one county outvotes a brief dip into
    another regardless of drive direction."""
    n = len(seq)
    if n == 0:
        return []
    sampled = _evenly_sample(seq, k)
    if len(sampled) >= n:
        return [(item, 1) for item in sampled]
    if len(sampled) <= 1:
        # A single sample stands for the whole sequence (its catchment is ``n``);
        # also guards the ``len(sampled) - 1`` divide below against zero.
        return [(item, n) for item in sampled]
    # Catchment = points nearest each sample index (a Voronoi split of the
    # index line). Direction-independent: reversing seq mirrors the split.
    idxs = [round(i * (n - 1) / (len(sampled) - 1)) for i in range(len(sampled))]
    weighted = []
    for pos, item in enumerate(sampled):
        lo = 0 if pos == 0 else (idxs[pos - 1] + idxs[pos]) // 2 + 1
        hi = n if pos == len(sampled) - 1 else (idxs[pos] + idxs[pos + 1]) // 2
        weighted.append((item, max(hi - lo, 1)))
    return weighted


def dominant_location(
    points, geocoder, *, max_samples: int = 9, min_votes: int = 1
) -> Optional[DominantLocation]:
    """Vote the modal ``(county, state)`` across a GPS trace.

    Filters noisy/stationary/null-island points, samples up to
    ``max_samples`` evenly, reverse-geocodes each (cached coords are free;
    rate-limit only bites on cache misses), and returns the county holding
    the plurality of samples. Two same-named counties in different states
    (Washington County TN/VA) are never merged, and an exact top-weight tie
    across two states abstains (None) rather than naming a state by alphabet.
    Returns None when there is no geocoder, no
    usable point, or no sample produced a county/state — callers fall back
    to their existing centroid path so the "total function" raise
    semantics are preserved.

    ``min_votes`` is a GEOCODE QUORUM FLOOR: the number of sampled
    points that must reverse-geocode to ANY ``(county, state)`` before the
    plurality winner is trusted. Below this floor the winning county is a
    weak plurality off a handful of samples — exactly the far-road case
    where a slim plurality off only a few geocoded points would stamp the
    wrong county. When fewer than ``min_votes`` samples produced
    a usable vote, this returns None so the caller degrades to its
    centroid/fallback label path instead of publishing the weak plurality.
    NOTE the quorum is a COUNT of successfully-geocoded samples, NOT the
    weighted vote total (votes are weighted by footage-span catchment, so
    the weighted sum is not a sample count). The default ``min_votes=1``
    preserves existing behaviour for every caller that does not opt in.
    """
    if geocoder is None:
        return None
    pts = list(points or [])
    if not pts:
        return None

    # Quality filter where the point shape supports it (dict/object);
    # positional tuples have no hdop/speed/time fields so they pass through.
    try:
        from .gps_quality import filter_high_quality_points

        filtered = filter_high_quality_points(pts)
        if filtered:
            pts = filtered
    except Exception as exc:  # noqa: BLE001 — filter is best-effort
        logger.debug("dominant_location: quality filter skipped: %s", exc)

    coords: list[tuple[float, float]] = []
    for p in pts:
        ll = _point_latlon(p)
        if ll is None:
            continue
        if ll[0] == 0.0 and ll[1] == 0.0:
            continue
        coords.append(ll)
    if not coords:
        return None

    votes: Counter = Counter()
    detail_by_key: dict[tuple, list] = {}
    # per-county (name, class_rank, weight) candidates in travel order —
    # the winning county's name is picked ACROSS its samples by class rank,
    # never first-sample-wins (hamlet "Cline" must not beat town "Wytheville"
    # just because it was driven through first).
    name_cands_by_key: dict[tuple, list[tuple[str, int, int]]] = {}
    ordered_places: list[str] = []  # route-ordered place labels
    geocode_quorum = 0  # COUNT of samples that yielded any (county, state)
    for (lat, lon), weight in _evenly_sample_weighted(coords, max_samples):
        try:
            detailed = geocoder.reverse_geocode_detailed(lat, lon) or {}
        except Exception as exc:  # noqa: BLE001 — log + skip this sample
            logger.debug("dominant_location: reverse_geocode_detailed failed: %s", exc)
            continue
        county = detailed.get("county")
        state = detailed.get("state")
        if not county and not state:
            continue
        geocode_quorum += 1
        key = (county, state)
        # Weight by footage span (catchment), not a flat +1, so a brief dip
        # into a neighbouring county can't tie a long stretch in the main one.
        votes[key] += weight
        detail_by_key.setdefault(key, []).append(detailed)
        # best town-level label for this sample (falls back to county),
        # consecutively deduped → the route's place sequence in travel order.
        cand = _sample_name_candidate(detailed)
        if cand is not None:
            name_cands_by_key.setdefault(key, []).append((cand[0], cand[1], weight))
        place = cand[0] if cand is not None else county
        if place and (not ordered_places or ordered_places[-1] != place):
            ordered_places.append(place)
    if not votes:
        return None

    # Geocode quorum floor: a plurality off too few successfully-geocoded
    # samples is unreliable (the far-road case — a slim plurality off a
    # handful of geocoded points can stamp the wrong county). Degrade to None so
    # the caller falls back to its centroid/single-geocode label path.
    if geocode_quorum < min_votes:
        logger.warning(
            "dominant_location: geocode quorum miss (%d < %d) — %d sample(s) "
            "produced a county/state; degrading to caller fallback",
            geocode_quorum,
            min_votes,
            geocode_quorum,
        )
        return None

    # county-first vote. One real county can reverse-geocode under two
    # "state" variants — an Irish county tagged once with its province-as-state
    # ("Munster") and once STATELESS — splitting its vote across
    # ``(county, "Munster")`` and ``(county, None)`` so a genuinely-minority
    # county can steal the plurality and stamp the WRONG county. A STATELESS
    # variant is therefore folded onto its county's single non-empty state.
    #
    # Two DISTINCT non-empty states under one county name are two different
    # real counties (Washington County TN/VA/MD/PA, Franklin PA/VA, ...) and
    # are NEVER merged: each keeps its own key and weight. Their stateless
    # samples cannot be attributed to either, so they are dropped from the
    # tally (DEBUG-logged) rather than handed to a state by alphabet/majority.
    states_by_county: dict = {}
    for c, s in votes:
        if s:
            states_by_county.setdefault(c, set()).add(s)
    if any(not s or len(states_by_county.get(c, ())) > 1 for c, s in votes):
        merged_votes: Counter = Counter()
        merged_detail: dict[tuple, list] = {}
        merged_names: dict[tuple, list[tuple[str, int, int]]] = {}
        for (c, s), n in votes.items():
            known = states_by_county.get(c, set())
            if s or not known:
                mkey = (c, s)
            elif len(known) == 1:
                mkey = (c, next(iter(known)))
            else:
                logger.debug(
                    "dominant_location: dropping %s stateless sample(s) of %r — "
                    "name exists in %d states (%s); cannot attribute",
                    n,
                    c,
                    len(known),
                    ", ".join(sorted(known)),
                )
                continue
            merged_votes[mkey] += n
            merged_detail.setdefault(mkey, []).extend(detail_by_key.get((c, s), []))
            merged_names.setdefault(mkey, []).extend(name_cands_by_key.get((c, s), []))
        votes, detail_by_key, name_cands_by_key = merged_votes, merged_detail, merged_names
        if not votes:
            return None

    # An exact weight tie between keys in two different non-empty states would
    # name a state by alphabet, not footage — abstain (None) so the caller
    # degrades like a quorum miss. A same-state tie keeps the tie-break below.
    top = max(votes.values())
    if len({k[1] for k in votes if votes[k] == top and k[1]}) > 1:
        logger.debug("dominant_location: cross-state weight tie at %s — abstaining", top)
        return None

    # Deterministic, direction-independent winner: highest weighted vote,
    # ties broken alphabetically by (county, state) — never insertion/sample
    # order (= drive direction), which is what ``most_common`` would use.
    (county, state) = min(
        votes,
        key=lambda k: (-votes[k], str(k[0] or ""), str(k[1] or "")),
    )
    # rank-aware name — best settlement class across ALL winning-county
    # samples, then accumulated footage weight, then travel order (see
    # ``_pick_ranked_name``). The county vote above is untouched.
    name = _pick_ranked_name(name_cands_by_key.get((county, state), []))
    # The via-road must be a road from the WINNING county's samples, never
    # the centroid's (which can fall in the runner-up county). AND it must
    # be a named through-route — a residential/track/service way or a business
    # name in OSM's ``road`` slot is dropped so the via-clause is omitted
    # rather than stamping junk ("Dickie's"). VOTE the road across all
    # winning-county samples rather than taking the first quality match — a
    # lone off-route local way (e.g. Marion's "South Main Street", OSM-classed
    # as a through ``highway`` so it clears the quality filter) must not outvote the road the
    # drive actually used (``I 81``). Credit EVERY member of a concurrency
    # ``ref`` (``I 77;I 81;US 52`` → I-77, I-81 AND US-52), not just the primary
    # — else an I-77/I-81 overlap collapses to "I-77" and outvotes the I-81 the
    # drive stays on once I-77 branches north.
    road_votes: Counter = Counter()
    road_refs: list[str] = []  # raw per-sample refs for the ladder's support check
    country: Optional[str] = None  # geocoded country of the winning county
    for detailed in detail_by_key[(county, state)]:
        if country is None:
            _c = detailed.get("country")
            if isinstance(_c, str) and _c.strip():
                country = _c.strip()
        rd = detailed.get("road")
        if (
            isinstance(rd, str)
            and rd.strip()
            and _road_passes_quality(
                rd,
                place_class=detailed.get("place_class"),
                match_type=detailed.get("match_type"),
            )
        ):
            road_refs.append(rd.strip())
            for member in _road_ref_members(rd.strip()):
                road_votes[member] += 1
    # Modal carried route with a VOTE-based, class-aware tie policy. The
    # old guard was SIGNATURE-based ("every sample inside ONE multi-route
    # concurrency → no road"), so a window spanning TWO overlap signatures that
    # share the same interstates — US-11 joining the I-77/I-81/US-52 overlap
    # mid-window (real Wytheville refs) — escaped it, and the alphabetical
    # tie-break stamped "via I-77" on an I-81 drive.
    # Policy: a single top-voted member wins outright; on a tie, a LONE
    # interstate beats its non-interstate companions ("I 81;US 11" → I-81,
    # where the old guard dropped the road); a tie holding two-plus interstates
    # — or none — is undistinguishable, so emit NO via-road (a road-less title
    # is safe; a wrong road is not) rather than pick one alphabetically.
    road: Optional[str] = None
    if road_votes:
        top = max(road_votes.values())
        tied = sorted(r for r, n in road_votes.items() if n == top)
        if len(tied) == 1:
            road = tied[0]
        else:
            interstates = [r for r in tied if _INTERSTATE_REF_RE.match(r)]
            if len(interstates) == 1:
                road = interstates[0]
    return DominantLocation(
        county=county,
        state=state,
        name=name,
        # A stateless winning county (Irish counties, US independent cities
        # like Washington DC) has ``s is None``; serialize it to an EMPTY slot,
        # never the literal ``"None"`` — the consumer partition would otherwise
        # read a truthy ``"None"`` and build a malformed ``"County, None"``
        # region that false-fails the cross-surface location invariant.
        votes={f"{c}|{s or ''}": n for (c, s), n in votes.items()},
        road=road,
        ordered_places=ordered_places,
        geocoded_sample_count=geocode_quorum,
        country=country,
        road_refs=road_refs,
        road_votes=dict(road_votes),
    )


def _validate_route_endpoints(
    geocoder,
    first_centroid,
    last_centroid,
    mid_coords,
    *,
    county_shares: Optional[dict[tuple[Optional[str], Optional[str]], float]] = None,
) -> tuple[bool, bool]:
    """Return ``(keep_start, keep_end)`` for a cross-boundary "A to B" claim.

    When start and end resolve to different counties/states, a stale-GPS or
    geocoder flake on one endpoint can fabricate a bogus route. Validate
    against the on-route midpoint, VOTED across several mid-route samples
    (``mid_coords``, a list of ``(lat, lon)``) so one flaky mid geocode can't
    fail the demotion open: the endpoint the voted midpoint does NOT
    corroborate (while the other IS) is the flake and is dropped. A genuine
    crossing (midpoint in a third area, or inconclusive) keeps both. Mirrors
    the Shape-B midpoint demotion in ``ingest/ost_formatter.py``.

    ``county_shares`` (``{(county, state): footage share}``) protects a real
    location: an endpoint whose county holds at least
    ``LOCATION_MATERIAL_SHARE`` of the footage-weighted vote is NEVER dropped
    (a flake has ~zero footage by definition), however long the drive spent in
    the other county. ``None`` keeps the midpoint-only behaviour.
    """
    from .validation.content_gates import LOCATION_MATERIAL_SHARE

    shares = county_shares or {}
    s_county, s_state = _endpoint_county_state(geocoder, first_centroid)
    e_county, e_state = _endpoint_county_state(geocoder, last_centroid)
    if (s_county, s_state) == (e_county, e_state):
        return (True, True)  # same area — nothing to validate
    if (s_county is None and s_state is None) or (e_county is None and e_state is None):
        return (True, True)  # missing data — don't second-guess
    m_county, m_state = _vote_mid_region(geocoder, mid_coords)
    if m_county is None and m_state is None:
        return (True, True)
    mid = (m_county, m_state)
    mid_matches_start = mid == (s_county, s_state)
    mid_matches_end = mid == (e_county, e_state)
    if mid_matches_start and not mid_matches_end:
        # end is the flake — unless its county holds material footage
        return (True, shares.get((e_county, e_state), 0.0) >= LOCATION_MATERIAL_SHARE)
    if mid_matches_end and not mid_matches_start:
        # start is the flake — unless its county holds material footage
        return (shares.get((s_county, s_state), 0.0) >= LOCATION_MATERIAL_SHARE, True)
    return (True, True)  # genuine crossing or inconclusive


def _county_shares(dom) -> Optional[dict[tuple[str, str], float]]:
    """``{(county, state): share}`` of the footage-weighted vote, or None."""
    votes = getattr(dom, "votes", None) or {}
    total = sum(votes.values())
    if not total:
        return None
    out: dict[tuple[str, str], float] = {}
    for key, weight in votes.items():
        county, _, state = key.partition("|")
        out[(county, state)] = weight / total
    return out


def _material_counties(dom, geocoder, first_centroid, last_centroid) -> list[tuple[str, str]]:
    """Counties holding >= ``LOCATION_MATERIAL_SHARE`` of the footage vote.

    Ordered start-to-end when the endpoint counties are both material (travel
    order), otherwise by descending share.
    """
    from .validation.content_gates import LOCATION_MATERIAL_SHARE

    shares = _county_shares(dom) or {}
    material = sorted(
        (k for k, v in shares.items() if k[0] and v >= LOCATION_MATERIAL_SHARE),
        key=lambda k: (-shares[k], k),
    )
    if len(material) < 2:
        return material
    start = _endpoint_county_state(geocoder, first_centroid)
    end = _endpoint_county_state(geocoder, last_centroid)
    if start != end and start in material and end in material:
        return [start, end] + [k for k in material if k not in (start, end)]
    return material


def _multi_county_label(counties: list[tuple[str, str]], region_label: str) -> Optional[str]:
    """Level 2b: ``"A & B County, ST"`` (two US counties in one state),
    ``"A County, S1 & B County, S2"`` across states, or ``"County A & County B"``
    for non-US labels. Uses the top two counties. None when not formable."""
    if len(counties) < 2:
        return None
    (a, a_state), (b, b_state) = counties[0], counties[1]
    if a.endswith(" County") and b.endswith(" County"):
        if a_state and a_state == b_state and region_label:
            return f"{a[: -len(' County')]} & {b}, {region_label}"
        if a_state and b_state:
            a_ab = normalize_state_abbr(a_state) or a_state
            b_ab = normalize_state_abbr(b_state) or b_state
            return f"{a}, {a_ab} & {b}, {b_ab}"
        return None
    return f"{a} & {b}"


def _first_last_centroids(
    segments,
) -> tuple[
    Optional[tuple[float, float]],
    Optional[tuple[float, float]],
]:
    """Return (first_centroid, last_centroid) from an LF's segment list.

    A centroid is ``(lat, lon)``. Returns ``(None, None)`` if the
    segments list is empty or centroids are unavailable.
    """
    if not segments:
        return (None, None)

    def _centroid(seg) -> Optional[tuple[float, float]]:
        lat = getattr(seg, "gps_center_lat", None)
        lon = getattr(seg, "gps_center_lon", None)
        if lat is None or lon is None:
            return None
        if lat == 0.0 and lon == 0.0:
            return None
        return (float(lat), float(lon))

    first = _centroid(segments[0])
    last = _centroid(segments[-1])
    return (first, last)


def _haversine_km_simple(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points (Haversine)."""
    import math

    earth_radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return earth_radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _centroid_from_gps_points(points) -> Optional[tuple[float, float]]:
    """Mean (lat, lon) across a list of GPS point dicts/objects."""
    if not points:
        return None
    lats: list[float] = []
    lons: list[float] = []
    for p in points:
        if isinstance(p, dict):
            lat = p.get("lat")
            lon = p.get("lon") if "lon" in p else p.get("lng")
        else:
            lat = getattr(p, "lat", None)
            lon = getattr(p, "lon", None) or getattr(p, "lng", None)
        if lat is None or lon is None:
            continue
        try:
            lats.append(float(lat))
            lons.append(float(lon))
        except (TypeError, ValueError):
            continue
    if not lats:
        return None
    return (sum(lats) / len(lats), sum(lons) / len(lons))


# Endpoint window: fraction of the trace averaged at each end when deriving
# the start/end centroids. Small on purpose — the mean of half a route sits
# ~¼-route inland (or off-route entirely on a curved half), so a half-trace
# "start" centroid geocodes to a mid-route town while the true origin is
# never named.
_ENDPOINT_WINDOW_FRACTION = 0.05


def _first_last_centroids_from_gps(
    points,
) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
    """Return (head_centroid, tail_centroid) from filtered points — used by
    build_lf_title when raw GPS is supplied.

    Each endpoint is the mean of a small HEAD/TAIL window (first/last
    ``_ENDPOINT_WINDOW_FRACTION`` of fixes, min 1) rather than a half-trace
    centroid, so the geocoded town hugs the true origin/destination instead
    of a point ~¼-route inland.
    """
    if not points:
        return (None, None)
    n = len(points)
    if n == 1:
        c = _centroid_from_gps_points(points)
        return (c, c)
    win = max(1, int(n * _ENDPOINT_WINDOW_FRACTION))
    return (
        _centroid_from_gps_points(points[:win]),
        _centroid_from_gps_points(points[-win:]),
    )


def _build_multi_road_title(
    start_town: str,
    end_town: str,
    roads: list,
    state_abbr: str,
) -> Optional[str]:
    """Render the multi-road "via R1 + R2 + R3" format. Returns None
    when no road names survive ``select_significant_roads``."""
    from .gps_quality import select_significant_roads

    selected = select_significant_roads(roads or [])
    names: list[str] = []
    for r in selected:
        name = r.get("name") if isinstance(r, dict) else getattr(r, "name", None)
        if name:
            names.append(str(name))
    if not names:
        return None
    roads_str = " + ".join(names)
    return f"{start_town} to {end_town} via {roads_str}, {state_abbr}"


def _resolve_via_road(
    resolved_road: Optional[str],
    dom,
    per_lf_location: str,
) -> Optional[str]:
    """Resolve the via-road component for the title via the winning-county cascade.

    The via-road must follow the VOTED county, not the centroid. Prefer
    (1) an explicit ``resolved_road`` from the canonical ``ResolvedLocation``
    (the winning county's road), then (2) the winning-county road voted
    in-place from the dominant location (``dom``), falling back to (3) the
    legacy ``per_lf_location`` ``(road)`` centroid suffix only when no trace is
    available — and only when that suffix passes the same quality filter
    ``dominant_location`` applies, so a junk residential/track suffix road
    degrades to the no-road county shape instead of surfacing at Level 4.
    """
    road = resolved_road or None
    if road is None and dom is not None and dom.road:
        road = dom.road
    if road is None:
        # Residual: the ``per_lf_location`` ``(road)`` suffix is written by
        # ``derive_per_lf_locations`` from an UNFILTERED centroid geocode, so it
        # can carry the same junk (residential/track/business) road the quality filter drops at
        # the ``dominant_location`` pick — but the suffix bypassed that filter
        # and surfaced at Level 4 ("Colleton County via Blocker Run Road"). Gate
        # the suffix road through the SAME quality filter (name-only: the suffix
        # carries no OSM class, so only a route-ref / through-suffix / byway
        # name passes). A junk suffix road degrades to the no-road county shape.
        _suffix_road = _extract_road_from_suffix(per_lf_location)
        if _suffix_road and _road_passes_quality(_suffix_road):
            road = _suffix_road
    return road


# ── Specificity ladder (dead-GPS graceful degradation) ───────────────
# A label may only be as specific as its evidence. When locked GPS covers
# enough of the footage window we trust the full town→town+road claim
# (tier 1 = today's cascade, byte-for-byte unchanged); as coverage dies the
# label DEMOTES down the ladder rather than fabricating endpoints from sparse
# fixes:
#   1 town→town + road  — locked GPS ≥ LOCKED_COVERAGE_TIER1_MIN of the window
#   2 corridor + region — ≥2 corroborations (operator/OCR/bracket)
#   3 county / region   — 1 corroboration covering the window
#   4 state only        — bracket route alone
#   5 no location claim — anything less
LOCKED_COVERAGE_TIER1_MIN = 0.60


def award_tier(coverage: float, corroborations: int, *, has_bracket_route: bool = False) -> int:
    """Map evidence strength to a specificity tier (1 = most specific).

    ``coverage`` is the per-LF locked-GPS coverage fraction (0.0–1.0, from
    :func:`gomoveshift.pipeline.gps_quality.locked_gps_coverage`);
    ``corroborations`` is the COUNT of non-GPS :class:`Corroboration` items
    covering the window. Pure, deterministic demotion — same inputs, same tier:

    * ≥ ``LOCKED_COVERAGE_TIER1_MIN`` locked coverage → tier 1 (trust the
      town→town+road endpoints the GPS supports — today's behaviour);
    * else ≥2 corroborations → tier 2 (corridor + region);
    * else 1 corroboration → tier 3 (county / region);
    * else a bracket route alone → tier 4 (state only);
    * else → tier 5 (no location claim).
    """
    if coverage >= LOCKED_COVERAGE_TIER1_MIN:
        return 1
    if corroborations >= 2:
        return 2
    if corroborations >= 1:
        return 3
    if has_bracket_route:
        return 4
    return 5


def build_lf_title(
    segments,
    geocoder,
    per_lf_location: str,
    state_abbr: Optional[str] = None,
    per_lf_ost_text: Optional[str] = None,
    gps_points: Optional[list] = None,
    roads: Optional[list] = None,
    resolved_road: Optional[str] = None,
    resolved_places: Optional[list] = None,
    resolved_destination: Optional[str] = None,
    place_corrections: Optional[PlaceCorrections] = None,
    road_coverage_measured: bool = False,
) -> str:
    """Build the canonical endpoint-places + via-road title for one LF.

    ``road_coverage_measured``: a "via {road}" clause is burned only when
    the road's retained-time coverage was measured. No caller measures it yet,
    so every road is dropped and the place/region text stays; the LF is not
    refused. Waypost's final-text review accepts a cited road without checking
    coverage, so an unmeasured road would ship as an unproven claim.

    ``place_corrections``: the corrections-ledger bundle
    (``validation.place_corrections``) that substitutes a mapped wrong
    place name for its correction, or refuses (falls back to
    ``per_lf_location``) a bare denylisted name that survives. None
    (every existing call site) loads it from ``GMS_CORRECTIONS_PATH``; a
    missing/empty ledger is a strict no-op.
    """
    title = _build_lf_title_cascade(
        segments,
        geocoder,
        per_lf_location,
        state_abbr=state_abbr,
        per_lf_ost_text=per_lf_ost_text,
        gps_points=gps_points,
        roads=roads,
        resolved_road=resolved_road,
        resolved_places=resolved_places,
        resolved_destination=resolved_destination,
        road_coverage_measured=road_coverage_measured,
    )
    from .validation.place_corrections import (
        apply_corrections,
        find_denylisted,
        load_place_corrections,
    )

    corrections = place_corrections if place_corrections is not None else load_place_corrections()
    if not corrections.mapping and not corrections.denylist:
        return title
    corrected = apply_corrections(title, corrections)
    if find_denylisted(corrected, corrections):
        logger.warning(
            "build_lf_title: refused denylisted place name in %r; falling back to %r",
            corrected,
            per_lf_location,
        )
        return per_lf_location
    return corrected


def _build_lf_title_cascade(
    segments,
    geocoder,
    per_lf_location: str,
    state_abbr: Optional[str] = None,
    per_lf_ost_text: Optional[str] = None,
    gps_points: Optional[list] = None,
    roads: Optional[list] = None,
    resolved_road: Optional[str] = None,
    resolved_places: Optional[list] = None,
    resolved_destination: Optional[str] = None,
    road_coverage_measured: bool = False,
) -> str:
    """Build the canonical endpoint-places + via-road title for one LF.

    Args:
        segments: Iterable of LF segments; each must expose
            ``gps_center_lat`` / ``gps_center_lon``. The first segment's
            centroid becomes the start, the last segment's the end.
        geocoder: Object with ``reverse_geocode(lat, lon)`` returning an
            object with ``city`` / ``locality`` / ``state`` / ``country``
            attributes. May be None — the builder falls back gracefully.
        per_lf_location: The string produced by
            ``derive_per_lf_locations`` — carries the county/state base
            and optionally a ``(road)`` disambiguator.
        state_abbr: Optional pre-computed state abbreviation. When None,
            the builder parses the state from ``per_lf_location`` and
            normalizes via ``_US_STATE_ABBR``.
        resolved_road: Optional road from the canonical
            :class:`~gomoveshift.models.ResolvedLocation` — a road
            sampled INSIDE the voted (winning) county. When supplied it wins
            over the ``per_lf_location`` ``(road)`` suffix, which carries the
            centroid's road and can name the wrong county on a sub-50km
            crossing. When None and ``gps_points`` are present, the
            winning-county road is voted in-place.
        resolved_destination: Optional destination POI name from the
            canonical :class:`~gomoveshift.models.ResolvedLocation`
            (``destination_resolver.resolve_destination`` over the LF's
            montage-peak footage). When non-empty it ANCHORS the title
            ahead of the endpoint A→B cascade — ``"{destination}, {region}"``
            — so the title names what the highlights are about rather than
            the broad ambient region. ``None``/empty falls through to the
            A→B cascade unchanged; the far-road location invariant still reads the
            county from ``ResolvedLocation`` (not the title text), so a
            destination title never desyncs the cross-surface gate.
        per_lf_ost_text: Optional canonical OST text for this LF (the
            single source of truth, stored in
            ``state["per_lf_ost_text"]``). When provided and non-empty,
            the builder returns it verbatim (after length enforcement)
            and skips geocoding entirely — OST burn, YouTube title, and
            thumbnail all derive from the same string. When None or
            empty, the builder falls back to the legacy geocoding
            cascade below.

    Returns:
        A non-empty title string matching one of the fallback levels.
        Empty input produces an empty string — callers must handle that
        as "no title available" and use the per-LF location as-is.
    """
    # ── Single-source-of-truth short-circuit ───────────────────────
    # When the caller already has canonical OST text,
    # adopt it verbatim — no geocoder call, no cascade. OST = title =
    # thumbnail.
    if per_lf_ost_text:
        return _enforce_length_simple(per_lf_ost_text)

    if not per_lf_location:
        return ""

    county_part, state_part = _split_location_parts(per_lf_location)
    if state_abbr is None:
        state_abbr = normalize_state_abbr(state_part)

    # region label used in the trailing ", {region}" of every
    # cascade shape. US drives carry a 2-letter ``state_abbr``; non-US
    # drives (Ireland etc.) yield "" from ``normalize_state_abbr``, so we
    # use the trailing region (``state_part``) verbatim instead of letting
    # every "A to B" branch fall through to the Level 5 raw fallback. With
    # this, a GPS-derived non-US title reads "Killarney to Dingle via N86,
    # Kerry". US output is unchanged (region_label == state_abbr).
    region_label = state_abbr or state_part

    # Level 0 — destination anchor (ahead of the A→B cascade). When the
    # canonical ResolvedLocation named a destination POI enclosing the LF's
    # featured/montage-peak footage (a WMA / nature reserve / protected-area
    # boundary / named viewpoint / coastal feature, resolved once by
    # ``destination_resolver.resolve_destination``), name what the highlights
    # are ABOUT — "Donnelley Wildlife Management Area, SC" — instead of the
    # broad ambient region the GPS span happens to enclose ("ACE Basin"). The
    # county is still carried in the ResolvedLocation the far-road location
    # invariant consumes (it reads dominant_county/regions/vote, never the title text),
    # so a destination title never trips the cross-surface gate. When None we
    # fall through to the existing endpoint A→B cascade unchanged.
    _destination = (resolved_destination or "").strip()
    if _destination:
        if region_label and not _destination.endswith(f", {region_label}"):
            anchored = f"{_destination}, {region_label}"
        else:
            anchored = _destination
        return _enforce_length_simple(anchored)

    # The via-road must follow the VOTED county, not the centroid.
    # Vote the dominant location once when a trace is available — it carries
    # BOTH the winning-county road and the travel-ordered place sequence
    # that backfills a recognizable settlement when the per-endpoint
    # geocode misses. ``_resolve_via_road`` runs the road cascade.
    # the in-place vote honours the SAME geocode quorum floor every other
    # production caller passes (gps_enrichment, the shorts region vote, the
    # cross-surface gate) — a plurality off a single successfully-geocoded
    # sample is too weak to anchor the burned OST. Function-local import: a
    # top-level one would cycle (gps_enrichment imports this module before it
    # defines the constant), the pattern content_gates.py already uses.
    from .gps_enrichment import GEOCODE_QUORUM_MIN

    _dom = (
        dominant_location(gps_points, geocoder, min_votes=GEOCODE_QUORUM_MIN)
        if gps_points
        else None
    )
    road = _resolve_via_road(resolved_road, _dom, per_lf_location)

    # The canonical travel-ordered place sequence — prefer the explicit
    # ResolvedLocation.ordered_place_sequence, else the in-place vote's
    # ordered_places. Recognizable settlements here (Cashiers, Franklin, Lake
    # Toxaway, Fort Fremont) anchor the title instead of collapsing to the
    # bare county when the per-endpoint geocode below returns nothing.
    place_sequence = list(resolved_places or (getattr(_dom, "ordered_places", None) or []))

    segments_list = list(segments or [])

    # Footage-honest title shape: a best-moments / highlight LF is a
    # montage of SCATTERED, non-contiguous clips — there was no continuous drive
    # from the first segment's place to the last's. An "A to B via {road}" /
    # "A to B" / "Loop" title asserts exactly that continuity, so it would lie
    # about the footage. ``LFSegment.contiguous_with_previous`` is False on any
    # segment that does NOT immediately follow its predecessor in the source
    # timeline; a single such gap makes the LF non-contiguous. For such an LF we
    # suppress every endpoint-pair / Loop / via-road shape and let the cascade
    # fall through to the feature/region anchor ("{place}, {region}"), which
    # makes no continuity claim. A genuinely contiguous LF (the
    # full_drive single-LF / contiguous path — every segment
    # contiguous_with_previous, or legacy segments lacking the attribute) keeps
    # its A→B / route / place-region shape. ``resolved_destination`` (Level 0
    # above) still wins outright when present, before this point is reached.
    _noncontiguous = any(
        not getattr(s, "contiguous_with_previous", True) for s in segments_list[1:]
    )

    # When raw GPS points are supplied, filter them first and
    # derive start/end centroids from the clean trace. Keeps noisy or
    # stationary fixes out of the centroid reducer, which otherwise
    # would drag endpoints into neighboring towns. When gps_points is
    # absent we fall back to the segment-centroid path (backward-compat).
    first_centroid: Optional[tuple[float, float]] = None
    last_centroid: Optional[tuple[float, float]] = None
    _gps_filtered: list = []
    if gps_points:
        from .gps_quality import filter_high_quality_points

        _gps_filtered = filter_high_quality_points(gps_points)
        first_centroid, last_centroid = _first_last_centroids_from_gps(_gps_filtered)
    if first_centroid is None and last_centroid is None:
        first_centroid, last_centroid = _first_last_centroids(segments_list)

    start_town = _geocode_town(geocoder, *first_centroid) if first_centroid else None
    end_town = _geocode_town(geocoder, *last_centroid) if last_centroid else None

    # A COUNTY name is not a settlement. ``_geocode_town`` returns
    # the geocoder's ``.name``/``.locality``, which at a rural centroid with no
    # town is the COUNTY itself ("Beaufort County"). Left as-is it makes
    # ``start_town``/``end_town`` non-None, which (a) suppresses the
    # settlement backfill (it only fills a None endpoint) and (b) wins the
    # Level-4 ``_single_town`` anchor over a real settlement in the sequence —
    # so "Fort Fremont" lost to "Beaufort County". Null a county-shaped endpoint
    # here so the recognizable-settlement ladder fires instead. A broad REGION
    # name ("ACE Basin") is the same class of non-settlement fallback and is
    # demoted symmetrically.
    if start_town and (
        _looks_like_county(start_town)
        or _looks_like_township(start_town)
        or _looks_like_region(start_town)
    ):
        start_town = None
    if end_town and (
        _looks_like_county(end_town)
        or _looks_like_township(end_town)
        or _looks_like_region(end_town)
    ):
        end_town = None

    # Via-road span guard: a single local "via {road}" can only describe the
    # drive when the endpoints are reasonably close to each other (and thus to
    # that road). On a long crossing the dominant road is far from one endpoint,
    # so suppress every via-road shape and fall through to the no-road cascade.
    _endpoints_far = (
        first_centroid is not None
        and last_centroid is not None
        and _haversine_km_simple(
            first_centroid[0], first_centroid[1], last_centroid[0], last_centroid[1]
        )
        > _VIA_ROAD_MAX_SPAN_KM
    )
    # the span guard targets LOCAL roads only. A road carrying a route
    # number (interstate/US/state highway — ``_ROUTE_REF_RE``) is a numbered
    # through-route engineered to connect distant places, so it honestly
    # describes a long crossing (e.g. a drive between two distant towns
    # stayed on I 81 for 100 km). Exempt it from the span guard so the dominant
    # interstate survives; a named-but-unnumbered road ("Sea Island Parkway")
    # is NOT a route-ref and is still span-guarded.
    _road_is_route_ref = bool(road and _ROUTE_REF_RE.search(road))
    _span_suppresses_road = _endpoints_far and not _road_is_route_ref
    # On a long crossing (>50 km between endpoints) the dominant LOCAL
    # road can't describe the drive in ANY shape — drop it outright so it can't
    # reattach in the single-endpoint shapes (Level 1 "A to B via road" AND the
    # Level 4 county fallback, whose via-road arrives via the per_lf_location
    # suffix and the per-level ``not _span_suppresses_road`` guards therefore miss).
    # Mirrors route_profile.title()'s road=None guard. Short same-area drives
    # (≤50 km) keep their local road.
    # A non-contiguous (best-moments) LF never traversed a single road
    # end-to-end either, so a "via {road}" clause is just as dishonest as the
    # A→B claim — drop the road for the same reason the long-crossing guard does,
    # so the Level-4 anchor degrades to the clean "{place}, {region}" rather
    # than "{place} via {road}". A route-ref survives the span guard but never
    # the non-contiguous guard (a montage crossed no single road end-to-end).
    if _span_suppresses_road or _noncontiguous:
        road = None
    # No measured retained-time coverage for the road, so no road claim in
    # any shape (single, multi or suffix road).
    if not road_coverage_measured:
        road = None
        roads = None

    # Remember whether each endpoint resolved from its OWN centroid
    # geocode BEFORE the flake guard runs. A backfill below only fills an
    # endpoint that was missing here (a geocode miss) — never one the flake
    # guard intentionally clears (that would re-fabricate the bogus route the
    # guard exists to kill). The guard only runs when BOTH are present, so a
    # missing endpoint here is always a genuine miss, not a demotion.
    _start_present_pre = start_town is not None
    _end_present_pre = end_town is not None

    # Cross-county/cross-state flake guard: drop a fabricated endpoint
    # before it reaches the "A to B" cascade. Uses the on-route midpoint
    # (a real GPS fix, not the mean) to decide which side is real. On
    # demotion the title degrades to the single-location cascade below,
    # whose county base is the dominant county from per_lf_location.
    if start_town and end_town and start_town != end_town:
        # Vote the midpoint across the CENTRAL THIRD of the trace, not a
        # single sample — one flaky mid geocode must not fail the demotion
        # open. Fall back to the endpoint-mean when no trace is available.
        mid_coords: list[tuple[float, float]] = []
        if _gps_filtered:
            for p in _central_window(_gps_filtered):
                ll = _point_latlon(p)
                if ll is not None:
                    mid_coords.append(ll)
        if not mid_coords and first_centroid and last_centroid:
            mid_coords = [
                (
                    (first_centroid[0] + last_centroid[0]) / 2,
                    (first_centroid[1] + last_centroid[1]) / 2,
                )
            ]
        keep_start, keep_end = _validate_route_endpoints(
            geocoder,
            first_centroid,
            last_centroid,
            mid_coords,
            county_shares=_county_shares(_dom),
        )
        if not keep_end:
            end_town = None
        elif not keep_start:
            start_town = None

    # Recognizable-place backfill. When a per-endpoint geocode missed
    # (the endpoint was None BEFORE the flake guard — not demotion-cleared),
    # anchor it on the travel-ordered place sequence's recognizable
    # settlements instead of letting the title collapse to the bare county.
    # This is what surfaces "Lake Toxaway to Cashiers" / "Cashiers to Franklin"
    # where the endpoint centroid geocode returned no town but the vote's
    # sequence carried the towns. ``_settlement_anchor`` (single recognizable
    # place) additionally replaces the county base in the Level-4 fallback so a
    # one-settlement LF reads "Fort Fremont, SC", not the county.
    _settlements = _recognizable_settlements(place_sequence, county_part)
    if _settlements:
        if start_town is None and not _start_present_pre:
            start_town = _settlements[0]
        if (
            end_town is None
            and not _end_present_pre
            and _settlements[-1] != start_town
            and len(_settlements) >= 2
        ):
            end_town = _settlements[-1]
    # the Level-4 anchor prefers the rank-aware winning-county pick
    # (``_dom.name`` — towns beat hamlets) when that pick is itself one of the
    # sequence's recognizable settlements; otherwise (plain-string sequence,
    # no dom, or a pick filtered out above) travel-order-first stands.
    # The endpoint backfills above stay travel-ordered on purpose — they
    # frame a crossing, not a single anchor.
    _settlement_anchor = _settlements[0] if _settlements else None
    _dom_name = (getattr(_dom, "name", None) or "").strip()
    if _dom_name and _dom_name in _settlements:
        _settlement_anchor = _dom_name

    # Shape A (cross-state): when the two endpoints geocode to DIFFERENT
    # states, the single dominant-county ``region_label`` in Levels 1/2 would
    # stamp the minority endpoint with the wrong state ("TownInSC to TownInNC,
    # NC"). Mirror the episode ladder's Shape-A cross-state form
    # (``ost_formatter.format_location_ladder``): each town carries its OWN
    # state — "{start}, {S1} to {end}, {S2}[ via {road}]". Fires only when
    # both endpoints resolved and both states are known strings; otherwise
    # the ordinary cascade below applies unchanged. Skipped for a
    # non-contiguous (best-moments) LF like every other A→B claim. The
    # geocodes are cache hits — the flake guard above already resolved both
    # endpoint (county, state) pairs.
    if start_town and end_town and start_town != end_town and not _noncontiguous:
        _s_state = _endpoint_county_state(geocoder, first_centroid)[1]
        _e_state = _endpoint_county_state(geocoder, last_centroid)[1]
        if (
            isinstance(_s_state, str)
            and isinstance(_e_state, str)
            and _s_state.strip()
            and _e_state.strip()
            and _s_state != _e_state
        ):
            _s_ab = normalize_state_abbr(_s_state) or _s_state
            _e_ab = normalize_state_abbr(_e_state) or _e_state
            cross = f"{start_town}, {_s_ab} to {end_town}, {_e_ab}"
            if road and not _span_suppresses_road:
                with_road = f"{cross} via {road}"
                if len(with_road) <= MAX_TITLE_LEN:
                    return with_road
            return _enforce_length(cross, per_lf_location, region_label)

    # ── Fallback cascade ────────────────────────────────────────────
    # Level 1a: start to end via MULTIPLE roads — only when
    # ``roads`` is supplied and selection yields ≥1 name. Falls back
    # to the single-road format below when the multi-road title would
    # exceed MAX_TITLE_LEN.
    if (
        start_town
        and end_town
        and start_town != end_town
        and roads
        and region_label
        and not _endpoints_far
        and not _noncontiguous
    ):
        multi = _build_multi_road_title(start_town, end_town, roads, region_label)
        if multi and len(multi) <= MAX_TITLE_LEN:
            return multi
        # Multi-road format too long → degrade to single-road below.

    # Level 1: start to end via road (canonical)
    if (
        start_town
        and end_town
        and start_town != end_town
        and road
        and region_label
        and not _span_suppresses_road
        and not _noncontiguous
    ):
        title = f"{start_town} to {end_town} via {road}, {region_label}"
        return _enforce_length(title, per_lf_location, region_label)

    # Level 2: start to end (no road)
    if start_town and end_town and start_town != end_town and region_label and not _noncontiguous:
        title = f"{start_town} to {end_town}, {region_label}"
        return _enforce_length(title, per_lf_location, region_label)

    # Level 3: Loop via road (start == end AND GPS confirms circular route).
    # Requires GPS centroids to be within 5 km of each other — same-town geocoding
    # without actual circular GPS is not sufficient, because a linear drive through
    # a small town also produces start_town == end_town and must NOT get "Loop".
    # Confirm circular route by centroid proximity.  Two different thresholds:
    # • GPS trace available (1 km): GPS must show the drive returned close to
    #   its origin — tighter because GPS-poor coverage can make a linear drive
    #   appear local even when it isn't.
    # • No GPS (0.1 km): only accept literally co-located segment endpoints
    #   (same GPS coordinate).  Same-town geocoding alone is NOT sufficient.
    _loop_dist_km = 1.0 if bool(_gps_filtered) else 0.1
    _gps_circular = (
        first_centroid is not None
        and last_centroid is not None
        and _haversine_km_simple(
            first_centroid[0], first_centroid[1], last_centroid[0], last_centroid[1]
        )
        < _loop_dist_km
    )
    if (
        start_town
        and end_town
        and start_town == end_town
        and road
        and region_label
        and _gps_circular
        and not _noncontiguous
    ):
        title = f"{start_town} Loop via {road}, {region_label}"
        return _enforce_length(title, per_lf_location, region_label)

    # Level 3b: Loop (start == end, no road, GPS confirms circular)
    if (
        start_town
        and end_town
        and start_town == end_town
        and region_label
        and _gps_circular
        and not _noncontiguous
    ):
        # "Loop" is factual (GPS-confirmed circular); drop the "Scenic" genre word.
        title = f"{start_town} Loop, {region_label}"
        return _enforce_length(title, per_lf_location, region_label)

    # Level 2b (placed after the loop levels so a confirmed loop wins): two counties hold material footage but an endpoint town is
    # still unresolved after the backfills — name both counties rather than
    # collapsing to the one the drive spent longer in.
    if _dom is not None and not (start_town and end_town):
        _multi = _multi_county_label(
            _material_counties(_dom, geocoder, first_centroid, last_centroid), region_label
        )
        if _multi:
            return _enforce_length(_multi, per_lf_location, region_label)

    # Level 4: single town or county fallback.
    # Strip road names that imply circular routing (Loop, Scenic Loop) when
    # GPS does not confirm a circular route — using them here would mislead
    # viewers into thinking the drive is a loop when it is not.
    if (county_part or _settlement_anchor) and region_label:
        _loop_re = re.compile(r"\s+Scenic\s+Loop\b|\s+\bLoop\b", re.IGNORECASE)
        # A confirmed loop keeps "Loop" but never the "Scenic" genre word.
        clean_county = (
            (
                re.sub(_loop_re, "", county_part).strip().rstrip(",").strip()
                if not _gps_circular
                else re.sub(r"\s+Scenic(?=\s+Loop\b)", "", county_part, flags=re.IGNORECASE)
            )
            if county_part
            else ""
        )
        # Anchor a single-location LF on a recognizable settlement, not
        # the bare county ("Fort Fremont, SC" > "Beaufort County, SC").
        # Priority: (1) a town that actually resolved at the
        # endpoints (a linear drive through one town, or the survivor of a
        # flake demotion) — never overridden by the sequence; (2) the sequence's
        # first recognizable settlement (the geocode-miss case); (3) the bare
        # county fallback.
        _single_town = (
            start_town if (start_town and (end_town is None or end_town == start_town)) else None
        )
        base = _single_town or _settlement_anchor or clean_county
        clean_road = road if (road and not re.search(r"\bLoop\b", road, re.IGNORECASE)) else None
        if clean_road:
            title = f"{base} via {clean_road}, {region_label}"
        else:
            # Route-only: place + region, no "Scenic Drive" genre claim.
            title = f"{base}, {region_label}"
        return _enforce_length(title, per_lf_location, region_label)

    # Level 5: total fallback — return the per-LF location verbatim
    return per_lf_location


def _drop_trailing_clauses(text: str, budget: int) -> str:
    """Fit ``text`` to ``budget`` chars, region-preserving.

    Drops whole trailing comma-separated clauses (", County Kerry") one at a
    time — keeping the distinguishing leading clause (the town) — before any
    character-level cut: "Kenmare, County Kerry" over budget degrades to
    "Kenmare", never "Kenmare, Count". Char-chops only as an absolute last
    resort (a single leading clause still over budget).
    """
    if len(text) <= budget:
        return text
    parts = [p.strip() for p in text.split(",")]
    while len(parts) > 1 and len(", ".join(parts)) > budget:
        parts = parts[:-1]
    joined = ", ".join(parts)
    return joined if len(joined) <= budget else joined[:budget]


def _enforce_length(title: str, fallback: str, state_abbr: str) -> str:
    """Ensure title is ≤ MAX_TITLE_LEN; degrade to a shorter form if not."""
    if len(title) <= MAX_TITLE_LEN:
        return title
    # Try trimming: drop the road segment if present
    shorter = re.sub(r"\s+via\s+[^,]+,", ",", title, count=1)
    if len(shorter) <= MAX_TITLE_LEN:
        return shorter
    # Final fallback — per-LF location with state, degraded region-preservingly
    # (drop ", County X" before any char-chop) so the town survives intact.
    if state_abbr:
        budget = MAX_TITLE_LEN - len(state_abbr) - 2
        return f"{_drop_trailing_clauses(fallback, budget)}, {state_abbr}"
    return _drop_trailing_clauses(fallback, MAX_TITLE_LEN)


def _enforce_length_simple(title: str) -> str:
    """Ensure title is ≤ MAX_TITLE_LEN without fallback context.

    Used by the per_lf_ost_text short-circuit where the caller has
    already committed to a canonical string; we have no county/state
    fallback to degrade to. First try the ``via {road},`` trim; then
    hard-truncate.
    """
    if len(title) <= MAX_TITLE_LEN:
        return title
    # Try trimming: drop the road segment if present
    shorter = re.sub(r"\s+via\s+[^,]+,", ",", title, count=1)
    if len(shorter) <= MAX_TITLE_LEN:
        return shorter
    return title[:MAX_TITLE_LEN]


_GENERIC_ROAD_SUFFIX_RE = re.compile(
    r"\s+(?:Road|Rd|Drive|Dr|Street|St|Lane|Ln|Court|Ct|Way|Avenue|Ave|"
    r"Boulevard|Blvd|Highway|Hwy|Parkway|Pkwy|Trail|Trl|Pike)\.?$",
    re.IGNORECASE,
)


def _feature_stem(road: str) -> str:
    """Strip a single generic road-type suffix so a road name reads as a place
    feature: ``"Blocker Run Road" -> "Blocker Run"``, ``"Donnelley Drive" ->
    "Donnelley"``, ``"Stocks Creek Road" -> "Stocks Creek"``. Idempotent; leaves
    a name with no generic suffix unchanged."""
    return _GENERIC_ROAD_SUFFIX_RE.sub("", road.strip()).strip()


def ranked_lf_features(points: Any, geocoder, *, max_samples: int = 25) -> list[tuple[str, int]]:
    """Footage-weighted, descending-ranked list of an LF's named geographic
    features (``(feature_stem, weight)``) for deterministic sibling
    differentiation.

    Reuses the same vote idiom as :func:`dominant_location` (sample the
    quality-filtered trace, reverse-geocode, weight by footage-span catchment).
    Drops possessive/business junk (``Dickie's`` via the possessive guard +
    ``is_valid_chapter_placename``) and bare numerics. The caller assigns a
    DISTINCT feature per colliding LF deterministically — this function never
    chooses; it only ranks. Empty when no geocoder / no usable road.
    """
    if geocoder is None:
        return []
    pts = list(points or [])
    if not pts:
        return []
    try:
        from .gps_quality import filter_high_quality_points

        filtered = filter_high_quality_points(pts)
        if filtered:
            pts = filtered
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.debug("ranked_lf_features: quality filter skipped: %s", exc)

    coords: list[tuple[float, float]] = []
    for p in pts:
        ll = _point_latlon(p)
        if ll is None or (ll[0] == 0.0 and ll[1] == 0.0):
            continue
        coords.append(ll)
    if not coords:
        return []

    from .chapters._shared import is_valid_chapter_placename

    votes: Counter = Counter()
    for (lat, lon), weight in _evenly_sample_weighted(coords, max_samples):
        try:
            detailed = geocoder.reverse_geocode_detailed(lat, lon) or {}
        except Exception as exc:  # noqa: BLE001 — log + skip
            logger.debug("ranked_lf_features: reverse_geocode_detailed failed: %s", exc)
            continue
        road = detailed.get("road")
        if not isinstance(road, str) or not road.strip():
            continue
        stem = _feature_stem(road)
        if not stem or re.fullmatch(r"[\w.]+'s", stem):  # possessive business
            continue
        if not re.search(r"[A-Za-z]", stem):  # bare numeric / symbol
            continue
        if not is_valid_chapter_placename(stem):
            continue
        votes[stem] += weight
    # Deterministic order: descending weight, then alphabetical (NEVER sample
    # order, which would track drive direction).
    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))


def _insert_feature(base: str, feature: str) -> str:
    """Insert ``feature`` into a ``"{place}, {ST}"`` label as
    ``"{place} — {feature}, {ST}"``; if the base has no trailing region the
    feature is appended as ``"{base} — {feature}"``."""
    base = base.strip()
    if ", " in base:
        place, _, region = base.rpartition(", ")
        return f"{place} — {feature}, {region}"
    return f"{base} — {feature}"


def differentiate_sibling_labels(
    labels: dict[int, str], features_by_lf: dict[int, list[tuple[str, int]]]
) -> dict[int, str]:
    """DETERMINISTIC sibling-LF differentiation (replaces the ``(Part N of M)``
    filler for the no-settlement collision case — the ACE-Basin class).

    When ≥2 LFs share an identical label (e.g. four ``"ACE Basin, SC"`` rural
    backroad LFs with no distinct town), differentiate each by its **dominant**
    (#1, modal) named geographic feature instead of a part-number. Fully
    deterministic: each LF contributes ONLY its top-ranked feature, the strongest
    claim wins each feature (sorted ``(-weight, lf_index, feature)``), and an LF
    whose dominant feature is already claimed by a stronger sibling — i.e. the
    two are the SAME road — is LEFT UNCHANGED on the clean region base rather
    than forced onto a weak 2nd-choice fallback that would manufacture a
    distinction the geography does not support. Unchanged LFs flow to
    :func:`ensure_unique_titles`'s ``(Part N)`` last resort. No human/LLM
    judgment: same input → same output.

    ``features_by_lf`` maps ``lf_index -> ranked [(feature, weight), ...]`` (see
    :func:`ranked_lf_features`). LFs absent from it (no GPS) are never touched.
    """
    if not labels:
        return {}
    groups: dict[str, list[int]] = {}
    for idx, label in labels.items():
        groups.setdefault(label, []).append(idx)

    result = dict(labels)
    for label, members in groups.items():
        if len(members) < 2:
            continue
        # DOMINANT-ONLY: each LF contributes ONLY its #1 (modal) feature — never
        # a weak 2nd choice. When two siblings share the same dominant feature
        # (they are the SAME road — e.g. two LFs on one named backroad), the
        # stronger claim wins it and the other gets NO feature (it keeps the
        # clean region base), rather than being forced onto a junk fallback like
        # a bare "Savage" track to manufacture a distinction the geography does
        # not support. Rural-backroad features are often all OSM
        # ``type=track`` so a class filter can't separate good from junk — the
        # honest signal is "is this the LF's dominant feature", not the road
        # class.
        triples: list[tuple[int, int, str]] = []  # (-weight, lf_index, feature)
        for lf_idx in members:
            ranked = features_by_lf.get(lf_idx, []) or []
            if ranked:
                feature, weight = ranked[0]
                triples.append((-weight, lf_idx, feature))
        triples.sort(key=lambda t: (t[0], t[1], t[2]))
        assigned: dict[int, str] = {}
        claimed: set[str] = set()
        for _negw, lf_idx, feature in triples:
            if lf_idx in assigned or feature in claimed:
                continue
            assigned[lf_idx] = feature
            claimed.add(feature)
        for lf_idx, feature in assigned.items():
            result[lf_idx] = _insert_feature(label, feature)
    return result


def build_superlong_part_title(
    base_label: str,
    part_index: int,
    part_count: int,
    trim_ratio: Optional[float] = None,
) -> str:
    """Build one superlong part's title: a trim-honest verb plus a
    DELIBERATE ``— Part N of M`` suffix driven by the day's
    ``SuperlongPartPlan`` (``part_index``/``part_count``).

    Distinct from :func:`ensure_unique_titles`'s dedup-only
    ``(Part N of M)`` suffix, which fires only on incidental title
    collisions — see ``_SUPERLONG_PART_SUFFIX_RE``, which that function
    checks so it never stacks its own suffix on top of this one.

    Verb: superlong content is the whole drive day's moving footage with
    only stops removed — never a curated highlight edit — so it earns the
    SAME ``trim_ratio`` honesty rule regular ``full_drive`` LFs use
    (``longform_metadata._select_verb``, reused rather than re-derived).
    ``trim_ratio=None`` (unmeasured) bands to "Driving", matching that
    function's existing default.

    Burned OST text never carries this suffix or verb (metadata-only);
    idempotent over a ``base_label`` that already carries either.
    """
    from .longform_metadata import _select_verb, _strip_leading_verb

    label = _strip_leading_verb((base_label or "").strip())
    title = f"{_select_verb(trim_ratio)} {label}".strip()
    if part_count > 1 and not _SUPERLONG_PART_SUFFIX_RE.search(title):
        title = f"{title} — Part {part_index} of {part_count}"
    return title


def append_full_drive_part_suffix(
    title: str,
    part_index: Optional[int],
    part_count: Optional[int],
    max_len: int = 100,
) -> str:
    """Append the superlong ``— Part N of M`` suffix to a split ``full_drive``
    LF's metadata title (same convention as :func:`build_superlong_part_title`).

    No-op when ``part_count`` is ``None``/``<= 1`` (an unsplit day) or the
    title already carries the suffix. The suffix is kept whole: the base is
    truncated on a word boundary to fit ``max_len`` (YouTube's title limit).
    """
    if not part_count or part_count <= 1 or part_index is None:
        return title
    if _SUPERLONG_PART_SUFFIX_RE.search(title):
        return title
    suffix = f" — Part {part_index} of {part_count}"
    base = title.rstrip()
    budget = max_len - len(suffix)
    if budget > 0 and len(base) > budget:
        cut = base[:budget]
        sp = cut.rfind(" ")
        base = (cut[:sp] if sp > 0 else cut).rstrip(" ,—-")
    return f"{base}{suffix}"


def ensure_unique_titles(
    lf_titles: dict[int, str],
    published_titles: set[str] | None = None,
    published_title_ids: dict[str, str] | None = None,
) -> dict[int, str]:
    """Post-process: append ``(Part N)`` to any duplicate titles.

    Input: ``{lf_index: title}``. Output: same dict shape, with
    collisions disambiguated by appending ``(Part N of M)`` where N is
    occurrence order and M is total colliding titles.

    Every collision check runs in the FINAL-title namespace. Inputs
    are pre-reshape labels ("A to B via US-221, NC") but the channel
    ledger (``longform_metadata.load_published_titles``) holds assembled
    final titles ("A to B, NC in 4K"), and the via-stripping reshape
    (``longform_metadata.assemble_final_title``) runs AFTER this dedup —
    so both the in-batch count and the channel comparison key on each
    label's assembled form, while the RETURNED values stay labels (plus
    any ``(Part …)`` suffix, which the reshape detaches and re-appends).
    Comparing raw labels missed (a) a repeat drive of a published
    corridor and (b) two same-day labels differing only in via-road that
    collapse to one identical final title after via-stripping.

    ``published_titles`` (optional): a label whose assembled form matches
    — case-insensitively, trimmed — an already-published title is
    disambiguated even when unique in-batch: a trailing ``(Part N)`` is
    bumped until the assembled form collides with neither a published
    channel title nor another title emitted this batch. When omitted the
    behaviour is in-batch dedup only.

    A label unique in-batch (no in-batch duplicate, no superlong
    suffix) whose UNSUFFIXED assembled form exactly matches a published
    title is a hard failure, not a ``(Part N)`` rename of a genuine
    duplicate — the bump stays for near-misses (in-batch dedup or
    already-suffixed forms that also collide). ``published_title_ids``
    (optional ``{title: video_id}``) names the live video in the error;
    absent an entry, the local live-catalog snapshot does (read only on
    this refusal path, never the network).

    Titles that are already unique (in-batch and vs the channel) are
    returned unchanged.
    """
    if not lf_titles:
        return {}
    # Count occurrences
    from collections import Counter

    from .longform_metadata import assemble_final_title

    _final_cache: dict[str, str] = {}

    def _final(label: str) -> str:
        """Normalised assembled-final-title form of a label, for comparison.

        The deliberate ``— Part N of M`` suffix is part of a title's
        IDENTITY, not decoration: two parts of one superlong day are
        different videos with different titles, never a collision. But
        ``assemble_final_title`` strips series suffixes, so it must be
        re-attached here — otherwise every part of a day collapses to one
        comparison key and the residual-collision bump below stacks a
        second ``(Part N)`` onto a title that already carries one.
        """
        if label not in _final_cache:
            base = assemble_final_title(label).strip().lower()
            match = _SUPERLONG_PART_SUFFIX_RE.search(label)
            _final_cache[label] = f"{base}{match.group(0).rstrip().lower()}" if match else base
        return _final_cache[label]

    # Normalised set of titles already live on the channel (final namespace).
    published = {t.strip().lower() for t in (published_titles or set()) if t and t.strip()}

    counts = Counter(_final(t) for t in lf_titles.values())
    collides_channel = any(_final(t) in published for t in lf_titles.values())
    if all(c == 1 for c in counts.values()) and not collides_channel:
        return dict(lf_titles)

    # Per-final-form occurrence counter
    seen: dict[str, int] = {}
    used: set[str] = set()  # normalised final forms already emitted this batch
    result: dict[int, str] = {}
    for lf_idx in sorted(lf_titles.keys()):
        title = lf_titles[lf_idx]
        total = counts[_final(title)]
        is_superlong = bool(_SUPERLONG_PART_SUFFIX_RE.search(title))
        if total > 1 and not is_superlong:
            seen[_final(title)] = seen.get(_final(title), 0) + 1
            candidate = f"{title} (Part {seen[_final(title)]} of {total})"
        else:
            # A title carrying the deliberate superlong ``— Part N of M``
            # suffix (build_superlong_part_title) is already disambiguated;
            # the while-loop bump below still guards a genuine residual
            # collision (e.g. against a published channel title) without
            # stacking a second "(Part N of M)" on top of this one.
            candidate = title
            if total == 1 and not is_superlong and _final(candidate) in published:
                # No in-batch/superlong suffix at all — bare form
                # already, exactly, live on the channel. Refuse rather than
                # rename a genuine duplicate before the OST burn.
                from ..cli.handlers.pub_status import LIVE_CATALOG_SNAPSHOT
                from .exceptions import PhaseBCExitContractError
                from .longform_metadata import live_catalog_title_ids

                video_id = (published_title_ids or {}).get(_final(candidate)) or (
                    live_catalog_title_ids().get(_final(candidate))
                )
                where = "" if video_id else f" (not in the local {LIVE_CATALOG_SNAPSHOT})"
                raise PhaseBCExitContractError(
                    f"Title {candidate!r} (LF {lf_idx}) exactly duplicates "
                    f"live video id={video_id or 'unknown'}{where} already published "
                    "under this title — refusing to auto-suffix a duplicate upload; "
                    "fix the LF label or confirm the live video is superseded."
                )
        # Bump a trailing "(Part N)" until the assembled form collides with
        # neither a published channel title nor another title in this batch.
        part = 1
        while _final(candidate) in published or _final(candidate) in used:
            part += 1
            candidate = f"{title} (Part {part})"
        used.add(_final(candidate))
        result[lf_idx] = candidate
    return result
