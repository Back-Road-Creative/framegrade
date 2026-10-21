"""Place-shape and midpoint-vote helpers shared by the OST formatter and the title builder.

These are pure geometry/string helpers over a geocoder's answers: whether an
endpoint "town" is really a county, township or broad region, and the modal
``(county, state)`` across the central samples of a trace. They live in the
engine because the OST formatter (``ingest/ost_formatter.py``) needs them;
``title_builder`` (publish-side) imports them from here, so the dependency runs
publish -> engine and never the other way.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Optional

logger = logging.getLogger("gomoveshift.pipeline.title_builder")


def _looks_like_county(town: Optional[str]) -> bool:
    """True when an endpoint ``town`` is actually a COUNTY name, not a settlement.

    ``_geocode_town`` reads the geocoder's ``.name``/``.locality``, which at a
    rural centroid with no town is the county itself ("Beaufort County" /
    "County Kerry"). Such a value must NOT count as a resolved settlement: it would block the settlement backfill and win the Level-4 anchor
    over a real place. US and Irish counties both carry the ``County`` token, so
    a substring test suffices — and crucially does NOT flag a real town that
    merely happens to equal the LF's location base (e.g. "Mars Hill").
    """
    return bool(town) and "county" in town.strip().lower()


def _looks_like_township(town: Optional[str]) -> bool:
    """True when an endpoint is a US administrative TOWNSHIP ("Cumberland
    Township"), not a settlement a viewer would search.

    Like a county, a township is the rural fallback the geocoder returns when no
    town/village resolves — it must not win the title over a real settlement in
    the trace's place sequence. A drive near Gettysburg could otherwise be
    labelled "Cumberland Township" even though Gettysburg is in the sequence;
    nulling the township endpoint lets the recognizable-settlement backfill
    surface "Gettysburg". Ireland has no townships, so this is US-only by construction
    and cannot demote a legitimate Irish hamlet endpoint.
    """
    return bool(town) and bool(re.search(r"\bTownship\b", town, re.IGNORECASE))


# Broad-area names that denote something LARGER than a settlement — a watershed,
# ecological region, or protected reserve. The geocoder hands one of these back
# at a rural centroid inside such an area (e.g. drives through Donnelley WMA
# resolved to "ACE Basin"); like a county or township it must NOT fill the town
# slot or it wins the anchor over the real settlement / destination. Word-boundary
# matched and case-insensitive so it demotes "ACE Basin" / "ACE Basin, SC" but
# leaves ordinary towns ("Beaufort", "Walterboro", "Preston") untouched. Kept
# deliberately conservative — extend the token set, never loosen to substrings.
_REGION_TOKENS_RE = re.compile(
    r"\b("
    r"Basin"
    r"|Lowcountry"
    r"|Watershed"
    r"|Region"
    r"|Wildlife Management Area"
    r"|Wildlife Refuge"
    r"|National Forest"
    r"|State Forest"
    r"|National Park"
    r"|State Park"
    r"|Conservation Area"
    r"|Nature Reserve"
    r"|Preserve"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_region(town: Optional[str]) -> bool:
    """True when an endpoint ``town`` names a broad AREA (basin/watershed/reserve),
    not a settlement a viewer would search.

    Sibling of ``_looks_like_county`` / ``_looks_like_township``: the geocoder
    returns the encompassing region at a rural centroid with no town, and that
    value must not count as a resolved settlement or it blocks the settlement backfill
    and wins the Level-4 anchor. A drive through a wildlife management area
    would otherwise get the ambient basin name (for example "ACE Basin") in the
    town slot for exactly this reason. The
    token set is curated and word-boundary matched so it cannot demote a real
    town whose name merely contains one of these as a substring.
    """
    return bool(town) and bool(_REGION_TOKENS_RE.search(town.strip()))


def _evenly_sample(seq: list, k: int) -> list:
    """Return at most ``k`` items spread evenly across ``seq`` (preserves
    order; endpoints always included when ``k >= 2``)."""
    n = len(seq)
    if k <= 0 or n <= k:
        return list(seq)
    if k == 1:
        # Single sample: take the median element (avoids a div-by-zero in the
        # ``step = (n - 1) / (k - 1)`` spread below when ``k == 1`` and ``n > 1``).
        return [seq[n // 2]]
    step = (n - 1) / (k - 1)
    return [seq[round(i * step)] for i in range(k)]


def _endpoint_county_state(geocoder, centroid) -> tuple[Optional[str], Optional[str]]:
    """Reverse-geocode ``centroid`` to ``(county, state)``; ``(None, None)``
    on missing input or geocoder failure."""
    if geocoder is None or not centroid:
        return (None, None)
    try:
        detailed = geocoder.reverse_geocode_detailed(*centroid) or {}
    except Exception as exc:  # noqa: BLE001 — log + treat as unknown
        logger.debug("title_builder._endpoint_county_state failed: %s", exc)
        return (None, None)
    return (detailed.get("county"), detailed.get("state"))


_MID_VOTE_SAMPLES = 5


def _central_window(seq: list, width: int = _MID_VOTE_SAMPLES) -> list:
    """Return up to ``width`` items from ``seq`` centred symmetrically on the
    midpoint. Centring (not a `[n//3:2n//3]` slice) keeps the window from
    leaning toward the start endpoint on an even split, so a genuine
    mid-route crossing still votes for the area its true centre falls in.
    """
    n = len(seq)
    if n <= width:
        return list(seq)
    half = width // 2
    mid = n // 2
    lo = max(0, mid - half)
    return seq[lo : lo + width]


def _vote_mid_region(geocoder, mid_coords, *, max_samples: int = _MID_VOTE_SAMPLES):
    """Vote the modal ``(county, state)`` across several mid-route samples.

    The flake demotion used to validate the on-route midpoint with ONE
    sample, so a single flaky mid geocode (or a brief dip across a county
    line at the exact midpoint) made the demotion *fail open* and keep both
    endpoints. Sampling several central points and taking the plurality
    ``(county, state)`` makes one flake unable to defeat the demotion.

    ``mid_coords`` is a list of ``(lat, lon)`` from the central portion of
    the trace. Returns ``(None, None)`` when there is no geocoder, no usable
    coord, no sample produced a county/state, or the top count is an exact
    tie between regions (a genuine crossing near the midpoint splits the
    samples evenly). The caller treats that as inconclusive → keep both.
    """
    if geocoder is None:
        return (None, None)
    coords = [c for c in (mid_coords or []) if c]
    if not coords:
        return (None, None)
    votes: Counter = Counter()
    for lat, lon in _evenly_sample(coords, max_samples):
        county, state = _endpoint_county_state(geocoder, (lat, lon))
        if county is None and state is None:
            continue
        votes[(county, state)] += 1
    if not votes:
        return (None, None)
    # An exact top tie is inconclusive -> (None, None), so callers keep both
    # endpoints. Unlike dominant_location (which NAMES a place and keeps its
    # same-state alphabetical tie), this vote only decides whether to DROP a
    # real endpoint, so any tie must not be broken by the alphabet.
    top = max(votes.values())
    winners = [k for k, n in votes.items() if n == top]
    if len(winners) > 1:
        return (None, None)
    return winners[0]


# ── US state abbreviation map ──────────────────────────────────────────
# Only 50 states + DC. Not an exhaustive i18n solution — deliberately
# scoped to current region (USA). Ireland + other regions need their
# own format rules; extend here when a non-USA project starts.
_US_STATE_ABBR: dict[str, str] = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}


# Reverse lookup set for abbreviation validation. Frozen at import so the
# membership test below is O(1) and never mutated.
_US_STATE_ABBR_VALUES: frozenset[str] = frozenset(_US_STATE_ABBR.values())


def normalize_state_abbr(state_str: Optional[str]) -> str:
    """Convert a full state name to its 2-letter abbreviation.

    Returns the input uppercased if it's already 2 chars (assumed
    abbreviation); returns the mapped abbreviation for known US states;
    returns empty string if unmapped.

    Case-insensitive; trims whitespace.
    """
    if not state_str:
        return ""
    s = state_str.strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()
    return _US_STATE_ABBR.get(s.lower(), "")


# Display-normalize an OSM road ``ref``/name for a title. OSM joins concurrent
# routes in the ``ref`` tag with ';' ("I 81;US 58") and stores refs space-
# separated ("I 81"); a title wants the primary route, hyphenated ("I-81"). A
# named road (no leading route-ref token, e.g. "Sea Island Parkway") is
# returned unchanged. Idempotent on an already-hyphenated ref ("US-64").
_ROUTE_REF_HYPHENATE_RE = re.compile(r"^([A-Za-z]{1,3})[\s-](\d+[A-Za-z]?)$")


def _hyphenate_route_ref(token: str) -> str:
    """Hyphenate a single route-ref token (``"I 81"`` → ``"I-81"``); a named
    road or already-hyphenated ref passes through unchanged. Shared by the
    display normalizer and the per-member vote-site splitter."""
    m = _ROUTE_REF_HYPHENATE_RE.match(token)
    if m:
        return f"{m.group(1).upper()}-{m.group(2).upper()}"
    return token


def _road_ref_members(road: str) -> list[str]:
    """Every ';'-separated member of an OSM concurrency ``ref``, each stripped
    and hyphenated (``"I 77;I 81;US 52"`` → ``["I-77", "I-81", "US-52"]``); a
    single road yields a one-element list. Used at the via-road vote site so a
    concurrency credits EVERY carried route, not just the primary — otherwise an
    I-77/I-81 overlap collapses to "I-77" and outvotes the I-81 the drive stays
    on after I-77 branches away."""
    return [_hyphenate_route_ref(m.strip()) for m in road.split(";") if m.strip()]


def render_degraded_tier_label(
    tier: int,
    *,
    road: Optional[str] = None,
    county: Optional[str] = None,
    region: Optional[str] = None,
    state: Optional[str] = None,
) -> str:
    """Render the DEGRADED burned-OST label for a below-tier-1 award.

    Conservative forms sourced from the LF's ``ResolvedLocation`` (no named
    corridors / genre words the pipeline can't mechanically produce):

    * tier 2 (corridor + region) → ``"{road}, {region}"``   (e.g. ``"I-81, VA"``)
    * tier 3 (county / region)   → ``"{county}, {region}"`` (e.g. ``"Franklin County, PA"``)
    * tier 4 (state only)        → bare ``"{state}"``       (e.g. ``"Virginia"``)

    ``region`` is the trailing abbreviation for tiers 2/3 (the tier-1 ``…, VA``
    shape); ``state`` is the full name the bare tier-4 claim prefers. A missing
    field FALLS to the broadest form it CAN support rather than a blank/fabricated
    component. Tier 1 (renders via :func:`build_lf_title`) and tier 5 (no-claim)
    return ``""`` — the caller keeps the original or hard-fails.
    """
    road = (road or "").strip()
    county = (county or "").strip()
    region = (region or "").strip()
    state = (state or "").strip()
    if tier == 2 and road and region:
        return f"{road}, {region}"
    if tier in (2, 3) and county and region:
        return f"{county}, {region}"
    if tier in (2, 3, 4):
        # tier-4 form (bare region); also the fall-through for a tier-2/3 award
        # whose more-specific field is absent. Full state name reads best.
        return state or region
    return ""
