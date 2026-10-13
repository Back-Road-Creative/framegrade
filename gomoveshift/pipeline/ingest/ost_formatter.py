"""On-screen text (OST) formatting from GPS route analysis.

Derives human-readable location text for video overlays using route analysis
data (road segments, towns passed through) from the gps_journey library.

Supports multiple output shapes:
  A. Cross-state: "{start}, {StateA} to {end}, {StateB}"
  B. Cross-county: "{StartCounty} to {EndCounty}, {State}"
  C. Two towns + road: "{A} to {B} via {Road}"
  D. One town + road: "{Road}, {Town}, {ST}"
  E. Same county: "{Road}, {County}, {ST}"
  F. Road only: "{Road}, {State}"
"""

import logging
from pathlib import Path
from typing import Optional

from ..geo_constants import US_STATE_ABBREVS
from .geocoding import _get_route_analyzer, _get_shared_geocoder

logger = logging.getLogger(__name__)


def _state_abbrev(state_name) -> str | None:
    """Return USPS abbreviation for a US state, or None if not a US state."""
    if not state_name or not isinstance(state_name, str):
        return None
    return US_STATE_ABBREVS.get(state_name)


# ============================================================================
# Junk Town Detection
# ============================================================================

# Rank order for OSM place_class — higher values indicate stronger
# classification and are less likely to be junk.
_PLACE_CLASS_RANK = {
    "city": 4,
    "town": 3,
    "village": 2,
    "hamlet": 1,
}

# Junk thresholds. Two rules fire:
#
# Rule A (highway-only brief): Nominatim returned class='highway' (meaning
# the sampled GPS point closest matched a road, not a place) AND the vehicle
# only spent a brief time in the area. This catches unincorporated crossroads
# like "Gneiss, NC" that appear in OSM as village names but have no place
# features — just a named dot on a road. Real towns you drive through briefly
# still typically have place/amenity/shop features that the geocoder prefers.
#
# Rule B (legacy weak-class fallback): Kept for test compatibility and to
# catch cases where place_class is explicitly "hamlet"/None/unclassified.
_JUNK_HIGHWAY_DURATION = 6  # minutes — highway class must be at least this long
_JUNK_DURATION_MIN = 3  # minutes
_JUNK_IMPORTANCE = 0.3
_JUNK_CLASSES = {None, "", "hamlet", "unclassified"}


def _is_junk_town(town: dict) -> bool:
    """True when a town is a pinpoint/crossroads unlikely to describe a drive."""
    duration = town.get("duration_minutes")
    place_class = town.get("place_class")

    # Rule A: matched a highway feature (not a real place), brief duration.
    if place_class == "highway" and duration is not None and duration < _JUNK_HIGHWAY_DURATION:
        return True

    # Rule B: brief duration + weak classification + low importance.
    if duration is None:
        # Legacy test data with no duration — treat as non-junk so callers
        # that don't populate durations keep working.
        return False
    if duration >= _JUNK_DURATION_MIN:
        return False

    if place_class not in _JUNK_CLASSES:
        return False  # real city/town/village wins despite short stay

    importance = town.get("importance")
    try:
        importance_val = float(importance) if importance is not None else 0.0
    except (TypeError, ValueError):
        importance_val = 0.0
    return importance_val < _JUNK_IMPORTANCE


def _select_endpoint_towns(towns: list[dict]) -> list[dict]:
    """Select the start and end towns for the OST, preserving chronology.

    Strategy:
      1. The first non-junk town chronologically is the start endpoint.
      2. The last non-junk town chronologically (after the start) is the
         end endpoint.
      3. If fewer than 2 non-junk towns survive, return whatever we have
         (the single-town shape in _format_ost_from_route will handle it).

    This replaces the old importance-based ranking, which let high-importance
    brief pass-throughs displace the real chronological endpoints.
    """
    if not towns:
        return []

    non_junk = [t for t in towns if not _is_junk_town(t)]

    if len(non_junk) == 0:
        return []
    if len(non_junk) == 1:
        return non_junk

    return [non_junk[0], non_junk[-1]]


# ============================================================================
# Endpoint Anchoring
# ============================================================================


def _endpoint_anchor_from_waypoint(geocoder, waypoint) -> dict:
    """Reverse-geocode a GPS waypoint to the best available place anchor.

    Prefers reverse_geocode_detailed() because its dict also carries road/POI/
    city_district fields the Location object doesn't. The Location now carries
    town/village/hamlet too, so the reverse_geocode() fallback below is
    equally settlement-aware when the detailed call isn't available (e.g. legacy
    test mocks that only stub reverse_geocode).

    Returns a dict with keys:
      name: str | None  — best place name (town > city > village > None)
      settlements: list[str]  — every town/city/village field the geocoder gave
      county: str | None  — county with " County" suffix preserved
      state: str | None  — full state name
      label: str  — best-effort display label: name, else county, else ""

    Returns an empty dict on any failure. Safe against missing fields.
    """
    if geocoder is None or waypoint is None:
        return {}

    def _str_val(obj, key):
        if isinstance(obj, dict):
            v = obj.get(key)
        else:
            v = getattr(obj, key, None)
        return v if isinstance(v, str) and v else None

    # Primary: reverse_geocode_detailed returns a dict with all address fields
    try:
        detailed = geocoder.reverse_geocode_detailed(waypoint.latitude, waypoint.longitude)
    except Exception:
        detailed = None

    if isinstance(detailed, dict):
        settlements = [v for v in (_str_val(detailed, k) for k in ("town", "city", "village")) if v]
        town = settlements[0] if settlements else None
        county = _str_val(detailed, "county")
        state = _str_val(detailed, "state")
        label = town or county or ""
        return {
            "name": town,
            "settlements": settlements,
            "county": county,
            "state": state,
            "label": label,
        }

    # Fallback: reverse_geocode returns a Location object which now carries
    # town/village/hamlet as well as city/county/state, so a rural
    # endpoint resolves here too.
    try:
        loc = geocoder.reverse_geocode(waypoint.latitude, waypoint.longitude)
    except Exception:
        return {}
    if loc is None:
        return {}

    settlements = [v for v in (_str_val(loc, k) for k in ("town", "city", "village")) if v]
    town = settlements[0] if settlements else None
    county = _str_val(loc, "county")
    state = _str_val(loc, "state")
    label = town or county or ""
    return {
        "name": town,
        "settlements": settlements,
        "county": county,
        "state": state,
        "label": label,
    }


def _pick_anchor_waypoint(waypoints, from_end: bool = False):
    """Pick a waypoint a bit inward from the start/end to dodge bogus GPS fixes.

    GoPro recordings sometimes start with a stale GPS fix left over from a
    previous session — the first waypoint can be hundreds of km from the
    actual drive start. Using waypoint[min(20, len//20)] skips ~20 seconds
    of potentially stale fixes without materially shifting the anchor point
    for legitimate drives (600m max forward progress at highway speed).
    """
    if not waypoints:
        return None
    n = len(waypoints)
    offset = min(20, max(1, n // 20))
    idx = n - 1 - offset if from_end else offset
    idx = max(0, min(idx, n - 1))
    return waypoints[idx]


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    from math import asin, cos, radians, sin, sqrt

    r = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


# A detected town is "the same place" as the start/end anchor only within this
# distance of the anchor waypoint. The anchor geocode sits ~20 fixes (under 1 km)
# inside the track and the geocoder accepts a settlement up to 3 km away; one
# RouteAnalyzer sample step adds up to ~3 km more. 50 km is ~8 steps: a town
# that far up the road is where the drive went, not where it started.
ANCHOR_TOWN_SAME_PLACE_KM = 6.0

_PLACE_PREFIXES = ("city of ", "town of ", "village of ")


def _place_key(name: str) -> str:
    key = name.casefold().strip()
    for prefix in _PLACE_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :].strip()
    return key


def _resolve_endpoint_name(detected: dict | None, anchor: dict | None, waypoint) -> str | None:
    """Name a drive endpoint from the GPS start/end, not a town up the road.

    The anchor is the settlement the start/end fixes geocode to. When it has one,
    it is the endpoint; a detected town is used only if it is the same place as
    ANY settlement field the geocoder returned (town/city/village; same name
    after dropping "City of"/"Town of"/"Village of"), keeping the detected
    spelling. A county/township/region name in the town slot is not a settlement
    and does not count. With no anchor settlement, a detected town is used only
    when its entry coordinates sit within ``ANCHOR_TOWN_SAME_PLACE_KM`` of the
    anchor waypoint; farther, None, so the caller falls back to the county/state
    label. RouteAnalyzer always records entry coordinates; a town without them
    (legacy fixtures) has nothing to contradict it and is kept. One rule for the
    OST string and the route profile.
    """
    from ..place_shape import _looks_like_county, _looks_like_region, _looks_like_township

    detected_name = (detected or {}).get("name") or None
    anchor = anchor or {}
    candidates = anchor.get("settlements")
    if candidates is None:
        candidates = [anchor["name"]] if anchor.get("name") else []
    candidates = [
        c
        for c in candidates
        if not (_looks_like_county(c) or _looks_like_township(c) or _looks_like_region(c))
    ]
    if candidates:
        if detected_name and _place_key(detected_name) in {_place_key(c) for c in candidates}:
            return detected_name
        return candidates[0]
    if not detected_name or waypoint is None:
        return None
    lat = detected.get("entry_latitude")
    lon = detected.get("entry_longitude")
    if lat is None or lon is None:
        return detected_name
    if _haversine_km(lat, lon, waypoint.latitude, waypoint.longitude) <= ANCHOR_TOWN_SAME_PLACE_KM:
        return detected_name
    return None


# ============================================================================
# Shared shape ladder (single source of truth)
# ============================================================================


def format_location_ladder(
    start: str | None = None,
    end: str | None = None,
    road: str | None = None,
    start_state: str | None = None,
    end_state: str | None = None,
    start_county: str | None = None,
    end_county: str | None = None,
    features: list[str] | None = None,
    is_cross_state: bool = False,
    is_cross_county: bool = False,
    suppress_road: bool = False,
    two_town_state_suffix: bool = True,
    cross_county_before_road: bool = False,
) -> str:
    """The single OST/title shape ladder both surfaces route through.

    ``_format_ost_from_route`` (geocode path) and ``RouteProfile.title``
    (structured path) both resolve their inputs to these components and
    delegate here, so the shape priority can no longer drift. Two historical
    policy differences survive as flags, not duplicated ladders:
    ``two_town_state_suffix`` (RouteProfile appends ``, {ST}`` to a bare "A to
    B"; the geocode path does not) and ``cross_county_before_road`` (the
    geocode path emits "County1 to County2" ahead of "Road, County").

    Priority: A cross-state span -> same-endpoint collapse -> two-town
    [+road] -> feature+road -> one town [+road] -> road/county <-> cross-county
    -> road+state -> "". Identical start/end labels never span "X to X".
    """
    features = features or []
    last_state = end_state or start_state
    state_suffix = _state_abbrev(last_state) or last_state
    road_ok = bool(road and not suppress_road)

    two_towns = bool(start and end and start != end)
    same_endpoint = bool(start and end and start == end)
    cross_county = bool(
        is_cross_county and start_county and end_county and start_county != end_county
    )

    def _cross_county() -> str:
        # A town label (when present on one side) substitutes for the bare
        # county, so a one-town crossing emits "{Town} to {County}".
        suffix = f", {state_suffix}" if state_suffix else ""
        return f"{start or start_county} to {end or end_county}{suffix}"

    def _road_county() -> str:
        return (
            f"{road}, {start_county}, {state_suffix}" if state_suffix else f"{road}, {start_county}"
        )

    # Shape A: cross-state, two genuinely distinct endpoints.
    if two_towns and is_cross_state and start_state and end_state and start_state != end_state:
        s_abbrev = _state_abbrev(start_state) or start_state
        e_abbrev = _state_abbrev(end_state) or end_state
        core = f"{start}, {s_abbrev} to {end}, {e_abbrev}"
        if road_ok:
            core = f"{core} via {road}"
        return core

    # Shape B (geocode path only): cross-county span ahead of the single-town
    # collapse, so a one-town crossing still reads as "{Town} to {County}".
    if cross_county_before_road and cross_county:
        return _cross_county()

    def _single_town() -> str:
        town = start or end
        town_state = start_state if start else end_state
        if road_ok:
            # Road + town: abbrev only (a full state name reads poorly mid-label,
            # e.g. "N56, Downings, Ulster"); drop when not a US state.
            abbrev = _state_abbrev(town_state)
            return f"{road}, {town}, {abbrev}" if abbrev else f"{road}, {town}"
        suffix = _state_abbrev(town_state) or town_state  # town only: abbrev or full name
        return f"{town}, {suffix}" if suffix else town

    # Same-endpoint "X to X" collapses to a single location before the
    # two-town/feature shapes so a roundtrip never spans itself.
    if same_endpoint:
        return _single_town()

    two_town_suffix = f", {state_suffix}" if (two_town_state_suffix and state_suffix) else ""
    if two_towns and road_ok:  # Shape 1: two towns + road
        return f"{start} to {end} via {road}{two_town_suffix}"
    if two_towns:  # Shape 2: two towns, no road
        return f"{start} to {end}{two_town_suffix}"
    if features and road_ok:  # Shape 3: feature + road (before a lone town + road)
        suffix = f", {state_suffix}" if state_suffix else ""
        return f"{features[0]} via {road}{suffix}"
    if (start and not end) or (end and not start):  # Shape 4: single town [+ road]
        return _single_town()
    if road_ok and start_county:  # Shape 5: road + county
        return _road_county()
    if cross_county:  # Shape 6: cross-county
        return _cross_county()
    if road_ok and last_state:  # Shape 7: road + state
        return f"{road}, {last_state}"
    return ""


# ============================================================================
# OST Formatting
# ============================================================================


def _format_ost_from_route(
    road_segments: list[dict], towns: list[dict], geocoder=None, waypoints=None
) -> str:
    """Format OST text from RouteAnalyzer results.

    Anchoring strategy:
      1. Start with RouteAnalyzer's detected towns (junk-filtered, chronological).
      2. Reverse-geocode the actual first and last GPS waypoints to get
         geocoded anchors (town, county, state).
      3. Name each endpoint with ``_resolve_endpoint_name``: the waypoint
         anchor's settlement wins; a detected town is used only when it is
         the same place (or, with no anchor settlement, within 6 km).

    Output shapes (priority order):
      A. Cross-state drive: "{start}, {StateA} to {end}, {StateB}"
      B. Cross-county same-state drive with no towns: "{StartCounty} to {EndCounty}, {State}"
      C. Two towns + road: "{A} to {B} via {Road}"
      D. One town + road: "{Road}, {Town}, {ST}"
      E. No towns, same county: "{Road}, {County}, {ST}"
      F. Road only: "{Road}, {State}"
      G. Empty: ""
    """
    top_road = ""
    if road_segments:
        sorted_segs = sorted(road_segments, key=lambda s: s.get("distance_km", 0), reverse=True)
        top_road = sorted_segs[0].get("road", "")

    # Select chronological start/end endpoints, dropping junk towns.
    named_towns = [t for t in towns if t.get("name")]
    endpoint_towns = _select_endpoint_towns(named_towns)

    # Waypoint geocodes for the actual drive endpoints (ground truth for
    # the drive's start/end location). Use _pick_anchor_waypoint to dodge
    # stale GPS fixes in the first/last few waypoints.
    start_wp = _pick_anchor_waypoint(waypoints, from_end=False)
    end_wp = _pick_anchor_waypoint(waypoints, from_end=True)
    start_anchor = _endpoint_anchor_from_waypoint(geocoder, start_wp)
    end_anchor = _endpoint_anchor_from_waypoint(geocoder, end_wp)

    start_state = start_anchor.get("state")
    end_state = end_anchor.get("state")
    start_county = start_anchor.get("county")
    end_county = end_anchor.get("county")

    # Name each endpoint where the GPS starts/ends (see _resolve_endpoint_name):
    # the waypoint anchor's settlement wins over a detected town up the road.
    first_detected = endpoint_towns[0] if endpoint_towns else None
    last_detected = endpoint_towns[-1] if len(endpoint_towns) >= 2 else None
    start_name = _resolve_endpoint_name(first_detected, start_anchor, start_wp)
    end_name = _resolve_endpoint_name(last_detected, end_anchor, end_wp)

    # Cross-boundary validation: verify the route actually crosses county/state
    # lines by checking a midpoint. Geocoder flakes can report a different
    # county OR state for nearby points on a single-region drive — the midpoint
    # check catches both.
    cross_state = bool(start_state and end_state and start_state != end_state)
    cross_county = bool(
        start_county and end_county and start_county != end_county and start_state == end_state
    )
    if (cross_county or cross_state) and waypoints and geocoder:
        # VOTE the midpoint across several central samples, not a single
        # waypoint — one flaky mid geocode must not fail the demotion open.
        # Shared modal-vote helper (see place_shape._vote_mid_region).
        from ..place_shape import _central_window, _vote_mid_region

        mid_coords = [
            (wp.latitude, wp.longitude)
            for wp in _central_window(list(waypoints))
            if getattr(wp, "latitude", None) is not None
            and getattr(wp, "longitude", None) is not None
        ]
        mid_county, mid_state = _vote_mid_region(geocoder, mid_coords)
        if cross_county and mid_county and mid_county == start_county:
            # Midpoint matches start county — likely a geocoder flake on the
            # end waypoint. Demote to single-county.
            cross_county = False
            end_county = start_county
        if cross_state and mid_state and mid_state == start_state:
            # Midpoint state matches the start state — the end-waypoint
            # geocode flaked across a state line on a single-state drive. Demote
            # to single-state so the ladder never burns a bogus "…, A to …, B".
            cross_state = False
            end_state = start_state

    # Delegate to the single shared shape ladder. For cross-state, the
    # anchor "label" (county/town) substitutes for a missing town name so the
    # span shape still fires; for cross-county the county is NOT injected into
    # start/end (that would read as a two-town span and steal a via-road) — the
    # ladder's cross-county shape uses start_county/end_county directly.
    ladder_start = start_name
    ladder_end = end_name
    if cross_state:
        ladder_start = ladder_start or (start_anchor.get("label") or None)
        ladder_end = ladder_end or (end_anchor.get("label") or None)

    out = format_location_ladder(
        start=ladder_start,
        end=ladder_end,
        road=top_road or None,
        start_state=start_state,
        end_state=end_state,
        start_county=start_county,
        end_county=end_county,
        is_cross_state=cross_state,
        is_cross_county=cross_county,
        two_town_state_suffix=False,
        cross_county_before_road=True,
    )
    if out:
        return out

    # Tail fallbacks the structured ladder doesn't carry (raw single town /
    # road with only a full state name).
    single_town = start_name or end_name
    if single_town and (start_state or end_state):
        return f"{single_town}, {start_state or end_state}"
    if single_town:
        return single_town
    if top_road and (start_state or end_state):
        return f"{top_road}, {start_state or end_state}"
    if top_road:
        return top_road
    return ""


def derive_episode_ost_text(gpx_path, route_analyzer=None, geocoder=None) -> str:
    """Derive per-episode OST using RouteAnalyzer road segments + towns.

    Uses gps_journey.analysis.RouteAnalyzer.get_road_segments() for actual
    road names and get_towns_passed_through() for town names.

    Args:
        gpx_path: Path to the episode's GPX file, or None.
        route_analyzer: Optional shared RouteAnalyzer instance.
        geocoder: Optional shared NominatimRoadsClient (for state fallback).
    """
    if not gpx_path:
        return ""

    try:
        from gps_journey.core.gpx_parser import GPXParser

        parser = GPXParser()
        journey = parser.parse(Path(gpx_path))

        if not journey.waypoints:
            return ""

        if route_analyzer is None:
            if geocoder is None:
                geocoder = _get_shared_geocoder()
            route_analyzer = _get_route_analyzer(geocoder)

        road_segments = route_analyzer.get_road_segments(journey.waypoints)
        towns = route_analyzer.get_towns_passed_through(journey.waypoints)

        return _format_ost_from_route(
            road_segments,
            towns,
            geocoder=geocoder,
            waypoints=journey.waypoints,
        )

    except Exception as e:
        logger.warning("  OST derivation failed for %s: %s", gpx_path, e)
        return ""


def _geocode_town_state(geocoder, latitude: float, longitude: float) -> tuple[str, str]:
    """Reverse-geocode a single point to (town_name, state_name).

    Returns ('', '') on any failure. The town is the most specific place
    available (town > city > village). State is the full state name, not the
    abbreviation — callers should map to abbreviation if needed.
    """
    try:
        loc = geocoder.reverse_geocode(latitude, longitude)
    except Exception:
        return "", ""

    if loc is None:
        return "", ""

    town = (
        getattr(loc, "town", None)
        or getattr(loc, "city", None)
        or getattr(loc, "village", None)
        or ""
    )
    state = getattr(loc, "state", None) or ""
    if not isinstance(town, str):
        town = ""
    if not isinstance(state, str):
        state = ""
    return town, state


def _parse_route_ost_endpoints(ost_text: str) -> Optional[tuple[str, str]]:
    """Extract (start_town, end_town) from an 'A to B via Road' shaped OST.

    Handles a trailing ' (XX)' direction differentiation suffix. Returns None
    for shapes that aren't parseable as a route (e.g. 'Road, Town, ST').
    """
    if not ost_text or " to " not in ost_text:
        return None

    # Strip direction differentiation suffix: "...SC (NW)" -> "...SC"
    cleaned = ost_text.rstrip()
    if cleaned.endswith(")"):
        last_open = cleaned.rfind(" (")
        if last_open != -1:
            cleaned = cleaned[:last_open].rstrip()

    # Drop trailing 'via Road' clause if present
    if " via " in cleaned:
        cleaned = cleaned.rsplit(" via ", 1)[0].rstrip()

    if " to " not in cleaned:
        return None
    parts = cleaned.split(" to ", 1)
    if len(parts) != 2:
        return None
    start = parts[0].strip()
    end = parts[1].strip()
    if not start or not end:
        return None
    return start, end


def _extract_trailing_state(text: str) -> str | None:
    """Extract a trailing 2-letter US state abbreviation from 'Place, ST'.

    Returns the abbreviation if found, None otherwise.
    """
    if ", " not in text:
        return None
    parts = text.rsplit(", ", 1)
    abbrev = parts[1].strip()
    if len(abbrev) == 2 and abbrev.isalpha() and abbrev.isupper():
        return abbrev
    return None


def parse_ost_components(ost_text: str) -> dict:
    """Extract structured components from any OST shape.

    DEPRECATED: Kept for backward compatibility with old state files and
    episode dedup. Active pipeline uses RouteProfile for title derivation.

    Returns dict with keys: start, end, road, state, direction,
    start_state (all str or None). start_state is set only for
    cross-state routes where the start endpoint has an embedded state.
    """
    if not ost_text or not isinstance(ost_text, str):
        return {}

    cleaned = ost_text.strip()
    direction = None
    result: dict = {}

    # Strip direction suffix: "...SC (NW)" → "...SC", direction="NW"
    if cleaned.endswith(")"):
        last_open = cleaned.rfind(" (")
        if last_open != -1:
            suffix = cleaned[last_open + 2 : -1].strip()
            if len(suffix) <= 3 and suffix.isalpha():
                direction = suffix
                cleaned = cleaned[:last_open].rstrip()
    if direction:
        result["direction"] = direction

    # Shape A/B/C: "X to Y [via Road]"
    if " to " in cleaned:
        road = None
        if " via " in cleaned:
            before_via, road = cleaned.rsplit(" via ", 1)
            road = road.strip()
        else:
            before_via = cleaned
        parts = before_via.split(" to ", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            start_raw = parts[0].strip()
            end_raw = parts[1].strip()
            if road:
                result["road"] = road
            # Strip trailing state abbreviation from end: "Beaufort County, SC" → "Beaufort County"
            end_state = _extract_trailing_state(end_raw)
            if end_state:
                result["end"] = end_raw[: -(len(end_state) + 2)].strip()
                result["state"] = end_state
            else:
                result["end"] = end_raw
            # Strip trailing state from start (cross-state: "Beaufort, SC to Cashiers, NC")
            start_state = _extract_trailing_state(start_raw)
            if start_state:
                result["start"] = start_raw[: -(len(start_state) + 2)].strip()
                result["start_state"] = start_state
            else:
                result["start"] = start_raw
            return result

    # Shape D/E: "Road, Location, ST"
    comma_parts = [p.strip() for p in cleaned.split(",")]
    if len(comma_parts) == 3:
        result["road"] = comma_parts[0]
        result["start"] = comma_parts[1]
        result["end"] = comma_parts[1]
        result["state"] = comma_parts[2]
        return result
    if len(comma_parts) == 2:
        second = comma_parts[1].strip()
        if len(second) == 2 and second.isalpha() and second.isupper():
            result["start"] = comma_parts[0]
            result["end"] = comma_parts[0]
            result["state"] = second
        elif _state_abbrev(second):
            result["start"] = comma_parts[0]
            result["end"] = comma_parts[0]
            result["state"] = _state_abbrev(second) or second
        else:
            result["road"] = comma_parts[0]
            result["start"] = second
            result["end"] = second
        return result
    if len(comma_parts) == 1 and cleaned:
        result["start"] = cleaned
        result["end"] = cleaned
        return result

    return result


def _drive_ost_from_episode_locations(episode_locations: list[str]) -> str:
    """Build drive-level OST by parsing per-episode OSTs.

    Scans forward for the first parseable 'A to B' episode (takes its start)
    and backward for the last parseable episode (takes its end).
    Returns an empty string when no episode is parseable as a route.
    """
    if not episode_locations:
        return ""

    start_town = None
    for ost in episode_locations:
        parsed = _parse_route_ost_endpoints(ost or "")
        if parsed is not None:
            start_town = parsed[0]
            break

    end_town = None
    for ost in reversed(episode_locations):
        parsed = _parse_route_ost_endpoints(ost or "")
        if parsed is not None:
            end_town = parsed[1]
            break

    if not start_town and not end_town:
        return ""
    if start_town and end_town:
        if start_town == end_town:
            return start_town
        return f"{start_town} to {end_town}"
    return start_town or end_town or ""


def derive_drive_ost_text(
    gpx_paths: list[Path],
    route_analyzer=None,
    geocoder=None,
    episode_locations: Optional[list[str]] = None,
    episode_profiles: Optional[list[dict]] = None,
) -> str:
    """Derive drive-level OST.

    Priority order:
    1. Structured episode_profiles (RouteProfile dicts) — start/end towns
       from first/last profiles.
    2. String-based episode_locations — parse 'A to B' shapes.
    3. Legacy GPX endpoint geocoding.

    Args:
        gpx_paths: List of GPX file paths (ordered).
        route_analyzer: Optional shared RouteAnalyzer (no longer required —
            kept for signature compatibility with callers).
        geocoder: Optional shared NominatimRoadsClient.
        episode_locations: Per-episode OST strings, if already derived.
        episode_profiles: Per-episode RouteProfile dicts, if already built.
    """
    # Preferred: use structured profiles for drive-level title
    if episode_profiles:
        from gomoveshift.pipeline.route_profile import RouteProfile

        profiles = [RouteProfile.from_dict(p) for p in episode_profiles if p]
        if profiles:
            start = profiles[0].start_town
            end = profiles[-1].end_town
            if start and end and start != end:
                return f"{start} to {end}"
            if start or end:
                return start or end
            # Profiles exist but yielded no endpoint towns — don't ship an
            # empty drive-level OST; fall through to the string-based and
            # GPX-geocode fallbacks below (a state-level span still burns).

    # Second: reuse already-derived episode OSTs
    if episode_locations:
        from_episodes = _drive_ost_from_episode_locations(episode_locations)
        if from_episodes:
            return from_episodes

    if not gpx_paths:
        return ""

    valid_paths = [Path(p) for p in gpx_paths if p]
    if not valid_paths:
        return ""

    try:
        from gps_journey.core.gpx_parser import GPXParser

        parser = GPXParser()
        if geocoder is None:
            geocoder = _get_shared_geocoder()

        first_journey = parser.parse(valid_paths[0])
        last_journey = parser.parse(valid_paths[-1]) if len(valid_paths) > 1 else first_journey

        first_wps = list(first_journey.waypoints or [])
        last_wps = list(last_journey.waypoints or [])
        if not first_wps or not last_wps:
            return ""

        # Strip null-island / teleport / low-quality fixes before anchoring.
        # _pick_anchor_waypoint only steps ~20 fixes inward, so a longer stale
        # head/tail run would still own the anchor and burn a wrong drive-wide
        # label that feeds every downstream consumer. Keep the raw trace if the
        # filter would evict everything (degenerate/legacy traces).
        from ..gps_quality import filter_high_quality_points

        try:
            first_wps = filter_high_quality_points(first_wps) or first_wps
            last_wps = filter_high_quality_points(last_wps) or last_wps
        except Exception as exc:  # noqa: BLE001 — never let the filter regress the raw anchor
            logger.debug("drive-OST quality filter skipped: %s", exc)

        # Anchor slightly inward from the raw endpoints — waypoint 0 can be a
        # stale previous-session GPS fix hundreds of km from the drive start
        # (see _pick_anchor_waypoint), which would burn a wrong "A to B" OST.
        start_wp = _pick_anchor_waypoint(first_wps, from_end=False)
        end_wp = _pick_anchor_waypoint(last_wps, from_end=True)

        start_town, start_state = _geocode_town_state(
            geocoder,
            start_wp.latitude,
            start_wp.longitude,
        )
        end_town, end_state = _geocode_town_state(
            geocoder,
            end_wp.latitude,
            end_wp.longitude,
        )

        if not start_town and not end_town:
            # Neither endpoint resolves to a town (rural road). Fall back to
            # a state-level span when the states are known and distinct.
            if start_state and end_state:
                if start_state == end_state:
                    return start_state
                return f"{start_state} to {end_state}"
            return start_state or end_state or ""

        # Same endpoint (or only one resolved) -> single town shape with state abbrev
        if not end_town or start_town == end_town:
            single_town = start_town or end_town
            state_abbrev = _state_abbrev(start_state or end_state)
            if state_abbrev:
                return f"{single_town}, {state_abbrev}"
            return single_town

        if not start_town:
            state_abbrev = _state_abbrev(end_state)
            if state_abbrev:
                return f"{end_town}, {state_abbrev}"
            return end_town

        return f"{start_town} to {end_town}"

    except Exception as e:
        logger.warning("  Drive-level OST derivation failed: %s", e)
        return ""
