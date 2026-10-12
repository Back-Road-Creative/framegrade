"""Structured GPS-backed route profiles for video pipeline location derivation.

Replaces lossy string-based location round-trips with a RouteProfile dataclass
that carries the full GPS hierarchy from Phase A through to title generation.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gomoveshift.pipeline.ingest.ost_formatter import (
    _endpoint_anchor_from_waypoint,
    _haversine_km,
    _pick_anchor_waypoint,
    _resolve_endpoint_name,
    _select_endpoint_towns,
    format_location_ladder,
)
from gomoveshift.pipeline.title_builder import (
    _looks_like_county,
    _looks_like_region,
    _looks_like_township,
    _road_passes_quality,
    _road_ref_members,
)

logger = logging.getLogger(__name__)

# Via-road span: endpoints farther apart than this are a transit, not a local loop.
_ENDPOINTS_FAR_APART_KM = 50.0


def _demote_nonsettlement(town: str | None) -> str | None:
    """Null an endpoint that is a bare county / township / broad-region name
    rather than a settlement a viewer would search.

    Shares ``build_lf_title``'s endpoint demotion instead of duplicating a
    weaker copy: at a rural centroid the geocoder hands back the encompassing
    township ("Cumberland Township"), county ("Beaufort County"), or region
    ("ACE Basin") when no town resolves, and that value must NOT ship as an
    "A to B" endpoint. Nulling it here lets the shape ladder fall through to
    the road/county/state shape. Reuses the shared ``title_builder``
    predicates so the two ladders can't drift.
    """
    if town and (
        _looks_like_county(town) or _looks_like_township(town) or _looks_like_region(town)
    ):
        return None
    return town


# Minimum total driven distance (km) a road must accumulate before it ships
# in ``roads_traversed``. RouteAnalyzer emits one segment per contiguous run of
# same-road reverse-geocode samples, so a single-sample GPS drift onto a
# parallel/service road makes a tiny-distance segment; without a floor it shipped
# verbatim as "Roads traveled: …", LF tags, and metadata["named_routes"].
# ``dominant_road`` already has a >50% plurality guard; this is the equivalent
# floor for the full list.
ROADS_TRAVERSED_MIN_KM = 1.0

# road-slot quality floors for an UNCLASSED road name. RouteAnalyzer
# segments carry only road/distance_km/timestamps — no OSM ``place_class`` —
# so the dominant-road slot cannot be gated class-aware here. Signals we DO
# have: NAME quality (``title_builder._road_passes_quality`` name-only —
# route-ref, curated byway, through-road suffix), which keeps the existing
# >50% plurality bar; and DOMINANCE, for a name that clears none of those
# ("Avenue John Brown", "South Main Street"): it is trusted only when it
# truly owns the drive — a strong distance-share supermajority AND an
# absolute driven-km floor (a street can own 91% of a 2 km hop; it can't own
# 20 km). A hard name-only filter was explicitly rejected: it would
# permanently bar a legitimately named scenic road (the Shape-5 named-road
# contract). Carrying ``place_class`` on gps-side segments is the future
# class-aware upgrade.
UNCLASSED_ROAD_MIN_SHARE = 0.75
UNCLASSED_ROAD_MIN_KM = 5.0


def _road_slot_quality_ok(name: str, *, share: float, distance_km: float) -> bool:
    """True when ``name`` may take the dominant-road slot."""
    if _road_passes_quality(name):
        return True
    return share >= UNCLASSED_ROAD_MIN_SHARE and distance_km >= UNCLASSED_ROAD_MIN_KM


def _dominant_by_distance(roads_by_distance: dict[str, float]) -> str | None:
    """Pick the road holding the most distance, deterministically.

    ``max(d, key=d.get)`` breaks ties by dict-insertion order — which on a
    route is traversal order, i.e. drive direction — so the same physical
    drive could pick a different dominant road depending on which way it was
    filmed. Tie-break alphabetically instead: highest distance first,
    then the lexicographically-first road name. Returns None when empty.
    """
    if not roads_by_distance:
        return None
    return min(roads_by_distance, key=lambda name: (-roads_by_distance[name], name))


# ============================================================================
# Data Models
# ============================================================================


@dataclass(frozen=True)
class ClipGeocode:
    """Geocoded location for a single clip/waypoint anchor."""

    country: str | None = None
    state: str | None = None
    county: str | None = None
    town: str | None = None
    road: str | None = None
    poi: str | None = None
    lat: float = 0.0
    lon: float = 0.0


@dataclass
class RouteProfile:
    """Structured route profile aggregated from GPS data.

    Carries the full GPS hierarchy so downstream stages never need to
    re-parse display strings.
    """

    lf_index: int = 0
    clips: list[ClipGeocode] = field(default_factory=list)
    start_town: str | None = None
    end_town: str | None = None
    dominant_road: str | None = None
    states_traversed: list[str] = field(default_factory=list)
    counties_traversed: list[str] = field(default_factory=list)
    towns_traversed: list[str] = field(default_factory=list)
    roads_traversed: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    is_cross_state: bool = False
    is_cross_county: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON persistence."""
        d = asdict(self)
        d["clips"] = [asdict(c) for c in self.clips]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RouteProfile:
        """Deserialize from a plain dict."""
        clips = [ClipGeocode(**c) for c in d.get("clips", [])]
        return cls(
            lf_index=d.get("lf_index", 0),
            clips=clips,
            start_town=d.get("start_town"),
            end_town=d.get("end_town"),
            dominant_road=d.get("dominant_road"),
            states_traversed=d.get("states_traversed", []),
            counties_traversed=d.get("counties_traversed", []),
            towns_traversed=d.get("towns_traversed", []),
            roads_traversed=d.get("roads_traversed", []),
            features=d.get("features", []),
            is_cross_state=d.get("is_cross_state", False),
            is_cross_county=d.get("is_cross_county", False),
        )

    def _endpoints_far_apart(self) -> bool:
        """True when the first/last clip centroids are >50 km apart.

        A ``via {road}`` suffix only describes the drive when both endpoints
        sit near that road. On a long crossing the dominant local road is far
        from one endpoint, so the via-road is suppressed (mirrors the
        ``ost_formatter`` / ``title_builder`` endpoint sanity).
        """
        if len(self.clips) < 2:
            return False
        a, b = self.clips[0], self.clips[-1]
        if not (a.lat or a.lon) or not (b.lat or b.lon):
            return False
        return _haversine_km(a.lat, a.lon, b.lat, b.lon) > _ENDPOINTS_FAR_APART_KM

    def title(self) -> str:
        """Generate a display title via the shared shape ladder.

        Delegates to ``ost_formatter.format_location_ladder`` so the title
        shape priority cannot drift from the geocode-anchor path.
        """
        states = self.states_traversed
        counties = self.counties_traversed
        # Cross-state endpoints carry their own state; the single-state path
        # uses the trailing state for the whole label.
        start_state = states[0] if states else None
        end_state = states[-1] if states else None

        return format_location_ladder(
            start=_demote_nonsettlement(self.start_town),
            end=_demote_nonsettlement(self.end_town),
            road=self.dominant_road,
            start_state=start_state,
            end_state=end_state,
            start_county=counties[0] if counties else None,
            end_county=counties[-1] if len(counties) >= 2 else None,
            features=self.features,
            is_cross_state=self.is_cross_state and len(states) >= 2,
            is_cross_county=self.is_cross_county and len(counties) >= 2,
            suppress_road=self._endpoints_far_apart(),
        )


# ============================================================================
# Builders
# ============================================================================


def build_route_profile(
    index: int,
    gpx_files: list[Path],
    episode_indices: list[int],
    geocoder,
    route_analyzer,
) -> RouteProfile:
    """Build a RouteProfile for a single episode span from GPX data.

    Args:
        index: The long-form index this profile belongs to.
        gpx_files: All GPX files in the manifest (indexed by episode).
        episode_indices: Which episodes to include in this profile.
        geocoder: NominatimRoadsClient or compatible geocoder.
        route_analyzer: RouteAnalyzer instance.

    Returns:
        A populated RouteProfile. Returns an empty profile on failure.
    """
    all_road_segments: list[dict] = []
    all_towns: list[dict] = []
    all_waypoints: list = []
    clips: list[ClipGeocode] = []

    for ep_idx in episode_indices:
        if ep_idx >= len(gpx_files):
            continue
        gpx_path = gpx_files[ep_idx]
        if not gpx_path or not Path(gpx_path).exists():
            continue

        try:
            from gps_journey.core.gpx_parser import GPXParser

            parser = GPXParser()
            journey = parser.parse(Path(gpx_path))
            if not journey.waypoints:
                continue

            wps = list(journey.waypoints)
            all_waypoints.extend(wps)

            road_segments = route_analyzer.get_road_segments(wps)
            towns = route_analyzer.get_towns_passed_through(wps)
            all_road_segments.extend(road_segments)
            all_towns.extend(towns)

        except Exception as e:
            logger.warning("GPX parse failed for episode %d: %s", ep_idx, e)
            continue

    if not all_waypoints:
        return RouteProfile(lf_index=index)

    # Compute dominant road (>50% of total distance)
    total_distance = sum(s.get("distance_km", 0) for s in all_road_segments)
    dominant_road = None
    roads_by_distance: dict[str, float] = {}
    for seg in all_road_segments:
        name = seg.get("road", "")
        if not name:
            continue
        dist = seg.get("distance_km", 0)
        # Member-credit: an OSM concurrency ``ref`` ("I 77;I 81;US 11;US 52")
        # carries several routes at once — key the distance on EVERY hyphenated
        # member, not the raw ref string, so I-81's mileage isn't split across
        # "I 81", "I 81;US 11", … variants and lose its >50% plurality (the
        # member-credit build_lf_title's vote site already applies).
        for member in _road_ref_members(name):
            roads_by_distance[member] = roads_by_distance.get(member, 0) + dist
    if roads_by_distance:
        top_road_name = _dominant_by_distance(roads_by_distance)
        # A road is "dominant" only when it holds a true distance plurality
        # (>50%). Declaring the longest road dominant at, say, 38% of the
        # route stamps a road the drive barely touched onto the title;
        # leave dominant_road None and let the shape ladder fall through to a
        # county/state shape instead. The slot is additionally quality-
        # gated (``_road_slot_quality_ok``) so a junk local name can't win on
        # a bare plurality; a failed gate emits NO road (road-less is safe, a
        # junk road is not) rather than promoting the runner-up.
        if total_distance > 0:
            _share = roads_by_distance[top_road_name] / total_distance
            if _share > 0.5 and _road_slot_quality_ok(
                top_road_name,
                share=_share,
                distance_km=roads_by_distance[top_road_name],
            ):
                dominant_road = top_road_name

    # Roads traversed (ordered by first appearance, deduplicated). A road is
    # kept only when its segments sum to >= ROADS_TRAVERSED_MIN_KM, so a one-sample
    # reverse-geocode blip onto a parallel/service road doesn't ship as a claim.
    roads_traversed: list[str] = []
    seen_roads: set[str] = set()
    for seg in all_road_segments:
        name = seg.get("road", "")
        if not name:
            continue
        # Same member-credit as the distance map: list each carried route once,
        # gated on its member-credited distance floor, so a concurrency ships
        # "I-81", "US-11" rather than the raw "I 81;US 11" ref.
        for member in _road_ref_members(name):
            if (
                member not in seen_roads
                and roads_by_distance.get(member, 0.0) >= ROADS_TRAVERSED_MIN_KM
            ):
                roads_traversed.append(member)
                seen_roads.add(member)

    # Filter junk towns and select endpoints
    named_towns = [t for t in all_towns if t.get("name")]
    endpoint_towns = _select_endpoint_towns(named_towns)

    # Geocode anchor waypoints for start/end
    start_wp = _pick_anchor_waypoint(all_waypoints, from_end=False)
    end_wp = _pick_anchor_waypoint(all_waypoints, from_end=True)
    start_anchor = _endpoint_anchor_from_waypoint(geocoder, start_wp)
    end_anchor = _endpoint_anchor_from_waypoint(geocoder, end_wp)

    # Name each endpoint where the GPS starts/ends: the waypoint anchor's
    # settlement wins over a detected town up the road (a small-town start, not a
    # larger city 25 km on), and also drops a far transit town (a mountain-lake
    # town in one state must not borrow another state's start anchor). A detected town
    # is used only when it is the same place as the anchor. One shared rule with
    # the OST string: ost_formatter._resolve_endpoint_name.
    first_detected = endpoint_towns[0] if endpoint_towns else None
    last_detected = endpoint_towns[-1] if len(endpoint_towns) >= 2 else None
    start_town = _resolve_endpoint_name(first_detected, start_anchor, start_wp)
    end_town = _resolve_endpoint_name(last_detected, end_anchor, end_wp)
    # demote a bare county/township/region endpoint at the SOURCE so the
    # drive-level OST (derive_drive_ost_text reads start_town/end_town directly,
    # bypassing title()) never ships it either.
    start_town = _demote_nonsettlement(start_town)
    end_town = _demote_nonsettlement(end_town)

    start_state = start_anchor.get("state")
    end_state = end_anchor.get("state")
    start_county = start_anchor.get("county")
    end_county = end_anchor.get("county")

    # Build ClipGeocode objects from anchors
    if start_wp:
        clips.append(
            ClipGeocode(
                state=start_state,
                county=start_county,
                town=start_anchor.get("name"),
                lat=start_wp.latitude,
                lon=start_wp.longitude,
            )
        )
    if end_wp and end_wp is not start_wp:
        clips.append(
            ClipGeocode(
                state=end_state,
                county=end_county,
                town=end_anchor.get("name"),
                lat=end_wp.latitude,
                lon=end_wp.longitude,
            )
        )

    # Aggregate traversal lists
    states_traversed: list[str] = []
    seen_states: set[str] = set()
    for s in [start_state, end_state]:
        if s and s not in seen_states:
            states_traversed.append(s)
            seen_states.add(s)

    counties_traversed: list[str] = []
    seen_counties: set[str] = set()
    for c in [start_county, end_county]:
        if c and c not in seen_counties:
            counties_traversed.append(c)
            seen_counties.add(c)

    towns_traversed: list[str] = []
    seen_towns: set[str] = set()
    for t in named_towns:
        name = t.get("name", "")
        if name and name not in seen_towns:
            towns_traversed.append(name)
            seen_towns.add(name)

    is_cross_state = len(states_traversed) >= 2
    is_cross_county = len(counties_traversed) >= 2

    # GPS-verified cross-county: geocode midpoint, compare to start county
    if is_cross_county and all_waypoints and geocoder:
        mid_wp = all_waypoints[len(all_waypoints) // 2]
        mid_anchor = _endpoint_anchor_from_waypoint(geocoder, mid_wp)
        mid_county = mid_anchor.get("county")
        if mid_county and mid_county == start_county:
            is_cross_county = False

    return RouteProfile(
        lf_index=index,
        clips=clips,
        start_town=start_town,
        end_town=end_town,
        dominant_road=dominant_road,
        states_traversed=states_traversed,
        counties_traversed=counties_traversed,
        towns_traversed=towns_traversed,
        roads_traversed=roads_traversed,
        features=[],
        is_cross_state=is_cross_state,
        is_cross_county=is_cross_county,
    )


def build_lf_profile(
    lf_index: int,
    episode_profiles: list[RouteProfile],
    gpx_files: list[Path],
    episode_indices: list[int],
    geocoder,
    route_analyzer,
) -> RouteProfile:
    """Aggregate per-episode RouteProfiles into a long-form level profile.

    Args:
        lf_index: The long-form index.
        episode_profiles: Per-episode RouteProfiles (may be empty).
        gpx_files: All GPX files in the manifest.
        episode_indices: Episode indices spanned by this LF.
        geocoder: NominatimRoadsClient or compatible.
        route_analyzer: RouteAnalyzer instance.

    Returns:
        An aggregated RouteProfile. Falls back to build_route_profile
        if no episode_profiles are provided.
    """
    if not episode_profiles:
        return build_route_profile(lf_index, gpx_files, episode_indices, geocoder, route_analyzer)

    # Start/end from first/last episode
    start_town = episode_profiles[0].start_town
    end_town = episode_profiles[-1].end_town

    # Merge roads by distance across all episodes
    merged_road_distance: dict[str, float] = {}
    for ep in episode_profiles:
        for rd in ep.roads_traversed:
            # We don't have per-road distances in the profile, so count
            # appearances. The dominant_road from each episode carries weight.
            merged_road_distance[rd] = merged_road_distance.get(rd, 0) + 1.0
        if ep.dominant_road:
            merged_road_distance[ep.dominant_road] = (
                merged_road_distance.get(ep.dominant_road, 0) + 10.0
            )

    # Mirror build_route_profile's >50% guard: a road is "dominant" only at a
    # true distance-weight plurality. Without this, the top-weighted road ships
    # as "via {Road}" even when the drive barely touched it — a barely-
    # driven road on a multi-road day would stamp the title/description.
    dominant_road = None
    if merged_road_distance:
        top_road = _dominant_by_distance(merged_road_distance)
        total_weight = sum(merged_road_distance.values())
        # the merged weights are appearance COUNTS (no distances), so the
        # episode gate's share/km floors can't run here. An unclassed name is
        # trusted only when some EPISODE's distance-based gate already accepted
        # it as that episode's dominant_road; a junk name arriving via
        # roads_traversed alone (trivially 100% of a one-road list) is dropped.
        _episode_dominants = {ep.dominant_road for ep in episode_profiles if ep.dominant_road}
        if (
            top_road
            and total_weight > 0
            and merged_road_distance[top_road] / total_weight > 0.5
            and (_road_passes_quality(top_road) or top_road in _episode_dominants)
        ):
            dominant_road = top_road

    # Merge traversal lists (preserve order, deduplicate)
    def _merge_ordered(lists: list[list[str]]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for lst in lists:
            for item in lst:
                if item not in seen:
                    result.append(item)
                    seen.add(item)
        return result

    states_traversed = _merge_ordered([ep.states_traversed for ep in episode_profiles])
    counties_traversed = _merge_ordered([ep.counties_traversed for ep in episode_profiles])
    towns_traversed = _merge_ordered([ep.towns_traversed for ep in episode_profiles])
    roads_traversed = _merge_ordered([ep.roads_traversed for ep in episode_profiles])
    features = _merge_ordered([ep.features for ep in episode_profiles])

    # Merge clips
    all_clips: list[ClipGeocode] = []
    for ep in episode_profiles:
        all_clips.extend(ep.clips)

    is_cross_state = len(states_traversed) >= 2
    is_cross_county = len(counties_traversed) >= 2

    return RouteProfile(
        lf_index=lf_index,
        clips=all_clips,
        start_town=start_town,
        end_town=end_town,
        dominant_road=dominant_road,
        states_traversed=states_traversed,
        counties_traversed=counties_traversed,
        towns_traversed=towns_traversed,
        roads_traversed=roads_traversed,
        features=features,
        is_cross_state=is_cross_state,
        is_cross_county=is_cross_county,
    )
