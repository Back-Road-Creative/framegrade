"""Overpass API queries, POI filtering, and response caching.

Split out of the chapter generator module.
"""

import errno
import hashlib
import json
import logging
import math
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Tried in order; ranked on a DAY-SIZED bbox, which reverses a toy bbox's
# ranking (see docs/pipeline-internals.md). Adding one widens the network
# allowlist and needs operator sign-off.
OVERPASS_ENDPOINTS = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
OVERPASS_TIMEOUT = 25  # seconds (increased for US bounding boxes)
OVERPASS_RETRY_DELAY = 5  # seconds — the fallback wait when a 429 omits Retry-After
OVERPASS_RETRY_MAX_SLEEP = 60  # seconds; a mirror asking for an hour is ignored
OVERPASS_ATTEMPTS_PER_ENDPOINT = 2

# Shared-cache lifetime. OSM POI/route data moves on edit timescales, not run
# timescales, so a long TTL is safe and is what makes re-processing one drive
# (LFs, then shorts, then superlong, then a re-run into a fresh --output-dir)
# free of network traffic. Bounded rather than infinite so genuine OSM edits
# do land eventually.
OVERPASS_CACHE_TTL_SEC = 90 * 24 * 60 * 60  # 90 days

# Descriptive User-Agent. Overpass rejects the Python-urllib default with HTTP
# 406. OSM-family APIs also expect a contact-identified UA per their usage
# policy, so set ``GMS_OVERPASS_CONTACT`` (an e-mail address or URL for whoever
# runs this install) and it is appended to the project identifier below.
OVERPASS_USER_AGENT = "gomoveshift-video/1.0 (+https://gomoveshift.com)"
OVERPASS_CONTACT_ENV = "GMS_OVERPASS_CONTACT"


def overpass_user_agent() -> str:
    """The request User-Agent: the project identifier plus the configured contact."""
    contact = os.environ.get(OVERPASS_CONTACT_ENV, "").strip()
    if not contact:
        return OVERPASS_USER_AGENT
    return f"{OVERPASS_USER_AGENT[:-1]}; {contact})"


# Sticky last-good endpoint, for the life of the process. A dead head endpoint
# otherwise makes every call re-pay the full failover budget from the top of
# OVERPASS_ENDPOINTS (many minutes wasted on a long run). Remembering the
# endpoint that last answered and trying it first amortizes that cost across
# a run without ever skipping an endpoint or blacklisting one permanently —
# a later failure still falls back to the full declared-order rotation.
_last_good_endpoint: Optional[str] = None


def reset_sticky_overpass_endpoint() -> None:
    """Test hook: forget the remembered endpoint so state can't leak between tests."""
    global _last_good_endpoint
    _last_good_endpoint = None


def _endpoints_to_try() -> tuple[str, ...]:
    """``OVERPASS_ENDPOINTS`` reordered so the last-good endpoint goes first.

    Every endpoint still appears exactly once, in declared order among
    themselves — only the sticky one is promoted to the front.
    """
    sticky = _last_good_endpoint
    if sticky is None or sticky not in OVERPASS_ENDPOINTS:
        return OVERPASS_ENDPOINTS
    rest = tuple(url for url in OVERPASS_ENDPOINTS if url != sticky)
    return (sticky, *rest)


# Max distance (m) from a named route's real geometry to a sampled GPS
# waypoint for the route to count as driven. ~1 km absorbs GPS drift and sparse
# Overpass vertices without the old 50 km centroid slop that turned far parallel
# roads into false positives.
ROUTE_MATCH_RADIUS_M = 1_000

# Overlap floor: distinct sampled waypoints that must fall within
# ROUTE_MATCH_RADIUS_M of a named route's real geometry before the route is
# claimed. A drive that merely CROSSES the route grazes 1-2 waypoints; one that
# actually FOLLOWS it touches many. Without this floor a single vertex within
# 1 km of a single waypoint stamped "Driving the {route}" titles and per-chapter
# route claims onto transits that only clipped the route's corner.
ROUTE_MIN_WAYPOINT_MATCHES = 3


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in meters between two lat/lon points."""
    earth_radius_m = 6_371_000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return earth_radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _route_vertices(elem: dict) -> list[tuple[float, float]]:
    """Flat (lat, lon) vertices of a way/relation's real ``out geom`` geometry.

    A way carries a flat ``geometry`` list; a relation carries geometry per
    member (its outer ways hold the polyline). Returns ``[]`` when the element
    has no geometry (a node, or a legacy ``out center`` response).
    """
    geom = elem.get("geometry")
    if geom:
        return [(p["lat"], p["lon"]) for p in geom if "lat" in p and "lon" in p]
    verts: list[tuple[float, float]] = []
    for member in elem.get("members", []) or []:
        for p in member.get("geometry", []) or []:
            if "lat" in p and "lon" in p:
                verts.append((p["lat"], p["lon"]))
    return verts


def _bbox_from_waypoints(waypoints, margin_km: float = 2.0) -> tuple[float, float, float, float]:
    """Compute (south, west, north, east) bounding box from waypoints with margin."""
    lats = [wp.latitude for wp in waypoints]
    lons = [wp.longitude for wp in waypoints]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    # ~0.009 degrees per km latitude
    margin_deg = margin_km * 0.009
    return (
        min_lat - margin_deg,
        min_lon - margin_deg * 1.5,  # longitude degrees are wider at mid-latitudes
        max_lat + margin_deg,
        max_lon + margin_deg * 1.5,
    )


def _shared_cache_entry(query: str) -> Optional[Path]:
    """Shared-cache path for one Overpass query, or ``None`` when disabled.

    Keyed on the FULL query text, not the bbox: the query embeds
    ``named_road_patterns``, so a bbox-only key served one pattern set's
    answer to another (the earlier ``_cache_path_for_bbox``
    behaviour, still used for the per-run stages copy).
    """
    from ...paths import gms_overpass_cache_dir

    cache_dir = gms_overpass_cache_dir()
    if cache_dir is None:
        return None
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"overpass_{digest}.json"


def _load_shared_cache(path: Optional[Path]) -> Optional[dict]:
    """Read a cache entry, or ``None`` if absent, stale, or unreadable."""
    if path is None or not path.exists():
        return None
    try:
        if time.time() - path.stat().st_mtime > OVERPASS_CACHE_TTL_SEC:
            logger.info("Overpass cache entry expired, refetching: %s", path.name)
            return None
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        # A corrupt or unreadable entry must never be fatal — it degrades to
        # a live query, which then overwrites it.
        logger.warning("Overpass cache entry unusable (%s): %s", path.name, exc)
        return None


def _store_shared_cache(path: Optional[Path], payload: dict) -> None:
    """Persist a successful response. Best-effort; never raises.

    Written to a temp path and renamed so a killed process can never leave a
    truncated entry that the next run would read as valid JSON.
    """
    if path is None or not payload:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(payload, fh)
        tmp.replace(path)
    except OSError as exc:
        logger.warning("Failed to write Overpass cache entry %s: %s", path.name, exc)


def _retry_after_seconds(exc: urllib.error.HTTPError) -> float:
    """Wait after a 429, from the server's own ``Retry-After`` — delta-seconds
    only, and capped: an encode must not wait an hour on a POI lookup."""
    header = getattr(exc, "headers", None)
    raw = header.get("Retry-After") if header is not None else None
    try:
        wait = float(raw) if raw is not None else float(OVERPASS_RETRY_DELAY)
    except (TypeError, ValueError):
        wait = float(OVERPASS_RETRY_DELAY)
    return max(0.0, min(wait, float(OVERPASS_RETRY_MAX_SLEEP)))


def _endpoint_is_unusable(exc: BaseException) -> bool:
    """True when no retry will change this endpoint's answer. Refusal, dead
    network and DNS miss are verdicts about reachability — spend the budget on
    the next mirror. ``URLError`` carries the distinguishing errno on ``.reason``."""
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, socket.gaierror):
        return True
    return isinstance(reason, OSError) and reason.errno in (
        errno.ECONNREFUSED,
        errno.ENETUNREACH,
        errno.EHOSTUNREACH,
    )


def _query_overpass(
    bbox: tuple[float, float, float, float],
    named_road_patterns: Optional[list[str]] = None,
) -> Optional[dict]:
    """
    Query Overpass for towns, POIs, and named roads in bbox.

    Answers from the shared cross-run cache when one is warm — every caller
    benefits, including ``destination_resolver``'s POI source, which passes
    no ``stages_dir`` and so had no cache at all.

    Returns raw JSON response. Raises ``RuntimeError`` (from
    ``post_overpass_query``) when every endpoint failed — callers that can
    degrade (``discover_area_pois``, ``destination_resolver``) catch it.
    """
    s, w, n, e = bbox
    bbox_str = f"{s},{w},{n},{e}"

    # Build named road/route filter
    road_filter = ""
    if named_road_patterns:
        pattern = "|".join(named_road_patterns)
        road_filter = (
            f'way["name"~"{pattern}"]({bbox_str});\n  relation["name"~"{pattern}"]({bbox_str});'
        )

    query = f"""[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  node["place"~"city|town|village"]({bbox_str});
  node["tourism"~"viewpoint|attraction"]({bbox_str});
  node["natural"~"peak|saddle"]({bbox_str});
  node["natural"~"beach|bay|cape|island"]["name"]({bbox_str});
  node["waterway"~"river"]["name"]({bbox_str});
  node["man_made"~"bridge"]["name"]({bbox_str});
  node["historic"~"monument|memorial"]["name"]({bbox_str});
  node["leisure"~"nature_reserve|park"]["name"]({bbox_str});
  way["leisure"~"nature_reserve|park"]["name"]({bbox_str});
  relation["leisure"~"nature_reserve|park"]["name"]({bbox_str});
  relation["boundary"~"national_park|protected_area"]["name"]({bbox_str});
  relation["route"="road"]["name"]({bbox_str});
  {road_filter}
);
out geom;"""

    cache_entry = _shared_cache_entry(query)
    cached = _load_shared_cache(cache_entry)
    if cached is not None:
        logger.info("Overpass shared-cache hit: %s", cache_entry.name)
        return cached

    payload = post_overpass_query(query)
    _store_shared_cache(cache_entry, payload)
    return payload


def post_overpass_query(query: str) -> dict:
    """POST ``query`` to each Overpass endpoint in turn; return the first answer.

    The failover loop lives here rather than inline in ``_query_overpass`` so
    that EVERY Overpass caller gets it. The coastal-band query in
    ``generator._coastline_points_for_bbox`` used to post straight to the
    primary endpoint alone, with no failover, and so died on a mirror the POI
    pass had already routed around — silently disabling the coastal band on a
    coastal drive.

    Tries the last endpoint that answered successfully FIRST (see
    ``_endpoints_to_try``); a failure there still falls back to every
    endpoint in declared order, so a stale sticky pick can never strand a
    call.

    Raises ``RuntimeError`` naming every endpoint's failure when all are down.
    """
    global _last_good_endpoint
    data = f"data={urllib.request.quote(query)}"
    failures: list[str] = []

    for url in _endpoints_to_try():
        for attempt in range(OVERPASS_ATTEMPTS_PER_ENDPOINT):
            last = attempt == OVERPASS_ATTEMPTS_PER_ENDPOINT - 1
            try:
                req = urllib.request.Request(
                    url,
                    data=data.encode("utf-8"),
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "User-Agent": overpass_user_agent(),
                    },
                )
                with urllib.request.urlopen(req, timeout=OVERPASS_TIMEOUT) as resp:
                    if resp.status != 200:
                        raise OSError(f"Overpass returned HTTP {resp.status}")
                    result = json.loads(resp.read().decode("utf-8"))
                    _last_good_endpoint = url
                    return result
            except urllib.error.HTTPError as exc:
                # 429 means "later", not "never" — wait it out in place.
                logger.error("Overpass %s failed: HTTP %s", url, exc.code)
                failures.append(f"{url}: HTTP {exc.code}")
                if exc.code == 429 and not last:
                    time.sleep(_retry_after_seconds(exc))
                    continue
                break
            except (OSError, TimeoutError, json.JSONDecodeError) as exc:
                # EVERY failure is logged: create_connection re-raises only its
                # final address error, which routinely names the wrong problem.
                logger.error("Overpass %s failed: %s", url, exc)
                failures.append(f"{url}: {exc}")
                if _endpoint_is_unusable(exc):
                    break  # a refusal is a verdict; fail over now, don't sleep
                if not last:
                    time.sleep(2**attempt)

    raise RuntimeError(
        "Overpass query failed on every endpoint — "
        + "; ".join(failures)
        + ". Check network connectivity or Overpass API status."
    )


def _cache_path_for_bbox(stages_dir: Path, bbox: tuple[float, float, float, float]) -> Path:
    """Deterministic cache path for an Overpass bbox query."""
    bbox_str = f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}"
    bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()[:12]
    return stages_dir / f"overpass_cache_{bbox_hash}.json"


class OverpassChapterSource:
    """Discover POIs and named routes via Overpass API.

    Handles spatial indexing, element classification, caching,
    and route detection for a GPS journey.
    """

    DEFAULT_ROAD_PATTERNS = [
        "Ring of Kerry",
        "Wild Atlantic Way",
        "Healy Pass",
        "Caha Pass",
        "Conor Pass",
        "Sky Road",
        "Slea Head",
        "Vee Road",
    ]

    @staticmethod
    def build_spatial_index(
        journey,
        gps_duration: float,
    ) -> list[tuple[float, float, float]]:
        """Build a sampled spatial index from journey waypoints.

        Samples one waypoint per ~30s mark of GPS time, choosing the waypoint
        whose REAL timestamp is nearest each mark and carrying that waypoint's
        actual elapsed time (from the journey start). Mapping by real time — not
        a proportional index — keeps a variable sample rate or a mid-trace
        dropout from stamping a POI at a time the drive wasn't there.

        Returns:
            List of (gps_time, latitude, longitude) tuples.
        """
        sample_interval = 30
        num_samples = max(1, int(gps_duration / sample_interval))
        start = journey.waypoints[0].timestamp
        timed = [((wp.timestamp - start).total_seconds(), wp) for wp in journey.waypoints]
        sampled_wps = []
        for i in range(num_samples):
            target_time = i * sample_interval
            gps_time, wp = min(timed, key=lambda tw, t=target_time: abs(tw[0] - t))
            sampled_wps.append((gps_time, wp.latitude, wp.longitude))
        return sampled_wps

    @staticmethod
    def classify_element(
        elem: dict,
        sampled_wps: list[tuple[float, float, float]],
        proximity_m: float,
        named_road_patterns: list[str],
        detected_routes: set[str],
        route_matches: Optional[dict[str, set[int]]] = None,
    ) -> Optional[tuple[float, float, float, str, float]]:
        """Classify an Overpass element and match it to the route.

        Returns (closest_gps_time, lat, lon, poi_type, distance_m) if
        the element is a matchable POI within proximity, or None to skip.

        Side effect: adds to detected_routes if a named route is found. When
        ``route_matches`` is supplied, also records the set of distinct sampled-
        waypoint indices that lie on the route's geometry, keyed by route name —
        the caller applies the ``ROUTE_MIN_WAYPOINT_MATCHES`` overlap floor to it.
        """
        # Get element coordinates. Under ``out geom`` way/relation elements carry
        # a ``bounds`` box but no ``center``; derive the point from the bounds
        # center (keeping ``center`` support for legacy/injected fixtures).
        if elem.get("type") == "node":
            elat = elem.get("lat", 0)
            elon = elem.get("lon", 0)
        elif elem.get("type") in ("way", "relation"):
            center = elem.get("center")
            bounds = elem.get("bounds")
            if isinstance(center, dict):
                elat = center.get("lat", 0)
                elon = center.get("lon", 0)
            elif isinstance(bounds, dict):
                elat = (bounds.get("minlat", 0) + bounds.get("maxlat", 0)) / 2
                elon = (bounds.get("minlon", 0) + bounds.get("maxlon", 0)) / 2
            else:
                return None
        else:
            return None

        tags = elem.get("tags", {})
        name = tags.get("name", "")
        if not name:
            return None

        # Extract tag categories
        place_type = tags.get("place", "")
        tourism = tags.get("tourism", "")
        natural = tags.get("natural", "")
        leisure = tags.get("leisure", "")
        boundary = tags.get("boundary", "")

        # Find nearest waypoint
        min_dist = float("inf")
        closest_gps_time = 0.0
        for gps_time, wlat, wlon in sampled_wps:
            d = _haversine(elat, elon, wlat, wlon)
            if d < min_dist:
                min_dist = d
                closest_gps_time = gps_time

        # Parks/reserves as ways/relations -- proximity-match like POI nodes
        is_park = bool(leisure or boundary in ("national_park", "protected_area"))
        if is_park and elem.get("type") in ("way", "relation"):
            if min_dist > proximity_m:
                return None
            # Fall through to POI type classification below
        # Named roads/relations (NOT parks): match the route only where its real
        # ``out geom`` geometry passes within ROUTE_MATCH_RADIUS_M of a sampled
        # waypoint. Element centers can be tens of km from where the GPS track
        # crosses the road, so measure vertex-to-waypoint, not center-to-waypoint.
        elif (
            elem.get("type") in ("way", "relation")
            and not place_type
            and not tourism
            and not natural
        ):
            verts = _route_vertices(elem)
            # Distinct sampled-waypoint indices with a route vertex within radius.
            # Indices (not coords) so multiple ways of one route union cleanly.
            matched_wp_idx = {
                wi
                for wi, (_gt, wlat, wlon) in enumerate(sampled_wps)
                if any(
                    _haversine(vlat, vlon, wlat, wlon) <= ROUTE_MATCH_RADIUS_M
                    for vlat, vlon in verts
                )
            }
            if matched_wp_idx:
                for pattern_name in named_road_patterns:
                    if pattern_name.lower() in name.lower():
                        detected_routes.add(pattern_name)
                        if route_matches is not None:
                            route_matches.setdefault(pattern_name, set()).update(matched_wp_idx)
                        break
            return None  # Routes are detected, not returned as POIs
        else:
            # Nodes (towns, POIs): find nearest waypoint
            if min_dist > proximity_m:
                return None

        # Determine POI type
        if place_type:
            poi_type = place_type
        elif tourism:
            poi_type = tourism
        elif natural:
            poi_type = natural
        elif leisure or boundary:
            poi_type = "park"
        else:
            poi_type = "other"

        return (closest_gps_time, elat, elon, poi_type, min_dist)

    def discover_area_pois(
        self,
        journey,
        gps_duration: float,
        original_duration: float,
        scale_gps_to_video,
        map_to_edited_time,
        keep_segments=None,
        stages_dir: Optional[Path] = None,
        proximity_m: float = 500.0,
        named_road_patterns: Optional[list[str]] = None,
    ) -> tuple[list[tuple[float, dict]], list[dict], list[str]]:
        """
        Discover towns, POIs, and named roads via a single Overpass API call.

        Args:
            journey: Parsed GPS journey with waypoints
            gps_duration: Total GPS duration in seconds
            original_duration: Original (pre-stop-removal) video duration
            scale_gps_to_video: Callable(gps_time, gps_duration, original_duration) -> float
            map_to_edited_time: Callable(original_time) -> float
            keep_segments: Optional list of (start, end) kept segments
            stages_dir: Directory for caching Overpass responses
            proximity_m: Max distance from route for POI matching
            named_road_patterns: List of named road/route patterns to detect

        Returns:
            Tuple of (matched_pois, route_pois, named_routes) where:
            - matched_pois: list of (edited_video_time, poi_dict) sorted by time
            - route_pois: list of poi dicts
            - named_routes: sorted list of detected route names
        """
        if not journey.waypoints:
            return [], [], []

        if named_road_patterns is None:
            named_road_patterns = list(self.DEFAULT_ROAD_PATTERNS)

        bbox = _bbox_from_waypoints(journey.waypoints)

        # Check cache first
        overpass_data = None
        cache_file = None
        if stages_dir:
            cache_file = _cache_path_for_bbox(stages_dir, bbox)
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        overpass_data = json.load(f)
                    logger.info(f"Loaded Overpass cache: {cache_file.name}")
                except Exception as exc:
                    logger.warning(
                        "Failed to load Overpass cache %s; preferred cached "
                        "response, falling back to fresh Overpass API query: %s",
                        cache_file.name,
                        exc,
                    )

        # Query Overpass if no cache
        # Look up _query_overpass through the parent package module to allow mock patching
        # (tests patch gomoveshift.pipeline.chapters._query_overpass)
        _pkg = sys.modules.get("gomoveshift.pipeline.chapters", None)
        _query_fn = getattr(_pkg, "_query_overpass", _query_overpass) if _pkg else _query_overpass
        if overpass_data is None:
            logger.info("Querying Overpass API for area POIs...")
            try:
                overpass_data = _query_fn(bbox, named_road_patterns)
            except RuntimeError as exc:
                # Every mirror is down. That is a verdict about Overpass, not
                # about this LF: the generator's Nominatim geocoding fallback
                # still yields chapters, so degrade LOUDLY instead of letting
                # the raise abort the whole chapter pass (otherwise 0
                # chapters on a long LF → metadata HARD FAIL while Nominatim
                # is answering fine).
                logger.error(
                    "Overpass area discovery unavailable — falling back to "
                    "Nominatim geocoding for chapters: %s",
                    exc,
                )
                return [], [], []
            if overpass_data and cache_file:
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, "w") as f:
                        json.dump(overpass_data, f)
                    logger.info(f"Cached Overpass response: {cache_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to cache Overpass response: {e}")

        if not overpass_data or "elements" not in overpass_data:
            logger.error("Overpass returned no data")
            return [], [], []

        elements = overpass_data["elements"]
        logger.info(f"Overpass returned {len(elements)} elements")

        # Build spatial index from sampled waypoints
        sampled_wps = self.build_spatial_index(journey, gps_duration)

        # Classify elements and match to route
        matched_pois: list[tuple[float, dict]] = []
        detected_routes: set[str] = set()
        route_matches: dict[str, set[int]] = {}

        for elem in elements:
            result = self.classify_element(
                elem,
                sampled_wps,
                proximity_m,
                named_road_patterns,
                detected_routes,
                route_matches,
            )
            if result is None:
                continue

            closest_gps_time, elat, elon, poi_type, min_dist = result
            name = elem.get("tags", {}).get("name", "")

            # Map GPS time -> original video time -> edited video time
            original_time = scale_gps_to_video(closest_gps_time, gps_duration, original_duration)
            edited_time = map_to_edited_time(original_time)

            matched_pois.append(
                (
                    edited_time,
                    {
                        "name": name,
                        "type": poi_type,
                        "lat": elat,
                        "lon": elon,
                        "distance_m": round(min_dist, 1),
                    },
                )
            )

        # Sort by video time
        matched_pois.sort(key=lambda x: x[0])

        # Store results
        route_pois = [poi for _, poi in matched_pois]

        # Overlap floor: keep only routes the GPS track actually followed
        # (>= ROUTE_MIN_WAYPOINT_MATCHES distinct sampled waypoints on the
        # route's geometry). Drops routes a transit merely clipped.
        detected_routes = {
            r
            for r in detected_routes
            if len(route_matches.get(r, set())) >= ROUTE_MIN_WAYPOINT_MATCHES
        }
        # On-route waypoint coords per surviving route, exposed for the per-chapter
        # suffix gate in MarkerMerger: a town is "on the route" only when it sits
        # near a point where the track actually ran along the route.
        self.route_match_coords: dict[str, list[tuple[float, float]]] = {
            r: [(sampled_wps[i][1], sampled_wps[i][2]) for i in sorted(idxs)]
            for r, idxs in route_matches.items()
            if r in detected_routes
        }

        # Sort by pattern list order (first pattern = highest priority)
        named_routes = sorted(
            detected_routes,
            key=lambda r: next(
                (i for i, p in enumerate(named_road_patterns) if p == r),
                len(named_road_patterns),
            ),
        )

        logger.info(
            f"Area discovery: {len(matched_pois)} POIs within {proximity_m}m, "
            f"{len(detected_routes)} named route(s): {list(detected_routes) or 'none'}"
        )

        return matched_pois, route_pois, named_routes

    @staticmethod
    def pois_to_locations(
        area_pois: list[tuple[float, dict]],
        journey_waypoints,
        min_chapters: int,
        is_valid_placename_fn=None,
        resolved_location=None,
    ) -> list:
        """Convert Overpass POIs to Location tuples, deduplicating by name.

        First pass: towns/cities/villages only. If insufficient, adds
        viewpoints/peaks/other POIs that pass the placename validator.

        Args:
            area_pois: list of (video_time, poi_dict) from discover_area_pois
            journey_waypoints: waypoints to extract county/country metadata from
            min_chapters: minimum number of chapters needed
            is_valid_placename_fn: callable(name) -> bool for filtering
            resolved_location: the LF's canonical ``ResolvedLocation``. When supplied,
                each POI's county comes from its OWN
                nearest geocoded waypoint and is reconciled against the
                resolved sampled regions (kept iff the resolver voted that
                county; a known own-county outside the regions snaps to the
                voted dominant county the title/OST also use; an ungeocoded
                POI with no own county OMITS it, never snaps).
                ``None`` preserves the legacy first-waypoint stamp
                (single-video / no-resolver path).

        Returns:
            list of (video_time, Location) tuples, sorted by time
        """
        from gps_journey.core.models import Location

        # Legacy fallback county/country: the FIRST geocoded waypoint;
        # this was stamped on EVERY chapter; with a resolver we derive each
        # POI's own county instead (see ``_county_for`` below).
        _county = None
        _country = None
        for wp in journey_waypoints:
            if hasattr(wp, "location") and wp.location:
                _county = getattr(wp.location, "county", None)
                _country = getattr(wp.location, "country", None)
                if _county:
                    break

        _region_counties = OverpassChapterSource._region_counties(resolved_location)
        _dominant = getattr(resolved_location, "dominant_county", None)

        def _county_for(poi: dict):
            # Legacy path (no resolver) keeps the first-waypoint
            # county. With a resolver, take this POI's OWN nearest-waypoint
            # county and reconcile it against the sampled regions so a chapter
            # can't display a county the trace's resolver never voted.
            if resolved_location is None:
                return _county
            own = OverpassChapterSource._nearest_waypoint_county(
                poi["lat"], poi["lon"], journey_waypoints
            )
            # An ungeocoded POI (own is None, common on raw GoPro GPX)
            # OMITS the county rather than inheriting the dominant one —
            # snapping would stamp a wrong-county chapter (e.g. a POI in one
            # county labelled with the dominant county's name). Only a *known*
            # own-county outside the resolver's sampled regions still snaps to
            # the dominant the title/OST use.
            if own is None:
                return None
            if _region_counties:
                return own if own in _region_counties else _dominant
            return own

        seen_names: set[str] = set()
        town_results: list[tuple[float, Location]] = []

        # Pass 1: towns/cities/villages
        for video_time, poi in area_pois:
            if poi["type"] not in ("city", "town", "village"):
                continue
            if poi["name"] in seen_names:
                continue
            seen_names.add(poi["name"])
            loc = Location(
                latitude=poi["lat"],
                longitude=poi["lon"],
                name=poi["name"],
                city=poi["name"],
                county=_county_for(poi),
                country=_country,
            )
            town_results.append((video_time, loc))

        if len(town_results) >= min_chapters:
            return town_results

        # Pass 2: add viewpoints/peaks/other POIs
        for video_time, poi in area_pois:
            if poi["type"] in ("city", "town", "village"):
                continue
            if poi["name"] in seen_names:
                continue
            if is_valid_placename_fn and not is_valid_placename_fn(poi["name"]):
                continue
            seen_names.add(poi["name"])
            loc = Location(
                latitude=poi["lat"],
                longitude=poi["lon"],
                name=poi["name"],
                county=_county_for(poi),
                country=_country,
            )
            town_results.append((video_time, loc))

        town_results.sort(key=lambda x: x[0])
        return town_results

    @staticmethod
    def _region_counties(resolved_location) -> set:
        """County tokens (the pre-comma part) of the resolved sampled regions."""
        out: set[str] = set()
        for label in getattr(resolved_location, "regions", None) or []:
            county = str(label).split(",")[0].strip()
            if county:
                out.add(county)
        return out

    @staticmethod
    def _nearest_waypoint_county(lat: float, lon: float, journey_waypoints):
        """County of the geocoded waypoint nearest ``(lat, lon)``.

        Linear scan over waypoints carrying a ``.location.county``; squared
        euclidean distance in degrees is monotone-correct for nearest-pick at
        chapter scale. Returns ``None`` when no waypoint is geocoded (common
        for raw GoPro GPX) — the caller then OMITS the county rather
        than snapping to the dominant, which would mislabel the chapter.
        """
        best = None
        best_d = None
        for wp in journey_waypoints:
            wlat = getattr(wp, "latitude", None)
            wlon = getattr(wp, "longitude", None)
            loc = getattr(wp, "location", None)
            county = getattr(loc, "county", None) if loc else None
            if wlat is None or wlon is None or not county:
                continue
            d = (wlat - lat) ** 2 + (wlon - lon) ** 2
            if best_d is None or d < best_d:
                best_d = d
                best = county
        return best
