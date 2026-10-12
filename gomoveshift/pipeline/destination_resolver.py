"""Destination-POI resolver for day-first LF titles/OST.

Names the *destination* a highlight LF is built around — a wildlife
management area, nature reserve, national/protected-area boundary, named
viewpoint/attraction, or a coastal natural feature — instead of the broad
ambient region the GPS span happens to enclose ("ACE Basin" → "Donnelley
Wildlife Management Area"). Returns ``None`` when no destination POI encloses
the featured footage; the caller's existing cascade then handles the LF.

Deterministic, no per-run judgment:

* **Vocabulary is reused from the CHAPTERS Overpass query** (see
  ``DESTINATION_TAGS``) — a WMA tagged ``leisure=nature_reserve`` OR
  ``boundary=protected_area`` is already caught. This REPLACED the narrower
  title-path ``geo_feature_resolver`` module (deleted as dead code).
* **Prominence weighting = ``composite_score``** (the montage's own peak
  weight, see ``longform_assembler._select_top_peaks``): a featured-cluster
  destination outvotes a long dull transit stretch.
* **Specificity beats a superset region's raw vote.** A broad region encloses
  EVERY peak an enclosed POI does, so it strictly outvotes the specific POI —
  yet the WMA, not ACE Basin, is the destination. Among candidates capturing at
  least ``MIN_PEAK_MASS_SHARE`` of the featured peak mass the SMALLEST (most
  specific) polygon wins, so the enclosed POI beats its superset instead of
  being outvoted by it; prominence + protect-class break residual ties. When no
  candidate clears the share bar the plain highest-vote winner stands (unless a
  hard sample quorum is required — see ``require_sample_quorum``).
* **The POI source is INJECTABLE** so tests run fully offline; the default
  delegates to the already-allowlisted chapters Overpass machinery, adding NO
  new network path.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# Destination-POI tags, mirroring the CHAPTERS query
# (overpass_source._query_overpass). Each row is
# (tag_key, accepted_values, specificity_rank); LOWER rank = more specific,
# a tie-break secondary key after enclosing area.
DESTINATION_TAGS: tuple[tuple[str, frozenset[str], int], ...] = (
    ("boundary", frozenset({"protected_area"}), 10),
    ("boundary", frozenset({"national_park"}), 20),
    ("leisure", frozenset({"nature_reserve"}), 20),
    ("natural", frozenset({"beach", "bay", "cape", "island"}), 30),
    ("tourism", frozenset({"viewpoint", "attraction"}), 40),
    ("leisure", frozenset({"park"}), 50),
)

# bbox -> iterable of Overpass-shaped elements (node lat/lon or way/relation
# bounds). Injected so tests stay offline.
POISource = Callable[[tuple[float, float, float, float]], Iterable[dict]]

# Minimum share of the featured peak/sample mass a POI must capture to anchor
# as the destination on SPECIFICITY. A broad region that merely
# encloses a specific POI's footage no longer outvotes it, and a short whose
# fixes only graze a polygon (a centroid-of-arc artifact) fails this quorum.
MIN_PEAK_MASS_SHARE = 0.5


def _classify(tags: dict) -> Optional[int]:
    """Specificity rank for a destination POI, or ``None`` if not one."""
    best: Optional[int] = None
    for key, values, rank in DESTINATION_TAGS:
        if tags.get(key) in values and (best is None or rank < best):
            best = rank
    return best


def _bbox_of(elem: dict) -> Optional[tuple[float, float, float, float]]:
    """(south, west, north, east) from ``bounds`` or a node/center point."""
    b = elem.get("bounds")
    if isinstance(b, dict):
        try:
            return (
                float(b["minlat"]),
                float(b["minlon"]),
                float(b["maxlat"]),
                float(b["maxlon"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    pt = elem if elem.get("type") == "node" else elem.get("center")
    if isinstance(pt, dict):
        lat, lon = pt.get("lat"), pt.get("lon")
        if lat is not None and lon is not None:
            return (float(lat), float(lon), float(lat), float(lon))
    # ``out geom`` geometry without ``bounds`` (e.g. injected test elements):
    # derive the bounds from the rings themselves.
    outers, _ = _rings_of(elem)
    verts = [v for ring in outers for v in ring]
    if verts:
        lats = [v[0] for v in verts]
        lons = [v[1] for v in verts]
        return (min(lats), min(lons), max(lats), max(lons))
    return None


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    s, w, n, e = bbox
    return abs((n - s) * (e - w))


def _ring_points(pts: Iterable[dict]) -> list[tuple[float, float]]:
    return [(p["lat"], p["lon"]) for p in (pts or []) if "lat" in p and "lon" in p]


def _stitch(fragments: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Join ``out geom`` member-way fragments end-to-end into rings.

    A multipolygon's outer ring is usually SPLIT across several member ways;
    ray-casting each fragment as its own "ring" would test sliver polygons
    instead of the real boundary. Endpoint-matching is exact — Overpass emits
    identical vertex coords on shared way endpoints.
    """
    frags = [list(f) for f in fragments if len(f) >= 2]
    rings: list[list[tuple[float, float]]] = []
    while frags:
        ring = frags.pop()
        grew = True
        while grew and ring[0] != ring[-1]:
            grew = False
            for i, f in enumerate(frags):
                if f[0] == ring[-1]:
                    ring = ring + f[1:]
                elif f[-1] == ring[-1]:
                    ring = ring + f[-2::-1]
                elif f[-1] == ring[0]:
                    ring = f[:-1] + ring
                elif f[0] == ring[0]:
                    ring = f[:0:-1] + ring
                else:
                    continue
                frags.pop(i)
                grew = True
                break
        if len(ring) >= 3:
            rings.append(ring)
    return rings


def _rings_of(
    elem: dict,
) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]]]:
    """``(outer_rings, inner_rings)`` from ``out geom`` geometry; ``([], [])``
    for a node / center-only / legacy ``out center`` element."""
    flat = _ring_points(elem.get("geometry"))
    if flat:
        return ([flat] if len(flat) >= 3 else [], [])
    outer_frags: list[list[tuple[float, float]]] = []
    inner_frags: list[list[tuple[float, float]]] = []
    for member in elem.get("members", []) or []:
        pts = _ring_points(member.get("geometry"))
        if len(pts) >= 2:
            (inner_frags if member.get("role") == "inner" else outer_frags).append(pts)
    return (_stitch(outer_frags), _stitch(inner_frags))


def _point_in_ring(lat: float, lon: float, ring: list[tuple[float, float]]) -> bool:
    """Even-odd ray cast; an unclosed ring is implicitly closed."""
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        lat_i, lon_i = ring[i]
        lat_j, lon_j = ring[j]
        if (lat_i > lat) != (lat_j > lat) and lon < (lon_j - lon_i) * (lat - lat_i) / (
            lat_j - lat_i
        ) + lon_i:
            inside = not inside
        j = i
    return inside


def _encloses(elem: dict, bbox: tuple[float, float, float, float], lat: float, lon: float) -> bool:
    """True POLYGON containment when the element carries ``out geom`` geometry;
    bbox containment only as the node/center fallback (no boundary to test).

    A bbox-only test let a diagonal/thin protected area "enclose" footage far
    outside its real boundary — and the winning name anchors the ENTIRE burned
    OST/title via Level 0 with no further check. The bbox stays as a cheap
    pre-reject (every ring is inside its own bounds).
    """
    s, w, n, e = bbox
    if not (s <= lat <= n and w <= lon <= e):
        return False
    outers, inners = _rings_of(elem)
    if not outers:
        return True  # node / center-only element: the bbox is all we have
    if any(_point_in_ring(lat, lon, ring) for ring in inners):
        return False
    return any(_point_in_ring(lat, lon, ring) for ring in outers)


def _segments_bbox(segs: list, margin_deg: float) -> Optional[tuple[float, float, float, float]]:
    lats = [float(s.gps_center_lat) for s in segs]
    lons = [float(s.gps_center_lon) for s in segs]
    return (
        min(lats) - margin_deg,
        min(lons) - margin_deg,
        max(lats) + margin_deg,
        max(lons) + margin_deg,
    )


def resolve_destination(
    featured_segments: Iterable[object],
    *,
    poi_source: Optional[POISource] = None,
    bbox_margin_deg: float = 0.02,
    require_sample_quorum: bool = False,
) -> Optional[str]:
    """Most-specific destination POI enclosing the featured (peak) footage.

    ``featured_segments`` are the LF's montage peaks (each carrying
    ``gps_center_lat``/``gps_center_lon`` and ``composite_score``).
    ``poi_source`` is an injectable ``bbox -> iterable[element]``; ``None``
    delegates to the chapters Overpass query (the only allowlisted network
    path, never invoked in tests). Returns the winning destination name or
    ``None``.

    SPECIFICITY over a superset region's raw vote: among candidates
    capturing at least ``MIN_PEAK_MASS_SHARE`` of the featured peak mass, the
    smallest (most specific) polygon wins, so an enclosed POI beats the broad
    region that necessarily outvotes it. ``require_sample_quorum`` (the shorts
    path, where each fix is one unit-score sample) makes that bar a HARD gate:
    if no candidate clears it, return ``None`` rather than name a feature the
    footage only grazed. With the flag off (the LF path) an unmet bar falls back
    to the plain highest-vote winner, so a scattered montage still resolves.
    """
    segs = [
        s
        for s in (featured_segments or [])
        if getattr(s, "gps_center_lat", None) is not None
        and getattr(s, "gps_center_lon", None) is not None
    ]
    if not segs:
        return None

    source = poi_source or _default_poi_source
    try:
        elements = list(source(_segments_bbox(segs, bbox_margin_deg)))
    except Exception as exc:  # noqa: BLE001 — degrade to the caller's cascade
        logger.warning("destination_resolver POI source failed: %s", exc)
        return None

    # Prominence vote: each featured segment adds its composite_score to every
    # destination POI whose polygon encloses it. ``meta`` keeps the smallest /
    # most-specific instance of each name for the tie-break.
    votes: dict[str, float] = {}
    meta: dict[str, tuple[float, int]] = {}  # name -> (area, specificity_rank)
    for elem in elements:
        tags = elem.get("tags") or {}
        name = (tags.get("name") or "").strip()
        rank = _classify(tags)
        bbox = _bbox_of(elem)
        if not name or rank is None or bbox is None:
            continue
        area = _bbox_area(bbox)
        if name not in meta or (area, rank) < meta[name]:
            meta[name] = (area, rank)
        for s in segs:
            if _encloses(elem, bbox, float(s.gps_center_lat), float(s.gps_center_lon)):
                votes[name] = votes.get(name, 0.0) + float(
                    getattr(s, "composite_score", 0.0) or 0.0
                )

    enclosing = {n: v for n, v in votes.items() if v > 0.0}
    if not enclosing:
        return None

    # A broad region encloses EVERY peak a nested POI does, so it
    # strictly outvotes the specific POI — the pre-fix prominence-first winner
    # anchored the OST on the superset region. Among candidates capturing at
    # least MIN_PEAK_MASS_SHARE of the featured mass, rank by SPECIFICITY
    # (smallest polygon → most-specific class), with prominence + name breaking
    # residual ties, so the enclosed POI wins instead of being outvoted.
    total_mass = sum(float(getattr(s, "composite_score", 0.0) or 0.0) for s in segs) or 1.0
    eligible = [n for n, v in enclosing.items() if v >= MIN_PEAK_MASS_SHARE * total_mass]
    if eligible:
        return min(eligible, key=lambda n: (meta[n][0], meta[n][1], -enclosing[n], n))
    if require_sample_quorum:
        # Shorts quorum: no POI encloses enough of the clip's own fixes —
        # a centroid-of-arc match with too little in-polygon footage. Do NOT
        # name a grazed feature; degrade to the caller's cascade.
        return None
    # LF fallback: nothing dominates the scattered montage — keep the
    # highest-prominence enclosing POI (pre-fix behaviour), never promote a
    # tiny grazed polygon just because it is the smallest.
    return min(enclosing, key=lambda n: (-enclosing[n], meta[n][0], meta[n][1], n))


def _default_poi_source(bbox: tuple[float, float, float, float]) -> list[dict]:
    """Production POI source: the chapters Overpass query (allowlisted host).

    Best-effort — any failure degrades to ``[]`` so the caller's cascade runs.
    """
    from .chapters.overpass_source import _query_overpass

    data = _query_overpass(bbox)
    if not data or "elements" not in data:
        return []
    return list(data["elements"])
