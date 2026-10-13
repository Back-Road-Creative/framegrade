"""Forward geocoding (Nominatim search) for the media engine.

The one HTTP path behind the (name) -> (lat, lon) and (name) -> settlement
candidates lookups. They live in the engine because Phase B/C and the legacy
shorts path need them, and ``chapters`` is a publishing package the engine must
not import (``tests/test_media_boundary.py``); ``chapters.nominatim_source`` re-imports
them under their old names.
"""

import json
import logging
import time
import urllib.error
import urllib.request  # noqa: F401 — used as urllib.request.Request/urlopen
from typing import Optional

logger = logging.getLogger("gomoveshift.pipeline.chapters.nominatim_source")

# Bound on the operator-supplied forward-geocode query (a place name is never
# anywhere near this long; the cap defends against a pathological input).
MAX_GEOCODE_QUERY_LEN = 200


def _sanitize_geocode_query(location_str: str) -> str:
    """Strip control characters and cap length of a forward-geocode query.

    The result is still URL-encoded by the caller; this only removes bytes that
    have no business in a place name (newlines, NUL, other control chars) so a
    hostile location string can't smuggle them into the request.
    """
    cleaned = "".join(ch for ch in location_str if ch.isprintable())
    return cleaned.strip()[:MAX_GEOCODE_QUERY_LEN]


def _nominatim_results(location_str: str, limit: int) -> Optional[list]:
    """Raw Nominatim search hits for a location string, best match first.

    The one HTTP path behind both forward geocoders below. ``None`` when every
    attempt failed on the network; a malformed body (``ValueError``) propagates.
    """
    import urllib.parse

    params = urllib.parse.urlencode(
        {
            "q": _sanitize_geocode_query(location_str),
            "format": "json",
            "limit": str(limit),
        }
    )
    url = f"https://nominatim.openstreetmap.org/search?{params}"

    for attempt in range(2):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "gomoveshift-video/0.1.0",
                },
            )
            time.sleep(1.0)  # OSM rate limit
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            logger.warning(f"Nominatim attempt {attempt + 1}/2 failed for '{location_str}': {exc}")

    return None


def _nominatim_search(location_str: str) -> Optional[tuple[float, float]]:
    """
    Forward-geocode a location string via Nominatim search API.

    Returns (latitude, longitude) or None. Uses direct HTTP -- no dependency
    on NominatimRoadsClient which only supports reverse geocoding.
    """
    results = _nominatim_results(location_str, 1)
    if not results:
        return None
    # KeyError/ValueError here means malformed response -- let it propagate
    return float(results[0]["lat"]), float(results[0]["lon"])


# Nominatim result classes that are settlements (village/town/hamlet/locality).
_SETTLEMENT_CLASSES = frozenset({"place"})
# A settlement Nominatim models only as an administrative boundary (a US
# census-designated town) carries its settlement kind in ``addresstype``; a
# district or county boundary of the same name does not.
_SETTLEMENT_ADDRESSTYPES = frozenset({"city", "town", "village", "hamlet"})


def _nominatim_places(location_str: str, limit: int = 5) -> list[tuple[str, float, float]]:
    """Settlement candidates ``(name, lat, lon)`` for a location string.

    The top search hit is not always the settlement: for "Goleen, County Cork"
    it is the electoral-division boundary, whose centroid sits 2.7 km from the
    village. Callers that need *the town* get every settlement-class hit and
    pick by their own evidence (e.g. distance to a GPS trace). A ``boundary``
    hit counts only when its ``addresstype`` is a settlement kind; no other
    class does, whatever its ``addresstype``. ``[]`` on no
    hits or network failure; a hit with a blank ``name`` falls back to the
    first ``display_name`` component.
    """
    out: list[tuple[str, float, float]] = []
    for hit in _nominatim_results(location_str, limit) or []:
        is_place = hit.get("class") in _SETTLEMENT_CLASSES
        is_boundary_town = (
            hit.get("class") == "boundary" and hit.get("addresstype") in _SETTLEMENT_ADDRESSTYPES
        )
        if not (is_place or is_boundary_town):
            continue
        name = (hit.get("name") or str(hit.get("display_name") or "").split(",")[0]).strip()
        if not name:
            continue
        out.append((name, float(hit["lat"]), float(hit["lon"])))
    return out


def forward_geocode(town: str):
    """Defensive forward-geocode for the ladder's endpoint-proximity check.

    Wraps ``_nominatim_search`` (the one (name)->(lat, lon) geocoder in the
    codebase) so any failure — import, network, malformed response — degrades
    to ``None`` and the gate skips the check instead of crashing pre-encode.
    """
    try:
        return _nominatim_search(town)
    except Exception:  # noqa: BLE001 — a geocoder failure never blocks an encode
        return None


def forward_geocode_places(query: str) -> list:
    """Defensive settlement-candidate lookup for trace-verifying label endpoints.

    Wraps ``_nominatim_places`` so any failure degrades to ``[]`` — nothing
    verified, and the location invariant then fails closed exactly as before.
    The failure is logged so it is not mistaken for "no settlement of that name".
    """
    try:
        return _nominatim_places(query)
    except Exception as exc:  # noqa: BLE001 — unverified, never a crash
        logger.warning("TRACE-UNVERIFIED-PLACE: %r — geocoder failed (%s)", query, exc)
        return []
