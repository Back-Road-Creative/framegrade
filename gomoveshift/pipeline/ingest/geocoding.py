"""GPS-based location auto-derivation.

Reverse-geocodes GPS waypoints from GPX files to determine drive locations.
Uses the gps_journey library's NominatimRoadsClient with a shared geocoding cache.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def derive_location_from_gpx(gpx_path: Path, *, photo_mode: bool = False) -> str:
    """Reverse-geocode multiple GPS waypoints and majority-vote on location.

    Samples up to 5 evenly-spaced waypoints, geocodes each, and uses a
    majority vote on (state, country) to derive the canonical location.

    Uses the existing NominatimRoadsClient from gps_journey with the
    shared geocoding cache.

    Args:
        photo_mode: opt-in for EXIF-derived photo fixes, where every waypoint
            is a deliberate shutter press at a real place rather than a
            continuous recording — so every fix votes and none is sampled
            away or skipped inward. The default (GPX/video) path is unchanged.

    Returns:
        Location string like "County Clare, Ireland" or empty string on failure.
    """
    try:
        from collections import Counter

        from gps_journey.core.gpx_parser import GPXParser
        from gps_journey.geocoding.cache import GeocodingCache
        from gps_journey.geocoding.nominatim_roads import NominatimRoadsClient

        from gomoveshift.config import DriveConfig

        parser = GPXParser()
        journey = parser.parse(gpx_path)

        if not journey.waypoints:
            return ""

        # ONE canonical geocode-cache namespace, shared with Phase BC / D
        # (DriveConfig.resolved_geocode_cache_dir) so a coordinate resolves
        # identically everywhere in a run; still honors GMS_GPS_CACHE_DIR.
        cache_dir = DriveConfig().resolved_geocode_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = GeocodingCache(cache_path=cache_dir / "geocoding_cache.json", ttl_days=90)
        geocoder = NominatimRoadsClient(
            user_agent="gomoveshift-video/0.1.0",
            cache=cache,
            rate_limit=1.0,
        )

        # Sample up to 5 evenly-spaced waypoints. Endpoint samples use
        # _pick_anchor_waypoint (slightly inward) so a stale previous-session
        # GPS fix at waypoint 0 / n-1 can't vote in the drive-wide label.
        from .ost_formatter import _pick_anchor_waypoint

        wps = journey.waypoints
        n = len(wps)
        # Photo mode has no stale clip-boundary fix to dodge, so the inward
        # skip would only disenfranchise the first and last photo of a set.
        if photo_mode or n <= 5:
            sample_wps = list(wps)
        else:
            sample_wps = [
                _pick_anchor_waypoint(wps, from_end=False),
                wps[n // 4],
                wps[n // 2],
                wps[3 * n // 4],
                _pick_anchor_waypoint(wps, from_end=True),
            ]

        votes: list[tuple[str, str]] = []
        for wp in sample_wps:
            location = geocoder.reverse_geocode(wp.latitude, wp.longitude)
            if location:
                state = location.state or location.county or ""
                country = location.country or ""
                if state and country:
                    votes.append((state, country))

        if not votes:
            return ""

        # Majority vote on (state, country)
        counter = Counter(votes)
        (winning_state, winning_country), _count = counter.most_common(1)[0]
        loc_str = f"{winning_state}, {winning_country}"
        logger.info(f"  Auto-derived location (majority vote, {len(votes)} samples): {loc_str}")
        return loc_str

    except Exception as e:
        logger.warning(f"  Location auto-derivation failed: {type(e).__name__}: {e}")

    return ""


def _get_shared_geocoder():
    """Return a NominatimRoadsClient with persistent cache."""
    from gps_journey.geocoding.cache import GeocodingCache
    from gps_journey.geocoding.nominatim_roads import NominatimRoadsClient as _NomClient

    from gomoveshift.config import DriveConfig

    # ONE canonical geocode-cache namespace, shared with Phase BC / D
    # (DriveConfig.resolved_geocode_cache_dir); still honors GMS_GPS_CACHE_DIR.
    cache_dir = DriveConfig().resolved_geocode_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = GeocodingCache(cache_path=cache_dir / "geocoding_cache.json", ttl_days=90)
    return _NomClient(
        user_agent="gomoveshift-video/0.1.0",
        cache=cache,
        rate_limit=1.0,
    )


def _get_route_analyzer(geocoder=None):
    """Return a RouteAnalyzer using the shared geocoder."""
    from gps_journey.analysis.route_analyzer import RouteAnalyzer

    if geocoder is None:
        geocoder = _get_shared_geocoder()
    return RouteAnalyzer(geocoder)
