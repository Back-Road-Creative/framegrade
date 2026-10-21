"""Nominatim geocoding with rate limiting and session budget.

Part of the ChapterGenerator split (see ``chapters/generator.py``).
"""

import logging
from pathlib import Path
from typing import Optional

from gps_journey.core.gpx_parser import GPXParser

from ..forward_geocode import (  # noqa: F401 — re-exported under the old names
    MAX_GEOCODE_QUERY_LEN,
    _nominatim_places,
    _nominatim_results,
    _nominatim_search,
    _sanitize_geocode_query,
)
from ._shared import is_valid_chapter_placename

logger = logging.getLogger(__name__)

# Cap Nominatim calls when Overpass fails
MAX_GEOCODE_FALLBACK = 200


def _road_chapter_label(
    road: Optional[str],
    *,
    place_class: Optional[str] = None,
    match_type: Optional[str] = None,
) -> Optional[str]:
    """Display label for a road used as an interval-fill chapter, or ``None``.

    This string is burned verbatim as a superlong lower third and
    ``is_valid_chapter_placename`` does not screen it, so the drive-title
    path's road filter and ref splitter are reused rather than reimplemented:
    a road failing the quality gate yields ``None`` (omit, never stamp) and an
    OSM concurrency join collapses to its hyphenated primary member — that
    ``;`` chains drawtext filters and is not escaped. See pipeline-internals.
    """
    # Function-local: title_builder imports chapters._shared, so a module-level
    # import here would close the cycle.
    from ..title_builder import _road_passes_quality, _road_ref_members

    if not road or not str(road).strip():
        return None
    if not _road_passes_quality(road, place_class=place_class, match_type=match_type):
        return None
    members = _road_ref_members(str(road))
    return members[0] if members else None


class NominatimChapterSource:
    """Nominatim geocoding with a per-session request budget.

    Enforces a cap of MAX_SESSION_REQUESTS to prevent IP bans from
    excessive Nominatim usage in a single pipeline run.
    """

    MAX_SESSION_REQUESTS = 50

    def __init__(self, geocoder, cache=None):
        """
        Args:
            geocoder: NominatimRoadsClient instance for reverse geocoding
            cache: Optional GeocodingCache for lookup before hitting API
        """
        self.geocoder = geocoder
        self.cache = cache
        self._session_request_count = 0

    @property
    def session_request_count(self) -> int:
        """Number of Nominatim requests made in this session."""
        return self._session_request_count

    def reverse_geocode(self, latitude: float, longitude: float):
        """Reverse-geocode with session budget enforcement.

        Returns a Location or None. When the session budget is exhausted,
        attempts cache lookup only; if no cache hit, returns None.
        """
        if self._session_request_count >= self.MAX_SESSION_REQUESTS:
            logger.warning(
                "Nominatim session budget exhausted (%d requests)",
                self.MAX_SESSION_REQUESTS,
            )
            # Try cache if available
            if self.cache is not None:
                try:
                    return self.cache.get(latitude, longitude)
                except Exception as exc:
                    logger.warning(
                        "Nominatim cache lookup failed for (%.5f, %.5f) after "
                        "budget exhaustion; preferred reverse_geocode, falling "
                        "back to None: %s",
                        latitude,
                        longitude,
                        exc,
                    )
            return None

        self._session_request_count += 1
        return self.geocoder.reverse_geocode(latitude, longitude)

    def reverse_geocode_detailed(self, latitude: float, longitude: float) -> dict:
        """Detailed reverse-geocode with session budget enforcement.

        Returns a dict of address components. When budget is exhausted,
        returns an empty dict.
        """
        if self._session_request_count >= self.MAX_SESSION_REQUESTS:
            logger.warning(
                "Nominatim session budget exhausted (%d requests)",
                self.MAX_SESSION_REQUESTS,
            )
            # Serve from the `detailed` cache namespace (the one the roads
            # geocoder writes) before giving up — a warm cache still yields
            # road-name chapters instead of a silent drop.
            if self.cache is not None:
                try:
                    cached = self.cache.get_detailed(latitude, longitude)
                    if cached:
                        return cached
                except Exception as exc:
                    logger.warning(
                        "Nominatim detailed-cache lookup failed for (%.5f, %.5f) "
                        "after budget exhaustion; preferred "
                        "reverse_geocode_detailed, falling back to empty dict: %s",
                        latitude,
                        longitude,
                        exc,
                    )
            return {}

        self._session_request_count += 1
        return self.geocoder.reverse_geocode_detailed(latitude, longitude)

    def geocode_fallback(
        self,
        journey,
        gps_duration: float,
        original_duration: float,
        scale_gps_to_video,
        map_to_edited_time,
        is_valid_placename_fn,
    ) -> list:
        """Capped Nominatim geocoding fallback when Overpass yields too few results.

        Samples waypoints at evenly spaced marks of GPS time (at most one per
        30 s, at most half the remaining session budget) — picking the waypoint
        whose REAL timestamp is nearest each mark and carrying that waypoint's
        actual elapsed time (from the journey start) rather than a proportional
        index, so a variable sample rate or a dropout can't map a place name to
        a time the drive wasn't there — geocodes them, and deduplicates by
        city/locality name.

        Returns list of (edited_video_time, Location) tuples.
        """
        from gps_journey.core.models import Location

        # At most one sample per 30 s, and never more than half the requests the
        # session has left: the gate-floor top-up geocodes road labels from the
        # same budget, and an LF whose samples spent it all ships short of the
        # chapter floor. The samples spread across the whole drive, so a long LF
        # still gets names in its back half.
        sample_budget = max(0, self.MAX_SESSION_REQUESTS - self._session_request_count) // 2
        num_samples = min(int(gps_duration / 30), MAX_GEOCODE_FALLBACK, sample_budget)
        sample_interval = gps_duration / num_samples if num_samples else 0.0

        logger.info(f"Geocoding {num_samples} waypoints from {len(journey.waypoints)} total")

        start = journey.waypoints[0].timestamp if journey.waypoints else None
        timed = [((wp.timestamp - start).total_seconds(), wp) for wp in journey.waypoints]

        all_geocoded: list[tuple[float, Location]] = []
        for i in range(num_samples):
            target_time = i * sample_interval
            gps_time, waypoint = min(timed, key=lambda tw, t=target_time: abs(tw[0] - t))

            if waypoint.location:
                location = waypoint.location
            else:
                try:
                    location = self.reverse_geocode(waypoint.latitude, waypoint.longitude)
                except Exception as e:
                    logger.warning(
                        "Geocoding failed for waypoint (%.5f, %.5f); preferred "
                        "reverse_geocode, falling back to skipping this "
                        "waypoint in chapter set: %s",
                        waypoint.latitude,
                        waypoint.longitude,
                        e,
                    )
                    continue

            if location is None:
                continue

            original_time = scale_gps_to_video(gps_time, gps_duration, original_duration)
            video_time = map_to_edited_time(original_time)
            all_geocoded.append((video_time, location))

        return all_geocoded

    @staticmethod
    def filter_unique(
        geocoded: list,
        seen_keys: set,
        level: str,
        is_valid_placename_fn=None,
    ) -> list:
        """Filter geocoded waypoints by naming level, deduplicating."""
        results = []
        for video_time, location in geocoded:
            if level == "city":
                name = location.city
            else:
                name = location.locality or location.name or location.suburb
            if not name:
                continue
            if is_valid_placename_fn and not is_valid_placename_fn(name):
                continue
            key = (name, location.county)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            results.append((video_time, location))
        return results

    def fill_interval_chapters(
        self,
        named: list,
        all_geocoded: list,
        video_duration: float,
        target: int,
        used_labels: Optional[set] = None,
    ) -> list:
        """Fill gaps with evenly-spaced interval chapters using coordinates."""
        from gps_journey.core.models import Location

        if not all_geocoded:
            return named

        interval = video_duration / (target + 1)
        target_times = [interval * (i + 1) for i in range(target)]

        named_times = {t for t, _ in named}
        needed_times = [t for t in target_times if not any(abs(t - nt) < 60 for nt in named_times)]

        if used_labels is None:
            used_labels = set()
            for _, loc in named:
                label = loc.city or loc.locality or loc.name or loc.suburb or ""
                if label:
                    used_labels.add(label)

        added = []
        for target_time in needed_times:
            closest = min(all_geocoded, key=lambda x: abs(x[0] - target_time))
            real_time, loc = closest

            place_name = loc.city or loc.locality or loc.name or loc.suburb
            if (
                place_name
                and place_name not in used_labels
                and is_valid_chapter_placename(place_name)
            ):
                used_labels.add(place_name)
                added.append((real_time, loc))
            else:
                road_name = None
                detailed: dict = {}
                try:
                    detailed = self.reverse_geocode_detailed(loc.latitude, loc.longitude) or {}
                    road_name = detailed.get("road")
                except Exception:
                    pass

                label = _road_chapter_label(
                    road_name,
                    place_class=detailed.get("place_class"),
                    match_type=detailed.get("match_type"),
                )
                if label and label not in used_labels:
                    used_labels.add(label)
                    synthetic = Location(
                        latitude=loc.latitude,
                        longitude=loc.longitude,
                        name=label,
                        county=loc.county,
                        country=loc.country,
                    )
                    added.append((real_time, synthetic))

        result = sorted(named + added, key=lambda x: x[0])
        logger.info(
            f"Filled to {len(result)} chapters ({len(named)} named + {len(added)} interval)"
        )
        return result


def detect_gps_date_for_location(
    gpx_path: Path,
    location_str: str,
    geocoder=None,
) -> Optional[str]:
    """
    For multi-day GPX tracks, find which date's waypoints best match the location.

    Forward-geocodes the location string to get a center coordinate, then checks
    each date in the GPX to find which has the most waypoints within 50km of that
    center.

    Returns YYYY-MM-DD string or None if no match found.
    """
    from collections import defaultdict

    from ._journey_filter import local_day
    from .overpass_source import _haversine

    # Forward-geocode the location to get a center point
    coords = _nominatim_search(location_str)
    if coords is None:
        logger.warning(f"Could not geocode '{location_str}' for GPS date detection")
        return None
    center_lat, center_lon = coords

    # Parse GPX
    parser = GPXParser()
    try:
        journey = parser.parse(gpx_path)
    except Exception:
        return None

    if not journey.waypoints:
        return None

    # Check if multi-day -- if single LOCAL day, no need to filter. Bucket by
    # longitude-local day so a drive crossing UTC midnight isn't mistaken for
    # two days (which would drop the post-midnight half via the majority vote).
    dates = {local_day(wp.timestamp, wp.longitude) for wp in journey.waypoints}
    if len(dates) <= 1:
        return None

    # Count waypoints within 50km of location center, per local day
    match_radius_m = 50_000
    date_counts: dict[str, int] = defaultdict(int)
    for wp in journey.waypoints:
        dist = _haversine(center_lat, center_lon, wp.latitude, wp.longitude)
        if dist <= match_radius_m:
            date_counts[local_day(wp.timestamp, wp.longitude).strftime("%Y-%m-%d")] += 1

    if not date_counts:
        logger.warning(
            f"No GPS waypoints within 50km of '{location_str}' ({center_lat:.2f}, {center_lon:.2f})"
        )
        return None

    # Return the date with most matching waypoints
    best_date = max(date_counts, key=date_counts.get)
    logger.info(
        f"GPS date detection: '{location_str}' -> {best_date} "
        f"({date_counts[best_date]} waypoints within 50km, "
        f"{len(date_counts)} candidate dates)"
    )
    return best_date
