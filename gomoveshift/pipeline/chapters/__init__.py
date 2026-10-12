"""YouTube Chapter Generator package.

Re-exports all public symbols from the original monolithic chapters.py
so that ``from gomoveshift.pipeline.chapters import X`` continues to work.
"""

from gps_journey.geocoding.nominatim_roads import NominatimRoadsClient

from ._shared import _KNOWN_DISPLAY_NAMES, is_valid_chapter_placename
from .generator import ChapterGenerator
from .marker_merger import MarkerMerger
from .nominatim_source import (
    MAX_GEOCODE_FALLBACK,
    NominatimChapterSource,
    _nominatim_search,
    detect_gps_date_for_location,
)
from .overpass_source import (
    OVERPASS_ENDPOINTS,
    OVERPASS_RETRY_DELAY,
    OVERPASS_TIMEOUT,
    OverpassChapterSource,
    _bbox_from_waypoints,
    _cache_path_for_bbox,
    _haversine,
    _query_overpass,
)

__all__ = [
    "ChapterGenerator",
    "NominatimRoadsClient",
    "MarkerMerger",
    "NominatimChapterSource",
    "OverpassChapterSource",
    "OVERPASS_ENDPOINTS",
    "OVERPASS_TIMEOUT",
    "OVERPASS_RETRY_DELAY",
    "MAX_GEOCODE_FALLBACK",
    "_KNOWN_DISPLAY_NAMES",
    "_bbox_from_waypoints",
    "_cache_path_for_bbox",
    "_haversine",
    "_nominatim_search",
    "_query_overpass",
    "detect_gps_date_for_location",
    "is_valid_chapter_placename",
]
