"""GPS-to-video timestamp scaling, marker/chapter merge, and deduplication.

Part of the ChapterGenerator split (see ``chapters/generator.py``).
"""

import logging
from typing import Optional

from .overpass_source import _haversine

logger = logging.getLogger(__name__)

# A chapter town gets the ", {route}" suffix only when it sits within this
# radius of a point where the GPS track actually ran along the detected route.
# On-route waypoints hug the route (<= ROUTE_MATCH_RADIUS_M ~1 km); towns beside
# the route are matched within a few km of the track, so ~5 km keeps genuine
# beside-route towns while excluding towns on a far leg the drive merely crossed.
ROUTE_SUFFIX_RADIUS_M = 5_000

# Decoration suffixes for terrain bands. Keep total chapter
# title under ~50 chars; the merger never decorates two same-band chapters
# in a row, so suffixes never compound.
_TERRAIN_BAND_SUFFIX = {
    "mountain": " — Mountain Climb",
    "motorway": " — Motorway",
    "coastal": " — Coastal",
}


class MarkerMerger:
    """Merge DaVinci Resolve markers with GPS chapters.

    Handles GPS-to-video timestamp scaling, piecewise time mapping
    for stop-removed videos, and deduplication of chapter names.
    """

    def __init__(
        self,
        named_routes: Optional[list[str]] = None,
        road_coverage_measured: bool = False,
    ):
        """
        Args:
            named_routes: List of detected named routes for chapter enrichment
            road_coverage_measured: the route suffix ships only when road
                coverage is measured over retained time
        """
        self.named_routes = named_routes or []
        self.road_coverage_measured = road_coverage_measured
        # (lat, lon) of the sampled waypoints where the drive actually ran along
        # named_routes[0]; proximity-gates the per-chapter route suffix.
        self.route_waypoints: list[tuple[float, float]] = []

    @staticmethod
    def scale_gps_to_video(gps_time: float, gps_duration: float, video_duration: float) -> float:
        """
        Scale GPS timestamp to original video timestamp (linear).

        Args:
            gps_time: Time in seconds from GPS start
            gps_duration: Total GPS duration in seconds
            video_duration: Original video duration in seconds

        Returns:
            Scaled original video time in seconds
        """
        scale_factor = video_duration / gps_duration
        return gps_time * scale_factor

    @staticmethod
    def map_to_edited_time(
        original_time: float, keep_segments: Optional[list[tuple[float, float]]] = None
    ) -> float:
        """
        Map original video time to edited (post-stop-removal) video time.

        Uses the keep_segments list to compute the piecewise mapping.
        Each keep_segment is a (start, end) in original video time.
        In the edited video, these segments are concatenated without gaps.

        Falls back to identity (no adjustment) when keep_segments is None.
        """
        if not keep_segments:
            return original_time

        elapsed = 0.0
        for seg_start, seg_end in keep_segments:
            if original_time <= seg_start:
                # Point is in a removed stop -- snap to end of previous segment
                return elapsed
            elif original_time <= seg_end:
                # Point is within this kept segment
                return elapsed + (original_time - seg_start)
            else:
                # Point is after this segment -- accumulate its duration
                elapsed += seg_end - seg_start

        # Past all segments -- return total edited duration
        return elapsed

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """
        Format seconds as YouTube timestamp (M:SS or H:MM:SS).

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string
        """
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60

        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"

    def merge_markers_with_chapters(
        self,
        waypoints_with_towns,
        markers: Optional[list[dict]],
        video_duration: float,
        start_location: str,
        end_location: str,
        format_location_fn=None,
        terrain_bands: Optional[list[Optional[str]]] = None,
    ) -> list[dict]:
        """Merge DaVinci Resolve markers with GPS chapters into final list.

        Markers take priority: GPS chapters within 30s of a marker are
        suppressed. Enforces minimum 60s gap between intermediate chapters
        and adds start/end chapters.

        Args:
            waypoints_with_towns: list of (video_time, Location) tuples
            markers: Optional list of marker dicts with 'sec' and 'name' keys
            video_duration: Total video duration in seconds
            start_location: Starting location string
            end_location: Ending location string
            format_location_fn: Callable(Location) -> str for formatting
        """
        # Build marker chapters (from DaVinci Resolve timeline markers)
        marker_chapters: list[tuple[float, str]] = []
        if markers:
            for m in markers:
                m_sec = m.get("sec", 0)
                m_name = m.get("name", "").strip()
                if m_name and 0 < m_sec < video_duration:
                    marker_chapters.append((m_sec, m_name))
            if marker_chapters:
                logger.info(f"Using {len(marker_chapters)} marker(s) as chapter anchors")

        # Build chapter list
        chapters = []

        # Start chapter -- use first marker if at/near 0:00, else use location
        if marker_chapters and marker_chapters[0][0] < 30:
            chapters.append(
                {
                    "timestamp": "0:00",
                    "title": marker_chapters[0][1],
                }
            )
            marker_chapters = marker_chapters[1:]  # consume it
        else:
            chapters.append(
                {
                    "timestamp": "0:00",
                    "title": f"Start - {start_location}",
                }
            )

        # Merge marker chapters and GPS chapters
        marker_times = {t for t, _ in marker_chapters}
        last_chapter_time = 0.0

        # Build combined list: markers + GPS, sorted by time.
        # Tuple shape: (time, title, source, band_or_None).
        combined: list[tuple[float, str, str, Optional[str]]] = []
        for m_time, m_title in marker_chapters:
            combined.append((m_time, m_title, "marker", None))
        # Named route suffix for chapter enrichment
        _route_suffix = ""
        if self.named_routes and self.road_coverage_measured:
            _route_suffix = f", {self.named_routes[0]}"

        bands = terrain_bands or [None] * len(waypoints_with_towns)
        for idx, (video_time, loc) in enumerate(waypoints_with_towns):
            if format_location_fn is not None:
                town_name = format_location_fn(loc)
            else:
                town_name = getattr(loc, "city", None) or getattr(loc, "name", None) or ""
            if not town_name:
                continue
            # Suppress GPS chapter if within 30s of any marker
            if any(abs(video_time - mt) < 30 for mt in marker_times):
                continue
            title = f"Entering {town_name}"
            if _route_suffix and town_name not in _route_suffix and self._on_named_route(loc):
                title += _route_suffix
            band = bands[idx] if idx < len(bands) else None
            combined.append((video_time, title, "gps", band))
        combined.sort(key=lambda x: x[0])

        # Add intermediate chapters. Terrain bands decorate the title only
        # when the band differs from the previously emitted chapter's band
        # — prevents 12-in-a-row "Mountain Climb" repetition.
        prev_band: Optional[str] = None
        for video_time, title, _source, band in combined:
            # A real chapter inside the 30s head/tail dead-zone used to be
            # dropped silently. Clamp it inward to a renderable slot that still
            # clears the 60s gap from the Start (0:00) / Finish chapters instead
            # of vanishing; the 60s-gap rule below then dedups it normally.
            if video_time < 60:
                video_time = 60.0
            elif video_time > (video_duration - 30):
                video_time = max(60.0, video_duration - 45)
            # Enforce minimum 60s gap between intermediate chapters
            if video_time - last_chapter_time < 60:
                continue
            decorated = title
            if band and band != prev_band and band in _TERRAIN_BAND_SUFFIX:
                decorated = title + _TERRAIN_BAND_SUFFIX[band]
            chapters.append(
                {
                    "timestamp": self.format_timestamp(video_time),
                    "title": decorated,
                }
            )
            last_chapter_time = video_time
            prev_band = band

        # End chapter -- place 15s before end so YouTube displays it
        finish_time = max(0, video_duration - 15)
        # Don't overlap with last intermediate chapter
        if last_chapter_time > 0 and finish_time - last_chapter_time < 30:
            finish_time = last_chapter_time + 30
        if finish_time < video_duration:
            chapters.append(
                {
                    "timestamp": self.format_timestamp(finish_time),
                    "title": f"Finish - {end_location}",
                }
            )

        return self._collapse_consecutive_duplicates(chapters)

    def _on_named_route(self, loc) -> bool:
        """True when ``loc`` sits within ROUTE_SUFFIX_RADIUS_M of a point where the
        drive actually ran along the detected route.

        Gates the per-chapter route suffix so a town on a leg far from the
        route (the drive merely crossed it) isn't labeled as on it. With no
        on-route waypoints threaded (``route_waypoints`` empty), no chapter is
        labeled — the gate never over-claims.
        """
        lat = getattr(loc, "latitude", None)
        lon = getattr(loc, "longitude", None)
        if lat is None or lon is None or not self.route_waypoints:
            return False
        return any(
            _haversine(lat, lon, rlat, rlon) <= ROUTE_SUFFIX_RADIUS_M
            for rlat, rlon in self.route_waypoints
        )

    @staticmethod
    def _collapse_consecutive_duplicates(chapters: list[dict]) -> list[dict]:
        """Drop any chapter whose title equals the previous one's.

        Two adjacent waypoints geocoding to the same town (or a marker
        repeating a GPS town name) produced consecutive identical titles. The
        earlier timestamp is kept; the duplicate is dropped.
        """
        collapsed: list[dict] = []
        for ch in chapters:
            if collapsed and _strip_chapter_decorations(
                str(collapsed[-1].get("title") or "")
            ) == _strip_chapter_decorations(str(ch.get("title") or "")):
                continue
            collapsed.append(ch)
        return collapsed


def _strip_chapter_decorations(title: str) -> str:
    """A chapter title with this module's own decorations removed — the terrain
    band suffix and the route suffix — so ONE helper defines "the same place"
    for every consumer.

    The route suffix drop is STRUCTURAL (last comma-part, only when it passes
    the road-quality gate). The old comma-counting ``parts[:2]`` assumed
    ``_format_location`` always contributes a county; it omits one whenever
    ``county`` is falsy, and the route then burned in the county slot
    (``"Yemassee, US-17"``). A real county never passes the road gate.
    """
    from ..title_builder import _road_passes_quality

    label = title[len("Entering ") :] if title.startswith("Entering ") else title
    for suffix in _TERRAIN_BAND_SUFFIX.values():
        if label.endswith(suffix):
            label = label[: -len(suffix)]
            break
    parts = [p.strip() for p in label.split(",")]
    if len(parts) > 1 and _road_passes_quality(parts[-1]):
        parts = parts[:-1]
    return ", ".join(parts[:2]).strip()


def chapter_town_events(chapters: Optional[list]) -> list[tuple[str, float]]:
    """Persisted superlong-part chapters -> ``(place_label, event_time_sec)``
    recurring LOCATION lower-third events for ``apply_ost``'s ``lower_thirds``.

    Keeps ONLY real town-transition chapters — the ``Entering <town>`` form
    built above by ``merge_markers_with_chapters`` — stripping the per-chapter
    route suffix (``, <route>``) and terrain-band decoration (the SAME
    ``_TERRAIN_BAND_SUFFIX`` this module burns, never re-literalled). Excludes
    the opener/end-card ``Start -``/``Finish -`` chapters (the OST opener/CTA
    already carry those) and the evenly-spaced SYNTHETIC ``En route`` padding
    chapters (``ChapterGenerator._pad_to_min_chapters``) — those exist only to
    satisfy YouTube's renderability minimum at fabricated times, never a real
    transition, so they never start with ``"Entering "`` and are excluded by
    construction. Never raises: a malformed chapter is skipped; an empty/
    None/garbage list returns ``[]``.

    Label reconstruction is comma-COUNTING, not a first-comma split:
    ``ChapterGenerator._format_location`` contributes AT MOST one comma
    (``name`` or ``name, county``) and the route suffix appends EXACTLY one
    more (``, <route>``) after it — so the first two comma-parts are always
    the place label (``Town`` or ``Town, County``) and anything from the
    third part on is the route suffix. A first-comma split would wrongly
    drop the county off every ``Town, County`` label.
    """
    from .generator import ChapterGenerator

    events: list[tuple[str, float]] = []
    for chapter in chapters or []:
        try:
            title = str(chapter.get("title") or "")
            if not title.startswith("Entering "):
                continue
            label = _strip_chapter_decorations(title)
            if not label:
                continue
            seconds = float(ChapterGenerator._parse_timestamp(chapter.get("timestamp", "")))
        except Exception:  # noqa: BLE001 -- one bad chapter must not fail the part
            continue
        # Adjacent duplicates: the upstream collapse compares DECORATED titles,
        # so two chapters differing only by a band/route decoration reach here
        # and reduce to the same burned label. Keep the earlier one.
        if events and events[-1][0] == label:
            continue
        events.append((label, seconds))
    return events


# YouTube drops a chapter list whose entries sit closer than this.
MIN_CHAPTER_GAP_S = 10
INSERT_MARKER_TITLE = "\U0001f4f8 Photo stop"


def _timestamp_seconds(stamp: str) -> int:
    seconds = 0
    for part in str(stamp).split(":"):
        seconds = seconds * 60 + int(part)
    return seconds


def add_insert_markers(chapters: list[dict], insert_times_s: list[float]) -> list[dict]:
    """``chapters`` plus a marker at each insert, in time order.

    A marker is skipped when it would sit within ``MIN_CHAPTER_GAP_S`` of any chapter
    or earlier marker (YouTube would drop the whole list), so a burst of photos shows
    as one stop. ``chapters`` is not mutated.
    """
    out = [dict(c) for c in chapters]
    taken = [_timestamp_seconds(c["timestamp"]) for c in out]
    for at in sorted(float(t) for t in insert_times_s):
        sec = int(at)
        if all(abs(sec - t) >= MIN_CHAPTER_GAP_S for t in taken):
            out.append(
                {"timestamp": MarkerMerger.format_timestamp(sec), "title": INSERT_MARKER_TITLE}
            )
            taken.append(sec)
    return sorted(out, key=lambda c: _timestamp_seconds(c["timestamp"]))
