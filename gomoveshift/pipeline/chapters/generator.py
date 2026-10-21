"""YouTube Chapter Generator orchestrator.

Coordinates OverpassChapterSource, NominatimChapterSource, and MarkerMerger
to generate YouTube chapters from GPS data.
"""

import hashlib
import json
import logging
import math
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from gps_journey.core.gpx_parser import GPXParser
from gps_journey.core.models import Journey, Location, Waypoint, to_local
from gps_journey.geocoding.cache import GeocodingCache
from gps_journey.geocoding.nominatim_roads import NominatimRoadsClient

from ...config import Config
from ._journey_filter import load_journey_and_origin
from ._shared import is_valid_chapter_placename
from .marker_merger import MarkerMerger
from .nominatim_source import NominatimChapterSource
from .overpass_source import OverpassChapterSource

# YouTube only renders a description chapter block when it has at least 3
# timestamps, the first at 0:00, and each at least 10s apart. The sparse-GPS
# basic fallback emits only 2 (Start/Finish), which YouTube silently refuses
# to render — so the gate (``_pad_to_min_chapters``) pads such a block.
YOUTUBE_MIN_CHAPTERS = 3
YOUTUBE_MIN_CHAPTER_GAP_SEC = 10

# Superlong chapter profile (LOAD-BEARING): validation.content_gates.
# check_metadata_quality requires >= max(1, int(video_duration/60/15))
# intermediate chapters for any GPS-bearing video over 600s — at the 4h
# superlong-part cap that's 16, while the default max_chapters=15 (cap-2 for
# Start/Finish) tops out at 13. ``generate(max_chapters=...)`` already
# accepts a per-call override on top of the constructor default, so a
# superlong caller passes ``superlong_max_chapters(video_duration)`` instead
# of widening the gate itself (more room to place chapters, never a
# rewritten gate). ``+2`` accounts for the Start/Finish
# chapters the gate's intermediate count always excludes.
SUPERLONG_CHAPTER_INTERVAL_SEC = 900  # 15 min, matches the gate's own /15 divisor


def superlong_max_chapters(video_duration_sec: float) -> int:
    """``max_chapters`` for a superlong part's :meth:`ChapterGenerator.generate`
    call, sized so the intermediate-chapter floor in
    ``content_gates.check_metadata_quality`` is clearable by construction."""
    return max(1, math.ceil(video_duration_sec / SUPERLONG_CHAPTER_INTERVAL_SEC)) + 2


# Metadata-gate chapter floor (LOAD-BEARING): validation.content_gates.
# check_metadata_quality imports both helpers below, so the generator sizes its
# chapters against the gate's own floor and count — they can't drift apart.
GATE_CHAPTER_MIN_VIDEO_SEC = 600


def min_intermediate_chapters(video_duration_sec: float) -> int:
    """Middle (non Start/Finish) chapters a GPS-bearing video needs to pass
    ``content_gates.check_metadata_quality``: none up to 10 min, then one per
    15 min, at least one."""
    if not video_duration_sec or video_duration_sec <= GATE_CHAPTER_MIN_VIDEO_SEC:
        return 0
    return max(1, int(video_duration_sec / 60 / 15))


def intermediate_chapters(chapters: list[dict]) -> list[dict]:
    """The chapters the gate's floor counts: every one but Start/Finish."""
    return [c for c in chapters if not c.get("title", "").startswith(("Start", "Finish"))]


def build_day_master_journey(
    gps_points: Optional[list[tuple[float, float, float]]],
) -> Optional[Journey]:
    """Build a ``Journey`` whose waypoint timestamps ARE day-master seconds.

    ``gps_points`` is ``[(t_master, lat, lon), ...]`` — GPS trackpoints
    already projected onto the day master's own 0-based timeline (see
    ``gps_enrichment._build_gps_points_by_date``), spanning every episode
    of the day, not just one. Anchoring each waypoint at an arbitrary
    epoch plus ``t_master`` seconds means the elapsed seconds between any
    two waypoints of the returned journey equal their ``t_master`` gap —
    the same second-0 basis ``ChapterGenerator.generate``'s
    ``included_segments`` expects, so a day-master
    ``SuperlongPartPlan.ranges`` window lines up with this journey with no
    per-episode rebasing. Returns ``None`` for missing/empty input (caller
    falls back to per-episode GPX chapters).
    """
    if not gps_points:
        return None
    ordered = sorted(gps_points, key=lambda p: p[0])
    epoch = datetime(1970, 1, 1)
    waypoints = tuple(
        Waypoint(latitude=lat, longitude=lon, timestamp=epoch + timedelta(seconds=t))
        for t, lat, lon in ordered
    )
    return Journey(
        start_date=waypoints[0].timestamp, end_date=waypoints[-1].timestamp, waypoints=waypoints
    )


def _longitude_local_start(journey) -> Optional[datetime]:
    """Journey start shifted to longitude-local solar time.

    GPX timestamps are UTC, but the recording DATE viewers (and the ``📅 Filmed:``
    line) mean is the drive's LOCAL calendar day. The canonical ``to_local``
    (one hour per 15° of longitude) — the same shift ``day_grouper`` /
    ``_journey_filter.local_day`` use — so a US evening drive stamps the right day
    (and month), not the next UTC day. Returns the raw start unchanged when no
    first-fix longitude is available.
    """
    start = getattr(journey, "start_date", None)
    waypoints = getattr(journey, "waypoints", None) or ()
    if start is None or not waypoints:
        return start
    lon = getattr(waypoints[0], "longitude", None)
    if lon is None:
        return start
    return to_local(start, lon)


# Terrain-band thresholds.
_MOUNTAIN_GAIN_PER_KM = 30.0  # m/km elevation gain → "Mountain Climb"
# Net-elevation-range floor for the "mountain" band. The gain_per_km metric
# above SUMS every positive consecutive-waypoint delta, so on a flat drive GPS
# elevation noise (±a few metres × hundreds of fixes) accumulates into a large
# bogus "gain" and a flat coastal chapter gets mis-tagged "Mountain Climb". A
# real mountain climb has a large NET elevation change (max−min over the slice);
# flat-with-noise has a near-zero net even when the summed-positive is large. We
# therefore require BOTH the per-km gain AND a meaningful net range before
# classifying "mountain". 100 m is conservative — a true climb clears it easily
# while ±few-metre noise cannot.
_MOUNTAIN_MIN_NET_RANGE_M = 100.0  # max−min elevation over slice → "Mountain Climb"
_MOTORWAY_MEDIAN_MPS = 27.0  # ≈ 60 mph → "Motorway"
_COASTLINE_PROXIMITY_M = 1000.0  # OSM natural=coastline within 1 km → "Coastal"
_MOTORWAY_MAX_GAIN_PER_KM = 10.0  # "elevation-flat" guard for motorway band


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in meters (duplicated locally to avoid cycle)."""
    earth_r = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return earth_r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _classify_chapter_band(
    journey_waypoints,
    wp_video_times: list[float],
    chapter_start_video_t: float,
    chapter_end_video_t: float,
    coastline_points: list[tuple[float, float]],
) -> Optional[str]:
    """Classify a chapter slice → 'mountain' / 'motorway' / 'coastal' / None.
    Priority: mountain > motorway > coastal > scenic. Missing data → None."""
    slice_wps = [
        wp
        for wp, vt in zip(journey_waypoints, wp_video_times, strict=False)
        if chapter_start_video_t <= vt <= chapter_end_video_t
    ]
    if len(slice_wps) < 2:
        return None

    distance_m = elevation_gain_m = 0.0
    has_elevation = False
    speeds: list[float] = []
    min_elev: Optional[float] = None
    max_elev: Optional[float] = None
    for prev, curr in zip(slice_wps, slice_wps[1:], strict=False):
        seg_d = _haversine_m(prev.latitude, prev.longitude, curr.latitude, curr.longitude)
        distance_m += seg_d
        if prev.elevation is not None and curr.elevation is not None:
            has_elevation = True
            if curr.elevation > prev.elevation:
                elevation_gain_m += curr.elevation - prev.elevation
        for elev in (prev.elevation, curr.elevation):
            if elev is None:
                continue
            min_elev = elev if min_elev is None else min(min_elev, elev)
            max_elev = elev if max_elev is None else max(max_elev, elev)
        dt = (curr.timestamp - prev.timestamp).total_seconds()
        if dt > 0:
            speeds.append(seg_d / dt)

    distance_km = distance_m / 1000.0
    gain_per_km = (elevation_gain_m / distance_km) if distance_km > 0 else 0.0
    median_mps = sorted(speeds)[len(speeds) // 2] if speeds else 0.0
    # NET elevation range over the slice: flat-with-noise has a near-zero
    # net even when the summed-positive gain_per_km is large, so require both.
    net_range_m = (max_elev - min_elev) if (min_elev is not None and max_elev is not None) else 0.0

    if (
        has_elevation
        and gain_per_km >= _MOUNTAIN_GAIN_PER_KM
        and net_range_m >= _MOUNTAIN_MIN_NET_RANGE_M
    ):
        return "mountain"
    motorway_flat = (not has_elevation) or gain_per_km < _MOTORWAY_MAX_GAIN_PER_KM
    if median_mps >= _MOTORWAY_MEDIAN_MPS and motorway_flat:
        return "motorway"
    if coastline_points:
        for wp in slice_wps:
            for clat, clon in coastline_points:
                if _haversine_m(wp.latitude, wp.longitude, clat, clon) <= _COASTLINE_PROXIMITY_M:
                    return "coastal"
    return None


def _coastline_points_for_bbox(
    waypoints, stages_dir: Optional[Path] = None
) -> list[tuple[float, float]]:
    """Query OSM natural=coastline within waypoint bbox (one Overpass call,
    cached to ``stages_dir`` when provided). Returns (lat, lon) samples."""
    if not waypoints:
        return []

    from . import overpass_source as _ops

    bbox = _ops._bbox_from_waypoints(waypoints, margin_km=5.0)
    cache_file = None
    if stages_dir:
        bbox_str = f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}"
        bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()[:12]
        cache_file = Path(stages_dir) / f"coastline_{bbox_hash}.json"
        if cache_file.exists():
            try:
                return [(p[0], p[1]) for p in json.loads(cache_file.read_text())]
            except Exception as exc:
                logger.warning(f"Coastline cache load failed: {exc}")

    s, w, n, e = bbox
    query = f'[out:json][timeout:25];(way["natural"="coastline"]({s},{w},{n},{e});); out body geom;'
    points: list[tuple[float, float]] = []
    try:
        # Every endpoint, not just the primary: a single dead mirror used to
        # disable the coastal band outright — see post_overpass_query.
        payload = _ops.post_overpass_query(query)
        for elem in payload.get("elements", []):
            for nd in elem.get("geometry", []) or []:
                if "lat" in nd and "lon" in nd:
                    points.append((nd["lat"], nd["lon"]))
        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(points))
            except Exception as exc:
                logger.warning(f"Coastline cache write failed: {exc}")
    except Exception as exc:
        logger.info(f"Coastline query skipped ({exc}); coastal band disabled")
        return []
    return points


logger = logging.getLogger(__name__)


def _rebuild_journey(journey: Journey, waypoints) -> Journey:
    """Return ``journey`` with ``waypoints`` and start/end dates recomputed
    from them.

    ``Journey.with_waypoints`` keeps the ORIGINAL ``start_date``/``end_date``,
    so after head/tail stripping or an ``included_segments`` trim the journey's
    span stays the raw, untrimmed one — and the ``video_duration / gps_duration``
    scale that places every chapter is then computed against footage that is no
    longer on-screen (the systematic chapter-drift bug). ``dataclasses.replace``
    overrides only ``waypoints`` + the two dates and PRESERVES every other field
    (photos / title / start_location / end_location / daily_breakdowns), so this
    is strictly safer than the lossy date-recomputing constructor in
    ``_journey_filter``. Empty input keeps the original dates (the caller then
    falls back to basic chapters).
    """
    wps = tuple(waypoints)
    if not wps:
        return replace(journey, waypoints=wps)
    return replace(
        journey,
        waypoints=wps,
        start_date=wps[0].timestamp,
        end_date=wps[-1].timestamp,
    )


def _strip_null_island_waypoints(journey: Journey) -> Journey:
    """Drop ``(0, 0)`` "null-island" cold-start fixes from a journey.

    GoPro / Insta360 cameras emit ``(0, 0)`` waypoints before acquiring a
    satellite lock. Reverse-geocoding those yields a bogus mid-ocean chapter
    and can fabricate the GPS-detected end location. Dropping them here — at
    the single point where the journey is materialised, before both the
    geocode-sampling fallback and the end-location detection consume it —
    keeps the bad fixes out of every downstream chapter path. Mirrors the
    null-island reject already applied on the drive-label path via
    ``gps_quality.filter_high_quality_points``.

    Returns the original journey unchanged when nothing is null-island, or
    when *every* waypoint is null-island (so the basic-chapter fallback still
    has data to work with rather than an empty journey).
    """
    waypoints = journey.waypoints
    if not waypoints:
        return journey
    clean = tuple(
        wp
        for wp in waypoints
        if not (
            float(getattr(wp, "latitude", 0.0) or 0.0) == 0.0
            and float(getattr(wp, "longitude", 0.0) or 0.0) == 0.0
        )
    )
    if not clean or len(clean) == len(waypoints):
        return journey
    logger.info(
        "Chapters: dropped %d null-island (0,0) waypoint(s) before geocoding",
        len(waypoints) - len(clean),
    )
    return _rebuild_journey(journey, clean)


# Teleport-run strip thresholds. Tuned so a cold-start
# stale-lock (e.g. ~15 fixes ~400 km away, then a continuous body)
# fires, but a genuine continuous traverse (a real ~250 mi drive whose
# *consecutive* steps stay small) never does — the discriminator is a single
# physically-impossible jump, NOT distance-from-centroid.
_TELEPORT_JUMP_FLOOR_KM = 25.0  # consecutive GPS fixes can't be >25 km apart at sub-sec spacing
_TELEPORT_K = 20.0  # a seam must also exceed 20× the median per-step distance
_TELEPORT_MAX_STRIP_FRAC = (
    0.05  # never strip >5% of points from an end — a real drive isn't a teleport
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return _haversine_m(lat1, lon1, lat2, lon2) / 1000.0


def _strip_teleport_runs(journey: Journey) -> Journey:
    """Drop a contiguous head/tail run separated from the trace by one big jump.

    A stale-nonzero cold-start fix (e.g. a GPS lock acquired far from the route
    start) survives the null-island strip. Its short leading/trailing run is
    joined to the real trace by a single spatial *jump* that's physically
    impossible between consecutive sub-second-spaced fixes — that discontinuity,
    not distance from the route's centroid, is the teleport signature. A genuine
    long traverse spreads continuously with no such single jump, so its true
    endpoints are preserved.

    Never strips more than ``_TELEPORT_MAX_STRIP_FRAC`` of points from an end
    (a teleport is a brief artifact, not half the drive), and returns the
    original journey unchanged if stripping would leave <2 waypoints — mirroring
    ``_strip_null_island_waypoints`` so the basic-chapter fallback keeps data.
    """
    waypoints = journey.waypoints
    n = len(waypoints)
    if n < 3:
        return journey

    from ..gps_quality import exceeds_physical_speed

    steps = [
        _haversine_km(
            waypoints[i].latitude,
            waypoints[i].longitude,
            waypoints[i + 1].latitude,
            waypoints[i + 1].longitude,
        )
        for i in range(n - 1)
    ]
    ordered = sorted(steps)
    mid = len(ordered) // 2
    median_step_km = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    seam_threshold = max(_TELEPORT_JUMP_FLOOR_KM, _TELEPORT_K * median_step_km)
    max_strip = int(n * _TELEPORT_MAX_STRIP_FRAC)

    def _step_dt_sec(i: int) -> Optional[float]:
        a = getattr(waypoints[i], "timestamp", None)
        b = getattr(waypoints[i + 1], "timestamp", None)
        if a is None or b is None:
            return None
        try:
            return (b - a).total_seconds()
        except (TypeError, AttributeError):
            return None

    def _is_teleport_seam(i: int) -> bool:
        # A seam is a teleport ONLY when it is both spatially large AND
        # implies a physically impossible speed. A big jump over a long dropout
        # (ferry/tunnel/bridge) clears the km threshold but is a sane speed, so it
        # is NOT stripped. When dt is unknown, fall back to the km-only signal.
        if steps[i] <= seam_threshold:
            return False
        dt = _step_dt_sec(i)
        if dt is None:
            return True
        return exceeds_physical_speed(steps[i], dt)

    start = 0
    # First seam scanning inward from the head bounds the leading stale run.
    for i in range(min(max_strip, n - 1)):
        if _is_teleport_seam(i):
            start = i + 1
            break

    end = n
    # Mirror from the tail.
    for j in range(min(max_strip, n - 1)):
        if _is_teleport_seam(n - 2 - j):
            end = n - 1 - j
            break

    if start == 0 and end == n:
        return journey
    if end - start < 2:
        return journey

    clean = waypoints[start:end]
    logger.info(
        "Chapters: dropped %d head + %d tail teleport waypoint(s) before geocoding",
        start,
        n - end,
    )
    return _rebuild_journey(journey, clean)


class ChapterGenerator:
    """Generate YouTube chapters from GPS data."""

    def __init__(
        self,
        config: Config,
        cache_path: Optional[Path] = None,
        max_chapters: int = 15,
    ):
        """
        Initialize chapter generator.

        Args:
            config: Application configuration
            cache_path: Path to geocoding cache JSON file (default: GPS repo cache)
            max_chapters: Maximum number of chapters (default: 15)
        """
        self.config = config
        self.max_chapters = max_chapters
        self._recording_date: Optional[datetime] = None

        # Use GPS repo cache by default (path to JSON file, not directory).
        # ONE canonical namespace shared with Phase A / Phase BC so a
        # coordinate resolves identically across title, label, and chapters.
        if cache_path is None:
            configured = config.drive.resolved_geocode_cache_dir()
            cache_dir = configured
            if not cache_dir.exists():
                cache_dir = Path(tempfile.gettempdir()) / "gomoveshift" / "geocoding_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                logger.warning(
                    "Configured geocoding cache dir %s does not exist, using fallback: %s",
                    configured,
                    cache_dir,
                )
            else:
                cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "geocoding_cache.json"

        # Initialize geocoding with cache (cache_path is a file, not directory)
        self.cache = GeocodingCache(cache_path=cache_path, ttl_days=90)
        # Look up NominatimRoadsClient through package module to support mock patching
        # (tests patch gomoveshift.pipeline.chapters.NominatimRoadsClient)
        _pkg = sys.modules.get(__name__.rsplit(".", 1)[0], None)
        _nrc = getattr(_pkg, "NominatimRoadsClient", NominatimRoadsClient)
        self.geocoder = _nrc(
            user_agent="gomoveshift-video/0.1.0",
            cache=self.cache,
            rate_limit=1.0,  # OSM policy: 1 req/sec
        )
        self.gpx_parser = GPXParser()

        # Sub-components
        self._overpass = OverpassChapterSource()
        self._merger = MarkerMerger()
        # A single budgeted Nominatim source shared across the run so the
        # MAX_SESSION_REQUESTS cap covers BOTH the end-location lookup and the
        # geocode_fallback pass (previously the end-location used the raw,
        # uncapped self.geocoder, bypassing the budget entirely).
        self._nom_source = NominatimChapterSource(geocoder=self.geocoder, cache=self.cache)

        # Area discovery results (populated after generate())
        self.route_pois: list[dict] = []
        self.named_routes: list[str] = []

        # Coastline cache — populated once per generate() call.
        self._coastline_points: list[tuple[float, float]] = []

        # Time-mapping state, (re)populated each generate() call. Initialised
        # here so direct calls to the _scale_gps_to_video / _map_to_edited_time
        # delegates (used by unit tests + area discovery) never hit an
        # AttributeError before generate() runs.
        self._keep_segments: Optional[list[tuple[float, float]]] = None
        self._included_segments: Optional[list[tuple[float, float]]] = None
        self._episode_offset: float = 0.0
        self._footage_origin = None
        self._footage_offset = 0.0

    @property
    def recording_date(self) -> Optional[datetime]:
        """Recording date from GPX data, available after generate()."""
        return self._recording_date

    def _load_and_filter_journey(
        self,
        gpx_path,
        gps_date,
        gps_range,
        _mock_journey,
    ) -> Optional[Journey]:
        """Parse GPX file and apply date/range filters."""
        journey, self._footage_origin = load_journey_and_origin(
            gpx_path,
            gps_date,
            gps_range,
            _mock_journey,
            self.gpx_parser,
        )
        return journey

    def _discover_area_pois(
        self,
        journey: Journey,
        gps_duration: float,
        original_duration: float,
        stages_dir: Optional[Path] = None,
        proximity_m: float = 500.0,
    ) -> list[tuple[float, dict]]:
        """Discover area POIs via Overpass, delegating to OverpassChapterSource.

        Also populates self.route_pois and self.named_routes.
        """
        matched_pois, route_pois, named_routes = self._overpass.discover_area_pois(
            journey=journey,
            gps_duration=gps_duration,
            original_duration=original_duration,
            scale_gps_to_video=self._scale_gps_to_video,
            map_to_edited_time=self._map_to_edited_time,
            keep_segments=self._keep_segments,
            stages_dir=stages_dir,
            proximity_m=proximity_m,
        )
        self.route_pois = route_pois
        self.named_routes = named_routes
        self._merger.named_routes = named_routes
        # On-route waypoint coords for the top route gate the per-chapter suffix
        # so only towns near the route get labeled as on it.
        self._merger.route_waypoints = (
            getattr(self._overpass, "route_match_coords", {}).get(named_routes[0], [])
            if named_routes
            else []
        )
        return matched_pois

    def generate(
        self,
        gpx_path: Path,
        video_duration: float,
        start_location: str,
        end_location: Optional[str] = None,
        max_chapters: Optional[int] = None,
        gps_range: Optional[str] = None,
        keep_segments: Optional[list[tuple[float, float]]] = None,
        markers: Optional[list[dict]] = None,
        gps_date: Optional[str] = None,
        stages_dir: Optional[Path] = None,
        original_duration: Optional[float] = None,
        included_segments: Optional[list[tuple[float, float]]] = None,
        resolved_location=None,
        day_journey: Optional[Journey] = None,
        _mock_journey: Optional[Journey] = None,
    ) -> list[dict]:
        """
        Generate YouTube chapters from GPS file, optionally merging
        DaVinci Resolve timeline markers.

        Args:
            gpx_path: Path to GPX file
            video_duration: Video duration in seconds (after stop removal)
            start_location: Starting location (e.g., "Ennis, County Clare")
            end_location: Ending location (defaults to start_location)
            max_chapters: Override default max chapters
            gps_range: Time range within GPS track in minutes ("START-END")
            keep_segments: List of (start, end) tuples in original video time
            markers: List of timeline marker dicts from DaVinci Resolve
            gps_date: Date filter for multi-day GPX tracks (YYYY-MM-DD)
            stages_dir: Directory for caching Overpass responses
            original_duration: True original video duration in seconds (from ffprobe).
                When provided with keep_segments, used instead of inferring from
                keep_segments[-1][1] which is wrong when there's a trailing stop.
            included_segments: Optional list of ``(start_sec, end_sec)`` ranges
                describing which footage made it into the clip, keyed from
                the loaded journey's own first waypoint BEFORE any head/tail
                strip. That's this GPX's EPISODE-LOCAL seconds when
                loading from ``gpx_path``, or DAY-MASTER seconds when
                ``day_journey`` is supplied (its own second 0 IS day-master
                zero) — never drive-global source seconds either way. When
                set, GPS waypoints that fall outside every included range
                are trimmed BEFORE chapter segmentation — so chapter titles
                describe what's actually on-screen.
                ``None`` preserves the no-trim behaviour.
            resolved_location: the LF's canonical ``ResolvedLocation``; reconciles each
                chapter's county against the sampled regions. ``None`` keeps the legacy
                first-waypoint county.
            day_journey: a pre-built, real (non-test) journey whose waypoint
                timestamps are already DAY-MASTER seconds (see
                ``build_day_master_journey``) — used instead of parsing
                ``gpx_path`` fresh. Pass this (spanning every episode of the
                day) together with ``included_segments`` in the SAME
                day-master seconds — never episode-local — so a superlong
                part windows onto its own slice of the whole day instead of
                one episode's GPX linearly scaled. ``None`` preserves the
                per-episode-GPX behaviour.
            _mock_journey: Injected journey for testing (internal)

        Returns:
            List of chapter dicts with 'timestamp' and 'title' keys
        """
        if end_location is None:
            end_location = start_location

        max_chapters = max_chapters or self.max_chapters
        self._keep_segments = keep_segments
        # ``included_segments`` live in episode-local source seconds (the
        # GPX's own second 0). On the drive path (included set, keep unset) the
        # time map rebases + piecewise-maps through them instead of one linear
        # scale that smears a non-contiguous LF's towns minutes off.
        # _episode_offset bridges the trimmed journey's waypoint base back to
        # that origin; it stays 0 on the single-video / linear paths.
        self._included_segments = included_segments
        self._episode_offset = 0.0
        self._stages_dir = stages_dir
        # The canonical ResolvedLocation, consumed by _extract_towns
        # so each chapter's county is reconciled against the sampled regions.
        self._resolved_location = resolved_location

        # Footage clock (gms_media_s) state; only a freshly parsed, fully
        # stamped GPX sets it, so reset before loading.
        self._footage_origin = None
        self._footage_offset = 0.0

        # Load and filter GPS journey. A pre-built day_journey (real,
        # day-master-time) always wins over parsing gpx_path fresh — see
        # the Args note above.
        journey = day_journey or self._load_and_filter_journey(
            gpx_path, gps_date, gps_range, _mock_journey
        )
        if journey is None:
            return self._generate_basic_chapters(video_duration, start_location, end_location)

        # Episode-local second 0 — the GPX's first waypoint BEFORE any
        # head/tail strip. The trim window and the drive-path time map are
        # both keyed to THIS origin; anchoring on the post-strip first waypoint
        # shifts every window forward by the stripped head's duration.
        if self._footage_origin is not None:
            # Stamped GPX: origin is footage second 0, so the trim and the
            # time map key on the same clock as included_segments.
            episode_origin = self._footage_origin
        else:
            episode_origin = (
                journey.waypoints[0].timestamp if journey.waypoints else journey.start_date
            )

        # Drop (0,0) cold-start fixes before any geocoding so a pre-lock GPS
        # sample can't fabricate a mid-ocean chapter or end-location.
        journey = _strip_null_island_waypoints(journey)
        # Drop a stale-nonzero cold-start teleport run (a short head/tail run
        # joined to the trace by one physically-impossible jump) — a (0,0)-style
        # artifact the null-island strip misses.
        journey = _strip_teleport_runs(journey)

        # Calculate GPS duration
        gps_duration = (journey.end_date - journey.start_date).total_seconds()
        self._recording_date = _longitude_local_start(journey)
        if gps_duration <= 0:
            logger.warning("GPS duration is zero or negative")
            return self._generate_basic_chapters(video_duration, start_location, end_location)

        if not journey.waypoints:
            logger.warning("Journey has no waypoints")
            return self._generate_basic_chapters(video_duration, start_location, end_location)

        # ── Trim waypoints to episode-local clip ranges ───────
        # Chapters must describe what's on-screen, not the raw source
        # footage. included_segments are
        # keyed to this GPX's episode-local time base, the same base
        # as j_start below.
        if included_segments:
            from ..gps_quality import trim_gps_to_clip_timeline

            # Key the trim window on the PRE-strip origin so a stripped
            # null-island / teleport head doesn't shift it by its duration.
            j_start = episode_origin
            original_count = len(journey.waypoints)
            trimmed = trim_gps_to_clip_timeline(
                journey.waypoints,
                included_segments,
                time_key=lambda wp: (wp.timestamp - j_start).total_seconds(),
            )
            if not trimmed:
                logger.warning(
                    "All waypoints trimmed by included_segments; falling back to basic chapters."
                )
                return self._generate_basic_chapters(video_duration, start_location, end_location)
            journey = _rebuild_journey(journey, trimmed)
            # Downstream gps_time is measured from the trimmed journey's
            # first surviving waypoint; record its offset from episode-local 0 so
            # _scale_gps_to_video can rebase before the piecewise map.
            self._episode_offset = (journey.waypoints[0].timestamp - episode_origin).total_seconds()
            logger.info(
                "Chapters: trimmed waypoints to %d ranges → %d / %d kept",
                len(included_segments),
                len(trimmed),
                original_count,
            )

        # Recompute the GPS time-base from the (possibly trimmed,
        # head/tail-stripped) journey. _rebuild_journey reset start/end_date to
        # the surviving span, so gps_duration now measures on-screen footage —
        # the scale gps→video below (and Overpass num_samples, sized off
        # gps_duration) match the trimmed waypoints instead of the raw source
        # span. recording_date follows the trimmed start.
        gps_duration = (journey.end_date - journey.start_date).total_seconds()
        self._recording_date = _longitude_local_start(journey)
        if gps_duration <= 0:
            logger.warning("GPS duration is zero or negative after trim/strip")
            return self._generate_basic_chapters(video_duration, start_location, end_location)
        if self._footage_origin is not None:
            self._footage_offset = (
                journey.waypoints[0].timestamp - self._footage_origin
            ).total_seconds()

        # Detect end_location from last GPS waypoint if not explicitly set
        if end_location == start_location and journey.waypoints:
            last_wp = journey.waypoints[-1]
            try:
                end_loc = self._nom_source.reverse_geocode(last_wp.latitude, last_wp.longitude)
                if end_loc and end_loc.city:
                    # Prefer the administrative city over end_loc.name: .name is the
                    # POI/business-class label with no at-point distance check, so a
                    # merely passed-near pub could otherwise become the final
                    # chapter's location.
                    detected_end = end_loc.city or end_loc.name
                    if detected_end != start_location:
                        end_location = detected_end
                        logger.info(f"End location from GPS: {end_location}")
            except Exception as e:
                logger.debug(f"Could not detect end location from GPS: {e}")

        # Compute original video duration from keep_segments if available
        if keep_segments:
            if original_duration is None:
                # Fallback: infer from last keep_segment end (wrong if trailing stop)
                original_duration = keep_segments[-1][1]
            logger.info(
                f"GPS duration: {gps_duration / 60:.1f}min, "
                f"Original video: {original_duration / 60:.1f}min, "
                f"Edited video: {video_duration / 60:.1f}min, "
                f"Stops removed: {len(keep_segments) - 1} gap(s)"
            )
        else:
            original_duration = video_duration
            logger.info(
                f"GPS duration: {gps_duration / 60:.1f}min, "
                f"Video duration: {video_duration / 60:.1f}min, "
                f"Scale factor: {video_duration / gps_duration:.3f}"
            )

        # Sample waypoints and geocode
        waypoints_with_towns = self._extract_towns(journey, gps_duration, original_duration)

        # Limit chapters
        if len(waypoints_with_towns) > (max_chapters - 2):
            waypoints_with_towns = self._prioritize_chapters(waypoints_with_towns, max_chapters - 2)

        def merge(towns, bands=None):
            # Merge markers with GPS chapters into a chapter list
            return self._merger.merge_markers_with_chapters(
                towns,
                markers,
                video_duration,
                start_location,
                end_location,
                format_location_fn=self._format_location,
                terrain_bands=bands,
            )

        # The metadata gate refuses a GPS-bearing LF below its middle-chapter
        # floor; top up from real GPS samples (e.g. several Overpass places
        # at one second can merge down to a single middle chapter).
        waypoints_with_towns = self._top_up_to_gate_floor(
            waypoints_with_towns,
            merge,
            video_duration,
            max_chapters,
            samples=lambda: self._gps_samples(journey, gps_duration, original_duration),
        )

        # Compute per-chapter terrain bands from the trace.
        bands = self._compute_terrain_bands(
            journey,
            waypoints_with_towns,
            gps_duration,
            original_duration,
            stages_dir=getattr(self, "_stages_dir", None),
        )

        chapters = merge(waypoints_with_towns, bands)
        # Still under the floor when no GPS sample names a new place (rural
        # West Cork, 2026-10-12): add trace-spaced chapters so the gate clears
        # by construction instead of failing after a multi-day render.
        chapters = self._pad_to_gate_floor(
            chapters,
            video_duration,
            self._gps_samples(journey, gps_duration, original_duration),
            self._trace_km_at(journey, gps_duration, original_duration),
        )

        logger.info(f"Generated {len(chapters)} chapters")
        return self._pad_to_min_chapters(chapters, video_duration, start_location)

    # ── YouTube renderability gate ────────────────────────────────────

    @staticmethod
    def _parse_timestamp(ts: str) -> int:
        """Parse a 'M:SS' / 'H:MM:SS' timestamp back into seconds."""
        secs = 0
        for part in ts.split(":"):
            secs = secs * 60 + int(part)
        return secs

    def _pad_to_min_chapters(
        self, chapters: list[dict], video_duration: float, label: Optional[str] = None
    ) -> list[dict]:
        """Pad a chapter list to YouTube's minimum renderable shape.

        YouTube only renders a description chapter block with >=
        ``YOUTUBE_MIN_CHAPTERS`` timestamps, the first at 0:00, each >=
        ``YOUTUBE_MIN_CHAPTER_GAP_SEC`` apart. The sparse-GPS basic fallback
        emits only 2 (Start/Finish); without padding YouTube silently shows
        none. Insert evenly-spaced interior chapters carrying the LF location
        so the block renders. No-op when already renderable; an LF too short
        to host the minimum spaced chapters is left untouched with a warning
        rather than crammed. Interior pads carry a real town name from
        ``ResolvedLocation.ordered_place_sequence`` (positioned by travel order)
        when one exists, falling back to the generic ``"En route — {place}"``.
        """
        if len(chapters) >= YOUTUBE_MIN_CHAPTERS:
            return chapters
        if video_duration < YOUTUBE_MIN_CHAPTERS * YOUTUBE_MIN_CHAPTER_GAP_SEC:
            logger.warning(
                "chapter gate: LF too short (%.0fs) to pad to %d renderable "
                "chapters — block may not render on YouTube",
                video_duration,
                YOUTUBE_MIN_CHAPTERS,
            )
            return chapters
        needed = YOUTUBE_MIN_CHAPTERS - len(chapters)
        logger.warning(
            "chapter gate: only %d chapter(s) for a %.0fs LF — padding %d interior "
            "chapter(s) so YouTube renders the block (needs >=%d, first 0:00, >=%ds apart)",
            len(chapters),
            video_duration,
            needed,
            YOUTUBE_MIN_CHAPTERS,
            YOUTUBE_MIN_CHAPTER_GAP_SEC,
        )
        place = label.split(" - ", 1)[-1].strip() if label else ""
        interior_title = f"En route — {place}" if place else "En route"

        # Prefer REAL town names from the canonical GPS place sequence
        # (ResolvedLocation.ordered_place_sequence) over the generic
        # "En route — {place}" — YouTube chapter names are indexed search terms,
        # so a searchable town outranks a generic interior label. The sequence
        # is the SAME canonical, vote-derived ordering the title/OST/description
        # use, so a wrong county can't leak in. Back-compat: an absent /
        # empty sequence (no resolver, single-video path, GPS-less LF) keeps the
        # generic name unchanged.
        resolved = getattr(self, "_resolved_location", None)
        place_seq = list(getattr(resolved, "ordered_place_sequence", None) or [])

        def _pad_title(t: float, left_title: str) -> str:
            """Name a pad at time ``t`` from the place sequence by travel order.

            Maps the pad's fractional position through the LF to an index in
            ``ordered_place_sequence`` (clamped in range). When the picked town
            equals the left neighbour's name, prefer a DIFFERENT town from the
            sequence before falling back to the timestamp disambiguator.
            """
            if not place_seq:
                title = interior_title
                if title == left_title:
                    title = f"{interior_title} ({self._format_timestamp(t)})"
                return title
            frac = t / video_duration if video_duration else 0.0
            idx = max(0, min(len(place_seq) - 1, round(frac * (len(place_seq) - 1))))
            town = place_seq[idx]
            if town == left_title:
                # Search outward for a different town before disambiguating.
                for off in range(1, len(place_seq)):
                    for j in (idx + off, idx - off):
                        if 0 <= j < len(place_seq) and place_seq[j] != left_title:
                            town = place_seq[j]
                            break
                    if town != left_title:
                        break
            if town == left_title:
                return f"{town} ({self._format_timestamp(t)})"
            return town

        # Place each pad in the CURRENT largest gap, midway between its
        # neighbors, but only if both halves clear YOUTUBE_MIN_CHAPTER_GAP_SEC.
        # Even-spreading the whole video ignored the real chapters' positions
        # and could land a pad within the min gap of one (YouTube then refuses
        # the block) or duplicate an adjacent chapter's name.
        chapters = sorted(chapters, key=lambda c: self._parse_timestamp(c["timestamp"]))
        placed = 0
        for _ in range(needed):
            times = [self._parse_timestamp(c["timestamp"]) for c in chapters]
            bounds = times + [video_duration]
            # Find the widest gap that can host a pad with min-gap on both sides.
            best_i, best_w = -1, 0.0
            for i in range(len(times)):
                width = bounds[i + 1] - bounds[i]
                if width > best_w:
                    best_w, best_i = width, i
            if best_i < 0 or best_w < 2 * YOUTUBE_MIN_CHAPTER_GAP_SEC:
                break
            t = (bounds[best_i] + bounds[best_i + 1]) / 2.0
            # Don't duplicate the neighbour name; prefer a real town.
            title = _pad_title(t, chapters[best_i]["title"])
            chapters.insert(best_i + 1, {"timestamp": self._format_timestamp(t), "title": title})
            placed += 1

        chapters.sort(key=lambda c: self._parse_timestamp(c["timestamp"]))
        # Validate own output — first at 0:00, every gap >= the minimum.
        times = [self._parse_timestamp(c["timestamp"]) for c in chapters]
        gaps_ok = all(b - a >= YOUTUBE_MIN_CHAPTER_GAP_SEC for a, b in zip(times, times[1:]))
        if placed < needed or not gaps_ok or (times and times[0] != 0):
            logger.warning(
                "chapter gate: padding produced %d/%d chapters (gaps_ok=%s) — "
                "block may not render on YouTube",
                len(chapters),
                YOUTUBE_MIN_CHAPTERS,
                gaps_ok,
            )
        return chapters

    # ── Delegate methods for backward compatibility ───────────────────

    def _scale_gps_to_video(
        self, gps_time: float, gps_duration: float, video_duration: float
    ) -> float:
        # Footage clock: gps_time is measured from the first surviving fix, whose
        # footage second is _footage_offset, so the sum is the exact footage
        # second; _map_to_edited_time maps it through keep/included segments.
        if self._footage_origin is not None and (self._included_segments or self._keep_segments):
            return gps_time + self._footage_offset
        # Drive path (included_segments set, keep_segments unset): the LF is
        # assembled from NON-contiguous source segments, so one linear scale
        # smears every town by the removed footage (minutes off). Rebase the
        # journey-relative gps_time to episode-local source seconds and let
        # _map_to_edited_time piecewise-map it through included_segments instead.
        if self._included_segments and not self._keep_segments:
            return gps_time + self._episode_offset
        return MarkerMerger.scale_gps_to_video(gps_time, gps_duration, video_duration)

    def _map_to_edited_time(self, original_time: float) -> float:
        segments = self._keep_segments or self._included_segments
        return MarkerMerger.map_to_edited_time(original_time, segments)

    def _format_timestamp(self, seconds: float) -> str:
        return MarkerMerger.format_timestamp(seconds)

    # ── Town extraction ──────────────────────────────────────────────

    def _extract_towns(
        self, journey: Journey, gps_duration: float, original_duration: float
    ) -> list[tuple[float, Location]]:
        """Extract town waypoints from journey and map to edited video time."""
        self._town_samples = None  # this journey's geocoded samples, once fetched
        if self._keep_segments:
            edited_duration = sum(end - start for start, end in self._keep_segments)
        else:
            edited_duration = original_duration
        min_chapters = max(2, int(edited_duration / 60 / 10))

        # Fast path: Overpass area discovery
        area_pois = self._discover_area_pois(
            journey,
            gps_duration,
            original_duration,
            stages_dir=getattr(self, "_stages_dir", None),
        )

        if area_pois:
            town_results = OverpassChapterSource.pois_to_locations(
                area_pois,
                journey.waypoints,
                min_chapters,
                is_valid_chapter_placename,
                resolved_location=getattr(self, "_resolved_location", None),
            )
            if len(town_results) >= min_chapters:
                logger.info(f"Overpass: {len(town_results)} chapters (target {min_chapters})")
                return town_results

            logger.info(
                f"Overpass found {len(town_results)} places, need {min_chapters} -- "
                "falling back to geocoding"
            )

        # Fallback: capped Nominatim geocoding (delegated to NominatimChapterSource).
        # Reuse the run-shared budgeted source so the end-location lookup and
        # this fallback share one MAX_SESSION_REQUESTS budget.
        nom = self._nom_source
        all_geocoded = self._gps_samples(journey, gps_duration, original_duration)

        seen_keys: set[tuple] = set()
        cities = nom.filter_unique(all_geocoded, seen_keys, "city", is_valid_chapter_placename)

        if len(cities) >= min_chapters:
            logger.info(f"Pass 1 (cities): {len(cities)} chapters (target {min_chapters})")
            return cities

        localities = nom.filter_unique(
            all_geocoded,
            seen_keys,
            "locality",
            is_valid_chapter_placename,
        )
        combined = sorted(cities + localities, key=lambda x: x[0])

        if len(combined) >= min_chapters:
            logger.info(
                f"Pass 2 (cities+localities): {len(combined)} chapters (target {min_chapters})"
            )
            return combined

        logger.info(
            f"Only {len(combined)} named places found (target {min_chapters}), "
            f"adding interval chapters"
        )
        return nom.fill_interval_chapters(combined, all_geocoded, edited_duration, min_chapters)

    def _gps_samples(
        self, journey: Journey, gps_duration: float, original_duration: float
    ) -> list[tuple[float, Location]]:
        """Reverse-geocoded GPS samples as (edited video time, Location), fetched
        once per ``_extract_towns`` so its fallback and the gate-floor top-up
        share one Nominatim session budget."""
        if getattr(self, "_town_samples", None) is None:
            self._town_samples = self._nom_source.geocode_fallback(
                journey,
                gps_duration,
                original_duration,
                self._scale_gps_to_video,
                self._map_to_edited_time,
                is_valid_chapter_placename,
            )
        return self._town_samples

    def _top_up_to_gate_floor(
        self,
        towns: list[tuple[float, Location]],
        merge,
        video_duration: float,
        max_chapters: int,
        samples,
    ) -> list[tuple[float, Location]]:
        """Add interval chapters until ``merge(towns)`` clears the metadata
        gate's middle-chapter floor (``min_intermediate_chapters``).

        Each added chapter is a reverse-geocoded GPS sample at its own
        timestamp (``fill_interval_chapters``), so it can only name a place the
        car was at. Denser targets are tried because the merger drops a chapter
        within 60 s of the one before. If none clears the floor the best list
        is returned and the shortfall logged, never padded with a guessed name:
        the gate then refuses the LF instead of shipping an invented chapter.
        """
        floor = min_intermediate_chapters(video_duration)
        best_n = len(intermediate_chapters(merge(towns))) if floor else 0
        if best_n >= floor:
            return towns
        geocoded = samples()
        best = towns
        for target in range(floor, 2 * floor + 3):
            trial = self._nom_source.fill_interval_chapters(towns, geocoded, video_duration, target)
            if len(trial) > max_chapters - 2:
                trial = self._prioritize_chapters(trial, max_chapters - 2)
            n = len(intermediate_chapters(merge(trial)))
            if n > best_n:
                best, best_n = trial, n
            if best_n >= floor:
                logger.info(f"Chapter floor: topped up to {best_n} middle chapters (need {floor})")
                return best
        logger.warning(
            "chapter floor: %d of %d middle chapters for a %.0f min video and no further "
            "GPS sample names one; the metadata gate will refuse it",
            best_n,
            floor,
            video_duration / 60,
        )
        return best

    def _trace_km_at(self, journey: Journey, gps_duration: float, original_duration: float):
        """Return ``f(video_time) -> km driven along the trace by then``."""
        j_start = journey.start_date
        times: list[float] = []
        kms: list[float] = []
        total, prev = 0.0, None
        for wp in journey.waypoints:
            if prev is not None:
                total += _haversine_m(prev.latitude, prev.longitude, wp.latitude, wp.longitude)
            prev = wp
            times.append(
                self._map_to_edited_time(
                    self._scale_gps_to_video(
                        (wp.timestamp - j_start).total_seconds(), gps_duration, original_duration
                    )
                )
            )
            kms.append(total / 1000.0)

        def km_at(t: float) -> float:
            if not times:
                return 0.0
            i = min(range(len(times)), key=lambda k: abs(times[k] - t))
            return kms[i]

        return km_at

    def _pad_to_gate_floor(
        self,
        chapters: list[dict],
        video_duration: float,
        samples: list[tuple[float, Location]],
        km_at,
    ) -> list[dict]:
        """Insert trace-spaced chapters until the gate's middle-chapter floor holds.

        Each pad sits midway in the widest gap whose halves both clear the
        merger's 60 s rule, and is named by the reverse-geocoded GPS sample
        nearest it plus the distance driven by then ("West Cork, County Cork —
        24 km in"), so it only names where the car was. No geocoded sample at
        all leaves the list short and the gate refuses it, as before.
        """
        floor = min_intermediate_chapters(video_duration)
        if len(intermediate_chapters(chapters)) >= floor:
            return chapters
        named = [(t, label) for t, loc in samples if (label := self._sample_label(loc))]
        if not named:
            return chapters
        chapters = sorted(chapters, key=lambda c: self._parse_timestamp(c["timestamp"]))
        titles = {c["title"] for c in chapters}
        while len(intermediate_chapters(chapters)) < floor:
            times = [self._parse_timestamp(c["timestamp"]) for c in chapters]
            gaps = [(b - a, i) for i, (a, b) in enumerate(zip(times, times[1:])) if b - a >= 120]
            if not gaps:
                break
            _, i = max(gaps)
            t = (times[i] + times[i + 1]) // 2
            label = min(named, key=lambda s: abs(s[0] - t))[1]
            title = f"{label} — {km_at(t):.0f} km in"
            if title in titles:
                title = f"{label} — {km_at(t):.1f} km in"
            if title in titles:
                break
            titles.add(title)
            chapters.insert(i + 1, {"timestamp": self._format_timestamp(t), "title": title})
        logger.info(
            "Chapter floor: %d middle chapters after trace-spaced padding (need %d)",
            len(intermediate_chapters(chapters)),
            floor,
        )
        return chapters

    def _sample_label(self, loc: Optional[Location]) -> str:
        """A geocoded sample's chapter label: its place, else its county."""
        if loc is None:
            return ""
        return self._format_location(loc) or (loc.county or "")

    def _prioritize_chapters(
        self, chapters: list[tuple[float, Location]], max_count: int
    ) -> list[tuple[float, Location]]:
        """Limit chapters to max_count by keeping evenly spaced chapters."""
        if len(chapters) <= max_count:
            return chapters

        step = len(chapters) / max_count
        selected = []

        for i in range(max_count):
            index = int(i * step)
            if index < len(chapters):
                selected.append(chapters[index])

        logger.info(f"Limited {len(chapters)} chapters to {len(selected)} (evenly spaced)")
        return selected

    _STRIP_SUFFIXES = [
        "Boat Landing Parking",
        "Shopping Center",
        "Shopping Centre",
        "Parking Lot",
        "Car Park",
        "Rest Area",
        "Subdivision",
        "Parking",
        "Marina",
    ]

    def _format_location(self, location: Location) -> str:
        """Format location as 'Town, County' using best available name.

        An unnamed POI returns "" (suppresses the chapter) instead of a
        raw coordinate string like "35.1234, -82.5678" — the caller drops
        empty labels, so a nameless place never ships an "Entering <coords>".
        """
        name = location.city or location.locality or location.name or location.suburb
        if not name:
            return ""
        for suffix in self._STRIP_SUFFIXES:
            if name.endswith(f" {suffix}"):
                name = name[: -(len(suffix) + 1)].strip()
                break
        if not name:
            return ""
        if not is_valid_chapter_placename(name):
            return ""
        if location.county and location.county not in name:
            return f"{name}, {location.county}"
        return name

    # ── Terrain-band classification ────────────────────

    def _compute_terrain_bands(
        self,
        journey: Journey,
        waypoints_with_towns: list[tuple[float, Location]],
        gps_duration: float,
        original_duration: float,
        stages_dir: Optional[Path] = None,
    ) -> list[Optional[str]]:
        """Parallel list of terrain bands for each `waypoints_with_towns` entry.

        Coastline points are queried ONCE per drive via the module-level
        helper; marker_merger never re-queries them.
        """
        if not waypoints_with_towns or not journey.waypoints:
            return [None] * len(waypoints_with_towns)

        self._coastline_points = _coastline_points_for_bbox(journey.waypoints, stages_dir)

        j_start = journey.start_date
        wp_video_times = [
            self._map_to_edited_time(
                self._scale_gps_to_video(
                    (wp.timestamp - j_start).total_seconds(), gps_duration, original_duration
                )
            )
            for wp in journey.waypoints
        ]

        bands: list[Optional[str]] = []
        prev = 0.0
        for video_time, _loc in waypoints_with_towns:
            bands.append(
                _classify_chapter_band(
                    journey.waypoints, wp_video_times, prev, video_time, self._coastline_points
                )
            )
            prev = video_time
        return bands

    def _generate_basic_chapters(
        self, video_duration: float, start_location: str, end_location: str
    ) -> list[dict]:
        """Generate basic start/end chapters when GPS unavailable."""
        logger.info("Generating basic chapters (no GPS data)")
        finish_time = max(0, video_duration - 15)
        basic = [
            {"timestamp": "0:00", "title": f"Start - {start_location}"},
            {
                "timestamp": self._format_timestamp(finish_time),
                "title": f"Finish - {end_location}",
            },
        ]
        return self._pad_to_min_chapters(basic, video_duration, start_location)

    def format_chapters(self, chapters: list[dict]) -> str:
        """Format chapters as YouTube description text."""
        lines = []
        for chapter in chapters:
            lines.append(f"{chapter['timestamp']} {chapter['title']}")
        return "\n".join(lines)
