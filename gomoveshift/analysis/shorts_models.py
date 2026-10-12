"""Data models and exceptions for shorts extraction."""

import json
import logging
import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Selection-algorithm version for the persisted ``shorts_analysis.json`` cache.
# Bumped whenever the short-SELECTION rule (candidate windows, scoring, stop
# exclusion) changes such that a cached analysis's windows can no longer be
# trusted — e.g. a change to stop-aware selection. ``ExtractionResult.to_dict``
# stamps it; ``cache.is_analysis_cached`` re-checks it so a selector change
# re-analyzes instead of shipping pre-fix windows. Mirrors GRADE_ALGO_VERSION.
SHORTS_SELECTOR_VERSION = 1

# A stale/teleport clock lock resolves to a fix FAR from the leading stuck
# coordinate (typically hundreds of km). A car legitimately
# parked at the true start pulls away only tens of metres. This threshold
# separates the two: a leading identical run is only treated as a stale lock
# when its first real fix is at least this far away, so a genuine parked start
# keeps its base offset instead of jumping ~300 s to the first moving fix.
_STALE_LOCK_MIN_JUMP_M = 1000.0

# Namespace-free GPX extension tag carrying a fix's footage second in the EPISODE
# concat timeline. Bare (no xmlns) on purpose: a namespaced extension is
# re-serialised as malformed XML when gpxpy rewrites it through a fresh GPX()
# (_merge_gpx_files / _extract_gpx_time_range); a bare tag survives every
# hop and reads back as {gpx-ns}gms_media_s.
GPX_MEDIA_S_TAG = "gms_media_s"


def footage_origin(gps_data) -> Optional[float]:
    """GPS-clock time of footage second 0, from stamped trackpoints; else None.

    Accepts GPSData or the dict form {"trackpoints": [dict, ...]}.
    First stamped point's timestamp - media_s.
    """
    if gps_data is None:
        return None
    tps = getattr(gps_data, "trackpoints", None)
    if tps is None and isinstance(gps_data, dict):
        tps = gps_data.get("trackpoints")
    for tp in tps or []:
        if isinstance(tp, dict):
            media_s, ts = tp.get("media_s"), tp.get("timestamp")
        else:
            media_s, ts = getattr(tp, "media_s", None), getattr(tp, "timestamp", None)
        if media_s is not None and ts is not None:
            return float(ts) - float(media_s)
    return None


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in meters between two lat/lon points."""
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def effective_offset(gps_data, stored_offset) -> float:
    """Parse-time GPS-offset correction shared by EVERY consumer.

    The stored offset (first trackpoint timestamp) is wrong when the trace
    opens with a GoPro cold-start anomaly — a leading null-island ``(0,0)`` run
    OR a leading repeated stale-nonzero lock (the teleport-clock
    case); both push segment queries into the wrong epoch and empty the
    point set. Returns the first trackpoint that is neither null-island nor part
    of the leading stuck-coordinate run; falls back to ``stored_offset``. Every
    site (LF labelling, the scorer, the shorts-OST path) calls this one helper,
    so none reads the raw offset.
    """
    base = float(stored_offset or 0.0)
    if gps_data is None:
        return base
    # A stamped GPX KNOWS where footage second 0 sits on the GPS clock; the
    # null-island / stale-lock logic below only guesses it from first-fix time.
    stamped = footage_origin(gps_data)
    if stamped is not None:
        return stamped
    tps = getattr(gps_data, "trackpoints", None)
    if tps is None and isinstance(gps_data, dict):
        tps = gps_data.get("trackpoints") or []
    tps = list(tps or [])
    if not tps:
        return base

    def _ll(tp):
        if isinstance(tp, dict):
            return float(tp.get("lat", 0.0) or 0.0), float(tp.get("lon", 0.0) or 0.0)
        return float(getattr(tp, "lat", 0.0) or 0.0), float(getattr(tp, "lon", 0.0) or 0.0)

    def _ts(tp):
        return float(tp["timestamp"]) if isinstance(tp, dict) else float(tp.timestamp)

    first_lat, first_lon = _ll(tps[0])
    leading_null = first_lat == 0.0 and first_lon == 0.0
    # A leading run of the SAME nonzero coordinate is a stale clock
    # lock ONLY when it is genuinely stale — the run resolves to a fix FAR from
    # the stuck coordinate (a teleport). A short-or-long identical run at the
    # TRUE start (a car parked ~5 min before pulling away) is legitimate: its
    # first moving fix is NEAR, so the base offset must be preserved.
    # Distinguish the two by the distance from the stuck coord to the first fix
    # (non-null, different coord) that breaks the run.
    leading_repeat = not leading_null and len(tps) > 1 and _ll(tps[1]) == (first_lat, first_lon)
    leading_stale = False
    if leading_repeat:
        for tp in tps:
            lat, lon = _ll(tp)
            if lat == 0.0 and lon == 0.0:
                continue  # null-island break candidate is invalid — keep scanning
            if (lat, lon) == (first_lat, first_lon):
                continue  # still inside the leading stuck run
            leading_stale = _haversine_m(first_lat, first_lon, lat, lon) >= _STALE_LOCK_MIN_JUMP_M
            break
    if not (leading_null or leading_stale):
        return base
    # Two independent skip reasons, kept SEPARATE so a mid-trace null embedded
    # in a stale-lock run can never be returned as the offset:
    #   * null-island ``(0,0)`` is ALWAYS invalid (a real fix is never at the
    #     equator/prime-meridian for this content) — reject it wherever it
    #     appears, not just in the leading run;
    #   * the leading stuck-coordinate run is skipped only while we are
    #     still consuming that exact leading coord.
    stuck = (first_lat, first_lon) if leading_stale else None
    for tp in tps:
        lat, lon = _ll(tp)
        if lat == 0.0 and lon == 0.0:
            # Null-island: invalid coordinate, skip regardless of which lead
            # anomaly we are correcting.
            continue
        if stuck is not None and (lat, lon) == stuck:
            # Still inside the leading stale-lock run.
            continue
        return _ts(tp)
    return base


class ShortsExtractorError(Exception):
    """Base exception for shorts extraction errors."""

    pass


class VideoNotFoundError(ShortsExtractorError):
    """Video file not found or not accessible."""

    pass


class InvalidVideoError(ShortsExtractorError):
    """Video file is invalid, corrupt, or has no video track."""

    pass


class ExtractionError(ShortsExtractorError):
    """Error during extraction process."""

    pass


class GPSError(Exception):
    """Error related to GPS data processing."""

    pass


# =============================================================================
# GPS Data Structures
# =============================================================================


@dataclass
class GPSTrackpoint:
    """A single GPS trackpoint."""

    lat: float
    lon: float
    elevation: Optional[float]
    timestamp: float  # Unix timestamp
    # Footage second in the episode concat timeline (gms_media_s); None on
    # unstamped (legacy / external) GPX.
    media_s: Optional[float] = None

    @property
    def has_elevation(self) -> bool:
        return self.elevation is not None


@dataclass
class GPSData:
    """Collection of GPS trackpoints with metadata."""

    trackpoints: List["GPSTrackpoint"]

    @property
    def start_time(self) -> Optional[float]:
        if not self.trackpoints:
            return None
        return self.trackpoints[0].timestamp

    @property
    def footage_origin(self) -> Optional[float]:
        return footage_origin(self)

    @property
    def end_time(self) -> Optional[float]:
        if not self.trackpoints:
            return None
        return self.trackpoints[-1].timestamp

    def get_points_in_range(self, start_time: float, end_time: float) -> List["GPSTrackpoint"]:
        """Get trackpoints within a time range."""
        return [tp for tp in self.trackpoints if start_time <= tp.timestamp <= end_time]


# =============================================================================
# Extraction Result Models
# =============================================================================


@dataclass
class ExtractedShort:
    """A short video segment extracted from long-form content."""

    index: int  # 1-based index
    rank: int  # Quality rank (1 = best)
    start_time: float  # seconds
    end_time: float  # seconds
    duration: float  # seconds
    avg_motion_energy: float
    peak_motion_energy: float
    source_video: str
    output_path: Optional[str] = None
    thumbnail_path: Optional[str] = None

    # Metadata for social media
    suggested_caption: Optional[str] = None

    # GPS-derived terrain classification
    terrain_type: Optional[str] = None  # "coastal", "mountain", "winding", "countryside"
    avg_elevation: Optional[float] = None  # meters
    turn_intensity: Optional[float] = None  # 0-100

    # GPS center for geographic deduplication
    gps_center_lat: Optional[float] = None
    gps_center_lon: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "rank": self.rank,
            "start_time": self.start_time,
            "start_formatted": str(timedelta(seconds=int(self.start_time))),
            "end_time": self.end_time,
            "end_formatted": str(timedelta(seconds=int(self.end_time))),
            "duration": self.duration,
            "avg_motion_energy": round(self.avg_motion_energy, 6),
            "peak_motion_energy": round(self.peak_motion_energy, 6),
            "source_video": self.source_video,
            "output_path": self.output_path,
            "thumbnail_path": self.thumbnail_path,
            "suggested_caption": self.suggested_caption,
            "terrain_type": self.terrain_type,
            "avg_elevation": self.avg_elevation,
            "turn_intensity": self.turn_intensity,
            "gps_center_lat": self.gps_center_lat,
            "gps_center_lon": self.gps_center_lon,
        }


@dataclass
class ExtractionResult:
    """Result of shorts extraction process."""

    source_video: str
    source_duration: float
    num_shorts_requested: int
    num_shorts_found: int
    shorts: List[ExtractedShort]
    dead_spots_avoided: int
    processing_time: float

    def to_dict(self) -> dict:
        return {
            "source_video": self.source_video,
            "source_duration": self.source_duration,
            "source_duration_formatted": str(timedelta(seconds=int(self.source_duration))),
            "num_shorts_requested": self.num_shorts_requested,
            "num_shorts_found": self.num_shorts_found,
            "shorts": [s.to_dict() for s in self.shorts],
            "dead_spots_avoided": self.dead_spots_avoided,
            "processing_time": round(self.processing_time, 2),
            # Selector-rule version — invalidates the cache on a selection-rule
            # change (see SHORTS_SELECTOR_VERSION / cache.is_analysis_cached).
            "selector_version": SHORTS_SELECTOR_VERSION,
        }

    def save(self, output_path: Path) -> None:
        """Save extraction results to JSON."""
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExtractionResult":
        """Load extraction results from JSON."""
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            logger.warning("Corrupt extraction results JSON at %s: %s", path, exc)
            raise ValueError(f"Could not parse extraction results: {path}") from exc
        shorts = []
        for s in data.get("shorts", []):
            shorts.append(
                ExtractedShort(
                    index=s["index"],
                    rank=s["rank"],
                    start_time=s["start_time"],
                    end_time=s["end_time"],
                    duration=s["duration"],
                    avg_motion_energy=s["avg_motion_energy"],
                    peak_motion_energy=s["peak_motion_energy"],
                    source_video=s["source_video"],
                    output_path=s.get("output_path"),
                    thumbnail_path=s.get("thumbnail_path"),
                    suggested_caption=s.get("suggested_caption"),
                    terrain_type=s.get("terrain_type"),
                    avg_elevation=s.get("avg_elevation"),
                    turn_intensity=s.get("turn_intensity"),
                    gps_center_lat=s.get("gps_center_lat"),
                    gps_center_lon=s.get("gps_center_lon"),
                )
            )
        return cls(
            source_video=data["source_video"],
            source_duration=data["source_duration"],
            num_shorts_requested=data["num_shorts_requested"],
            num_shorts_found=data["num_shorts_found"],
            shorts=shorts,
            dead_spots_avoided=data["dead_spots_avoided"],
            processing_time=data["processing_time"],
        )


@dataclass
class ScoringWeights:
    """Configurable weights for segment scoring."""

    motion_energy: float = 0.33  # Weight for average motion energy
    peak_motion: float = 0.20  # Weight for peak motion (exciting moments)
    duration_match: float = 0.10  # Weight for matching target duration
    position: float = 0.05  # Weight for position in video (avoid edges)
    spread: float = 0.10  # Weight for spread across video
    visual_interest: float = 0.07  # Weight for visual interest (colors, contrast)
    # Enabled by default (funded by motion_energy 0.40 → 0.33) so dark/flat
    # night or stopped footage is demoted — mirrors the day-first path's
    # brightness signal (drive.scoring_weights["brightness"] = 0.07). Cost:
    # ~5 sampled frames per candidate segment, negligible next to the full
    # motion-analysis pass. Set to 0.0 to skip frame sampling entirely.
    scenic_location: float = 0.15  # Weight for GPS-based scenic scoring
    # Enabled by default — neutral (50) score when no GPS data available

    def validate(self):
        """Ensure weights sum to 1.0."""
        total = (
            self.motion_energy
            + self.peak_motion
            + self.duration_match
            + self.position
            + self.spread
            + self.visual_interest
            + self.scenic_location
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")
