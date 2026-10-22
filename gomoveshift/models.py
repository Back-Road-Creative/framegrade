"""Data models for video analysis results."""

import json
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class LocationType(str, Enum):
    """Type of location marker detected."""

    TOWN_ENTRY = "town_entry"
    TOWN_EXIT = "town_exit"
    COUNTY_BOUNDARY = "county_boundary"
    STATE_BOUNDARY = "state_boundary"
    ROUTE_MARKER = "route_marker"
    LANDMARK = "landmark"
    RIVER_CROSSING = "river_crossing"
    ROAD_SIGN = "road_sign"
    BUSINESS = "business"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence level of detection."""

    HIGH = "high"  # EasyOCR > 0.8 or Claude confirmed
    MEDIUM = "medium"  # EasyOCR 0.5-0.8
    LOW = "low"  # EasyOCR < 0.5, needs review


@dataclass
class DetectedLocation:
    """A detected location/sign from the video."""

    timestamp: float  # seconds from start
    text: str  # raw OCR text
    location_name: str  # parsed location name
    location_type: LocationType
    confidence: ConfidenceLevel
    confidence_score: float  # 0-1
    frame_path: Optional[str] = None  # path to saved frame
    source: str = "easyocr"  # "easyocr" or "claude"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "timestamp_formatted": str(timedelta(seconds=int(self.timestamp))),
            "text": self.text,
            "location_name": self.location_name,
            "location_type": self.location_type.value,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "frame_path": self.frame_path,
            "source": self.source,
        }


@dataclass
class DeadSpot:
    """A detected dead spot (vehicle stopped)."""

    start_time: float  # seconds
    end_time: float  # seconds
    motion_energy: float  # average motion during this period
    suggested_trim: bool = True  # whether to suggest trimming

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "start_formatted": str(timedelta(seconds=int(self.start_time))),
            "end_time": self.end_time,
            "end_formatted": str(timedelta(seconds=int(self.end_time))),
            "duration": self.duration,
            "motion_energy": self.motion_energy,
            "suggested_trim": self.suggested_trim,
        }


@dataclass
class HighMotionSegment:
    """A high-motion segment suitable for shorts extraction."""

    start_time: float  # seconds
    end_time: float  # seconds
    avg_motion_energy: float  # average motion during this period
    peak_motion_energy: float  # peak motion during this period
    rank: int = 0  # 1 = highest motion, 2 = second highest, etc.

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "start_formatted": str(timedelta(seconds=int(self.start_time))),
            "end_time": self.end_time,
            "end_formatted": str(timedelta(seconds=int(self.end_time))),
            "duration": self.duration,
            "avg_motion_energy": self.avg_motion_energy,
            "peak_motion_energy": self.peak_motion_energy,
            "rank": self.rank,
        }


@dataclass
class CutPoint:
    """A suggested cut point for segmentation."""

    timestamp: float
    reason: str  # "location_change", "duration_target", "dead_spot_end"
    priority: int  # 1 = best, higher = worse
    location: Optional[DetectedLocation] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "timestamp_formatted": str(timedelta(seconds=int(self.timestamp))),
            "reason": self.reason,
            "priority": self.priority,
            "location": self.location.to_dict() if self.location else None,
        }


@dataclass
class VideoAnalysis:
    """Complete analysis results for a video file."""

    video_path: str
    duration: float  # seconds
    fps: float
    width: int
    height: int
    locations: list[DetectedLocation] = field(default_factory=list)
    dead_spots: list[DeadSpot] = field(default_factory=list)
    suggested_cuts: list[CutPoint] = field(default_factory=list)
    processing_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "duration": self.duration,
            "duration_formatted": str(timedelta(seconds=int(self.duration))),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "locations": [loc.to_dict() for loc in self.locations],
            "dead_spots": [ds.to_dict() for ds in self.dead_spots],
            "suggested_cuts": [cp.to_dict() for cp in self.suggested_cuts],
            "processing_stats": self.processing_stats,
        }

    def save(self, output_path: Path) -> None:
        """Save analysis to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, input_path: Path) -> "VideoAnalysis":
        """Load analysis from JSON file."""
        with open(input_path) as f:
            data = json.load(f)

        locations = [
            DetectedLocation(
                timestamp=loc["timestamp"],
                text=loc["text"],
                location_name=loc["location_name"],
                location_type=LocationType(loc["location_type"]),
                confidence=ConfidenceLevel(loc["confidence"]),
                confidence_score=loc["confidence_score"],
                frame_path=loc.get("frame_path"),
                # Legacy artifacts predate the "source" field and were produced
                # by the retired PaddleOCR engine; keep attributing them to it
                # rather than relabel history to the current engine.
                source=loc.get("source", "paddleocr"),
            )
            for loc in data.get("locations", [])
        ]

        dead_spots = [
            DeadSpot(
                start_time=ds["start_time"],
                end_time=ds["end_time"],
                motion_energy=ds["motion_energy"],
                suggested_trim=ds.get("suggested_trim", True),
            )
            for ds in data.get("dead_spots", [])
        ]

        suggested_cuts = [
            CutPoint(
                timestamp=cp["timestamp"],
                reason=cp["reason"],
                priority=cp["priority"],
            )
            for cp in data.get("suggested_cuts", [])
        ]

        return cls(
            video_path=data["video_path"],
            duration=data["duration"],
            fps=data["fps"],
            width=data["width"],
            height=data["height"],
            locations=locations,
            dead_spots=dead_spots,
            suggested_cuts=suggested_cuts,
            processing_stats=data.get("processing_stats", {}),
        )


@dataclass
class ProcessingJob:
    """A video processing job."""

    video_path: Path
    status: str = "pending"  # pending, analyzing, reviewed, processing, complete, error
    analysis: Optional[VideoAnalysis] = None
    approved_cuts: list[float] = field(default_factory=list)
    approved_trims: list[tuple[float, float]] = field(default_factory=list)
    output_files: list[Path] = field(default_factory=list)
    error_message: Optional[str] = None
    # The web /api/approve VideoAssembler render path has NO stop/audio/blur
    # validation — set to the reason string so its outputs are EXPLICITLY marked
    # ungated (a manual/local review surface), never silently passed off as gated.
    ungated_reason: Optional[str] = None


# ── Drive Pipeline Models ────────────────────────────────────────────


@dataclass
class LFSegment:
    """A segment selected for inclusion in a regular long-form video."""

    source_file_index: int  # Index into processed files list
    source_path: Path
    start_time: float  # Within the processed file (seconds)
    end_time: float  # Within the processed file (seconds)
    original_start_time: float  # In original drive timeline (seconds)
    original_end_time: float  # In original drive timeline (seconds)
    composite_score: float  # 0-1 from unified scorer
    gps_center_lat: float
    gps_center_lon: float
    terrain_type: str  # "coastal", "mountain", "winding", "countryside"
    contiguous_with_previous: bool  # Same source file, adjacent in time
    avg_motion_energy: float = 0.0  # Average motion energy across segment samples

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "source_file_index": self.source_file_index,
            "source_path": str(self.source_path),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "original_start_time": self.original_start_time,
            "original_end_time": self.original_end_time,
            "composite_score": self.composite_score,
            "gps_center_lat": self.gps_center_lat,
            "gps_center_lon": self.gps_center_lon,
            "terrain_type": self.terrain_type,
            "contiguous_with_previous": self.contiguous_with_previous,
            "avg_motion_energy": self.avg_motion_energy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LFSegment":
        return cls(
            source_file_index=d["source_file_index"],
            source_path=Path(d["source_path"]),
            start_time=d["start_time"],
            end_time=d["end_time"],
            original_start_time=d["original_start_time"],
            original_end_time=d["original_end_time"],
            composite_score=d["composite_score"],
            gps_center_lat=d["gps_center_lat"],
            gps_center_lon=d["gps_center_lon"],
            terrain_type=d["terrain_type"],
            contiguous_with_previous=d["contiguous_with_previous"],
            avg_motion_energy=d.get("avg_motion_energy", 0.0),
        )


@dataclass
class RegularLFPlan:
    """Assembly plan for one regular long-form video."""

    index: int  # 1-based
    segments: list[LFSegment]
    montage_peaks: list[LFSegment]  # Top peaks for 30s highlight montage
    shorts_target: int  # Minimum shorts this LF must yield
    region: str  # "ireland" or "usa"
    total_duration: float  # Sum of segment durations + transitions
    undersized: bool = False  # True if total_duration < min_lf_duration
    # full_drive balanced split: 1-based part within the day, and the day's
    # part count. None for every non-split LF.
    part_index: Optional[int] = None
    part_count: Optional[int] = None

    def to_dict(self) -> dict:
        d = {
            "index": self.index,
            "segments": [s.to_dict() for s in self.segments],
            "montage_peaks": [s.to_dict() for s in self.montage_peaks],
            "shorts_target": self.shorts_target,
            "region": self.region,
            "total_duration": self.total_duration,
            "undersized": self.undersized,
        }
        if self.part_index is not None:
            d["part_index"] = self.part_index
            d["part_count"] = self.part_count
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RegularLFPlan":
        return cls(
            index=d["index"],
            segments=[LFSegment.from_dict(s) for s in d["segments"]],
            montage_peaks=[LFSegment.from_dict(s) for s in d["montage_peaks"]],
            shorts_target=d["shorts_target"],
            region=d["region"],
            total_duration=d["total_duration"],
            undersized=d.get("undersized", False),
            part_index=d.get("part_index"),
            part_count=d.get("part_count"),
        )


@dataclass
class SuperlongPartPlan:
    """One part of a superlong (whole-day) rendition — distinct from
    RegularLFPlan/LFSegment; the RegularLF pack/merge/classify machinery
    must never see these parts."""

    part_index: int  # 1-based
    part_count: int
    ranges: list[tuple[float, float]]  # ordered (start_sec, end_sec) unmasked spans
    kept_duration_sec: float  # sum of range durations (masked seconds already excluded)

    def to_dict(self) -> dict:
        return {
            "part_index": self.part_index,
            "part_count": self.part_count,
            "ranges": [[start, end] for start, end in self.ranges],
            "kept_duration_sec": self.kept_duration_sec,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SuperlongPartPlan":
        return cls(
            part_index=d["part_index"],
            part_count=d["part_count"],
            ranges=[(r[0], r[1]) for r in d["ranges"]],
            kept_duration_sec=d["kept_duration_sec"],
        )


@dataclass
class DrivePlan:
    """Complete assembly plan for an entire drive."""

    source_files: list[Path]
    processed_files: list[Path]
    regular_lfs: list[RegularLFPlan]
    score_threshold: float
    total_source_duration: float
    total_highlight_duration: float
    best_moments_filtered: bool = False  # True when best_moments_mode filtered segments
    # Phase B+C code version that built this plan; phase B+C rebuilds any other.
    plan_code_version: Optional[int] = None

    @property
    def highlight_ratio(self) -> float:
        if self.total_source_duration <= 0:
            return 0.0
        return self.total_highlight_duration / self.total_source_duration

    def to_dict(self) -> dict:
        return {
            "source_files": [str(p) for p in self.source_files],
            "processed_files": [str(p) for p in self.processed_files],
            "regular_lfs": [lf.to_dict() for lf in self.regular_lfs],
            "score_threshold": self.score_threshold,
            "total_source_duration": self.total_source_duration,
            "total_highlight_duration": self.total_highlight_duration,
            "highlight_ratio": self.highlight_ratio,
            "best_moments_filtered": self.best_moments_filtered,
            "plan_code_version": self.plan_code_version,
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DrivePlan":
        """Reconstruct a ``DrivePlan`` from its ``to_dict()`` output.

        Used by ``PipelineState.from_dict`` to re-hydrate the ``early_plan``
        and ``drive_plan`` fields when loading a checkpoint — without this,
        resume paths that skip Phase A/BC would hand downstream code a raw
        dict and break on attribute access (``drive_plan.regular_lfs``).
        The ``highlight_ratio`` key emitted by ``to_dict()`` is a computed
        property and is intentionally not passed to the constructor.
        """
        return cls(
            source_files=[Path(p) for p in d["source_files"]],
            processed_files=[Path(p) for p in d["processed_files"]],
            regular_lfs=[RegularLFPlan.from_dict(lf) for lf in d["regular_lfs"]],
            score_threshold=d["score_threshold"],
            total_source_duration=d["total_source_duration"],
            total_highlight_duration=d["total_highlight_duration"],
            best_moments_filtered=d.get("best_moments_filtered", False),
            plan_code_version=d.get("plan_code_version"),
        )

    @classmethod
    def load(cls, path: Path) -> "DrivePlan":
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)


@dataclass
class ResolvedLocation:
    """Canonical, once-resolved location for ONE long-form video.

    The structural fix for re-deriving the label at four separate sites.
    Computed once (by the builder) on the quality-filtered,
    null-island/teleport-stripped GPS trace and persisted into
    ``PipelineState.per_lf_resolved_location`` so resume reloads instead of
    re-deriving. Every surface (title/OST, description, chapters, country/
    units/playlist) is meant to consume THIS instead of independently
    parsing display strings.

    Plain dataclass with JSON-native field types ONLY so the state
    serializer (``dataclasses.asdict``) and ``_deserialize_dataclass``
    (``cls(**value)``) round-trip without a custom ``to_dict``/``from_dict``.

    Fields
    ------
    dominant_county / dominant_state:
        The modal ``(county, state)`` voted across sampled, quality-filtered
        points (the ``title_builder.dominant_location`` reducer's output).
    country_code / is_us:
        Country derivation (a trailing US state ⇒ ``is_us`` True, miles/feet,
        US playlist) — resolved once, not re-parsed per surface.
    ordered_place_sequence:
        Ordered place labels along the route (chapters / description body).
    regions:
        Union of sampled quality-filtered counties — the ``location_invariant`` gate's
        "title county ∈ sampled set" membership half.
    vote:
        Per-county sample shares (``{county_label: share}``) — the ``location_invariant``
        gate's coverage half (needs both ``regions`` AND ``vote``).
    start_label / end_label:
        Resolved endpoint labels for the ``"A to B"`` title.
    endpoints:
        Raw endpoint coordinates ``{"start": [lat, lon], "end": [lat, lon]}``
        for downstream road disambiguation. A ``start`` / ``end`` key is
        WITHHELD (absent) when the segment supplying it lost its head / tail
        GPS waypoints, so a shifted coordinate is never emitted as authoritative
        — consumers must treat either key as optional.
    endpoint_regions:
        ``{"start": "County, State", "end": "County, State"}`` — the county each
        trace endpoint reverse-geocodes to (same label form as ``regions``), so
        the location coverage gate can tell which county an ``"A to B"`` title's
        towns name. A key is absent when its endpoint was withheld or the
        geocode missed; empty on a legacy model.
    road:
        The via-road sampled INSIDE the winning (voted) county. The
        title's ``(road)`` disambiguator must follow the vote, not the
        arithmetic-mean centroid, which on a sub-50km county crossing lands
        in the runner-up county and stamps its road. ``None`` when no road
        was geocoded for the winning county.
    road_refs:
        RAW quality-passing road strings from the winning county's samples
        (one per sample, OSM shape — e.g. ``"I 81;US 11"``), for the
        specificity ladder's via-road support check. Normalized to
        hyphenated members at the evidence site, never here.
    road_votes:
        The FULL winning-county road tally (``{hyphenated_member: count}``) the
        via-road winner was chosen from — carried so the per-LF provenance
        sidecar can persist what the vote actually looked like, not just the
        winner. Empty on a degraded / road-less model.
    destination:
        The most-specific enclosing destination POI for the LF's featured
        (montage-peak) footage — a wildlife management area, nature reserve,
        protected-area/national-park boundary, named viewpoint/attraction, or
        coastal natural feature — resolved by
        ``destination_resolver.resolve_destination`` so the title/OST can name
        what the highlights are ABOUT rather than the broad ambient region
        ("Donnelley Wildlife Management Area", not "ACE Basin"). ``None`` when
        no destination POI encloses the footage. Set by ``_populate_resolved_locations``
        in the day-first phase; read by the title builder, the location content gate,
        the social-text generator and the sign-verify surfaces.
    ost_signature:
        The fresh per-LF OST/title text (``state["per_lf_ost_text"]``) this
        model was derived AGAINST, stamped at build time. ``per_lf_ost_text``
        is recomputed on every resume, but the persisted model was previously
        reused unconditionally — so a changed trace/geocoder shipped a title that
        silently diverged from the model feeding units/playlist/tags and the
        content gates. On resume ``_populate_resolved_locations`` re-derives the
        model whenever this signature no longer matches the freshly-recomputed
        OST text, so title and model can never diverge. ``None`` on a legacy /
        pre-signature state file (routes to a one-time re-derive on the next resume).
    trace_verified_places:
        Endpoint towns of the ``ost_signature`` label proven by geometry: a
        settlement of exactly that name lies within 2 km of this LF's own GPS
        fixes. Town-level truth for the location invariant where the reverse
        geocoder buckets the route under a district ("West Cork"), never a
        relaxation — an endpoint missing here still fails. ``None`` = not
        computed (legacy state; filled on the next populate); ``[]`` =
        computed, nothing verified.
    """

    dominant_county: Optional[str] = None
    dominant_state: Optional[str] = None
    country_code: Optional[str] = None
    is_us: bool = False
    ordered_place_sequence: list = field(default_factory=list)
    regions: list = field(default_factory=list)
    vote: dict = field(default_factory=dict)
    start_label: Optional[str] = None
    end_label: Optional[str] = None
    endpoints: dict = field(default_factory=dict)
    endpoint_regions: dict = field(default_factory=dict)
    road: Optional[str] = None
    road_refs: list = field(default_factory=list)
    road_votes: dict = field(default_factory=dict)
    destination: Optional[str] = None
    ost_signature: Optional[str] = None
    trace_verified_places: Optional[list] = None


@dataclass
class DrivePipelineResult:
    """Result of a complete Drive Pipeline run."""

    success: bool
    source_dir: Path
    output_dir: Path
    drive_plan: Any  # DrivePlan or None
    regular_lf_paths: list[Path]
    shorts_per_lf: dict[int, list]  # LF index -> ExtractedShort list
    stages_completed: list[str]
    errors: list[str]
    timings: dict
