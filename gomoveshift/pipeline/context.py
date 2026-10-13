"""PipelineContext — the central data object that flows through all stages.

Replaces the 20 boolean skip flags, scattered local variables, and
implicit state passing in the monolithic run() method. Every stage reads
from and writes to this single context object.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config import Config
from .video_detect import VideoProperties


@dataclass
class PipelinePaths:
    """Single source of truth for all pipeline file locations."""

    output_base: Path
    project_dir: Path
    stages_dir: Path
    state_file: Path
    staging_dir: Path


@dataclass
class PipelineContext:
    """All state needed by pipeline stages. Replaces 20 boolean flags.

    The CLI maps --skip-color, --skip-stops, etc. into
    skip_stages={"color_correction", "stop_removal"} at construction time.
    Resume logic populates skip_stages from stages_completed.
    """

    # ── Identity ──────────────────────────────────────────────────────
    project_name: str
    source_video: Path
    source_hash: str

    # ── Configuration ─────────────────────────────────────────────────
    config: Config
    location: str
    publish_day: str
    paths: PipelinePaths

    # ── Skip manifest (declarative, not 20 booleans) ──────────────────
    skip_stages: set[str] = field(default_factory=set)

    # ── Mutable state (updated by stages) ─────────────────────────────
    current_video: Optional[Path] = None
    video_properties: Optional[VideoProperties] = None
    keep_segments: Optional[list[tuple[float, float]]] = None
    blur_regions_path: Optional[Path] = None  # blur-region sidecar; the mask stage consumes it
    blur_mask_path: Optional[Path] = (
        None  # frame-accurate mask video; the encode alpha-composites it
    )
    recording_date: Optional[datetime] = None
    route_title: Optional[str] = None
    metadata: Optional[dict] = None
    shorts_data: Optional[dict] = None
    schedule: Any = None  # ReleaseSchedule — avoid circular import
    social_posts: Optional[list] = None
    upload_manifest: Any = None  # UploadManifest — avoid circular import
    # The publish-side long-form metadata generator class (a ``LongFormMetadataGenerator``),
    # injected by the caller so the engine never imports publishing code.
    metadata_generator: Any = None

    # ── Tracking ──────────────────────────────────────────────────────
    stages_completed: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    # ── Optional parameters ───────────────────────────────────────────
    lut_path: Optional[Path] = None
    crossfade_duration: float = 0.5
    gps_path: Optional[Path] = None
    gps_date: Optional[str] = None
    gps_range: Optional[str] = None
    draft: bool = False
    fresh: bool = False

    # ── Run-time extras (populated during _build_context) ────────────
    params: dict = field(default_factory=dict)
    num_shorts: int = 0
    metadata_file: Optional[Path] = None
    final_video: Optional[Path] = None

    def require_metadata_generator(self):
        """Return the injected long-form metadata generator class, or raise.

        The engine never imports the publish-side generator; the caller (the CLI,
        the batch runner) hands it to ``PipelineOrchestrator(metadata_generator=...)``.
        """
        if self.metadata_generator is None:
            raise RuntimeError(
                "no metadata generator was injected: construct the orchestrator with "
                "PipelineOrchestrator(config, metadata_generator=LongFormMetadataGenerator)"
            )
        return self.metadata_generator

    # ── Stage names ───────────────────────────────────────────────────
    # Canonical stage name constants. Used by skip_stages, stages_completed,
    # and stage wrapper .name attributes.
    STAGE_VALIDATION = "validation"
    STAGE_ANALYTICS_SYNC = "analytics_sync"
    STAGE_DETECT_PROPERTIES = "detect_properties"
    STAGE_COPY_TO_WSL = "copy_to_wsl"
    STAGE_STOP_REMOVAL = "stop_removal"
    STAGE_BLUR_DETECTION = "blur_detection"
    STAGE_COLOR_CORRECTION = "color_correction"
    STAGE_OST_OVERLAYS = "ost_overlays"
    STAGE_AUDIO_NORMALIZATION = "audio_normalization"
    STAGE_METADATA_GENERATION = "metadata_generation"
    STAGE_CHAPTER_GENERATION = "chapter_generation"
    STAGE_THUMBNAIL_GENERATION = "thumbnail_generation"
    STAGE_SHORTS_ANALYSIS = "shorts_analysis"
    STAGE_SCHEDULE_GENERATION = "schedule_generation"
    STAGE_SOCIAL_GENERATION = "social_generation"
    STAGE_DRIVE_UPLOAD = "drive_upload"
    STAGE_CSV_GENERATION = "csv_generation"
    STAGE_D_DRIVE_COPY = "d_drive_copy"
    STAGE_VALIDATION_GATE = "validation_gate"
    STAGE_CLEANUP = "cleanup"
    STAGE_PRE_RENDER_REVIEW = "pre_render_review"


# ── Skip flag mapping ─────────────────────────────────────────────────
# Maps single-video pipeline skip toggles to stage names for
# PipelineContext.skip_stages. Most keys are argparse dests (--skip-color etc.);
# a few (skip_audio, skip_ost, skip_metadata) are API-only — fed by
# RunOptions.from_legacy_flags(...)/PipelineOrchestrator.run(skip_audio=), not a
# --skip-* CLI flag on the single-video subparser. NOTE: this mapping is the
# SINGLE-VIDEO path only. The production DRIVE path does not use these stages;
# its audio-skip switch is config.drive.skip_audio (--no-audio, threaded via the
# flag registry into the day-master encode).
SKIP_FLAG_TO_STAGE: dict[str, str] = {
    "skip_color": PipelineContext.STAGE_COLOR_CORRECTION,
    "skip_stops": PipelineContext.STAGE_STOP_REMOVAL,
    "skip_ost": PipelineContext.STAGE_OST_OVERLAYS,
    "skip_audio": PipelineContext.STAGE_AUDIO_NORMALIZATION,
    "skip_metadata": PipelineContext.STAGE_METADATA_GENERATION,
    "skip_chapters": PipelineContext.STAGE_CHAPTER_GENERATION,
    "skip_thumbnails": PipelineContext.STAGE_THUMBNAIL_GENERATION,
    "skip_shorts": PipelineContext.STAGE_SHORTS_ANALYSIS,
    "skip_schedule": PipelineContext.STAGE_SCHEDULE_GENERATION,
    "skip_social": PipelineContext.STAGE_SOCIAL_GENERATION,
    "skip_upload": PipelineContext.STAGE_DRIVE_UPLOAD,
    "skip_csv": PipelineContext.STAGE_CSV_GENERATION,
    "skip_d_copy": PipelineContext.STAGE_D_DRIVE_COPY,
    "skip_cleanup": PipelineContext.STAGE_CLEANUP,
}
