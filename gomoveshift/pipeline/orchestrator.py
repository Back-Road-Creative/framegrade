"""
Pipeline Orchestrator — master coordinator for long form video processing.

Takes an arranged video export and produces a polished long form with stops
removed, color corrected, OST overlays, and normalized audio.

run() is a stage loop. Each stage is a BaseStage subclass in pipeline/stages/.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import Config, get_config
from ..paths import gms_output_dir
from .context import SKIP_FLAG_TO_STAGE, PipelineContext, PipelinePaths
from .csv_generator import CSVGenerator  # noqa: F401 — stages import via _orch
from .ffmpeg_runner import validate_video
from .pipeline_validator import PipelineValidator  # noqa: F401
from .scheduler import PipelineScheduler  # noqa: F401
from .shorts_extractor import PipelineShortsExtractor  # noqa: F401
from .single_pass import SinglePassProcessor  # noqa: F401
from .social_generator import SocialGenerator  # noqa: F401
from .state_manager import can_resume, load_state, save_state
from .stop_processor import DEFAULT_CACHE_DIR as MOTION_CACHE_DIR  # noqa: F401
from .stop_processor import (
    StopProcessor,  # noqa: F401
    compute_file_hash,
)
from .sub_pipelines import (
    IngestAndPrep,
    LightMetadata,
    PublishAndArchive,
    RenderAndDerive,
)
from .thumbnail_text_overlay import (
    generate_thumbnails_from_video,  # noqa: F401 — stages/thumbnails.py imports via _orch
)
from .uploader import PipelineUploader, UploadManifest  # noqa: F401
from .video_detect import VideoProperties, detect_video_properties
from .wsl_mount import ensure_wsl_mount


def extract_series_info(project_name: str) -> tuple:
    """Extract series base name and part number from project name."""
    import re

    m = re.match(r"^([a-z]+[a-z])(\d+)(?:_\d+p)?$", project_name, re.IGNORECASE)
    if m:
        return (m.group(1).lower(), int(m.group(2)))
    return (None, None)


def make_series_title(named_route, part_number, total_parts, resolution_label):
    """Generate a series-aware title for multi-part videos."""
    return f"Driving the {named_route} Part {part_number} | {resolution_label} POV"


try:
    from process_tracker import ProcessTracker
except ImportError:
    ProcessTracker = None

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a complete pipeline run."""

    success: bool
    project_name: str
    output_dir: Path
    final_video: Optional[Path] = None
    metadata_file: Optional[Path] = None
    video_properties: Optional[VideoProperties] = None
    stages_completed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


@dataclass
class RunOptions:
    """All inputs for a pipeline run.

    Replaces the 27-parameter ``run()`` signature. The ``skip`` set holds
    canonical stage names (values from ``SKIP_FLAG_TO_STAGE``) and replaces
    14 individual ``skip_*`` booleans.

    Use ``RunOptions.from_legacy()`` to convert old-style ``skip_*`` kwargs.
    """

    video_path: Path
    location: str
    publish_day: str
    output_base: Path = field(default_factory=gms_output_dir)
    lut_path: Optional[Path] = None
    crossfade_duration: float = 0.5
    gps_path: Optional[Path] = None
    gps_date: Optional[str] = None
    gps_range: Optional[str] = None
    draft: bool = False
    fresh: bool = False
    skip: set[str] = field(default_factory=set)

    @classmethod
    def from_legacy(  # noqa: PLR0913 — transitional bridge
        cls,
        video_path: Path,
        location: str,
        publish_day: str,
        output_base: Optional[Path] = None,
        lut_path: Optional[Path] = None,
        crossfade_duration: float = 0.5,
        gps_path: Optional[Path] = None,
        gps_date: Optional[str] = None,
        gps_range: Optional[str] = None,
        draft: bool = False,
        fresh: bool = False,
        skip_color: bool = False,
        skip_stops: bool = False,
        skip_ost: bool = False,
        skip_audio: bool = False,
        skip_metadata: bool = False,
        skip_chapters: bool = False,
        skip_thumbnails: bool = False,
        skip_shorts: bool = False,
        skip_schedule: bool = False,
        skip_social: bool = False,
        skip_upload: bool = False,
        skip_csv: bool = False,
        skip_d_copy: bool = False,
        skip_cleanup: bool = False,
    ) -> RunOptions:
        """Build ``RunOptions`` from legacy ``skip_*`` boolean kwargs."""
        skip_flags = {
            "skip_color": skip_color,
            "skip_stops": skip_stops,
            "skip_ost": skip_ost,
            "skip_audio": skip_audio,
            "skip_metadata": skip_metadata,
            "skip_chapters": skip_chapters,
            "skip_thumbnails": skip_thumbnails,
            "skip_shorts": skip_shorts,
            "skip_schedule": skip_schedule,
            "skip_social": skip_social,
            "skip_upload": skip_upload,
            "skip_csv": skip_csv,
            "skip_d_copy": skip_d_copy,
            "skip_cleanup": skip_cleanup,
        }
        skip_stages = {
            SKIP_FLAG_TO_STAGE[f] for f, v in skip_flags.items() if v and f in SKIP_FLAG_TO_STAGE
        }
        return cls(
            video_path=video_path,
            location=location,
            publish_day=publish_day,
            output_base=output_base if output_base is not None else gms_output_dir(),
            lut_path=lut_path,
            crossfade_duration=crossfade_duration,
            gps_path=gps_path,
            gps_date=gps_date,
            gps_range=gps_range,
            draft=draft,
            fresh=fresh,
            skip=skip_stages,
        )


@dataclass
class _PipelinePaths:
    """Single source of truth for all pipeline file locations."""

    output_base: Path
    project_dir: Path
    stages_dir: Path
    state_file: Path
    staging_dir: Path

    @classmethod
    def resolve(
        cls, output_base: Path, project_name: str, intermediate_folder: str = ""
    ) -> "_PipelinePaths":
        project_dir = output_base / project_name
        from ..paths import gms_intermediate_dir, gms_staging_dir

        stages_dir = gms_intermediate_dir(output_base, project_name, intermediate_folder)
        staging_dir = gms_staging_dir()
        # State file stays on drvfs (small, needs persistence across /tmp wipes)
        drvfs_stages = project_dir / "_stages"
        return cls(
            output_base=output_base,
            project_dir=project_dir,
            stages_dir=stages_dir,
            state_file=drvfs_stages / "pipeline_state.json",
            staging_dir=staging_dir,
        )


class PipelineOrchestrator:
    """Master coordinator that manages all pipeline stages in order."""

    PUBLISH_DAYS = {
        "mon": "Monday",
        "tue": "Tuesday",
        "wed": "Wednesday",
        "thu": "Thursday",
        "fri": "Friday",
        "sat": "Saturday",
        "sun": "Sunday",
    }

    def __init__(self, config: Optional[Config] = None, metadata_generator=None):
        self.config = config or get_config()
        # Publish-side ``LongFormMetadataGenerator`` class, handed in by the
        # caller; the metadata, chapter and analytics-sync stages read it from
        # the run context so the engine imports no publishing code.
        self.metadata_generator = metadata_generator

    @staticmethod
    def _ensure_wsl_mount(path, retries=3, delay=2.0):
        return ensure_wsl_mount(path, retries=retries, delay=delay)

    def _save_state(
        self,
        state_file,
        project_name,
        source_video,
        source_hash,
        stages_completed,
        current_video,
        video_properties,
        timings,
        params,
        keep_segments=None,
    ):
        save_state(
            state_file=state_file,
            project_name=project_name,
            source_video=source_video,
            source_hash=source_hash,
            stages_completed=stages_completed,
            current_video=current_video,
            video_properties=video_properties,
            timings=timings,
            params=params,
            keep_segments=keep_segments,
        )

    def _save_state_after_stage(self, ctx):
        """Persist ctx after a stage completes — shared by sub-pipelines and the legacy loop.

        Guard: never write before PropertyDetectionStage has populated
        ``ctx.video_properties`` (state-file format requires a real
        VideoProperties block).
        """
        if ctx.video_properties is None:
            return
        self._save_state(
            state_file=ctx.paths.state_file,
            project_name=ctx.project_name,
            source_video=ctx.source_video,
            source_hash=ctx.source_hash,
            stages_completed=ctx.stages_completed,
            current_video=ctx.current_video,
            video_properties=ctx.video_properties,
            timings=ctx.timings,
            params=ctx.params,
            keep_segments=ctx.keep_segments,
        )

    def _load_state(self, state_file):
        return load_state(state_file)

    def _can_resume(self, state, source_video, params):
        source_hash = compute_file_hash(source_video)
        return can_resume(state, source_hash, params)

    def _build_context(
        self,
        video_path,
        location,
        publish_day,
        output_base,
        lut_path,
        crossfade_duration,
        fresh,
        gps_path,
        gps_date,
        gps_range,
        draft,
        skip_stages,
    ):
        video_path = Path(video_path).resolve()
        publish_day = publish_day.lower()[:3]
        if publish_day not in self.PUBLISH_DAYS:
            raise ValueError(f"Invalid publish day: '{publish_day}'.")
        project_name = video_path.stem.replace(" ", "_").lower()
        paths_obj = _PipelinePaths.resolve(
            output_base,
            project_name,
            intermediate_folder=self.config.paths.intermediate_folder,
        )
        paths_obj.project_dir.mkdir(parents=True, exist_ok=True)
        paths_obj.stages_dir.mkdir(exist_ok=True)
        params = {
            "location": location,
            "publish_day": publish_day,
            "skip_color": PipelineContext.STAGE_COLOR_CORRECTION in skip_stages,
            "skip_stops": PipelineContext.STAGE_STOP_REMOVAL in skip_stages,
            "gps_path": str(gps_path) if gps_path else None,
            "gps_date": gps_date,
            "gps_range": gps_range,
        }
        source_hash = compute_file_hash(video_path)
        ctx_paths = PipelinePaths(
            output_base=paths_obj.output_base,
            project_dir=paths_obj.project_dir,
            stages_dir=paths_obj.stages_dir,
            state_file=paths_obj.state_file,
            staging_dir=paths_obj.staging_dir,
        )
        ctx = PipelineContext(
            project_name=project_name,
            source_video=video_path,
            source_hash=source_hash,
            config=self.config,
            location=location,
            publish_day=publish_day,
            paths=ctx_paths,
            skip_stages=skip_stages,
            current_video=video_path,
            lut_path=lut_path,
            crossfade_duration=crossfade_duration,
            gps_path=gps_path,
            gps_date=gps_date,
            gps_range=gps_range,
            draft=draft,
            fresh=fresh,
            params=params,
            metadata_generator=self.metadata_generator,
        )
        existing_state = None if fresh else self._load_state(paths_obj.state_file)
        if existing_state and self._can_resume(existing_state, video_path, params):
            self._restore_from_state(ctx, existing_state, paths_obj)
        elif existing_state and not fresh:
            logger.warning("  Cannot resume - starting from scratch")
        return ctx

    def _restore_from_state(self, ctx, existing_state, paths):
        logger.info("=" * 60)
        logger.info("  RESUMING PIPELINE")
        logger.info(f"  Stages already complete: {len(existing_state['stages_completed'])}")
        logger.info("=" * 60)
        ctx.stages_completed = existing_state["stages_completed"]
        ctx.timings = existing_state.get("timings", {})
        ctx.current_video = Path(existing_state["current_video"])
        saved_params = existing_state.get("params", {})
        if ctx.gps_path is None and saved_params.get("gps_path"):
            ctx.gps_path = Path(saved_params["gps_path"])
            ctx.params["gps_path"] = saved_params["gps_path"]
        if ctx.gps_date is None and saved_params.get("gps_date"):
            ctx.gps_date = saved_params["gps_date"]
            ctx.params["gps_date"] = ctx.gps_date
        if ctx.gps_range is None and saved_params.get("gps_range"):
            ctx.gps_range = saved_params["gps_range"]
            ctx.params["gps_range"] = ctx.gps_range
        if str(ctx.current_video).startswith("/mnt/"):
            wsl_copy = paths.staging_dir / ctx.current_video.name
            if wsl_copy.exists() and wsl_copy.stat().st_size == ctx.current_video.stat().st_size:
                wsl_val = validate_video(wsl_copy, require_audio=False)
                if wsl_val.valid:
                    ctx.current_video = wsl_copy
        props_data = existing_state["video_properties"]
        props = VideoProperties(
            width=props_data["width"],
            height=props_data["height"],
            fps=props_data["fps"],
            duration=props_data["duration"],
            bitrate_kbps=props_data.get("bitrate_kbps", 0),
            codec=props_data.get("codec", "unknown"),
            audio_codec=props_data.get("audio_codec", "unknown"),
            file_size_bytes=props_data.get("file_size_bytes", 0),
        )
        ctx.video_properties = props
        if props.bitrate_kbps == 0:
            raise ValueError("Resumed state has bitrate_kbps=0. Use --fresh.")
        saved_ks = existing_state.get("keep_segments")
        if saved_ks is not None:
            ctx.keep_segments = [(s, e) for s, e in saved_ks]

    def run(  # noqa: PLR0913 — legacy kwargs kept for backward compat
        self,
        options: Optional[RunOptions] = None,
        *,
        video_path: Optional[Path] = None,
        location: Optional[str] = None,
        publish_day: Optional[str] = None,
        output_base: Optional[Path] = None,
        lut_path: Optional[Path] = None,
        crossfade_duration: float = 0.5,
        skip_color: bool = False,
        skip_stops: bool = False,
        fresh: bool = False,
        skip_ost: bool = False,
        skip_audio: bool = False,
        skip_metadata: bool = False,
        skip_chapters: bool = False,
        skip_thumbnails: bool = False,
        skip_shorts: bool = False,
        skip_schedule: bool = False,
        skip_social: bool = False,
        skip_upload: bool = False,
        skip_csv: bool = False,
        skip_d_copy: bool = False,
        gps_path: Optional[Path] = None,
        gps_date: Optional[str] = None,
        gps_range: Optional[str] = None,
        draft: bool = False,
        skip_cleanup: bool = False,
    ) -> PipelineResult:
        """Run the full long form processing pipeline.

        Preferred: pass a single ``RunOptions`` instance.
        Legacy: pass individual keyword arguments (deprecated, will be removed).
        """
        if options is None:
            # Legacy path — build RunOptions from kwargs
            if video_path is None or location is None or publish_day is None:
                raise TypeError(
                    "run() requires either a RunOptions instance or "
                    "video_path, location, and publish_day keyword arguments"
                )
            warnings.warn(
                "Passing individual keyword arguments to PipelineOrchestrator.run() "
                "is deprecated. Use RunOptions instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            options = RunOptions.from_legacy(
                video_path=video_path,
                location=location,
                publish_day=publish_day,
                output_base=output_base,
                lut_path=lut_path,
                crossfade_duration=crossfade_duration,
                gps_path=gps_path,
                gps_date=gps_date,
                gps_range=gps_range,
                draft=draft,
                fresh=fresh,
                skip_color=skip_color,
                skip_stops=skip_stops,
                skip_ost=skip_ost,
                skip_audio=skip_audio,
                skip_metadata=skip_metadata,
                skip_chapters=skip_chapters,
                skip_thumbnails=skip_thumbnails,
                skip_shorts=skip_shorts,
                skip_schedule=skip_schedule,
                skip_social=skip_social,
                skip_upload=skip_upload,
                skip_csv=skip_csv,
                skip_d_copy=skip_d_copy,
                skip_cleanup=skip_cleanup,
            )
        total_start = time.time()
        skip_stages = set(options.skip)
        try:
            ctx = self._build_context(
                video_path=options.video_path,
                location=options.location,
                publish_day=options.publish_day,
                output_base=options.output_base,
                lut_path=options.lut_path,
                crossfade_duration=options.crossfade_duration,
                fresh=options.fresh,
                gps_path=options.gps_path,
                gps_date=options.gps_date,
                gps_range=options.gps_range,
                draft=options.draft,
                skip_stages=skip_stages,
            )
        except ValueError as e:
            return self._fail(str(e))
        result = PipelineResult(
            success=False,
            project_name=ctx.project_name,
            output_dir=ctx.paths.project_dir,
            video_properties=ctx.video_properties,
            stages_completed=ctx.stages_completed,
        )
        logger.info(f"Output directory: {ctx.paths.project_dir}")
        # The 19 long-form stages are grouped into four sub-pipelines that
        # run in fixed order. Each sub-pipeline iterates its slice with the
        # same skip/resume/fatal-abort semantics as the legacy loop; the
        # canonical flat ordering still lives in pipeline.stages.STAGE_ORDER
        # (validated by tests/test_stage_ordering.py and
        # tests/test_sub_pipelines.py).
        for sub in (
            IngestAndPrep(),
            LightMetadata(),
            RenderAndDerive(),
            PublishAndArchive(),
        ):
            sub_result = sub.run(ctx, self._save_state_after_stage)
            if not sub_result.success and sub_result.fatal:
                result.stages_completed = ctx.stages_completed
                result.timings = ctx.timings
                return self._fail(sub_result.error, result)
        result.stages_completed = ctx.stages_completed
        result.timings = ctx.timings
        result.video_properties = ctx.video_properties
        result.final_video = ctx.final_video
        result.metadata_file = ctx.metadata_file
        result.success = True
        total_time = time.time() - total_start
        result.timings["total"] = total_time
        self._log_header("PIPELINE COMPLETE")
        logger.info(f"  Project: {ctx.project_name}")
        logger.info(f"  Final video: {result.final_video}")
        logger.info(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        logger.info(f"  Stages: {' -> '.join(result.stages_completed)}")
        from . import encoding_constants as ec

        props = ctx.video_properties
        summary = {
            "project_name": ctx.project_name,
            "source_video": str(ctx.source_video),
            "location": ctx.location,
            "publish_day": self.PUBLISH_DAYS[ctx.publish_day],
            "final_video": str(result.final_video),
            "metadata_file": str(result.metadata_file),
            "video_properties": {
                "width": props.width,
                "height": props.height,
                "fps": props.fps,
                "duration": props.duration,
                "resolution": props.resolution_label,
            }
            if props
            else {},
            "encoding_params": {
                "longform_crf": ec.LONGFORM_CRF,
                "longform_nvenc_cq": ec.LONGFORM_NVENC_CQ,
                "longform_nvenc_preset": ec.LONGFORM_NVENC_PRESET,
                "longform_preset": ec.LONGFORM_PRESET,
                "longform_profile": ec.LONGFORM_PROFILE,
                "longform_pix_fmt": ec.LONGFORM_PIX_FMT,
                "longform_audio_lufs": ec.LONGFORM_AUDIO_LUFS,
            },
            "stages_completed": result.stages_completed,
            "timings": result.timings,
        }
        with open(ctx.paths.project_dir / f"{ctx.project_name}_pipeline_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return result

    def _detect_video_properties(self, video_path):
        return detect_video_properties(video_path)

    @staticmethod
    def _extract_recording_date(gpx_date=None, ffprobe_creation_time=None, project_name=""):
        from datetime import datetime as dt

        if gpx_date is not None:
            if isinstance(gpx_date, str):
                try:
                    return dt.strptime(gpx_date, "%Y-%m-%d")
                except ValueError:
                    pass
            return gpx_date
        if ffprobe_creation_time:
            for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
                try:
                    return dt.strptime(ffprobe_creation_time, fmt)
                except ValueError:
                    continue
        import re

        m = re.match(r"(\d{4})-(\d{2})-(\d{2})", project_name)
        if m:
            try:
                return dt(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
        m = re.match(r"(\d{4})(\d{2})(\d{2})", project_name)
        if m:
            try:
                return dt(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
        return None

    def _fail(self, message, result=None, tracker=None):
        if tracker:
            tracker.fail(message)
        logger.error(f"PIPELINE FAILED: {message}")
        if result is None:
            result = PipelineResult(success=False, project_name="unknown", output_dir=Path("."))
        result.success = False
        result.errors.append(message)
        try:
            if result and hasattr(result, "output_dir") and result.output_dir:
                sf = result.output_dir / "_stages" / "pipeline_state.json"
                vp = getattr(result, "video_properties", None)
                if vp is None:
                    vp = VideoProperties(
                        width=0,
                        height=0,
                        fps=0.0,
                        duration=0.0,
                        codec="",
                        audio_codec="",
                        file_size_bytes=0,
                    )
                self._save_state(
                    state_file=sf,
                    project_name=getattr(result, "project_name", "unknown"),
                    source_video=Path("."),
                    source_hash="",
                    stages_completed=getattr(result, "stages_completed", []),
                    current_video=getattr(result, "final_video", None) or Path("."),
                    video_properties=vp,
                    timings={},
                    params={"failed": True, "error": message},
                )
        except Exception as e:
            logger.warning("Failed to save failure state: %s", e)
        return result

    @staticmethod
    def _count_series_parts(output_base, series_base):
        import re

        count = 0
        if output_base.exists():
            pat = re.compile(rf"^{re.escape(series_base)}(\d+)", re.IGNORECASE)
            for d in output_base.iterdir():
                if d.is_dir() and pat.match(d.name):
                    count += 1
        return max(count, 1)

    def _log_header(self, title):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  {title}")
        logger.info("=" * 60)
