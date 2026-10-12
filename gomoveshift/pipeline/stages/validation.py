"""Stage: Input validation and setup."""

import logging
import re

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)

_GPS_REQUIRED_REGIONS = {
    "Ireland",
    "Kerry",
    "Clare",
    "Galway",
    "Tipperary",
    "Cork",
    "Donegal",
    "Mayo",
    "Sligo",
    "Wicklow",
    "Connemara",
    "Burren",
    "Dingle",
    "Portumna",
    "Cloonlee",
    "North Carolina",
    "NC",
}


class ValidationStage(BaseStage):
    name = PipelineContext.STAGE_VALIDATION

    def should_run(self, ctx):
        return True

    def run(self, ctx):
        from ..wsl_mount import ensure_wsl_mount

        for mount_path in [ctx.source_video, ctx.paths.output_base]:
            if str(mount_path).startswith("/mnt/") and not ensure_wsl_mount(mount_path):
                return StageResult(
                    success=False, error=f"Windows drive mount not accessible: {mount_path}"
                )
        if not ctx.source_video.exists():
            return StageResult(success=False, error=f"Video file not found: {ctx.source_video}")
        source_file_size = ctx.source_video.stat().st_size
        if source_file_size < 1024:
            return StageResult(
                success=False,
                error=f"Source file too small ({source_file_size} bytes): {ctx.source_video}",
            )
        if ctx.source_video.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mxf"}:
            return StageResult(
                success=False, error=f"Unsupported video format: {ctx.source_video.suffix}"
            )
        if ctx.gps_path is None:
            try:
                from ..gopro_gps import detect_gpmf_stream, extract_gopro_gpx

                if detect_gpmf_stream(ctx.source_video):
                    logger.info("GoPro GPMF telemetry detected — extracting GPS...")
                    _extracted = extract_gopro_gpx(ctx.source_video, ctx.paths.stages_dir)
                    if _extracted:
                        ctx.gps_path = _extracted
                        logger.info(f"  Using extracted GPS: {_extracted.name}")
                        if ctx.gps_date is None:
                            from ..gopro_gps import extract_recording_date

                            _rec_date = extract_recording_date(ctx.source_video)
                            if _rec_date:
                                ctx.gps_date = _rec_date.strftime("%Y-%m-%d")
            except ImportError:
                pass
        if (not ctx.location or not ctx.location.strip()) and ctx.gps_path:
            from ..ingest import derive_location_from_gpx

            location = derive_location_from_gpx(ctx.gps_path)
            if location:
                ctx.location = location
        if not ctx.location or not ctx.location.strip():
            return StageResult(success=False, error="Location cannot be empty.")
        ctx.location = ctx.location.strip()
        _loc_lower = ctx.location.lower()
        _needs_gps = any(
            re.search(r"\b" + re.escape(r.lower()) + r"\b", _loc_lower)
            for r in _GPS_REQUIRED_REGIONS
        )
        if _needs_gps and ctx.gps_path is None:
            _auto_gpx = ctx.source_video.with_suffix(".gpx")
            _dir_gpx = list(ctx.source_video.parent.glob("*.gpx"))
            if not _auto_gpx.exists() and not _dir_gpx:
                return StageResult(
                    success=False,
                    error=f"GPS data REQUIRED for '{ctx.location}' but no --gps file provided.",
                )
        return StageResult(success=True)
