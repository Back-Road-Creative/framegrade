"""Stage: Detect video properties via ffprobe."""

import logging

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult
from ..video_detect import detect_video_properties

logger = logging.getLogger(__name__)


class PropertyDetectionStage(BaseStage):
    name = PipelineContext.STAGE_DETECT_PROPERTIES

    def run(self, ctx):
        try:
            props = detect_video_properties(ctx.source_video)
        except Exception as e:
            return StageResult(success=False, error=f"Failed to detect video properties: {e}")
        if props.bitrate_kbps == 0:
            return StageResult(
                success=False, error="Source bitrate_kbps=0 — ffprobe could not determine bitrate."
            )
        ctx.video_properties = props
        logger.info(f"  Resolution: {props.width}x{props.height} ({props.resolution_label})")
        logger.info(f"  FPS: {props.fps}")
        logger.info(f"  Duration: {props.duration:.1f}s ({props.duration / 60:.1f} min)")
        logger.info(f"  Codec: {props.codec} / {props.audio_codec}")
        logger.info(f"  Size: {props.file_size_bytes / 1024 / 1024:.1f} MB")
        return StageResult(success=True)
