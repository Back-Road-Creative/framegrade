"""Stage: Copy source video to WSL filesystem for faster I/O."""

import logging
import time

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


class WSLCopyStage(BaseStage):
    name = PipelineContext.STAGE_COPY_TO_WSL

    def should_run(self, ctx):
        if self.name in ctx.skip_stages or self.name in ctx.stages_completed:
            return False
        if PipelineContext.STAGE_STOP_REMOVAL in ctx.stages_completed:
            return False
        if not str(ctx.source_video).startswith("/mnt/"):
            return False
        return True

    def run(self, ctx):
        try:
            ctx.paths.staging_dir.mkdir(parents=True, exist_ok=True)
            wsl_copy = ctx.paths.staging_dir / ctx.source_video.name
            if wsl_copy.exists() and wsl_copy.stat().st_size == ctx.source_video.stat().st_size:
                logger.info(f"  File already copied: {wsl_copy}")
            else:
                from .. import fs_safety

                props = ctx.video_properties
                size_gb = props.file_size_bytes / (1024**3) if props else 0
                logger.info(f"  Copying {size_gb:.1f} GB to native WSL filesystem...")
                t0 = time.time()
                fs_safety.safe_copy(str(ctx.source_video), str(wsl_copy))
                if wsl_copy.stat().st_size != ctx.source_video.stat().st_size:
                    wsl_copy.unlink()
                    raise IOError(
                        f"Copy incomplete: {wsl_copy.stat().st_size} vs {ctx.source_video.stat().st_size}"
                    )
                logger.info(f"  Copy complete ({time.time() - t0:.0f}s)")
            if wsl_copy.exists():
                ctx.current_video = wsl_copy
        except Exception as e:
            logger.warning(f"  WSL copy failed ({e}), continuing with original path")
        return StageResult(success=True)
