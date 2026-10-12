"""Stage: Post-pipeline cleanup."""

import logging

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


class CleanupStage(BaseStage):
    name = PipelineContext.STAGE_CLEANUP

    def run(self, ctx):
        cleaned = 0
        try:
            if ctx.paths.staging_dir.exists():
                for f in sorted(ctx.paths.staging_dir.glob("*.mp4")):
                    if ctx.project_name.lower() in f.name.lower():
                        cleaned += f.stat().st_size
                        f.unlink()
                for f in sorted(ctx.paths.staging_dir.glob("*_color_analysis.json")):
                    if ctx.project_name.lower() in f.name.lower():
                        f.unlink()
            sd = ctx.paths.stages_dir
            if sd.exists():
                for f in sorted(sd.iterdir()):
                    if f.is_file() and f.name != "pipeline_state.json":
                        cleaned += f.stat().st_size
                        f.unlink()
            logger.info(f"  Freed {cleaned / (1024**3):.1f} GB")
        except Exception as e:
            logger.warning(f"  Cleanup error (non-fatal): {e}")
        return StageResult(success=True)
