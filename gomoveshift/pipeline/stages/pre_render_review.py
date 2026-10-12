"""Stage: Pre-render review gate — human approval before OST burn-in."""

import logging

from ..context import PipelineContext
from ..review_gate import ReviewFailedError, ReviewGate
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


class PreRenderReviewStage(BaseStage):
    name = PipelineContext.STAGE_PRE_RENDER_REVIEW

    def run(self, ctx: PipelineContext) -> StageResult:
        try:
            from ..ffmpeg_filters import shorten_ost_for_shorts

            # Show preview info
            logger.info("Pre-render review: checking OST text before burn-in")
            if ctx.location:
                logger.info(f"  OST location: {ctx.location}")
                short_text = shorten_ost_for_shorts(ctx.location)
                if short_text != ctx.location:
                    logger.info(f"  Shorts text:  {short_text}")
            if ctx.metadata:
                logger.info(f"  Title: {ctx.metadata.get('title', '(none)')}")

            # Run the review gate
            gate = ReviewGate(interactive=True)
            gate.gate_pre_render(
                lf_index=0,
                ost_text=ctx.location,
                output_dir=ctx.paths.project_dir,
            )
        except ReviewFailedError as e:
            return StageResult(success=False, error=f"Review rejected: {e}", fatal=True)
        except Exception as e:
            return StageResult(success=False, error=f"Review gate error: {e}", fatal=True)
        return StageResult(success=True)
