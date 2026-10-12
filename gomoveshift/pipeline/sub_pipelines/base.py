"""SubPipeline base — groups a slice of pipeline stages into a named unit."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

from ..context import PipelineContext
from ..stage_protocol import BaseStage

logger = logging.getLogger(__name__)

# Callback the orchestrator supplies so sub-pipelines persist state through
# the same code path used by the legacy loop (single state file, single
# write-guard on ctx.video_properties).
SaveStateFn = Callable[[PipelineContext], None]


@dataclass
class SubPipelineResult:
    """Outcome of one sub-pipeline run.

    ``fatal=True`` signals the outer orchestrator to abort the sequence,
    matching the ``StageResult.fatal`` contract of the legacy loop.
    """

    success: bool
    error: str = ""
    fatal: bool = False


class SubPipeline:
    """Iterates an ordered list of stages with the legacy loop's semantics.

    Subclasses set ``name`` and ``stages``. The behavior here mirrors the
    inner loop body of ``PipelineOrchestrator.run()``, with no observable
    change to resume, skip, or fail-fast.
    """

    name: str = ""
    stages: list[BaseStage] = []

    def run(self, ctx: PipelineContext, save_state: SaveStateFn) -> SubPipelineResult:
        for stage in self.stages:
            if not stage.should_run(ctx):
                if stage.name in ctx.skip_stages:
                    logger.info(f"{stage.name}: SKIPPED (user flag)")
                elif stage.name in ctx.stages_completed:
                    logger.info(f"{stage.name}: SKIPPED (already complete)")
                continue
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"  {stage.name.upper().replace('_', ' ')}")
            logger.info("=" * 60)
            t0 = time.time()
            stage_result = stage.run(ctx)
            ctx.timings[stage.name] = time.time() - t0
            if stage_result.success:
                if stage.name not in ctx.stages_completed:
                    ctx.stages_completed.append(stage.name)
                save_state(ctx)
            elif stage_result.fatal:
                return SubPipelineResult(success=False, error=stage_result.error, fatal=True)
            else:
                logger.warning(f"  {stage.name} failed (non-fatal): {stage_result.error}")
        return SubPipelineResult(success=True)
