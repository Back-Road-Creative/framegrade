"""Stage: YouTube analytics sync."""

import logging

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


class AnalyticsSyncStage(BaseStage):
    name = PipelineContext.STAGE_ANALYTICS_SYNC

    def run(self, ctx):
        try:
            from .. import orchestrator as _orch

            LongFormMetadataGenerator = _orch.LongFormMetadataGenerator  # noqa: N806
            meta_gen = LongFormMetadataGenerator()
            meta_gen._sync_youtube_analytics()
            return StageResult(success=True)
        except FileNotFoundError as e:
            # Missing analytics credentials — warn and continue
            logger.warning(f"  Analytics sync skipped (file not found): {e}")
            return StageResult(success=True)
        except Exception as e:
            logger.error(f"  Analytics sync FAILED: {e}")
            return StageResult(success=False, error=str(e), fatal=False)
