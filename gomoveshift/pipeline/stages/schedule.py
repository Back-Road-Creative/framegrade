"""Stage: Release schedule generation."""

import json
import logging

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


class ScheduleGenerationStage(BaseStage):
    name = PipelineContext.STAGE_SCHEDULE_GENERATION

    def run(self, ctx):
        from .. import orchestrator as _orch

        PipelineScheduler = _orch.PipelineScheduler  # noqa: N806
        output_dir = ctx.paths.project_dir
        if ctx.shorts_data is None:
            af = output_dir / "shorts" / "shorts_analysis.json"
            if af.exists():
                try:
                    with open(af) as f:
                        ctx.shorts_data = json.load(f)
                except Exception:
                    pass
        ctx.num_shorts = len(ctx.shorts_data["segments"]) if ctx.shorts_data else 0
        try:
            scheduler = PipelineScheduler()
            _pub_date = None
            _mp = output_dir / f"{ctx.project_name}_youtube.json"
            if _mp.exists():
                try:
                    with open(_mp) as f:
                        _meta = json.load(f)
                    _np = _meta.get("next_publish", {})
                    if _np.get("date"):
                        _pub_date = _np["date"]
                except Exception:
                    pass
            if not _pub_date:
                from ..seo_slots import default_publish_slot

                _pub_date = default_publish_slot(ctx.config, ctx.publish_day).strftime("%Y-%m-%d")
            from datetime import datetime as dt2

            rd = dt2.strptime(_pub_date, "%Y-%m-%d")
            apd = rd.strftime("%a").lower()  # the date's day label, not a slot decision
            ns = ctx.num_shorts if ctx.num_shorts > 0 else 15
            ctx.schedule = scheduler.generate_schedule(
                publish_date=_pub_date,
                publish_day=apd,
                num_shorts=ns,
                project_name=ctx.project_name,
                location=ctx.location,
            )
            sp = output_dir / "release_schedule.yaml"
            ctx.schedule.save(sp)
            logger.info(f"  Scheduled {len(ctx.schedule.slots)} shorts slots")
        except Exception as e:
            # Non-fatal: a missing/broken shorts schedule doesn't invalidate
            # the long-form deliverable already produced upstream, so the
            # run continues rather than aborting. But it must not be
            # reported as success — a truncated/unwritten schedule reaching
            # the Metricool CSV generator silently ships with slots missing.
            # Not marking stages_completed also means a resume retries this
            # stage instead of treating it as done.
            logger.error(f"  Schedule generation FAILED: {e}")
            return StageResult(success=False, error=str(e), fatal=False)
        return StageResult(success=True)
