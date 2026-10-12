"""Stage: Metricool CSV generation."""

import json
import logging

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


class CSVGenerationStage(BaseStage):
    name = PipelineContext.STAGE_CSV_GENERATION

    def run(self, ctx):
        from .. import orchestrator as _orch

        CSVGenerator = _orch.CSVGenerator  # noqa: N806
        UploadManifest = _orch.UploadManifest  # noqa: N806
        output_dir = ctx.paths.project_dir
        try:
            social_posts = ctx.social_posts
            if social_posts is None:
                pf = output_dir / "shorts" / "social_posts.json"
                if pf.exists():
                    with open(pf) as f:
                        social_posts = json.load(f)
            schedule = ctx.schedule
            if schedule is None:
                import yaml

                sf = output_dir / "release_schedule.yaml"
                if sf.exists():
                    with open(sf) as f:
                        sd = yaml.safe_load(f)
                    from ..scheduler import ReleaseSchedule, ScheduledSlot

                    slots = [ScheduledSlot(**s) for s in sd.get("schedule", [])]
                    schedule = ReleaseSchedule(
                        project_name=sd.get("project_name", ctx.project_name),
                        location=sd.get("location", ctx.location),
                        long_form_publish=sd.get("long_form_publish", ""),
                        window_start=sd.get("window_start", ""),
                        window_end=sd.get("window_end", ""),
                        num_shorts=sd.get("num_shorts", 0),
                        num_slots=sd.get("num_slots", 0),
                        slots=slots,
                    )
            from ..social_generator import SocialPost

            po = [SocialPost(**p) if isinstance(p, dict) else p for p in (social_posts or [])]
            um = ctx.upload_manifest
            if um is None:
                mf = output_dir / "shorts" / "gdrive_manifest.json"
                if mf.exists():
                    um = UploadManifest.load(mf)
            manifest = um or UploadManifest(project_name=ctx.project_name)
            cg = CSVGenerator()
            csv_path = cg.generate(
                project_name=ctx.project_name,
                social_posts=po,
                schedule=schedule,
                manifest=manifest,
                output_dir=str(output_dir),
                validate=bool(manifest.files),
            )
            logger.info(f"  CSV: {csv_path}")
        except Exception as e:
            logger.error(f"  CSV generation failed: {e}")
            return StageResult(success=False, error=f"CSV generation failed: {e}")
        return StageResult(success=True)
