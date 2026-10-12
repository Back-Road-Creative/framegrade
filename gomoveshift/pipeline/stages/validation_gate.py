"""Stage: Pre-publish validation gate."""

import json
import logging

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


class ValidationGateStage(BaseStage):
    name = PipelineContext.STAGE_VALIDATION_GATE

    def should_run(self, ctx):
        if self.name in ctx.skip_stages or self.name in ctx.stages_completed:
            return False
        if not ctx.final_video or not ctx.final_video.exists():
            return False
        if ctx.final_video.stat().st_size <= 1024 * 1024:
            return False
        return True

    def run(self, ctx):
        from .. import orchestrator as _orch

        PipelineValidator = _orch.PipelineValidator  # noqa: N806
        output_dir = ctx.paths.project_dir
        props = ctx.video_properties
        gv = PipelineValidator(
            max_lufs=ctx.config.output.audio_max_lufs,
            max_true_peak=ctx.config.output.audio_target_tp,
            ffprobe_timeout_sec=ctx.config.drive.ffprobe_timeout_sec,
        )
        gm, gsp = None, None
        mp = output_dir / f"{ctx.project_name}_youtube.json"
        if mp.exists():
            with open(mp) as f:
                gm = json.load(f)
        sp = output_dir / "shorts" / "social_posts.json"
        if sp.exists():
            with open(sp) as f:
                gsp = json.load(f)
        skip_audio = PipelineContext.STAGE_AUDIO_NORMALIZATION in ctx.skip_stages
        gr = gv.validate(
            video_path=ctx.final_video,
            expected_duration=props.duration if props else None,
            metadata_json=gm,
            social_posts=gsp,
            skip_audio=skip_audio,
        )
        if not gr.passed:
            # fatal=True (explicit): a failed audio/content gate MUST abort the
            # sub-pipeline before the irreversible Drive/YouTube upload — never
            # warn-continue into d_drive_copy with an invalid encode.
            return StageResult(
                success=False, error=f"Validation gate failed:\n{gr.summary()}", fatal=True
            )
        logger.info(f"  All gates passed:\n{gr.summary()}")

        # ── Shorts validation ──────────────────────────────────────────
        shorts_dir = output_dir / "shorts"
        if shorts_dir.is_dir() and any(shorts_dir.glob("short_*.mp4")):
            logger.info("Running shorts quality validation...")
            sr = gv.validate_shorts(shorts_dir)
            if not sr.passed:
                return StageResult(
                    success=False,
                    error=f"Shorts validation failed:\n{sr.summary()}",
                    fatal=True,
                )
            logger.info(f"  Shorts gates passed:\n{sr.summary()}")

        return StageResult(success=True)
