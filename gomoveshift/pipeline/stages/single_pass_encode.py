"""Stage: Single-pass encode (color + OST + audio)."""

import logging

from .. import blur_gate
from ..color_filters import cadence_kwargs
from ..context import PipelineContext
from ..ffmpeg_runner import check_disk_space
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


class SinglePassEncodeStage(BaseStage):
    name = "single_pass_encode"

    def should_run(self, ctx):
        if self.name in ctx.stages_completed:
            return False
        cc = PipelineContext.STAGE_COLOR_CORRECTION
        ost = PipelineContext.STAGE_OST_OVERLAYS
        aud = PipelineContext.STAGE_AUDIO_NORMALIZATION
        if (
            cc in ctx.stages_completed
            and ost in ctx.stages_completed
            and aud in ctx.stages_completed
        ):
            return False
        if cc in ctx.skip_stages and ost in ctx.skip_stages and aud in ctx.skip_stages:
            return False
        return True

    def run(self, ctx):
        cc = PipelineContext.STAGE_COLOR_CORRECTION
        ost = PipelineContext.STAGE_OST_OVERLAYS
        aud = PipelineContext.STAGE_AUDIO_NORMALIZATION
        if cc in ctx.skip_stages and ost in ctx.skip_stages and aud in ctx.skip_stages:
            ctx.final_video = ctx.current_video
            return StageResult(success=True)

        from .. import orchestrator as _orch

        validate_video = _orch.validate_video
        _SinglePassProcessor = _orch.SinglePassProcessor  # noqa: N806
        props = ctx.video_properties
        output_dir = ctx.paths.project_dir
        aud = PipelineContext.STAGE_AUDIO_NORMALIZATION

        all_complete = (
            cc in ctx.stages_completed
            and ost in ctx.stages_completed
            and aud in ctx.stages_completed
        )
        if all_complete:
            final_output = output_dir / f"{ctx.project_name}_FINAL.mp4"
            if not final_output.exists():
                return StageResult(
                    success=False,
                    error="Resume: FINAL.mp4 missing. Use --fresh to re-encode.",
                )
            if props.bitrate_kbps > 0 and props.duration > 0:
                output_size = final_output.stat().st_size
                output_bitrate_kbps = int(output_size * 8 / props.duration / 1000)
                min_expected = int(props.bitrate_kbps * 0.90)
                if output_bitrate_kbps < min_expected:
                    return StageResult(
                        success=False,
                        error=(
                            f"Resume QUALITY GATE: Output {output_bitrate_kbps}kbps "
                            f"< 90% of source {props.bitrate_kbps}kbps."
                        ),
                    )
            # Privacy gate: a FINAL accepted without re-encoding must prove it was
            # blurred — never ship a pre-blur master once blur is enabled.
            blur_err = blur_gate.check_applied(ctx, final_output)
            if blur_err:
                return StageResult(success=False, error=blur_err)
            ctx.final_video = final_output
            return StageResult(success=True)

        skip_color = cc in ctx.skip_stages
        skip_ost = ost in ctx.skip_stages
        skip_audio = aud in ctx.skip_stages

        if skip_color:
            # --skip-color is a diagnostic escape: the FINAL is UNGRADED (raw
            # camera color), not a production master. Say so loudly so an
            # ungraded output can't be mistaken for a publishable one. (The
            # day-first drive path never sets this — encode_day_master always
            # grades — so production cannot reach here ungraded.)
            logger.warning(
                "⚠ --skip-color active for %s: the FINAL is UNGRADED (raw "
                "camera color) — a diagnostic output only; do not publish it as "
                "a production master.",
                ctx.project_name,
            )

        try:
            # Preflight disk space check: require 1.5x source size headroom
            source_size = 0
            try:
                source_size = ctx.current_video.stat().st_size
            except OSError:
                pass
            if source_size > 0:
                check_disk_space(output_dir, int(source_size * 1.5))

            processor = _SinglePassProcessor(ctx.config)
            final_output = output_dir / f"{ctx.project_name}_FINAL.mp4"

            if final_output.exists():
                stale_val = validate_video(final_output, require_audio=False)
                if not stale_val.valid:
                    logger.warning(f"  Removing corrupt FINAL: {stale_val.reason}")
                    final_output.unlink()

            # Privacy gate: hard-fail (and repopulate ctx.blur_mask_path on resume)
            # BEFORE the irreversible encode, so a present mask == a blurred FINAL.
            blur_err = blur_gate.check_mask_ready(ctx)
            if blur_err:
                return StageResult(success=False, error=blur_err)
            # Occupant-coverage gate: a corroborated car whose cabin the mask misses
            # would ship its driver exposed — fail BEFORE the irreversible encode.
            cabin_err = blur_gate.check_cabin_coverage(ctx)
            if cabin_err:
                return StageResult(success=False, error=cabin_err)

            sp_result = processor.process(
                video_path=ctx.current_video,
                output_path=final_output,
                location=ctx.location,
                video_duration=props.duration,
                lut_path=ctx.lut_path,
                skip_color=skip_color,
                skip_ost=skip_ost,
                skip_audio=skip_audio,
                video_info={
                    "duration": props.duration,
                    "height": props.height,
                    "width": props.width,
                    "fps": props.fps,
                    "bitrate_kbps": props.bitrate_kbps,
                },
                # The blur stage sets this only when blur ran and produced a frame-aligned
                # mask; None (the default) keeps the legacy no-blur -vf path.
                blur_mask_path=ctx.blur_mask_path,
                **cadence_kwargs(ctx.config),
            )

            if sp_result.success:
                # Stamp the durable blur marker so a later resume can prove this
                # master was encoded with the privacy composite (see blur_gate).
                if blur_gate.blur_required(ctx.config):
                    blur_gate.mark_applied(final_output)
                applied = "applied" if sp_result.ost_applied else "skipped"
                normalized = "normalized" if sp_result.audio_normalized else "skipped"
                nvenc = "yes" if sp_result.nvenc_used else "no (CPU)"
                logger.info(f"  Color: {sp_result.color_method}")
                if sp_result.tonemap:
                    logger.info(f"  HDR tonemap: {sp_result.tonemap}")
                logger.info(f"  OST: {applied}")
                logger.info(f"  Audio: {normalized}")
                logger.info(f"  NVENC: {nvenc}")
                logger.info(f"  Encode time: {sp_result.encoding_time:.0f}s")

                # IMMEDIATE QUALITY GATE
                if final_output.exists() and props.bitrate_kbps > 0 and props.duration > 0:
                    output_size = final_output.stat().st_size
                    output_bitrate_kbps = int(output_size * 8 / props.duration / 1000)
                    min_expected = int(props.bitrate_kbps * 0.90)
                    if output_bitrate_kbps < min_expected:
                        return StageResult(
                            success=False,
                            error=(
                                f"QUALITY GATE: Output {output_bitrate_kbps}kbps "
                                f"< 90% of source {props.bitrate_kbps}kbps."
                            ),
                        )
                if str(final_output).startswith("/mnt/"):
                    try:
                        from .. import fs_safety

                        wsl_final = ctx.paths.staging_dir / final_output.name
                        fs_safety.safe_copy(str(final_output), str(wsl_final))
                        if wsl_final.stat().st_size != final_output.stat().st_size:
                            wsl_final.unlink()
                            raise IOError("Copy incomplete")
                        ctx.current_video = wsl_final
                    except Exception as e:
                        logger.warning(f"  WSL FINAL copy failed ({e}), using original /mnt/ path")
                        ctx.current_video = final_output
                else:
                    ctx.current_video = final_output
            else:
                raise RuntimeError("Single-pass encode failed")

        except Exception as e:
            return StageResult(success=False, error=f"Single-pass processing failed: {e}")

        for stage_name in [cc, ost, aud]:
            if stage_name not in ctx.stages_completed:
                ctx.stages_completed.append(stage_name)
        ctx.final_video = final_output
        return StageResult(success=True)
