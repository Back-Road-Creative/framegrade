"""Stage: Detect and remove stops from driving video."""

import logging

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


class StopRemovalStage(BaseStage):
    name = PipelineContext.STAGE_STOP_REMOVAL

    def run(self, ctx):
        try:
            from .. import orchestrator as _orch

            StopProcessor = _orch.StopProcessor  # noqa: N806
            cache_dir = ctx.paths.stages_dir / "motion_cache"
            # GPS trace: when the LF has a GPX, parse it so optical
            # stops can be corroborated against GPS speed. A parse failure or an
            # absent trace leaves gps_data=None — optical-only behavior preserved.
            gps_data = None
            if ctx.gps_path is not None:
                try:
                    from ...analysis.gps_parser import parse_gpx_file

                    gps_data = parse_gpx_file(ctx.gps_path)
                except Exception as gps_exc:  # noqa: BLE001
                    logger.warning(
                        "  Could not parse GPS trace for stop corroboration (%s) — "
                        "falling back to optical-only stop detection",
                        gps_exc,
                    )
            stop_proc = StopProcessor(
                ctx.config,
                crossfade_duration=ctx.crossfade_duration,
                cache_dir=cache_dir,
                video_height=ctx.video_properties.height,
                gps_data=gps_data,
            )
            stops_output = ctx.paths.stages_dir / f"{ctx.project_name}_no_stops.mp4"
            source_duration = stop_proc._get_duration(ctx.current_video)
            stops_result = stop_proc.process(ctx.current_video, stops_output)
            if stops_result["stops_found"] > 0:
                logger.info(f"  Found {stops_result['stops_found']} stop(s)")
                for i, stop in enumerate(stops_result["stops"], 1):
                    logger.info(
                        f"    {i}. {stop['start']:.1f}s - {stop['end']:.1f}s (duration: {stop['duration']:.1f}s)"
                    )
                logger.info(f"  Removed {stops_result['total_removed_duration']:.1f}s of stops")
                keep_segments = stops_result.get("keep_segments")
                # UPSTREAM GATE: prove stops were cut before the encode.
                if getattr(ctx.config.drive, "skip_stop_verify", False):
                    logger.warning("  STOP-VERIFY: skipped (--skip-stop-verify)")
                else:
                    from ..stop_verify import verify_stops_removed

                    verify_stops_removed(
                        stops_output,
                        keep_segments,
                        source_duration,
                        ctx.config,
                        crossfade_duration=ctx.crossfade_duration,
                        _stop_processor=stop_proc,
                    )
                ctx.current_video = stops_output
                ctx.keep_segments = keep_segments
            else:
                logger.info("  No stops detected — skipping removal")
            # Independent GPS-speed check of the RETAINED timeline — runs
            # regardless of stops_found (even the no-stops branch above), using
            # the GPS trace rather than the blind optical re-scan. Hard-fails
            # when a parked GPS span the optics missed survives in the footage
            # about to be encoded. No-op for GPS-less LFs / --skip-stop-verify.
            if not getattr(ctx.config.drive, "skip_stop_verify", False):
                from ..stop_verify import verify_retained_timeline_moving

                verify_retained_timeline_moving(
                    stops_result.get("keep_segments"),
                    source_duration,
                    ctx.config,
                    stop_proc,
                )
        except Exception as e:
            return StageResult(success=False, error=f"Stop processing failed: {e}")
        return StageResult(success=True)
