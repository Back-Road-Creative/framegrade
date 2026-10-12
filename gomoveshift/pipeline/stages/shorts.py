"""Stage: Shorts analysis and rendering."""

import json
import logging

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


class ShortsAnalysisStage(BaseStage):
    name = PipelineContext.STAGE_SHORTS_ANALYSIS

    def run(self, ctx):
        from .. import orchestrator as _orch

        validate_video = _orch.validate_video
        PipelineShortsExtractor = _orch.PipelineShortsExtractor  # noqa: N806
        MOTION_CACHE_DIR = _orch.MOTION_CACHE_DIR  # noqa: N806
        props = ctx.video_properties
        output_dir = ctx.paths.project_dir
        try:
            shorts_dir = output_dir / "shorts"
            shorts_dir.mkdir(parents=True, exist_ok=True)
            extractor = PipelineShortsExtractor()
            from ..gopro_gps import resolve_corresponding_gpx

            # No blind glob("*.gpx")[0] — see chapters stage. Adopt only an
            # explicit --gps, an exact <stem>.gpx sidecar, or a GPX that passes
            # the correspondence gate, so a wrong-drive GPX can't label shorts.
            _gpx = resolve_corresponding_gpx(
                ctx.source_video, explicit_gpx=ctx.gps_path, video_duration=props.duration
            )
            _gps_dur, _pois = 0.0, None
            try:
                from ..chapters import ChapterGenerator

                if PipelineContext.STAGE_CHAPTER_GENERATION in ctx.stages_completed:
                    _cg = ChapterGenerator(ctx.config)
                    if _gpx:
                        _j = _cg.gpx_parser.parse(_gpx)
                        _gps_dur = (_j.end_date - _j.start_date).total_seconds()
            except Exception as exc:
                logger.debug("Chapter parsing for shorts GPS duration failed: %s", exc)
            sr = extractor.extract_and_render(
                video_path=str(ctx.current_video),
                output_dir=str(shorts_dir),
                location=ctx.location,
                num_shorts=15,
                render=True,
                motion_cache_dir=MOTION_CACHE_DIR
                if PipelineContext.STAGE_STOP_REMOVAL in ctx.stages_completed
                else None,
                gpx_path=_gpx,
                keep_segments=ctx.keep_segments,
                gps_duration=_gps_dur,
                original_duration=props.duration,
                route_pois=_pois,
                # None = "derive per-clip labels from the GPX inside
                # extract_and_render, falling back to the operator --location"
                # — NOT "burn the drive-wide label
                # on every short".
                location_map=None,
            )
            segments = [s.to_dict() for s in sr.shorts]
            ctx.shorts_data = {"segments": segments}
            with open(shorts_dir / "shorts_analysis.json", "w") as f:
                json.dump(ctx.shorts_data, f, indent=2)
            logger.info(f"  Found {len(segments)} shorts segments")
            mp4s = sorted(shorts_dir.glob("short_*.mp4"))
            corrupted = [
                f"{m.name}: {validate_video(m, require_audio=False).reason}"
                for m in mp4s
                if not validate_video(m, require_audio=False).valid
            ]
            if corrupted:
                raise RuntimeError(f"{len(corrupted)} corrupted short(s)")
        except Exception as e:
            logger.warning(f"  Shorts analysis failed: {e}")
            return StageResult(success=False, error=f"Shorts analysis failed: {e}", fatal=False)
        return StageResult(success=True)
