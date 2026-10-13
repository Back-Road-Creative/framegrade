"""Stage: YouTube chapter generation from GPS data."""

import json
import logging
import re
import traceback
from pathlib import Path

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


class ChapterGenerationStage(BaseStage):
    name = PipelineContext.STAGE_CHAPTER_GENERATION

    def run(self, ctx):
        from .. import orchestrator as _orch
        from ..chapters import ChapterGenerator

        props = ctx.video_properties
        output_dir = ctx.paths.project_dir
        try:
            from ..gopro_gps import resolve_corresponding_gpx

            # No blind glob("*.gpx")[0]. Adopt only an explicit --gps, an
            # exact <stem>.gpx sidecar, or an arbitrary GPX that passes the
            # correspondence gate — else None (never a wrong-drive GPX).
            gpx_file = resolve_corresponding_gpx(
                ctx.source_video, explicit_gpx=ctx.gps_path, video_duration=props.duration
            )
            chapter_gen = ChapterGenerator(ctx.config)
            dur = sum(e - s for s, e in ctx.keep_segments) if ctx.keep_segments else props.duration
            _eff_date = ctx.gps_date
            if _eff_date is None and gpx_file and gpx_file.exists():
                from ..chapters import detect_gps_date_for_location

                detected = detect_gps_date_for_location(
                    gpx_path=gpx_file, location_str=ctx.location
                )
                if detected:
                    _eff_date = detected
            chapters = chapter_gen.generate(
                gpx_path=gpx_file if gpx_file else Path("no_gps_file.gpx"),
                video_duration=dur,
                start_location=ctx.location,
                end_location=ctx.location,
                gps_range=ctx.gps_range,
                keep_segments=ctx.keep_segments,
                gps_date=_eff_date,
                stages_dir=ctx.paths.stages_dir,
                original_duration=props.duration if ctx.keep_segments else None,
                # ``keep_segments`` are source-timeline ranges of the
                # surviving footage; pass them as ``included_segments`` too so
                # waypoints outside the kept ranges are trimmed BEFORE chapter
                # segmentation — "describe only surviving footage" was
                # previously drive-path-only.
                included_segments=ctx.keep_segments,
            )
            if chapter_gen.recording_date is not None:
                ctx.recording_date = chapter_gen.recording_date
            chapter_text = chapter_gen.format_chapters(chapters)
            cp = output_dir / f"{ctx.project_name}_chapters.txt"
            with open(cp, "w") as f:
                f.write(chapter_text)
            mp = output_dir / f"{ctx.project_name}_youtube.json"
            if mp.exists():
                with open(mp) as f:
                    metadata = json.load(f)
                metadata["chapters"] = chapters
                metadata["chapter_text"] = chapter_text
                if "description" in metadata:
                    metadata["description"] = ctx.require_metadata_generator()._embed_chapters(
                        metadata["description"], chapter_text
                    )
                if ctx.recording_date and "description" in metadata:
                    from ..social_generator import format_recording_date

                    # Route through the SHARED formatter:
                    # emit the canonical "Mon YYYY" shape, never the forbidden
                    # "%B %d, %Y" full-date form, and matching the longform
                    # "📅 Filmed:" line. ctx.recording_date is already
                    # longitude-local (chapters generator), so the day (hence
                    # month) reflects the drive's LOCAL calendar, not raw UTC.
                    ds = format_recording_date(ctx.recording_date)
                    if ds:
                        # Match the canonical "📅 Filmed: <date>" line AND the
                        # legacy "📹 Recorded: <date>" line so already-published
                        # descriptions still get patched.
                        metadata["description"] = re.sub(
                            "\U0001f4c5 Filmed: .+",
                            f"\U0001f4c5 Filmed: {ds}",
                            metadata["description"],
                        )
                        metadata["description"] = re.sub(
                            "\U0001f4f9 Recorded: .+",
                            f"\U0001f4c5 Filmed: {ds}",
                            metadata["description"],
                        )
                try:
                    from ..route_context import extract_route_context

                    rc = extract_route_context(ctx.location, chapters)
                    if rc.gps_towns:
                        ctx.route_title = rc.make_default_title()
                        metadata["title"] = ctx.route_title
                except Exception:
                    pass
                if chapter_gen.named_routes:
                    named = chapter_gen.named_routes[0]
                    sb, sn = _orch.extract_series_info(ctx.project_name)
                    if sn is not None:
                        tp = _orch.PipelineOrchestrator._count_series_parts(output_dir.parent, sb)
                        nt = _orch.make_series_title(named, sn, tp, props.resolution_label)
                        metadata["series"] = {"name": named, "part": sn, "total": tp}
                    else:
                        nt = f"Driving the {named} | {props.resolution_label} POV"
                    metadata["title"] = nt
                    ctx.route_title = nt
                with open(mp, "w") as f:
                    json.dump(metadata, f, indent=2)
                ctx.metadata = metadata
            logger.info(f"  Generated {len(chapters)} chapters")
            if gpx_file and gpx_file.exists() and len(chapters) < 2:
                return StageResult(success=False, error="GPS available but <2 chapters generated.")
        except Exception as e:
            logger.error(f"  Chapter generation failed: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            return StageResult(success=False, error=f"Chapter generation failed: {e}")
        return StageResult(success=True)
