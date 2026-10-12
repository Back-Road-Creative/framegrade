"""Stage: Thumbnail generation."""

import logging

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult
from ..thumbnail_text_overlay import MIN_THUMBNAIL_BYTES

logger = logging.getLogger(__name__)


class ThumbnailGenerationStage(BaseStage):
    name = PipelineContext.STAGE_THUMBNAIL_GENERATION

    def run(self, ctx):
        from .. import orchestrator as _orch

        generate_thumbnails_from_video = _orch.generate_thumbnails_from_video
        props = ctx.video_properties
        output_dir = ctx.paths.project_dir
        try:
            thumb_dir = output_dir / "thumbnails" / "styled"
            thumb_dir.mkdir(parents=True, exist_ok=True)
            existing = list(thumb_dir.glob("*.jpg")) + list(thumb_dir.glob("*.png"))
            if len(existing) >= 9:
                logger.info(f"  {len(existing)} thumbnails already exist")
            else:
                if generate_thumbnails_from_video is None:
                    raise ImportError("thumbnail_text_overlay not available")
                duration = props.duration if props else 3600.0
                try:
                    # The drive path's picker — never brightest-wins again.
                    from ..drive_phases import thumbnail_generator as _tg

                    margin = duration * 0.05
                    step = (duration - 2 * margin) / 19
                    cands = [margin + i * step for i in range(20)]
                    picked = _tg.select_best_timestamps(ctx.current_video, cands, count=3)
                    if not picked:
                        raise RuntimeError("All frame-quality measurements failed")
                    timestamps = [int(t) for t in picked]
                except Exception:
                    timestamps = [int(duration * 0.15), int(duration * 0.35), int(duration * 0.55)]
                title = ctx.route_title or ctx.location
                sub = f"{props.resolution_label} POV" if props else "4K POV"
                thumb_results = generate_thumbnails_from_video(
                    video_path=str(ctx.current_video),
                    title=title,
                    subtitle=sub,
                    output_dir=str(thumb_dir),
                    frame_timestamps=timestamps,
                    base_name=ctx.project_name,
                )
                logger.info(f"  Generated {sum(len(s) for s in thumb_results.values())} thumbnails")
        except Exception as e:
            logger.error(f"  Thumbnail generation failed: {e}")
            return StageResult(success=False, error=f"Thumbnail generation failed: {e}")

        # ── Thumbnail quality gate ──────────────────────────────────
        # HARD FAIL, not a warning: this stage runs upstream of the Drive
        # upload / D-drive copy stages, so breakage caught here costs a
        # re-run rather than shipping a video with no usable thumbnail.
        # An operator wanting none skips the stage (--skip-thumbnails),
        # which never reaches this code.
        thumb_dir = output_dir / "thumbnails" / "styled"
        all_thumbs = []
        if thumb_dir.exists():
            all_thumbs = list(thumb_dir.glob("*.jpg")) + list(thumb_dir.glob("*.png"))
        if not all_thumbs:
            msg = f"No thumbnails produced in {thumb_dir}"
            logger.error(f"  {msg}")
            return StageResult(success=False, error=msg)

        undersized = [f for f in all_thumbs if f.stat().st_size < MIN_THUMBNAIL_BYTES]
        if undersized:
            names = ", ".join(f"{f.name} ({f.stat().st_size}B)" for f in undersized)
            msg = f"{len(undersized)} thumbnail(s) below {MIN_THUMBNAIL_BYTES}B: {names}"
            logger.error(f"  {msg}")
            return StageResult(success=False, error=msg)

        total_size = sum(f.stat().st_size for f in all_thumbs)
        logger.info(
            f"  Thumbnail validation passed: {len(all_thumbs)} files, "
            f"{total_size / 1024:.0f}KB total"
        )
        return StageResult(success=True)
