"""Stage: Google Drive upload."""

import logging

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


def _planned_short_names(ctx):
    """Return the set of short filenames THIS run planned, or ``None``.

    Reads the ``output_path`` of each segment in ``ctx.shorts_data`` (produced
    by ``ShortsAnalysisStage``). ``None`` means the run produced no plan
    (legacy / no-shorts-stage callers), which the upload stage treats as
    "fall back to the directory glob".
    """
    from pathlib import Path

    shorts_data = getattr(ctx, "shorts_data", None)
    if not shorts_data:
        return None
    segments = shorts_data.get("segments") if isinstance(shorts_data, dict) else None
    if segments is None:
        return None
    return {
        Path(seg["output_path"]).name
        for seg in segments
        if isinstance(seg, dict) and seg.get("output_path")
    }


class DriveUploadStage(BaseStage):
    name = PipelineContext.STAGE_DRIVE_UPLOAD

    def run(self, ctx):
        from .. import orchestrator as _orch

        validate_video = _orch.validate_video
        PipelineUploader = _orch.PipelineUploader  # noqa: N806
        output_dir = ctx.paths.project_dir
        try:
            # When stop removal was explicitly skipped, the shorts derive
            # from RAW (unstopped) video. Part-1 (the shorts extractor now avoids
            # dead spots at stop_detection sensitivity) already lowers the risk of
            # a 5-30s stop leaking, but it can't GUARANTEE a stop-free clip. Make
            # the residual risk EXPLICIT with a loud warning to the operator,
            # rather than silently shipping potentially-parked shorts as if they
            # were stop-gated. The warning is the whole protection: this is a
            # manual/local surface, the drive pipeline never skips stops, and no
            # consumer reads a durable ungated marker off the run.
            if PipelineContext.STAGE_STOP_REMOVAL in getattr(ctx, "skip_stages", set()):
                logger.warning(
                    "  Drive upload UNGATED (--skip-stops): shorts derive from raw "
                    "video and may contain stops. Re-run without --skip-stops for a "
                    "stop-gated upload."
                )
            shorts_dir = output_dir / "shorts"
            # Ship exactly the shorts THIS run planned/rendered
            # (``ctx.shorts_data`` segments' ``output_path``), never a bare glob
            # of whatever ``*.mp4`` happen to sit in the dir — a pre-fix or stale
            # short left from an earlier run must not reach Drive + the Metricool
            # CSV. When the run produced no plan (legacy path), fall back to the
            # glob so behaviour is unchanged for those callers.
            planned = _planned_short_names(ctx)
            on_disk = sorted(shorts_dir.glob("*.mp4")) if shorts_dir.exists() else []
            if planned is not None:
                unplanned = [p.name for p in on_disk if p.name not in planned]
                if unplanned:
                    raise RuntimeError(
                        f"unplanned short(s) on disk not in this run's plan: "
                        f"{sorted(unplanned)} — refusing to glob-ship stale/pre-fix shorts"
                    )
                mp4s = [shorts_dir / name for name in sorted(planned)]
                missing = [m.name for m in mp4s if not m.exists()]
                if missing:
                    raise RuntimeError(f"planned short(s) missing on disk: {sorted(missing)}")
            else:
                mp4s = sorted(shorts_dir.glob("short_*.mp4"))
            if not mp4s:
                logger.warning("  No rendered shorts found")
            else:
                bad = [
                    f"{m.name}: {validate_video(m, require_audio=False).reason}"
                    for m in mp4s
                    if not validate_video(m, require_audio=False).valid
                ]
                if bad:
                    raise RuntimeError(f"{len(bad)} corrupted short(s)")
                uploader = PipelineUploader()
                ctx.upload_manifest = uploader.upload_shorts(
                    shorts_dir=str(shorts_dir),
                    project_name=ctx.project_name,
                    drive_subfolder=ctx.project_name,
                )
                logger.info(f"  Uploaded {len(ctx.upload_manifest.files)} files")
        except Exception as e:
            logger.warning(f"  Drive upload failed: {e}")
            return StageResult(success=False, error=f"Drive upload failed: {e}")
        return StageResult(success=True)
