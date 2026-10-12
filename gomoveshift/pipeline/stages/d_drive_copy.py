"""Stage: Copy deliverables to D: drive."""

import logging

from ..context import PipelineContext
from ..stage_protocol import BaseStage, StageResult

logger = logging.getLogger(__name__)


class DDriveCopyStage(BaseStage):
    name = PipelineContext.STAGE_D_DRIVE_COPY

    def run(self, ctx):
        output_dir = ctx.paths.project_dir
        # Pre-flight: re-verify the WSL /mnt/<letter>/ mount before copying.
        # ValidationStage runs the same check at pipeline start, but D: drive copy
        # happens hours later post-encode, and 9p mounts can go stale after Windows
        # sleep/hibernate (see wsl_mount.py docstring). A typed pre-flight error
        # is far more actionable than the OSError shutil.copy2 would raise on a
        # stale mount.
        from ..wsl_mount import DriveNotMountedError, assert_wsl_mount

        try:
            assert_wsl_mount(ctx.paths.output_base)
        except DriveNotMountedError as e:
            logger.error(f"  D: drive copy aborted — {e}")
            return StageResult(success=False, error=str(e))
        try:
            from .. import fs_safety

            d_out = ctx.paths.output_base / ctx.project_name
            d_out.mkdir(parents=True, exist_ok=True)
            deliverables = [
                output_dir / f"{ctx.project_name}_FINAL.mp4",
                output_dir / f"{ctx.project_name}_youtube.txt",
                output_dir / f"{ctx.project_name}_metricool.csv",
            ]
            copied = 0
            for src in deliverables:
                if not src.exists():
                    continue
                dst = d_out / src.name
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    continue
                fs_safety.safe_copy(str(src), str(dst))
                if dst.stat().st_size != src.stat().st_size:
                    dst.unlink()
                    raise IOError(f"Copy incomplete: {dst.stat().st_size} vs {src.stat().st_size}")
                copied += 1
            logger.info(f"  {copied} file(s) copied to {d_out}")
        except Exception as e:
            logger.warning(f"  D: drive copy failed: {e}")
            return StageResult(success=False, error=f"D: drive copy failed: {e}")
        return StageResult(success=True)
