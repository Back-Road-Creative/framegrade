"""Regression tests for silent-success bugs in DriveUploadStage / DDriveCopyStage.

Both stages historically wrapped their entire body in a bare ``try/except
Exception`` that logged a warning and then fell through to ``return
StageResult(success=True)``. Net effect: the orchestrator marked the stage
complete and downstream gates proceeded with missing / corrupt artifacts.

On exception, the stages return ``success=False`` with a descriptive error so
the orchestrator can surface the failure.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.context import PipelineContext, PipelinePaths
from gomoveshift.pipeline.stages.d_drive_copy import DDriveCopyStage
from gomoveshift.pipeline.stages.upload import DriveUploadStage


def _make_ctx(tmp_path: Path, **overrides) -> PipelineContext:
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    paths = PipelinePaths(
        output_base=tmp_path / "out",
        project_dir=project_dir,
        stages_dir=tmp_path / "stages",
        state_file=tmp_path / "state.json",
        staging_dir=tmp_path / "staging",
    )
    paths.output_base.mkdir(parents=True, exist_ok=True)
    defaults = dict(
        project_name="testproj",
        source_video=tmp_path / "src.mp4",
        source_hash="abc123",
        config=MagicMock(),
        location="A to B via Road",
        publish_day="wed",
        paths=paths,
    )
    defaults.update(overrides)
    return PipelineContext(**defaults)


class TestDriveUploadStageSilentSuccess:
    """DriveUploadStage must not report success=True after swallowing an exception."""

    def test_validate_video_failure_returns_failure(self, tmp_path, monkeypatch, caplog):
        """When a rendered short fails validation, the resulting RuntimeError
        must propagate as ``success=False`` — not be swallowed into success=True.
        """
        shorts_dir = tmp_path / "project" / "shorts"
        shorts_dir.mkdir(parents=True, exist_ok=True)
        (shorts_dir / "short_01.mp4").write_bytes(b"fake")

        from gomoveshift.pipeline import orchestrator as _orch

        monkeypatch.setattr(
            _orch,
            "validate_video",
            lambda path, require_audio=False: MagicMock(valid=False, reason="no audio"),
        )

        stage = DriveUploadStage()
        ctx = _make_ctx(tmp_path)
        result = stage.run(ctx)

        assert result.success is False, (
            "Stage swallowed the corrupt-short RuntimeError and returned success=True. "
            "stages/upload.py must propagate exceptions as "
            "StageResult(success=False)."
        )
        assert result.error, "success=False must carry a non-empty error message"

    def test_uploader_exception_returns_failure(self, tmp_path, monkeypatch):
        """When PipelineUploader.upload_shorts raises, the stage must fail loudly."""
        shorts_dir = tmp_path / "project" / "shorts"
        shorts_dir.mkdir(parents=True, exist_ok=True)
        (shorts_dir / "short_01.mp4").write_bytes(b"fake")

        from gomoveshift.pipeline import orchestrator as _orch

        monkeypatch.setattr(
            _orch,
            "validate_video",
            lambda path, require_audio=False: MagicMock(valid=True, reason=""),
        )

        class _FailingUploader:
            def upload_shorts(self, **_kwargs):
                raise OSError("simulated drive auth failure")

        monkeypatch.setattr(_orch, "PipelineUploader", _FailingUploader)

        stage = DriveUploadStage()
        ctx = _make_ctx(tmp_path)
        result = stage.run(ctx)

        assert result.success is False
        assert "simulated drive auth failure" in result.error

    def test_no_shorts_directory_is_not_a_failure(self, tmp_path):
        """Empty / missing shorts dir is the documented no-op path — still
        success=True (the stage logs a warning but does no work). This pins
        that the fix narrows the failure surface to actual exceptions.
        """
        stage = DriveUploadStage()
        ctx = _make_ctx(tmp_path)
        result = stage.run(ctx)
        assert result.success is True


class TestDDriveCopyStageSilentSuccess:
    """DDriveCopyStage must not report success=True after swallowing an exception."""

    def test_copy_failure_returns_failure(self, tmp_path, monkeypatch):
        """When the copy raises, the stage must fail loudly — not log a
        warning and pretend the deliverable landed on the D: drive.
        """
        ctx = _make_ctx(tmp_path)
        # Create one of the deliverables so the copy loop actually runs.
        (ctx.paths.project_dir / f"{ctx.project_name}_FINAL.mp4").write_bytes(b"final")

        from gomoveshift.pipeline import fs_safety

        def _boom(src, dst):
            raise OSError("simulated permission denied on D:")

        monkeypatch.setattr(fs_safety, "safe_copy", _boom)

        stage = DDriveCopyStage()
        result = stage.run(ctx)

        assert result.success is False, (
            "Stage swallowed the OSError from fs_safety.safe_copy and returned "
            "success=True. stages/d_drive_copy.py must "
            "propagate exceptions as StageResult(success=False)."
        )
        assert "simulated permission denied" in result.error

    def test_copy_to_drvfs_destination_survives_copystat_denial(self, tmp_path, monkeypatch):
        """A drvfs copystat PermissionError (a known drvfs bug class)
        must NOT fail the stage: bytes must still land on D: and the stage
        must report success=True.
        """
        import shutil

        ctx = _make_ctx(tmp_path)
        final_bytes = b"final-deliverable-bytes"
        (ctx.paths.project_dir / f"{ctx.project_name}_FINAL.mp4").write_bytes(final_bytes)

        def _boom(_src, _dst, *, follow_symlinks=True):
            raise PermissionError("simulated drvfs copystat denial")

        monkeypatch.setattr(shutil, "copystat", _boom)

        stage = DDriveCopyStage()
        result = stage.run(ctx)

        assert result.success is True, (
            f"Stage failed on a copystat-only error (drvfs quirk): {result.error}"
        )
        dst = ctx.paths.output_base / ctx.project_name / f"{ctx.project_name}_FINAL.mp4"
        assert dst.read_bytes() == final_bytes

    def test_no_deliverables_present_is_not_a_failure(self, tmp_path):
        """No deliverables in project_dir is the documented no-op path — still
        success=True (loop body never executes, copied=0).
        """
        ctx = _make_ctx(tmp_path)
        stage = DDriveCopyStage()
        result = stage.run(ctx)
        assert result.success is True


class TestDDriveCopyStageWslMountPreflight:
    """DDriveCopyStage must pre-flight the WSL ``/mnt/<letter>/`` mount before
    attempting shutil.copy2 and surface a typed, operator-friendly error when
    the drive is stale.
    """

    def test_unmounted_drive_raises_clear_named_error(self, tmp_path, monkeypatch):
        """When ctx.paths.output_base lives on /mnt/d/ but the mount is stale,
        the stage must short-circuit with a DriveNotMountedError-derived
        StageResult.error that names the missing mount + remediation command.
        """
        from gomoveshift.pipeline import wsl_mount as _wsl

        # Force ensure_wsl_mount to report "not accessible" for any /mnt/ path
        # so the assert_wsl_mount helper raises DriveNotMountedError.
        monkeypatch.setattr(_wsl, "ensure_wsl_mount", lambda p, **kw: False)

        ctx = _make_ctx(tmp_path)
        # Override output_base to a /mnt/d/ path so the pre-flight is gated on.
        ctx.paths.output_base = Path("/mnt/d/videos")
        # Drop a deliverable so we know the failure is the pre-flight, not the
        # "no files to copy" no-op path.
        (ctx.paths.project_dir / f"{ctx.project_name}_FINAL.mp4").write_bytes(b"final")

        stage = DDriveCopyStage()
        result = stage.run(ctx)

        assert result.success is False
        assert "/mnt/d" in result.error
        assert "mount -t drvfs" in result.error, (
            f"Operator-friendly error must include the recovery command. got: {result.error!r}"
        )

    def test_non_mnt_path_skips_preflight(self, tmp_path, monkeypatch):
        """Pre-flight is a no-op for non-/mnt/ output bases (e.g. ext4 staging
        dirs). It must not call ensure_wsl_mount at all in that case.
        """
        from gomoveshift.pipeline import wsl_mount as _wsl

        called = {"n": 0}

        def _spy(p, **kw):
            called["n"] += 1
            return True

        monkeypatch.setattr(_wsl, "ensure_wsl_mount", _spy)

        ctx = _make_ctx(tmp_path)  # output_base under tmp_path, not /mnt/
        stage = DDriveCopyStage()
        result = stage.run(ctx)

        assert result.success is True
        assert called["n"] == 0, "ensure_wsl_mount must not be called for non-/mnt/ paths"


class TestAssertWslMount:
    """assert_wsl_mount must convert ensure_wsl_mount's boolean into a typed
    DriveNotMountedError carrying mount_point + remediation guidance.
    """

    def test_raises_drive_not_mounted_error_with_mount_point(self, monkeypatch):
        from gomoveshift.pipeline import wsl_mount as _wsl

        monkeypatch.setattr(_wsl, "ensure_wsl_mount", lambda p, **kw: False)

        with pytest.raises(_wsl.DriveNotMountedError) as exc_info:
            _wsl.assert_wsl_mount(Path("/mnt/d/videos/run_a"))

        assert exc_info.value.mount_point == Path("/mnt/d")
        assert "/mnt/d" in str(exc_info.value)
        assert "mount -t drvfs D:" in str(exc_info.value)

    def test_noop_for_non_mnt_path(self, monkeypatch):
        from gomoveshift.pipeline import wsl_mount as _wsl

        called = {"n": 0}
        monkeypatch.setattr(
            _wsl, "ensure_wsl_mount", lambda p, **kw: called.__setitem__("n", 1) or True
        )

        # Should not raise and should not call ensure_wsl_mount.
        _wsl.assert_wsl_mount(Path("/var/tmp/videos"))
        assert called["n"] == 0

    def test_noop_when_mount_accessible(self, monkeypatch):
        from gomoveshift.pipeline import wsl_mount as _wsl

        monkeypatch.setattr(_wsl, "ensure_wsl_mount", lambda p, **kw: True)
        # Should not raise.
        _wsl.assert_wsl_mount(Path("/mnt/d/videos"))
