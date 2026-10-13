"""Interim guard: ``--skip-stop-verify`` is a NO-OP on drive runs.

Its only consumer is the single-video stop-removal stage
(``config.drive.skip_stop_verify``), which the drive path doesn't run (drive
stop handling is the per-day stop mask). A no-op *safety* flag is worse than none
— it implies a verification step was deliberately skipped when nothing happened —
so a drive invocation carrying it must ERROR at handler entry.

``--skip-blur`` is NO LONGER rejected: privacy blur is wired into the drive
day master (``encode_day_master`` → ``blur_gate``), so the flag is FUNCTIONAL —
it sets ``config.drive.skip_blur`` and the day-master encode honors it, exactly
like the single-video ``BlurDetectionStage``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.config import Config


def _drive_args(**overrides):
    defaults = dict(
        directory="/tmp/nonexistent",
        output_dir="/tmp/out",
        region="ireland",
        # cmd_drive now refuses when
        # neither --publish nor --lf-day is given.
        publish="sun",
        location="",
        start_date=None,
        shorts_per_day=None,
        shorts_target=None,
        shorts_gap=None,
        fresh=False,
        no_tmux=True,
        preview=False,
        no_review=True,
        no_audio=False,
        skip_validation=False,
        skip_upload=True,
        skip_drive_upload=None,
        skip_youtube_upload=None,
        skip_csv=False,
        youtube_upload=None,
        phase=None,
        keep_intermediates=False,
        keep_masters=False,
        force_sidecar_reset=None,
        allow_seo_degraded=None,
        dry_run=True,
        force=True,
        skip_invariants=False,
        force_unlock=False,
        lf_day=None,
        lf_time=None,
        shorts_times=None,
        ingest_mode=None,
        lf_mode=None,
        lf_target_duration=None,
        no_youtube_upload=None,
        ost_audit_confirmed=None,
        gate_cd_confirmed=None,
        allow_cross_fs_source=False,
        allow_degraded_weather=False,
        lf_publish_date=None,
        force_no_rebase=False,
        per_lf_locations=None,
        list_lfs=False,
        skip_stop_verify=False,
        skip_blur=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _run_cmd_drive(args, cfg):
    from gomoveshift.cli.handlers import drive as drive_handler

    with (
        patch.object(drive_handler, "is_in_tmux", return_value=True),
        patch.object(drive_handler, "configure_logging"),
        patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator") as orch_cls,
    ):
        orch_cls.return_value.run.return_value = SimpleNamespace(
            success=True, stages_completed=[], drive_plan=None
        )
        return drive_handler.cmd_drive(args, cfg)


class TestDriveRejectsNoOpSafetyFlags:
    def test_skip_blur_accepted_and_sets_config(self, tmp_path, capsys):
        """--skip-blur is functional on drive runs — accepted (not rc=2) and
        mirrored to config.drive.skip_blur, which the day-master encode honors."""
        cfg = Config()
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            skip_blur=True,
        )
        rc = _run_cmd_drive(args, cfg)
        assert rc != 2
        err = capsys.readouterr().err
        assert "--skip-blur" not in err
        assert cfg.drive.skip_blur is True

    def test_skip_stop_verify_rejected(self, tmp_path, capsys):
        cfg = Config()
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            skip_stop_verify=True,
        )
        rc = _run_cmd_drive(args, cfg)
        assert rc == 2
        err = capsys.readouterr().err
        assert "--skip-stop-verify" in err
        assert "not supported on drive runs" in err

    def test_skip_stop_verify_still_rejected_even_with_skip_blur(self, tmp_path, capsys):
        cfg = Config()
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            skip_blur=True,
            skip_stop_verify=True,
        )
        rc = _run_cmd_drive(args, cfg)
        assert rc == 2
        err = capsys.readouterr().err
        assert "--skip-stop-verify" in err
        assert "--skip-blur" not in err  # skip-blur is functional now, not rejected

    def test_no_flags_not_rejected(self, tmp_path, capsys):
        cfg = Config()
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
        )
        _run_cmd_drive(args, cfg)
        err = capsys.readouterr().err
        assert "not supported on drive runs" not in err
        # The reject must never set the no-op config mirror on a clean run.
        assert cfg.drive.skip_blur is False
        assert cfg.drive.skip_stop_verify is False


class TestSingleVideoConsumerUntouched:
    """The single-video stage still honors the config flag — the drive-run
    rejection must not disturb the real consumer."""

    def _blur_ctx(self, *, blur_enabled: bool, skip_blur: bool):
        return SimpleNamespace(
            skip_stages=set(),
            stages_completed=set(),
            config=SimpleNamespace(
                drive=SimpleNamespace(blur_enabled=blur_enabled, skip_blur=skip_blur)
            ),
        )

    def test_blur_stage_skips_when_skip_blur_set(self):
        from gomoveshift.pipeline.stages.blur_detection import BlurDetectionStage

        ctx = self._blur_ctx(blur_enabled=True, skip_blur=True)
        assert BlurDetectionStage().should_run(ctx) is False

    def test_blur_stage_runs_when_skip_blur_clear(self):
        from gomoveshift.pipeline.stages.blur_detection import BlurDetectionStage

        ctx = self._blur_ctx(blur_enabled=True, skip_blur=False)
        assert BlurDetectionStage().should_run(ctx) is True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
