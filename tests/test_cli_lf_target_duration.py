"""``--lf-target-duration N`` (minutes) must:

  1. Enable ``config.drive.best_moments_mode`` for the run.
  2. Clamp ``best_moments_target_min == best_moments_target_max == N*60``.
  3. Widen ``config.drive.max_lf_duration`` when target > current max so
     the packer can hold the target in a single LF (default max is 720s
     = 12min, target 15min would otherwise split across two LFs).
  4. Reject ``--lf-mode full_drive`` (incompatible).
  5. Reject ``--lf-target-duration <= 0``.

Fills the gap between ``--lf-mode best_moments`` (variable duration via
sliding-window planner + adaptive threshold) and ``--lf-mode full_drive``
(whole timeline). The single-file action-camera use case (one 30-min source,
operator wants exactly one 15-min highlight) had no first-class CLI
expression before this flag.
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
        # cmd_drive refuses when
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


class TestLfTargetDurationClamps:
    def test_15min_target_enables_best_moments_and_clamps_filter(self, tmp_path):
        cfg = Config()
        cfg.drive.best_moments_mode = False
        cfg.drive.best_moments_target_min = 900.0
        cfg.drive.best_moments_target_max = 1500.0
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            lf_target_duration=15.0,
        )
        _run_cmd_drive(args, cfg)
        assert cfg.drive.best_moments_mode is True
        assert cfg.drive.best_moments_target_min == 900.0
        assert cfg.drive.best_moments_target_max == 900.0

    def test_target_above_max_lf_widens_packer_ceiling(self, tmp_path):
        cfg = Config()
        # Default max_lf_duration is 720s (12min) — target 15min must widen it
        assert cfg.drive.max_lf_duration == 720.0
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            lf_target_duration=15.0,
        )
        _run_cmd_drive(args, cfg)
        assert cfg.drive.max_lf_duration >= 900.0

    def test_target_below_min_lf_lowers_packer_floor(self, tmp_path):
        cfg = Config()
        # Default min_lf_duration is 480s (8min) — target 5min must lower it
        assert cfg.drive.min_lf_duration == 480.0
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            lf_target_duration=5.0,
        )
        _run_cmd_drive(args, cfg)
        assert cfg.drive.min_lf_duration <= 300.0

    def test_target_within_default_bracket_leaves_bracket_alone(self, tmp_path):
        cfg = Config()
        # Default bracket is [480, 720]; target 10min (600s) fits → no change
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            lf_target_duration=10.0,
        )
        _run_cmd_drive(args, cfg)
        assert cfg.drive.min_lf_duration == 480.0
        assert cfg.drive.max_lf_duration == 720.0
        assert cfg.drive.best_moments_target_min == 600.0
        assert cfg.drive.best_moments_target_max == 600.0


class TestLfTargetDurationRejects:
    def test_full_drive_mode_rejected(self, tmp_path, capsys):
        cfg = Config()
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            lf_mode="full_drive",
            lf_target_duration=15.0,
        )
        rc = _run_cmd_drive(args, cfg)
        assert rc == 2
        err = capsys.readouterr().err
        assert "incompatible" in err.lower()
        assert "full_drive" in err

    @pytest.mark.parametrize("bad", [0, -1.0, -15.0])
    def test_non_positive_rejected(self, tmp_path, capsys, bad):
        cfg = Config()
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            lf_target_duration=bad,
        )
        rc = _run_cmd_drive(args, cfg)
        assert rc == 2
        err = capsys.readouterr().err
        assert "> 0" in err


class TestLfTargetDurationOmitted:
    def test_no_flag_leaves_config_untouched(self, tmp_path):
        cfg = Config()
        baseline = (
            cfg.drive.best_moments_mode,
            cfg.drive.best_moments_target_min,
            cfg.drive.best_moments_target_max,
            cfg.drive.min_lf_duration,
            cfg.drive.max_lf_duration,
        )
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            lf_target_duration=None,
        )
        _run_cmd_drive(args, cfg)
        assert (
            cfg.drive.best_moments_mode,
            cfg.drive.best_moments_target_min,
            cfg.drive.best_moments_target_max,
            cfg.drive.min_lf_duration,
            cfg.drive.max_lf_duration,
        ) == baseline
