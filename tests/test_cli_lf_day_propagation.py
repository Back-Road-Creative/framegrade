"""``--lf-day`` must override ``config.drive.long_form_publish_days``
and that override must reach the schedule generator that picks
``next_publish.date``.

Failure scenario: pipeline invoked with ``--lf-day mon --publish mon``.
The log emitted::

    generate_drive_schedule: publish_day=mon (dow=0) is not in
    long_form_days=[2] — scheduling off-cadence

The shipped LF metadata then had ``publish_day: "Monday"`` but a
``next_publish.date`` falling on a Wednesday. Two failure modes:

  1. ``cmd_drive`` must merge ``args.lf_day`` into
     ``config.drive.long_form_publish_days`` before any downstream
     consumer reads it (CLI overrides beat persisted state).
  2. Phase D's ``PipelineScheduler()`` construction must read cadence
     from ``config``, not from the import-time snapshot of
     ``LONG_FORM_DAYS`` that ``schedule_constants.apply_config`` can't
     rebind across module boundaries.
"""

from __future__ import annotations

from datetime import time as dt_time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import schedule_constants
from gomoveshift.pipeline.scheduler import PipelineScheduler


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def restore_schedule_module():
    """``apply_config`` rebinds module-level constants; restore after test
    so other tests that read identity-bindings aren't poisoned.
    """
    saved = (
        schedule_constants.LONG_FORM_DAYS,
        schedule_constants.LONG_FORM_TIME,
        schedule_constants.SHORTS_TIMES,
    )
    yield
    schedule_constants.LONG_FORM_DAYS = saved[0]
    schedule_constants.LONG_FORM_TIME = saved[1]
    schedule_constants.SHORTS_TIMES = saved[2]
    from gomoveshift.pipeline import scheduler as _sched

    _sched.LONG_FORM_DAYS = saved[0]
    _sched.LONG_FORM_TIME = saved[1]
    _sched.SHORTS_TIMES = saved[2]


def _drive_args(**overrides):
    defaults = dict(
        directory="/tmp/nonexistent",
        output_dir="/tmp/out",
        region="ireland",
        # cmd_drive refuses when
        # neither --publish nor --lf-day is given, so give these
        # cadence-merge tests a harmless default publish day.
        publish="mon",
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
        drive_handler.cmd_drive(args, cfg)


# ── Tests ──────────────────────────────────────────────────────────────


class TestLfDayOverridesConfig:
    """The handler-level merge: ``args.lf_day`` → ``config.drive.long_form_publish_days``."""

    def test_lf_day_mon_overrides_config_long_form_publish_days(
        self, tmp_path, restore_schedule_module
    ):
        cfg = Config()
        cfg.drive.long_form_publish_days = [2]  # Wed in config file
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            lf_day="mon",
        )
        _run_cmd_drive(args, cfg)
        assert cfg.drive.long_form_publish_days == [0]
        # And the module-level mirror used by call sites that don't have
        # a config in hand must also reflect Monday.
        assert schedule_constants.LONG_FORM_DAYS == frozenset({0})

    @pytest.mark.parametrize(
        "flag,expected_dow",
        [
            ("mon", 0),
            ("tue", 1),
            ("wed", 2),
            ("thu", 3),
            ("fri", 4),
            ("sat", 5),
            ("sun", 6),
        ],
    )
    def test_all_seven_weekdays(self, tmp_path, restore_schedule_module, flag, expected_dow):
        cfg = Config()
        cfg.drive.long_form_publish_days = [2]  # Wed baseline
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            lf_day=flag,
        )
        _run_cmd_drive(args, cfg)
        assert cfg.drive.long_form_publish_days == [expected_dow]
        assert schedule_constants.LONG_FORM_DAYS == frozenset({expected_dow})

    def test_no_lf_day_uses_config_default(self, tmp_path, restore_schedule_module):
        cfg = Config()
        cfg.drive.long_form_publish_days = [2]  # Wed in config
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            lf_day=None,
        )
        _run_cmd_drive(args, cfg)
        assert cfg.drive.long_form_publish_days == [2]


class TestLfDayReachesScheduler:
    """Once the merge lands on ``config``, the scheduler must observe it
    so ``next_publish.date`` lands on the operator-requested weekday.
    """

    def test_from_config_honours_lf_day_monday(self):
        cfg = Config()
        cfg.drive.long_form_publish_days = [0]  # Mon (post-merge state)
        sched = PipelineScheduler.from_config(cfg)
        assert sched.long_form_days == frozenset({0})

    def test_lf_day_propagates_to_next_publish_date(self):
        """Fixed ``start_date=2026-05-11`` (Mon). With cadence={Mon},
        the first scheduled LF lands on 2026-05-11, not 2026-05-13.
        """
        from gomoveshift.models import LFSegment, RegularLFPlan
        from pathlib import Path

        cfg = Config()
        cfg.drive.long_form_publish_days = [0]  # Mon
        sched = PipelineScheduler.from_config(cfg)
        seg = LFSegment(
            source_file_index=0,
            source_path=Path("/tmp/x.mp4"),
            start_time=0.0,
            end_time=600.0,
            original_start_time=0.0,
            original_end_time=600.0,
            composite_score=0.8,
            gps_center_lat=51.77,
            gps_center_lon=-9.98,
            terrain_type="mountain",
            contiguous_with_previous=False,
        )
        lf = RegularLFPlan(
            index=1,
            segments=[seg],
            montage_peaks=[seg],
            shorts_target=7,
            region="ireland",
            total_duration=600.0,
        )
        schedule = sched.generate_drive_schedule(
            regular_lfs=[lf],
            region="ireland",
            start_date="2026-05-11",  # Monday
            publish_day="mon",
        )
        assert schedule.lf_slots[0].date == "2026-05-11"
        assert schedule.lf_slots[0].day_of_week == "Mon"


class TestPhaseDBuildsSchedulerFromConfig:
    """The unfixed source: ``phase_d.py`` used to call ``PipelineScheduler()``
    with no args, which reads the import-time ``scheduler.LONG_FORM_DAYS``
    snapshot — invisible to ``schedule_constants.apply_config``'s
    ``globals()`` rebind. Must now call ``PipelineScheduler.from_config``
    so ``config.drive.long_form_publish_days`` flows through.
    """

    def test_phase_d_constructs_scheduler_via_from_config(self):
        """Verify phase_d builds the scheduler via ``from_config`` rather
        than the bare constructor — so cadence overrides on
        ``config.drive`` flow through (CLI overrides beat persisted state).

        The schedule step lives in a helper in ``phase_d_scheduling.py``;
        the call is ``scheduler_cls.from_config(config)`` (parameterised on
        the injected class so test mocks still work). The source grep
        targets that module.
        """
        from pathlib import Path

        phase_d_src = (
            Path(__file__).parent.parent
            / "gomoveshift"
            / "pipeline"
            / "drive_phases"
            / "phase_d_scheduling.py"
        ).read_text()
        assert "scheduler_cls.from_config(config)" in phase_d_src, (
            "phase_d_scheduling must build PipelineScheduler via from_config "
            "so cadence overrides on config.drive flow through (CLI overrides "
            "beat persisted state)."
        )

    def test_phase_d_scheduler_observes_config_cadence(self, restore_schedule_module):
        """End-to-end-ish: with ``config.drive.long_form_publish_days=[0]``
        the scheduler built by phase_d's construction expression must
        report ``long_form_days == {0}``.
        """
        cfg = Config()
        cfg.drive.long_form_publish_days = [0]  # Mon
        # Mirror the construction in phase_d.py § Drive schedule:
        scheduler = PipelineScheduler.from_config(cfg)
        custom_shorts_times = None
        if custom_shorts_times is not None:
            scheduler.shorts_times = custom_shorts_times
        assert scheduler.long_form_days == frozenset({0})

    def test_phase_d_scheduler_respects_custom_shorts_times(self):
        """``--shorts-per-day N`` selects the first N entries of
        ``config.drive.shorts_publish_times`` (single source of truth)
        and they flow through ``from_config``. Mirrors the construction
        in phase_d_scheduling._build_and_persist_drive_schedule.
        """
        from gomoveshift.pipeline.schedule_constants import _parse_hhmm

        cfg = Config()
        cfg.drive.long_form_publish_days = [0]
        cfg.drive.shorts_publish_times = ["15:00", "18:00"]
        scheduler = PipelineScheduler.from_config(cfg)
        shorts_per_day = 2
        cfg_times = [_parse_hhmm(t) for t in cfg.drive.shorts_publish_times]
        custom_shorts_times = cfg_times[:shorts_per_day]
        if custom_shorts_times is not None:
            scheduler.shorts_times = custom_shorts_times
        assert scheduler.shorts_times == [dt_time(15, 0, 0), dt_time(18, 0, 0)]


class TestLfDaySetsPublishDayConsistency:
    """Consistency fix: ``--lf-day`` must also default ``publish_day`` to the
    same weekday so it can't diverge from ``long_form_publish_days`` — the
    divergence (e.g. ``--lf-day tue`` on a USA drive → ``publish_day='sat'`` vs
    ``[tue]``) tripped the ``_check_publish_day`` state invariant.
    """

    def _captured_run_kwargs(self, args, cfg):
        from gomoveshift.cli.handlers import drive as drive_handler

        with (
            patch.object(drive_handler, "is_in_tmux", return_value=True),
            patch.object(drive_handler, "configure_logging"),
            patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator") as orch_cls,
        ):
            orch_cls.return_value.run.return_value = SimpleNamespace(
                success=True, stages_completed=[], drive_plan=None
            )
            drive_handler.cmd_drive(args, cfg)
            return orch_cls.return_value.run.call_args.kwargs

    def test_lf_day_defaults_publish_day_to_same_weekday(self, tmp_path, restore_schedule_module):
        cfg = Config()
        kw = self._captured_run_kwargs(
            _drive_args(
                directory=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                lf_day="tue",
                publish=None,
                region="usa",
                dry_run=False,
            ),
            cfg,
        )
        assert kw["publish_day"] == "tue"  # matches long_form_publish_days=[1]

    def test_explicit_publish_wins_over_lf_day(self, tmp_path, restore_schedule_module):
        cfg = Config()
        kw = self._captured_run_kwargs(
            _drive_args(
                directory=str(tmp_path),
                output_dir=str(tmp_path / "out"),
                lf_day="tue",
                publish="fri",
                region="usa",  # tue and fri are both Waypost days for usa
                dry_run=False,
            ),
            cfg,
        )
        assert kw["publish_day"] == "fri"

    def test_neither_flag_refuses_launch(self, tmp_path, restore_schedule_module):
        """Neither flag present is no
        longer a silent auto-from-region default — cmd_drive refuses before
        any Phase A work."""
        from gomoveshift.cli.handlers import drive as drive_handler

        cfg = Config()
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            publish=None,
            lf_day=None,
            dry_run=False,
        )
        with pytest.raises(SystemExit):
            drive_handler.cmd_drive(args, cfg)


class TestScheduleSpecDriftGate:
    """Single-source drift gate: when no explicit --shorts-times, the resolved
    shorts window MUST equal the region's determined spec
    (Waypost's ``shorts_times`` for the region). A divergence = a second source crept
    back in (a second source of truth) → SystemExit before any work.
    """

    def test_passes_on_normal_region_resolution(self, tmp_path, restore_schedule_module):
        cfg = Config()
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            region="usa",
            lf_day="tue",
        )
        _run_cmd_drive(args, cfg)  # no raise
        assert cfg.drive.shorts_publish_times == ["14:00", "18:00"]

    def test_fires_when_a_second_source_drifts(self, tmp_path, restore_schedule_module):
        # Simulate a re-introduced second source: resolve_shorts_times (which
        # SETS config) returns a window that doesn't match the region spec
        # (resolve_shorts_times_for_region, unpatched). The gate must abort.
        cfg = Config()
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            region="usa",
            lf_day="tue",
        )
        with patch.object(
            schedule_constants, "resolve_shorts_times", return_value=["09:00", "21:00"]
        ):
            with pytest.raises(SystemExit, match="SCHEDULE-SPEC-DRIFT"):
                _run_cmd_drive(args, cfg)

    def test_explicit_shorts_times_bypasses_gate(self, tmp_path, restore_schedule_module):
        cfg = Config()
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            region="usa",
            lf_day="tue",
            shorts_times="09:00,21:00",
        )
        _run_cmd_drive(args, cfg)  # explicit override allowed to differ
        assert cfg.drive.shorts_publish_times == ["09:00", "21:00"]
