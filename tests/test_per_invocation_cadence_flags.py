"""Per-invocation cadence flags: --lf-day, --lf-time, --shorts-times.

Asserts:
  - DriveConfig.shorts_publish_times validates HH:MM and rejects malformed.
  - schedule_constants.shorts_times(config) honors config.drive override and
    falls back to module SHORTS_TIMES when config is None (monkeypatch
    contract preserved).
  - apply_config(config) pushes shorts_publish_times into module SHORTS_TIMES.
  - PipelineScheduler.from_config wires shorts_times from config.
  - cmd_drive merges --lf-day / --lf-time / --shorts-times into config.drive
    BEFORE schedule_constants observes them, beating config-file values.
"""

from datetime import time as dt_time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config, DriveConfig
from gomoveshift.pipeline import schedule_constants
from gomoveshift.pipeline.scheduler import PipelineScheduler


class TestDriveConfigShortsPublishTimes:
    def test_default_is_afternoon_online_window(self):
        cfg = DriveConfig()
        assert cfg.shorts_publish_times == ["15:00", "18:00"]

    def test_accepts_arbitrary_hhmm_list(self):
        cfg = DriveConfig(shorts_publish_times=["14:00", "18:00"])
        assert cfg.shorts_publish_times == ["14:00", "18:00"]

    def test_rejects_empty_list(self):
        with pytest.raises(ValueError, match="at least one"):
            DriveConfig(shorts_publish_times=[])

    def test_rejects_malformed_entry(self):
        with pytest.raises(ValueError, match="HH:MM"):
            DriveConfig(shorts_publish_times=["14:00", "25:99"])

    def test_rejects_non_string_entry(self):
        with pytest.raises(ValueError):
            DriveConfig(shorts_publish_times=["14:00", 1800])  # type: ignore[list-item]


class TestShortsPublishTimesByRegion:
    def test_default_spread(self):
        cfg = DriveConfig()
        # Ireland is a flat window; USA is a per-LF-day map (Tue/Fri stagger so
        # the two concurrent USA drives don't share a slot) with a _default.
        assert cfg.shorts_publish_times_by_region == {
            "ireland": ["08:00", "12:00"],
            "usa": {
                "tue": ["14:00", "18:00"],
                "fri": ["10:00", "16:00"],
                "_default": ["14:00", "18:00"],
            },
        }

    def test_accepts_custom_map(self):
        cfg = DriveConfig(shorts_publish_times_by_region={"usa": ["10:00", "18:00"]})
        assert cfg.shorts_publish_times_by_region == {"usa": ["10:00", "18:00"]}

    def test_empty_map_is_valid(self):
        cfg = DriveConfig(shorts_publish_times_by_region={})
        assert cfg.shorts_publish_times_by_region == {}

    def test_rejects_empty_region_list(self):
        with pytest.raises(ValueError, match="at least one"):
            DriveConfig(shorts_publish_times_by_region={"usa": []})

    def test_rejects_malformed_entry(self):
        with pytest.raises(ValueError, match="HH:MM"):
            DriveConfig(shorts_publish_times_by_region={"usa": ["14:00", "26:00"]})

    def test_accepts_per_day_map(self):
        cfg = DriveConfig(
            shorts_publish_times_by_region={
                "usa": {"tue": ["14:00", "18:00"], "_default": ["14:00", "18:00"]}
            }
        )
        assert cfg.shorts_publish_times_by_region["usa"]["tue"] == ["14:00", "18:00"]

    def test_per_day_map_requires_default(self):
        with pytest.raises(ValueError, match="_default"):
            DriveConfig(shorts_publish_times_by_region={"usa": {"tue": ["14:00", "18:00"]}})

    def test_per_day_map_validates_each_window(self):
        with pytest.raises(ValueError, match="HH:MM"):
            DriveConfig(
                shorts_publish_times_by_region={"usa": {"tue": ["26:00"], "_default": ["14:00"]}}
            )


class TestResolveShortsTimesForRegion:
    def test_region_in_map_returns_region_window(self):
        cfg = Config()
        assert schedule_constants.resolve_shorts_times_for_region(cfg, "ireland") == [
            "08:00",
            "12:00",
        ]

    def test_region_absent_falls_back_to_flat(self):
        cfg = Config()
        cfg.drive.shorts_publish_times = ["15:00", "18:00"]
        assert schedule_constants.resolve_shorts_times_for_region(cfg, "japan") == [
            "15:00",
            "18:00",
        ]

    def test_none_region_falls_back_to_flat(self):
        cfg = Config()
        cfg.drive.shorts_publish_times = ["15:00", "18:00"]
        assert schedule_constants.resolve_shorts_times_for_region(cfg, None) == [
            "15:00",
            "18:00",
        ]


class TestShortsTimesAccessor:
    def test_returns_module_default_when_config_none(self, monkeypatch):
        sentinel = [dt_time(7, 0, 0)]
        monkeypatch.setattr(schedule_constants, "SHORTS_TIMES", sentinel)
        assert schedule_constants.shorts_times(None) == sentinel

    def test_honors_config_when_supplied(self):
        cfg = Config()
        cfg.drive.shorts_publish_times = ["14:00", "18:00"]
        assert schedule_constants.shorts_times(cfg) == [
            dt_time(14, 0, 0),
            dt_time(18, 0, 0),
        ]


@pytest.fixture
def restore_schedule_module():
    """apply_config rebinds module-level LONG_FORM_DAYS/TIME/SHORTS_TIMES
    via globals() assignment. The cross-module identity tests in
    test_schedule_constants.py rely on these objects staying the same
    across import sites, so any test that calls apply_config must
    restore the originals after the test.
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
    # Re-bind the names that other modules captured at import time
    # (`from schedule_constants import LONG_FORM_DAYS`) so identity
    # checks downstream see the originals again.
    from gomoveshift.pipeline import scheduler as _sched

    _sched.LONG_FORM_DAYS = saved[0]
    _sched.LONG_FORM_TIME = saved[1]
    _sched.SHORTS_TIMES = saved[2]


class TestApplyConfigPushesShortsTimes:
    def test_apply_config_updates_module_shorts_times(self, restore_schedule_module):
        cfg = Config()
        cfg.drive.shorts_publish_times = ["09:30", "21:45"]
        schedule_constants.apply_config(cfg)
        assert schedule_constants.SHORTS_TIMES == [
            dt_time(9, 30, 0),
            dt_time(21, 45, 0),
        ]


class TestSchedulerFromConfig:
    def test_from_config_wires_shorts_times(self):
        cfg = Config()
        cfg.drive.long_form_publish_days = [4]  # Fri
        cfg.drive.long_form_publish_time = "11:00"
        cfg.drive.shorts_publish_times = ["14:00", "18:00"]
        sched = PipelineScheduler.from_config(cfg)
        assert sched.shorts_times == [dt_time(14, 0, 0), dt_time(18, 0, 0)]
        assert sched.long_form_time == dt_time(11, 0, 0)
        assert sched.long_form_days == frozenset({4})


class TestCmdDriveCadenceMerge:
    """The drive handler must merge --lf-day / --lf-time / --shorts-times
    into config.drive BEFORE schedule_constants observes them, so a
    config-file Wed @ 11/12,16 cohort can be redirected to Fri @ 11/14,18
    for one run without a config edit.
    """

    def _args(self, **overrides):
        defaults = dict(
            directory="/tmp/nonexistent",
            output_dir="/tmp/out",
            region="usa",
            # The schedule-flag guard requires --publish or
            # --lf-day; these tests exercise the cadence merge, not the
            # guard, so give it a harmless default publish day unless a
            # test overrides lf_day/publish itself.
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
            list_lfs=False,
            label_dry_run=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _run(self, args, cfg):
        # Stub the orchestrator + tmux so we exercise only the merge.
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

    def test_lf_day_overrides_config(self, tmp_path, restore_schedule_module):
        cfg = Config()
        cfg.drive.long_form_publish_days = [2]  # Wed in config file
        args = self._args(directory=str(tmp_path), output_dir=str(tmp_path / "out"), lf_day="fri")
        self._run(args, cfg)
        assert cfg.drive.long_form_publish_days == [4]
        assert schedule_constants.LONG_FORM_DAYS == frozenset({4})

    def test_lf_time_overrides_config(self, tmp_path, restore_schedule_module):
        cfg = Config()
        cfg.drive.long_form_publish_time = "12:00"
        args = self._args(
            directory=str(tmp_path), output_dir=str(tmp_path / "out"), lf_time="11:00"
        )
        self._run(args, cfg)
        assert cfg.drive.long_form_publish_time == "11:00"
        assert schedule_constants.LONG_FORM_TIME == dt_time(11, 0, 0)

    def test_shorts_times_overrides_config(self, tmp_path, restore_schedule_module):
        cfg = Config()
        cfg.drive.shorts_publish_times = ["12:00", "16:00"]
        args = self._args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            shorts_times="14:00,18:00",
        )
        self._run(args, cfg)
        assert cfg.drive.shorts_publish_times == ["14:00", "18:00"]
        assert schedule_constants.SHORTS_TIMES == [
            dt_time(14, 0, 0),
            dt_time(18, 0, 0),
        ]

    def test_invalid_shorts_time_raises(self, tmp_path, restore_schedule_module):
        cfg = Config()
        args = self._args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            shorts_times="14:00,25:99",
        )
        with pytest.raises(Exception):
            self._run(args, cfg)

    def test_region_default_applied_when_no_shorts_times(self, tmp_path, restore_schedule_module):
        # No --shorts-times → the region's default window (ireland) is used.
        cfg = Config()
        args = self._args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            region="ireland",
            shorts_times=None,
        )
        self._run(args, cfg)
        assert cfg.drive.shorts_publish_times == ["08:00", "12:00"]

    def test_region_default_usa(self, tmp_path, restore_schedule_module):
        cfg = Config()
        args = self._args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            region="usa",
            shorts_times=None,
        )
        self._run(args, cfg)
        assert cfg.drive.shorts_publish_times == ["14:00", "18:00"]

    def test_explicit_shorts_times_beats_region_default(self, tmp_path, restore_schedule_module):
        # --shorts-times wins even when the region has a default window.
        cfg = Config()
        args = self._args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            region="ireland",
            shorts_times="13:00,21:00",
        )
        self._run(args, cfg)
        assert cfg.drive.shorts_publish_times == ["13:00", "21:00"]

    def test_region_default_reaches_module_when_lf_day_also_passed(
        self, tmp_path, restore_schedule_module
    ):
        # When another cadence flag (--lf-day) is present, apply_config runs
        # and the region shorts default propagates to module SHORTS_TIMES too.
        # Uses a NO-SLOT day (wed) so the precedence falls through to the region
        # default — a day WITH a publish_slot (sun/tue/fri) would correctly win
        # over the region default per resolve_shorts_times.
        cfg = Config()
        args = self._args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            region="ireland",
            shorts_times=None,
            lf_day="wed",
        )
        self._run(args, cfg)
        assert cfg.drive.shorts_publish_times == ["08:00", "12:00"]
        assert schedule_constants.SHORTS_TIMES == [dt_time(8, 0, 0), dt_time(12, 0, 0)]

    def test_unmapped_region_keeps_flat_default(self, tmp_path, restore_schedule_module):
        # A region absent from the map leaves the flat default untouched.
        cfg = Config()
        args = self._args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            region="canada",
            shorts_times=None,
        )
        self._run(args, cfg)
        assert cfg.drive.shorts_publish_times == ["15:00", "18:00"]
