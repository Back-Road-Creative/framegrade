"""Tests for schedule constant dedup and SEO type validation.

TDD: verify single source of truth for schedule constants.
"""

from datetime import time as dt_time
from zoneinfo import ZoneInfo


SEED_CADENCE = True  # opt out of conftest.legacy_test_cadence


class TestScheduleConstantsSingleSource:
    """All three files must import from schedule_constants, not define their own."""

    def test_shared_module_exists(self):
        from gomoveshift.pipeline.schedule_constants import (
            EST,
            LONG_FORM_DAYS,
            LONG_FORM_TIME,
            SHORTS_TIMES,
        )

        # The import-time mirrors come from Waypost's packaged seed: usa tue + fri,
        # ireland sun; the time is the earliest day's (tue) 16:00.
        assert LONG_FORM_DAYS == {1, 4, 6}
        assert LONG_FORM_TIME == dt_time(16, 0, 0)
        assert SHORTS_TIMES == [dt_time(15, 0, 0), dt_time(18, 0, 0)], (
            "Afternoon audience-online window: 3pm + 6pm EST"
        )
        assert EST == ZoneInfo("America/New_York")

    def test_metadata_imports_from_shared(self):
        """longform_metadata must use schedule_constants, not local defs."""
        from gomoveshift.pipeline import longform_metadata, schedule_constants

        assert longform_metadata.EST is schedule_constants.EST
        assert longform_metadata.LONG_FORM_DAYS is schedule_constants.LONG_FORM_DAYS
        assert longform_metadata.LONG_FORM_TIME is schedule_constants.LONG_FORM_TIME

    def test_ledger_imports_from_shared(self):
        """content_ledger must use schedule_constants, not local defs."""
        from gomoveshift.pipeline import content_ledger, schedule_constants

        assert content_ledger.EST is schedule_constants.EST
        assert content_ledger.LONG_FORM_DAYS is schedule_constants.LONG_FORM_DAYS
        assert content_ledger.LONG_FORM_TIME is schedule_constants.LONG_FORM_TIME
        assert content_ledger.SHORTS_TIMES is schedule_constants.SHORTS_TIMES

    def test_scheduler_imports_from_shared(self):
        """scheduler must use schedule_constants, not local defs."""
        from gomoveshift.pipeline import schedule_constants, scheduler

        assert scheduler.EST is schedule_constants.EST
        assert scheduler.LONG_FORM_DAYS is schedule_constants.LONG_FORM_DAYS
        assert scheduler.LONG_FORM_TIME is schedule_constants.LONG_FORM_TIME
        assert scheduler.SHORTS_TIMES is schedule_constants.SHORTS_TIMES


class TestConfigurableCadence:
    """Cadence (LF days + time) must be configurable via ``config.drive.*``.

    Defaults are empty / unset: Waypost's cadence answers. The drive fields
    survive only as the per-run ``--lf-day`` / ``--lf-time`` transport.
    """

    def test_long_form_days_accessor_reads_config_override(self):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.schedule_constants import long_form_days

        cfg = Config()
        cfg.drive.long_form_publish_days = [4]  # Fri
        assert long_form_days(cfg) == frozenset({4})

    def test_long_form_time_accessor_reads_config_override(self):
        from datetime import time as dt_time

        from gomoveshift.config import Config
        from gomoveshift.pipeline.schedule_constants import long_form_time

        cfg = Config()
        cfg.drive.long_form_publish_time = "09:30"
        assert long_form_time(cfg) == dt_time(9, 30, 0)

    def test_accessors_default_to_module_constants(self):
        from gomoveshift.pipeline.schedule_constants import (
            LONG_FORM_DAYS,
            LONG_FORM_TIME,
            long_form_days,
            long_form_time,
        )

        # No config passed → accessors return the same values as the module
        # attributes (which remain the hardwired defaults).
        assert long_form_days() == frozenset(LONG_FORM_DAYS)
        assert long_form_time() == LONG_FORM_TIME

    def test_config_defaults_are_waypost_not_hand_set(self):
        from datetime import time as dt_time

        from gomoveshift.config import Config

        cfg = Config()
        cfg.seo.storage_path = "/nonexistent/seo"  # no decisions: the packaged seed answers
        assert cfg.drive.long_form_publish_days == []
        assert cfg.drive.long_form_publish_time is None
        from gomoveshift.pipeline.schedule_constants import long_form_days, long_form_time

        assert long_form_days(cfg) == frozenset({1, 4, 6})  # seed: usa tue+fri, ireland sun
        assert long_form_time(cfg) == dt_time(16, 0, 0)  # the earliest day's (tue) time

    def test_config_rejects_invalid_day(self):
        import pytest
        from pydantic import ValidationError

        from gomoveshift.config import DriveConfig

        with pytest.raises(ValidationError):
            DriveConfig(long_form_publish_days=[7])  # out of [0,6]

    def test_config_rejects_invalid_time(self):
        import pytest
        from pydantic import ValidationError

        from gomoveshift.config import DriveConfig

        with pytest.raises(ValidationError):
            DriveConfig(long_form_publish_time="25:00")  # not HH:MM

    def test_scheduler_honors_config_override(self):
        """``PipelineScheduler.from_config`` threads config cadence fields through."""
        from datetime import time as dt_time

        from gomoveshift.config import Config
        from gomoveshift.pipeline.scheduler import PipelineScheduler

        cfg = Config()
        cfg.drive.long_form_publish_days = [4]  # Fri
        cfg.drive.long_form_publish_time = "11:00"
        scheduler = PipelineScheduler.from_config(cfg)
        assert scheduler.long_form_days == frozenset({4})
        assert scheduler.long_form_time == dt_time(11, 0, 0)


class TestShortsPoolRegionFence:
    """A region's slot pool must never contain ANOTHER region's default windows.

    Defect: one market's pool shipped with another market's afternoon slot, so
    a USA drive whose defaults were contested substituted into Ireland's
    window and USA shorts landed on Ireland slots. The pool is fenced
    (USA owns 10/14/16/18, Ireland owns 08/12 — 16:00 is a USA slot, USA shorts +
    long-forms publish there) and a validator makes reintroducing the bleed
    impossible.
    """

    def test_usa_pool_excludes_ireland_windows(self):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.schedule_constants import resolve_shorts_pool_for_region

        usa_pool = resolve_shorts_pool_for_region(Config(), "usa")
        # Ireland owns 08:00 + 12:00; a USA drive must never draw from them.
        assert "08:00" not in usa_pool and "12:00" not in usa_pool
        # USA keeps its own four-slot grid (10/14/16/18 EST).
        assert set(usa_pool) == {"14:00", "18:00", "10:00", "16:00"}

    def test_ireland_pool_excludes_usa_windows(self):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.schedule_constants import resolve_shorts_pool_for_region

        ie_pool = resolve_shorts_pool_for_region(Config(), "ireland")
        assert not (set(ie_pool) & {"14:00", "18:00", "10:00", "16:00"})

    def test_validator_rejects_pool_polluted_with_other_region_window(self):
        import pytest
        from pydantic import ValidationError

        from gomoveshift.config import DriveConfig

        with pytest.raises(ValidationError):
            DriveConfig(
                shorts_publish_times_by_region={
                    "ireland": ["08:00", "16:00"],
                    "usa": ["14:00", "18:00"],
                },
                shorts_slot_pool_by_region={
                    "ireland": ["08:00", "16:00"],
                    "usa": ["14:00", "18:00", "16:00"],  # 16:00 is Ireland's → reject
                },
            )


class TestRequiredShortsPerLf:
    """The per-LF shorts requirement
    is DERIVED (shorts windows/day × LF cadence interval), never hand-copied."""

    @staticmethod
    def _config(**overrides):
        from gomoveshift.config import Config, DriveConfig

        cfg = Config()
        cfg.seo.storage_path = "/nonexistent/seo"  # no decisions: the packaged seed answers
        cfg.drive = DriveConfig(**overrides)
        return cfg

    def test_ireland_derives_14(self):
        from gomoveshift.pipeline.schedule_constants import required_shorts_per_lf

        assert required_shorts_per_lf(self._config(), region="ireland") == 14

    def test_usa_derives_14_for_each_lf_day(self):
        from gomoveshift.pipeline.schedule_constants import required_shorts_per_lf

        cfg = self._config()
        assert required_shorts_per_lf(cfg, region="usa", lf_day="tue") == 14
        assert required_shorts_per_lf(cfg, region="usa", lf_day="fri") == 14

    def test_three_window_plan_derives_21(self):
        """Changing the publish plan changes the requirement — no second edit."""
        from gomoveshift.pipeline.schedule_constants import required_shorts_per_lf

        cfg = self._config(shorts_publish_times=["08:00", "12:00", "16:00"])
        assert required_shorts_per_lf(cfg) == 21

    def test_no_config_raises_instead_of_guessing(self):
        import pytest

        from gomoveshift.pipeline.schedule_constants import required_shorts_per_lf

        with pytest.raises(ValueError, match="no publish plan"):
            required_shorts_per_lf(None)

    def test_derived_count_equals_scheduler_slot_supply(self):
        """The derivation and generate_drive_schedule share LF_PUBLISH_INTERVAL_DAYS:
        a 1-LF drive's generated shorts slots == required_shorts_per_lf."""
        from gomoveshift.pipeline.schedule_constants import required_shorts_per_lf
        from gomoveshift.pipeline.scheduler import PipelineScheduler

        cfg = self._config()
        schedule = PipelineScheduler.from_config(cfg).generate_drive_schedule(
            regular_lfs=[object()], region="ireland", start_date="2026-07-01"
        )
        assert len(schedule.shorts_slots) == required_shorts_per_lf(cfg)

    def test_shorts_per_day_caps_the_requirement(self):
        """``--shorts-per-day N`` truncates the scheduler to N windows, so
        the requirement must cap the window count at N — not derive from the
        full ``shorts_publish_times``. A cap at/above the window count is a
        no-op (``min``)."""
        from gomoveshift.pipeline.schedule_constants import required_shorts_per_lf

        cfg = self._config()  # default 2 windows
        assert required_shorts_per_lf(cfg, region="ireland") == 14  # uncapped
        assert required_shorts_per_lf(cfg, region="ireland", shorts_per_day=1) == 7
        # Cap above the configured window count never inflates the requirement.
        assert required_shorts_per_lf(cfg, region="ireland", shorts_per_day=5) == 14

    def test_shorts_per_day_requirement_matches_scheduler_supply(self):
        """The capped requirement equals the
        scheduler's slot supply when it truncates to N windows — the invariant
        that eliminates slotless posts on every ``--shorts-per-day N`` run."""
        from gomoveshift.pipeline.schedule_constants import required_shorts_per_lf
        from gomoveshift.pipeline.scheduler import PipelineScheduler

        cfg = self._config()
        schedule = PipelineScheduler.from_config(cfg).generate_drive_schedule(
            regular_lfs=[object()],
            region="ireland",
            start_date="2026-07-01",
            shorts_per_day=1,
        )
        assert len(schedule.shorts_slots) == required_shorts_per_lf(cfg, shorts_per_day=1)
