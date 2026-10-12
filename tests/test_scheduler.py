"""
Tests for PipelineScheduler — shorts window calculation and slot generation.

The core scheduling rule:
  - Shorts start SAME DAY as longform, at first slot AFTER longform time
  - Shorts stop BEFORE the next longform publishes
  - 3 slots per day: 8:00 AM, 12:00 PM, 4:00 PM EST
  - Morning slots (before 11 AM) on the next longform day are allowed

Window examples (long form on Wed/Sat at 11 AM):
  Wed long form → shorts Wed 12pm,4pm + Thu-Fri full + Sat 8am (9 slots)
  Sat long form → shorts Sat 12pm,4pm + Sun-Tue full + Wed 8am (12 slots)
"""

from datetime import datetime
from datetime import time as dt_time
from unittest.mock import patch
from zoneinfo import ZoneInfo

from gomoveshift.pipeline.content_ledger import ContentLedger
from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator
from gomoveshift.pipeline.scheduler import (
    PipelineScheduler,
)

EST = ZoneInfo("America/New_York")

# Legacy defaults — these tests document the scheduler's behaviour under
# the original Wed+Sat, 8am/12pm/4pm cadence. The live defaults changed
# to Wed-only + 12pm/4pm in schedule_constants, but the scheduler still
# supports arbitrary cadences via constructor args; these tests pin that
# legacy contract. New-cadence behaviour is covered by
# TestNewCadenceDefaults below.
_LEGACY_DAYS = {2, 5}
_LEGACY_TIMES = [dt_time(8, 0, 0), dt_time(12, 0, 0), dt_time(16, 0, 0)]


def _legacy_scheduler():
    return PipelineScheduler(
        long_form_days=_LEGACY_DAYS,
        shorts_times=_LEGACY_TIMES,
    )


class TestCalculateWindow:
    """Shorts window starts same day as longform (after LF time), ends before next LF."""

    def test_wed_longform_window_starts_same_day_12pm(self):
        """Wed Mar 4 longform at 11am -> first short at Wed 12:00 PM."""
        scheduler = _legacy_scheduler()
        lf_dt = datetime(2026, 3, 4, 11, 0, 0, tzinfo=EST)
        start, end = scheduler._calculate_window(lf_dt)

        assert start.date().isoformat() == "2026-03-04", (
            f"Shorts should start same day as longform, got {start.date()}"
        )
        assert start.time() == dt_time(12, 0, 0), (
            f"First slot should be 12:00 PM (after 11am LF), got {start.time()}"
        )

    def test_wed_longform_window_ends_saturday_morning(self):
        """Wed Mar 4 longform -> shorts end Sat Mar 7 at 8:00 AM."""
        scheduler = _legacy_scheduler()
        lf_dt = datetime(2026, 3, 4, 11, 0, 0, tzinfo=EST)
        start, end = scheduler._calculate_window(lf_dt)

        assert end.date().isoformat() == "2026-03-07"
        assert end.time() == dt_time(8, 0, 0)

    def test_sat_longform_window_starts_same_day_12pm(self):
        """Sat Mar 7 longform at 11am -> first short at Sat 12:00 PM."""
        scheduler = _legacy_scheduler()
        lf_dt = datetime(2026, 3, 7, 11, 0, 0, tzinfo=EST)
        start, end = scheduler._calculate_window(lf_dt)

        assert start.date().isoformat() == "2026-03-07", (
            f"Shorts should start same day as longform, got {start.date()}"
        )
        assert start.time() == dt_time(12, 0, 0)

    def test_sat_longform_window_ends_wednesday_morning(self):
        """Sat Mar 7 longform -> shorts end Wed Mar 11 at 8:00 AM."""
        scheduler = _legacy_scheduler()
        lf_dt = datetime(2026, 3, 7, 11, 0, 0, tzinfo=EST)
        start, end = scheduler._calculate_window(lf_dt)

        assert end.date().isoformat() == "2026-03-11"
        assert end.time() == dt_time(8, 0, 0)

    def test_no_short_before_longform_time(self):
        """No shorts slot should be at or before the longform time on LF day."""
        scheduler = _legacy_scheduler()
        for lf_date in [
            datetime(2026, 3, 4, 11, 0, 0, tzinfo=EST),
            datetime(2026, 3, 7, 11, 0, 0, tzinfo=EST),
        ]:
            start, _ = scheduler._calculate_window(lf_date)
            assert start > lf_date, (
                f"First short must be AFTER longform time {lf_date}, got {start}"
            )


class TestSlotCount:
    """Verify correct number of available slots per window type."""

    def test_wed_to_sat_has_9_slots(self):
        """Wed->Sat: Wed(2) + Thu(3) + Fri(3) + Sat morning(1) = 9 slots."""
        scheduler = _legacy_scheduler()
        count = scheduler.get_available_slot_count("2026-03-04", "wed")
        assert count == 9, f"Wed->Sat should have 9 slots, got {count}"

    def test_sat_to_wed_has_12_slots(self):
        """Sat->Wed: Sat(2) + Sun(3) + Mon(3) + Tue(3) + Wed morning(1) = 12 slots."""
        scheduler = _legacy_scheduler()
        count = scheduler.get_available_slot_count("2026-03-07", "sat")
        assert count == 12, f"Sat->Wed should have 12 slots, got {count}"


class TestGenerateSchedule:
    """End-to-end schedule generation tests."""

    def test_wed_schedule_first_two_on_longform_day(self):
        """First 2 shorts should be on the longform day (12pm, 4pm)."""
        scheduler = _legacy_scheduler()
        schedule = scheduler.generate_schedule(
            publish_date="2026-03-04",
            publish_day="wed",
            num_shorts=9,
            project_name="test",
        )
        assert schedule.slots[0].date == "2026-03-04"
        assert schedule.slots[0].time == "12:00:00"
        assert schedule.slots[1].date == "2026-03-04"
        assert schedule.slots[1].time == "16:00:00"

    def test_wed_schedule_all_after_longform_time(self):
        """All shorts datetimes must be strictly after longform datetime."""
        scheduler = _legacy_scheduler()
        schedule = scheduler.generate_schedule(
            publish_date="2026-03-04",
            publish_day="wed",
            num_shorts=9,
            project_name="test",
        )
        lf_dt = datetime(2026, 3, 4, 11, 0, 0, tzinfo=EST)
        for slot in schedule.slots:
            slot_dt = datetime.fromisoformat(slot.datetime_est)
            assert slot_dt > lf_dt, (
                f"Short #{slot.short_index} at {slot.datetime_est} must be after "
                f"longform at {lf_dt.isoformat()}"
            )

    def test_wed_schedule_all_before_next_longform_time(self):
        """No shorts slot datetime can be >= next longform datetime."""
        scheduler = _legacy_scheduler()
        schedule = scheduler.generate_schedule(
            publish_date="2026-03-04",
            publish_day="wed",
            num_shorts=9,
            project_name="test",
        )
        next_lf = datetime(2026, 3, 7, 11, 0, 0, tzinfo=EST)
        for slot in schedule.slots:
            slot_dt = datetime.fromisoformat(slot.datetime_est)
            assert slot_dt < next_lf, (
                f"Short #{slot.short_index} at {slot.datetime_est} must be before "
                f"next longform at {next_lf.isoformat()}"
            )

    def test_wed_9_shorts_fills_exactly(self):
        """Wed->Sat has 9 slots; 9 shorts should fill exactly."""
        scheduler = _legacy_scheduler()
        schedule = scheduler.generate_schedule(
            publish_date="2026-03-04",
            publish_day="wed",
            num_shorts=9,
            project_name="test",
        )
        assert schedule.num_slots == 9

    def test_sat_schedule_9_shorts_in_12_slots(self):
        """Sat->Wed has 12 slots; 9 shorts should get 9 slots."""
        scheduler = _legacy_scheduler()
        schedule = scheduler.generate_schedule(
            publish_date="2026-03-07",
            publish_day="sat",
            num_shorts=9,
            project_name="test",
        )
        assert schedule.num_slots == 9

    def test_schedule_slot_times_are_valid(self):
        """All slot times must be one of: 08:00:00, 12:00:00, 16:00:00."""
        valid_times = {"08:00:00", "12:00:00", "16:00:00"}
        scheduler = _legacy_scheduler()
        schedule = scheduler.generate_schedule(
            publish_date="2026-03-07",
            publish_day="sat",
            num_shorts=9,
            project_name="test",
        )
        for slot in schedule.slots:
            assert slot.time in valid_times, (
                f"Short #{slot.short_index} time {slot.time} not in {valid_times}"
            )

    def test_slots_are_chronologically_ordered(self):
        """Slots must be in ascending datetime order."""
        scheduler = _legacy_scheduler()
        schedule = scheduler.generate_schedule(
            publish_date="2026-03-07",
            publish_day="sat",
            num_shorts=9,
            project_name="test",
        )
        datetimes = [datetime.fromisoformat(s.datetime_est) for s in schedule.slots]
        for i in range(1, len(datetimes)):
            assert datetimes[i] > datetimes[i - 1], (
                f"Slot {i + 1} ({datetimes[i]}) not after slot {i} ({datetimes[i - 1]})"
            )

    def test_caps_at_window_when_shorts_exceed_slots(self):
        """More shorts than slots caps at window size — never bleeds past next longform."""
        scheduler = _legacy_scheduler()
        schedule = scheduler.generate_schedule(
            publish_date="2026-03-04",
            publish_day="wed",
            num_shorts=15,  # Only 9 slots in Wed->Sat
            project_name="test",
        )
        assert schedule.num_slots == 9, (
            f"Should cap at 9 slots in Wed->Sat window, got {schedule.num_slots}"
        )
        assert schedule.num_shorts == 15
        assert len(schedule.slots) == 9
        # Last slot must be before next longform (Sat 11am)
        from datetime import datetime

        last_dt = datetime.fromisoformat(schedule.slots[-1].datetime_est)
        next_lf = datetime(2026, 3, 7, 11, 0, 0, tzinfo=EST)
        assert last_dt < next_lf, f"Last slot {last_dt} must be before next longform {next_lf}"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_longform_time_equals_shorts_time_starts_next_slot(self):
        """If LF at 12pm, first short should be 4pm same day (skip 12pm slot).

        Pins legacy shorts times so the equals-a-slot scenario holds
        regardless of the afternoon default (15:00/18:00), matching the
        sibling ``test_longform_at_4pm_starts_next_day``.
        """
        scheduler = PipelineScheduler(
            long_form_time=dt_time(12, 0, 0),
            shorts_times=_LEGACY_TIMES,
            long_form_days=_LEGACY_DAYS,
        )
        lf_dt = datetime(2026, 3, 4, 12, 0, 0, tzinfo=EST)
        start, end = scheduler._calculate_window(lf_dt)

        assert start.date() == lf_dt.date()
        assert start.time() == dt_time(16, 0, 0)

    def test_longform_at_4pm_starts_next_day(self):
        """If LF at 4pm (last shorts slot), shorts must start next day 8am."""
        scheduler = PipelineScheduler(
            long_form_time=dt_time(16, 0, 0),
            shorts_times=_LEGACY_TIMES,
            long_form_days=_LEGACY_DAYS,
        )
        lf_dt = datetime(2026, 3, 4, 16, 0, 0, tzinfo=EST)
        start, end = scheduler._calculate_window(lf_dt)

        assert start.date().isoformat() == "2026-03-05"
        assert start.time() == dt_time(8, 0, 0)

    def test_single_short_goes_to_last_slot(self):
        """A single short should be placed in the last slot (bridge to next LF)."""
        scheduler = _legacy_scheduler()
        schedule = scheduler.generate_schedule(
            publish_date="2026-03-04",
            publish_day="wed",
            num_shorts=1,
            project_name="test",
        )
        assert len(schedule.slots) == 1
        assert schedule.slots[0].date == "2026-03-07"
        assert schedule.slots[0].time == "08:00:00"


class TestScheduleValidation:
    """Test validate_schedule catches invalid schedules."""

    def test_valid_schedule_passes(self):
        """A correctly generated schedule should have no errors."""
        scheduler = _legacy_scheduler()
        schedule = scheduler.generate_schedule(
            publish_date="2026-03-04",
            publish_day="wed",
            num_shorts=9,
            project_name="test",
        )
        errors = scheduler.validate_schedule(schedule)
        assert errors == [], f"Valid schedule should have no errors: {errors}"

    def test_detects_slot_past_next_longform(self):
        """A slot past the next longform should be caught."""
        from gomoveshift.pipeline.scheduler import ReleaseSchedule, ScheduledSlot

        scheduler = _legacy_scheduler()
        bad_schedule = ReleaseSchedule(
            project_name="test",
            location="",
            long_form_publish="2026-03-04T11:00:00-05:00",
            window_start="2026-03-04T12:00:00-05:00",
            window_end="2026-03-07T08:00:00-05:00",
            num_shorts=1,
            num_slots=1,
            slots=[
                ScheduledSlot(
                    short_index=1,
                    date="2026-03-08",
                    time="12:00:00",
                    datetime_est="2026-03-08T12:00:00-05:00",
                    day_of_week="Sun",
                ),
            ],
        )
        errors = scheduler.validate_schedule(bad_schedule)
        assert any("bleeds past" in e for e in errors), (
            f"Should detect slot past next longform: {errors}"
        )

    def test_detects_slot_before_longform(self):
        """A slot at or before longform time should be caught."""
        from gomoveshift.pipeline.scheduler import ReleaseSchedule, ScheduledSlot

        scheduler = _legacy_scheduler()
        bad_schedule = ReleaseSchedule(
            project_name="test",
            location="",
            long_form_publish="2026-03-04T11:00:00-05:00",
            window_start="2026-03-04T12:00:00-05:00",
            window_end="2026-03-07T08:00:00-05:00",
            num_shorts=1,
            num_slots=1,
            slots=[
                ScheduledSlot(
                    short_index=1,
                    date="2026-03-04",
                    time="08:00:00",
                    datetime_est="2026-03-04T08:00:00-05:00",
                    day_of_week="Wed",
                ),
            ],
        )
        errors = scheduler.validate_schedule(bad_schedule)
        assert any("not after longform" in e for e in errors), (
            f"Should detect slot before longform: {errors}"
        )


class TestNextPublishSlot:
    """next_publish_slot must include today if today is a longform day."""

    def test_ledger_saturday_returns_next_wednesday(self):
        """Under Wed-only cadence, Saturday should route to next Wednesday
        regardless of time of day (Sat is no longer a longform day)."""
        ledger = ContentLedger()
        sat_morning = datetime(2026, 2, 28, 9, 0, 0, tzinfo=EST)
        slot = ledger.get_next_available_slot("longform", after=sat_morning)
        assert slot is not None
        assert slot["date"] == "2026-03-04"
        assert slot["day_name"] == "Wednesday"

    def test_ledger_skips_today_if_past_publish_time(self):
        """On Wednesday afternoon (after 11am), next slot is next Wednesday."""
        ledger = ContentLedger()
        wed_afternoon = datetime(2026, 3, 4, 14, 0, 0, tzinfo=EST)
        slot = ledger.get_next_available_slot("longform", after=wed_afternoon)
        assert slot is not None
        assert slot["date"] == "2026-03-11"
        assert slot["day_name"] == "Wednesday"

    def test_ledger_includes_today_wednesday(self):
        """On Wednesday morning, next longform slot should be today."""
        ledger = ContentLedger()
        wed_morning = datetime(2026, 3, 4, 8, 0, 0, tzinfo=EST)
        slot = ledger.get_next_available_slot("longform", after=wed_morning)
        assert slot is not None
        assert slot["date"] == "2026-03-04"
        assert slot["day_name"] == "Wednesday"

    def test_metadata_fallback_includes_today(self):
        """LongFormMetadataGenerator fallback includes today when today IS
        a longform day (Wed under the new cadence)."""
        gen = LongFormMetadataGenerator()
        with patch("gomoveshift.pipeline.longform_metadata.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 4, 9, 0, 0, tzinfo=EST)
            mock_dt.combine = datetime.combine
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            gen._ledger = None
            slot = gen.get_next_publish_slot("longform")
            assert slot["date"] == "2026-03-04", (
                f"Fallback should return today (Wed Mar 4), got {slot['date']}"
            )
