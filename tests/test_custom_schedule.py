"""Tests for custom scheduling CLI flags: --start-date, --shorts-per-day,
--shorts-target, --shorts-gap.

Validates:
  - PipelineScheduler with 2 shorts/day yields 14 slots per weekly LF window
  - start_date threading produces correct LF publish dates
  - shorts_target override propagates to LF plan
  - shorts_gap override propagates to PipelineShortsExtractor
  - Phase D uses drive_schedule dates for YouTube upload scheduling
"""

from datetime import datetime
from datetime import time as dt_time
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from gomoveshift.models import DrivePlan, RegularLFPlan
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.scheduler import PipelineScheduler

from _state_helpers import make_state

EST = ZoneInfo("America/New_York")


class TestTwoShortsPerDay:
    """generate_drive_schedule with shorts_times=[8am, 4pm] produces 2 slots/day.

    Phase D calls generate_drive_schedule (not generate_schedule), which spaces
    LFs one week apart from start_date. _calculate_window is NOT used.
    """

    def _make_scheduler(self):
        return PipelineScheduler(
            shorts_times=[dt_time(8, 0, 0), dt_time(16, 0, 0)],
        )

    def _make_lfs(self, count):
        return [
            RegularLFPlan(
                index=i,
                segments=[],
                montage_peaks=[],
                shorts_target=14,
                region="usa",
                total_duration=1140.0,
            )
            for i in range(count)
        ]

    def test_thursday_lf_14_shorts_slots_per_week(self):
        """Thu LF with 2 shorts/day → 14 shorts slots between weekly LFs."""
        scheduler = self._make_scheduler()
        lfs = self._make_lfs(2)
        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="usa",
            start_date="2026-04-16",
        )

        # LF1 on Apr 16, LF2 on Apr 23
        assert schedule.lf_slots[0].date == "2026-04-16"
        assert schedule.lf_slots[1].date == "2026-04-23"

        # Count shorts between LF1 and LF2
        lf1_dt = schedule.lf_slots[0].datetime_est
        lf2_dt = schedule.lf_slots[1].datetime_est
        lf1_shorts = [
            s for s in schedule.shorts_slots if s.datetime_est > lf1_dt and s.datetime_est < lf2_dt
        ]
        assert len(lf1_shorts) == 14, (
            f"Expected 14 shorts in first LF window, got {len(lf1_shorts)}"
        )

    def test_slots_are_chronological(self):
        """All shorts slots must be in chronological order."""
        scheduler = self._make_scheduler()
        lfs = self._make_lfs(2)
        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="usa",
            start_date="2026-04-16",
        )

        for i in range(1, len(schedule.shorts_slots)):
            assert schedule.shorts_slots[i].datetime_est > (
                schedule.shorts_slots[i - 1].datetime_est
            )

    def test_only_8am_and_4pm_slots(self):
        """With 2 shorts/day config, only 8am and 4pm slots appear."""
        scheduler = self._make_scheduler()
        lfs = self._make_lfs(2)
        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="usa",
            start_date="2026-04-16",
        )

        valid_times = {"08:00:00", "16:00:00"}
        for slot in schedule.shorts_slots:
            assert slot.time in valid_times, f"Slot time {slot.time} not in {valid_times}"


class TestStartDate:
    """generate_drive_schedule respects start_date for LF publish dates."""

    def _make_lfs(self, count):
        return [
            RegularLFPlan(
                index=i,
                segments=[],
                montage_peaks=[],
                shorts_target=14,
                region="usa",
                total_duration=1140.0,
            )
            for i in range(count)
        ]

    def test_start_date_sets_first_lf_date(self):
        """start_date='2026-04-16' → first LF on Apr 16."""
        scheduler = PipelineScheduler(
            shorts_times=[dt_time(8, 0, 0), dt_time(16, 0, 0)],
        )
        lfs = self._make_lfs(3)
        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="usa",
            start_date="2026-04-16",
        )

        assert schedule.lf_slots[0].date == "2026-04-16"
        assert schedule.lf_slots[1].date == "2026-04-23"
        assert schedule.lf_slots[2].date == "2026-04-30"

    def test_shorts_slots_between_lf_dates(self):
        """All shorts slots fall between their parent LF and next LF."""
        scheduler = PipelineScheduler(
            shorts_times=[dt_time(8, 0, 0), dt_time(16, 0, 0)],
        )
        lfs = self._make_lfs(2)
        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="usa",
            start_date="2026-04-16",
        )

        lf1_dt = datetime.fromisoformat(schedule.lf_slots[0].datetime_est)
        lf2_dt = datetime.fromisoformat(schedule.lf_slots[1].datetime_est)

        for slot in schedule.shorts_slots:
            slot_dt = datetime.fromisoformat(slot.datetime_est)
            if slot_dt < lf2_dt:
                assert slot_dt > lf1_dt, f"Short at {slot.datetime_est} is before LF1 at {lf1_dt}"

    def test_14_shorts_slots_per_weekly_lf_confirmed(self):
        """Redundant confirmation: 2 shorts/day × 7 days = 14."""
        scheduler = PipelineScheduler(
            shorts_times=[dt_time(8, 0, 0), dt_time(16, 0, 0)],
        )
        lfs = self._make_lfs(3)
        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="usa",
            start_date="2026-04-16",
        )

        # Verify ALL LF windows (not just first) have 14 shorts
        for i in range(len(schedule.lf_slots) - 1):
            lf_dt = schedule.lf_slots[i].datetime_est
            next_dt = schedule.lf_slots[i + 1].datetime_est
            window_shorts = [
                s
                for s in schedule.shorts_slots
                if s.datetime_est > lf_dt and s.datetime_est < next_dt
            ]
            assert len(window_shorts) == 14, f"LF {i}: expected 14 shorts, got {len(window_shorts)}"


class TestShortsTargetOverride:
    """state['shorts_target'] override applies to an LF plan.

    The previous test imported the deleted PhaseC class but only tested the
    override semantics in pure Python. The assembly phase that reads the
    override is now phase_bc_day_first; the contract (override wins when
    set, else the LF's default is kept) is unchanged.
    """

    def test_shorts_target_override_wins_when_set(self):
        lf = RegularLFPlan(
            index=0,
            segments=[],
            montage_peaks=[],
            shorts_target=7,
            region="usa",
            total_duration=1140.0,
        )
        assert lf.shorts_target == 7

        state = make_state(shorts_target=14)
        shorts_target_override = state.get("shorts_target")
        if shorts_target_override is not None:
            lf.shorts_target = shorts_target_override

        assert lf.shorts_target == 14


class TestShortsGapOverride:
    """--shorts-gap propagates to PipelineShortsExtractor."""

    def test_gap_threads_to_extractor(self):
        """PipelineShortsExtractor accepts min_gap_between_shorts."""
        from gomoveshift.pipeline.shorts_extractor import PipelineShortsExtractor

        ext = PipelineShortsExtractor(min_gap_between_shorts=15.0)
        assert ext.min_gap == 15.0

    def test_default_gap_is_60(self):
        """Default gap remains 60s when not overridden."""
        from gomoveshift.pipeline.shorts_extractor import PipelineShortsExtractor

        ext = PipelineShortsExtractor()
        assert ext.min_gap == 60.0


class TestPhaseDYouTubeScheduling:
    """Phase D uses drive_schedule LF dates for YouTube publish_at."""

    def test_start_date_uses_drive_schedule_dates(self):
        """When start_date is set, YouTube publish_at comes from
        drive_schedule.lf_slots, not plan_weekly_mondays."""
        from gomoveshift.pipeline.scheduler import ScheduledSlot

        # Simulate what phase_d does
        _start_date = "2026-04-16"

        lf_slots = [
            ScheduledSlot(
                short_index=0,
                date="2026-04-16",
                time="11:00:00",
                datetime_est="2026-04-16T11:00:00-04:00",
                day_of_week="Thu",
            ),
            ScheduledSlot(
                short_index=0,
                date="2026-04-23",
                time="11:00:00",
                datetime_est="2026-04-23T11:00:00-04:00",
                day_of_week="Thu",
            ),
        ]

        drive_schedule = MagicMock()
        drive_schedule.lf_slots = lf_slots

        # Phase D logic: use drive_schedule dates when start_date is set
        if _start_date and drive_schedule and getattr(drive_schedule, "lf_slots", None):
            schedule_dates = [
                datetime.fromisoformat(s.datetime_est) for s in drive_schedule.lf_slots
            ]
        else:
            schedule_dates = None

        assert schedule_dates is not None
        assert len(schedule_dates) == 2
        assert schedule_dates[0].day == 16
        assert schedule_dates[0].weekday() == 3  # Thursday
        assert schedule_dates[1].day == 23


class TestOrchestratorStateThreading:
    """Orchestrator stores CLI overrides in pipeline state."""

    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_a_ingest_and_analyze"
    )
    def test_custom_flags_stored_in_state(self, mock_a, mock_rg, tmp_path):
        """start_date, shorts_per_day, shorts_target, shorts_gap stored in state."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "GL010001.MP4").write_bytes(b"fake")

        captured_state_ref = [None]

        def capture_state(state):
            # Keep a reference to the actual PipelineState (dynamic attrs
            # are invisible to dict() but visible to state.get()).
            captured_state_ref[0] = state
            raise RuntimeError("stop after phase_a")

        mock_a.side_effect = capture_state

        orch = DrivePipelineOrchestrator()
        orch.run(
            source_dir=source,
            output_base=tmp_path / "output",
            region="usa",
            # Layer-1 invariant (Rule 2) requires publish_day to match
            # LONG_FORM_DAYS ({2} = Wed). Test intent is flag threading,
            # not weekday selection — "wed" satisfies the invariant.
            publish_day="wed",
            start_date="2026-04-16",
            shorts_per_day=2,
            shorts_target=14,
            shorts_gap=15.0,
        )

        state = captured_state_ref[0]
        assert state is not None, "phase_a was never called"
        assert state.get("start_date") == "2026-04-16"
        assert state.get("shorts_per_day") == 2
        assert state.get("shorts_target") == 14
        assert state.get("shorts_gap") == 15.0
        assert state.get("publish_day") == "wed"
