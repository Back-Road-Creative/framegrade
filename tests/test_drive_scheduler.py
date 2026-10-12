"""Tests for region-aware drive scheduling."""

from pathlib import Path

from gomoveshift.models import LFSegment, RegularLFPlan
from gomoveshift.pipeline.scheduler import PipelineScheduler
from gomoveshift.pipeline.shorts_extractor import RenderedShort


def _make_lf_plan(index: int, duration: float = 1200.0, region: str = "ireland") -> RegularLFPlan:
    seg = LFSegment(
        source_file_index=0,
        source_path=Path("/tmp/file_00.mp4"),
        start_time=0.0,
        end_time=duration,
        original_start_time=0.0,
        original_end_time=duration,
        composite_score=0.8,
        gps_center_lat=51.77,
        gps_center_lon=-9.98,
        terrain_type="mountain",
        contiguous_with_previous=False,
    )
    return RegularLFPlan(
        index=index,
        segments=[seg],
        montage_peaks=[seg],
        shorts_target=7,
        region=region,
        total_duration=duration,
    )


class TestGenerateDriveSchedule:
    """Region-aware scheduling for drive pipeline."""

    def test_ireland_uses_wednesday(self):
        """Ireland LFs should be scheduled on Wednesdays."""
        scheduler = PipelineScheduler()
        lfs = [_make_lf_plan(1, region="ireland"), _make_lf_plan(2, region="ireland")]

        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="ireland",
            start_date="2026-03-25",  # This is a Wednesday
        )

        assert len(schedule.lf_slots) >= 2
        # First LF on 2026-03-25 (Wed)
        assert schedule.lf_slots[0].day_of_week == "Wed"

    def test_usa_uses_saturday(self):
        """USA LFs should be scheduled on Saturdays."""
        scheduler = PipelineScheduler()
        lfs = [_make_lf_plan(1, region="usa")]

        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="usa",
            start_date="2026-03-28",  # This is a Saturday
        )

        assert len(schedule.lf_slots) >= 1
        assert schedule.lf_slots[0].day_of_week == "Sat"

    def test_shorts_allocated_per_lf(self):
        """Each LF gets its proportional shorts allocation."""
        scheduler = PipelineScheduler()
        lfs = [_make_lf_plan(1, region="ireland")]

        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="ireland",
            start_date="2026-03-25",
        )

        # Should have shorts slots between this LF and the next week's LF
        assert len(schedule.shorts_slots) >= 7  # At least 7 shorts

    def test_one_lf_per_week(self):
        """LFs are spaced 1 per week."""
        scheduler = PipelineScheduler()
        lfs = [_make_lf_plan(i, region="ireland") for i in range(1, 4)]

        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="ireland",
            start_date="2026-03-25",
        )

        # LFs should be 7 days apart
        for i in range(1, len(schedule.lf_slots)):
            prev = schedule.lf_slots[i - 1].date
            curr = schedule.lf_slots[i].date
            # Parse dates and check diff
            from datetime import datetime

            d_prev = datetime.strptime(prev, "%Y-%m-%d")
            d_curr = datetime.strptime(curr, "%Y-%m-%d")
            assert (d_curr - d_prev).days == 7


def _make_rendered_short(index: int, rank: int) -> RenderedShort:
    return RenderedShort(
        index=index,
        rank=rank,
        start_time=0.0,
        end_time=35.0,
        duration=35.0,
        avg_motion_energy=0.5,
        peak_motion_energy=0.8,
        source_video="/tmp/lf_00.mp4",
        output_path=f"/tmp/short_{index:02d}.mp4",
        location_text="Test Location",
        subscribe_text="@TestChannel",
    )


class TestRankTieBreaking:
    """Deterministic ordering when shorts share the same rank."""

    def test_same_rank_deterministic_order(self):
        """Shorts with identical ranks should be sorted deterministically by index."""
        scheduler = PipelineScheduler()
        lfs = [_make_lf_plan(1, region="ireland")]

        # All shorts have rank=1 — order should be deterministic by index
        shorts = [
            _make_rendered_short(5, rank=1),
            _make_rendered_short(2, rank=1),
            _make_rendered_short(4, rank=1),
            _make_rendered_short(1, rank=1),
            _make_rendered_short(3, rank=1),
        ]

        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="ireland",
            start_date="2026-03-25",
            shorts_per_lf={1: shorts},
        )

        # With tie-breaking by index, the first 5 slots should be assigned
        # in index order: 1, 2, 3, 4, 5
        first_five = schedule.shorts_slots[:5]
        assigned_indices = [s.short_index for s in first_five]
        assert assigned_indices == [1, 2, 3, 4, 5], (
            f"Expected deterministic order [1, 2, 3, 4, 5] for same-rank "
            f"shorts, got {assigned_indices}"
        )

    def test_tie_breaking_preserves_rank_priority(self):
        """Tie-breaking should not override primary rank ordering."""
        scheduler = PipelineScheduler()
        lfs = [_make_lf_plan(1, region="ireland")]

        # Mix of ranks: rank 1 shorts should come before rank 2 shorts,
        # and within each rank, index order should apply
        shorts = [
            _make_rendered_short(10, rank=2),
            _make_rendered_short(7, rank=1),
            _make_rendered_short(3, rank=2),
            _make_rendered_short(9, rank=1),
        ]

        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="ireland",
            start_date="2026-03-25",
            shorts_per_lf={1: shorts},
        )

        first_four = schedule.shorts_slots[:4]
        assigned_indices = [s.short_index for s in first_four]
        # Rank 1 shorts first (index 7, 9), then rank 2 shorts (index 3, 10)
        assert assigned_indices == [7, 9, 3, 10], (
            f"Expected [7, 9, 3, 10] (rank then index order), got {assigned_indices}"
        )


class TestExplicitPublishDay:
    """``publish_day`` kwarg must win over the regional default.

    Regression: a user passed ``--publish wed``
    (landed in state as ``publish_day="wed"``) but the scheduler
    ignored it and used the hardcoded ``region == "usa" → Saturday``
    rule, producing 7 LF slots on Saturdays. ``LONG_FORM_DAYS`` had since changed
    to {Wed} but this call site hardcoded 5 (Sat) for USA regardless. Fix: thread
    ``publish_day`` through; USA default now picks
    ``min(LONG_FORM_DAYS)`` instead of hardcoded Sat.
    """

    def test_publish_day_wed_overrides_usa_default(self):
        """USA region + publish_day='wed' → Wed slots, not Sat."""
        scheduler = PipelineScheduler()
        lfs = [_make_lf_plan(1, region="usa"), _make_lf_plan(2, region="usa")]

        # The base date is decided by seo_manager (Phase D) and handed in.
        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="usa",
            start_date="2031-01-01",  # a Wednesday
            publish_day="wed",
        )

        assert len(schedule.lf_slots) >= 2
        for slot in schedule.lf_slots:
            assert slot.day_of_week == "Wed", (
                f"publish_day='wed' must produce Wed slots under any "
                f"region; got {slot.day_of_week} for slot {slot.date}"
            )

    def test_publish_day_sat_overrides_ireland_default(self):
        """Ireland region + publish_day='sat' → Sat slots, not Wed.

        Symmetric check: the override works in both directions.
        """
        scheduler = PipelineScheduler(
            # Legacy constraint set to allow Sat — under the default
            # LONG_FORM_DAYS={2}, Sat would warn but still land.
            long_form_days={2, 5},
        )
        lfs = [_make_lf_plan(1, region="ireland"), _make_lf_plan(2, region="ireland")]

        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="ireland",
            start_date="2031-01-04",  # a Saturday
            publish_day="sat",
        )

        assert all(s.day_of_week == "Sat" for s in schedule.lf_slots), (
            f"publish_day='sat' must produce Sat slots regardless of "
            f"region; got {[s.day_of_week for s in schedule.lf_slots]}"
        )

    def test_invalid_publish_day_raises(self):
        """Unknown publish_day raises ValueError, not silently fall back."""
        import pytest

        scheduler = PipelineScheduler()
        lfs = [_make_lf_plan(1, region="usa")]

        with pytest.raises(ValueError, match="Invalid publish_day"):
            scheduler.generate_drive_schedule(
                regular_lfs=lfs,
                region="usa",
                publish_day="xyz",
            )

    def test_usa_default_follows_long_form_days_set(self):
        """Without an explicit publish_day, USA picks min(long_form_days)
        instead of hardcoding Saturday. Under the new cadence
        (LONG_FORM_DAYS={2}), that means Wed."""
        scheduler = PipelineScheduler()  # uses constants default: {2}
        lfs = [_make_lf_plan(1, region="usa"), _make_lf_plan(2, region="usa")]

        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="usa",
            start_date="2031-01-01",  # a Wednesday
        )
        for slot in schedule.lf_slots:
            assert slot.day_of_week == "Wed", (
                f"Under LONG_FORM_DAYS={{2}}, USA default should produce "
                f"Wed slots; got {slot.day_of_week}"
            )

    def test_start_date_still_wins_over_publish_day(self):
        """``start_date`` is the most specific signal — if both are
        given, start_date's day wins for the first LF (subsequent LFs
        still follow weekly spacing from that anchor)."""
        scheduler = PipelineScheduler(long_form_days={2, 5})
        lfs = [_make_lf_plan(1, region="usa")]

        # start_date is a Saturday (2026-04-25). publish_day says wed.
        # start_date wins — first LF lands on the date given.
        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="usa",
            start_date="2026-04-25",  # Saturday
            publish_day="wed",
        )

        assert schedule.lf_slots[0].date == "2026-04-25"
        assert schedule.lf_slots[0].day_of_week == "Sat"
