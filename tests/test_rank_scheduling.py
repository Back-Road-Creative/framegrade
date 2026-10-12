"""
TDD tests for rank-aware shorts scheduling.

Shorts with lower rank numbers (better content) should get the
earliest/best time slots. Currently generate_drive_schedule ignores
extraction rank entirely.
"""

from pathlib import Path

from gomoveshift.models import LFSegment, RegularLFPlan
from gomoveshift.pipeline.scheduler import PipelineScheduler
from gomoveshift.pipeline.shorts_extractor import RenderedShort


def _make_lf_plan(index: int, duration: float = 1200.0) -> RegularLFPlan:
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
        shorts_target=5,
        region="ireland",
        total_duration=duration,
    )


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


class TestRankAwareScheduling:
    """Shorts with better ranks should get earlier/better time slots."""

    def test_shorts_sorted_by_rank_in_schedule(self):
        """Shorts with rank 1 should get the earliest slot, rank 5 the latest."""
        scheduler = PipelineScheduler()
        lfs = [_make_lf_plan(1)]

        # Create shorts with scrambled ranks
        shorts = [
            _make_rendered_short(0, rank=3),
            _make_rendered_short(1, rank=1),
            _make_rendered_short(2, rank=5),
            _make_rendered_short(3, rank=2),
            _make_rendered_short(4, rank=4),
        ]

        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="ireland",
            start_date="2026-03-25",
            shorts_per_lf={1: shorts},
        )

        # The shorts_slots should be ordered: best rank gets earliest slot
        # short_index in the slots should reflect rank ordering
        assert len(schedule.shorts_slots) >= 5, (
            f"Expected at least 5 shorts slots, got {len(schedule.shorts_slots)}"
        )

        # Extract the short_index assignments for the first 5 slots.
        # _apply_rank_ordering sorts by rank then assigns each slot the
        # RenderedShort's .index (original extraction order), NOT .rank.
        # rank=1→index=1, rank=2→index=3, rank=3→index=0, rank=4→index=4, rank=5→index=2
        first_five = schedule.shorts_slots[:5]
        assigned_indices = [s.short_index for s in first_five]

        assert assigned_indices == [1, 3, 0, 4, 2], (
            f"Expected index order [1, 3, 0, 4, 2] (sorted by rank), got {assigned_indices}"
        )

    def test_no_shorts_per_lf_preserves_existing_behavior(self):
        """When shorts_per_lf is not provided, behavior unchanged."""
        scheduler = PipelineScheduler()
        lfs = [_make_lf_plan(1)]

        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="ireland",
            start_date="2026-03-25",
        )

        # Should still produce numbered shorts slots (sequential)
        assert len(schedule.shorts_slots) >= 7
        # First slot index should be 1
        assert schedule.shorts_slots[0].short_index == 1

    def test_rank_1_gets_earliest_slot(self):
        """The short with rank=1 must get the earliest time slot."""
        scheduler = PipelineScheduler()
        lfs = [_make_lf_plan(1)]

        shorts = [
            _make_rendered_short(0, rank=5),
            _make_rendered_short(1, rank=3),
            _make_rendered_short(2, rank=1),
            _make_rendered_short(3, rank=4),
            _make_rendered_short(4, rank=2),
        ]

        schedule = scheduler.generate_drive_schedule(
            regular_lfs=lfs,
            region="ireland",
            start_date="2026-03-25",
            shorts_per_lf={1: shorts},
        )

        # First slot should be the rank-1 short, which has index=2.
        # _apply_rank_ordering assigns .index (extraction order), not .rank.
        assert schedule.shorts_slots[0].short_index == 2, (
            f"First slot should have short_index=2 (rank 1 short's index), "
            f"got {schedule.shorts_slots[0].short_index}"
        )
