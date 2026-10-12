"""Tests for the pure scheduling math: ladder volume, ordering, stagger.

See docs/pipeline/149-dynamic-scheduling.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from gomoveshift.publishing.dynamic_schedule import (
    order_shorts,
    staggered_by_platform,
    weekly_volume_from_ladder,
)


@dataclass(frozen=True)
class _Policy:
    daily_ladder: Optional[Tuple[int, ...]] = None
    weekly_ceiling: Optional[int] = None


class TestWeeklyVolumeFromLadder:
    def test_healthy_uses_ladder_ceiling_times_seven(self):
        policy = _Policy(daily_ladder=(1, 2, 3))
        assert weekly_volume_from_ladder(policy, "HEALTHY", healthy_days=0) == 21

    def test_suppressed_floors_at_one_per_day(self):
        policy = _Policy(daily_ladder=(1, 2, 3))
        assert weekly_volume_from_ladder(policy, "SUPPRESSED", healthy_days=99) == 7

    def test_recovering_climbs_the_ladder(self):
        policy = _Policy(daily_ladder=(1, 2, 3))
        # 7 healthy days -> rung 1 -> cap 2/day -> 14/week.
        assert weekly_volume_from_ladder(policy, "RECOVERING", healthy_days=7) == 14

    def test_no_policy_falls_back_to_default_ladder(self):
        assert weekly_volume_from_ladder(None, "HEALTHY", healthy_days=0) == 21

    def test_weekly_ceiling_caps_the_result(self):
        policy = _Policy(daily_ladder=(1, 2, 3), weekly_ceiling=10)
        assert weekly_volume_from_ladder(policy, "HEALTHY", healthy_days=0) == 10

    def test_weekly_ceiling_above_ladder_result_is_a_no_op(self):
        policy = _Policy(daily_ladder=(1, 2, 3), weekly_ceiling=999)
        assert weekly_volume_from_ladder(policy, "HEALTHY", healthy_days=0) == 21


def _record(short_index, region, location, scheduled_at, **extra):
    row = {
        "short_index": short_index,
        "region": region,
        "location": location,
        "scheduled_at": scheduled_at,
    }
    row.update(extra)
    return row


class TestOrderShorts:
    def test_higher_region_weight_sorts_first(self):
        records = [
            _record(1, "usa", "cork", "2026-09-07T10:00:00+00:00"),
            _record(2, "ireland", "cork", "2026-09-07T10:00:00+00:00"),
        ]
        ordered = order_shorts(
            records,
            region_weight={"usa": 1.0, "ireland": 5.0},
            last_posted_by_location=None,
            now=datetime(2026, 9, 7, tzinfo=timezone.utc),
        )
        assert [r["short_index"] for r in ordered] == [2, 1]

    def test_longer_since_last_posted_sorts_first(self):
        now = datetime(2026, 9, 7, tzinfo=timezone.utc)
        records = [
            _record(1, "usa", "cork", "2026-09-07T10:00:00+00:00"),
            _record(2, "usa", "dublin", "2026-09-07T10:00:00+00:00"),
        ]
        ordered = order_shorts(
            records,
            region_weight=None,
            last_posted_by_location={
                "cork": now - timedelta(days=1),
                "dublin": now - timedelta(days=10),
            },
            now=now,
        )
        assert [r["short_index"] for r in ordered] == [2, 1]

    def test_location_never_posted_outranks_a_recent_one(self):
        now = datetime(2026, 9, 7, tzinfo=timezone.utc)
        records = [
            _record(1, "usa", "cork", "2026-09-07T10:00:00+00:00"),
            _record(2, "usa", "galway", "2026-09-07T10:00:00+00:00"),
        ]
        ordered = order_shorts(
            records,
            region_weight=None,
            last_posted_by_location={"cork": now - timedelta(days=1)},
            now=now,
        )
        assert [r["short_index"] for r in ordered] == [2, 1]

    def test_score_of_multiplies_priority(self):
        now = datetime(2026, 9, 7, tzinfo=timezone.utc)
        records = [
            _record(1, "usa", "cork", "2026-09-07T10:00:00+00:00"),
            _record(2, "usa", "cork", "2026-09-07T10:00:00+00:00"),
        ]

        def score_of(record):
            return 10.0 if record["short_index"] == 1 else None

        ordered = order_shorts(
            records,
            region_weight=None,
            last_posted_by_location=None,
            now=now,
            score_of=score_of,
        )
        assert [r["short_index"] for r in ordered] == [1, 2]

    def test_absent_score_of_defaults_every_record_to_one(self):
        now = datetime(2026, 9, 7, tzinfo=timezone.utc)
        records = [
            _record(1, "usa", "cork", "2026-09-07T11:00:00+00:00"),
            _record(2, "usa", "cork", "2026-09-07T10:00:00+00:00"),
        ]
        ordered = order_shorts(records, region_weight=None, last_posted_by_location=None, now=now)
        # Tied priority -> tie-break by (scheduled_at, short_index).
        assert [r["short_index"] for r in ordered] == [2, 1]


class TestStaggeredByPlatform:
    def test_fb_and_ig_move_stagger_hours_after_base(self):
        base = datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc)
        result = staggered_by_platform(
            base, stagger_hours=2, platforms=("youtube", "instagram", "facebook", "tiktok")
        )
        assert result == {
            "instagram": base + timedelta(hours=2),
            "facebook": base + timedelta(hours=2),
        }

    def test_zero_or_none_stagger_is_a_no_op(self):
        base = datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc)
        assert staggered_by_platform(base, stagger_hours=None, platforms=("facebook",)) == {}
        assert staggered_by_platform(base, stagger_hours=0, platforms=("facebook",)) == {}

    def test_facebook_lead_is_clamped_to_max_schedule_lead(self):
        from gomoveshift.publishing.facebook_adapter import MAX_SCHEDULE_LEAD

        base = datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc)
        result = staggered_by_platform(
            base, stagger_hours=24 * 40, platforms=("facebook", "instagram")
        )
        assert result["facebook"] == base + MAX_SCHEDULE_LEAD
        # Instagram carries no such cap.
        assert result["instagram"] == base + timedelta(hours=24 * 40)
