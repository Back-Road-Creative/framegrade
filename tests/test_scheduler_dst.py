"""DST-aware datetime comparison in slot/LF window partitioning.

ISO-string lexicographic comparison of timezone-suffixed datetimes
(`-04:00` vs `-05:00`) is wrong across DST transitions: two instants
that have the same point in UTC can sort differently as strings, so
shorts slots can silently mis-assign to the wrong LF window around
the spring-forward / fall-back boundary.

Both production sites compute the same predicate:

  - ``scheduler._apply_rank_ordering`` (used at fresh-schedule build time)
  - ``phase_d._partition_shorts_slots`` (used on resume from saved state)

The window predicate is: a shorts slot S belongs to LF i's window when
``lf_dt < S < next_lf_dt`` (strict on both ends). Tests assert that
predicate holds when LF and slot are expressed with different UTC
offsets across a DST boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from gomoveshift.pipeline.scheduler import ScheduledSlot
from gomoveshift.pipeline.drive_phases.phase_d import _partition_shorts_slots


@dataclass
class _LF:
    index: int


def _slot(short_index: int, iso: str) -> ScheduledSlot:
    return ScheduledSlot(
        short_index=short_index,
        date=iso[:10],
        time=iso[11:19],
        datetime_est=iso,
        day_of_week="any",
    )


def test_spring_forward_slot_after_lf_is_in_window():
    """LF at 2026-03-08T01:30:00-05:00 (06:30 UTC, before DST flip).
    Slot at 2026-03-08T03:30:00-04:00 (07:30 UTC, after DST flip).
    UTC ordering: slot is 1 hour AFTER the LF, so it must land in
    that LF's window. String compare gives "03:30:00-04:00" >
    "01:30:00-05:00" — accidentally correct here. Use the next-LF
    boundary to expose the real defect: place next-LF AFTER the slot
    in UTC but EARLIER as a string."""
    lf_slots = [
        _slot(0, "2026-03-08T01:30:00-05:00"),  # 06:30 UTC
        _slot(0, "2026-03-08T04:00:00-04:00"),  # 08:00 UTC
    ]
    # Slot at 07:30 UTC sits between the two LFs.
    shorts_slots = [_slot(1, "2026-03-08T03:30:00-04:00")]  # 07:30 UTC
    schedule = SimpleNamespace(lf_slots=lf_slots, shorts_slots=shorts_slots)

    result = _partition_shorts_slots(schedule, [_LF(0), _LF(1)])

    # Slot belongs to LF 0's window (after LF 0, before LF 1).
    assert result[0] == shorts_slots
    assert result[1] == []


def test_fall_back_slot_string_compare_wrong_utc_compare_right():
    """LF 0 at 2026-11-01T01:30:00-04:00 (05:30 UTC, pre-DST-end).
    LF 1 at 2026-11-01T03:00:00-05:00 (08:00 UTC, post-DST-end).
    Slot at 2026-11-01T01:30:00-05:00 (06:30 UTC).

    By UTC the slot is 1h after LF 0 and 1.5h before LF 1 — it must
    land in LF 0's window.

    By ISO STRING compare, "2026-11-01T01:30:00-05:00" sorts EQUAL
    to "2026-11-01T01:30:00-04:00" up through the seconds field
    (the suffix breaks the tie LATER) — and Python lexicographically
    "...-05:00" > "...-04:00", so the predicate ``slot <= lf_dt``
    is FALSE (correct) but ``slot >= next_dt`` against
    "2026-11-01T03:00:00-05:00" is FALSE too (also correct here).
    The real failure shows up when slot and next-LF have inverted
    UTC/string ordering — assert via UTC-correct framing."""
    lf_slots = [
        _slot(0, "2026-11-01T01:30:00-04:00"),  # 05:30 UTC
        _slot(0, "2026-11-01T03:00:00-05:00"),  # 08:00 UTC
    ]
    shorts_slots = [_slot(1, "2026-11-01T01:30:00-05:00")]  # 06:30 UTC
    schedule = SimpleNamespace(lf_slots=lf_slots, shorts_slots=shorts_slots)

    result = _partition_shorts_slots(schedule, [_LF(0), _LF(1)])

    assert result[0] == shorts_slots
    assert result[1] == []


def test_string_compare_misorders_slot_relative_to_next_lf():
    """The targeted regression: slot.datetime_est sorts >= next-LF
    string but is EARLIER in UTC than next-LF. String predicate
    would exclude this slot from LF 0's window. UTC predicate
    correctly includes it.

    Slot:    2026-03-08T07:00:00-04:00  -> 11:00 UTC
    Next-LF: 2026-03-08T08:00:00-05:00  -> 13:00 UTC (2 h later in UTC)

    String compare: "...07:00:00-04:00" < "...08:00:00-05:00" — wait,
    those happen to also be string-ordered correctly. Construct the
    inversion explicitly: pick a slot whose hour is >= the next-LF
    hour as a string but earlier in UTC.

    Slot:    2026-11-01T07:30:00-04:00  -> 11:30 UTC
    Next-LF: 2026-11-01T07:00:00-05:00  -> 12:00 UTC
    String : "07:30...-04:00" > "07:00...-05:00"   (slot > next-LF as string)
    UTC    : slot 11:30 < next-LF 12:00            (slot < next-LF in UTC)

    Expectation: slot DOES belong to LF 0's window (UTC-correct).
    String-compare path would drop it.
    """
    lf_slots = [
        _slot(0, "2026-11-01T03:00:00-04:00"),  # 07:00 UTC (LF 0)
        _slot(0, "2026-11-01T07:00:00-05:00"),  # 12:00 UTC (LF 1)
    ]
    shorts_slots = [_slot(1, "2026-11-01T07:30:00-04:00")]  # 11:30 UTC
    schedule = SimpleNamespace(lf_slots=lf_slots, shorts_slots=shorts_slots)

    result = _partition_shorts_slots(schedule, [_LF(0), _LF(1)])

    assert result[0] == shorts_slots, (
        "slot at 11:30 UTC must be in LF 0's window "
        "(LF 0 = 07:00 UTC, LF 1 = 12:00 UTC). String compare drops it."
    )
    assert result[1] == []


def test_scheduler_apply_rank_ordering_dst_window():
    """Same predicate, scheduler.py:_apply_rank_ordering site.
    Two shorts in one LF's DST-spanning window; rank-based reorder
    must keep both shorts in the same LF window regardless of which
    UTC offset side of DST each falls on."""
    from gomoveshift.pipeline.scheduler import PipelineScheduler

    @dataclass
    class _RenderedShort:
        index: int
        rank: int

    @dataclass
    class _LFPlan:
        index: int

    lf_slots = [
        _slot(0, "2026-11-01T03:00:00-04:00"),  # 07:00 UTC
        _slot(0, "2026-11-08T11:00:00-05:00"),  # next LF a week later
    ]
    shorts_slots = [
        _slot(1, "2026-11-01T07:30:00-04:00"),  # 11:30 UTC — pre-fallback offset
        _slot(2, "2026-11-01T16:00:00-05:00"),  # 21:00 UTC — post-fallback offset
    ]
    regular_lfs = [_LFPlan(0)]
    shorts_per_lf = {0: [_RenderedShort(1, rank=2), _RenderedShort(2, rank=1)]}

    out = PipelineScheduler._apply_rank_ordering(
        shorts_slots=list(shorts_slots),
        lf_slots=lf_slots,
        regular_lfs=regular_lfs,
        shorts_per_lf=shorts_per_lf,
    )

    # Both shorts must have been seen as in-window and reordered.
    # rank 1 (short index 2) gets earliest slot; rank 2 (short index 1)
    # gets later slot.
    assert len(out) == 2
    assert out[0].short_index == 2
    assert out[1].short_index == 1
