"""Cross-run shorts-slot interleaving (ledger-aware collision avoidance).

Two USA drives must not double-book the same (date, time). The scheduler draws
`shorts_per_day` shorts from a priority-ordered `slot_pool`, skipping any slot
already booked by another run, so the second drive falls to the next free times.
"""

from __future__ import annotations

import datetime as dt

from gomoveshift.pipeline.scheduler import PipelineScheduler

# Region-fenced USA pool (matches config.drive.shorts_slot_pool_by_region["usa"]):
# USA owns 10/14/16/18 — Ireland's 08:00/12:00 are deliberately absent so a
# contended USA drive can never substitute into an Ireland window.
USA_POOL = [
    dt.time(14, 0),
    dt.time(18, 0),
    dt.time(10, 0),
    dt.time(16, 0),
]


def _lf(idx):
    return type("LF", (), {"index": idx})()


def _shorts_times(schedule):
    return sorted({s.time for s in schedule.shorts_slots})


def test_first_drive_claims_priority_times():
    """With nothing booked, a USA drive uses the top-priority pair 14:00/18:00."""
    sched = PipelineScheduler(shorts_times=[dt.time(14, 0), dt.time(18, 0)])
    out = sched.generate_drive_schedule(
        regular_lfs=[_lf(0)],
        region="usa",
        start_date="2026-07-07",  # a Tuesday
        slot_pool=USA_POOL,
        shorts_per_day=2,
    )
    assert _shorts_times(out) == ["14:00:00", "18:00:00"]


def test_second_drive_interleaves_to_free_times():
    """When 14:00/18:00 are booked every day, the next drive falls to 10:00/16:00."""
    sched = PipelineScheduler(shorts_times=[dt.time(14, 0), dt.time(18, 0)])
    # Book 14:00 + 18:00 across a wide window (simulating the SC queue).
    occupied = set()
    d = dt.date(2026, 7, 1)
    for _ in range(60):
        occupied.add((d.strftime("%Y-%m-%d"), "14:00:00"))
        occupied.add((d.strftime("%Y-%m-%d"), "18:00:00"))
        d += dt.timedelta(days=1)
    out = sched.generate_drive_schedule(
        regular_lfs=[_lf(0)],
        region="usa",
        start_date="2026-07-07",
        occupied_slots=occupied,
        slot_pool=USA_POOL,
        shorts_per_day=2,
    )
    times = set(_shorts_times(out))
    # Never reuses an SC-occupied slot...
    assert not ({(s.date, s.time) for s in out.shorts_slots} & occupied)
    assert "14:00:00" not in times and "18:00:00" not in times
    # ...and the region-fenced USA fallback pair (10:00/16:00) is what it falls
    # to — NEVER Ireland's 08:00/12:00 (defect A).
    assert {"10:00:00", "16:00:00"} <= times
    # Every time used is drawn from the fenced USA pool, never Ireland's pair.
    assert times <= {"10:00:00", "16:00:00"}


def test_per_day_count_respected():
    """Exactly `shorts_per_day` shorts are placed per day even with a 6-slot pool."""
    sched = PipelineScheduler(shorts_times=[dt.time(14, 0), dt.time(18, 0)])
    out = sched.generate_drive_schedule(
        regular_lfs=[_lf(0)],
        region="usa",
        start_date="2026-07-07",
        slot_pool=USA_POOL,
        shorts_per_day=2,
    )
    by_date: dict[str, int] = {}
    for s in out.shorts_slots:
        by_date[s.date] = by_date.get(s.date, 0) + 1
    assert by_date and all(n <= 2 for n in by_date.values())


def test_backward_compatible_without_pool_or_occupied():
    """Default call (no pool/occupied) reproduces the legacy 2-slot behavior."""
    sched = PipelineScheduler(shorts_times=[dt.time(14, 0), dt.time(18, 0)])
    legacy = sched.generate_drive_schedule(
        regular_lfs=[_lf(0)], region="usa", start_date="2026-07-07"
    )
    assert _shorts_times(legacy) == ["14:00:00", "18:00:00"]


# Ireland posts 08:00 + 12:00 — both verified collision-free vs USA's grid
# (USA owns 10/14/16/18). 16:00 is a USA slot, NOT Ireland's, so a USA 16:00
# long-form/short no longer lands on an Ireland shorts slot (the old defect).
IRELAND_POOL = [dt.time(8, 0), dt.time(12, 0)]


def test_ireland_2lf_unaffected_by_usa_16():
    """A 2-LF Ireland drive at 08:00/12:00 places all its shorts even when USA
    content occupies 16:00 (the old double-owned slot) — never touches 16:00,
    nothing dropped."""
    sched = PipelineScheduler(shorts_times=[dt.time(8, 0), dt.time(12, 0)])
    base = sched.generate_drive_schedule(
        regular_lfs=[_lf(0), _lf(1)],
        region="ireland",
        start_date="2026-06-21",
        slot_pool=IRELAND_POOL,
        shorts_per_day=2,
    )
    usa16 = {("2026-06-23", "16:00:00"), ("2026-06-30", "16:00:00")}
    out = sched.generate_drive_schedule(
        regular_lfs=[_lf(0), _lf(1)],
        region="ireland",
        start_date="2026-06-21",
        occupied_slots=usa16,
        slot_pool=IRELAND_POOL,
        shorts_per_day=2,
    )
    assert len(out.shorts_slots) == len(base.shorts_slots)  # USA 16:00 irrelevant
    assert {s.time for s in out.shorts_slots} <= {"08:00:00", "12:00:00"}  # never 16:00


def test_ireland_config_grid_disjoint_from_usa():
    """Shipped config: Ireland's slots (08/12) and USA's pool (10/14/16/18) are
    fully disjoint — no double-owned time like the old 16:00."""
    from gomoveshift.config import DriveConfig

    d = DriveConfig()
    ie = set(d.shorts_publish_times_by_region["ireland"])
    usa = set(d.shorts_slot_pool_by_region["usa"])
    assert ie == {"08:00", "12:00"}
    assert not (ie & usa)


def test_run_locks_one_pair_and_never_drifts_back():
    """A drive keeps ONE pair run-wide; it must NOT revert to 14:00/18:00 on the
    tail days where the other drive's queue has ended (a past regression)."""
    sched = PipelineScheduler(shorts_times=[dt.time(14, 0), dt.time(18, 0)])
    # Occupy 14:00/18:00 for the first 42 days only; the run spans ~8 weeks, so
    # the last ~2 weeks have 14:00/18:00 free. Per-date logic would drift there.
    occupied = set()
    d = dt.date(2026, 7, 7)
    for _ in range(42):
        occupied.add((d.strftime("%Y-%m-%d"), "14:00:00"))
        occupied.add((d.strftime("%Y-%m-%d"), "18:00:00"))
        d += dt.timedelta(days=1)
    out = sched.generate_drive_schedule(
        regular_lfs=[_lf(i) for i in range(8)],
        region="usa",
        start_date="2026-07-07",
        occupied_slots=occupied,
        slot_pool=USA_POOL,
        shorts_per_day=2,
    )
    times = set(_shorts_times(out))
    # Locked to the fenced USA pair 10:00/16:00 for the WHOLE run — no drift back
    # to the freed 14:00/18:00, and never Ireland's 08:00/12:00 (defect A).
    assert "14:00:00" not in times and "18:00:00" not in times
    assert times <= {"10:00:00", "16:00:00"}
    assert {"10:00:00", "16:00:00"} <= times
