"""A drive's OWN partial-day booking must never be dodged as a rival.

Regression test: the identity-time exemption must not assume that a drive
always books its complete identity pair together each day, so a single-time
hit could not be its own re-seen booking.

**That premise is false in production.** Partial days occur naturally at the
edges of a schedule. A ledger for a drive with an identity pair
(10:00 + 16:00) can carry several of them::

    2031-03-03 [16:00]  2031-03-10 [16:00]  2031-03-14 [10:00]
    2031-03-21 [16:00]  2031-03-28 [10:00]

If each is classified as a rival, the drive dodges its own bookings, runs out
of placeable slots, and leaves shorts with no slot. The CSV exporter then
refuses to write the file (correctly -- a slotless row imports as an immediate
post), failing the run and booking a month of spurious slots on the way.

The proper rule needs actual ownership -- ``ContentLedger`` entries carry
``project_name``, which is the drive's identity -- rather than a per-date
pattern proxy. Whatever shape that takes, THIS invariant must hold: a
resumed drive re-seeing its own booking, whole day or partial, must reuse
that slot rather than treat it as contested.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from gomoveshift.pipeline.scheduler import PipelineScheduler

_IDENTITY = [dt.time(10, 0), dt.time(16, 0)]


def _lf(idx):
    return SimpleNamespace(index=idx)


def _times_on(schedule, date_str):
    return {s.time for s in schedule.shorts_slots if s.date == date_str}


def _run(occupied):
    sched = PipelineScheduler(shorts_times=list(_IDENTITY))
    return sched.generate_drive_schedule(
        regular_lfs=[_lf(0)],
        region="usa",
        start_date="2099-07-07",
        occupied_slots=occupied,
        slot_pool=[*_IDENTITY, dt.time(12, 0), dt.time(18, 0)],
        shorts_per_day=2,
        locked_identity_times=list(_IDENTITY),
    )


def test_own_partial_day_booking_is_reused_not_dodged():
    """One identity time booked on a date, the other free — the resumed run's
    own partial day. It must still place at BOTH identity times, not
    substitute the booked one away."""
    partial_date = "2099-07-08"
    out = _run({(partial_date, "16:00:00")})

    placed = _times_on(out, partial_date)
    assert "16:00:00" in placed, (
        "the drive dodged its OWN partial-day booking at 16:00 on "
        f"{partial_date} — that is the regression this test guards. Placed: {placed}"
    )
    assert "10:00:00" in placed, f"the free identity time was not placed either: {placed}"


def test_own_full_day_booking_is_reused_not_dodged():
    """The already-working case, pinned so a future change cannot regress it
    while fixing the partial-day case above."""
    full_date = "2099-07-08"
    out = _run({(full_date, "10:00:00"), (full_date, "16:00:00")})

    placed = _times_on(out, full_date)
    assert placed == {"10:00:00", "16:00:00"}, (
        f"a resumed run must return its locked pair verbatim, got: {placed}"
    )
