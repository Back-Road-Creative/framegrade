"""Superlong slots are ONE shared weekly queue, not a per-region reservation.

Every region resolves the SAME ``superlong_publish_day`` absent an explicit
per-region override in ``config.drive.superlong_publish_day_by_region`` — and
no config sets one (Saturday is the intended shared day).
So unlike a regular long-form, whose regions sit on genuinely disjoint
weekdays (USA Tue/Fri, Ireland Sun), two regions' superlong parts compete
for the SAME Saturday. ``next_free_superlong_slot`` must block cross-region
by default so a taken Saturday is taken for everyone and the next producer
(whichever region has content ready) takes the next free one — never two
regions silently double-booking the same slot.

Regular long-form ``next_free_slot`` keeps its own disjoint-day default
unchanged: a USA LF must still be free to take a day an Ireland LF holds.
"""

from __future__ import annotations

from datetime import datetime

from gomoveshift.config import Config
from gomoveshift.pipeline.content_ledger import (
    ContentEntry,
    ContentLedger,
    next_free_slot,
    next_free_superlong_slot,
)
from gomoveshift.pipeline.schedule_constants import EST

ANCHOR = datetime(2026, 8, 10, 9, 0, tzinfo=EST)  # a Monday


def _superlong_entry(date: str, region: str) -> ContentEntry:
    return ContentEntry(
        date=date,
        time="11:00:00",
        content_type="longform",
        title="Driving the Ring of Kerry — Part 1 of 3",
        project_name="superlong",
        status="scheduled",
        source="pipeline",
        region=region,
    )


def _lf_entry(date: str, region: str) -> ContentEntry:
    return ContentEntry(
        date=date,
        time="14:00:00",
        content_type="longform",
        title="Booked Drive",
        project_name="booked_drive",
        status="scheduled",
        source="pipeline",
        region=region,
    )


class TestSuperlongSlotsAreASharedQueue:
    def test_usa_moves_to_a_later_saturday_when_ireland_already_holds_this_one(self):
        ledger = ContentLedger()
        ledger.entries.append(_superlong_entry("2026-08-15", "ireland"))  # a Saturday
        slot = next_free_superlong_slot(ledger, Config(), "usa", after=ANCHOR)
        assert slot.date().isoformat() != "2026-08-15"
        assert slot.date().isoformat() == "2026-08-22"

    def test_ireland_moves_to_a_later_saturday_when_usa_already_holds_this_one(self):
        """Converse of the above — the fix is not direction-specific."""
        ledger = ContentLedger()
        ledger.entries.append(_superlong_entry("2026-08-15", "usa"))  # a Saturday
        slot = next_free_superlong_slot(ledger, Config(), "ireland", after=ANCHOR)
        assert slot.date().isoformat() != "2026-08-15"
        assert slot.date().isoformat() == "2026-08-22"

    def test_regular_longform_next_free_slot_is_unaffected(self):
        """The disjoint-day premise still holds for regular LF cadence — a
        USA LF must remain free to take a day an Ireland LF holds."""
        config = Config()
        config.drive.long_form_publish_days = [6]  # Sunday, shared for this check
        config.drive.long_form_publish_time = "14:00"
        ledger = ContentLedger()
        ledger.entries.append(_lf_entry("2026-08-16", "ireland"))  # a Sunday
        slot = next_free_slot(ledger, config, "usa", after=ANCHOR)
        assert slot.date().isoformat() == "2026-08-16"
