"""``build_schedule_report``'s ``next_available`` availability MATRIX — the next
free long-form slot per (region, designed day) and that slot's associated shorts.

A matrix rather than one row per region on purpose: no config field binds a
region to a weekday (the "Ireland publishes Sunday" pairing lives only in
the free-text ``cadence_line_by_region``), so collapsing would assert a binding
the config does not contain."""

from __future__ import annotations

from datetime import datetime

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import schedule_constants
from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger
from gomoveshift.pipeline.schedule_report import build_schedule_report

ANCHOR = datetime(2026, 8, 10, 9, 0, tzinfo=schedule_constants.EST)  # a Monday
DESIGNED = {"sun": "14:00", "tue": "16:00", "fri": "15:00"}  # publish_slots lf_times


def _lf(date: str, region: str) -> ContentEntry:
    return ContentEntry(
        date=date,
        time="16:00:00",
        content_type="longform",
        title="Booked Drive",
        project_name="booked",
        status="scheduled",
        source="pipeline",
        region=region,
    )


@pytest.fixture
def ledger() -> ContentLedger:
    return ContentLedger()


def _rows(ledger, config=None):
    return build_schedule_report(ledger, config or Config(), after=ANCHOR)["next_available"]


def _earliest(rows, region: str) -> str:
    return next(r["slot"] for r in rows if r["region"] == region and r["earliest_for_region"])


class TestMatrixShape:
    def test_every_region_gets_a_row_per_designed_day(self, ledger):
        rows = _rows(ledger)
        assert {(r["region"], r["day"]) for r in rows} == {
            (reg, d) for reg in ("ireland", "usa") for d in ("tue", "fri", "sun")
        }
        assert all(datetime.fromisoformat(r["slot"]) > ANCHOR for r in rows)

    def test_exactly_one_row_per_region_is_marked_earliest(self, ledger):
        rows = _rows(ledger)
        for region in ("ireland", "usa"):
            mine = [r for r in rows if r["region"] == region]
            marked = [r for r in mine if r["earliest_for_region"]]
            assert len(marked) == 1
            assert all(marked[0]["slot"] <= r["slot"] for r in mine)


class TestOccupancy:
    def test_a_booked_day_is_skipped_for_that_region_only(self, ledger):
        before = _rows(ledger)
        usa_before, ireland_before = _earliest(before, "usa"), _earliest(before, "ireland")
        ledger.entries = [_lf(datetime.fromisoformat(usa_before).strftime("%Y-%m-%d"), "usa")]
        after = _rows(ledger)
        assert _earliest(after, "usa") != usa_before, "USA must move off its booked day"
        assert _earliest(after, "ireland") == ireland_before, "Ireland — different audience"


class TestSlotTime:
    def test_time_is_the_designed_one_not_the_flat_default(self, ledger):
        """publish_slots designs sun 14:00 / tue 16:00 / fri 15:00. next_free_slot
        stamps the flat long_form_publish_time (11:00) on whatever day it lands
        on — a time the pipeline never publishes at — so it must be substituted."""
        for row in _rows(ledger):
            assert datetime.fromisoformat(row["slot"]).strftime("%H:%M") == DESIGNED[row["day"]]

    def test_a_designed_time_earlier_than_the_flat_default_is_never_backdated(self, ledger):
        """A day designed EARLIER than the flat long_form_publish_time (11:00)
        must have its true time honoured by the SEARCH itself — patching the
        time on afterward can return a slot already before ``after`` (the
        search's own "not yet occurred" check ran against the wrong time)."""
        config = Config()
        config.drive.publish_slots["sun"].lf_time = "09:00"
        after = datetime(2026, 8, 16, 10, 0, tzinfo=schedule_constants.EST)  # a Sunday, 10:00
        rows = build_schedule_report(ledger, config, after=after)["next_available"]
        sun_row = next(r for r in rows if r["day"] == "sun" and r["region"] == "usa")
        assert datetime.fromisoformat(sun_row["slot"]) > after


class TestAssociatedShorts:
    def test_every_slot_carries_its_regions_windows_and_count(self, ledger):
        rows = _rows(ledger)
        for row in rows:
            assert row["shorts_windows"], f"{row['region']}/{row['day']} resolved no windows"
            # required_shorts_per_lf = windows x LF_PUBLISH_INTERVAL_DAYS (7)
            assert row["shorts_required"] == len(row["shorts_windows"]) * 7
        ireland = next(r for r in rows if r["region"] == "ireland")
        assert ireland["shorts_windows"] == ["08:00", "12:00"]

    def test_usa_windows_stagger_by_day_per_the_anti_cannibalisation_map(self, ledger):
        """USA's shorts map is a per-day dict, so Tue and Fri must differ."""
        usa = {r["day"]: r["shorts_windows"] for r in _rows(ledger) if r["region"] == "usa"}
        assert usa["tue"] == ["14:00", "18:00"]
        assert usa["fri"] == ["10:00", "16:00"]


class TestNoMutation:
    def test_the_matrix_never_touches_the_global_cadence_mirror(self, ledger):
        """Each candidate day is passed to next_free_slot as an EXPLICIT
        publish_day, so LONG_FORM_DAYS — which a drive run narrows to its own
        --lf-day — is never read or written."""
        days, time_ = schedule_constants.LONG_FORM_DAYS, schedule_constants.LONG_FORM_TIME
        _rows(ledger)
        assert schedule_constants.LONG_FORM_DAYS == days
        assert schedule_constants.LONG_FORM_TIME == time_
