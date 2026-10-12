"""``schedule_report.build_schedule_report`` — the read-only answer to "what is
scheduled, when does it end, what is the next available long form, and therefore
which shorts". Composes existing primitives; must never mutate cadence state."""

from __future__ import annotations

from datetime import datetime

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import schedule_constants
from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger
from gomoveshift.pipeline.schedule_report import build_schedule_report

# Monday, so the next tue/fri/sun all fall in the same week.
ANCHOR = datetime(2026, 8, 10, 9, 0, tzinfo=schedule_constants.EST)


def _entry(date: str, region: str, kind: str, time: str) -> ContentEntry:
    return ContentEntry(
        date=date,
        time=time,
        content_type=kind,
        title="Booked Drive",
        project_name="booked",
        status="scheduled",
        source="pipeline",
        region=region,
    )


def _lf(date: str, region: str) -> ContentEntry:
    return _entry(date, region, "longform", "16:00:00")


def _short(date: str, region: str) -> ContentEntry:
    return _entry(date, region, "shorts", "14:00:00")


@pytest.fixture
def ledger() -> ContentLedger:
    return ContentLedger()


def _report(ledger, config=None):
    return build_schedule_report(ledger, config or Config(), after=ANCHOR)


class TestCapacity:
    @pytest.mark.parametrize(
        "drop,capacity,days",
        [
            (None, 3, ["tue", "fri", "sun"]),  # Python weekday order, not prose order
            ("fri", 2, ["tue", "sun"]),  # capacity follows config, never a hardcoded 3
        ],
    )
    def test_capacity_and_days_come_from_publish_slots(self, ledger, drop, capacity, days):
        config = Config()
        if drop:
            config.drive.publish_slots = {
                k: v for k, v in config.drive.publish_slots.items() if k != drop
            }
        report = _report(ledger, config)
        assert report["weekly_longform_capacity"] == capacity
        assert report["designed_days"] == days


class TestWhatIsScheduledAndWhenItEnds:
    def test_reports_booked_longforms_and_shorts_counts(self, ledger):
        ledger.entries = [
            _lf("2026-08-11", "usa"),
            _lf("2026-08-16", "ireland"),
            _short("2026-08-11", "usa"),
        ]
        report = _report(ledger)
        assert [e["date"] for e in report["scheduled_longforms"]] == ["2026-08-11", "2026-08-16"]
        assert report["scheduled_shorts_count"] == 1

    @pytest.mark.parametrize(
        "entries,expected",
        [
            ([], None),  # nothing booked
            ([_lf("2026-08-11", "usa"), _lf("2026-09-01", "usa")], "2026-09-01"),
            # a trailing SHORT must never extend the long-form queue end
            ([_lf("2026-08-11", "usa"), _short("2026-12-25", "usa")], "2026-08-11"),
        ],
    )
    def test_queue_ends_is_the_last_booked_longform_date(self, ledger, entries, expected):
        ledger.entries = list(entries)
        report = _report(ledger)
        assert report["queue_ends"] == expected
        if not entries:
            assert report["scheduled_longforms"] == []


class TestNoMutation:
    def test_building_a_report_does_not_touch_global_cadence(self, ledger):
        days, time_ = schedule_constants.LONG_FORM_DAYS, schedule_constants.LONG_FORM_TIME
        _report(ledger)
        assert schedule_constants.LONG_FORM_DAYS == days
        assert schedule_constants.LONG_FORM_TIME == time_
