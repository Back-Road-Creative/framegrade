"""The Schedule tab's shorts count must narrow under a journey
filter like every other figure on the page, instead of staying a whole-queue
total under a heading that says the view is filtered.

``build_schedule_report`` now reports shorts per region
(``scheduled_shorts_by_region``) alongside the existing whole-queue
``scheduled_shorts_count`` (unchanged -- the CLI and the unfiltered console
view still print it verbatim). A shorts entry with no recorded region lands
in its own named ``"unassigned"`` bucket -- never silently folded into a
region, never dropped from the total.

``console.schedule._filter_report_by_region`` then narrows
``scheduled_shorts_count`` itself to the matched region's bucket, mirroring
how it already narrows ``scheduled_longforms``/``queue_ends``. An unassigned
short can never be attributed to a specific region, so it is excluded from
every filtered (region-matched) view -- same "filter to nothing, never to
everything" rule already governing an unmatched journey.
"""

from __future__ import annotations

from datetime import datetime

from gomoveshift.config import Config
from gomoveshift.console.schedule import _filter_report_by_region
from gomoveshift.pipeline import schedule_constants
from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger
from gomoveshift.pipeline.schedule_report import build_schedule_report

ANCHOR = datetime(2026, 8, 10, 9, 0, tzinfo=schedule_constants.EST)


def _short(date: str, region) -> ContentEntry:
    return ContentEntry(
        date=date,
        time="14:00:00",
        content_type="shorts",
        title="Booked Short",
        project_name="booked",
        status="scheduled",
        source="pipeline",
        region=region,
    )


def _report(ledger, config=None):
    return build_schedule_report(ledger, config or Config(), after=ANCHOR)


class TestShortsByRegion:
    def test_reports_shorts_broken_down_by_region(self):
        ledger = ContentLedger()
        ledger.entries = [
            _short("2026-08-11", "usa"),
            _short("2026-08-12", "usa"),
            _short("2026-08-13", "ireland"),
        ]
        report = _report(ledger)
        assert report["scheduled_shorts_count"] == 3
        assert report["scheduled_shorts_by_region"] == {"usa": 2, "ireland": 1}

    def test_region_none_shorts_land_in_a_named_unassigned_bucket(self):
        ledger = ContentLedger()
        ledger.entries = [
            _short("2026-08-11", "usa"),
            _short("2026-08-12", None),
        ]
        report = _report(ledger)
        # never vanishes from the whole-queue total
        assert report["scheduled_shorts_count"] == 2
        # never silently attributed to a region
        assert report["scheduled_shorts_by_region"] == {"usa": 1, "unassigned": 1}


class TestFilteredShortsCount:
    def test_filter_narrows_shorts_count_to_the_matched_region(self):
        report = {
            "scheduled_longforms": [],
            "next_available": [],
            "next_available_superlong": [],
            "scheduled_shorts_count": 5,
            "scheduled_shorts_by_region": {"usa": 3, "ireland": 2},
        }
        filtered = _filter_report_by_region(report, "usa")
        assert filtered["scheduled_shorts_count"] == 3

    def test_unmatched_journey_filters_shorts_count_to_zero_not_everything(self):
        report = {
            "scheduled_longforms": [],
            "next_available": [],
            "next_available_superlong": [],
            "scheduled_shorts_count": 5,
            "scheduled_shorts_by_region": {"usa": 3, "ireland": 2},
        }
        filtered = _filter_report_by_region(report, None)
        assert filtered["scheduled_shorts_count"] == 0

    def test_unassigned_shorts_never_leak_into_a_region_filtered_view(self):
        report = {
            "scheduled_longforms": [],
            "next_available": [],
            "next_available_superlong": [],
            "scheduled_shorts_count": 4,
            "scheduled_shorts_by_region": {"usa": 3, "unassigned": 1},
        }
        filtered = _filter_report_by_region(report, "usa")
        assert filtered["scheduled_shorts_count"] == 3
