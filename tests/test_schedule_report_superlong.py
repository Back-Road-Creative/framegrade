"""``build_schedule_report`` must surface the superlong publish day as a
first-class prospective slot, and must label an already-booked superlong
part distinctly from a regular long-form — today it is invisible in the
first case and indistinguishable in the second."""

from __future__ import annotations

from datetime import datetime

import pytest

from gomoveshift.cli.handlers.schedule import _render
from gomoveshift.config import Config
from gomoveshift.pipeline import schedule_constants
from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger
from gomoveshift.pipeline.schedule_report import build_schedule_report

ANCHOR = datetime(2026, 8, 10, 9, 0, tzinfo=schedule_constants.EST)  # a Monday


@pytest.fixture
def ledger() -> ContentLedger:
    return ContentLedger()


def _report(ledger, config=None):
    return build_schedule_report(ledger, config or Config(), after=ANCHOR)


class TestSuperlongProspectiveSlot:
    def test_the_superlong_day_appears_as_a_prospective_slot_per_region(self, ledger):
        report = _report(ledger)
        rows = report["next_available_superlong"]
        assert {r["region"] for r in rows} == {"ireland", "usa"}
        for row in rows:
            assert row["day"] == "sat"  # config default, resolved not hardcoded
            assert datetime.fromisoformat(row["slot"]) > ANCHOR


class TestBookedSuperlongPartIsLabeled:
    def test_a_booked_superlong_part_is_distinguished_from_a_regular_longform(self, ledger):
        ledger.entries = [
            ContentEntry(
                date="2026-08-15",
                time="15:00:00",
                content_type="longform",
                title="Driving the Ring of Kerry — Part 1 of 3",
                project_name="superlong",
                status="scheduled",
                source="pipeline",
                region="ireland",
            ),
            ContentEntry(
                date="2026-08-16",
                time="14:00:00",
                content_type="longform",
                title="A to B, NC in 4K",
                project_name="regular",
                status="scheduled",
                source="pipeline",
                region="usa",
            ),
        ]
        report = _report(ledger)
        by_title = {e["title"]: e["kind"] for e in report["scheduled_longforms"]}
        assert by_title["Driving the Ring of Kerry — Part 1 of 3"] == "superlong_part"
        assert by_title["A to B, NC in 4K"] == "longform"

    def test_the_rendered_report_shows_both(self, ledger):
        """A field the renderer never reads is dead scaffolding: ``gomoveshift
        schedule`` prints ``_render``, not the JSON, so the operator sees the
        superlong day only if the text output carries it."""
        ledger.entries = [
            ContentEntry(
                date="2026-08-15",
                time="15:00:00",
                content_type="longform",
                title="Driving the Ring of Kerry — Part 1 of 3",
                project_name="superlong",
                status="scheduled",
                source="pipeline",
                region="ireland",
            ),
        ]
        out = _render(_report(ledger))
        assert "Next available superlong part:" in out
        assert "sat" in out.split("Next available superlong part:")[1]
        assert "SL" in [ln.split()[2] for ln in out.splitlines() if "Ring of Kerry" in ln][0]
