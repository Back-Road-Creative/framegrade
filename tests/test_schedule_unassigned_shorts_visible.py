"""An unfiltered shorts total whose parts do not add up says so on the page.

``scheduled_shorts_count`` is the whole queue. The per-region buckets exclude
shorts whose ``region`` was never recorded, because an unassigned short cannot
honestly be attributed to a region. That is the right call -- but it leaves the
HTML view showing a total the visible regions do not account for, with the
remainder reachable only through ``--json``. A number whose parts silently do
not sum is the same defect one layer down: the reader concludes the regions ARE
the total.

So the unfiltered page names the unassigned count whenever there is one, and
stays quiet when there is not -- an always-on "(0 unassigned)" trains the eye to
skip the line that matters.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from gomoveshift.console import schedule as console_schedule
from gomoveshift.console.app import create_app

_BASE = {
    "as_of": "2026-08-10T09:00:00-04:00",
    "weekly_longform_capacity": 3,
    "designed_days": ["tue", "fri", "sun"],
    "scheduled_longforms": [],
    "queue_ends": None,
    "next_available": [],
    "next_available_superlong": [],
}


def _report(**over) -> dict:
    return {**_BASE, **over}


def _render(monkeypatch, report) -> str:
    monkeypatch.setattr(
        console_schedule, "build_schedule_report", lambda ledger, config, **_kw: report
    )
    resp = TestClient(create_app()).get("/schedule")
    assert resp.status_code == 200
    return resp.text


def test_unassigned_shorts_are_named_on_the_unfiltered_page(monkeypatch):
    text = _render(
        monkeypatch,
        _report(
            scheduled_shorts_count=12,
            scheduled_shorts_by_region={"usa": 6, "ireland": 4, "unassigned": 2},
        ),
    )

    assert "Scheduled shorts: 12" in text
    assert "unassigned" in text.lower(), (
        "the page shows a total of 12 while its regions account for 10; the "
        "missing 2 must be visible here, not only in --json"
    )
    assert "2" in text


def test_no_unassigned_shorts_means_no_noise(monkeypatch):
    text = _render(
        monkeypatch,
        _report(
            scheduled_shorts_count=10,
            scheduled_shorts_by_region={"usa": 6, "ireland": 4},
        ),
    )

    assert "Scheduled shorts: 10" in text
    assert "unassigned" not in text.lower()


def test_a_report_with_no_breakdown_at_all_still_renders(monkeypatch):
    """A cached/legacy report predating the breakdown has no key to read."""
    text = _render(monkeypatch, _report(scheduled_shorts_count=7))

    assert "Scheduled shorts: 7" in text
    assert "unassigned" not in text.lower()
