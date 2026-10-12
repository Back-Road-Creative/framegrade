"""Console Schedule tab: read-only view over
``pipeline.schedule_report.build_schedule_report``. Never re-implements
scheduling; ``schedule_view`` composes the same primitives the CLI's
``cmd_schedule`` does."""

from __future__ import annotations

from fastapi.testclient import TestClient

from gomoveshift.console import schedule as console_schedule
from gomoveshift.console.app import NAV_ENTRIES, create_app


def _client() -> TestClient:
    return TestClient(create_app())


_FAKE_REPORT = {
    "as_of": "2026-08-10T09:00:00-04:00",
    "weekly_longform_capacity": 3,
    "designed_days": ["tue", "fri", "sun"],
    "scheduled_longforms": [
        {
            "date": "2026-08-11",
            "time": "16:00:00",
            "region": "usa",
            "title": "Booked Drive Through The Hills",
            "kind": "longform",
        }
    ],
    "scheduled_shorts_count": 4,
    "queue_ends": "2026-08-11",
    "next_available": [
        {
            "region": "usa",
            "day": "tue",
            "slot": "2026-08-18T16:00:00-04:00",
            "shorts_windows": ["14:00", "18:00"],
            "shorts_required": 2,
            "earliest_for_region": True,
        }
    ],
    "next_available_superlong": [
        {"region": "usa", "day": "sat", "slot": "2026-08-15T15:00:00-04:00"}
    ],
}


def test_schedule_view_returns_report_dict(monkeypatch):
    monkeypatch.setattr(
        console_schedule, "build_schedule_report", lambda ledger, config: _FAKE_REPORT
    )
    result = console_schedule.schedule_view()
    assert result == _FAKE_REPORT


def test_schedule_view_missing_ledger_returns_empty_report_with_hint(monkeypatch):
    def _raise(ledger, config):
        raise RuntimeError("boom")

    monkeypatch.setattr(console_schedule, "build_schedule_report", _raise)
    result = console_schedule.schedule_view()
    assert result["scheduled_longforms"] == []
    assert result["next_available"] == []
    assert result["next_available_superlong"] == []
    assert result["hint"]


def test_schedule_route_renders_booked_title_and_matrix(monkeypatch):
    monkeypatch.setattr(
        console_schedule, "build_schedule_report", lambda ledger, config: _FAKE_REPORT
    )
    resp = _client().get("/schedule")
    assert resp.status_code == 200
    assert "Booked Drive Through The Hills" in resp.text
    assert "usa" in resp.text
    assert "tue" in resp.text


def test_nav_contains_schedule():
    assert ("schedule", "Schedule") in NAV_ENTRIES
    resp = _client().get("/")
    assert 'href="/schedule"' in resp.text


def test_examplebrand_brand_disabled_tab_is_404(shipped_brands, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_BRAND", "examplebrand")
    client = TestClient(create_app())
    from gomoveshift.console import settings as console_settings

    disabled = console_settings.brand_disabled_tabs("examplebrand")
    assert disabled == {"drives", "shorts"}
    for slug in sorted(disabled):
        assert client.get(f"/{slug}").status_code == 404
    # A tab the brand does not disable is still served.
    assert client.get("/schedule").status_code == 200
