"""The Jobs console page must never auto-reload itself.

A `<meta http-equiv="refresh">` interrupts the required schedule-name and
interval-minutes inputs on `/jobs` mid-entry -- nobody types a schedule name
in under five seconds, so the old behaviour made adding a schedule a race
against a timer. It's also a WCAG 2.2.1 (Timing Adjustable) violation: an
auto-updating page must be pausable, stoppable, or adjustable.

The fix: drop the meta refresh entirely and give the operator an explicit,
no-JS refresh control instead.
"""

from fastapi.testclient import TestClient


def test_job_list_never_carries_a_meta_refresh_even_while_running(monkeypatch, tmp_path):
    import gomoveshift.console.app as console_app

    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    running_rows = [
        {
            "id": 1,
            "name": "status",
            "argv": ["status"],
            "status": "running",
            "rc": None,
            "created_at": "2026-09-01T12:00:00+00:00",
            "finished_at": None,
            "log_path": "",
        }
    ]
    monkeypatch.setattr(console_app.jobs, "run_due", lambda *a, **k: [])
    monkeypatch.setattr(console_app.jobs, "list_jobs", lambda store: running_rows)
    monkeypatch.setattr(console_app.jobs, "list_schedules", lambda store: [])

    resp = TestClient(console_app.create_app()).get("/jobs")
    assert resp.status_code == 200
    assert "http-equiv=" not in resp.text.lower()


def test_job_list_offers_an_explicit_refresh_link(monkeypatch, tmp_path):
    import gomoveshift.console.app as console_app

    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(console_app.jobs, "run_due", lambda *a, **k: [])
    monkeypatch.setattr(console_app.jobs, "list_jobs", lambda store: [])
    monkeypatch.setattr(console_app.jobs, "list_schedules", lambda store: [])

    resp = TestClient(console_app.create_app()).get("/jobs")
    assert resp.status_code == 200
    assert '<a href="/jobs">' in resp.text
