"""Console errors render an HTML page with the nav (not raw JSON) for
browser requests, keep JSON for Accept: application/json, and the job list
shows created/finished timestamps + rc, highlights failures. The job list
never meta-refreshes itself (that wiped the schedule-name/interval inputs
mid-entry and violated WCAG 2.2.1) -- see test_console_jobs_no_autorefresh.py."""

from fastapi.testclient import TestClient
from markupsafe import escape

from gomoveshift.console.app import NAV_ENTRIES, create_app


def _client(**kwargs) -> TestClient:
    return TestClient(create_app(), **kwargs)


def test_unknown_route_html_request_gets_nav_and_status():
    resp = _client().get("/does-not-exist", headers={"accept": "text/html"})
    assert resp.status_code == 404
    assert "text/html" in resp.headers["content-type"]
    for _slug, title in NAV_ENTRIES:
        assert str(escape(title)) in resp.text
    assert "404" in resp.text


def test_unknown_route_json_request_gets_json():
    resp = _client().get("/does-not-exist", headers={"accept": "application/json"})
    assert resp.status_code == 404
    assert "application/json" in resp.headers["content-type"]
    assert "detail" in resp.json()


def _add_boom_route(app):
    """Register /boom ahead of the catch-all ``/{slug}`` route (added last by
    ``create_app()``), since routes match in registration order."""

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    boom_route = app.router.routes.pop()
    app.router.routes.insert(len(app.router.routes) - 1, boom_route)


def test_unhandled_exception_html_request_renders_error_page():
    app = create_app()
    _add_boom_route(app)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/boom", headers={"accept": "text/html"})
    assert resp.status_code == 500
    assert "text/html" in resp.headers["content-type"]
    for _slug, title in NAV_ENTRIES:
        assert str(escape(title)) in resp.text
    assert "kaboom" not in resp.text


def test_unhandled_exception_json_request_gets_json():
    app = create_app()
    _add_boom_route(app)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/boom", headers={"accept": "application/json"})
    assert resp.status_code == 500
    assert "application/json" in resp.headers["content-type"]
    assert "kaboom" not in resp.text


def test_job_list_shows_timestamps_rc_and_never_meta_refreshes(monkeypatch, tmp_path):
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
        },
        {
            "id": 2,
            "name": "drive",
            "argv": ["drive"],
            "status": "failed",
            "rc": 1,
            "created_at": "2026-09-01T11:00:00+00:00",
            "finished_at": "2026-09-01T11:05:00+00:00",
            "log_path": "",
        },
    ]
    monkeypatch.setattr(console_app.jobs, "run_due", lambda *a, **k: [])
    monkeypatch.setattr(console_app.jobs, "list_jobs", lambda store: running_rows)
    monkeypatch.setattr(console_app.jobs, "list_schedules", lambda store: [])

    resp = TestClient(console_app.create_app()).get("/jobs")
    assert resp.status_code == 200
    assert "2026-09-01T12:00:00+00:00" in resp.text
    assert "2026-09-01T11:05:00+00:00" in resp.text
    assert 'http-equiv="refresh"' not in resp.text
    assert 'class="job-failed"' in resp.text


def test_job_list_no_refresh_when_nothing_running(monkeypatch, tmp_path):
    import gomoveshift.console.app as console_app

    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    done_rows = [
        {
            "id": 1,
            "name": "status",
            "argv": ["status"],
            "status": "done",
            "rc": 0,
            "created_at": "2026-09-01T12:00:00+00:00",
            "finished_at": "2026-09-01T12:01:00+00:00",
            "log_path": "",
        }
    ]
    monkeypatch.setattr(console_app.jobs, "run_due", lambda *a, **k: [])
    monkeypatch.setattr(console_app.jobs, "list_jobs", lambda store: done_rows)
    monkeypatch.setattr(console_app.jobs, "list_schedules", lambda store: [])

    resp = TestClient(console_app.create_app()).get("/jobs")
    assert resp.status_code == 200
    assert 'http-equiv="refresh"' not in resp.text
