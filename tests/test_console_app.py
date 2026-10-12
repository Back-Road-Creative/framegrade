"""Console skeleton -- /healthz, index nav, unknown-tab 404."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from markupsafe import escape

import gomoveshift.console.app as console_app
from gomoveshift.console import jobs, pubqueue, tenant
from gomoveshift.console.app import NAV_ENTRIES, create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_healthz_returns_ok_and_brands(monkeypatch):
    monkeypatch.delenv("GMS_CONSOLE_NOTIFY_URL", raising=False)
    body = _client().get("/healthz").json()
    assert body["status"] == "ok"
    assert "gomoveshift" in body["brands"]
    assert body["observer_ok"] is None  # ticker disabled -- no observer running
    assert body["notify_configured"] is False


def test_healthz_reports_notify_configured_when_the_webhook_url_is_set(monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", "http://example/hook")
    resp = _client().get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["notify_configured"] is True


def test_index_lists_the_nine_nav_entries():
    assert len(NAV_ENTRIES) == 11
    response = _client().get("/")
    assert response.status_code == 200
    for _slug, title in NAV_ENTRIES:
        assert str(escape(title)) in response.text


def test_known_slug_renders_not_built_yet_placeholder(monkeypatch):
    # Every NAV_ENTRIES slug now has a real tab; exercise the
    # placeholder mechanism itself with a synthetic future nav entry rather
    # than a real, already-built one.
    monkeypatch.setattr(console_app, "NAV_ENTRIES", NAV_ENTRIES + [("future-tab", "Future Tab")])
    monkeypatch.setattr(console_app, "_NAV_SLUGS", console_app._NAV_SLUGS | {"future-tab"})
    response = TestClient(console_app.create_app()).get("/future-tab")
    assert response.status_code == 200
    assert "Not built yet" in response.text
    assert str(escape("Future Tab")) in response.text


def test_unknown_slug_is_404():
    assert _client().get("/does-not-exist").status_code == 404


def test_unhandled_exception_is_logged_and_notified(monkeypatch, caplog):
    """An unhandled exception was previously swallowed to a bare 500 page --
    no log line, no notify. It must do both while still rendering the page."""
    app = create_app()

    @app.get("/boom")
    def _boom():
        raise RuntimeError("kaboom")

    boom_route = app.router.routes.pop()
    app.router.routes.insert(len(app.router.routes) - 1, boom_route)

    calls = []
    monkeypatch.setattr(
        console_app.notify,
        "notify",
        lambda event, identity, status: calls.append((event, identity, status)),
    )
    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level("ERROR"):
        resp = client.get("/boom")

    assert resp.status_code == 500
    assert "kaboom" in caplog.text
    assert len(calls) == 1
    event, identity, status = calls[0]
    assert event == "console_error"
    assert identity["path"] == "/boom"
    assert identity["exc_type"] == "RuntimeError"
    assert status == "kaboom"


def _isolated_storage_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pubqueue.Config,
        "load",
        staticmethod(
            lambda *a, **k: SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
        ),
    )


def test_lifespan_claims_tenant_storage_on_startup(tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_CONSOLE_BRAND", raising=False)
    _isolated_storage_config(tmp_path, monkeypatch)

    with TestClient(create_app()):
        pass

    assert (tmp_path / ".waypost-tenant").read_text().strip() == "gomoveshift"


def test_lifespan_refuses_a_mismatched_brand_and_never_sweeps_or_mutates(tmp_path, monkeypatch):
    stamp = tmp_path / ".waypost-tenant"
    stamp.write_text("some-other-brand")
    _isolated_storage_config(tmp_path, monkeypatch)
    monkeypatch.setenv("GMS_CONSOLE_BRAND", "gomoveshift")

    swept = []
    monkeypatch.setattr(console_app.jobs, "sweep_orphaned_jobs", lambda *a, **k: swept.append(True))

    with pytest.raises(tenant.TenantMismatchError) as excinfo:
        with TestClient(create_app()):
            pass

    message = str(excinfo.value)
    assert "gomoveshift" in message
    assert "some-other-brand" in message
    assert swept == []
    assert stamp.read_text().strip() == "some-other-brand"
