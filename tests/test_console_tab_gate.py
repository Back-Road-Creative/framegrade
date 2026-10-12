"""A brand's disabled tabs must be closed, not just hidden from the nav.

``_require_tab`` (app.py) is called from every drives/shorts/library/
publishing/schedule/jobs route, but ten routes across photos, seo, comments,
journeys and settings never call it -- a direct GET/POST reaches the handler
and runs even when the brand's ``brand.yaml`` disables that tab. This exists
structurally now (a middleware keyed off the request path's first segment),
not as ten more call sites.
"""

import pytest
from fastapi.testclient import TestClient

import gomoveshift.console.app as console_app
import gomoveshift.console.tabs as console_tabs
from gomoveshift.console.app import create_app

#: (method, path, owning tab) for every route that did NOT already call
#: ``_require_tab`` -- verified by reading every ``@app.get``/``@app.post``
#: decorator in app.py.
_UNGATED_ROUTES = [
    ("GET", "/photos", "photos"),
    ("POST", "/photos/run", "photos"),
    ("GET", "/seo", "seo"),
    ("POST", "/seo/corrections", "seo"),
    ("GET", "/comments", "comments"),
    ("GET", "/journeys", "journeys"),
    ("POST", "/journeys/sync-site", "journeys"),
    ("POST", "/journeys/publish-journal-replay", "journeys"),
    ("GET", "/settings", "settings"),
    ("POST", "/settings", "settings"),
]


def _client_with_disabled(monkeypatch, disabled: set[str]) -> TestClient:
    monkeypatch.setenv("GMS_CONSOLE_BRAND", "gomoveshift")
    monkeypatch.setattr(console_app, "brand_disabled_tabs", lambda _slug: disabled)
    monkeypatch.setattr(console_tabs, "brand_disabled_tabs", lambda _slug: disabled)
    return TestClient(create_app())


@pytest.mark.parametrize("method,path,tab", _UNGATED_ROUTES)
def test_disabled_tab_closes_the_route_not_just_the_nav_link(monkeypatch, method, path, tab):
    client = _client_with_disabled(monkeypatch, {tab})
    response = client.request(method, path, data={})
    assert response.status_code == 404


def test_an_enabled_tab_still_works(monkeypatch):
    client = _client_with_disabled(monkeypatch, {"photos"})
    assert client.get("/seo").status_code == 200


def test_healthz_is_unaffected(monkeypatch):
    client = _client_with_disabled(monkeypatch, {"photos", "seo"})
    assert client.get("/healthz").status_code == 200


def test_default_brand_has_no_disabled_tabs_and_every_tab_is_reachable(monkeypatch):
    monkeypatch.delenv("GMS_CONSOLE_BRAND", raising=False)
    client = TestClient(create_app())
    for _method, path, _tab in _UNGATED_ROUTES:
        # GET-able entries only -- POST bodies aren't the point of this check.
        if _method == "GET":
            assert client.get(path).status_code == 200
