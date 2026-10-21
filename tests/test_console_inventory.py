"""docs/console-inventory.md lists every console route.

The inventory is only useful while it is complete: a route added without a row
would be an enabled page nobody has classified as proven, blocked or unavailable.
"""

from pathlib import Path

import pytest
from fastapi.routing import APIRoute

from gomoveshift.console.app import create_app

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "console-inventory.md"

_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
_STATUSES = {"proven", "proven-test", "blocked", "unverified", "unavailable"}


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path))
    return create_app()


def _flatten(routes, prefix=""):
    """Yield every APIRoute, descending into FastAPI's lazily-included routers."""
    for route in routes:
        if isinstance(route, APIRoute):
            yield prefix + route.path, route
        elif hasattr(route, "original_router"):
            ctx = route.include_context
            yield from _flatten(route.original_router.routes, prefix + (ctx.prefix or ""))


def _mounted(app):
    """(METHOD, path) for every route the default app really serves, any decorator form."""
    return {
        (method, path)
        for path, route in _flatten(app.routes)
        for method in route.methods
        if method in _METHODS
    }


def _documented():
    rows = {}
    for line in DOC.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 4 and cells[0] in _METHODS:
            rows[(cells[0], cells[1].strip("`"))] = cells[3]
    return rows


def test_every_registered_route_has_an_inventory_row(app):
    missing = sorted(_mounted(app) - set(_documented()))
    assert not missing, f"routes with no row in docs/console-inventory.md: {missing}"


def test_the_inventory_lists_no_route_that_no_longer_exists(app):
    stale = sorted(set(_documented()) - _mounted(app))
    assert not stale, f"inventory rows for routes that are gone: {stale}"


def test_every_row_uses_a_known_status():
    bad = {k: v for k, v in _documented().items() if v not in _STATUSES}
    assert not bad, bad


def test_the_registered_set_is_not_vacuous(app):
    assert len(_mounted(app)) > 40


@pytest.mark.parametrize("method", ["put", "delete", "patch"])
def test_a_route_added_in_any_form_without_a_row_is_caught(app, method):
    app.add_api_route("/zz-new", lambda: None, methods=[method.upper()])
    assert (method.upper(), "/zz-new") in _mounted(app) - set(_documented())


def test_every_unverified_row_names_an_owner(app):
    rows = [
        line
        for line in DOC.read_text(encoding="utf-8").splitlines()
        if line.startswith("| ")
        and line.split("|")[1].strip() in _METHODS
        and "| unverified |" in line
    ]
    assert rows
    bad = [r for r in rows if "owner:" not in r]
    assert not bad, bad
