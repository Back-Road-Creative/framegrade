"""Second console brand (examplebrand) -- same image,
GMS_CONSOLE_BRAND selects which tabs are enabled."""

from fastapi.testclient import TestClient

from gomoveshift.console import settings as console_settings
from gomoveshift.console.app import NAV_ENTRIES, create_app


def test_known_brands_lists_both_gomoveshift_and_examplebrand(shipped_brands):
    brands = console_settings.known_brands()
    assert "gomoveshift" in brands
    assert "examplebrand" in brands


def test_default_app_is_unchanged_all_tabs_present(shipped_brands, monkeypatch):
    monkeypatch.delenv("GMS_CONSOLE_BRAND", raising=False)
    client = TestClient(create_app())

    response = client.get("/")
    assert response.status_code == 200
    for slug, _title in NAV_ENTRIES:
        assert f'href="/{slug}"' in response.text

    assert client.get("/drives").status_code == 200
    assert client.get("/shorts").status_code == 200


def test_examplebrand_brand_hides_drives_and_shorts(shipped_brands, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_BRAND", "examplebrand")
    client = TestClient(create_app())

    response = client.get("/")
    assert response.status_code == 200
    assert 'href="/drives"' not in response.text
    assert 'href="/shorts"' not in response.text
    assert 'href="/photos"' in response.text
    assert 'href="/seo"' in response.text

    assert client.get("/drives").status_code == 404
    assert client.get("/shorts").status_code == 404


def test_examplebrand_brand_keeps_enabled_tabs_rendering(shipped_brands, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_BRAND", "examplebrand")
    client = TestClient(create_app())

    assert client.get("/photos").status_code == 200
    assert client.get("/seo").status_code == 200
    assert client.get("/comments").status_code == 200
    assert client.get("/journeys").status_code == 200
    assert client.get("/settings").status_code == 200
    assert client.get("/jobs").status_code == 200
