"""Console Media library tab -- ``site_path`` is tenant config only."""

import inspect

from fastapi.testclient import TestClient

from gomoveshift.console import library
from gomoveshift.console.app import NAV_ENTRIES, create_app

JOURNEY_A = (
    "---\ndays:\n- date: '2025-01-01'\n  photos:\n  - photos: [photo-001.jpg, photo-legacy.jpg]\n"
    "    photo_hashes: {photo-001.jpg: abc123}\n---\nBody text\n"
)

BAD_YAML_JOURNEY = "---\ntitle: [unterminated\ndays: []\n---\nBody text\n"

ROUTING_YAML = "routes: {vid-1: {journey: journey-a, date: '2025-01-01'}}\n"


def _site(tmp_path, *, damaged=False):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True)
    (jdir / "journey-a.md").write_text(JOURNEY_A)
    if damaged:
        (jdir / "journey-bad.md").write_text(BAD_YAML_JOURNEY)
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "video_routing.yaml").write_text(ROUTING_YAML)
    return site


def _client() -> TestClient:
    return TestClient(create_app())


def _spy_read_catalog(monkeypatch):
    captured = {}
    real_read_catalog = library.read_catalog

    def spy(path, **kwargs):
        captured["path"] = str(path)
        captured.update(kwargs)
        return real_read_catalog(path, **kwargs)

    monkeypatch.setattr(library, "read_catalog", spy)
    return captured


def test_resolve_site_path_takes_no_path_argument_and_reads_env(monkeypatch, tmp_path):
    # The signature is the guarantee: nothing here can accept an
    # operator-supplied filesystem path.
    params = inspect.signature(library.resolve_site_path).parameters
    assert "site_path" not in params and "path" not in params
    monkeypatch.setenv("GMS_SITE_PATH", str(tmp_path))
    assert library.resolve_site_path() == str(tmp_path)


def test_filters_pass_through_to_read_catalog_not_applied_in_template(monkeypatch, tmp_path):
    site = _site(tmp_path)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    captured = _spy_read_catalog(monkeypatch)
    library.build_context(media_type="photo", journey="journey-a", date="2025-01-01")
    assert captured == {
        "path": str(site),
        "media_type": "photo",
        "journey": "journey-a",
        "date": "2025-01-01",
        "catalog_id": None,
    }


def test_damaged_journey_summary_excludes_path_and_raw_error(monkeypatch, tmp_path):
    site = _site(tmp_path, damaged=True)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    ctx = library.build_context()
    assert ctx["damaged"] == [{"source": "journey-bad", "reason": "could not be read"}]
    assert any(e.journey == "journey-a" for e in ctx["entries"])


def test_library_is_first_nav_entry():
    assert NAV_ENTRIES[0][0] == "library"


def test_library_route_renders_mixed_collection_never_says_verified(monkeypatch, tmp_path):
    site = _site(tmp_path)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    response = _client().get("/library")
    assert response.status_code == 200
    assert "photo:journey-a" in response.text
    assert "video:vid-1" in response.text
    assert "verified" not in response.text.lower()
    assert "no hash recorded" in response.text.lower()


def test_library_route_ignores_operator_supplied_site_path(monkeypatch, tmp_path):
    site = _site(tmp_path)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    captured = _spy_read_catalog(monkeypatch)
    response = _client().get("/library", params={"site_path": "/etc/passwd"})
    assert response.status_code == 200
    assert captured["path"] == str(site)


def test_library_route_damage_banner_never_leaks_path_or_raw_error(monkeypatch, tmp_path):
    site = _site(tmp_path, damaged=True)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    response = _client().get("/library")
    assert response.status_code == 200
    assert str(site) not in response.text
    assert "unterminated" not in response.text
    assert "journey-bad" in response.text
    assert "photo:journey-a" in response.text


def test_library_route_empty_and_damaged_only_render_differently(monkeypatch, tmp_path):
    empty_site = tmp_path / "empty"
    (empty_site / "content" / "journeys").mkdir(parents=True)
    monkeypatch.setenv("GMS_SITE_PATH", str(empty_site))
    empty_text = _client().get("/library").text

    damaged_site = tmp_path / "damaged"
    jdir = damaged_site / "content" / "journeys"
    jdir.mkdir(parents=True)
    (jdir / "journey-bad.md").write_text(BAD_YAML_JOURNEY)
    monkeypatch.setenv("GMS_SITE_PATH", str(damaged_site))
    damaged_text = _client().get("/library").text

    assert empty_text != damaged_text


def test_library_respects_brand_disabled_tabs(monkeypatch):
    import gomoveshift.console.app as console_app

    monkeypatch.setattr(console_app, "brand_disabled_tabs", lambda slug: {"library"})
    response = TestClient(console_app.create_app()).get("/library")
    assert response.status_code == 404


def test_a_damaged_routing_file_is_named_for_what_it_is(monkeypatch, tmp_path):
    """The catalog reads two kinds of source and only one is a journey.
    Labelling the routing file as the journey "video_routing" would send an
    operator looking through content/journeys for a file that is not there."""
    site = _site(tmp_path)
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "video_routing.yaml").write_text("routes:\n  v:\n    journey: [bad\n")
    monkeypatch.setattr(library, "resolve_site_path", lambda: site)

    sources = [d["source"] for d in library.build_context()["damaged"]]
    assert "the video routing file" in sources
