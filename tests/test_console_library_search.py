"""Free-form tags on library items + free-text search across the
console library. Tags live in per-brand storage (path separation, the same
tenant-isolation primitive as the rest of the console) -- never a shared
store keyed by brand column, so brand A's tags must never surface in brand
B's library."""

from fastapi.testclient import TestClient

from gomoveshift.catalog import tags as library_tags
from gomoveshift.console import library
from gomoveshift.console.app import create_app

JOURNEY_MD = (
    "---\ndays:\n- date: '2025-01-01'\n  photos:\n  - photos: ['dublin.jpg']\n"
    "    photo_hashes: {'dublin.jpg': abc123}\n---\nBody\n"
)
ROUTING_YAML = "routes: {}\n"


def _site(tmp_path):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "ireland.md").write_text(JOURNEY_MD)
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "video_routing.yaml").write_text(ROUTING_YAML)
    return site


def _client() -> TestClient:
    return TestClient(create_app())


def _catalog_id():
    return "photo:ireland:2025-01-01:dublin.jpg"


def test_add_tag_then_filter_by_search_finds_it(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path)))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    client = _client()
    catalog_id = _catalog_id()

    response = client.post(f"/library/{catalog_id}/tags", data={"tag": "Roadtrip", "action": "add"})
    assert response.status_code in (200, 303)

    search = client.get("/library", params={"q": "roadtrip"})
    assert search.status_code == 200
    assert "dublin.jpg" in search.text

    miss = client.get("/library", params={"q": "no-such-tag"})
    assert "dublin.jpg" not in miss.text


def test_remove_tag_drops_it_from_search(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path)))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    client = _client()
    catalog_id = _catalog_id()

    client.post(f"/library/{catalog_id}/tags", data={"tag": "keeper", "action": "add"})
    client.post(f"/library/{catalog_id}/tags", data={"tag": "keeper", "action": "remove"})

    search = client.get("/library", params={"q": "keeper"})
    assert "dublin.jpg" not in search.text


def test_search_matches_title_and_journey_case_insensitively(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path)))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    client = _client()

    by_title = client.get("/library", params={"q": "DUBLIN"})
    assert "dublin.jpg" in by_title.text

    by_journey = client.get("/library", params={"q": "IRELAND"})
    assert "dublin.jpg" in by_journey.text

    no_match = client.get("/library", params={"q": "nonexistent-place"})
    assert "dublin.jpg" not in no_match.text


def test_search_combines_with_structural_filters(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path)))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    client = _client()

    matches_type = client.get("/library", params={"q": "dublin", "media_type": "photo"})
    assert "dublin.jpg" in matches_type.text

    wrong_type = client.get("/library", params={"q": "dublin", "media_type": "video"})
    assert "dublin.jpg" not in wrong_type.text


def test_tags_are_escaped_in_the_template(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path)))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    client = _client()
    catalog_id = _catalog_id()

    client.post(
        f"/library/{catalog_id}/tags",
        data={"tag": "<script>alert(1)</script>", "action": "add"},
    )

    page = client.get("/library")
    assert "<script>alert(1)</script>" not in page.text
    assert "&lt;script&gt;" in page.text


def test_brand_a_tags_never_appear_in_brand_b_library(monkeypatch, tmp_path):
    data_dir = tmp_path / "console"
    site = _site(tmp_path)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(data_dir))
    catalog_id = _catalog_id()

    monkeypatch.setenv("GMS_CONSOLE_BRAND", "brand-a")
    client_a = _client()
    client_a.post(f"/library/{catalog_id}/tags", data={"tag": "brand-a-only", "action": "add"})
    search_a = client_a.get("/library", params={"q": "brand-a-only"})
    assert "dublin.jpg" in search_a.text

    monkeypatch.setenv("GMS_CONSOLE_BRAND", "brand-b")
    client_b = _client()
    search_b = client_b.get("/library", params={"q": "brand-a-only"})
    assert "dublin.jpg" not in search_b.text

    # And directly at the storage layer -- brand B's own tag file never
    # contains brand A's tag for the same catalog_id.
    tags_b = library_tags.load_tags(data_dir, "brand-b")
    assert catalog_id not in tags_b


def test_library_tags_for_returns_added_tags(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path)))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    catalog_id = _catalog_id()

    library.add_library_tag(catalog_id, "scenic")
    library.add_library_tag(catalog_id, "coast")

    context = library.build_context()
    row = context["display"][catalog_id]
    assert set(row["tags"]) == {"scenic", "coast"}
