"""The collection view must let an operator recognise their own asset.

Type | Journey | Date | Catalog ID | Hash gave every row two machine
identifiers and nothing a person recognises. These pin the added Name/Kind/
Treatment/Privacy columns and, deliberately, the absence of any `<img>` in
the collection table: `CatalogEntry` (`catalog/reader.py`) and the journal
frontmatter it reads carry no blur/privacy signal for any entry, so a
thumbnail here would be an unverifiable image -- the same hard rule that
makes an unproven blur a hard fail on the publish side.
"""

from fastapi.testclient import TestClient

from gomoveshift.console import library
from gomoveshift.console.app import create_app

JOURNEY_A = (
    "---\ndays:\n- date: '2025-01-01'\n  photos:\n  - photos: [photo-001.jpg, photo-legacy.jpg]\n"
    "    photo_hashes: {photo-001.jpg: abc123}\nphoto_provenance: {photo-001.jpg: raw}\n---\nBody\n"
)

ROUTING_YAML = "routes: {vid-1: {journey: journey-a, date: '2025-01-01'}}\n"


def _site(tmp_path):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True)
    (jdir / "journey-a.md").write_text(JOURNEY_A)
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "video_routing.yaml").write_text(ROUTING_YAML)
    return site


def _client() -> TestClient:
    return TestClient(create_app())


def test_display_name_for_a_photo_is_its_export_filename():
    entry = library.CatalogEntry(
        catalog_id="photo:journey-a:2025-01-01:photo-001.jpg",
        media_type="photo",
        journey="journey-a",
        date="2025-01-01",
    )
    assert library._display_name(entry) == "photo-001.jpg"


def test_display_name_for_a_video_is_its_routing_id():
    entry = library.CatalogEntry(
        catalog_id="video:vid-1",
        media_type="video",
        journey="journey-a",
        date="2025-01-01",
        routing_id="vid-1",
    )
    assert library._display_name(entry) == "vid-1"


def test_privacy_status_never_claims_verified():
    entry = library.CatalogEntry(
        catalog_id="photo:journey-a:2025-01-01:photo-001.jpg",
        media_type="photo",
        journey="journey-a",
        date="2025-01-01",
    )
    status = library._privacy_status(entry)
    assert "verified" not in status.lower()
    assert status  # a real, honest state -- never blank


def test_build_context_carries_a_display_row_per_entry(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path)))
    ctx = library.build_context()
    for entry in ctx["entries"]:
        row = ctx["display"][entry.catalog_id]
        assert row["name"]
        assert row["kind"] in ("Photo", "Video")
        assert "privacy_status" in row
        assert "treatment" in row


def test_library_page_shows_name_kind_and_privacy_columns(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path)))
    text = _client().get("/library").text
    assert "photo-001.jpg" in text
    assert "vid-1" in text
    assert "Privacy" in text
    assert "verified" not in text.lower()


def test_library_page_renders_no_image_tags_in_the_collection_table(monkeypatch, tmp_path):
    """No `<img>` anywhere: the catalog carries no privacy signal, so no
    thumbnail may be served on this page regardless of whether the bytes
    happen to be reachable through the existing media route."""
    monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path)))
    text = _client().get("/library").text
    assert "<img" not in text
