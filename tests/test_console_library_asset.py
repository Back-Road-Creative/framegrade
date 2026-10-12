"""The asset page and its photo byte-serving route.

``GET /library/{catalog_id}`` and ``GET /library/media/{catalog_id}``.
"""

from fastapi.testclient import TestClient

from gomoveshift.console.app import create_app

JOURNEY_A = (
    "---\ndays:\n- date: '2025-01-01'\n  photos:\n  - photos: [photo-001.jpg]\n"
    "    photo_hashes: {photo-001.jpg: abc123}\nphoto_provenance: {photo-001.jpg: raw}\n---\nBody\n"
)

# A photo entry whose filename is recorded but the file is missing on disk.
JOURNEY_MISSING_FILE = (
    "---\ndays:\n- date: '2025-01-01'\n  photos:\n  - photos: [ghost.jpg]\n"
    "    photo_hashes: {}\n---\nBody\n"
)

# A photo entry whose filename tries to escape the media root -- the fixture
# genuinely carries a traversal segment, per the design.
JOURNEY_TRAVERSAL = (
    "---\ndays:\n- date: '2025-01-01'\n  photos:\n  - photos: ['../evil.jpg']\n"
    "    photo_hashes: {}\n---\nBody\n"
)

JOURNEY_BAD_SUFFIX = (
    "---\ndays:\n- date: '2025-01-01'\n  photos:\n  - photos: [notes.txt]\n"
    "    photo_hashes: {}\n---\nBody\n"
)

JOURNEY_SYMLINK = (
    "---\ndays:\n- date: '2025-01-01'\n  photos:\n  - photos: [link.jpg]\n"
    "    photo_hashes: {}\n---\nBody\n"
)

ROUTING_YAML = "routes: {vid-1: {journey: journey-a, date: '2025-01-01'}}\n"


def _site(tmp_path, journey_name, journey_body, *, with_photo_file=False, with_symlink=False):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / f"{journey_name}.md").write_text(journey_body)
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "video_routing.yaml").write_text(ROUTING_YAML)
    if with_photo_file:
        images_dir = site / "static" / "images" / "journeys" / journey_name
        images_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / "photo-001.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    if with_symlink:
        images_dir = site / "static" / "images" / "journeys" / journey_name
        images_dir.mkdir(parents=True, exist_ok=True)
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(b"secret")
        (images_dir / "link.jpg").symlink_to(outside)
    return site


def _client() -> TestClient:
    return TestClient(create_app())


def test_photo_asset_page_renders_journey_hash_and_decoder(monkeypatch, tmp_path):
    site = _site(tmp_path, "journey-a", JOURNEY_A, with_photo_file=True)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    response = _client().get("/library/photo:journey-a:2025-01-01:photo-001.jpg")
    assert response.status_code == 200
    assert "journey-a" in response.text
    # The recorded hash itself, not the word "recorded": the page names the value
    # and says which file it identifies (the original, not the export shown).
    assert "abc123" in response.text
    assert "raw" in response.text


def test_video_asset_page_has_no_media_link_and_says_url_is_preview(monkeypatch, tmp_path):
    site = _site(tmp_path, "journey-a", JOURNEY_A)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    response = _client().get("/library/video:vid-1")
    assert response.status_code == 200
    assert "vid-1" in response.text
    assert "/library/media/video:vid-1" not in response.text
    assert "published url" in response.text.lower() or "preview" in response.text.lower()


def test_unknown_catalog_id_is_404(monkeypatch, tmp_path):
    site = _site(tmp_path, "journey-a", JOURNEY_A)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    response = _client().get("/library/photo:nope:2025-01-01:x.jpg")
    assert response.status_code == 404


def test_served_photo_returns_bytes_and_content_type(monkeypatch, tmp_path):
    site = _site(tmp_path, "journey-a", JOURNEY_A, with_photo_file=True)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    response = _client().get("/library/media/photo:journey-a:2025-01-01:photo-001.jpg")
    assert response.status_code == 200
    assert response.content == b"\xff\xd8\xff\xe0fakejpeg"
    assert response.headers["content-type"].startswith("image/")


def test_missing_file_renders_unavailable_state_on_asset_page(monkeypatch, tmp_path):
    site = _site(tmp_path, "journey-missing", JOURNEY_MISSING_FILE)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    response = _client().get("/library/photo:journey-missing:2025-01-01:ghost.jpg")
    assert response.status_code == 200
    assert "unavailable" in response.text.lower()

    media_response = _client().get("/library/media/photo:journey-missing:2025-01-01:ghost.jpg")
    assert media_response.status_code == 404


def test_available_photo_page_differs_from_unavailable_page(monkeypatch, tmp_path):
    site_ok = _site(tmp_path / "ok", "journey-a", JOURNEY_A, with_photo_file=True)
    monkeypatch.setenv("GMS_SITE_PATH", str(site_ok))
    ok_text = _client().get("/library/photo:journey-a:2025-01-01:photo-001.jpg").text

    site_missing = _site(tmp_path / "missing", "journey-missing", JOURNEY_MISSING_FILE)
    monkeypatch.setenv("GMS_SITE_PATH", str(site_missing))
    missing_text = _client().get("/library/photo:journey-missing:2025-01-01:ghost.jpg").text

    assert ok_text != missing_text
    assert "unavailable" not in ok_text.lower()


def test_traversal_filename_is_refused_not_rewritten(monkeypatch, tmp_path):
    site = _site(tmp_path, "journey-trav", JOURNEY_TRAVERSAL)
    # Plant the file the traversal targets, one level above the media root,
    # to prove the route refuses to serve it rather than merely 404ing on a
    # coincidentally-absent file.
    (tmp_path / "site" / "static" / "images" / "evil.jpg").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "site" / "static" / "images" / "evil.jpg").write_bytes(b"escaped")
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    response = _client().get("/library/media/photo:journey-trav:2025-01-01:../evil.jpg")
    assert response.status_code == 404
    assert response.content != b"escaped"


def test_symlink_escaping_media_root_is_refused(monkeypatch, tmp_path):
    site = _site(tmp_path, "journey-sym", JOURNEY_SYMLINK, with_symlink=True)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    response = _client().get("/library/media/photo:journey-sym:2025-01-01:link.jpg")
    assert response.status_code == 404
    assert response.content != b"secret"


def test_non_image_suffix_is_404_not_downloaded(monkeypatch, tmp_path):
    site = _site(tmp_path, "journey-txt", JOURNEY_BAD_SUFFIX)
    images_dir = site / "static" / "images" / "journeys" / "journey-txt"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "notes.txt").write_text("not an image")
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    response = _client().get("/library/media/photo:journey-txt:2025-01-01:notes.txt")
    assert response.status_code == 404
