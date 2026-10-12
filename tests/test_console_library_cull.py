"""The culling score (sharpness, exposure clipping, near-duplicate
phash) surfaced read-only on the console asset page. Never a verdict --
no auto-delete, no auto-reject; a photo that can't be decoded just carries
no cull info instead of failing the page.
"""

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from gomoveshift.console import library
from gomoveshift.console.app import create_app

JOURNEY_TWO_PHOTOS = (
    "---\ndays:\n- date: '2025-01-01'\n  photos:\n  - photos: [photo-001.jpg, photo-002.jpg]\n"
    "    photo_hashes: {}\n---\nBody\n"
)

JOURNEY_ONE_PHOTO = (
    "---\ndays:\n- date: '2025-01-01'\n  photos:\n  - photos: [photo-001.jpg]\n"
    "    photo_hashes: {}\n---\nBody\n"
)


def _write_sharp_jpeg(path):
    rng = np.random.default_rng(4)
    Image.fromarray(rng.integers(0, 256, (60, 80, 3), dtype=np.uint8), "RGB").save(path, "JPEG")


def _site(tmp_path, journey_body, filenames):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True)
    (jdir / "journey-a.md").write_text(journey_body)
    images_dir = site / "static" / "images" / "journeys" / "journey-a"
    images_dir.mkdir(parents=True)
    for name in filenames:
        _write_sharp_jpeg(images_dir / name)
    return site


def _client() -> TestClient:
    return TestClient(create_app())


def test_asset_context_carries_a_cull_score_for_a_decodable_photo(monkeypatch, tmp_path):
    site = _site(tmp_path, JOURNEY_ONE_PHOTO, ["photo-001.jpg"])
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    ctx = library.build_asset_context("photo:journey-a:2025-01-01:photo-001.jpg")
    assert ctx["cull"] is not None
    assert ctx["cull"]["sharpness"] >= 0.0
    assert ctx["cull"]["exposure_black_fraction"] == 0.0
    assert ctx["cull"]["near_duplicates"] == []


def test_undecodable_photo_has_no_cull_info_but_page_still_renders(monkeypatch, tmp_path):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True)
    (jdir / "journey-a.md").write_text(JOURNEY_ONE_PHOTO)
    images_dir = site / "static" / "images" / "journeys" / "journey-a"
    images_dir.mkdir(parents=True)
    (images_dir / "photo-001.jpg").write_bytes(b"\xff\xd8\xff\xe0notreallyajpeg")
    monkeypatch.setenv("GMS_SITE_PATH", str(site))

    ctx = library.build_asset_context("photo:journey-a:2025-01-01:photo-001.jpg")
    assert ctx["cull"] is None

    response = _client().get("/library/photo:journey-a:2025-01-01:photo-001.jpg")
    assert response.status_code == 200


def test_identical_sibling_photos_are_flagged_as_near_duplicates(monkeypatch, tmp_path):
    site = _site(tmp_path, JOURNEY_TWO_PHOTOS, [])
    images_dir = site / "static" / "images" / "journeys" / "journey-a"
    # Same bytes on disk -- the same pHash by construction.
    _write_sharp_jpeg(images_dir / "photo-001.jpg")
    import shutil

    shutil.copyfile(images_dir / "photo-001.jpg", images_dir / "photo-002.jpg")
    monkeypatch.setenv("GMS_SITE_PATH", str(site))

    ctx = library.build_asset_context("photo:journey-a:2025-01-01:photo-001.jpg")
    assert ctx["cull"]["near_duplicates"] == ["photo-002.jpg"]

    response = _client().get("/library/photo:journey-a:2025-01-01:photo-001.jpg")
    assert response.status_code == 200
    assert "photo-002.jpg" in response.text
    assert "near-duplicate" in response.text.lower()


def test_video_asset_context_carries_no_cull_key_value(monkeypatch, tmp_path):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True)
    (jdir / "journey-a.md").write_text(JOURNEY_ONE_PHOTO)
    (site / "data").mkdir(parents=True)
    (site / "data" / "video_routing.yaml").write_text(
        "routes: {vid-1: {journey: journey-a, date: '2025-01-01'}}\n"
    )
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    ctx = library.build_asset_context("video:vid-1")
    assert ctx["cull"] is None
