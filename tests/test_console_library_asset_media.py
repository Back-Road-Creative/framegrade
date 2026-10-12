"""The asset page's photo ``<img src>`` must survive round-trip.

A ``catalog_id`` is built from a real card filename (``library.py``'s
``_photo_filename``), and some capture tools burst-suffix duplicate frames
with ``#`` (e.g. ``burst#2.jpg``). ``#`` is a URL fragment delimiter: an
unescaped ``src`` truncates there and the browser requests a *different*
resource than the one the page names.
"""

import re
from urllib.parse import quote

from fastapi.testclient import TestClient

from gomoveshift.console.app import create_app

# The filename itself carries the awkward character -- a real burst-capture
# naming convention, not an invented worst case.
JOURNEY_HASH_FILENAME = (
    "---\ndays:\n- date: '2025-01-01'\n  photos:\n  - photos: ['burst#2.jpg']\n"
    "    photo_hashes: {'burst#2.jpg': abc123}\n---\nBody\n"
)

ROUTING_YAML = "routes: {}\n"


def _site(tmp_path):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "journey-a.md").write_text(JOURNEY_HASH_FILENAME)
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "video_routing.yaml").write_text(ROUTING_YAML)
    images_dir = site / "static" / "images" / "journeys" / "journey-a"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "burst#2.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    return site


def _client() -> TestClient:
    return TestClient(create_app())


def test_asset_page_media_src_survives_hash_in_filename(monkeypatch, tmp_path):
    site = _site(tmp_path)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    client = _client()

    # The initial request goes through the encoded form -- exactly what a
    # real link (whose `href` is already `|urlencode`d) would actually send; the raw "#" belongs only to the
    # bug under test, which lives in this page's own <img src>.
    catalog_id = "photo:journey-a:2025-01-01:burst#2.jpg"
    response = client.get("/library/" + quote(catalog_id, safe=""))
    assert response.status_code == 200

    match = re.search(r'<img src="([^"]+)"', response.text)
    assert match is not None, "asset page did not render an <img> tag"
    src = match.group(1)

    # Follow the page's own output -- a test that hardcodes the expected URL
    # can pass by agreeing with a buggy encoder.
    media_response = client.get(src)
    assert media_response.status_code == 200
    assert media_response.content == b"\xff\xd8\xff\xe0fakejpeg"
