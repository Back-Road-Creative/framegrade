"""The asset page serves a derivative and says so.

`photo_hashes[filename]` is the SHA-256 of the ORIGINAL card file --
`_source_hash`'s docstring is explicit that it is "never derived from the
developed output" (see `_source_hash` in `pipeline/photos/ingest.py`). What
`GET /library/media/{id}` serves is the developed export under
`static/images/journeys/`. The page showed the image and a bare "hash
recorded" line beside it, so an operator who hashed the bytes they were
looking at would get a different number and conclude the file was corrupt.
These pin that the page says which file each fact is about, and never claims
the original is here.
"""

from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import library
from gomoveshift.console.app import create_app

JOURNEY = """---
days:
- date: '2025-01-01'
  photos:
  - location: Place A
    photos:
      - photo-001.jpg
    photo_hashes:
      photo-001.jpg: abc123
photo_provenance:
  photo-001.jpg: rawtherapee
---
Body
"""

LEGACY_JOURNEY = """---
days:
- date: '2025-01-01'
  photos:
  - location: Place A
    photos:
      - photo-001.jpg
---
Body
"""

PHOTO_ID = "photo:journey-a:2025-01-01:photo-001.jpg"
ROUTING_YAML = "routes: {vid-1: {journey: journey-a, date: '2025-01-01'}}\n"
JPEG = b"\xff\xd8\xff\xdb" + b"derivative bytes" * 8


def _site(tmp_path, monkeypatch, *, frontmatter=JOURNEY, with_file=True, routing=False):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True)
    (jdir / "journey-a.md").write_text(frontmatter)
    if with_file:
        images = site / library.IMAGES_BASE / "journey-a"
        images.mkdir(parents=True)
        (images / "photo-001.jpg").write_bytes(JPEG)
    if routing:
        (site / "data").mkdir(parents=True, exist_ok=True)
        (site / "data" / "video_routing.yaml").write_text(ROUTING_YAML)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    return site


def test_the_recorded_hash_is_labelled_as_the_originals_not_the_files(tmp_path, monkeypatch):
    _site(tmp_path, monkeypatch)

    context = library.build_asset_context(PHOTO_ID)

    assert context["original"]["content_hash"] == "abc123"
    assert context["original"]["present_here"] is False


def test_the_page_never_calls_a_recorded_hash_a_verification(tmp_path, monkeypatch):
    """The reader is offline and opens no media -- nothing has compared these
    bytes to anything, and the page must not imply otherwise."""
    _site(tmp_path, monkeypatch)

    text = TestClient(create_app()).get(f"/library/{PHOTO_ID}").text.lower()

    for word in ("verified", "matches", "checksum ok", "integrity confirmed"):
        assert word not in text


def test_what_is_served_is_described_as_the_developed_export(tmp_path, monkeypatch):
    _site(tmp_path, monkeypatch)

    context = library.build_asset_context(PHOTO_ID)

    assert context["derivative"]["relative_path"] == (
        f"{library.IMAGES_BASE}/journey-a/photo-001.jpg"
    )
    assert context["derivative"]["size_bytes"] == len(JPEG)


def test_the_page_says_the_original_is_not_in_this_checkout(tmp_path, monkeypatch):
    """The honest half of the answer. The original lives on the card the
    photos were ingested from; nothing in the site checkout holds it."""
    _site(tmp_path, monkeypatch)

    text = TestClient(create_app()).get(f"/library/{PHOTO_ID}").text

    assert "not in this site" in text.lower()


def test_a_legacy_photo_with_no_recorded_hash_invents_nothing(tmp_path, monkeypatch):
    _site(tmp_path, monkeypatch, frontmatter=LEGACY_JOURNEY)

    context = library.build_asset_context(PHOTO_ID)
    text = TestClient(create_app()).get(f"/library/{PHOTO_ID}").text

    assert context["original"]["content_hash"] is None
    assert context["original"]["decoder"] is None
    assert "no hash recorded" in text.lower()


def test_a_missing_export_is_reported_at_the_place_it_should_be(tmp_path, monkeypatch):
    """ "Photo unavailable" alone does not tell an operator where to look."""
    _site(tmp_path, monkeypatch, with_file=False)

    context = library.build_asset_context(PHOTO_ID)

    assert context["derivative"]["present"] is False
    assert context["derivative"]["relative_path"] == (
        f"{library.IMAGES_BASE}/journey-a/photo-001.jpg"
    )
    assert context["derivative"]["size_bytes"] is None


def test_a_video_row_claims_no_photo_derivative(tmp_path, monkeypatch):
    """A video's bytes are on the destination platform, not here. Borrowing
    the photo shape would state a file that does not exist."""
    _site(tmp_path, monkeypatch, routing=True)

    context = library.build_asset_context("video:vid-1")

    assert context["derivative"] is None
    assert context["original"] is None


def test_a_photo_with_everything_recorded_still_renders_its_image(tmp_path, monkeypatch):
    """The accept-case: the new section is added beside the preview, never
    instead of it, and the bytes still serve."""
    _site(tmp_path, monkeypatch)
    client = TestClient(create_app())

    page = client.get(f"/library/{PHOTO_ID}")
    media = client.get(f"/library/media/{PHOTO_ID}")

    # The raw-id form is incidental to what the test is about (the preview
    # survives beside the new section and the bytes still serve). A
    # catalog_id is built from a real filename, so the template encodes it --
    # one `#` in a name would otherwise truncate the src at the fragment.
    # Both forms reach the {catalog_id:path} route; the media fetch above
    # still uses the raw id and still gets 200.
    assert f'src="/library/media/{quote(PHOTO_ID)}"' in page.text
    assert media.status_code == 200
    assert media.content == JPEG


def test_the_decoder_is_named_as_what_developed_the_export(tmp_path, monkeypatch):
    _site(tmp_path, monkeypatch)

    context = library.build_asset_context(PHOTO_ID)

    assert context["original"]["decoder"] == "rawtherapee"


@pytest.mark.parametrize("missing", [True, False])
def test_the_context_always_carries_both_halves_for_a_photo(tmp_path, monkeypatch, missing):
    """The template reads both unconditionally for a photo row; a missing key
    is a 500 on a page whose whole job is to degrade visibly."""
    _site(tmp_path, monkeypatch, with_file=not missing)

    context = library.build_asset_context(PHOTO_ID)

    assert set(context["original"]) == {"content_hash", "decoder", "present_here"}
    assert set(context["derivative"]) == {"relative_path", "present", "size_bytes"}
