"""Tests for ``photo_image_url.image_url_for_media_path`` --
the piece that maps a ``MediaFingerprint.path`` onto the
``<slug>/<filename>`` pair ``resolve_photo_image_url`` wants, WITHOUT ever
guessing a slug/filename from a stem parse.
"""

from __future__ import annotations

import pytest

from gomoveshift.publishing.photo_image_url import (
    PhotoImageUrlError,
    image_url_for_media_path,
)


def _write_hugo_config(site_root, base_url):
    config_dir = site_root / "config" / "_default"
    config_dir.mkdir(parents=True)
    (config_dir / "hugo.toml").write_text(f'baseURL = "{base_url}"\n')


def test_derives_url_for_a_path_under_the_served_images_layout(tmp_path):
    _write_hugo_config(tmp_path, "https://gomoveshift.com")
    media_path = tmp_path / "static" / "images" / "journeys" / "example-2026-08" / "photo-002.jpg"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"x")

    url = image_url_for_media_path(str(media_path), site_root=tmp_path)

    assert url == "https://gomoveshift.com/images/journeys/example-2026-08/photo-002.jpg"


def test_raises_named_error_for_a_path_outside_the_served_layout(tmp_path):
    _write_hugo_config(tmp_path, "https://gomoveshift.com")
    # A resized social copy, e.g. under output/photo_resized -- flattened
    # naming, not served by the site at all.
    resized_path = tmp_path.parent / "photo_resized" / "example-2026-08_photo-002.jpg"

    with pytest.raises(PhotoImageUrlError, match=str(resized_path)):
        image_url_for_media_path(str(resized_path), site_root=tmp_path)


def test_raises_named_error_for_a_path_missing_the_slug_directory(tmp_path):
    _write_hugo_config(tmp_path, "https://gomoveshift.com")
    # Directly inside images/journeys/ with no slug subdirectory.
    media_path = tmp_path / "static" / "images" / "journeys" / "photo-002.jpg"

    with pytest.raises(PhotoImageUrlError, match=str(media_path)):
        image_url_for_media_path(str(media_path), site_root=tmp_path)
