"""Tests for ``publishing/photo_image_url.py``.

No network call ever happens here -- every reachability check injects a
fake ``session`` object standing in for ``requests.Session``.
"""

from __future__ import annotations

import requests

from gomoveshift.publishing.photo_image_url import (
    NOT_FOUND,
    UNREACHABLE,
    WRONG_CONTENT_TYPE,
    resolve_photo_image_url,
    site_base_url,
    verify_image_url,
)


def _write_hugo_config(site_root, base_url):
    config_dir = site_root / "config" / "_default"
    config_dir.mkdir(parents=True)
    (config_dir / "hugo.toml").write_text(f'baseURL = "{base_url}"\n')


class _Response:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class _FakeSession:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self._exc is not None:
            raise self._exc
        return self._response


def test_site_base_url_reads_hugo_toml(tmp_path):
    _write_hugo_config(tmp_path, "https://gomoveshift.com/")
    assert site_base_url(tmp_path) == "https://gomoveshift.com"


def test_site_base_url_missing_config_returns_none(tmp_path):
    assert site_base_url(tmp_path) is None


def test_site_base_url_override_wins(tmp_path):
    _write_hugo_config(tmp_path, "https://gomoveshift.com/")
    assert site_base_url(tmp_path, override="https://waypost.example/") == "https://waypost.example"


def test_resolve_photo_image_url_from_site_base_url(tmp_path):
    _write_hugo_config(tmp_path, "https://gomoveshift.com")
    url = resolve_photo_image_url("example-2026-08", "photo-002.jpg", site_root=tmp_path)
    assert url == "https://gomoveshift.com/images/journeys/example-2026-08/photo-002.jpg"


def test_resolve_photo_image_url_falls_back_to_drive(tmp_path):
    drive_url = "https://drive.google.com/file/d/ABC123/view?usp=sharing"
    url = resolve_photo_image_url(
        "example-2026-08",
        "photo-002.jpg",
        site_root=tmp_path,
        drive_share_url=drive_url,
    )
    assert url == "https://drive.google.com/uc?export=download&id=ABC123"


def test_resolve_photo_image_url_raises_without_any_source(tmp_path):
    import pytest

    from gomoveshift.publishing.photo_image_url import PhotoImageUrlError

    with pytest.raises(PhotoImageUrlError, match="example-2026-08/photo-002.jpg"):
        resolve_photo_image_url("example-2026-08", "photo-002.jpg", site_root=tmp_path)


def test_verify_image_url_ok(tmp_path):
    session = _FakeSession(_Response(200, {"Content-Type": "image/jpeg"}))
    result = verify_image_url("https://x/y.jpg", session=session, site_root=tmp_path)
    assert result.ok is True
    assert result.reason == ""


def test_verify_image_url_not_found_names_deploy_step(tmp_path):
    site_root = tmp_path / "example-site"
    site_root.mkdir()
    session = _FakeSession(_Response(404))
    result = verify_image_url("https://x/y.jpg", session=session, site_root=site_root)
    assert result.ok is False
    assert result.reason.startswith(NOT_FOUND)
    assert "deploy the site (example-site)" in result.reason


def test_verify_image_url_wrong_content_type(tmp_path):
    session = _FakeSession(_Response(200, {"Content-Type": "text/html"}))
    result = verify_image_url("https://x/y.jpg", session=session, site_root=tmp_path)
    assert result.ok is False
    assert result.reason.startswith(WRONG_CONTENT_TYPE)


def test_verify_image_url_unreachable_on_request_exception(tmp_path):
    session = _FakeSession(exc=requests.ConnectionError("boom"))
    result = verify_image_url("https://x/y.jpg", session=session, site_root=tmp_path)
    assert result.ok is False
    assert result.reason.startswith(UNREACHABLE)


def test_verify_image_url_unreachable_on_other_status(tmp_path):
    session = _FakeSession(_Response(500))
    result = verify_image_url("https://x/y.jpg", session=session, site_root=tmp_path)
    assert result.ok is False
    assert result.reason.startswith(UNREACHABLE)
