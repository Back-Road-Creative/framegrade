"""Tests for ``publishing/photo_graph_client.py`` .

No network call ever happens here -- every test injects a fake ``session``
object standing in for ``requests.Session``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gomoveshift.publishing.photo_adapter import PhotoAuthBlockedError, PhotoPermanentError
from gomoveshift.publishing.photo_graph_client import (
    MetaPhotoGraphClient,
    PhotoImageUrlUnverifiedError,
    PhotoPublishingLimitError,
)

SITE_ROOT = Path("/example/site")
CREDS = {"ig_user_id": "17841", "page_access_token": "tok"}
IMAGE_URL = "https://gomoveshift.com/images/journeys/example-2026-08/photo-002.jpg"


class _Response:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.headers = headers or {}

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self, responses):
        # queue of (method, response) consumed in order
        self._responses = list(responses)
        self.calls = []

    def _next(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._responses.pop(0)

    def post(self, url, **kwargs):
        return self._next("post", url, **kwargs)

    def get(self, url, **kwargs):
        return self._next("get", url, **kwargs)


def _quota_ok():
    return _Response(200, {"data": [{"quota_usage": 1, "config": {"quota_total": 25}}]})


def _quota_exhausted():
    return _Response(200, {"data": [{"quota_usage": 25, "config": {"quota_total": 25}}]})


def test_account_id_is_the_ig_user_id():
    client = MetaPhotoGraphClient(CREDS, session=_FakeSession([]), site_root=SITE_ROOT)
    assert client.account_id == "17841"


def test_create_single_container_verifies_url_then_posts_image_url():
    session = _FakeSession(
        [
            _Response(200, headers={"Content-Type": "image/jpeg"}),  # verification GET
            _Response(200, {"id": "container_1"}),  # media POST
        ]
    )
    client = MetaPhotoGraphClient(CREDS, session=session, site_root=SITE_ROOT)

    container_id = client.create_single_container(IMAGE_URL, "caption text")

    assert container_id == "container_1"
    verify_call, post_call = session.calls
    assert verify_call[0] == "get"
    assert verify_call[1] == IMAGE_URL
    method, url, kwargs = post_call
    assert method == "post"
    assert url == "https://graph.facebook.com/v21.0/17841/media"
    assert kwargs["data"]["image_url"] == IMAGE_URL
    assert kwargs["data"]["caption"] == "caption text"


def test_create_single_container_refuses_when_url_not_verified():
    session = _FakeSession([_Response(404)])
    client = MetaPhotoGraphClient(CREDS, session=session, site_root=SITE_ROOT)

    with pytest.raises(PhotoImageUrlUnverifiedError, match="not_found"):
        client.create_single_container(IMAGE_URL, "caption")

    # never reaches the Graph API once the URL fails verification.
    assert len(session.calls) == 1


def test_create_carousel_item_container_verifies_url_then_posts_is_carousel_item():
    session = _FakeSession(
        [
            _Response(200, headers={"Content-Type": "image/png"}),
            _Response(200, {"id": "item_1"}),
        ]
    )
    client = MetaPhotoGraphClient(CREDS, session=session, site_root=SITE_ROOT)

    container_id = client.create_carousel_item_container(IMAGE_URL)

    assert container_id == "item_1"
    method, url, kwargs = session.calls[1]
    assert kwargs["data"]["image_url"] == IMAGE_URL
    assert kwargs["data"]["is_carousel_item"] == "true"


def test_create_carousel_parent_container_posts_media_type_carousel():
    session = _FakeSession([_Response(200, {"id": "container_1"})])
    client = MetaPhotoGraphClient(CREDS, session=session, site_root=SITE_ROOT)

    container_id = client.create_carousel_parent_container(("item1", "item2"), "caption text")

    assert container_id == "container_1"
    method, url, kwargs = session.calls[0]
    assert method == "post"
    assert url == "https://graph.facebook.com/v21.0/17841/media"
    assert kwargs["data"]["media_type"] == "CAROUSEL"
    assert kwargs["data"]["children"] == "item1,item2"
    assert kwargs["data"]["caption"] == "caption text"


def test_publish_container_checks_quota_then_publishes_and_fetches_permalink():
    session = _FakeSession(
        [
            _quota_ok(),
            _Response(200, {"id": "post_1"}),
            _Response(200, {"permalink": "https://instagram.com/p/x"}),
        ]
    )
    client = MetaPhotoGraphClient(CREDS, session=session, site_root=SITE_ROOT)

    result = client.publish_container("container_1")

    assert result == {"post_id": "post_1", "permalink": "https://instagram.com/p/x"}
    # quota check happens before the publish call, every time.
    assert session.calls[0][1].endswith("content_publishing_limit")
    assert session.calls[1][1].endswith("media_publish")


def test_publish_container_refuses_when_quota_exhausted():
    session = _FakeSession([_quota_exhausted()])
    client = MetaPhotoGraphClient(CREDS, session=session, site_root=SITE_ROOT)

    with pytest.raises(PhotoPublishingLimitError, match="publishing limit reached"):
        client.publish_container("container_1")

    # never reaches media_publish once quota is exhausted.
    assert len(session.calls) == 1


def test_publish_container_permalink_lookup_failure_never_fails_the_publish():
    session = _FakeSession(
        [
            _quota_ok(),
            _Response(200, {"id": "post_1"}),
            _Response(400, {"error": {"message": "no permalink field", "code": 100}}),
        ]
    )
    client = MetaPhotoGraphClient(CREDS, session=session, site_root=SITE_ROOT)

    result = client.publish_container("container_1")

    assert result == {"post_id": "post_1", "permalink": None}


def test_auth_error_raises_photo_auth_blocked():
    session = _FakeSession([_Response(400, {"error": {"code": 190, "message": "token expired"}})])
    client = MetaPhotoGraphClient(CREDS, session=session, site_root=SITE_ROOT)

    with pytest.raises(PhotoAuthBlockedError):
        client.create_carousel_parent_container(("item1",), "caption")


def test_unrecognised_error_raises_photo_permanent():
    session = _FakeSession([_Response(400, {"error": {"code": 100, "message": "invalid field"}})])
    client = MetaPhotoGraphClient(CREDS, session=session, site_root=SITE_ROOT)

    with pytest.raises(PhotoPermanentError):
        client.create_carousel_parent_container(("item1",), "caption")


def test_site_root_unset_everywhere_raises_a_config_error_naming_the_key():
    from gomoveshift.config import ConfigError

    with pytest.raises(ConfigError, match="photos.site_root"):
        MetaPhotoGraphClient(CREDS, session=_FakeSession([]))
