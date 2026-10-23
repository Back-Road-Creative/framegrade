"""Tests for ``publishing/meta_graph.py``, the one Graph request helper the
Facebook, Instagram and photo clients share.

No network call ever happens here: every test injects a fake session.
"""

from __future__ import annotations

import pytest

from gomoveshift.publishing import meta_graph
from gomoveshift.publishing.facebook_adapter import (
    FacebookAuthBlockedError,
    FacebookPermanentError,
    FacebookReelsAdapter,
    FacebookRetryableError,
)
from gomoveshift.publishing.instagram_adapter import (
    InstagramAuthBlockedError,
    InstagramPermanentError,
    InstagramReelsAdapter,
    InstagramRetryableError,
)
from gomoveshift.publishing.meta_auth import DEFAULT_GRAPH_VERSION
from gomoveshift.publishing.photo_adapter import (
    PhotoAuthBlockedError,
    PhotoPermanentError,
    PhotoRetryableError,
)
from gomoveshift.publishing.photo_graph_client import MetaPhotoGraphClient


class _Response:
    def __init__(self, status_code, body=None, json_error=False):
        self.status_code = status_code
        self._body = body
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._body


class _Session:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        return self._response

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        return self._response


class _RejectedError(Exception):
    pass


def _hook(status_code, body):
    raise _RejectedError((status_code, body))


def test_base_url_uses_version_and_default():
    assert meta_graph.graph_base_url("v19.0") == "https://graph.facebook.com/v19.0"
    assert meta_graph.graph_base_url(None) == f"https://graph.facebook.com/{DEFAULT_GRAPH_VERSION}"


@pytest.mark.parametrize("method", ["post", "get"])
def test_success_returns_body_and_forwards_kwargs(method):
    session = _Session(_Response(200, {"id": "9"}))
    body = getattr(meta_graph, f"graph_{method}")(session, "https://x/y", _hook, data={"a": 1})
    assert body == {"id": "9"}
    assert session.calls == [(method, "https://x/y", {"data": {"a": 1}})]


@pytest.mark.parametrize("method", ["post", "get"])
def test_http_error_calls_hook_with_status_and_body(method):
    err = {"error": {"code": 100, "message": "bad"}}
    with pytest.raises(_RejectedError) as exc:
        getattr(meta_graph, f"graph_{method}")(_Session(_Response(400, err)), "u", _hook)
    assert exc.value.args[0] == (400, err)


@pytest.mark.parametrize("method", ["post", "get"])
def test_error_object_in_a_200_body_still_calls_hook(method):
    err = {"error": {"code": 100, "message": "bad"}}
    with pytest.raises(_RejectedError) as exc:
        getattr(meta_graph, f"graph_{method}")(_Session(_Response(200, err)), "u", _hook)
    assert exc.value.args[0] == (200, err)


@pytest.mark.parametrize("method", ["post", "get"])
def test_non_json_success_body_is_empty_dict(method):
    session = _Session(_Response(200, json_error=True))
    assert getattr(meta_graph, f"graph_{method}")(session, "u", _hook) == {}


def test_non_json_error_status_calls_hook_with_empty_body():
    with pytest.raises(_RejectedError) as exc:
        meta_graph.graph_post(_Session(_Response(502, json_error=True)), "u", _hook)
    assert exc.value.args[0] == (502, {})


_FB_CREDS = {"page_id": "1", "page_access_token": "tok"}
_IG_CREDS = {"ig_user_id": "2", "page_access_token": "tok"}

_PLATFORMS = [
    (
        lambda s: FacebookReelsAdapter(_FB_CREDS, session=s),
        FacebookAuthBlockedError,
        FacebookRetryableError,
        FacebookPermanentError,
    ),
    (
        lambda s: InstagramReelsAdapter(_IG_CREDS, session=s),
        InstagramAuthBlockedError,
        InstagramRetryableError,
        InstagramPermanentError,
    ),
    (
        lambda s: MetaPhotoGraphClient(_IG_CREDS, session=s, site_root="/site"),
        PhotoAuthBlockedError,
        PhotoRetryableError,
        PhotoPermanentError,
    ),
]


@pytest.mark.parametrize("make,auth,retryable,permanent", _PLATFORMS)
@pytest.mark.parametrize("verb", ["_post", "_get"])
def test_each_client_maps_graph_errors_to_its_own_error_type(
    make, auth, retryable, permanent, verb
):
    cases = [
        (200, {"error": {"code": 190, "message": "x"}}, auth),
        (400, {"error": {"code": 4, "message": "x"}}, retryable),
        (503, {"error": {"code": 1, "message": "x"}}, retryable),
        (400, {"error": {"code": 100, "message": "x"}}, permanent),
    ]
    for status, body, expected in cases:
        client = make(_Session(_Response(status, body)))
        with pytest.raises(expected):
            getattr(client, verb)("https://x/y")
