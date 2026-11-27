"""Tests for ``publishing/meta_graph.py``, the one Graph request helper the
Facebook, Instagram and photo clients share.

No network call ever happens here: every test injects a fake session, or
talks to a loopback server it starts itself.
"""

from __future__ import annotations

import io
import socket
import threading
import time

import pytest
import requests

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
    assert session.calls == [
        (method, "https://x/y", {"data": {"a": 1}, "timeout": meta_graph.GRAPH_TIMEOUT})
    ]


@pytest.mark.parametrize("method", ["post", "get"])
def test_every_request_carries_a_finite_timeout(method):
    """A Graph call with no timeout blocks on a dead connection forever: on
    2026-11-11 one hung `dispatch-queue --apply` 21 h, held the dispatch lock
    (no social post went out) and deferred every staging sync behind it."""
    session = _Session(_Response(200, {"id": "9"}))
    getattr(meta_graph, f"graph_{method}")(session, "u", _hook)
    connect, read = session.calls[0][2]["timeout"]
    assert 0 < connect <= 30 and 0 < read <= 300


@pytest.mark.parametrize("method", ["post", "get"])
def test_caller_timeout_wins(method):
    session = _Session(_Response(200, {"id": "9"}))
    getattr(meta_graph, f"graph_{method}")(session, "u", _hook, timeout=5)
    assert session.calls[0][2]["timeout"] == 5


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


@pytest.mark.parametrize("make", [p[0] for p in _PLATFORMS])
@pytest.mark.parametrize("verb", ["_post", "_get"])
def test_each_client_request_carries_a_timeout(make, verb):
    session = _Session(_Response(200, {"id": "9"}))
    getattr(make(session), verb)("https://x/y")
    assert session.calls[0][2]["timeout"] == meta_graph.GRAPH_TIMEOUT


# --- a byte upload's write is bounded per block, not as a whole ---------------


def _slow_reader_server(read_block: int, pause_s: float):
    """Loopback HTTP server that reads the request body slowly but steadily,
    like an uplink that keeps making progress, then answers ``{"ok": true}``.
    Returns ``(url, thread)``."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, read_block)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(10)

    def serve():
        with srv:
            conn, _ = srv.accept()
            with conn:
                conn.settimeout(10)
                head = b""
                while b"\r\n\r\n" not in head:
                    head += conn.recv(read_block)
                header, _, body = head.partition(b"\r\n\r\n")
                length = next(
                    int(line.split(b":")[1])
                    for line in header.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
                got = len(body)
                while got < length:
                    chunk = conn.recv(read_block)
                    if not chunk:
                        return
                    got += len(chunk)
                    time.sleep(pause_s)
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    b"Content-Length: 12\r\n\r\n" + b'{"ok": true}'
                )

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{srv.getsockname()[1]}/upload", thread


def test_a_byte_upload_that_keeps_progressing_is_not_cut_off_by_its_bound():
    # 22 and 27 Nov 2026: reel uploads failed with "The write operation timed
    # out". urllib3 sends a body under the CONNECT bound, and a bytes body
    # went out in one sendall whose deadline covered all of it, so a steady
    # upload that took longer than that bound was cut off. Here the body
    # takes well over a second to drain at ~6 MB/s against a 0.3 s bound.
    url, thread = _slow_reader_server(read_block=64 * 1024, pause_s=0.01)
    session = requests.Session()
    session.trust_env = False  # never through a proxy
    body = meta_graph.graph_post(
        session,
        url,
        _hook,
        data=b"x" * (8 * 1024 * 1024),
        headers={"offset": "0"},
        timeout=(0.3, 10),
    )
    thread.join(10)
    assert body == {"ok": True}


def test_a_byte_upload_still_sends_its_exact_length():
    # rupload reads Content-Length and checks it against ``file_size``: the
    # streamed body must not switch to chunked encoding.
    session = _Session(_Response(200, {"ok": True}))
    meta_graph.graph_post(session, "https://rupload.facebook.com/x", _hook, data=b"abc")
    sent = session.calls[0][2]["data"]
    assert sent.read() == b"abc"
    headers = requests.Request("POST", "https://x/", data=io.BytesIO(b"abc")).prepare().headers
    assert headers["Content-Length"] == "3" and "Transfer-Encoding" not in headers
