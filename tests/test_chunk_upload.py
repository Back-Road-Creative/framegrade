"""Tests for ``pipeline/_chunk_upload.py``: what the YouTube and Drive
resumable uploads share. Every socket here is loopback, opened by the test."""

from __future__ import annotations

import socket
import threading
import time
from unittest.mock import patch

import httplib2
import pytest
from googleapiclient.http import DEFAULT_HTTP_TIMEOUT_SEC

from gomoveshift.pipeline import _chunk_upload
from gomoveshift.pipeline._chunk_upload import (
    UPLOAD_CHUNKSIZE,
    ChunkStillRunningError,
    next_chunk_with_timeout,
)


def _server_that_never_reads():
    """Accepts one connection and reads nothing, like a dead uplink. Returns
    ``(url, release)``; set ``release`` to let the server close."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    release = threading.Event()

    def serve():
        with srv:
            conn, _ = srv.accept()
            release.wait(30)
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    return f"http://127.0.0.1:{srv.getsockname()[1]}/upload", release


class _Request:
    """A resumable request whose chunk is a real httplib2 PUT; counts the
    chunks still sending."""

    def __init__(self, url):
        self.http = httplib2.Http(timeout=30, proxy_info=None)
        self.url, self.running, self.lock = url, 0, threading.Lock()

    def next_chunk(self):
        with self.lock:
            self.running += 1
        try:
            self.http.request(self.url, "PUT", body=b"x" * (16 * 1024 * 1024))
        finally:
            with self.lock:
                self.running -= 1
        return None, {"id": "v"}


def test_a_timed_out_chunk_is_stopped_before_the_caller_can_retry():
    # The worker was abandoned on timeout and kept sending, so the caller's
    # retry sent the same chunk again beside it on the same connection.
    url, release = _server_that_never_reads()
    request = _Request(url)
    try:
        with pytest.raises(TimeoutError) as exc_info:
            next_chunk_with_timeout(request, 0.3, "YouTube")
        assert request.running == 0, "the timed-out chunk is still sending"
        assert not isinstance(exc_info.value, ChunkStillRunningError)
    finally:
        release.set()


def test_a_chunk_that_will_not_stop_is_never_retried_beside(monkeypatch):
    monkeypatch.setattr(_chunk_upload, "ABORT_GRACE_SEC", 0.2)
    gate = threading.Event()

    class _Stuck:
        http = None

        def next_chunk(self):
            gate.wait(10)
            return None, {"id": "v"}

    try:
        with pytest.raises(ChunkStillRunningError):
            next_chunk_with_timeout(_Stuck(), 0.1, "YouTube")
    finally:
        gate.set()


def test_youtube_does_not_retry_a_chunk_that_is_still_running():
    from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

    calls = []

    def still_running(request):
        calls.append(request)
        raise ChunkStillRunningError("still running")

    up = yu.YouTubeUploader.__new__(yu.YouTubeUploader)
    with (
        patch.object(yu.YouTubeUploader, "_next_chunk_with_timeout", staticmethod(still_running)),
        patch.object(yu.time, "sleep"),
        pytest.raises(ChunkStillRunningError),
    ):
        up._execute_resumable(object())
    assert len(calls) == 1


def test_a_chunk_can_be_sent_within_the_socket_timeout_on_a_slow_uplink():
    # googleapiclient's socket timeout bounds a chunk's whole send: a 256 MB
    # chunk needed ~36 Mbit/s or it timed out, on every retry.
    mbit_per_s = UPLOAD_CHUNKSIZE * 8 / 1e6 / DEFAULT_HTTP_TIMEOUT_SEC
    assert mbit_per_s <= 2
    assert UPLOAD_CHUNKSIZE % (256 * 1024) == 0


def test_a_chunk_body_send_is_bounded_as_a_whole_by_the_socket_timeout():
    # The fact the chunk size rests on: steady progress does not reset it.
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    body = 8 * 1024 * 1024

    def serve():
        with srv:
            conn, _ = srv.accept()
            with conn:
                conn.settimeout(10)
                try:
                    while conn.recv(65536):
                        time.sleep(0.01)  # ~6 MB/s, never stalled
                except OSError:
                    pass

    threading.Thread(target=serve, daemon=True).start()
    http = httplib2.Http(timeout=0.3, proxy_info=None)
    with pytest.raises(TimeoutError):
        http.request(f"http://127.0.0.1:{srv.getsockname()[1]}/", "PUT", body=b"x" * body)
