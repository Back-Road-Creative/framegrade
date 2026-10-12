"""Tests for ancillary-call retry and chunk-timeout thread safety.

Retry: ``set_thumbnail`` / ``update_privacy`` ``.execute()`` calls had no retry,
unlike the chunked video upload. A transient 5xx after the video published left
the video with no thumbnail / wrong privacy. The fix routes both ancillary
calls through ``_execute_with_retry`` reusing the chunk-path backoff policy.

Timeout: ``_next_chunk_with_timeout`` used ``with ThreadPoolExecutor(...) as pool``,
whose ``__exit__`` calls ``shutdown(wait=True)`` — on TimeoutError that blocks
on the orphaned ``next_chunk()`` worker. The fix uses an explicit pool and calls
``shutdown(wait=False, cancel_futures=True)`` on timeout before raising, so the
caller is freed promptly.

Fully mocks googleapiclient — no network.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_http_error(status_code: int, reason: str | None = None, message: str = "error"):
    """Build a googleapiclient HttpError with the given status code."""
    from googleapiclient.errors import HttpError

    class _Resp:
        def __init__(self, status):
            self.status = status
            self.reason = "Error"

    errors = [{"message": reason or message, "reason": reason}]
    body = json.dumps(
        {"error": {"code": status_code, "message": message, "errors": errors}}
    ).encode("utf-8")
    return HttpError(_Resp(status_code), body, uri="fake://")


@pytest.fixture
def fake_creds_dict():
    return {
        "token": "ya29.fake",
        "refresh_token": "1//fake",
        "client_id": "fake.apps.googleusercontent.com",
        "client_secret": "fakesecret",  # pragma: allowlist secret
        "token_uri": "https://oauth2.googleapis.com/token",
        "scopes": ["https://www.googleapis.com/auth/youtube.upload"],
    }


def _make_uploader(mock_service, fake_creds_dict):
    from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

    with (
        patch.object(yu, "load_youtube_credentials", return_value=fake_creds_dict),
        patch.object(
            yu, "_build_credentials_obj", return_value=MagicMock(valid=True, expired=False)
        ),
        patch.object(yu, "build", return_value=mock_service),
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
    ):
        up = yu.YouTubeUploader(credentials_dir=Path("/tmp/fake-creds"))
        assert up._build_service() is mock_service
    return up, yu


# ---------------------------------------------------------------- retry


def test_set_thumbnail_retries_through_two_503s(fake_creds_dict, tmp_path):
    """503 twice then success → ancillary thumbnail call retries to success.

    Mirrors the chunk-path 5xx backoff: a transient 5xx after the video is
    already published must not leave the video without a custom thumbnail.
    """
    svc = MagicMock()
    execute = svc.thumbnails.return_value.set.return_value.execute
    execute.side_effect = [
        _make_http_error(503, message="Service Unavailable"),
        _make_http_error(503, message="Service Unavailable"),
        {},
    ]
    up, yu = _make_uploader(svc, fake_creds_dict)
    thumb = tmp_path / "thumb.jpg"
    thumb.write_bytes(b"\xff\xd8\xff")

    with (
        patch.object(yu, "MediaFileUpload", return_value=MagicMock()),
        patch.object(yu.time, "sleep"),
    ):
        up.set_thumbnail("VID1", thumb)

    assert execute.call_count == 3


def test_update_privacy_retries_through_two_503s(fake_creds_dict):
    """503 twice then success → ancillary privacy update retries to success.

    A transient 5xx must not leave a scheduled video stuck at the wrong
    privacy status after publish.
    """
    svc = MagicMock()
    execute = svc.videos.return_value.update.return_value.execute
    execute.side_effect = [
        _make_http_error(503, message="Service Unavailable"),
        _make_http_error(503, message="Service Unavailable"),
        {},
    ]
    up, yu = _make_uploader(svc, fake_creds_dict)

    with patch.object(yu.time, "sleep"):
        up.update_privacy("VID1", privacy="unlisted")

    assert execute.call_count == 3


def test_ancillary_permanent_error_not_retried(fake_creds_dict):
    """A permanent 404 on an ancillary call must NOT retry — raised verbatim."""
    svc = MagicMock()
    execute = svc.videos.return_value.update.return_value.execute
    execute.side_effect = _make_http_error(404, message="Not Found")
    up, yu = _make_uploader(svc, fake_creds_dict)

    from googleapiclient.errors import HttpError

    with patch.object(yu.time, "sleep"):
        with pytest.raises(HttpError):
            up.update_privacy("VID1", privacy="unlisted")

    assert execute.call_count == 1


# ----------------------------------------------------- timeout thread


def test_chunk_timeout_does_not_block_on_shutdown(fake_creds_dict):
    """On timeout the helper raises promptly, not after the worker finishes.

    Regression for the ``with ThreadPoolExecutor`` auto ``shutdown(wait=True)``
    that blocked the caller until the orphaned ``next_chunk()`` worker returned.
    The worker sleeps far longer than the timeout; the helper must raise in
    well under that worker duration.
    """
    from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

    worker_sleep = 5.0  # far longer than the (patched) timeout

    request = MagicMock()
    request.next_chunk.side_effect = lambda: time.sleep(worker_sleep)

    start = time.monotonic()
    with patch.object(yu, "CHUNK_TIMEOUT_SEC", 0.2):
        with pytest.raises(TimeoutError):
            yu.YouTubeUploader._next_chunk_with_timeout(request)
    elapsed = time.monotonic() - start

    # Must return on the timeout (~0.2s), NOT block for the full worker sleep.
    assert elapsed < worker_sleep / 2, (
        f"timeout helper blocked {elapsed:.2f}s — auto shutdown(wait=True) "
        f"is waiting on the orphaned worker (worker_sleep={worker_sleep}s)"
    )
