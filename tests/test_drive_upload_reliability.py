"""Tests for Drive upload reliability: stall detection and 429 retry."""

import ssl
import socket
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.uploader import _SSL_RETRY_DELAYS, PipelineUploader


def _make_uploader(tmp_path):
    """Return a PipelineUploader in legacy mode so no SOPS decrypt is needed."""
    creds = tmp_path / "credentials.json"
    token = tmp_path / "token.json"
    creds.write_text('{"installed":{}}')
    token.write_text("{}")
    return PipelineUploader(credentials_path=creds, token_path=token)


class TestUploadStallDetection:
    """_upload_file must raise if no progress for STALL_TIMEOUT seconds."""

    def test_raises_on_stall(self, tmp_path):
        """next_chunk() making no progress for > 120s must raise RuntimeError."""
        uploader = _make_uploader(tmp_path)

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video content")

        mock_drive = MagicMock()
        mock_request = MagicMock()
        # next_chunk() always returns no progress and no response
        mock_request.next_chunk.return_value = (None, None)
        mock_drive.files.return_value.create.return_value = mock_request

        frozen_time = [1000.0]

        def advancing_time():
            frozen_time[0] += 130  # 130s jump — past 120s stall threshold
            return frozen_time[0]

        with patch("gomoveshift.pipeline.uploader.time") as mock_time:
            mock_time.time.side_effect = advancing_time
            mock_time.sleep = MagicMock()
            with patch.object(uploader, "_refresh_credentials_if_needed"):
                with pytest.raises(RuntimeError, match="[Ss]tall"):
                    uploader._upload_file(
                        drive=mock_drive,
                        file_path=video,
                        folder_id="fake_folder_id",
                        make_shareable=False,
                    )

    def test_no_raise_when_upload_completes(self, tmp_path):
        """A normally completing upload must not raise."""
        uploader = _make_uploader(tmp_path)

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video content")

        mock_drive = MagicMock()
        mock_request = MagicMock()
        progress_status = MagicMock()
        progress_status.progress.return_value = 0.5
        final_response = {"id": "file_abc123"}
        mock_request.next_chunk.side_effect = [
            (progress_status, None),
            (None, final_response),
        ]
        mock_drive.files.return_value.create.return_value = mock_request

        with patch.object(uploader, "_refresh_credentials_if_needed"):
            result = uploader._upload_file(
                drive=mock_drive,
                file_path=video,
                folder_id="fake_folder_id",
                make_shareable=False,
            )
        assert result is not None


class TestDriveApiRateLimit:
    """Drive API .execute() calls must retry on HTTP 429 with backoff."""

    def test_folder_list_retries_on_429(self, tmp_path):
        """_find_or_create_folder must retry list().execute() once on 429."""
        from googleapiclient.errors import HttpError

        uploader = _make_uploader(tmp_path)
        mock_drive = MagicMock()

        resp_429 = MagicMock()
        resp_429.status = 429
        error_429 = HttpError(resp=resp_429, content=b"Rate Limit Exceeded")

        # First call 429; second succeeds with a folder
        mock_drive.files.return_value.list.return_value.execute.side_effect = [
            error_429,
            {"files": [{"id": "folder_id_123", "name": "TestFolder"}]},
        ]

        sleep_calls = []
        with patch("gomoveshift.pipeline.uploader.time") as mock_time:
            mock_time.sleep.side_effect = lambda s: sleep_calls.append(s)
            mock_time.time.return_value = 0.0
            result = uploader._find_or_create_folder(mock_drive, "TestFolder")

        assert result == "folder_id_123"
        assert len(sleep_calls) >= 1, "Should sleep at least once after 429"

    def test_non_429_http_error_propagates_immediately(self, tmp_path):
        """Non-429 HTTP errors (403, 500) must not be retried — propagate immediately."""
        from googleapiclient.errors import HttpError

        uploader = _make_uploader(tmp_path)
        mock_drive = MagicMock()

        resp_403 = MagicMock()
        resp_403.status = 403
        error_403 = HttpError(resp=resp_403, content=b"Permission Denied")
        mock_drive.files.return_value.list.return_value.execute.side_effect = error_403

        with pytest.raises(HttpError):
            uploader._find_or_create_folder(mock_drive, "TestFolder")

    def test_exhausting_retries_raises(self, tmp_path):
        """Persistent 429 after max retries must raise RuntimeError."""
        from googleapiclient.errors import HttpError

        uploader = _make_uploader(tmp_path)
        mock_drive = MagicMock()

        resp_429 = MagicMock()
        resp_429.status = 429
        error_429 = HttpError(resp=resp_429, content=b"Rate Limit Exceeded")
        mock_drive.files.return_value.list.return_value.execute.side_effect = error_429

        with patch("gomoveshift.pipeline.uploader.time") as mock_time:
            mock_time.sleep = MagicMock()
            mock_time.time.return_value = 0.0
            with pytest.raises((RuntimeError, HttpError)):
                uploader._find_or_create_folder(mock_drive, "TestFolder")


class TestSSLRetry:
    """upload_shorts retries on SSL/connection errors and uses sequential mode."""

    def test_ssl_error_retries_and_succeeds(self, tmp_path):
        """First call raises SSLError, second call succeeds — upload completes."""
        uploader = _make_uploader(tmp_path)
        video = tmp_path / "short.mp4"
        video.write_bytes(b"data")

        call_count = {"n": 0}

        def mock_upload(drive, file_path, folder_id, make_shareable=True):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ssl.SSLError("DECRYPTION_FAILED_OR_BAD_RECORD_MAC")
            return "https://drive.google.com/file/d/abc123/view?usp=sharing"

        uploader._upload_file = mock_upload
        uploader._build_drive_service = lambda: MagicMock()
        sleeps = []
        with patch("gomoveshift.pipeline.uploader.time.sleep", side_effect=sleeps.append):
            result = uploader.upload_shorts([str(video)], folder_id="folder1")
        assert len(result) == 1
        assert "abc123" in result[0]
        assert call_count["n"] == 2
        # one jittered wait, [0.5, 1.5) x the first delay, between the two attempts
        assert len(sleeps) == 1
        assert _SSL_RETRY_DELAYS[0] * 0.5 <= sleeps[0] < _SSL_RETRY_DELAYS[0] * 1.5

    def test_exhausted_ssl_retries_raises(self, tmp_path):
        """Three SSL failures → DriveUploadError raised."""
        uploader = _make_uploader(tmp_path)
        video = tmp_path / "short.mp4"
        video.write_bytes(b"data")

        def always_fail(drive, file_path, folder_id, make_shareable=True):
            raise ssl.SSLError("CIPHER_OPERATION_FAILED")

        uploader._upload_file = always_fail
        uploader._build_drive_service = lambda: MagicMock()
        from gomoveshift.pipeline.uploader import DriveUploadError

        sleeps = []
        with patch("gomoveshift.pipeline.uploader.time.sleep", side_effect=sleeps.append):
            with pytest.raises(DriveUploadError):
                uploader.upload_shorts([str(video)], folder_id="folder1")
        # a wait only BETWEEN attempts: none after the last one, which can only raise
        assert len(sleeps) == len(_SSL_RETRY_DELAYS) - 1
        for slept, base in zip(sleeps, _SSL_RETRY_DELAYS):
            assert base * 0.5 <= slept < base * 1.5

    def test_sequential_upload_by_default(self, tmp_path):
        """upload_shorts runs files sequentially (no ThreadPoolExecutor) by default."""
        uploader = _make_uploader(tmp_path)
        order = []
        videos = []
        for i in range(3):
            v = tmp_path / f"short_{i}.mp4"
            v.write_bytes(b"x")
            videos.append(str(v))

        def mock_upload(drive, file_path, folder_id, make_shareable=True):
            order.append(str(file_path))
            return "https://drive.google.com/file/d/id/view"

        uploader._upload_file = mock_upload
        uploader._build_drive_service = lambda: MagicMock()
        result = uploader.upload_shorts(videos, folder_id="folder1")
        assert len(result) == 3
        # Order must match input order (sequential)
        assert order == videos
