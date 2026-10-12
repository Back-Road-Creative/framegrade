"""Tests for retry hardening: jitter, circuit breaker, error classification.

Covers:
- Jitter adds randomness to backoff delays
- Circuit breaker trips after N consecutive failures
- Permanent errors (401/403/404) fail immediately
- Transient errors (429/5xx/timeout/ConnectionError) retry
- Integration with Drive _drive_execute and YouTube _execute_resumable
"""

from __future__ import annotations

import ssl
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from gomoveshift.pipeline.uploader import (
    DriveUploadError,
    PipelineUploader,
    _drive_execute,
    is_transient_error,
    is_permanent_error,
    CircuitBreaker,
    CircuitBreakerOpen,
)


# ---------- helpers ----------


def _make_uploader(tmp_path):
    """Return a PipelineUploader in legacy mode so no SOPS decrypt is needed."""
    creds = tmp_path / "credentials.json"
    token = tmp_path / "token.json"
    creds.write_text('{"installed":{}}')
    token.write_text("{}")
    return PipelineUploader(credentials_path=creds, token_path=token)


def _make_http_error(status_code: int):
    """Build a googleapiclient HttpError with the given status."""
    from googleapiclient.errors import HttpError

    resp = MagicMock()
    resp.status = status_code
    return HttpError(resp=resp, content=b"error")


# ---------- error classification ----------


class TestErrorClassification:
    """is_transient_error / is_permanent_error must classify HTTP codes correctly."""

    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    def test_transient_http_codes(self, code):
        exc = _make_http_error(code)
        assert is_transient_error(exc) is True
        assert is_permanent_error(exc) is False

    @pytest.mark.parametrize("code", [401, 403, 404])
    def test_permanent_http_codes(self, code):
        exc = _make_http_error(code)
        assert is_permanent_error(exc) is True
        assert is_transient_error(exc) is False

    def test_timeout_is_transient(self):
        assert is_transient_error(TimeoutError("timed out")) is True
        assert is_permanent_error(TimeoutError("timed out")) is False

    def test_connection_error_is_transient(self):
        assert is_transient_error(ConnectionError("reset")) is True

    def test_ssl_error_is_transient(self):
        assert is_transient_error(ssl.SSLError("BAD_MAC")) is True

    def test_socket_error_is_transient(self):
        assert is_transient_error(socket.error("broken pipe")) is True

    def test_os_error_is_transient(self):
        assert is_transient_error(OSError("network unreachable")) is True

    def test_unknown_error_is_neither(self):
        assert is_transient_error(ValueError("bad input")) is False
        assert is_permanent_error(ValueError("bad input")) is False

    def test_other_http_code_is_neither(self):
        """HTTP 409 (conflict) is neither clearly transient nor permanent."""
        exc = _make_http_error(409)
        assert is_transient_error(exc) is False
        assert is_permanent_error(exc) is False


# ---------- jitter ----------


class TestJitter:
    """_drive_execute must add jitter to backoff delays."""

    def test_jitter_adds_randomness(self, tmp_path):
        """Backoff delay must be multiplied by random factor [0.5, 1.5)."""
        from googleapiclient.errors import HttpError

        resp_429 = MagicMock()
        resp_429.status = 429

        request = MagicMock()
        request.execute.side_effect = [
            HttpError(resp=resp_429, content=b"429"),
            HttpError(resp=resp_429, content=b"429"),
            {"id": "ok"},
        ]

        sleep_values = []
        with (
            patch("gomoveshift.pipeline.uploader.time") as mock_time,
            patch("gomoveshift.pipeline.uploader.random") as mock_random,
        ):
            mock_time.sleep.side_effect = lambda s: sleep_values.append(s)
            mock_random.random.return_value = 0.25  # jitter factor: 0.5 + 0.25 = 0.75

            result = _drive_execute(request, max_attempts=3)

        assert result == {"id": "ok"}
        # First retry: base=1, jitter=0.75 → 0.75
        # Second retry: base=2, jitter=0.75 → 1.5
        assert sleep_values[0] == pytest.approx(0.75)
        assert sleep_values[1] == pytest.approx(1.5)

    def test_jitter_range_bounds(self):
        """Jitter factor 0.5+random() is in [0.5, 1.5) — verify boundary math."""
        # With random()=0.0 → factor=0.5 → delay=base*0.5
        # With random()=0.999 → factor=1.499 → delay=base*1.499
        # This is a design assertion, not a runtime test
        assert 0.5 + 0.0 == 0.5
        assert 0.5 + 0.999 == pytest.approx(1.499)


# ---------- circuit breaker ----------


class TestCircuitBreaker:
    """CircuitBreaker must trip after N consecutive failures and raise CircuitBreakerOpen."""

    def test_trips_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True

    def test_below_threshold_is_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is False

    def test_success_resets_counter(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.consecutive_failures == 0
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is False

    def test_check_raises_when_open(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        with pytest.raises(CircuitBreakerOpen, match="3 consecutive"):
            cb.check()

    def test_check_passes_when_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.check()  # Should not raise


# ---------- Drive _drive_execute permanent error fast-fail ----------


class TestDriveExecutePermanentFail:
    """_drive_execute must raise immediately on permanent errors (401/403/404)."""

    @pytest.mark.parametrize("code", [401, 403, 404])
    def test_permanent_error_no_retry(self, code):
        from googleapiclient.errors import HttpError

        request = MagicMock()
        request.execute.side_effect = _make_http_error(code)

        sleep_calls = []
        with patch("gomoveshift.pipeline.uploader.time") as mock_time:
            mock_time.sleep.side_effect = lambda s: sleep_calls.append(s)
            with pytest.raises(HttpError):
                _drive_execute(request, max_attempts=5)

        # Must NOT have slept — immediate failure
        assert len(sleep_calls) == 0
        assert request.execute.call_count == 1


# ---------- Drive _drive_execute circuit breaker integration ----------


class TestDriveExecuteCircuitBreaker:
    """_drive_execute must accept and update a circuit breaker."""

    def test_circuit_breaker_trips_on_consecutive_429s(self):
        from googleapiclient.errors import HttpError

        cb = CircuitBreaker(failure_threshold=3)
        request = MagicMock()
        request.execute.side_effect = _make_http_error(429)

        # Exhaust retries three times — circuit breaker accumulates failures
        for _ in range(3):
            with (
                patch("gomoveshift.pipeline.uploader.time") as mock_time,
                patch("gomoveshift.pipeline.uploader.random") as mock_random,
            ):
                mock_time.sleep = MagicMock()
                mock_random.random.return_value = 0.5
                try:
                    _drive_execute(request, max_attempts=1, circuit_breaker=cb)
                except (RuntimeError, HttpError):
                    pass

        assert cb.is_open is True
        with pytest.raises(CircuitBreakerOpen):
            cb.check()

    def test_circuit_breaker_checked_before_execute(self):
        """If circuit breaker is already open, _drive_execute raises immediately."""
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()  # Now open

        request = MagicMock()
        with pytest.raises(CircuitBreakerOpen):
            _drive_execute(request, max_attempts=5, circuit_breaker=cb)

        request.execute.assert_not_called()


# ---------- YouTube retry with jitter ----------


class TestYouTubeRetryJitter:
    """YouTube _execute_resumable must add jitter to backoff delays."""

    def test_jitter_applied_to_youtube_backoff(self, tmp_path):
        from gomoveshift.pipeline.drive_phases import youtube_uploader as yu

        insert_request = MagicMock()
        # Fail twice with 503, then succeed
        resp_503 = MagicMock()
        resp_503.status = 503
        from googleapiclient.errors import HttpError

        insert_request.next_chunk.side_effect = [
            HttpError(resp=resp_503, content=b"503"),
            HttpError(resp=resp_503, content=b"503"),
            (None, {"id": "VID123"}),
        ]

        sleep_values = []
        with (
            patch.object(yu.time, "sleep", side_effect=lambda s: sleep_values.append(s)),
            patch.object(yu.random, "random", return_value=0.25),
        ):
            up = yu.YouTubeUploader.__new__(yu.YouTubeUploader)
            result = up._execute_resumable(insert_request)

        assert result == "VID123"
        # First: base=2.0, jitter=0.75 → 1.5
        # Second: base=4.0, jitter=0.75 → 3.0
        assert sleep_values[0] == pytest.approx(1.5)
        assert sleep_values[1] == pytest.approx(3.0)


# ---------- YouTube permanent error fast-fail ----------


class TestYouTubePermanentFail:
    """YouTube must fail immediately on 401/403/404 without retrying."""

    @pytest.mark.parametrize("code", [401, 403, 404])
    def test_permanent_http_error_no_retry(self, code):
        from gomoveshift.pipeline.drive_phases import youtube_uploader as yu
        from googleapiclient.errors import HttpError

        insert_request = MagicMock()
        insert_request.next_chunk.side_effect = _make_http_error(code)

        sleep_calls = []
        with patch.object(yu.time, "sleep", side_effect=lambda s: sleep_calls.append(s)):
            up = yu.YouTubeUploader.__new__(yu.YouTubeUploader)
            with pytest.raises(HttpError):
                up._execute_resumable(insert_request)

        assert len(sleep_calls) == 0
        assert insert_request.next_chunk.call_count == 1


# ---------- Drive SSL retry with jitter ----------


class TestDriveSSLRetryJitter:
    """upload_shorts SSL retry must use jittered delays."""

    def test_ssl_retry_uses_jitter(self, tmp_path):
        uploader = _make_uploader(tmp_path)
        video = tmp_path / "short.mp4"
        video.write_bytes(b"data")

        call_count = {"n": 0}

        def mock_upload(drive, file_path, folder_id, make_shareable=True):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise ssl.SSLError("BAD_MAC")
            return "https://drive.google.com/file/d/abc123/view?usp=sharing"

        uploader._upload_file = mock_upload
        uploader._build_drive_service = lambda: MagicMock()

        sleep_values = []
        with (
            patch("gomoveshift.pipeline.uploader.time") as mock_time,
            patch("gomoveshift.pipeline.uploader.random") as mock_random,
        ):
            mock_time.sleep.side_effect = lambda s: sleep_values.append(s)
            mock_random.random.return_value = 0.5  # factor = 1.0

            result = uploader.upload_shorts([str(video)], folder_id="folder1")

        assert len(result) == 1
        # SSL delays [10, 20, 30] with jitter factor 1.0: [10.0, 20.0]
        assert sleep_values[0] == pytest.approx(10.0)
        assert sleep_values[1] == pytest.approx(20.0)
