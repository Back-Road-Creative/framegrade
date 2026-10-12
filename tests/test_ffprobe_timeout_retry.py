"""FFprobeClient retries transient timeouts, fails fast on corruption.

A valid file probed while a heavy concurrent encode pins the CPU can exceed
the 30s ffprobe timeout and raise ``subprocess.TimeoutExpired``. Upstream
treats that as corruption/fatal — a false failure that can abort a long run.
The client now retries a timeout a few times (transient load, not corruption)
before raising, while NON-timeout errors stay fatal immediately.
"""

import subprocess
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.validation.ffprobe_client import (
    FFPROBE_TIMEOUT_RETRIES,
    FFprobeClient,
    FFprobeError,
)


def _ok_result(stdout: str = '{"format": {"duration": "12.5"}}'):
    return MagicMock(returncode=0, stdout=stdout, stderr="")


def _timeout(*_args, **_kwargs):
    raise subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=30)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # Keep the backoff from slowing the suite; the retry path still exercises.
    monkeypatch.setattr(
        "gomoveshift.pipeline.validation.ffprobe_client.time.sleep",
        lambda _s: None,
    )


def test_timeout_twice_then_success_returns_value():
    """Times out twice, succeeds on the third attempt → returns the value."""
    sub = MagicMock()
    sub.run.side_effect = [
        subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=30),
        subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=30),
        _ok_result(),
    ]

    client = FFprobeClient(sub)
    data = client.probe("/some/file.mp4")

    assert data == {"format": {"duration": "12.5"}}
    assert sub.run.call_count == 3  # retried, did not give up on first timeout


def test_timeout_every_attempt_raises_after_n_retries():
    """Times out on every attempt → raises TimeoutExpired after N attempts."""
    sub = MagicMock()
    sub.run.side_effect = _timeout

    client = FFprobeClient(sub)
    with pytest.raises(subprocess.TimeoutExpired) as exc_info:
        client.probe("/some/file.mp4")

    # Raised only after exhausting all retries, not on the first timeout.
    assert sub.run.call_count == FFPROBE_TIMEOUT_RETRIES
    # Message makes clear it was timeout-after-retries, not corruption.
    assert "not corruption" in (exc_info.value.output or "")


def test_non_timeout_error_raises_immediately_without_retry():
    """A non-zero return code (genuine corruption) fails fast, no retry."""
    sub = MagicMock()
    sub.run.return_value = MagicMock(returncode=1, stdout="", stderr="moov atom not found")

    client = FFprobeClient(sub)
    with pytest.raises(FFprobeError):
        client.probe("/some/file.mp4")

    # Corruption is never retried — exactly one subprocess invocation.
    assert sub.run.call_count == 1


def test_subprocess_exception_is_not_retried():
    """A non-timeout exception from subprocess.run propagates immediately."""
    sub = MagicMock()
    sub.run.side_effect = OSError("ffprobe not found")

    client = FFprobeClient(sub)
    with pytest.raises(OSError):
        client.probe("/some/file.mp4")

    assert sub.run.call_count == 1
