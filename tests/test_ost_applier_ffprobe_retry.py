"""ost_applier._probe_video_info retries a TRANSIENT ffprobe timeout.

A valid LF montage intermediate probed while a heavy concurrent encode pins the
CPU can exceed the 30s ffprobe window and raise TimeoutExpired — that is
transient load, NOT corruption. Previously this aborted the whole OST burn
mid-day. The probe now retries ONLY on
TimeoutExpired; every non-timeout failure stays fatal on the first attempt.
"""

import subprocess

import pytest

from gomoveshift.pipeline import ost_applier
from gomoveshift.pipeline.exceptions import ProbeFailedError

_GOOD = subprocess.CompletedProcess(
    ["ffprobe"],
    0,
    stdout=(
        b'{"streams":[{"width":3840,"height":2160,"r_frame_rate":"30/1"}],'
        b'"format":{"duration":"100.0","bit_rate":"1000000"}}'
    ),
    stderr=b"",
)


@pytest.fixture
def _video(tmp_path):
    p = tmp_path / "lf_01.mp4"
    p.write_bytes(b"\x00")
    return p


def test_retries_transient_timeout_then_succeeds(monkeypatch, _video):
    calls = {"n": 0}

    def fake_run(cmd, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise subprocess.TimeoutExpired(cmd, 30)
        return _GOOD

    monkeypatch.setattr(ost_applier.subprocess, "run", fake_run)
    monkeypatch.setattr(ost_applier.time, "sleep", lambda *a, **k: None)

    info = ost_applier._probe_video_info(_video)
    assert calls["n"] == 3  # two retries, succeeded on the third
    assert info["width"] == 3840 and info["height"] == 2160


def test_persistent_timeout_raises_after_retries(monkeypatch, _video):
    calls = {"n": 0}

    def always_timeout(cmd, **kw):
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd, 30)

    monkeypatch.setattr(ost_applier.subprocess, "run", always_timeout)
    monkeypatch.setattr(ost_applier.time, "sleep", lambda *a, **k: None)

    from gomoveshift.pipeline.validation.ffprobe_client import FFPROBE_TIMEOUT_RETRIES

    with pytest.raises(ProbeFailedError):
        ost_applier._probe_video_info(_video)
    assert calls["n"] == FFPROBE_TIMEOUT_RETRIES  # exhausted all attempts


def test_non_timeout_error_is_fatal_immediately(monkeypatch, _video):
    calls = {"n": 0}

    def os_error(cmd, **kw):
        calls["n"] += 1
        raise OSError("ffprobe not found")

    monkeypatch.setattr(ost_applier.subprocess, "run", os_error)

    with pytest.raises(ProbeFailedError):
        ost_applier._probe_video_info(_video)
    assert calls["n"] == 1  # NOT retried
