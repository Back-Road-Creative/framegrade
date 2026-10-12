"""Tests for color-analysis progress heartbeat.

Operability fix: the ffmpeg subprocess inside ``sample_frames`` blocks for
60-90 minutes on 4K HEVC sources without emitting any log line. Operator
cannot distinguish "still working" from "deadlocked" without /proc/PID/io
inspection. We add a wall-clock heartbeat thread that logs every 30s while
the subprocess is running, and exits cleanly when ffmpeg returns.

These tests are TDD-first: the module changes that make them pass are in
``gomoveshift/pipeline/color_analysis.py``.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _frame_bytes(n: int = 3) -> bytes:
    """Return n frames worth of 160x90 RGB24 raw bytes."""
    return b"\x80" * (160 * 90 * 3 * n)


class _FakePopen:
    """Minimal Popen stand-in: blocks in communicate() for ``sleep`` seconds.

    Mirrors the real subprocess.Popen API surface that sample_frames uses:
      - returncode (set after communicate returns)
      - communicate(timeout=...) -> (stdout_bytes, stderr_bytes)
      - kill() / poll()
    """

    def __init__(self, sleep_s: float, payload: bytes, returncode: int = 0):
        self._sleep_s = sleep_s
        self._payload = payload
        self._returncode_after = returncode
        self.returncode = None  # not yet finished
        self._killed = threading.Event()
        self.pid = 99999

    def communicate(self, timeout=None):
        # Honor the heartbeat thread by blocking on an event so .kill() works.
        finished = self._killed.wait(timeout=self._sleep_s)
        if finished:
            self.returncode = -9
            return (b"", b"killed")
        self.returncode = self._returncode_after
        return (self._payload, b"")

    def kill(self):
        self._killed.set()

    def poll(self):
        return self.returncode


def test_heartbeat_emits_at_least_two_lines_during_long_run(caplog, monkeypatch, tmp_path):
    """Long color analysis (~70s simulated) must log >=2 heartbeats at <=30s cadence."""
    from gomoveshift.pipeline import color_analysis

    # Speed-test by collapsing the heartbeat interval to 0.5s and simulating
    # a 2.5s subprocess. We then assert the heartbeat fired at least twice.
    monkeypatch.setattr(color_analysis, "_HEARTBEAT_INTERVAL_SEC", 0.5, raising=False)

    fake = _FakePopen(sleep_s=2.5, payload=_frame_bytes(2))

    with patch.object(color_analysis.subprocess, "Popen", return_value=fake):
        with caplog.at_level(logging.INFO, logger="gomoveshift.pipeline.color_analysis"):
            frames = color_analysis.sample_frames(
                video_path=tmp_path / "src.mp4",
                duration=600.0,  # well above 60s, so num_samples > 0
                interval=10.0,
                has_nvenc=False,
            )

    # Data path: still got frames back (shape preserved)
    assert len(frames) == 2
    assert frames[0].shape == (90, 160, 3)

    # Operability path: heartbeat fired at least twice
    heartbeat_lines = [
        rec.getMessage()
        for rec in caplog.records
        if "color analysis" in rec.getMessage().lower()
        and ("elapsed" in rec.getMessage().lower() or "still" in rec.getMessage().lower())
    ]
    assert len(heartbeat_lines) >= 2, (
        f"Expected >=2 heartbeat lines during 2.5s simulated run with 0.5s interval, "
        f"got {len(heartbeat_lines)}: {heartbeat_lines}"
    )


def test_heartbeat_thread_terminates_when_subprocess_exits(monkeypatch, tmp_path):
    """The daemon heartbeat thread must not outlive the ffmpeg subprocess."""
    from gomoveshift.pipeline import color_analysis

    monkeypatch.setattr(color_analysis, "_HEARTBEAT_INTERVAL_SEC", 0.2, raising=False)

    fake = _FakePopen(sleep_s=0.6, payload=_frame_bytes(1))

    threads_before = {t.ident for t in threading.enumerate()}

    with patch.object(color_analysis.subprocess, "Popen", return_value=fake):
        color_analysis.sample_frames(
            video_path=tmp_path / "src.mp4",
            duration=120.0,
            interval=10.0,
            has_nvenc=False,
        )

    # Allow a brief grace window for the heartbeat thread to observe the stop event
    deadline = time.monotonic() + 2.0
    leaked = []
    while time.monotonic() < deadline:
        leaked = [
            t
            for t in threading.enumerate()
            if t.ident not in threads_before
            and t.is_alive()
            and "heartbeat" in (t.name or "").lower()
        ]
        if not leaked:
            break
        time.sleep(0.05)

    assert not leaked, f"Heartbeat thread leaked: {[t.name for t in leaked]}"


def test_short_run_still_returns_correct_frames(monkeypatch, tmp_path):
    """A fast subprocess (< heartbeat interval) must still return frames unchanged.

    Regression guard: the heartbeat wrapping must not corrupt or drop the
    rawvideo bytes coming out of ffmpeg.
    """
    from gomoveshift.pipeline import color_analysis

    # 3 frames, returns immediately
    fake = _FakePopen(sleep_s=0.0, payload=_frame_bytes(3))

    with patch.object(color_analysis.subprocess, "Popen", return_value=fake):
        frames = color_analysis.sample_frames(
            video_path=tmp_path / "src.mp4",
            duration=300.0,
            interval=10.0,
            has_nvenc=False,
        )

    assert len(frames) == 3
    for f in frames:
        assert f.shape == (90, 160, 3)
        assert f.dtype == np.uint8
