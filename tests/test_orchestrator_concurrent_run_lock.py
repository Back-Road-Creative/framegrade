"""Tests for the orchestrator-entry concurrent-run lock.

Two operators (or two tmux sessions) pointing at the same project output
dir would otherwise race on ``pipeline_state.json`` writes and overwrite
each other's day-master encodes. ``_acquire_run_lock`` takes an
``fcntl.flock(LOCK_EX | LOCK_NB)`` on a sidecar file under
``stages_base/pipeline_state.lock``; on contention it probes the holder
PID via ``os.kill(pid, 0)`` and reclaims if the holder is dead, else
raises :class:`ConcurrentRunError` with the holder's PID + start time.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from gomoveshift.pipeline.drive_orchestrator import (
    ConcurrentRunError,
    _acquire_run_lock,
    _pid_is_alive,
    _release_run_lock,
)


# ── Path placement ──────────────────────────────────────────────────────


def test_lock_path_under_stages_base(tmp_path: Path):
    """Lock file lands under stages_base, not output_base."""
    stages = tmp_path / "stages"
    output = tmp_path / "output"
    fd, lock_path = _acquire_run_lock(stages, output)
    try:
        assert lock_path == stages / "pipeline_state.lock"
        assert lock_path.exists()
        assert not (output / "pipeline_state.lock").exists()
    finally:
        _release_run_lock(fd, lock_path)


def test_lock_metadata_contains_pid_and_started(tmp_path: Path):
    """Acquired lock writes our PID + ISO8601 start + output_base into the file."""
    stages = tmp_path / "stages"
    output = tmp_path / "output"
    fd, lock_path = _acquire_run_lock(stages, output)
    try:
        meta = json.loads(lock_path.read_text())
        assert meta["pid"] == os.getpid()
        assert meta["output_base"] == str(output)
        # ISO8601 with timezone — sanity-check it parses.
        from datetime import datetime

        parsed = datetime.fromisoformat(meta["started"])
        assert parsed.tzinfo is not None
    finally:
        _release_run_lock(fd, lock_path)


# ── Live holder rejection ───────────────────────────────────────────────


def _hold_lock_then_block(stages_str: str, output_str: str, ready_fd: int) -> None:
    """Worker: acquire lock, signal ready, sit in a wait loop."""
    fd, _ = _acquire_run_lock(Path(stages_str), Path(output_str))  # noqa: F841 — held
    os.write(ready_fd, b"1")
    os.close(ready_fd)
    # Sit until parent kills us.
    time.sleep(60)


def test_second_acquire_against_live_holder_raises(tmp_path: Path):
    """A second _acquire_run_lock against a live holder raises ConcurrentRunError
    with the holder's PID and start time in the message."""
    stages = tmp_path / "stages"
    output = tmp_path / "output"
    r, w = os.pipe()
    ctx = mp.get_context("fork")
    proc = ctx.Process(target=_hold_lock_then_block, args=(str(stages), str(output), w))
    proc.start()
    os.close(w)
    try:
        # Wait for the child to signal it has the lock.
        assert os.read(r, 1) == b"1"
        os.close(r)
        with pytest.raises(ConcurrentRunError) as exc_info:
            _acquire_run_lock(stages, output)
        msg = str(exc_info.value)
        assert str(proc.pid) in msg
        assert "holder_pid" in msg
        assert "started" in msg
        assert str(output) in msg
        assert str(stages / "pipeline_state.lock") in msg
    finally:
        proc.terminate()
        proc.join(timeout=5)


# ── Stale-lock reclamation ──────────────────────────────────────────────


def _make_stale_pid() -> int:
    """Fork a child that exits immediately, return its now-dead PID."""
    ctx = mp.get_context("fork")
    proc = ctx.Process(target=lambda: None)
    proc.start()
    proc.join()
    assert proc.exitcode is not None
    return proc.pid


def test_stale_lock_with_dead_pid_is_reclaimed(tmp_path: Path):
    """Lock file written by a dead PID is reclaimed by the next acquire."""
    stages = tmp_path / "stages"
    stages.mkdir()
    output = tmp_path / "output"
    lock_path = stages / "pipeline_state.lock"
    dead_pid = _make_stale_pid()
    lock_path.write_text(
        json.dumps(
            {"pid": dead_pid, "started": "2026-01-01T00:00:00+00:00", "output_base": str(output)}
        )
    )
    fd, returned_path = _acquire_run_lock(stages, output)
    try:
        assert returned_path == lock_path
        meta = json.loads(lock_path.read_text())
        # Reclaim overwrites with our PID.
        assert meta["pid"] == os.getpid()
    finally:
        _release_run_lock(fd, lock_path)


def test_stale_lock_with_corrupt_metadata_is_reclaimed(tmp_path: Path):
    """A corrupt/empty lock file is treated as stale and reclaimed."""
    stages = tmp_path / "stages"
    stages.mkdir()
    output = tmp_path / "output"
    lock_path = stages / "pipeline_state.lock"
    lock_path.write_text("not-json{")
    fd, _ = _acquire_run_lock(stages, output)
    try:
        meta = json.loads(lock_path.read_text())
        assert meta["pid"] == os.getpid()
    finally:
        _release_run_lock(fd, lock_path)


# ── Release ──────────────────────────────────────────────────────────────


def test_release_unlinks_lock_file(tmp_path: Path):
    """_release_run_lock deletes the lock-file from disk."""
    stages = tmp_path / "stages"
    output = tmp_path / "output"
    fd, lock_path = _acquire_run_lock(stages, output)
    assert lock_path.exists()
    _release_run_lock(fd, lock_path)
    assert not lock_path.exists()


def test_release_is_safe_with_none_fd(tmp_path: Path):
    """Pre-acquire failure paths pass fd=None to release; must not raise."""
    _release_run_lock(None, None)  # no-op
    _release_run_lock(None, tmp_path / "no-such.lock")  # no-op


def test_acquire_release_acquire_round_trip(tmp_path: Path):
    """After clean release, a fresh acquire from the same process succeeds."""
    stages = tmp_path / "stages"
    output = tmp_path / "output"
    fd1, lock_path = _acquire_run_lock(stages, output)
    _release_run_lock(fd1, lock_path)
    assert not lock_path.exists()
    fd2, _ = _acquire_run_lock(stages, output)
    try:
        # Lock metadata reflects the second acquisition (fresh started timestamp).
        assert lock_path.exists()
        meta = json.loads(lock_path.read_text())
        assert meta["pid"] == os.getpid()
    finally:
        _release_run_lock(fd2, lock_path)


# ── --force-unlock escape hatch ─────────────────────────────────────────


def test_force_unlock_breaks_existing_lock(tmp_path: Path):
    """--force-unlock removes the lock file before acquire, even if held."""
    stages = tmp_path / "stages"
    stages.mkdir()
    output = tmp_path / "output"
    lock_path = stages / "pipeline_state.lock"
    # Pretend a live holder wrote this — _acquire would normally raise.
    lock_path.write_text(json.dumps({"pid": os.getpid(), "started": "2026-01-01T00:00:00+00:00"}))
    fd, _ = _acquire_run_lock(stages, output, force_unlock=True)
    try:
        meta = json.loads(lock_path.read_text())
        assert meta["pid"] == os.getpid()
        assert meta["started"] != "2026-01-01T00:00:00+00:00"
    finally:
        _release_run_lock(fd, lock_path)


# ── _pid_is_alive ───────────────────────────────────────────────────────


def test_pid_is_alive_for_self():
    assert _pid_is_alive(os.getpid()) is True


def test_pid_is_alive_for_dead_process():
    dead = _make_stale_pid()
    assert _pid_is_alive(dead) is False


def test_pid_is_alive_rejects_zero_and_negative():
    assert _pid_is_alive(0) is False
    assert _pid_is_alive(-1) is False
