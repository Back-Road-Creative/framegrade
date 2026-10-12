"""The shared GPU lock, and its file-level contract with a cooperating process.

The two parties never import each other; they interoperate through the lock
file alone, so the contract is asserted here rather than assumed.
"""

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from gomoveshift.pipeline.orchestrator_lock import (
    GPU_LOCK_OWNER,
    GpuLockBusyError,
    acquire_gpu_lock,
    gpu_lock_path,
    release_gpu_lock,
)


def test_disabled_when_path_is_empty(monkeypatch):
    monkeypatch.setenv("GMS_GPU_LOCK_PATH", "")
    assert gpu_lock_path() is None
    assert acquire_gpu_lock() == (None, None)


def test_default_path_is_gpu_lock_under_cache_home(monkeypatch, tmp_path):
    monkeypatch.delenv("GMS_GPU_LOCK_PATH", raising=False)
    monkeypatch.setenv("GMS_CACHE_HOME", str(tmp_path))
    assert gpu_lock_path() == tmp_path / "gpu.lock"


def test_env_path_overrides_default(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_CACHE_HOME", str(tmp_path / "cache"))
    override = tmp_path / "custom.lock"
    monkeypatch.setenv("GMS_GPU_LOCK_PATH", str(override))
    assert gpu_lock_path() == override


def test_acquire_writes_owner_and_pid(tmp_path):
    path = tmp_path / "gpu.lock"
    fd, held = acquire_gpu_lock(wait_s=0, path=path)
    try:
        holder = json.loads(path.read_text())
        assert holder["owner"] == GPU_LOCK_OWNER
        assert holder["pid"] == os.getpid()
        assert held == path
    finally:
        release_gpu_lock(fd, held)


def test_release_does_not_unlink(tmp_path):
    """A deleted-and-recreated lock file is two inodes, hence two holders."""
    path = tmp_path / "gpu.lock"
    fd, held = acquire_gpu_lock(wait_s=0, path=path)
    inode = path.stat().st_ino
    release_gpu_lock(fd, held)
    assert path.exists(), "release must leave the lock file in place"
    assert path.stat().st_ino == inode
    assert json.loads(path.read_text()) == {}


def test_reacquirable_after_release(tmp_path):
    path = tmp_path / "gpu.lock"
    fd, held = acquire_gpu_lock(wait_s=0, path=path)
    release_gpu_lock(fd, held)
    fd2, held2 = acquire_gpu_lock(wait_s=0, path=path)
    release_gpu_lock(fd2, held2)


def _hold(path_str, ready, stop):
    fd, held = acquire_gpu_lock(wait_s=0, path=Path(path_str))
    ready.set()
    stop.wait(30)
    release_gpu_lock(fd, held)


def test_live_holder_blocks_and_the_error_names_it(tmp_path):
    path = tmp_path / "gpu.lock"
    ctx = mp.get_context("spawn")
    ready, stop = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=_hold, args=(str(path), ready, stop))
    proc.start()
    try:
        # A spawn child re-imports this module (and gomoveshift) from scratch; inside a
        # long parallel suite that can take more than 10 s. Only the failure path waits
        # this long.
        assert ready.wait(120), "holder process did not start"
        started = time.monotonic()
        with pytest.raises(GpuLockBusyError) as exc:
            acquire_gpu_lock(wait_s=1.0, path=path)
        assert time.monotonic() - started >= 1.0, "must wait out the holder"
        assert str(proc.pid) in str(exc.value)
    finally:
        stop.set()
        proc.join(10)


def test_a_foreign_holder_is_honoured(tmp_path):
    """A foreign holder writes ``owner``/``pid`` too; flock is the real mutex."""
    path = tmp_path / "gpu.lock"
    import fcntl

    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.write(fd, json.dumps({"owner": "other-process", "pid": os.getpid()}).encode())
    try:
        with pytest.raises(GpuLockBusyError, match="other-process"):
            acquire_gpu_lock(wait_s=0, path=path)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class _SentinelError(RuntimeError):
    pass


def test_drive_orchestrator_run_takes_the_gpu_lock(tmp_path, monkeypatch):
    """The wiring, not just the helper — an untaken lock is worse than none.

    Patches the acquire call to raise and asserts the exception escapes
    ``run()``, which can only happen if the call site is on the live path.
    """
    from gomoveshift.config import Config
    from gomoveshift.pipeline import drive_orchestrator as mod

    def _boom(*_a, **_k):
        raise _SentinelError("acquire_gpu_lock reached")

    monkeypatch.setattr(mod, "acquire_gpu_lock", _boom)
    source = tmp_path / "src"
    source.mkdir()
    orch = mod.DrivePipelineOrchestrator(config=Config())
    with pytest.raises(_SentinelError):
        orch.run(source_dir=source, output_base=tmp_path / "out")
    # A GPU lock that cannot be taken must roll the run locks back, or every
    # later run on this project dir refuses against a lock nobody holds.
    assert not list(tmp_path.rglob("*.lock")), "run locks leaked on the failure path"
