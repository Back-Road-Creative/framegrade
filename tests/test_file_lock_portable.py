"""The portable file lock: msvcrt on Windows, fcntl unchanged on POSIX.

The Windows branch runs here through a fake ``msvcrt`` so it is exercised on
every host; the Windows CI leg also runs this file against the real module.
"""

from __future__ import annotations

import ast
import errno
import os
import sys
import threading
from pathlib import Path

import pytest

from gomoveshift.pipeline import _file_lock

ROOT = Path(__file__).resolve().parents[1]


class FakeMsvcrt:
    """Records calls; ``busy`` attempts at LK_NBLCK fail before one succeeds."""

    LK_UNLCK = 0
    LK_LOCK = 1
    LK_NBLCK = 2
    LK_RLCK = 3
    LK_NBRLCK = 4

    def __init__(self, busy: int = 0):
        self.busy = busy
        self.calls: list[tuple[int, int, int]] = []  # (mode, offset, nbytes)

    def locking(self, fd, mode, nbytes):
        self.calls.append((mode, os.lseek(fd, 0, os.SEEK_CUR), nbytes))
        if mode == self.LK_NBLCK and self.busy > 0:
            self.busy -= 1
            raise OSError(errno.EACCES, "Permission denied")


@pytest.fixture
def windows(monkeypatch):
    fake = FakeMsvcrt()
    monkeypatch.setattr(_file_lock, "_WINDOWS", True)
    monkeypatch.setattr(_file_lock, "msvcrt", fake, raising=False)
    monkeypatch.setattr(_file_lock.time, "sleep", lambda s: None)
    return fake


def _fd(tmp_path, offset=5):
    fd = os.open(tmp_path / "x.lock", os.O_CREAT | os.O_RDWR, 0o600)
    os.write(fd, b"0123456789")
    os.lseek(fd, offset, os.SEEK_SET)
    return fd


def test_windows_blocking_retries_until_free(windows, tmp_path):
    windows.busy = 3
    fd = _fd(tmp_path)
    try:
        _file_lock.lock_exclusive(fd)
    finally:
        os.close(fd)
    modes = [c[0] for c in windows.calls]
    assert modes == [FakeMsvcrt.LK_NBLCK] * 4
    assert FakeMsvcrt.LK_LOCK not in modes
    assert all(c[1] == 0 and c[2] == 1 for c in windows.calls)


def test_windows_nonblocking_raises_blockingioerror(windows, tmp_path):
    windows.busy = 1
    fd = _fd(tmp_path)
    try:
        with pytest.raises(BlockingIOError) as info:
            _file_lock.lock_exclusive(fd, blocking=False)
    finally:
        os.close(fd)
    assert info.value.errno in (errno.EWOULDBLOCK, errno.EAGAIN)
    assert len(windows.calls) == 1


def test_windows_nonblocking_other_error_is_not_contention(windows, monkeypatch, tmp_path):
    def boom(fd, mode, nbytes):
        raise OSError(errno.EBADF, "bad fd")

    monkeypatch.setattr(windows, "locking", boom)
    fd = _fd(tmp_path)
    try:
        with pytest.raises(OSError) as info:
            _file_lock.lock_exclusive(fd, blocking=False)
    finally:
        os.close(fd)
    assert not isinstance(info.value, BlockingIOError)


def test_windows_unlock_seeks_to_zero_and_unlocks(windows, tmp_path):
    fd = _fd(tmp_path)
    try:
        _file_lock.unlock(fd)
    finally:
        os.close(fd)
    assert windows.calls == [(FakeMsvcrt.LK_UNLCK, 0, 1)]


def test_windows_accepts_file_objects(windows, tmp_path):
    with open(tmp_path / "f.lock", "a") as handle:
        _file_lock.lock_exclusive(handle, blocking=False)
        _file_lock.unlock(handle)
    assert [c[0] for c in windows.calls] == [FakeMsvcrt.LK_NBLCK, FakeMsvcrt.LK_UNLCK]
    assert all(c[1] == 0 for c in windows.calls)


def test_windows_blocking_lock_context_manager(windows, tmp_path):
    lock = _file_lock.make_blocking_file_lock(threading.local())
    with lock(tmp_path / "res.json"):
        with lock(tmp_path / "res.json"):  # reentrant: no second lock call
            pass
    modes = [c[0] for c in windows.calls]
    assert modes == [FakeMsvcrt.LK_NBLCK, FakeMsvcrt.LK_UNLCK]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX branch")
def test_posix_nonblocking_serialises_two_holders(tmp_path):
    path = tmp_path / "a.lock"
    first = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    second = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        _file_lock.lock_exclusive(first, blocking=False)
        with pytest.raises(BlockingIOError) as info:
            _file_lock.lock_exclusive(second, blocking=False)
        assert info.value.errno in (errno.EWOULDBLOCK, errno.EAGAIN)
        _file_lock.unlock(first)
        _file_lock.lock_exclusive(second, blocking=False)
        _file_lock.unlock(second)
    finally:
        os.close(first)
        os.close(second)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX branch")
def test_posix_blocking_waits_for_holder(tmp_path):
    lock = _file_lock.make_blocking_file_lock(threading.local())
    target = tmp_path / "res.json"
    order: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    def holder():
        with lock(target):
            order.append("a-in")
            entered.set()
            release.wait(5)
            order.append("a-out")

    other_state = threading.local()
    other = _file_lock.make_blocking_file_lock(other_state)

    def waiter():
        entered.wait(5)
        with other(target):
            order.append("b-in")

    threads = [threading.Thread(target=holder), threading.Thread(target=waiter)]
    for t in threads:
        t.start()
    entered.wait(5)
    release.set()
    for t in threads:
        t.join(10)
    assert order == ["a-in", "a-out", "b-in"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX branch")
def test_posix_uses_flock(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(_file_lock.fcntl, "flock", lambda fd, op: calls.append(op))
    fd = _fd(tmp_path)
    try:
        _file_lock.lock_exclusive(fd)
        _file_lock.lock_exclusive(fd, blocking=False)
        _file_lock.unlock(fd)
    finally:
        os.close(fd)
    f = _file_lock.fcntl
    assert calls == [f.LOCK_EX, f.LOCK_EX | f.LOCK_NB, f.LOCK_UN]


def test_only_file_lock_imports_fcntl():
    offenders = []
    for path in sorted((ROOT / "gomoveshift").rglob("*.py")):
        if path.name == "_file_lock.py" and path.parent.name == "pipeline":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import) and any(a.name == "fcntl" for a in node.names):
                offenders.append(str(path.relative_to(ROOT)))
            elif isinstance(node, ast.ImportFrom) and node.module == "fcntl":
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_media_admission_runs_without_posix_only_os_names(monkeypatch, tmp_path):
    """Windows Python has no ``os.getuid`` or ``os.O_NOFOLLOW``."""
    from gomoveshift.pipeline import media_admission

    monkeypatch.delattr(os, "getuid", raising=False)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    monkeypatch.delenv("GMS_NIGHT_ADMISSION_LOCK", raising=False)
    monkeypatch.setattr(media_admission.tempfile, "gettempdir", lambda: str(tmp_path))
    with media_admission.media_admission() as admitted:
        assert admitted is True
    assert len(list(tmp_path.glob("gomoveshift-night-*.lock"))) == 1


def test_queue_schema_lock_runs_without_fchmod(monkeypatch, tmp_path):
    """``os.fchmod`` exists on Windows only from Python 3.13; the bundle is 3.12."""
    from gomoveshift.publishing import queue

    monkeypatch.delattr(os, "fchmod", raising=False)
    with queue._schema_lock(tmp_path / "q.db"):
        pass
