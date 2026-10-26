"""Blocking inter-process file lock helper.

Eliminates the bug class where a full read-modify-write (load -> mutate ->
save) races another process and one writer's update is silently lost. The
pattern (per-thread reentrancy tracking + a ``.lock`` sidecar + a blocking
``flock``) already existed inline, near-verbatim, in
``content_ledger._ledger_lock`` and ``pending_journal.pending_lock``. This
module centralises it — same precedent as ``_atomic_write.py`` for the
temp-file + ``os.replace`` dance — so a third RMW site never has to
re-implement it, and a future fix reaches every call site at once.

This is NOT the right primitive for ``journal_publisher.journey_lock``: that
lock is non-blocking (raises/returns ``False`` on contention rather than
waiting) and delegates to the orchestrator's stale-PID-aware
``_acquire_run_lock``/``_release_run_lock`` — different semantics entirely,
left untouched.

Portability: this is the ONLY module that imports ``fcntl`` (POSIX) or
``msvcrt`` (Windows), chosen by ``sys.platform`` below; every other lock site
calls :func:`lock_exclusive` / :func:`unlock`. On POSIX they are
``fcntl.flock`` exactly as before, and a contended non-blocking attempt raises
``BlockingIOError`` (EWOULDBLOCK). On Windows they are ``msvcrt.locking`` on
byte 0 of the open file: the blocking mode loops on ``LK_NBLCK`` with a short
sleep (``LK_LOCK`` gives up after about ten seconds), and a contended
non-blocking attempt raises ``BlockingIOError`` with ``errno.EAGAIN``, so a
caller's ``except BlockingIOError`` and errno check work on both. The Windows
lock is mandatory, not advisory, and is per open handle, which is why the
reentrancy tracking below matters there too.
"""

from __future__ import annotations

import contextlib
import errno
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict

_WINDOWS = sys.platform == "win32"
if _WINDOWS:
    import msvcrt

    fcntl = None
else:
    import fcntl

    msvcrt = None

_WINDOWS_RETRY_S = 0.05
# msvcrt.locking reports a held region as EACCES (LK_NBLCK) or EDEADLOCK (LK_LOCK).
_WINDOWS_BUSY = (errno.EACCES, errno.EAGAIN, getattr(errno, "EDEADLOCK", errno.EACCES))


def _fileno(target) -> int:
    return target if isinstance(target, int) else target.fileno()


def lock_exclusive(target, *, blocking: bool = True) -> None:
    """Take an exclusive lock on an open fd or file object.

    ``blocking=False`` raises ``BlockingIOError`` when another holder has it.
    """
    if not _WINDOWS:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(target, flags)
        return
    fd = _fileno(target)
    while True:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if exc.errno not in _WINDOWS_BUSY:
                raise
            if not blocking:
                raise BlockingIOError(errno.EAGAIN, "file is locked by another holder") from exc
        time.sleep(_WINDOWS_RETRY_S)


def unlock(target) -> None:
    """Release a lock taken by :func:`lock_exclusive` (``OSError`` if it fails)."""
    if not _WINDOWS:
        fcntl.flock(target, fcntl.LOCK_UN)
        return
    fd = _fileno(target)
    os.lseek(fd, 0, os.SEEK_SET)
    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def make_blocking_file_lock(state: threading.local):
    """Build a reentrant, blocking file-lock context manager keyed on a
    resolved path, backed by ``.lock`` sidecar files.

    ``state`` is a caller-owned ``threading.local`` so each lock "family"
    (the ledger vs. the pending sidecar) tracks its own reentrancy depth
    independently — sharing one ``threading.local`` across unrelated
    resources would let a ledger lock held in one thread mask a pending-lock
    acquire for a same-named path in that same thread.

    Two concurrent writers (e.g. the documented two-USA-drives mode) would
    otherwise lose-update each other: both load the same snapshot, each
    saves back only its own change, last writer wins and the other's update
    is lost — permanently, if nothing downstream re-derives it. The lock is
    a blocking exclusive lock on a ``.lock`` sidecar beside the
    resource; the second writer serializes behind the first so both updates
    survive. Advisory + auto-released on fd close / process death.
    Reentrant within a thread so a nested acquire (e.g. a ``save()`` called
    from a caller that already holds the lock) doesn't self-deadlock — flock
    is tied to the open file description, so a naive second acquire in the
    same thread would open a fresh fd and block forever on itself.
    """

    @contextlib.contextmanager
    def _lock(path: Path):
        key = str(Path(path).resolve())
        held: Dict[str, int] | None = getattr(state, "held", None)
        if held is None:
            held = state.held = {}
        if held.get(key):
            held[key] += 1
            try:
                yield
            finally:
                held[key] -= 1
            return
        lock_path = Path(path).parent / (Path(path).name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o664)
        # Group members (two OS users sharing a group) must keep write access
        # to the sidecar, mirroring the resource's own 0o664; the creator's
        # umask may have narrowed it. Best-effort — a foreign creator's chmod
        # fails harmlessly.
        with contextlib.suppress(OSError):
            os.chmod(lock_path, 0o664)
        try:
            lock_exclusive(fd)
            held[key] = 1
            yield
        finally:
            held.pop(key, None)
            with contextlib.suppress(OSError):
                unlock(fd)
            os.close(fd)

    return _lock
