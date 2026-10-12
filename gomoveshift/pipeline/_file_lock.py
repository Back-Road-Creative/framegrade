"""Blocking inter-process file lock helper.

Eliminates the bug class where a full read-modify-write (load -> mutate ->
save) races another process and one writer's update is silently lost. The
pattern (per-thread reentrancy tracking + a ``.lock`` sidecar + a blocking
``fcntl.flock``) already existed inline, near-verbatim, in
``content_ledger._ledger_lock`` and ``pending_journal.pending_lock``. This
module centralises it — same precedent as ``_atomic_write.py`` for the
temp-file + ``os.replace`` dance — so a third RMW site never has to
re-implement it, and a future fix reaches every call site at once.

This is NOT the right primitive for ``journal_publisher.journey_lock``: that
lock is non-blocking (raises/returns ``False`` on contention rather than
waiting) and delegates to the orchestrator's stale-PID-aware
``_acquire_run_lock``/``_release_run_lock`` — different semantics entirely,
left untouched.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import threading
from pathlib import Path
from typing import Dict


def make_blocking_file_lock(state: threading.local):
    """Build a reentrant, blocking ``fcntl.flock`` context manager keyed on a
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
    a blocking ``fcntl.flock(LOCK_EX)`` on a ``.lock`` sidecar beside the
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
            fcntl.flock(fd, fcntl.LOCK_EX)
            held[key] = 1
            yield
        finally:
            held.pop(key, None)
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    return _lock
