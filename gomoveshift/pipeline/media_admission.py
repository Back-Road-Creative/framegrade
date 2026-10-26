"""Nonblocking local admission shared by participating heavy photo commands.

Preserves the original night-project lock path for mixed-version workers. The
lock fences cooperating processes; it neither reserves RAM nor replaces each
workload's memory/disk preflight. Never unlink a live lock file.
"""

from __future__ import annotations

import getpass
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from gomoveshift.pipeline._file_lock import lock_exclusive


class MediaAdmissionError(OSError):
    """The admission fence could not be opened or checked safely."""


def _user_key() -> str:
    """The uid on POSIX (the original lock name); the user name on Windows."""
    return str(os.getuid()) if hasattr(os, "getuid") else getpass.getuser()


@contextmanager
def media_admission():
    """Yield whether admission was acquired; release on return or exception.

    A nested acquisition returns False instead of blocking on its caller. Keep
    one acquisition around the outer heavy operation, including child workers.
    """
    path = Path(
        os.environ.get(
            "GMS_NIGHT_ADMISSION_LOCK",
            str(Path(tempfile.gettempdir()) / f"gomoveshift-night-{_user_key()}.lock"),
        )
    )
    try:
        # Windows has no O_NOFOLLOW; its per-user temp dir is not shared.
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise MediaAdmissionError(f"Cannot open media admission lock: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise MediaAdmissionError("Media admission lock must be a regular file")
        try:
            lock_exclusive(fd, blocking=False)
        except BlockingIOError:
            acquired = False
        except OSError as exc:
            raise MediaAdmissionError(f"Cannot acquire media admission lock: {exc}") from exc
        else:
            acquired = True
        yield acquired
    finally:
        os.close(fd)
