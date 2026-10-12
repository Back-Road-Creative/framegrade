"""Nonblocking local admission shared by participating heavy photo commands.

Preserves the original night-project lock path for mixed-version workers. The
lock fences cooperating processes; it neither reserves RAM nor replaces each
workload's memory/disk preflight. Never unlink a live lock file.
"""

from __future__ import annotations

import fcntl
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path


class MediaAdmissionError(OSError):
    """The admission fence could not be opened or checked safely."""


@contextmanager
def media_admission():
    """Yield whether admission was acquired; release on return or exception.

    A nested acquisition returns False instead of blocking on its caller. Keep
    one acquisition around the outer heavy operation, including child workers.
    """
    path = Path(
        os.environ.get(
            "GMS_NIGHT_ADMISSION_LOCK",
            str(Path(tempfile.gettempdir()) / f"gomoveshift-night-{os.getuid()}.lock"),
        )
    )
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise MediaAdmissionError(f"Cannot open media admission lock: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise MediaAdmissionError("Media admission lock must be a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            acquired = False
        except OSError as exc:
            raise MediaAdmissionError(f"Cannot acquire media admission lock: {exc}") from exc
        else:
            acquired = True
        yield acquired
    finally:
        os.close(fd)
