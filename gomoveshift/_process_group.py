"""Portable "child in its own process group" start and kill.

POSIX: ``start_new_session`` plus ``os.killpg``. Windows: a new process group
plus CTRL_BREAK for the polite stop and ``taskkill /T /F`` for the hard kill.
The platform is read at call time so tests can switch it.

Errors propagate (``ProcessLookupError`` / ``PermissionError`` on POSIX); each
caller decides what an already-exited child means.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Any


def _is_windows() -> bool:
    return os.name == "nt"


def popen_kwargs() -> dict[str, Any]:
    """Popen kwargs that put the child in its own process group."""
    if _is_windows():
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def terminate_group(proc: subprocess.Popen) -> None:
    """Ask the child's group to stop (SIGTERM / CTRL_BREAK)."""
    if _is_windows():
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        os.killpg(proc.pid, signal.SIGTERM)


def kill_group(proc: subprocess.Popen) -> None:
    """Force-kill the child and its descendants."""
    if not _is_windows():
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return
    try:
        result = subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            return
    except (OSError, subprocess.SubprocessError):
        pass
    proc.kill()
