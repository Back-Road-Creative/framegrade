"""WSL mount detection and recovery.

WSL2 9p mounts can become stale after Windows sleep/hibernate. This module
provides mount accessibility checks and automatic remount recovery.
"""

import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class DriveNotMountedError(RuntimeError):
    """Raised when a required WSL ``/mnt/<letter>/`` drive is not accessible.

    Carries the mount point that failed (e.g. ``/mnt/d``) so callers can surface
    an operator-friendly remediation message naming the missing drive.
    """

    def __init__(self, mount_point: Path, attempts: int) -> None:
        self.mount_point = Path(mount_point)
        self.attempts = attempts
        drive_letter = self.mount_point.name.upper() if self.mount_point.name else "?"
        super().__init__(
            f"Windows drive {self.mount_point} is not accessible after {attempts} probe "
            f"attempt(s). The WSL 9p mount is likely stale (post-sleep/hibernate). "
            f"Recover with: sudo mount -t drvfs {drive_letter}: {self.mount_point}"
        )


def assert_wsl_mount(path: Path, retries: int = 3, delay: float = 2.0) -> None:
    """Pre-flight check — raise :class:`DriveNotMountedError` if ``path`` lives
    on an inaccessible WSL ``/mnt/<letter>/`` mount.

    No-op for paths outside ``/mnt/``. Wraps :func:`ensure_wsl_mount` and converts
    its boolean result into a typed exception with operator guidance, so callers
    that copy or open files on Windows drives can surface a clear failure rather
    than a cryptic ``OSError`` from deep inside ``shutil``/``open``.
    """
    if not str(path).startswith("/mnt/"):
        return
    if ensure_wsl_mount(path, retries=retries, delay=delay):
        return
    parts = Path(path).parts
    mount_point = Path(parts[0]) / parts[1] / parts[2] if len(parts) >= 3 else Path(path)
    raise DriveNotMountedError(mount_point, retries)


def ensure_wsl_mount(path: Path, retries: int = 3, delay: float = 2.0) -> bool:
    """Verify a WSL /mnt/ drive is accessible, attempting remount if stale.

    WSL2 9p mounts can become stale after Windows sleep/hibernate.
    This checks accessibility and tries `mount -t drvfs` to recover.

    Returns True if the mount is accessible after checks.
    """
    if not str(path).startswith("/mnt/"):
        return True

    # Extract the drive mount point (e.g., /mnt/d)
    parts = Path(path).parts
    if len(parts) < 3:
        return True
    mount_point = Path(parts[0]) / parts[1] / parts[2]  # /mnt/d
    drive_letter = parts[2].upper()  # D

    for attempt in range(retries):
        try:
            # Test if mount is responsive (ls with timeout)
            result = subprocess.run(
                ["ls", str(mount_point)],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            logger.warning(
                f"  Mount {mount_point} not responding (attempt {attempt + 1}/{retries})"
            )
        except Exception as e:
            logger.warning(f"  Mount check failed: {e}")

        if attempt < retries - 1:
            if not os.isatty(0):
                logger.warning(
                    f"  Skipping remount of {mount_point} — non-interactive context "
                    f"(no tty). Run manually: sudo mount -t drvfs {drive_letter}: {mount_point}"
                )
                continue
            logger.info(f"  Attempting to remount {drive_letter}: → {mount_point}...")
            try:
                subprocess.run(
                    ["sudo", "mount", "-t", "drvfs", f"{drive_letter}:", str(mount_point)],
                    capture_output=True,
                    timeout=10,
                )
                time.sleep(delay)
            except Exception as e:
                logger.warning(f"  Remount attempt failed: {e}")

    logger.error(
        f"  Mount {mount_point} is not accessible after {retries} attempts.\n"
        f"  Try manually: sudo mount -t drvfs {drive_letter}: {mount_point}"
    )
    return False
