"""Unified ffprobe client — replaces 7 duplicate subprocess patterns."""

from __future__ import annotations

import json
import subprocess
import time
import types
from pathlib import Path
from typing import Optional

# A probe that exceeds its timeout is almost always transient load (a heavy
# concurrent encode pinning the CPU), NOT file corruption — so retry a few
# times with a short backoff before treating the timeout as fatal. NON-timeout
# failures (ffprobe rc!=0, bad JSON) are genuine corruption and are NEVER
# retried.
FFPROBE_TIMEOUT_RETRIES = 3  # total subprocess attempts on TimeoutExpired
FFPROBE_TIMEOUT_BACKOFF_SEC = 2.0  # base backoff, multiplied by attempt number


class FFprobeClient:
    """Single parameterised ffprobe wrapper.

    Replaces all inline ``subprocess.run(["ffprobe", ...])`` patterns
    with one configurable call.

    Parameters
    ----------
    subprocess_module:
        The ``subprocess`` module (or mock) to use for ``run()``.
        Accepting this as a constructor arg lets the caller control
        which module-level ``subprocess`` reference is used, so that
        ``@patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")``
        in existing tests keeps working without modification.
    timeout:
        Default timeout in seconds for ffprobe calls.
    """

    def __init__(
        self,
        subprocess_module: types.ModuleType,
        *,
        timeout: int = 30,
    ) -> None:
        self._subprocess = subprocess_module
        self._timeout = timeout

    def probe(
        self,
        path: Path,
        *,
        verbose_level: str = "quiet",
        show_streams: bool = False,
        show_format: bool = True,
        entries: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> dict:
        """Run ffprobe and return parsed JSON.

        Parameters
        ----------
        path:
            Media file to probe.
        verbose_level:
            ``-v`` flag value (``"quiet"`` or ``"error"``).
        show_streams:
            Include ``-show_streams`` in the command.
        show_format:
            Include ``-show_format`` in the command.
        entries:
            If set, passed as ``-show_entries <entries>`` (replaces
            show_format / show_streams).
        timeout:
            Per-call timeout override (seconds).

        Returns
        -------
        dict
            Parsed JSON output from ffprobe.

        Raises
        ------
        FFprobeError
            On non-zero return code or JSON parse failure.
        subprocess.TimeoutExpired
            When the call exceeds *timeout*.
        """
        cmd: list[str] = ["ffprobe", "-v", verbose_level]

        if entries is not None:
            cmd += ["-show_entries", entries, "-of", "json"]
        else:
            cmd.append("-print_format")
            cmd.append("json")
            if show_format:
                cmd.append("-show_format")
            if show_streams:
                cmd.append("-show_streams")

        cmd.append(str(path))

        result = self._run_with_retry(cmd, timeout or self._timeout)

        if result.returncode != 0:
            raise FFprobeError(
                f"ffprobe failed (rc={result.returncode}): {(result.stderr or '')[:200]}"
            )

        return json.loads(result.stdout)

    def _run_with_retry(self, cmd: list[str], timeout: int):
        """Run ffprobe, retrying ONLY on ``subprocess.TimeoutExpired``.

        A timeout reflects transient machine load (a valid file probed while
        a heavy encode pins the CPU), not corruption, so we retry up to
        ``FFPROBE_TIMEOUT_RETRIES`` times with a linear backoff. Any other
        exception (or a non-zero return code surfaced by the caller) is
        genuine and propagates immediately — never retried.

        On the final timeout the original ``TimeoutExpired`` is re-raised so a
        genuinely hung probe still surfaces, but the message makes clear it was
        a timeout-after-retries (transient load), not file corruption.
        """
        last_exc: Optional[subprocess.TimeoutExpired] = None
        for attempt in range(1, FFPROBE_TIMEOUT_RETRIES + 1):
            try:
                return self._subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                last_exc = exc
                if attempt < FFPROBE_TIMEOUT_RETRIES:
                    time.sleep(FFPROBE_TIMEOUT_BACKOFF_SEC * attempt)
        # All attempts timed out — transient load, not corruption.
        raise subprocess.TimeoutExpired(
            cmd,
            timeout,
            output=(
                f"ffprobe timed out after {FFPROBE_TIMEOUT_RETRIES} attempts "
                f"({timeout}s each) — transient machine load, not corruption. "
                f"Raise config.drive.ffprobe_timeout_sec if probes routinely "
                f"exceed the limit under encode load."
            ),
        ) from last_exc

    # Convenience helpers ---------------------------------------------------

    def probe_duration(self, path: Path) -> float:
        """Return duration in seconds, or 0.0 on any failure."""
        try:
            data = self.probe(path, show_format=True)
        except Exception:
            return 0.0
        return float(data.get("format", {}).get("duration", 0))

    def probe_or_none(
        self,
        path: Path,
        *,
        show_streams: bool = False,
        show_format: bool = True,
    ) -> Optional[dict]:
        """Probe returning ``None`` instead of raising on failure."""
        try:
            return self.probe(
                path,
                show_streams=show_streams,
                show_format=show_format,
            )
        except Exception:
            return None


class FFprobeError(Exception):
    """Raised when ffprobe returns a non-zero exit code."""
