"""File spool the console shares with the host GPU worker (``gomoveshift.hostworker``).

The console container has no GPU, so a GPU subcommand is never run here: the
console writes a request into ``<spool>/pending`` and reads the worker's status
file. Layout and file contracts: ``docs/host-worker.md``. Nothing in this module
runs a process, and nothing falls back to one when the spool is unusable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

SPOOL_ENV = "GMS_HOST_SPOOL_DIR"

#: Subcommands that need the GPU (NVENC / CUDA). With a spool configured these
#: go to the host worker; there is no local path for them.
GPU_SUBCOMMANDS = {"drive", "route-story-film"}

#: Seconds without a worker heartbeat before the worker counts as gone. The
#: worker refreshes every ~3 s, so this is many missed beats, not one.
HEARTBEAT_STALE_S = 60.0

TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


class HostRefusedError(ValueError):
    """The host executor will not take this job; the message is the plain reason."""


def spool_dir() -> Path | None:
    raw = os.environ.get(SPOOL_ENV)
    return Path(raw) if raw else None


def is_gpu_argv(argv: list[str]) -> bool:
    return len(argv) > 3 and argv[3] in GPU_SUBCOMMANDS


def dedupe_key(argv: list[str], brand: str | None) -> str:
    return hashlib.sha256(json.dumps([brand, argv]).encode()).hexdigest()[:32]


def spool_id(brand: str | None, job_id: int) -> str:
    """Name a job in the spool. Each brand's console numbers jobs from 1 and they
    share one spool, so the brand goes in the name."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", brand or "console").strip("-").lower() or "console"
    return f"{slug}-{job_id}"


def log_path(spool: Path, sid: str) -> Path:
    return spool / "logs" / f"{sid}.log"


def _heartbeat_age(spool: Path) -> float | None:
    try:
        ts = float(json.loads((spool / "heartbeat.json").read_text())["ts"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return max(0.0, time.time() - ts)


def worker_gone_reason(spool: Path) -> str | None:
    age = _heartbeat_age(spool)
    if age is None:
        return "the host worker has never reported a heartbeat (is gms-host-worker running?)"
    if age > HEARTBEAT_STALE_S:
        return f"the host worker's heartbeat is {int(age)}s old (is gms-host-worker running?)"
    return None


def require_ready(spool: Path | None) -> Path:
    """The spool to write to, or ``HostRefusedError`` saying what is wrong."""
    if spool is None:
        raise HostRefusedError(f"the host executor is not configured ({SPOOL_ENV} is unset)")
    for sub in ("pending", "cancel"):
        target = spool / sub
        if not target.is_dir() or not os.access(target, os.W_OK):
            raise HostRefusedError(f"the host spool {target} is missing or not writable")
    reason = worker_gone_reason(spool)
    if reason:
        raise HostRefusedError(f"{reason}; nothing was queued")
    return spool


def active_duplicate(spool: Path, key: str) -> str | None:
    """Spool id of a pending or running request carrying ``key``, else ``None``."""
    for sub in ("pending", "running"):
        for path in sorted((spool / sub).glob("*.json")):
            try:
                if json.loads(path.read_text()).get("dedupe_key") == key:
                    return path.stem
            except (OSError, ValueError):
                continue
    return None


def write_request(
    spool: Path, sid: str, brand: str | None, argv: list[str], key: str, env: dict | None = None
) -> None:
    body = {
        "job_id": sid,
        "brand": brand,
        # the container's interpreter path means nothing on the host; the worker substitutes its own
        "argv": ["python3", *argv[1:]],
        "dedupe_key": key,
        "env": dict(env or {}),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    final = spool / "pending" / f"{sid}.json"
    tmp = final.with_name(final.name + ".tmp")
    tmp.write_text(json.dumps(body))
    os.replace(tmp, final)


def read_status(spool: Path, sid: str) -> dict | None:
    try:
        data = json.loads((spool / f"{sid}.status.json").read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def request_present(spool: Path, sid: str) -> bool:
    return any((spool / sub / f"{sid}.json").exists() for sub in ("pending", "running"))


def request_finished(spool: Path, sid: str) -> bool:
    return (spool / "done" / f"{sid}.json").exists()


def write_cancel(spool: Path, sid: str) -> None:
    (spool / "cancel" / sid).touch()
