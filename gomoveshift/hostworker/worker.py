"""Run console-spooled GPU requests on the host, one at a time.

Spool layout (``GMS_HOST_SPOOL_DIR``)::

    pending/<job_id>.json      written by the console
    running/<job_id>.json      claimed by atomic rename
    done/<job_id>.json         finished (any outcome)
    cancel/<job_id>            empty marker written by the console
    logs/<job_id>.log          child stdout+stderr (the worker writes it)
    <job_id>.status.json       state, pid, phase, progress, exit_code, error
    heartbeat.json             refreshed every few seconds while alive

The worker adds no pipeline gates: it re-checks the argv allowlist, takes the
shared GPU lock for commands that do not take it themselves, and runs the child.
Drive owns its own whole-run GPU lock. It never falls back to the CPU.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from gomoveshift import run_origin
from gomoveshift._process_group import kill_group, popen_kwargs, terminate_group
from gomoveshift.console.jobs import ALLOWED_SUBCOMMANDS
from gomoveshift.console.settings import load_registry
from gomoveshift.pipeline.orchestrator_lock import (
    GpuLockBusyError,
    acquire_gpu_lock,
    gpu_lock_path,
    release_gpu_lock,
)

logger = logging.getLogger(__name__)

SPOOL_ENV = "GMS_HOST_SPOOL_DIR"
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PROGRESS = re.compile(rb"^GMS_PROGRESS\s+([0-9.]+)\s*(.*)$", re.MULTILINE)
_DRIVE_MEDIA_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mxf"})


#: never taken from a request: the worker's own spool and the shared GPU lock
_PROTECTED_ENV = frozenset({SPOOL_ENV, "GMS_GPU_LOCK_PATH"})


class HostWorkerError(RuntimeError):
    """The worker cannot start or a request is unusable."""


def check_nvidia_smi() -> None:
    """Raise unless a GPU is visible (``nvidia-smi`` runs and exits 0)."""
    try:
        res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostWorkerError(f"nvidia-smi failed: {exc}") from exc
    if res.returncode != 0:
        raise HostWorkerError(f"nvidia-smi exited {res.returncode}: no GPU visible")


def _validate_argv(argv: object) -> list[str]:
    ok = (
        isinstance(argv, list)
        and len(argv) >= 4
        and all(isinstance(a, str) for a in argv)
        and argv[1:3] == ["-m", "gomoveshift"]
        and (argv[0] == sys.executable or os.path.basename(argv[0]).startswith("python"))
        and argv[3] in ALLOWED_SUBCOMMANDS
    )
    if not ok:
        raise HostWorkerError(f"argv not on the allowlist: {argv!r}")
    # The console builds argv[0] from its container's interpreter, which does not exist on
    # the host: run the checked request under the worker's own interpreter instead.
    return [sys.executable, *argv[1:]]  # type: ignore[index]


def _request_env(raw: object) -> dict[str, str]:
    """The request's brand overrides: registry names only, string values only.

    Anything else (a non-dict, unknown names, the spool/lock vars) is dropped,
    never fatal -- the request still runs on the worker's own environment."""
    if not isinstance(raw, dict):
        return {}
    allowed = {row.name for row in load_registry()} - _PROTECTED_ENV
    return {k: v for k, v in raw.items() if k in allowed and isinstance(v, str)}


def _drive_source_error(argv: list[str]) -> Optional[str]:
    """Check that the worker can read a drive source before starting its child."""
    if argv[3] != "drive":
        return None
    if len(argv) < 5:
        return "drive source directory is missing"
    source = Path(argv[4])
    if not source.is_absolute():
        return f"drive source must be an absolute host path: {source}"
    if not source.is_dir():
        return f"drive source directory is missing or inaccessible: {source}"
    try:
        with os.scandir(source) as entries:
            for entry in entries:
                if Path(entry.name).suffix.lower() not in _DRIVE_MEDIA_SUFFIXES:
                    continue
                try:
                    fd = os.open(entry.path, os.O_RDONLY | os.O_NONBLOCK)
                except OSError as exc:
                    return f"drive source media is unreadable: {entry.path}: {exc}"
                else:
                    os.close(fd)
    except OSError as exc:
        return f"drive source directory is unreadable: {source}: {exc}"
    return None


class HostWorker:
    def __init__(
        self,
        spool: Optional[Path] = None,
        *,
        gpu_checker: Callable[[], None] = check_nvidia_smi,
        lock_path: Optional[Path] = None,
        lock_wait_s: Optional[float] = None,
        grace_s: float = 15.0,
        heartbeat_s: float = 3.0,
        poll_s: float = 1.0,
        spawn: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        raw = spool if spool is not None else os.environ.get(SPOOL_ENV)
        if not raw:
            raise HostWorkerError(f"no spool dir: pass one or set {SPOOL_ENV}")
        self.spool = Path(raw)
        self.lock_path = lock_path if lock_path is not None else gpu_lock_path()
        if self.lock_path is None:
            raise HostWorkerError("no GPU lock path (GMS_GPU_LOCK_PATH): refusing to run unlocked")
        gpu_checker()
        self.lock_wait_s = lock_wait_s
        self.grace_s = grace_s
        self.heartbeat_s = heartbeat_s
        self.poll_s = poll_s
        self._spawn = spawn
        #: set by ``serve`` (or a caller): stop the running child and return
        self.stop_event = threading.Event()
        for sub in ("pending", "running", "done", "cancel", "logs"):
            (self.spool / sub).mkdir(parents=True, exist_ok=True)

    # -- files ---------------------------------------------------------------
    def _atomic_json(self, path: Path, payload: dict) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)

    def write_heartbeat(self) -> None:
        self._atomic_json(self.spool / "heartbeat.json", {"ts": time.time(), "pid": os.getpid()})

    def _write_status(self, job_id: str, state: str, **fields: object) -> None:
        payload = {
            "job_id": job_id,
            "state": state,
            "pid": None,
            "phase": None,
            "progress": None,
            "exit_code": None,
            "error": None,
            "updated_at": time.time(),
        }
        payload.update(fields)
        self._atomic_json(self.spool / f"{job_id}.status.json", payload)

    # -- claim ---------------------------------------------------------------
    def claim_next(self) -> Optional[Path]:
        """Atomically move the oldest pending request to running/."""
        pending = self.spool / "pending"
        entries = []
        for p in pending.glob("*.json"):
            try:
                entries.append((p.stat().st_mtime, p.name, p))
            except FileNotFoundError:
                continue
        for _, name, p in sorted(entries):
            target = self.spool / "running" / name
            try:
                os.rename(p, target)
            except FileNotFoundError:
                continue  # another claimer won
            return target
        return None

    def _finish(self, claimed: Path) -> None:
        shutil.move(str(claimed), str(self.spool / "done" / claimed.name))

    # -- run -----------------------------------------------------------------
    def run_once(self) -> bool:
        """Process at most one request; False when nothing was pending."""
        claimed = self.claim_next()
        if claimed is None:
            return False
        try:
            self._process(claimed)
        finally:
            self._finish(claimed)
        return True

    def _process(self, claimed: Path) -> None:
        try:
            req = json.loads(claimed.read_text())
            job_id = req["job_id"]
            if not isinstance(job_id, str) or not _JOB_ID.match(job_id) or ".." in job_id:
                raise HostWorkerError(f"invalid job_id: {job_id!r}")
        except (OSError, ValueError, KeyError, TypeError, HostWorkerError) as exc:
            logger.error("unusable request %s: %s", claimed.name, exc)
            return
        try:
            argv = _validate_argv(req.get("argv"))
        except HostWorkerError as exc:
            self._write_status(job_id, "failed", error=str(exc))
            return
        source_error = _drive_source_error(argv)
        if source_error:
            self._write_status(job_id, "failed", error=source_error)
            return
        # Drive acquires this same lock inside its orchestrator after its run
        # locks. Taking it here would make the child wait on its own parent.
        # The worker still serializes spool jobs while drive owns the lock.
        fd, lock = None, None
        if argv[3] != "drive":
            try:
                fd, lock = acquire_gpu_lock(wait_s=self.lock_wait_s, path=self.lock_path)
            except (GpuLockBusyError, OSError) as exc:
                self._write_status(job_id, "failed", error=f"GPU lock unavailable: {exc}")
                return
        env = _request_env(req.get("env"))
        if argv[3] == "drive":  # set here, never taken from the request: the spool is untrusted
            env[run_origin.LAUNCHED_BY_ENV] = "hostworker"
        try:
            self._run_child(job_id, argv, env)
        finally:
            release_gpu_lock(fd, lock)

    def _run_child(self, job_id: str, argv: list[str], env: dict[str, str]) -> None:
        log = self.spool / "logs" / f"{job_id}.log"
        cancel = self.spool / "cancel" / job_id
        try:
            with open(log, "ab") as logf:
                proc = self._spawn(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    **popen_kwargs(),
                    env={**os.environ, **env, "GMS_GPU_LOCK_PATH": str(self.lock_path)},
                )
        except OSError as exc:
            self._write_status(job_id, "failed", error=f"could not start: {exc}")
            return
        phase: Optional[str] = None
        progress: Optional[float] = None
        offset = 0
        self._write_status(job_id, "running", pid=proc.pid)
        cancelled = False
        stopped = False
        while proc.poll() is None:
            if self.stop_event.is_set():
                stopped = True
                self._stop(proc)
                break
            if cancel.exists():
                cancelled = True
                self._stop(proc)
                break
            offset, found = self._scan_progress(log, offset)
            if found and found != (progress, phase):
                progress, phase = found
                self._write_status(job_id, "running", pid=proc.pid, phase=phase, progress=progress)
            time.sleep(0.1)
        rc = proc.wait()
        if stopped:
            self._write_status(
                job_id,
                "failed",
                pid=proc.pid,
                exit_code=rc,
                phase=phase,
                progress=progress,
                error="host worker stopped while this job was running; it was interrupted",
            )
        elif cancelled:
            cancel.unlink(missing_ok=True)
            self._write_status(
                job_id, "cancelled", pid=proc.pid, exit_code=rc, phase=phase, progress=progress
            )
        elif rc == 0:
            self._write_status(
                job_id,
                "succeeded",
                pid=proc.pid,
                exit_code=0,
                phase=phase,
                progress=1.0 if progress is None else progress,
            )
        else:
            self._write_status(
                job_id,
                "failed",
                pid=proc.pid,
                exit_code=rc,
                phase=phase,
                progress=progress,
                error=f"exited with code {rc}",
            )

    @staticmethod
    def _scan_progress(log: Path, offset: int) -> tuple[int, Optional[tuple[float, str]]]:
        try:
            with open(log, "rb") as f:
                f.seek(offset)
                data = f.read()
        except OSError:
            return offset, None
        last = None
        for m in _PROGRESS.finditer(data):
            try:
                last = (float(m.group(1)), m.group(2).decode("utf-8", "replace").strip())
            except ValueError:
                continue
        return offset + len(data), last

    def _stop(self, proc: subprocess.Popen) -> None:
        """SIGTERM, grace, SIGKILL: our own child's process group only."""
        self._signal(terminate_group, proc)
        try:
            proc.wait(timeout=self.grace_s)
        except subprocess.TimeoutExpired:
            self._signal(kill_group, proc)

    @staticmethod
    def _signal(action: Callable[[subprocess.Popen], None], proc: subprocess.Popen) -> None:
        try:
            action(proc)  # child is its own group leader
        except (ProcessLookupError, PermissionError):
            pass

    # -- service loop --------------------------------------------------------
    def serve(self, stop: Optional[threading.Event] = None) -> None:
        stop = stop or threading.Event()
        self.stop_event = stop

        def beat() -> None:
            while not stop.is_set():
                try:
                    self.write_heartbeat()
                except OSError:
                    logger.exception("heartbeat write failed")
                stop.wait(self.heartbeat_s)

        t = threading.Thread(target=beat, name="gms-hostworker-heartbeat", daemon=True)
        t.start()
        try:
            while not stop.is_set():
                if not self.run_once():
                    stop.wait(self.poll_s)
        finally:
            stop.set()
            t.join(timeout=5)
