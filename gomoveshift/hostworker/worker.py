"""Run console-spooled GPU requests on the host; jobs outlive the worker.

Spool layout (``GMS_HOST_SPOOL_DIR``)::

    pending/<job_id>.json      written by the console
    running/<job_id>.json      claimed by atomic rename
    done/<job_id>.json         finished (any outcome)
    cancel/<job_id>            empty marker written by the console
    logs/<job_id>.log          child stdout+stderr (the launcher writes it)
    jobs/<job_id>.*.json       launch spec, launcher identity, exit record
    <job_id>.status.json       state, pid, phase, progress, exit_code, error
    heartbeat.json             refreshed every few seconds while alive
    capabilities.json          what this worker can do (read by host-reconcile 5u)

Each job runs in its own transient unit (``gms-job-<id>``) through
``gomoveshift.hostworker.launch``, or in its own session where transient units are
refused, so a worker restart, deploy or unit change never signals it. On startup the
worker re-adopts every ``running/`` entry and only watches it: nothing is relaunched.
``GMS_HOST_WORKER_LANES`` (default 2) claim loops run side by side, so a short job
never waits behind a long one; a job that needs the shared GPU lock waits for it inside
its own launcher (``queued``), never failed. Drive takes its own whole-run GPU lock.
The worker adds no pipeline gates and never falls back to the CPU.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from gomoveshift import run_origin
from gomoveshift.console.jobs import ALLOWED_SUBCOMMANDS
from gomoveshift.console.settings import load_registry
from gomoveshift.pipeline.orchestrator_lock import gpu_lock_path

from .launch import proc_start

logger = logging.getLogger(__name__)

SPOOL_ENV = "GMS_HOST_SPOOL_DIR"
RUNNER_ENV = "GMS_HOST_JOB_RUNNER"  # systemd-run (default) | detached
LANES_ENV = "GMS_HOST_WORKER_LANES"
DEFAULT_LANES = 2
#: host-reconcile 5u restarts the worker at once only when the running worker wrote this
CAPABILITIES = {"adopts_running_jobs": True}
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


def _lanes() -> int:
    try:
        return max(1, int(os.environ.get(LANES_ENV, DEFAULT_LANES)))
    except ValueError:
        return DEFAULT_LANES


def _signal_group(pid: Optional[int], start: Optional[str], sig: int) -> None:
    """Signal ``pid``'s process group only while ``pid`` is still the process that
    recorded ``start`` (a reused pid is a stranger's)."""
    if pid is None or start is None or proc_start(pid) != start:
        return
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


class HostWorker:
    def __init__(
        self,
        spool: Optional[Path] = None,
        *,
        gpu_checker: Callable[[], None] = check_nvidia_smi,
        lock_path: Optional[Path] = None,
        grace_s: float = 15.0,
        heartbeat_s: float = 3.0,
        poll_s: float = 1.0,
        spawn: Callable[..., subprocess.Popen] = subprocess.Popen,
        runner: Optional[str] = None,
        lanes: Optional[int] = None,
        adopt_grace_s: float = 30.0,
        child_argv: Callable[[list[str]], list[str]] = lambda argv: argv,
    ) -> None:
        raw = spool if spool is not None else os.environ.get(SPOOL_ENV)
        if not raw:
            raise HostWorkerError(f"no spool dir: pass one or set {SPOOL_ENV}")
        self.spool = Path(raw)
        self.lock_path = lock_path if lock_path is not None else gpu_lock_path()
        if self.lock_path is None:
            raise HostWorkerError("no GPU lock path (GMS_GPU_LOCK_PATH): refusing to run unlocked")
        gpu_checker()
        self.grace_s = grace_s
        self.heartbeat_s = heartbeat_s
        self.poll_s = poll_s
        self.runner = runner or os.environ.get(RUNNER_ENV) or "systemd-run"
        self.lanes = lanes if lanes is not None else _lanes()
        self.adopt_grace_s = adopt_grace_s
        self._spawn = spawn
        self._child_argv = child_argv
        self._launched: dict[str, subprocess.Popen] = {}  # detached launchers, to reap
        #: set by ``serve`` (or a caller): stop claiming and watching; jobs keep running
        self.stop_event = threading.Event()
        for sub in ("pending", "running", "done", "cancel", "logs", "jobs"):
            (self.spool / sub).mkdir(parents=True, exist_ok=True)
        self._atomic_json(self.spool / "capabilities.json", {**CAPABILITIES, "pid": os.getpid()})

    # -- files ---------------------------------------------------------------
    def _atomic_json(self, path: Path, payload: dict) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)

    def _read_json(self, path: Path) -> Optional[dict]:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

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
        self._handle(claimed, adopt=False)
        return True

    def _handle(self, claimed: Path, *, adopt: bool) -> None:
        """Launch (or, adopting, only watch) one claimed job; done/ once it ended."""
        ended = True
        try:
            ended = self._process(claimed, adopt=adopt)
        finally:
            if ended:
                self._finish(claimed)

    def _process(self, claimed: Path, *, adopt: bool) -> bool:
        try:
            req = json.loads(claimed.read_text())
            job_id = req["job_id"]
            if not isinstance(job_id, str) or not _JOB_ID.match(job_id) or ".." in job_id:
                raise HostWorkerError(f"invalid job_id: {job_id!r}")
        except (OSError, ValueError, KeyError, TypeError, HostWorkerError) as exc:
            logger.error("unusable request %s: %s", claimed.name, exc)
            return True
        if adopt:
            if not (self.spool / "jobs" / f"{job_id}.launch.json").exists():
                # claimed by a worker that ran jobs as its own children: that worker is
                # gone, so its child is too. Never relaunched.
                self._write_status(
                    job_id,
                    "failed",
                    error="host worker stopped while this job was running; it was interrupted",
                )
                return True
            logger.info("re-adopting running job %s (not relaunched)", job_id)
            return self._watch(job_id)
        try:
            argv = _validate_argv(req.get("argv"))
        except HostWorkerError as exc:
            self._write_status(job_id, "failed", error=str(exc))
            return True
        source_error = _drive_source_error(argv)
        if source_error:
            self._write_status(job_id, "failed", error=source_error)
            return True
        env = _request_env(req.get("env"))
        if argv[3] == "drive":  # set here, never taken from the request: the spool is untrusted
            env[run_origin.LAUNCHED_BY_ENV] = "hostworker"
        # A job waits for the GPU as long as it takes: queued, never failed. Drive takes
        # the lock itself after its run locks (taking it in the launcher would make the
        # child wait on its own parent); the launcher takes it for everything else.
        env["GMS_GPU_LOCK_WAIT_S"] = "inf"
        if not self._launch(job_id, argv, env, take_lock=argv[3] != "drive"):
            return True
        return self._watch(job_id)

    def _launch(
        self, job_id: str, argv: list[str], env: dict[str, str], *, take_lock: bool
    ) -> bool:
        jobs = self.spool / "jobs"
        for suffix in ("proc", "exit"):
            (jobs / f"{job_id}.{suffix}.json").unlink(missing_ok=True)
        spec = {
            "argv": self._child_argv(argv),
            "env": {**os.environ, **env, "GMS_GPU_LOCK_PATH": str(self.lock_path)},
            "log": str(self.spool / "logs" / f"{job_id}.log"),
            "lock_path": str(self.lock_path),
            "take_lock": take_lock,
            "launched_at": time.time(),
        }
        spec_path = jobs / f"{job_id}.launch.json"
        fd = os.open(
            spec_path.with_name(spec_path.name + ".tmp"),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(fd, "w") as f:
            json.dump(spec, f)
        os.replace(spec_path.with_name(spec_path.name + ".tmp"), spec_path)
        cmd = [sys.executable, "-m", "gomoveshift.hostworker.launch", str(self.spool), job_id]
        if self.runner == "systemd-run":
            unit = [
                "systemd-run",
                f"--unit=gms-job-{job_id}",
                "--collect",
                "--quiet",
                f"--uid={os.getuid()}",
                f"--gid={os.getgid()}",
                f"--working-directory={os.getcwd()}",
            ]
            try:
                res = subprocess.run([*unit, *cmd], capture_output=True, timeout=60, check=False)
            except (OSError, subprocess.SubprocessError) as exc:
                reason = str(exc)
            else:
                if res.returncode == 0:
                    return True
                reason = f"exit {res.returncode}: {res.stderr.decode('utf-8', 'replace').strip()}"
            logger.warning(
                "systemd-run refused job %s (%s); running it in its own session", job_id, reason
            )
        try:
            proc = self._spawn(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            self._write_status(job_id, "failed", error=f"could not start: {exc}")
            return False
        if proc is None:  # a test spawn that records and starts nothing
            return False
        self._launched[job_id] = proc
        return True

    def _watch(self, job_id: str) -> bool:
        """Follow a launched job until its exit record; False when the worker is
        stopping (the job keeps running and the next worker re-adopts it)."""
        jobs = self.spool / "jobs"
        log = self.spool / "logs" / f"{job_id}.log"
        cancel = self.spool / "cancel" / job_id
        spec = self._read_json(jobs / f"{job_id}.launch.json") or {}
        launched_at = float(spec.get("launched_at") or 0)
        prev = self._read_json(self.spool / f"{job_id}.status.json") or {}
        phase, progress = prev.get("phase"), prev.get("progress")
        offset, cancelled, shown = 0, False, None
        while True:
            reaper = self._launched.get(job_id)
            if reaper is not None:
                reaper.poll()
            ident = self._read_json(jobs / f"{job_id}.proc.json")
            done = self._read_json(jobs / f"{job_id}.exit.json")
            offset, found = self._scan_progress(log, offset)
            if found:
                progress, phase = found
            pid = (ident or {}).get("child_pid") or (ident or {}).get("pid")
            if done is not None:
                break
            launcher_alive = ident is not None and proc_start(ident.get("pid")) == ident.get(
                "start"
            )
            if not launcher_alive and (
                ident is not None or time.time() - launched_at > self.adopt_grace_s
            ):
                # re-read: the launcher may have written its exit record just before exiting
                done = self._read_json(jobs / f"{job_id}.exit.json")
                if done is None:
                    done = {"exit_code": None, "error": "the job ended without an exit record"}
                break
            if self.stop_event.is_set():
                self._launched.pop(job_id, None)
                return False
            if cancel.exists() and not cancelled and ident is not None:
                cancelled = True
                self._cancel(ident)
                continue
            state = "queued" if (ident or {}).get("waiting_gpu") else "running"
            now = (state, pid, phase, progress)
            if now != shown:
                shown = now
                extra = (
                    {"phase": "waiting for the GPU lock"} if state == "queued" else {"phase": phase}
                )
                self._write_status(job_id, state, pid=pid, progress=progress, **extra)
            time.sleep(0.1)
        self._launched.pop(job_id, None)
        rc, error = done.get("exit_code"), done.get("error")
        fields = {"pid": pid, "exit_code": rc, "phase": phase, "progress": progress}
        if cancelled or cancel.exists():
            cancel.unlink(missing_ok=True)
            self._write_status(job_id, "cancelled", **fields)
        elif rc == 0:
            fields["progress"] = 1.0 if progress is None else progress
            self._write_status(job_id, "succeeded", **fields)
        else:
            self._write_status(job_id, "failed", error=error or f"exited with code {rc}", **fields)
        return True

    def _cancel(self, ident: Optional[dict]) -> None:
        """SIGTERM, grace, SIGKILL: this job's own child group only (or its launcher,
        while it still waits for the GPU lock)."""
        ident = ident or {}
        if ident.get("child_pid"):
            pid, start = ident["child_pid"], ident.get("child_start")
        else:
            pid, start = ident.get("pid"), ident.get("start")
        _signal_group(pid, start, signal.SIGTERM)
        deadline = time.monotonic() + self.grace_s
        while time.monotonic() < deadline and proc_start(pid) == start and start is not None:
            for reaper in list(self._launched.values()):
                reaper.poll()
            time.sleep(0.05)
        _signal_group(pid, start, signal.SIGKILL)

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

    # -- service loop --------------------------------------------------------
    def serve(self, stop: Optional[threading.Event] = None) -> None:
        """Re-adopt what is running, then run ``lanes`` claim loops until ``stop``.
        Stopping never signals a job."""
        stop = stop or threading.Event()
        self.stop_event = stop
        adopt = sorted((self.spool / "running").glob("*.json"))
        take = threading.Lock()

        def beat() -> None:
            while not stop.is_set():
                try:
                    self.write_heartbeat()
                except OSError:
                    logger.exception("heartbeat write failed")
                stop.wait(self.heartbeat_s)

        def lane() -> None:
            while not stop.is_set():
                with take:
                    claimed = adopt.pop(0) if adopt else None
                try:
                    if claimed is not None:
                        self._handle(claimed, adopt=True)
                    elif not self.run_once():
                        stop.wait(self.poll_s)
                except Exception:  # noqa: BLE001 - one bad job never stops a lane
                    logger.exception("host worker lane error")
                    stop.wait(self.poll_s)

        threads = [threading.Thread(target=beat, name="gms-hostworker-heartbeat", daemon=True)]
        threads += [
            threading.Thread(target=lane, name=f"gms-hostworker-lane-{i}", daemon=True)
            for i in range(self.lanes)
        ]
        for t in threads:
            t.start()
        try:
            while not stop.is_set():
                stop.wait(0.5)
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=5)
