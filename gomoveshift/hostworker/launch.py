"""``python -m gomoveshift.hostworker.launch SPOOL JOB_ID``: one spooled job, start to end.

The worker starts this in its own transient unit (``systemd-run --unit=gms-job-<id>``) or,
where that is refused, in its own session. It outlives the worker: a worker restart,
deploy or unit change never signals it. It reads ``jobs/<id>.launch.json`` (argv, env, log,
whether to take the GPU lock), takes the shared GPU lock itself when asked, waiting as long
as ``lock_wait_s`` allows (``queued``); a lock still held then (a render) is recorded as
``waiting`` and the worker puts the request back in pending, never failed. It runs the
child in its own process group and writes:

    jobs/<id>.proc.json   {pid, start, child_pid, child_start, waiting_gpu}  (identity)
    jobs/<id>.exit.json   {exit_code, error[, waiting]}                       (last write)

The worker that is running at the time (the one that launched it, or a new one that
re-adopted it) turns those into the job's status file. Nothing here relaunches anything.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from gomoveshift._process_group import popen_kwargs, terminate_group
from gomoveshift.pipeline.orchestrator_lock import (
    GpuLockBusyError,
    acquire_gpu_lock,
    release_gpu_lock,
)


def proc_start(pid: Optional[int]) -> Optional[str]:
    """Field 22 of ``/proc/<pid>/stat`` (start ticks), or None when gone or a zombie."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        content = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = content[content.rfind(")") + 2 :].split()
    if len(fields) <= 19 or fields[0] in ("Z", "X"):
        return None
    return fields[19]


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _forward(proc: subprocess.Popen) -> None:
    try:
        terminate_group(proc)
    except (ProcessLookupError, PermissionError):
        pass


def main(argv: Optional[list[str]] = None) -> int:
    spool, job_id = (argv if argv is not None else sys.argv[1:])[:2]
    jobs = Path(spool) / "jobs"
    spec = json.loads((jobs / f"{job_id}.launch.json").read_text())
    me = {"pid": os.getpid(), "start": proc_start(os.getpid()), "child_pid": None}
    me["waiting_gpu"] = bool(spec.get("take_lock"))
    _atomic_json(jobs / f"{job_id}.proc.json", me)
    child: list[subprocess.Popen] = []

    def on_term(*_: object) -> None:
        # A stop (a cancel, `systemctl stop gms-job-<id>`) goes to the child once it
        # exists; before that it ends this process cleanly, so the exit record is written.
        if not child:
            raise SystemExit(143)
        _forward(child[0])

    signal.signal(signal.SIGTERM, on_term)
    fd, lock = None, None
    rc: Optional[int] = None
    error: Optional[str] = None
    waiting: Optional[str] = None
    try:
        if spec.get("take_lock"):
            try:
                fd, lock = acquire_gpu_lock(
                    wait_s=spec.get("lock_wait_s"), path=Path(spec["lock_path"])
                )
            except GpuLockBusyError as exc:
                waiting = f"waiting for the GPU lock: {exc}"
                return 0
            except OSError as exc:
                error = f"GPU lock unavailable: {exc}"
                return 0
        with open(spec["log"], "ab") as logf:
            child.append(
                subprocess.Popen(
                    spec["argv"],
                    stdin=subprocess.DEVNULL,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    env=spec["env"],
                    **popen_kwargs(),
                )
            )
        pid = child[0].pid
        me.update(child_pid=pid, child_start=proc_start(pid), waiting_gpu=False)
        _atomic_json(jobs / f"{job_id}.proc.json", me)
        rc = child[0].wait()
    except OSError as exc:
        error = f"could not start: {exc}"
    except SystemExit:
        error = "stopped before the job started"
    finally:
        release_gpu_lock(fd, lock)
        record: dict = {"exit_code": rc, "error": error}
        if waiting:  # never started: the worker puts it back in pending
            record["waiting"] = waiting
        _atomic_json(jobs / f"{job_id}.exit.json", record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
