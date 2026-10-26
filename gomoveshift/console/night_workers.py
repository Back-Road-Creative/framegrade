"""Cross-process night launches and durable, fail-closed retry intent."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import time
from contextlib import closing, contextmanager
from pathlib import Path

from gomoveshift.pipeline._file_lock import lock_exclusive

from ..pipeline.photos.night_project import build_identity, runtime_identity, worker_alive
from . import jobs, settings

HEARTBEAT_DEADLINE_S = 90
STAGE_STALL_DEADLINES_S = {
    "best_stills": 2 * 3600,
    "deep_still": 4 * 3600,
    "timelapse": 12 * 3600,
    "star_trails": 12 * 3600,
    "private_video": 12 * 3600,
}


def live_worker_observation(row: dict, *, now: float) -> str:
    """Describe liveness and milestone age without authorizing replacement."""
    heartbeat = row.get("heartbeat_at")
    if not isinstance(heartbeat, (int, float)) or now - heartbeat > HEARTBEAT_DEADLINE_S:
        age = (
            "unavailable" if not isinstance(heartbeat, (int, float)) else f"{int(now - heartbeat)}s"
        )
        return (
            f"Live process identity, but worker heartbeat is {age} old; "
            "supervisor will not relaunch it."
        )
    stage = row.get("current_stage")
    progress_at = row.get("last_progress_at")
    deadline = STAGE_STALL_DEADLINES_S.get(stage)
    if (
        deadline is not None
        and isinstance(progress_at, (int, float))
        and now - progress_at > deadline
    ):
        return (
            f"Live heartbeat; {stage} has reported no milestone for {int(now - progress_at)}s, "
            f"past its {deadline}s stage stall review threshold."
        )
    progress = row.get("stage_progress")
    if isinstance(progress, dict) and stage:
        return (
            f"Live worker; {stage} milestone {progress.get('completed', '?')}/"
            f"{progress.get('total', '?')}; no duplicate launch."
        )
    return "Live worker heartbeat is current; no duplicate launch."


@contextmanager
def state_db(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "supervision.sqlite3"
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Supervision store must not be a symlink")
    db = sqlite3.connect(path, timeout=1)
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS retries (id TEXT, brand TEXT, data TEXT, enabled INTEGER, next_due REAL, PRIMARY KEY(id, brand))"
        )
        with db:
            yield db
    finally:
        db.close()


def read_state(root, brand, project):
    if not (Path(root) / "supervision.sqlite3").exists():
        return None
    with state_db(root) as db:
        row = db.execute(
            "SELECT CASE WHEN length(data)<=1048576 THEN data END FROM retries WHERE id=? AND brand=?",
            (project, brand),
        ).fetchone()
    if not row:
        return None
    state = json.loads(row[0])
    if not isinstance(state, dict) or not isinstance(state.get("enabled"), bool):
        raise ValueError("Invalid supervision metadata")
    if state.get("claim") not in {"idle", "launching"}:
        raise ValueError("Invalid launch claim")
    if state["claim"] == "launching" and (
        type(state.get("jobs_floor")) is not int or state["jobs_floor"] < 0
    ):
        raise ValueError("Invalid claim evidence boundary")
    for field in ("launches", "crash_retries"):
        if type(state.get(field)) is not int or not 0 <= state[field] <= 10**9:
            raise ValueError("Invalid supervision counters")
    if not isinstance(state.get("next_retry_at"), (int, float)) or not math.isfinite(
        state["next_retry_at"]
    ):
        raise ValueError("Invalid retry time")
    return state


def observation(root, brand, project):
    """Fail-closed, displayable supervision health without breaking project view."""
    try:
        return read_state(root, brand, project)
    except (OSError, sqlite3.Error, ValueError, TypeError, RecursionError, KeyError):
        return {
            "enabled": False,
            "launches": "unknown",
            "next_retry_at": "unavailable",
            "observation": "Supervision metadata unavailable; manual review required.",
        }


def _save(root, brand, project, state):
    with state_db(root) as db:
        db.execute(
            "INSERT OR REPLACE INTO retries VALUES (?, ?, ?, ?, ?)",
            (project, brand, json.dumps(state), int(state["enabled"]), state["next_retry_at"]),
        )


def project_metadata(root, brand, project):
    path = Path(root) / "projects.sqlite3"
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Project store must not be a symlink")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)) as db:
        found = db.execute(
            "SELECT CASE WHEN length(data)<=1048576 THEN data END FROM night_projects WHERE id=? AND brand=?",
            (project, brand),
        ).fetchone()
    if not found:
        raise KeyError(project)
    row = json.loads(found[0])
    if row["id"] != project or row["brand"] != brand:
        raise ValueError("Project identity mismatch")
    return row


def _argv(root, brand, project):
    return [
        sys.executable,
        "-m",
        "gomoveshift",
        "photos",
        "night",
        "--project",
        project,
        "--workspace",
        str(root),
        "--brand",
        brand,
    ]


def _live(job_store, row):
    process = jobs.tracked_process(job_store, row["id"])
    if process is not None:
        return process.poll() is None
    return row["pid"] is not None and jobs._process_matches(row["pid"], row["started_fingerprint"])


def launch(store, project, *, automatic=False, now=None):
    """Launch at most once across HTTP/tickers; unknown Popen outcomes stay held.

    Claims have no expiring lease. An unidentified child after a crash requires
    operator reconciliation; time alone never makes another launch safe.
    """
    now = time.time() if now is None else now
    brand = settings.current_brand()
    store.root.mkdir(parents=True, exist_ok=True)
    fd = os.open(store.root / "supervision.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as lock:
        try:
            lock_exclusive(lock, blocking=False)
        except BlockingIOError:
            return False
        row = project_metadata(store.root, brand, project)
        state = read_state(store.root, brand, project)
        if automatic and (not state or not state["enabled"] or now < state["next_retry_at"]):
            return False
        state = state or {
            "enabled": True,
            "launches": 0,
            "crash_retries": 0,
            "next_retry_at": now,
            "claim": "idle",
        }
        if row.get("cancel_requested") or (automatic and row["status"] == "cancelled"):
            state.update(enabled=False, observation="Cancelled; explicit Resume required.")
            _save(store.root, brand, project, state)
            return False
        if row.get("pause_requested") or (automatic and row["status"] == "paused"):
            state.update(enabled=False, observation="Paused; explicit Resume required.")
            _save(store.root, brand, project, state)
            return False
        if (
            (automatic and row["status"] == "complete")
            or row["status"] == "blocked"
            or row.get("build") != build_identity()
            or row.get("runtime") != runtime_identity()
            or row.get("interpreter") != sys.executable
        ):
            state.update(
                enabled=False,
                observation="Project is complete or build-blocked; no automatic launch.",
            )
            _save(store.root, brand, project, state)
            return False
        if not automatic:
            state["enabled"] = True
        job_store = jobs.JobStore(settings.console_data_dir() / "jobs.sqlite3")
        try:
            argv = _argv(store.root, brand, project)
            matching = [
                dict(job)
                for job in job_store._conn.execute(
                    "SELECT id, pid, started_fingerprint, status FROM jobs WHERE brand=? AND argv=?",
                    (brand, json.dumps(argv)),
                ).fetchall()
            ]
            live = any(_live(job_store, job) for job in matching)
            if live or worker_alive(row.get("worker")):
                state.update(
                    observation=live_worker_observation(row, now=now),
                    next_retry_at=now + 60,
                )
                _save(store.root, brand, project, state)
                return False
            uncertain = any(
                (
                    job["status"] in {"queued", "running"}
                    or job["id"] == state.get("job_id")
                    or (state["claim"] == "launching" and job["id"] > state["jobs_floor"])
                )
                and jobs.tracked_process(job_store, job["id"]) is None
                and (job["pid"] is None or not job["started_fingerprint"])
                for job in matching
            )
            if uncertain:
                state.update(
                    claim="launching",
                    jobs_floor=state.get("jobs_floor", 0),
                    observation="Worker identity unknown; manual process reconciliation required.",
                    next_retry_at=now + 3600,
                )
                _save(store.root, brand, project, state)
                return False
            if state["claim"] == "launching":
                # An unfinished claim may have created a child before the parent
                # could persist its PID. Only a positively identified dead child
                # lets us clear it; an empty/queued Jobs row is not that evidence.
                evidence = [
                    job
                    for job in matching
                    if job["id"] > state.get("jobs_floor", 0)
                    and job["pid"] is not None
                    and job["started_fingerprint"]
                ]
                if not evidence:
                    state.update(
                        observation="Launch outcome unknown; manual process reconciliation required.",
                        next_retry_at=now + 3600,
                    )
                    _save(store.root, brand, project, state)
                    return False
                state["claim"] = "idle"
                state["job_id"] = max(job["id"] for job in evidence)
            if automatic:
                transient = row["status"] == "waiting" and row.get("wait_kind") in {
                    "memory",
                    "admission",
                    "disk",
                }
                acknowledged_dead = any(
                    job["id"] == state.get("job_id")
                    and job["pid"] is not None
                    and job["started_fingerprint"]
                    and not _live(job_store, job)
                    for job in matching
                )
                crashed = (
                    row["status"] in {"interrupted", "running"}
                    or (row["status"] == "queued" and acknowledged_dead)
                ) and not worker_alive(row.get("worker"))
                if not transient and not crashed:
                    state.update(
                        enabled=False,
                        observation="Manual review required for this outcome; automatic retry stopped.",
                    )
                    _save(store.root, brand, project, state)
                    return False
                if crashed:
                    if state["crash_retries"] >= 3:
                        state.update(
                            enabled=False,
                            observation="Worker restart limit reached; manual review required.",
                        )
                        _save(store.root, brand, project, state)
                        return False
                    state["crash_retries"] += 1
            else:
                state["enabled"] = True
            # Persist before Popen. A killed launcher cannot forget uncertainty.
            state.update(
                claim="launching",
                jobs_floor=max([job["id"] for job in matching], default=0),
                last_launch_at=now,
                launcher_pid=os.getpid(),
                interpreter=sys.executable,
                build=row.get("build"),
                observation="Launching worker.",
            )
            _save(store.root, brand, project, state)
            # A control recorded during claim preparation is authoritative.
            controls = project_metadata(store.root, brand, project)
            if controls.get("cancel_requested"):
                state.update(
                    enabled=False,
                    claim="idle",
                    observation="Cancelled before launch; explicit Resume required.",
                )
                _save(store.root, brand, project, state)
                return False
            if controls.get("pause_requested"):
                state.update(enabled=False, claim="idle", observation="Paused before launch.")
                _save(store.root, brand, project, state)
                return False
            try:
                job_id = jobs.launch(
                    job_store,
                    "photos night",
                    argv,
                    settings.console_data_dir() / "job_logs",
                    env=settings.brand_launch_env(),
                    brand=brand,
                )
            except Exception as exc:
                state.update(
                    last_error=str(exc),
                    observation="Launch outcome unknown; manual process reconciliation required.",
                )
                _save(store.root, brand, project, state)
                raise
            state["launches"] += 1
            delay = min(3600, 60 * 2 ** min(state["launches"] - 1, 6))
            state.update(
                claim="idle",
                job_id=job_id,
                next_retry_at=now + delay,
                last_error=None,
                observation="Worker launched; next automatic check is persisted.",
            )
            _save(store.root, brand, project, state)
            return True
        finally:
            job_store._conn.close()
