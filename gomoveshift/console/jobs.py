"""Console job runner: SQLite-backed record of console-launched CLI jobs.

``launch()`` only Popens an allowlisted argv list (never a shell string); no
background thread, the live ``Popen`` is reaped on read. A GPU subcommand is not
Popen'd at all when a host spool is configured: it is written to the spool for
the host worker (``hostspool``) and its progress is read back from status files.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .. import run_origin
from . import hostspool, notify, runs, settings
from .night_job_evidence import project_outcome

logger = logging.getLogger(__name__)

HostRefusedError = hostspool.HostRefusedError

#: Subcommands the console is allowed to launch (with a real argv the
#: caller built). ``status`` is NOT a top-level ``gomoveshift`` subcommand
#: (see ``cli/__init__.py``'s ``_COMMAND_MAP``) and never belonged here.
ALLOWED_SUBCOMMANDS = {
    "drive",
    "photos",
    "sync-site",
    "publish-journal",
    "enqueue-plan",
    "dispatch-queue",
    "schedule",
    "publishing-status",
    "reconcile-playlists",
    "route-story-film",
}

#: Subset of ``ALLOWED_SUBCOMMANDS`` that genuinely runs with ZERO extra
#: args -- the only ones ``build_argv`` may hand to the generic job/
#: schedule form, which never collects any argument beyond the
#: subcommand name. Every other allowlisted subcommand has a required
#: positional or flag (``drive``'s directory, ``sync-site``'s
#: ``--site-path``, ...) and would guaranteed-fail if launched bare.
#: ``metrics`` is deliberately excluded -- it requires ``collect <project>``
#: and would guaranteed-fail bare (a parameterized argv would be needed to add it).
GENERIC_SUBCOMMANDS = {
    "dispatch-queue",
    "schedule",
    "publishing-status",
    "reconcile-playlists",
}

#: (store identity, job id) -> live Popen handle, for jobs launched by this
#: process. Every ``JobStore`` numbers its jobs from 1, so a bare job id names
#: a different job in each store and ``JobStore.identity`` is what tells them
#: apart. Read and write it only through ``_track``, ``tracked_process``,
#: ``_untrack`` and ``_tracked_job_ids``, which build the key in one place.
_PROCESSES: dict[tuple[Path, int], subprocess.Popen] = {}

#: Serializes ``run_due`` so a concurrent GET /jobs drain and ticker tick
#: (or two overlapping drains) can't double-launch the same due schedule.
_RUN_DUE_LOCK = threading.Lock()


def build_argv(subcommand: str) -> list[str]:
    """Build the argv for a subcommand safe to launch with zero args."""
    if subcommand not in GENERIC_SUBCOMMANDS:
        raise ValueError(f"subcommand not allowlisted: {subcommand!r}")
    return [sys.executable, "-m", "gomoveshift", subcommand]


def _validate_argv(argv: list[str]) -> None:
    """Refuse anything that isn't ``sys.executable -m gomoveshift <allowed>``."""
    prefix = [sys.executable, "-m", "gomoveshift"]
    if len(argv) < 4 or argv[:3] != prefix or argv[3] not in ALLOWED_SUBCOMMANDS:
        raise ValueError(f"argv not allowlisted: {argv!r}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """SQLite-backed record of console-launched jobs."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        #: Which database this store is: the resolved path of the file sqlite
        #: just opened, taken once. Stores opened on the same file share it
        #: however the path was spelled, so a process launched through one is
        #: found through another (the console builds a fresh store per request).
        self.identity = self.db_path.resolve()
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name TEXT NOT NULL, argv TEXT NOT NULL, status TEXT NOT NULL, rc INTEGER, "
            "created_at TEXT NOT NULL, finished_at TEXT, log_path TEXT)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schedules (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name TEXT NOT NULL, subcommand TEXT NOT NULL, argv TEXT NOT NULL, "
            "interval_minutes INTEGER NOT NULL, enabled INTEGER NOT NULL, last_run_at TEXT)"
        )
        self._ensure_brand_column()
        self._ensure_liveness_columns()
        self._ensure_rerun_column()
        self._ensure_host_columns()
        self._conn.commit()

    def _ensure_brand_column(self) -> None:
        """``CREATE TABLE IF NOT EXISTS`` is a no-op against a jobs table
        that predates this column -- the store is live on two hosts, so an
        idempotent ``ALTER TABLE`` is the only way an existing DB picks it
        up. Legacy rows get ``NULL`` (unknown), never a guessed brand."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        if "brand" not in cols:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN brand TEXT")

    def _ensure_liveness_columns(self) -> None:
        """A running row needs durable evidence of the child it
        launched -- its pid plus a process-start fingerprint, so a reused
        pid after a restart is never mistaken for the original child. Same
        idempotent-``ALTER TABLE`` pattern as ``_ensure_brand_column`` --
        the store is live on two hosts. Legacy rows get ``NULL`` (no
        evidence), never a guessed pid."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        if "pid" not in cols:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN pid INTEGER")
        if "started_fingerprint" not in cols:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN started_fingerprint TEXT")
        if "outcome_evidence" not in cols:
            # WHY a row was marked ``interrupted`` -- which of three
            # different observations the console actually made. NULL on every
            # row that predates this column and on every row that ended with a
            # real exit code, where there is nothing to hedge about.
            self._conn.execute("ALTER TABLE jobs ADD COLUMN outcome_evidence TEXT")

    def _ensure_rerun_column(self) -> None:
        """Which job this one was started again from, or NULL.

        Same idempotent ``ALTER TABLE`` as the two above -- the store is live
        on two hosts. NULL means "not started from another job", which is a
        different fact from "started from job 0"; nothing here ever guesses an
        origin for a row written before the column existed.
        """
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        if "rerun_of" not in cols:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN rerun_of INTEGER")

    def _ensure_host_columns(self) -> None:
        """Host-executor jobs: which executor ran it (NULL = this process), its name in the
        spool, the worker's last progress/phase, and the worker's error text."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        for name, kind in (
            ("executor", "TEXT"),
            ("spool_id", "TEXT"),
            ("progress", "REAL"),
            ("phase", "TEXT"),
            ("host_error", "TEXT"),
        ):
            if name not in cols:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {kind}")  # noqa: S608

    def create(
        self,
        name: str,
        argv: list[str],
        log_path: str,
        brand: str | None = None,
        rerun_of: int | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO jobs (name, argv, status, created_at, log_path, brand, rerun_of) "
            "VALUES (?, ?, 'queued', ?, ?, ?, ?)",
            (name, json.dumps(argv), _now(), log_path, brand, rerun_of),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    #: `fields` in `update()` is internal-only (status/rc/finished_at/log_path), never user input.
    def update(self, job_id: int, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        query = f"UPDATE jobs SET {cols} WHERE id = ?"  # noqa: S608
        self._conn.execute(query, (*fields.values(), job_id))
        self._conn.commit()

    def get(self, job_id: int) -> dict | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def list(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
        return [_row_to_dict(row) for row in rows]

    def create_schedule(
        self, name: str, subcommand: str, argv: list[str], interval_minutes: int
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO schedules (name, subcommand, argv, interval_minutes, enabled, "
            "last_run_at) VALUES (?, ?, ?, ?, 1, NULL)",
            (name, subcommand, json.dumps(argv), interval_minutes),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_schedules(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM schedules ORDER BY id").fetchall()
        return [_schedule_row_to_dict(row) for row in rows]

    def toggle_schedule(self, schedule_id: int) -> None:
        self._conn.execute(
            "UPDATE schedules SET enabled = 1 - enabled WHERE id = ?", (schedule_id,)
        )
        self._conn.commit()

    def delete_schedule(self, schedule_id: int) -> None:
        self._conn.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
        self._conn.commit()

    def mark_schedule_ran(self, schedule_id: int, when: str) -> None:
        self._conn.execute("UPDATE schedules SET last_run_at = ? WHERE id = ?", (when, schedule_id))
        self._conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["argv"] = json.loads(data["argv"])
    return data


def _schedule_row_to_dict(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["argv"] = json.loads(data["argv"])
    data["enabled"] = bool(data["enabled"])
    return data


def _process_key(store: JobStore, job_id: int) -> tuple[Path, int]:
    """The one place a ``_PROCESSES`` key is built: database file plus job id."""
    return (store.identity, job_id)


def _track(store: JobStore, job_id: int, proc: subprocess.Popen) -> None:
    _PROCESSES[_process_key(store, job_id)] = proc


def tracked_process(store: JobStore, job_id: int) -> subprocess.Popen | None:
    """The ``Popen`` this process launched for ``job_id`` of ``store``, or
    ``None`` (never launched here, already reaped, or launched before a restart)."""
    return _PROCESSES.get(_process_key(store, job_id))


def _untrack(store: JobStore, job_id: int) -> None:
    _PROCESSES.pop(_process_key(store, job_id), None)


def _tracked_job_ids(store: JobStore) -> list[int]:
    """Ids of ``store``'s jobs with a tracked process. ``list(_PROCESSES)``
    snapshots the keys first: the ticker and request threads launch and reap
    concurrently, and iterating the live dict could raise mid-resize."""
    return [job_id for identity, job_id in list(_PROCESSES) if identity == store.identity]


def launch(
    store: JobStore,
    name: str,
    argv: list[str],
    log_dir: Path,
    env: dict | None = None,
    brand: str | None = None,
    rerun_of: int | None = None,
    executor: str | None = None,
) -> int:
    """Validate ``argv``, Popen it with output appended to a log file, record it.

    ``executor="host"`` (the default for a GPU subcommand when ``GMS_HOST_SPOOL_DIR`` is set)
    writes a request for the host worker instead of forking, and raises ``HostRefusedError``
    -- never falling back to a local Popen -- when the spool is unusable, the worker's
    heartbeat is stale, or the same argv for the same brand is already pending or running.

    ``env=None`` (the default) inherits this process's environment, same as
    before; pass ``settings.brand_launch_env()`` to apply the launching
    brand's saved overrides. ``brand`` is recorded on the job row so a
    later rerun can replay the same brand it actually launched as --
    defaults to the current instance's brand, matching the default ``env``.

    ``rerun_of`` is the job this one was started again from, recorded so the
    detail page can say so. It changes nothing about how the job runs: the
    ``argv`` handed in is the old job's, while ``env`` is resolved by the
    caller from the settings as they stand now. Those are two different
    moments, and the column is what lets a reader see that they are.
    """
    _validate_argv(argv)
    if brand is None:
        brand = settings.current_brand()
    if executor is None and hostspool.is_gpu_argv(argv) and hostspool.spool_dir() is not None:
        executor = "host"
    if executor == "host":
        return _launch_host(store, name, argv, brand, rerun_of)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    job_id = store.create(name, argv, log_path="", brand=brand, rerun_of=rerun_of)
    log_path = log_dir / f"{job_id}.log"
    log_fh = log_path.open("ab")
    # The child records itself as console-launched, so its own run_origin.json agrees with the job row.
    child_env = {**(os.environ if env is None else env), run_origin.ORIGIN_ENV: "console"}
    if name == "drive":  # the drive gate accepts only tracked launchers
        child_env[run_origin.LAUNCHED_BY_ENV] = "console"
    proc = subprocess.Popen(argv, stdout=log_fh, stderr=subprocess.STDOUT, env=child_env)
    _track(store, job_id, proc)
    # ``getattr`` rather than ``proc.pid`` -- test doubles that fake a
    # finished ``Popen`` don't always implement ``.pid``.
    pid = getattr(proc, "pid", None)
    fingerprint = _read_proc_starttime(pid) if pid is not None else None
    store.update(
        job_id,
        status="running",
        log_path=str(log_path),
        pid=pid,
        started_fingerprint=fingerprint,
    )
    return job_id


def _launch_host(
    store: JobStore, name: str, argv: list[str], brand: str | None, rerun_of: int | None
) -> int:
    spool = hostspool.require_ready(hostspool.spool_dir())
    key = hostspool.dedupe_key(argv, brand)
    twin = hostspool.active_duplicate(spool, key)
    if twin is not None:
        raise HostRefusedError(
            f"the same command for this brand is already queued or running ({twin}); "
            "wait for it to finish or cancel it"
        )
    job_id = store.create(name, argv, log_path="", brand=brand, rerun_of=rerun_of)
    sid = hostspool.spool_id(brand, job_id)
    store.update(
        job_id,
        executor="host",
        spool_id=sid,
        log_path=str(hostspool.log_path(spool, sid)),
    )
    try:
        hostspool.write_request(
            spool, sid, brand, argv, key, env=settings.brand_env_overrides(brand)
        )
    except OSError as exc:
        store.update(job_id, status="failed", finished_at=_now(), host_error=str(exc))
        raise HostRefusedError(f"could not write the request to the host spool: {exc}") from exc
    return job_id


def _read_proc_starttime(pid: int) -> str | None:
    """Field 22 (``starttime``) of ``/proc/<pid>/stat`` -- fingerprints
    "this pid, this process" (not just this pid, since pids get reused).
    ``None`` if the pid is already gone. On Windows (no ``/proc``) the psutil
    process create time stands in, as a string."""
    if sys.platform == "win32":
        try:
            import psutil

            return str(psutil.Process(pid).create_time())
        except Exception:  # noqa: BLE001 - gone/denied/psutil missing: no fingerprint
            return None
    try:
        content = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # ``comm`` (field 2) is parenthesized and may itself hold ")"/spaces --
    # split after the LAST ")"; field 22 is then 19 further along.
    rparen = content.rfind(")")
    if rparen == -1:
        return None
    fields = content[rparen + 2 :].split()
    return fields[19] if len(fields) > 19 else None


def _process_matches(pid: int, fingerprint: str | None) -> bool:
    """True only if ``pid`` is alive AND is still the process that
    recorded ``fingerprint`` -- a live pid with no/mismatched fingerprint
    is a stranger's process (pid reuse), not the job's own child."""
    return fingerprint is not None and _read_proc_starttime(pid) == fingerprint


#: The host worker stopped reporting while a job was in flight, or its request vanished
#: without a status. Like the other evidence values this is what was SEEN.
EVIDENCE_HOST_WORKER_LOST = "host_worker_lost"


def _is_host(row: dict | None) -> bool:
    return bool(row) and row.get("executor") == "host"


def _reap_host(store: JobStore, row: dict) -> None:
    """Settle a host job from the worker's status file (the console never has an exit code
    of its own for it)."""
    job_id = row["id"]
    spool = hostspool.spool_dir()
    if spool is None or row["status"] not in ("queued", "running"):
        return
    sid = row["spool_id"]
    status = hostspool.read_status(spool, sid)
    state = (status or {}).get("state")
    if state in hostspool.TERMINAL_STATES:
        rc = status.get("exit_code")
        fields = {"progress": status.get("progress"), "phase": status.get("phase")}
        if state == "succeeded":
            store.update(job_id, status="done", rc=rc, finished_at=_now(), **fields)
            return
        error = status.get("error")
        if state == "cancelled":
            store.update(job_id, status="cancelled", rc=rc, finished_at=_now(), **fields)
            return
        store.update(job_id, status="failed", rc=rc, finished_at=_now(), host_error=error, **fields)
        notify.notify_job_failed(store.get(job_id))
        return
    if state == "running":
        store.update(
            job_id,
            status="running",
            progress=status.get("progress"),
            phase=status.get("phase"),
        )
        if hostspool.worker_gone_reason(spool):
            _mark_interrupted(store, job_id, EVIDENCE_HOST_WORKER_LOST)
        return
    if not hostspool.request_present(spool, sid):
        # Neither queued nor running nor a status: it was taken and lost, or deleted.
        _mark_interrupted(store, job_id, EVIDENCE_HOST_WORKER_LOST)


def _reap(store: JobStore, job_id: int) -> None:
    row = store.get(job_id)
    if _is_host(row):
        _reap_host(store, row)
        return
    proc = tracked_process(store, job_id)
    rc = proc.poll() if proc is not None else None
    if proc is None or rc is None:
        return
    evidence = project_outcome(store.get(job_id) or {}) if rc != 0 else None
    if evidence and evidence["status"] in {"paused", "waiting", "cancelled"}:
        store.update(job_id, **evidence, finished_at=_now())
        _untrack(store, job_id)
        return
    store.update(job_id, status="done" if rc == 0 else "failed", rc=rc, finished_at=_now())
    _untrack(store, job_id)
    if rc != 0:
        row = store.get(job_id)
        if row is not None:
            notify.notify_job_failed(row)


#: The console launches jobs out of process, so for a row it did not fork it
#: can never obtain an exit code -- ``wait()`` belongs to the parent. It wrote
#: ``interrupted`` anyway, which reads as "this job was cut short", and it
#: collapsed three genuinely different observations into that one word. In two
#: of them the job may have finished all its work successfully. These record
#: what was SEEN; the status stays ``interrupted`` because it is the terminal
#: bucket the list page and the rerun guard already understand.

#: The process was not there when the console started up -- either absent, or
#: alive under a start-time fingerprint that makes it a stranger's process
#: holding a reused pid. Both are the same knowledge: what was recorded is gone.
EVIDENCE_GONE_AT_STARTUP = "gone_at_startup"

#: The row carries no pid at all (written before pids were recorded). No
#: observation was ever possible for it.
EVIDENCE_NO_PID_RECORDED = "no_pid_recorded"

#: The startup sweep saw the process alive, and a later pass found it gone.
#: The strongest evidence available here, and still not an exit code.
EVIDENCE_EXITED_WHILE_WATCHED = "exited_while_watched"

_EVIDENCE_SENTENCES = {
    EVIDENCE_GONE_AT_STARTUP: (
        "the process this job launched was no longer running when this console "
        "started, so whether it finished its work is unknown"
    ),
    EVIDENCE_NO_PID_RECORDED: (
        "this console never recorded a process id for this job, so whether it "
        "finished its work is unknown"
    ),
    EVIDENCE_HOST_WORKER_LOST: (
        "the host worker stopped reporting for this job (or its request disappeared) before "
        "it wrote a final status, so whether it finished its work is unknown"
    ),
    EVIDENCE_EXITED_WHILE_WATCHED: (
        "the process this job launched was running and has since exited; this "
        "console is not its parent, so it never saw an exit code and whether "
        "the work completed is unknown"
    ),
}

#: An ``interrupted`` row written before this column existed.
_EVIDENCE_UNRECORDED = (
    "this job was marked interrupted before this console recorded why, so what "
    "happened to it is unknown"
)


def interrupted_evidence_sentence(row: dict) -> str | None:
    """What the console actually observed about ``row``, or ``None``.

    Only an ``interrupted`` row gets a sentence: every other terminal status
    came from a real exit code this console read itself, and hedging about one
    of those would be noise. Returned as a sentence rather than a code so the
    page states the observation in the operator's words instead of showing a
    status that asserts more than anyone here knows.
    """
    if row.get("status") != "interrupted":
        return None
    return _EVIDENCE_SENTENCES.get(row.get("outcome_evidence"), _EVIDENCE_UNRECORDED)


def _mark_interrupted(store: JobStore, job_id: int, evidence: str) -> bool:
    durable = project_outcome(store.get(job_id) or {})
    if durable:
        store.update(job_id, **durable, finished_at=_now())
        if durable["status"] == "failed":
            notify.notify_job_failed(store.get(job_id))
        return False
    store.update(job_id, status="interrupted", finished_at=_now(), outcome_evidence=evidence)
    notify.notify_job_interrupted(store.get(job_id))
    return True


def sweep_orphaned_jobs(store: JobStore) -> list[int]:
    """Reconcile DB rows stuck ``running`` with no live in-process
    ``Popen`` handle -- ``_PROCESSES`` is in-memory only, so a restart
    empties it whether or not the child actually died. Call once at
    startup, before anything else reads the jobs table.

    No pid recorded (legacy row) -> no evidence either way, interrupted,
    as before. Pid alive with a matching fingerprint -> genuinely still
    running, left ``running``, no notify. Pid gone or alive under a
    mismatched fingerprint (pid reuse) -> interrupted. A row left
    ``running`` has no ``Popen`` to reap; ``reap_finished`` re-checks it
    on every later pass so it isn't stranded forever."""
    interrupted = []
    for row in store.list():
        if row["status"] != "running" or tracked_process(store, row["id"]) is not None:
            continue
        if _is_host(row):
            continue
        pid = row["pid"]
        if pid is None:
            if _mark_interrupted(store, row["id"], EVIDENCE_NO_PID_RECORDED):
                interrupted.append(row["id"])
        elif not _process_matches(pid, row["started_fingerprint"]):
            if _mark_interrupted(store, row["id"], EVIDENCE_GONE_AT_STARTUP):
                interrupted.append(row["id"])
    return interrupted


def _reconcile_untracked_running(store: JobStore) -> None:
    """Re-check every ``running`` row with no ``Popen`` (left running by
    ``sweep_orphaned_jobs`` on live evidence) so a since-exited outside
    process is eventually marked ``interrupted``, not stranded forever. A
    pid-less legacy row was already resolved by the startup sweep."""
    for row in store.list():
        if row["status"] != "running" or tracked_process(store, row["id"]) is not None:
            continue
        if _is_host(row):
            continue
        pid = row["pid"]
        if pid is not None and not _process_matches(pid, row["started_fingerprint"]):
            _mark_interrupted(store, row["id"], EVIDENCE_EXITED_WHILE_WATCHED)


def reap_finished(store: JobStore) -> None:
    """Reap every process this instance is tracking for ``store`` whose
    ``Popen`` has exited, updating its DB row and notifying on a nonzero rc.
    Another store's processes are not this store's to settle. Shared by
    ``get_job``/``list_jobs`` (pull) and the ``Ticker`` (push, so a failure
    notifies within one tick instead of waiting for someone to open /jobs).
    Also closes the loop on rows ``sweep_orphaned_jobs`` left ``running``
    with no ``Popen`` handle -- see ``_reconcile_untracked_running``."""
    for job_id in _tracked_job_ids(store):
        _reap(store, job_id)
    for row in store.list():
        if _is_host(row) and row["status"] in ("queued", "running"):
            _reap_host(store, row)
    _reconcile_untracked_running(store)


def cancel_job(store: JobStore, job_id: int) -> None:
    """Ask the host worker to stop a queued or running host job by writing its cancel marker.

    The worker does the stopping; the row settles to ``cancelled`` on a later reap, from the
    worker's status file. Only host jobs can be cancelled this way."""
    row = store.get(job_id)
    if row is None:
        raise ValueError(f"Unknown job: {job_id}")
    if not _is_host(row):
        raise ValueError(
            f"job {job_id} did not run on the host worker; it cannot be cancelled here"
        )
    if row["status"] not in ("queued", "running"):
        raise ValueError(f"job {job_id} has already finished ({row['status']})")
    spool = hostspool.require_ready(hostspool.spool_dir())
    hostspool.write_cancel(spool, row["spool_id"])


#: The one status a rerun must refuse. Everything else -- done, failed,
#: interrupted -- is exactly what the button is for.
_RUNNING = "running"


def rerun_blocked_reason(row: dict) -> str | None:
    """Why ``row`` must not be relaunched, or ``None`` when it may be.

    A job still ``running`` is the whole list. The rerun route used to read
    the row and launch it, checking nothing, so a second click while a
    ``dispatch-queue --apply`` job was mid-drain started a second drain over
    the same queue -- and the second worker picks up rows the first is
    already publishing, which is the double-post the lease exists to stop.

    Returned as a reason rather than a bool so the route's 409 and the
    detail page's hidden button say the same thing, and the page never
    offers a control that is going to refuse.
    """
    if row.get("status") == _RUNNING:
        return (
            f"job {row.get('id')} is still running; re-running it now would start a "
            "second copy of the same work. Wait for it to finish, or stop it first"
        )
    return None


#: Seconds since a `running` job's log was last written before its liveness
#: label switches from "working" to "may be stalled". Modeled on
#: ``ffmpeg_runner.py``'s per-stage ``stall_timeout`` default (600s): a
#: console job launches a whole ``gomoveshift`` subcommand, which is at
#: least as coarse-grained as a single ffmpeg stage, so the same window
#: isn't too eager to flag a merely-quiet-but-healthy job. (``uploader.py``'s
#: ``_DRIVE_STALL_TIMEOUT`` of 120s is the other precedent, but that watches
#: one upload's byte progress, a tighter-grained signal than a job log.)
STALL_THRESHOLD_SEC = 600


def _format_age(seconds: int) -> str:
    """Render a duration for the liveness label at the precedent's
    granularity (seconds under a minute, else whole minutes) -- never false
    precision."""
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m"


def _host_liveness(row: dict) -> str:
    progress, phase = row.get("progress"), row.get("phase")
    if progress is None and not phase:
        return "running on the host worker (no progress reported yet)"
    parts = []
    if progress is not None:
        parts.append(f"{round(float(progress) * 100)}%")
    if phase:
        parts.append(str(phase))
    return "running on the host worker: " + " -- ".join(parts)


def liveness_label(row: dict, now: datetime | None = None) -> str | None:
    """Evidence about a running job's log, or ``None`` when there is none.

    Compares ``now`` (defaults to the real UTC clock; tests inject a fixed
    value so the threshold boundary is deterministic) against the mtime of
    ``row["log_path"]`` -- the one per-job signal every subcommand's log
    already provides, with nothing new to wire up. The label states only
    what was observed: a log written within ``STALL_THRESHOLD_SEC`` reads
    "working", one older reads "may be stalled". It never claims the
    process is dead -- that verdict belongs to ``sweep_orphaned_jobs``,
    unchanged. A missing or unreadable ``log_path`` yields ``None``, never
    a "stalled" claim it cannot back up.
    """
    if _is_host(row):
        return _host_liveness(row)
    log_path = row.get("log_path")
    if not log_path:
        return None
    try:
        mtime = Path(log_path).stat().st_mtime
    except OSError:
        return None
    if now is None:
        now = datetime.now(timezone.utc)
    age_seconds = max(0, int(now.timestamp() - mtime))
    age = _format_age(age_seconds)
    if age_seconds <= STALL_THRESHOLD_SEC:
        return f"working (log updated {age} ago)"
    return f"quiet {age} -- may be stalled"


def _with_liveness(row: dict | None) -> dict | None:
    """Add a ``liveness`` key to ``row`` when (and only when) it is
    ``running`` -- every other status renders exactly as before, with no
    new key at all, not merely one that happens to be falsy."""
    if row is not None and _is_host(row) and row.get("status") == "queued":
        return {**row, "liveness": "waiting for the host worker to pick this up"}
    if row is None or row.get("status") != "running":
        return row
    return {**row, "liveness": liveness_label(row)}


def get_job(store: JobStore, job_id: int) -> dict | None:
    """Return the job row, reaping a finished process first."""
    _reap(store, job_id)
    return _with_liveness(store.get(job_id))


def list_jobs(store: JobStore) -> list[dict]:
    """Return all job rows, reaping any finished processes first."""
    reap_finished(store)
    return [_with_liveness(row) for row in store.list()]


def add_schedule(store: JobStore, name: str, subcommand: str, interval_minutes: int) -> int:
    """Add a schedule that launches ``subcommand``'s default argv on an interval."""
    # The form marks the interval `min="1"`, which a browser enforces and a
    # direct POST does not. Zero or negative meant a schedule due on every tick,
    # forever -- the console's own guard belongs here, beside the allowlist.
    if interval_minutes < 1:
        raise ValueError(f"Interval (minutes): {interval_minutes} is not at least 1")
    argv = build_argv(subcommand)  # raises ValueError on a non-allowlisted subcommand
    # A second identical POST (double-click, retry) used to add a second row that ran twice.
    for row in store.list_schedules():
        if (row["name"], row["subcommand"], row["interval_minutes"]) == (
            name,
            subcommand,
            interval_minutes,
        ):
            raise ValueError(
                f"A schedule named {name!r} for {subcommand} every {interval_minutes} "
                "minutes already exists"
            )
    return store.create_schedule(name, subcommand, argv, interval_minutes)


def list_schedules(store: JobStore) -> list[dict]:
    return store.list_schedules()


def toggle_schedule(store: JobStore, schedule_id: int) -> None:
    store.toggle_schedule(schedule_id)


def delete_schedule(store: JobStore, schedule_id: int) -> None:
    store.delete_schedule(schedule_id)


def due_schedules(store: JobStore, now: datetime) -> list[dict]:
    """Return enabled schedules that have never run, or are past their interval."""
    due = []
    for row in store.list_schedules():
        if not row["enabled"]:
            continue
        if row["last_run_at"] is None:
            due.append(row)
            continue
        last_run = datetime.fromisoformat(row["last_run_at"])
        if last_run + timedelta(minutes=row["interval_minutes"]) <= now:
            due.append(row)
    return due


def run_due(store: JobStore, now: datetime, log_dir: Path) -> list[int]:
    """Launch every due schedule's argv through the allowlisted ``launch()``, stamping it ran.

    Locked: the GET /jobs opportunistic drain and the background
    ``Ticker`` can otherwise race and double-launch the same due
    schedule.
    """
    with _RUN_DUE_LOCK:
        launched = []
        for row in due_schedules(store, now):
            argv = build_argv(row["subcommand"])
            job_id = launch(
                store, row["subcommand"], argv, log_dir, env=settings.brand_launch_env()
            )
            store.mark_schedule_ran(row["id"], _now())
            launched.append(job_id)
        return launched


class Ticker:
    """Background daemon thread that calls ``run_due`` every ``tick_seconds``
    until ``stop()`` -- the replacement for the opportunistic
    ``run_due`` call on ``GET /jobs``. Slot precision is NOT this thread's
    job: it lives in the delivery queue rows (``publishing/queue.py``'s
    ``next_attempt_at``), so a schedule only needs to drain often enough,
    never at the exact minute."""

    def __init__(
        self,
        store: JobStore,
        log_dir: Path,
        tick_seconds: float,
        wait: Callable[[float], bool] | None = None,
    ) -> None:
        self._store = store
        self._log_dir = log_dir
        self._tick_seconds = tick_seconds
        self._stop_event = threading.Event()
        #: Tick-boundary wait seam -- defaults to the real stop-aware sleep;
        #: tests inject a lock-step fake so tick counts are exact instead of
        #: a wall-clock race (see ``StepWaiter`` in test_console_ticker.py).
        self._wait = wait if wait is not None else self._stop_event.wait
        self._thread = threading.Thread(target=self._run, daemon=True)
        #: Timestamp of the last tick/observer pass that completed without
        #: raising -- /healthz's staleness check, since a wedged tick body
        #: can leave the thread ``is_alive()`` forever.
        self.last_tick_ok_at: datetime | None = None
        self.last_observer_ok_at: datetime | None = None
        self.observer_failures = 0
        self.last_night_ok_at: datetime | None = None
        self.last_night_error: str | None = None
        #: Process-start reference for /healthz -- a never-completed first
        #: pass otherwise looks healthy forever (``last_tick_ok_at is None``).
        self.started_at: datetime = datetime.now(timezone.utc)

    def _observe_gates(self) -> None:
        """Alert for pending review gates in every known run dir -- console jobs and runs the
        console did not launch (CLI/cron), via the backend-run registry."""
        notify.check_pending_gates(runs.known_output_dirs(self._store))

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self, timeout: float = 5.0) -> None:
        """Idempotent: safe to call more than once."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                run_due(self._store, datetime.now(timezone.utc), self._log_dir)
                reap_finished(self._store)
                self.last_tick_ok_at = datetime.now(timezone.utc)
            except Exception:
                # One bad tick (a due schedule's argv build blows up, a
                # transient DB error, ...) must not kill the thread --
                # /healthz's ``ticker_alive`` is the only signal an
                # operator has that draining stopped.
                logger.exception("console ticker tick failed; continuing")
            try:
                from .night_supervision import supervise

                supervise()
                self.last_night_ok_at = datetime.now(timezone.utc)
                self.last_night_error = None
            except Exception as exc:  # noqa: BLE001 -- independent observer
                self.last_night_error = str(exc)
                logger.exception("console night supervision skipped; continuing")
            try:
                notify.check_queue_regressions()
                self.last_observer_ok_at = datetime.now(timezone.utc)
            except Exception:  # noqa: BLE001 -- an observer must never kill the drain
                self.observer_failures += 1
                logger.exception("console ticker queue observer skipped; continuing")
            try:
                self._observe_gates()
                self.last_observer_ok_at = datetime.now(timezone.utc)
            except Exception:  # noqa: BLE001 -- an observer must never kill the drain
                self.observer_failures += 1
                logger.exception("console ticker gate observer skipped; continuing")
            try:
                # The only observer that fires on something NOT happening.
                # It runs after run_due above, so a slot this very tick drained
                # is already out of the window it would be judged against.
                notify.check_missed_slots()
                self.last_observer_ok_at = datetime.now(timezone.utc)
            except Exception:  # noqa: BLE001 -- an observer must never kill the drain
                self.observer_failures += 1
                logger.exception("console ticker missed-slot observer skipped; continuing")
            self._wait(self._tick_seconds)


def start_ticker(store: JobStore, log_dir: Path, tick_seconds: float) -> Ticker:
    """Build and start a ``Ticker`` draining due schedules on an interval."""
    ticker = Ticker(store, log_dir, tick_seconds)
    ticker.start()
    return ticker


def tail_log(log_path: str | Path, lines: int = 100) -> list[str]:
    """Return the last ``lines`` lines of a job's log file (empty if missing)."""
    path = Path(log_path)
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-lines:]
