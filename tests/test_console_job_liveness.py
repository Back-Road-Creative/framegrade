"""Restart reconciliation for console jobs.

``_PROCESSES`` is in-memory only, so after a console restart every row
still marked ``running`` has no live ``Popen`` handle -- but that is not
evidence the child process actually exited. These tests pin the new
contract: a running row whose recorded pid still belongs to the same
process (matched by /proc start-time fingerprint) survives the sweep;
only genuine absence or a mismatched fingerprint (pid reuse) is treated
as interrupted.
"""

import sqlite3
import subprocess
import sys
import time

import pytest

from gomoveshift.console import jobs
from gomoveshift.console.jobs import JobStore


def _store(tmp_path):
    return JobStore(tmp_path / "jobs.sqlite3")


@pytest.fixture(autouse=True)
def _isolate_processes():
    jobs._PROCESSES.clear()
    yield
    jobs._PROCESSES.clear()


@pytest.fixture
def sleeper():
    """A real short-lived child process, terminated at teardown."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_live_process_with_matching_fingerprint_survives_sweep(tmp_path, monkeypatch, sleeper):
    calls = []
    monkeypatch.setattr(jobs.notify, "notify_job_interrupted", lambda job: calls.append(job))
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    fingerprint = jobs._read_proc_starttime(sleeper.pid)
    assert fingerprint is not None
    store.update(job_id, status="running", pid=sleeper.pid, started_fingerprint=fingerprint)

    interrupted = jobs.sweep_orphaned_jobs(store)

    assert interrupted == []
    row = store.get(job_id)
    assert row["status"] == "running"
    assert row["rc"] is None
    assert calls == []


def test_dead_pid_becomes_interrupted_and_notifies(tmp_path, monkeypatch, sleeper):
    calls = []
    monkeypatch.setattr(jobs.notify, "notify_job_interrupted", lambda job: calls.append(job))
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    fingerprint = jobs._read_proc_starttime(sleeper.pid)
    pid = sleeper.pid
    sleeper.terminate()
    sleeper.wait(timeout=5)

    store.update(job_id, status="running", pid=pid, started_fingerprint=fingerprint)

    interrupted = jobs.sweep_orphaned_jobs(store)

    assert interrupted == [job_id]
    row = store.get(job_id)
    assert row["status"] == "interrupted"
    assert row["rc"] is None
    assert len(calls) == 1


def test_pid_reuse_mismatch_becomes_interrupted(tmp_path, monkeypatch, sleeper):
    """The pid is alive, but it isn't the process this row launched --
    a stranger's process is not your job."""
    calls = []
    monkeypatch.setattr(jobs.notify, "notify_job_interrupted", lambda job: calls.append(job))
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    real_fingerprint = jobs._read_proc_starttime(sleeper.pid)
    wrong_fingerprint = real_fingerprint + "1"
    store.update(job_id, status="running", pid=sleeper.pid, started_fingerprint=wrong_fingerprint)

    interrupted = jobs.sweep_orphaned_jobs(store)

    assert interrupted == [job_id]
    row = store.get(job_id)
    assert row["status"] == "interrupted"
    assert row["rc"] is None
    assert len(calls) == 1


def test_legacy_null_pid_becomes_interrupted(tmp_path, monkeypatch):
    """A row written before this change carries no pid -- no evidence,
    so the sweep preserves today's behaviour."""
    calls = []
    monkeypatch.setattr(jobs.notify, "notify_job_interrupted", lambda job: calls.append(job))
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    store.update(job_id, status="running")

    interrupted = jobs.sweep_orphaned_jobs(store)

    assert interrupted == [job_id]
    row = store.get(job_id)
    assert row["status"] == "interrupted"
    assert row["rc"] is None
    assert len(calls) == 1


def test_left_running_row_is_interrupted_by_reap_once_process_exits(tmp_path, monkeypatch, sleeper):
    """A row the sweep leaves ``running`` has no ``Popen`` to reap -- prove
    the normal reap path picks it back up once the outside process exits,
    instead of stranding it ``running`` forever."""
    calls = []
    monkeypatch.setattr(jobs.notify, "notify_job_interrupted", lambda job: calls.append(job))
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    fingerprint = jobs._read_proc_starttime(sleeper.pid)
    store.update(job_id, status="running", pid=sleeper.pid, started_fingerprint=fingerprint)

    assert jobs.sweep_orphaned_jobs(store) == []
    assert store.get(job_id)["status"] == "running"

    sleeper.terminate()
    sleeper.wait(timeout=5)

    jobs.reap_finished(store)

    row = store.get(job_id)
    assert row["status"] == "interrupted"
    assert row["rc"] is None
    assert len(calls) == 1


def test_terminal_rows_never_touched_by_reconciliation(tmp_path, sleeper):
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    fingerprint = jobs._read_proc_starttime(sleeper.pid)
    sleeper.terminate()
    sleeper.wait(timeout=5)
    store.update(job_id, status="done", rc=0, pid=sleeper.pid, started_fingerprint=fingerprint)

    assert jobs.sweep_orphaned_jobs(store) == []
    jobs.reap_finished(store)

    assert store.get(job_id)["status"] == "done"


def test_alter_table_idempotent_across_two_opens(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    JobStore(path)
    JobStore(path)  # must not raise "duplicate column"


def test_alter_table_idempotent_against_old_schema(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "name TEXT NOT NULL, argv TEXT NOT NULL, status TEXT NOT NULL, rc INTEGER, "
        "created_at TEXT NOT NULL, finished_at TEXT, log_path TEXT)"
    )
    conn.commit()
    conn.close()

    store = JobStore(path)  # must add brand/pid/started_fingerprint without error
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    assert store.get(job_id)["status"] == "queued"
