"""An "interrupted" job records what this console SAW, not a verdict.

The console launches jobs out of process. When a row is still ``running`` and
no ``Popen`` handle is left -- after a restart, or because the child was never
this process's to begin with -- the console has no exit code and never will:
you cannot ``wait()`` on a process you did not fork. It nevertheless wrote
``interrupted``, which reads as "this job was cut short". Three genuinely
different situations were collapsed into that one word, and in two of them the
job may well have finished successfully.

These pin the evidence, and pin that the page stops asserting a conclusion the
console cannot support.
"""

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import jobs
from gomoveshift.console.app import create_app
from gomoveshift.console.jobs import JobStore


@pytest.fixture(autouse=True)
def _isolate_processes():
    jobs._PROCESSES.clear()
    yield
    jobs._PROCESSES.clear()


@pytest.fixture(autouse=True)
def _quiet_notify(monkeypatch):
    monkeypatch.setattr(jobs.notify, "notify_job_interrupted", lambda job: None)


@pytest.fixture
def sleeper():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def _store(tmp_path):
    return JobStore(tmp_path / "jobs.sqlite3")


def _dead_pid(sleeper):
    fingerprint = jobs._read_proc_starttime(sleeper.pid)
    pid = sleeper.pid
    sleeper.terminate()
    sleeper.wait(timeout=5)
    return pid, fingerprint


def test_a_pid_already_gone_at_startup_records_that_and_not_a_verdict(tmp_path, sleeper):
    """The console restarted and the process was not there. It may have
    finished its work perfectly before the restart -- nothing here can tell."""
    pid, fingerprint = _dead_pid(sleeper)
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    store.update(job_id, status="running", pid=pid, started_fingerprint=fingerprint)

    jobs.sweep_orphaned_jobs(store)

    assert store.get(job_id)["outcome_evidence"] == jobs.EVIDENCE_GONE_AT_STARTUP


def test_a_reused_pid_records_the_same_evidence_as_an_absent_one(tmp_path, sleeper):
    """A live pid under a different start-time fingerprint is a stranger's
    process. What this console knows is the same thing either way: the
    process it recorded is not there."""
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    wrong = jobs._read_proc_starttime(sleeper.pid) + "1"
    store.update(job_id, status="running", pid=sleeper.pid, started_fingerprint=wrong)

    jobs.sweep_orphaned_jobs(store)

    assert store.get(job_id)["outcome_evidence"] == jobs.EVIDENCE_GONE_AT_STARTUP


def test_a_row_with_no_pid_records_that_there_was_no_evidence(tmp_path):
    """A row written before pids were recorded. Calling this "interrupted"
    was always a guess; now it says so."""
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    store.update(job_id, status="running")

    jobs.sweep_orphaned_jobs(store)

    assert store.get(job_id)["outcome_evidence"] == jobs.EVIDENCE_NO_PID_RECORDED


def test_a_process_seen_alive_and_later_gone_is_recorded_as_exactly_that(tmp_path, sleeper):
    """The strongest evidence available, and still not an exit code: the
    sweep left the row running because the process was alive, and a later
    pass found it gone."""
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    fingerprint = jobs._read_proc_starttime(sleeper.pid)
    store.update(job_id, status="running", pid=sleeper.pid, started_fingerprint=fingerprint)
    assert jobs.sweep_orphaned_jobs(store) == []

    sleeper.terminate()
    sleeper.wait(timeout=5)
    jobs.reap_finished(store)

    row = store.get(job_id)
    assert row["status"] == "interrupted"
    assert row["outcome_evidence"] == jobs.EVIDENCE_EXITED_WHILE_WATCHED


@pytest.mark.parametrize(
    "evidence",
    [
        jobs.EVIDENCE_GONE_AT_STARTUP,
        jobs.EVIDENCE_NO_PID_RECORDED,
        jobs.EVIDENCE_EXITED_WHILE_WATCHED,
    ],
)
def test_no_sentence_claims_the_job_failed_or_was_cut_short(evidence):
    """Every sentence describes an observation and says the outcome is
    unknown. None of them asserts the job did not finish -- that is the
    whole defect."""
    sentence = jobs.interrupted_evidence_sentence(
        {"status": "interrupted", "outcome_evidence": evidence}
    )

    assert sentence
    lowered = sentence.lower()
    assert "unknown" in lowered or "cannot" in lowered or "never" in lowered
    for forbidden in ("failed", "cut short", "did not finish", "was killed"):
        assert forbidden not in lowered


def test_a_finished_job_has_no_evidence_sentence():
    """The accept-case. A job this console actually reaped has an exit
    code, so there is nothing to hedge about and nothing to say."""
    assert jobs.interrupted_evidence_sentence({"status": "done", "rc": 0}) is None
    assert jobs.interrupted_evidence_sentence({"status": "failed", "rc": 1}) is None
    assert jobs.interrupted_evidence_sentence({"status": "running"}) is None


def test_an_interrupted_row_from_before_this_change_still_renders(tmp_path):
    """Rows already in the table carry no evidence. They must still say
    something honest rather than raise or render an empty page."""
    sentence = jobs.interrupted_evidence_sentence(
        {"status": "interrupted", "outcome_evidence": None}
    )

    assert sentence
    assert "unknown" in sentence.lower()


def test_the_detail_page_says_what_was_seen(tmp_path, monkeypatch, sleeper):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    pid, fingerprint = _dead_pid(sleeper)
    store = JobStore(tmp_path / "jobs.sqlite3")
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    store.update(job_id, status="running", pid=pid, started_fingerprint=fingerprint)
    jobs.sweep_orphaned_jobs(store)

    response = TestClient(create_app()).get(f"/jobs/{job_id}")

    assert response.status_code == 200
    assert "no longer running when this console started" in response.text
    assert "unknown" in response.text.lower()


def test_the_detail_page_of_a_finished_job_carries_no_hedge(tmp_path, monkeypatch):
    """The accept-case on the page: a job with a real exit code must not
    grow a paragraph about what could not be determined."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    store = JobStore(tmp_path / "jobs.sqlite3")
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    store.update(job_id, status="done", rc=0)

    response = TestClient(create_app()).get(f"/jobs/{job_id}")

    assert response.status_code == 200
    assert "could not be determined" not in response.text.lower()
    assert "unknown" not in response.text.lower()


def test_the_column_is_added_to_a_table_that_predates_it(tmp_path):
    """Same idempotent-ALTER contract the pid columns already have: an
    existing DB can predate the column and has to pick it up."""
    import sqlite3

    path = tmp_path / "jobs.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, "
        "argv TEXT NOT NULL, status TEXT NOT NULL, rc INTEGER, created_at TEXT NOT NULL, "
        "finished_at TEXT, log_path TEXT)"
    )
    conn.commit()
    conn.close()

    JobStore(path)
    JobStore(path)

    conn = sqlite3.connect(path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    conn.close()
    assert "outcome_evidence" in cols
