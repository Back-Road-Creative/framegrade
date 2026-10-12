"""A rerun says whose settings it ran under.

"Re-run" replays the original job's ``argv`` but resolves ``brand_launch_env``
fresh, so it is neither of the two things an operator could mean by the word.
It is not "retry that job" -- a setting saved since then silently applies. It
is not "start again with current settings" -- the command line is the old one,
written before those settings existed. Nothing on either page said which, or
that the new job came from an older one at all.

The fix does not change what a rerun DOES: the hybrid is the useful behaviour
(replay this command, under the configuration this instance has now). It makes
the hybrid legible -- the new row records which job it came from, and the page
states both halves.
"""

import sqlite3

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


def _store(tmp_path):
    return JobStore(tmp_path / "jobs.sqlite3")


def _finished(store, *, rerun_of=None):
    job_id = store.create(
        "drive", ["/bin/true"], str(store.db_path.parent / "x.log"), rerun_of=rerun_of
    )
    store.update(job_id, status="done", rc=0)
    return job_id


def test_a_plain_launch_records_no_origin(tmp_path):
    """``None``, never 0 and never the job's own id: "this did not come from
    another job" is a different fact from "this came from job 0"."""
    store = _store(tmp_path)

    job_id = store.create("drive", ["/bin/true"], "")

    assert store.get(job_id)["rerun_of"] is None


def test_the_store_persists_the_origin(tmp_path):
    store = _store(tmp_path)
    original = _finished(store)

    rerun = store.create("drive", ["/bin/true"], "", rerun_of=original)

    assert store.get(rerun)["rerun_of"] == original


def test_an_existing_store_gains_the_column(tmp_path):
    """An existing store can predate the column, so it has to arrive by an
    idempotent ``ALTER`` against a table that predates it -- the same way
    ``brand``, ``pid`` and ``outcome_evidence`` did. A legacy row gets NULL
    (unknown), never a guessed origin."""
    path = tmp_path / "jobs.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, "
        "argv TEXT NOT NULL, status TEXT NOT NULL, rc INTEGER, created_at TEXT NOT NULL, "
        "finished_at TEXT, log_path TEXT)"
    )
    conn.execute(
        "INSERT INTO jobs (name, argv, status, created_at, log_path) "
        "VALUES ('drive', '[]', 'done', '2026-01-01T00:00:00', '')"
    )
    conn.commit()
    conn.close()

    store = JobStore(path)

    assert store.get(1)["rerun_of"] is None


def test_launch_threads_the_origin_through_to_the_row(tmp_path, monkeypatch):
    """``launch`` is what the route calls; the column is useless if the
    origin stops at ``create``."""
    store = _store(tmp_path)
    original = _finished(store)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakeProc())

    new_id = jobs.launch(
        store,
        "dispatch-queue",
        jobs.build_argv("dispatch-queue"),
        tmp_path,
        rerun_of=original,
    )

    assert store.get(new_id)["rerun_of"] == original


class _FakeProc:
    pid = None

    def poll(self):
        return None


def test_the_rerun_route_records_the_job_it_came_from(tmp_path, monkeypatch):
    store = _store(tmp_path)
    original = _finished(store)
    seen = {}
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(jobs, "launch", lambda *a, **k: (seen.update(k), 7)[1])

    response = TestClient(create_app()).post(f"/jobs/{original}/rerun", follow_redirects=False)

    assert response.status_code == 303
    assert seen["rerun_of"] == original


def test_the_rerun_route_still_resolves_the_environment_fresh(tmp_path, monkeypatch):
    """The behaviour the page now describes, pinned. If a rerun ever starts
    replaying a stored environment instead, the sentence on the page becomes
    a lie and this fails."""
    store = _store(tmp_path)
    original = _finished(store)
    seen = {}
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(jobs, "launch", lambda *a, **k: (seen.update(k), 7)[1])
    monkeypatch.setattr(
        "gomoveshift.console.app.brand_launch_env",
        lambda *a, **k: {"GMS_MARKER": "resolved-now"},
    )

    TestClient(create_app()).post(f"/jobs/{original}/rerun", follow_redirects=False)

    assert seen["env"] == {"GMS_MARKER": "resolved-now"}


def test_the_page_states_both_halves_of_what_a_rerun_is(tmp_path, monkeypatch):
    store = _store(tmp_path)
    original = _finished(store)
    rerun = _finished(store, rerun_of=original)
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    text = TestClient(create_app()).get(f"/jobs/{rerun}").text

    # Which job, reachable; and both halves of what was replayed.
    assert f'href="/jobs/{original}"' in text
    assert f"job #{original}" in text.lower()
    assert "command" in text.lower()
    assert "settings" in text.lower()


def test_a_job_that_is_not_a_rerun_says_nothing_about_one(tmp_path, monkeypatch):
    """No dangling "started again from job #" on an ordinary job."""
    store = _store(tmp_path)
    job_id = _finished(store)
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    text = TestClient(create_app()).get(f"/jobs/{job_id}").text

    assert "started again from" not in text.lower()
