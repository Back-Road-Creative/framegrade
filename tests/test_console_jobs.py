"""Console job runner -- SQLite store, allowlisted Popen launch, log tail, rerun,
and interval schedules (a `schedules` table replacing crontab for allowlisted subcommands)."""

import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import jobs, pubqueue
from gomoveshift.console.app import create_app
from gomoveshift.console.jobs import JobStore


@pytest.fixture(autouse=True)
def _isolate_tenant_storage(tmp_path, monkeypatch):
    """``_lifespan`` claims tenant storage (``console/tenant.py``) off
    ``Config.load().seo.storage_path`` before anything else -- isolate it from
    whatever ``seo.storage_path`` the ambient config names so a `with
    TestClient(...)` in this file (which DOES run lifespan) never writes a
    stamp outside ``tmp_path``."""
    monkeypatch.setattr(
        pubqueue.Config,
        "load",
        staticmethod(
            lambda *a, **k: SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
        ),
    )


def _store(tmp_path):
    return JobStore(tmp_path / "jobs.sqlite3")


@pytest.fixture(autouse=True)
def _isolate_processes():
    """``_PROCESSES`` is a process-global dict keyed by store file and job
    id. An entry left by a prior test (a launched real subprocess not yet
    reaped before that test's deadline) would stay tracked for the rest of
    the xdist worker's life, so every test starts and ends with an empty
    registry and sees only the processes it launched itself."""
    jobs._PROCESSES.clear()
    yield
    jobs._PROCESSES.clear()


def _wait_done(store, job_id, timeout=5.0):
    deadline = time.time() + timeout
    row = jobs.get_job(store, job_id)
    while row["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        row = jobs.get_job(store, job_id)
    return row


def test_store_roundtrip(tmp_path):
    store = _store(tmp_path)
    job_id = store.create("status", ["a", "b"], str(tmp_path / "x.log"))
    row = store.get(job_id)
    assert row["name"] == "status"
    assert row["argv"] == ["a", "b"]
    assert row["status"] == "queued"
    assert row["rc"] is None


def test_build_argv_refuses_non_allowlisted_subcommand():
    with pytest.raises(ValueError):
        jobs.build_argv("upload-everything")


def test_launch_reaps_to_done_and_tail_log_reads_output(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_validate_argv", lambda argv: None)
    store = _store(tmp_path)
    argv = [sys.executable, "-c", "print('hello from job')"]
    job_id = jobs.launch(store, "status", argv, tmp_path / "logs")
    row = _wait_done(store, job_id)
    assert row["status"] == "done"
    assert row["rc"] == 0
    assert row["finished_at"]
    assert any("hello from job" in line for line in jobs.tail_log(row["log_path"], 100))


def test_launch_passes_env_kwarg_through_to_popen(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_validate_argv", lambda argv: None)
    store = _store(tmp_path)
    captured = {}

    class _FakePopen:
        def __init__(self, argv, stdout=None, stderr=None, env=None):
            captured["env"] = env

        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", _FakePopen)
    jobs.launch(store, "status", ["a"], tmp_path / "logs", env={"GMS_OUTPUT_DIR": "/override"})

    # The override passes through; the console also stamps the run origin on every child.
    assert captured["env"] == {"GMS_OUTPUT_DIR": "/override", "GMS_RUN_ORIGIN": "console"}


def test_jobs_routes_launch_status_and_rerun(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    class _FakePopen:
        def __init__(self, *a, **k):
            self._env = k.get("env")

        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", _FakePopen)
    client = TestClient(create_app())

    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert "Jobs" in resp.text

    resp = client.post("/jobs", data={"subcommand": "dispatch-queue"}, follow_redirects=False)
    assert resp.status_code == 303
    detail_url = resp.headers["location"]

    resp = client.get(detail_url)
    assert resp.status_code == 200
    store = JobStore(tmp_path / "jobs.sqlite3")
    job_id = int(detail_url.rsplit("/", 1)[-1])
    _wait_done(store, job_id)

    resp = client.post(f"/jobs/{job_id}/rerun", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] != detail_url

    resp = client.post("/jobs", data={"subcommand": "not-allowed"}, follow_redirects=False)
    assert resp.status_code == 400


def test_get_jobs_skips_the_opportunistic_drain_when_ticker_enabled(tmp_path, monkeypatch):
    """With GMS_CONSOLE_TICKER=1 the background Ticker is the single
    drainer -- GET /jobs must not also call run_due."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_CONSOLE_TICKER", "1")
    monkeypatch.setenv("GMS_CONSOLE_TICK_SECONDS", "9999")
    calls = []
    first_tick = threading.Event()

    def _fake_run_due(store, now, log_dir):
        calls.append(now)
        first_tick.set()
        return []

    monkeypatch.setattr(jobs, "run_due", _fake_run_due)
    with TestClient(create_app()) as client:
        # Deterministically wait for the ticker's own startup tick to land
        # before clearing -- with a 9999s interval no further tick is due,
        # so nothing can land after the clear (a fixed sleep window here
        # raced under a loaded CI runner).
        assert first_tick.wait(timeout=5.0)
        calls.clear()
        resp = client.get("/jobs")
        assert resp.status_code == 200
    assert calls == []


def test_get_jobs_never_drains_even_when_ticker_disabled(tmp_path, monkeypatch):
    """A read must never start work: GET /jobs launches nothing regardless
    of ticker state -- only the explicit POST /jobs/drain route does."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("GMS_CONSOLE_TICKER", raising=False)
    calls = []
    monkeypatch.setattr(jobs, "run_due", lambda store, now, log_dir: calls.append(now) or [])
    client = TestClient(create_app())
    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert calls == []
    assert "not draining" in resp.text.lower()

    resp = client.post("/jobs/drain", follow_redirects=False)
    assert resp.status_code == 303
    assert len(calls) == 1


def test_status_is_not_a_real_subcommand_and_is_refused(tmp_path):
    """``status`` was never a top-level ``gomoveshift`` subcommand -- it
    must not appear in the generic-form allowlist nor build a real argv."""
    assert "status" not in jobs.ALLOWED_SUBCOMMANDS
    assert "status" not in jobs.GENERIC_SUBCOMMANDS
    assert jobs.GENERIC_SUBCOMMANDS <= jobs.ALLOWED_SUBCOMMANDS
    with pytest.raises(ValueError):
        jobs.build_argv("status")


def test_add_schedule_round_trip_and_list(tmp_path):
    store = _store(tmp_path)
    sched_id = jobs.add_schedule(store, "hourly status", "dispatch-queue", interval_minutes=60)
    rows = jobs.list_schedules(store)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == sched_id
    assert row["name"] == "hourly status"
    assert row["subcommand"] == "dispatch-queue"
    assert row["argv"] == jobs.build_argv("dispatch-queue")
    assert row["interval_minutes"] == 60
    assert row["enabled"] is True
    assert row["last_run_at"] is None


@pytest.mark.parametrize("subcommand", ["schedule", "publishing-status", "reconcile-playlists"])
def test_build_argv_accepts_new_zero_arg_subcommands(subcommand):
    assert jobs.build_argv(subcommand) == [sys.executable, "-m", "gomoveshift", subcommand]


def test_metrics_is_not_zero_arg_and_is_refused():
    """``metrics`` requires ``collect <project>`` -- it must never be
    launchable bare (a parameterized argv would have to add it properly)."""
    assert "metrics" not in jobs.GENERIC_SUBCOMMANDS
    with pytest.raises(ValueError):
        jobs.build_argv("metrics")


@pytest.mark.parametrize("subcommand", ["schedule", "publishing-status", "reconcile-playlists"])
def test_jobs_route_launches_new_subcommands(tmp_path, monkeypatch, subcommand):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    class _FakePopen:
        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())
    client = TestClient(create_app())

    resp = client.post("/jobs", data={"subcommand": subcommand}, follow_redirects=False)
    assert resp.status_code == 303


def test_jobs_page_lists_new_subcommands_in_the_dropdown(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    client = TestClient(create_app())
    resp = client.get("/jobs")
    assert resp.status_code == 200
    for subcommand in ("schedule", "publishing-status", "reconcile-playlists"):
        assert f'value="{subcommand}"' in resp.text


@pytest.mark.parametrize("subcommand", ["schedule", "publishing-status", "reconcile-playlists"])
def test_add_schedule_accepts_new_subcommands(tmp_path, subcommand):
    store = _store(tmp_path)
    sched_id = jobs.add_schedule(store, "recurring", subcommand, interval_minutes=60)
    row = jobs.list_schedules(store)[0]
    assert row["id"] == sched_id
    assert row["subcommand"] == subcommand
    assert row["argv"] == jobs.build_argv(subcommand)


def test_add_schedule_refuses_non_allowlisted_subcommand(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        jobs.add_schedule(store, "bad", "upload-everything", interval_minutes=60)


def test_toggle_schedule_flips_enabled(tmp_path):
    store = _store(tmp_path)
    sched_id = jobs.add_schedule(store, "hourly status", "dispatch-queue", interval_minutes=60)
    jobs.toggle_schedule(store, sched_id)
    assert jobs.list_schedules(store)[0]["enabled"] is False
    jobs.toggle_schedule(store, sched_id)
    assert jobs.list_schedules(store)[0]["enabled"] is True


def test_delete_schedule_removes_row(tmp_path):
    store = _store(tmp_path)
    sched_id = jobs.add_schedule(store, "hourly status", "dispatch-queue", interval_minutes=60)
    jobs.delete_schedule(store, sched_id)
    assert jobs.list_schedules(store) == []


def test_schema_is_idempotent_on_existing_db_file(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    JobStore(db_path)
    store2 = JobStore(db_path)  # reopening an existing DB must not raise
    sched_id = jobs.add_schedule(store2, "again", "dispatch-queue", interval_minutes=5)
    assert jobs.list_schedules(store2)[0]["id"] == sched_id


def test_due_schedules_boundary_cases(tmp_path):
    store = _store(tmp_path)
    now = datetime.now(timezone.utc)

    never_ran = jobs.add_schedule(store, "never ran", "dispatch-queue", interval_minutes=60)
    disabled = jobs.add_schedule(store, "disabled", "dispatch-queue", interval_minutes=1)
    jobs.toggle_schedule(store, disabled)
    not_due = jobs.add_schedule(store, "not due", "dispatch-queue", interval_minutes=60)
    store._conn.execute(
        "UPDATE schedules SET last_run_at = ? WHERE id = ?",
        ((now - timedelta(minutes=5)).isoformat(), not_due),
    )
    store._conn.commit()
    overdue = jobs.add_schedule(store, "overdue", "dispatch-queue", interval_minutes=10)
    store._conn.execute(
        "UPDATE schedules SET last_run_at = ? WHERE id = ?",
        ((now - timedelta(minutes=15)).isoformat(), overdue),
    )
    store._conn.commit()

    due_ids = {row["id"] for row in jobs.due_schedules(store, now)}
    assert due_ids == {never_ran, overdue}
    assert disabled not in due_ids
    assert not_due not in due_ids


def test_run_due_launches_via_allowlist_and_stamps_last_run_at(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_validate_argv", lambda argv: None)
    store = _store(tmp_path)
    sched_id = jobs.add_schedule(store, "hourly status", "dispatch-queue", interval_minutes=60)
    monkeypatch.setattr(
        jobs, "build_argv", lambda subcommand: [sys.executable, "-c", "print('scheduled')"]
    )

    now = datetime.now(timezone.utc)
    launched = jobs.run_due(store, now, tmp_path / "logs")
    assert len(launched) == 1
    _wait_done(store, launched[0])

    row = jobs.list_schedules(store)[0]
    assert row["last_run_at"] is not None

    # Not due again immediately after running.
    assert jobs.due_schedules(store, now) == []


def test_sweep_orphaned_jobs_marks_stranded_running_rows_interrupted(tmp_path):
    """``_PROCESSES`` is in-memory only -- a restart loses every live
    Popen handle, so a DB row left ``running`` from before the restart is
    stranded there forever unless something sweeps it."""
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    store.update(job_id, status="running")

    interrupted = jobs.sweep_orphaned_jobs(store)

    assert interrupted == [job_id]
    assert store.get(job_id)["status"] == "interrupted"
    assert store.get(job_id)["finished_at"]


def test_sweep_orphaned_jobs_leaves_live_process_running(tmp_path):
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    store.update(job_id, status="running")

    class _FakeProc:
        pass

    jobs._track(store, job_id, _FakeProc())
    try:
        assert jobs.sweep_orphaned_jobs(store) == []
        assert store.get(job_id)["status"] == "running"
    finally:
        jobs._untrack(store, job_id)


def _two_stores(tmp_path):
    """Two job stores in one process. Each numbers its jobs from 1, so the
    same id names a different job in each."""
    return JobStore(tmp_path / "a" / "jobs.sqlite3"), JobStore(tmp_path / "b" / "jobs.sqlite3")


def _fake_popen(monkeypatch, rc):
    """Stand in for ``Popen``: every launched process reports ``rc`` from
    ``poll()`` (``None`` = still running)."""
    monkeypatch.setattr(jobs, "_validate_argv", lambda argv: None)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: SimpleNamespace(poll=lambda: rc))


def test_a_finished_process_settles_only_its_own_stores_job(tmp_path, monkeypatch):
    """An earlier test once left a finished rc=1 process tracked
    as job 1 of its own store, and reaping a second store settled
    that store's seeded job 1 as ``failed`` with that exit code."""
    failures = []
    monkeypatch.setattr(jobs.notify, "notify_job_failed", failures.append)
    _fake_popen(monkeypatch, rc=1)
    store_a, store_b = _two_stores(tmp_path)
    job_a = jobs.launch(store_a, "drive", ["a"], tmp_path / "logs")
    job_b = store_b.create("drive", ["b"], str(tmp_path / "b.log"))
    store_b.update(job_b, status="done", rc=0, finished_at="2026-09-01T00:00:00+00:00")
    assert job_a == job_b == 1

    jobs.list_jobs(store_b)
    jobs.get_job(store_b, job_b)
    jobs.reap_finished(store_b)

    row_b = store_b.get(job_b)
    assert (row_b["status"], row_b["rc"], row_b["finished_at"]) == (
        "done",
        0,
        "2026-09-01T00:00:00+00:00",
    )
    assert failures == []
    # The process is still store A's, and reaping store A settles it there.
    assert jobs.get_job(store_a, job_a)["status"] == "failed"
    assert [row["argv"] for row in failures] == [["a"]]


def test_sweep_ignores_a_live_process_tracked_for_another_store(tmp_path, monkeypatch):
    """A live process launched as job 1 of store A says nothing about job 1 of
    store B: B's row, stranded ``running`` with no pid, is still swept."""
    monkeypatch.setattr(jobs.notify, "notify_job_interrupted", lambda row: None)
    _fake_popen(monkeypatch, rc=None)
    store_a, store_b = _two_stores(tmp_path)
    job_a = jobs.launch(store_a, "drive", ["a"], tmp_path / "logs")
    job_b = store_b.create("dispatch-queue", ["b"], str(tmp_path / "b.log"))
    store_b.update(job_b, status="running")
    assert job_a == job_b == 1

    assert jobs.sweep_orphaned_jobs(store_b) == [job_b]
    assert store_b.get(job_b)["status"] == "interrupted"
    assert jobs.sweep_orphaned_jobs(store_a) == []
    assert store_a.get(job_a)["status"] == "running"


def test_reap_reconciles_a_dead_row_despite_another_stores_live_process(tmp_path, monkeypatch):
    """``reap_finished`` re-checks untracked ``running`` rows; a live process
    tracked for another store's job with the same id must not shield this
    store's row, whose recorded process is gone."""
    monkeypatch.setattr(jobs.notify, "notify_job_interrupted", lambda row: None)
    monkeypatch.setattr(jobs, "_process_matches", lambda pid, fingerprint: False)
    _fake_popen(monkeypatch, rc=None)
    store_a, store_b = _two_stores(tmp_path)
    job_a = jobs.launch(store_a, "drive", ["a"], tmp_path / "logs")
    job_b = store_b.create("dispatch-queue", ["b"], str(tmp_path / "b.log"))
    store_b.update(job_b, status="running", pid=4242, started_fingerprint="birth")
    assert job_a == job_b == 1

    jobs.reap_finished(store_b)

    row_b = store_b.get(job_b)
    assert (row_b["status"], row_b["outcome_evidence"]) == (
        "interrupted",
        jobs.EVIDENCE_EXITED_WHILE_WATCHED,
    )
    assert store_a.get(job_a)["status"] == "running"


def test_a_process_is_reaped_through_any_store_opened_on_the_same_file(tmp_path, monkeypatch):
    """``app.py`` opens a fresh ``JobStore`` per request, so a process launched
    through one instance must be reaped through another on the same database
    file, however the path to it is spelled."""
    _fake_popen(monkeypatch, rc=0)
    first = JobStore(tmp_path / "jobs.sqlite3")
    job_id = jobs.launch(first, "drive", ["a"], tmp_path / "logs")
    second = JobStore(tmp_path / "alias" / ".." / "jobs.sqlite3")

    row = jobs.get_job(second, job_id)

    assert (row["status"], row["rc"]) == ("done", 0)


def test_sweep_orphaned_jobs_ignores_terminal_rows(tmp_path):
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    store.update(job_id, status="done", rc=0)

    assert jobs.sweep_orphaned_jobs(store) == []
    assert store.get(job_id)["status"] == "done"


def test_sweep_orphaned_jobs_notifies_each_interrupted_row(tmp_path, monkeypatch):
    """A startup sweep silently stranding a row as ``interrupted`` is as
    invisible as a failure nobody reaps -- it must notify too."""
    calls = []
    monkeypatch.setattr(jobs.notify, "notify_job_interrupted", lambda job: calls.append(job))
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["a"], str(tmp_path / "x.log"))
    store.update(job_id, status="running")

    interrupted = jobs.sweep_orphaned_jobs(store)

    assert interrupted == [job_id]
    assert len(calls) == 1
    assert calls[0]["id"] == job_id


def test_run_due_serializes_concurrent_drains(tmp_path, monkeypatch):
    """Two concurrent drains (a ticker tick racing a GET /jobs drain)
    must not both launch the same due schedule."""
    monkeypatch.setattr(jobs, "_validate_argv", lambda argv: None)
    monkeypatch.setattr(
        jobs, "build_argv", lambda subcommand: [sys.executable, "-c", "print('scheduled')"]
    )
    store = _store(tmp_path)
    jobs.add_schedule(store, "hourly", "dispatch-queue", interval_minutes=60)
    now = datetime.now(timezone.utc)

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(jobs.run_due(store, now, tmp_path / "logs")))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    total_launched = sum(len(r) for r in results)
    assert total_launched == 1


def test_schedules_routes_create_toggle_delete_and_reject_bad_subcommand(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    client = TestClient(create_app())

    resp = client.post(
        "/jobs/schedules",
        data={"name": "hourly status", "subcommand": "dispatch-queue", "interval_minutes": "60"},
        follow_redirects=False,
    )
    assert resp.status_code == 303

    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert "hourly status" in resp.text

    store = JobStore(tmp_path / "jobs.sqlite3")
    sched_id = jobs.list_schedules(store)[0]["id"]

    resp = client.post(f"/jobs/schedules/{sched_id}/toggle", follow_redirects=False)
    assert resp.status_code == 303
    assert jobs.list_schedules(store)[0]["enabled"] is False

    resp = client.post(f"/jobs/schedules/{sched_id}/delete", follow_redirects=False)
    assert resp.status_code == 303
    assert jobs.list_schedules(store) == []

    resp = client.post(
        "/jobs/schedules",
        data={"name": "bad", "subcommand": "not-allowed", "interval_minutes": "60"},
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_migration_adds_brand_column_to_existing_db_and_preserves_rows(tmp_path):
    """An existing jobs DB can predate the ``brand`` column --
    opening an existing DB must ALTER TABLE in, not just CREATE TABLE IF NOT
    EXISTS (a no-op against an existing table), and existing rows survive."""
    db_path = tmp_path / "jobs.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "name TEXT NOT NULL, argv TEXT NOT NULL, status TEXT NOT NULL, rc INTEGER, "
        "created_at TEXT NOT NULL, finished_at TEXT, log_path TEXT)"
    )
    conn.execute(
        "INSERT INTO jobs (name, argv, status, created_at, log_path) VALUES "
        "('drive', '[\"a\"]', 'done', '2026-01-01T00:00:00+00:00', '/x.log')"
    )
    conn.commit()
    conn.close()

    store = JobStore(db_path)  # must not raise, must add the column

    rows = store.list()
    assert len(rows) == 1
    assert rows[0]["name"] == "drive"
    assert rows[0]["brand"] is None  # legacy row: unknown, never a guess

    # And it's still usable after the migration.
    job_id = store.create("status", ["a"], str(tmp_path / "x.log"), brand="acme")
    assert store.get(job_id)["brand"] == "acme"


def test_launch_records_the_brand_it_launched_as(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_validate_argv", lambda argv: None)
    monkeypatch.setattr(
        jobs.subprocess, "Popen", lambda *a, **k: type("P", (), {"poll": lambda s: 0})()
    )
    store = _store(tmp_path)
    job_id = jobs.launch(store, "status", ["a"], tmp_path / "logs", brand="acme")
    assert store.get(job_id)["brand"] == "acme"


def test_launch_defaults_brand_to_current_brand_when_not_given(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_validate_argv", lambda argv: None)
    monkeypatch.setattr(
        jobs.subprocess, "Popen", lambda *a, **k: type("P", (), {"poll": lambda s: 0})()
    )
    monkeypatch.setenv("GMS_CONSOLE_BRAND", "examplebrand")
    store = _store(tmp_path)
    job_id = jobs.launch(store, "status", ["a"], tmp_path / "logs")
    assert store.get(job_id)["brand"] == "examplebrand"


def test_run_due_passes_brand_launch_env_so_scheduled_jobs_see_brand_overrides(
    tmp_path, monkeypatch
):
    """A scheduled job used to launch with no env at all,
    silently losing the brand's saved overrides that every manual route
    already passes."""
    monkeypatch.setattr(jobs, "_validate_argv", lambda argv: None)
    monkeypatch.setattr(jobs.settings, "brand_launch_env", lambda: {"GMS_OUTPUT_DIR": "/branded"})
    store = _store(tmp_path)
    sched_id = jobs.add_schedule(store, "hourly", "dispatch-queue", interval_minutes=60)

    captured = {}

    def _fake_popen(argv, stdout=None, stderr=None, env=None):
        captured["env"] = env
        return type("P", (), {"poll": lambda s: 0})()

    monkeypatch.setattr(jobs.subprocess, "Popen", _fake_popen)

    jobs.run_due(store, datetime.now(timezone.utc), tmp_path / "logs")

    assert captured["env"] == {"GMS_OUTPUT_DIR": "/branded", "GMS_RUN_ORIGIN": "console"}
    assert jobs.list_schedules(store)[0]["id"] == sched_id


def test_rerun_route_replays_the_recorded_launch_brand_not_the_current_one(tmp_path, monkeypatch):
    """Rerunning a job launched as brand A while the
    instance's current brand is B must still rerun as A."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    envs_seen = []

    class _FakePopen:
        def __init__(self, argv, stdout=None, stderr=None, env=None):
            envs_seen.append(env)

        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", _FakePopen)

    monkeypatch.setenv("GMS_CONSOLE_BRAND", "brand-a")
    client = TestClient(create_app())
    resp = client.post("/jobs", data={"subcommand": "dispatch-queue"}, follow_redirects=False)
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    store = JobStore(tmp_path / "jobs.sqlite3")
    assert store.get(job_id)["brand"] == "brand-a"

    monkeypatch.setenv("GMS_CONSOLE_BRAND", "brand-b")
    resp = client.post(f"/jobs/{job_id}/rerun", follow_redirects=False)
    assert resp.status_code == 303
    new_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    assert store.get(new_id)["brand"] == "brand-a"


class _FormControls(HTMLParser):
    """Collects the Jobs page's form controls and its ``<label for=...>``
    targets, so a test can assert every control has an accessible name."""

    def __init__(self):
        super().__init__()
        self.controls: list[dict] = []
        self.label_targets: set[str] = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "label" and attrs.get("for"):
            self.label_targets.add(attrs["for"])
        elif tag == "select" or (tag == "input" and attrs.get("type") in {"text", "number"}):
            self.controls.append({"tag": tag, **attrs})


def _jobs_form_controls(tmp_path, monkeypatch) -> _FormControls:
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    with TestClient(create_app()) as client:
        resp = client.get("/jobs")
    assert resp.status_code == 200
    parser = _FormControls()
    parser.feed(resp.text)
    return parser


def test_every_jobs_form_control_has_an_accessible_name(tmp_path, monkeypatch):
    """axe-core found the two ``subcommand`` selects unnamed. A
    ``placeholder`` is not a name either -- it is a hint, it vanishes on
    first keystroke, and several screen readers never announce it. Every
    control needs a real ``<label for>`` or an ``aria-label``."""
    parser = _jobs_form_controls(tmp_path, monkeypatch)
    assert parser.controls, "the Jobs page should render form controls"
    unnamed = [
        c
        for c in parser.controls
        if not c.get("aria-label")
        and not c.get("aria-labelledby")
        and not (c.get("id") and c["id"] in parser.label_targets)
    ]
    assert not unnamed, f"form controls with no accessible name: {unnamed}"


def test_the_two_subcommand_selects_are_named_distinctly(tmp_path, monkeypatch):
    """Both forms post a field called ``subcommand`` -- one launches now,
    one adds a schedule. Identical names would leave a screen-reader user
    two indistinguishable controls, which is the same defect wearing a
    label."""
    parser = _jobs_form_controls(tmp_path, monkeypatch)
    selects = [c for c in parser.controls if c["tag"] == "select"]
    assert len(selects) == 2
    ids = [c.get("id") for c in selects]
    assert all(ids) and len(set(ids)) == 2, f"selects need distinct ids, got {ids}"


class _NeverExits:
    """A launched child that is still running: ``poll()`` returns None, so
    nothing reaps it and the row stays ``running``."""

    def __init__(self, argv, stdout=None, stderr=None, env=None):
        self.pid = 424242

    def poll(self):
        return None


def _running_job(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(jobs.subprocess, "Popen", _NeverExits)
    client = TestClient(create_app())
    resp = client.post("/jobs", data={"subcommand": "dispatch-queue"}, follow_redirects=False)
    return client, int(resp.headers["location"].rsplit("/", 1)[-1])


def test_rerunning_a_job_that_is_still_running_is_refused(tmp_path, monkeypatch):
    """The rerun route read the row and relaunched it, checking nothing. On a
    `dispatch-queue --apply` job that means two drains against the same queue
    at once -- the second worker picks up rows the first is already
    publishing, which is exactly the double-post the lease exists to stop."""
    client, job_id = _running_job(tmp_path, monkeypatch)

    resp = client.post(f"/jobs/{job_id}/rerun", follow_redirects=False)

    assert resp.status_code == 409


def test_a_refused_rerun_launches_nothing(tmp_path, monkeypatch):
    """The status code is not the point; not starting a second pipeline is."""
    client, job_id = _running_job(tmp_path, monkeypatch)
    store = JobStore(tmp_path / "jobs.sqlite3")
    before = len(store.list())

    client.post(f"/jobs/{job_id}/rerun", follow_redirects=False)

    assert len(store.list()) == before


def test_the_refusal_says_the_job_is_still_running(tmp_path, monkeypatch):
    """A bare 409 reads like a bug in the console. It has to name the reason,
    because the operator's next move (wait, or stop the job) depends on it.
    The console answers a browser with its error page, so the reason is read
    off the page the operator actually lands on."""
    client, job_id = _running_job(tmp_path, monkeypatch)

    resp = client.post(f"/jobs/{job_id}/rerun", follow_redirects=False)

    assert "still running" in resp.text.lower()


def test_the_detail_page_does_not_offer_rerun_while_a_job_runs(tmp_path, monkeypatch):
    """A button that always 409s tells an operator an action is available and
    then refuses it."""
    client, job_id = _running_job(tmp_path, monkeypatch)

    body = client.get(f"/jobs/{job_id}").text

    assert f"/jobs/{job_id}/rerun" not in body


def test_a_finished_job_can_still_be_rerun(tmp_path, monkeypatch):
    """The accept-case: rerun is the whole point of the page for a job that
    failed or was interrupted. A guard with no accept-case is a guard that
    breaks the feature."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    class _Exits:
        def __init__(self, argv, stdout=None, stderr=None, env=None):
            self.pid = 424243

        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", _Exits)
    client = TestClient(create_app())
    resp = client.post("/jobs", data={"subcommand": "dispatch-queue"}, follow_redirects=False)
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    client.get(f"/jobs/{job_id}")  # reaps it to "done"

    resp = client.post(f"/jobs/{job_id}/rerun", follow_redirects=False)

    assert resp.status_code == 303


def test_the_detail_page_offers_rerun_for_a_finished_job(tmp_path, monkeypatch):
    """The other half of the accept-case: the button is still there when it
    would actually work."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    store = _store(tmp_path)
    job_id = store.create("dispatch-queue", ["gomoveshift", "dispatch-queue"], str(tmp_path / "l"))
    store.update(job_id, status="failed", rc=1)

    body = TestClient(create_app()).get(f"/jobs/{job_id}").text

    assert f"/jobs/{job_id}/rerun" in body


def test_route_story_film_is_allowlisted():
    """The console job door for a route-story film launch."""
    assert "route-story-film" in jobs.ALLOWED_SUBCOMMANDS
    assert "route-story-film" not in jobs.GENERIC_SUBCOMMANDS


def _set_log_mtime(log_path, now: datetime, age_seconds: float) -> None:
    """Deterministic log age via ``os.utime`` -- never a sleep."""
    stamp = now.timestamp() - age_seconds
    os.utime(log_path, (stamp, stamp))


def test_liveness_label_reports_working_for_a_recently_written_log(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("hello")
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _set_log_mtime(log_path, now, 12)

    label = jobs.liveness_label({"log_path": str(log_path)}, now=now)

    assert label == "working (log updated 12s ago)"


def test_liveness_label_reports_may_be_stalled_past_the_threshold(tmp_path):
    log_path = tmp_path / "job.log"
    log_path.write_text("hello")
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _set_log_mtime(log_path, now, jobs.STALL_THRESHOLD_SEC + 14 * 60)

    label = jobs.liveness_label({"log_path": str(log_path)}, now=now)

    assert label == "quiet 24m -- may be stalled"


def test_liveness_label_never_claims_the_process_is_dead(tmp_path):
    """The reaper, not this label, is the only thing allowed to say a job
    is no longer running."""
    log_path = tmp_path / "job.log"
    log_path.write_text("hello")
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _set_log_mtime(log_path, now, jobs.STALL_THRESHOLD_SEC + 3600)

    label = jobs.liveness_label({"log_path": str(log_path)}, now=now)

    assert "dead" not in label
    assert "not running" not in label


def test_liveness_label_is_none_for_a_missing_log_file(tmp_path):
    row = {"log_path": str(tmp_path / "never-written.log")}
    assert jobs.liveness_label(row) is None


def test_liveness_label_is_none_for_no_log_path_recorded():
    assert jobs.liveness_label({"log_path": ""}) is None
    assert jobs.liveness_label({"log_path": None}) is None
    assert jobs.liveness_label({}) is None


def test_get_job_and_list_jobs_add_liveness_only_to_running_rows(tmp_path):
    store = _store(tmp_path)
    log_path = tmp_path / "r.log"
    log_path.write_text("x")
    running_id = store.create("dispatch-queue", ["a"], str(log_path))
    store.update(running_id, status="running")

    row = jobs.get_job(store, running_id)
    assert "liveness" in row
    assert row["liveness"].startswith("working")

    rows = jobs.list_jobs(store)
    by_id = {r["id"]: r for r in rows}
    assert "liveness" in by_id[running_id]

    for status in ("queued", "done", "failed", "interrupted", "cancelled"):
        job_id = store.create("dispatch-queue", ["a"], str(log_path))
        store.update(job_id, status=status)

        assert "liveness" not in jobs.get_job(store, job_id)
        rows = jobs.list_jobs(store)
        assert "liveness" not in {r["id"]: r for r in rows}[job_id]


def test_jobs_list_and_detail_pages_render_working_for_a_running_job(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    store = JobStore(tmp_path / "jobs.sqlite3")
    log_path = tmp_path / "job_logs" / "1.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("still going")
    job_id = store.create("dispatch-queue", ["a"], str(log_path))
    store.update(job_id, status="running")

    client = TestClient(create_app())

    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert "working (log updated" in resp.text

    resp = client.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    assert "working (log updated" in resp.text


def test_jobs_list_and_detail_pages_render_may_be_stalled_for_an_old_log(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    store = JobStore(tmp_path / "jobs.sqlite3")
    log_path = tmp_path / "job_logs" / "1.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("nothing new")
    job_id = store.create("dispatch-queue", ["a"], str(log_path))
    store.update(job_id, status="running")
    # An epoch-zero mtime is past STALL_THRESHOLD_SEC on any real clock, so
    # the route's own `now` needs no freezing here.
    os.utime(log_path, (0, 0))

    client = TestClient(create_app())

    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert "may be stalled" in resp.text

    resp = client.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    assert "may be stalled" in resp.text


def test_jobs_list_and_detail_pages_show_no_liveness_text_for_a_finished_job(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    store = JobStore(tmp_path / "jobs.sqlite3")
    log_path = tmp_path / "job_logs" / "1.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("done")
    job_id = store.create("dispatch-queue", ["a"], str(log_path))
    store.update(job_id, status="done", rc=0, finished_at="2026-01-01T00:00:00+00:00")

    client = TestClient(create_app())

    resp = client.get("/jobs")
    assert "working (log updated" not in resp.text
    assert "may be stalled" not in resp.text

    resp = client.get(f"/jobs/{job_id}")
    assert "working (log updated" not in resp.text
    assert "may be stalled" not in resp.text


def test_add_schedule_refuses_an_exact_duplicate(tmp_path):
    store = _store(tmp_path)
    jobs.add_schedule(store, "hourly status", "dispatch-queue", interval_minutes=60)
    with pytest.raises(ValueError, match="already"):
        jobs.add_schedule(store, "hourly status", "dispatch-queue", interval_minutes=60)
    assert len(jobs.list_schedules(store)) == 1


def test_add_schedule_allows_a_same_name_with_another_interval_or_subcommand(tmp_path):
    store = _store(tmp_path)
    jobs.add_schedule(store, "status", "dispatch-queue", interval_minutes=60)
    jobs.add_schedule(store, "status", "dispatch-queue", interval_minutes=30)
    jobs.add_schedule(store, "status", "publishing-status", interval_minutes=60)
    assert len(jobs.list_schedules(store)) == 3


def test_posting_the_same_schedule_twice_refuses_the_second(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    store = JobStore(tmp_path / "jobs.sqlite3")
    client = TestClient(create_app())
    form = {"name": "nightly", "subcommand": "dispatch-queue", "interval_minutes": "60"}
    first = client.post("/jobs/schedules", data=form, follow_redirects=False)
    second = client.post("/jobs/schedules", data=form, follow_redirects=False)
    assert first.status_code in (200, 303)
    assert second.status_code == 400
    assert "already" in second.text
    assert len(jobs.list_schedules(store)) == 1


def test_jobs_page_with_no_jobs_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = TestClient(create_app()).get("/jobs")
    assert "No jobs yet." in resp.text


def test_jobs_page_with_a_job_does_not_say_no_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    JobStore(tmp_path / "jobs.sqlite3").create("dispatch-queue", ["true"], None)
    html = TestClient(create_app()).get("/jobs").text
    assert "dispatch-queue" in html
    assert "No jobs yet." not in html


def test_a_console_drive_child_carries_the_console_launcher_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_validate_argv", lambda argv: None)
    captured = {}

    class _FakePopen:
        def __init__(self, argv, stdout=None, stderr=None, env=None):
            captured["env"] = env

        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", _FakePopen)
    jobs.launch(_store(tmp_path), "drive", ["a"], tmp_path / "logs", env={}, executor="local")
    assert captured["env"]["GMS_LAUNCHED_BY"] == "console"
