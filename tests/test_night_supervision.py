"""Durable retry policy never confuses waiting or uncertain launch with death."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gomoveshift.console import jobs, night_workers as workers
from gomoveshift.pipeline.photos.night_project import NightStore

# How long a spawned worker process may take to start and hand back. 10 s was
# hit by load alone (`launched.wait(10)` false on a busy box); the bound guards a hang, not a speed.
CHILD_SPAWN_BUDGET_S = 60


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(workers.settings, "console_data_dir", lambda: tmp_path)
    monkeypatch.setattr(workers.settings, "current_brand", lambda: "brand")
    monkeypatch.setattr(workers.settings, "brand_launch_env", lambda: {})
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    store = NightStore(tmp_path / "night-projects")
    identifier = store.create("brand", [source], [tmp_path], "natural")
    return store, identifier


def outcome(store, identifier, status, **extra):
    row = store.get("brand", identifier)
    row.update(status=status, **extra)
    store.save(row)


def test_transient_retry_backoff_survives_restart(setup, monkeypatch):
    from gomoveshift.console.night_supervision import supervise

    store, identifier = setup
    launch = Mock(return_value=1)
    monkeypatch.setattr(jobs, "launch", launch)
    workers.launch(store, identifier, now=100)
    outcome(store, identifier, "waiting", wait_kind="memory")
    assert supervise(store.root, "brand", now=159) == 0
    assert supervise(store.root, "brand", now=160) == 1
    assert supervise(store.root, "brand", now=161) == 0
    assert launch.call_count == 2
    state = workers.read_state(store.root, "brand", identifier)
    assert state["next_retry_at"] == 280


@pytest.mark.parametrize(
    "status,kind",
    [
        ("complete", None),
        ("blocked", None),
        ("failed", None),
        ("waiting", "budget"),
        ("waiting", None),
    ],
)
def test_nontransient_outcomes_never_auto_retry(setup, monkeypatch, status, kind):
    from gomoveshift.console.night_supervision import supervise

    store, identifier = setup
    launch = Mock(return_value=1)
    monkeypatch.setattr(jobs, "launch", launch)
    workers.launch(store, identifier, now=100)
    outcome(store, identifier, status, wait_kind=kind)
    assert supervise(store.root, "brand", now=10000) == 0
    assert launch.call_count == 1


def test_pause_disables_retry_and_live_waiter_is_preserved(setup, monkeypatch):
    from gomoveshift.console.night_supervision import supervise

    store, identifier = setup
    monkeypatch.setattr(
        jobs.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123, poll=lambda: None)
    )
    monkeypatch.setattr(jobs, "_read_proc_starttime", lambda pid: "birth")
    workers.launch(store, identifier, now=100)
    outcome(store, identifier, "waiting", wait_kind="memory")
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    monkeypatch.setattr(jobs, "_process_matches", lambda *args: True)
    launch = Mock()
    monkeypatch.setattr(jobs, "launch", launch)
    assert supervise(store.root, "brand", now=10000) == 0
    store.pause("brand", identifier)
    assert supervise(store.root, "brand", now=10061) == 0
    assert workers.read_state(store.root, "brand", identifier)["enabled"] is False
    launch.assert_not_called()


def test_launch_exception_is_unknown_not_time_expiring_permission(setup, monkeypatch):
    from gomoveshift.console.night_supervision import supervise

    store, identifier = setup
    monkeypatch.setattr(jobs, "launch", Mock(side_effect=OSError("persistence failed")))
    with pytest.raises(OSError):
        workers.launch(store, identifier, now=100)
    outcome(store, identifier, "waiting", wait_kind="memory")
    launch = Mock()
    monkeypatch.setattr(jobs, "launch", launch)
    assert supervise(store.root, "brand", now=10**9) == 0
    workers.launch(store, identifier, now=10**9)
    launch.assert_not_called()
    assert "unknown" in workers.read_state(store.root, "brand", identifier)["observation"]


def test_historical_cli_and_foreign_brand_never_enrolled(setup, monkeypatch):
    from gomoveshift.console.night_supervision import supervise

    store, identifier = setup
    outcome(store, identifier, "waiting", wait_kind="memory")
    launch = Mock()
    monkeypatch.setattr(jobs, "launch", launch)
    assert supervise(store.root, "brand", now=1000) == 0
    assert not (store.root / "supervision.sqlite3").exists()
    launch.return_value = 1
    workers.launch(store, identifier, now=100)
    assert supervise(store.root, "foreign", now=1000) == 0
    assert launch.call_count == 1


def test_pause_between_claim_and_popen_wins(setup, monkeypatch):
    store, identifier = setup
    original = workers._save

    def save(root, brand, project, state):
        original(root, brand, project, state)
        if state["claim"] == "launching":
            store.pause(brand, project)

    monkeypatch.setattr(workers, "_save", save)
    launch = Mock()
    monkeypatch.setattr(jobs, "launch", launch)
    assert workers.launch(store, identifier, now=100) is False
    launch.assert_not_called()


def test_cancelled_project_requires_explicit_resume(setup, monkeypatch):
    store, identifier = setup
    store.cancel("brand", identifier)
    launch = Mock()
    monkeypatch.setattr(jobs, "launch", launch)

    assert workers.launch(store, identifier, now=100) is False

    launch.assert_not_called()
    state = workers.read_state(store.root, "brand", identifier)
    assert state["enabled"] is False
    assert state["observation"] == "Cancelled; explicit Resume required."


def test_identified_dead_child_reconciles_uncertain_launch(setup, monkeypatch):
    from gomoveshift.console.night_supervision import supervise

    store, identifier = setup

    def crash(job_store, name, argv, log_dir, **kwargs):
        job = job_store.create(name, argv, log_path="", brand="brand")
        job_store.update(job, pid=999999, started_fingerprint="dead", status="running")
        raise OSError("parent lost acknowledgement")

    monkeypatch.setattr(jobs, "launch", crash)
    with pytest.raises(OSError):
        workers.launch(store, identifier, now=100)
    outcome(store, identifier, "waiting", wait_kind="admission")
    monkeypatch.setattr(jobs, "_process_matches", lambda *args: False)
    launch = Mock(return_value=2)
    monkeypatch.setattr(jobs, "launch", launch)
    assert supervise(store.root, "brand", now=160) == 1
    launch.assert_called_once()


def test_two_processes_cannot_claim_same_launch(setup, monkeypatch):
    import multiprocessing
    import os

    store, identifier = setup
    context = multiprocessing.get_context("fork")
    launched, release = context.Event(), context.Event()
    results = context.Queue()

    def spawn(job_store, name, argv, log_dir, **kwargs):
        job = job_store.create(name, argv, log_path="", brand="brand")
        job_store.update(
            job,
            pid=os.getpid(),
            started_fingerprint=jobs._read_proc_starttime(os.getpid()),
            status="running",
        )
        launched.set()
        assert release.wait(CHILD_SPAWN_BUDGET_S)
        return job

    monkeypatch.setattr(jobs, "launch", spawn)

    def run():
        results.put(workers.launch(store, identifier, now=100))

    first, second = context.Process(target=run), context.Process(target=run)
    try:
        first.start()
        assert launched.wait(CHILD_SPAWN_BUDGET_S)
        second.start()
        assert results.get(timeout=CHILD_SPAWN_BUDGET_S) is False
        release.set()
        assert results.get(timeout=CHILD_SPAWN_BUDGET_S) is True
        first.join(CHILD_SPAWN_BUDGET_S)
        second.join(CHILD_SPAWN_BUDGET_S)
        assert first.exitcode == second.exitcode == 0
    finally:
        release.set()
        for child in (first, second):
            if child.pid is not None:
                child.join(10)
                if child.is_alive():
                    child.terminate()
                    child.join(10)


def test_malformed_project_does_not_stop_next_due_project(setup, monkeypatch):
    from gomoveshift.console.night_supervision import supervise

    store, identifier = setup
    source = store.root.parent / "source"
    other = store.create("brand", [source], [source.parent], "accent")
    launch = Mock(return_value=1)
    monkeypatch.setattr(jobs, "launch", launch)
    workers.launch(store, identifier, now=100)
    workers.launch(store, other, now=101)
    with store.connection() as db:
        db.execute(
            "UPDATE night_projects SET data=? WHERE id=?",
            ("[" * 20000 + "0" + "]" * 20000, identifier),
        )
    outcome(store, other, "waiting", wait_kind="disk")
    assert supervise(store.root, "brand", now=200) == 1
    assert launch.call_count == 3
    assert workers.read_state(store.root, "brand", identifier)["enabled"] is False


def test_crash_retries_are_finite_and_build_drift_stops(setup, monkeypatch):
    from gomoveshift.console.night_supervision import supervise

    store, identifier = setup
    launch = Mock(return_value=1)
    monkeypatch.setattr(jobs, "launch", launch)
    workers.launch(store, identifier, now=100)
    outcome(store, identifier, "interrupted")
    for now in (1000, 2000, 4000, 8000):
        supervise(store.root, "brand", now=now)
    assert launch.call_count == 4
    assert workers.read_state(store.root, "brand", identifier)["enabled"] is False
    outcome(store, identifier, "waiting", wait_kind="memory", build={"changed": "build"})
    workers.launch(store, identifier, now=9000)
    assert launch.call_count == 4


def test_runtime_drift_stops_before_launch(setup, monkeypatch):
    store, identifier = setup
    outcome(store, identifier, "queued", runtime={"changed": "runtime"})
    launch = Mock()
    monkeypatch.setattr(jobs, "launch", launch)

    assert workers.launch(store, identifier, now=100) is False

    launch.assert_not_called()
    assert workers.read_state(store.root, "brand", identifier)["enabled"] is False


def test_ticker_night_failure_cannot_stop_other_observers(tmp_path, monkeypatch):
    from gomoveshift.console import night_supervision

    monkeypatch.setattr(jobs, "run_due", Mock())
    monkeypatch.setattr(jobs, "reap_finished", Mock())
    observer = Mock()
    monkeypatch.setattr(jobs.notify, "check_queue_regressions", observer)
    monkeypatch.setattr(jobs.notify, "check_pending_gates", Mock())
    monkeypatch.setattr(jobs.notify, "check_missed_slots", Mock())
    monkeypatch.setattr(
        night_supervision, "supervise", Mock(side_effect=ValueError("broken metadata"))
    )
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    ticker = jobs.Ticker(store, tmp_path, 1)
    ticker._wait = lambda _: ticker._stop_event.set()
    ticker._run()
    assert ticker.last_night_error == "broken metadata"
    assert ticker.last_night_ok_at is None
    observer.assert_called_once()
    store._conn.close()


def test_running_job_without_fingerprint_is_unknown_not_dead(setup, monkeypatch):
    from gomoveshift.console.night_supervision import supervise

    store, identifier = setup
    monkeypatch.setattr(
        jobs.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123, poll=lambda: None)
    )
    monkeypatch.setattr(jobs, "_read_proc_starttime", lambda pid: None)
    workers.launch(store, identifier, now=100)
    outcome(store, identifier, "waiting", wait_kind="memory")
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    launch = Mock()
    monkeypatch.setattr(jobs, "launch", launch)
    # Reaping uses durable waiting evidence when PID identity is missing;
    # that must not erase the supervisor's uncertainty about a live child.
    reaped = jobs.JobStore(store.root.parent / "jobs.sqlite3")
    jobs.reap_finished(reaped)
    reaped._conn.close()
    assert supervise(store.root, "brand", now=200) == 0
    assert "unknown" in workers.read_state(store.root, "brand", identifier)["observation"]
    launch.assert_not_called()


def test_explicit_resume_reenables_paused_project(setup, monkeypatch):
    store, identifier = setup
    launch = Mock(return_value=1)
    monkeypatch.setattr(jobs, "launch", launch)
    workers.launch(store, identifier, now=100)
    store.pause("brand", identifier)
    workers.launch(store, identifier, now=200)
    assert workers.read_state(store.root, "brand", identifier)["enabled"] is False
    outcome(store, identifier, "paused")
    store.pause("brand", identifier, requested=False)
    assert workers.launch(store, identifier, now=300) is True
    assert workers.read_state(store.root, "brand", identifier)["enabled"] is True


def test_resume_live_polling_worker_restores_retry_intent(setup, monkeypatch):
    store, identifier = setup
    monkeypatch.setattr(
        jobs.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123, poll=lambda: None)
    )
    monkeypatch.setattr(jobs, "_read_proc_starttime", lambda pid: "birth")
    workers.launch(store, identifier, now=100)
    store.pause("brand", identifier)
    workers.launch(store, identifier, now=101)
    assert workers.read_state(store.root, "brand", identifier)["enabled"] is False
    store.pause("brand", identifier, requested=False)
    assert workers.launch(store, identifier, now=102) is False
    assert workers.read_state(store.root, "brand", identifier)["enabled"] is True


def test_dead_acknowledged_startup_worker_retries_queued_project(setup, monkeypatch):
    from gomoveshift.console.night_supervision import supervise

    store, identifier = setup
    monkeypatch.setattr(
        jobs.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123, poll=lambda: None)
    )
    monkeypatch.setattr(jobs, "_read_proc_starttime", lambda pid: "birth")
    workers.launch(store, identifier, now=100)
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    monkeypatch.setattr(jobs, "_process_matches", lambda *args: False)
    launch = Mock(return_value=2)
    monkeypatch.setattr(jobs, "launch", launch)
    assert supervise(store.root, "brand", now=160) == 1
    launch.assert_called_once()


@pytest.mark.parametrize("claim", ["CORRUPT", None])
def test_invalid_claim_never_authorizes_launch(setup, monkeypatch, claim):
    from gomoveshift.console.night_supervision import supervise

    store, identifier = setup
    launch = Mock(return_value=1)
    monkeypatch.setattr(jobs, "launch", launch)
    workers.launch(store, identifier, now=100)
    outcome(store, identifier, "waiting", wait_kind="memory")
    state = workers.read_state(store.root, "brand", identifier)
    state["claim"] = claim
    with workers.state_db(store.root) as db:
        db.execute("UPDATE retries SET data=? WHERE id=?", (json.dumps(state), identifier))
    assert supervise(store.root, "brand", now=200) == 0
    assert launch.call_count == 1


def test_live_observation_distinguishes_missing_heartbeat_from_stage_stall():
    row = {
        "updated_at": 100,
        "heartbeat_at": 49_980,
        "last_progress_at": 1_000,
        "current_stage": "best_stills",
        "stage_progress": {"completed": 2, "total": 10},
    }

    stalled = workers.live_worker_observation(row, now=50_000)
    missed = workers.live_worker_observation(row | {"heartbeat_at": 49_000}, now=50_000)

    assert "stage stall" in stalled and "best_stills" in stalled
    assert "heartbeat" in missed and "will not relaunch" in missed
