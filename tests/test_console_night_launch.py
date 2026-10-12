"""Repeated resume requests preserve a single live waiting worker."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gomoveshift.console import jobs, night


@pytest.fixture
def waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(night.settings, "console_data_dir", lambda: tmp_path)
    monkeypatch.setattr(night.settings, "current_brand", lambda: "brand")
    monkeypatch.setattr(night.settings, "brand_launch_env", lambda: {})
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    monkeypatch.setattr(
        jobs.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123, poll=lambda: None)
    )
    monkeypatch.setattr(jobs, "_read_proc_starttime", lambda pid: "birth")
    import json
    import sys
    from gomoveshift.pipeline.photos.night_project import (
        NightStore,
        build_identity,
        runtime_identity,
    )

    store = NightStore(tmp_path / "night-projects")
    with store.connection() as db:
        for project in ("project", "other-project"):
            row = {
                "id": project,
                "brand": "brand",
                "status": "waiting",
                "pause_requested": False,
                "build": build_identity(),
                "runtime": runtime_identity(),
                "interpreter": sys.executable,
            }
            db.execute(
                "INSERT INTO night_projects VALUES (?, ?, ?)", (project, "brand", json.dumps(row))
            )
    return store


@pytest.mark.parametrize("live", [True, False])
def test_waiting_project_launch_checks_actual_worker(waiting, monkeypatch, live):
    # Simulate a waiting child adopted by a restarted console without Popen.
    night._launch(waiting, "project")
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    monkeypatch.setattr(jobs, "_process_matches", lambda pid, fingerprint: live)
    launch = Mock(wraps=jobs.launch)
    monkeypatch.setattr(jobs, "launch", launch)
    night._launch(waiting, "project")
    assert launch.call_count == (0 if live else 1)


def test_waiting_project_ignores_different_live_project(waiting, monkeypatch):
    night._launch(waiting, "other-project")
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    monkeypatch.setattr(jobs, "_process_matches", lambda *a: True)
    launch = Mock(wraps=jobs.launch)
    monkeypatch.setattr(jobs, "launch", launch)
    night._launch(waiting, "project")
    launch.assert_called_once()


def test_waiting_project_ignores_a_live_process_tracked_for_another_store(
    waiting, tmp_path, monkeypatch
):
    """Every job store numbers its jobs from 1, so another store's live job 1
    is not this project's worker: with its own process gone, it relaunches."""
    night._launch(waiting, "project")
    [night_job] = jobs.JobStore(tmp_path / "jobs.sqlite3").list()
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    monkeypatch.setattr(jobs, "_process_matches", lambda pid, fingerprint: False)
    monkeypatch.setattr(jobs, "_validate_argv", lambda argv: None)
    other = jobs.JobStore(tmp_path / "other" / "jobs.sqlite3")
    assert jobs.launch(other, "drive", ["x"], tmp_path / "other-logs") == night_job["id"]
    launch = Mock(wraps=jobs.launch)
    monkeypatch.setattr(jobs, "launch", launch)
    night._launch(waiting, "project")
    assert launch.call_count == 1


@pytest.mark.parametrize("returncode", [None, 0])
def test_waiting_worker_uses_owned_process_exit_status(waiting, monkeypatch, returncode):
    night._launch(waiting, "project")
    process = next(iter(jobs._PROCESSES.values()))
    process.poll = lambda: returncode
    # A recycled PID must not override the exit code of our own child.
    monkeypatch.setattr(jobs, "_process_matches", lambda *a: True)
    launch = Mock(wraps=jobs.launch)
    monkeypatch.setattr(jobs, "launch", launch)
    night._launch(waiting, "project")
    assert launch.call_count == (0 if returncode is None else 1)


def test_concurrent_resume_clicks_launch_one_waiting_worker(waiting, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    # Loose bounds, not speed checks: on a loaded CI runner one launch (real
    # sqlite stores, identity hashing) has taken over 5 s.
    guard = 60
    ready = threading.Barrier(2)
    original_launch = jobs.launch

    def slow_launch(*args, **kwargs):
        time.sleep(0.03)  # both requests reach the launch window
        return original_launch(*args, **kwargs)

    launch = Mock(side_effect=slow_launch)
    monkeypatch.setattr(jobs, "launch", launch)

    def click():
        ready.wait(timeout=guard)
        night._launch(waiting, "project")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(click) for _ in range(2)]
        for future in futures:
            future.result(timeout=guard)
    launch.assert_called_once()
