"""Console ``host`` executor: GPU subcommands are spooled to the host worker, never Popen'd here."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

from gomoveshift.console import jobs, pubqueue
from gomoveshift.console.app import create_app
from gomoveshift.console.jobs import JobStore

DRIVE = [sys.executable, "-m", "gomoveshift", "drive", "/in/day1"]
DISPATCH = [sys.executable, "-m", "gomoveshift", "dispatch-queue"]


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    jobs._PROCESSES.clear()
    monkeypatch.setattr(
        pubqueue.Config,
        "load",
        staticmethod(
            lambda *a, **k: SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
        ),
    )
    yield
    jobs._PROCESSES.clear()


@pytest.fixture
def spool(tmp_path, monkeypatch):
    root = tmp_path / "spool"
    for sub in ("pending", "running", "done", "cancel", "logs"):
        (root / sub).mkdir(parents=True)
    monkeypatch.setenv("GMS_HOST_SPOOL_DIR", str(root))
    return root


@pytest.fixture
def no_popen(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a GPU job must never be Popen'd in the console")

    monkeypatch.setattr(jobs.subprocess, "Popen", boom)


@pytest.fixture
def alerts(monkeypatch):
    seen = {"failed": [], "interrupted": []}
    monkeypatch.setattr(jobs.notify, "notify_job_failed", lambda row: seen["failed"].append(row))
    monkeypatch.setattr(
        jobs.notify, "notify_job_interrupted", lambda row: seen["interrupted"].append(row)
    )
    return seen


def _beat(spool: Path, age: float = 0.0) -> None:
    (spool / "heartbeat.json").write_text(json.dumps({"ts": time.time() - age, "pid": 1}))


def _status(spool: Path, spool_id: str, state: str, **fields) -> None:
    body = {"job_id": spool_id, "state": state, "pid": 77, "phase": None, "progress": None}
    body.update(fields)
    (spool / f"{spool_id}.status.json").write_text(json.dumps(body))


def _store(tmp_path):
    return JobStore(tmp_path / "jobs.sqlite3")


def _launch(store, tmp_path, argv=DRIVE, brand="gomoveshift", name="drive"):
    return jobs.launch(store, name, argv, tmp_path / "logs", brand=brand)


def _spool_id(store, job_id):
    return store.get(job_id)["spool_id"]


def test_gpu_launch_writes_a_request_and_runs_nothing(tmp_path, spool, no_popen):
    _beat(spool)
    store = _store(tmp_path)
    job_id = _launch(store, tmp_path)
    row = store.get(job_id)
    assert row["executor"] == "host" and row["status"] == "queued"
    request = json.loads((spool / "pending" / f"{row['spool_id']}.json").read_text())
    assert request["job_id"] == row["spool_id"]
    assert request["brand"] == "gomoveshift"
    assert request["argv"] == ["python3", *DRIVE[1:]]  # no container interpreter on the host
    assert request["dedupe_key"] and request["submitted_at"]
    assert row["log_path"] == str(spool / "logs" / f"{row['spool_id']}.log")
    assert row["pid"] is None


def test_a_non_gpu_subcommand_still_runs_locally(tmp_path, spool, monkeypatch):
    _beat(spool)
    calls = []

    class Fake:
        pid = 4242

        def poll(self):
            return None

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: calls.append(a) or Fake())
    store = _store(tmp_path)
    job_id = _launch(store, tmp_path, argv=DISPATCH, name="dispatch-queue")
    assert calls and store.get(job_id)["executor"] in (None, "local")
    assert not list((spool / "pending").iterdir())


def test_unwritable_spool_refuses_without_a_local_fallback(tmp_path, monkeypatch, no_popen):
    monkeypatch.setenv("GMS_HOST_SPOOL_DIR", str(tmp_path / "missing"))
    store = _store(tmp_path)
    with pytest.raises(jobs.HostRefusedError, match="spool"):
        _launch(store, tmp_path)
    assert store.list() == []


def test_explicit_host_without_a_spool_is_refused(tmp_path, monkeypatch, no_popen):
    monkeypatch.delenv("GMS_HOST_SPOOL_DIR", raising=False)
    with pytest.raises(jobs.HostRefusedError, match="not configured"):
        jobs.launch(_store(tmp_path), "drive", DRIVE, tmp_path / "logs", executor="host")


@pytest.mark.parametrize("age", [None, 600.0])
def test_missing_or_stale_heartbeat_refuses(tmp_path, spool, no_popen, age):
    if age is not None:
        _beat(spool, age)
    store = _store(tmp_path)
    with pytest.raises(jobs.HostRefusedError, match="heartbeat"):
        _launch(store, tmp_path)
    assert store.list() == [] and not list((spool / "pending").iterdir())


def test_duplicate_is_refused_while_pending_then_allowed_after_it_finishes(
    tmp_path, spool, no_popen
):
    _beat(spool)
    store = _store(tmp_path)
    first = _launch(store, tmp_path)
    with pytest.raises(jobs.HostRefusedError, match="already"):
        _launch(store, tmp_path)
    assert len(store.list()) == 1
    sid = _spool_id(store, first)
    (spool / "pending" / f"{sid}.json").rename(spool / "running" / f"{sid}.json")
    with pytest.raises(jobs.HostRefusedError, match="already"):
        _launch(store, tmp_path)
    (spool / "running" / f"{sid}.json").rename(spool / "done" / f"{sid}.json")
    assert _launch(store, tmp_path) != first


def test_same_argv_for_another_brand_is_not_a_duplicate(tmp_path, spool, no_popen):
    _beat(spool)
    store = _store(tmp_path)
    a = _launch(store, tmp_path, brand="gomoveshift")
    b = _launch(store, tmp_path, brand="backroadcreative")
    assert _spool_id(store, a) != _spool_id(store, b)


def test_status_file_drives_progress_and_done(tmp_path, spool, no_popen, alerts):
    _beat(spool)
    store = _store(tmp_path)
    job_id = _launch(store, tmp_path)
    sid = _spool_id(store, job_id)
    _status(spool, sid, "running", progress=0.4, phase="encode")
    row = jobs.get_job(store, job_id)
    assert row["status"] == "running" and row["progress"] == 0.4 and row["phase"] == "encode"
    assert "40%" in row["liveness"] and "encode" in row["liveness"]
    _status(spool, sid, "succeeded", exit_code=0, progress=1.0)
    row = jobs.get_job(store, job_id)
    assert row["status"] == "done" and row["rc"] == 0 and row["finished_at"]
    assert alerts == {"failed": [], "interrupted": []}


def test_failed_status_marks_failed_and_notifies(tmp_path, spool, no_popen, alerts):
    _beat(spool)
    store = _store(tmp_path)
    job_id = _launch(store, tmp_path)
    _status(spool, _spool_id(store, job_id), "failed", exit_code=3, error="GPU lock unavailable")
    jobs.reap_finished(store)
    row = store.get(job_id)
    assert row["status"] == "failed" and row["rc"] == 3
    assert "GPU lock unavailable" in row["host_error"]
    assert len(alerts["failed"]) == 1
    jobs.reap_finished(store)
    assert len(alerts["failed"]) == 1


def test_cancelled_status_is_cancelled_without_a_failure_alert(tmp_path, spool, no_popen, alerts):
    _beat(spool)
    store = _store(tmp_path)
    job_id = _launch(store, tmp_path)
    _status(spool, _spool_id(store, job_id), "cancelled", exit_code=-15)
    assert jobs.get_job(store, job_id)["status"] == "cancelled"
    assert alerts["failed"] == []


def test_running_job_whose_worker_went_silent_is_interrupted_and_notified(
    tmp_path, spool, no_popen, alerts
):
    _beat(spool)
    store = _store(tmp_path)
    job_id = _launch(store, tmp_path)
    _status(spool, _spool_id(store, job_id), "running", progress=0.2)
    _beat(spool, 600.0)
    jobs.reap_finished(store)
    row = store.get(job_id)
    assert row["status"] == "interrupted"
    assert "host worker" in jobs.interrupted_evidence_sentence(row)
    assert len(alerts["interrupted"]) == 1


def test_request_that_vanished_without_a_status_is_interrupted(tmp_path, spool, no_popen, alerts):
    _beat(spool)
    store = _store(tmp_path)
    job_id = _launch(store, tmp_path)
    (spool / "pending" / f"{_spool_id(store, job_id)}.json").unlink()
    jobs.reap_finished(store)
    assert store.get(job_id)["status"] == "interrupted"


def test_startup_sweep_leaves_a_host_job_alone(tmp_path, spool, no_popen, alerts):
    _beat(spool)
    store = _store(tmp_path)
    job_id = _launch(store, tmp_path)
    _status(spool, _spool_id(store, job_id), "running")
    jobs.reap_finished(store)
    assert jobs.sweep_orphaned_jobs(store) == []
    assert store.get(job_id)["status"] == "running"


def test_cancel_writes_a_marker_for_an_active_host_job(tmp_path, spool, no_popen):
    _beat(spool)
    store = _store(tmp_path)
    job_id = _launch(store, tmp_path)
    jobs.cancel_job(store, job_id)
    assert (spool / "cancel" / _spool_id(store, job_id)).exists()


def test_cancel_refuses_a_local_or_finished_job(tmp_path, spool, no_popen):
    _beat(spool)
    store = _store(tmp_path)
    local = store.create("dispatch-queue", DISPATCH, "")
    with pytest.raises(ValueError, match="host"):
        jobs.cancel_job(store, local)
    job_id = _launch(store, tmp_path)
    _status(spool, _spool_id(store, job_id), "succeeded", exit_code=0)
    jobs.get_job(store, job_id)
    with pytest.raises(ValueError, match="finished"):
        jobs.cancel_job(store, job_id)


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    return TestClient(create_app())


def test_detail_page_shows_progress_and_a_cancel_button(tmp_path, spool, no_popen, monkeypatch):
    _beat(spool)
    client = _client(tmp_path, monkeypatch)
    store = JobStore(tmp_path / "console" / "jobs.sqlite3")
    job_id = _launch(store, tmp_path)
    _status(spool, _spool_id(store, job_id), "running", progress=0.5, phase="encode")
    page = client.get(f"/jobs/{job_id}")
    assert page.status_code == 200
    assert "50%" in page.text and "encode" in page.text
    assert f'action="/jobs/{job_id}/cancel"' in page.text
    resp = client.post(f"/jobs/{job_id}/cancel", follow_redirects=False)
    assert resp.status_code == 303
    assert (spool / "cancel" / _spool_id(store, job_id)).exists()


def test_cancel_route_says_why_for_a_local_job(tmp_path, spool, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    store = JobStore(tmp_path / "console" / "jobs.sqlite3")
    local = store.create("dispatch-queue", DISPATCH, "")
    resp = client.post(f"/jobs/{local}/cancel", follow_redirects=False)
    assert resp.status_code == 409 and "host" in resp.text


def test_a_refused_launch_reads_as_a_409_with_the_reason(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    def refuse(*a, **k):
        raise jobs.HostRefusedError("the host worker's heartbeat is 600s old")

    monkeypatch.setattr(jobs, "launch", refuse)
    resp = client.post("/jobs", data={"subcommand": "dispatch-queue"})
    assert resp.status_code == 409 and "heartbeat" in resp.text


def test_compose_binds_the_host_spool():
    compose = Path(__file__).resolve().parent.parent / "docker" / "console-compose.yaml"
    base = yaml.safe_load(compose.read_text())["services"]["console-base"]
    assert any(str(v).endswith(":/host-spool") for v in base["volumes"])
    assert "GMS_HOST_SPOOL_DIR=/host-spool" in base["environment"]


def test_request_env_holds_only_the_brand_overrides(tmp_path, spool, no_popen, monkeypatch):
    from gomoveshift.console import settings

    name = settings.load_registry()[0].name
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    monkeypatch.setenv("SOME_SECRET_TOKEN", "do-not-leak")
    settings.save_brand_settings(
        "gomoveshift", {name: "override-value", "NOT_IN_REGISTRY": "x", "GMS_UNSET": None}
    )
    _beat(spool)
    store = _store(tmp_path)
    job_id = _launch(store, tmp_path)
    request = json.loads((spool / "pending" / f"{_spool_id(store, job_id)}.json").read_text())
    assert request["env"] == {name: "override-value"}
    assert "do-not-leak" not in json.dumps(request)


def test_request_env_is_empty_without_overrides(tmp_path, spool, no_popen, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    _beat(spool)
    store = _store(tmp_path)
    job_id = _launch(store, tmp_path)
    request = json.loads((spool / "pending" / f"{_spool_id(store, job_id)}.json").read_text())
    assert request["env"] == {}


# -- mutant pins: hostspool contracts the job-level tests do not reach -----------


def test_both_gpu_subcommands_are_spooled():
    from gomoveshift.console import hostspool

    story = [sys.executable, "-m", "gomoveshift", "route-story-film", "/in"]
    assert hostspool.is_gpu_argv(DRIVE)
    assert hostspool.is_gpu_argv(story)
    assert not hostspool.is_gpu_argv(DISPATCH)


def test_two_different_renders_for_one_brand_are_not_duplicates(spool):
    from gomoveshift.console import hostspool

    other = [*DRIVE[:-1], "/in/day2"]
    k1 = hostspool.dedupe_key(DRIVE, "gomoveshift")
    k2 = hostspool.dedupe_key(other, "gomoveshift")
    assert k1 != k2
    assert hostspool.dedupe_key(DRIVE, "gomoveshift") == k1
    assert hostspool.dedupe_key(DRIVE, "otherbrand") != k1
    hostspool.write_request(spool, "gomoveshift-1", "gomoveshift", DRIVE, k1)
    assert hostspool.active_duplicate(spool, k1) == "gomoveshift-1"
    assert hostspool.active_duplicate(spool, k2) is None


@pytest.mark.parametrize("sub", ["pending", "cancel"])
def test_a_spool_dir_the_console_cannot_write_is_refused(spool, sub):
    from gomoveshift.console import hostspool

    _beat(spool)
    assert hostspool.require_ready(spool) == spool
    (spool / sub).chmod(0o500)
    try:
        with pytest.raises(hostspool.HostRefusedError, match="not writable"):
            hostspool.require_ready(spool)
    finally:
        (spool / sub).chmod(0o700)


def test_spool_id_names_the_brand_so_two_consoles_never_collide():
    from gomoveshift.console import hostspool

    assert hostspool.spool_id("gomoveshift", 1) != hostspool.spool_id("otherbrand", 1)
    assert hostspool.spool_id("gomoveshift", 1) == "gomoveshift-1"
    assert hostspool.spool_id("Back Road!", 7) == "back-road-7"
    assert hostspool.spool_id(None, 3) == "console-3"


def test_non_string_override_values_reach_the_request_as_strings(tmp_path, monkeypatch):
    from gomoveshift.console import settings

    names = [row.name for row in settings.load_registry()][:2]
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    settings.save_brand_settings("gomoveshift", {names[0]: 5, names[1]: True})
    got = settings.brand_env_overrides("gomoveshift")
    assert got == {names[0]: "5", names[1]: "True"}
    assert all(isinstance(v, str) for v in got.values())
