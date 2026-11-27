"""Host worker: claim, validate, lock, run, status, heartbeat, cancel."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time

import pytest

from gomoveshift.hostworker import HostWorker, HostWorkerError
from gomoveshift.hostworker import worker as worker_mod


def _ok_gpu() -> None:
    return None


def _no_gpu() -> None:
    raise HostWorkerError("nvidia-smi failed")


def _script(script: str):
    """Swap the allowlisted argv for a python -c script (keeps log/env/lock wiring)."""
    return lambda argv: [sys.executable, "-c", script]


def _make(tmp_path, script="print('hi')", **kw):
    kw.setdefault("gpu_checker", _ok_gpu)
    kw.setdefault("lock_path", tmp_path / "gpu.lock")
    kw.setdefault("lock_wait_s", 0.3)
    kw.setdefault("grace_s", 0.5)
    kw.setdefault("heartbeat_s", 0.1)
    kw.setdefault("runner", "detached")
    kw.setdefault("child_argv", _script(script))
    return HostWorker(tmp_path / "spool", **kw)


def _submit(tmp_path, job_id="j1", argv=None, **extra):
    spool = tmp_path / "spool"
    (spool / "pending").mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    req = {
        "job_id": job_id,
        "brand": "gms",
        "argv": argv or [sys.executable, "-m", "gomoveshift", "drive", str(source)],
        "dedupe_key": "k",
        "submitted_at": "2026-10-17T00:00:00+00:00",
    }
    req.update(extra)
    (spool / "pending" / f"{job_id}.json").write_text(json.dumps(req))
    return req


def _status(tmp_path, job_id="j1"):
    return json.loads((tmp_path / "spool" / f"{job_id}.status.json").read_text())


def _log(tmp_path, job_id="j1"):
    return (tmp_path / "spool" / "logs" / f"{job_id}.log").read_text()


def _cancel(tmp_path, job_id="j1"):
    d = tmp_path / "spool" / "cancel"
    d.mkdir(parents=True, exist_ok=True)
    (d / job_id).write_text("")


def test_refuses_to_start_without_gpu(tmp_path):
    with pytest.raises(HostWorkerError):
        _make(tmp_path, gpu_checker=_no_gpu)


def test_check_nvidia_smi_fails_when_binary_missing(monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(HostWorkerError):
        worker_mod.check_nvidia_smi()


def test_spool_dir_defaults_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_HOST_SPOOL_DIR", str(tmp_path / "envspool"))
    w = HostWorker(None, gpu_checker=_ok_gpu, lock_path=tmp_path / "l")
    assert w.spool == tmp_path / "envspool"
    assert (tmp_path / "envspool" / "pending").is_dir()
    assert (tmp_path / "envspool" / "logs").is_dir()


def test_no_spool_dir_anywhere_is_an_error(monkeypatch):
    monkeypatch.delenv("GMS_HOST_SPOOL_DIR", raising=False)
    with pytest.raises(HostWorkerError):
        HostWorker(None, gpu_checker=_ok_gpu)


def test_runs_request_and_moves_to_done(tmp_path):
    _submit(tmp_path)
    assert _make(tmp_path, "print('hello-log')").run_once() is True
    st = _status(tmp_path)
    assert st["state"] == "succeeded" and st["exit_code"] == 0
    assert st["pid"] > 0 and st["error"] is None
    assert "hello-log" in _log(tmp_path)
    spool = tmp_path / "spool"
    assert (spool / "done" / "j1.json").exists()
    assert not list((spool / "pending").iterdir())
    assert not list((spool / "running").iterdir())


def test_request_log_path_is_ignored(tmp_path):
    _submit(tmp_path, log_path=str(tmp_path / "elsewhere.log"))
    _make(tmp_path, "print('x')").run_once()
    assert not (tmp_path / "elsewhere.log").exists()
    assert "x" in _log(tmp_path)


def test_idle_returns_false(tmp_path):
    assert _make(tmp_path).run_once() is False


def test_nonzero_exit_is_failed(tmp_path):
    _submit(tmp_path)
    _make(tmp_path, "import sys; sys.exit(3)").run_once()
    st = _status(tmp_path)
    assert st["state"] == "failed" and st["exit_code"] == 3
    assert "3" in st["error"]


def test_drive_missing_source_is_refused_before_child_starts(tmp_path):
    missing = tmp_path / "missing"
    _submit(tmp_path, argv=[sys.executable, "-m", "gomoveshift", "drive", str(missing)])
    _make(tmp_path, spawn=lambda *a, **k: pytest.fail("started an unreadable source")).run_once()
    st = _status(tmp_path)
    assert st["state"] == "failed"
    assert "source" in st["error"].lower() and str(missing) in st["error"]


def test_drive_relative_source_is_refused_before_child_starts(tmp_path):
    _submit(tmp_path, argv=[sys.executable, "-m", "gomoveshift", "drive", "source"])
    _make(tmp_path, spawn=lambda *a, **k: pytest.fail("started a relative host source")).run_once()
    st = _status(tmp_path)
    assert st["state"] == "failed" and "absolute host path" in st["error"]


def test_drive_unreadable_source_directory_is_refused(tmp_path, monkeypatch):
    source = tmp_path / "blocked"
    source.mkdir()
    _submit(tmp_path, argv=[sys.executable, "-m", "gomoveshift", "drive", str(source)])
    real_scandir = os.scandir

    def unreadable(path):
        if os.fspath(path) == str(source):
            raise PermissionError("directory cannot be listed")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", unreadable)
    _make(tmp_path, spawn=lambda *a, **k: pytest.fail("started an unreadable source")).run_once()
    st = _status(tmp_path)
    assert st["state"] == "failed" and "source" in st["error"].lower()


def test_drive_broken_media_symlink_is_refused(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "GX010046.MP4").symlink_to(tmp_path / "missing.MP4")
    _submit(tmp_path, argv=[sys.executable, "-m", "gomoveshift", "drive", str(source)])
    _make(tmp_path, spawn=lambda *a, **k: pytest.fail("started a broken media source")).run_once()
    st = _status(tmp_path)
    assert st["state"] == "failed" and "GX010046.MP4" in st["error"]


@pytest.mark.parametrize(
    "argv",
    [
        ["/bin/sh", "-c", "id"],
        ["/bin/sh", "-m", "gomoveshift", "drive"],
        ["/tmp/evil/notpython", "-m", "gomoveshift", "drive"],
        [sys.executable, "-m", "gomoveshift", "rm-rf"],
        [sys.executable, "-m", "other", "drive"],
        [sys.executable, "-m", "gomoveshift"],
        "drive",
    ],
)
def test_rejects_non_allowlisted_argv_without_spawning(tmp_path, argv):
    _submit(tmp_path, argv=argv)
    spawned = []
    w = _make(tmp_path, spawn=lambda *a, **k: spawned.append(a))
    w.run_once()
    assert not spawned
    st = _status(tmp_path)
    assert st["state"] == "failed" and "allowlist" in st["error"]
    assert (tmp_path / "spool" / "done" / "j1.json").exists()


@pytest.mark.parametrize("job_id", ["../evil", "a/b", "", 7])
def test_invalid_job_id_never_runs(tmp_path, job_id):
    spool = tmp_path / "spool"
    (spool / "pending").mkdir(parents=True)
    (spool / "pending" / "bad.json").write_text(
        json.dumps({"job_id": job_id, "argv": [sys.executable, "-m", "gomoveshift", "drive"]})
    )
    spawned = []
    w = _make(tmp_path, spawn=lambda *a, **k: spawned.append(a))
    w.run_once()
    assert not spawned
    assert (spool / "done" / "bad.json").exists()
    assert not (tmp_path / "evil.status.json").exists()


def test_malformed_json_moves_to_done_without_running(tmp_path):
    spool = tmp_path / "spool"
    (spool / "pending").mkdir(parents=True)
    (spool / "pending" / "junk.json").write_text("{nope")
    w = _make(tmp_path, spawn=lambda *a, **k: pytest.fail("spawned"))
    assert w.run_once() is True
    assert (spool / "done" / "junk.json").exists()


def test_claim_is_atomic_one_at_a_time_oldest_first(tmp_path):
    _submit(tmp_path, "a")
    time.sleep(0.02)
    _submit(tmp_path, "b")
    w = _make(tmp_path)
    claimed = w.claim_next()
    assert claimed.name == "a.json" and claimed.parent.name == "running"
    assert (tmp_path / "spool" / "pending" / "b.json").exists()
    # a lost race (file already taken) is skipped, not an error
    (tmp_path / "spool" / "pending" / "b.json").rename(tmp_path / "taken.json")
    assert w.claim_next() is None


def test_worker_holds_gpu_lock_for_child_without_its_own_lock(tmp_path):
    _submit(tmp_path, argv=[sys.executable, "-m", "gomoveshift", "route-story-film", "x"])
    lock = tmp_path / "gpu.lock"
    probe = (
        "import fcntl,os\n"
        f"fd=os.open({str(lock)!r},os.O_RDWR)\n"
        "try:\n fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);print('FREE')\n"
        "except BlockingIOError:\n print('HELD')\n"
    )
    _make(tmp_path, probe).run_once()
    assert "HELD" in _log(tmp_path)
    fd = os.open(lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # released afterwards
    os.close(fd)


def test_drive_child_can_acquire_gpu_lock_without_worker_self_deadlock(tmp_path):
    _submit(tmp_path)
    script = (
        "from pathlib import Path\n"
        "from gomoveshift.pipeline.orchestrator_lock import acquire_gpu_lock,release_gpu_lock\n"
        "fd,path=acquire_gpu_lock(wait_s=0.1)\n"
        f"assert path==Path({str(tmp_path / 'gpu.lock')!r})\n"
        "print('DRIVE_LOCK_ACQUIRED',flush=True)\n"
        "release_gpu_lock(fd,path)\n"
    )
    _make(tmp_path, script).run_once()
    assert _status(tmp_path)["state"] == "succeeded"
    assert "DRIVE_LOCK_ACQUIRED" in _log(tmp_path)


def test_drive_child_refuses_foreign_gpu_lock(tmp_path):
    _submit(tmp_path)
    lock = tmp_path / "gpu.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        script = (
            "from gomoveshift.pipeline.orchestrator_lock import acquire_gpu_lock\n"
            "acquire_gpu_lock(wait_s=0.1)\n"
            "print('UNSAFE_GPU_USE',flush=True)\n"
        )
        _make(tmp_path, script).run_once()
        assert _status(tmp_path)["state"] == "failed"
        assert "UNSAFE_GPU_USE" not in _log(tmp_path)
    finally:
        os.close(fd)


_FILM = [sys.executable, "-m", "gomoveshift", "route-story-film", "x"]


def _hold_gpu(tmp_path) -> int:
    """A render holding the shared GPU lock (the caller closes the fd)."""
    fd = os.open(tmp_path / "gpu.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.write(fd, json.dumps({"owner": "gomoveshift", "pid": os.getpid()}).encode())
    return fd


# 2026-11-26: a route-story-film request was failed after 120 s whenever a drive held the card;
# renders run back to back for up to 45 h, so it could never run. It now waits its turn.
def test_a_busy_gpu_lock_keeps_the_request_queued_and_never_runs_it(tmp_path):
    _submit(tmp_path, argv=_FILM)
    fd = _hold_gpu(tmp_path)
    try:
        w = _make(tmp_path, "print('RAN', flush=True)")
        assert w.run_once() is False  # the lane polls instead of spinning on it
    finally:
        os.close(fd)
    st = _status(tmp_path)
    assert st["state"] == "queued" and "GPU lock" in st["waiting"] and st["error"] is None
    assert (tmp_path / "spool" / "pending" / "j1.json").exists()
    assert not (tmp_path / "spool" / "done" / "j1.json").exists()
    assert not (tmp_path / "spool" / "logs" / "j1.log").exists()  # the child never started


def test_a_request_that_waited_for_the_lock_runs_once_it_is_free(tmp_path):
    _submit(tmp_path, argv=_FILM)
    fd = _hold_gpu(tmp_path)
    try:
        _make(tmp_path).run_once()
    finally:
        os.close(fd)
    assert _make(tmp_path).run_once() is True
    assert _status(tmp_path)["state"] == "succeeded"
    assert (tmp_path / "spool" / "done" / "j1.json").exists()


def test_a_waiting_request_goes_behind_the_next_one_and_can_be_cancelled(tmp_path):
    _submit(tmp_path, "j1", argv=_FILM)
    _submit(tmp_path, "j2")  # a drive request, which takes its own lock
    os.utime(tmp_path / "spool" / "pending" / "j1.json", (1, 1))  # j1 was first in line
    os.utime(tmp_path / "spool" / "pending" / "j2.json", (2, 2))
    fd = _hold_gpu(tmp_path)
    try:
        _make(tmp_path).run_once()  # j1 waits: back of the line, so j2 is next
        assert _make(tmp_path).claim_next().name == "j2.json"
        _cancel(tmp_path, "j1")
        _make(tmp_path).run_once()
    finally:
        os.close(fd)
    assert _status(tmp_path, "j1")["state"] == "cancelled"
    assert (tmp_path / "spool" / "done" / "j1.json").exists()


def test_progress_lines_update_status(tmp_path):
    _submit(tmp_path)
    script = "import time;print('GMS_PROGRESS 0.5 encoding',flush=True);time.sleep(0.8)"
    w = _make(tmp_path, script)
    seen = []
    orig = w._write_status

    def spy(*a, **k):
        orig(*a, **k)
        seen.append(_status(tmp_path))

    w._write_status = spy
    w.run_once()
    assert any(s["progress"] == 0.5 and s["phase"] == "encoding" for s in seen)


def test_heartbeat_file_written(tmp_path):
    w = _make(tmp_path)
    w.write_heartbeat()
    hb = json.loads((tmp_path / "spool" / "heartbeat.json").read_text())
    assert hb["pid"] == os.getpid() and hb["ts"] > 0


def test_cancel_sigterm_marks_cancelled(tmp_path):
    _submit(tmp_path)
    _cancel(tmp_path)
    _make(tmp_path, "import time;time.sleep(30)").run_once()
    st = _status(tmp_path)
    assert st["state"] == "cancelled"
    assert not (tmp_path / "spool" / "cancel" / "j1").exists()
    assert (tmp_path / "spool" / "done" / "j1.json").exists()


def test_cancel_escalates_to_sigkill_when_term_ignored(tmp_path):
    _submit(tmp_path)
    _cancel(tmp_path)
    script = (
        "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "print('up',flush=True);time.sleep(30)"
    )
    t0 = time.monotonic()
    _make(tmp_path, script).run_once()
    assert time.monotonic() - t0 < 10
    st = _status(tmp_path)
    assert st["state"] == "cancelled"
    with pytest.raises(ProcessLookupError):
        os.kill(st["pid"], 0)


def test_cancel_for_other_job_is_ignored(tmp_path):
    _submit(tmp_path)
    _cancel(tmp_path, "other")
    _make(tmp_path).run_once()
    assert _status(tmp_path)["state"] == "succeeded"
    assert (tmp_path / "spool" / "cancel" / "other").exists()


def test_main_refuses_to_start_without_spool(monkeypatch, capsys):
    from gomoveshift.hostworker.__main__ import main

    monkeypatch.delenv("GMS_HOST_SPOOL_DIR", raising=False)
    assert main([]) == 2
    assert "refusing to start" in capsys.readouterr().err


def test_serve_runs_requests_and_beats_until_stopped(tmp_path):
    import threading

    _submit(tmp_path)
    w = _make(tmp_path, poll_s=0.05)
    stop = threading.Event()
    t = threading.Thread(target=w.serve, args=(stop,))
    t.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not (tmp_path / "spool" / "done" / "j1.json").exists():
        time.sleep(0.05)
    stop.set()
    t.join(timeout=5)
    assert _status(tmp_path)["state"] == "succeeded"
    assert (tmp_path / "spool" / "heartbeat.json").exists()


_ECHO = "import os; print('X=' + str(os.environ.get('GMS_STAGING_DIR')))"


def test_request_env_is_applied_on_top_of_the_worker_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_STAGING_DIR", "/worker/default")
    _submit(tmp_path, env={"GMS_STAGING_DIR": "/brand/override"})
    _make(tmp_path, _ECHO).run_once()
    assert "X=/brand/override" in _log(tmp_path)


def test_request_env_only_accepts_registry_names(tmp_path, monkeypatch):
    monkeypatch.delenv("NOT_A_GMS_SETTING", raising=False)
    _submit(tmp_path, env={"NOT_A_GMS_SETTING": "x", "PATH": "/evil", "GMS_STAGING_DIR": "/ok"})
    _make(
        tmp_path,
        "import os; print('P=' + str(os.environ.get('NOT_A_GMS_SETTING')),"
        " 'S=' + os.environ['GMS_STAGING_DIR'], 'EVIL=' + str(os.environ['PATH'] == '/evil'))",
    ).run_once()
    log = _log(tmp_path)
    assert "P=None" in log and "S=/ok" in log and "EVIL=False" in log


def test_request_env_never_overrides_the_spool_or_the_gpu_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))
    _submit(tmp_path, env={"GMS_HOST_SPOOL_DIR": "/evil", "GMS_GPU_LOCK_PATH": "/evil/lock"})
    _make(
        tmp_path,
        "import os; print('L=' + os.environ['GMS_GPU_LOCK_PATH'],"
        " 'D=' + str(os.environ.get('GMS_HOST_SPOOL_DIR')))",
    ).run_once()
    log = _log(tmp_path)
    assert f"L={tmp_path / 'gpu.lock'}" in log and "/evil" not in log


@pytest.mark.parametrize("bad", ["a string", ["GMS_STAGING_DIR"], {"GMS_STAGING_DIR": {"x": 1}}])
def test_a_malformed_request_env_is_ignored_not_fatal(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("GMS_STAGING_DIR", "/worker/default")
    _submit(tmp_path, env=bad)
    _make(tmp_path, _ECHO).run_once()
    assert _status(tmp_path)["state"] == "succeeded"
    assert "X=/worker/default" in _log(tmp_path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait(cond, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


def _state(tmp_path, job_id="j1"):
    try:
        return _status(tmp_path, job_id)["state"]
    except (OSError, ValueError):
        return None


def _held_job(tmp_path, name="j1"):
    """A job that records each start and any SIGTERM, then waits for ``<name>.go``."""
    go, mark = tmp_path / f"{name}.go", tmp_path / f"{name}.marks"
    return (
        (
            "import os,signal,time\n"
            f"m=open({str(mark)!r},'a',buffering=1)\n"
            "signal.signal(signal.SIGTERM, lambda *a: m.write('TERM\\n'))\n"
            "m.write('START\\n')\n"
            "print('up', flush=True)\n"
            "t=time.time()\n"
            f"while not os.path.exists({str(go)!r}) and time.time()-t<30: time.sleep(0.05)\n"
        ),
        go,
        mark,
    )


def _serve(w):
    import threading

    stop = threading.Event()
    t = threading.Thread(target=w.serve, args=(stop,), daemon=True)
    t.start()
    return stop, t


def test_a_worker_restart_leaves_the_job_unsignalled_and_the_new_worker_re_adopts_it(tmp_path):
    """Joe's rule: a deploy, restart or unit change goes ahead at once and the render
    runs to completion. The old worker killed it and marked it failed."""
    script, go, mark = _held_job(tmp_path)
    _submit(tmp_path)
    old = _make(tmp_path, script, poll_s=0.05)
    stop, t = _serve(old)
    ident = tmp_path / "spool" / "jobs" / "j1.proc.json"
    assert _wait(lambda: mark.exists() and "START" in mark.read_text())
    assert _wait(
        lambda: (
            _state(tmp_path) == "running"
            and ident.exists()
            and _status(tmp_path)["pid"] == json.loads(ident.read_text())["child_pid"]
        )
    )
    pid = _status(tmp_path)["pid"]
    stop.set()  # what SIGTERM does in __main__
    t.join(timeout=10)
    assert not t.is_alive()
    time.sleep(0.5)
    assert _pid_alive(pid), "the worker stop killed the job"
    assert "TERM" not in mark.read_text(), "the worker stop signalled the job"
    assert (tmp_path / "spool" / "running" / "j1.json").exists()
    assert _state(tmp_path) == "running"

    new = _make(tmp_path, child_argv=lambda argv: pytest.fail("relaunched"), poll_s=0.05)
    stop, t = _serve(new)
    try:
        time.sleep(0.5)
        assert _pid_alive(pid)
        go.write_text("")
        assert _wait(lambda: (tmp_path / "spool" / "done" / "j1.json").exists())
    finally:
        go.write_text("")
        stop.set()
        t.join(timeout=10)
    st = _status(tmp_path)
    assert st["state"] == "succeeded" and st["exit_code"] == 0
    assert mark.read_text().split() == ["START"], "relaunched, or signalled"


def test_drive_runs_in_its_own_lane_so_a_short_job_never_waits_behind_a_render(tmp_path):
    long_script, go, _ = _held_job(tmp_path, "long")
    _submit(tmp_path, "long")  # a drive render, first in line
    time.sleep(0.02)
    _submit(tmp_path, "render2")  # a second drive: waits in the drive lane
    time.sleep(0.02)
    _submit(tmp_path, "short", argv=_FILM)

    def child(argv):
        if argv[3] != "drive":
            return [sys.executable, "-c", "print('short done')"]
        return [sys.executable, "-c", long_script]

    w = _make(tmp_path, child_argv=child, poll_s=0.05, lanes=2)
    stop, t = _serve(w)
    try:
        assert _wait(lambda: _state(tmp_path, "short") == "succeeded", timeout=15)
        assert _state(tmp_path, "long") == "running"
        assert (tmp_path / "spool" / "pending" / "render2.json").exists()  # one render at a time
    finally:
        go.write_text("")
        assert _wait(lambda: _state(tmp_path, "long") == "succeeded")
        assert _wait(lambda: _state(tmp_path, "render2") == "succeeded")
        stop.set()
        t.join(timeout=10)


def test_a_lane_claims_only_its_own_kind(tmp_path):
    _submit(tmp_path, "d1")
    time.sleep(0.02)
    _submit(tmp_path, "f1", argv=_FILM)
    w = _make(tmp_path)
    assert w.claim_next(drive=False).name == "f1.json"
    assert w.claim_next(drive=False) is None
    assert w.claim_next(drive=True).name == "d1.json"


def test_the_lane_count_defaults_to_two(tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_HOST_WORKER_LANES", raising=False)
    assert _make(tmp_path, lanes=None).lanes == 2
    monkeypatch.setenv("GMS_HOST_WORKER_LANES", "3")
    assert _make(tmp_path, lanes=None).lanes == 3


def test_the_worker_writes_its_adoption_capability_for_host_reconcile(tmp_path):
    _make(tmp_path)
    cap = json.loads((tmp_path / "spool" / "capabilities.json").read_text())
    assert cap == {"adopts_running_jobs": True, "pid": os.getpid()}


def test_a_running_entry_left_by_a_child_running_worker_is_failed_never_relaunched(tmp_path):
    _submit(tmp_path)
    spool = tmp_path / "spool"
    (spool / "running").mkdir(parents=True, exist_ok=True)
    (spool / "pending" / "j1.json").rename(spool / "running" / "j1.json")
    w = _make(tmp_path, child_argv=lambda argv: pytest.fail("relaunched"), poll_s=0.05)
    stop, t = _serve(w)
    assert _wait(lambda: (spool / "done" / "j1.json").exists())
    stop.set()
    t.join(timeout=10)
    st = _status(tmp_path)
    assert st["state"] == "failed" and "interrupted" in st["error"]


def test_an_adopted_job_whose_launcher_died_without_an_exit_record_is_failed(tmp_path):
    _submit(tmp_path)
    spool = tmp_path / "spool"
    w = _make(tmp_path, child_argv=lambda argv: pytest.fail("relaunched"), poll_s=0.05)
    (spool / "pending" / "j1.json").rename(spool / "running" / "j1.json")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (spool / "jobs" / "j1.launch.json").write_text(json.dumps({"launched_at": 1}))
    (spool / "jobs" / "j1.proc.json").write_text(json.dumps({"pid": dead.pid, "start": "1"}))
    stop, t = _serve(w)
    assert _wait(lambda: (spool / "done" / "j1.json").exists())
    stop.set()
    t.join(timeout=10)
    st = _status(tmp_path)
    assert st["state"] == "failed" and "without an exit record" in st["error"]


def _shim(tmp_path, body):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    shim = bin_dir / "systemd-run"
    shim.write_text("#!/bin/sh\n" + body)
    shim.chmod(0o755)
    return bin_dir


def test_each_job_gets_its_own_transient_unit(tmp_path, monkeypatch):
    args = tmp_path / "systemd-run.args"
    body = (
        f'printf "%s\\n" "$@" > {args}\n'
        'while [ "${1#-}" != "$1" ]; do shift; done\n'
        'setsid "$@" </dev/null >/dev/null 2>&1 &\n'
    )
    monkeypatch.setenv("PATH", f"{_shim(tmp_path, body)}:{os.environ['PATH']}")
    _submit(tmp_path)
    spawned = []
    w = _make(
        tmp_path, "print('in unit')", runner="systemd-run", spawn=lambda *a, **k: spawned.append(a)
    )
    w.run_once()
    got = args.read_text().split("\n")
    assert "--unit=gms-job-j1" in got and "--collect" in got
    assert got[got.index("-m") + 1] == "gomoveshift.hostworker.launch"
    assert not spawned, "ran outside its unit"
    assert _status(tmp_path)["state"] == "succeeded" and "in unit" in _log(tmp_path)


def test_a_refused_transient_unit_falls_back_to_its_own_session(tmp_path, monkeypatch, caplog):
    body = 'echo "Failed to start transient service unit: Access denied" >&2\nexit 1\n'
    monkeypatch.setenv("PATH", f"{_shim(tmp_path, body)}:{os.environ['PATH']}")
    _submit(tmp_path)
    _make(tmp_path, "print('fallback')", runner="systemd-run").run_once()
    assert _status(tmp_path)["state"] == "succeeded" and "fallback" in _log(tmp_path)
    assert "Access denied" in caplog.text


# -- mutant pins (each one fails if the named guard is removed) -----------------


@pytest.mark.parametrize("rc", [1, 9])
def test_nvidia_smi_nonzero_exit_is_refused(monkeypatch, rc):
    monkeypatch.setattr(
        worker_mod.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, rc, b"", b""),
    )
    with pytest.raises(HostWorkerError, match="no GPU visible"):
        worker_mod.check_nvidia_smi()


def test_nvidia_smi_exit_zero_is_accepted(monkeypatch):
    monkeypatch.setattr(
        worker_mod.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, b"", b"")
    )
    worker_mod.check_nvidia_smi()


def test_a_worker_with_the_gpu_lock_disabled_refuses_to_run_unlocked(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_GPU_LOCK_PATH", "")
    with pytest.raises(HostWorkerError, match="unlocked"):
        HostWorker(tmp_path / "spool", gpu_checker=_ok_gpu, lock_path=None)


@pytest.mark.parametrize("job_id", ["a..b", "x..", "a...b"])
def test_a_dotdot_in_the_job_id_never_runs(tmp_path, job_id):
    _submit(tmp_path, job_id=job_id)
    spawned = []
    w = _make(tmp_path, spawn=lambda *a, **k: spawned.append(a))
    w.run_once()
    assert not spawned
    assert not list((tmp_path / "spool").glob("*.status.json"))


def _cancel_once_up(tmp_path, job_id="j1"):
    """Cancel only after the child printed 'up' (its signal handler is installed)."""
    import threading

    log = tmp_path / "spool" / "logs" / f"{job_id}.log"

    def go():
        for _ in range(100):
            if log.exists() and b"up" in log.read_bytes():
                break
            time.sleep(0.05)
        _cancel(tmp_path, job_id)

    th = threading.Thread(target=go, daemon=True)
    th.start()
    return th


def test_stop_escalates_to_sigkill_only_after_the_grace(tmp_path):
    _submit(tmp_path)
    _cancel_once_up(tmp_path)
    script = (
        "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "print('up',flush=True);time.sleep(30)"
    )
    w = _make(tmp_path, script, grace_s=1.0)
    t0 = time.monotonic()
    w.run_once()
    elapsed = time.monotonic() - t0
    assert elapsed >= 1.0, "SIGKILL came before the grace elapsed"
    assert elapsed < 10
    st = _status(tmp_path)
    assert st["state"] == "cancelled" and st["exit_code"] == -9
    with pytest.raises(ProcessLookupError):
        os.kill(st["pid"], 0)


def test_cancel_sends_sigterm_first_so_the_child_can_clean_up(tmp_path):
    _submit(tmp_path)
    _cancel_once_up(tmp_path)
    marker = tmp_path / "term-seen"
    script = (
        "import signal,sys,time\n"
        f"def h(*a):\n    open({str(marker)!r}, 'w').write('x'); sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, h)\n"
        "print('up', flush=True)\n"
        "time.sleep(30)\n"
    )
    t0 = time.monotonic()
    _make(tmp_path, script, grace_s=5.0).run_once()
    assert time.monotonic() - t0 < 5.0
    assert marker.read_text() == "x"
    st = _status(tmp_path)
    assert st["state"] == "cancelled" and st["exit_code"] == 0


def test_progress_parsing_takes_the_last_line_and_ignores_noise(tmp_path):
    _submit(tmp_path)
    script = (
        "print('noise 0.9 nope')\n"
        "print('GMS_PROGRESS 0.25 first pass')\n"
        "print('GMS_PROGRESS 1.2.3 broken')\n"
        "print('GMS_PROGRESS 0.75 final encode', flush=True)\n"
        "import time; time.sleep(0.8)\n"
    )
    w = _make(tmp_path, script)
    seen = []
    orig = w._write_status

    def spy(*a, **k):
        orig(*a, **k)
        seen.append(_status(tmp_path))

    w._write_status = spy
    w.run_once()
    running = [s for s in seen if s["state"] == "running" and s["progress"] is not None]
    assert running and running[-1]["progress"] == 0.75 and running[-1]["phase"] == "final encode"
    assert all(s["progress"] != 0.9 for s in seen)
    done = _status(tmp_path)
    assert done["state"] == "succeeded" and done["progress"] == 0.75
    assert done["phase"] == "final encode"


def test_request_env_drops_the_spool_dir_and_lock_path_even_if_registered(monkeypatch):
    from types import SimpleNamespace

    names = ["GMS_HOST_SPOOL_DIR", "GMS_GPU_LOCK_PATH", "GMS_OK"]
    monkeypatch.setattr(
        worker_mod, "load_registry", lambda: [SimpleNamespace(name=n) for n in names]
    )
    got = worker_mod._request_env({n: "/evil" for n in names})
    assert got == {"GMS_OK": "/evil"}


@pytest.mark.parametrize("sig", ["SIGTERM", "SIGINT"])
def test_the_real_entry_point_turns_a_signal_into_the_stop_event(sig):
    code = (
        "import os, signal\n"
        "from gomoveshift.hostworker import __main__ as m\n"
        "class W:\n"
        "    def __init__(self, *a): pass\n"
        "    def serve(self, stop):\n"
        f"        os.kill(os.getpid(), signal.{sig})\n"
        "        print('STOP' if stop.wait(5) else 'NOSTOP', flush=True)\n"
        "m.HostWorker = W\n"
        "raise SystemExit(m.main([]))\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert res.stdout.strip() == "STOP", (res.returncode, res.stderr)
    assert res.returncode == 0


@pytest.mark.parametrize("subcommand", ["drive", "route-story-film"])
@pytest.mark.parametrize("interp", ["/usr/local/bin/python3.12", "python3", sys.executable])
def test_the_child_runs_under_the_workers_own_interpreter(tmp_path, subcommand, interp):
    """A console container's interpreter path means nothing on the host."""
    source = tmp_path / "source"
    source.mkdir()
    argument = str(source) if subcommand == "drive" else "x"
    _submit(tmp_path, argv=[interp, "-m", "gomoveshift", subcommand, argument, "--flag"])
    spawned = []

    def child(argv):
        spawned.append(argv)
        return [sys.executable, "-c", "pass"]

    _make(tmp_path, child_argv=child).run_once()
    assert spawned == [[sys.executable, "-m", "gomoveshift", subcommand, argument, "--flag"]]
    assert _status(tmp_path)["state"] == "succeeded"


_MARK = "import os; print('M=' + str(os.environ.get('GMS_LAUNCHED_BY')))"


def test_the_worker_marks_a_drive_child_as_host_launched(tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_LAUNCHED_BY", raising=False)
    _submit(tmp_path)
    _make(tmp_path, _MARK).run_once()
    assert "M=hostworker" in _log(tmp_path)


def test_a_marker_in_the_spool_request_cannot_choose_the_launcher(tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_LAUNCHED_BY", raising=False)
    _submit(tmp_path, env={"GMS_LAUNCHED_BY": "health-backlog"})
    _make(tmp_path, _MARK).run_once()
    assert "M=hostworker" in _log(tmp_path)
