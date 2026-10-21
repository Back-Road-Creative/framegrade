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


def _spawn_script(script: str):
    """Swap the allowlisted argv for a python -c script (keeps log/env wiring)."""

    def spawn(argv, **kw):
        return subprocess.Popen([sys.executable, "-c", script], **kw)

    return spawn


def _make(tmp_path, script="print('hi')", **kw):
    kw.setdefault("gpu_checker", _ok_gpu)
    kw.setdefault("lock_path", tmp_path / "gpu.lock")
    kw.setdefault("lock_wait_s", 0.3)
    kw.setdefault("grace_s", 0.5)
    kw.setdefault("heartbeat_s", 0.1)
    kw.setdefault("spawn", _spawn_script(script))
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


def test_busy_gpu_lock_fails_loudly_and_never_runs(tmp_path):
    _submit(tmp_path, argv=[sys.executable, "-m", "gomoveshift", "route-story-film", "x"])
    fd = os.open(tmp_path / "gpu.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        w = _make(tmp_path, spawn=lambda *a, **k: pytest.fail("ran without lock"))
        w.run_once()
    finally:
        os.close(fd)
    st = _status(tmp_path)
    assert st["state"] == "failed" and "GPU lock" in st["error"]
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


def test_a_stop_request_mid_job_kills_the_child_and_marks_it_failed(tmp_path):
    import threading

    _submit(tmp_path)
    w = _make(tmp_path, "import time; print('started', flush=True); time.sleep(25)")
    w.stop_event = threading.Event()
    threading.Timer(0.8, w.stop_event.set).start()
    started = time.time()
    assert w.run_once() is True
    assert time.time() - started < 20
    st = _status(tmp_path)
    assert st["state"] == "failed" and "stopped" in st["error"]
    assert not _pid_alive(st["pid"])
    assert (tmp_path / "spool" / "done" / "j1.json").exists()


def test_serve_hands_its_stop_event_to_the_running_job(tmp_path):
    import threading

    _submit(tmp_path)
    w = _make(tmp_path, "import time; time.sleep(25)")
    stop = threading.Event()
    threading.Timer(0.8, stop.set).start()
    w.serve(stop)
    st = _status(tmp_path)
    assert st["state"] == "failed" and not _pid_alive(st["pid"])


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

    def spawn(argv, **kw):
        spawned.append(argv)
        return subprocess.Popen([sys.executable, "-c", "pass"], **kw)

    _make(tmp_path, spawn=spawn).run_once()
    assert spawned == [[sys.executable, "-m", "gomoveshift", subcommand, argument, "--flag"]]
    assert _status(tmp_path)["state"] == "succeeded"
