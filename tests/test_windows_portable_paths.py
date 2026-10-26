"""Windows portability of the Linux-only /proc, /tmp and /mnt sites.

Windows is simulated: ``sys.platform`` is patched where a module reads it at
call time, a missing ``/proc`` is faked by making ``Path.read_text`` raise
``FileNotFoundError``, and psutil is replaced by a fake. Linux behaviour is
asserted unchanged beside each Windows case.
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import pytest

from gomoveshift.console import jobs
from gomoveshift.health import heal
from gomoveshift.pipeline import orchestrator_preflight, preflight
from gomoveshift.pipeline.photos import astro_benchmark, develop
from gomoveshift.pipeline.photos import orchestrator as photos_orch


class _FakeProcess:
    def __init__(self, pid):
        self.pid = pid

    def create_time(self):
        return 1700000000.5


def _fake_psutil(raise_for_pid=None):
    def _process(pid):
        if pid == raise_for_pid:
            raise ProcessLookupError(pid)
        return _FakeProcess(pid)

    return types.SimpleNamespace(
        Process=_process,
        boot_time=lambda: 1699990000.25,
        NoSuchProcess=ProcessLookupError,
        Error=OSError,
        virtual_memory=lambda: types.SimpleNamespace(total=16 * 1024**3, available=8 * 1024**3),
    )


@pytest.fixture
def no_proc(monkeypatch):
    """Every /proc read raises FileNotFoundError, as on Windows."""
    real = Path.read_text

    def _read_text(self, *a, **k):
        if str(self).startswith("/proc"):
            raise FileNotFoundError(str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", _read_text)


@pytest.fixture
def windows(monkeypatch, no_proc):
    monkeypatch.setattr(sys, "platform", "win32")
    fake = _fake_psutil(raise_for_pid=-1)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    return fake


# --- console/jobs.py: pid fingerprint ---------------------------------------


def test_jobs_fingerprint_linux_is_proc_field_22():
    import os

    fp = jobs._read_proc_starttime(os.getpid())
    stat = Path(f"/proc/{os.getpid()}/stat").read_text()
    assert fp == stat[stat.rfind(")") + 2 :].split()[19]


def test_jobs_fingerprint_windows_uses_psutil_create_time(windows):
    assert jobs._read_proc_starttime(1234) == "1700000000.5"


def test_jobs_fingerprint_windows_dead_pid_is_none(windows):
    assert jobs._read_proc_starttime(-1) is None


def test_jobs_process_matches_windows(windows):
    assert jobs._process_matches(1234, "1700000000.5")
    assert not jobs._process_matches(1234, "1")


# --- health/heal.py: boot id + start ticks ----------------------------------


def test_heal_linux_identity_unchanged():
    import os

    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    stat = Path(f"/proc/{os.getpid()}/stat").read_text()
    ticks = stat.rsplit(")", 1)[1].split()[19]
    assert heal._proc_alive({"pid": os.getpid(), "boot_id": boot, "start_ticks": ticks})
    assert not heal._proc_alive({"pid": os.getpid(), "boot_id": "other", "start_ticks": ticks})


def test_heal_windows_identity_from_psutil(windows):
    ident = {"pid": 1234, "boot_id": "1699990000.25", "start_ticks": "1700000000.5"}
    assert heal._proc_alive(ident)
    assert not heal._proc_alive({**ident, "boot_id": "other"})
    assert not heal._proc_alive({**ident, "start_ticks": "1"})
    assert not heal._proc_alive({**ident, "pid": -1})


# --- preflight / orchestrator_preflight --------------------------------------


def test_machine_spec_windows_uses_psutil(windows):
    cores, gb = orchestrator_preflight._machine_spec()
    assert gb == 16.0


def test_virtual_memory_windows_uses_psutil(monkeypatch):
    monkeypatch.undo()  # drop conftest's autouse pin so the real probe runs
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil())
    assert preflight._virtual_memory().available == 8 * 1024**3


def test_is_wsl_false_on_windows(windows):
    assert preflight._is_wsl() is False


def test_read_mounts_empty_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert preflight._read_mounts() == []


# --- astro_benchmark: named refusal -------------------------------------------


def test_astro_benchmark_refuses_on_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(astro_benchmark.LinuxOnlyBenchmarkError, match="Linux-only"):
        astro_benchmark.run_benchmark([], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_astro_benchmark_reset_peak_rss_false_without_proc(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert astro_benchmark._reset_peak_rss() is False


# --- config temp folder ---------------------------------------------------------


def test_temp_folder_default_follows_gettempdir(monkeypatch, tmp_path):
    from gomoveshift.config import PathsConfig

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    assert PathsConfig().temp_folder == str(tmp_path / "gomoveshift")


def test_temp_folder_default_linux_is_tmp(monkeypatch):
    from gomoveshift.config import PathsConfig

    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", None)
    assert PathsConfig().temp_folder == "/tmp/gomoveshift"


def test_repo_config_yaml_does_not_pin_tmp():
    text = (Path(__file__).resolve().parent.parent / "config.yaml").read_text()
    assert "/tmp/gomoveshift" not in text


# --- photos orchestrator: PowerShell export ------------------------------------


def test_export_copies_plainly_on_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(Path, "exists", lambda self: True)  # even if a path "exists"
    calls = []
    monkeypatch.setattr(photos_orch.fs_safety, "safe_copy", lambda s, d: calls.append((s, d)))
    monkeypatch.setattr(
        photos_orch.subprocess, "check_output", lambda *a, **k: pytest.fail("wslpath on Windows")
    )
    csv = tmp_path / "a.csv"
    csv.write_text("x")
    photos_orch.PhotoOrchestrator._export_to_windows(csv, tmp_path / "exp")
    assert calls == [(csv, tmp_path / "exp" / "a.csv")]


# --- develop.py: DNG converter -------------------------------------------------


def test_dng_probe_list_has_native_windows_path():
    paths = develop.HE_DNG_CONVERTER_DEFAULT_PATHS
    assert paths[0].startswith("/mnt/c/Program Files/Adobe")
    assert "C:\\Program Files\\Adobe\\Adobe DNG Converter\\Adobe DNG Converter.exe" in paths


def test_native_windows_exe_path_is_not_wsl_converted():
    exe = "C:\\Program Files\\Adobe\\Adobe DNG Converter\\Adobe DNG Converter.exe"
    assert not exe.startswith("/mnt/")
    assert develop._windows_path("D:\\shots\\a.nef") == "D:\\shots\\a.nef"
    assert develop._windows_path("/mnt/d/shots/a.nef") == "D:\\shots\\a.nef"


# --- /mnt/ startswith checks stay inert for Windows paths -----------------------


def test_wsl_mount_checks_inert_for_windows_path():
    from gomoveshift.pipeline import wsl_mount

    win = Path("D:\\clips")
    assert wsl_mount.ensure_wsl_mount(win) is True
    wsl_mount.assert_wsl_mount(win)  # no raise, no subprocess


def test_paths_mnt_check_inert_for_windows_path():
    from gomoveshift import paths

    assert not str(Path("D:\\out")).startswith("/mnt/")
    assert paths is not None
