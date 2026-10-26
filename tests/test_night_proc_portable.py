"""Night worker identity and memory gates run on Windows (psutil) and keep /proc on Linux."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

from gomoveshift.pipeline.photos import night_products, night_project

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def windows(monkeypatch):
    fake = types.SimpleNamespace(
        boot_time=lambda: 1700000000.0,
        Process=lambda pid: types.SimpleNamespace(create_time=lambda: 1700000123.25),
        virtual_memory=lambda: types.SimpleNamespace(available=8 * 1024**3),
    )
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "psutil", fake)
    return fake


def test_windows_identity_comes_from_psutil(windows):
    assert night_project.boot_id() == "1700000000.0"
    assert night_project.process_start(1234) == "1700000123.25"


def test_windows_worker_alive_round_trips(windows):
    worker = {
        "pid": 1234,
        "started_at": night_project.process_start(1234),
        "boot_id": night_project.boot_id(),
    }
    assert night_project.worker_alive(worker)
    assert not night_project.worker_alive({**worker, "boot_id": "other"})


def test_windows_dead_process_is_not_alive(windows, monkeypatch):
    def gone(pid):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(windows, "Process", gone)
    assert not night_project.worker_alive({"pid": 1, "started_at": "x", "boot_id": "1700000000.0"})


def test_windows_available_memory_comes_from_psutil(windows):
    assert night_project.available_memory_bytes() == 8 * 1024**3


@pytest.mark.skipif(not Path("/proc/meminfo").exists(), reason="Linux /proc")
def test_linux_reads_proc_unchanged():
    import os

    assert night_project.boot_id() == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    stat = Path(f"/proc/{os.getpid()}/stat").read_text()
    assert night_project.process_start(os.getpid()) == stat.rsplit(")", 1)[1].split()[19]
    assert night_project.available_memory_bytes() > 0


def test_night_modules_read_proc_only_through_the_helpers():
    """``/proc`` literals live only inside the Linux branch of the helpers."""
    allowed = {"process_start", "boot_id", "available_memory_bytes"}
    for module in (night_project, night_products):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name in allowed:
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    assert "/proc" not in node.value, f"{module.__name__}.{fn.name}"
                if isinstance(node, ast.JoinedStr):
                    assert "/proc" not in ast.unparse(node), f"{module.__name__}.{fn.name}"
