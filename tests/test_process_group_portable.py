"""Portable process-group start/kill: POSIX unchanged, Windows via taskkill."""

from __future__ import annotations

import ast
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from gomoveshift import _process_group as pg

PKG = Path(pg.__file__).parent

_PARENT = textwrap.dedent(
    """
    import subprocess, sys, time
    g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    print(g.pid, flush=True)
    time.sleep(60)
    """
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_posix_kill_group_reaps_child_and_grandchild():
    proc = subprocess.Popen(
        [sys.executable, "-c", _PARENT], stdout=subprocess.PIPE, text=True, **pg.popen_kwargs()
    )
    grandchild = int(proc.stdout.readline())
    assert _alive(grandchild)
    pg.kill_group(proc)
    proc.wait(timeout=10)
    deadline = time.time() + 10
    while _alive(grandchild) and time.time() < deadline:
        time.sleep(0.1)
    assert not _alive(grandchild)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_posix_terminate_group_sends_sigterm_to_the_group(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    pg.terminate_group(type("P", (), {"pid": 77})())
    assert calls == [(77, signal.SIGTERM)]


def test_posix_popen_kwargs(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    assert pg.popen_kwargs() == {"start_new_session": True}


def test_windows_popen_kwargs(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    assert pg.popen_kwargs() == {"creationflags": 512}


class _FakeProc:
    pid = 4321

    def __init__(self):
        self.sent = []
        self.killed = False

    def send_signal(self, sig):
        self.sent.append(sig)

    def kill(self):
        self.killed = True


def test_windows_terminate_sends_ctrl_break(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(signal, "CTRL_BREAK_EVENT", 1, raising=False)
    proc = _FakeProc()
    pg.terminate_group(proc)
    assert proc.sent == [1]


def test_windows_kill_group_runs_taskkill_tree(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    seen = []

    def fake_run(argv, **kw):
        seen.append((argv, kw))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    proc = _FakeProc()
    pg.kill_group(proc)
    argv, kw = seen[0]
    assert argv == ["taskkill", "/T", "/F", "/PID", "4321"]
    assert not kw.get("shell")
    assert not proc.killed


def test_windows_kill_group_falls_back_to_proc_kill(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")

    def boom(argv, **kw):
        raise FileNotFoundError("taskkill")

    monkeypatch.setattr(subprocess, "run", boom)
    proc = _FakeProc()
    pg.kill_group(proc)
    assert proc.killed

    proc2 = _FakeProc()
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 128)
    )
    pg.kill_group(proc2)
    assert proc2.killed


@pytest.mark.skipif(os.name != "nt", reason="real Windows process groups")
def test_windows_real_group_start_and_kill():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], **pg.popen_kwargs()
    )
    assert proc.poll() is None
    pg.kill_group(proc)
    assert proc.wait(timeout=15) is not None


def test_posix_group_calls_live_only_in_the_helper():
    offenders = []
    for path in PKG.rglob("*.py"):
        if path.name == "_process_group.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Attribute) and node.attr in ("killpg", "getpgid"):
                name = node.attr
            elif isinstance(node, ast.keyword) and node.arg == "start_new_session":
                name = node.arg
            if name:
                offenders.append(f"{path.relative_to(PKG)}:{name}")
    assert offenders == []
