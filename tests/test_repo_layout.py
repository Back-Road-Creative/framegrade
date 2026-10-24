"""Repo layout guards: test modules live in a ``tests/`` directory, never loose."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    return [p for p in out.decode().split("\0") if p]


def test_no_test_modules_outside_tests_dir():
    stray = [
        p
        for p in _tracked_files()
        if Path(p).name.startswith("test_")
        and Path(p).suffix == ".py"
        and "tests" not in Path(p).parts[:-1]
    ]
    assert stray == [], f"test_*.py outside a tests/ directory: {stray}"


def test_scripts_dir_is_gone():
    assert not (REPO_ROOT / "scripts").exists()
