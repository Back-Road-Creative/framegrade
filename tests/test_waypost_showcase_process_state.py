"""`bin/waypost-showcase.py` boots the console in-process to capture its pages,
so it must hand the process back as it found it. The console environment it
sets for the fixture points into a temporary directory that is deleted when
``main`` returns; left in place, every later caller in the same process (the
next test on an xdist worker) reads a data dir that no longer exists.

Loaded by path, like ``test_waypost_showcase.py``: a hyphenated filename
cannot be imported as a module.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

GENERATOR = Path(__file__).resolve().parents[1] / "bin" / "waypost-showcase.py"
_spec = importlib.util.spec_from_file_location("waypost_showcase_process_state", GENERATOR)
assert _spec is not None and _spec.loader is not None, f"{GENERATOR} is not importable"
showcase = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(showcase)

CONSOLE_ENV = (
    "GMS_CONSOLE_DATA_DIR",
    "GMS_CONSOLE_BRAND",
    "GMS_CONSOLE_TOKEN",
    "GMS_CONSOLE_TICKER",
)


@pytest.mark.parametrize("caller_set", [True, False], ids=["caller-set", "caller-unset"])
def test_generating_the_bundle_leaves_the_environment_as_it_found_it(
    tmp_path, monkeypatch, caller_set
):
    for name in CONSOLE_ENV:
        if caller_set:
            monkeypatch.setenv(name, f"caller-{name.lower()}")
        else:
            monkeypatch.delenv(name, raising=False)
    before = dict(os.environ)
    assert showcase.main(["--out", str(tmp_path / "out"), "--source", "abc1234"]) == 0
    assert dict(os.environ) == before


def test_generating_the_bundle_leaves_the_working_directory_as_it_found_it(tmp_path):
    cwd = Path.cwd()
    assert showcase.main(["--out", str(tmp_path / "out"), "--source", "abc1234"]) == 0
    assert Path.cwd() == cwd
