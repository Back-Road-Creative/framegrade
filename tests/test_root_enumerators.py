"""tests/root_enumerators.json: the service tests that walk this package.

bin/pytest-affected.sh adds a test named here when a changed path matches one
of its globs, so a package walker (an ast/rglob sweep that imports nothing by
name) runs on the change that can break it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SVC = Path(__file__).resolve().parents[1]
REGISTRY = SVC / "tests" / "root_enumerators.json"
SELECTOR = SVC.parents[1] / "bin" / "root-enumerators.py"


def _tests() -> dict[str, list[str]]:
    return json.loads(REGISTRY.read_text())["tests"]


def _selector():
    spec = importlib.util.spec_from_file_location("root_enumerators", SELECTOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_registered_test_exists():
    missing = [t for t in _tests() if not (SVC / t).is_file()]
    assert not missing, f"registry names tests that do not exist: {missing}"


def test_every_glob_is_a_whole_subtree():
    bad = [
        (t, g)
        for t, globs in _tests().items()
        for g in globs
        if g != "**" and not g.endswith("/**")
    ]
    assert not bad, f"declare a whole subtree (<dir>/**), not a partial glob: {bad}"


@pytest.mark.skipif(
    not SELECTOR.is_file(),
    reason="the selector ships with the monorepo harness (bin/pytest-affected.sh), not this package",
)
def test_a_package_change_selects_the_platform_client_boundary_walker():
    # 2026-10-15: a gomoveshift/ change at 209a8d418 broke this walker and the
    # service selection never ran it.
    picked = _selector().selected(_tests(), ["gomoveshift/cli/handlers/queue_recover.py"])
    assert "tests/test_platform_client_boundary.py" in picked
