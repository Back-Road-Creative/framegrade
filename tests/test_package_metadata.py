"""Guard: package metadata in ``pyproject.toml`` and the pins in ``requirements.lock``.

``ci/install-cpu.sh`` feeds ``requirements.lock`` to ``uv pip install --constraint``, so a
dependency dropped from ``pyproject.toml`` must also leave the lock; otherwise the lock keeps
pinning a package nothing asks for.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]

# Declared once, imported by nothing under gomoveshift/. ``future`` was only ever the
# transitive requirement of ffmpeg-python.
REMOVED_DEPS = {
    "ffmpeg-python": "ffmpeg",
    "watchdog": "watchdog",
    "aiofiles": "aiofiles",
}
REMOVED_LOCK_PINS = {*REMOVED_DEPS, "future"}


def _name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement.strip())
    assert match, requirement
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def _declared() -> set[str]:
    names = {_name(r) for r in PROJECT["dependencies"]}
    for extra in PROJECT.get("optional-dependencies", {}).values():
        names.update(_name(r) for r in extra)
    return names


def _imported_top_level() -> set[str]:
    found: set[str] = set()
    for path in (ROOT / "gomoveshift").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def test_python_floor_is_3_12() -> None:
    assert PROJECT["requires-python"] == ">=3.12"


def test_description_is_not_the_stale_sign_detection_text() -> None:
    description = PROJECT["description"].lower()
    assert "sign detection" not in description
    assert "ocr" not in description
    assert description.strip()


def test_removed_dependencies_are_not_declared() -> None:
    assert not (REMOVED_DEPS.keys() & _declared())


def test_removed_dependencies_are_not_imported() -> None:
    imported = _imported_top_level()
    assert not (set(REMOVED_DEPS.values()) & imported)


def test_lock_carries_no_pin_for_removed_dependencies() -> None:
    pins = {
        _name(line.split("==")[0])
        for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(("#", "-e ")) and "==" in line
    }
    assert not (REMOVED_LOCK_PINS & pins)
