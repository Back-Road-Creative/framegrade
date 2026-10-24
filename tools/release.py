"""Prepare a release commit: ``python3 -I tools/release.py 0.3.5``.

Bumps the version in ``pyproject.toml`` and ``gomoveshift/__init__.py``, gives
the ``[Unreleased]`` entries a dated ``## [X.Y.Z]`` heading and moves the
CHANGELOG link references on (the shape ``tests/test_changelog.py`` checks).
Each file keeps its own line endings. Nothing is written unless every edit
applies. Committing, the PR and the tag stay with the caller (README, Releasing).
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

REPO_URL = "https://github.com/Back-Road-Creative/framegrade"
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


class ReleaseError(Exception):
    """The tree is not in a releasable shape; nothing was written."""


def _read(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as fh:
        return fh.read()


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def _replace_once(name: str, text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ReleaseError(f"{name}: expected exactly one {old!r}")
    return text.replace(old, new)


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("."))


def release(root: Path, version: str, date: str) -> str:
    """Write release ``version`` dated ``date`` into ``root``; return the old version."""
    pyproject = root / "pyproject.toml"
    init = root / "gomoveshift" / "__init__.py"
    changelog = root / "CHANGELOG.md"
    py_text, init_text, log_text = _read(pyproject), _read(init), _read(changelog)

    found = re.search(r'^version = "([^"]+)"', py_text, flags=re.MULTILINE)
    if not found:
        raise ReleaseError("pyproject.toml: no [project] version")
    current = found.group(1)
    if not _VERSION.match(version) or _key(version) <= _key(current):
        raise ReleaseError(f"{version!r} is not an X.Y.Z version after {current}")

    py_text = _replace_once(
        "pyproject.toml", py_text, f'version = "{current}"', f'version = "{version}"'
    )
    init_text = _replace_once(
        "gomoveshift/__init__.py",
        init_text,
        f'__version__ = "{current}"',
        f'__version__ = "{version}"',
    )

    nl = "\r\n" if "\r\n" in log_text else "\n"
    head = f"## [Unreleased]{nl}{nl}"
    if log_text.count(head) != 1:
        raise ReleaseError("CHANGELOG.md: expected exactly one [Unreleased] heading")
    if log_text.split(head, 1)[1].startswith("## ["):
        raise ReleaseError("CHANGELOG.md: [Unreleased] is empty, nothing to release")
    log_text = log_text.replace(head, f"{head}## [{version}] - {date}{nl}{nl}")
    log_text = _replace_once(
        "CHANGELOG.md",
        log_text,
        f"[Unreleased]: {REPO_URL}/compare/v{current}...HEAD{nl}",
        f"[Unreleased]: {REPO_URL}/compare/v{version}...HEAD{nl}"
        f"[{version}]: {REPO_URL}/compare/v{current}...v{version}{nl}",
    )

    _write(pyproject, py_text)
    _write(init, init_text)
    _write(changelog, log_text)
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="the new X.Y.Z version")
    parser.add_argument(
        "--date", default=datetime.date.today().isoformat(), help="release date (default today)"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    try:
        old = release(args.root, args.version, args.date)
    except ReleaseError as exc:
        print(f"release: {exc}", file=sys.stderr)
        return 1
    print(f"{old} -> {args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
