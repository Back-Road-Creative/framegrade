"""Prepare a release commit: ``python3 -I tools/release.py 0.3.5``.

Bumps the version in ``pyproject.toml`` and ``gomoveshift/__init__.py``, folds
each ``changelog.d/*.md`` fragment into the ``[Unreleased]`` entries (by
section, then deletes it), gives those entries a dated ``## [X.Y.Z]`` heading
and moves the CHANGELOG link references on (the shape
``tests/test_changelog.py`` checks).
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
#: Keep a Changelog sections, in the order a release lists them. A change adds
#: ``changelog.d/<name>.md`` (one or more ``### <Section>`` blocks of bullets)
#: instead of editing CHANGELOG.md, so parallel PRs never conflict on it.
SECTIONS = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")
FRAGMENTS_DIR = "changelog.d"
_SECTION = re.compile(r"^### (.+?)\s*$", flags=re.MULTILINE)


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


def _sections(name: str, text: str) -> dict[str, list[str]]:
    """``### <Section>`` blocks of ``text`` (``\\n`` line endings) -> their entries."""
    found = list(_SECTION.finditer(text))
    if text.strip() and (not found or text[: found[0].start()].strip()):
        raise ReleaseError(f"{name}: must start with a '### <Section>' heading")
    out: dict[str, list[str]] = {}
    for match, nxt in zip(found, [*found[1:], None]):
        body = text[match.end() : nxt.start() if nxt else len(text)].strip("\n")
        if not body.strip():
            raise ReleaseError(f"{name}: '### {match.group(1)}' has no entries")
        out.setdefault(match.group(1), []).append(body)
    return out


def fragment_paths(root: Path) -> list[Path]:
    """The ``changelog.d`` fragments the next release folds in, in name order."""
    frags = root / FRAGMENTS_DIR
    if not frags.is_dir():
        return []
    return sorted(p for p in frags.glob("*.md") if p.name != "README.md")


def read_fragment(path: Path) -> dict[str, list[str]]:
    """One fragment's entries by section; refuses a section outside ``SECTIONS``."""
    name = f"{FRAGMENTS_DIR}/{path.name}"
    sections = _sections(name, _read(path).replace("\r\n", "\n"))
    if not sections:
        raise ReleaseError(f"{name}: must start with a '### <Section>' heading")
    unknown = [s for s in sections if s not in SECTIONS]
    if unknown:
        raise ReleaseError(f"{name}: unknown section {unknown[0]!r} (use one of {SECTIONS})")
    return sections


def _fold(body: str, fragments: list[dict[str, list[str]]]) -> str:
    """``[Unreleased]`` ``body`` plus ``fragments``, one block per section, ``\\n`` endings."""
    merged = _sections("CHANGELOG.md [Unreleased]", body)
    for frag in fragments:
        for section, blocks in frag.items():
            merged.setdefault(section, []).extend(blocks)
    order = [s for s in SECTIONS if s in merged] + [s for s in merged if s not in SECTIONS]
    return "".join(f"### {s}\n\n" + "\n".join(merged[s]) + "\n\n" for s in order)


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
    before, rest = log_text.split(head, 1)
    cut = re.search(r"^## \[", rest, flags=re.MULTILINE)
    if cut is None:
        raise ReleaseError("CHANGELOG.md: no released version under [Unreleased]")
    paths = fragment_paths(root)
    entries = _fold(rest[: cut.start()].replace(nl, "\n"), [read_fragment(p) for p in paths])
    if not entries:
        raise ReleaseError("CHANGELOG.md: [Unreleased] is empty, nothing to release")
    entries = entries.replace("\n", nl)
    log_text = f"{before}{head}## [{version}] - {date}{nl}{nl}{entries}{rest[cut.start() :]}"
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
    for path in paths:
        path.unlink()
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
