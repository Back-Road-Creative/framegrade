#!/usr/bin/env python3
"""Import every module of a package and hold the failures to a written list.

The Windows CI job runs this against ``gomoveshift`` with
``ci/windows-import-failures.txt``: a module that newly fails to import fails
the job, and so does a listed module that now imports, so the list only
shrinks. Each failure prints one ``IMPORT_SMOKE`` line with its error.

  python ci/import_smoke.py [--package gomoveshift] [--expected FILE]
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from pathlib import Path


def modules(package: str) -> list[str]:
    """The package and every module under it, sorted, leaving out ``__main__``."""
    root = importlib.import_module(package)
    found = [package]
    for info in pkgutil.walk_packages(
        root.__path__, f"{package}.", onerror=lambda name: None
    ):
        if info.name.rsplit(".", 1)[-1] != "__main__":
            found.append(info.name)
    return sorted(found)


def read_expected(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            names.add(name)
    return names


def check(package: str, expected: set[str]) -> int:
    names = modules(package)
    failed = {}
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # any import-time error counts, not just ImportError
            failed[name] = f"{type(exc).__name__}: {exc}"
    problems = []
    for name in sorted(set(failed) - expected):
        problems.append(f"IMPORT_SMOKE unexpected={name} {failed[name]}")
    for name in sorted(expected - set(failed)):
        if name in names:
            problems.append(f"IMPORT_SMOKE fixed={name} (delete it from the list)")
        else:
            problems.append(f"IMPORT_SMOKE unknown={name} (no such module)")
    for line in problems:
        print(line)
    if problems:
        return 1
    print(f"IMPORT_SMOKE ok modules={len(names)} expected_failures={len(expected)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", default="gomoveshift")
    parser.add_argument(
        "--expected", type=Path, help="modules allowed to fail, one per line"
    )
    args = parser.parse_args(argv)
    expected = read_expected(args.expected) if args.expected else set()
    return check(args.package, expected)


if __name__ == "__main__":
    sys.exit(main())
