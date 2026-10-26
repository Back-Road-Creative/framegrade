"""Release gate and checksums for the Windows release workflow.

Run by .github/workflows/windows-release.yml:

    python packaging/windows/release_gate.py gate          # the gate job, tag runs
    python packaging/windows/release_gate.py sums FILE...  # the build job

``gate`` refuses a tag unless (1) the tagged commit is on main, (2) the tag is
``vX.Y.Z`` and names the version in pyproject.toml, and (3) ci.yml has a
completed, successful run for that exact commit. Every problem is printed and
the exit code is 1, so nothing is built.

``sums`` size-gates every asset (GitHub Releases' 2 GiB per-file limit, the
same gate build_release.py applies to its outputs) and writes ``SHA256SUMS``
next to them, one ``<sha256>  <filename>`` line per file, so
``sha256sum -c SHA256SUMS`` checks a download.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import tomllib

ROOT = Path(__file__).resolve().parents[2]
_TAG = re.compile(r"v(\d+\.\d+\.\d+)")


def tag_version(tag: str) -> str | None:
    """``0.3.10`` for ``v0.3.10``; None when the tag is not the release form."""
    found = _TAG.fullmatch(tag)
    return found.group(1) if found else None


def gate_problems(
    sha: str,
    tag: str,
    pyproject_version: str,
    *,
    on_main: bool,
    runs: list[dict[str, Any]],
) -> list[str]:
    """What stops the tag from being released; empty means it may go ahead."""
    problems = []
    version = tag_version(tag)
    if version is None:
        problems.append(f"tag {tag!r} is not a release tag: use vX.Y.Z")
    elif version != pyproject_version:
        problems.append(
            f"tag {tag} means version {version}, pyproject.toml says {pyproject_version}"
        )
    if not on_main:
        problems.append(f"commit {sha} is not on main: tag a commit that was merged to main")
    passed = any(
        r.get("head_sha") == sha
        and r.get("status") == "completed"
        and r.get("conclusion") == "success"
        for r in runs
    )
    if not passed:
        problems.append(
            f"no successful ci.yml run for commit {sha}: wait for main's CI to pass, "
            "then re-run this workflow"
        )
    return problems


def sums_text(paths: list[Path]) -> str:
    """``<sha256>  <filename>`` lines, sorted by filename."""
    lines = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
        for p in sorted(paths, key=lambda p: p.name)
    ]
    return "\n".join(lines) + "\n"


def check_sizes(paths: list[Path], limit: int | None = None) -> None:
    """Exit non-zero naming any asset over the per-file limit."""
    spec = importlib.util.spec_from_file_location(
        "build_release", Path(__file__).resolve().with_name("build_release.py")
    )
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load build_release.py")
    build_release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_release)
    build_release.size_gate(paths, build_release.MAX_ASSET_BYTES if limit is None else limit)


def _on_main(sha: str) -> bool:
    done = subprocess.run(  # noqa: S603
        ["git", "merge-base", "--is-ancestor", sha, "origin/main"],  # noqa: S607
        check=False,
    )
    return done.returncode == 0


def _ci_runs(repo: str, sha: str) -> list[dict[str, Any]]:
    out = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "gh",
            "api",
            f"repos/{repo}/actions/workflows/ci.yml/runs?head_sha={sha}&per_page=100",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    runs: list[dict[str, Any]] = json.loads(out)["workflow_runs"]
    return runs


def _gate() -> int:
    sha, tag, repo = (
        os.environ["GITHUB_SHA"],
        os.environ["GITHUB_REF_NAME"],
        os.environ["GITHUB_REPOSITORY"],
    )
    with (ROOT / "pyproject.toml").open("rb") as fh:
        version = tomllib.load(fh)["project"]["version"]
    problems = gate_problems(sha, tag, version, on_main=_on_main(sha), runs=_ci_runs(repo, sha))
    for p in problems:
        print(f"release refused: {p}", file=sys.stderr)
    if not problems:
        print(f"release gate passed: {tag} is {sha}, on main, version matches, ci.yml passed it")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["gate"] and len(args) == 1:
        return _gate()
    if args[:1] == ["sums"] and len(args) > 1:
        paths = [Path(a) for a in args[1:]]
        check_sizes(paths)
        out = paths[0].parent / "SHA256SUMS"
        out.write_text(sums_text(paths), encoding="ascii", newline="\n")
        print(out.read_text(encoding="ascii"), end="")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
