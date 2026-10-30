"""Release gate and checksums for the Windows release workflow.

Run by .github/workflows/windows-release.yml:

    python packaging/windows/release_gate.py gate          # the gate job, tag runs
    python packaging/windows/release_gate.py sums FILE...  # the build job

``gate`` refuses a tag unless (1) the tagged commit is on main, (2) the tag is
``vX.Y.Z`` and names the version in pyproject.toml, and (3) ci.yml has a
completed, successful run for that exact commit. While ci.yml is still running
on the commit (a tag pushed right after the merge), the gate waits for it, up to
``CI_WAIT_SECONDS``, instead of refusing. Every problem is printed and the exit
code is 1, so nothing is built.

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
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tomllib

ROOT = Path(__file__).resolve().parents[2]
_TAG = re.compile(r"v(\d+\.\d+\.\d+)")
# main's ci.yml takes 10-12 minutes; the gate job's timeout sits above this.
CI_WAIT_SECONDS = 30 * 60
CI_POLL_SECONDS = 30
_UNFINISHED = {"queued", "in_progress", "waiting", "requested", "pending"}


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


def wait_for_ci(
    fetch: Callable[[], list[dict[str, Any]]],
    sha: str,
    *,
    timeout: float,
    poll: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    """ci.yml runs for ``sha``, re-fetched while one is unfinished and none passed.

    Returns as soon as there is nothing to wait for (a run passed, or no run for
    the commit is still going) or ``timeout`` seconds have gone by.
    """
    deadline = clock() + timeout
    runs = fetch()
    while clock() + poll <= deadline:
        mine = [r for r in runs if r.get("head_sha") == sha]
        if any(r.get("conclusion") == "success" for r in mine):
            break
        if not any(r.get("status") in _UNFINISHED for r in mine):
            break
        print(f"ci.yml is still running on {sha}; checking again in {poll:g}s", file=sys.stderr)
        sleep(poll)
        runs = fetch()
    return runs


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
    runs = wait_for_ci(
        lambda: _ci_runs(repo, sha), sha, timeout=CI_WAIT_SECONDS, poll=CI_POLL_SECONDS
    )
    problems = gate_problems(sha, tag, version, on_main=_on_main(sha), runs=runs)
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
