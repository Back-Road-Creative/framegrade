#!/usr/bin/env python3
"""Trusted data-only PR path comparison and fail-closed CI aggregation."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

SHA = re.compile(r"[0-9a-f]{40}")


def git(root, *args):
    result = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "--literal-pathspecs",
            "-C",
            str(root),
            "-c",
            "core.fsmonitor=false",
            *args,
        ],
        capture_output=True,
    )
    if result.returncode:
        raise ValueError(f"Git {args[0]} failed (exit {result.returncode})")
    return result.stdout


def protected(name):
    # Protect both guard dependencies and the product's quality helper scripts.
    return (
        name.startswith(("tools/", "ci/", ".github/workflows/", ".github/actions/"))
        or PurePosixPath(name).name == ".gitignore"
    )


def inspect(root, base, head):
    if not SHA.fullmatch(base) or not SHA.fullmatch(head):
        raise ValueError("base/head must be explicit lowercase 40-digit commit SHAs")
    root = root.resolve(strict=True)
    actual = Path(os.fsdecode(git(root, "rev-parse", "--show-toplevel")).rstrip("\n"))
    if actual.resolve() != root:
        raise ValueError("candidate must be the exact Git root")
    for commit in (base, head):
        git(root, "cat-file", "-e", f"{commit}^{{commit}}")
    if git(root, "rev-parse", "HEAD").decode().strip() != head:
        raise ValueError("candidate checkout HEAD does not equal the requested head SHA")
    changed = git(
        root,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--name-only",
        "-z",
        base,
        head,
        "--",
    ).split(b"\0")
    result = {
        "maintenance_changed": str(any(protected(os.fsdecode(p)) for p in changed if p)).lower(),
        "inspected_sha": head,
    }
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as stream:
            for key, value in result.items():
                stream.write(f"{key}={value}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


def aggregate():
    expected = os.environ.get("EXPECTED_SHA", "")
    inspected = os.environ.get("INSPECTED_SHA", "")
    changed = os.environ.get("MAINTENANCE_CHANGED", "")
    inspection = os.environ.get("INSPECTION_RESULT", "")
    approval = os.environ.get("APPROVAL_RESULT", "")
    allowed = (changed == "false" and approval == "skipped") or (
        changed == "true" and approval == "success"
    )
    if (
        not SHA.fullmatch(expected)
        or inspected != expected
        or inspection != "success"
        or not allowed
    ):
        print(
            "cleanliness-ci: FAIL: incomplete inspection, identity, or approval",
            file=sys.stderr,
        )
        return 1
    print(f"cleanliness-ci: PASS: inspected {inspected}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("inspect")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--base-sha", required=True)
    check.add_argument("--head-sha", required=True)
    commands.add_parser("aggregate")
    args = parser.parse_args()
    try:
        if args.command == "aggregate":
            return aggregate()
        return inspect(args.root, args.base_sha, args.head_sha)
    except (OSError, ValueError) as error:
        print(f"cleanliness-ci: ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
