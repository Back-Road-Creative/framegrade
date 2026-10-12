#!/usr/bin/env python3
"""Reject private/runtime paths, generated artifacts, and dirty product builds.

This is a tracked-file and workspace cleanliness check, not a secret scanner.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

PRIVATE_DIRS = {".git", ".aws", ".ssh", ".gnupg", ".data", ".runtime", ".state"}
GENERATED_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "htmlcov",
}
PRIVATE_NAMES = {
    "credentials",
    "credentials.json",
    "secrets.json",
    "token.json",
    "tokens.json",
    "youtube_upload_token.json",
    "id_rsa",
    "id_ed25519",
    "id_dsa",
    "id_ecdsa",
}
GENERATED_NAMES = {".coverage", "coverage.xml", ".ds_store", "thumbs.db"}
ENV_EXAMPLES = {".env.example", ".env.sample", ".env.template"}


class CheckError(Exception):
    """The check could not safely inspect the requested repository."""


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
            "-c",
            "core.untrackedCache=false",
            *args,
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        # Do not relay raw Git output: it can include remote URLs or credentials.
        raise CheckError(f"Git {args[0]} failed (exit {result.returncode})")
    return result.stdout


def tracked_files(root):
    entries = {}
    for record in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, name = record.split(b"\t", 1)
        mode, _oid, stage = metadata.decode("ascii").split()
        if stage != "0":
            raise CheckError("Git index contains an unresolved merge")
        entries[os.fsdecode(name)] = mode
    return entries


def path_problem(name):
    path = PurePosixPath(name.lower())
    parts, base = set(path.parts), path.name
    if parts & PRIVATE_DIRS:
        return "private runtime/credential directory"
    if base == ".env" or (base.startswith(".env.") and base not in ENV_EXAMPLES):
        return "environment credentials/configuration"
    if base in PRIVATE_NAMES or base.endswith((".enc", ".key", ".p12", ".pfx", ".jks")):
        return "credential or encrypted private file"
    if re.fullmatch(
        r"(?:credentials?|secrets?|tokens?|client_secret[^/]*|service_account[^/]*)\.(?:json|ya?ml|toml|ini)",
        base,
    ):
        return "credential configuration file"
    if re.search(r"\.(?:db|sqlite|sqlite3)(?:-(?:wal|shm|journal))?$", base):
        return "runtime database"
    if parts & GENERATED_DIRS or any(part.endswith(".egg-info") for part in path.parts):
        return "generated build/cache directory"
    if base in GENERATED_NAMES or base.startswith(".coverage.") or base.endswith((".pyc", ".pyo")):
        return "generated build/cache artifact"
    return None


def read_policy(root, policy_path, entries):
    path = root / policy_path
    if path.is_symlink():
        raise CheckError("policy must be a regular tracked file, not a symlink")
    try:
        relative = path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise CheckError("policy must be inside the repository") from error
    if relative not in entries or entries[relative] not in {"100644", "100755"}:
        raise CheckError("policy must be a tracked regular file")
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckError("policy is missing, unreadable, or invalid JSON") from error
    if (
        not isinstance(policy, dict)
        or set(policy) != {"version", "fixture_exceptions"}
        or type(policy["version"]) is not int
        or policy["version"] != 1
        or not isinstance(policy["fixture_exceptions"], list)
    ):
        raise CheckError("policy requires version 1 and fixture_exceptions only")
    exceptions = {}
    for item in policy["fixture_exceptions"]:
        if not isinstance(item, dict) or set(item) != {"path", "reason", "sha256"}:
            raise CheckError("each fixture exception requires path, reason, sha256")
        name, reason, digest = item["path"], item["reason"], item["sha256"]
        if (
            not isinstance(name, str)
            or not name.startswith("tests/fixtures/")
            or str(PurePosixPath(name)) != name
            or ".." in PurePosixPath(name).parts
            or any(char in name for char in "*?[]\\")
            or not isinstance(reason, str)
            or not reason.strip()
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise CheckError(
                "fixture exceptions require an exact tests/fixtures path, reason, SHA-256"
            )
        if (
            name in exceptions
            or name not in entries
            or entries[name] not in {"100644", "100755"}
            or not path_problem(name)
        ):
            raise CheckError(f"duplicate, stale, or unnecessary fixture exception: {name!r}")
        exceptions[name] = digest
    return exceptions


def safe_link(root, name, target):
    # Absolute links are non-portable even if they happen to point inside this clone.
    if os.path.isabs(target):
        return False
    try:
        relative = (root / name).parent.joinpath(target).resolve().relative_to(root)
    except (ValueError, RuntimeError, OSError):
        return False
    return path_problem(relative.as_posix()) is None


def inspect_tree(root, entries, exceptions):
    problems = []
    for record in git(root, "ls-files", "-v", "-z").split(b"\0"):
        if record and (record[:1].islower() or record[:1] == b"S"):
            problems.append(
                f"{os.fsdecode(record[2:])!r}: hidden index flag (assume-unchanged/skip-worktree)"
            )
    for name, mode in entries.items():
        path = root / name
        if mode not in {"100644", "100755", "120000"}:
            problems.append(f"{name!r}: opaque gitlink or unsupported tracked file mode")
            continue
        if not path.exists() and not path.is_symlink():
            problems.append(f"{name!r}: tracked file missing from checkout")
            continue
        if path.is_symlink() and not safe_link(root, name, os.readlink(path)):
            problems.append(
                f"{name!r}: checkout symlink has unsafe target (outside/private/generated/cyclic)"
            )
            continue
        if mode == "120000":
            target = os.fsdecode(git(root, "cat-file", "blob", f":{name}"))
            if not safe_link(root, name, target):
                problems.append(
                    f"{name!r}: tracked symlink has unsafe target (outside/private/generated/cyclic)"
                )
                continue
        problem = path_problem(name)
        if not problem:
            continue
        if name in exceptions and not path.is_symlink():
            if hashlib.sha256(path.read_bytes()).hexdigest() == exceptions[name]:
                continue
            problem = "synthetic fixture changed: reviewed SHA-256 no longer matches"
        problems.append(f"{name!r}: {problem}")
    return problems


def dirty_paths(root):
    records = iter(
        git(
            root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignored=no",
            "--ignore-submodules=none",
        ).split(b"\0")
    )
    problems = []
    for record in records:
        if not record:
            continue
        status = record[:2].decode("ascii")
        name = os.fsdecode(record[3:])
        problems.append(f"{name!r}: unexpected Git status {status!r}")
        if "R" in status or "C" in status:
            next(records)  # NUL-delimited original name follows a rename/copy.
    return problems


def unexpected_ignored_paths(root):
    problems = []
    # Enumerate whole ignored directories, not every dependency in a virtualenv.
    records = git(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--directory",
        "--no-empty-directory",
        "-z",
    ).split(b"\0")
    for record in records:
        if not record:
            continue
        name = os.fsdecode(record).rstrip("/")
        problem = path_problem(name)
        if problem in {
            "generated build/cache directory",
            "generated build/cache artifact",
        }:
            continue
        problems.append(
            f"{name!r}: ignored {problem or 'unexpected output'}; only known build/cache output is allowed"
        )
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="exact product Git root")
    parser.add_argument("--policy", default="tools/public-tree-policy.json")
    parser.add_argument(
        "--check-clean",
        action="store_true",
        help="also reject tracked/untracked changes",
    )
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        actual_root = Path(
            os.fsdecode(git(root, "rev-parse", "--show-toplevel")).rstrip("\n")
        ).resolve()
        if actual_root != root:
            raise CheckError("--root must name the exact Git root; refusing a parent repository")
        entries = tracked_files(root)
        exceptions = read_policy(root, args.policy, entries)
        problems = inspect_tree(root, entries, exceptions)
        if args.check_clean:
            problems.extend(dirty_paths(root))
            problems.extend(unexpected_ignored_paths(root))
    except (CheckError, OSError, ValueError, RuntimeError) as error:
        print(f"public-tree: ERROR: {error}", file=sys.stderr)
        return 2
    if problems:
        for problem in problems:
            print(f"public-tree: FAIL: {problem}", file=sys.stderr)
        return 1
    print(
        f"public-tree: PASS ({len(entries)} tracked files"
        + (", clean checkout)" if args.check_clean else ")")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
