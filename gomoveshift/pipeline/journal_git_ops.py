"""Git operations for the publish-journal replay-pending sweep.

Lives in :mod:`gomoveshift.pipeline` so the CLI handler can extend its
replay mode with ``--auto-commit`` / ``--auto-push`` without growing
shell-out logic in two places.

Design contract:

- **Skip-on-dirty, never auto-stash.** If the target repo has uncommitted
  changes to any tracked file, the sweep refuses to write a commit and
  logs the conflict. Untracked files do NOT count as dirty (they're
  typically WIP scratch artifacts).
- **Per-repo grouping.** Multiple journey files in the same git repo
  collapse into one commit, not one per journey, to keep the history
  legible and the push count small.
- **Push failures are loud.** No retry, no rebase. A non-zero exit is the
  signal the operator's cron alerting should fire on.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class GitOpError(RuntimeError):
    """Raised on any git subprocess failure during the replay sweep."""


@dataclass
class CommitResult:
    repo_root: Path
    commit_sha: Optional[str]  # None when nothing was committed (no diff after stage)
    pushed: bool
    skipped_reason: Optional[str] = None  # "dirty", "no changes", etc.


def git_toplevel(path: Path) -> Optional[Path]:
    """Return the git working-tree root for ``path``, or None when not in a repo.

    Accepts either a file or directory; ``git -C`` requires a directory so
    we resolve the parent dir for file paths (or use the path itself when
    it's a dir).
    """
    cwd = path if path.is_dir() else path.parent
    try:
        result = _run(["git", "-C", str(cwd), "rev-parse", "--show-toplevel"])
    except GitOpError:
        return None
    return Path(result.strip())


def is_tracked_dirty(repo_root: Path) -> bool:
    """True iff the working tree has uncommitted changes to tracked files.

    Untracked files are NOT dirty under this definition: a stray WIP file
    elsewhere in the repo should not block a journey flush.
    """
    # `git diff --quiet` exits 1 iff there are unstaged changes to tracked files.
    # `git diff --cached --quiet` exits 1 iff there are staged but uncommitted
    # changes. Either signal means dirty.
    unstaged = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--quiet"],
        check=False,
    )
    if unstaged.returncode == 1:
        return True
    staged = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--cached", "--quiet"],
        check=False,
    )
    return staged.returncode == 1


def commit_paths(
    repo_root: Path,
    paths: list[Path],
    message: str,
    push: bool,
) -> CommitResult:
    """Stage ``paths``, commit with ``message``, optionally push.

    Caller is responsible for verifying ``is_tracked_dirty`` BEFORE writing
    the paths — by the time we get here, the only diff against HEAD should
    be the paths we wrote.

    Returns a :class:`CommitResult`. ``commit_sha=None`` means the stage
    produced no actual diff (idempotent re-run of a flush that was already
    committed); this is treated as success, not error.
    """
    if not paths:
        return CommitResult(
            repo_root=repo_root, commit_sha=None, pushed=False, skipped_reason="no paths"
        )

    rel = [str(p.relative_to(repo_root)) for p in paths]
    _run(["git", "-C", str(repo_root), "add", "--", *rel])

    # If nothing actually changed (idempotent re-run), commit would error.
    diff = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--cached", "--quiet"],
        check=False,
    )
    if diff.returncode == 0:
        return CommitResult(
            repo_root=repo_root,
            commit_sha=None,
            pushed=False,
            skipped_reason="no diff after stage",
        )

    _run(["git", "-C", str(repo_root), "commit", "-m", message])
    sha = _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"]).strip()

    pushed = False
    if push:
        # Loud failure: let GitOpError bubble; the caller turns it into a
        # cron-visible non-zero exit. Pull-rebase is intentionally NOT done
        # here — silently rebasing journey commits onto unseen remote work
        # would mask conflicts a human should resolve.
        _run(["git", "-C", str(repo_root), "push"])
        pushed = True

    return CommitResult(repo_root=repo_root, commit_sha=sha, pushed=pushed)


def _run(cmd: list[str]) -> str:
    """Run a git command, raising :class:`GitOpError` on non-zero exit.

    Captures both stdout and stderr so the error message names the actual
    failure rather than the operator's invocation.
    """
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise GitOpError(
            f"git command failed (rc={result.returncode}): {' '.join(cmd)}\n"
            f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()}"
        )
    return result.stdout
