"""Unit tests for journal_git_ops — real tmp git repos, no mocks.

Real git is cheap (sub-second per test) and catches subtle subprocess
issues that a mocked git would let slide.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gomoveshift.pipeline.journal_git_ops import (
    GitOpError,
    commit_paths,
    git_toplevel,
    is_tracked_dirty,
)


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    # Need a baseline commit + identity so subsequent ops work.
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"],
        check=True,
    )
    (path / "README.md").write_text("seed\n")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "seed"], check=True)


def test_git_toplevel_resolves_repo_root(tmp_path: Path):
    _init_repo(tmp_path)
    nested = tmp_path / "content" / "journeys"
    nested.mkdir(parents=True)
    (nested / "trip.md").write_text("---\ntitle: x\n---\n")
    assert git_toplevel(nested / "trip.md") == tmp_path


def test_git_toplevel_returns_none_outside_repo(tmp_path: Path):
    # tmp_path itself is not a git repo here.
    assert git_toplevel(tmp_path / "loose.txt") is None


def test_is_tracked_dirty_false_on_clean(tmp_path: Path):
    _init_repo(tmp_path)
    assert is_tracked_dirty(tmp_path) is False


def test_is_tracked_dirty_true_on_unstaged_modification(tmp_path: Path):
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("dirty\n")
    assert is_tracked_dirty(tmp_path) is True


def test_is_tracked_dirty_true_on_staged_modification(tmp_path: Path):
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("dirty\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    assert is_tracked_dirty(tmp_path) is True


def test_is_tracked_dirty_false_when_only_untracked(tmp_path: Path):
    """Untracked files do NOT make the repo dirty — they're WIP scratch."""
    _init_repo(tmp_path)
    (tmp_path / "scratch.md").write_text("notes\n")
    assert is_tracked_dirty(tmp_path) is False


def test_commit_paths_commits_modified_file(tmp_path: Path):
    _init_repo(tmp_path)
    (tmp_path / "journey.md").write_text("v1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "journey.md"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "add journey"], check=True)

    # Now modify and commit via the helper.
    (tmp_path / "journey.md").write_text("v2\n")
    result = commit_paths(
        tmp_path,
        [tmp_path / "journey.md"],
        message="chore: bump journey",
        push=False,
    )
    assert result.commit_sha is not None
    assert result.pushed is False
    log = subprocess.run(
        ["git", "-C", str(tmp_path), "log", "-1", "--pretty=%s"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert log.stdout.strip() == "chore: bump journey"


def test_commit_paths_no_diff_after_stage_is_not_error(tmp_path: Path):
    """Idempotent re-run: file content unchanged → return without committing."""
    _init_repo(tmp_path)
    (tmp_path / "journey.md").write_text("v1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "journey.md"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "add journey"], check=True)

    # Same content, no changes — commit_paths should report no-op.
    result = commit_paths(
        tmp_path,
        [tmp_path / "journey.md"],
        message="should not commit",
        push=False,
    )
    assert result.commit_sha is None
    assert result.skipped_reason == "no diff after stage"
    assert result.pushed is False


def test_commit_paths_push_failure_raises(tmp_path: Path):
    """A repo without a remote raises on push — caller decides what to do."""
    _init_repo(tmp_path)
    (tmp_path / "journey.md").write_text("v1\n")
    with pytest.raises(GitOpError, match="git command failed"):
        commit_paths(
            tmp_path,
            [tmp_path / "journey.md"],
            message="will fail push",
            push=True,
        )


def test_commit_paths_push_success_via_local_bare_remote(tmp_path: Path):
    """End-to-end push against a local bare remote — proves the happy path."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)

    work = tmp_path / "work"
    work.mkdir()
    _init_repo(work)
    subprocess.run(
        ["git", "-C", str(work), "remote", "add", "origin", str(bare)],
        check=True,
    )
    # Push the seed commit first so origin/master exists; otherwise the
    # bare-remote default-branch dance breaks on push.
    subprocess.run(["git", "-C", str(work), "push", "-u", "origin", "master"], check=True)

    (work / "journey.md").write_text("v1\n")
    result = commit_paths(work, [work / "journey.md"], message="chore: ship journey", push=True)
    assert result.commit_sha is not None
    assert result.pushed is True

    # The bare remote received it.
    remote_log = subprocess.run(
        ["git", "-C", str(bare), "log", "-1", "--pretty=%s"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert remote_log.stdout.strip() == "chore: ship journey"
