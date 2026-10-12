"""Tests for the worktree-code-tree lease guard.

A live drive running from a disposable worktree checkout (as recognised by
``orchestrator_lock._is_disposable_worktree``) must hold the lease name
``wt-<branch>`` that a worktree-cleanup tool checks before removing a worktree.
The lease is taken through an external helper executable named by
``GMS_LEASE_BIN``; while that variable is unset the guard is off. Extra
disposable directory names come from ``GMS_DISPOSABLE_WORKTREE_DIRS``. Without
the lease, a long-running drive's code tree could be legally deleted mid-run by
an unrelated cleanup pass.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import drive_orchestrator
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.orchestrator_lock import (
    WorktreeLeaseError,
    _acquire_worktree_lease,
    _is_disposable_worktree,
    _release_worktree_lease,
    _renew_worktree_lease,
)

_LEASE_BIN = "/opt/lease-helper/lease"


@pytest.fixture(autouse=True)
def _clean_lease_env(monkeypatch):
    """Every test starts with the guard unconfigured and no extra disposable names."""
    monkeypatch.delenv("GMS_LEASE_BIN", raising=False)
    monkeypatch.delenv("GMS_DISPOSABLE_WORKTREE_DIRS", raising=False)


@pytest.fixture
def lease_configured(monkeypatch):
    monkeypatch.setenv("GMS_LEASE_BIN", _LEASE_BIN)
    return _LEASE_BIN


# ── Disposable-tree detection ───────────────────────────────────────────


def test_main_tree_is_not_disposable(tmp_path):
    assert _is_disposable_worktree(tmp_path / "repo") is False


def test_dot_worktrees_is_disposable():
    assert _is_disposable_worktree(Path("/srv/repo/.worktrees/my-branch")) is True


def test_configured_directory_name_is_disposable(monkeypatch):
    """A name in GMS_DISPOSABLE_WORKTREE_DIRS marks any tree under it disposable."""
    tree = Path("/srv/repo/scratch-trees/my-branch")
    assert _is_disposable_worktree(tree) is False
    monkeypatch.setenv("GMS_DISPOSABLE_WORKTREE_DIRS", "scratch-trees")
    assert _is_disposable_worktree(tree) is True


def test_several_configured_names_are_separated_by_pathsep(monkeypatch):
    monkeypatch.setenv("GMS_DISPOSABLE_WORKTREE_DIRS", os.pathsep.join(["one", "two"]))
    assert _is_disposable_worktree(Path("/srv/repo/one/a")) is True
    assert _is_disposable_worktree(Path("/srv/repo/two/a")) is True
    assert _is_disposable_worktree(Path("/srv/repo/three/a")) is False


def test_linked_worktree_git_file_is_disposable(tmp_path):
    """A linked git worktree has a ``.git`` file (not a directory)."""
    tree = tmp_path / "linked"
    tree.mkdir()
    (tree / ".git").write_text("gitdir: /elsewhere\n")
    assert _is_disposable_worktree(tree) is True


def test_git_directory_checkout_is_not_disposable(tmp_path):
    tree = tmp_path / "main"
    (tree / ".git").mkdir(parents=True)
    assert _is_disposable_worktree(tree) is False


def test_submodule_pointer_is_not_disposable(tmp_path):
    """A ``.git`` file pointing under ``<main>/.git/modules`` is a submodule checkout."""
    target = tmp_path / "main" / ".git" / "modules" / "sub"
    target.mkdir(parents=True)
    tree = tmp_path / "main" / "sub"
    tree.mkdir()
    (tree / ".git").write_text(f"gitdir: {target}\n")
    assert _is_disposable_worktree(tree) is False


def test_separate_git_dir_pointer_is_not_disposable(tmp_path):
    """A ``.git`` file pointing at a plain separate git dir is a main checkout."""
    target = tmp_path / "gitdirs" / "repo.git"
    target.mkdir(parents=True)
    tree = tmp_path / "checkout"
    tree.mkdir()
    (tree / ".git").write_text(f"gitdir: {target}\n")
    assert _is_disposable_worktree(tree) is False


def test_real_linked_worktree_pointer_is_disposable(tmp_path):
    """A ``.git`` file pointing at ``<main>/.git/worktrees/<name>`` with a commondir file."""
    target = tmp_path / "main" / ".git" / "worktrees" / "wt"
    target.mkdir(parents=True)
    (target / "commondir").write_text("../..\n")
    tree = tmp_path / "wt"
    tree.mkdir()
    (tree / ".git").write_text(f"gitdir: {target}\n")
    assert _is_disposable_worktree(tree) is True


def test_submodule_pointer_never_invokes_lease_helper(tmp_path, monkeypatch, lease_configured):
    target = tmp_path / "main" / ".git" / "modules" / "sub"
    target.mkdir(parents=True)
    tree = tmp_path / "main" / "sub"
    tree.mkdir()
    (tree / ".git").write_text(f"gitdir: {target}\n")
    with patch("gomoveshift.pipeline.orchestrator_lock.subprocess.run") as mock_run:
        handle = _acquire_worktree_lease(code_dir=tree)
    mock_run.assert_not_called()
    assert handle is None


# ── Lease acquisition ────────────────────────────────────────────────────


def test_acquire_is_noop_for_main_tree(tmp_path, lease_configured):
    """A non-disposable code_dir never shells out to the lease helper at all."""
    with patch("gomoveshift.pipeline.orchestrator_lock.subprocess.run") as mock_run:
        handle = _acquire_worktree_lease(code_dir=tmp_path)
    mock_run.assert_not_called()
    assert handle is None


def test_acquire_without_lease_bin_returns_none_and_never_shells_out(tmp_path, monkeypatch):
    """GMS_LEASE_BIN unset: the guard reports not-configured and the run proceeds."""
    worktree = tmp_path / ".worktrees" / "my-branch"
    worktree.mkdir(parents=True)
    monkeypatch.setattr(
        "gomoveshift.pipeline.orchestrator_lock._git_branch",
        lambda code_dir: "my-branch",
    )
    with patch("gomoveshift.pipeline.orchestrator_lock.subprocess.run") as mock_run:
        handle = _acquire_worktree_lease(code_dir=worktree)
    mock_run.assert_not_called()
    assert handle is None


def test_renew_without_lease_bin_never_shells_out():
    """A handle left over from a configured run is a no-op once the helper is unset."""
    from gomoveshift.pipeline.orchestrator_lock import WorktreeLeaseHandle

    handle = WorktreeLeaseHandle(name="wt-my-branch", code_dir=Path("/srv/repo"), fence_token="3")  # noqa: S106
    with patch("gomoveshift.pipeline.orchestrator_lock.subprocess.run") as mock_run:
        _renew_worktree_lease(handle)
    mock_run.assert_not_called()


def test_acquire_raises_when_lease_held_elsewhere(tmp_path, monkeypatch, lease_configured):
    """A disposable code_dir whose lease is already held elsewhere refuses to
    start — WorktreeLeaseError, raised BEFORE any phase work."""
    worktree = tmp_path / ".worktrees" / "my-branch"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text("gitdir: /elsewhere\n")
    monkeypatch.setattr(
        "gomoveshift.pipeline.orchestrator_lock._git_branch",
        lambda code_dir: "my-branch",
    )
    held = MagicMock(returncode=3, stdout="", stderr="lease HELD by someone for 10s")
    with patch(
        "gomoveshift.pipeline.orchestrator_lock.subprocess.run", return_value=held
    ) as mock_run:
        with pytest.raises(WorktreeLeaseError) as excinfo:
            _acquire_worktree_lease(code_dir=worktree)
    assert "wt-my-branch" in str(excinfo.value)
    args = mock_run.call_args[0][0]
    assert args[0] == lease_configured
    assert args[1:3] == ["acquire", "wt-my-branch"]


def test_acquire_succeeds_and_parses_fence(tmp_path, monkeypatch, lease_configured):
    worktree = tmp_path / ".worktrees" / "my-branch"
    worktree.mkdir(parents=True)
    monkeypatch.setattr(
        "gomoveshift.pipeline.orchestrator_lock._git_branch",
        lambda code_dir: "my-branch",
    )
    ok = MagicMock(
        returncode=0,
        stdout="lease ACQUIRED: wt-my-branch (ttl 1800s, fence 4) — gomoveshift drive run\n",
        stderr="",
    )
    with patch("gomoveshift.pipeline.orchestrator_lock.subprocess.run", return_value=ok):
        handle = _acquire_worktree_lease(code_dir=worktree)
    assert handle is not None
    assert handle.name == "wt-my-branch"
    assert handle.fence_token == "4"  # noqa: S105 — a fence counter, not a secret


def test_renew_and_release_are_noop_for_none_handle():
    """No handle (main tree, or lease disabled) — renew/release never raise."""
    _renew_worktree_lease(None)
    _release_worktree_lease(None)


def test_renew_passes_fence_token(tmp_path, monkeypatch, lease_configured):
    worktree = tmp_path / ".worktrees" / "my-branch"
    worktree.mkdir(parents=True)
    monkeypatch.setattr(
        "gomoveshift.pipeline.orchestrator_lock._git_branch",
        lambda code_dir: "my-branch",
    )
    ok = MagicMock(
        returncode=0,
        stdout="lease ACQUIRED: wt-my-branch (ttl 1800s, fence 7) — gomoveshift drive run\n",
        stderr="",
    )
    with patch("gomoveshift.pipeline.orchestrator_lock.subprocess.run", return_value=ok):
        handle = _acquire_worktree_lease(code_dir=worktree)
    with patch("gomoveshift.pipeline.orchestrator_lock.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        _renew_worktree_lease(handle)
    args, kwargs = mock_run.call_args
    assert args[0][0] == lease_configured
    assert args[0][1:3] == ["renew", "wt-my-branch"]
    assert kwargs["env"]["LEASE_FENCE_TOKEN"] == "7"  # noqa: S105 — a fence counter, not a secret


# ── Orchestrator wiring: refuses to start before Phase A ────────────────


def test_run_refuses_before_phase_a_when_lease_held(tmp_path):
    """Orchestrator.run() acquires the worktree lease before any phase work;
    a WorktreeLeaseError aborts the run before phase_a is ever invoked."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    output_base = tmp_path / "out"
    output_base.mkdir()

    phase_a_called = False

    def _record_phase_a(self, state):
        nonlocal phase_a_called
        phase_a_called = True

    with (
        patch.object(
            drive_orchestrator,
            "_acquire_worktree_lease",
            side_effect=WorktreeLeaseError(
                "code tree is a disposable worktree and its lease wt-my-branch is held elsewhere"
            ),
        ),
        patch.object(
            DrivePipelineOrchestrator,
            "_phase_a_ingest_and_analyze",
            autospec=True,
            side_effect=_record_phase_a,
        ),
    ):
        orch = DrivePipelineOrchestrator()
        with pytest.raises(WorktreeLeaseError):
            orch.run(
                source_dir=source_dir,
                output_base=output_base,
                fresh=True,
                force=True,
            )

    assert phase_a_called is False


def test_run_releases_lease_when_run_lock_acquisition_fails(tmp_path):
    """The lease is taken before ``_acquire_run_locks`` — a
    ``ConcurrentRunError`` from that call must not leak the lease, or
    relaunching from this same worktree is refused for the full TTL."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    output_base = tmp_path / "out"
    output_base.mkdir()
    sentinel_handle = object()
    released = []

    with (
        patch.object(drive_orchestrator, "_acquire_worktree_lease", return_value=sentinel_handle),
        patch.object(
            drive_orchestrator,
            "_release_worktree_lease",
            side_effect=released.append,
        ),
        patch.object(
            drive_orchestrator,
            "_acquire_run_locks",
            side_effect=drive_orchestrator.ConcurrentRunError("held elsewhere"),
        ),
    ):
        orch = DrivePipelineOrchestrator()
        with pytest.raises(drive_orchestrator.ConcurrentRunError):
            orch.run(
                source_dir=source_dir,
                output_base=output_base,
                fresh=True,
                force=True,
            )

    assert released == [sentinel_handle], "lease was not released on run-lock failure"


def test_run_releases_lease_when_gpu_lock_acquisition_fails(tmp_path):
    """Same guarantee on the GPU-lock failure path — the existing rollback
    releases both run locks but must also release the worktree lease."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    output_base = tmp_path / "out"
    output_base.mkdir()
    sentinel_handle = object()
    released = []

    with (
        patch.object(drive_orchestrator, "_acquire_worktree_lease", return_value=sentinel_handle),
        patch.object(
            drive_orchestrator,
            "_release_worktree_lease",
            side_effect=released.append,
        ),
        patch.object(
            drive_orchestrator,
            "acquire_gpu_lock",
            side_effect=RuntimeError("gpu lock busy"),
        ),
    ):
        orch = DrivePipelineOrchestrator()
        with pytest.raises(RuntimeError, match="gpu lock busy"):
            orch.run(
                source_dir=source_dir,
                output_base=output_base,
                fresh=True,
                force=True,
            )

    assert released == [sentinel_handle], "lease was not released on gpu-lock failure"
