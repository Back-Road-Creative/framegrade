"""Run-lock helpers extracted from ``drive_orchestrator``."""

import errno
import fcntl
import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional

from ..paths import gms_cache_home

# Logger name preserved so grep-trails stay valid post-extraction.
logger = logging.getLogger("gomoveshift.pipeline.drive_orchestrator")


# ─── Orchestrator-entry concurrent-run lock ──────────────────────────────
#
# Two operators (or two tmux sessions) pointing at the same project
# output dir would otherwise race on ``pipeline_state.json`` writes and
# overwrite each other's day-master encodes. The lock lives at
# ``stages_base / pipeline_state.lock`` (sibling of the state file, on
# ext4 always — never NTFS where ``fcntl.flock`` semantics are
# unreliable). Held against the open file descriptor, so a tmux detach
# or reattach is invisible to the lock — only process death releases.
#
# Stale recovery: on EWOULDBLOCK we read the holder PID from the lock
# file and probe with ``os.kill(pid, 0)``. ``ProcessLookupError`` →
# stale, reclaim. Any other state → live holder, raise
# :class:`ConcurrentRunError` with the holder's PID + start time.
#
# Escape hatch: ``--force-unlock`` removes the lock unconditionally.
# Operator-only; logs WARNING. Use only after verifying via
# ``pgrep -af 'gomoveshift drive'`` that no other run is active.
class ConcurrentRunError(RuntimeError):
    """Another orchestrator already holds the run lock for this output dir."""


def _acquire_run_lock(
    stages_base: Path,
    output_base: Path,
    *,
    force_unlock: bool = False,
    lock_name: str = "pipeline_state.lock",
) -> tuple[int, Path]:
    """Acquire an exclusive non-blocking lock on the run lock-file.

    ``lock_name`` selects the sidecar filename under ``stages_base`` so the
    photo pipeline can take an independent run lock through this same
    stale-PID-recovery machinery instead of growing a second scheme.

    Writes ``{"pid": ..., "started": ..., "output_base": ...}`` into the
    lock file on success. On ``BlockingIOError`` (EWOULDBLOCK) reads the
    existing PID and probes; if dead, reclaim once; if live, raise
    :class:`ConcurrentRunError`.

    Returns
    -------
    tuple[int, Path]
        The held file descriptor and the lock-file path. Pass both to
        :func:`_release_run_lock` when done.

    Raises
    ------
    ConcurrentRunError
        Another live orchestrator holds the lock.
    """
    stages_base.mkdir(parents=True, exist_ok=True)
    lock_path = stages_base / lock_name

    if force_unlock and lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text())
            logger.warning(
                "--force-unlock: removing existing lock (was held by PID=%s started=%s)",
                existing.get("pid"),
                existing.get("started"),
            )
        except (json.JSONDecodeError, OSError):
            logger.warning("--force-unlock: removing existing lock (unparsable)")
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return _write_lock_metadata(fd, lock_path, output_base)
        except BlockingIOError as exc:
            # EWOULDBLOCK / EAGAIN: someone holds it. Probe.
            if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                raise
            holder = _read_lock_metadata(lock_path)
            holder_pid = holder.get("pid") if isinstance(holder, dict) else None
            if isinstance(holder_pid, int) and _pid_is_alive(holder_pid):
                # Live holder. Surface a clear error.
                _raise_concurrent_run_error(holder, lock_path, output_base)
            # Stale (dead PID, missing PID, or unparsable metadata).
            logger.warning(
                "Stale run lock at %s (holder_pid=%s) — reclaiming.",
                lock_path,
                holder_pid,
            )
            # One retry. If THIS fails, surface whatever the new holder is.
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                holder = _read_lock_metadata(lock_path)
                _raise_concurrent_run_error(holder, lock_path, output_base)
            return _write_lock_metadata(fd, lock_path, output_base)
    except BaseException:
        os.close(fd)
        raise


# Lock filename for the project-dir lock (distinct from the stages-base
# ``pipeline_state.lock`` so both can coexist under different roots).
PROJECT_RUN_LOCK_NAME = "pipeline_project.lock"


def _acquire_run_locks(
    project_dir: Path,
    stages_base: Path,
    output_base: Path,
    *,
    force_unlock: bool = False,
) -> tuple[int, Path, int, Path]:
    """Acquire BOTH run locks and return ``(stage_fd, stage_path, proj_fd, proj_path)``.

    Two runs against the same ``project_dir`` collide on the shared
    ``output_base`` artifacts (``youtube_uploads.json`` sidecar, Metricool CSV)
    and double-book upload slots even when their ``GMS_INTERMEDIATE_DIR`` (hence
    ``stages_base``) differs — so the stages-base ``pipeline_state.lock`` alone
    is NOT sufficient. The **project-dir** lock is the authoritative cross-
    staging guard; the stages-base lock preserves the existing same-staging
    refusal. Acquired project-first (consistent order across processes); if the
    stages-base lock can't be taken, the project lock is rolled back so a
    failure never leaks a held fd.

    Raises :class:`ConcurrentRunError` if either lock is held by a live run.
    """
    proj_fd, proj_path = _acquire_run_lock(
        project_dir, output_base, force_unlock=force_unlock, lock_name=PROJECT_RUN_LOCK_NAME
    )
    try:
        stage_fd, stage_path = _acquire_run_lock(
            stages_base, output_base, force_unlock=force_unlock
        )
    except BaseException:
        _release_run_lock(proj_fd, proj_path)
        raise
    return stage_fd, stage_path, proj_fd, proj_path


def _write_lock_metadata(fd: int, lock_path: Path, output_base: Path) -> tuple[int, Path]:
    """Write our PID + start time to the lock file (caller already holds flock)."""
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "started": datetime.now(tz=timezone.utc).isoformat(),
            "output_base": str(output_base),
        }
    )
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, payload.encode("utf-8"))
    os.fsync(fd)
    return fd, lock_path


def _read_lock_metadata(lock_path: Path) -> dict:
    """Best-effort read of the lock-file JSON. Empty dict on any failure."""
    try:
        text = lock_path.read_text()
    except OSError:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _pid_is_alive(pid: int) -> bool:
    """Probe via ``os.kill(pid, 0)``. ``PermissionError`` → alive (different uid)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _raise_concurrent_run_error(holder: dict, lock_path: Path, output_base: Path) -> None:
    """Format a clear error message and raise ConcurrentRunError."""
    holder_pid = holder.get("pid", "<unknown>")
    started = holder.get("started", "<unknown>")
    msg = (
        "Another orchestrator already holds this output dir.\n"
        f"  output_base: {output_base}\n"
        f"  lock_file:   {lock_path}\n"
        f"  holder_pid:  {holder_pid}\n"
        f"  started:     {started}\n"
        "If you are sure no other run is active, remove the lock file:\n"
        f"  rm {lock_path}\n"
        "Or pass --force-unlock to break it (use only after verifying no other "
        "gomoveshift process is running: pgrep -af 'gomoveshift drive')."
    )
    logger.error(msg)
    raise ConcurrentRunError(msg)


def _release_run_lock(fd: Optional[int], lock_path: Optional[Path]) -> None:
    """Release the lock and unlink the lock-file. No-op if fd is None."""
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass
    if lock_path is not None:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # Best-effort cleanup; a stale lock-file on disk is harmless
            # because the next run's PID-probe path reclaims it.
            pass


# ─── Shared GPU lock (cross-process) ──────────────────────────────────────
#
# A host that runs this pipeline's nvenc encodes may also run other
# GPU-hungry work on the same card (local LLM inference, CI runners). A
# cooperating process that takes an exclusive lock on the file below before
# loading a model makes the mutual exclusion two-sided: without it a drive
# run could start mid-inference and lose several GiB of VRAM to a resident
# model.
#
# The processes interoperate through the *file*, not through an import —
# no cross-package Python dependency exists or should. The contract is small
# and is asserted by ``tests/test_gpu_lock.py``:
#
#   * ``fcntl.flock(LOCK_EX)`` on the file is the actual mutex;
#   * the JSON body is diagnostics, and carries at least ``owner`` and ``pid``;
#   * **release never unlinks.** A lock file that is deleted and re-created is
#     two different inodes, so a releasing holder can delete a path a waiting
#     holder has already opened, and both then "hold" the lock. The file is
#     permanent; releasing writes an empty holder and drops the flock.
#
# The default location is ``gpu.lock`` under the cache home; every
# cooperating process must be pointed at the same file with
# ``GMS_GPU_LOCK_PATH``.

GPU_LOCK_OWNER = "gomoveshift"
DEFAULT_GPU_LOCK_WAIT_S = 120.0


class GpuLockBusyError(RuntimeError):
    """Local inference holds the shared GPU lock."""


def gpu_lock_path() -> Optional[Path]:
    """Configured shared-GPU-lock path, or ``None`` when the lock is disabled.

    Unset, the lock is ``gpu.lock`` under the cache home, resolved on every
    call. ``GMS_GPU_LOCK_PATH=""`` disables it, for a host that does not run
    local inference at all.
    """
    raw = os.environ.get("GMS_GPU_LOCK_PATH")
    if raw is None:
        return gms_cache_home() / "gpu.lock"
    return Path(raw) if raw else None


def _gpu_lock_wait_s() -> float:
    try:
        return float(os.environ.get("GMS_GPU_LOCK_WAIT_S", DEFAULT_GPU_LOCK_WAIT_S))
    except ValueError:
        return DEFAULT_GPU_LOCK_WAIT_S


def _write_gpu_holder(fd: int, *, owner: str) -> None:
    payload = json.dumps(
        {
            "owner": owner,
            "pid": os.getpid(),
            "acquired": datetime.now(tz=timezone.utc).isoformat(),
        }
    )
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, payload.encode("utf-8"))
    os.fsync(fd)


def acquire_gpu_lock(
    *,
    wait_s: Optional[float] = None,
    path: Optional[Path] = None,
) -> tuple[Optional[int], Optional[Path]]:
    """Take the shared GPU lock, waiting out a live holder.

    Waits because the other holder is an inference turn measured in tens of
    seconds, and failing a multi-hour encode over that would be absurd; fails
    afterwards rather than proceeding because the failure it prevents — an
    nvenc OOM partway through a day master — is worse than a clear refusal at
    the start. Returns ``(None, None)`` when the lock is disabled.

    Raises
    ------
    GpuLockBusyError
        A live holder still had it when ``wait_s`` elapsed.
    """
    lock_path = path if path is not None else gpu_lock_path()
    if lock_path is None:
        return None, None
    if wait_s is None:
        wait_s = _gpu_lock_wait_s()

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + wait_s
    announced = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise
                holder = _read_lock_metadata(lock_path)
                if not announced:
                    logger.info(
                        "Waiting up to %.0fs for the shared GPU lock at %s (held by %s pid=%s).",
                        wait_s,
                        lock_path,
                        holder.get("owner", "?"),
                        holder.get("pid", "?"),
                    )
                    announced = True
                if time.monotonic() >= deadline:
                    raise GpuLockBusyError(
                        f"Shared GPU lock at {lock_path} is held by "
                        f"{holder.get('owner', 'another process')} "
                        f"(pid {holder.get('pid', '?')}) after waiting "
                        f"{wait_s:.0f}s. Local inference is using the card; "
                        "wait for the turn to finish, or stop the inference "
                        "service. Set GMS_GPU_LOCK_WAIT_S to wait longer."
                    ) from None
                time.sleep(0.5)
                continue
            _write_gpu_holder(fd, owner=GPU_LOCK_OWNER)
            return fd, lock_path
    except BaseException:
        os.close(fd)
        raise


def release_gpu_lock(fd: Optional[int], lock_path: Optional[Path]) -> None:
    """Drop the shared GPU lock. Deliberately does **not** unlink — see above."""
    if fd is None:
        return
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, b"{}")
    except OSError:
        pass
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


# ─── Worktree-code-tree lease guard ───────────────────────────────────────
#
# A live drive running from a disposable git worktree checkout has nothing
# stopping an unrelated cleanup pass from deleting that tree mid-run. A
# cleanup tool that checks for an active ``wt-<branch>`` lease before it
# removes a worktree can be honoured only if the pipeline acquires that lease:
# acquire it before Phase A, renew it at phase boundaries, release it on
# exit. A main-tree checkout (or any non-disposable code_dir) is a no-op —
# no lease is taken, nothing shells out.
#
# The lease helper is an external executable that is not part of this
# package, so it has no default: ``GMS_LEASE_BIN`` names it, and while that
# variable is unset the guard is off (it logs that it is not configured and
# the run proceeds). The helper is invoked as ``<bin> acquire <name> <ttl>
# <note>``, ``<bin> renew <name> <ttl>`` and ``<bin> release <name>``; a
# successful acquire prints ``fence <N>`` and the token is passed back on
# renew/release as ``LEASE_FENCE_TOKEN``.
DEFAULT_WORKTREE_LEASE_TTL_S = 1800
_LEASE_BIN_ENV = "GMS_LEASE_BIN"
# Extra directory names (separated by ``os.pathsep``) that mark a code tree as
# disposable, on top of ``.worktrees`` and any linked git worktree.
_DISPOSABLE_DIRS_ENV = "GMS_DISPOSABLE_WORKTREE_DIRS"
_DEFAULT_DISPOSABLE_DIRS = (".worktrees",)
_FENCE_RE = re.compile(r"fence (\d+)")


class WorktreeLeaseError(RuntimeError):
    """A live drive cannot (continue to) run from its code tree's worktree.

    Raised when the code tree is a disposable worktree checkout and the
    ``wt-<branch>`` lease that a worktree-cleanup tool checks is already
    held by someone else — starting anyway would let a concurrent cleanup
    pass legally delete the tree out from under this run.
    """


class WorktreeLeaseHandle(NamedTuple):
    name: str
    code_dir: Path
    fence_token: Optional[str]


def _code_dir() -> Path:
    """Root of the checkout this module is running from.

    ``orchestrator_lock.py`` lives at
    ``<code_dir>/gomoveshift/pipeline/``, so two ``.parent`` hops up from
    this file is the checkout root — the same root whether that's the main
    tree or a disposable worktree.
    """
    return Path(__file__).resolve().parents[2]


def _is_disposable_worktree(code_dir: Path) -> bool:
    """True when ``code_dir`` is a tree a cleanup pass may remove.

    That is a linked git worktree (its ``.git`` is a file pointing at the
    main repository), or a tree that sits under one of the directory names in
    ``.worktrees`` plus ``GMS_DISPOSABLE_WORKTREE_DIRS``.
    """
    names = set(_DEFAULT_DISPOSABLE_DIRS)
    names.update(
        n.strip() for n in os.environ.get(_DISPOSABLE_DIRS_ENV, "").split(os.pathsep) if n.strip()
    )
    if any(part in names for part in code_dir.parts):
        return True
    return _is_linked_worktree(code_dir)


def _is_linked_worktree(code_dir: Path) -> bool:
    """True when ``code_dir/.git`` is the pointer file of a linked worktree.

    A ``.git`` *file* also appears in a git submodule checkout and in a clone
    made with a separate git dir; both are main checkouts, not linked
    worktrees. The pointer target tells them apart: a linked worktree's git
    dir sits under ``<common>/worktrees/<name>`` and holds a ``commondir``
    file. A pointer whose target cannot be inspected is treated as a linked
    worktree (the conservative reading: a lease is only ever taken when
    ``GMS_LEASE_BIN`` is configured).
    """
    git_path = code_dir / ".git"
    try:
        if not git_path.is_file():
            return False
        first_line = git_path.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    except OSError:
        return False
    if not first_line or not first_line[0].startswith("gitdir:"):
        return False
    target = Path(first_line[0][len("gitdir:") :].strip())
    if not target.is_absolute():
        target = code_dir / target
    try:
        if (target / "commondir").is_file():
            return True
        if target.is_dir():
            return target.parent.name == "worktrees"
    except OSError:
        return True
    return True


def _git_branch(code_dir: Path) -> Optional[str]:
    """Current branch of ``code_dir``, or None (detached HEAD / not a repo)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(code_dir), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    if not branch or branch == "HEAD":
        return None  # detached HEAD — a cleanup pass skips these
    return branch


def _lease_bin() -> Optional[Path]:
    """The configured lease helper, or ``None`` when ``GMS_LEASE_BIN`` is unset."""
    override = os.environ.get(_LEASE_BIN_ENV, "").strip()
    return Path(override) if override else None


def _acquire_worktree_lease(
    code_dir: Optional[Path] = None,
    *,
    ttl: int = DEFAULT_WORKTREE_LEASE_TTL_S,
) -> Optional[WorktreeLeaseHandle]:
    """Take the ``wt-<branch>`` lease before any phase work, if applicable.

    No-op (returns None, never shells out) for a main-tree checkout, a
    disposable tree in detached HEAD (cleanup already skips those), or when
    ``GMS_LEASE_BIN`` is not configured.

    Raises
    ------
    WorktreeLeaseError
        The lease is already held by another session — refuse to start
        rather than run unprotected against a legal mid-run delete.
    """
    code_dir = code_dir if code_dir is not None else _code_dir()
    if not _is_disposable_worktree(code_dir):
        return None
    branch = _git_branch(code_dir)
    if branch is None:
        return None
    lease_bin = _lease_bin()
    if lease_bin is None:
        logger.info(
            "worktree lease guard not configured: %s is not set, so %s is not leased "
            "(set it to the lease helper executable to protect a disposable worktree).",
            _LEASE_BIN_ENV,
            code_dir,
        )
        return None
    name = f"wt-{branch}"
    result = subprocess.run(
        [str(lease_bin), "acquire", name, str(ttl), "gomoveshift drive run"],
        cwd=str(code_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise WorktreeLeaseError(
            f"Refusing to start: code tree {code_dir} is a disposable worktree "
            f"and its lease {name!r} is held elsewhere — a concurrent cleanup "
            "pass could legally delete this tree mid-run. "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    fence_match = _FENCE_RE.search(result.stdout)
    return WorktreeLeaseHandle(
        name=name, code_dir=code_dir, fence_token=fence_match.group(1) if fence_match else None
    )


def _renew_worktree_lease(
    handle: Optional[WorktreeLeaseHandle], *, ttl: int = DEFAULT_WORKTREE_LEASE_TTL_S
) -> None:
    """Renew the lease at a phase boundary. Best-effort: a renew failure is
    logged, never raised — a multi-hour drive must not abort mid-run over a
    renewal hiccup; the acquire-time refusal is the real gate."""
    if handle is None:
        return
    lease_bin = _lease_bin()
    if lease_bin is None:
        return
    env = dict(os.environ)
    if handle.fence_token is not None:
        env["LEASE_FENCE_TOKEN"] = handle.fence_token
    result = subprocess.run(
        [str(lease_bin), "renew", handle.name, str(ttl)],
        cwd=str(handle.code_dir),
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        logger.warning(
            "worktree lease renew failed for %s (continuing run): %s",
            handle.name,
            result.stderr.strip() or result.stdout.strip(),
        )


def _release_worktree_lease(handle: Optional[WorktreeLeaseHandle]) -> None:
    """Release the lease on exit. Best-effort — never raises."""
    if handle is None:
        return
    lease_bin = _lease_bin()
    if lease_bin is None:
        return
    env = dict(os.environ)
    if handle.fence_token is not None:
        env["LEASE_FENCE_TOKEN"] = handle.fence_token
    try:
        subprocess.run(
            [str(lease_bin), "release", handle.name],
            cwd=str(handle.code_dir),
            capture_output=True,
            text=True,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        pass
