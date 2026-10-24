"""Atomic JSON write helper.

Eliminates the bug class where a crash mid-write leaves a corrupt JSON
sidecar. The pattern (tempfile in same dir + ``os.replace``) already
existed inline in several pipeline modules — most notably
``youtube_sidecar.save_sidecar`` and ``_drive_state.save_drive_state``.
This module centralises the pattern so every JSON write site benefits
from it without re-implementing the dance.

Failure mode prevented: a SIGKILL / power loss between ``open(path,
"w")`` and ``json.dump`` finishing would previously leave ``path`` as a
truncated, unparseable file. With this helper, the existing file at
``path`` is replaced atomically only after the temp file is fully
written and closed; a mid-write crash leaves the old file intact and
the temp file orphaned (best-effort cleanup in the ``except`` branch).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import yaml


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> None:
    """Write JSON to ``path`` atomically.

    A crash mid-write leaves the existing file at ``path`` untouched.
    Parent directories are created if missing.

    Args:
        path: Destination path. Parent dirs are auto-created.
        payload: Any ``json.dump``-serializable object.
        indent: Indent passed through to ``json.dump`` (default 2).
        sort_keys: Pass-through to ``json.dump`` for deterministic
            operator-visible files (default ``False``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=indent, sort_keys=sort_keys)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError, OSError):
            os.unlink(tmp)
        raise


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically — same contract as ``atomic_write_json``.

    The one shared text writer: site journey markdown, ``videos.yaml``, GPX
    files, the video routing table and the JSON state files (health incident
    rows, the pending-publish sidecar, the trail-opening marker). This helper
    decides only *how* the string reaches disk, never what it contains, so
    comment-preserving round-trips are unaffected. A torn write on published
    site content is worse than a skipped one — the merge is replayable, a
    truncated page is a corrupt artifact that blocks the deploy.

    The text is staged in a uniquely named sibling file, flushed and fsynced,
    then ``os.replace``d over ``path``: a reader sees the old file or the new
    one, never a torn one, and any failure removes the staging file and leaves
    the old file intact. The unique name keeps concurrent writers from
    clobbering each other's staging file. The staging file takes the process
    umask (like ``Path.write_text``), not the owner-only mode ``mkstemp``
    forces, so a file other users read keeps the permissions it had. The
    guarantee is atomicity of the swap; the parent directory is not fsynced,
    so the rename itself can still be lost to a power cut. Encodes as utf-8 by
    default so a non-utf-8 locale can't mangle place names. Parent
    directories are created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)  # & ~umask
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError, OSError):
            os.unlink(tmp)
        raise


def atomic_write_yaml(path: Path, payload: Any, **dump_kwargs: Any) -> None:
    """Write YAML to ``path`` atomically — same contract as ``atomic_write_json``.

    Used for operator-reviewed sidecars (e.g. photo caption review files),
    where a truncated file costs hand-review work that cannot be replayed.
    ``dump_kwargs`` pass through to ``yaml.dump``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(payload, f, **dump_kwargs)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError, OSError):
            os.unlink(tmp)
        raise
