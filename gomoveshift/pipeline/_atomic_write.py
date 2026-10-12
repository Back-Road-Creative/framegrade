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


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically — same contract as ``atomic_write_json``.

    For files whose bytes are produced by a caller-owned serializer (site
    journey markdown, ``videos.yaml``): this helper decides only *how* the
    string reaches disk, never what it contains, so comment-preserving
    round-trips are unaffected. A torn write on published site content is
    worse than a skipped one — the merge is replayable, a truncated page is
    a corrupt artifact that blocks the deploy. Encodes as utf-8 so a
    non-utf-8 locale can't mangle place names.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
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
