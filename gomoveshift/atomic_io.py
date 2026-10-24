"""Crash-safe text writes: stage in a sibling file, then swap it in.

``Path.write_text`` truncates the target the moment it opens it, so a kill or
a full disk mid-write leaves an empty or half-written file. ``write_text_atomic``
writes a uniquely named sibling, flushes it to disk and only then ``os.replace``s
it over the target: a reader sees the old file or the new one, never a torn one,
and a failure leaves the old file intact with no stray temp file.

The temp file is created with the process umask (like ``write_text``), not the
owner-only mode ``tempfile.mkstemp`` forces, so a file other users read keeps
the permissions it had before. Stdlib only, so any layer may import it.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Union


def write_text_atomic(path: Union[str, Path], text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically; the parent directory must exist.

    A unique temp name per call keeps concurrent writers from clobbering each
    other's staging file. Any failure removes the temp file and re-raises.
    """
    target = Path(path)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "x", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
