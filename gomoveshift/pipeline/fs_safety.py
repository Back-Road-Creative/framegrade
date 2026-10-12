"""Filesystem operations hardened against drvfs/9p mount quirks.

WSL's drvfs (/mnt/<letter>) and 9p mounts routinely refuse the mode/utime
metadata update that ``shutil.copy2`` performs after copying bytes, raising
PermissionError/OSError even though the file content copied fine (seen in
cron-driven metrics runs, and handled at several other copy sites such as
tracking/store.py, redrive_ost.py and drive_phases/phase_d.py).

Every pipeline call site that needs a stat-preserving copy should go through
``safe_copy`` instead of ``shutil.copy2`` so this failure class can't
resurface at a new call site — a structural test
(``tests/test_fs_safety_gate.py``) bans the raw stdlib call outside this
module.
"""

import shutil
from pathlib import Path
from typing import Union

_PathLike = Union[str, Path]


def safe_copy(src: _PathLike, dst: _PathLike) -> None:
    """Copy bytes from ``src`` to ``dst``, best-effort preserving stat.

    The byte copy is not tolerant of failure — a genuine I/O error (missing
    source, disk full, real permission denial on the destination) still
    raises. Only the metadata-preservation step (mode/utime) is guarded,
    matching the pattern already proven at drive_phases/phase_d.py: a drvfs
    mount refusing that step must not fail a copy whose bytes landed fine.
    """
    shutil.copyfile(src, dst)
    try:
        shutil.copystat(src, dst)
    except OSError:
        pass
