"""The ``day_<YYYY-MM-DD>`` day-directory layout, defined once.

A day-first run writes everything for one recording day under
``<stages>/day_<YYYY-MM-DD>/``: ``regular_lfs/lf_NN.mp4``,
``shorts/short_NN.mp4`` and the ``day_<DATE>_*`` artifacts beside them.
Every module that names, recognises or parses that directory goes through
this one module, so the pattern cannot drift between copies.

Pure string and path logic; no pipeline imports, so any layer may use it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

DAY_DIR_PREFIX = "day_"
LF_SUBDIR = "regular_lfs"

#: Matches exactly a day directory name; group 1 is the ISO date.
DAY_DIR_RE = re.compile(r"^day_(\d{4}-\d{2}-\d{2})$")


def day_dir_name(date: str) -> str:
    """``"2025-09-15"`` -> ``"day_2025-09-15"``."""
    return f"{DAY_DIR_PREFIX}{date}"


def date_from_day_dir_name(name: str) -> Optional[str]:
    """The ISO date of a day directory name, or ``None`` when it is not one."""
    m = DAY_DIR_RE.match(name)
    return m.group(1) if m else None


def day_date_from_lf_path(lf_path) -> Optional[str]:
    """Extract ``YYYY-MM-DD`` from the ``day_<DATE>`` dir anchoring an LF path.

    Canonical layout: ``<stages>/day_<YYYY-MM-DD>/regular_lfs/lf_NN.mp4``.
    The date directory is the parent of the ``regular_lfs`` dir (the LF
    file's grandparent). Anchoring to that exact component, rather than
    searching the whole path string, means a relocated path whose ANCESTORS
    happen to contain a different ``day_*`` token cannot skew the date.

    Falls back to scanning path components for a single ``day_<DATE>`` dir
    when ``regular_lfs`` is absent (e.g. a flattened test fixture); returns
    ``None`` when no anchored match exists.
    """
    parts = Path(str(lf_path)).parts
    for i, comp in enumerate(parts):
        if comp == LF_SUBDIR and i > 0:
            date = date_from_day_dir_name(parts[i - 1])
            if date is not None:
                return date
    matches = [d for comp in parts if (d := date_from_day_dir_name(comp)) is not None]
    return matches[0] if len(matches) == 1 else None
