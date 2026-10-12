"""Graded pixels on disk, keyed on what produced them.

The Route Story grade (haze, exposure, vibrance) is slow -- 2.7 s per
megapixel -- and a pure function of the image's bytes and the code that grades
it. Opening rasters (``layers._graded_raster_image``) and photo stills
(``hud._load_still``) both keep their graded result here, so an image is graded
once per machine rather than once (or twice) per render.

Opt out with ``GMS_ROUTE_STORY_GRADE_CACHE=0``. The directory follows the
render's intermediate dir; the size cap (``GMS_ROUTE_STORY_GRADE_CACHE_MAX_GB``,
default 8) trims the oldest entries. A cache that cannot be read or written is
a slower render, never a failed one.
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path
from types import ModuleType

import numpy as np
from PIL import Image

from ...paths import gms_cache_home

CACHE_ENV = "GMS_ROUTE_STORY_GRADE_CACHE"
MAX_GB_ENV = "GMS_ROUTE_STORY_GRADE_CACHE_MAX_GB"
DEFAULT_MAX_GB = 8.0


def cache_dir() -> Path:
    # Resolved per call, never at import: a module-level default freezes
    # whatever HOME was when the module loaded.
    intermediate = os.environ.get("GMS_INTERMEDIATE_DIR", "")
    if intermediate:
        return Path(intermediate) / "route-story-grade-cache"
    return gms_cache_home() / "route-story" / "graded-rasters"


@functools.lru_cache(maxsize=None)
def code_identity(*modules: ModuleType) -> str:
    """Changes whenever the pixels could: any edit to one of ``modules``' files.

    Whole files, not single functions: a key that misses a helper the grade
    calls serves stale pixels, while one that over-invalidates costs a regrade.
    """
    digest = hashlib.sha256()
    for module in modules:
        digest.update(Path(module.__file__ or "").read_bytes())
    return digest.hexdigest()


def cache_path(source: Path, *identity: str) -> Path | None:
    """Where ``source``'s graded pixels live, keyed on its CONTENT plus ``identity``.

    Content, not path or mtime: the bundle's opening rasters are re-written
    into a fresh temporary directory by every render, and one photo may sit on
    two cards. Hashing a 30 MB file costs ~0.05 s against a minute of grading.
    ``None`` when the cache is off or the source cannot be read.
    """
    if os.environ.get(CACHE_ENV, "1") == "0":
        return None
    try:
        digest = hashlib.sha256(Path(source).read_bytes())
    except OSError:
        return None
    for part in identity:
        digest.update(part.encode())
    return cache_dir() / f"{digest.hexdigest()}.npy"


def load(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    try:
        return Image.fromarray(np.load(path, allow_pickle=False), "RGB")
    except (OSError, ValueError):
        return None  # absent or unreadable: the caller grades and stores it


def store(path: Path | None, graded: Image.Image) -> None:
    """Write atomically, then trim the oldest entries past the size cap."""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.stem}.{os.getpid()}.partial.npy")
        np.save(partial, np.asarray(graded.convert("RGB"), dtype=np.uint8), allow_pickle=False)
        os.replace(partial, path)
        trim(path.parent, keep=path)
    except OSError:
        return


def trim(directory: Path, *, keep: Path) -> None:
    try:
        cap_gb = float(os.environ.get(MAX_GB_ENV, "") or DEFAULT_MAX_GB)
    except ValueError:
        cap_gb = DEFAULT_MAX_GB
    entries = sorted(
        (entry for entry in directory.glob("*.npy") if not entry.name.endswith(".partial.npy")),
        key=lambda entry: entry.stat().st_mtime,
    )
    total = sum(entry.stat().st_size for entry in entries)
    for entry in entries:
        if total <= cap_gb * 1024**3:
            break
        if entry == keep:
            continue
        total -= entry.stat().st_size
        entry.unlink(missing_ok=True)
