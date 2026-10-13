"""Privacy gate for route-story stills: blur privacy regions and strip EXIF
into a working copy BEFORE any still is composited or exported into the film.
The film composites/exports the WORKING COPIES only -- the original photo
path is never read past this gate.

Reuses the photos privacy contract verbatim (never a second copy of it):
blur detection/compositing from ``photos.face_filter``, the GPS-survival
check from ``photos.resizer``. Fail-closed: a default run (blur on) whose
detector cannot run raises before any frame is rendered -- ``--skip-blur``
is the named diagnostic escape, never a silent unblurred film.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2

from ..day_master_encoder import REQUIRED_PRIVACY_CLASSES
from ..photos.face_filter import (
    FaceBlurError,
    attestable_privacy_classes,
    blur_enabled,
    blur_faces_for_publish,
    should_withhold_photo,
)
from ..photos.resizer import _assert_no_gps

logger = logging.getLogger(__name__)

#: Subdirectory of ``work_dir`` holding the blurred/EXIF-stripped copies the
#: film actually composites -- never the operator's source photos.
WORKING_SUBDIR = "privacy_gated_stills"


@dataclass(frozen=True)
class PrivacyResult:
    """``photos``: the input sequence with each ``.source`` swapped to its
    gated working copy (withheld photos dropped). ``withheld``: names of
    photos dropped as unreadable/unblurrable. ``attestation``: the
    ``receipt["privacy"]`` / ``publish.json`` block, written in ``dispatch``'s
    own vocabulary so ``privacy_hold_reason`` needs no translation."""

    photos: Tuple[Any, ...]
    withheld: Tuple[str, ...] = ()
    attestation: Dict[str, Any] = field(default_factory=dict)


def _strip_gps(dest: Path) -> None:
    """Re-save ``dest`` with no EXIF/XMP so no GPS tag can survive, then
    assert it -- the still analogue of the photos resize gate."""
    from PIL import Image

    with Image.open(dest) as img:
        img.load()
        clean = img.convert(img.mode)
    clean.save(dest, exif=b"")
    _assert_no_gps(dest)


def _gate_one(photo: Any, *, blur: Optional[bool], gated_dir: Path) -> Optional[Path]:
    """Return the gated working-copy path, or ``None`` when withheld."""
    src = Path(photo.source)
    if should_withhold_photo(str(src), blur=blur):
        return None

    img = cv2.imread(str(src))
    if img is None:
        raise FaceBlurError(f"unreadable photo, cannot privacy-gate: {src}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        out_rgb, _ = blur_faces_for_publish(rgb, str(src), blur=blur)
    except FaceBlurError as e:
        raise FaceBlurError(
            f"{e} -- privacy detection is required before compositing "
            f"{src.name}; --skip-blur is the diagnostic escape (ships with "
            "unproven privacy, never a silent default)"
        ) from e

    dest = gated_dir / src.name
    cv2.imwrite(str(dest), cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
    _strip_gps(dest)
    return dest


def apply_privacy_gate(
    photos: Sequence[Any], *, blur: Optional[bool], work_dir: Path
) -> PrivacyResult:
    """Blur + EXIF-strip every still into ``work_dir`` before compositing.

    Every downstream composite/export must read the returned ``photos[i]``
    (the working copy) -- never the caller's original list."""
    gated_dir = Path(work_dir) / WORKING_SUBDIR
    gated_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    withheld = []
    for photo in photos:
        dest = _gate_one(photo, blur=blur, gated_dir=gated_dir)
        if dest is None:
            withheld.append(Path(photo.source).name)
            continue
        kept.append(replace(photo, source=dest))

    effective_blur = blur_enabled() if blur is None else blur
    classes = attestable_privacy_classes(blur)
    missing = set(REQUIRED_PRIVACY_CLASSES) - set(classes)
    if missing:
        raise ValueError(f"privacy attestation missing required classes: {sorted(missing)}")
    attestation = {"classes": classes, "blur": effective_blur, "detector": sorted(classes)}

    if withheld:
        logger.info("privacy gate withheld %d photo(s): %s", len(withheld), ", ".join(withheld))

    return PrivacyResult(photos=tuple(kept), withheld=tuple(withheld), attestation=attestation)
