"""Panorama stitching: an ordered set of overlapping photos -> one wide
frame, via OpenCV's ``Stitcher``.

The stitch is upstream-gated, not late-checked: a degenerate result (blown
out seams, a homography that collapsed the frame) never reaches disk. QA runs
on the stitched frame BEFORE any write, so a rejected stitch leaves ``out_path``
untouched -- callers never have to clean up a partial file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#: Below this fraction of non-black pixels, the stitch is treated as a failed
#: seam rather than a usable panorama -- OpenCV's Stitcher can return
#: ``Stitcher_OK`` for a technically-valid homography that still leaves most
#: of the canvas black (barely-overlapping frames, a near-degenerate warp).
MIN_VALID_AREA_FRACTION = 0.5

_STATUS_NAMES = {
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "need more overlapping images",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: (
        "homography estimation failed (images likely don't overlap)"
    ),
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "camera parameter adjustment failed",
}


class PanoramaError(ValueError):
    """A stitch request this module refuses -- too few images, unreadable
    files, a Stitcher failure, or a result that fails seam QA. Never partial:
    raising this always means nothing was written to ``out_path``."""


@dataclass(frozen=True)
class PanoramaResult:
    """A successful stitch. ``out_path`` always exists when this is returned."""

    out_path: Path
    width: int
    height: int
    valid_area_fraction: float
    status: str
    message: str


def _valid_area_fraction(frame: np.ndarray) -> float:
    """Fraction of ``frame`` that is not pure black -- the Stitcher pads the
    warped canvas with black where no source image reaches."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(gray > 0)) / float(gray.size)


def stitch_panorama(paths, out_path) -> PanoramaResult:
    """Stitch ``paths`` (2+ ordered, overlapping images) into one panorama at
    ``out_path``. Raises ``PanoramaError`` -- never writes a partial output --
    when there are too few images, a file can't be decoded, the Stitcher
    itself fails, or the result fails seam QA (mostly-black canvas, or an
    output narrower than the widest input, both signs of a degenerate warp).
    """
    paths = [Path(p) for p in paths]
    if len(paths) < 2:
        raise PanoramaError(f"stitch_panorama needs >=2 images, got {len(paths)}")

    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise PanoramaError(f"{path}: could not be read as an image")
        images.append(image)

    # SCANS mode fits this pipeline's inputs: overlapping frames taken from
    # roughly the same spot with a horizontal pan, not a wide-FOV spherical
    # rig. PANORAMA mode's rotational/spherical warp model over-crops a
    # near-planar pan (observed: a 396px-wide input warped down to 220px on
    # this module's own test scene) -- exactly the degenerate shrink the seam
    # QA below exists to catch, so picking the matching model is the
    # upstream, robust-by-construction fix rather than loosening the gate.
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, stitched = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        reason = _STATUS_NAMES.get(status, f"stitcher error code {status}")
        raise PanoramaError(f"panorama stitch failed: {reason}")

    valid_fraction = _valid_area_fraction(stitched)
    max_input_width = max(image.shape[1] for image in images)
    if valid_fraction < MIN_VALID_AREA_FRACTION:
        raise PanoramaError(
            "panorama stitch rejected by seam QA: only "
            f"{valid_fraction:.0%} of the output is valid image data "
            f"(need >= {MIN_VALID_AREA_FRACTION:.0%})"
        )
    if stitched.shape[1] < max_input_width:
        raise PanoramaError(
            "panorama stitch rejected by seam QA: output width "
            f"{stitched.shape[1]}px is narrower than the widest input "
            f"({max_input_width}px)"
        )

    out_path = Path(out_path)
    if not cv2.imwrite(str(out_path), stitched):
        raise PanoramaError(f"{out_path}: failed to write stitched panorama")

    return PanoramaResult(
        out_path=out_path,
        width=int(stitched.shape[1]),
        height=int(stitched.shape[0]),
        valid_area_fraction=valid_fraction,
        status="ok",
        message="",
    )
