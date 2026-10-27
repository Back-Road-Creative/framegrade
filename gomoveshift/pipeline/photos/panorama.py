"""Panorama stitching: an ordered set of overlapping photos -> one wide
frame, via OpenCV's ``Stitcher``.

The stitch is upstream-gated, not late-checked: a degenerate result (blown
out seams, a homography that collapsed the frame) never reaches disk. QA runs
on the stitched frame BEFORE any write, so a rejected stitch leaves ``out_path``
untouched -- callers never have to clean up a partial file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Below this fraction of non-black pixels, the stitch is treated as a failed
#: seam rather than a usable panorama -- OpenCV's Stitcher can return
#: ``Stitcher_OK`` for a technically-valid homography that still leaves most
#: of the canvas black (barely-overlapping frames, a near-degenerate warp).
MIN_VALID_AREA_FRACTION = 0.5

#: Seam measurement. Each adjacent source pair is registered with an ORB
#: feature-match similarity fit (the Stitcher does not expose its per-pair
#: registration), the next frame is warped onto the previous one, and the
#: overlap is compared on blurred luma.
#:
#: A pixel is "ghosted" when it still differs by more than this many luma
#: levels (of 255) after the seam's mean-luma step is removed, so an exposure
#: change alone never reads as a ghost.
GHOST_PIXEL_DIFF = 40
#: A seam is flagged when more than this fraction of its overlap is ghosted.
#: A moving person or car covers well over 2% of a frame's overlap, while the
#: sub-pixel registration error on a clean seam leaves a few tenths of a percent.
#: Chosen on synthetic scenes; not yet tuned on real handheld pans.
MAX_SEAM_GHOST_FRACTION = 0.02
#: A seam is flagged when the mean luma of the overlap differs by more than
#: this many levels (of 255) between the two frames -- about 1/6 stop at
#: mid-grey, a step the blender cannot hide. Auto-exposure drift between
#: handheld frames is usually below this; a step of 45 (the test) is far above.
MAX_SEAM_LUMA_STEP = 12.0
_MIN_SEAM_INLIERS = 12
#: Largest vertical feature shift (fraction of frame height) a seam match may show.
_MAX_VERTICAL_SHIFT = 0.15

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
    #: One entry per measured adjacent pair: fraction of the overlap that
    #: still differs after registration (ghosting), and the absolute
    #: mean-luma difference across the overlap (exposure step). A pair with
    #: too few feature matches to register is skipped, so these can be
    #: shorter than ``len(paths) - 1``.
    seam_ghost_fractions: tuple[float, ...] = ()
    seam_luma_steps: tuple[float, ...] = ()


def _valid_area_fraction(frame: np.ndarray) -> float:
    """Fraction of ``frame`` that is not pure black -- the Stitcher pads the
    warped canvas with black where no source image reaches."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(gray > 0)) / float(gray.size)


def measure_seam(prev: np.ndarray, nxt: np.ndarray) -> tuple[float, float] | None:
    """Register ``nxt`` onto ``prev`` and measure their overlap.

    Returns ``(ghost_fraction, luma_step)``, or ``None`` when too few feature
    matches survive RANSAC to register the pair (nothing to measure)."""
    gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gray_next = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(4000)
    kp_p, des_p = orb.detectAndCompute(gray_prev, None)
    kp_n, des_n = orb.detectAndCompute(gray_next, None)
    if des_p is None or des_n is None:
        return None
    # Lowe ratio test: a repeating texture (tiles, windows) otherwise matches
    # a feature to a neighbour one period over and registers the pair wrongly.
    matches = []
    for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_n, des_p, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            matches.append(pair[0])
    if len(matches) < _MIN_SEAM_INLIERS:
        return None
    src = np.float32([kp_n[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_p[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # SCANS inputs pan horizontally, so a match that moves a feature far
    # vertically is a false one (a symmetric scene also fits a 180-degree
    # rotation, which no real pan is).
    keep = np.abs(dst[:, 0, 1] - src[:, 0, 1]) < _MAX_VERTICAL_SHIFT * gray_prev.shape[0]
    src, dst = src[keep], dst[keep]
    if len(src) < _MIN_SEAM_INLIERS:
        return None
    # A similarity (shift, rotation, uniform scale) is all a hand-held pan
    # between two frames needs, and it is far harder to fool than a full
    # homography.
    affine, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if affine is None or int(inliers.sum()) < _MIN_SEAM_INLIERS:
        return None
    homography = np.vstack([affine, [0.0, 0.0, 1.0]])
    h, w = gray_prev.shape
    warped = cv2.warpPerspective(gray_next, homography, (w, h))
    mask = cv2.warpPerspective(np.full(gray_next.shape, 255, np.uint8), homography, (w, h))
    # Erode so warp-edge interpolation never counts as overlap content.
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
    if not mask.any():
        return None
    a = cv2.GaussianBlur(gray_prev, (5, 5), 0).astype(np.float32)[mask]
    b = cv2.GaussianBlur(warped, (5, 5), 0).astype(np.float32)[mask]
    step = float(a.mean() - b.mean())
    ghost = float(np.mean(np.abs(a - (b + step)) > GHOST_PIXEL_DIFF))
    return ghost, abs(step)


def stitch_panorama(paths, out_path, *, strict_seams: bool = False) -> PanoramaResult:
    """Stitch ``paths`` (2+ ordered, overlapping images) into one panorama at
    ``out_path``. Raises ``PanoramaError`` -- never writes a partial output --
    when there are too few images, a file can't be decoded, the Stitcher
    itself fails, or the result fails seam QA (mostly-black canvas, or an
    output narrower than the widest input, both signs of a degenerate warp).

    Seams over ``MAX_SEAM_GHOST_FRACTION`` / ``MAX_SEAM_LUMA_STEP`` are flagged:
    by default the panorama is still written and the reasons are logged at
    WARNING and returned in ``message`` (the thresholds are not yet tuned on
    real handheld pans, and the Stitcher's seam finder and exposure
    compensator often hide what the raw overlap shows). ``strict_seams=True``
    refuses instead, before the write.
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

    ghosts: list[float] = []
    steps: list[float] = []
    flags: list[str] = []
    for index, (prev, nxt) in enumerate(zip(images, images[1:]), start=1):
        measured = measure_seam(prev, nxt)
        if measured is None:
            continue
        ghost, step = measured
        ghosts.append(ghost)
        steps.append(step)
        if ghost > MAX_SEAM_GHOST_FRACTION:
            flags.append(
                f"ghosting at seam {index} ({ghost:.1%} of the overlap differs, "
                f"max {MAX_SEAM_GHOST_FRACTION:.1%})"
            )
        if step > MAX_SEAM_LUMA_STEP:
            flags.append(
                f"exposure step at seam {index} ({step:.1f} luma levels, "
                f"max {MAX_SEAM_LUMA_STEP:.0f})"
            )
    seam_message = "; ".join(flags)
    if flags and strict_seams:
        raise PanoramaError(f"panorama stitch rejected by seam QA: {seam_message}")
    if flags:
        logger.warning("panorama seam QA: %s", seam_message)

    out_path = Path(out_path)
    if not cv2.imwrite(str(out_path), stitched):
        raise PanoramaError(f"{out_path}: failed to write stitched panorama")

    return PanoramaResult(
        out_path=out_path,
        width=int(stitched.shape[1]),
        height=int(stitched.shape[0]),
        valid_area_fraction=valid_fraction,
        status="ok",
        message=seam_message,
        seam_ghost_fractions=tuple(ghosts),
        seam_luma_steps=tuple(steps),
    )
