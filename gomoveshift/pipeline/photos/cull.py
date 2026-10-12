"""Per-photo culling score: read-only measurement surfaced in the
console library review so an operator can spot a soft/blown/duplicate photo
without opening every file.

The measurements (``CullScore``) and the near-duplicate grouping stay pure
numbers. ``cull_decision`` turns one score into a keep/skip verdict against
``CullFloors``; ``PhotoSelector`` uses it to skip soft and clipped photos and
to collapse near-duplicates, and the console review page keeps showing the
raw scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

from ..quality import sharpness_score

#: Bounds the Laplacian/pHash pass on a full-res photo (mirrors
#: ``thumbnail_generator.SHARPNESS_MAX_DIM``): sharpness and the perceptual
#: hash are both low-frequency-ish properties, so a downscale costs nothing
#: comparatively while keeping every photo measured on the same footing.
SHARPNESS_MAX_DIM = 960

#: A pixel at/below this 0..255 luma reads as clipped black (shadows with no
#: recoverable detail) -- 2, not 0, because a real photo's darkest pixel is
#: rarely bit-exact 0 even when the shadow is genuinely clipped.
CLIPPED_BLACK_LEVEL = 2

#: A pixel at/above this luma reads as clipped white (blown highlights) --
#: mirrors ``thumbnail_generator.BLOWN_LEVEL`` (250), same reasoning: real
#: sensor noise rarely lands on bit-exact 255 even at a genuine highlight
#: clip.
CLIPPED_WHITE_LEVEL = 250

#: Side length of the DCT pHash working image. 32 is the standard pHash
#: size: large enough that the low-frequency DCT coefficients below carry
#: real structure, small enough that the DCT itself is instant.
PHASH_DCT_SIZE = 32

#: The pHash keeps the top-left PHASH_HASH_SIZE x PHASH_HASH_SIZE block of
#: DCT coefficients, drops the DC term (overall brightness, not structure),
#: and thresholds the remaining 8*8 - 1 = 63 coefficients against their own
#: median -- the classic pHash shape, fitting in a 64-bit int with one bit
#: to spare.
PHASH_HASH_SIZE = 8

#: Two photos whose pHash Hamming distance is at or below this are treated
#: as near-duplicates (burst shots, near-identical re-takes). 64-bit pHash,
#: literature and this module's own noise test (±2 luma jitter) both put a
#: genuine re-encode/noise gap at single-digit bit flips; 5 gives headroom
#: over that without folding in visually distinct frames (see
#: test_phash_differs_for_visually_different_frames).
NEAR_DUPLICATE_HAMMING_THRESHOLD = 5


@dataclass(frozen=True)
class CullScore:
    """Measured culling signal for one photo. Every field is a plain
    measurement -- this dataclass carries no verdict (no ``.should_cull``)."""

    sharpness: float  # variance of the Laplacian of luma, on a bounded downscale
    exposure_black_fraction: float  # share of pixels at/below CLIPPED_BLACK_LEVEL
    exposure_white_fraction: float  # share of pixels at/above CLIPPED_WHITE_LEVEL
    phash: int  # 64-bit perceptual hash, for near-duplicate grouping


@dataclass(frozen=True)
class CullFloors:
    """Selection gates applied to a ``CullScore``.

    PROVISIONAL: these first values are educated guesses, to be tuned from one
    card's console review, not measured optima.
    """

    min_sharpness: float = 30.0  # Laplacian variance on the bounded downscale
    max_white_fraction: float = 0.25  # blown-highlight ceiling
    max_black_fraction: float = 0.50  # crushed-shadow ceiling (dusk shots run dark)
    duplicate_hamming: int = NEAR_DUPLICATE_HAMMING_THRESHOLD


@dataclass(frozen=True)
class CullDecision:
    keep: bool
    reason: str = ""


def cull_decision(scores: CullScore, *, floors: CullFloors = CullFloors()) -> CullDecision:
    """Keep/skip verdict for one photo's measured score. Duplicates are a
    group property, decided by the caller with ``floors.duplicate_hamming``."""
    if scores.sharpness < floors.min_sharpness:
        return CullDecision(
            False, f"soft: sharpness {scores.sharpness:.1f} < floor {floors.min_sharpness:g}"
        )
    if scores.exposure_white_fraction > floors.max_white_fraction:
        return CullDecision(
            False,
            f"highlight clipping {scores.exposure_white_fraction:.2f} "
            f"> ceiling {floors.max_white_fraction:g}",
        )
    if scores.exposure_black_fraction > floors.max_black_fraction:
        return CullDecision(
            False,
            f"shadow clipping {scores.exposure_black_fraction:.2f} "
            f"> ceiling {floors.max_black_fraction:g}",
        )
    return CullDecision(True)


def _bounded_gray(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if max(h, w) > SHARPNESS_MAX_DIM:
        k = SHARPNESS_MAX_DIM / float(max(h, w))
        size = (max(1, int(w * k)), max(1, int(h * k)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return gray


def _sharpness(gray: np.ndarray) -> float:
    return sharpness_score(gray)


def _exposure_clip(gray: np.ndarray) -> tuple[float, float]:
    total = gray.size
    if total == 0:
        return 0.0, 0.0
    black = float((gray <= CLIPPED_BLACK_LEVEL).sum()) / total
    white = float((gray >= CLIPPED_WHITE_LEVEL).sum()) / total
    return black, white


def _phash(gray: np.ndarray) -> int:
    """64-bit DCT perceptual hash: resize to a fixed small grid, DCT, keep
    the low-frequency block (dropping the DC term), threshold each
    coefficient against the block's own median. Two visually-similar photos
    land on nearly the same bits; a photo's exact resolution/crop-adjacent
    noise does not move the low-frequency DCT coefficients much, which is
    what makes this robust where a raw pixel hash is not."""
    small = cv2.resize(gray, (PHASH_DCT_SIZE, PHASH_DCT_SIZE), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small.astype(np.float64))
    block = dct[:PHASH_HASH_SIZE, :PHASH_HASH_SIZE].copy()
    block[0, 0] = 0.0  # DC term carries only overall brightness, not structure
    bits = block.flatten()[1:]  # drop the zeroed DC slot, keep the 63 AC coefficients
    median = float(np.median(bits))
    value = 0
    for i, coeff in enumerate(bits):
        if coeff > median:
            value |= 1 << i
    return value


def compute_cull_score(rgb: np.ndarray) -> CullScore:
    """Measure sharpness, exposure clipping, and a perceptual hash for one
    developed RGB photo frame. Raises ``ValueError`` on anything that is not
    an HxWx3 uint8 array -- the same shape contract ``frame_qa`` uses."""
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
    ):
        raise ValueError(
            f"expected an HxWx3 uint8 array, got "
            f"shape={getattr(rgb, 'shape', None)} dtype={getattr(rgb, 'dtype', None)}"
        )
    gray = _bounded_gray(rgb)
    sharpness = _sharpness(gray)
    black, white = _exposure_clip(gray)
    phash = _phash(gray)
    return CullScore(
        sharpness=sharpness,
        exposure_black_fraction=black,
        exposure_white_fraction=white,
        phash=phash,
    )


def hamming_distance(a: int, b: int) -> int:
    """Bit-flip distance between two 64-bit pHash values."""
    return bin(a ^ b).count("1")


def group_near_duplicates(hashes: Dict[str, int]) -> Dict[str, List[str]]:
    """For each key in *hashes*, the OTHER keys whose pHash is within
    ``NEAR_DUPLICATE_HAMMING_THRESHOLD`` bits -- an O(n^2) pairwise compare,
    fine at the per-day/per-journey-page scale this is called at (review is
    one page of photos, not the whole library).

    Every key in the input is present in the result, mapping to ``[]`` when
    it has no near-duplicate. This never decides which of a pair to keep --
    that is an operator call.
    """
    keys = list(hashes)
    groups: Dict[str, List[str]] = {k: [] for k in keys}
    for i, ki in enumerate(keys):
        for kj in keys[i + 1 :]:
            if hamming_distance(hashes[ki], hashes[kj]) <= NEAR_DUPLICATE_HAMMING_THRESHOLD:
                groups[ki].append(kj)
                groups[kj].append(ki)
    return groups
