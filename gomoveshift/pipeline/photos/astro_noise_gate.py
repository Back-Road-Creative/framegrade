"""Measure whether a deep stack's noise actually averages down, instead of assuming a depth.

A fixed chunk-count cap would be a property of one night's data, and the split-half
estimator carries +/-3.4 pp on its own, so a fixed threshold can pass a stack that does not
meet the gate.  This module measures the thing such a constant would stand in for.

The estimator is a split-half one:

* master noise      -- half the MAD-sigma of the difference between two balanced half-stacks.
                       Differencing cancels everything common to both halves, so what survives is
                       noise, not structure; the /2 undoes the doubling from differencing two
                       independent halves.
* per-chunk noise   -- MAD-sigma of adjacent-chunk differences / sqrt(2), the same trick one
                       level down.
* ideal gain        -- sqrt(effective_contributors), the weighted effective-sample-size
                       (sum r)^2 / sum r^2 over each box's per-chunk weights normalised by
                       frames_per_chunk -- equal to sqrt(chunk_count) when every chunk weighs a
                       full frames_per_chunk (a quality-weighted chunk no longer does).

Efficiency is (chunk_noise / master_noise) / ideal_gain.  100% means the stack averaged exactly
as sqrt(N) predicts; the left box on one measured night reaches 71.5% at 38 chunks because
something in it does not average at all (diffuse, large-scale, cause still unknown).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

# Both from a measured contract.  The floor is the original gate; the sample count is what
# the measurements used, and 41 splits leave +/-3.4 pp of sampling spread -- raise it if a
# decision ever rests on a margin tighter than that.
NOISE_EFFICIENCY_FLOOR_PCT = 80.0
NOISE_SPLIT_SAMPLES = 41
MAD_TO_SIGMA = 1.4826


class NoiseGateError(RuntimeError):
    """The efficiency measurement could not be made (not: was made and failed)."""


@dataclass(frozen=True)
class NoiseFloorReport:
    passed: bool
    efficiency_pct: Mapping[str, float]
    failing_boxes: tuple[str, ...]
    floor_pct: float


def _mad_sigma(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(finite - np.median(finite))))


def detrend(box: np.ndarray) -> np.ndarray:
    """Remove a 2nd-degree background so a gradient is not scored as noise."""
    box = np.asarray(box, dtype=np.float64)
    rows, cols = box.shape
    y, x = np.mgrid[0:rows, 0:cols]
    x = (x / max(cols, 1)) - 0.5
    y = (y / max(rows, 1)) - 0.5
    design = np.stack(
        [
            np.ones_like(x).ravel(),
            x.ravel(),
            y.ravel(),
            (x * x).ravel(),
            (x * y).ravel(),
            (y * y).ravel(),
        ],
        axis=1,
    )
    flat = box.ravel()
    ok = np.isfinite(flat)
    if int(ok.sum()) < 3 * design.shape[1]:
        return np.full_like(box, np.nan)
    coef, *_ = np.linalg.lstsq(design[ok], flat[ok], rcond=None)
    return (flat - design @ coef).reshape(box.shape)


def effective_contributors(counts: Sequence[np.ndarray], *, frames_per_chunk: int) -> np.ndarray:
    """Effective per-pixel chunk-equivalent count, generalising ``sum(counts) / frames_per_chunk``.

    Each chunk's per-pixel weight is normalised to "fraction of one full chunk"
    (``count / frames_per_chunk``) and combined via the standard weighted effective-sample-size
    formula, ``(sum r)^2 / sum r^2``. When every chunk carries exactly ``frames_per_chunk``
    (the historical, unweighted case: a literal frame count), every ``r`` is 1 and this reduces
    exactly to the chunk count -- the same answer ``sum(counts) / frames_per_chunk`` gave before.
    A chunk's contribution is a quality weight rather than a literal frame count, so a
    naive sum would let one badly-weighted chunk masquerade as several independent ones;
    this formula does not.
    """
    normalized = [np.asarray(c, dtype=np.float64) / float(frames_per_chunk) for c in counts]
    weight_sum = sum(normalized)
    weight_sq_sum = sum(w * w for w in normalized)
    return np.divide(
        weight_sum**2,
        weight_sq_sum,
        out=np.zeros_like(weight_sum, dtype=np.float64),
        where=weight_sq_sum > 0,
    )


def _stack(indices: Sequence[int], means: Sequence[np.ndarray], counts: Sequence[np.ndarray]):
    total = sum(means[i] * counts[i] for i in indices)
    weight = sum(counts[i] for i in indices)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight > 0, total / np.where(weight > 0, weight, 1.0), np.nan)


def noise_efficiency(means, counts, *, frames_per_chunk: int, rng) -> float:
    """Percent of the ideal sqrt(N) improvement this stack actually achieved, in one box."""
    n = len(means)
    if n < 2:
        raise NoiseGateError(f"need >= 2 chunks to split-half; got {n}")
    order = list(range(n))
    samples = []
    for _ in range(NOISE_SPLIT_SAMPLES):
        half = sorted(rng.permutation(order)[: n // 2])
        other = [i for i in order if i not in set(half)]
        samples.append(
            _mad_sigma(detrend(_stack(half, means, counts) - _stack(other, means, counts))) / 2.0
        )
    finite = np.asarray([s for s in samples if np.isfinite(s)])
    if finite.size == 0:
        raise NoiseGateError("split-half differences were not finite anywhere")
    master_noise = float(np.median(finite))
    pairwise = [_mad_sigma(detrend(means[i] - means[i + 1])) / np.sqrt(2.0) for i in range(n - 1)]
    chunk_noise = float(np.median([p for p in pairwise if np.isfinite(p)]))
    if not np.isfinite(chunk_noise) or master_noise <= 0.0:
        raise NoiseGateError("could not measure a per-chunk noise level")
    contributors = effective_contributors(counts, frames_per_chunk=frames_per_chunk)
    positive = contributors[contributors > 0]
    if positive.size == 0:
        raise NoiseGateError("no contributing pixels in this box")
    ideal = float(np.sqrt(float(np.median(positive))))
    return 100.0 * (chunk_noise / master_noise) / ideal


def check_noise_floor(boxes, *, frames_per_chunk: int, seed: int = 20260820) -> NoiseFloorReport:
    """Measure every sample box and report which, if any, fall below the floor."""
    efficiency = {
        name: noise_efficiency(
            means, counts, frames_per_chunk=frames_per_chunk, rng=np.random.default_rng(seed)
        )
        for name, (means, counts) in boxes.items()
    }
    failing = tuple(sorted(n for n, e in efficiency.items() if e < NOISE_EFFICIENCY_FLOOR_PCT))
    return NoiseFloorReport(
        passed=not failing,
        efficiency_pct=efficiency,
        failing_boxes=failing,
        floor_pct=NOISE_EFFICIENCY_FLOOR_PCT,
    )


# Sample-box geometry from the measurements: 512 px boxes at mid-height,
# one centred and one at each of +/-25% of the width.  All three sit inside the shipped scoped
# frame -- gating on a box outside the crop would refuse a product over pixels nobody receives.
# The left box is the one that has ever bound; centre and right stayed at 97-101% at every depth
# measured, and why the left differs is still unexplained.
NOISE_BOX_PX = 512
NOISE_BOX_OFFSET_FRACTION = 0.25


def noise_sample_boxes(shape) -> dict[str, tuple[slice, slice]]:
    """Row/column slices of the sample boxes, or {} when the frame is too small to hold them."""
    height, width = int(shape[0]), int(shape[1])
    box = NOISE_BOX_PX
    offset = int(NOISE_BOX_OFFSET_FRACTION * width)
    row0 = height // 2 - box // 2
    centre_col0 = width // 2 - box // 2
    starts = {
        "left": centre_col0 - offset,
        "centre": centre_col0,
        "right": centre_col0 + offset,
    }
    if row0 < 0 or min(starts.values()) < 0 or max(starts.values()) + box > width:
        return {}
    return {
        name: (slice(row0, row0 + box), slice(col0, col0 + box)) for name, col0 in starts.items()
    }
