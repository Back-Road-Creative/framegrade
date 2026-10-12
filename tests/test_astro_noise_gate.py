"""The deep combine measures its own noise efficiency instead of trusting a depth constant.

An earlier run established that the scoped product passes its gates up to 30 chunks and fails at 34, and
recommended AGAINST shipping 30 as a constant: it is one night's property, and a later measurement found the
split-half estimator itself carrying +/-3.4 pp, so a fixed threshold can silently drift out of
validity.  These tests pin the measurement that replaces it.
"""

import numpy as np
import pytest

from gomoveshift.pipeline.photos import astro_noise_gate as gate


def _chunks(n, shape, rng, *, drift=0.0, fixed_amp=0.0, frames=10):
    """n chunk box-means of pure noise, optionally plus structure.

    `fixed_amp` adds a pattern IDENTICAL in every chunk; `drift` ramps that pattern across the
    chunk order.  The distinction matters and is the point of two separate tests: a perfectly
    fixed pattern cancels in the split-half difference AND in the adjacent-chunk difference, so
    corrected efficiency is rightly blind to it.  Slowly-varying structure does not cancel
    equally on both sides, and that is what the gate is for.
    """
    pattern = rng.normal(0.0, 1.0, size=shape)
    means, counts = [], []
    for k in range(n):
        noise = rng.normal(0.0, 1.0, size=shape) / np.sqrt(frames)
        ramp = drift * pattern * (k / max(n - 1, 1))
        means.append(noise + fixed_amp * pattern + ramp)
        counts.append(np.full(shape, float(frames)))
    return means, counts


def test_pure_noise_reaches_near_ideal_efficiency():
    rng = np.random.default_rng(7)
    means, counts = _chunks(12, (64, 64), rng)
    eff = gate.noise_efficiency(means, counts, frames_per_chunk=10, rng=np.random.default_rng(1))
    assert 90.0 <= eff <= 110.0


def test_a_perfectly_fixed_pattern_is_correctly_invisible():
    """Differencing cancels what both sides share, so a static pattern must NOT score as noise."""
    rng = np.random.default_rng(7)
    means, counts = _chunks(12, (64, 64), rng, fixed_amp=0.6)
    eff = gate.noise_efficiency(means, counts, frames_per_chunk=10, rng=np.random.default_rng(1))
    assert eff > 90.0


def test_slowly_varying_structure_drags_efficiency_below_the_floor():
    """The real failure mode: structure that changes across the stack does not average away."""
    rng = np.random.default_rng(7)
    means, counts = _chunks(12, (64, 64), rng, drift=2.0)
    eff = gate.noise_efficiency(means, counts, frames_per_chunk=10, rng=np.random.default_rng(1))
    assert eff < 80.0


def test_floor_is_eighty_percent():
    assert gate.NOISE_EFFICIENCY_FLOOR_PCT == 80.0


def test_check_passes_clean_boxes_and_names_the_failing_one():
    rng = np.random.default_rng(3)
    good_m, good_c = _chunks(12, (64, 64), rng)
    bad_m, bad_c = _chunks(12, (64, 64), rng, drift=2.0)
    report = gate.check_noise_floor(
        {"centre": (good_m, good_c), "left": (bad_m, bad_c)}, frames_per_chunk=10, seed=1
    )
    assert report.passed is False
    assert report.failing_boxes == ("left",)
    assert report.efficiency_pct["centre"] > 80.0
    assert report.efficiency_pct["left"] < 80.0


def test_check_passes_when_every_box_clears():
    rng = np.random.default_rng(5)
    a, b = _chunks(12, (64, 64), rng)
    c, d = _chunks(12, (64, 64), rng)
    report = gate.check_noise_floor({"centre": (a, b), "left": (c, d)}, frames_per_chunk=10, seed=1)
    assert report.passed is True
    assert report.failing_boxes == ()


def test_too_few_chunks_to_split_is_refused_not_guessed():
    rng = np.random.default_rng(5)
    means, counts = _chunks(1, (32, 32), rng)
    with pytest.raises(gate.NoiseGateError):
        gate.noise_efficiency(means, counts, frames_per_chunk=10, rng=np.random.default_rng(1))


def test_effective_contributors_matches_chunk_count_for_equal_weights():
    """Five full (frames_per_chunk-weighted) chunks are worth exactly 5 chunk-equivalents,
    same as the unweighted `sum(counts) / frames_per_chunk` this replaces."""
    counts = [np.full((4, 4), 10.0) for _ in range(5)]
    effective = gate.effective_contributors(counts, frames_per_chunk=10)
    assert np.allclose(effective, 5.0)


def test_effective_contributors_is_dominated_by_one_outsized_weight():
    """(sum w)^2 / sum w^2, over weights normalised by frames_per_chunk: one chunk carrying
    nearly all the weight should NOT count as if every chunk contributed a full chunk's worth.
    """
    counts = [np.full((2, 2), 1.0), np.full((2, 2), 1.0), np.full((2, 2), 100.0)]
    effective = gate.effective_contributors(counts, frames_per_chunk=1)
    naive_sum = 102.0
    expected = (naive_sum**2) / (1.0 + 1.0 + 100.0 * 100.0)
    assert np.allclose(effective, expected)
    assert np.all(effective < naive_sum)
