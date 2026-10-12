"""Per-photo culling score: sharpness, exposure clipping, and a
perceptual hash for near-duplicate grouping. Read-only measurement -- this
module never deletes or rejects a photo, it only scores one.
"""

import numpy as np
import pytest

from gomoveshift.pipeline.photos import cull


def _flat(value=128, size=(120, 160)):
    return np.full((*size, 3), value, dtype=np.uint8)


def _textured(size=(120, 160), seed=3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (*size, 3), dtype=np.uint8)


def _checkerboard(size=(120, 160), square=4):
    h, w = size
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    board = (((yy // square) + (xx // square)) % 2) * 255
    frame = board.astype(np.uint8)
    return np.repeat(frame[:, :, np.newaxis], 3, axis=2)


# ── sharpness ──────────────────────────────────────────────────────────


def test_sharp_frame_scores_higher_than_flat_frame():
    flat_score = cull.compute_cull_score(_flat())
    sharp_score = cull.compute_cull_score(_checkerboard())
    assert sharp_score.sharpness > flat_score.sharpness
    assert flat_score.sharpness == pytest.approx(0.0, abs=1e-6)


# ── exposure clipping ───────────────────────────────────────────────────


def test_all_black_frame_is_fully_clipped_black():
    score = cull.compute_cull_score(_flat(value=0))
    assert score.exposure_black_fraction == pytest.approx(1.0)
    assert score.exposure_white_fraction == pytest.approx(0.0)


def test_all_white_frame_is_fully_clipped_white():
    score = cull.compute_cull_score(_flat(value=255))
    assert score.exposure_white_fraction == pytest.approx(1.0)
    assert score.exposure_black_fraction == pytest.approx(0.0)


def test_midtone_frame_has_no_clipping():
    score = cull.compute_cull_score(_flat(value=128))
    assert score.exposure_black_fraction == pytest.approx(0.0)
    assert score.exposure_white_fraction == pytest.approx(0.0)


def test_compute_cull_score_rejects_non_rgb_uint8():
    with pytest.raises(ValueError):
        cull.compute_cull_score(np.zeros((10, 10), dtype=np.uint8))


# ── perceptual hash ──────────────────────────────────────────────────────


def test_phash_is_64_bit():
    score = cull.compute_cull_score(_textured())
    assert 0 <= score.phash < (1 << 64)


def test_phash_is_stable_under_jpeg_like_noise():
    """A tiny per-pixel wobble (re-encode noise) must not flip the hash."""
    base = _textured()
    noisy = np.clip(
        base.astype(np.int16) + np.random.default_rng(1).integers(-2, 3, base.shape),
        0,
        255,
    ).astype(np.uint8)
    a = cull.compute_cull_score(base).phash
    b = cull.compute_cull_score(noisy).phash
    assert cull.hamming_distance(a, b) <= cull.NEAR_DUPLICATE_HAMMING_THRESHOLD


def test_phash_differs_for_visually_different_frames():
    a = cull.compute_cull_score(_textured(seed=11)).phash
    b = cull.compute_cull_score(_textured(seed=97)).phash
    assert cull.hamming_distance(a, b) > cull.NEAR_DUPLICATE_HAMMING_THRESHOLD


# ── near-duplicate grouping ──────────────────────────────────────────────


def test_group_near_duplicates_pairs_close_hashes():
    a_hash = cull.compute_cull_score(_textured(seed=1)).phash
    b_hash = a_hash  # identical frame, e.g. a burst duplicate
    c_hash = cull.compute_cull_score(_checkerboard()).phash
    groups = cull.group_near_duplicates({"a": a_hash, "b": b_hash, "c": c_hash})
    assert groups["a"] == ["b"]
    assert groups["b"] == ["a"]
    assert groups.get("c", []) == []


def test_group_near_duplicates_empty_when_all_distinct():
    a_hash = cull.compute_cull_score(_textured(seed=11)).phash
    b_hash = cull.compute_cull_score(_textured(seed=97)).phash
    groups = cull.group_near_duplicates({"a": a_hash, "b": b_hash})
    assert groups.get("a", []) == []
    assert groups.get("b", []) == []


def test_hamming_distance_symmetric_and_zero_for_equal():
    assert cull.hamming_distance(0b1010, 0b1010) == 0
    assert cull.hamming_distance(0b1010, 0b0101) == 4
    assert cull.hamming_distance(0b1111, 0b0000) == cull.hamming_distance(0b0000, 0b1111)
