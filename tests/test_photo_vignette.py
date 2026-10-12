"""The spatial seam: the reusable ``_apply_gain_mask`` primitive plus the
first position-aware stage built on it, a resolution-independent radial vignette.

The seam is what later spatial stages (clarity, grain, halation, sky-mask) reuse:
each computes an (H,W) per-pixel gain and hands it to ``_apply_gain_mask``. The
hard invariant is that an all-ones gain (amount 0) is a byte-identical no-op, so
a stage that measures "no change" costs nothing and the default develop is
unchanged from before the seam existed.
"""

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.photos import develop as develop_mod


def _write_photo(path, size=(80, 100)):
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _corner_mean(a, k=6):
    return float(np.mean([a[:k, :k], a[:k, -k:], a[-k:, :k], a[-k:, -k:]]))


def _center_mean(a, k=4):
    h, w = a.shape[:2]
    return float(np.mean(a[h // 2 - k : h // 2 + k, w // 2 - k : w // 2 + k]))


# ── the reusable gain-mask seam ──────────────────────────────────────


def test_apply_gain_mask_ones_is_byte_identical():
    """The load-bearing no-op: gain==1.0 everywhere returns the exact input bytes."""
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
    out = develop_mod._apply_gain_mask(rgb, np.ones((20, 30), dtype=np.float32))
    assert out.dtype == np.uint8
    assert np.array_equal(out, rgb)


def test_apply_gain_mask_accepts_hw1_gain_and_clips():
    rgb = np.full((4, 5, 3), 200, dtype=np.uint8)
    halved = develop_mod._apply_gain_mask(rgb, np.full((4, 5, 1), 0.5, dtype=np.float32))
    assert np.all(halved == 100)
    # gains above 1 clip at 255 rather than wrapping.
    boosted = develop_mod._apply_gain_mask(rgb, np.full((4, 5), 2.0, dtype=np.float32))
    assert np.all(boosted == 255)


# ── the radial vignette mask ─────────────────────────────────────────


def test_radial_vignette_mask_zero_amount_is_all_ones():
    mask = develop_mod._radial_vignette_mask(30, 40, 0.0)
    assert mask.shape == (30, 40)
    assert np.array_equal(mask, np.ones((30, 40), dtype=mask.dtype))


def test_radial_vignette_mask_darkens_corners_more_than_center():
    amount = 0.5
    mask = develop_mod._radial_vignette_mask(50, 70, amount)
    center = mask[25, 35]
    corners = [mask[0, 0], mask[0, -1], mask[-1, 0], mask[-1, -1]]
    assert center == pytest.approx(1.0, abs=1e-3), "the center is never darkened"
    for c in corners:
        assert c < center
        assert c == pytest.approx(1.0 - amount, abs=1e-3), "corner gain == 1 - amount"


def test_radial_vignette_mask_is_resolution_independent():
    """Same normalized falloff at any frame size — the corner reads identically."""
    small = develop_mod._radial_vignette_mask(50, 50, 0.4)
    large = develop_mod._radial_vignette_mask(600, 900, 0.4)
    assert small[0, 0] == pytest.approx(large[0, 0], abs=1e-4)
    assert small[0, 0] == pytest.approx(1.0 - 0.4, abs=1e-3)


def test_vignette_feather_participates_in_the_algo_version(monkeypatch):
    """The shape knob is an ALL_CAPS tunable, so a retune self-invalidates the cache."""
    baseline = develop_mod._develop_algo_version()
    monkeypatch.setattr(develop_mod, "VIGNETTE_FEATHER", develop_mod.VIGNETTE_FEATHER + 1.0)
    assert develop_mod._develop_algo_version() != baseline


# ── end to end through the develop stage ─────────────────────────────


def test_develop_variants_default_is_byte_identical_to_no_vignette(tmp_path):
    photo = _write_photo(tmp_path / "IMG_A.jpg")
    plain = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    off = develop_mod.DevelopOptions(vignette=0.0)
    zero = dict(develop_mod.develop_variants(photo, ["natural"], options=off))["natural"].rgb
    assert np.array_equal(plain, zero)


def test_develop_variants_vignette_darkens_corners_more_than_center(tmp_path):
    photo = _write_photo(tmp_path / "IMG_B.jpg")
    base = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    on = develop_mod.DevelopOptions(vignette=0.6)
    vig = dict(develop_mod.develop_variants(photo, ["natural"], options=on))["natural"].rgb
    corner_drop = _corner_mean(base) - _corner_mean(vig)
    center_drop = _center_mean(base) - _center_mean(vig)
    assert corner_drop > center_drop
    assert center_drop == pytest.approx(0.0, abs=2.0), "the center stays put"


def test_develop_single_photo_vignette_darkens_corners(tmp_path):
    photo = _write_photo(tmp_path / "IMG_C.jpg")
    base = develop_mod.develop(photo).rgb
    vig = develop_mod.develop(photo, vignette=0.6).rgb
    assert _corner_mean(base) - _corner_mean(vig) > _center_mean(base) - _center_mean(vig)
