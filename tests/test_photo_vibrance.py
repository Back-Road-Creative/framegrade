"""Tests for low-saturation-weighted vibrance in the photo develop chain.

``_apply_vibrance`` is a photo-develop-only look: a chroma boost around BT.709
luma whose per-pixel push is weighted by ``(1 - s)`` (``s`` = the pixel's
``(max-min)/max`` saturation), so muted pixels take the full ``vibrance`` lift
while already-rich pixels barely move — finer than vivid's scene-level headroom
gate. It rides ``_grade`` right after ``_apply_saturation``. Default-OFF
(``vibrance == 0``) is bit-identical, a neutral gray stays gray, and it is
deliberately NOT wired into ``params_to_lut3d_cube``.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from gomoveshift.pipeline import color_correction as cc
from gomoveshift.pipeline.photos import develop as dev


def _saturation(px: np.ndarray) -> float:
    """A pixel's ``(max-min)/max`` saturation, the grade engine's basis."""
    mx = float(px.max())
    return 0.0 if mx <= 0.0 else (mx - float(px.min())) / mx


def test_default_off_is_bit_identical():
    """``vibrance == 0`` leaves every pixel untouched (early ``rgb.copy()``)."""
    rng = np.random.default_rng(0)
    rgb = rng.random((256, 3)).astype(np.float32)

    assert np.array_equal(cc._apply_vibrance(rgb, cc.CorrectionParams()), rgb)


def test_grade_default_off_matches_the_chain_without_vibrance():
    """The default photo grade is byte-identical to the pre-vibrance chain."""
    frame = np.linspace(0, 255, 48 * 32 * 3, dtype=np.uint8).reshape(32, 48, 3)
    params = cc.CorrectionParams()

    flat = frame.reshape(-1, 3).astype(np.float32) / 255.0
    out = cc._apply_curves(flat, params)
    out = cc._apply_colorbalance(out, params)
    out = cc._apply_saturation(out, params.saturation_mult)
    out = cc._apply_hsl(out, params)
    out = cc._apply_gamma(out, params.gamma)
    expected = np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8).reshape(frame.shape)

    assert np.array_equal(dev._grade(frame, params), expected)


def test_muted_patch_gains_more_saturation_than_a_rich_patch():
    """The defining low-sat weighting: at ONE vibrance the muted patch gains more
    saturation than an already-rich one (skies/foliage protected from cartooning),
    while both still gain some."""
    muted = np.array([[0.5, 0.45, 0.40]], dtype=np.float32)  # low saturation
    rich = np.array([[0.9, 0.10, 0.10]], dtype=np.float32)  # already saturated
    params = cc.CorrectionParams(vibrance=1.0)

    muted_gain = _saturation(cc._apply_vibrance(muted, params)[0]) - _saturation(muted[0])
    rich_gain = _saturation(cc._apply_vibrance(rich, params)[0]) - _saturation(rich[0])

    assert muted_gain > 0.0  # a real effect
    assert rich_gain > 0.0
    assert muted_gain > rich_gain  # muted takes the larger push


def test_neutral_gray_stays_gray():
    """A fully neutral patch has zero chroma to scale, so it never tints."""
    gray = np.full((16, 3), 0.5, dtype=np.float32)
    out = cc._apply_vibrance(gray, cc.CorrectionParams(vibrance=2.0))

    assert np.allclose(out, gray, atol=1e-6)
    assert np.allclose(out[:, 0], out[:, 1]) and np.allclose(out[:, 1], out[:, 2])


def test_is_deterministic():
    rgb = np.random.default_rng(1).random((512, 3)).astype(np.float32)
    params = cc.CorrectionParams(vibrance=0.7)

    assert np.array_equal(cc._apply_vibrance(rgb, params), cc._apply_vibrance(rgb, params))


def test_vibrance_is_clamped_to_a_sane_range():
    """An absurd vibrance clamps to the applier ceiling rather than blowing chroma."""
    rgb = np.random.default_rng(3).random((128, 3)).astype(np.float32)

    assert np.array_equal(
        cc._apply_vibrance(rgb, cc.CorrectionParams(vibrance=100.0)),
        cc._apply_vibrance(rgb, cc.CorrectionParams(vibrance=2.0)),
    )


def test_vibrance_feeds_the_develop_cache_version():
    """A retune of the vibrance applier must invalidate the developed cache."""
    assert cc._apply_vibrance in dev._HASHED_SOURCES


def _write_photo(path: Path) -> Path:
    rng = np.random.default_rng(11)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def test_vibrance_threads_onto_every_styled_variant_but_not_the_base(tmp_path):
    """A non-zero vibrance lands on EVERY styled variant's params (natural too),
    but never the base — LUT looks ride a clean film-emulation base."""
    photo = _write_photo(tmp_path / "IMG_0001.jpg")
    plan = dev.plan_variants(photo, ["natural", "moody"], options=dev.DevelopOptions(vibrance=0.6))

    for _, params in plan.styled:
        assert params.vibrance == 0.6
    assert plan.base.vibrance == 0.0
