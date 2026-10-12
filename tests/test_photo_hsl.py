"""Tests for the per-hue HSL adjustment stage of the photo develop grade chain.

``_apply_hsl`` is a photo-develop-only look control: three optional length-6 (or
length-7, orange included) tuples steer hue rotation / saturation / value across
the fixed anchor hues. Default-OFF must be bit-identical; a blue-band boost must
leave a red patch untouched; and every shipped length-6 recipe must still render
BIT-IDENTICALLY now that a seventh anchor exists. It is deliberately NOT wired
into ``params_to_lut3d_cube``.
"""

import numpy as np
import pytest

from gomoveshift.cli.handlers.photos import _resolve_hue_band
from gomoveshift.pipeline import color_correction as cc
from gomoveshift.pipeline.photos import develop as dev
from gomoveshift.pipeline.photos.looks import BUILTIN_LOOKS


def _sat(px):
    """Simple HSV saturation of one RGB pixel in [0, 1]."""
    mx, mn = float(np.max(px)), float(np.min(px))
    return 0.0 if mx <= 1e-12 else (mx - mn) / mx


def _six_anchor_reference(rgb, shift=(), sat_ctrl=(), lum_ctrl=()):
    """The six-anchor applier as it shipped BEFORE orange existed, verbatim.

    Reproduced (not imported) on purpose: it is the fixed reference every length-6
    control must still match bit-for-bit, and recomputing from the recipe keeps a
    legitimate look retune green while a change to the six-anchor geometry or the
    interpolation goes red."""
    anchors = np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0])

    def _band(ctrl, base):
        arr = np.full(6, base) if not ctrl else np.asarray(ctrl, dtype=np.float64)
        return np.append(arr, arr[0])

    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = np.max(rgb, axis=1).astype(np.float64)
    chroma = mx - np.min(rgb, axis=1)
    nz = chroma > 1e-12
    hue = np.zeros_like(mx)
    is_r = nz & (mx == r)
    is_g = nz & (mx == g) & ~is_r
    is_b = nz & ~is_r & ~is_g
    hue[is_r] = ((g - b)[is_r] / chroma[is_r]) % 6.0
    hue[is_g] = ((b - r)[is_g] / chroma[is_g]) + 2.0
    hue[is_b] = ((r - g)[is_b] / chroma[is_b]) + 4.0
    hue *= 60.0
    sat = np.where(mx > 1e-12, chroma / np.where(nz, mx, 1.0), 0.0)
    val = mx
    new_hue = np.mod(hue + np.interp(hue, anchors, _band(shift, 0.0)), 360.0)
    sat = np.clip(sat * np.interp(hue, anchors, _band(sat_ctrl, 1.0)), 0.0, 1.0)
    val = np.clip(val * np.interp(hue, anchors, _band(lum_ctrl, 1.0)), 0.0, 1.0)
    h6 = np.mod(new_hue / 60.0, 6.0)
    i = np.floor(h6).astype(np.int64)
    f = h6 - i
    p, q = val * (1.0 - sat), val * (1.0 - sat * f)
    t = val * (1.0 - sat * (1.0 - f))
    conds = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    out = np.stack(
        [
            np.select(conds, [val, q, p, p, t, val]),
            np.select(conds, [t, val, val, q, p, p]),
            np.select(conds, [p, p, t, val, val, q]),
        ],
        axis=1,
    )
    return np.clip(out, 0.0, 1.0).astype(rgb.dtype, copy=False)


def test_neutral_params_are_a_bit_identical_no_op():
    """An all-neutral (or empty) params leaves every pixel untouched."""
    rng = np.random.default_rng(0)
    rgb = rng.random((256, 3)).astype(np.float32)

    assert np.array_equal(cc._apply_hsl(rgb, cc.CorrectionParams()), rgb)
    neutral = cc.CorrectionParams(hue_shift=(0.0,) * 6, hue_sat=(1.0,) * 6, hue_lum=(1.0,) * 6)
    assert np.array_equal(cc._apply_hsl(rgb, neutral), rgb)


def test_grade_default_off_matches_the_chain_without_hsl():
    """The default photo grade is byte-identical to the pre-HSL applier chain."""
    frame = np.linspace(0, 255, 48 * 32 * 3, dtype=np.uint8).reshape(32, 48, 3)
    params = cc.CorrectionParams()

    flat = frame.reshape(-1, 3).astype(np.float32) / 255.0
    out = cc._apply_curves(flat, params)
    out = cc._apply_colorbalance(out, params)
    out = cc._apply_saturation(out, params.saturation_mult)
    out = cc._apply_gamma(out, params.gamma)
    expected = np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8).reshape(frame.shape)

    assert np.array_equal(dev._grade(frame, params), expected)


def test_blue_band_boost_spares_red_and_saturates_blue():
    """A blue-band saturation boost measurably raises a blue patch while a red
    patch (sitting on a neutral anchor) is left effectively unchanged."""
    red = [1.0, 0.5, 0.5]
    blue = [0.5, 0.5, 1.0]
    rgb = np.array([red, blue], dtype=np.float32)
    # index 4 == blue anchor (240°)
    params = cc.CorrectionParams(hue_sat=(1.0, 1.0, 1.0, 1.0, 1.6, 1.0))

    out = cc._apply_hsl(rgb, params)

    assert np.allclose(out[0], rgb[0], atol=1e-6)  # red untouched
    assert _sat(out[1]) > _sat(rgb[1]) + 0.1  # blue clearly more saturated


def test_hue_shift_rotates_the_targeted_band():
    """A red-band hue shift moves a red patch toward orange (green channel up)
    without touching an out-of-band blue patch."""
    rgb = np.array([[1.0, 0.2, 0.2], [0.2, 0.2, 1.0]], dtype=np.float32)
    params = cc.CorrectionParams(hue_shift=(30.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    out = cc._apply_hsl(rgb, params)

    assert out[0, 1] > rgb[0, 1] + 0.05  # red's green channel rises (toward orange)
    assert np.allclose(out[1], rgb[1], atol=1e-6)  # blue untouched


def test_is_deterministic():
    rgb = np.random.default_rng(1).random((512, 3)).astype(np.float32)
    params = cc.CorrectionParams(hue_lum=(0.8, 1.0, 1.2, 1.0, 0.9, 1.1))

    assert np.array_equal(cc._apply_hsl(rgb, params), cc._apply_hsl(rgb, params))


def test_wrong_length_control_is_rejected():
    with pytest.raises(ValueError, match="length 6"):
        cc.CorrectionParams(hue_sat=(1.0, 1.0, 1.0))


def test_hsl_stage_feeds_the_develop_cache_version():
    """A retune of the HSL applier must invalidate the developed-photo cache."""
    assert cc._apply_hsl in dev._HASHED_SOURCES


# ── the seventh anchor: orange at 30° (skin and the golden-hour range) ──

# The shipped six-slot recipes, whose pixels must not change.
SIX_SLOT_LOOKS = [
    (name, tuple(knobs["hue_sat"]))
    for name, knobs in BUILTIN_LOOKS.items()
    if len(knobs.get("hue_sat", ())) == 6
]


def test_orange_band_moves_orange_and_spares_red_and_yellow():
    """A length-7 control boosts the ~30° band while red (0°) and yellow (60°)
    sit on neutral anchors — the band separation six anchors cannot express."""
    rgb = np.array([[1.0, 0.3, 0.3], [1.0, 0.65, 0.3], [1.0, 1.0, 0.3]], dtype=np.float32)
    seven = cc.CorrectionParams(hue_sat=(1.0, 1.6, 1.0, 1.0, 1.0, 1.0, 1.0))

    out = cc._apply_hsl(rgb, seven)

    assert np.allclose(out[0], rgb[0], atol=1e-6)  # red untouched
    assert _sat(out[1]) > _sat(rgb[1]) + 0.1  # orange clearly more saturated
    assert np.allclose(out[2], rgb[2], atol=1e-6)  # yellow untouched
    # Six anchors have no orange slot, so the same push has to drag red with it.
    six = cc._apply_hsl(rgb, cc.CorrectionParams(hue_sat=(1.6, 1.0, 1.0, 1.0, 1.0, 1.0)))
    assert _sat(six[0]) > _sat(rgb[0]) + 0.1


def test_length_six_controls_stay_bit_identical():
    """Every shipped six-slot recipe — and a six-slot shift/lum pair — renders
    byte-for-byte as it did before orange existed."""
    rgb = np.random.default_rng(7).random((512, 3)).astype(np.float32)

    assert len(SIX_SLOT_LOOKS) >= 8, "the shipped six-slot recipes are the regression bar"
    for name, band in SIX_SLOT_LOOKS:
        got = cc._apply_hsl(rgb, cc.CorrectionParams(hue_sat=band))
        assert np.array_equal(got, _six_anchor_reference(rgb, sat_ctrl=band)), name

    shift = (10.0, -5.0, 0.0, 3.0, -8.0, 2.0)
    lum = (0.9, 1.1, 1.0, 1.05, 0.95, 1.2)
    got = cc._apply_hsl(rgb, cc.CorrectionParams(hue_shift=shift, hue_lum=lum))
    assert np.array_equal(got, _six_anchor_reference(rgb, shift=shift, lum_ctrl=lum))


@pytest.mark.parametrize("length", [5, 8])
def test_illegal_lengths_are_refused_naming_both_legal_ones(length):
    """Neither door accepts anything but empty / 6 / 7, and both say so."""
    band = (1.0,) * length
    with pytest.raises(ValueError, match="6.*7|7.*6"):
        cc.CorrectionParams(hue_sat=band)
    with pytest.raises(ValueError, match="6.*7|7.*6"):
        _resolve_hue_band(band, [], "--hue-sat", 0.0, 5.0)


def test_seven_band_never_reaches_the_baked_lut(tmp_path):
    """``_apply_hsl`` stays excluded from ``params_to_lut3d_cube``, orange and all."""
    base = cc.CorrectionParams(saturation_mult=1.1)
    orange = cc.CorrectionParams(saturation_mult=1.1, hue_sat=(1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    a = cc.params_to_lut3d_cube(base, cube_size=17, output_path=tmp_path / "a.cube")
    b = cc.params_to_lut3d_cube(orange, cube_size=17, output_path=tmp_path / "b.cube")

    assert a.read_bytes() == b.read_bytes()
