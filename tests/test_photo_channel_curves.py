"""Tests for the operator-drawn tone curves in the photo develop chain.

``_apply_channel_curves`` is a photo-develop-only look: an independent ADDITIONAL
monotone tone curve per R/G/B channel, riding ``_grade`` right after the master
curve. Default-OFF (and a y=x identity curve) must be bit-identical; a classic
cross-process move — red midtones down, blue midtones up — must shift a mid-gray
patch cooler while pinning pure white and pure black. It is deliberately NOT
wired into ``params_to_lut3d_cube``.
"""

from dataclasses import replace

import numpy as np
import pytest

from gomoveshift.config import check_layers
from gomoveshift.pipeline import color_correction as cc
from gomoveshift.pipeline.photos import develop as dev


def test_default_off_is_bit_identical():
    """Empty channel curves leave every pixel untouched; a y=x identity curve is
    numerically indistinguishable from a pass-through."""
    rng = np.random.default_rng(0)
    rgb = rng.random((256, 3)).astype(np.float32)

    assert np.array_equal(cc._apply_channel_curves(rgb, cc.CorrectionParams()), rgb)

    identity = cc.CorrectionParams(
        curve_r=(0.0, 0.0, 1.0, 1.0),
        curve_g=(0.0, 0.0, 1.0, 1.0),
        curve_b=(0.0, 0.0, 1.0, 1.0),
    )
    assert np.allclose(cc._apply_channel_curves(rgb, identity), rgb, atol=1e-6)


def test_grade_default_off_matches_the_chain_without_channel_curves():
    """The default photo grade is byte-identical to the pre-channel-curve chain."""
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


def test_cross_process_shifts_midgray_cooler_and_pins_endpoints():
    """Red midtones down + blue midtones up cools a mid-gray patch, while pure
    white and pure black stay anchored (curves pass through their endpoints)."""
    rgb = np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    params = cc.CorrectionParams(
        curve_r=(0.0, 0.0, 0.5, 0.35, 1.0, 1.0),  # red midtones pulled DOWN
        curve_b=(0.0, 0.0, 0.5, 0.65, 1.0, 1.0),  # blue midtones pushed UP
    )

    out = cc._apply_channel_curves(rgb, params)

    # Mid-gray: red drops, blue rises, green (no curve) untouched → clearly cooler.
    assert out[0, 0] < rgb[0, 0] - 0.1
    assert out[0, 2] > rgb[0, 2] + 0.1
    assert out[0, 1] == pytest.approx(rgb[0, 1])
    assert out[0, 2] - out[0, 0] > 0.2  # blue now well above red
    # Endpoints anchored on every channel.
    assert np.allclose(out[1], rgb[1], atol=1e-6)  # pure white
    assert np.allclose(out[2], rgb[2], atol=1e-6)  # pure black


def test_channels_without_a_curve_pass_through():
    """A red-only curve leaves the untouched green/blue channels exactly equal."""
    rng = np.random.default_rng(2)
    rgb = rng.random((128, 3)).astype(np.float32)
    params = cc.CorrectionParams(curve_r=(0.0, 0.1, 1.0, 0.9))

    out = cc._apply_channel_curves(rgb, params)

    assert np.array_equal(out[:, 1], rgb[:, 1])
    assert np.array_equal(out[:, 2], rgb[:, 2])
    assert not np.array_equal(out[:, 0], rgb[:, 0])


def test_is_deterministic():
    rgb = np.random.default_rng(1).random((512, 3)).astype(np.float32)
    params = cc.CorrectionParams(curve_g=(0.0, 0.05, 0.5, 0.4, 1.0, 0.95))

    assert np.array_equal(
        cc._apply_channel_curves(rgb, params), cc._apply_channel_curves(rgb, params)
    )


def test_odd_length_control_is_rejected():
    with pytest.raises(ValueError, match="even-length"):
        cc.CorrectionParams(curve_r=(0.0, 0.0, 1.0))


def test_single_point_control_is_rejected():
    with pytest.raises(ValueError, match=r"2"):
        cc.CorrectionParams(curve_b=(0.0, 0.0))


def test_non_monotone_x_is_rejected():
    with pytest.raises(ValueError, match="ascending"):
        cc.CorrectionParams(curve_g=(0.0, 0.0, 0.5, 0.4, 0.5, 0.9))


def test_channel_curves_feed_the_develop_cache_version():
    """A retune of the channel-curve applier must invalidate the developed cache."""
    assert cc._apply_channel_curves in dev._HASHED_SOURCES


# ── curve_l: the MASTER tone curve the operator actually draws ───────
#
# The parametric skeleton ``_apply_curves`` builds is fixed at four knots; the
# only master an operator could draw was three channel curves carrying one
# knot list (what ``styles.py``'s matte presets hand-roll). ``curve_l`` is that
# move as one field, run through the same interpolant, before the per-channel
# curves ride on top of it. Knots below are shaped for WHERE THE CURVE RUNS —
# early in ``_grade``, where a real graded frame occupies roughly [0, 0.6].

#: A lifted toe: shadows up, decaying to identity by the top of the occupied range.
TOE_LIFT = (0.0, 0.10, 0.15, 0.23, 0.35, 0.40, 0.6, 0.62, 1.0, 1.0)
#: A contrast S inside that same occupied band — lows down, upper mids up.
CONTRAST_S = (0.0, 0.0, 0.15, 0.12, 0.3, 0.27, 0.45, 0.48, 0.6, 0.66, 1.0, 1.0)


def test_master_curve_lifts_the_toe_with_no_channel_separation():
    """A lifted-toe ``curve_l`` raises shadow luma, and a neutral patch comes back
    neutral on every channel — one curve on all three means no cast, by
    construction (the property the three-channel workaround buys by hand)."""
    rgb = np.array([[0.05, 0.05, 0.05], [0.2, 0.2, 0.2], [0.5, 0.5, 0.5]], dtype=np.float32)

    out = cc._apply_luma_curve(rgb, cc.CorrectionParams(curve_l=TOE_LIFT))

    assert out[0, 0] > rgb[0, 0] + 0.05, "the toe must actually lift"
    assert out[1, 0] > rgb[1, 0], "the low mids ride the lift too"
    for row in out:
        assert row[0] == row[1] == row[2], "a master curve may not separate the channels"


def test_master_curve_equals_the_same_knots_on_all_three_channel_curves():
    """THE equivalence that makes the hand-rolled workaround redundant: one
    ``curve_l`` is bit-for-bit the same render as that knot list duplicated onto
    ``curve_r``/``curve_g``/``curve_b``."""
    rgb = np.random.default_rng(7).random((512, 3)).astype(np.float32) * 0.6

    master = cc._apply_luma_curve(rgb, cc.CorrectionParams(curve_l=CONTRAST_S))
    tripled = cc._apply_channel_curves(
        rgb, cc.CorrectionParams(curve_r=CONTRAST_S, curve_g=CONTRAST_S, curve_b=CONTRAST_S)
    )

    assert np.array_equal(master, tripled)


def test_empty_master_curve_is_bit_identical_and_never_reaches_a_baked_lut(tmp_path):
    """Empty is OFF by construction, and — like every photo-only applier —
    ``curve_l`` sits outside ``params_to_lut3d_cube``, so video output and every
    baked LUT stay byte-for-byte what they were."""
    rgb = np.random.default_rng(3).random((256, 3)).astype(np.float32)
    assert np.array_equal(cc._apply_luma_curve(rgb, cc.CorrectionParams()), rgb)

    base = cc.CorrectionParams(shadow_lift=6, contrast_curve=0.04, saturation_mult=1.1)
    plain = cc.params_to_lut3d_cube(base, cube_size=8, output_path=tmp_path / "plain.cube")
    curved = cc.params_to_lut3d_cube(
        replace(base, curve_l=CONTRAST_S), cube_size=8, output_path=tmp_path / "curved.cube"
    )
    assert plain.read_bytes() == curved.read_bytes()


def test_master_curve_runs_under_the_channel_curves_in_the_grade():
    """The documented order: the master tone shape lands FIRST, then the
    cross-process channel curves ride on top of it — the same relationship
    ``_apply_channel_curves`` already has with the parametric ``_apply_curves``."""
    frame = np.linspace(0, 150, 32 * 24 * 3, dtype=np.uint8).reshape(24, 32, 3)
    params = cc.CorrectionParams(curve_l=CONTRAST_S, curve_r=(0.0, 0.0, 0.3, 0.2, 1.0, 1.0))

    flat = frame.reshape(-1, 3).astype(np.float32) / 255.0
    out = cc._apply_curves(flat, params)
    out = cc._apply_luma_curve(out, params)
    out = cc._apply_channel_curves(out, params)
    out = cc._apply_colorbalance(out, params)
    out = cc._apply_saturation(out, params.saturation_mult)
    out = cc._apply_gamma(out, params.gamma)
    expected = np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8).reshape(frame.shape)

    assert np.array_equal(dev._grade(frame, params), expected)


@pytest.mark.parametrize(
    "pts,message",
    [
        ((0.0, 0.0, 1.0), "even-length"),
        ((0.0, 0.5), "2"),
        ((0.0, 0.0, 0.5, 0.4, 0.5, 0.9), "ascending"),
    ],
)
def test_a_malformed_master_curve_is_refused_at_the_door(pts, message):
    """Refused where the params are built — before any decode, not mid-batch."""
    with pytest.raises(ValueError, match=message):
        cc.CorrectionParams(curve_l=pts)


def test_a_look_recipe_and_a_masked_layer_can_both_draw_one():
    """Reachable from the two operator surfaces a colour knob has: a named look's
    own knob record, and one masked layer riding a region of the frame."""
    look = dev.look_variant_options({"curve_l": list(TOE_LIFT)}, dev.DevelopOptions())
    assert look.curve_l == TOE_LIFT
    assert dev._look_params(cc.CorrectionParams(), look).curve_l == TOE_LIFT

    check_layers("--layers", [{"mask": "sky", "curve_l": list(TOE_LIFT)}])
    frame = np.full((48, 64, 3), 90, dtype=np.uint8)
    layered = dev._layered(frame, ({"mask": "linear", "curve_l": TOE_LIFT},))
    assert not np.array_equal(layered, frame), "a masked layer drew no tone move"


def test_the_master_curve_feeds_the_develop_cache_version():
    """A retune of the master-curve applier must invalidate the developed cache."""
    assert cc._apply_luma_curve in dev._HASHED_SOURCES
