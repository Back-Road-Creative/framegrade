"""Black-point tone move in the shared grade chain (photo styles only).

``CorrectionParams.black_point`` is a signed 8-bit-levels move applied as the
final step of ``_apply_curves``: POSITIVE lifts the output floor (matte fade,
``out = bp/255 + (1 - bp/255) * out``), NEGATIVE cuts the input floor
(crushed blacks, ``out = clip((out - |bp|/255) / (1 - |bp|/255))``). The video
pipeline never sets the field, so anything under the no-op threshold must
leave the output BIT-IDENTICAL — the LUT baker runs this same applier.
"""

from dataclasses import fields, replace

import numpy as np
import pytest

from gomoveshift.pipeline.color_correction import CorrectionParams, _apply_curves
from gomoveshift.pipeline.photos import develop as develop_mod


def _rgb(*values):
    """(N, 3) float64 gray pixels in [0, 1] — one row per value."""
    return np.array([[v, v, v] for v in values], dtype=np.float64)


# ── Fade (positive: lift the output floor) ───────────────────────────


def test_fade_lifts_black_to_exact_floor():
    out = _apply_curves(_rgb(0.0, 1.0), CorrectionParams(black_point=10.0))
    np.testing.assert_array_equal(out[0], np.full(3, 10.0 / 255.0))
    assert out[1] == pytest.approx(1.0, abs=1e-12)


def test_fade_is_monotone_in_midtones():
    out = _apply_curves(_rgb(0.25, 0.5), CorrectionParams(black_point=10.0))
    floor = 10.0 / 255.0
    assert out[0, 0] == pytest.approx(floor + (1.0 - floor) * 0.25)
    assert out[1, 0] == pytest.approx(floor + (1.0 - floor) * 0.5)


# ── Crush (negative: cut the input floor) ────────────────────────────


def test_crush_maps_cut_level_to_zero_and_keeps_white():
    cut = 8.0 / 255.0
    out = _apply_curves(_rgb(cut, 0.5, 1.0), CorrectionParams(black_point=-8.0))
    np.testing.assert_array_equal(out[0], np.zeros(3))
    assert out[1, 0] == pytest.approx((0.5 - cut) / (1.0 - cut))
    np.testing.assert_array_equal(out[2], np.ones(3))


def test_crush_clips_below_cut_to_zero():
    out = _apply_curves(_rgb(3.0 / 255.0), CorrectionParams(black_point=-8.0))
    np.testing.assert_array_equal(out[0], np.zeros(3))


# ── Exact no-op under the threshold (video bit-identity) ─────────────


def test_zero_and_subthreshold_black_point_are_bitwise_noops():
    rng = np.random.default_rng(7)
    rgb = rng.random((256, 3), dtype=np.float64)
    baseline = _apply_curves(rgb, CorrectionParams())
    for bp in (0.0, 0.4, -0.4):
        out = _apply_curves(rgb, CorrectionParams(black_point=bp))
        assert np.array_equal(out, baseline), f"black_point={bp} must be an exact no-op"


def test_subthreshold_noop_holds_with_active_curve_params():
    rng = np.random.default_rng(11)
    rgb = rng.random((256, 3), dtype=np.float64)
    curve = CorrectionParams(shadow_lift=8.0, highlight_pull=6.0, contrast_curve=0.05)
    baseline = _apply_curves(rgb, curve)
    out = _apply_curves(rgb, replace(curve, black_point=0.4))
    assert np.array_equal(out, baseline)


def test_float32_input_keeps_float32_dtype():
    rgb = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(-1, 3)
    out = _apply_curves(rgb, CorrectionParams(black_point=10.0))
    assert out.dtype == np.float32


# ── Clamp to the documented authority ─────────────────────────────────


def test_clamp_constants_document_the_authority():
    from gomoveshift.pipeline.color_correction import BLACK_POINT_MAX, BLACK_POINT_MIN

    assert BLACK_POINT_MIN == -10.0
    assert BLACK_POINT_MAX == 20.0


def test_over_authority_values_apply_at_the_rails():
    rgb = _rgb(0.0, 10.0 / 255.0, 0.5, 1.0)
    lifted = _apply_curves(rgb, CorrectionParams(black_point=50.0))
    np.testing.assert_array_equal(lifted, _apply_curves(rgb, CorrectionParams(black_point=20.0)))
    np.testing.assert_array_equal(lifted[0], np.full(3, 20.0 / 255.0))
    crushed = _apply_curves(rgb, CorrectionParams(black_point=-30.0))
    np.testing.assert_array_equal(crushed, _apply_curves(rgb, CorrectionParams(black_point=-10.0)))
    np.testing.assert_array_equal(crushed[1], np.zeros(3))


# ── Distance normalization (variant distinctness) ─────────────────────


def test_params_distance_full_authority_black_point_scores_one():
    a = CorrectionParams(black_point=-10.0)
    assert develop_mod._params_distance(a, replace(a, black_point=20.0)) == pytest.approx(1.0)


def test_params_distance_black_point_scales_linearly():
    a = CorrectionParams()
    one = develop_mod._params_distance(a, replace(a, black_point=3.0))
    two = develop_mod._params_distance(a, replace(a, black_point=6.0))
    assert one > 0.0
    assert two == pytest.approx(2.0 * one)


# ── Dataclass contract ────────────────────────────────────────────────


def test_default_round_trips():
    p = CorrectionParams()
    assert "black_point" in {f.name for f in fields(p)}
    assert p.black_point == 0.0
    q = replace(p, black_point=12.0)
    assert q.black_point == 12.0
    assert p.black_point == 0.0
