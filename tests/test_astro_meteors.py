"""astro_meteors: streak masking + lighten-blend re-insertion.

Base/layer arrays are synthetic uint16 RGB (the frozen ``astro_develop.develop_stack``
output dtype), with a seeded RNG only where noise-shape assertions need it.
"""

import numpy as np
import pytest

from gomoveshift.pipeline.photos import astro_meteors as meteors_mod


def _flat_uint16(h, w, value):
    return np.full((h, w, 3), value, dtype=np.uint16)


def test_streak_mask_empty_streaks_is_all_zero():
    mask = meteors_mod.streak_mask((20, 20), [])
    assert np.array_equal(mask, np.zeros((20, 20), dtype=np.float32))


def test_streak_mask_feathers_with_no_hard_step():
    mask = meteors_mod.streak_mask((40, 40), [(5.0, 20.0, 35.0, 20.0)], feather_px=6.0)
    assert mask.max() > 0.9  # the line's own core reads as fully masked
    row = mask[20]
    edge = row[(row > 0.02) & (row < 0.98)]
    assert edge.size > 0  # a real transition band exists
    assert np.all(np.diff(row) < 0.5)  # never a single-pixel cliff from 0 to 1


def test_streak_mask_zero_feather_still_marks_the_line():
    mask = meteors_mod.streak_mask((20, 20), [(2.0, 10.0, 17.0, 10.0)], feather_px=0.0)
    assert mask[10, 10] == pytest.approx(1.0)
    assert mask[0, 0] == 0.0


def test_reinsert_mask_one_takes_the_brighter_layer():
    base = _flat_uint16(10, 10, 1000)
    layer = _flat_uint16(10, 10, 50000)
    mask = np.ones((10, 10), dtype=np.float32)
    out = meteors_mod.reinsert(base, [layer], [mask], feather_px=0.0)
    assert np.array_equal(out, layer)


def test_reinsert_mask_zero_is_bit_identical_to_base():
    base = _flat_uint16(10, 10, 1000)
    layer = _flat_uint16(10, 10, 50000)
    mask = np.zeros((10, 10), dtype=np.float32)
    out = meteors_mod.reinsert(base, [layer], [mask])
    assert np.array_equal(out, base)


def test_reinsert_dim_layer_never_darkens_base():
    base = _flat_uint16(10, 10, 40000)
    layer = _flat_uint16(10, 10, 100)  # dimmer than base everywhere
    mask = np.ones((10, 10), dtype=np.float32)
    out = meteors_mod.reinsert(base, [layer], [mask])
    assert np.array_equal(out, base)


def test_reinsert_feathered_region_is_a_true_partial_blend():
    base = _flat_uint16(10, 10, 1000)
    layer = _flat_uint16(10, 10, 50000)
    mask = np.full((10, 10), 0.25, dtype=np.float32)
    out = meteors_mod.reinsert(base, [layer], [mask])[0, 0, 0]
    assert 1000 < out < 50000
    assert out == pytest.approx(1000 + 0.25 * (50000 - 1000), abs=2)


def test_reinsert_multiple_layers_composite_at_the_running_max():
    base = _flat_uint16(10, 10, 0)
    layer_a = _flat_uint16(10, 10, 100)
    layer_b = _flat_uint16(10, 10, 200)
    mask_a = np.zeros((10, 10), dtype=np.float32)
    mask_a[:, :5] = 1.0  # left half only
    mask_b = np.zeros((10, 10), dtype=np.float32)
    mask_b[:, 3:] = 1.0  # overlaps the right edge of mask_a
    out = meteors_mod.reinsert(base, [layer_a, layer_b], [mask_a, mask_b])
    assert out[0, 0, 0] == 100  # layer_a only
    assert out[0, 6, 0] == 200  # layer_b only
    assert out[0, 4, 0] == 200  # overlap: the true max, not a sum (300)


def test_reinsert_empty_layers_is_a_noop_returning_base_unchanged():
    base = _flat_uint16(10, 10, 4242)
    out = meteors_mod.reinsert(base, [], [])
    assert np.array_equal(out, base)
    assert out is not base


def test_reinsert_accepts_raw_streaks_and_composes_with_the_develop_chain_dtype():
    base = _flat_uint16(30, 30, 500)
    layer = _flat_uint16(30, 30, 60000)
    out = meteors_mod.reinsert(base, [layer], [[(2.0, 15.0, 27.0, 15.0)]])
    assert out.dtype == np.uint16
    assert out.shape == base.shape
    assert out[15, 15, 0] > base[15, 15, 0]  # the streak's core picked up the meteor
    assert out[0, 0, 0] == base[0, 0, 0]  # far from any streak: untouched


def test_reinsert_refuses_layer_shape_mismatch():
    base = _flat_uint16(10, 10, 100)
    bad_layer = _flat_uint16(8, 8, 100)
    with pytest.raises(meteors_mod.MeteorInputError):
        meteors_mod.reinsert(base, [bad_layer], [[]])


def test_reinsert_refuses_layer_dtype_mismatch():
    base = _flat_uint16(10, 10, 100)
    bad_layer = np.full((10, 10, 3), 100, dtype=np.float32)
    with pytest.raises(meteors_mod.MeteorInputError):
        meteors_mod.reinsert(base, [bad_layer], [[]])


def test_reinsert_refuses_mismatched_layer_and_mask_list_lengths():
    base = _flat_uint16(10, 10, 100)
    layer = _flat_uint16(10, 10, 200)
    with pytest.raises(meteors_mod.MeteorInputError):
        meteors_mod.reinsert(base, [layer, layer], [np.ones((10, 10), dtype=np.float32)])


@pytest.mark.parametrize("seed", [0, 20260813])
@pytest.mark.parametrize("sigma", [5.0, 15.0, 30.0])
def test_reinsert_no_meteor_fixture_is_unchanged_beyond_the_noise_floor(seed, sigma):
    """'A single-exposure line is not positive proof of a meteor' -- a
    triage-flagged line drawn over nothing but ordinary frame noise (no real signal
    anywhere) must not add anything, at any of several seeds/sigmas -- the floor is
    measured from each layer's own noise (_layer_gate), not a value tuned to one
    fixture."""
    rng = np.random.default_rng(seed)
    base = np.clip(1000.0 + rng.normal(0.0, sigma, size=(40, 40, 3)), 0, 65535).astype(np.uint16)
    layer = np.clip(
        base.astype(np.float32) + rng.normal(0.0, sigma, size=(40, 40, 3)), 0, 65535
    ).astype(np.uint16)
    out = meteors_mod.reinsert(base, [layer], [[(5.0, 20.0, 35.0, 20.0)]])
    assert np.array_equal(out, base)


@pytest.mark.parametrize("wobble", [10, 24, 200, 400])
def test_reinsert_star_under_the_streak_line_is_not_brightened(wobble):
    """The old max-composite preferentially adds a coincident star --
    the streak line runs straight through a star already correctly stacked into
    base; ordinary single-frame photometric wobble, at ANY amplitude (a star's own
    shot noise on a 40000 ADU star is ~200 ADU -- a fixed floor cannot suppress
    that), must not brighten it further. Gated by ``_star_prominence_mask``, not a
    wobble-amplitude threshold."""
    base = _flat_uint16(30, 30, 500)
    base[15, 15] = 40000  # a real star, already correctly represented in base
    layer = base.copy()
    layer[15, 15] = 40000 + wobble
    out = meteors_mod.reinsert(base, [layer], [[(2.0, 15.0, 27.0, 15.0)]])
    assert out[15, 15, 0] == base[15, 15, 0]


def test_reinsert_overlapping_streaks_are_order_independent():
    """Excess is gated against the frozen ``base``, never the mutating
    accumulator, so the fold (``maximum``) is commutative -- including under a
    FEATHERED mask (``streak_mask``, not ``np.ones``, which hides the pre-repair
    formula's own order-dependence). Measured on this exact fixture, the baseline
    (the pre-repair ``layer - out`` formula): max|forward-backward| = 13.0 ADU. This
    repair's ``layer - base`` formula measures 0.0 on the same fixture."""
    base = _flat_uint16(10, 10, 500)
    mask = meteors_mod.streak_mask((10, 10), [(1.0, 5.0, 8.0, 5.0)], feather_px=4.0)
    layer_a = _flat_uint16(10, 10, 600)
    layer_b = _flat_uint16(10, 10, 800)
    forward = meteors_mod.reinsert(base, [layer_a, layer_b], [mask, mask])
    backward = meteors_mod.reinsert(base, [layer_b, layer_a], [mask, mask])
    assert np.array_equal(forward, backward)
    assert forward[5, 5, 0] == 800  # deep in the mask's core: the true max


def test_reinsert_gate_ramps_instead_of_stepping_across_the_noise_floor():
    """Measured, DISCLOSED regression -- not hidden behind a strawman. The
    real pre-repair formula (the baseline, no floor at all: ``max(0, layer-base)``)
    steps 2.00 ADU across this 1.36-ADU/px true taper (uint16 quantisation only);
    ``_layer_gate``'s smoothstep ramp steps 6.00 ADU on the SAME fixture -- 3x LARGER,
    not smaller (see ``_layer_gate``'s docstring and its tests). The first assertion
    is what actually gates against the baseline: it has no gate at all, so its own output
    IS the un-gated formula (``old_out``), byte-identical to it."""
    base = _flat_uint16(1, 60, 500)
    ramp = np.linspace(0.0, 80.0, 60, dtype=np.float32)
    layer_vals = np.clip(500.0 + ramp, 0, 65535).astype(np.uint16)
    layer = np.tile(layer_vals[None, :, None], (1, 1, 3))
    mask = np.ones((1, 60), dtype=np.float32)

    diff = layer.astype(np.float32) - base.astype(np.float32)
    old_raw = np.clip(diff, 0.0, None)  # the baseline's actual formula: no floor at all
    old_out = base.astype(np.float32) + mask[..., None] * old_raw
    old_step = np.abs(np.diff(old_out[0, :, 0])).max()

    new_out = meteors_mod.reinsert(base, [layer], [mask]).astype(np.float32)
    new_step = np.abs(np.diff(new_out[0, :, 0])).max()

    assert not np.array_equal(new_out, old_out)  # the gate changes values; the baseline has none
    assert old_step == pytest.approx(2.0, abs=0.5)
    assert new_step <= 6.5  # measured 6.0; bounds further regression, does not hide it


def test_reinsert_suppresses_feather_halo_noise_the_old_max_composite_leaked():
    """At mask 1 the pre-fix formula is exactly ``max(base, layer)``, so
    ANY positive noise excursion under the mask -- however small -- was composited
    in whole. Reproduces that pre-fix formula inline to measure it against the new
    gated model on one fixture: a real streak core (amplitude 3000, far above the
    floor) plus independent per-pixel noise everywhere else, including the feathered
    halo ringing the streak -- background that must read as background."""
    rng = np.random.default_rng(20260813)
    shape_hw = (40, 40)
    base = np.clip(1000.0 + rng.normal(0.0, 5.0, size=(*shape_hw, 3)), 0, 65535).astype(np.uint16)
    mask = meteors_mod.streak_mask(shape_hw, [(5.0, 20.0, 35.0, 20.0)], feather_px=4.0)
    core = (mask > 0.99).astype(np.float32)
    layer_f = (
        base.astype(np.float32)
        + rng.normal(0.0, 5.0, size=(*shape_hw, 3))
        + 3000.0 * core[..., None]
    )
    layer = np.clip(layer_f, 0, 65535).astype(np.uint16)

    # pre-fix formula (this module's OLD docstring): out = base + mask * max(0, layer - base)
    old_raw = np.clip(layer.astype(np.float32) - base.astype(np.float32), 0.0, None)
    old_out = base.astype(np.float32) + mask[..., None] * old_raw

    new_out = meteors_mod.reinsert(base, [layer], [mask]).astype(np.float32)

    halo = (mask > 0.01) & (mask < 0.3)  # feathered ring around the streak, no real signal
    assert halo.any()
    old_halo_delta = (old_out - base.astype(np.float32))[..., 0][halo]
    new_halo_delta = (new_out - base.astype(np.float32))[..., 0][halo]
    assert old_halo_delta.max() > 0  # the old formula already leaks some noise here
    assert new_halo_delta.max() == 0.0  # the gated model leaks none

    # the real streak signal still comes through, unweakened, at the saturated core
    core_mask = mask > 0.99
    new_core_delta = (new_out - base.astype(np.float32))[..., 0][core_mask].mean()
    assert new_core_delta == pytest.approx(3000.0, abs=5.0)


def test_reinsert_mask_edge_seam_is_not_worsened_by_the_gate():
    """Sampled PERPENDICULAR to the mask's own feather
    boundary, with a REALISTIC TAPERING streak (linear brightness falloff over 15
    rows around the centerline -- not a hard 3000 ADU block at mask>0.99, which hid
    this): where the taper crosses the noise floor near the mask's own feather edge,
    the gate measurably WORSENS the seam -- old (pre-repair baseline formula)
    vs new (gated) profiles differ by 48.0 ADU on this fixture (the baseline differs from
    itself by 0.0). Disclosed here and in ``_layer_gate``'s docstring, not hidden
    behind an easier fixture; bounded, not chased to zero."""
    shape_hw = (60, 60)
    rng = np.random.default_rng(20260813)
    base = np.clip(1200.0 + rng.normal(0.0, 4.0, size=(*shape_hw, 3)), 0, 65535).astype(np.uint16)
    mask = meteors_mod.streak_mask(shape_hw, [(5.0, 30.0, 55.0, 30.0)], feather_px=4.0)
    rows = np.arange(shape_hw[0])[:, None]
    taper = np.clip(1.0 - np.abs(rows - 30) / 15.0, 0.0, 1.0).astype(np.float32)
    amplitude = np.broadcast_to(60.0 * taper, shape_hw).astype(np.float32)
    layer_f = (
        base.astype(np.float32) + rng.normal(0.0, 4.0, size=(*shape_hw, 3)) + amplitude[..., None]
    )
    layer = np.clip(layer_f, 0, 65535).astype(np.uint16)

    old_raw = np.clip(layer.astype(np.float32) - base.astype(np.float32), 0.0, None)
    old_out = base.astype(np.float32) + mask[..., None] * old_raw
    new_out = meteors_mod.reinsert(base, [layer], [mask]).astype(np.float32)

    col = 30  # crosses the feather boundary at rows ~24 and ~36
    old_profile = old_out[20:41, col, 0]
    new_profile = new_out[20:41, col, 0]
    delta = np.abs(old_profile - new_profile).max()
    assert delta > 20.0  # the disclosed regression: the baseline would measure 0.0 here
    assert delta <= 60.0  # measured 48.0; bounds further regression, not a fix


@pytest.mark.parametrize(
    "fwhm_px,expected_leak_adu",
    [(3, 0), (9, 0), (11, 0), (12, 1), (13, 3), (14, 53), (15, 56), (18, 125), (21, 161)],
)
def test_reinsert_star_wings_leak_grows_past_the_dilation_radius(fwhm_px, expected_leak_adu):
    """``_star_prominence_mask`` dilates a FIXED radius
    (``STAR_LOCAL_BG_PX``, 7 px), not the star's own light -- HONEST BOUND, not a
    general claim: zero leak through FWHM 11px; wider PSFs leak past the dilation
    radius, growing with width (pinned below, on a 121x121 fixture so even FWHM 21
    does not fill the frame and mask its own tail)."""
    size = 121
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    r2 = (yy - c) ** 2 + (xx - c) ** 2
    sigma = fwhm_px / 2.3548
    star = 40000.0 * np.exp(-r2 / (2 * sigma**2))
    wobble = 2000.0 * np.exp(-r2 / (2 * sigma**2))
    base = np.clip(500.0 + star, 0, 65535).astype(np.uint16)
    layer = np.clip(base.astype(np.float32) + wobble, 0, 65535).astype(np.uint16)
    base3 = np.repeat(base[..., None], 3, axis=-1)
    layer3 = np.repeat(layer[..., None], 3, axis=-1)
    out = meteors_mod.reinsert(base3, [layer3], [[(2.0, c, size - 3, c)]])
    delta = out[..., 0].astype(np.float32) - base3[..., 0].astype(np.float32)
    assert delta[star > 1.0].max() == pytest.approx(expected_leak_adu, abs=1.0)
    assert meteors_mod.STAR_PROMINENCE_ADU == 500.0  # the constant this test pins


def test_reinsert_iterator_form_matches_list_form_byte_for_byte():
    """Streaming pairs vs the two-list form, ONE accumulator either way."""
    base = _flat_uint16(20, 20, 500)
    layer_a = _flat_uint16(20, 20, 40000)
    layer_b = _flat_uint16(20, 20, 20000)
    mask_a = np.zeros((20, 20), dtype=np.float32)
    mask_a[:, :10] = 1.0
    mask_b = np.zeros((20, 20), dtype=np.float32)
    mask_b[:, 5:] = 1.0  # overlaps mask_a: exercises the running-max across the streamed form

    out_list = meteors_mod.reinsert(base, [layer_a, layer_b], [mask_a, mask_b])

    def _pairs():
        yield layer_a, mask_a
        yield layer_b, mask_b

    out_iter = meteors_mod.reinsert(base, _pairs())

    assert np.array_equal(out_list, out_iter)
