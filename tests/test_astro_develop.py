"""TDD for astro_develop: "gradient fit recovers a
known polynomial, MTF lands background at target" — synthetic, seeded RNG."""

import numpy as np
import pytest

import astro_fixtures as fx
from gomoveshift.pipeline.photos import astro_develop as develop_mod
from gomoveshift.pipeline.photos.astro_develop import (
    BLACK_POINT_SIGMAS,
    GRADIENT_DEGREE_DEFAULT,
    STRETCH_TARGET_BACKGROUND,
    background_wb,
    black_point,
    develop_stack,
    develop_stack_with_params,
    mtf,
    remove_gradient,
    remove_gradient_detailed,
    solve_m,
    stretch,
)
from gomoveshift.pipeline.photos.astro_quality import (
    CONTRAST_RETENTION_MIN,
    broad_feature_contrast,
    contrast_retention,
)

H, W = 160, 192
LUMA = np.array([0.2126, 0.7152, 0.0722])


def test_gradient_removal_builds_the_full_frame_design_matrix_once(monkeypatch):
    rgb = np.full((40, 50, 3), 0.2, dtype=np.float32)
    real_design = develop_mod._design_matrix
    full_frame_calls = 0

    def counting_design(x, y, degree):
        nonlocal full_frame_calls
        if x.shape == rgb.shape[:2]:
            full_frame_calls += 1
        return real_design(x, y, degree)

    monkeypatch.setattr(develop_mod, "_design_matrix", counting_design)

    remove_gradient(rgb)

    assert full_frame_calls == 1


def _field(a, b, c, d=0.0, e=0.0, f=0.0):
    """A known degree-2 polynomial gradient over the full frame."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    xn, yn = x / (W - 1), y / (H - 1)
    return a + b * xn + c * yn + d * xn**2 + e * xn * yn + f * yn**2


def _stars(rng, n=25, amp=0.5, sigma=1.2):
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W))
    for y0, x0 in zip(rng.integers(5, H - 5, n), rng.integers(5, W - 5, n)):
        img += amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))
    return img


class TestRemoveGradient:
    def test_flattens_a_known_polynomial_gradient(self):
        rng = np.random.default_rng(1)
        rgb = np.stack(
            [
                _field(0.15, 0.10, 0.05, d=0.02) + _stars(rng),
                _field(0.15, -0.04, 0.08) + _stars(rng),
                _field(0.15, 0.06, -0.05, f=0.03) + _stars(rng),
            ],
            axis=-1,
        ).astype(np.float32)
        out = remove_gradient(rgb, degree=2)
        for c in range(3):
            corners = (out[:20, :20, c], out[-20:, -20:, c], out[:20, -20:, c], out[-20:, :20, c])
            medians = [float(np.median(patch)) for patch in corners]
            assert max(medians) - min(medians) < 0.02  # residual slope near zero

    def test_rejects_a_bright_extended_blob_instead_of_fitting_it(self):
        yy, xx = np.mgrid[0:H, 0:W]
        blob = 0.5 * np.exp(-((xx - 130) ** 2 + (yy - 40) ** 2) / (2 * 25.0**2))
        rgb = np.stack([_field(0.12, 0.03, 0.02) + blob] * 3, axis=-1).astype(np.float32)
        out = remove_gradient(rgb, degree=2)
        peak = float(out[40, 130, 0])
        background = float(np.median(out[:20, :20, 0]))
        assert peak - background > 0.35  # most of the injected 0.5 amplitude survives

    def test_differing_per_channel_gradients_come_out_neutral(self):
        stars = _stars(np.random.default_rng(2))  # same stars in every channel
        r = _field(0.15, 0.12, -0.06, d=0.05) + stars
        g = _field(0.15, -0.02, 0.03) + stars
        b = _field(0.15, 0.08, 0.10, f=-0.04) + stars
        before_cast = float(np.std((r - g)[10:-10, 10:-10]))
        rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
        out = remove_gradient(rgb, degree=2)
        after_cast = float(np.std((out[:, :, 0] - out[:, :, 1])[10:-10, 10:-10]))
        assert after_cast < before_cast * 0.25  # the spatial colour cast is gone

    def test_rejects_an_invalid_degree(self):
        with pytest.raises(ValueError):
            remove_gradient(np.zeros((8, 8, 3), dtype=np.float32), degree=5)


def _corner_local_bump(amplitude, background=0.12, sigma_frac=0.10):
    """A smooth bump sitting on each of the frame's four literal corners and decaying
    toward the centre -- a shape no degree-2 surface can follow (it has no basis term
    that's near-zero over most of the frame and large only at the four corners), and
    the sigma-rejecting fit (``GRADIENT_REJECT_ITERATIONS``) throws these bump-carrying
    grid cells out as an outlier "extended feature" exactly as it would a real corner
    aberration/vignetting residual a cropped centre never exposed the fit to."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    sigma_x, sigma_y = sigma_frac * W, sigma_frac * H
    bump = np.zeros((H, W))
    for cy in (0, H - 1):
        for cx in (0, W - 1):
            bump += amplitude * np.exp(-(((x - cx) / sigma_x) ** 2 + ((y - cy) / sigma_y) ** 2))
    return background + bump


class TestRemoveGradientDetailed:
    def test_unmodeled_corner_bump_reports_large_corner_residual_small_centre_residual(self):
        rng = np.random.default_rng(5)
        bump = _corner_local_bump(amplitude=1.0)
        rgb = np.stack([bump + _stars(rng)] * 3, axis=-1).astype(np.float32)
        _out, residual = remove_gradient_detailed(rgb, degree=2)
        assert residual.center is not None
        assert residual.corners is not None
        assert residual.corners > 5 * residual.center

    def test_flat_background_reports_near_zero_residual_everywhere(self):
        rgb = np.full((H, W, 3), 0.2, dtype=np.float32)
        _out, residual = remove_gradient_detailed(rgb, degree=2)
        assert residual.center is not None
        assert residual.corners is not None
        assert residual.center < 1e-6
        assert residual.corners < 1e-6

    def test_fully_masked_corner_region_reports_none_not_a_fabricated_zero(self):
        center_only_mask, _corners_mask = develop_mod._gradient_regions(H, W, None)
        rgb = np.full((H, W, 3), 0.2, dtype=np.float32)
        _out, residual = remove_gradient_detailed(rgb, degree=2, sky_mask=center_only_mask)
        assert residual.center is not None
        assert residual.corners is None

    def test_remove_gradient_and_remove_gradient_detailed_agree_on_the_array(self):
        rng = np.random.default_rng(6)
        rgb = np.stack([_field(0.15, 0.10, 0.05, d=0.02) + _stars(rng)] * 3, axis=-1).astype(
            np.float32
        )
        plain = remove_gradient(rgb, degree=2)
        detailed, _residual = remove_gradient_detailed(rgb, degree=2)
        np.testing.assert_array_equal(plain, detailed)


class TestMtf:
    def test_identity_at_m_half(self):
        x = np.linspace(0, 1, 11)
        assert np.allclose(mtf(x, 0.5), x)

    def test_endpoints_are_fixed_for_any_m(self):
        for m in (0.2, 0.4, 0.6, 0.8):
            assert mtf(0.0, m) == pytest.approx(0.0)
            assert mtf(1.0, m) == pytest.approx(1.0)


class TestSolveM:
    @pytest.mark.parametrize("b,t", [(0.05, 0.12), (0.2, 0.1), (0.3, 0.15), (0.02, 0.12)])
    def test_round_trips_through_mtf(self, b, t):
        assert float(mtf(b, solve_m(b, t))) == pytest.approx(t, abs=1e-6)


class TestStretch:
    def _linear_image(self):
        rng = np.random.default_rng(4)
        base = rng.random((H, W)) * 0.06 + 0.01
        yy, xx = np.mgrid[0:H, 0:W]
        for y0, x0 in zip(rng.integers(5, H - 5, 30), rng.integers(5, W - 5, 30)):
            base += 0.6 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 1.5**2))
        return np.clip(np.stack([base] * 3, axis=-1), 0, 1).astype(np.float32)

    def test_lands_the_luma_background_near_target(self):
        out = stretch(self._linear_image())
        background = float(np.median(out.astype(np.float64) @ LUMA)) / 65535.0
        assert abs(background - STRETCH_TARGET_BACKGROUND) < 0.02

    def test_is_monotonic_per_channel_and_uint16_full_range(self):
        ramp = np.linspace(0.0, 1.0, W)
        rgb = np.clip(np.tile(ramp, (H, 1))[:, :, None] * np.array([1.0, 0.8, 1.2]), 0, 1)
        out = stretch(rgb.astype(np.float32))
        assert out.dtype == np.uint16
        assert out.max() > 60000
        for c in range(3):
            assert np.all(np.diff(out[H // 2, :, c].astype(np.int64)) >= 0)


def _cast_background(rng, r_bg=0.08, g_bg=0.02, b_bg=0.02):
    """A red-heavy sky background: shared low-amplitude noise plus a
    deliberate per-channel offset (the colour cast ``background_wb`` must
    remove)."""
    noise = rng.random((H, W)) * 0.01
    rgb = np.stack([noise + r_bg, noise + g_bg, noise + b_bg], axis=-1)
    return rgb.astype(np.float32)


class TestBackgroundWb:
    def test_equalizes_a_red_heavy_background(self):
        rgb = _cast_background(np.random.default_rng(21))
        out, gains = background_wb(rgb)
        medians = [float(np.median(out[:, :, c])) for c in range(3)]
        assert max(medians) - min(medians) < 1e-3
        assert gains[0] < 1.0 < gains[1]  # red pulled down, green pulled up

    def test_a_bright_saturated_star_does_not_move_the_measured_background(self):
        base = _cast_background(np.random.default_rng(22))
        yy, xx = np.mgrid[0:H, 0:W]
        star = 5.0 * np.exp(-((xx - 96) ** 2 + (yy - 80) ** 2) / (2 * 1.0**2))
        starred = base.copy()
        starred[:, :, 0] += star.astype(np.float32)  # one very bright, very red star
        _, gains_base = background_wb(base)
        _, gains_starred = background_wb(starred)
        # the star's faint Gaussian wings touch a handful of pixels near
        # the median rank; the shift is negligible next to the >0.1 gain
        # spread the deliberate colour cast itself produces (below).
        assert gains_base == pytest.approx(gains_starred, abs=1e-3)

    def test_an_already_neutral_frame_is_unchanged(self):
        rng = np.random.default_rng(23)
        base = (rng.random((H, W)) * 0.05 + 0.05).astype(np.float32)
        rgb = np.stack([base, base, base], axis=-1)
        out, gains = background_wb(rgb)
        assert gains == pytest.approx((1.0, 1.0, 1.0), abs=1e-6)
        assert np.allclose(out, rgb, atol=1e-4)

    def test_a_zero_or_negative_background_does_not_raise_or_nan(self):
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[:, :, 0] -= 0.5  # this channel's median goes negative
        out, gains = background_wb(rgb)
        assert np.all(np.isfinite(out))
        assert not any(np.isnan(g) for g in gains)


class TestDevelopStack:
    def test_applies_background_wb_and_returns_a_valid_result(self):
        rgb = _cast_background(np.random.default_rng(24))
        result = develop_stack(rgb)
        assert result.rgb_uint16.dtype == np.uint16
        assert result.rgb_uint16.shape == (H, W, 3)
        assert max(result.background_gains) - min(result.background_gains) > 0.1
        assert result.peak > 0.0

        # the WB step, not just its recorded gains, actually reaches the
        # output: skipping it leaves a wider colour cast than keeping it.
        no_wb = stretch(remove_gradient(rgb))
        with_wb_meds = [float(np.median(result.rgb_uint16[:, :, c])) for c in range(3)]
        no_wb_meds = [float(np.median(no_wb[:, :, c])) for c in range(3)]
        with_wb_spread = max(with_wb_meds) - min(with_wb_meds)
        no_wb_spread = max(no_wb_meds) - min(no_wb_meds)
        assert with_wb_spread < no_wb_spread

    def test_records_the_gradient_fit_corner_residual(self):
        rng = np.random.default_rng(25)
        bump = _corner_local_bump(amplitude=1.0)
        rgb = np.stack([bump + _stars(rng)] * 3, axis=-1).astype(np.float32)
        result = develop_stack(rgb)
        assert result.gradient_residual is not None
        assert result.gradient_residual.center is not None
        assert result.gradient_residual.corners is not None
        assert result.gradient_residual.corners > result.gradient_residual.center


class TestSkyMaskedBlackPoint:
    """A bright ground silhouette in the WHOLE-FRAME luma inflates
    ``np.std`` and drags the black point up, crushing faint sky signal.
    ``sky_mask`` restricts the black point (and the stretch's own background
    read) to sky samples only."""

    def test_black_point_uses_only_sky_mask_samples(self):
        h, w = 96, 96
        frame, truth = fx.sky_frame(
            shape=(h, w),
            background=0.02,
            noise_sigma=0.002,
            ground_profile_frac=np.full(w, 0.7, dtype=np.float32),
            ground_level=0.9,  # a lit building / bright horizon glow
            seed=5,
        )
        sky_mask = ~truth["ground_mask"]
        luma = frame.astype(np.float64) @ LUMA
        expected = float(
            max(np.median(luma[sky_mask]) - BLACK_POINT_SIGMAS * np.std(luma[sky_mask]), 0.0)
        )
        assert black_point(frame, sky_mask=sky_mask) == pytest.approx(expected, rel=1e-9)
        # the bright ground materially moves the whole-frame statistic --
        # proving the mask isn't a no-op.
        whole_frame = black_point(frame)
        assert whole_frame != pytest.approx(expected, rel=0.02)

    def test_sky_mask_none_reproduces_the_original_whole_frame_formula_exactly(self):
        rng = np.random.default_rng(9)
        frame = np.clip(0.05 + 0.01 * rng.standard_normal((64, 64, 3)), 0.0, None).astype(
            np.float32
        )
        luma = frame.astype(np.float64) @ LUMA
        expected = float(max(np.median(luma) - BLACK_POINT_SIGMAS * np.std(luma), 0.0))
        assert black_point(frame) == pytest.approx(expected, rel=1e-9)
        assert black_point(frame, sky_mask=None) == pytest.approx(expected, rel=1e-9)

    def test_a_degenerate_sky_mask_raises_or_warns_rather_than_reverting_to_whole_frame(self):
        """Falling back to the whole frame would silently restore the very statistic the
        mask exists to remove, and ``_fit_gradient`` already raises on the same input.
        Measured on a flat 0.05 frame with 0.01-sigma noise (32x32): a 4-pixel mask reads
        a black point of 0.0400 against the whole frame's 0.0288 -- 39% high, silently."""
        rng = np.random.default_rng(45)
        frame = (0.05 + 0.01 * rng.standard_normal((32, 32, 3))).astype(np.float32)
        with pytest.raises(ValueError, match="no sky pixels"):
            black_point(frame, sky_mask=np.zeros((32, 32), dtype=bool))
        thin = np.zeros((32, 32), dtype=bool)
        thin[0, :4] = True
        with pytest.warns(RuntimeWarning, match="too few sky pixels"):
            assert np.isfinite(black_point(frame, sky_mask=thin))

    @pytest.mark.parametrize("ground_frac", [0.3, 0.5, 0.8])
    def test_develop_stack_threads_sky_mask_into_the_stretch_and_recovers_faint_nebulosity(
        self, ground_frac
    ):
        """A ground silhouette covering ``ground_frac`` of the frame (boundary row at
        ``1 - ground_frac`` of height), brighter than the sky, contaminates the WHOLE-FRAME
        luma median/std, so the whole-frame black point misplaces itself against the sky's
        own faint nebulosity (0.04 contrast over a 0.02 background); the sky-only one
        (formula below) leaves headroom. Measured end-to-end on this exact fixture at
        ground_frac 0.3/0.5/0.8: unmasked broad_feature_contrast 0.044/0.065/-0.009, masked
        0.404/0.403/0.335 -- masked wins at every fraction (what is asserted), while an
        absolute ``abs(unmasked) < 0.05`` premise would hold only at 0.8."""
        h, w = 160, 120
        frame, truth = fx.sky_frame(
            shape=(h, w),
            background=0.02,
            gradient_amplitude=0.0,
            nebulosity_contrast=0.04,
            nebulosity_center=(0.10, 0.5),
            nebulosity_radius_frac=0.10,
            ground_profile_frac=np.full(w, 1.0 - ground_frac, dtype=np.float32),
            ground_level=0.05,
            noise_sigma=0.0005,
            seed=3,
        )
        sky_mask = ~truth["ground_mask"]
        neb_mask = truth["nebulosity_mask"] & sky_mask
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = 0.10 * h, 0.5 * w
        radius = 0.10 * min(h, w)
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        core_mask = neb_mask & (dist < radius * 0.6)
        ring_mask = sky_mask & (dist > radius) & (dist < radius + 6)

        unmasked = develop_stack(frame)
        masked = develop_stack(frame, sky_mask=sky_mask)

        # mechanism proof: the recorded black point must equal the sky-only formula
        # applied to the SAME gradient-removed, white-balanced array develop_stack
        # itself produces -- not a tuned threshold.
        degraded_expected = remove_gradient(frame, GRADIENT_DEGREE_DEFAULT, sky_mask)
        balanced_expected, _ = background_wb(degraded_expected)
        luma_expected = balanced_expected.astype(np.float64) @ LUMA
        expected_bp = float(
            max(
                np.median(luma_expected[sky_mask])
                - BLACK_POINT_SIGMAS * np.std(luma_expected[sky_mask]),
                0.0,
            )
        )
        assert masked.black_point == pytest.approx(expected_bp, rel=1e-6)
        # not a no-op; the whole-frame value errs LOW at small ground fractions and HIGH
        # at large (measured on this fixture: 0.0122/0.0140/0.0270 whole-frame against
        # 0.0190/0.0190/0.0185 sky-only), so pin the magnitude rather than the sign.
        assert masked.black_point != pytest.approx(unmasked.black_point, rel=0.2)

        # same mechanism proof for the stretch's OWN background read, not only the
        # black point -- a mask that reaches black_point but not the per-step
        # background target is still whole-frame-contaminated.
        shifted_expected = np.clip(balanced_expected.astype(np.float64) - expected_bp, 0.0, None)
        peak_expected = float(shifted_expected.max())
        working_expected = (
            shifted_expected / peak_expected if peak_expected > 0 else shifted_expected
        )
        expected_before = float(np.median((working_expected @ LUMA)[sky_mask]))
        assert masked.background_before == pytest.approx(expected_before, rel=1e-6)

        unmasked_contrast = broad_feature_contrast(
            unmasked.rgb_uint16.astype(np.float64), core_mask, ring_mask
        )
        masked_contrast = broad_feature_contrast(
            masked.rgb_uint16.astype(np.float64), core_mask, ring_mask
        )
        assert np.isfinite(unmasked_contrast) and np.isfinite(masked_contrast)
        assert masked_contrast > unmasked_contrast  # measured above, at every fraction
        assert masked_contrast > 0.25  # recovered: measured 0.335-0.404
        # the ratio above RISES as the frame collapses toward black (a whole-frame per-step
        # background read scored 0.4167 on this fixture at ground_frac 0.8 while the render
        # went near-black), so also pin the absolute level the stretch exists to place.
        assert abs(masked.background_after - STRETCH_TARGET_BACKGROUND) < 0.02

    def test_default_develop_stack_output_is_unchanged_by_the_new_optional_argument(self):
        """The ``sky_mask=None`` path must stay byte-for-byte the pre-existing
        whole-frame output -- this is a shipped render path."""
        rgb = _cast_background(np.random.default_rng(41))
        omitted = develop_stack(rgb)
        explicit_none = develop_stack(rgb, sky_mask=None)
        np.testing.assert_array_equal(omitted.rgb_uint16, explicit_none.rgb_uint16)
        assert omitted.black_point == pytest.approx(explicit_none.black_point, rel=1e-12)


class TestDevelopStackWithParams:
    """The ``astro_meteors`` replay path: a meteor layer must share base's tone
    exactly, never re-derive its own."""

    def test_replaying_the_base_reproduces_its_own_output_exactly(self):
        rgb = _cast_background(np.random.default_rng(31))
        base = develop_stack(rgb)

        replayed = develop_stack_with_params(rgb, base)

        np.testing.assert_array_equal(replayed, base.rgb_uint16)

    def test_a_masked_base_replays_byte_for_byte_without_the_caller_repeating_the_mask(self):
        """The meteor path replays layers against a base now developed under a sky mask.
        The gradient fit is the one step ``develop_stack`` masks and this replay re-runs,
        so the mask must travel ON the base: fitting the replay whole-frame against a
        masked base is off by up to 20173 of 65535 on this fixture (160x192, sky = the top
        half) while every recorded scalar still matches."""
        rgb = _cast_background(np.random.default_rng(34))
        sky_mask = np.zeros((H, W), dtype=bool)
        sky_mask[: H // 2] = True
        base = develop_stack(rgb, sky_mask=sky_mask)
        np.testing.assert_array_equal(base.sky_mask, sky_mask)
        np.testing.assert_array_equal(develop_stack_with_params(rgb, base), base.rgb_uint16)

    def test_a_different_layer_gets_its_own_gradient_but_bases_recorded_tone(self):
        rng = np.random.default_rng(32)
        base_rgb = np.clip(
            np.stack([_field(0.10, 0.02, 0.01) + _stars(rng, n=20)] * 3, axis=-1), 0.0, 1.0
        ).astype(np.float32)
        base = develop_stack(base_rgb)
        # a differently-exposed, differently-gradiented "meteor frame" -- if this
        # recomputed its own black point/peak/m instead of reusing base's, the
        # hand-derived expectation below (built from base's OWN recorded numbers)
        # would not match.
        rng2 = np.random.default_rng(33)
        layer = np.clip(
            np.stack(
                [_field(0.14, -0.05, 0.06, d=0.03) + _stars(rng2, n=15, amp=0.4)] * 3, axis=-1
            ),
            0.0,
            1.0,
        ).astype(np.float32)

        replayed = develop_stack_with_params(layer, base)

        degraded = remove_gradient(layer)
        balanced = (degraded.astype(np.float64) * np.array(base.background_gains)).astype(
            np.float32
        )
        shifted = np.clip(balanced.astype(np.float64) - base.black_point, 0.0, None)
        working = shifted / base.peak if base.peak > 0 else shifted
        for m in base.m_values:
            working = mtf(working, m)
        expected = (np.clip(working, 0.0, 1.0) * 65535).round().astype(np.uint16)
        assert expected.max() > 0  # not a degenerate all-clipped-to-zero expectation
        np.testing.assert_array_equal(replayed, expected)


def test_gradient_removal_preserves_each_channel_level():
    """The pedestal added back must be the fitted surface's MEAN, not its value at
    the (0, 0) corner. The basis is x/y normalised to [0, 1] rather than centred, so
    ``coeffs[0]`` is the corner — the DARKEST point of a vignetted night frame — and
    adding it back while subtracting a whole surface pushes most of the frame below
    zero. Downstream that is silent and total: ``background_wb`` sees a non-positive
    median, falls back to gains of exactly 1.0 (no white balance at all), and the
    stretch clips the negatives to black. Measured on a real 5-frame stack, red and
    green went 70% and 58% negative and the render came out monochrome blue.
    """
    h, w = 64, 96
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    # A vignette: brightest mid-frame, falling off to every corner.
    vignette = 0.03 - 0.02 * (((x / (w - 1)) - 0.5) ** 2 + ((y / (h - 1)) - 0.5) ** 2)
    rgb = np.repeat(vignette[:, :, None], 3, axis=2).astype(np.float32)

    out = remove_gradient(rgb)

    for c in range(3):
        assert out[..., c].mean() == pytest.approx(rgb[..., c].mean(), rel=1e-3), (
            f"channel {c} level not preserved"
        )
        assert (out[..., c] < 0).mean() < 0.01, f"channel {c} driven negative"


class TestMaskedBackgroundFit:
    """``sky_mask`` excludes pixels from the fit; ``core_mask``/
    ``local_ring`` are LOCAL so retained-contrast isn't gradient-diluted."""

    SHAPE = (128, 128)
    RADIUS_FRAC = 0.35

    def _fixture(self, nebulosity_contrast, gradient_amplitude=0.02):
        return fx.sky_frame(
            shape=self.SHAPE,
            background=0.02,
            gradient_amplitude=gradient_amplitude,
            gradient_angle_deg=30.0,
            nebulosity_contrast=nebulosity_contrast,
            nebulosity_center=(0.5, 0.5),
            nebulosity_radius_frac=self.RADIUS_FRAC,
            noise_sigma=0.0,
            seed=0,
        )

    def _masks(self):
        h, w = self.SHAPE
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
        radius = self.RADIUS_FRAC * min(h, w)
        core_mask = dist < 0.08 * min(h, w)  # nebula core: near-peak profile
        local_ring = (dist > radius) & (dist < radius + 4)  # just outside the patch
        return core_mask, local_ring

    def _grad_spread(self, img):
        corners = (img[:20, :20, 0], img[-20:, -20:, 0], img[:20, -20:, 0], img[-20:, :20, 0])
        medians = [float(np.median(c)) for c in corners]
        return max(medians) - min(medians)

    @pytest.mark.parametrize("nebulosity_contrast", [0.03, 0.01])
    def test_masked_path_beats_unmasked_and_clears_the_gate(self, nebulosity_contrast):
        """The documented primary contract: regenerate the fixture with
        ``gradient_amplitude=0.0`` (everything else identical) and use THAT as
        ``reference`` -- no hand-reconstructed nuisance array to drift from the
        fixture's real formula again. Measured (128x128, contrast 0.03/0.01): masked
        retention ~1.005-1.009 -- it CLEARS ``CONTRAST_RETENTION_MIN``, reversing the
        earlier broken-reconstruction verdict. Bounded both sides: a bare ``>= MIN``
        would also pass at 3x over-correction."""
        frame, truth = self._fixture(nebulosity_contrast)
        clean, _ = self._fixture(nebulosity_contrast, gradient_amplitude=0.0)
        sky_mask = ~truth["nebulosity_mask"]
        core_mask, local_ring = self._masks()
        out_masked = remove_gradient(frame, degree=GRADIENT_DEGREE_DEFAULT, sky_mask=sky_mask)
        out_unmasked = remove_gradient(frame, degree=GRADIENT_DEGREE_DEFAULT)
        ret_masked = contrast_retention(clean, out_masked, core_mask, local_ring)
        ret_unmasked = contrast_retention(clean, out_unmasked, core_mask, local_ring)
        assert CONTRAST_RETENTION_MIN <= ret_masked < 1.2  # own measured ~1.01
        assert ret_masked > ret_unmasked  # masking still measurably helps
        grad_before, grad_after = self._grad_spread(frame), self._grad_spread(out_masked)
        assert grad_after < grad_before * 0.05  # the nuisance gradient is materially gone

    def test_masked_out_ground_and_bright_sources_never_tilt_the_fitted_surface(self):
        """A bright ground light entirely inside the excluded region can't move the fit."""
        frame, truth = fx.sky_frame(
            shape=self.SHAPE,
            background=0.02,
            gradient_amplitude=0.02,
            gradient_angle_deg=30.0,
            ground_profile_frac=np.full(self.SHAPE[1], 0.85, dtype=np.float32),
            ground_level=0.01,
            seed=0,
        )
        ground_mask = truth["ground_mask"]
        sky_mask = ~ground_mask
        yy, xx = np.mgrid[0 : self.SHAPE[0], 0 : self.SHAPE[1]]
        ground_light = 3.0 * np.exp(-((xx - 64) ** 2 + (yy - 118) ** 2) / (2 * 4.0**2))
        ground_light = ground_light * ground_mask  # confined to the excluded band
        lit = frame + ground_light.astype(np.float32)[:, :, None]

        out_clean = remove_gradient(frame, degree=GRADIENT_DEGREE_DEFAULT, sky_mask=sky_mask)
        out_lit = remove_gradient(lit, degree=GRADIENT_DEGREE_DEFAULT, sky_mask=sky_mask)

        np.testing.assert_array_equal(out_clean[sky_mask], out_lit[sky_mask])

    def test_default_sky_mask_reproduces_todays_whole_frame_output_exactly(self):
        """Omitting ``sky_mask`` and passing ``None`` must be the SAME code path."""
        rng = np.random.default_rng(7)
        rgb = np.clip(0.03 + 0.005 * rng.standard_normal((*self.SHAPE, 3)), 0.0, None).astype(
            np.float32
        )
        omitted = remove_gradient(rgb, degree=2)
        explicit_none = remove_gradient(rgb, degree=2, sky_mask=None)
        np.testing.assert_array_equal(omitted, explicit_none)

        develop_omitted = develop_stack(rgb)
        develop_none = develop_stack(rgb, sky_mask=None)
        np.testing.assert_array_equal(develop_omitted.rgb_uint16, develop_none.rgb_uint16)

        # below the term count, unmasked skips the fallback (a no-op would equal input).
        for shape, degree in [((1, 5), 2), ((2, 7), 4)]:
            h, w = shape
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
            ramp = 0.02 + 0.01 * (xx / max(w - 1, 1)) + 0.015 * (yy / max(h - 1, 1))
            tiny = np.repeat(ramp[:, :, None], 3, axis=2).astype(np.float32)
            assert not np.array_equal(remove_gradient(tiny, degree=degree), tiny)

    def test_leaky_mask_boundary_still_flattens_the_gradient(self):
        """Regression: one-sided left 1.36% residual on a 4-row leak; two-sided none."""
        h, w = self.SHAPE
        frame = np.full((h, w, 3), 0.05, dtype=np.float32)
        frame += (0.02 * (np.arange(w) / w)).astype(np.float32)[None, :, None]
        frame[100:, :, :] = 0.002  # dark foreground below the true horizon
        leaky_mask = np.ones((h, w), dtype=bool)
        leaky_mask[104:, :] = False  # true boundary is row 100 -- 4 dark rows leak in
        out = remove_gradient(frame, sky_mask=leaky_mask)
        sky = out[:90, :, 0]
        assert float(sky.max() - sky.min()) < 0.001


def test_fit_gradient_guards_underdetermined_and_empty_masks():
    """Too few samples falls back to a no-op and WARNS; an all-excluded mask raises; a
    count just above n_terms stays finite; enough samples but too few distinct grid
    ROWS (ill-conditioned, not underdetermined by count) also falls back and warns."""
    rng = np.random.default_rng(0)
    base = (0.05 + 0.02 * rng.standard_normal((128, 128))).astype(np.float32)
    rgb = np.repeat(base[:, :, None], 3, axis=2)
    tiny_mask = np.zeros((128, 128), dtype=bool)
    tiny_mask[:15, :15] = True  # 4 grid-cell samples survive; degree 4 needs 15 terms
    with pytest.warns(RuntimeWarning, match="too few samples"):
        out = remove_gradient(rgb, degree=4, sky_mask=tiny_mask)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, rgb, atol=1e-5)
    with pytest.raises(ValueError, match="no sample cells"):
        remove_gradient(rgb, sky_mask=np.zeros((128, 128), dtype=bool))
    with pytest.raises(ValueError, match="!= frame shape"):
        remove_gradient(rgb, sky_mask=np.zeros((64, 64), dtype=bool))

    h, w = 200, 260
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    gradient = 0.02 + 0.02 * (x / (w - 1)) + 0.015 * (y / (h - 1))
    frame = (gradient + 0.004 * rng.standard_normal((h, w))).astype(np.float32)
    rgb2 = np.repeat(frame[:, :, None], 3, axis=2)
    corner_mask = np.zeros((h, w), dtype=bool)
    corner_mask[:30, :40] = True  # 9 cells, below 3*n_terms=18 -- constant-fallback no-op
    with pytest.warns(RuntimeWarning, match="too few samples"):
        out3 = remove_gradient(rgb2, degree=2, sky_mask=corner_mask)
    np.testing.assert_allclose(out3, rgb2, atol=1e-5)

    # 32 samples clears the 3*n_terms=18 count guard but spans only 2 of 16 grid rows
    # (measured condition number ~1.5e18, far past CONDITION_NUMBER_MAX) -- count alone
    # would have let this through to a real, non-no-op fit off a near-singular system.
    row_band_mask = np.zeros((h, w), dtype=bool)
    row_band_mask[:20, :] = True
    with pytest.warns(RuntimeWarning, match="ill-conditioned"):
        out2 = remove_gradient(rgb2, degree=2, sky_mask=row_band_mask)
    np.testing.assert_allclose(out2, rgb2, atol=1e-5)


# --- chroma-only denoise ----------------------------------------------------
def _speckled_sky(seed=7):
    """A flat sky carrying LUMA detail (stars, identical in all three channels)
    and independent per-channel CHROMA noise -- the two things the colour step
    has to tell apart. Measured on a real master, clear sky holds Lab chroma sigma a=1.33 b=1.71 against luma sigma 8.15:
    the speckle is colour, the detail is brightness."""
    rng = np.random.default_rng(seed)
    h = w = 128
    img = np.full((h, w, 3), 0.30, dtype=np.float64)
    img += rng.normal(0.0, 0.010, (h, w, 1))  # luma: same in all channels
    img += rng.normal(0.0, 0.014, (h, w, 3))  # chroma: independent per channel
    for y, x in [(30, 40), (70, 90), (100, 20)]:
        img[y - 1 : y + 2, x - 1 : x + 2] += 0.35  # stars
    return np.clip(img, 0.0, 1.0)


def _chroma_sigma(rgb_uint16):
    f = np.asarray(rgb_uint16, dtype=np.float64) / 65535.0
    return float((f - f.mean(axis=2, keepdims=True)).std())


def _luma_sigma(rgb_uint16):
    f = np.asarray(rgb_uint16, dtype=np.float64) / 65535.0
    return float(f.mean(axis=2).std())


def test_chroma_denoise_takes_the_colour_speckle_and_leaves_the_stars():
    noisy = (_speckled_sky() * 65535).round().astype(np.uint16)
    cleaned = develop_mod.denoise_chroma(noisy)
    assert cleaned.dtype == np.uint16 and cleaned.shape == noisy.shape
    assert _chroma_sigma(cleaned) < 0.5 * _chroma_sigma(noisy), "colour speckle must go"
    kept = _luma_sigma(cleaned) / _luma_sigma(noisy)
    assert 0.95 <= kept <= 1.05, f"luma detail must survive, kept {kept:.3f}"


def test_chroma_sigma_reads_colour_speckle_and_ignores_brightness():
    neutral = np.full((32, 32, 3), 20000, dtype=np.uint16)
    assert develop_mod.chroma_sigma(neutral) == pytest.approx(0.0, abs=1e-12)
    # Twice as bright, same neutrality: still zero. It measures colour, not level.
    assert develop_mod.chroma_sigma(neutral * 2) == pytest.approx(0.0, abs=1e-12)
    speckled = (_speckled_sky() * 65535).round().astype(np.uint16)
    assert develop_mod.chroma_sigma(speckled) > 0.005


def test_develop_stack_leaves_the_master_invertible():
    """The colour blur belongs to the WRITER, not to develop_stack: every step
    develop_stack takes is reversible from the numbers it records, which is what
    astro_contact.recover_balanced walks back. A blur is not reversible, so
    develop_stack must not have applied one."""
    rgb = _speckled_sky(seed=3).astype(np.float32)
    result = develop_mod.develop_stack(rgb)
    assert develop_mod.chroma_sigma(result.rgb_uint16) > develop_mod.chroma_sigma(
        develop_mod.denoise_chroma(result.rgb_uint16)
    )


def test_replaying_a_develop_still_reproduces_it_byte_for_byte():
    """``develop_stack_with_params``' standing promise, which the colour step
    must not break: it is a deterministic function of the image alone, so it
    replays identically."""
    rgb = _speckled_sky(seed=11).astype(np.float32)
    base = develop_mod.develop_stack(rgb)
    assert np.array_equal(develop_mod.develop_stack_with_params(rgb, base), base.rgb_uint16)


# --- the viewable preview beside the master ---------------------------------
SKY_ROWS = slice(0, 60)  # above the treeline in ``_master_like``


def _master_like():
    """A developed master's shape, built to the percentiles of a real one.

    Measured on a real developed master: luma p25 0.091, p50 0.120, p75 0.151, p90 0.175, p99.8 0.255. The two
    features that matter are both consequences of the master ALREADY carrying the
    MTF stretch: the sky sits at the archival 0.12 target rather than near zero, and
    the brightest star cores are only about twice the sky, not five or ten times it.
    A fixture with a near-black sky and blazing stars is an undeveloped frame, and a
    preview tuned against one is tuned against the wrong picture.
    """
    rng = np.random.default_rng(5)
    h, w = 96, 128
    band = np.linspace(0.09, 0.16, w)[None, :, None] * np.ones((h, 1, 3))
    sky = band + rng.normal(0.0, 0.02, (h, w, 3))
    sky[70:, :, :] *= 0.15  # treeline
    for y, x in [(20, 30), (40, 80), (15, 110)]:
        sky[y - 1 : y + 2, x - 1 : x + 2] += 0.13
    return (np.clip(sky, 0, 1) * 65535).round().astype(np.uint16)


def test_the_preview_is_brighter_and_punchier_than_the_archival_master():
    master = _master_like()
    view = develop_mod.preview_render(master)
    assert view.dtype == np.uint8 and view.shape == master.shape
    as_u8 = (master / 257).astype(np.uint8)
    assert view[SKY_ROWS].mean() > as_u8[SKY_ROWS].mean(), "the sky must come up off the floor"
    assert view[SKY_ROWS].std() > as_u8[SKY_ROWS].std(), "and carry more separation, not less"


def test_the_preview_separates_the_band_from_the_sky_floor_without_blowing_stars():
    """The property a percentile black point can actually deliver.

    An earlier version of this test demanded under 2% of the sky clip to black.
    That is unsatisfiable by construction for any percentile black point: cutting at
    the Nth percentile clips exactly N% of the frame, so only a black point down at
    p1 could pass -- and p1 on an already-stretched master double-lifts the sky to a
    washed 180/255 grey with no picture in it. What a viewer actually needs is
    SEPARATION: the bright side of the band clearly above the faint side, the star
    cores still below saturation.
    """
    view = develop_mod.preview_render(_master_like())
    lum = view[SKY_ROWS].mean(axis=2)
    faint, bright = np.percentile(lum, 10), np.percentile(lum, 90)
    assert bright - faint > 40, "the band must stand off the sky floor, not sit flat on it"
    assert bright < 250, "and the bright side must not be pinned at the top of the range"
    assert (view == 255).mean() < 0.02, "blown stars hide the picture the other way"


def test_rendering_a_preview_does_not_touch_the_master():
    master = _master_like()
    before = master.copy()
    develop_mod.preview_render(master)
    assert np.array_equal(master, before)
