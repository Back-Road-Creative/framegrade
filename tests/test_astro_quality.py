"""Pure-math tests for astro_quality's metric helpers -- no other astro_* module involved,
except TestA3ByRadialThird, whose subject reuses astro_triage's star detector/shape
measurer (see astro_quality.a3_by_radial_third's docstring)."""

import zlib

import numpy as np
import pytest

from gomoveshift.pipeline.photos.astro_quality import (
    A3_RADIAL_GROWTH_BOUNDARY,
    a3_by_radial_third,
    background_noise,
    boundary_seam,
    broad_feature_contrast,
    contrast_retention,
    coverage_stats,
    false_rejection_rate,
    noise_improvement,
    stellar_color_ratios,
)

_GAUSSIAN_SIGMA_TO_FWHM = 2.3548200450309493  # 2*sqrt(2*ln 2) -- same convention as astro_triage.


def _det_seed(*parts):  # deterministic across runs -- hash() of a str is PYTHONHASHSEED-salted.
    return zlib.crc32(repr(parts).encode()) % (2**32)


def _patched(shape, feature_slice, bg, fg):
    image = np.full(shape, bg, dtype=np.float64)
    feature_mask = np.zeros(shape, dtype=bool)
    feature_mask[feature_slice] = True
    image[feature_mask] = fg
    return image, feature_mask, ~feature_mask


def _gradient_field(shape, kind):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    x, y = xx / (w - 1), yy / (h - 1)
    if kind == "ramp":
        return 0.30 * x + 0.20 * y
    if kind == "dome":
        return 0.45 * np.exp(-(((y - 0.1) ** 2 + (x - 0.5) ** 2) / 0.35))
    if kind == "bowl":  # radial bowl -- null-space shape, see class docstrings below
        return 0.40 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)
    if kind == "glow":  # centred gaussian glow
        return 0.40 * np.exp(-(((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.03))
    if kind == "saddle":
        return 0.30 + 0.60 * ((x - 0.5) ** 2 - (y - 0.5) ** 2)
    return 0.35 * (1.0 - ((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.5)  # vignette


def _oracle_masks(shape, geometry):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy, r = {
        "centred_small": (w / 2, h / 2, 12.0),
        "centred_large": (w / 2, h / 2, 40.0),
        "edge_crossing": (0.0, h / 2, 30.0),
        "corner": (0.0, 0.0, 30.0),
    }[geometry]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    feature_mask = dist <= r
    background_mask = (dist >= 1.6 * r) & (dist <= 2.2 * r)
    return feature_mask, background_mask


class TestAstroQualityMetrics:
    def test_broad_feature_contrast_known_amplitude_outliers_and_degenerate_cases(self):
        image, feature, background = _patched((6, 6), np.s_[2:4, 2:4], 10.0, 15.0)
        assert broad_feature_contrast(image, feature, background) == pytest.approx(0.5)
        image, feature, background = _patched((3, 3), np.s_[0, 0], 10.0, 20.0)
        image[1, 1] = 10_000.0  # one wild outlier -- median of the other 8 stays 10.0
        assert broad_feature_contrast(image, feature, background) == pytest.approx(1.0)
        flat, ones3, zeros3 = np.full((3, 3), 10.0), np.ones((3, 3), bool), np.zeros((3, 3), bool)
        feature_only = np.arange(9).reshape(3, 3) == 0  # True only at [0, 0]
        zero_bg = feature_only * 5.0  # 0 everywhere except [0, 0] == 5.0
        empty_arr, empty_mask = np.zeros((0, 0)), np.zeros((0, 0), bool)
        assert broad_feature_contrast(flat, zeros3, ones3) == 0.0  # empty feature mask
        assert broad_feature_contrast(flat, ones3, zeros3) == 0.0  # empty background mask
        assert broad_feature_contrast(empty_arr, empty_mask, empty_mask) == 0.0  # empty image
        assert broad_feature_contrast(zero_bg, feature_only, ~feature_only) == 0.0  # bg median 0

    def test_contrast_retention_known_ratios_and_zero_reference(self):
        image, feature, background = _patched((6, 6), np.s_[2:4, 2:4], 10.0, 15.0)
        assert contrast_retention(image, image, feature, background) == pytest.approx(1.0)
        reference, feature, background = _patched((6, 6), np.s_[2:4, 2:4], 10.0, 20.0)
        after, _, _ = _patched((6, 6), np.s_[2:4, 2:4], 10.0, 19.0)
        # reference contrast 1.0, after contrast 0.9 -> ratio 0.9
        assert contrast_retention(reference, after, feature, background) == pytest.approx(0.9)
        flat = np.full((3, 3), 10.0)
        ones3 = np.ones((3, 3), bool)
        assert contrast_retention(flat, flat, ones3, ones3) == 0.0  # reference contrast is 0
        after_nonzero, feat3, bg3 = _patched((3, 3), np.s_[0, 0], 10.0, 20.0)
        # DISJOINT masks, so `after` really does carry contrast (1.0) under them -- with
        # feature == background every image measures 0.0 and the case below is vacuous.
        assert broad_feature_contrast(after_nonzero, feat3, bg3) == pytest.approx(1.0)
        # Reference contrast is still 0, so retention fails CLOSED to 0.0 rather than
        # handing back after's own contrast as if it had been retained. This pins
        # PRE-EXISTING behaviour (the earlier implementation returned 0.0 here too) -- it is a
        # regression guard for the rewrite around it, not proof of new behaviour.
        assert contrast_retention(flat, after_nonzero, feat3, bg3) == 0.0

    def test_background_noise_hand_built_gaussian_and_degenerate_sigma(self):
        # median 14, abs deviations [4,2,0,2,4] -> median 2 -> sigma 1.4826*2
        image = np.array([[10, 12, 14, 16, 18]], dtype=np.float64)
        ones5 = np.ones_like(image, dtype=bool)
        assert background_noise(image, ones5) == pytest.approx(1.4826 * 2.0)
        rng = np.random.default_rng(20260813)
        gaussian = rng.normal(loc=100.0, scale=5.0, size=(200, 200))
        ones_g = np.ones_like(gaussian, dtype=bool)
        assert background_noise(gaussian, ones_g) == pytest.approx(5.0, rel=0.1)
        assert background_noise(np.full((3, 3), 10.0), np.zeros((3, 3), bool)) == 0.0
        assert background_noise(np.zeros((0, 0)), np.zeros((0, 0), bool)) == 0.0

    def test_noise_improvement_known_zero_n_and_zero_stacked_noise_cases(self):
        cases = [
            (10.0, 5.0, 4, 2.0, 2.0, 1.0),  # exact ideal
            (9.0, 6.0, 4, 1.5, 2.0, 0.75),  # below ideal
            (10.0, 5.0, 0, 2.0, 0.0, 0.0),  # n <= 0 -> no ideal curve to compare against
        ]
        for single, stacked, n, measured, ideal, ratio in cases:
            result = noise_improvement(single, stacked, n)
            assert result.measured_factor == pytest.approx(measured)
            assert result.ideal_factor == pytest.approx(ideal)
            assert result.ratio_to_ideal == pytest.approx(ratio)
        # stacked_noise == 0 floors the denominator instead of dividing by zero
        assert np.isfinite(noise_improvement(10.0, 0.0, 4).measured_factor)
        assert noise_improvement(10.0, 0.0, 4).measured_factor > 1e6

    def test_false_rejection_rate_exact_k_over_total_and_empty(self):
        mask = np.zeros((5, 4), dtype=bool)
        mask[0, 0] = mask[3, 0] = mask[1, 1] = True  # k=3; pixel cols 0 (x2) and 1
        result = false_rejection_rate(mask)
        assert result.rate == pytest.approx(3.0 / 20.0)
        assert result.any_rejected_fraction == pytest.approx(2.0 / 4.0)  # cols 0,1 of 4
        empty = false_rejection_rate(np.zeros((0, 4), dtype=bool))
        assert (empty.rate, empty.any_rejected_fraction) == (0.0, 0.0)

    def test_stellar_color_ratios_valid_flags_spread_and_degenerate_cases(self):
        rgb = np.zeros((10, 10, 3), dtype=np.float64)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 2.0, 4.0, 1.0
        rgb[7:, 7:, 0] = 6.0  # a differently-colored patch for the "genuine drift" case
        uniform_xy = np.array([[2.0, 2.0], [5.0, 5.0], [2.0, 5.0], [-100.0, -100.0]])
        result = stellar_color_ratios(rgb, uniform_xy, aperture_px=1.5)
        assert list(result.valid) == [True, True, True, False]
        assert result.rg_ratios[:3] == pytest.approx([0.5, 0.5, 0.5])
        assert result.bg_ratios[0] == pytest.approx(0.25)
        assert result.rg_ratios[3] == 0.0  # off-frame: unmeasured, not "0 ratio"
        # 3 identical, color-stable on-frame stars + 1 off-frame star: spread must
        # track color stability, never star placement.
        assert result.rg_spread == 0.0 and result.bg_spread == 0.0
        drifted = stellar_color_ratios(rgb, np.array([[2.0, 2.0], [8.0, 8.0]]), aperture_px=1.5)
        assert drifted.rg_ratios == pytest.approx([0.5, 1.5])
        assert drifted.rg_spread == pytest.approx(0.5)  # std([0.5, 1.5]) -- genuine drift
        empty = stellar_color_ratios(rgb, np.zeros((0, 2)), aperture_px=2.0)
        assert empty.rg_ratios.shape == empty.valid.shape == (0,) and empty.rg_spread == 0.0
        no_green = np.zeros((10, 10, 3), dtype=np.float64)
        no_green[..., 0] = 5.0  # G stays 0
        unmeasured = stellar_color_ratios(no_green, uniform_xy[:1], aperture_px=1.5)
        assert unmeasured.rg_ratios[0] == 0.0 and not unmeasured.valid[0]

    def test_coverage_stats_hand_built_and_empty_cases(self):
        result = coverage_stats(np.array([1, 2, 3, 4, 5]), floor=3)
        assert result.min_count == 1
        assert result.median_count == pytest.approx(3.0)
        assert result.below_floor_fraction == pytest.approx(2.0 / 5.0)
        empty = coverage_stats(np.zeros((0,), dtype=int), floor=3)
        assert (empty.min_count, empty.median_count, empty.below_floor_fraction) == (0, 0.0, 0.0)

    def test_boundary_seam_background_sigma_units_identical_degenerate_and_floor_cases(self):
        def band(step):  # MAD=1 on EACH side (sigma=1.4826) regardless of step -- must
            image = np.zeros((4, 4))  # track background noise, never the saturating
            image[0, :], image[1, :] = 9.0, 11.0  # combined-band spread.
            image[2, :], image[3, :] = 9.0 + step, 11.0 + step
            return image

        mask = np.arange(16).reshape(4, 4) < 8  # True for rows 0-1
        assert boundary_seam(band(3 * 1.4826), mask) == pytest.approx(3.0)
        assert boundary_seam(band(10 * 1.4826), mask) == pytest.approx(10.0)
        assert boundary_seam(band(10 * 1.4826), mask) > boundary_seam(band(3 * 1.4826), mask)
        assert boundary_seam(np.full((4, 4), 10.0), mask) == 0.0  # identical sides
        for degenerate_mask in [
            np.ones((4, 4), dtype=bool),
            np.zeros((4, 4), dtype=bool),
            np.zeros((0, 0), dtype=bool),
        ]:
            assert boundary_seam(np.full(degenerate_mask.shape, 10.0), degenerate_mask) == 0.0
        # majority value 10 (15 px) + one outlier 100 -> each side's own MAD is 0
        # (>=50% at the median), but the means genuinely differ -- sigma floors.
        image = np.full((4, 4), 10.0)
        image[0, 0] = 100.0
        outlier_mask = np.arange(16).reshape(4, 4) == 0
        assert np.isfinite(boundary_seam(image, outlier_mask))
        assert boundary_seam(image, outlier_mask) > 1e6


class TestContrastRetentionOracleMatrix:
    """contrast_retention(frame, oracle_after, ..., known_nuisance=gradient) scores
    ~1.0 when `known_nuisance` is `frame`'s true injected nuisance and `after` is a
    perfect, known-truth correction -- across the same 6 gradient shapes x 4 geometries
    x 2 contrasts the earlier implementation got wrong. Measured against it directly:
    contrast_retention(frame, oracle_after, feature_mask, background_mask) -- no
    known_nuisance correction, its signature at the time -- over this same 48-combination
    sweep ranges -28.32 to +4.30 and lands within 0.05 of 1.0 for only 2/48 combinations.
    The earlier bug was dividing by contrast measured on the UNCORRECTED frame; `known_nuisance`
    fixes that by correcting the denominator before measuring it. It does NOT verify the
    caller's nuisance is right for `after` -- see the module docstring and
    astro-quality-contract.md."""

    SHAPE = (256, 256)
    SKY = 0.10
    NOISE_SIGMA = 0.002

    @pytest.mark.parametrize("contrast", [0.01, 0.03])
    @pytest.mark.parametrize(
        "geometry", ["centred_small", "centred_large", "edge_crossing", "corner"]
    )
    @pytest.mark.parametrize(
        "gradient_kind", ["ramp", "dome", "bowl", "glow", "saddle", "vignette"]
    )
    def test_oracle_gradient_removal_scores_near_one(self, gradient_kind, geometry, contrast):
        seed = _det_seed(gradient_kind, geometry, contrast)
        rng = np.random.default_rng(seed)
        feature_mask, background_mask = _oracle_masks(self.SHAPE, geometry)
        reference = np.full(self.SHAPE, self.SKY, dtype=np.float64)
        reference[feature_mask] += contrast
        reference += rng.normal(0.0, self.NOISE_SIGMA, size=self.SHAPE)
        gradient = _gradient_field(self.SHAPE, gradient_kind)
        frame = reference + gradient  # nuisance broad structure injected
        oracle_after = frame - gradient  # perfect, known-truth removal
        score = contrast_retention(
            frame, oracle_after, feature_mask, background_mask, known_nuisance=gradient
        )
        assert score == pytest.approx(1.0, abs=1e-6)


def _ring(center, radius, count, angle0=0.0):
    angles = np.linspace(angle0, angle0 + 2 * np.pi, count, endpoint=False)
    return [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]


class TestA3ByRadialThird:
    """Synthetic star fields -- same discipline as test_astro_triage.py (hand-built float32
    linear luminance, deterministic RNG). Frame is 300x300 so a max-grown adaptive window
    (61px) never clips at any placed star, including the outer-third ones near the corners."""

    SHAPE = (300, 300)
    CENTER_PX = ((SHAPE[1] - 1) / 2.0, (SHAPE[0] - 1) / 2.0)
    MAX_RADIUS_PX = float(np.hypot(*CENTER_PX))
    NARROW_SIGMA_PX = 1.8
    BROAD_SIGMA_PX = 1.8 * 1.55  # ~55% FWHM growth -- same order as the measured broadening.

    # Radial fractions picked well inside each third's [1/3, 2/3) boundary, and (for "outer")
    # only on the diagonals so the 61px-max adaptive window never runs off a 300x300 frame.
    CENTER_STARS = _ring(CENTER_PX, 0.15 * MAX_RADIUS_PX, 4)
    MIDDLE_STARS = _ring(CENTER_PX, 0.50 * MAX_RADIUS_PX, 4)
    OUTER_STARS = _ring(CENTER_PX, 0.70 * MAX_RADIUS_PX, 4, angle0=np.pi / 4)

    def _place(self, positions_sigmas, noise=0.002, seed=0, flux=0.6, background=0.05):
        h, w = self.SHAPE
        luma = np.clip(
            np.random.default_rng(seed).normal(background, noise, size=self.SHAPE), 0.0, None
        )
        yy, xx = (a.astype(np.float32) for a in np.mgrid[0:h, 0:w])
        for cx, cy, sigma_px in positions_sigmas:
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            luma += flux * np.exp(-r2 / (2.0 * sigma_px**2))
        return np.clip(luma, 0.0, 1.0).astype(np.float32)

    def test_broadened_outer_third_fails_while_center_passes(self):
        positions = (
            [(x, y, self.NARROW_SIGMA_PX) for x, y in self.CENTER_STARS]
            + [(x, y, self.NARROW_SIGMA_PX) for x, y in self.MIDDLE_STARS]
            + [(x, y, self.BROAD_SIGMA_PX) for x, y in self.OUTER_STARS]
        )
        luma = self._place(positions, seed=11)
        frame_median_fwhm = self.NARROW_SIGMA_PX * _GAUSSIAN_SIGMA_TO_FWHM
        result = a3_by_radial_third(luma, frame_median_fwhm)

        assert result["center"].n == len(self.CENTER_STARS)
        assert result["center"].passed is True

        assert result["outer"].n == len(self.OUTER_STARS)
        assert result["outer"].growth_fraction > A3_RADIAL_GROWTH_BOUNDARY
        assert result["outer"].passed is False

    def test_uniform_field_passes_all_thirds(self):
        positions = [
            (x, y, self.NARROW_SIGMA_PX)
            for x, y in self.CENTER_STARS + self.MIDDLE_STARS + self.OUTER_STARS
        ]
        luma = self._place(positions, seed=12)
        frame_median_fwhm = self.NARROW_SIGMA_PX * _GAUSSIAN_SIGMA_TO_FWHM
        result = a3_by_radial_third(luma, frame_median_fwhm)
        for name in ("center", "middle", "outer"):
            assert result[name].n > 0
            assert result[name].passed is True

    def test_a_band_with_no_detected_stars_reports_honest_absence(self):
        luma = self._place([(x, y, self.NARROW_SIGMA_PX) for x, y in self.CENTER_STARS], seed=13)
        frame_median_fwhm = self.NARROW_SIGMA_PX * _GAUSSIAN_SIGMA_TO_FWHM
        result = a3_by_radial_third(luma, frame_median_fwhm)
        assert result["outer"].n == 0
        assert result["outer"].fwhm_px is None
        assert result["outer"].growth_fraction is None
        assert result["outer"].passed is None

    def test_band_edges_match_the_registration_convention(self):
        # Just inside the center/middle boundary (radial_inlier_residuals_px: center < 1/3).
        target_radius = (self.MAX_RADIUS_PX / 3.0) - 1.0
        cx, cy = self.CENTER_PX[0] + target_radius, self.CENTER_PX[1]
        luma = self._place([(cx, cy, self.NARROW_SIGMA_PX)], seed=14)
        frame_median_fwhm = self.NARROW_SIGMA_PX * _GAUSSIAN_SIGMA_TO_FWHM
        result = a3_by_radial_third(luma, frame_median_fwhm)
        assert result["center"].n == 1
        assert result["middle"].n == 0
