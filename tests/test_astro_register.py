"""Registration recovers a known similarity transform from synthetic star fields.

Deterministic RNG throughout — no real card in CI. Reference
and frame star lists are built by applying a KNOWN rotation + scale +
translation to one random field, rendered as Gaussian-PSF luminance images
so ``coarse_shift`` has something real to phase-correlate.
"""

import functools

import cv2
import numpy as np
import pytest

from gomoveshift.pipeline.photos import astro_register as astro_register_mod
from gomoveshift.pipeline.photos.astro_register import (
    MATCH_TOLERANCE_PX,
    MIN_INLIER_STARS,
    MIN_MATCH_STARS,
    MIN_OUTER_THIRD_INLIERS,
    MIN_VALID_COVERAGE,
    RegistrationError,
    coarse_shift,
    common_support_crop,
    contributor_count,
    fit_residuals_px,
    fit_similarity,
    match_progressive,
    match_stars,
    prepare_reference,
    radial_inlier_residuals_px,
    register,
    warp_coverage,
    warp_to_reference,
    warp_validity,
)

SEED = 20260813
SHAPE = (200, 200)  # (h, w)
ANGLE_DEG = 0.3
SCALE = 1.001
SHIFT = (5.2, -3.1)


def _similarity_xy(xy, angle_deg=ANGLE_DEG, scale=SCALE, shift=SHIFT):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64) * scale
    return xy @ rot.T + np.asarray(shift)


def _star_field(rng, n, shape=SHAPE, margin=20.0):
    xy = rng.uniform(margin, shape[1] - margin, size=(n, 2))
    flux = rng.uniform(400.0, 4000.0, size=n)
    return np.column_stack([xy, flux]).astype(np.float32)


def _render(stars, shape=SHAPE, sigma=1.6):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w), dtype=np.float32)
    for x, y, flux in stars:
        img += flux * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)))
    return img


def _reference_and_frame(n=50, seed=SEED):
    ref = _star_field(np.random.default_rng(seed), n)
    frame_xy = _similarity_xy(ref[:, :2])
    frame = np.column_stack([frame_xy, ref[:, 2]]).astype(np.float32)
    return ref, frame


def _rms_to_reference(transform, frame_xy, ref_xy):
    predicted = frame_xy @ transform[:, :2].T + transform[:, 2]
    return float(np.sqrt(np.mean(np.sum((predicted - ref_xy) ** 2, axis=1))))


class TestRegisterRecoversTheKnownTransform:
    def test_recovers_reference_positions_to_subpixel_rms(self):
        ref, frame = _reference_and_frame()
        result = register(_render(ref), ref, _render(frame), frame)
        assert result.n_matched >= MIN_MATCH_STARS
        assert result.inlier_fraction > 0.9
        rms = _rms_to_reference(result.transform, frame[:, :2], ref[:, :2])
        assert rms < 0.5
        # Geometry evidence: near-zero residuals for a
        # near-exact synthetic recovery, and a match yield close to 1 since every
        # ref star has a corresponding frame star within tolerance.
        assert result.residual_median_px < 0.5
        assert result.residual_p95_px < 1.0
        assert result.match_yield > 0.9
        # Where the matched pairs actually came from, stage by stage.
        assert set(result.match_yield_by_stage) == {
            "seed",
            "wide",
            "tighten_10px",
            "tighten_4px",
            "final",
        }
        assert result.match_yield_by_stage["final"] == result.n_matched


class TestOuterThirdSupportFloor:
    """A fit that only succeeded in the frame's centre must be refused, not shipped
    silently -- MIN_OUTER_THIRD_INLIERS (PROVISIONAL, see its comment)."""

    def test_thin_outer_third_support_is_refused_with_the_actual_count(self, monkeypatch):
        ref, frame = _reference_and_frame()

        real_radial = astro_register_mod.radial_inlier_residuals_px

        def thin_outer(transform, matched_ref_xy, matched_frame_xy, inliers, frame_shape):
            result = real_radial(transform, matched_ref_xy, matched_frame_xy, inliers, frame_shape)
            result["outer"] = {**result["outer"], "n": MIN_OUTER_THIRD_INLIERS - 1}
            return result

        monkeypatch.setattr(astro_register_mod, "radial_inlier_residuals_px", thin_outer)

        with pytest.raises(RegistrationError, match=str(MIN_OUTER_THIRD_INLIERS - 1)):
            register(_render(ref), ref, _render(frame), frame)

    def test_outer_third_support_at_the_floor_is_accepted(self):
        # The real synthetic fixture's outer band sits right at the floor (8) --
        # proof this doesn't refuse a healthy fit that just happens to be small.
        ref, frame = _reference_and_frame()
        result = register(_render(ref), ref, _render(frame), frame)
        assert result.radial_inlier_residuals_px["outer"]["n"] >= MIN_OUTER_THIRD_INLIERS


class TestRansacRejectsCorruptedPairs:
    def test_fit_still_converges_with_ten_percent_bad_pairs(self):
        ref, frame = _reference_and_frame(n=60)
        rng = np.random.default_rng(SEED + 1)
        bad = rng.choice(len(frame), size=6, replace=False)  # 10%
        corrupted_frame_xy = frame[:, :2].copy()
        corrupted_frame_xy[bad] = rng.uniform(0, SHAPE[1], size=(len(bad), 2))
        transform = fit_similarity(ref[:, :2], corrupted_frame_xy)
        good = np.setdiff1d(np.arange(len(ref)), bad)
        rms = _rms_to_reference(transform, corrupted_frame_xy[good], ref[good, :2])
        assert rms < 0.5


class TestTooFewMatchesRaises:
    def test_below_the_floor_raises_the_typed_error(self):
        rng = np.random.default_rng(SEED)
        ref_xy = rng.uniform(0, SHAPE[1], size=(MIN_MATCH_STARS - 1, 2))
        frame_xy = _similarity_xy(ref_xy)
        with pytest.raises(RegistrationError):
            fit_similarity(ref_xy, frame_xy)


class TestInlierCountFloor:
    """4 points share one exact similarity (identity); 4 more are independent random
    scatter over the whole frame, so no transform can plausibly place them within
    RANSAC_REPROJ_THRESHOLD_PX of each other or of the good 4. RANSAC's best
    consensus is therefore the 4 good points: inlier_fraction lands at exactly 0.5
    (clears MIN_INLIER_FRACTION) while the raw count (4) sits below MIN_INLIER_STARS
    -- isolating the count floor from the fraction floor.
    """

    def _matched_pairs(self):
        good_ref = np.array([[40.0, 40.0], [160.0, 40.0], [40.0, 160.0], [100.0, 100.0]])
        good_frame = good_ref.copy()  # identity transform, zero residual
        rng = np.random.default_rng(SEED + 2)
        bad_ref = rng.uniform(0, SHAPE[1], size=(4, 2))
        bad_frame = rng.uniform(0, SHAPE[1], size=(4, 2))  # unrelated to bad_ref
        ref_xy = np.vstack([good_ref, bad_ref])
        frame_xy = np.vstack([good_frame, bad_frame])
        return ref_xy, frame_xy

    def test_too_few_inliers_despite_meeting_the_fraction_floor_raises(self):
        ref_xy, frame_xy = self._matched_pairs()
        with pytest.raises(RegistrationError, match="inliers"):
            fit_similarity(ref_xy, frame_xy)

    def test_a_healthy_fit_well_above_the_floor_still_succeeds(self):
        ref, frame = _reference_and_frame(n=50)
        transform = fit_similarity(ref[:, :2], frame[:, :2])
        rms = _rms_to_reference(transform, frame[:, :2], ref[:, :2])
        assert rms < 0.5


class TestMutualBestMatching:
    def test_an_ambiguous_pair_is_dropped_not_guessed(self):
        # Two ref stars equidistant from one frame star: only the mutual best survives.
        stars_ref = np.array([[100.0, 100.0, 1000.0], [102.0, 100.0, 1000.0]], dtype=np.float32)
        stars_frame = np.array([[101.0, 100.0, 1000.0]], dtype=np.float32)
        pairs = match_stars(stars_ref, stars_frame, prior_shift=(0.0, 0.0), tolerance_px=3.0)
        assert pairs.shape == (1, 2)
        assert pairs[0].tolist() == [0, 0]


def _patch_estimator(monkeypatch, transform, n_points):
    """Stub ``cv2.estimateAffinePartial2D`` to return a FIXED transform with every
    point marked an inlier, so the sanity guard can be probed directly instead of
    fighting RANSAC for a specific degenerate geometry."""

    def fake_estimate(frame_pts, ref_pts, **kwargs):
        return transform, np.ones((n_points, 1), dtype=np.uint8)

    monkeypatch.setattr(astro_register_mod.cv2, "estimateAffinePartial2D", fake_estimate)


class TestFitSanityGuards:
    def _matched_pairs(self):
        rng = np.random.default_rng(SEED)
        ref_xy = rng.uniform(0, SHAPE[1], size=(MIN_MATCH_STARS, 2))
        return ref_xy, ref_xy.copy()

    def test_non_finite_transform_is_refused(self, monkeypatch):
        transform = np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        ref_xy, frame_xy = self._matched_pairs()
        _patch_estimator(monkeypatch, transform, len(ref_xy))
        with pytest.raises(RegistrationError, match="non-finite"):
            fit_similarity(ref_xy, frame_xy)

    def test_degenerate_near_zero_scale_is_refused(self, monkeypatch):
        transform = np.array([[1e-6, 0.0, 0.0], [0.0, 1e-6, 0.0]], dtype=np.float32)
        ref_xy, frame_xy = self._matched_pairs()
        _patch_estimator(monkeypatch, transform, len(ref_xy))
        with pytest.raises(RegistrationError, match="scale"):
            fit_similarity(ref_xy, frame_xy)

    def test_absurdly_large_scale_is_refused(self, monkeypatch):
        transform = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float32)
        ref_xy, frame_xy = self._matched_pairs()
        _patch_estimator(monkeypatch, transform, len(ref_xy))
        with pytest.raises(RegistrationError, match="scale"):
            fit_similarity(ref_xy, frame_xy)

    def test_rotation_beyond_cap_is_refused(self, monkeypatch):
        theta = np.radians(45.0)
        c, s = np.cos(theta), np.sin(theta)
        transform = np.array([[c, -s, 0.0], [s, c, 0.0]], dtype=np.float32)
        ref_xy, frame_xy = self._matched_pairs()
        _patch_estimator(monkeypatch, transform, len(ref_xy))
        with pytest.raises(RegistrationError, match="rotation"):
            fit_similarity(ref_xy, frame_xy)

    def test_translation_beyond_frame_fraction_is_refused(self, monkeypatch):
        # Half the frame's longer side is already a generous bound (200px frame);
        # 5000px either axis is not a plausible frame-to-frame drift.
        transform = np.array([[1.0, 0.0, 5000.0], [0.0, 1.0, 5000.0]], dtype=np.float32)
        ref, frame = _reference_and_frame()
        _patch_estimator(monkeypatch, transform, len(ref))
        with pytest.raises(RegistrationError, match="translation"):
            register(_render(ref), ref, _render(frame), frame)


class TestFitResidualsPx:
    def test_residual_median_and_p95_reflect_one_displaced_pair(self):
        transform = np.eye(2, 3, dtype=np.float64)
        ref_xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]])
        frame_xy = ref_xy.copy()
        frame_xy[2] = frame_xy[2] + np.array([3.0, 4.0])  # one displaced pair -> 5.0px residual

        median, p95 = fit_residuals_px(transform, ref_xy, frame_xy)

        assert median == pytest.approx(0.0, abs=1e-9)
        assert p95 == pytest.approx(4.0)

    def test_radial_residuals_bin_only_ransac_inliers(self):
        transform = np.eye(2, 3, dtype=np.float64)
        ref_xy = np.array([[50.0, 50.0], [75.0, 50.0], [100.0, 100.0], [0.0, 0.0]])
        frame_xy = ref_xy + np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [100.0, 0.0]])

        radial = radial_inlier_residuals_px(
            transform, ref_xy, frame_xy, np.array([True, True, True, False]), (101, 101)
        )

        assert radial["center"] == {"n": 1, "median_px": 1.0, "p95_px": 1.0}
        assert radial["middle"]["median_px"] == pytest.approx(2.0)
        assert radial["outer"]["p95_px"] == pytest.approx(3.0)


class TestPrepareReferenceCachesDownscale:
    def test_reference_downscale_happens_once_across_multiple_frames_via_register(
        self, monkeypatch
    ):
        # Proof runs through register() -- the orchestrator's one production caller --
        # not a direct coarse_shift call, so the cache is proven reachable from the
        # real pipeline path, not just from a test-only entry point.
        real_downscale = astro_register_mod._downscale
        call_count = {"n": 0}

        def counting_downscale(luma, max_dim):
            call_count["n"] += 1
            return real_downscale(luma, max_dim)

        monkeypatch.setattr(astro_register_mod, "_downscale", counting_downscale)

        ref, frame = _reference_and_frame()
        luma_ref = _render(ref)
        ref_prepared = prepare_reference(luma_ref)
        assert call_count["n"] == 1

        for _ in range(4):
            register(luma_ref, ref, _render(frame), frame, ref_prepared=ref_prepared)

        # Each register() call downscales only the frame side; the reference's one
        # downscale from prepare_reference above is never repeated.
        assert call_count["n"] == 1 + 4


class TestWarpToReference:
    def test_a_known_blob_moves_to_its_predicted_position(self):
        h, w = 120, 120
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy, sigma = 30.0, 40.0, 2.0
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
        rgb = np.stack([blob, blob, blob], axis=-1).astype(np.float32)
        transform = np.array([[1.0, 0.0, 15.0], [0.0, 1.0, 8.0]], dtype=np.float64)

        warped = warp_to_reference(rgb, transform, (h, w))

        luma = warped[:, :, 0]
        total = float(luma.sum())
        centroid_x = float((xx * luma).sum() / total)
        centroid_y = float((yy * luma).sum() / total)
        assert abs(centroid_x - (cx + 15.0)) < 0.2
        assert abs(centroid_y - (cy + 8.0)) < 0.2
        assert warped.dtype == np.float32


class TestWarpCoverageAndValidity:
    def test_pure_integer_translation_leaves_exact_border_invalid(self):
        h, w = 40, 40
        in_shape = (h, w, 3)
        out_shape = (h, w)
        transform = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]], dtype=np.float64)

        coverage = warp_coverage(transform, in_shape, out_shape)
        valid = warp_validity(transform, in_shape, out_shape)

        expected_valid = np.zeros((h, w), dtype=bool)
        expected_valid[3:, 5:] = True
        assert np.array_equal(valid, expected_valid)
        # An exact-integer translation samples only exact source pixels -- a clean
        # 0/1 step, no fractional blending (the next test covers a fractional one).
        assert set(np.unique(coverage).tolist()) <= {0.0, 1.0}

    def test_majority_zero_fill_edge_pixel_is_invalid_and_matches_the_rgb_warp(self):
        h, w = 40, 40
        rgb = np.full((h, w, 3), 10.0, dtype=np.float32)
        transform = np.array([[1.0, 0.0, 5.75], [0.0, 1.0, 0.0]], dtype=np.float64)

        coverage = warp_coverage(transform, rgb.shape, (h, w))
        valid = warp_validity(transform, rgb.shape, (h, w))
        warped = warp_to_reference(rgb, transform, (h, w))

        # Output column 5 samples source x=-0.75: 25% real pixel, 75% zero-fill
        # border -- below MIN_VALID_COVERAGE, the "border zeros became the
        # majority" case, so it must be marked invalid.
        assert coverage[10, 5] == pytest.approx(0.25, abs=1e-4)
        assert not valid[10, 5]
        # A constant source, warped through zero-fill border, is exactly the
        # source value scaled by coverage everywhere -- proof coverage describes
        # the SAME footprint the RGB warp actually produced.
        np.testing.assert_allclose(warped[:, :, 0], coverage * 10.0, atol=1e-3)


class TestContributorCount:
    def test_counts_overlap_across_differently_shifted_masks(self):
        m1 = np.array([[True, True], [False, True]])
        m2 = np.array([[True, False], [False, True]])
        m3 = np.array([[False, False], [False, True]])

        counts = contributor_count([m1, m2, m3])

        assert counts.tolist() == [[2, 1], [0, 3]]


class TestCommonSupportCrop:
    def test_returns_the_expected_rectangle(self):
        counts = np.array(
            [
                [1, 1, 1, 1],
                [1, 3, 3, 1],
                [1, 3, 3, 1],
                [1, 1, 1, 1],
            ]
        )

        crop = common_support_crop(counts, min_contributors=3)

        assert crop == (slice(1, 3), slice(1, 3))

    def test_no_valid_crop_returns_none_not_a_bare_exception(self):
        counts = np.zeros((5, 5), dtype=np.int32)
        assert common_support_crop(counts, min_contributors=1) is None


class TestWarpToReferenceOutputUnchanged:
    def test_output_matches_the_pinned_pre_change_cv2_call(self):
        h, w = 50, 60
        rgb = np.random.default_rng(SEED).uniform(0, 255, size=(h, w, 3)).astype(np.float32)
        transform = np.array([[0.999, -0.01, 4.3], [0.01, 0.999, -2.1]], dtype=np.float64)
        dsize = (w, h)

        expected = cv2.warpAffine(
            rgb,
            transform.astype(np.float32),
            dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        actual = warp_to_reference(rgb, transform, (h, w))

        assert np.array_equal(actual, expected)


class TestWarpAcceptsRemapPair:
    """The warp trio also accepts a ``build_remap``-shaped ``(map_x, map_y)``
    pair, alongside today's 2x3 affine matrix (which the trails/timelapse callers
    keep using unchanged -- see ``TestWarpToReferenceOutputUnchanged`` above)."""

    def test_identity_remap_leaves_a_blob_in_place(self):
        h, w = 60, 60
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy, sigma = 20.0, 25.0, 2.0
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
        rgb = np.stack([blob, blob, blob], axis=-1).astype(np.float32)
        map_x, map_y = xx.copy(), yy.copy()

        warped = warp_to_reference(rgb, (map_x, map_y), (h, w))

        assert warped.dtype == np.float32
        luma = warped[:, :, 0]
        total = float(luma.sum())
        centroid_x = float((xx * luma).sum() / total)
        centroid_y = float((yy * luma).sum() / total)
        assert abs(centroid_x - cx) < 0.2
        assert abs(centroid_y - cy) < 0.2

    def test_remap_pair_matches_the_equivalent_cv2_remap_call(self):
        h, w = 50, 60
        rgb = np.random.default_rng(SEED).uniform(0, 255, size=(h, w, 3)).astype(np.float32)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = xx + 1.5
        map_y = yy - 0.5

        expected = cv2.remap(
            rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        actual = warp_to_reference(rgb, (map_x, map_y), (h, w))

        assert np.array_equal(actual, expected)

    def test_warp_coverage_and_validity_accept_a_remap_pair(self):
        h, w = 40, 40
        rgb = np.full((h, w, 3), 10.0, dtype=np.float32)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # cv2.remap is destination-to-source: dst(x) samples src(map_x(x)), the
        # opposite direction of warpAffine's forward M -- so matching the 2x3
        # test's "output column 5 samples source x=-0.75" needs map_x = xx - 5.75.
        map_x = xx - 5.75
        map_y = yy.copy()

        coverage = warp_coverage((map_x, map_y), rgb.shape, (h, w))
        valid = warp_validity((map_x, map_y), rgb.shape, (h, w))
        warped = warp_to_reference(rgb, (map_x, map_y), (h, w))

        # Same fractional-coverage contract as the 2x3-matrix test above: output
        # column 5 samples source x=5.75, 25% real / 75% zero-fill.
        assert coverage[10, 5] == pytest.approx(0.25, abs=1e-4)
        assert not valid[10, 5]
        np.testing.assert_allclose(warped[:, :, 0], coverage * 10.0, atol=1e-3)

    def test_2x3_matrix_path_is_unaffected_by_the_remap_branch(self):
        h, w = 40, 40
        rgb = np.full((h, w, 3), 10.0, dtype=np.float32)
        transform = np.array([[1.0, 0.0, 5.75], [0.0, 1.0, 0.0]], dtype=np.float64)

        coverage = warp_coverage(transform, rgb.shape, (h, w))
        valid = warp_validity(transform, rgb.shape, (h, w))
        warped = warp_to_reference(rgb, transform, (h, w))

        assert coverage[10, 5] == pytest.approx(0.25, abs=1e-4)
        assert not valid[10, 5]
        np.testing.assert_allclose(warped[:, :, 0], coverage * 10.0, atol=1e-3)


# --- Stationary-foreground bias -------------------------------------------------------
#
# A landscape astro frame is not a pure star field: the bottom third is trees and
# ridgeline, high-contrast and IDENTICAL frame to frame while the sky rotates. Whole-frame
# phase correlation weights that stationary structure far above the point-like stars, so
# the returned prior is dragged toward zero. Because match_stars then shifts by that prior
# and keeps only pairs inside MATCH_TOLERANCE_PX, a prior a few px wrong loses EVERY pair
# at once -- 0 matches, not a degraded count -- and the frame is rejected outright.
#
# On a real card, 52% of frames were rejected, all with
# "only 0 matched star pairs". These fixtures are the CI-side reproduction.

# The real rig's frame, not a convenient miniature. Prior accuracy scales with the
# downscale factor, so a small fixture flatters every MAX_COARSE_DIM equally and would
# prove nothing about the one the pipeline actually runs: measured on this fixture at
# (1400, 2100) the sky crop lands within 0.3 px at 512, 1024 AND 2048, while at the real
# 3728x5600 it is 5.32 px at 512 (past MATCH_TOLERANCE_PX -- every pair lost) and 0.37 px
# at 1024. Sizing this down is how the coarse-dimension half of the fix stops being tested.
FG_SHAPE = (3728, 5600)  # (h, w) -- a 20.9 MP body
FG_SKY_FRACTION = 0.6  # stars live above this row fraction; treeline below it
FG_TRUE_SHIFT = (37.0, 23.0)  # frame xy + this == ref xy


def _render_fast(stars, shape, sigma=1.8, radius=7):
    """``_render`` for large frames: each star is drawn into its own small window rather
    than across a full-frame meshgrid, which is what makes a 1400x2100 fixture tolerable
    in CI. Same Gaussian PSF, so correlation behaviour is unchanged."""
    h, w = shape
    img = np.zeros((h, w), dtype=np.float32)
    off = np.arange(-radius, radius + 1, dtype=np.float32)
    for x, y, flux in stars:
        cx, cy = int(round(float(x))), int(round(float(y)))
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        dx = (x0 - cx) + off[: x1 - x0] + (cx - float(x))
        dy = (y0 - cy) + off[: y1 - y0] + (cy - float(y))
        img[y0:y1, x0:x1] += flux * np.exp(
            -((dx[None, :] ** 2 + dy[:, None] ** 2) / (2 * sigma**2))
        )
    return img


def _stationary_foreground(shape=FG_SHAPE, seed=SEED):
    """A bright jagged treeline filling the bottom of the frame -- the SAME array in
    reference and frame, because real foreground does not move with the sky."""
    h, w = shape
    rng = np.random.default_rng(seed + 1)
    ridge = int(h * FG_SKY_FRACTION) + np.cumsum(rng.integers(-3, 4, size=w))
    ridge = np.clip(ridge, int(h * 0.5), h - 1)
    rows = np.arange(h)[:, None]
    fg = np.where(rows >= ridge[None, :], 3000.0, 0.0).astype(np.float32)
    # Bark/branch texture, so the foreground carries broadband detail rather than one
    # flat step -- a flat block would under-state how strongly it grabs the correlation.
    fg *= 0.6 + 0.4 * rng.random((h, w)).astype(np.float32)
    return fg


@functools.lru_cache(maxsize=1)
def _sky_and_foreground_pair(n_stars=120, seed=SEED):
    """``(luma_ref, luma_frame, ref_stars, frame_stars, sky_mask)`` -- a sky shifted by a
    known amount over a foreground that did not move.

    Cached: at full sensor size each luminance plane is ~80 MB, so the four tests below
    share one build rather than paying it four times. They only read these arrays; anything
    that mutates them must copy first.
    """
    h, w = FG_SHAPE
    rng = np.random.default_rng(seed)
    sky_bottom = h * FG_SKY_FRACTION
    xy = np.column_stack(
        [rng.uniform(60.0, w - 60.0, n_stars), rng.uniform(60.0, sky_bottom - 60.0, n_stars)]
    )
    flux = rng.uniform(400.0, 4000.0, n_stars)
    ref_stars = np.column_stack([xy, flux]).astype(np.float32)
    frame_stars = np.column_stack([xy - np.asarray(FG_TRUE_SHIFT), flux]).astype(np.float32)

    foreground = _stationary_foreground()
    luma_ref = _render_fast(ref_stars, FG_SHAPE) + foreground
    luma_frame = _render_fast(frame_stars, FG_SHAPE) + foreground

    sky_mask = np.zeros(FG_SHAPE, dtype=bool)
    sky_mask[: int(sky_bottom), :] = True
    return luma_ref, luma_frame, ref_stars, frame_stars, sky_mask


def _prior_error_px(prior, true_shift=FG_TRUE_SHIFT):
    return float(np.hypot(prior[0] - true_shift[0], prior[1] - true_shift[1]))


class TestCoarseShiftUnderStationaryForeground:
    def test_the_fixture_really_is_adversarial_without_a_sky_mask(self):
        # Guards the two tests below from going vacuous: if whole-frame correlation ever
        # solves this fixture on its own, the sky-mask assertions stop proving anything.
        # This pins the fixture's difficulty, NOT the unmasked path as desired behaviour.
        luma_ref, luma_frame, _, _, _ = _sky_and_foreground_pair()
        err = _prior_error_px(coarse_shift(luma_ref, luma_frame))
        assert err > MATCH_TOLERANCE_PX

    def test_a_sky_mask_recovers_the_true_shift(self):
        luma_ref, luma_frame, _, _, sky_mask = _sky_and_foreground_pair()
        prior = coarse_shift(luma_ref, luma_frame, sky_mask=sky_mask)
        assert _prior_error_px(prior) <= MATCH_TOLERANCE_PX

    def test_a_cached_reference_carries_the_same_sky_crop(self):
        # The cache and the live path must crop identically -- a ref_prepared built over
        # the sky with a frame cropped over the whole image would correlate two different
        # windows and return a plausible-looking, wrong prior.
        luma_ref, luma_frame, _, _, sky_mask = _sky_and_foreground_pair()
        cached = coarse_shift(
            luma_ref,
            luma_frame,
            ref_prepared=prepare_reference(luma_ref, sky_mask=sky_mask),
        )
        live = coarse_shift(luma_ref, luma_frame, sky_mask=sky_mask)
        assert cached == pytest.approx(live, abs=1e-9)


class TestRegisterUnderStationaryForeground:
    def test_stars_match_when_registration_is_given_the_sky_mask(self):
        luma_ref, luma_frame, ref_stars, frame_stars, sky_mask = _sky_and_foreground_pair()

        # Without the mask this is the real card's failure: the prior misses, so every
        # candidate pair falls outside MATCH_TOLERANCE_PX and the frame is thrown away.
        with pytest.raises(RegistrationError, match="matched star pairs"):
            register(luma_ref, ref_stars, luma_frame, frame_stars)

        result = register(luma_ref, ref_stars, luma_frame, frame_stars, sky_mask=sky_mask)

        assert result.n_matched >= MIN_MATCH_STARS
        rms = _rms_to_reference(result.transform, frame_stars[:, :2], ref_stars[:, :2])
        assert rms < 0.5


# A big-sensor shape (~5.4k x 3.6k, close to a real 20.9 MP card) so a small
# frame-to-frame rotation displaces corner stars far past MATCH_TOLERANCE_PX
# while leaving central stars comfortably inside it -- the exact edge-starves
# failure mode this test pins.
WIDE_SHAPE = (3600, 5400)  # (h, w)
WIDE_ROTATION_DEG = 0.4
WIDE_SHIFT = (6.0, -4.0)
WIDE_N_STARS = 600


def _wide_star_field(rng, n, shape, margin=20.0):
    # NOT ``_star_field``: that helper samples both axes over ``shape[1]``
    # (width), which is silently correct only for the square ``SHAPE`` fixtures
    # above it -- WIDE_SHAPE is rectangular, so x and y need their own bounds.
    height, width = shape
    x = rng.uniform(margin, width - margin, size=n)
    y = rng.uniform(margin, height - margin, size=n)
    flux = rng.uniform(400.0, 4000.0, size=n)
    return np.column_stack([x, y, flux]).astype(np.float32)


def _wide_field(seed=SEED):
    ref = _wide_star_field(np.random.default_rng(seed), WIDE_N_STARS, shape=WIDE_SHAPE)
    frame_xy = _similarity_xy(ref[:, :2], angle_deg=WIDE_ROTATION_DEG, scale=1.0, shift=WIDE_SHIFT)
    frame = np.column_stack([frame_xy, ref[:, 2]]).astype(np.float32)
    return ref, frame


def _radius_fraction(xy, shape=WIDE_SHAPE):
    height, width = shape
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    max_radius = max(np.linalg.norm(center), 1.0)
    return np.linalg.norm(np.asarray(xy) - center, axis=1) / max_radius


def _outer_third_count(pairs, ref_stars):
    if len(pairs) == 0:
        return 0
    radius = _radius_fraction(ref_stars[pairs[:, 0], :2])
    return int(np.sum(radius >= 2 / 3))


WIDE_PRIOR_SHIFT = (-WIDE_SHIFT[0], -WIDE_SHIFT[1])  # frame = rotate(ref) + WIDE_SHIFT


class TestMatchProgressiveRecoversEdgeStars:
    def test_recovers_outer_third_support_not_just_a_higher_total(self):
        # A total-count assertion alone passes under a cascade that narrows the
        # search window before a model capable of reaching the frame edge exists
        # -- that defect stays invisible unless outer-third SHARE is pinned
        # directly, not inferred from the total. Bin by normalised radius (same
        # convention as ``radial_inlier_residuals_px``, astro_register.py).
        ref, frame = _wide_field()

        old_pairs = match_stars(ref, frame, prior_shift=WIDE_PRIOR_SHIFT)
        new_pairs, stage_yields = match_progressive(
            ref, frame, prior_shift=WIDE_PRIOR_SHIFT, frame_shape=WIDE_SHAPE
        )

        # The single-tolerance/translation-only matcher starves once rotation
        # displaces corner stars past MATCH_TOLERANCE_PX -- pinned as the known-bad
        # baseline this fix must beat, not as desired behaviour.
        old_outer = _outer_third_count(old_pairs, ref)

        assert len(new_pairs) >= 300
        new_outer = _outer_third_count(new_pairs, ref)
        outer_share = new_outer / len(new_pairs)
        # A cascade that only widens the FINAL tolerance (not the intermediate
        # search window) still starves the outer third to well under 1% on this
        # fixture -- the specific defect this pins against. >= 10% is a
        # substantial share, clear of that failure mode by an order of magnitude.
        assert outer_share >= 0.10
        assert new_outer > old_outer

    def test_per_stage_yields_are_reported_and_ordered_sensibly(self):
        ref, frame = _wide_field()
        _, stage_yields = match_progressive(
            ref, frame, prior_shift=WIDE_PRIOR_SHIFT, frame_shape=WIDE_SHAPE
        )

        assert set(stage_yields) == {"seed", "wide", "tighten_10px", "tighten_4px", "final"}
        assert all(isinstance(v, int) for v in stage_yields.values())
        # "wide" widens the search window over the seed-fitted similarity -- it can
        # only add candidates over the central-band-only seed, never lose them.
        assert stage_yields["wide"] >= stage_yields["seed"]
        assert stage_yields["final"] > 0

    def test_pure_translation_pair_matches_at_least_as_well_as_before(self):
        rng = np.random.default_rng(SEED + 7)
        ref_xy = rng.uniform(20.0, SHAPE[1] - 20.0, size=(50, 2))
        flux = rng.uniform(400.0, 4000.0, size=50)
        ref = np.column_stack([ref_xy, flux]).astype(np.float32)
        frame_xy = ref_xy + np.array(SHIFT)
        frame = np.column_stack([frame_xy, flux]).astype(np.float32)
        prior_shift = (-SHIFT[0], -SHIFT[1])  # frame = ref + SHIFT

        old_pairs = match_stars(ref, frame, prior_shift=prior_shift)
        new_pairs, _ = match_progressive(ref, frame, prior_shift=prior_shift, frame_shape=SHAPE)

        assert len(new_pairs) >= len(old_pairs)
        assert len(old_pairs) >= 40  # crowded-field mutual-best drops a few, not most
