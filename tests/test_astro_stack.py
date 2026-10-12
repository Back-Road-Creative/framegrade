"""astro_stack: sigma-clipped sky mean, median ground, horizon blend.

Every scene here is synthetic float32 linear RGB in [0, 1], the frozen interface's own
units, with a seeded RNG so the noise-reduction and streak-rejection assertions are
deterministic.
"""

import tracemalloc

import cv2
import numpy as np
import pytest

from gomoveshift.pipeline.photos import astro_stack as stack_mod
from gomoveshift.pipeline.photos.astro_stack import MIN_AUTO_HORIZON_STARS


def _noisy_frames(n, h=20, w=20, background=0.30, sigma=0.02, seed=1):
    rng = np.random.default_rng(seed)
    return [
        (background + sigma * rng.standard_normal((h, w, 3))).astype(np.float32) for _ in range(n)
    ]


def test_masked_median_matches_nanmedian_for_odd_even_and_missing_samples():
    rng = np.random.default_rng(20260818)
    values = rng.normal(size=(10, 4, 5, 3)).astype(np.float32)
    valid = rng.random((10, 4, 5)) > 0.25
    valid[:, 0, 0] = False
    masked = np.where(valid[..., None], values, np.nan)
    with pytest.warns(RuntimeWarning, match="All-NaN slice"):
        expected = np.nanmedian(masked, axis=0)

    actual = stack_mod._masked_median(values, valid)

    np.testing.assert_array_equal(actual, expected)


def test_stack_sky_reduces_noise_by_sqrt_n_and_is_unbiased():
    n, sigma, background = 25, 0.02, 0.30
    out = stack_mod.stack_sky(_noisy_frames(n, sigma=sigma, background=background))
    assert float(out.std()) < sigma / np.sqrt(n) * 1.6  # comfortably inside the win
    assert abs(float(out.mean()) - background) < 3 * sigma / np.sqrt(n)


def test_bright_streak_is_rejected_from_sky_stack_but_not_from_a_plain_mean():
    frames = _noisy_frames(12, sigma=0.01, background=0.30)
    frames[3][5, 5] = 5.0  # a streak: one frame, one pixel, far above background
    clipped = stack_mod.stack_sky(frames)[5, 5, 0]
    plain = np.mean([f[5, 5, 0] for f in frames])
    assert abs(clipped - 0.30) < 0.05  # sigma-clip: back to background
    assert plain - 0.30 > 0.3  # a plain mean would have kept most of the streak


def test_stack_result_retains_unblended_linear_components_and_post_rejection_count():
    """A deep combine must weight each sky pixel by samples that actually survived.

    The public blended RGB remains the current product, while the retained sky
    numerator/count and fixed-coordinate ground make a later linear combine
    possible without warping the foreground or treating rejected samples as real.
    """
    frames = _noisy_frames(12, sigma=0.01, background=0.30)
    frames[3][5, 5] = 5.0
    mask = np.ones((20, 20), dtype=np.float32)

    result = stack_mod.stack(frames, frames, mask)

    assert result.sky_sum is not None
    assert result.sky_count is not None
    assert result.ground is not None
    assert result.sky_count.dtype == np.uint16
    assert int(result.sky_count.max()) == 12
    assert int(result.sky_count[5, 5]) == 11
    sky_mean = result.sky_sum / result.sky_count[..., None]
    assert np.allclose(stack_mod.blend(sky_mean, result.ground, mask), result.rgb)


def test_stack_sky_refuses_too_few_or_mismatched_frames():
    with pytest.raises(stack_mod.StackInputError):
        stack_mod.stack_sky(_noisy_frames(2))
    frames = _noisy_frames(3)
    frames[0] = frames[0][:-1]
    with pytest.raises(stack_mod.StackInputError):
        stack_mod.stack_sky(frames)


def test_stack_ground_median_keeps_static_block_rejects_moving_star():
    n, h, w = 9, 20, 20
    frames = []
    for i in range(n):
        frame = np.full((h, w, 3), 0.10, dtype=np.float32)
        frame[8:12, 8:12] = 0.60  # the fixed foreground
        frame[i % h, (i * 3) % w] = 0.95  # a "star": a different pixel every frame
        frames.append(frame)
    out = stack_mod.stack_ground(frames)
    assert np.allclose(out[8:12, 8:12], 0.60, atol=1e-5)  # foreground converges exactly
    assert out[0, 0, 0] < 0.11  # the moving dot never dominates any one pixel's median


def test_horizon_mask_explicit_frac_is_one_above_zero_below_and_monotone():
    mask = stack_mod.horizon_mask((200, 10), horizon_frac=0.5)
    assert mask[0, 0] == pytest.approx(1.0, abs=1e-3)
    assert mask[-1, 0] == pytest.approx(0.0, abs=1e-3)
    assert np.all(np.diff(mask[:, 0]) <= 1e-9)  # never brightens going down the frame


def test_horizon_mask_requires_exactly_one_mode():
    with pytest.raises(ValueError):
        stack_mod.horizon_mask((10, 10))
    with pytest.raises(ValueError):
        stack_mod.horizon_mask((10, 10), horizon_frac=0.5, star_density=np.zeros((10, 10)))


def test_horizon_mask_star_density_finds_the_known_ridge_row():
    h, w, ridge = 200, 10, 60
    density = np.zeros((h, w), dtype=np.float32)
    density[:ridge] = 1.0  # dense sky above the ridge, none below
    mask = stack_mod.horizon_mask((h, w), star_density=density)
    boundary = float(np.argmin(mask[:, 0] > 0.5))
    assert abs(boundary - ridge) <= 5


@pytest.mark.parametrize("ridge", [1, 10, 25, 40, 49, 50, 51, 60, 100, 160, 199])
def test_horizon_mask_star_density_places_the_ridge_at_every_sky_depth(ridge):
    """Fail-OPEN regression. When more than 75% of rows are starless the profile's own
    upper quartile is 0, so the pre-fix ``profile >= threshold`` was true for EVERY row
    of any non-negative density and the boundary came back as the frame HEIGHT -- "sky
    everywhere, no masking at all", the opposite direction from the docstring's claimed
    "ground everywhere". Swept the whole 1..199 range on this fixture (200 rows, uniform
    density above the ridge and nothing below): the pre-fix rule was wrong for every
    ridge <= 49 and right for 50..199, so the failure envelope is exactly a sky shallower
    than 0.25 of frame height; the fixed rule is exact (boundary == ridge) for all 199."""
    h, w = 200, 10
    density = np.zeros((h, w), dtype=np.float32)
    density[:ridge] = 1.0
    mask = stack_mod.horizon_mask((h, w), star_density=density)
    assert int((mask[:, 0] > 0.5).sum()) == ridge


def test_horizon_mask_star_density_with_no_stars_at_all_reads_ground_not_sky():
    """``horizon_mask``'s documented "a profile that never clears the threshold reads no
    sky found" branch was UNREACHABLE before this fix: ``profile >= 0`` holds for every
    row of a non-negative density, so ``sky_rows`` was never empty and a wholly starless
    frame came back as full sky. It must fail CLOSED -- no pixel above the blend midpoint."""
    mask = stack_mod.horizon_mask((200, 10), star_density=np.zeros((200, 10), dtype=np.float32))
    assert float(mask.max()) <= 0.5
    assert float(mask[int(stack_mod.FEATHER_ROWS) :].max()) == 0.0


def _extinction_field(n_stars, ridge_row, seed, height=200, width=200):
    """A synthetic night frame: flat 0.02 sky + noise + ``n_stars`` Gaussian PSF stars
    above ``ridge_row``, thinned toward the horizon by a 0.5 vertical extinction
    gradient -- the atmospheric falloff present on essentially every real night frame."""
    rng = np.random.default_rng(seed)
    luma = 0.02 + 0.002 * rng.standard_normal((height, width))
    ys = []
    while len(ys) < n_stars:
        y = rng.uniform(2.0, ridge_row - 2.0)
        if rng.uniform() <= 1.0 - 0.5 * (y / ridge_row):
            ys.append(y)
    yy, xx = np.mgrid[0:height, 0:width]
    xs = rng.uniform(2.0, width - 2.0, n_stars)
    for x, y, amp in zip(xs, ys, rng.uniform(0.05, 0.30, n_stars)):
        luma += amp * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 1.4**2)))
    return luma.astype(np.float32)


def _auto_boundary_errors(n_stars, seeds, ridge_row=160.0):
    """Boundary error as % of frame height over ``seeds`` fields, built exactly the way
    ``astro_orchestrator.stack_run`` builds it: real ``detect_stars`` -> impulse map ->
    ``cv2.blur`` -> ``horizon_mask``."""
    from gomoveshift.pipeline.photos.astro_triage import STAR_DETECT_SIGMA, detect_stars

    errors = []
    for seed in range(seeds):
        luma = _extinction_field(n_stars, ridge_row, seed)
        height, width = luma.shape
        stars = detect_stars(luma, sigma=STAR_DETECT_SIGMA)
        impulses = np.zeros(luma.shape, dtype=np.float32)
        impulses[stars[:, 1].round().astype(int), stars[:, 0].round().astype(int)] = stars[:, 2]
        density = cv2.blur(impulses, (max(3, width // 20), max(3, height // 20)))
        mask = stack_mod.horizon_mask(luma.shape, star_density=density)
        errors.append(abs(float((mask[:, 0] > 0.5).sum()) - ridge_row) / height * 100.0)
    return np.array(errors)


def test_the_automatic_boundary_is_only_trustworthy_above_the_star_floor():
    """Evidence behind ``MIN_AUTO_HORIZON_STARS`` (24 seeds, 200x200, ridge at row 160,
    extinction falloff 0.5). The 3600-frame sweep the constant cites reads, as % of
    frame height with gross = off by more than 10%: 44.5% gross below 10 detections,
    24.8% at 10-14, 16.0% at 15-19, 8.5% at 20-24, 5.5% at 30-34, 2.3% at 35-39 and
    0.8% at 40-44. This pins the two ends cheaply, with each bound carried past its own
    cutover ON THIS HELPER: the 40-star median reads 2.2 here and 1.2/2.0/2.0/2.0/1.5
    at 8/16/48/96/160 seeds, so the <= 4.0 bound is not an artifact of this seed count,
    and walking the count DOWN it survives 35/30/25/20 (2.0/2.2/3.8/2.8 -- 25 is the
    nearest pass) and first FAILS at 15 (6.5). The 10-star cell measures 9 gross of
    these 24 seeds (25.0%/29.2%/31.2% at 16/48/96 seeds), and that bound is the one
    with a floor under it: at 8 seeds it reads 0 of 8, so 24 is not shrinkable."""
    good = _auto_boundary_errors(MIN_AUTO_HORIZON_STARS, seeds=24)
    bad = _auto_boundary_errors(10, seeds=24)

    assert float(np.median(good)) <= 4.0
    assert int((bad > 10.0).sum()) >= 3
    assert float(np.median(bad)) > float(np.median(good))


def test_blend_identity_at_mask_extremes():
    sky = np.full((4, 4, 3), 0.8, dtype=np.float32)
    ground = np.full((4, 4, 3), 0.2, dtype=np.float32)
    assert np.array_equal(stack_mod.blend(sky, ground, np.ones((4, 4))), sky)
    assert np.array_equal(stack_mod.blend(sky, ground, np.zeros((4, 4))), ground)


def test_banding_is_invariant_to_band_size(monkeypatch):
    frames = _noisy_frames(6, h=17, w=5, seed=7)
    monkeypatch.setattr(stack_mod, "BAND_ROWS", 3)
    banded_sky, banded_ground = stack_mod.stack_sky(frames), stack_mod.stack_ground(frames)
    monkeypatch.setattr(stack_mod, "BAND_ROWS", 1000)
    whole_sky, whole_ground = stack_mod.stack_sky(frames), stack_mod.stack_ground(frames)
    assert np.allclose(banded_sky, whole_sky, atol=1e-6)
    assert np.array_equal(banded_ground, whole_ground)


def test_stack_composes_sky_ground_and_blend_and_reports_n_frames():
    aligned = _noisy_frames(5, h=200, w=20, background=0.7, sigma=0.0, seed=2)
    unaligned = _noisy_frames(5, h=200, w=20, background=0.2, sigma=0.0, seed=3)
    mask = stack_mod.horizon_mask((200, 20), horizon_frac=0.5)
    result = stack_mod.stack(aligned, unaligned, mask)
    assert isinstance(result, stack_mod.StackResult)
    assert result.n_frames == 5
    assert 0.0 <= result.mean_rejected_fraction <= 1.0
    assert result.rgb[0, 0, 0] == pytest.approx(0.7, abs=1e-4)  # far above the ridge: sky
    assert result.rgb[-1, 0, 0] == pytest.approx(0.2, abs=1e-4)  # far below: ground


def test_stack_refuses_mismatched_aligned_unaligned_counts():
    with pytest.raises(stack_mod.StackInputError):
        stack_mod.stack(_noisy_frames(4), _noisy_frames(3), np.ones((20, 20)))


def test_stack_sky_and_stack_ground_never_materialize_a_full_stacked_array():
    """Neither function may allocate a whole ``(N, H, W, C)`` copy -- peak traced
    allocation must stay well under one such array's size, the old ``np.stack`` cost."""
    n, h, w = 12, 4096, 64
    frames = _noisy_frames(n, h=h, w=w, seed=11)
    full_stack_bytes = n * h * w * 3 * 4  # what a whole-array np.stack copy would cost

    for fn in (stack_mod.stack_sky, stack_mod.stack_ground):
        tracemalloc.start()
        tracemalloc.reset_peak()
        fn(frames)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < full_stack_bytes * 0.5, (
            f"{fn.__name__} peak {peak} B (full copy would be {full_stack_bytes} B)"
        )


def test_stack_output_matches_known_good_reference_values():
    """Regression pin, RE-PINNED under leave-one-out (was 0.7480573654174805 at [1, 1, 0]
    -- the old inert clip never rejected the streak at n=6)."""
    rng = np.random.default_rng(42)
    n, h, w = 6, 5, 4
    frames = [(0.3 + 0.02 * rng.standard_normal((h, w, 3))).astype(np.float32) for _ in range(n)]
    frames[2][1, 1] = 3.0  # a streak
    sky = stack_mod.stack_sky(frames)
    ground = stack_mod.stack_ground(frames)
    assert sky[0, 0, 0] == pytest.approx(0.29918280243873596)
    assert sky[1, 1, 0] == pytest.approx(0.2976689338684082)  # streak now rejected
    assert float(sky.sum()) == pytest.approx(17.987287521362305)
    assert ground[0, 0, 0] == pytest.approx(0.2942785620689392)
    assert float(ground.sum()) == pytest.approx(17.991985321044922)


def test_clean_noise_false_sample_rejection_stays_under_one_percent():
    """On CLEAN Gaussian stacks (no planted outlier), the per-n calibrated leave-one-out
    rejector must not fire on more than ~1% of samples (see SHORT_STACK_KAPPA_BY_N)."""
    for n, h, w, seed, sigma in (
        (5, 600, 600, 21, 0.02),  # true rate 0.9615%; 120x120's MC noise flipped ~16% red
        (20, 60, 60, 22, 0.02),
        (5, 120, 120, 41, 5e-4),  # regression guard: float32 cancellation; NOT 1e-4
    ):
        aligned = _noisy_frames(n, h=h, w=w, sigma=sigma, background=0.30, seed=seed)
        unaligned = _noisy_frames(n, h=h, w=w, sigma=0.0, background=0.10, seed=seed + 1)
        mask = np.ones((h, w), dtype=np.float32)
        result = stack_mod.stack(aligned, unaligned, mask)
        assert result.mean_rejected_fraction < 0.01
        assert result.rejection_mode == "leave_one_out_bright"


def test_isolated_bright_artifact_is_rejected_at_n5_and_n20():
    """One pixel in one frame, ~470x sigma up -- rejected at both ends of the calibrated range."""
    for n, seed in ((5, 30), (20, 31)):
        frames = _noisy_frames(n, sigma=0.01, background=0.30, seed=seed)
        frames[1][5, 5] = 5.0
        out = stack_mod.stack_sky(frames)
        assert abs(float(out[5, 5, 0]) - 0.30) < 0.03


def test_isolated_bright_artifact_is_rejected_at_low_per_frame_sigma():
    """The OLD ABS_DEV_FLOOR (2e-4) disabled rejection below it -- the real dark-sky regime."""
    for sigma in (1e-4, 3e-4, 5e-4):
        frames = _noisy_frames(5, h=6, w=6, background=0.02, sigma=sigma, seed=17)
        frames[1][2, 2] = 1.0
        out = stack_mod.stack_sky(frames)
        assert abs(float(out[2, 2, 0]) - 0.02) < 0.02


def test_masked_five_of_twenty_false_rejection_stays_under_one_percent():
    """kappa must come from the pixel's own valid COUNT, not ``len(frames)`` -- else
    5-of-20 is judged against kappa 3.3 (n=20) instead of 8.8 (n=5), measured 12x over.
    ``mean_rejected_fraction`` is denominated in VALID samples only (the 15/20 invalid
    frames never enter the count -- see test_mean_rejected_fraction_excludes_invalid_
    footprint_not_just_outliers), so this IS the false-reject rate directly."""
    n, h, w = 20, 400, 400
    aligned = _noisy_frames(n, h=h, w=w, sigma=0.02, background=0.30, seed=71)
    validity = [np.full((h, w), i < 5, dtype=bool) for i in range(n)]  # 5 of 20 valid
    unaligned = _noisy_frames(n, h=h, w=w, sigma=0.0, background=0.10, seed=72)
    mask = np.ones((h, w), dtype=np.float32)
    result = stack_mod.stack(aligned, unaligned, mask, sky_validity=validity)
    assert result.mean_rejected_fraction < 0.01  # measured 0.96% at this fixed seed


def test_mean_rejected_fraction_excludes_invalid_footprint_not_just_outliers():
    """``mean_rejected_fraction`` must mean "the sigma clip's own reject rate", never
    conflate an out-of-footprint (invalid) sample with a genuine outlier. Six IDENTICAL
    frames -- zero outliers by construction -- with a drifting real per-frame footprint
    (the edge-wedge shape) must report a rate of exactly zero; the pre-fix
    arithmetic instead reported exactly the invalid-pixel fraction (0.375 here)."""
    n, h, w = 6, 20, 20
    aligned = [np.full((h, w, 3), 0.3, dtype=np.float32) for _ in range(n)]
    unaligned = [np.full((h, w, 3), 0.1, dtype=np.float32) for _ in range(n)]
    validity = []
    for i in range(n):
        v = np.zeros((h, w), dtype=bool)
        v[:, i * 3 :] = True  # a drifting footprint, same shape as real inter-frame drift
        validity.append(v)
    invalid_fraction = float(np.mean([~v for v in validity]))
    assert invalid_fraction > 0.1  # sanity: the fixture really has a lot of invalid pixels

    mask = np.ones((h, w), dtype=np.float32)
    result = stack_mod.stack(aligned, unaligned, mask, sky_validity=validity)
    assert result.mean_rejected_fraction == pytest.approx(0.0, abs=1e-9)

    # No-op sanity check: with validity=None every sample is trivially valid, so this
    # fix must not move that result at all.
    result_none = stack_mod.stack(aligned, unaligned, mask)
    assert result_none.mean_rejected_fraction == pytest.approx(0.0, abs=1e-9)


def test_mean_rejected_fraction_denominator_excludes_invalid_samples_exactly():
    """Exact arithmetic pin, fully deterministic (no RNG): 5 valid frame-values at one
    pixel plus 1 INVALID frame carrying a wild value (9.0) that must never enter the
    count either way. Frame index 4 (5.0) is a genuine bright outlier among the 5 valid
    frames and gets rejected; frame index 5 is invalid, not rejected -- rejection and
    invalidity are different things. Correct arithmetic: 1 rejected / 5 valid = 0.2.
    Pre-fix arithmetic instead diluted by the invalid frame: 2 / 6 = 0.333."""
    values = [0.30, 0.31, 0.29, 0.30, 5.0, 9.0]
    frames = [np.full((1, 1, 3), v, dtype=np.float32) for v in values]
    validity = [np.ones((1, 1), dtype=bool) for _ in values]
    validity[5][:] = False  # out-of-footprint sample, not an outlier
    out, frac, mode = stack_mod._short_stack_reject(frames, validity)
    assert mode == "leave_one_out_bright"
    assert float(out[0, 0, 0]) == pytest.approx(0.3, abs=1e-4)  # the outlier was rejected
    assert frac == pytest.approx(0.2)  # 1 rejected / 5 valid -- never diluted to 2/6


def test_dark_outlier_survives_by_default_and_is_rejected_when_flipped():
    """``bright_only=True`` (default) keeps a dark outlier in the mean; ``False`` rejects it."""
    n, h, w = 8, 20, 20
    frames = _noisy_frames(n, h=h, w=w, sigma=0.01, background=0.30, seed=61)
    frames[3][5, 5] = 0.0
    default_out = float(stack_mod.stack_sky(frames)[5, 5, 0])
    flipped_out = float(stack_mod.stack_sky(frames, bright_only=False)[5, 5, 0])
    plain = float(np.mean([f[5, 5, 0] for f in frames]))
    assert abs(default_out - plain) < 1e-4  # survives: default is bright-only
    assert abs(flipped_out - 0.30) < 0.03  # rejected: recovers close to background


def test_pure_red_outlier_rejects_the_whole_sample_not_just_red():
    """ONE boolean per (frame, pixel): a pure-red streak must drop that sample's G too."""
    n, h, w = 5, 20, 20
    frames = _noisy_frames(n, h=h, w=w, sigma=0.01, background=0.30, seed=5)
    y, x = 5, 5
    frames[2][y, x, 0] = 5.0  # pure-red outlier: only R is a streak
    frames[2][y, x, 1] += 0.05  # a modest, non-outlier-sized G deviation, same sample
    out = stack_mod.stack_sky(frames)
    g_excluding_frame2 = np.mean([f[y, x, 1] for i, f in enumerate(frames) if i != 2])
    g_including_all = np.mean([f[y, x, 1] for f in frames])
    assert abs(g_including_all - g_excluding_frame2) > 0.005  # the two differ measurably
    assert abs(float(out[y, x, 1]) - g_excluding_frame2) < 1e-5  # G dropped WITH R


def test_short_stack_fallback_below_five_frames_does_not_reject():
    """n=3 and n=4 fall back to a plain mean (see SHORT_STACK_MIN_REJECT_FRAMES) -- a
    bright artifact survives unrejected, pinned deliberately."""
    for n in (3, 4):
        aligned = _noisy_frames(n, h=6, w=6, background=0.30, sigma=0.01, seed=9)
        aligned[0][2, 2, 0] = 9.0  # a bright artifact -- NOT rejected at this n
        expected = np.mean([f[2, 2, 0] for f in aligned])
        out = stack_mod.stack_sky(aligned)
        assert float(out[2, 2, 0]) == pytest.approx(expected, abs=1e-4)

        unaligned = _noisy_frames(n, h=6, w=6, background=0.10, sigma=0.0, seed=10)
        mask = np.ones((6, 6), dtype=np.float32)
        result = stack_mod.stack(aligned, unaligned, mask)
        assert result.rejection_mode == "no_reject_fallback"


def test_validity_mask_recovers_sky_from_zero_filled_warp_borders():
    """Without a mask, a zero-fill border enters as valid; at 4-of-5 zero-fill the
    all-zero "others" group is bitwise-degenerate, so rejection is refused -- diluted,
    not zeroed. With the mask, the real value recovers exactly."""
    n, h, w = 5, 4, 4
    background = 0.5
    aligned = [np.full((h, w, 3), background, dtype=np.float32) for _ in range(n)]
    y, x = 2, 2
    validity = [np.ones((h, w), dtype=bool) for _ in range(n)]
    for i in (0, 1, 2, 3):  # 4 of 5 frames are zero-filled warp border at this one pixel
        aligned[i][y, x] = 0.0
        validity[i][y, x] = False
    without_mask = stack_mod.stack_sky(aligned)
    with_mask = stack_mod.stack_sky(aligned, validity=validity)

    assert float(without_mask[y, x, 0]) == pytest.approx(0.1, abs=1e-4)  # diluted, not zeroed
    assert float(with_mask[y, x, 0]) == pytest.approx(background, abs=1e-4)  # recovered


def test_one_degenerate_channel_does_not_veto_rejection_for_the_others():
    """A single bitwise-degenerate channel must not veto rejection for the others."""
    frames = _noisy_frames(5, h=6, w=6, sigma=0.01, background=0.30, seed=51)
    y, x = 2, 2
    for f in frames:
        f[y, x, 2] = 0.0  # B pinned exactly 0.0 every frame; R/G stay noisy
    frames[1][y, x, 0] = 5.0  # bright R artifact, one frame
    out = stack_mod.stack_sky(frames)
    assert abs(float(out[y, x, 0]) - 0.30) < 0.05  # rejected: R recovers near background


def test_partial_validity_below_min_reject_frames_falls_back_to_mean():
    """Valid count under SHORT_STACK_MIN_REJECT_FRAMES falls back to the plain mean."""
    aligned = _noisy_frames(8, h=2, w=2, background=0.30, sigma=0.01, seed=13)
    aligned[6][0, 0, 0] = 9.0  # a bright artifact among the 3 frames left valid
    validity = [np.full((2, 2), i >= 5, dtype=bool) for i in range(8)]  # 3 of 8 valid
    expected = np.mean([f[0, 0, 0] for i, f in enumerate(aligned) if i >= 5])
    out = stack_mod.stack_sky(aligned, validity=validity)
    assert float(out[0, 0, 0]) == pytest.approx(expected, abs=1e-4)  # NOT rejected


def test_dead_channel_single_code_deviation_does_not_trigger_false_rejection():
    """A degenerate channel's z must not DRIVE rejection for the sample it shares."""
    frames = _noisy_frames(5, h=6, w=6, sigma=0.01, background=0.30, seed=81)
    for f in frames:
        f[2, 2, 2] = 0.0  # B bitwise 0.0 every frame -- dark-clip regime
    frames[3][2, 2, 2] = 1.0 / 65535.0  # one 16-bit code above 0, one frame
    out = stack_mod.stack_sky(frames)
    assert float(out[2, 2, 2]) == pytest.approx((4 * 0.0 + 1.0 / 65535.0) / 5, abs=1e-6)


def test_zero_filled_border_with_two_real_samples_is_not_zeroed():
    """2 real frames among n-2 zero-filled ones (warp border) must not both reject."""
    aligned = _noisy_frames(20, h=6, w=6, sigma=5e-4, background=0.02, seed=91)
    for i in range(20):
        if i not in (3, 11):
            aligned[i][2, 2] = 0.0
    out = stack_mod.stack_sky(aligned)
    assert float(out[2, 2, 0]) != 0.0  # diluted, never bitwise zeroed


def test_stack_reports_and_marks_thin_coverage_from_a_staggered_drift_gradient():
    """The edge-wedge scenario: real inter-frame drift trims a DIFFERENT amount off
    each warped frame's border, so contributor count ramps from 1 (barely any frames
    reach that far) up to n (every frame reaches it) -- a genuine gradient, not uniform
    coverage with one hole. Frame i is staggered to be valid only from column i onward,
    same shape as a real drifting sequence's per-frame footprint."""
    n, h, w = 15, 100, 20
    aligned = _noisy_frames(n, h=h, w=w, sigma=0.02, background=0.30, seed=201)
    unaligned = _noisy_frames(n, h=h, w=w, sigma=0.0, background=0.10, seed=202)
    validity = []
    for i in range(n):
        v = np.zeros((h, w), dtype=bool)
        v[:, i:] = True
        validity.append(v)
    mask = np.ones((h, w), dtype=np.float32)
    result = stack_mod.stack(aligned, unaligned, mask, sky_validity=validity)

    # counts per column = min(col + 1, n): 1, 2, 3, ..., 15, 15, 15, 15, 15
    assert result.coverage.min_count == 1
    assert result.coverage.median_count == pytest.approx(10.5)
    assert result.coverage.below_floor_fraction == pytest.approx(0.1)  # cols 0,1 of 20

    thin = result.thin_coverage
    assert thin.shape == (h, w)
    assert thin[:, 0].all()  # count 1 < MIN_STACK_CONTRIBUTORS(3): fires
    assert thin[:, 1].all()  # count 2 < 3: fires
    assert not thin[:, 2].any()  # count 3: at the floor, not below it -- stays out of the way
    assert not thin[:, -1].any()  # count 15 (full): stays out of the way

    # rgb itself is UNCHANGED by the mark -- stack_sky's own math already recovers a
    # real n=1 sample correctly (see test_validity_mask_recovers_sky_from_zero_filled_
    # warp_borders); the policy is "mark", not "crop or blank".
    plain_n1 = float(aligned[0][:, 0, 0].mean())  # only frame 0 is valid at col 0 (v[:, 0:])
    assert float(result.rgb[:, 0, 0].mean()) == pytest.approx(plain_n1, abs=1e-4)

    # MEASURE, not assert: the actual noise gap this policy exists to flag.
    thin_noise = float(result.rgb[:, :2, 0].std())
    full_noise = float(result.rgb[:, 14:, 0].std())
    print(f"thin-coverage(n<=2) std={thin_noise:.5f}  full-coverage(n=15) std={full_noise:.5f}")
    assert thin_noise > full_noise * 2  # thin columns keep close to single-frame noise


def test_stack_sky_validity_none_leaves_coverage_fields_at_none():
    """``sky_validity=None`` is today's only real caller -- byte-identical, including the
    new fields staying unset rather than silently computing something for it."""
    aligned = _noisy_frames(5, h=6, w=6, background=0.3, sigma=0.01, seed=15)
    unaligned = _noisy_frames(5, h=6, w=6, background=0.1, sigma=0.0, seed=16)
    result = stack_mod.stack(aligned, unaligned, np.ones((6, 6), dtype=np.float32))
    assert result.coverage is None
    assert result.thin_coverage is None


def test_stack_refuses_all_invalid_sky_validity_instead_of_a_black_master():
    """The all-invalid stack: sky_validity is False EVERYWHERE, every pixel, every
    frame -- nothing valid anywhere to average. Must hard-fail, not return a
    numerically-fine-looking all-zero ``rgb`` with no exception/warning/NaN."""
    n, h, w = 3, 6, 6
    aligned = [np.full((h, w, 3), 0.2, dtype=np.float32) for _ in range(n)]
    unaligned = [np.full((h, w, 3), 0.2, dtype=np.float32) for _ in range(n)]
    validity = [np.zeros((h, w), dtype=bool) for _ in range(n)]
    mask = np.ones((h, w), dtype=np.float32)
    with pytest.raises(stack_mod.StackInputError, match="valid"):
        stack_mod.stack(aligned, unaligned, mask, sky_validity=validity)


def test_stack_sky_refuses_all_invalid_validity_directly():
    """Same guard, exercised through ``stack_sky``'s own ``validity`` kwarg -- the
    shared ``_short_stack_reject`` path both callers go through."""
    frames = _noisy_frames(4, h=5, w=5, seed=50)
    validity = [np.zeros((5, 5), dtype=bool) for _ in range(4)]
    with pytest.raises(stack_mod.StackInputError):
        stack_mod.stack_sky(frames, validity=validity)


def test_stack_partially_invalid_sky_validity_still_succeeds():
    """A validity mask with a zero-contributor CORNER, but real coverage everywhere
    else, must NOT trip the all-invalid guard -- that's the existing thin_coverage
    POLICY's job (mark, not reject; see test_stack_reports_and_marks_thin_coverage_
    from_a_staggered_drift_gradient), a different case from 'nothing valid anywhere'."""
    n, h, w = 5, 6, 6
    aligned = _noisy_frames(n, h=h, w=w, background=0.3, sigma=0.0, seed=44)
    unaligned = _noisy_frames(n, h=h, w=w, background=0.1, sigma=0.0, seed=45)
    validity = [np.ones((h, w), dtype=bool) for _ in range(n)]
    for v in validity:
        v[0:2, 0:2] = False  # top-left 2x2 corner: zero contributors, every frame
    mask = np.ones((h, w), dtype=np.float32)
    result = stack_mod.stack(aligned, unaligned, mask, sky_validity=validity)
    assert result.thin_coverage[0:2, 0:2].all()  # marked, not rejected
    assert result.rgb[3, 3, 0] == pytest.approx(0.3, abs=1e-4)  # rest stacks fine


def test_stack_sky_all_zero_content_without_validity_is_not_rejected():
    """The guard is validity-scoped, not content-scoped: real all-zero-VALUE frames
    with sky_validity=None legitimately stack to zero and must NOT be treated as the
    'nothing valid' defect. Item 5: this confirms the silent-black path is NOT
    reachable without an explicit sky_validity -- validity=None makes every pixel
    trivially valid, so an all-zero result here is real content, not a coverage
    failure. Stated as the CONTRACT, not a caller census: whether or not any caller
    currently passes sky_validity, None means every pixel is valid."""
    frames = [np.zeros((5, 5, 3), dtype=np.float32) for _ in range(4)]
    out = stack_mod.stack_sky(frames)
    assert float(out.max()) == 0.0  # legitimate all-black content, not an error


def test_every_pixel_thin_but_nonzero_still_stacks_instead_of_raising():
    """Pins the docstring's load-bearing claim -- 'a structural zero-vs-nonzero test,
    never a tunable floor'. Without this, rewriting the guard as the very floor it says
    it is NOT (``not np.any(counts >= MIN_STACK_CONTRIBUTORS)``) leaves the whole suite
    green while killing real renders: six frames on DISJOINT stripes give every pixel
    exactly 1 contributor -- below the floor everywhere, but nonzero everywhere -- which
    is the documented keep-it-but-mark-it case, not the nothing-to-average case.
    The pre-existing partial-validity test cannot catch that mutant: its non-corner
    pixels still carry the full n contributors, so the floor is satisfied somewhere."""
    height, width, count = 12, 8, 6
    frames = [np.full((height, width, 3), 0.4, dtype=np.float32) for _ in range(count)]
    validity = []
    for i in range(count):
        v = np.zeros((height, width), dtype=bool)
        v[2 * i : 2 * i + 2, :] = True  # 2 rows each, disjoint -> exactly 1 contributor
        validity.append(v)
    counts = stack_mod.contributor_count(validity)
    assert counts.min() == counts.max() == 1  # thin everywhere, empty nowhere
    assert 1 < stack_mod.MIN_STACK_CONTRIBUTORS  # so the floor really is unmet

    result = stack_mod.stack(
        frames, frames, np.zeros((height, width), dtype=np.float32), sky_validity=validity
    )
    assert result.rgb[0, 0, 0] == pytest.approx(0.4)  # kept and averaged, not refused
    assert bool(result.thin_coverage.all())  # ... and marked thin, which is the policy


def test_dark_clipped_channel_false_sample_rejection_stays_under_one_percent():
    """A dark-clipped channel (mostly exact 0.0) must not drive false rejection via the
    shared reject boolean -- no artifact anywhere."""
    rng = np.random.default_rng(95)
    aligned = _noisy_frames(5, h=300, w=300, sigma=0.02, background=0.30, seed=96)
    dark = np.round(np.clip(rng.standard_normal((5, 300, 300)) * 1.5e-5, 0.0, None) * 65535.0)
    for i in range(5):
        aligned[i][:, :, 2] = (dark[i] / 65535.0).astype(np.float32)
    unaligned = _noisy_frames(5, h=300, w=300, sigma=0.0, background=0.10, seed=97)
    result = stack_mod.stack(aligned, unaligned, np.ones((300, 300), dtype=np.float32))
    assert result.mean_rejected_fraction < 0.01
