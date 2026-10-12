"""Synthetic astro quality fixtures (astro_fixtures.py). Every assertion checks a
real, injected property against a measurement, never a tautology on the builder's own math.
"""

import numpy as np

import astro_fixtures as fx


def _default_ground_profile(width):
    """A deterministic, non-flat silhouette (two sine components) -- never a flat row."""
    x = np.arange(width, dtype=np.float32)
    profile = (
        0.75
        + 0.05 * np.sin(2 * np.pi * x / width * 3)
        + 0.03 * np.sin(2 * np.pi * x / width * 7 + 1.0)
    )
    return np.clip(profile, 0.05, 0.95).astype(np.float32)


def test_gaussian_stack_seed_reproducible_and_seed_sensitive():
    frames_a, _ = fx.gaussian_stack(5, seed=7)
    frames_b, _ = fx.gaussian_stack(5, seed=7)
    frames_c, _ = fx.gaussian_stack(5, seed=8)
    for a, b in zip(frames_a, frames_b):
        assert np.array_equal(a, b)  # same seed -> byte-identical
    assert not np.array_equal(frames_a[0], frames_c[0])  # different seed -> different draw


def test_gaussian_stack_dtype_no_negatives_and_std_matches_sigma_at_n20():
    sigma = 0.02
    frames, _ = fx.gaussian_stack(20, shape=(64, 64), sigma=sigma, background=0.30, seed=3)
    for frame in frames:
        assert frame.dtype == np.float32
        assert (frame >= 0.0).all()
    measured = float(np.stack(frames, axis=0).std(axis=0).mean())
    assert abs(measured - sigma) / sigma < 0.05  # within 5% at n=20


def test_sky_frame_nebulosity_contrast_matches_request():
    # gradient_amplitude nonzero: pins "over local sky" (contract 2), not scalar background.
    kwargs, contrast = dict(background=0.02, gradient_amplitude=0.05), 0.02
    frame, truth = fx.sky_frame(nebulosity_contrast=contrast, **kwargs)
    reference, _ = fx.sky_frame(**kwargs)  # gradient-only, no nebulosity -- the true local sky
    assert frame.dtype == np.float32 and (frame >= 0.0).all()
    cy, cx = 64, 64  # DEFAULT_SHAPE (128, 128) center == nebulosity_center default
    local_sky = float(reference[cy, cx, 0])
    measured_contrast = (float(frame[cy, cx, 0]) - local_sky) / local_sky
    assert abs(measured_contrast - contrast) / contrast < 0.02
    assert truth["nebulosity_contrast"] == contrast
    assert bool(truth["nebulosity_mask"][cy, cx])


def test_sky_frame_warp_border_is_exactly_zero_inside_mask():
    frame, truth = fx.sky_frame(background=0.5, warp_border_px=6, noise_sigma=0.01, seed=1)
    b = truth["warp_border_px"]
    assert (frame[:b, :] == 0.0).all() and (frame[-b:, :] == 0.0).all()
    assert (frame[:, :b] == 0.0).all() and (frame[:, -b:] == 0.0).all()
    assert frame[b + 5, b + 5, 0] > 0.0  # interior untouched


def test_sky_frame_ground_silhouette_matches_profile_and_is_not_flat():
    width = fx.DEFAULT_SHAPE[1]
    profile = _default_ground_profile(width)
    frame, truth = fx.sky_frame(ground_profile_frac=profile, ground_level=0.01, background=0.9)
    mask = truth["ground_mask"]
    assert mask.shape == fx.DEFAULT_SHAPE
    boundary_rows = {int(np.argmax(mask[:, x])) for x in range(width)}
    assert len(boundary_rows) > 1  # irregular silhouette, not a single flat row
    assert np.allclose(frame[mask], 0.01) and np.allclose(frame[~mask], 0.9)


def test_multi_frame_sequence_sensor_artifacts_repeat_but_single_streak_appears_once():
    sites = ((10, 20), (100, 5))
    fixed, single = ((5, 5), (5, 40)), ((80, 10), (80, 60))
    frames, truth = fx.multi_frame_sequence(
        4,
        background=0.02,
        hot_pixel_sites=sites,
        fixed_streak=fixed,
        streak=single,
        streak_value=0.9,
        noise_sigma=0.01,
        seed=2,
    )
    assert truth["hot_pixel_sites"] == sites
    assert truth["streak"] == single
    for frame in frames:
        for ry, rx in sites:
            assert frame[ry, rx, 0] == 1.0  # exact, every frame -- painted after noise
        assert frame[5, 20, 0] == 0.9  # sensor-fixed streak: every frame
    hits = sum(frame[80, 30, 0] == 0.9 for frame in frames)
    assert hits == 1 and truth["streak_frame_index"] == 2  # single-frame trail, n // 2 default
