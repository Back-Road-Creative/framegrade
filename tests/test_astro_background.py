"""Star-masked mesh background removal.

An audit measured the LEFT sample box averaging down at 74.8% of the sqrt(N) ideal
at 38 chunks while centre and right sat at 98.4/96.8.  The cause is sky-fixed structure in the
8-128 px band (camera-frame pairwise correlation 0.021 against 0.550 in the sky frame), which is
why every polynomial background fit missed it: a degree-2 surface has no freedom at those scales
and scored 0.00 efficacy at all three synthetic control scales.

A star-masked 32 px tile-median mesh took the left box to 97.8% with centre and right moving by
<= 0.3 pp.  These tests are the two synthetic controls that qualified it -- white noise must
survive untouched, and star flux must survive -- plus the shape handling production needs and the
research box never exercised (3728 is not a multiple of 32, and real frames carry 3 channels).
"""

import numpy as np
import pytest

from gomoveshift.pipeline.photos import astro_background as bg


def _flat(shape, value=0.02):
    return np.full(shape, value, dtype=np.float64)


def test_mesh_removal_strengthens_with_scale_and_spares_small_structure():
    """The mesh must be a HIGH-pass keeper: it takes the big stuff and leaves the small stuff.

    Thresholds here are measured, not chosen.  Injecting a blurred field of known scale and
    reading back what survives gives 0.908 / 0.705 / 0.566 / 0.448 / 0.267 of the original
    amplitude at 16 / 32 / 48 / 96 / 192 px (seed 20260820).  That ordering is the same one the
    audit's synthetic control measured as efficacy 0.40 / 0.56 / 0.76 at 48 / 96 / 192 px.

    The 16 px end is the safety-relevant one: structure at the scale of a star must come through
    essentially untouched, or the mesh is eating signal rather than background.
    """
    import cv2

    valid = np.ones((256, 256), dtype=bool)
    survived = {}
    for scale in (16, 48, 192):
        structure = cv2.blur(
            np.random.default_rng(20260820).normal(0.0, 1.0, (256, 256)), (scale, scale)
        )
        structure = 3e-3 * structure / structure.std()
        field = bg.mesh_background((_flat((256, 256)) + structure)[..., None], valid, tile=32)
        survived[scale] = float((structure - field[..., 0]).std() / structure.std())

    assert survived[16] > 0.85, f"small-scale structure was eaten: {survived[16]:.3f}"
    assert survived[192] < 0.35, f"large-scale structure survived: {survived[192]:.3f}"
    assert survived[16] > survived[48] > survived[192]


def test_white_noise_survives_removal():
    """Control A: the remover must not eat independent noise -- that would fake an improvement."""
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 3e-4, (256, 256, 1))
    mean = _flat((256, 256, 1)) + noise
    valid = np.ones((256, 256), dtype=bool)

    cleaned = mean - bg.mesh_background(mean, valid, tile=32)

    ratio = float(cleaned.std() / noise.std())
    assert 0.95 <= ratio <= 1.05, ratio


def test_star_flux_survives_removal():
    """Control: the gate a synthetic control can honestly certify -- signal must not be eaten."""
    rng = np.random.default_rng(11)
    mean = _flat((256, 256, 1))
    truth = np.zeros((256, 256), dtype=np.float64)
    for row, col in rng.integers(20, 236, size=(40, 2)):
        truth[row - 2 : row + 3, col - 2 : col + 3] += 0.05
    mean = mean + truth[..., None]
    valid = np.ones((256, 256), dtype=bool)

    cleaned = mean - bg.mesh_background(mean, valid, tile=32)

    kept = float((cleaned[..., 0] - np.median(cleaned)).sum() / truth.sum())
    assert kept >= 0.95, f"only {kept:.1%} of star flux survived"


def test_shape_not_divisible_by_the_tile_is_handled():
    """Production frames are 3728 x 5600; 3728 % 32 == 16, so the mesh must pad, not truncate."""
    valid = np.ones((3728 // 8, 5600 // 8), dtype=bool)
    mean = _flat((*valid.shape, 3))
    assert valid.shape[0] % 32 != 0

    field = bg.mesh_background(mean, valid, tile=32)

    assert field.shape == mean.shape
    assert np.all(np.isfinite(field))


def test_each_channel_gets_its_own_background():
    """A colour gradient is per-channel; one shared mesh would leave two thirds of it behind."""
    valid = np.ones((128, 128), dtype=bool)
    ramp = np.linspace(0.0, 4e-3, 128)[None, :] * np.ones((128, 1))
    mean = np.stack([_flat((128, 128)), _flat((128, 128)) + ramp, _flat((128, 128)) - ramp], -1)

    field = bg.mesh_background(mean, valid, tile=32)

    assert field[..., 1].std() > 1e-4
    assert field[..., 2].std() > 1e-4
    assert field[..., 0].std() < field[..., 1].std() / 10


def test_the_sky_pedestal_survives_so_only_structure_is_removed():
    """An uncentred mesh IS the sky level; subtracting it would ship a frame with a zero sky."""
    valid = np.ones((128, 128), dtype=bool)
    mean = _flat((128, 128, 1), value=0.2)

    cleaned = mean - bg.mesh_background(mean, valid, tile=32)

    assert float(np.median(cleaned)) == pytest.approx(0.2, abs=1e-6)


def test_invalid_pixels_never_produce_a_nan_background():
    """A tile can be entirely outside the warped footprint; it must not poison the field."""
    valid = np.ones((128, 128), dtype=bool)
    valid[:64, :64] = False
    mean = _flat((128, 128, 1))
    mean[~valid] = 0.0

    field = bg.mesh_background(mean, valid, tile=32)

    assert np.all(np.isfinite(field))


def _dense_star_field(rng, shape=(512, 512), n_stars=4000, amp=0.01):
    """A star-dense box, like the real ones -- the research star mask covers 83% of centre."""
    import cv2

    seeds = np.zeros(shape, dtype=np.float64)
    for row, col in rng.integers(10, shape[0] - 10, size=(n_stars, 2)):
        seeds[row, col] += amp * rng.pareto(1.5)
    return cv2.GaussianBlur(seeds, (0, 0), 2.0)


def test_star_mask_does_not_swallow_the_background_it_is_meant_to_expose():
    """Regression for the ported-background defect.

    The first shipped `_star_mask` thresholded on ``mono - blur(mono, 2*tile)``.  High-passing at
    twice the tile size strips out the same 8-128 px structure the mesh exists to remove, which
    collapses the MAD sigma; 5 sigma then becomes a low bar, ordinary star wings clear it, and a
    15 px dilation multiplies every extra seed into a large masked area.  On the real LEFT box
    that inflated mask coverage from 12.3% to 31.2% -- so on the ONE box carrying the defect, the
    mask hid the defect from the mesh, and the shipped left-box efficiency came out at 91.2%
    instead of the 97.8% the audit measured (a 6.6 pp miss, outside the +/-3.4 pp estimator band).

    Thresholding on a 2nd-degree-detrended residual keeps the structure in the residual, so sigma
    stays representative.  That restored the real-night figure to 97.6%.

    A single-frame synthetic cannot reproduce the efficiency number -- efficiency is a cross-chunk
    property -- but it does reproduce the mask inflation that caused it: 12.7% against 5.6% here,
    a 2.3x factor next to the real 2.5x.  That is what this pins.
    """
    import cv2

    rng = np.random.default_rng(20260820)
    structure = cv2.blur(rng.normal(0.0, 1.0, (512, 512)), (48, 48))
    structure = 3e-3 * structure / structure.std()
    mono = 0.01 + structure + _dense_star_field(rng) + rng.normal(0.0, 3e-4, (512, 512))

    coverage = float(bg._star_mask(mono, 32).mean())

    assert coverage < 0.08, (
        f"star mask covers {coverage:.1%} of a structure-rich box; the high-pass-threshold "
        "regression covered 12.7% here and 31.2% on the real left box"
    )
