"""``astro_geometry``: joint lens + per-frame rotation solve.

Synthetic ground truth only (``tests/fixtures/astro_synthetic.py``) --
pure geometry, no real card.
"""

import numpy as np
import pytest

from fixtures.astro_synthetic import (
    add_noisy_rotation_frame,
    add_non_rotation_frame,
    build_synthetic_night,
)
from gomoveshift.pipeline.photos.astro_geometry import (
    FrameMatches,
    GeometrySolveError,
    solve_night,
)
from gomoveshift.pipeline.photos.astro_lens import LensParams

# ``solve_night`` used with the default k1-start ladder is not cheap (7
# starting points x a real least_squares solve each) -- a flat k1=0 initial
# guess is deliberately the hard case this module needs to recover from (see
# ``K1_START_OFFSETS``'s own comment), so tests pay for that instead of
# cheating with an initial guess already near the answer.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def night():
    return build_synthetic_night()


@pytest.fixture(scope="module")
def flat_initial(night):
    # k1=k2=0, f_px exact -- mirrors what a real run has (EXIF gives f_px,
    # `LensParams.from_exif` defaults k1/k2 to 0.0), not a value already
    # close to the synthetic ground truth.
    return LensParams(
        k1=0.0, k2=0.0, cx=night.true_lens.cx, cy=night.true_lens.cy, f_px=night.true_lens.f_px
    )


@pytest.fixture(scope="module")
def solved(night, flat_initial):
    return solve_night(night.matches, night.frame_shape, flat_initial)


def test_recovers_shared_lens_from_a_flat_initial_guess(night, solved):
    # Centroid noise is 0.15 px on ~40 train points/frame across 5 frames
    # (200 train points total) constraining 3 shared unknowns -- tight
    # recovery is expected. Tolerances are ~10x the noise floor to absorb
    # normal fit variance, not padding to make a bad fit pass.
    assert solved.converged
    assert solved.lens.k1 == pytest.approx(night.true_lens.k1, abs=0.02)
    assert solved.lens.k2 == pytest.approx(night.true_lens.k2, abs=0.02)
    assert solved.lens.f_px == pytest.approx(night.true_lens.f_px, abs=10.0)
    # cx/cy are pinned, never optimized -- must come back exactly what was
    # supplied (the central finding: freeing them overfits badly).
    assert solved.lens.cx == night.true_lens.cx
    assert solved.lens.cy == night.true_lens.cy


def test_recovers_each_frame_rotation(night, solved):
    for frame_solution in solved.frames:
        true_rotation = night.true_rotations[frame_solution.frame_id]
        assert frame_solution.method == "rotation"
        # Rodrigues-vector error in radians; noise-floor precision measured
        # at ~5e-4 rad on this fixture, generous 10x margin below.
        error = np.linalg.norm(frame_solution.rotation - true_rotation)
        assert error < 5e-3


def test_held_out_residual_is_small_across_all_three_radial_thirds(solved):
    for frame_solution in solved.frames:
        bins = frame_solution.held_out_residuals_px
        assert set(bins) == {"center", "middle", "outer"}
        for name, stats in bins.items():
            assert stats["n"] > 0, f"{frame_solution.frame_id}/{name} had no held-out points"
            # Median well under 1 px, p95 under 1 px too -- centroid noise
            # alone is 0.15 px per coordinate on each side of a match.
            assert stats["median_px"] < 1.0
            assert stats["p95_px"] < 1.0


def test_non_converged_solve_raises_never_returns_success(night, flat_initial):
    with pytest.raises(GeometrySolveError):
        solve_night(night.matches, night.frame_shape, flat_initial, max_nfev=1)


def test_homography_fallback_triggers_on_non_rotation_frame(night, flat_initial):
    with_bad_frame = add_non_rotation_frame(night)
    result = solve_night(with_bad_frame.matches, with_bad_frame.frame_shape, flat_initial)

    by_id = {f.frame_id: f for f in result.frames}
    assert by_id["frame_bad"].method == "homography"
    assert by_id["frame_bad"].homography is not None
    assert by_id["frame_bad"].homography.shape == (3, 3)
    assert by_id["frame_bad"].rotation is None
    # frame_bad's held-out residual is still reported, same shape as every
    # rotation-method frame.
    assert set(by_id["frame_bad"].held_out_residuals_px) == {"center", "middle", "outer"}

    # The genuinely-rotating frames are unaffected by the one bad frame.
    for frame_id, true_rotation in night.true_rotations.items():
        assert by_id[frame_id].method == "rotation"
        error = np.linalg.norm(by_id[frame_id].rotation - true_rotation)
        assert error < 5e-3


def test_noisy_rotation_frame_keeps_the_model_that_generalizes(night, flat_initial):
    """A rotation measured badly must not be handed to the homography.

    The rotation gate reads the rotation's TRAIN residual, and an 8-DOF homography can
    always drive a train residual lower than a 3-DOF rotation -- it has the spare freedom
    to bend toward the noise. On a thin, noisy match set that makes the gate select the
    overfitting model. Measured on a real night: every chunk that failed to
    register chose the homography, and chunk 11 -- the single chunk that chose the
    rotation -- produced the best fit of the night (inlier fraction 0.92, 33 outer-third
    inliers). Held-out points, which neither candidate was fitted on, decide it.
    """
    noisy = add_noisy_rotation_frame(night)
    result = solve_night(noisy.matches, noisy.frame_shape, flat_initial)
    frame = {f.frame_id: f for f in result.frames}["frame_noisy"]

    assert frame.train_residual_p95_px > 3.0, (
        "fixture must actually trip the rotation gate, else this proves nothing"
    )
    assert frame.method == "rotation", (
        f"held-out error should have kept the rotation, got {frame.method} "
        f"(train p95 {frame.train_residual_p95_px:.2f} px)"
    )
    assert frame.rotation is not None


def test_frame_with_too_few_matches_raises():
    lens = LensParams(k1=0.0, k2=0.0, cx=99.5, cy=99.5, f_px=200.0)
    sparse_matches = [
        FrameMatches(
            frame_id="frame_sparse",
            ref_xy=np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]),
            frame_xy=np.array([[11.0, 11.0], [21.0, 21.0], [31.0, 31.0]]),
        )
    ]
    with pytest.raises(GeometrySolveError):
        solve_night(sparse_matches, (200, 200), lens)
