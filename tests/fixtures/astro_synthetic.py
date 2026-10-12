"""Synthetic star field rendered through a KNOWN lens + rotation sequence.

Pure math, no real card. Gives ground truth (`LensParams`, per-frame Rodrigues
rotation vectors) that `test_astro_geometry.py` asks `astro_geometry.solve_night`
to recover, plus a held-out split the solver's own gate is scored against.

Geometry convention matches `astro_lens.build_remap` exactly: a star's ray in
the REFERENCE frame's ideal (undistorted) coordinate system is rotated into a
given frame's ray via ``v_source = R_i @ v_output``, projected, then
re-distorted through the SAME shared lens (one physical camera all night) to
land at that frame's own raw pixel position. `ref_xy`/`frame_xy` below are
both in raw (distorted) pixel space, exactly what star registration would
hand the solver.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from gomoveshift.pipeline.photos.astro_geometry import FrameMatches
from gomoveshift.pipeline.photos.astro_lens import LensParams, distort, undistort

# A small synthetic sensor -- large enough for a real radial-thirds spread at
# the corner, small enough that the whole suite stays fast. The lens
# magnitudes below are order-1 in astro_lens's own f_px-normalized
# convention (see its module docstring), not tied to sensor pixel count.
WIDTH = 1200
HEIGHT = 800

# Modelled on a typical rig scale (k1 near +0.24 in astro_lens's forward
# convention, f_px of several thousand pixels), scaled down for this synthetic
# sensor so distortion is still visible at the corner without the fixture
# depending on any real card's numbers.
TRUE_LENS = LensParams(k1=0.20, k2=0.03, cx=(WIDTH - 1) / 2, cy=(HEIGHT - 1) / 2, f_px=900.0)

# Per-match centroid noise (px), applied to BOTH ref_xy and frame_xy
# independently -- realistic star-centroiding jitter, not perfectly
# noise-free synthetic data (that would let a broken solver "converge" onto
# an off-by-construction result and still score zero residual).
CENTROID_NOISE_PX = 0.15


@dataclass(frozen=True)
class SyntheticNight:
    matches: list  # list[FrameMatches]
    frame_shape: tuple  # (height, width)
    true_lens: LensParams
    true_rotations: dict  # frame_id -> Rodrigues 3-vector (np.ndarray)


def _sample_ref_pixels(rng, n_stars, margin=0.15):
    """Ideal-space star positions in the reference frame, spread a bit beyond
    the frame edges (``margin``) so a rotated frame still has plenty of
    in-bounds matches near its own corners."""
    lo_x, hi_x = -margin * WIDTH, WIDTH * (1 + margin)
    lo_y, hi_y = -margin * HEIGHT, HEIGHT * (1 + margin)
    xs = rng.uniform(lo_x, hi_x, n_stars)
    ys = rng.uniform(lo_y, hi_y, n_stars)
    return np.stack([xs, ys], axis=-1)


def _project_to_frame(ref_pixels_ideal, rotation_vec, lens: LensParams):
    """`ref_pixels_ideal` (ideal/undistorted reference coords) -> that
    frame's raw (distorted) pixel coords, mirroring `astro_lens.build_remap`
    (`v_source = R @ v_output`)."""
    xn = (ref_pixels_ideal[..., 0] - lens.cx) / lens.f_px
    yn = (ref_pixels_ideal[..., 1] - lens.cy) / lens.f_px
    rays_output = np.stack([xn, yn, np.ones_like(xn)], axis=-1)
    rotation_matrix = Rotation.from_rotvec(rotation_vec).as_matrix()
    rays_source = rays_output @ rotation_matrix.T
    z = rays_source[..., 2]
    xu_source = rays_source[..., 0] / z
    yu_source = rays_source[..., 1] / z
    ideal_source = np.stack(
        [lens.cx + lens.f_px * xu_source, lens.cy + lens.f_px * yu_source], axis=-1
    )
    return distort(ideal_source, lens)


def build_synthetic_night(
    n_frames=5,
    n_stars=80,
    seed=20260914,
    # 0.08 rad (~4.6 deg) -- large enough to decorrelate the shared lens from
    # each frame's own rotation. On this fixture a near-identity
    # rotation scale (0.01 rad) leaves the shared
    # k1/f_px almost unconstrained -- every k1 start offset converges (a real
    # scipy stopping condition) to a DIFFERENT wrong k1, all with similarly
    # low cost, because tiny per-frame rotations barely probe the lens away
    # from the frame centre. 0.08 rad recovers k1 to within 0.005 and f_px to
    # within 3 px of ground truth.
    rotation_scale_rad=0.08,
    lens=TRUE_LENS,
    centroid_noise_px=CENTROID_NOISE_PX,
):
    """A synthetic night: `n_frames` frames, each with its own rotation
    relative to a fixed reference frame, sharing `lens`. Use
    `add_non_rotation_frame` to append a frame whose true motion is NOT a
    pure rotation (for the homography-fallback test)."""
    rng = np.random.default_rng(seed)
    frame_shape = (HEIGHT, WIDTH)
    ref_pixels_ideal = _sample_ref_pixels(rng, n_stars)
    ref_xy_true = distort(ref_pixels_ideal, lens)

    matches = []
    true_rotations = {}
    for i in range(n_frames):
        frame_id = f"frame_{i:03d}"
        rotation_vec = rng.normal(scale=rotation_scale_rad, size=3)
        true_rotations[frame_id] = rotation_vec
        frame_xy_true = _project_to_frame(ref_pixels_ideal, rotation_vec, lens)

        ref_xy = ref_xy_true + rng.normal(scale=centroid_noise_px, size=ref_xy_true.shape)
        frame_xy = frame_xy_true + rng.normal(scale=centroid_noise_px, size=frame_xy_true.shape)

        # Keep only matches that land within the frame -- a real star match
        # never claims a position off the sensor.
        in_bounds = (
            (frame_xy[:, 0] >= 0)
            & (frame_xy[:, 0] < WIDTH)
            & (frame_xy[:, 1] >= 0)
            & (frame_xy[:, 1] < HEIGHT)
        )
        matches.append(
            FrameMatches(
                frame_id=frame_id,
                ref_xy=ref_xy[in_bounds],
                frame_xy=frame_xy[in_bounds],
            )
        )

    return SyntheticNight(
        matches=matches, frame_shape=frame_shape, true_lens=lens, true_rotations=true_rotations
    )


def add_noisy_rotation_frame(
    night: SyntheticNight,
    frame_id="frame_noisy",
    seed=4242,
    n_stars=40,
    centroid_noise_px=3.0,
):
    """Append a frame whose true motion IS a rotation, measured badly.

    This is the thin, noisy match set a far chunk produces on a real night: the
    geometry is genuinely a rotation on the sphere, but centroid noise alone pushes
    the rotation's TRAIN residual past ``ROTATION_RESIDUAL_GATE_PX``. An 8-DOF
    homography can always drive that train residual lower -- it has the spare degrees
    of freedom to bend toward the noise -- while predicting held-out points worse.
    A gate that reads only the rotation's train residual therefore picks the model
    that fits the noise. The true rotation is returned so a test can check which
    model actually generalizes.
    """
    rng = np.random.default_rng(seed)
    ref_pixels_ideal = _sample_ref_pixels(rng, n_stars)
    rotation_vec = rng.normal(scale=2e-3, size=3)
    ref_xy = distort(ref_pixels_ideal, night.true_lens) + rng.normal(
        scale=centroid_noise_px, size=(n_stars, 2)
    )
    frame_xy = _project_to_frame(ref_pixels_ideal, rotation_vec, night.true_lens) + rng.normal(
        scale=centroid_noise_px, size=(n_stars, 2)
    )
    in_bounds = (
        (frame_xy[:, 0] >= 0)
        & (frame_xy[:, 0] < WIDTH)
        & (frame_xy[:, 1] >= 0)
        & (frame_xy[:, 1] < HEIGHT)
    )
    true_rotations = dict(night.true_rotations)
    true_rotations[frame_id] = rotation_vec
    return SyntheticNight(
        matches=list(night.matches)
        + [FrameMatches(frame_id=frame_id, ref_xy=ref_xy[in_bounds], frame_xy=frame_xy[in_bounds])],
        frame_shape=night.frame_shape,
        true_lens=night.true_lens,
        true_rotations=true_rotations,
    )


def add_non_rotation_frame(night: SyntheticNight, frame_id="frame_bad", seed=99):
    """Append one extra frame whose ref->frame mapping is an arbitrary affine
    warp (independent per-axis scale + shear + translation) instead of a
    rotation on the sphere -- no Rodrigues vector can fit it well, so
    `astro_geometry.solve_night`'s rotation-residual gate must reject it and
    fall back to the 8-DOF homography path.
    """
    rng = np.random.default_rng(seed)
    n_stars = 60
    ref_pixels_ideal = _sample_ref_pixels(rng, n_stars)
    ref_xy = distort(ref_pixels_ideal, night.true_lens) + rng.normal(
        scale=CENTROID_NOISE_PX, size=(n_stars, 2)
    )

    # A non-rigid affine: independent x/y scale plus shear. A pure rotation
    # is always a similarity-like map on the sphere; this deliberately is
    # not expressible as any Rodrigues rotation of the shared lens.
    affine = np.array([[1.15, 0.12], [-0.05, 0.85]])
    translation = np.array([40.0, -25.0])
    ideal_ref_centered = undistort(ref_xy, night.true_lens) - np.array(
        [night.true_lens.cx, night.true_lens.cy]
    )
    ideal_frame_centered = ideal_ref_centered @ affine.T + translation
    ideal_frame = ideal_frame_centered + np.array([night.true_lens.cx, night.true_lens.cy])
    frame_xy_true = distort(ideal_frame, night.true_lens)
    frame_xy = frame_xy_true + rng.normal(scale=CENTROID_NOISE_PX, size=(n_stars, 2))

    width, height = WIDTH, HEIGHT
    in_bounds = (
        (frame_xy[:, 0] >= 0)
        & (frame_xy[:, 0] < width)
        & (frame_xy[:, 1] >= 0)
        & (frame_xy[:, 1] < height)
    )
    new_matches = list(night.matches) + [
        FrameMatches(frame_id=frame_id, ref_xy=ref_xy[in_bounds], frame_xy=frame_xy[in_bounds])
    ]
    return SyntheticNight(
        matches=new_matches,
        frame_shape=night.frame_shape,
        true_lens=night.true_lens,
        true_rotations=dict(night.true_rotations),
    )
