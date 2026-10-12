"""One shared lens fitted across a whole night, one rotation per frame.

Pure array math -- numpy/scipy/cv2 only -- consuming matched star
pairs the way registration hands them off: per frame, `(N, 2)` reference
pixel positions and the SAME stars' `(N, 2)` positions in that frame, both in
each frame's own raw (distorted) pixel grid.

Model: `scipy.optimize.least_squares` (soft_l1 loss) jointly over every
frame's matches at once. Unknowns are the SHARED lens (`k1`, `k2`, `f_px` --
NOT `cx`/`cy`, pinned, see the paragraph below) plus 3 Rodrigues rotation
components per frame. A frame's residuals depend on the shared lens AND that
frame's own rotation, never another frame's -- the Jacobian sparsity passed
to the solver encodes exactly that block structure.

Ray convention mirrors `astro_lens.build_remap` exactly: a star's ray in the
reference frame's ideal (undistorted) coordinate system is rotated into a
given frame's ray via ``v_source = R_i @ v_output``, then re-distorted
through the SAME shared lens to predict that frame's own raw pixel position.
`ref_xy` is therefore always evaluated with `undistort`, `frame_xy` always
with `distort` -- both in astro_lens's OWN forward-distortion convention, not
cv2's inverse one (measured 54 px corner disagreement between the two on the
real card).

Principal point pinned: `cx`/`cy` are never optimized here. Freeing them
on 30 real frames drove the fit 1459-2295 px off frame centre
and TRIPLED the held-out outer-third error (1.10 -> 5.86 px) -- overfitting,
not a discovered centre. `cx`/`cy` on the returned lens are always exactly
what `initial_lens` supplied.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .astro_lens import LensParams, distort, undistort

# Number of shared lens unknowns actually optimized: k1, k2, f_px. cx/cy are
# pinned (see module docstring); adding a k3 term changed the held-out
# result by 0.00 px, so it is not a parameter here either.
N_SHARED_PARAMS = 3

# Rodrigues rotation vector: 3 components per frame.
N_ROTATION_PARAMS = 3

# Generous evaluation cap. Solving ALL
# unknowns jointly (including cx/cy) hits scipy's default max_nfev and
# returns status 0 (exceeded, not converged) with a meaningless answer --
# this module treats status <= 0 as a hard failure for every k1 start (see
# `solve_night`), so the cap here only needs to be generous enough that a
# well-posed pinned-centre solve reaches an actual stopping condition.
DEFAULT_MAX_NFEV = 5000

# Rotation-model gate (PROVISIONAL, no multi-night baseline exists yet): a
# frame whose rotation-only fit can't get its own TRAIN residual p95 under
# this many px is not well described by a pure rotation (e.g. a tripod bump
# mid-night) and falls back to a general 8-DOF homography instead. Centroid
# noise in this module's own synthetic fixture is ~0.15 px; a genuinely
# non-rotational frame's rotation-fit residual is measured in the tens of px
# (see `add_non_rotation_frame`), so this gate has wide margin either way.
ROTATION_RESIDUAL_GATE_PX = 3.0

# A rotation has 3 DOF; with the shared lens itself constrained by every
# other frame, 2 matched pairs is already an exact solve for one frame's
# rotation alone. Demanding real redundancy on BOTH the train and held-out
# halves (this module always splits even/odd) means at least this many
# matches before a frame is even attempted.
MIN_MATCHES_PER_FRAME = 8

# k1 starting offsets (relative to `initial_lens.k1`), swept because this
# joint cost surface has real local minima the trust-region solver can get
# stuck in: measured on this module's own synthetic fixture, starting flat
# at k1=0 converges (a real scipy stopping condition, `status > 0`) to a
# WRONG minimum -- k1 off by 0.22, f_px off by 45 px, held-out residual
# several px -- while sweeping this offset ladder and keeping the lowest-cost
# converged result lands on the true parameters (sub-0.001 k1 error). Spans
# the sign and rough magnitude the module docstring expects for
# this rig (k1 order 0.01-0.3).
K1_START_OFFSETS = (0.0, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3)


class GeometrySolveError(RuntimeError):
    """The joint solve did not converge, or a frame had too few matches."""


@dataclass(frozen=True)
class FrameMatches:
    """One frame's matched star pairs against the fixed reference frame.

    ``ref_xy``/``frame_xy``: ``(N, 2)`` float arrays, raw (distorted) pixel
    coordinates in the reference frame's own grid and this frame's own grid
    respectively -- exactly what star registration produces.
    """

    frame_id: str
    ref_xy: np.ndarray
    frame_xy: np.ndarray


@dataclass(frozen=True)
class FrameSolution:
    """One frame's fitted geometry plus its held-out (odd-indexed) residual."""

    frame_id: str
    method: str  # "rotation" or "homography"
    held_out_residuals_px: dict = field(default_factory=dict)
    rotation: np.ndarray | None = None  # Rodrigues 3-vector, when method == "rotation"
    homography: np.ndarray | None = None  # 3x3, when method == "homography"
    train_residual_p95_px: float = 0.0


@dataclass(frozen=True)
class SolveResult:
    lens: LensParams
    frames: list
    converged: bool
    cost: float


def _split_train_holdout(n):
    idx = np.arange(n)
    return idx[idx % 2 == 0], idx[idx % 2 == 1]


def _rays_from_ref(ref_xy, lens: LensParams):
    """Reference raw pixels -> unit-scale pinhole rays (``z = 1``) in the
    reference frame's ideal coordinate system."""
    ideal_ref = undistort(ref_xy, lens)
    xn = (ideal_ref[..., 0] - lens.cx) / lens.f_px
    yn = (ideal_ref[..., 1] - lens.cy) / lens.f_px
    return np.stack([xn, yn, np.ones_like(xn)], axis=-1)


def _predict_frame_xy(rays_output, rotation_vec, lens: LensParams):
    """`build_remap`'s convention: ``v_source = R @ v_output``, projected and
    re-distorted through the SAME shared lens, to predict raw pixel
    positions in the frame the rotation belongs to."""
    rotation_matrix = Rotation.from_rotvec(rotation_vec).as_matrix()
    rays_source = rays_output @ rotation_matrix.T
    z = rays_source[..., 2]
    xu_source = rays_source[..., 0] / z
    yu_source = rays_source[..., 1] / z
    ideal_source = np.stack(
        [lens.cx + lens.f_px * xu_source, lens.cy + lens.f_px * yu_source], axis=-1
    )
    return distort(ideal_source, lens)


def _lens_from_params(shared, initial_lens: LensParams):
    k1, k2, f_px = shared
    return LensParams(k1=k1, k2=k2, cx=initial_lens.cx, cy=initial_lens.cy, f_px=f_px)


def _residuals(params, train_ref_xy, train_frame_xy, initial_lens):
    # `undistort` here uses the CURRENT (evolving) shared k1/k2, not the
    # starting guess -- both sides of the model (reference ray in, predicted
    # frame pixel out) must share the SAME lens estimate at every iteration,
    # or k1/k2 only ever influence the "distort" half of the round trip and
    # the fit becomes only weakly sensitive to it. Measured: precomputing
    # rays once from the initial (typically k1=k2=0) lens converged to a
    # wrong minimum every time; recomputing them here from `lens` recovers
    # the true parameters (see `K1_START_OFFSETS` for the remaining
    # multi-start need on top of this fix).
    shared = params[:N_SHARED_PARAMS]
    lens = _lens_from_params(shared, initial_lens)
    out = []
    offset = N_SHARED_PARAMS
    for ref_xy, target in zip(train_ref_xy, train_frame_xy):
        rotation_vec = params[offset : offset + N_ROTATION_PARAMS]
        offset += N_ROTATION_PARAMS
        rays = _rays_from_ref(ref_xy, lens)
        predicted = _predict_frame_xy(rays, rotation_vec, lens)
        out.append((predicted - target).ravel())
    return np.concatenate(out)


def _build_jac_sparsity(train_ref_xy):
    """Each frame's residuals depend only on the shared lens columns and
    that frame's own 3 rotation columns -- never another frame's."""
    n_frames = len(train_ref_xy)
    n_params = N_SHARED_PARAMS + n_frames * N_ROTATION_PARAMS
    n_residuals = sum(2 * ref_xy.shape[0] for ref_xy in train_ref_xy)
    sparsity = lil_matrix((n_residuals, n_params), dtype=np.int8)
    row = 0
    for i, ref_xy in enumerate(train_ref_xy):
        n_rows = 2 * ref_xy.shape[0]
        sparsity[row : row + n_rows, :N_SHARED_PARAMS] = 1
        col = N_SHARED_PARAMS + i * N_ROTATION_PARAMS
        sparsity[row : row + n_rows, col : col + N_ROTATION_PARAMS] = 1
        row += n_rows
    return sparsity.tocsr()


def _radial_bins(xy, frame_shape):
    """Mirrors `astro_register.radial_inlier_residuals_px`'s banding: radius
    from frame centre, normalized by centre-to-corner distance, thirds named
    center/middle/outer."""
    height, width = frame_shape[:2]
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    max_radius = np.linalg.norm(center)
    radius = np.linalg.norm(np.asarray(xy) - center, axis=1) / max(max_radius, 1.0)
    return {
        "center": radius < 1 / 3,
        "middle": (radius >= 1 / 3) & (radius < 2 / 3),
        "outer": radius >= 2 / 3,
    }


def _bin_stats(residual_norms, bins):
    result = {}
    for name, selected in bins.items():
        values = residual_norms[selected]
        result[name] = {
            "n": int(values.size),
            "median_px": float(np.median(values)) if values.size else None,
            "p95_px": float(np.percentile(values, 95)) if values.size else None,
        }
    return result


def _fit_homography(ref_xy, frame_xy, lens: LensParams):
    """8-DOF homography between undistorted reference and frame coordinates
    -- the fallback for a frame the rotation-on-the-sphere model can't fit.
    """
    ideal_ref = undistort(ref_xy, lens).astype(np.float32)
    ideal_frame = undistort(frame_xy, lens).astype(np.float32)
    homography, _ = cv2.findHomography(ideal_ref, ideal_frame, method=0)
    if homography is None:
        raise GeometrySolveError("homography fallback did not converge (degenerate point set)")
    return homography.astype(np.float64)


def _apply_homography(ref_xy, homography, lens: LensParams):
    ideal_ref = undistort(ref_xy, lens)
    ones = np.ones((ideal_ref.shape[0], 1))
    homogeneous = np.concatenate([ideal_ref, ones], axis=1) @ homography.T
    ideal_frame = homogeneous[:, :2] / homogeneous[:, 2:3]
    return distort(ideal_frame, lens)


def solve_night(
    matches,
    frame_shape,
    initial_lens: LensParams,
    rotation_gate_p95_px=ROTATION_RESIDUAL_GATE_PX,
    max_nfev=DEFAULT_MAX_NFEV,
):
    """Jointly fit one shared lens (k1, k2, f_px; cx/cy pinned) plus one
    Rodrigues rotation per frame, over every frame's matches at once.

    ``matches``: list of `FrameMatches`. ``initial_lens``: starting point
    (and the pinned cx/cy) -- typically `LensParams.from_exif(...)`.

    Raises `GeometrySolveError` if the solve does not converge, or if any
    frame has fewer than `MIN_MATCHES_PER_FRAME` matches. Returns a
    `SolveResult` whose lens always carries `initial_lens`'s cx/cy unchanged.
    """
    for m in matches:
        if len(m.ref_xy) < MIN_MATCHES_PER_FRAME:
            raise GeometrySolveError(
                f"frame {m.frame_id!r} has {len(m.ref_xy)} matches, need >= {MIN_MATCHES_PER_FRAME}"
            )

    train_idx = []
    holdout_idx = []
    train_ref_xy = []
    train_targets = []
    for m in matches:
        tr, ho = _split_train_holdout(len(m.ref_xy))
        train_idx.append(tr)
        holdout_idx.append(ho)
        train_ref_xy.append(m.ref_xy[tr])
        train_targets.append(m.frame_xy[tr])

    n_frames = len(matches)
    jac_sparsity = _build_jac_sparsity(train_ref_xy)

    best_result = None
    for offset in K1_START_OFFSETS:
        x0 = np.concatenate(
            [
                [initial_lens.k1 + offset, initial_lens.k2, initial_lens.f_px],
                np.zeros(n_frames * N_ROTATION_PARAMS),
            ]
        )
        result = least_squares(
            _residuals,
            x0,
            args=(train_ref_xy, train_targets, initial_lens),
            loss="soft_l1",
            method="trf",
            tr_solver="lsmr",
            jac_sparsity=jac_sparsity,
            max_nfev=max_nfev,
            x_scale="jac",
        )
        if result.status <= 0:
            continue
        if best_result is None or result.cost < best_result.cost:
            best_result = result
    if best_result is None:
        raise GeometrySolveError(
            f"least_squares did not converge from any k1 start offset ({K1_START_OFFSETS!r})"
        )
    result = best_result

    shared = result.x[:N_SHARED_PARAMS]
    fitted_lens = _lens_from_params(shared, initial_lens)

    frames = []
    offset = N_SHARED_PARAMS
    for i, m in enumerate(matches):
        rotation_vec = result.x[offset : offset + N_ROTATION_PARAMS]
        offset += N_ROTATION_PARAMS

        tr, ho = train_idx[i], holdout_idx[i]
        train_rays_fitted = _rays_from_ref(train_ref_xy[i], fitted_lens)
        train_predicted = _predict_frame_xy(train_rays_fitted, rotation_vec, fitted_lens)
        train_residual_norms = np.linalg.norm(train_predicted - m.frame_xy[tr], axis=1)
        train_p95 = (
            float(np.percentile(train_residual_norms, 95)) if train_residual_norms.size else 0.0
        )

        holdout_ref = m.ref_xy[ho]
        holdout_frame = m.frame_xy[ho]

        holdout_bins = _radial_bins(holdout_frame, frame_shape)
        holdout_rays = _rays_from_ref(holdout_ref, fitted_lens)
        rotation_holdout = np.linalg.norm(
            _predict_frame_xy(holdout_rays, rotation_vec, fitted_lens) - holdout_frame, axis=1
        )

        if train_p95 <= rotation_gate_p95_px:
            frames.append(
                FrameSolution(
                    frame_id=m.frame_id,
                    method="rotation",
                    rotation=rotation_vec,
                    train_residual_p95_px=train_p95,
                    held_out_residuals_px=_bin_stats(rotation_holdout, holdout_bins),
                )
            )
            continue

        # The rotation missed its TRAIN gate, which is not by itself evidence that a
        # homography describes this frame better. An 8-DOF model can always drive a train
        # residual below a 3-DOF one -- it has the spare freedom to bend toward centroid
        # noise -- so selecting on the rotation's train residual alone hands every thin,
        # noisy match set to the model that overfits it. Measured on a real night:
        # every chunk that failed to register had been described with a homography, while
        # the one chunk described with a rotation produced the best fit of the night.
        # Both candidates are therefore scored on the held-out split, which neither was
        # fitted on, and the better one is kept. That split is a validation split for this
        # decision -- it arbitrates the choice as well as reporting it -- and the reported
        # residuals are the winner's own held-out numbers, never the train numbers.
        homography = _fit_homography(m.ref_xy[tr], m.frame_xy[tr], fitted_lens)
        homography_holdout = np.linalg.norm(
            _apply_homography(holdout_ref, homography, fitted_lens) - holdout_frame, axis=1
        )
        rotation_wins = rotation_holdout.size > 0 and float(
            np.percentile(rotation_holdout, 95)
        ) <= float(np.percentile(homography_holdout, 95))
        frames.append(
            FrameSolution(
                frame_id=m.frame_id,
                method="rotation" if rotation_wins else "homography",
                rotation=rotation_vec if rotation_wins else None,
                homography=None if rotation_wins else homography,
                train_residual_p95_px=train_p95,
                held_out_residuals_px=_bin_stats(
                    rotation_holdout if rotation_wins else homography_holdout, holdout_bins
                ),
            )
        )

    return SolveResult(lens=fitted_lens, frames=frames, converged=True, cost=float(result.cost))
