"""Pure validation and planning primitives for bounded-memory astro deep combines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

import cv2
import numpy as np

from .astro_background import mesh_background
from .astro_geometry import (
    FrameMatches,
    GeometrySolveError,
    _apply_homography,  # astro_geometry's own undistort -> homography -> re-distort composition
    solve_night,
)
from .astro_lens import LensParams, build_remap
from .astro_linear import LinearArtifactError, load_linear_artifact
from .astro_noise_gate import check_noise_floor, noise_sample_boxes
from .astro_quality import A3_RADIAL_GROWTH_BOUNDARY, a3_by_radial_third
from .astro_register import (
    MIN_INLIER_FRACTION,
    MIN_INLIER_STARS,
    MIN_OUTER_THIRD_INLIERS,
    RANSAC_REPROJ_THRESHOLD_PX,
    RegistrationError,
    RegistrationResult,
    _is_remap_pair,  # the predicate warp_to_reference itself dispatches on -- never a second copy
    coarse_shift,
    match_progressive,
    radial_inlier_residuals_px,
    register,
    warp_to_reference,
)
from .astro_stack import (
    MIN_STACK_CONTRIBUTORS,
    SHORT_STACK_KAPPA_BY_N,
    SHORT_STACK_MIN_REJECT_FRAMES,
    blend,
    horizon_mask,
)
from .astro_triage import STAR_DETECT_SIGMA, detect_stars

MIN_DEEP_CHUNKS = 2
_CAPTURE_IDENTITY_FIELDS = ("camera", "raw_shape", "cfa_pattern", "iso", "exposure_s")


class AstroCombineError(ValueError):
    """The proposed artifact set cannot produce an honest deep combine."""


@dataclass(frozen=True)
class ValidatedArtifact:
    """Small validated descriptor; full payload arrays are reopened only when streamed."""

    path: Path
    manifest: dict

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.manifest["payloads"]["sky_sum.npy"]["shape"])


@dataclass(frozen=True)
class ValidatedArtifactSet:
    artifacts: tuple[ValidatedArtifact, ...]
    reference_index: int

    @property
    def reference(self) -> ValidatedArtifact:
        return self.artifacts[self.reference_index]


@dataclass(frozen=True)
class ArtifactTransform:
    """One artifact-to-global-reference transform and the geometry path that produced it.

    Exactly one of ``transform`` (2x3 affine) and ``remap`` (a ``(map_x, map_y)`` pair from
    ``astro_lens.build_remap``) describes the geometry; ``warp`` is whichever it is. The
    lens+rotation model (``fit_matched_pairs``) has no matrix to report, so its ``transform``
    is ``None`` rather than an invented affine -- what fit the pair is in ``diagnostics``
    under ``geometry_method``. Keeping the remap out of ``transform`` also keeps the deep
    sidecar's per-chunk ``transform.matrix`` a matrix or null, never two full-frame
    resampling grids serialized as JSON.
    """

    transform: np.ndarray | None
    method: str
    links: tuple[tuple[int, int], ...]
    diagnostics: tuple[dict, ...] = ()
    depth: int = 0  # composed links between this artifact and the reference; 0 = the reference.
    remap: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def warp(self) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """What ``_warped_components`` applies -- the remap when there is one, else the 2x3."""
        return self.transform if self.remap is None else self.remap


@dataclass(frozen=True)
class ChunkPhotometry:
    offset: float
    scale: float


@dataclass(frozen=True)
class DeepSkyResult:
    sky_sum: np.ndarray
    sky_count: np.ndarray
    photometry: tuple[ChunkPhotometry, ...]
    rejected_pixel_fraction: float
    # Per-chunk crops of the sample boxes, harvested from the second pass at no extra I/O cost.
    # {box: (list_of_chunk_means, list_of_chunk_counts)} -- the input the noise gate needs.
    noise_samples: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DeepGroundResult:
    ground: np.ndarray
    weight: np.ndarray
    photometry: tuple[ChunkPhotometry, ...]
    rejected_pixel_fraction: float


# The deep product ships the region that passes its own star-shape gate, not the whole sensor.
# With per-artifact polynomial placement the outer radial
# third grows by more than 50% against a 10% A3 boundary at every useful depth,
# while centre+middle pass with margin.  Keeping r <= (2/3)*rmax at the
# frame's own aspect ratio is exactly the central two thirds of each axis, and is the LARGEST
# rectangle wholly inside the passing region -- so nothing that passes is discarded.
# On a 5600x3728 sensor that is 3732x2484, 9.3 MP of 20.9 MP.  Shipping the full frame instead
# would ship a default output that fails the gate the product is measured against.
SCOPED_FRAME_FRACTION = 2.0 / 3.0
# Concrete crop geometries `scoped_frame_slice` knows how to cut. "auto" (below) is a THIRD,
# caller-facing choice that never reaches this function directly -- it is resolved to one of
# these two by `resolve_auto_scope` first.
OUTPUT_SCOPES = ("scoped", "full")
AUTO_OUTPUT_SCOPE = "auto"
#: Every scope `deep_combine`'s `scope=` argument accepts.
CLI_OUTPUT_SCOPES = OUTPUT_SCOPES + (AUTO_OUTPUT_SCOPE,)
# auto replaces "scoped" as the default -- it measures the actual frame
# instead of assuming the worst case, and ships the full sensor whenever that measurement
# proves it clean. "scoped" is kept only as auto's own fallback rectangle (see
# `resolve_auto_scope`) and as an explicit opt-out.
DEFAULT_OUTPUT_SCOPE = AUTO_OUTPUT_SCOPE
# Same weights `fit_artifact_pair` uses to build a luma plane for star detection.
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def scoped_frame_slice(shape, scope: str = "scoped") -> tuple[slice, slice]:
    """Row/column slices selecting the shipped region of a full-frame deep product.

    Only understands the two concrete crop geometries in ``OUTPUT_SCOPES`` -- "auto" is
    resolved to one of them by ``resolve_auto_scope`` before this function is ever called."""
    if scope not in OUTPUT_SCOPES:
        raise ValueError(f"unknown output scope {scope!r}; expected one of {OUTPUT_SCOPES}")
    height, width = int(shape[0]), int(shape[1])
    if scope == "full":
        return slice(0, height), slice(0, width)
    keep_h = int(height * SCOPED_FRAME_FRACTION)
    keep_w = int(width * SCOPED_FRAME_FRACTION)
    # Force the retained extent to the same parity as the frame so the crop is exactly centred.
    # With an odd margin the box sits half a pixel off centre, which pushes its far corner
    # OUTSIDE r = (2/3)*rmax -- i.e. into the zone that fails A3, defeating the point of the
    # crop.  Measured: 2485x3733 on a 3728x5600 frame puts a corner at r/rmax 0.66674.
    if (height - keep_h) % 2:
        keep_h -= 1
    if (width - keep_w) % 2:
        keep_w -= 1
    return (
        slice((height - keep_h) // 2, (height - keep_h) // 2 + keep_h),
        slice((width - keep_w) // 2, (width - keep_w) // 2 + keep_w),
    )


def _frame_median_fwhm_px(artifacts: Sequence[ValidatedArtifact]) -> float | None:
    """Median per-frame star FWHM feeding this combine, pooled across every artifact's
    ``registration_provenance[].triage.fwhm_px`` (astro_orchestrator's per-frame triage,
    additive since linear artifact schema 2 -- see astro_linear.py). ``None`` if not one
    artifact carries it (schema 1, or a schema-2 artifact predating triage), so a caller can
    tell "no stars measured" apart from "measured, extremely sharp"."""
    values = [
        entry["triage"]["fwhm_px"]
        for artifact in artifacts
        for entry in artifact.manifest.get("registration_provenance", ())
        if isinstance(entry, dict)
        and isinstance(entry.get("triage"), dict)
        and entry["triage"].get("fwhm_px") is not None
    ]
    return float(median(values)) if values else None


def resolve_auto_scope(rgb: np.ndarray, frame_median_fwhm_px: float | None) -> tuple[str, dict]:
    """Decide what ``scope="auto"`` ships: measure A3-by-radial-third (``astro_quality
    .a3_by_radial_third``) on the blended FULL-frame RGB -- BEFORE any crop, so the
    measurement sees exactly the corners a crop would otherwise hide -- and ship the whole
    sensor only when its OUTER third measurably passes the gate already pinned
    (``A3_RADIAL_GROWTH_BOUNDARY``). That is the largest rectangle wholly inside the passing
    region: every pixel is closer to centre than the worst corner, so if the worst corner
    passes, nothing shipped can fail. Anything else -- outer fails, or no reference FWHM
    exists to grow against, so a pass can never be proven -- falls back to the scoped
    rectangle, which was measured to pass at every useful depth.

    No new threshold: this only decides WHICH of the three radial bands gates a full-frame
    ship (the outer one, because a full-frame ship is only as good as its worst corner)."""
    luma = np.sum(np.asarray(rgb, dtype=np.float32) * _LUMA_WEIGHTS, axis=2)
    outer = a3_by_radial_third(luma, frame_median_fwhm_px)["outer"]
    passed = outer.passed is True
    decision = {
        "requested_scope": AUTO_OUTPUT_SCOPE,
        "resolved_scope": "full" if passed else "scoped",
        "gate": "astro_quality.a3_by_radial_third.outer",
        "boundary": A3_RADIAL_GROWTH_BOUNDARY,
        "frame_median_fwhm_px": frame_median_fwhm_px,
        "outer_fwhm_px": outer.fwhm_px,
        "outer_growth_fraction": outer.growth_fraction,
        "outer_star_count": outer.n,
        "outer_passed": outer.passed,
    }
    return decision["resolved_scope"], decision


@dataclass(frozen=True)
class DeepCoverage:
    nominal_contributors: int
    floor: float
    min_contributors: float
    median_contributors: float
    below_floor_fraction: float


@dataclass(frozen=True)
class DeepCombineResult:
    rgb: np.ndarray
    mask: np.ndarray
    scope: str
    frame_origin: tuple[int, int]
    sky: DeepSkyResult
    ground: DeepGroundResult
    transforms: tuple[ArtifactTransform, ...]
    coverage: DeepCoverage
    artifacts: ValidatedArtifactSet
    noise_floor: object | None = None
    # The scope the caller asked for ("auto"/"full"/"scoped") -- `scope` above is always the
    # RESOLVED, concrete geometry actually shipped. `requested_scope` defaults to None (never
    # "auto" without one) so a result built directly still serialises sensibly.
    requested_scope: str | None = None
    # `resolve_auto_scope`'s full decision record, only populated when `requested_scope`
    # was "auto" -- an explicit "full"/"scoped" request never runs the measurement.
    scope_decision: dict | None = None


def _sky_mean(sky_sum: np.ndarray, sky_count: np.ndarray) -> np.ndarray:
    out = np.zeros(sky_sum.shape, dtype=np.float32)
    np.divide(
        sky_sum,
        np.asarray(sky_count, dtype=np.float32)[..., None],
        out=out,
        where=np.asarray(sky_count)[..., None] > 0,
    )
    return out


# --- lens-aware chunk-to-chunk registration -----------------------------------------
#
# A chunk pair is separated by minutes of sky, and `astro_register.register`'s only
# estimator is a 4-DOF similarity (rotation, uniform scale, translation). That model has
# no term that can absorb Brown radial distortion, so its fit residual grows monotonically
# with radius and RANSAC labels every corner star an outlier. On a wide-field stack the
# centre and middle stars still match at about a pixel median while the matched
# outer-third pairs do not. MIN_OUTER_THIRD_INLIERS is right to refuse such a fit;
# the MODEL is what is wrong. `astro_geometry
# .solve_night` fits the physically correct one -- undistort, 3-DOF rotation on the sphere,
# re-distort, with the lens solved jointly -- and `astro_lens.build_remap` composes it into
# the (map_x, map_y) pair `astro_register.warp_to_reference` already knows how to apply.

# Seed focal length for the chunk-pair solve, as a fraction of the frame's pixel width:
# the rig astro_lens documents (Nikon Z DX 18-140) at its wide end, 18 mm on a 23.5 mm-wide
# DX sensor -- the same rig fact the per-frame stacking path seeds its own solve with. A
# STARTING POINT only: solve_night fits f_px itself, and a seed that sends the fit
# somewhere wrong is caught by the inlier/residual gates below, which then refuse the fit.
LENS_SEED_F_PX_PER_FRAME_WIDTH = 18.0 / 23.5


#: `solve_night` describes a frame with a rotation on the sphere or, when no Rodrigues
#: vector fits it (a tripod that shifted as well as turned), an 8-DOF homography. BOTH are
#: lens-aware results: `astro_geometry` fits that homography in UNDISTORTED space, after the
#: Brown model has been removed, and a sky rotation IS a homography in gnomonic coordinates
#: -- so accepting it is the module's own documented contract, not a weakening. It is NOT
#: the homography inside `match_progressive`, which that function's docstring rightly
#: forbids as a registration transform: THAT one is fitted in raw DISTORTED pixels and
#: exists only to decide where the next matching stage looks. Do not merge the two.
GEOMETRY_METHOD_BY_SOLUTION = {"rotation": "lens_rotation", "homography": "lens_homography"}
#: Reserved for a genuine non-convergence -- never for a method this module simply declined.
SIMILARITY_FALLBACK_METHOD = "similarity_fallback"


class PairFitError(RegistrationError):
    """A refused chunk-pair fit that names WHICH model was refused.

    ``geometry_method`` is the model the gates actually measured, and ``fallback_reason``
    is set only when this pair had already been downgraded to the similarity -- so the
    planner's abort message can say a downgrade happened, at the point it kills the run.
    """

    def __init__(self, message: str, *, geometry_method: str, fallback_reason: str | None = None):
        super().__init__(message)
        self.geometry_method = geometry_method
        self.fallback_reason = fallback_reason


@dataclass(frozen=True)
class LensPairFit(RegistrationResult):
    """A chunk-pair fit, plus WHICH geometry model produced it.

    ``transform`` is a ``(map_x, map_y)`` remap pair for either lens-aware model
    (``"lens_rotation"`` from ``astro_lens.build_remap``, ``"lens_homography"`` from the
    same composition around ``astro_geometry``'s homography), and the old 2x3 similarity
    when ``geometry_method`` is ``"similarity_fallback"`` -- in which case
    ``fallback_reason`` names why the lens solve was refused. Every other field carries the
    fitted model's own residuals, so the gates that already run against a
    `RegistrationResult` measure what will actually be warped.
    """

    geometry_method: str = "lens_rotation"
    fallback_reason: str | None = None
    # `solve_night`'s odd-indexed held-out residuals by radial third (even-indexed pairs
    # train), i.e. out-of-sample evidence beside this fit's own in-sample numbers.
    held_out_residuals_px: dict | None = None


def _remap_at(maps, xy: np.ndarray) -> np.ndarray:
    """Where ``maps`` sends each reference position ``xy``, sampled from the SAME remap the
    pixels will be warped through rather than re-deriving the model's forward math."""
    map_x, map_y = maps
    points = np.ascontiguousarray(xy, dtype=np.float32).reshape(-1, 1, 2)
    sampled = [
        cv2.remap(
            np.asarray(plane, dtype=np.float32),
            np.ascontiguousarray(points[..., 0]),
            np.ascontiguousarray(points[..., 1]),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1)
        for plane in (map_x, map_y)
    ]
    return np.stack(sampled, axis=-1).astype(np.float64)


def _homography_remap(shape, lens: LensParams, homography: np.ndarray):
    """``build_remap``'s composition for the homography branch: undistort each OUTPUT pixel,
    apply the model, re-distort into the source frame's raw grid.

    The middle step is ``astro_geometry``'s own ``_apply_homography`` -- the same call
    ``solve_night`` measured this solution's held-out residuals with -- so the maps that
    warp the pixels cannot drift from the numbers that accepted the fit.
    """
    height, width = int(shape[0]), int(shape[1])
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=-1)
    source = _apply_homography(grid, homography, lens)
    return (
        source[:, 0].reshape(height, width).astype(np.float32),
        source[:, 1].reshape(height, width).astype(np.float32),
    )


def fit_matched_pairs(
    matched_ref_xy: np.ndarray,
    matched_frame_xy: np.ndarray,
    frame_shape,
    *,
    match_yield: float = 0.0,
    match_yield_by_stage: dict | None = None,
) -> LensPairFit:
    """Fit one chunk pair with the shared-lens model, gated on its own residuals.

    Both of ``solve_night``'s answers are accepted (see ``GEOMETRY_METHOD_BY_SOLUTION``):
    a rotation on the sphere, or the homography it falls back to for a pair no Rodrigues
    vector describes. Raises ``GeometrySolveError`` only when the joint solve genuinely
    does not converge, and ``PairFitError`` when a fitted model clears the solver but not
    the SAME inlier floors ``register`` applies -- those floors are never relaxed here,
    only measured against a model that can actually reach the sensor's corners.
    """
    ref_xy = np.asarray(matched_ref_xy, dtype=np.float64)
    frame_xy = np.asarray(matched_frame_xy, dtype=np.float64)
    height, width = int(frame_shape[0]), int(frame_shape[1])
    seed = LensParams(
        k1=0.0,
        k2=0.0,
        cx=(width - 1) / 2,
        cy=(height - 1) / 2,
        f_px=LENS_SEED_F_PX_PER_FRAME_WIDTH * width,
    )
    solved = solve_night(
        [FrameMatches(frame_id="moving", ref_xy=ref_xy, frame_xy=frame_xy)], (height, width), seed
    )
    solution = solved.frames[0]
    geometry_method = GEOMETRY_METHOD_BY_SOLUTION.get(solution.method)
    if geometry_method is None:
        raise GeometrySolveError(
            f"the joint fit described this chunk pair with an unknown model {solution.method!r}"
        )
    maps = (
        build_remap((height, width), solved.lens, solution.rotation)
        if solution.method == "rotation"
        else _homography_remap((height, width), solved.lens, solution.homography)
    )
    predicted = _remap_at(maps, ref_xy)
    residuals = np.linalg.norm(predicted - frame_xy, axis=1)
    inliers = residuals <= RANSAC_REPROJ_THRESHOLD_PX
    n_inliers = int(inliers.sum())
    if n_inliers < MIN_INLIER_STARS:
        raise PairFitError(
            f"fit kept only {n_inliers} inliers, need >= {MIN_INLIER_STARS}",
            geometry_method=geometry_method,
        )
    inlier_fraction = n_inliers / len(residuals)
    if inlier_fraction < MIN_INLIER_FRACTION:
        raise PairFitError(
            f"fit inlier fraction {inlier_fraction:.2f} below floor {MIN_INLIER_FRACTION}",
            geometry_method=geometry_method,
        )
    # The predictions are already in the moving chunk's own pixel grid, so the identity is
    # what turns `radial_inlier_residuals_px` into a pure banding of THIS model's residuals
    # -- same radial thirds, same statistics, no second copy of either.
    radial = radial_inlier_residuals_px(
        np.eye(2, 3, dtype=np.float64), frame_xy, predicted, inliers, (height, width)
    )
    outer_n = radial["outer"]["n"]
    if outer_n < MIN_OUTER_THIRD_INLIERS:
        raise PairFitError(
            f"fit kept only {outer_n} outer-third inliers, need >= {MIN_OUTER_THIRD_INLIERS} "
            "-- this transform only fit the frame's centre, not the whole sensor",
            geometry_method=geometry_method,
        )
    kept = residuals[inliers]
    return LensPairFit(
        transform=maps,
        n_matched=len(residuals),
        inlier_fraction=inlier_fraction,
        residual_median_px=float(np.median(kept)),
        residual_p95_px=float(np.percentile(kept, 95)),
        match_yield=match_yield,
        radial_inlier_residuals_px=radial,
        match_yield_by_stage=match_yield_by_stage,
        geometry_method=geometry_method,
        held_out_residuals_px=solution.held_out_residuals_px,
    )


def fit_lens_aware(
    luma_ref: np.ndarray,
    stars_ref: np.ndarray,
    luma_moving: np.ndarray,
    stars_moving: np.ndarray,
    sky_mask: np.ndarray | None = None,
) -> LensPairFit:
    """Match two chunk sky means' stars, then fit them with the lens+rotation model.

    Matching is ``register``'s own cascade (coarse prior, then ``match_progressive``)
    unchanged -- only the MODEL fitted to the resulting pairs is new. A solve that does not
    converge at all (``GeometrySolveError``) falls back to ``register``'s similarity,
    reported as ``geometry_method="similarity_fallback"`` with the refusal text -- on the
    result when the similarity passes its gates, and on the raised ``PairFitError`` when it
    does not, because a downgrade is invisible exactly where it matters otherwise: the
    message that kills the run. The fallback re-runs ``register`` rather than reusing the
    pairs above so the similarity path stays byte-for-byte the one this service already
    ships, gates included.
    """
    prior_shift = coarse_shift(luma_ref, luma_moving, sky_mask=sky_mask)
    frame_shape = luma_ref.shape[:2]
    pairs, stage_yields = match_progressive(stars_ref, stars_moving, prior_shift, frame_shape)
    matched_ref_xy = stars_ref[pairs[:, 0], :2] if len(pairs) else pairs
    matched_moving_xy = stars_moving[pairs[:, 1], :2] if len(pairs) else pairs
    smaller_detected = min(len(stars_ref), len(stars_moving))
    match_yield = len(pairs) / smaller_detected if smaller_detected else 0.0
    try:
        return fit_matched_pairs(
            matched_ref_xy,
            matched_moving_xy,
            frame_shape,
            match_yield=match_yield,
            match_yield_by_stage=stage_yields,
        )
    except GeometrySolveError as exc:
        try:
            fallback = register(luma_ref, stars_ref, luma_moving, stars_moving, sky_mask=sky_mask)
        except RegistrationError as refused:
            raise PairFitError(
                str(refused),
                geometry_method=SIMILARITY_FALLBACK_METHOD,
                fallback_reason=str(exc),
            ) from refused
        return LensPairFit(
            transform=fallback.transform,
            n_matched=fallback.n_matched,
            inlier_fraction=fallback.inlier_fraction,
            residual_median_px=fallback.residual_median_px,
            residual_p95_px=fallback.residual_p95_px,
            match_yield=fallback.match_yield,
            radial_inlier_residuals_px=fallback.radial_inlier_residuals_px,
            match_yield_by_stage=fallback.match_yield_by_stage,
            geometry_method=SIMILARITY_FALLBACK_METHOD,
            fallback_reason=str(exc),
        )


def fit_artifact_pair(
    moving: ValidatedArtifact,
    reference: ValidatedArtifact,
    *,
    loader=load_linear_artifact,
) -> RegistrationResult:
    """Fit one artifact sky mean into another under their persisted common sky boundary."""
    moving_loaded = loader(moving.path)
    moving_mean = _sky_mean(moving_loaded.sky_sum, moving_loaded.sky_count)
    del moving_loaded
    reference_loaded = loader(reference.path)
    reference_mean = _sky_mean(reference_loaded.sky_sum, reference_loaded.sky_count)
    del reference_loaded
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    moving_luma = np.sum(moving_mean * weights, axis=2, dtype=np.float32)
    reference_luma = np.sum(reference_mean * weights, axis=2, dtype=np.float32)
    boundaries = [
        float(artifact.manifest["mask_provenance"]["resolved_boundary_row"])
        for artifact in (moving, reference)
    ]
    boundary = int(np.clip(round(min(boundaries)), 0, reference_luma.shape[0]))
    sky_mask = np.broadcast_to(
        np.arange(reference_luma.shape[0])[:, None] < boundary, reference_luma.shape
    )
    reference_stars = detect_stars(reference_luma, sigma=STAR_DETECT_SIGMA, mask=sky_mask)
    moving_stars = detect_stars(moving_luma, sigma=STAR_DETECT_SIGMA, mask=sky_mask)
    return fit_lens_aware(
        reference_luma,
        reference_stars,
        moving_luma,
        moving_stars,
        sky_mask=sky_mask,
    )


def estimate_chunk_photometry(
    moving: np.ndarray, reference: np.ndarray, valid: np.ndarray
) -> tuple[float, float]:
    """Robust scalar ``(offset, scale)`` for ``reference = (moving - offset) * scale``."""
    moving_luma = np.asarray(moving, dtype=np.float64).mean(axis=2)
    reference_luma = np.asarray(reference, dtype=np.float64).mean(axis=2)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(moving_luma) & np.isfinite(reference_luma)
    if not np.any(valid):
        raise AstroCombineError("chunk photometry has no overlapping valid sky pixels")
    moving_pedestal = float(np.median(moving_luma[valid]))
    reference_pedestal = float(np.median(reference_luma[valid]))
    moving_signal = moving_luma - moving_pedestal
    reference_signal = reference_luma - reference_pedestal
    moving_floor = float(np.percentile(moving_signal[valid], 90))
    reference_floor = float(np.percentile(reference_signal[valid], 90))
    strong = valid & (moving_signal >= moving_floor) & (reference_signal >= reference_floor)
    ratios = reference_signal[strong] / np.maximum(moving_signal[strong], 1e-8)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    scale = float(np.median(ratios)) if ratios.size >= 4 else 1.0
    if not 0.5 <= scale <= 2.0:
        raise AstroCombineError(f"chunk photometric scale is implausible: {scale:.4f}")
    offset = moving_pedestal - reference_pedestal / scale
    return offset, scale


def _warped_components(artifact: ValidatedArtifact, transform, shape, loader):
    """Warp one chunk's sky sum and count into the reference grid.

    Routed through ``astro_register.warp_to_reference`` rather than calling ``cv2
    .warpAffine`` here, so BOTH transform shapes work: a 2x3 affine (identity reference,
    similarity fallback) and a ``(map_x, map_y)`` lens remap. Same interpolation, same
    border, whichever it is -- the sum and the count can never be warped differently.
    """
    loaded = loader(artifact.path)
    out_shape = shape[:2]
    sky_sum = warp_to_reference(np.asarray(loaded.sky_sum), transform, out_shape).astype(np.float32)
    sky_count = warp_to_reference(
        np.asarray(loaded.sky_count, dtype=np.float32), transform, out_shape
    ).astype(np.float32)
    del loaded
    return sky_sum, sky_count


def _rejection_kappa(chunk_count: np.ndarray) -> np.ndarray:
    kappa = np.full(chunk_count.shape, SHORT_STACK_KAPPA_BY_N[5], dtype=np.float64)
    for threshold, value in sorted(SHORT_STACK_KAPPA_BY_N.items()):
        kappa = np.where(chunk_count >= threshold, value, kappa)
    return kappa


# Only the calmest (least wind-shaken) chunks feed the frozen ground band -- a chunk's own
# sky can be perfectly usable (sky is combined separately, untouched by this) while its ground
# is blurred by camera shake, and averaging that blur in would soften the whole band. There is
# no persisted per-chunk "wind score" (`astro_triage.FrameMetrics` is a per-RAW-FRAME record
# from triage, long since folded into one already-stacked `.linear` chunk by the time this
# module sees it), so the ranking is DERIVED here from the chunk's own ground pixels: variance
# of the Laplacian, restricted to the GROUND rows only (`_ground_region_sharpness`) -- a wind
# blur flattens high-frequency detail, so a blurred chunk scores low.
GROUND_BAND_FRACTION = 0.30


def _ground_region_sharpness(ground: np.ndarray, boundary_row: float) -> float:
    """Variance-of-Laplacian sharpness of a chunk's GROUND rows only (``row >= boundary_row``).

    Deliberately excludes the sky rows: ``ground`` is a per-chunk MEDIAN of raw, unaligned
    frames (``astro_stack.stack_ground``), so above the horizon its stars sit at a different
    pixel every frame and read as high-frequency noise that says nothing about how still the
    camera held the actual foreground -- scoring the whole frame would let sky noise swamp the
    ground signal this ranking exists to measure. A boundary at/past the frame height (sky
    fills the whole frame -- e.g. a not-yet-cropped synthetic fixture) falls back to the whole
    array rather than an empty slice.
    """
    height = ground.shape[0]
    row_start = int(np.clip(round(boundary_row), 0, height))
    region = ground[row_start:] if row_start < height else ground
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luma = np.sum(np.asarray(region, dtype=np.float32) * weights, axis=2, dtype=np.float32)
    laplacian = cv2.Laplacian(luma, cv2.CV_32F, ksize=3)
    return float(np.var(laplacian, dtype=np.float64))


def _ground_band_keep_mask(scores: Sequence[float]) -> list[bool]:
    """Per-chunk keep flag: the calmest ``GROUND_BAND_FRACTION`` of ``scores``, by RANK not by
    list position -- so the kept chunks are wherever the sharpest ground actually is, never
    "the first N".  The cutoff is a SCORE, not a fixed count: every chunk tied with the boundary
    rank's score is kept too, so a run where nothing distinguishes the chunks (equally sharp, or
    equally degenerate-flat -- the common case in a small synthetic fixture) keeps all of them
    rather than dropping an arbitrary majority on a tie-break that has no evidence behind it.
    """
    if not scores:
        return []
    keep_count = max(1, int(np.ceil(len(scores) * GROUND_BAND_FRACTION)))
    boundary_score = sorted(scores, reverse=True)[keep_count - 1]
    return [score >= boundary_score for score in scores]


def combine_sky_two_pass(
    artifacts: ValidatedArtifactSet,
    transforms: Sequence[ArtifactTransform],
    *,
    loader=load_linear_artifact,
) -> DeepSkyResult:
    """Photometrically map, warp, reject, and accumulate sky in two O(1)-in-N passes."""
    if len(transforms) != len(artifacts.artifacts):
        raise AstroCombineError("one artifact transform is required per chunk")
    shape = artifacts.reference.shape
    ref_sum, ref_count = _warped_components(
        artifacts.reference, np.eye(2, 3, dtype=np.float64), shape, loader
    )
    reference_mean = _sky_mean(ref_sum, ref_count)
    del ref_sum, ref_count
    sum_means = np.zeros(shape, dtype=np.float64)
    sum_sq_means = np.zeros(shape, dtype=np.float64)
    chunk_count = np.zeros(shape[:2], dtype=np.uint16)
    photometry = []
    for index, (artifact, plan) in enumerate(zip(artifacts.artifacts, transforms)):
        sky_sum, sky_count = _warped_components(artifact, plan.warp, shape, loader)
        mean = _sky_mean(sky_sum, sky_count)
        valid = sky_count > 0
        offset, scale = (
            (0.0, 1.0)
            if index == artifacts.reference_index
            else estimate_chunk_photometry(mean, reference_mean, valid)
        )
        corrected = (mean.astype(np.float64) - offset) * scale
        # Both passes remove the SAME background field, and both do it after photometry.  If
        # only the second pass removed it, `others_mean` below would come from unremoved sums
        # and every rejection z-score would be measured against the wrong baseline.
        corrected -= mesh_background(corrected, valid)
        sum_means += np.where(valid[..., None], corrected, 0.0)
        sum_sq_means += np.where(valid[..., None], corrected * corrected, 0.0)
        chunk_count += valid.astype(np.uint16)
        photometry.append(ChunkPhotometry(offset, scale))
        del sky_sum, sky_count, mean, corrected
    final_sum = np.zeros(shape, dtype=np.float64)
    final_count = np.zeros(shape[:2], dtype=np.float64)
    sample_boxes = noise_sample_boxes(shape)
    noise_samples = {name: ([], []) for name in sample_boxes}
    rejected_pixels = valid_pixels = 0
    kappa = _rejection_kappa(chunk_count)
    for artifact, plan, adjustment in zip(artifacts.artifacts, transforms, photometry):
        sky_sum, sky_count = _warped_components(artifact, plan.warp, shape, loader)
        corrected_sum = sky_sum.astype(np.float64) - adjustment.offset * sky_count[..., None]
        corrected_sum *= adjustment.scale
        mean = _sky_mean(corrected_sum, sky_count).astype(np.float64)
        valid = sky_count > 0
        # Subtract from the SUM as well as the mean, weighted by each pixel's frame count, so
        # `mean` stays exactly `_sky_mean(corrected_sum, sky_count)` and what the noise gate
        # samples below is what actually lands in `final_sum`.
        background = mesh_background(mean, valid)
        mean -= background
        corrected_sum -= background * sky_count[..., None]
        others_n = chunk_count.astype(np.float64) - valid
        others_floor = np.maximum(others_n, 1.0)
        others_mean = (sum_means - np.where(valid[..., None], mean, 0.0)) / others_floor[..., None]
        others_sq = (sum_sq_means - np.where(valid[..., None], mean * mean, 0.0)) / others_floor[
            ..., None
        ]
        variance = np.maximum(others_sq - others_mean * others_mean, 0.0)
        z = (mean - others_mean) / (np.sqrt(variance) + 1e-6)
        reject = (
            valid & (chunk_count >= SHORT_STACK_MIN_REJECT_FRAMES) & (np.max(z, axis=2) > kappa)
        )
        keep = valid & ~reject
        final_sum += np.where(keep[..., None], corrected_sum, 0.0)
        final_count += np.where(keep, sky_count, 0.0)
        rejected_pixels += int(reject.sum())
        valid_pixels += int(valid.sum())
        for name, (brow, bcol) in sample_boxes.items():
            kept_mean = np.where(keep[..., None], mean, np.nan)[brow, bcol]
            noise_samples[name][0].append(kept_mean.mean(axis=2))
            noise_samples[name][1].append(
                np.where(keep, sky_count, 0.0)[brow, bcol].astype(np.float64)
            )
        del sky_sum, sky_count, corrected_sum, mean, variance, z
    return DeepSkyResult(
        sky_sum=final_sum.astype(np.float32),
        sky_count=final_count.astype(np.float32),
        photometry=tuple(photometry),
        rejected_pixel_fraction=rejected_pixels / valid_pixels if valid_pixels else 0.0,
        noise_samples=noise_samples,
    )


def _loaded_ground(artifact: ValidatedArtifact, loader) -> np.ndarray:
    loaded = loader(artifact.path)
    ground = np.array(loaded.ground, dtype=np.float32, copy=True)
    del loaded
    return ground


def combine_ground_two_pass(
    artifacts: ValidatedArtifactSet, *, loader=load_linear_artifact
) -> DeepGroundResult:
    """Robust fixed-sensor-coordinate ground combine, weighted by chunk frame count.

    Before either statistics pass, every chunk's ground-region sharpness is ranked
    (``_ground_region_sharpness``) and only the calmest ``GROUND_BAND_FRACTION`` of chunks
    (``_ground_band_keep_mask``) ever contribute pixels to ``ground``/``weight`` -- an excluded
    chunk's SKY is untouched, since ``combine_sky_two_pass`` never sees this selection.
    ``photometry`` still carries one entry per artifact, in ``artifacts.artifacts`` order,
    EXCLUDED chunks included -- ``astro_combine_output.py`` zips it 1:1 against the full
    artifact/transform list, so shrinking it would silently misalign that sidecar.
    """
    shape = artifacts.reference.shape
    reference = _loaded_ground(artifacts.reference, loader)
    scores = []
    for artifact in artifacts.artifacts:
        ground = _loaded_ground(artifact, loader)
        boundary_row = float(artifact.manifest["mask_provenance"]["resolved_boundary_row"])
        scores.append(_ground_region_sharpness(ground, boundary_row))
        del ground
    keep = _ground_band_keep_mask(scores)

    sum_chunks = np.zeros(shape, dtype=np.float64)
    sum_sq_chunks = np.zeros(shape, dtype=np.float64)
    chunk_count = np.zeros(shape[:2], dtype=np.uint16)
    photometry = []
    for index, artifact in enumerate(artifacts.artifacts):
        ground = _loaded_ground(artifact, loader)
        valid = np.all(np.isfinite(ground), axis=2)
        offset, scale = (
            (0.0, 1.0)
            if index == artifacts.reference_index
            else estimate_chunk_photometry(ground, reference, valid)
        )
        corrected = (ground.astype(np.float64) - offset) * scale
        photometry.append(ChunkPhotometry(offset, scale))
        if keep[index]:
            sum_chunks += np.where(valid[..., None], corrected, 0.0)
            sum_sq_chunks += np.where(valid[..., None], corrected * corrected, 0.0)
            chunk_count += valid.astype(np.uint16)
        del ground, corrected
    weighted_sum = np.zeros(shape, dtype=np.float64)
    weight = np.zeros(shape[:2], dtype=np.float64)
    rejected_pixels = valid_pixels = 0
    kappa = _rejection_kappa(chunk_count)
    for index, (artifact, adjustment) in enumerate(zip(artifacts.artifacts, photometry)):
        if not keep[index]:
            continue
        ground = _loaded_ground(artifact, loader)
        corrected = (ground.astype(np.float64) - adjustment.offset) * adjustment.scale
        valid = np.all(np.isfinite(corrected), axis=2)
        others_n = chunk_count.astype(np.float64) - valid
        others_floor = np.maximum(others_n, 1.0)
        others_mean = (sum_chunks - np.where(valid[..., None], corrected, 0.0)) / others_floor[
            ..., None
        ]
        others_sq = (
            sum_sq_chunks - np.where(valid[..., None], corrected * corrected, 0.0)
        ) / others_floor[..., None]
        variance = np.maximum(others_sq - others_mean * others_mean, 0.0)
        z = (corrected - others_mean) / (np.sqrt(variance) + 1e-6)
        reject = (
            valid & (chunk_count >= SHORT_STACK_MIN_REJECT_FRAMES) & (np.max(z, axis=2) > kappa)
        )
        keep_pixel = valid & ~reject
        frame_weight = float(artifact.manifest["n_frames"])
        weighted_sum += np.where(keep_pixel[..., None], corrected * frame_weight, 0.0)
        weight += np.where(keep_pixel, frame_weight, 0.0)
        rejected_pixels += int(reject.sum())
        valid_pixels += int(valid.sum())
        del ground, corrected, variance, z
    combined = np.zeros(shape, dtype=np.float32)
    np.divide(weighted_sum, weight[..., None], out=combined, where=weight[..., None] > 0)
    return DeepGroundResult(
        ground=combined,
        weight=weight.astype(np.float32),
        photometry=tuple(photometry),
        rejected_pixel_fraction=rejected_pixels / valid_pixels if valid_pixels else 0.0,
    )


def deep_combine(
    paths: Sequence[Path],
    *,
    reference: Path | None = None,
    fit: Callable[[ValidatedArtifact, ValidatedArtifact], np.ndarray | RegistrationResult]
    | None = None,
    loader=load_linear_artifact,
    scope: str = DEFAULT_OUTPUT_SCOPE,
    enforce_noise_floor: bool = True,
) -> DeepCombineResult:
    """Build one scene-linear deep product; performs no develop, export, or sidecar I/O."""
    if scope not in CLI_OUTPUT_SCOPES:
        raise AstroCombineError(
            f"unknown output scope {scope!r}; expected one of {CLI_OUTPUT_SCOPES}"
        )
    artifacts = validate_artifact_set(paths, reference=reference)

    def trusted_loader(path):
        return loader(path, verify_hashes=False)

    pair_fit = fit or (
        lambda moving, target: fit_artifact_pair(moving, target, loader=trusted_loader)
    )
    transforms = plan_direct_transforms(artifacts, pair_fit)
    sky = combine_sky_two_pass(artifacts, transforms, loader=trusted_loader)
    ground = combine_ground_two_pass(artifacts, loader=trusted_loader)
    sky_mean = _sky_mean(sky.sky_sum, sky.sky_count)
    height = sky_mean.shape[0]
    boundary = float(artifacts.reference.manifest["mask_provenance"]["resolved_boundary_row"])
    # Soft horizon seam: `horizon_mask` already feathers this boundary across
    # `astro_stack.FEATHER_ROWS` (24) rather than switching sky/ground at one row -- see that
    # constant's own comment for why a hard edge seams any ridgeline that isn't perfectly flat.
    # Nothing here re-derives or narrows that width; the frozen (never-warped) ground band from
    # `combine_ground_two_pass` above is blended into the sky through the exact same feather.
    mask = horizon_mask(sky_mean.shape[:2], horizon_frac=boundary / height)
    rgb = blend(sky_mean, ground.ground, mask)
    # Blend at full frame FIRST -- the horizon row is defined against the full height, so
    # cropping before the mask is built would place the horizon wrong -- then ship the scoped
    # region.  Coverage is reported over the SHIPPED pixels, not over pixels nobody receives.
    # "auto" is resolved HERE, against this same pre-crop `rgb` -- measuring after the crop
    # would defeat the point (a cropped frame's own corners are never its worst corners).
    if scope == AUTO_OUTPUT_SCOPE:
        resolved_scope, scope_decision = resolve_auto_scope(
            rgb, _frame_median_fwhm_px(artifacts.artifacts)
        )
    else:
        resolved_scope, scope_decision = scope, None
    rows, cols = scoped_frame_slice(rgb.shape, scope=resolved_scope)
    rgb = rgb[rows, cols]
    mask = mask[rows, cols]
    shipped_count = sky.sky_count[rows, cols]
    nominal = sum(int(artifact.manifest["n_frames"]) for artifact in artifacts.artifacts)
    floor = max(float(MIN_STACK_CONTRIBUTORS), nominal * 0.5)
    coverage = DeepCoverage(
        nominal_contributors=nominal,
        floor=floor,
        min_contributors=float(np.min(shipped_count)),
        median_contributors=float(np.median(shipped_count)),
        below_floor_fraction=float(np.mean(shipped_count < floor)),
    )
    # Upstream gate: measure before any irreversible artifact exists, never after.  A
    # fixed depth constant is not shipped because the passing depth is one night's property
    # and the estimator has its own error band; this measures the stack in hand instead.
    frames = int(median(int(a.manifest["n_frames"]) for a in artifacts.artifacts))
    noise_floor = check_noise_floor(sky.noise_samples, frames_per_chunk=max(frames, 1))
    if enforce_noise_floor and not noise_floor.passed:
        measured = ", ".join(
            f"{name} {noise_floor.efficiency_pct[name]:.1f}%" for name in noise_floor.failing_boxes
        )
        raise AstroCombineError(
            f"deep stack noise does not average down in {measured} "
            f"(floor {noise_floor.floor_pct:.0f}%); combine fewer chunks, or pass "
            f"enforce_noise_floor=False to publish it anyway"
        )
    return DeepCombineResult(
        rgb=rgb,
        mask=mask,
        scope=resolved_scope,
        requested_scope=scope,
        scope_decision=scope_decision,
        noise_floor=noise_floor,
        frame_origin=(int(rows.start), int(cols.start)),
        sky=sky,
        ground=ground,
        transforms=transforms,
        coverage=coverage,
        artifacts=artifacts,
    )


def compose_affine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compose 2x3 transforms: apply ``first``, then ``second``."""
    first_h = np.vstack([np.asarray(first, dtype=np.float64), [0.0, 0.0, 1.0]])
    second_h = np.vstack([np.asarray(second, dtype=np.float64), [0.0, 0.0, 1.0]])
    if first_h.shape != (3, 3) or second_h.shape != (3, 3):
        raise AstroCombineError("artifact transforms must be 2x3 affine matrices")
    return (second_h @ first_h)[:2]


def _resolved_fit(
    fit_result: np.ndarray | RegistrationResult, moving_index: int, target_index: int
) -> tuple[np.ndarray | None, tuple | None, dict]:
    """Turn a caller's raw fit return into ``(2x3 affine, remap, diagnostic dict)``, validated.

    The transform is either a 2x3 affine or a ``(map_x, map_y)`` lens remap -- the two
    shapes ``_warped_components``/``astro_register.warp_to_reference`` apply.
    """
    if isinstance(fit_result, RegistrationResult):
        raw = fit_result.transform
        diagnostic = {
            "n_matched": fit_result.n_matched,
            "inlier_fraction": fit_result.inlier_fraction,
            "residual_median_px": fit_result.residual_median_px,
            "residual_p95_px": fit_result.residual_p95_px,
            "match_yield": fit_result.match_yield,
            "radial_inlier_residuals_px": fit_result.radial_inlier_residuals_px,
        }
        if isinstance(fit_result, LensPairFit):
            diagnostic["geometry_method"] = fit_result.geometry_method
            diagnostic["fallback_reason"] = fit_result.fallback_reason
            diagnostic["held_out_residuals_px"] = fit_result.held_out_residuals_px
    else:
        raw = fit_result
        diagnostic = {}
    if _is_remap_pair(raw):
        maps = tuple(np.asarray(plane, dtype=np.float32) for plane in raw)
        if not all(np.all(np.isfinite(plane)) for plane in maps):
            raise AstroCombineError(f"invalid artifact transform {moving_index}->{target_index}")
        return None, maps, diagnostic
    value = np.asarray(raw, dtype=np.float64)
    if value.shape != (2, 3) or not np.all(np.isfinite(value)):
        raise AstroCombineError(f"invalid artifact transform {moving_index}->{target_index}")
    return value, None, diagnostic


def _fit_failure_detail(exc: RegistrationError) -> str:
    """The geometry context a refused pair carries, appended to the planner's abort message.

    Empty for a plain ``RegistrationError`` (a caller-supplied fit, or the pre-lens path):
    this adds context where it exists and invents none where it does not.
    """
    if not isinstance(exc, PairFitError):
        return ""
    detail = f" [geometry_method={exc.geometry_method}"
    if exc.fallback_reason:
        detail += f"; downgraded from the lens model because {exc.fallback_reason}"
    return detail + "]"


def plan_direct_transforms(
    artifacts: ValidatedArtifactSet,
    fit: Callable[[ValidatedArtifact, ValidatedArtifact], np.ndarray | RegistrationResult],
) -> tuple[ArtifactTransform, ...]:
    """Every artifact registers directly to the single global reference; no chaining.

    Composing a transform through a neighbour compounds that neighbour's own registration
    error, and on a long night the deepest chunk would end up many composed links from the
    reference. ``plan_artifact_transforms`` below is the chained variant, kept only for
    callers that still need it. A chunk that cannot register straight to the reference is a hard failure now, never a
    silently-composed one -- and that failure names the model it was measuring
    (``_fit_failure_detail``), so a run that died after a pair was downgraded to the
    similarity says so rather than reading exactly like the pre-lens behaviour.

    Every pair is attempted before any refusal is raised. Aborting on the first one made a
    night pay the whole per-pair fit cost (around a minute or more each under the lens
    model) and learn
    about exactly one chunk, so a re-run was needed to discover the next. The refusal now
    names every chunk that could not register and the widest run of chunks around the
    reference that could, which is the subset of paths a re-run can actually ship.
    """
    reference_index = artifacts.reference_index
    resolved: list[ArtifactTransform | None] = []
    failures: list[tuple[int, RegistrationError]] = []
    for index in range(len(artifacts.artifacts)):
        if index == reference_index:
            resolved.append(
                ArtifactTransform(np.eye(2, 3, dtype=np.float64), "identity", (), (), 0)
            )
            continue
        try:
            fit_result = fit(artifacts.artifacts[index], artifacts.artifacts[reference_index])
        except RegistrationError as exc:
            failures.append((index, exc))
            resolved.append(None)
            continue
        transform, remap, diagnostic = _resolved_fit(fit_result, index, reference_index)
        resolved.append(
            ArtifactTransform(
                transform, "direct", ((index, reference_index),), (diagnostic,), 1, remap
            )
        )
    if failures:
        raise AstroCombineError(
            _unregisterable_report(failures, resolved, reference_index)
        ) from failures[0][1]
    return tuple(plan for plan in resolved if plan is not None)


def _registrable_span(
    resolved: Sequence[ArtifactTransform | None], reference_index: int
) -> tuple[int, int]:
    """The inclusive run of registered chunks CONTAINING the reference.

    Not the longest run anywhere: a stretch that registers but sits behind a gap still cannot
    be warped into the reference's frame as one product, because the gap is exactly the part
    that has no transform into it.
    """
    start = reference_index
    while start > 0 and resolved[start - 1] is not None:
        start -= 1
    end = reference_index
    while end + 1 < len(resolved) and resolved[end + 1] is not None:
        end += 1
    return start, end


def _unregisterable_report(
    failures: Sequence[tuple[int, RegistrationError]],
    resolved: Sequence[ArtifactTransform | None],
    reference_index: int,
) -> str:
    first_index, first_exc = failures[0]
    lines = [
        f"required direct transform {first_index}->{reference_index} failed: {first_exc}"
        f"{_fit_failure_detail(first_exc)}"
    ]
    for index, exc in failures[1:]:
        lines.append(f"  also refused: {index}->{reference_index}: {exc}{_fit_failure_detail(exc)}")
    start, end = _registrable_span(resolved, reference_index)
    lines.append(
        f"  registrable span around reference {reference_index}: chunks {start}-{end} "
        f"({end - start + 1} of {len(resolved)})"
    )
    return "\n".join(lines)


def plan_artifact_transforms(
    artifacts: ValidatedArtifactSet,
    fit: Callable[[ValidatedArtifact, ValidatedArtifact], np.ndarray | RegistrationResult],
) -> tuple[ArtifactTransform, ...]:
    """Fit every artifact directly to the midpoint, chaining onto its nearest RESOLVED
    neighbour only on failure.

    ``fit(moving, reference)`` returns a 2x3 moving-to-reference transform and raises
    ``RegistrationError`` when the star fit cannot establish a valid consensus.  Every fallback
    link is recorded; the geometry path is never silently substituted.

    Artifacts are planned OUTWARD from the reference, so when a direct fit fails, the neighbour
    one step toward the reference already carries a finished transform. Chaining onto that
    neighbour costs a single new link. Walking the whole way to the reference instead
    re-derives geometry that is already known and composes its error
    again at every step: that compounds links on every outer chunk of a long combine, so the outer
    chunks' stars can be destroyed while adjacent pairs still come out clean. Each plan reports its ``depth`` so a caller can refuse a run whose
    worst artifact is composed too deeply to trust."""
    reference_index = artifacts.reference_index
    cache: dict[tuple[int, int], tuple[np.ndarray, dict]] = {}

    def fitted(moving_index: int, target_index: int) -> tuple[np.ndarray, dict]:
        key = moving_index, target_index
        if key not in cache:
            fit_result = fit(artifacts.artifacts[moving_index], artifacts.artifacts[target_index])
            transform, remap, diagnostic = _resolved_fit(fit_result, moving_index, target_index)
            if remap is not None:
                # A remap is a resampling grid, not a matrix: chaining one would resample
                # twice. This retired planner composes, so it takes affines only.
                raise AstroCombineError(
                    f"transform {moving_index}->{target_index} is a lens remap, "
                    "which the adjacent-chain planner cannot compose"
                )
            cache[key] = transform, diagnostic
        return cache[key]

    total = len(artifacts.artifacts)
    resolved: dict[int, ArtifactTransform] = {
        reference_index: ArtifactTransform(np.eye(2, 3, dtype=np.float64), "identity", (), (), 0)
    }

    # Outward from the reference, so the neighbour one step inward is always already resolved.
    for index in sorted(range(total), key=lambda i: abs(i - reference_index)):
        if index == reference_index:
            continue
        try:
            direct, diagnostic = fitted(index, reference_index)
        except RegistrationError:
            pass
        else:
            resolved[index] = ArtifactTransform(
                direct, "direct", ((index, reference_index),), (diagnostic,), 1
            )
            continue

        step_target = index + (1 if index < reference_index else -1)
        anchor = resolved.get(step_target)
        if anchor is None:  # unreachable while planning outward; a guard, not a fallback.
            raise AstroCombineError(
                f"artifact {index} has no resolved neighbour at {step_target} to chain onto"
            )
        try:
            step, diagnostic = fitted(index, step_target)
        except RegistrationError as exc:
            raise AstroCombineError(
                f"required adjacent transform {index}->{step_target} failed: {exc}"
            ) from exc
        resolved[index] = ArtifactTransform(
            compose_affine(step, anchor.transform),
            "adjacent_chain",
            ((index, step_target),) + anchor.links,
            (diagnostic,) + anchor.diagnostics,
            anchor.depth + 1,
        )

    return tuple(resolved[index] for index in range(total))


def _load_descriptor(path: Path) -> ValidatedArtifact:
    if path.suffix != ".linear" or not path.is_dir():
        raise AstroCombineError(f"deep combine inputs must be .linear artifact directories: {path}")
    try:
        loaded = load_linear_artifact(path)
    except LinearArtifactError as exc:
        raise AstroCombineError(f"linear artifact {path}: {exc}") from exc
    return ValidatedArtifact(path=path, manifest=loaded.manifest)


def _reference_index(artifacts: Sequence[ValidatedArtifact]) -> int:
    """Temporal midpoint among the top quality quartile; residual, frame count and index
    form the even-night tie-break within that quartile.

    Quality is the persisted per-chunk ``sky_weight`` (schema >= 3; absent/older manifests
    default to 1.0, the same "contribute as before" value ``chunk_sky_weight`` itself returns
    when it has nothing to measure). The quartile is a THRESHOLD on that weight (its own 75th
    percentile), not a fixed top-N slice, so a night where every chunk ties at the same weight
    still lets every chunk compete on proximity to the midpoint.
    Every chunk now registers directly to whichever chunk this picks (no more chaining), so a
    low-quality chunk that merely happens to sit closest to the midpoint must never anchor the
    whole stack.
    """
    total = len(artifacts)
    midpoint = (total - 1) / 2
    weights = [float(artifact.manifest.get("sky_weight", 1.0)) for artifact in artifacts]
    quartile_floor = float(np.percentile(weights, 75)) if total > 1 else weights[0]
    candidates = [index for index, weight in enumerate(weights) if weight >= quartile_floor]

    def score(index: int):
        manifest = artifacts[index].manifest
        residuals = [
            float(item["residual_p95_px"])
            for item in manifest.get("registration", ())
            if isinstance(item, dict) and "residual_p95_px" in item
        ]
        quality = median(residuals) if residuals else float("inf")
        return abs(index - midpoint), quality, -int(manifest.get("n_frames", 0)), index

    return min(candidates, key=score)


def validate_artifact_set(
    paths: Sequence[Path], *, reference: Path | None = None
) -> ValidatedArtifactSet:
    """Hash-check and validate the complete set before any output is written."""
    if len(paths) < MIN_DEEP_CHUNKS:
        raise AstroCombineError(f"deep combine needs at least {MIN_DEEP_CHUNKS} chunk artifacts")
    artifacts = tuple(_load_descriptor(Path(path)) for path in paths)
    first = artifacts[0].manifest
    algorithm = first.get("algorithm_version")
    shape = artifacts[0].shape
    identity = first.get("capture_identity")
    if not algorithm or not isinstance(identity, dict):
        raise AstroCombineError("first artifact is missing algorithm or capture identity")
    identity_key = tuple(identity.get(field) for field in _CAPTURE_IDENTITY_FIELDS)
    mask = first.get("mask_provenance")
    mask_mode = mask.get("mode") if isinstance(mask, dict) else None
    if mask_mode not in {"automatic", "explicit"}:
        raise AstroCombineError("first artifact has an unsupported mask mode")
    explicit_horizon = mask.get("requested_horizon_frac") if mask_mode == "explicit" else None
    seen_sources: set[str] = set()
    for artifact in artifacts:
        manifest = artifact.manifest
        if manifest.get("algorithm_version") != algorithm:
            raise AstroCombineError(f"artifact algorithm mismatch: {artifact.path}")
        if artifact.shape != shape:
            raise AstroCombineError(f"artifact shape mismatch: {artifact.path}")
        member_identity = manifest.get("capture_identity")
        member_key = (
            tuple(member_identity.get(field) for field in _CAPTURE_IDENTITY_FIELDS)
            if isinstance(member_identity, dict)
            else None
        )
        if member_key != identity_key:
            raise AstroCombineError(f"artifact capture identity mismatch: {artifact.path}")
        member_mask = manifest.get("mask_provenance")
        member_mode = member_mask.get("mode") if isinstance(member_mask, dict) else None
        if member_mode != mask_mode:
            raise AstroCombineError(f"artifact mask mode mismatch: {artifact.path}")
        if (
            mask_mode == "explicit"
            and member_mask.get("requested_horizon_frac") != explicit_horizon
        ):
            raise AstroCombineError(f"artifact explicit mask provenance mismatch: {artifact.path}")
        sources = manifest.get("source_frames")
        if (
            not isinstance(sources, list)
            or not sources
            or not all(isinstance(p, str) for p in sources)
        ):
            raise AstroCombineError(f"artifact source-frame provenance is invalid: {artifact.path}")
        overlap = seen_sources.intersection(sources)
        if overlap:
            raise AstroCombineError(
                f"artifact source frames overlap at {sorted(overlap)[0]}: {artifact.path}"
            )
        seen_sources.update(sources)
    if reference is None:
        reference_index = _reference_index(artifacts)
    else:
        wanted = Path(reference).resolve()
        matches = [i for i, artifact in enumerate(artifacts) if artifact.path.resolve() == wanted]
        if not matches:
            raise AstroCombineError(f"explicit reference is not in the artifact set: {reference}")
        reference_index = matches[0]
    return ValidatedArtifactSet(artifacts=artifacts, reference_index=reference_index)
