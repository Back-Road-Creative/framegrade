"""Star-field registration: a similarity transform per frame.

Pure array math — numpy/cv2 only, no other ``astro_*``
module — consuming star lists shaped as ``astro_triage.detect_stars`` emits
them: ``(N, 3) float32`` rows of ``(x, y, flux)``. Detecting stars is
triage's job; picking the reference frame is the orchestrator's.

Three stages compose into ``register``: ``coarse_shift`` (phase correlation
on downscaled luminance, a translation prior good to a few px),
``match_progressive`` (a coarse-to-fine cascade over ``match_stars``'s
nearest-neighbour/mutual-best core -- see its docstring), ``fit_similarity``
(RANSAC similarity fit). The fitted 2x3 maps a FRAME point onto REFERENCE
coordinates, so ``warp_to_reference`` hands it straight to ``cv2.warpAffine``
with no inversion.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Longest side a luminance frame is downscaled to before phaseCorrelate. The prior
# needs to land inside MATCH_TOLERANCE_PX, and 512 does not: on a real 20.9 MP card
# it returned priors tens of px off and lost every star pair, while 1024
# recovered them (one sampled frame: 0 -> 159 matched pairs). 1024 is not slower --
# measured per pair on that card, 512 = 683 ms, 1024 = 577 ms, 2048 = 952 ms -- because
# the correlation peak is sharper and cv2's FFT sizes are friendlier; 2048 buys under
# 2% more pairs for ~1.6x the time, so 1024 is the knee.
MAX_COARSE_DIM = 1024

# Nearest-neighbour radius (px) after shifting frame stars by the coarse
# prior. Spec's motion scale is ~1-2 px frame-to-frame plus small rotation,
# so 3 px covers that drift without also catching a crowded field's neighbour.
MATCH_TOLERANCE_PX = 3.0

# Matching cost is O(len(ref) * len(frame)); capping each side to its
# brightest N (bright = well-centroided) bounds that cost. Applies to
# ``match_stars``'s single-pass call only -- ``match_progressive`` (below) must
# see every detected star at its mid/final stages, or a star that only survives
# past its brightness cutoff once rotation is corrected for is lost for good.
MAX_MATCH_STARS = 200

# ``match_progressive`` stage 1 (seed): normalised radius (same convention as
# ``radial_inlier_residuals_px``) below which a star counts as "central" and is
# eligible for the wide-tolerance seed match. A frame-to-frame rotation
# displaces a star by roughly ``radius * angle_rad``, so restricting the seed
# to the central band keeps that displacement small even for a rotation that
# would blow well past MATCH_TOLERANCE_PX at the frame edge.
MATCH_SEED_RADIUS_FRACTION = 0.4

# ``match_progressive`` stage 1 tolerance (px): wide enough to catch a central
# star's residual displacement under prior_shift alone (translation only, no
# rotation correction yet).
MATCH_SEED_TOLERANCE_PX = 12.0

# ``match_progressive`` stage 3 tolerance (px): re-match ALL stars once a
# similarity fit from the central seed exists -- WIDENS, does not narrow. A
# similarity fitted only on central points is itself wrong by more than a few
# px at the frame edge (that's the whole problem this cascade exists to fix),
# so a narrow window here would throw away edge stars before any model capable
# of placing them has been fitted. Measured against a real card sequence: a 6 px
# window here (this cascade's first draft) cost 99.7% of the outer-third
# matches a 25 px window recovers.
MATCH_WIDE_TOLERANCE_PX = 25.0

# ``match_progressive`` stage 5 successive tightening tolerances (px), each
# re-fitting the homography (see MATCH_HOMOGRAPHY_RANSAC_THRESHOLD_PX) before
# the next. The last entry is MATCH_TOLERANCE_PX itself -- the final pass uses
# the same acceptance window ``match_stars`` always has, just under a model
# that can actually reach the frame edge.
MATCH_TIGHTEN_TOLERANCES_PX = (10.0, 4.0, MATCH_TOLERANCE_PX)

# RANSAC reprojection threshold (px) for the stage-4/5 homography fits.
# Generous relative to RANSAC_REPROJ_THRESHOLD_PX because this homography is a
# SEARCH-WINDOW model only -- it decides where to look for a counterpart in the
# tightening stages, it is never returned as, or used as, the registration
# transform. register() still fits its own similarity from whatever
# correspondence pairs come out of this cascade.
MATCH_HOMOGRAPHY_RANSAC_THRESHOLD_PX = 5.0

# A similarity transform has 4 DOF — 2 points are exactly determined. RANSAC
# needs real redundancy to tell a consensus from a coincidence; 6 is the
# floor below which "converged" is indistinguishable from "guessed".
MIN_MATCH_STARS = 6

# Below this inlier share the RANSAC consensus is itself suspect (a
# coincidental cluster, not a real match), so the fit is refused.
MIN_INLIER_FRACTION = 0.5

# A similarity transform still has 4 DOF once it survives RANSAC, so a consensus
# of 3 points is one point of redundancy over an exact solve -- an interpolation,
# not a fit, and indistinguishable from a coincidence no matter how large the
# matched set behind it was. MIN_INLIER_FRACTION alone can't catch this: a small
# matched set can clear 0.5 with only 3-4 inliers. Mirrors MIN_MATCH_STARS's floor.
MIN_INLIER_STARS = 6

# RANSAC reprojection threshold (px) for a point to count as an inlier.
# Centroid noise plus the coarse prior's slack put a true match within a
# couple of px of the fitted model; outliers sit far beyond that.
RANSAC_REPROJ_THRESHOLD_PX = 2.0

# Floor on RANSAC-inlier support in the outer third of the sensor (PROVISIONAL --
# no rig-specific study has set this, it is a judgement call). Justified by the
# Measured finding (astro_combine.py SCOPED_FRAME_FRACTION comment):
# the outer radial third is measurably the weakest region of a real fit, so a
# transform whose fit barely touched it -- even one that otherwise clears every
# other geometry gate -- is a fit that only succeeded in the frame's centre, not
# across the whole sensor. Below this floor the fit is refused outright rather
# than shipped with silently thin edge support.
MIN_OUTER_THIRD_INLIERS = 8

# Fitted-transform sanity bounds (PROVISIONAL) -- deliberately broad: no cadence/FOV
# derived data exists yet for this rig. These
# exist only to catch a degenerate or runaway RANSAC fit -- garbage correspondences that
# still clear MIN_INLIER_FRACTION -- not to characterize real frame-to-frame motion;
# retune once a real card sequence establishes actual drift.
FIT_MIN_SCALE = 0.5  # A fit outside 2x either way from 1.0 is a bad fit, not a lens zoom.
FIT_MAX_SCALE = 2.0
FIT_MAX_ROTATION_DEG = 20.0  # Broad cap on frame-to-frame field rotation.
FIT_MAX_TRANSLATION_FRAME_FRACTION = 0.5  # Translation magnitude vs. the frame's longer side.


class RegistrationError(RuntimeError):
    """Star matching or the similarity fit could not produce a usable transform."""


@dataclass(frozen=True)
class RegistrationResult:
    """The composition of the three fit stages for one frame against the reference.

    ``inlier_fraction`` is conditional on ``match_stars``'s 3 px mutual-best
    pre-filter -- it measures RANSAC consensus among pairs already agreed by nearest-
    neighbour matching, not among raw detections, so a value near 1.0 reflects a clean
    matched set rather than an unusually reliable fit. Read it alongside
    ``match_yield``, which relates the matched count back to what was actually
    detected in each frame.
    """

    transform: np.ndarray  # 2x3 float64: frame (x, y, 1) -> reference coordinates
    n_matched: int
    inlier_fraction: float
    # Geometry evidence -- defaulted so existing positional/
    # keyword construction elsewhere (orchestrator, sidecar, their tests) is unaffected.
    residual_median_px: float = 0.0  # Median per-pair fit residual (px) under the transform.
    residual_p95_px: float = 0.0  # 95th-percentile per-pair fit residual (px).
    match_yield: float = 0.0  # matched pairs / min(detected ref, detected frame) star count.
    radial_inlier_residuals_px: dict | None = None
    # ``match_progressive``'s stage-by-stage match counts (seed/wide/tighten_10px/
    # tighten_4px/final) -- where a fit's matched pairs actually came from, so a fit
    # that only succeeded once the search window widened is distinguishable from one
    # that matched cleanly from the first pass. In-sample: describes THIS fit's own
    # matching, not held-out geometry evidence.
    match_yield_by_stage: dict | None = None


def _downscale(luma, max_dim):
    h, w = luma.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return np.ascontiguousarray(luma, dtype=np.float32), 1.0
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    small = cv2.resize(luma.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    return small, scale


def sky_rows(sky_mask, shape_hw):
    """``(row_start, row_stop)`` -- the row band spanning every sky pixel, or the whole
    frame when there is no mask (or a mask with no sky in it).

    A RECTANGULAR band, deliberately, not the mask itself. Multiplying the frame by the
    mask was measured to make the prior WORSE, not better (43.6 px error at every coarse
    dimension on the fixture in ``tests/test_astro_register.py``): zeroing the ground
    replaces soft foreground texture with a hard, perfectly stationary horizon edge that
    correlates harder than the trees it was meant to suppress. Cropping removes that
    structure from the correlation entirely instead of re-shaping it.
    """
    if sky_mask is None:
        return 0, shape_hw[0]
    rows = np.flatnonzero(np.any(np.asarray(sky_mask, dtype=bool), axis=1))
    if rows.size == 0:
        return 0, shape_hw[0]
    return int(rows[0]), int(rows[-1]) + 1


def prepare_reference(luma_ref, sky_mask=None):
    """Precompute the reference's sky crop and ``_downscale`` result once: pass
    this as ``coarse_shift``'s ``ref_prepared`` across every frame registered against the
    same reference so it isn't re-cropped and re-downscaled on each call.

    Returns ``(ref_small, scale, rows)``. ``rows`` travels with the cached array so
    ``coarse_shift`` crops the frame to the SAME band -- correlating a sky-cropped
    reference against a whole-frame frame returns a confident, wrong prior.
    """
    rows = sky_rows(sky_mask, luma_ref.shape[:2])
    small, scale = _downscale(luma_ref[rows[0] : rows[1]], MAX_COARSE_DIM)
    return small, scale, rows


def coarse_shift(luma_ref, luma_frame, ref_prepared=None, sky_mask=None):
    """``(dx, dy)`` such that FRAME xy + (dx, dy) lands near REFERENCE xy.

    ``cv2.phaseCorrelate`` on downscaled float32 luminance, rescaled back up. Both sides
    are first cropped to ``sky_rows(sky_mask, ...)``, because a landscape frame's
    stationary foreground otherwise dominates the correlation and drags the prior toward
    zero -- and a prior outside ``MATCH_TOLERANCE_PX`` costs EVERY star pair at once, not
    a few. Cropping is a translation-preserving operation: the same fixed window is taken
    from both frames, so the shift measured inside it is the shift between them.

    ``ref_prepared`` is an optional ``prepare_reference(luma_ref, sky_mask)`` result;
    when given, the reference is NOT re-cropped or re-downscaled, and its cached row band
    is used for the frame too (``sky_mask`` is then redundant and ignored). Omitted with
    no ``sky_mask``, behavior is the whole-frame correlation this has always done.
    """
    if ref_prepared is not None:
        ref_small, scale, rows = ref_prepared
    else:
        ref_small, scale, rows = prepare_reference(luma_ref, sky_mask)
    frame_small, _ = _downscale(luma_frame[rows[0] : rows[1]], MAX_COARSE_DIM)
    window = cv2.createHanningWindow((ref_small.shape[1], ref_small.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(frame_small, ref_small, window)
    return dx / scale, dy / scale


def _brightest_indices(stars, limit):
    if len(stars) <= limit:
        return np.arange(len(stars))
    return np.argsort(-stars[:, 2])[:limit]


def _mutual_best_local_pairs(ref_xy, frame_xy, tolerance_px):
    """``[(i, j), ...]`` positions into ``ref_xy``/``frame_xy`` (already the caller's
    candidate arrays, already shifted/transformed into the same coordinate space) that
    are each other's nearest neighbour within ``tolerance_px`` — the mutual-best core
    shared by ``match_stars`` and ``match_progressive``."""
    if len(ref_xy) == 0 or len(frame_xy) == 0:
        return []
    diff = ref_xy[:, None, :] - frame_xy[None, :, :]
    dists = np.hypot(diff[..., 0], diff[..., 1])
    nearest_frame = np.argmin(dists, axis=1)
    nearest_ref = np.argmin(dists, axis=0)
    return [
        (i, j)
        for i, j in enumerate(nearest_frame)
        if nearest_ref[j] == i and dists[i, j] <= tolerance_px
    ]


def match_stars(stars_ref, stars_frame, prior_shift, tolerance_px=MATCH_TOLERANCE_PX):
    """``(M, 2)`` int index pairs ``(ref_idx, frame_idx)``, mutual-best only.

    Frame positions are shifted by ``prior_shift``, then matched to the
    nearest reference star; a pair survives only when each side is also the
    OTHER's nearest candidate, within ``tolerance_px`` — an ambiguous pair
    (two stars equally near one counterpart) is dropped, not guessed.

    Each side is capped to its ``MAX_MATCH_STARS`` brightest candidates first —
    a single wide-field, single-tolerance pass with no correction for
    frame-to-frame rotation. ``match_progressive`` below composes this same
    mutual-best core into a cascade that isn't limited by either constraint.
    """
    dx, dy = prior_shift
    ref_sel = _brightest_indices(stars_ref, MAX_MATCH_STARS)
    frame_sel = _brightest_indices(stars_frame, MAX_MATCH_STARS)
    if len(ref_sel) == 0 or len(frame_sel) == 0:
        return np.empty((0, 2), dtype=int)
    ref_xy = stars_ref[ref_sel, :2].astype(np.float64)
    frame_xy = stars_frame[frame_sel, :2].astype(np.float64) + np.array([dx, dy])
    local_pairs = _mutual_best_local_pairs(ref_xy, frame_xy, tolerance_px)
    if not local_pairs:
        return np.empty((0, 2), dtype=int)
    pairs = [(ref_sel[i], frame_sel[j]) for i, j in local_pairs]
    return np.array(pairs, dtype=int).reshape(-1, 2)


def _radius_fraction(xy, frame_shape):
    """Normalised distance from frame centre, same convention as
    ``radial_inlier_residuals_px``: 0 at centre, ~1 at the corner."""
    height, width = frame_shape[:2]
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    max_radius = max(np.linalg.norm(center), 1.0)
    return np.linalg.norm(np.asarray(xy, dtype=np.float64) - center, axis=1) / max_radius


def _match_with_projector(
    stars_ref, stars_frame, project_frame_xy, tolerance_px, ref_idx=None, frame_idx=None
):
    """``match_stars``'s mutual-best core, but under an arbitrary ``project_frame_xy``
    (frame xy -> reference-coordinate xy) instead of a plain translation, and over
    caller-chosen index subsets (``None`` means "every star") instead of a
    brightness cap. Returns ``(M, 2)`` global index pairs. Shared by the affine and
    homography search-window matchers below."""
    ref_idx = np.arange(len(stars_ref)) if ref_idx is None else np.asarray(ref_idx)
    frame_idx = np.arange(len(stars_frame)) if frame_idx is None else np.asarray(frame_idx)
    if len(ref_idx) == 0 or len(frame_idx) == 0:
        return np.empty((0, 2), dtype=int)
    ref_xy = stars_ref[ref_idx, :2].astype(np.float64)
    frame_xy_raw = stars_frame[frame_idx, :2].astype(np.float64)
    frame_xy = project_frame_xy(frame_xy_raw)
    local_pairs = _mutual_best_local_pairs(ref_xy, frame_xy, tolerance_px)
    if not local_pairs:
        return np.empty((0, 2), dtype=int)
    pairs = [(ref_idx[i], frame_idx[j]) for i, j in local_pairs]
    return np.array(pairs, dtype=int).reshape(-1, 2)


def _match_under_transform(
    stars_ref, stars_frame, transform, tolerance_px, ref_idx=None, frame_idx=None
):
    """``_match_with_projector`` under a full 2x3 similarity ``transform``
    (frame -> reference)."""
    transform = np.asarray(transform, dtype=np.float64)

    def _project(xy):
        return xy @ transform[:, :2].T + transform[:, 2]

    return _match_with_projector(
        stars_ref, stars_frame, _project, tolerance_px, ref_idx=ref_idx, frame_idx=frame_idx
    )


def _match_under_homography(
    stars_ref, stars_frame, homography, tolerance_px, ref_idx=None, frame_idx=None
):
    """``_match_with_projector`` under a full 3x3 ``homography`` (frame ->
    reference), perspective-divided -- ``match_progressive``'s stage 4/5
    search-window model, which has the freedom a 4-DOF similarity lacks to
    follow a frame edge."""
    homography = np.asarray(homography, dtype=np.float64)

    def _project(xy):
        ones = np.ones((len(xy), 1), dtype=np.float64)
        projected = np.hstack([xy, ones]) @ homography.T
        w = projected[:, 2:3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        return projected[:, :2] / w

    return _match_with_projector(
        stars_ref, stars_frame, _project, tolerance_px, ref_idx=ref_idx, frame_idx=frame_idx
    )


def _refit_or_keep(stars_ref, stars_frame, pairs, current_transform, frame_shape):
    """Refit a similarity from ``pairs`` and return it, or ``current_transform``
    unchanged when there aren't enough pairs to attempt a fit or the fit itself is
    refused (``RegistrationError``) — the graceful-degradation step between
    ``match_progressive`` stages."""
    if len(pairs) < MIN_MATCH_STARS:
        return current_transform
    matched_ref_xy = stars_ref[pairs[:, 0], :2]
    matched_frame_xy = stars_frame[pairs[:, 1], :2]
    try:
        transform, _, _ = _estimate_transform(
            matched_ref_xy, matched_frame_xy, frame_shape=frame_shape
        )
        return transform
    except RegistrationError:
        return current_transform


def _fit_homography_or_keep(stars_ref, stars_frame, pairs, current_homography):
    """Refit a search-window homography from ``pairs`` and return it, or
    ``current_homography`` unchanged when there aren't enough pairs or
    ``cv2.findHomography`` itself refuses (returns ``None``/non-finite) — the
    homography-side counterpart to ``_refit_or_keep``. Never validated against
    ``_validate_transform``: this homography is not a candidate registration
    transform, only a search window for the next matching pass."""
    if len(pairs) < MIN_MATCH_STARS:
        return current_homography
    ref_pts = np.ascontiguousarray(stars_ref[pairs[:, 0], :2], dtype=np.float32).reshape(-1, 1, 2)
    frame_pts = np.ascontiguousarray(stars_frame[pairs[:, 1], :2], dtype=np.float32).reshape(
        -1, 1, 2
    )
    homography, _mask = cv2.findHomography(
        frame_pts,
        ref_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=MATCH_HOMOGRAPHY_RANSAC_THRESHOLD_PX,
    )
    if homography is None or not np.all(np.isfinite(homography)):
        return current_homography
    return homography.astype(np.float64)


def match_progressive(stars_ref, stars_frame, prior_shift, frame_shape):
    """``(pairs, stage_yields)`` — a coarse-to-fine matching cascade that recovers
    edge stars a single-tolerance ``match_stars`` pass starves on.

    ``coarse_shift``'s ``prior_shift`` is a translation only; a real frame-to-frame
    motion also carries a small ROTATION, which displaces a star by roughly
    ``radius * angle_rad`` — negligible near the frame centre, tens of px at the
    corner. A single ``MATCH_TOLERANCE_PX`` (3.0) window can't cover both, so this
    bootstraps a model from the safe central band and WIDENS before it ever
    narrows again:

    1. **seed** — match only the central ``MATCH_SEED_RADIUS_FRACTION`` band, under
       ``prior_shift`` alone, at the wide ``MATCH_SEED_TOLERANCE_PX`` (12 px).
    2. Fit a similarity from that seed (``_estimate_transform`` — the same path
       ``fit_similarity``/``register`` use).
    3. **wide** — re-match EVERY star (no brightness cap) under that similarity, at
       the GENEROUS ``MATCH_WIDE_TOLERANCE_PX`` (25 px). A similarity fitted only on
       central points is itself off by more than a few px at the frame edge, so
       narrowing the window here -- before any model capable of placing an edge
       star exists -- is exactly what starves it; this stage widens instead.
    4. Fit a homography (``cv2.findHomography``, RANSAC) from that wide set — the
       first model in the cascade with the freedom (8 DOF) to actually follow a
       frame edge a similarity can't. This homography is a SEARCH-WINDOW model
       ONLY: it decides where the next stages look for a counterpart. It is never
       returned as, or used as, the registration transform — ``register()`` still
       fits its own similarity from whatever pairs this function returns, so this
       cascade does not change the registration geometry, only what starves before
       reaching it.
    5. **tighten_10px / tighten_4px / final** — re-match EVERY star under the
       homography at each of ``MATCH_TIGHTEN_TOLERANCES_PX`` (10, 4, then
       ``MATCH_TOLERANCE_PX`` = 3.0, unchanged), refitting the homography after
       each step. The last step's pairs are the returned set.

    A step that can't support a fit (too few pairs, or the fit itself is refused)
    degrades gracefully: the previous model carries forward rather than raising.
    The returned pairs are the largest-yield set seen across every stage, so a
    late stage that ends up worse than an earlier one never throws away better
    matches already found. ``stage_yields`` maps each stage name above to its
    match count, so a caller can see where a fit starved.
    """
    dx, dy = prior_shift
    affine = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    best_pairs = np.empty((0, 2), dtype=int)
    stage_yields = {}

    ref_band = np.flatnonzero(
        _radius_fraction(stars_ref[:, :2], frame_shape) < MATCH_SEED_RADIUS_FRACTION
    )
    frame_band = np.flatnonzero(
        _radius_fraction(stars_frame[:, :2], frame_shape) < MATCH_SEED_RADIUS_FRACTION
    )
    seed_pairs = _match_under_transform(
        stars_ref,
        stars_frame,
        affine,
        MATCH_SEED_TOLERANCE_PX,
        ref_idx=ref_band,
        frame_idx=frame_band,
    )
    stage_yields["seed"] = len(seed_pairs)
    if len(seed_pairs) > len(best_pairs):
        best_pairs = seed_pairs
    affine = _refit_or_keep(stars_ref, stars_frame, seed_pairs, affine, frame_shape)

    wide_pairs = _match_under_transform(stars_ref, stars_frame, affine, MATCH_WIDE_TOLERANCE_PX)
    stage_yields["wide"] = len(wide_pairs)
    if len(wide_pairs) > len(best_pairs):
        best_pairs = wide_pairs

    homography = _fit_homography_or_keep(stars_ref, stars_frame, wide_pairs, None)
    final_pairs = wide_pairs
    tighten_labels = ("tighten_10px", "tighten_4px", "final")
    for label, tolerance_px in zip(tighten_labels, MATCH_TIGHTEN_TOLERANCES_PX):
        if homography is None:
            stage_yields[label] = 0
            continue
        tightened = _match_under_homography(stars_ref, stars_frame, homography, tolerance_px)
        stage_yields[label] = len(tightened)
        if len(tightened) > len(best_pairs):
            best_pairs = tightened
        if len(tightened) > 0:
            final_pairs = tightened
        homography = _fit_homography_or_keep(stars_ref, stars_frame, tightened, homography)

    if len(final_pairs) >= len(best_pairs):
        best_pairs = final_pairs

    return best_pairs, stage_yields


def _validate_transform(transform, frame_shape=None):
    """Reject a fitted transform that RANSAC converged on but that isn't a plausible
    frame-to-frame similarity: non-finite entries, a degenerate/absurd scale, an
    excessive rotation, or (when ``frame_shape`` is known) a translation implausibly
    large for the frame. ``cv2.estimateAffinePartial2D``'s 4-DOF form is
    ``[[a, -b, tx], [b, a, ty]]``, so ``a*a + b*b`` can never be negative -- a
    reflection check would be dead code here, not a real failure mode.
    """
    if not np.all(np.isfinite(transform)):
        raise RegistrationError(f"fitted transform has non-finite entries: {transform.tolist()}")
    a, b = float(transform[0, 0]), float(transform[1, 0])
    scale = float(np.hypot(a, b))
    if not (FIT_MIN_SCALE <= scale <= FIT_MAX_SCALE):
        raise RegistrationError(
            f"fitted scale {scale:.4f} outside broad sanity band [{FIT_MIN_SCALE}, {FIT_MAX_SCALE}]"
        )
    rotation_deg = float(np.degrees(np.arctan2(b, a)))
    if abs(rotation_deg) > FIT_MAX_ROTATION_DEG:
        raise RegistrationError(
            f"fitted rotation {rotation_deg:.2f} deg exceeds broad cap {FIT_MAX_ROTATION_DEG} deg"
        )
    if frame_shape is not None:
        tx, ty = float(transform[0, 2]), float(transform[1, 2])
        translation_px = float(np.hypot(tx, ty))
        max_translation_px = FIT_MAX_TRANSLATION_FRAME_FRACTION * max(frame_shape[:2])
        if translation_px > max_translation_px:
            raise RegistrationError(
                f"fitted translation {translation_px:.1f}px exceeds "
                f"{FIT_MAX_TRANSLATION_FRAME_FRACTION:.0%} of the frame's longer side "
                f"({max_translation_px:.1f}px)"
            )


def fit_residuals_px(transform, matched_ref_xy, matched_frame_xy):
    """``(median, p95)`` px residual between the fitted transform's prediction and
    each matched reference position -- the per-pair fit error, not the RANSAC
    inlier/outlier split (geometry gates). Empty input has
    no residual to report."""
    if len(matched_frame_xy) == 0:
        return 0.0, 0.0
    frame_xy = np.asarray(matched_frame_xy, dtype=np.float64)
    ref_xy = np.asarray(matched_ref_xy, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    predicted = frame_xy @ transform[:, :2].T + transform[:, 2]
    diff = predicted - ref_xy
    residuals = np.hypot(diff[:, 0], diff[:, 1])
    return float(np.median(residuals)), float(np.percentile(residuals, 95))


def radial_inlier_residuals_px(transform, matched_ref_xy, matched_frame_xy, inliers, frame_shape):
    """Median/p95 fit residuals for RANSAC inliers in radial sensor thirds."""
    ref_xy = np.asarray(matched_ref_xy, dtype=np.float64)
    frame_xy = np.asarray(matched_frame_xy, dtype=np.float64)
    inliers = np.asarray(inliers, dtype=bool).reshape(-1)
    predicted = frame_xy @ np.asarray(transform)[:, :2].T + np.asarray(transform)[:, 2]
    residuals = np.linalg.norm(predicted - ref_xy, axis=1)
    height, width = frame_shape[:2]
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    max_radius = np.linalg.norm(center)
    radius = np.linalg.norm(ref_xy - center, axis=1) / max(max_radius, 1.0)
    bins = {
        "center": radius < 1 / 3,
        "middle": (radius >= 1 / 3) & (radius < 2 / 3),
        "outer": radius >= 2 / 3,
    }
    result = {}
    for name, selected in bins.items():
        values = residuals[selected & inliers]
        result[name] = {
            "n": int(values.size),
            "median_px": float(np.median(values)) if values.size else None,
            "p95_px": float(np.percentile(values, 95)) if values.size else None,
        }
    return result


def _estimate_transform(matched_ref_xy, matched_frame_xy, frame_shape=None):
    if len(matched_ref_xy) < MIN_MATCH_STARS:
        raise RegistrationError(
            f"only {len(matched_ref_xy)} matched star pairs, need >= {MIN_MATCH_STARS}"
        )
    # A sliced (non-contiguous) input must be copied first — cv2's point-count
    # assertion reads the buffer's own strides.
    ref_pts = np.ascontiguousarray(matched_ref_xy, dtype=np.float32).reshape(-1, 1, 2)
    frame_pts = np.ascontiguousarray(matched_frame_xy, dtype=np.float32).reshape(-1, 1, 2)
    transform, inliers = cv2.estimateAffinePartial2D(
        frame_pts, ref_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD_PX
    )
    if transform is None:
        raise RegistrationError("similarity fit did not converge (degenerate point set)")
    transform = transform.astype(np.float64)
    _validate_transform(transform, frame_shape=frame_shape)
    n_inliers = int(inliers.sum())
    if n_inliers < MIN_INLIER_STARS:
        raise RegistrationError(f"fit kept only {n_inliers} inliers, need >= {MIN_INLIER_STARS}")
    inlier_fraction = float(inliers.sum()) / len(inliers)
    if inlier_fraction < MIN_INLIER_FRACTION:
        raise RegistrationError(
            f"fit inlier fraction {inlier_fraction:.2f} below floor {MIN_INLIER_FRACTION}"
        )
    return transform, inlier_fraction, np.asarray(inliers, dtype=bool).reshape(-1)


def fit_similarity(matched_ref_xy, matched_frame_xy):
    """2x3 float64 similarity transform, matched FRAME xy -> matched REFERENCE xy.

    RANSAC similarity fit; raises ``RegistrationError`` for too few matches, a
    non-convergent fit, a fitted transform outside the broad sanity bounds, or an
    inlier share below the floor.
    """
    transform, _, _ = _estimate_transform(matched_ref_xy, matched_frame_xy)
    return transform


def register(luma_ref, stars_ref, luma_frame, stars_frame, ref_prepared=None, sky_mask=None):
    """Register one frame against the reference: the three stages, composed.

    ``sky_mask`` is the same boolean sky/ground split the caller already detects stars
    under; passing it keeps stationary foreground out of the coarse prior (see
    ``coarse_shift``). Callers that have one should always pass it -- without it, a
    landscape frame's prior can miss by enough to lose every star pair.

    ``ref_prepared`` is an optional ``prepare_reference(luma_ref, sky_mask)`` result:
    pass the same one across every frame registered against a shared reference so
    ``coarse_shift`` doesn't re-crop and re-downscale it each call; it carries its own row
    band, so ``sky_mask`` is redundant alongside it. Omitted with no ``sky_mask``, behavior
    is whole-frame correlation, with the reference downscaled on every call.
    """
    prior_shift = coarse_shift(luma_ref, luma_frame, ref_prepared=ref_prepared, sky_mask=sky_mask)
    frame_shape = luma_ref.shape[:2]
    pairs, stage_yields = match_progressive(stars_ref, stars_frame, prior_shift, frame_shape)
    matched_ref_xy = stars_ref[pairs[:, 0], :2] if len(pairs) else pairs
    matched_frame_xy = stars_frame[pairs[:, 1], :2] if len(pairs) else pairs
    transform, inlier_fraction, inliers = _estimate_transform(
        matched_ref_xy, matched_frame_xy, frame_shape=frame_shape
    )
    residual_median_px, residual_p95_px = fit_residuals_px(
        transform, matched_ref_xy[inliers], matched_frame_xy[inliers]
    )
    radial_residuals = radial_inlier_residuals_px(
        transform, matched_ref_xy, matched_frame_xy, inliers, frame_shape
    )
    outer_n = radial_residuals["outer"]["n"]
    if outer_n < MIN_OUTER_THIRD_INLIERS:
        raise RegistrationError(
            f"fit kept only {outer_n} outer-third inliers, need >= {MIN_OUTER_THIRD_INLIERS} "
            "-- this transform only fit the frame's centre, not the whole sensor"
        )
    smaller_detected = min(len(stars_ref), len(stars_frame))
    match_yield = len(pairs) / smaller_detected if smaller_detected else 0.0
    return RegistrationResult(
        transform=transform,
        n_matched=len(pairs),
        inlier_fraction=inlier_fraction,
        residual_median_px=residual_median_px,
        residual_p95_px=residual_p95_px,
        match_yield=match_yield,
        radial_inlier_residuals_px=radial_residuals,
        match_yield_by_stage=stage_yields,
    )


# cv2 warp call configuration shared by ``warp_to_reference`` (RGB) and
# ``warp_coverage`` (validity footprint) -- named constants, not duplicated
# literals, so the two calls can never silently drift apart and coverage stops
# describing what the RGB warp actually did ("use the SAME interpolation
# footprint as the RGB warp").
_WARP_INTERPOLATION = cv2.INTER_LINEAR
_WARP_BORDER_MODE = cv2.BORDER_CONSTANT
_WARP_BORDER_VALUE = 0

# Coverage fraction below which a warped pixel counts as invalid
# (PROVISIONAL). Under BORDER_CONSTANT=0 + INTER_LINEAR, a partially-covered edge
# pixel's interpolated value is a straight blend of real signal and zero-fill, and
# this coverage fraction IS exactly the real-signal share of that blend. 0.5 is the
# threshold the design notes name -- "border zeros can become the majority" -- so a
# pixel is valid only while real coverage still outweighs zero-fill contamination.
MIN_VALID_COVERAGE = 0.5


def _is_remap_pair(transform):
    """``True`` when ``transform`` is a ``(map_x, map_y)`` pair (``astro_lens.
    build_remap``'s return shape) rather than today's 2x3 affine matrix.

    A 2x3 matrix is itself a 2-element sequence of rows, so this checks each
    element's shape, not just ``len(transform) == 2`` -- a plain
    ``(row0, row1)`` affine would otherwise be misread as a remap pair.
    """
    if isinstance(transform, np.ndarray):
        return False
    if len(transform) != 2:
        return False
    map_x, map_y = transform
    map_x = np.asarray(map_x)
    map_y = np.asarray(map_y)
    return map_x.ndim == 2 and map_x.shape == map_y.shape


def _warp_array(array, transform, dsize):
    """Shared ``cv2`` call behind ``warp_to_reference``/``warp_coverage``: a 2x3
    affine matrix goes through ``cv2.warpAffine`` (today's path, unchanged for
    the trails/timelapse callers); a ``(map_x, map_y)`` pair (``astro_lens.
    build_remap``) goes through ``cv2.remap``. Same interpolation flag, same
    border mode/value, either way -- the two callers can never silently drift
    apart regardless of which transform shape they're given."""
    if _is_remap_pair(transform):
        map_x, map_y = transform
        return cv2.remap(
            array,
            np.asarray(map_x, dtype=np.float32),
            np.asarray(map_y, dtype=np.float32),
            interpolation=_WARP_INTERPOLATION,
            borderMode=_WARP_BORDER_MODE,
            borderValue=_WARP_BORDER_VALUE,
        )
    return cv2.warpAffine(
        array,
        np.asarray(transform, dtype=np.float32),
        dsize,
        flags=_WARP_INTERPOLATION,
        borderMode=_WARP_BORDER_MODE,
        borderValue=_WARP_BORDER_VALUE,
    )


def warp_to_reference(rgb, transform, out_shape):
    """Warp ``rgb`` into the reference frame using either a ``register`` 2x3
    similarity transform (trails/timelapse, unchanged) or a ``astro_lens.
    build_remap`` ``(map_x, map_y)`` pair.

    Pure, ``INTER_LINEAR``, zero-filled border, float32 preserved —
    consumed later by ``astro_stack``/``astro_meteors`` via the orchestrator.
    """
    dsize = (int(out_shape[1]), int(out_shape[0]))
    return _warp_array(np.asarray(rgb, dtype=np.float32), transform, dsize)


def warp_coverage(transform, in_shape, out_shape):
    """Float32 coverage weight in ``[0, 1]`` per output pixel: how much
    of that pixel's ``warp_to_reference`` value traces back to real source pixels
    rather than the zero-fill border.

    Warps a ones-valued array shaped like the RGB warp's INPUT (``in_shape``)
    through the identical warp configuration ``warp_to_reference`` uses -- same
    transform (2x3 matrix or ``build_remap`` pair), same interpolation flag,
    same border mode/value, same output size -- so a partially-covered edge
    pixel is judged by the exact footprint the RGB warp actually produced
    there, not an approximation of it.
    """
    dsize = (int(out_shape[1]), int(out_shape[0]))
    unit = np.ones(tuple(in_shape[:2]), dtype=np.float32)
    return _warp_array(unit, transform, dsize)


def warp_validity(transform, in_shape, out_shape, min_coverage=MIN_VALID_COVERAGE):
    """Boolean validity mask: ``True`` where ``warp_coverage`` meets
    ``min_coverage`` (default ``MIN_VALID_COVERAGE`` -- see its comment for why)."""
    return warp_coverage(transform, in_shape, out_shape) >= min_coverage


def contributor_count(validity_masks):
    """Per-pixel count of how many ``warp_validity`` masks call that pixel valid --
    the coverage/contributor contract: a pixel backed by only one registered
    frame has no stacking gain even though it individually clears
    ``MIN_VALID_COVERAGE``."""
    stacked = np.stack([np.asarray(m, dtype=bool) for m in validity_masks], axis=0)
    return stacked.sum(axis=0).astype(np.int32)


def _max_rectangle_in_histogram(heights):
    """``(area, col_start, col_stop, height)`` of the largest all-nonzero
    rectangle in a 1-D histogram -- the per-row step of the largest-rectangle-in-
    binary-matrix algorithm ``common_support_crop`` builds on."""
    stack = []
    best = (0, 0, 0, 0)
    n = len(heights)
    for i in range(n + 1):
        h = heights[i] if i < n else 0
        start = i
        while stack and stack[-1][1] >= h:
            idx, height = stack.pop()
            area = height * (i - idx)
            if area > best[0]:
                best = (area, idx, i, height)
            start = idx
        stack.append((start, h))
    return best


def common_support_crop(counts, min_contributors):
    """Largest axis-aligned ``(row_slice, col_slice)`` crop of ``counts`` (a
    ``contributor_count`` result) where every pixel meets ``min_contributors``, or
    ``None`` when no pixel does.

    Registration can rotate as well as translate, so the meets-floor region isn't
    guaranteed to be a clean bounding box; this runs the histogram/monotonic-stack
    largest-rectangle-in-binary-matrix algorithm (O(rows*cols)) so the returned
    crop is guaranteed ENTIRELY valid, never a box that merely contains some
    invalid pixels too.

    Documented fallback: ``None`` means no rectangle clears the floor at
    all -- the caller must lower ``min_contributors`` or treat the frame set as
    unusable for this stack; never slice on a ``None`` result.
    """
    mask = np.asarray(counts) >= min_contributors
    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=np.int64)
    best_area = 0
    best = None
    for r in range(rows):
        heights = np.where(mask[r], heights + 1, 0)
        area, left, right, height = _max_rectangle_in_histogram(heights)
        if area > best_area:
            best_area = area
            best = (r + 1 - height, r + 1, left, right)
    if best is None:
        return None
    row_start, row_stop, col_start, col_stop = best
    return slice(row_start, row_stop), slice(col_start, col_stop)
