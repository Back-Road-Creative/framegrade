"""Best-of-look-alike-group photo selection for a route-story film.

Photographers shoot several frames of the same view; a film should show only the
best one of each look-alike group and must never drop a photo merely because it
is close in time or place to another — the visual test is what actually decides
membership.

Pure library: the caller decodes RAW/JPEG into an RGB preview and supplies a
capture time and (optionally) a placement; nothing here touches the filesystem
or EXIF. Bracketed exposures (``gomoveshift/pipeline/photos/bracket.py``) are
out of scope for this module — that detector needs file paths/EXIF and groups a
different concept (metered exposure runs, not look-alike views).

Grouping is chain-linked along the capture-time-sorted sequence: consecutive
candidates join one group only when EVERY test passes between that adjacent
pair — capture-time gap, placement distance (skipped when either candidate
lacks one), and the visual "same view" test (ORB keypoint matching with a
histogram-correlation fallback for low-texture frames). A group is scored by
sharpness first (tile-based Laplacian variance, so a shallow-focus subject
isn't punished by a soft background), with ties (within ``tie_fraction`` of
the group's best) broken by exposure, then by colour vibrancy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import asin, cos, radians, sin, sqrt
from typing import Any, Sequence, Union

import cv2
import numpy as np
from PIL import Image

from gomoveshift.analysis.visual_scorer import score_color_vibrancy

# Visual "same view" test: ORB on a downsized grayscale copy, ratio-tested and
# RANSAC-verified; falls back to HSV histogram correlation when either frame
# yields too few keypoints (haze, open sky).
ORB_MATCH_WIDTH = 1024
MIN_KEYPOINTS_FOR_ORB = 20
LOWE_RATIO_THRESHOLD = 0.75
MIN_GOOD_MATCHES = 4
RANSAC_REPROJ_THRESHOLD = 5.0
HIST_MATCH_WIDTH = 256

# Sharpness: mean of the top-10% tile-variance grid, so a soft background
# behind a sharp shallow-focus subject doesn't drag the frame down.
SHARPNESS_WIDTH = 1500
SHARPNESS_GRID = 10
SHARPNESS_TOP_FRACTION = 0.1

# Exposure tie-break: fraction of near-clipped pixels, smaller wins -- but only
# a real difference decides (see ``SelectionOptions.exposure_tie_fraction``).
EXPOSURE_WIDTH = 512
EXPOSURE_HIGH_CLIP = 250
EXPOSURE_LOW_CLIP = 5

EARTH_RADIUS_M = 6_371_000.0

Placement = tuple[float, float]
ImageLike = Union[Image.Image, np.ndarray]


@dataclass(frozen=True)
class PhotoCandidate:
    """One decoded photo preview offered to the selector."""

    name: str
    image: ImageLike
    captured_at: float
    placement: Placement | None = None


@dataclass(frozen=True)
class SelectionOptions:
    """Tunable thresholds; defaults are tuned on real look-alike bursts."""

    enabled: bool = True
    max_gap_s: float = 120.0
    max_distance_m: float = 50.0
    min_inliers: int = 25
    min_inlier_ratio: float = 0.6
    hist_corr_threshold: float = 0.9
    tie_fraction: float = 0.10
    # Clipped-pixel fractions closer than this tie: a handful of blown pixels
    # is sensor noise, not an exposure fault, and must not pick the winner.
    exposure_tie_fraction: float = 0.002
    keep: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class PhotoScore:
    """Per-photo scoring signals recorded for the receipt."""

    name: str
    sharpness: float
    exposure_badness: float
    color_vibrancy: float


@dataclass(frozen=True)
class GroupResult:
    """One look-alike group: who's in it, who won, and why."""

    members: tuple[str, ...]
    winner: str
    kept: tuple[str, ...]
    losers: tuple[str, ...]
    scores: dict[str, PhotoScore]
    reason: str


@dataclass(frozen=True)
class SelectionResult:
    """The full selection: every group plus the flat kept list."""

    groups: tuple[GroupResult, ...]
    kept: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, JSON-serialisable for a receipt."""
        return asdict(self)


def _to_rgb(image: ImageLike) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    return array


def _resize_to_width(rgb: np.ndarray, width: int) -> np.ndarray:
    height, source_width = rgb.shape[:2]
    if source_width <= width:
        return rgb
    scale = width / source_width
    target = (width, max(1, int(round(height * scale))))
    return cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)


def _grayscale(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _distance_m(a: Placement, b: Placement) -> float:
    lat1, lon1 = radians(a[0]), radians(a[1])
    lat2, lon2 = radians(b[0]), radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    haversine = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(min(1.0, haversine)))


def _hist_same_view(rgb1: np.ndarray, rgb2: np.ndarray, options: SelectionOptions) -> bool:
    small1 = cv2.resize(rgb1, (HIST_MATCH_WIDTH, HIST_MATCH_WIDTH), interpolation=cv2.INTER_AREA)
    small2 = cv2.resize(rgb2, (HIST_MATCH_WIDTH, HIST_MATCH_WIDTH), interpolation=cv2.INTER_AREA)
    hsv1 = cv2.cvtColor(small1, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(small2, cv2.COLOR_RGB2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [8, 8], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return bool(correlation >= options.hist_corr_threshold)


def _orb_same_view(rgb1: np.ndarray, rgb2: np.ndarray, options: SelectionOptions) -> bool:
    gray1 = _grayscale(_resize_to_width(rgb1, ORB_MATCH_WIDTH))
    gray2 = _grayscale(_resize_to_width(rgb2, ORB_MATCH_WIDTH))
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    if (
        descriptors1 is None
        or descriptors2 is None
        or len(keypoints1) < MIN_KEYPOINTS_FOR_ORB
        or len(keypoints2) < MIN_KEYPOINTS_FOR_ORB
    ):
        return _hist_same_view(rgb1, rgb2, options)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < LOWE_RATIO_THRESHOLD * second.distance:
            good.append(best)
    if len(good) < MIN_GOOD_MATCHES:
        return False

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(points1, points2, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
    if mask is None:
        return False
    inliers = int(mask.sum())
    ratio = inliers / len(good)
    return inliers >= options.min_inliers and ratio >= options.min_inlier_ratio


def _same_view_pair(a: PhotoCandidate, b: PhotoCandidate, options: SelectionOptions) -> bool:
    if abs(b.captured_at - a.captured_at) > options.max_gap_s:
        return False
    if a.placement is not None and b.placement is not None:
        if _distance_m(a.placement, b.placement) > options.max_distance_m:
            return False
    return _orb_same_view(_to_rgb(a.image), _to_rgb(b.image), options)


def _sharpness_score(rgb: np.ndarray) -> float:
    gray = _grayscale(_resize_to_width(rgb, SHARPNESS_WIDTH))
    height, width = gray.shape
    tile_h = max(1, height // SHARPNESS_GRID)
    tile_w = max(1, width // SHARPNESS_GRID)
    variances = []
    for row in range(SHARPNESS_GRID):
        for col in range(SHARPNESS_GRID):
            tile = gray[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w]
            if tile.size == 0:
                continue
            variances.append(float(cv2.Laplacian(tile, cv2.CV_64F).var()))
    if not variances:
        return 0.0
    variances.sort(reverse=True)
    top_n = max(1, int(len(variances) * SHARPNESS_TOP_FRACTION))
    return float(np.mean(variances[:top_n]))


def _exposure_badness(rgb: np.ndarray) -> float:
    gray = _grayscale(_resize_to_width(rgb, EXPOSURE_WIDTH))
    total = gray.size
    if total == 0:
        return 0.0
    bad = int(np.count_nonzero((gray >= EXPOSURE_HIGH_CLIP) | (gray <= EXPOSURE_LOW_CLIP)))
    return bad / total


def _color_vibrancy(rgb: np.ndarray) -> float:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return float(score_color_vibrancy(bgr))


def _score_candidate(candidate: PhotoCandidate) -> PhotoScore:
    rgb = _to_rgb(candidate.image)
    return PhotoScore(
        name=candidate.name,
        sharpness=_sharpness_score(rgb),
        exposure_badness=_exposure_badness(rgb),
        color_vibrancy=_color_vibrancy(rgb),
    )


def _pick_winner(
    members: Sequence[PhotoCandidate],
    scores: dict[str, PhotoScore],
    options: SelectionOptions,
) -> tuple[str, str]:
    best_sharpness = max(scores[m.name].sharpness for m in members)
    sharpness_floor = best_sharpness * (1.0 - options.tie_fraction)
    tied = [m for m in members if scores[m.name].sharpness >= sharpness_floor]
    if len(tied) == 1:
        return tied[0].name, "sharpest"

    best_exposure = min(scores[m.name].exposure_badness for m in tied)
    tied_exposure = [
        m
        for m in tied
        if scores[m.name].exposure_badness - best_exposure <= options.exposure_tie_fraction
    ]
    if len(tied_exposure) == 1:
        return tied_exposure[0].name, "sharpness tie broken by exposure"

    best_color = max(scores[m.name].color_vibrancy for m in tied_exposure)
    tied_color = [m for m in tied_exposure if scores[m.name].color_vibrancy == best_color]
    winner = min(tied_color, key=lambda m: m.name)
    if len(tied_color) == 1:
        return winner.name, "sharpness/exposure tie broken by colour vibrancy"
    return winner.name, "sharpness/exposure/colour tie broken by name order"


def select_best_photos(
    candidates: Sequence[PhotoCandidate],
    options: SelectionOptions | None = None,
) -> SelectionResult:
    """Pick the best photo of each look-alike group.

    Deterministic and input-order independent: candidates are processed in
    ``(captured_at, name)`` order regardless of the order they were given in.
    """
    resolved_options = options or SelectionOptions()
    ordered = sorted(candidates, key=lambda c: (c.captured_at, c.name))

    if not resolved_options.enabled:
        groups = tuple(
            GroupResult(
                members=(c.name,),
                winner=c.name,
                kept=(c.name,),
                losers=(),
                scores={c.name: _score_candidate(c)},
                reason="selection disabled",
            )
            for c in ordered
        )
        return SelectionResult(groups=groups, kept=tuple(c.name for c in ordered))

    grouped: list[list[PhotoCandidate]] = []
    for candidate in ordered:
        if grouped and _same_view_pair(grouped[-1][-1], candidate, resolved_options):
            grouped[-1].append(candidate)
        else:
            grouped.append([candidate])

    group_results = []
    kept_names: set[str] = set()
    for members in grouped:
        scores = {m.name: _score_candidate(m) for m in members}
        member_names = tuple(m.name for m in members)
        if len(members) == 1:
            winner_name, reason = member_names[0], "only member"
        else:
            winner_name, reason = _pick_winner(members, scores, resolved_options)

        kept_set = {winner_name} | (set(member_names) & resolved_options.keep)
        if kept_set - {winner_name}:
            reason = f"{reason} (kept override applied)"
        kept_ordered = tuple(name for name in member_names if name in kept_set)
        losers = tuple(name for name in member_names if name not in kept_set)

        group_results.append(
            GroupResult(
                members=member_names,
                winner=winner_name,
                kept=kept_ordered,
                losers=losers,
                scores=scores,
                reason=reason,
            )
        )
        kept_names.update(kept_ordered)

    kept = tuple(c.name for c in ordered if c.name in kept_names)
    return SelectionResult(groups=tuple(group_results), kept=kept)
