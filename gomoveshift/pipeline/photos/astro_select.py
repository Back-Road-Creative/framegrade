"""Frame selection: picks the frames a stack is built from out of a night's measured
metrics. PURE decision function over numbers -- no I/O, no decoding, no orchestrator
import. Nothing in this module is wired into the pipeline yet; a later change builds
``FrameRecord``s from ``astro_triage.FrameMetrics`` + EXIF exposure + a
``astro_triage.TriageReport`` and calls ``select_frames``.

Selection order:
  1. exposure hold -- drop any frame whose exposure isn't within
     ``astro_sequence.MAX_HOLD_RATIO`` of the run's median exposure (the modal hold).
  2. condition/quality gates already judged elsewhere -- unusable, cloud, dew.
  3. score the survivors on measured sharpness/roundness/sky darkness/star count.
  4. keep the best ``keep_fraction``, floored at ``min_frames`` (or everything, if the
     run has fewer candidate frames than the floor to begin with).
"""

from dataclasses import dataclass
from pathlib import Path

from .astro_sequence import MAX_HOLD_RATIO

# PROVISIONAL: first guesses, not measured optima.
DEFAULT_KEEP_FRACTION = 0.70
DEFAULT_MIN_FRAMES = 100

REASON_EXPOSURE_UNMEASURED = "exposure_unmeasured"
REASON_EXPOSURE_OUTLIER = "exposure_outlier"
REASON_UNUSABLE = "unusable"
REASON_CLOUD = "cloud"
REASON_DEW = "dew"
REASON_UNMEASURED_QUALITY = "unmeasured_quality"
REASON_BELOW_KEEP_THRESHOLD = "below_keep_threshold"


@dataclass(frozen=True)
class FrameRecord:
    """One frame's selection inputs -- the narrow subset ``select_frames`` needs,
    decoupled from ``astro_triage.FrameMetrics`` (which carries no exposure) and from
    any EXIF/orchestrator type. ``None`` on any quality field means that quantity was
    never measured for this frame -- it is NEVER treated as a zero/best/worst score;
    see ``select_frames`` for exactly how an unmeasured field is handled.
    """

    path: Path
    exposure_s: float | None  # None = exposure unknown (EXIF missing/unreadable)
    usable: bool  # astro_triage.FrameMetrics.usable
    cloud: bool  # this frame's path was in the run's TriageReport.cloud_frames
    dew: bool  # this frame's path was in the run's TriageReport.dew_frames
    star_count: int | None = None
    background_median: float | None = None
    fwhm_px: float | None = None
    ellipticity: float | None = None


@dataclass(frozen=True)
class SelectionPolicy:
    """PROVISIONAL defaults -- first guesses, not measured optima."""

    keep_fraction: float = DEFAULT_KEEP_FRACTION
    min_frames: int = DEFAULT_MIN_FRAMES
    max_hold_ratio: float = MAX_HOLD_RATIO


@dataclass(frozen=True)
class FrameSelection:
    """One frame's outcome. ``score`` is None only for a frame dropped before scoring
    (exposure outlier/unmeasured, unusable, cloud, dew, or all quality fields
    unmeasured); every scored frame keeps its score whether selected or not, so a
    dropped-at-the-threshold frame is still inspectable."""

    path: Path
    score: float | None
    selected: bool
    reason: str | None  # None iff selected


@dataclass(frozen=True)
class SelectionResult:
    selections: tuple[FrameSelection, ...]  # one entry per input frame, input order
    selected: tuple[Path, ...]  # kept paths, best score first
    floor_applied: bool  # min_frames raised the keep count above the raw fraction
    kept_all_below_floor: bool  # fewer scorable candidates than min_frames: kept all


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _within_hold(a, b, ratio):
    """Mirrors astro_sequence's own hold check: the larger of the pair divided by the
    smaller must be strictly under ``ratio``."""
    if a <= 0 or b <= 0:
        return False
    return max(a, b) / min(a, b) < ratio


def _normalize(value, lo, hi):
    """[0, 1], 0=worst-of-range 1=best-of-range on the RAW value's own scale (caller
    flips direction). A flat range (lo == hi) carries no discriminating information --
    neutral 0.5 for every frame rather than an arbitrary 0 or 1."""
    if hi <= lo:
        return 0.5
    return (value - lo) / (hi - lo)


def _quality_terms(record, ranges):
    """List of (0=worst, 1=best) term scores for whichever of the four quality fields
    ``record`` actually has measured. A field left at None contributes NOTHING --
    never a synthetic 0 -- so a frame's score is the mean of what was actually
    measured about it, not diluted or inflated by absent data."""
    terms = []
    if record.fwhm_px is not None:  # sharper (lower) is better
        lo, hi = ranges["fwhm_px"]
        terms.append(1.0 - _normalize(record.fwhm_px, lo, hi))
    if record.ellipticity is not None:  # rounder (lower) is better
        lo, hi = ranges["ellipticity"]
        terms.append(1.0 - _normalize(record.ellipticity, lo, hi))
    if record.background_median is not None:  # darker sky (lower) is better
        lo, hi = ranges["background_median"]
        terms.append(1.0 - _normalize(record.background_median, lo, hi))
    if record.star_count is not None:  # more stars is better
        lo, hi = ranges["star_count"]
        terms.append(_normalize(record.star_count, lo, hi))
    return terms


def select_frames(records, policy=None):
    """Select the frames a stack should be built from.

    ``records``: an iterable of ``FrameRecord``. ``policy``: ``SelectionPolicy``,
    defaults applied when omitted. Returns a ``SelectionResult`` covering every input
    frame, selected or not, each with its score (if reachable) and, when dropped,
    exactly which rule dropped it -- the record that later feeds stack provenance.

    Scoring (step 3): each survivor's score is the MEAN, over whichever of
    {fwhm_px, ellipticity, background_median, star_count} it has measured, of that
    field's min-max position across this run's survivors, oriented so 1.0 is always
    "better" (fwhm/ellipticity/background_median inverted since lower is better;
    star_count is not, since higher is better). A field the frame never measured is
    dropped from ITS OWN mean, never substituted with a zero -- an honest absence
    changes what is averaged, not the frame's apparent quality. A frame with NONE of
    the four measured cannot be ranked at all and is dropped outright
    (``unmeasured_quality``), not silently included at the bottom.
    """
    policy = policy or SelectionPolicy()
    records = tuple(records)
    selections_by_path = {}
    order = []

    if not records:
        return SelectionResult((), (), False, False)

    exposures = [r.exposure_s for r in records if r.exposure_s is not None and r.exposure_s > 0]
    modal_exposure = _median(exposures) if exposures else None

    survivors = []
    for r in records:
        order.append(r.path)
        if r.exposure_s is None:
            selections_by_path[r.path] = FrameSelection(
                r.path, None, False, REASON_EXPOSURE_UNMEASURED
            )
            continue
        if modal_exposure is None or not _within_hold(
            r.exposure_s, modal_exposure, policy.max_hold_ratio
        ):
            selections_by_path[r.path] = FrameSelection(
                r.path, None, False, REASON_EXPOSURE_OUTLIER
            )
            continue
        if not r.usable:
            selections_by_path[r.path] = FrameSelection(r.path, None, False, REASON_UNUSABLE)
            continue
        if r.cloud:
            selections_by_path[r.path] = FrameSelection(r.path, None, False, REASON_CLOUD)
            continue
        if r.dew:
            selections_by_path[r.path] = FrameSelection(r.path, None, False, REASON_DEW)
            continue
        survivors.append(r)

    scorable = []
    for r in survivors:
        if (
            r.fwhm_px is None
            and r.ellipticity is None
            and r.background_median is None
            and r.star_count is None
        ):
            selections_by_path[r.path] = FrameSelection(
                r.path, None, False, REASON_UNMEASURED_QUALITY
            )
        else:
            scorable.append(r)

    ranges = {}
    for field in ("fwhm_px", "ellipticity", "background_median", "star_count"):
        values = [getattr(r, field) for r in scorable if getattr(r, field) is not None]
        ranges[field] = (min(values), max(values)) if values else (0.0, 0.0)

    scored = []
    for r in scorable:
        terms = _quality_terms(r, ranges)
        score = sum(terms) / len(terms)
        scored.append((r.path, score))

    scored.sort(key=lambda item: item[1], reverse=True)

    n = len(scored)
    if n <= policy.min_frames:
        keep_count = n
        kept_all_below_floor = True
        floor_applied = False
    else:
        frac_count = int(round(policy.keep_fraction * n))
        keep_count = max(frac_count, policy.min_frames)
        floor_applied = keep_count > frac_count
        kept_all_below_floor = False

    kept_paths = tuple(path for path, _score in scored[:keep_count])
    kept_set = set(kept_paths)
    for path, score in scored:
        if path in kept_set:
            selections_by_path[path] = FrameSelection(path, score, True, None)
        else:
            selections_by_path[path] = FrameSelection(
                path, score, False, REASON_BELOW_KEEP_THRESHOLD
            )

    selections = tuple(selections_by_path[p] for p in order)
    return SelectionResult(selections, kept_paths, floor_applied, kept_all_below_floor)
