"""Short-clip segment planner (day-first architecture).

Produces range selections for 15-55 second shorts from a scored timeline
(typically the day master's or a regular LF's per-second composite score).
Mirrors the structure of `regular_lf_planner` but with shorts-specific
constraints.

Key differences from regular LF planning:

  * Target duration is ~38 seconds (vs ~20 minutes for regular LFs)
  * Target count is ~9 per source (vs ~4 for regular LFs) — enough to
    feed a week of daily shorts publishing from one drive
  * Results return in RANK order (best integrated score first), not
    chronological order — rank becomes the publish sequence, so
    `short_01` is the day's best content.
  * `min_gap_sec` enforces spacing between chosen windows so shorts
    cover the drive's full arc rather than clustering on one peak.

Like `regular_lf_planner`, this module is pure Python with numpy. No
subprocess, no I/O. The renderer wires the planner to `render_short_ffmpeg`
(with its `skip_ost` flag) and `ost_applier` with `layout="short"`.
"""

import logging
from dataclasses import dataclass

import numpy as np

from .destination_resolver import MIN_PEAK_MASS_SHARE

logger = logging.getLogger(__name__)

# Score-array column indices (mirror unified_scorer)
COL_TIMESTAMP = 0
COL_COMPOSITE = 1

# Canonical short duration bounds (clip length
# follows its moment rather than a fixed target) — 15s covers a punchy
# single moment, 55s a sustained one; YouTube Shorts hard ceiling is 60s
# (video_types.SHORTS_MAX_DURATION).
DEFAULT_MIN_DURATION_SEC = 15.0
DEFAULT_MAX_DURATION_SEC = 55.0
DEFAULT_TARGET_DURATION_SEC = 38.0
DEFAULT_TARGET_COUNT = 9
DEFAULT_MIN_GAP_SEC = 60.0
DEFAULT_STRIDE_SEC = 5.0
# A window must beat the bar by more than float summation noise: on a flat
# timeline a window's mean can differ from the timeline mean by ~1e-17.
_BAR_EPSILON = 1e-9
# A Short is judged on how it opens: the mean composite of a window's first
# OPENING_SECONDS seconds must clear the quality bar, and ranking weights it.
OPENING_SECONDS = 4
# A window inside a measured landmark polygon ranks higher: its pick-order score
# is multiplied by (1 + LANDMARK_BONUS). Soft and ranking-only. The window must
# hold at least the resolver's Shorts quorum (MIN_PEAK_MASS_SHARE of its seconds
# inside a polygon) so a polygon the fixes only graze earns nothing.
LANDMARK_BONUS = 0.25
LANDMARK_QUORUM = MIN_PEAK_MASS_SHARE


@dataclass
class ShortRange:
    """A scored short-clip range selected from a source timeline.

    `rank` is 1-based, 1 = highest opening-weighted score
    (``integrated_score * (1 + opening_mean)``); ``integrated_score`` stays
    the plain window sum, so it need not fall with rank. Publish order
    follows rank: `short_01` is the day's best, `short_02` second, etc.
    """

    start_sec: float
    end_sec: float
    integrated_score: float
    rank: int
    # "sustained" when `_grow_to_moment_length` actually grew this
    # window past its fixed search width (still gaining strength at the
    # window edge), "peak" otherwise (a fixed-strength moment — plateau or
    # isolated peak — that never grew). `None` only when growth was never
    # attempted (empty timeline / no picks).
    moment_kind: "str | None" = None
    # Measured score behind the pick: mean composite of the final clip, mean
    # composite of its first OPENING_SECONDS seconds, and the quality bar both
    # had to beat. None when the range was not produced by `plan_shorts`.
    window_mean: "float | None" = None
    opening_mean: "float | None" = None
    quality_bar: "float | None" = None
    # True when the final clip holds the landmark quorum (it sat inside a measured
    # polygon). Observability only; ranking already used it. Outside the audit hash.
    landmark: bool = False

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def _build_candidate_windows(
    timestamps: np.ndarray,
    composites: np.ndarray,
    window_sec: float,
    stride_sec: float,
    stop_mask: np.ndarray | None = None,
) -> list[tuple[float, float, float]]:
    """Slide a window across the timeline and compute integrated scores.

    Returns a list of (start_sec, end_sec, integrated_score) tuples.

    When ``stop_mask`` is supplied (one bool per timeline second, ``True`` ==
    STOP), any window overlapping a masked second is DROPPED — a structural
    veto so the caller's relaxation ladder can only relax gap/stride, never
    reach parked footage. Default None keeps the historical behavior.
    """
    if len(timestamps) == 0:
        return []

    masked_ts: np.ndarray | None = None
    if stop_mask is not None:
        m = np.asarray(stop_mask, dtype=bool)
        if m.shape[0] == len(timestamps) and m.any():
            masked_ts = timestamps[m]

    total_duration = float(timestamps[-1] - timestamps[0] + 1.0)
    if total_duration < window_sec:
        start = float(timestamps[0])
        end = start + total_duration
        if masked_ts is not None and np.any((masked_ts >= start) & (masked_ts < end)):
            return []
        integrated = float(composites.sum())
        return [(start, end, integrated)]

    windows: list[tuple[float, float, float]] = []
    start_t = float(timestamps[0])
    max_start = float(timestamps[-1]) - window_sec + 1.0
    t = start_t
    while t <= max_start:
        end_t = t + window_sec
        if masked_ts is not None and np.any((masked_ts >= t) & (masked_ts < end_t)):
            t += stride_sec
            continue
        mask = (timestamps >= t) & (timestamps < end_t)
        integrated = float(composites[mask].sum())
        windows.append((t, end_t, integrated))
        t += stride_sec
    return windows


def _greedy_pick_with_gap(
    windows: list[tuple[float, float, float]],
    target_count: int,
    min_gap_sec: float,
    preselected: list[tuple[float, float, float]] | None = None,
) -> list[tuple[float, float, float]]:
    """Pick up to `target_count` windows in descending score order, enforcing
    that consecutive chosen windows are separated by at least `min_gap_sec`
    (measured from previous end to next start). Returns windows in the
    order they were picked (best score first — becomes the rank sequence).

    `preselected` seeds the chosen list with picks from an earlier ladder
    rung: they are immutable (never dropped or reordered) and count toward
    `target_count`; new picks must clear `min_gap_sec` against them too.
    Sorting is fully deterministic — score desc, then start-time asc —
    so the same windows in any input order produce the same picks.
    """
    chosen: list[tuple[float, float, float]] = list(preselected) if preselected else []
    if not windows:
        return chosen

    sorted_by_score = sorted(
        windows,
        key=lambda w: (-w[2], w[0]),  # score desc, earlier start as tiebreak
    )

    for w in sorted_by_score:
        if len(chosen) >= target_count:
            break
        start, end, _score = w
        # Check min gap against all already-chosen windows
        conflict = False
        for c_start, c_end, _ in chosen:
            if end <= c_start - min_gap_sec or start >= c_end + min_gap_sec:
                continue  # far enough away
            conflict = True
            break
        if conflict:
            continue
        chosen.append(w)
    return chosen


def timeline_quality_bar(scores: np.ndarray, stop_mask: np.ndarray | None = None) -> float | None:
    """Mean per-second composite of the eligible (non-stopped) timeline.

    A candidate window clears the quality bar only when its own mean composite is
    strictly above this number. The bar is parameter-free: it is read from the
    day's own footage, never a tuned constant. Returns ``None`` when there is no
    eligible second (empty scores, or every second stopped).
    """
    if scores is None or scores.size == 0:
        return None
    composites = scores[:, COL_COMPOSITE]
    if stop_mask is not None:
        m = np.asarray(stop_mask, dtype=bool)
        if m.shape[0] == len(composites):
            composites = composites[~m]
    if len(composites) == 0:
        return None
    return float(composites.mean())


def _window_mean(timestamps: np.ndarray, composites: np.ndarray, start: float, end: float) -> float:
    """Mean composite of the timeline rows inside ``[start, end)``."""
    in_window = (timestamps >= start) & (timestamps < end)
    n = int(in_window.sum())
    return float(composites[in_window].sum()) / n if n else 0.0


def _opening_mean(
    timestamps: np.ndarray, composites: np.ndarray, start: float, end: float
) -> float:
    """Mean composite of the first ``OPENING_SECONDS`` seconds from ``start``,
    never past the window's ``end`` (a window shorter than the opening is its
    own opening)."""
    return _window_mean(timestamps, composites, start, min(start + OPENING_SECONDS, end))


def _landmark_share(
    timestamps: np.ndarray, landmark_mask: np.ndarray | None, start: float, end: float
) -> float:
    """Share of the timeline rows in ``[start, end)`` that sit inside a landmark."""
    if landmark_mask is None:
        return 0.0
    in_window = (timestamps >= start) & (timestamps < end)
    n = int(in_window.sum())
    return float(landmark_mask[in_window].sum()) / n if n else 0.0


def _ranking_score(window_sum: float, opening_mean: float) -> float:
    """Pick-order score: the window's integrated sum weighted by its opening.

    ``sum * (1 + opening_mean)``: a window that opens strongly outranks one with
    a slightly larger sum and a flat opening. Composites lie in [0, 1], so the
    weight is in [1, 2] and the sum still dominates. Ties fall to the earlier
    start in `_greedy_pick_with_gap`, so the order is deterministic.
    """
    return window_sum * (1.0 + opening_mean)


def plan_shorts(
    scores: np.ndarray,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
    target_duration_sec: float = DEFAULT_TARGET_DURATION_SEC,
    target_count: int = DEFAULT_TARGET_COUNT,
    min_gap_sec: float = DEFAULT_MIN_GAP_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    stop_mask: np.ndarray | None = None,
    quality_bar: float | None = None,
    landmark_mask: np.ndarray | None = None,
) -> list[ShortRange]:
    """Plan 15-55s short-clip ranges from a scored timeline.

    Args:
        scores: numpy array of shape (N, 9) from
            `UnifiedSegmentScorer.merge_file_scores`. Column 0 = timestamp
            in seconds, column 1 = composite score in [0, 1].
        min_duration_sec: minimum short duration (default 15).
        max_duration_sec: maximum short duration (default 55).
        target_duration_sec: exact window width used for candidate scoring
            (default 38 — a comfortable middle of the 15-55 band).
        target_count: MAXIMUM number of shorts returned (default 9). The
            planner never lowers its standards to reach it.
        min_gap_sec: required gap in seconds between consecutive shorts
            (end of prior → start of next). Default 60. A hard constraint:
            it is never relaxed to reach `target_count`.
        stride_sec: slide step for candidate window generation. Default 5.
        stop_mask: optional bool array (one entry per timeline second, aligned
            to ``scores`` rows; ``True`` == STOP). Any candidate window
            overlapping a masked second is structurally dropped at BOTH build
            sites, so the relaxation ladder can never surface parked footage.
            Default None = unchanged behavior. Root-cause fix: a published
            short once shipped extended parked/roadside footage.
        quality_bar: optional explicit bar (mean per-second composite a window
            must STRICTLY exceed). Default None = the mean composite of the
            eligible (non-stopped) timeline in ``scores``. A caller planning
            over a sub-window of a day (the per-LF refill) passes the DAY's bar
            so a sub-window cannot lower it.
        landmark_mask: optional bool array (one entry per timeline second) marking
            seconds inside a measured destination polygon. A window holding at
            least ``LANDMARK_QUORUM`` of its seconds there has its pick-order
            score multiplied by ``1 + LANDMARK_BONUS``. Ranking only: it never
            lets a window past the bar, the opening check, the masks or the gap.
            ``None`` (or a mask not aligned to ``scores``) = no bonus.

    Returns:
        List of `ShortRange` objects sorted by rank (best score first,
        rank 1 = highest opening-weighted score). Only windows whose mean
        composite AND opening mean are strictly above the quality bar are
        eligible, so a
        flat timeline, or a day with nothing above its own mean, returns
        fewer than `target_count` — possibly none. Fewer is a valid plan;
        this never raises and the SHORTS-CONTRACT gate does not read a
        bar- or gap-limited plan as a failure.
    """
    if scores is None or scores.size == 0:
        return []

    timestamps = scores[:, COL_TIMESTAMP]
    composites = scores[:, COL_COMPOSITE]

    if len(timestamps) == 0:
        return []

    # Clamp target to [min, max]
    target = max(min_duration_sec, min(max_duration_sec, target_duration_sec))

    total_duration = float(timestamps[-1] - timestamps[0] + 1.0)
    if total_duration < min_duration_sec:
        logger.debug(
            "Scored timeline too short for a short clip: %.1fs < %.1fs",
            total_duration,
            min_duration_sec,
        )
        return []

    # If the full timeline is shorter than the target, shrink the window
    # to fit while still meeting the minimum duration check above.
    effective_window = min(target, total_duration)

    windows = _build_candidate_windows(
        timestamps,
        composites,
        effective_window,
        stride_sec,
        stop_mask=stop_mask,
    )
    if not windows:
        return []

    # ── Quality bar + hard gap ────────────────────────────────────────────
    # No count-filling: a window qualifies only when its mean per-second
    # composite is strictly above the bar (the eligible timeline's mean), and
    # the gap and stride are never relaxed. ``target_count`` is a maximum.
    bar = quality_bar if quality_bar is not None else timeline_quality_bar(scores, stop_mask)
    if bar is None:
        return []
    lm_mask = None
    if landmark_mask is not None:
        lm = np.asarray(landmark_mask, dtype=bool)
        if lm.shape[0] == len(timestamps) and lm.any():
            lm_mask = lm
    # A window qualifies only when BOTH its whole-window mean and its opening
    # mean are strictly above the bar: a weak opening never ships. Pick order
    # uses the opening-weighted score; the plain sum is restored afterwards.
    qualifying = []
    plain_sum: dict[float, float] = {}
    below_bar = 0
    weak_opening = 0
    for w_start, w_end, w_sum in windows:
        if _window_mean(timestamps, composites, w_start, w_end) <= bar + _BAR_EPSILON:
            below_bar += 1
            continue
        opening = _opening_mean(timestamps, composites, w_start, w_end)
        if opening <= bar + _BAR_EPSILON:
            weak_opening += 1
            continue
        plain_sum[w_start] = w_sum
        rank_score = _ranking_score(w_sum, opening)
        if _landmark_share(timestamps, lm_mask, w_start, w_end) >= LANDMARK_QUORUM:
            rank_score *= 1.0 + LANDMARK_BONUS
        qualifying.append((w_start, w_end, rank_score))
    chosen = _greedy_pick_with_gap(qualifying, target_count, min_gap_sec)
    # What the gap alone cost: qualifying windows a gap of 0 would also have
    # fit (non-overlapping), up to the target. Measured, never used to pick.
    no_gap = _greedy_pick_with_gap(qualifying, target_count, 0.0)
    lost_to_gap = max(0, len(no_gap) - len(chosen))
    logger.info(
        "SHORTS-PLAN: %d of target %d clear the quality bar "
        "(%d candidates below the timeline mean, %d with a weak opening, "
        "%d lost to the %gs gap)",
        len(chosen),
        target_count,
        below_bar,
        weak_opening,
        lost_to_gap,
        min_gap_sec,
    )

    chosen = [(start, end, plain_sum[start]) for start, end, _ in chosen]
    pre_growth_ends = [end for _, end, _ in chosen]
    chosen = _grow_to_moment_length(
        chosen,
        timestamps,
        composites,
        max_duration_sec,
        min_gap_sec,
        stop_mask=stop_mask,
    )
    chosen = _trim_to_moment_length(chosen, timestamps, composites, min_duration_sec, bar)

    for i, (start, end, _score) in enumerate(chosen):
        logger.info(
            "SHORTS-PLAN: short %d %.0f-%.0fs mean composite %.4f (bar %.4f)",
            i + 1,
            start,
            end,
            _window_mean(timestamps, composites, start, end),
            bar,
        )

    # Assign ranks in the order chosen (already sorted by score desc).
    # A window that grew past its pre-growth end was still gaining
    # strength at the fixed search width ("sustained"); one that didn't
    # grow is a fixed-strength moment ("peak").
    return [
        ShortRange(
            start_sec=start,
            end_sec=end,
            integrated_score=score,
            rank=i + 1,
            moment_kind=("sustained" if end > pre_growth_ends[i] else "peak"),
            window_mean=_window_mean(timestamps, composites, start, end),
            opening_mean=_opening_mean(timestamps, composites, start, end),
            quality_bar=bar,
            landmark=_landmark_share(timestamps, lm_mask, start, end) >= LANDMARK_QUORUM,
        )
        for i, (start, end, score) in enumerate(chosen)
    ]


def _grow_to_moment_length(
    chosen: list[tuple[float, float, float]],
    timestamps: np.ndarray,
    composites: np.ndarray,
    max_duration_sec: float,
    min_gap_sec: float,
    stop_mask: np.ndarray | None = None,
) -> list[tuple[float, float, float]]:
    """Grow each picked window's end to match its moment's extent (clip length follows the moment, target duration is
    a fallback only).

    Every candidate window is built at one fixed search width (`target_duration_sec`
    clamped to bounds); a moment that is still gaining
    strength right past that width should keep going. Extends one second
    at a time for as long as the next second's composite score is
    STRICTLY greater than the window's own running mean -- a flat or
    declining moment (the common case: plateaus, isolated peaks) never
    grows, so this changes nothing for a fixed-strength moment and only
    lengthens a genuinely still-rising one. Growth stops at
    `max_duration_sec`, the next chosen window's start minus
    `min_gap_sec` (never invades another short's gap), a stop-masked
    second, or the timeline end.
    """
    if not chosen or len(timestamps) == 0:
        return chosen

    mask = None
    if stop_mask is not None:
        m = np.asarray(stop_mask, dtype=bool)
        if m.shape[0] == len(timestamps):
            mask = m

    timeline_end = float(timestamps[-1]) + 1.0
    order = sorted(range(len(chosen)), key=lambda i: chosen[i][0])
    grown = list(chosen)
    for pos, idx in enumerate(order):
        start, end, score = grown[idx]
        ceiling = min(start + max_duration_sec, timeline_end)
        if pos + 1 < len(order):
            next_start = grown[order[pos + 1]][0]
            ceiling = min(ceiling, next_start - min_gap_sec)
        window_mask = (timestamps >= start) & (timestamps < end)
        if not window_mask.any():
            continue
        running_sum = float(composites[window_mask].sum())
        running_n = int(window_mask.sum())
        new_end = end
        t = end
        while t < ceiling:
            if mask is not None:
                sec_idx = int(np.searchsorted(timestamps, t))
                if sec_idx < len(mask) and mask[sec_idx]:
                    break
            sec_mask = (timestamps >= t) & (timestamps < t + 1.0)
            if not sec_mask.any():
                break
            sec_score = float(composites[sec_mask].mean())
            running_mean = running_sum / running_n
            if sec_score <= running_mean:
                break
            running_sum += sec_score
            running_n += sec_mask.sum()
            new_end = t + 1.0
            t += 1.0
        grown[idx] = (start, new_end, score)
    return grown


def _trim_to_moment_length(
    chosen: list[tuple[float, float, float]],
    timestamps: np.ndarray,
    composites: np.ndarray,
    min_duration_sec: float,
    bar: float,
) -> list[tuple[float, float, float]]:
    """Pull each pick's end back to the last second whose composite is above
    the bar, never shorter than ``min_duration_sec``.

    Together with `_grow_to_moment_length` this lets a clip end anywhere in
    ``[min, max]`` following its moment. Trimming only moves an end earlier, so
    it can never reach a second the pick did not already cover (and so never a
    stop/county/GPS-gap masked second, which candidate windows exclude).
    """
    trimmed = []
    for start, end, score in chosen:
        floor = start + min_duration_sec
        in_clip = (timestamps >= start) & (timestamps < end)
        above = in_clip & (composites > bar + _BAR_EPSILON)
        if above.any():
            last = float(timestamps[above].max()) + 1.0
            end = min(end, max(last, floor))
        else:
            end = min(end, floor)
        trimmed.append((start, end, score))
    return trimmed
