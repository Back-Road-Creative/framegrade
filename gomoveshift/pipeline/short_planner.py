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


@dataclass
class ShortRange:
    """A scored short-clip range selected from a source timeline.

    `rank` is 1-based, 1 = highest integrated score. Publish order
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


def plan_shorts(
    scores: np.ndarray,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
    target_duration_sec: float = DEFAULT_TARGET_DURATION_SEC,
    target_count: int = DEFAULT_TARGET_COUNT,
    min_gap_sec: float = DEFAULT_MIN_GAP_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    stop_mask: np.ndarray | None = None,
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
        target_count: maximum number of shorts returned (default 9).
        min_gap_sec: preferred gap in seconds between consecutive shorts
            (end of prior → start of next). Default 60. This is a rung-1
            preference, not a hard constraint: when it would leave the
            plan short of `target_count`, the relaxation ladder fills the
            remainder at half gap, then gap 0, then half stride + gap 0.
        stride_sec: slide step for candidate window generation. Default 5.
        stop_mask: optional bool array (one entry per timeline second, aligned
            to ``scores`` rows; ``True`` == STOP). Any candidate window
            overlapping a masked second is structurally dropped at BOTH build
            sites, so the relaxation ladder can never surface parked footage.
            Default None = unchanged behavior. Root-cause fix: a published
            short once shipped extended parked/roadside footage.

    Returns:
        List of `ShortRange` objects sorted by rank (best score first,
        rank 1 = highest integrated score). Empty if the timeline is
        shorter than `min_duration_sec`. Returns fewer than
        `target_count` ONLY when the timeline physically can't hold that
        many non-overlapping windows (ladder exhausted) — never raises;
        the day-pipeline SHORTS-CONTRACT gate owns that failure.
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

    # ── Deterministic relaxation ladder ───────────────────────────────────
    # Rung 1 runs the historical pick. When it falls short of target,
    # later rungs KEEP the picks already chosen (immutable across rungs)
    # and only ADD fills under progressively relaxed constraints, so the
    # planner under-delivers ONLY when the timeline physically can't hold
    # `target_count` non-overlapping windows — never because of tunable
    # gap spacing. Best-score-first order is preserved within each rung.
    chosen = _greedy_pick_with_gap(windows, target_count, min_gap_sec)
    rung_counts = [len(chosen)]
    if len(chosen) < target_count:
        # rung 2: half gap; rung 3: gap 0 (adjacent, non-overlapping);
        # rung 4: halve the candidate stride (denser windows), gap 0.
        rungs: list[tuple[float, float]] = [
            (min_gap_sec * 0.5, stride_sec),
            (0.0, stride_sec),
            (0.0, stride_sec / 2.0),
        ]
        for gap, rung_stride in rungs:
            cand = windows
            if rung_stride != stride_sec:
                cand = _build_candidate_windows(
                    timestamps, composites, effective_window, rung_stride, stop_mask=stop_mask
                )
            before = len(chosen)
            chosen = _greedy_pick_with_gap(cand, target_count, gap, preselected=chosen)
            rung_counts.append(len(chosen) - before)
            if len(chosen) >= target_count:
                break
        breakdown = " ".join(f"rung{i + 1}={n}" for i, n in enumerate(rung_counts))
        if len(chosen) >= target_count:
            logger.info(
                "SHORTS-PLAN: target %d reached at rung %d (%s)",
                target_count,
                len(rung_counts),
                breakdown,
            )
        else:
            logger.info(
                "SHORTS-PLAN: target %d unreachable — ladder exhausted at rung %d (%s); "
                "timeline physically holds only %d windows",
                target_count,
                len(rung_counts),
                breakdown,
                len(chosen),
            )

    pre_growth_ends = [end for _, end, _ in chosen]
    chosen = _grow_to_moment_length(
        chosen,
        timestamps,
        composites,
        max_duration_sec,
        min_gap_sec,
        stop_mask=stop_mask,
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
