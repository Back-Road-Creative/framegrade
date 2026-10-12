"""Regular LF segment planner (day-first architecture).

Produces contiguous range selections for regular long-form videos from a
day's per-second scored timeline. Regular LFs under the new architecture
are the best CONTINUOUS ranges of the day's super LF, in TEMPORAL ORDER
— not highlight compilations with cuts.

Algorithm:
  1. Extract (timestamp, composite_score) rows from the score array.
  2. Build candidate windows via sliding-window scan at `stride_sec`
     intervals. Each candidate is the target duration wide and its
     integrated score is the sum of composite scores inside.
  3. Sort candidates by integrated score (desc).
  4. Greedy-pick non-overlapping candidates from the sorted list until
     `max_count` are chosen.
  5. Return the chosen ranges sorted by `start_sec` (chronological).

Locked behavior:
  * Each returned range is contiguous (no cuts).
  * Ranges never overlap.
  * Ranges come back in chronological order regardless of which had the
    highest integrated score.
  * At most `max_count` ranges returned; fewer if the day is too short
    or the scored timeline doesn't contain enough distinct zones.
  * Individual range duration is bounded by
    `[min_duration_sec, max_duration_sec]`.

The algorithm is deliberately simple: fixed `target_window_sec` width
for all candidates. This guarantees predictable output and is easy to
reason about. Variable-width optimization can come later if needed.
"""

import logging
import math
import sys
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Score-array column indices (mirror unified_scorer)
COL_TIMESTAMP = 0
COL_COMPOSITE = 1

# Canonical LF duration bounds from gomoveshift.video_types
# (kept here as module defaults to avoid a circular import)
DEFAULT_MIN_DURATION_SEC = 480.0  # 8 min
DEFAULT_MAX_DURATION_SEC = 1800.0  # 30 min
DEFAULT_TARGET_WINDOW_SEC = 1200.0  # 20 min — middle of the range
DEFAULT_STRIDE_SEC = 30.0
DEFAULT_MAX_COUNT = 4

# Default ceiling for ``full_drive`` mode (Insta360 / full-drive LF).
# 2 hours is a sane upper bound for a single LF — protects against runaway
# encodes when the operator forgets to trim the source. Mirrored in
# ``config.DriveConfig.full_drive_max_duration_sec``.
DEFAULT_FULL_DRIVE_MAX_DURATION_SEC = 7200

LFSelectionMode = Literal["best_moments", "full_drive"]
_VALID_MODES = ("best_moments", "full_drive")


@dataclass
class RegularLFRange:
    """A contiguous range of the day master's timeline chosen as a regular LF."""

    start_sec: float
    end_sec: float
    integrated_score: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def duration_min(self) -> float:
        return self.duration_sec / 60.0


def _score_at_each_second(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps, composite_scores) as two 1-D arrays."""
    if scores is None or scores.size == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    return scores[:, COL_TIMESTAMP], scores[:, COL_COMPOSITE]


def _build_candidate_windows(
    timestamps: np.ndarray,
    composites: np.ndarray,
    window_sec: float,
    stride_sec: float,
) -> list[RegularLFRange]:
    """Build all sliding windows of the given width.

    Each window is a RegularLFRange with its integrated score computed
    from the composite array. Windows that run off the end of the
    timeline are truncated — they still count but their score reflects
    only the in-bounds portion.
    """
    if len(timestamps) == 0:
        return []

    total_duration = float(timestamps[-1] - timestamps[0] + 1.0)  # 1s sample resolution
    if total_duration < window_sec:
        # Single window covering everything
        integrated = float(composites.sum())
        return [
            RegularLFRange(
                start_sec=float(timestamps[0]),
                end_sec=float(timestamps[0] + total_duration),
                integrated_score=integrated,
            )
        ]

    windows: list[RegularLFRange] = []
    start_t = float(timestamps[0])
    # Stride through start positions
    max_start = float(timestamps[-1]) - window_sec + 1.0
    t = start_t
    while t <= max_start:
        end_t = t + window_sec
        # Slice composite scores for [t, end_t)
        mask = (timestamps >= t) & (timestamps < end_t)
        integrated = float(composites[mask].sum())
        windows.append(
            RegularLFRange(
                start_sec=t,
                end_sec=end_t,
                integrated_score=integrated,
            )
        )
        t += stride_sec
    return windows


def _greedy_pick_nonoverlapping(
    windows: list[RegularLFRange],
    max_count: int,
) -> list[RegularLFRange]:
    """Pick up to `max_count` non-overlapping windows in descending integrated
    score order. Returns them in chronological order (sorted by start_sec).
    """
    if not windows:
        return []
    # Sort by score desc; tiebreak on earlier start for determinism
    sorted_by_score = sorted(
        windows,
        key=lambda w: (-w.integrated_score, w.start_sec),
    )
    chosen: list[RegularLFRange] = []
    for w in sorted_by_score:
        if len(chosen) >= max_count:
            break
        # Does this window overlap any already-chosen window?
        overlaps = any(not (w.end_sec <= c.start_sec or w.start_sec >= c.end_sec) for c in chosen)
        if overlaps:
            continue
        chosen.append(w)
    return sorted(chosen, key=lambda w: w.start_sec)


def _full_drive_unmasked_ranges(
    timestamps: np.ndarray,
    composites: np.ndarray,
    stop_mask: Optional[np.ndarray],
) -> list[RegularLFRange]:
    """Maximal contiguous UNMASKED spans of the timeline as full-drive LF ranges.

    Scores are 1 Hz — row ``i`` is timestamp ``timestamps[i]``. A contiguous run
    of unmasked rows ``[i, j)`` becomes one range ``[timestamps[i],
    timestamps[j-1] + 1.0)`` (the ``+1`` mirrors the 1s sample resolution used
    for ``total_duration``). ``stop_mask`` marks STOP seconds (``True``) to cut
    out; when it is ``None`` (or shape-mismatched) every second is kept, giving
    the single whole-timeline span the legacy full_drive path produced.
    """
    n = len(timestamps)
    if n == 0:
        return []
    if stop_mask is None:
        keep = np.ones(n, dtype=bool)
    else:
        m = np.asarray(stop_mask, dtype=bool)
        keep = ~m if m.shape[0] == n else np.ones(n, dtype=bool)

    ranges: list[RegularLFRange] = []
    i = 0
    while i < n:
        if keep[i]:
            j = i
            while j < n and keep[j]:
                j += 1
            ranges.append(
                RegularLFRange(
                    start_sec=float(timestamps[i]),
                    end_sec=float(timestamps[j - 1]) + 1.0,
                    integrated_score=float(composites[i:j].sum()),
                )
            )
            i = j
        else:
            i += 1
    return ranges


def split_ranges_balanced(
    ranges: list[RegularLFRange],
    *,
    min_sec: int,
    max_sec: int,
) -> list[list[RegularLFRange]]:
    """Split a day's kept ranges into balanced long-form parts.

    ``kept`` is the total kept (unmasked) seconds. ``K = ceil(kept/max_sec)``;
    when ``K > 1`` and ``kept/K < min_sec`` the count drops by one, so a part
    may run over ``max_sec`` but never under ``min_sec`` (a day just over the
    max ships as one long part, not as runts). ``K == 1`` returns the ranges
    unchanged. Each cut lands on the stop-mask boundary (a range edge) nearest
    the even-split target if one lies in the feasible window, else mid-range at
    the target clamped into the window. Real timestamps are kept; a bisected
    range splits ``integrated_score`` (a SUM of per-second composites)
    proportionally to duration. Parts tile the input with no gap or overlap.
    """
    if min_sec <= 0 or max_sec < min_sec:
        raise ValueError(
            f"split_ranges_balanced: need 0 < min_sec <= max_sec (got {min_sec}, {max_sec})"
        )
    kept = sum(r.duration_sec for r in ranges)
    if kept <= 0:
        return [list(ranges)]
    k_parts = math.ceil(kept / max_sec)
    if k_parts > 1 and kept / k_parts < min_sec:
        k_parts -= 1
    if k_parts <= 1:
        return [list(ranges)]

    eff_max = max(float(max_sec), math.ceil(kept / k_parts))
    # Cumulative kept-seconds at each interior range edge.
    edges: list[float] = []
    acc = 0.0
    for r in ranges[:-1]:
        acc += r.duration_sec
        edges.append(acc)

    cuts: list[float] = []
    prev = 0.0
    for k in range(1, k_parts):
        target = k * kept / k_parts
        lo = max(prev + min_sec, kept - (k_parts - k) * eff_max)
        hi = min(prev + eff_max, kept - (k_parts - k) * min_sec)
        if lo > hi + 1e-9:
            raise ValueError(
                f"split_ranges_balanced: no feasible cut for part {k}/{k_parts} "
                f"(window {lo:.1f}-{hi:.1f}s)"
            )
        inside = [e for e in edges if lo - 1e-9 <= e <= hi + 1e-9]
        cut = (
            min(inside, key=lambda e: (abs(e - target), e)) if inside else min(max(target, lo), hi)
        )
        cuts.append(cut)
        prev = cut

    parts: list[list[RegularLFRange]] = [[] for _ in range(k_parts)]
    pos = 0.0
    part_idx = 0
    for r in ranges:
        start, dur = r.start_sec, r.duration_sec
        consumed = 0.0
        while True:
            end_cum = pos + dur - consumed
            if part_idx < len(cuts) and pos + 1e-9 < cuts[part_idx] < end_cum - 1e-9:
                take = cuts[part_idx] - pos
                cut_t = start + consumed + take
                frac = take / dur
                parts[part_idx].append(
                    RegularLFRange(start + consumed, cut_t, r.integrated_score * frac)
                )
                consumed += take
                pos += take
                part_idx += 1
                continue
            piece_dur = dur - consumed
            if consumed == 0.0:
                parts[part_idx].append(r)
            else:
                parts[part_idx].append(
                    RegularLFRange(
                        start + consumed,
                        r.end_sec,
                        r.integrated_score * (piece_dur / dur),
                    )
                )
            pos += piece_dur
            if part_idx < len(cuts) and abs(pos - cuts[part_idx]) <= 1e-9:
                part_idx += 1
            break

    for i, part in enumerate(parts):
        d = sum(x.duration_sec for x in part)
        if d < min_sec - 1e-6 or d > eff_max + 1e-6:
            raise ValueError(
                f"split_ranges_balanced: part {i + 1}/{k_parts} is {d:.1f}s, outside "
                f"[{min_sec}, {eff_max:.0f}]s"
            )
    return parts


def plan_full_drive_parts(
    scores: np.ndarray,
    *,
    min_duration_sec: float,
    full_drive_max_duration_sec: int,
    stop_mask: Optional[np.ndarray],
    part_min_sec: Optional[int] = None,
    part_max_sec: Optional[int] = None,
) -> list[list[RegularLFRange]]:
    """``plan_regular_lfs(mode="full_drive")`` grouped into long-form parts.

    With either part bound ``None`` this is exactly today's behaviour: one
    part holding the planner's ranges (total checked against the ceiling).
    With both set the day is split by :func:`split_ranges_balanced` and the
    480 s minimum and ``full_drive_max_duration_sec`` ceiling apply PER PART
    (a part over the ceiling raises; nothing is truncated).
    """
    if part_min_sec is None or part_max_sec is None:
        return [
            plan_regular_lfs(
                scores,
                mode="full_drive",
                min_duration_sec=min_duration_sec,
                full_drive_max_duration_sec=full_drive_max_duration_sec,
                stop_mask=stop_mask,
            )
        ]
    ranges = plan_regular_lfs(
        scores,
        mode="full_drive",
        min_duration_sec=min_duration_sec,
        full_drive_max_duration_sec=sys.maxsize,  # ceiling is enforced per part below
        stop_mask=stop_mask,
    )
    parts = split_ranges_balanced(ranges, min_sec=part_min_sec, max_sec=part_max_sec)
    for i, part in enumerate(parts, 1):
        d = sum(r.duration_sec for r in part)
        if d < min_duration_sec:
            raise ValueError(
                f"plan_full_drive_parts: part {i}/{len(parts)} {d:.1f}s is shorter than "
                f"the {min_duration_sec:.0f}s minimum LF duration."
            )
        if d > float(full_drive_max_duration_sec):
            raise ValueError(
                f"plan_full_drive_parts: part {i}/{len(parts)} {d:.1f}s exceeds the "
                f"{full_drive_max_duration_sec}s ceiling. Raise "
                "config.drive.full_drive_max_duration_sec or lower "
                "full_drive_part_max_sec — no silent truncation."
            )
    return parts


def plan_regular_lfs(
    scores: np.ndarray,
    *,
    mode: LFSelectionMode,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
    target_window_sec: Optional[float] = None,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    max_count: int = DEFAULT_MAX_COUNT,
    full_drive_max_duration_sec: int = DEFAULT_FULL_DRIVE_MAX_DURATION_SEC,
    stop_mask: Optional[np.ndarray] = None,
) -> list[RegularLFRange]:
    """Plan contiguous regular LF ranges from a day's scored timeline.

    ``mode`` is a REQUIRED keyword-only argument (no default). Callers
    that forget it crash at the function boundary with ``TypeError`` —
    robust by construction. Passing the wrong
    string raises ``ValueError`` with the valid set.

    Args:
        scores: a numpy array of shape (N, 9) from
            `UnifiedSegmentScorer.merge_file_scores`. Column 0 is the
            timestamp in seconds (already offset to be continuous across
            episodes). Column 1 is the composite score in [0, 1].
        mode: ``"best_moments"`` runs the sliding-window scan + greedy
            non-overlapping pick (legacy behavior). ``"full_drive"``
            spans the whole day MINUS masked stop spans — the Insta360
            use case where the operator wants the entire drive, not
            highlights. Requires ``stop_mask``; it emits the maximal
            contiguous UNMASKED ranges (parked spans physically cut out).
            Nothing upstream strips stops — the mask is the only thing
            that does.
        min_duration_sec: minimum duration for each LF (default 480 = 8 min).
            Applies in both modes — full_drive sources shorter than this
            are misconfigured and raise ``ValueError``.
        max_duration_sec: maximum duration for each LF (default 1800 = 30 min).
            ``best_moments`` only — irrelevant for ``full_drive`` (the
            full-drive ceiling is ``full_drive_max_duration_sec``).
        target_window_sec: ``best_moments`` only. Exact window width for
            candidate windows. When None, defaults to the midpoint of
            [min, max]. Clamped to [min_duration_sec, max_duration_sec].
        stride_sec: ``best_moments`` only. Slide step between candidate
            windows (default 30s).
        max_count: ``best_moments`` only. Maximum number of regular LFs
            returned (default 4). ``full_drive`` always returns one.
        full_drive_max_duration_sec: ``full_drive`` only. Hard ceiling
            for the LF's KEPT (unmasked) duration. When it exceeds this,
            ``ValueError`` is raised — no silent truncation.
            Default ``7200`` (2h); operator raises explicitly via the
            CLI / config for longer single sources.
        stop_mask: REQUIRED for ``full_drive`` — per-second bool array
            aligned to ``scores`` rows (``True`` == a STOP second to cut
            out). Splits the full-drive LF at masked spans. ``None`` raises
            ``ValueError``; a mask that keeps every second (all-``False``)
            is the explicit way to get the old whole-timeline single span.
            Ignored by ``best_moments``.

    Returns:
        A list of `RegularLFRange` objects sorted chronologically by
        `start_sec`. In ``full_drive`` mode the list is one element per
        contiguous unmasked span (a single whole-timeline element when no
        mask is supplied). Empty if the timeline is shorter than
        `min_duration_sec` (best_moments) or raises (full_drive).
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"plan_regular_lfs: invalid mode {mode!r} — must be one of {_VALID_MODES}")

    if scores is None or scores.size == 0:
        return []

    timestamps, composites = _score_at_each_second(scores)
    if len(timestamps) == 0:
        return []

    total_duration = float(timestamps[-1] - timestamps[0] + 1.0)

    if mode == "full_drive":
        # A full-drive LF spans the day MINUS masked (parked/stopped)
        # spans. A stop_mask is REQUIRED — without one this would degrade
        # to a single whole-timeline span (parked time included) and the
        # duration/ceiling checks below would measure wall-time instead of
        # moving-time, silently shipping a stops-included "full drive"
        # video. Robust-by-construction: refuse rather than guess.
        if stop_mask is None:
            raise ValueError(
                "plan_regular_lfs(mode='full_drive') requires a stop_mask — "
                "maskless full_drive planning would silently include parked "
                "stops in the shipped LF and measure the duration ceiling "
                "against wall-time instead of moving-time. Build the "
                "per-day stop mask and pass it as stop_mask=...; use "
                "mode='best_moments' if no mask applies."
            )
        # With a per-second stop mask we emit the maximal contiguous
        # UNMASKED ranges so parked spans are physically cut out. Nothing
        # upstream strips stops for us — the mask is the only thing that
        # does (the old "the scorer already stripped stops" claim was
        # false: the scored timeline is the raw drive, parked seconds
        # included).
        ranges = _full_drive_unmasked_ranges(timestamps, composites, stop_mask)
        kept_duration = sum(r.duration_sec for r in ranges)
        if kept_duration < min_duration_sec:
            raise ValueError(
                f"plan_regular_lfs(mode='full_drive'): kept (unmasked) timeline "
                f"{kept_duration:.1f}s is shorter than the {min_duration_sec:.0f}s "
                "minimum LF duration — source is misconfigured (or the whole day is "
                "parked). Either trim/extend the footage; a sub-8-minute 'full "
                "drive' is not a long-form video."
            )
        if kept_duration > float(full_drive_max_duration_sec):
            raise ValueError(
                f"plan_regular_lfs(mode='full_drive'): kept (unmasked) timeline "
                f"{kept_duration:.1f}s exceeds the {full_drive_max_duration_sec}s "
                "ceiling. Either raise config.drive.full_drive_max_duration_sec "
                "explicitly or split the source — no silent truncation."
            )
        return ranges

    # Clamp the target to the valid duration band
    if target_window_sec is None:
        target_window_sec = (min_duration_sec + max_duration_sec) / 2.0
    target_window_sec = max(min_duration_sec, min(max_duration_sec, target_window_sec))

    if total_duration < min_duration_sec:
        logger.debug(
            "Scored timeline too short for a regular LF: %.1fs < %.1fs min",
            total_duration,
            min_duration_sec,
        )
        return []

    # If the full timeline is shorter than the target, shrink the target
    # to span what's available (still >= min_duration_sec per the check above).
    effective_window = min(target_window_sec, total_duration)

    windows = _build_candidate_windows(
        timestamps,
        composites,
        effective_window,
        stride_sec,
    )
    if not windows:
        return []

    return _greedy_pick_nonoverlapping(windows, max_count)
