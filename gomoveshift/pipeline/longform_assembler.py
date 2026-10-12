"""Long-Form Assembler — segment selection and grouping for drive pipeline.

Takes a scored time-series and produces assembly plans for regular long-form
videos. The core logic of the drive pipeline's content selection.

Algorithm:
  1. Score threshold = percentile of composite scores (e.g., top 40%)
  2. Find contiguous regions above threshold, minimum 90 seconds
  3. Group segments into regular LFs (8-30 min each)
  4. Ensure each LF yields enough shorts (>=7 score peaks)
  5. Select montage peaks for each LF's 30-second highlight intro
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import DriveConfig
from ..models import DrivePlan, LFSegment, RegularLFPlan

logger = logging.getLogger(__name__)

# Column indices matching unified_scorer.py
COL_TIMESTAMP = 0
COL_COMPOSITE = 1
COL_MOTION_ENERGY = 2  # Matches unified_scorer.py column order


def _has_gps(gps_per_episode: Optional[list]) -> bool:
    """True when any episode carries parsed GPS data ((gps_data, offset) tuples)."""
    return bool(gps_per_episode) and any(
        entry and entry[0] is not None for entry in gps_per_episode
    )


def _absolute_floor_filter(segments: list[LFSegment], floor: float) -> list[LFSegment]:
    """Keep every segment whose composite_score clears a FIXED absolute floor.

    Unlike the percentile path, coverage scales with footage quality: a scenic
    drive keeps many segments (→ many LFs), a sparse one keeps few (→ one LF).
    Nothing is padded to a duration target. When no segment clears the floor the
    single best segment is kept so a day never collapses to zero LFs.
    """
    kept = [s for s in segments if s.composite_score >= floor]
    if not kept:
        kept = [max(segments, key=lambda s: s.composite_score)]
    kept_sorted = sorted(kept, key=lambda s: s.original_start_time)
    logger.info(
        "best_moments_mode (absolute floor=%.4f): kept %d/%d segments (%.0fs)",
        floor,
        len(kept_sorted),
        len(segments),
        sum(s.duration for s in kept_sorted),
    )
    return kept_sorted


def _dynamic_best_moments_filter(
    segments: list[LFSegment], config: "DriveConfig", has_gps: bool = False
) -> list[LFSegment]:
    """Filter segments using percentile-based quality scaling.

    Sort segments by composite_score descending. Accumulate duration from top.
    Stop when:
      duration >= effective_min AND (
        next segment score < quality_floor
        OR duration >= target_max
      )

    quality_floor is derived from best_moments_percentile: e.g. percentile=75
    means quality_floor = 75th percentile score, keeping the top 25%.
    percentile=50 keeps the top 50% — useful for footage with uniformly good
    quality where statistical thresholds are too aggressive.

    effective_min = max(target_min, min_keep_ratio * total_qualifying_duration).
    This ensures we never discard more than min_keep_ratio of qualifying footage.
    """
    if not segments:
        return segments

    # Opt-in absolute floor (visual-only). GPS-present drives have a higher,
    # shifted composite distribution (GPS mass is live, not redistributed), so
    # the GPS-less floor would be miscalibrated — fall back to the percentile
    # bar until a GPS floor is calibrated.
    if config.best_moments_absolute_floor_enabled:
        if has_gps:
            logger.info(
                "best_moments_mode: absolute floor requested but GPS present — "
                "using percentile bar (GPS floor not yet calibrated)"
            )
        else:
            return _absolute_floor_filter(segments, config.best_moments_absolute_floor)

    scores = sorted(s.composite_score for s in segments)
    # Percentile-based quality floor from config.best_moments_percentile.
    # percentile=75 → floor at 75th percentile → keep top 25% by score.
    pct = config.best_moments_percentile
    idx = int(len(scores) * pct / 100.0)
    idx = min(idx, len(scores) - 1)
    quality_floor = scores[idx]

    # Compute effective minimum: never cut more than min_keep_ratio of qualifying footage
    total_qualifying_duration = sum(s.duration for s in segments)
    effective_min = max(
        config.best_moments_target_min,
        config.best_moments_min_keep_ratio * total_qualifying_duration,
    )
    if effective_min > config.best_moments_target_min:
        logger.info(
            "best_moments_mode: effective_min raised to %.0fs "
            "(%.0f%% of %.0fs qualifying footage, target_min was %.0fs)",
            effective_min,
            config.best_moments_min_keep_ratio * 100,
            total_qualifying_duration,
            config.best_moments_target_min,
        )

    # Sort by score descending, accumulate
    ranked = sorted(segments, key=lambda s: s.composite_score, reverse=True)
    included: list[LFSegment] = []
    included_set: set[int] = set()  # track by id for O(1) lookup
    total_duration = 0.0

    for idx, seg in enumerate(ranked):
        included.append(seg)
        included_set.add(id(seg))
        total_duration += seg.duration

        if total_duration >= effective_min:
            # Check if we should stop
            remaining = [s for s in ranked[idx + 1 :] if id(s) not in included_set]
            if not remaining:
                break
            next_score = remaining[0].composite_score
            if next_score < quality_floor:
                break  # target_max is soft; only quality floor stops

    last_score = included[-1].composite_score if included else 0.0
    original_count = len(segments)
    logger.info(
        "best_moments_mode: kept %d/%d segments (%.0fs), quality_floor=%.4f, stopped at score=%.4f",
        len(included),
        original_count,
        total_duration,
        quality_floor,
        last_score,
    )

    # Re-sort by original timeline order
    included.sort(key=lambda s: s.original_start_time)
    return included


class LongFormAssembler:
    """Selects and groups highlight segments into regular long-form videos."""

    def __init__(self, config: DriveConfig):
        self.config = config

    def find_segments_above_threshold(
        self,
        scores: np.ndarray,
        threshold: float,
        file_boundaries: list[tuple[float, float]],
        gps_per_episode: Optional[list] = None,
        stop_mask: Optional[np.ndarray] = None,
    ) -> list[LFSegment]:
        """Find contiguous regions where composite score exceeds threshold.

        Regions shorter than min_segment_duration are dropped.

        Args:
            scores: Score array (N, 9) from UnifiedSegmentScorer.
            threshold: Minimum composite score for inclusion.
            file_boundaries: List of (start_time, end_time) per file in the
                drive-wide timeline.
            stop_mask: Optional per-second bool array aligned to ``scores``
                (``True`` == STOP). When supplied, a masked second can NEVER be
                inside an above-threshold segment — a structural veto so parked
                footage cannot land in an LF. Default None = unchanged.

        Returns:
            List of LFSegment objects for above-threshold regions.
        """
        composites = scores[:, COL_COMPOSITE]

        above = composites >= threshold
        # Also gate on per-sample motion energy when min_motion_energy is configured
        if self.config.min_motion_energy > 0:
            motion = scores[:, COL_MOTION_ENERGY]
            above = above & (motion >= self.config.min_motion_energy)
        # Structural stop veto: exclude masked (parked/roadside) seconds.
        if stop_mask is not None:
            m = np.asarray(stop_mask, dtype=bool)
            if m.shape[0] == above.shape[0]:
                above = above & ~m
        segments = []

        # Find contiguous runs of True
        in_segment = False
        seg_start_idx = 0

        for i in range(len(above)):
            if above[i] and not in_segment:
                seg_start_idx = i
                in_segment = True
            elif not above[i] and in_segment:
                self._maybe_add_segment(
                    segments,
                    scores,
                    seg_start_idx,
                    i,
                    file_boundaries,
                    gps_per_episode=gps_per_episode,
                )
                in_segment = False

        # Handle final segment
        if in_segment:
            self._maybe_add_segment(
                segments,
                scores,
                seg_start_idx,
                len(above),
                file_boundaries,
                gps_per_episode=gps_per_episode,
            )

        return segments

    def _maybe_add_segment(
        self,
        segments: list[LFSegment],
        scores: np.ndarray,
        start_idx: int,
        end_idx: int,
        file_boundaries: list[tuple[float, float]],
        gps_per_episode: Optional[list] = None,
    ) -> None:
        """Add a segment if it meets minimum duration."""
        timestamps = scores[:, COL_TIMESTAMP]
        composites = scores[:, COL_COMPOSITE]

        start_time = float(timestamps[start_idx])
        # end_idx is exclusive; the segment covers up to the next sample boundary
        last_idx = min(end_idx - 1, len(timestamps) - 1)
        # Add 1.0 to account for the duration of the last sample interval
        end_time = float(timestamps[last_idx]) + 1.0
        duration = end_time - start_time

        if duration < self.config.min_segment_duration:
            return

        # Cap long segments at max_segment_duration (0 disables cap)
        if self.config.max_segment_duration > 0 and duration > self.config.max_segment_duration:
            end_time = start_time + self.config.max_segment_duration

        avg_score = float(np.mean(composites[start_idx:end_idx]))
        avg_motion = float(np.mean(scores[start_idx:end_idx, COL_MOTION_ENERGY]))

        # Gate on minimum motion energy if configured
        if self.config.min_motion_energy > 0 and avg_motion < self.config.min_motion_energy:
            logger.debug(
                "Skipping segment %.0f-%.0fs: avg_motion=%.3f < threshold=%.3f",
                start_time,
                end_time,
                avg_motion,
                self.config.min_motion_energy,
            )
            return

        # Determine which source file this segment belongs to
        file_idx = 0
        for fi, (fb_start, fb_end) in enumerate(file_boundaries):
            if fb_start <= start_time < fb_end:
                file_idx = fi
                break

        # Calculate time within the source file
        fb_start = file_boundaries[file_idx][0]
        fb_end = file_boundaries[file_idx][1]
        local_start = start_time - fb_start
        local_end = end_time - fb_start

        # Clamp end_time to file boundary to prevent extraction past EOF
        # (stops removed before segment start shift offsets; guard prevents overrun)
        file_duration = fb_end - fb_start
        if local_end > file_duration - 0.5:
            if local_end > file_duration:
                logger.warning(
                    "Segment %.0f-%.0fs: end_time %.1fs exceeds file duration %.1fs — clamping",
                    start_time,
                    end_time,
                    local_end,
                    file_duration,
                )
            else:
                logger.debug(
                    "Segment %.0f-%.0fs: end_time %.1fs clamping into the 0.5s EOF guard band "
                    "(file duration %.1fs)",
                    start_time,
                    end_time,
                    local_end,
                    file_duration,
                )
            local_end = file_duration - 0.5

        # Populate GPS metadata when per-episode GPS is available
        center_lat = 0.0
        center_lon = 0.0
        terrain = "unknown"
        if gps_per_episode and file_idx < len(gps_per_episode):
            gps_data, gps_offset = gps_per_episode[file_idx]
            # Reconstitute GPSData from dict after JSON checkpoint round-trip
            if isinstance(gps_data, dict) and "trackpoints" in gps_data:
                from ..analysis.shorts_models import GPSData, GPSTrackpoint

                gps_data = GPSData(
                    trackpoints=[GPSTrackpoint(**tp) for tp in gps_data["trackpoints"]]
                )
            if gps_data is not None:
                from ..analysis.shorts_models import effective_offset

                # Idempotent: a stamped track pins footage second 0 whatever
                # offset was stored (legacy checkpoints hold the raw first fix).
                gps_offset = effective_offset(gps_data, gps_offset)
                gps_start = local_start + gps_offset
                gps_end = local_end + gps_offset
                points = gps_data.get_points_in_range(gps_start, gps_end)
                if points:
                    center_lat = round(sum(p.lat for p in points) / len(points), 6)
                    center_lon = round(sum(p.lon for p in points) / len(points), 6)
                    elevs = [p.elevation for p in points if p.elevation is not None]
                    avg_elev = sum(elevs) / len(elevs) if elevs else None
                    if avg_elev is not None:
                        from ..analysis.shorts_extractor import ScenicScorer
                        from .segment_context import classify_terrain

                        scorer = ScenicScorer(gps_data)
                        gain_rate = scorer.get_elevation_gain_rate(gps_start, gps_end)
                        turn_int = scorer.get_turn_intensity(gps_start, gps_end)
                        # Net elevation range (max-min) guards the mountain class
                        # against summed GPS jitter on a flat plateau. near_coast
                        # stays False: no coastline signal reaches this path, so
                        # "coastal" is withheld rather than faked from a low
                        # elevation (see classify_terrain).
                        net_range = (max(elevs) - min(elevs)) if elevs else 0.0
                        terrain = classify_terrain(
                            gain_rate,
                            avg_elev,
                            turn_int,
                            net_elevation_range_m=net_range,
                        )

        is_contiguous = False
        if segments:
            prev = segments[-1]
            if (
                prev.source_file_index == file_idx
                and abs((prev.end_time + fb_start) - start_time) < 2.0
            ):
                is_contiguous = True

        segments.append(
            LFSegment(
                source_file_index=file_idx,
                source_path=Path(f"file_{file_idx:02d}.mp4"),
                start_time=local_start,
                end_time=local_end,
                original_start_time=start_time,
                original_end_time=end_time,
                composite_score=avg_score,
                gps_center_lat=center_lat,
                gps_center_lon=center_lon,
                terrain_type=terrain,
                contiguous_with_previous=is_contiguous,
                avg_motion_energy=avg_motion,
            )
        )

    def group_into_regular_lfs(
        self,
        segments: list[LFSegment],
        has_gps: bool = False,
    ) -> list[RegularLFPlan]:
        """Bin-pack segments into regular long-form videos.

        Constraints:
          - Each LF is min_lf_duration to max_lf_duration
          - Segments maintain chronological order within each LF
          - Each LF index is 1-based

        When best_moments_mode is enabled, segments are filtered using
        duration-aware quality scaling (_dynamic_best_moments_filter) which
        accumulates top-scoring segments until target_min is reached, then
        continues while quality stays above the floor (mean - 0.5*std) up to
        target_max. Montage peak selection draws from the pre-filter segment
        pool (scoped to each LF's time range) so the branded intro always
        picks from the broadest available candidates.

        Args:
            segments: Sorted list of LFSegments (by original_start_time).

        Returns:
            List of RegularLFPlan objects.
        """
        if not segments:
            return []

        # Sort by original timeline order
        segments = sorted(segments, key=lambda s: s.original_start_time)

        # best_moments_mode: keep only the top-scoring segments for LF grouping.
        # Save the full pool first so montage peaks can still draw from all candidates.
        pre_filter_segments = segments
        if self.config.best_moments_mode:
            segments = _dynamic_best_moments_filter(segments, self.config, has_gps=has_gps)

        # GPS-dead days can't derive distinct location labels for multiple LFs, so
        # pack into the fewest LFs — each up to the hard ceiling (LF_MAX_DURATION)
        # rather than the operational target — so one coarse label covers the whole
        # merged span. GPS-present days pack at max_lf_duration, unchanged.
        if has_gps:
            pack_target = self.config.max_lf_duration
        else:
            from ..video_types import LF_MAX_DURATION

            pack_target = LF_MAX_DURATION

        lfs: list[RegularLFPlan] = []
        current_segments: list[LFSegment] = []
        current_duration = 0.0

        for seg in segments:
            # Warn about single oversized segments
            if seg.duration > self.config.max_lf_duration:
                logger.warning(
                    "Segment %.0fs exceeds max LF duration %.0fs — including as-is",
                    seg.duration,
                    self.config.max_lf_duration,
                )

            # Account for crossfade overhead when checking max duration.
            # Non-contiguous segments require a crossfade transition.
            xfade_penalty = 0.0
            if current_segments and not seg.contiguous_with_previous:
                xfade_penalty = self.config.crossfade_duration

            # Would adding this segment (+ crossfade) exceed the packing target?
            if (
                current_duration + seg.duration + xfade_penalty > pack_target
                and current_duration >= self.config.min_lf_duration
            ):
                # Close current LF and start new one
                lfs.append(
                    self._create_lf_plan(current_segments, len(lfs) + 1, pre_filter_segments)
                )
                current_segments = []
                current_duration = 0.0
                xfade_penalty = 0.0  # First segment in new LF has no crossfade

            current_segments.append(seg)
            current_duration += seg.duration + xfade_penalty

        # Handle remaining segments — check for undersized final group.
        # When the final group is below min_lf_duration, try in order:
        #   1. Direct merge into previous LF (if fits under max_lf_duration)
        #   2. Redistribute segments between prev and current so both are valid
        #   3. Merge into previous LF up to a hard ceiling (LF_MAX_DURATION)
        #   4. Produce undersized LF as last resort
        if current_segments:
            remaining_dur = sum(s.duration for s in current_segments)

            if remaining_dur < self.config.min_lf_duration and lfs:
                prev_lf = lfs[-1]
                merged_segments, merged_dur, strategy = self._rebalance_undersized_tail(
                    prev_lf, current_segments
                )

                if strategy == "direct_merge":
                    lfs[-1] = self._create_lf_plan(
                        merged_segments, prev_lf.index, pre_filter_segments
                    )
                    logger.info(
                        "Merged undersized remainder (%.0fs) into LF %d (now %.0fs)",
                        remaining_dur,
                        prev_lf.index,
                        lfs[-1].total_duration,
                    )
                elif strategy == "redistribute":
                    new_prev_segs, new_tail_segs = merged_segments
                    lfs[-1] = self._create_lf_plan(
                        new_prev_segs, prev_lf.index, pre_filter_segments
                    )
                    new_tail_plan = self._create_lf_plan(
                        new_tail_segs, len(lfs) + 1, pre_filter_segments
                    )
                    lfs.append(new_tail_plan)
                    logger.info(
                        "Redistributed undersized tail: LF %d now %.0fs, new LF %d %.0fs",
                        prev_lf.index,
                        lfs[-2].total_duration,
                        new_tail_plan.index,
                        new_tail_plan.total_duration,
                    )
                elif strategy == "oversized_merge":
                    lfs[-1] = self._create_lf_plan(
                        merged_segments, prev_lf.index, pre_filter_segments
                    )
                    logger.warning(
                        "Merged undersized remainder (%.0fs) into LF %d "
                        "above max_lf_duration (%.0fs > %.0fs) — accepted up to "
                        "hard ceiling to avoid publishing an undersized LF",
                        remaining_dur,
                        prev_lf.index,
                        lfs[-1].total_duration,
                        self.config.max_lf_duration,
                    )
                else:
                    # strategy == "undersized" — all options exhausted
                    logger.warning(
                        "Undersized LF %d: %.0fs < min %.0fs — rebalance failed, "
                        "publishing as-is with undersized tag",
                        len(lfs) + 1,
                        remaining_dur,
                        self.config.min_lf_duration,
                    )
                    plan = self._create_lf_plan(current_segments, len(lfs) + 1, pre_filter_segments)
                    plan.undersized = True
                    lfs.append(plan)
            else:
                plan = self._create_lf_plan(current_segments, len(lfs) + 1, pre_filter_segments)
                if remaining_dur < self.config.min_lf_duration:
                    # First (and only) LF is undersized — no previous to merge with
                    logger.warning(
                        "Undersized LF %d: %.0fs < min %.0fs — no previous LF to merge into",
                        plan.index,
                        remaining_dur,
                        self.config.min_lf_duration,
                    )
                    plan.undersized = True
                lfs.append(plan)

        return lfs

    def _rebalance_undersized_tail(
        self,
        prev_lf: "RegularLFPlan",
        tail_segments: list["LFSegment"],
    ) -> tuple:
        """Decide how to handle an undersized tail group against the prior LF.

        Returns (result, total_duration, strategy) where strategy is one of:
          - "direct_merge": result is merged segment list; fits under max_lf_duration
          - "redistribute": result is (new_prev_segs, new_tail_segs), both valid
          - "oversized_merge": result is merged segment list; exceeds max but under
             the hard ceiling (LF_MAX_DURATION); preferred over publishing an
             undersized LF
          - "undersized": result is tail_segments unchanged; no valid option found

        The caller applies the returned strategy.
        """
        from ..video_types import LF_MAX_DURATION  # hard ceiling, separate from op default

        tail_dur = sum(s.duration for s in tail_segments)
        prev_dur = prev_lf.total_duration
        combined_dur = prev_dur + tail_dur
        merged_segments = list(prev_lf.segments) + list(tail_segments)

        # Option 1: direct merge under the operational ceiling
        if combined_dur <= self.config.max_lf_duration:
            return (merged_segments, combined_dur, "direct_merge")

        # Option 2: redistribute so both LFs end up valid.
        # Walk the combined segments and find a split point where:
        #   - left side >= min_lf_duration
        #   - right side >= min_lf_duration
        #   - both sides <= max_lf_duration
        # Prefer the split nearest the existing prev/tail boundary for stability.
        min_d = self.config.min_lf_duration
        max_d = self.config.max_lf_duration

        running = 0.0
        valid_splits: list[tuple[int, float, float]] = []
        for i, seg in enumerate(merged_segments[:-1]):
            running += seg.duration
            left_dur = running
            right_dur = combined_dur - running
            if min_d <= left_dur <= max_d and min_d <= right_dur <= max_d:
                valid_splits.append((i + 1, left_dur, right_dur))

        if valid_splits:
            # Prefer the split nearest the natural prev/tail boundary.
            natural_idx = len(prev_lf.segments)
            valid_splits.sort(key=lambda x: abs(x[0] - natural_idx))
            split_idx, left_dur, right_dur = valid_splits[0]
            new_prev = merged_segments[:split_idx]
            new_tail = merged_segments[split_idx:]
            return ((new_prev, new_tail), combined_dur, "redistribute")

        # Option 3: accept oversized merge up to the hard ceiling.
        # Publishing a slightly-too-long LF is strictly better than publishing an
        # undersized one that can't stand alone.
        if combined_dur <= LF_MAX_DURATION:
            return (merged_segments, combined_dur, "oversized_merge")

        # Option 4: all paths exhausted — caller publishes undersized.
        return (list(tail_segments), tail_dur, "undersized")

    def _create_lf_plan(
        self,
        segments: list[LFSegment],
        index: int,
        peak_pool: Optional[list[LFSegment]] = None,
    ) -> RegularLFPlan:
        """Create a RegularLFPlan from a list of segments.

        Args:
            segments: Segments that form this LF's content (may be best-moments filtered).
            index: 1-based LF index.
            peak_pool: Optional broader candidate set for montage peak selection
                (e.g. pre-best-moments segments). When provided, peaks are chosen from
                candidates whose time range overlaps this LF, giving the montage access
                to all footage quality rather than just the filtered set.
        """
        total_dur = sum(s.duration for s in segments)
        # Account for crossfade transitions between non-contiguous segments
        num_transitions = sum(
            1 for i in range(1, len(segments)) if not segments[i].contiguous_with_previous
        )
        total_dur += num_transitions * self.config.crossfade_duration

        # Montage peak selection: use the full pre-filter pool scoped to this LF's
        # time range when available, so the branded intro draws from all candidates.
        if peak_pool and segments:
            lf_start = min(s.original_start_time for s in segments)
            lf_end = max(s.original_end_time for s in segments)
            scoped_pool = [
                s
                for s in peak_pool
                if s.original_start_time >= lf_start and s.original_end_time <= lf_end
            ]
            peak_candidates = scoped_pool if scoped_pool else segments
        else:
            peak_candidates = segments
        montage_peaks = self._select_top_peaks(peak_candidates, self.config.montage_peak_count)

        return RegularLFPlan(
            index=index,
            segments=segments,
            montage_peaks=montage_peaks,
            shorts_target=max(self.config.min_shorts_per_lf, self.config.shorts_target_count),
            region="",  # Set by caller
            total_duration=total_dur,
        )

    def _select_top_peaks(
        self,
        segments: list[LFSegment],
        count: int,
    ) -> list[LFSegment]:
        """Select top N segments by composite score."""
        sorted_segs = sorted(segments, key=lambda s: s.composite_score, reverse=True)
        return sorted_segs[:count]

    def select_montage_peaks(
        self,
        lf: RegularLFPlan,
    ) -> list[LFSegment]:
        """Select top peaks for montage from an existing LF plan."""
        return self._select_top_peaks(lf.segments, self.config.montage_peak_count)

    def _adaptive_threshold_search(
        self,
        day_scores: np.ndarray,
        day_file_boundaries: list[tuple[float, float]],
        gps_per_episode: Optional[list] = None,
        stop_mask: Optional[np.ndarray] = None,
    ) -> tuple[list["LFSegment"], float]:
        """Search for a threshold that produces >=min_lf_duration of content.

        Lowers percentile in steps of 5, then lowers min_segment_duration in
        steps of 10s. Respects configured floors for both.

        ``stop_mask`` (per-second bool aligned to ``day_scores``, ``True`` ==
        STOP) has two effects: the percentile threshold is computed over
        UNMASKED composites only (else a heavily-parked day lowers the relative
        bar enough to KEEP parked footage), and it is passed through to
        ``find_segments_above_threshold`` so masked seconds can't enter a
        segment. Default None = unchanged.

        Returns:
            (segments, threshold_used)
        """
        composites = day_scores[:, COL_COMPOSITE]
        if composites.size == 0:
            return [], 0.0
        # Percentile is computed over UNMASKED composites so a mostly-parked day
        # can't drop the relative threshold enough to re-admit its own stops.
        pct_composites = composites
        if stop_mask is not None:
            m = np.asarray(stop_mask, dtype=bool)
            if m.shape[0] == composites.shape[0] and (~m).any():
                pct_composites = composites[~m]
        percentile = self.config.score_threshold_percentile
        min_seg = self.config.min_segment_duration

        percentile_floor = float(self.config.adaptive_percentile_floor)
        min_seg_floor = float(self.config.adaptive_min_segment_floor_s)

        # When best_moments_mode is enabled, the qualifying bar should be
        # best_moments_target_min (e.g. 900s = 15 min) so the downstream
        # filter has enough content to produce LFs in the 15-25 min range.
        # Without this, the search stops at min_lf_duration (480s = 8 min)
        # and the best_moments filter can't add content that wasn't selected.
        qualifying_min = self.config.min_lf_duration
        if self.config.best_moments_mode:
            qualifying_min = max(qualifying_min, self.config.best_moments_target_min)

        best_segments: list[LFSegment] = []
        best_threshold = 0.0

        while True:
            threshold = float(np.percentile(pct_composites, percentile))
            # Temporarily override min_segment_duration for this search step
            orig_min_seg = self.config.min_segment_duration
            self.config.min_segment_duration = min_seg

            segments = self.find_segments_above_threshold(
                day_scores,
                threshold,
                day_file_boundaries,
                gps_per_episode=gps_per_episode,
                stop_mask=stop_mask,
            )

            self.config.min_segment_duration = orig_min_seg

            total_qualifying = sum(s.duration for s in segments)
            if total_qualifying >= qualifying_min:
                logger.info(
                    "[adaptive_threshold] p%.0f/%.0fs → qualifying=%.1fmin (%d segments)",
                    percentile,
                    min_seg,
                    total_qualifying / 60,
                    len(segments),
                )
                return segments, threshold

            # Track best attempt
            if total_qualifying > sum(s.duration for s in best_segments):
                best_segments = segments
                best_threshold = threshold

            # Step 1: lower percentile
            if percentile > percentile_floor:
                percentile = max(percentile - 5, percentile_floor)
                continue

            # Step 2: lower min_segment_duration
            if min_seg > min_seg_floor:
                min_seg = max(min_seg - 10, min_seg_floor)
                # Reset percentile to try all percentile levels with new min_seg
                percentile = self.config.score_threshold_percentile
                continue

            # Both floors hit — use best available
            logger.warning(
                "[adaptive_threshold] floors exhausted (p%.0f/%.0fs) — best=%.1fmin (%d segments)",
                percentile_floor,
                min_seg_floor,
                sum(s.duration for s in best_segments) / 60,
                len(best_segments),
            )
            return best_segments, best_threshold

    def build_drive_plan_per_day(
        self,
        scores: np.ndarray,
        processed_files: list[Path],
        file_boundaries: list[tuple[float, float]],
        episode_dates: list[str],
        gps_per_episode: Optional[list] = None,
        region: str = "",
        skip_file_indices: Optional[set[int]] = None,
        stop_mask: Optional[np.ndarray] = None,
    ) -> DrivePlan:
        """Build a drive plan with per-day adaptive thresholds.

        Groups episodes by recording date, runs adaptive threshold search
        per day, and ensures each day produces at least one LF.

        Falls back to global threshold (build_drive_plan) when
        adaptive_threshold_enabled is False.

        Args:
            scores: Drive-wide score array (N, 9).
            processed_files: Processed video file paths (aligned with file_boundaries).
            file_boundaries: (start, end) timestamps per file.
            episode_dates: YYYY-MM-DD date per file (aligned with file_boundaries).
            gps_per_episode: GPS data per file (optional).
            region: Content region.
            skip_file_indices: File indices to exclude.
            stop_mask: Optional per-second bool array aligned to the DRIVE-WIDE
                ``scores`` (``True`` == STOP). Sliced per day with the same
                boolean mask used to slice ``day_scores`` and threaded into the
                adaptive search, so parked seconds can neither lower the
                percentile bar nor enter a segment. Default None = unchanged.

        Returns:
            Complete DrivePlan.
        """
        skip_file_indices = skip_file_indices or set()

        drive_stop_mask: Optional[np.ndarray] = None
        if stop_mask is not None:
            _m = np.asarray(stop_mask, dtype=bool)
            if _m.shape[0] == scores.shape[0]:
                drive_stop_mask = _m

        if len(episode_dates) != len(file_boundaries):
            raise ValueError(
                f"episode_dates length ({len(episode_dates)}) must match "
                f"file_boundaries length ({len(file_boundaries)})"
            )

        # Full-drive mode bypasses the adaptive-threshold scoring
        # path entirely. One LF per day spanning the day's full processed
        # timeline. plan_regular_lfs enforces min/max duration upstream.
        if getattr(self.config, "lf_selection_mode", "best_moments") == "full_drive":
            return self._build_full_drive_plan_per_day(
                scores=scores,
                processed_files=processed_files,
                file_boundaries=file_boundaries,
                episode_dates=episode_dates,
                gps_per_episode=gps_per_episode,
                region=region,
                skip_file_indices=skip_file_indices,
                stop_mask=drive_stop_mask,
            )

        # Fallback: delegate to global threshold path
        if not self.config.adaptive_threshold_enabled:
            return self.build_drive_plan(
                scores=scores,
                processed_files=processed_files,
                file_boundaries=file_boundaries,
                gps_per_episode=gps_per_episode,
                region=region,
                skip_file_indices=skip_file_indices,
                stop_mask=drive_stop_mask,
            )

        total_duration = float(scores[-1, COL_TIMESTAMP]) if len(scores) > 0 else 0.0

        # Group file indices by date
        day_groups: dict[str, list[int]] = {}
        for i, date in enumerate(episode_dates):
            if i not in skip_file_indices:
                day_groups.setdefault(date, []).append(i)

        all_segments: list[LFSegment] = []
        all_lfs: list[RegularLFPlan] = []
        global_lf_index = 1
        thresholds_used: list[float] = []

        for date in sorted(day_groups.keys()):
            file_indices = day_groups[date]
            day_boundaries = [file_boundaries[i] for i in file_indices]

            # Slice score array to this day's time range
            day_start = min(fb[0] for fb in day_boundaries)
            day_end = max(fb[1] for fb in day_boundaries)
            mask = (scores[:, COL_TIMESTAMP] >= day_start) & (scores[:, COL_TIMESTAMP] < day_end)
            day_scores = scores[mask]

            if len(day_scores) == 0:
                logger.warning("[adaptive_threshold] day=%s: no scores, skipping", date)
                continue

            # Slice the drive-wide stop mask to this day with the SAME boolean
            # selector used for day_scores, so rows stay aligned.
            day_stop_mask = drive_stop_mask[mask] if drive_stop_mask is not None else None

            # GPS for this day's files
            day_gps = None
            if gps_per_episode:
                day_gps = [gps_per_episode[i] for i in file_indices]

            logger.info(
                "[adaptive_threshold] day=%s: %d files, %.1fmin of footage",
                date,
                len(file_indices),
                (day_end - day_start) / 60,
            )

            segments, threshold = self._adaptive_threshold_search(
                day_scores,
                day_boundaries,
                gps_per_episode=day_gps,
                stop_mask=day_stop_mask,
            )
            thresholds_used.append(threshold)

            # Remap source_file_index from day-local (0-based within day)
            # to global (0-based within entire drive). find_segments_above_threshold
            # assigns indices relative to the passed file_boundaries list.
            for seg in segments:
                local_idx = seg.source_file_index
                if local_idx < len(file_indices):
                    seg.source_file_index = file_indices[local_idx]

            # Filter out segments from failed files
            segments = [s for s in segments if s.source_file_index not in skip_file_indices]

            # Update source paths
            for seg in segments:
                if seg.source_file_index < len(processed_files):
                    seg.source_path = processed_files[seg.source_file_index]

            all_segments.extend(segments)

            # Group this day's segments into LFs
            day_lfs = self.group_into_regular_lfs(segments, has_gps=_has_gps(day_gps))
            for lf in day_lfs:
                lf.index = global_lf_index
                lf.region = region
                global_lf_index += 1
            all_lfs.extend(day_lfs)

            logger.info(
                "[adaptive_threshold] day=%s: threshold=%.4f → %d segments, %d LFs",
                date,
                threshold,
                len(segments),
                len(day_lfs),
            )

        highlight_duration = sum(s.duration for s in all_segments)
        avg_threshold = sum(thresholds_used) / len(thresholds_used) if thresholds_used else 0.0

        logger.info(
            "Per-day adaptive: %d LFs from %d days, %.0fs highlights / %.0fs total (%.0f%%)",
            len(all_lfs),
            len(day_groups),
            highlight_duration,
            total_duration,
            (highlight_duration / total_duration * 100) if total_duration > 0 else 0,
        )

        return DrivePlan(
            source_files=processed_files,
            processed_files=processed_files,
            regular_lfs=all_lfs,
            score_threshold=avg_threshold,
            total_source_duration=total_duration,
            total_highlight_duration=highlight_duration,
        )

    def build_drive_plan(
        self,
        scores: np.ndarray,
        processed_files: list[Path],
        file_boundaries: list[tuple[float, float]],
        gps_per_episode: Optional[list] = None,
        region: str = "",
        skip_file_indices: Optional[set[int]] = None,
        stop_mask: Optional[np.ndarray] = None,
    ) -> DrivePlan:
        """Top-level: scores -> threshold -> segments -> group -> plan.

        Args:
            scores: Drive-wide score array (N, 9).
            processed_files: List of processed video file paths.
            file_boundaries: (start, end) timestamps per file.
            gps_per_episode: Per-file GPS data: [(gps_data, gps_offset), ...].
            region: Content region ("ireland" or "usa").
            skip_file_indices: Set of file indices to exclude (e.g. failed encodes).

        Returns:
            Complete DrivePlan ready for assembly.
        """
        skip_file_indices = skip_file_indices or set()

        # Full-drive mode bypasses score-thresholded selection. The
        # legacy non-day-first path is not first-class, but completeness
        # avoids a future invocation that hits this method and silently
        # falls back to highlights logic.
        if getattr(self.config, "lf_selection_mode", "best_moments") == "full_drive":
            return self._build_full_drive_plan(
                scores=scores,
                processed_files=processed_files,
                file_boundaries=file_boundaries,
                gps_per_episode=gps_per_episode,
                region=region,
                skip_file_indices=skip_file_indices,
                stop_mask=stop_mask,
            )

        composites = scores[:, COL_COMPOSITE]
        total_duration = float(scores[-1, COL_TIMESTAMP]) if len(scores) > 0 else 0.0

        if composites.size == 0:
            logger.warning("build_drive_plan called with empty scores; returning empty plan")
            return DrivePlan(
                source_files=processed_files,
                processed_files=processed_files,
                regular_lfs=[],
                score_threshold=0.0,
                total_source_duration=total_duration,
                total_highlight_duration=0.0,
            )

        # Compute threshold from percentile over UNMASKED composites so a
        # heavily-parked drive can't lower the relative bar and re-admit stops.
        pct_composites = composites
        if stop_mask is not None:
            _m = np.asarray(stop_mask, dtype=bool)
            if _m.shape[0] == composites.shape[0] and (~_m).any():
                pct_composites = composites[~_m]
        threshold = float(np.percentile(pct_composites, self.config.score_threshold_percentile))
        logger.info(
            "Score threshold at %.0fth percentile: %.3f | "
            "Score range: min=%.3f, max=%.3f, mean=%.3f",
            self.config.score_threshold_percentile,
            threshold,
            float(np.min(composites)),
            float(np.max(composites)),
            float(np.mean(composites)),
        )

        # Find segments above threshold
        segments = self.find_segments_above_threshold(
            scores,
            threshold,
            file_boundaries,
            gps_per_episode=gps_per_episode,
            stop_mask=stop_mask,
        )

        # Filter out segments from failed files
        if skip_file_indices:
            before = len(segments)
            segments = [s for s in segments if s.source_file_index not in skip_file_indices]
            if before != len(segments):
                logger.warning(
                    "Dropped %d segments from failed file indices %s",
                    before - len(segments),
                    sorted(skip_file_indices),
                )

        # Update source paths from processed_files
        for seg in segments:
            if seg.source_file_index < len(processed_files):
                seg.source_path = processed_files[seg.source_file_index]

        total_above = len(segments)
        total_samples = len(composites)
        logger.info(
            "Found %d segments above threshold (total %.0fs from %d/%d samples)",
            total_above,
            sum(s.duration for s in segments),
            sum(1 for c in composites if c >= threshold),
            total_samples,
        )

        # Group into regular LFs
        lfs = self.group_into_regular_lfs(segments, has_gps=_has_gps(gps_per_episode))

        # Set region on all LFs
        for lf in lfs:
            lf.region = region

        highlight_duration = sum(s.duration for s in segments)

        logger.info(
            f"Assembled {len(lfs)} regular long forms from "
            f"{highlight_duration:.0f}s of highlights ({total_duration:.0f}s total)"
        )

        return DrivePlan(
            source_files=processed_files,
            processed_files=processed_files,
            regular_lfs=lfs,
            score_threshold=threshold,
            total_source_duration=total_duration,
            total_highlight_duration=highlight_duration,
        )

    # ------------------------------------------------------------------
    # Full-drive LF construction
    # ------------------------------------------------------------------

    def _build_full_drive_lf_segments(
        self,
        day_scores: np.ndarray,
        day_file_boundaries: list[tuple[float, float]],
        file_indices: list[int],
        processed_files: list[Path],
        ranges: Optional[list] = None,
    ) -> list[LFSegment]:
        """Build LFSegments spanning a day's processed timeline MINUS stops.

        ``ranges`` are the planner's maximal UNMASKED spans
        (``plan_regular_lfs(mode="full_drive")`` → ``RegularLFRange`` in the
        same timestamps as ``day_scores``). Each range is intersected with
        each source file, one LFSegment per piece, so a parked span is
        physically cut out of the LF — this used to emit one whole-file
        segment per file and the mask only validated the kept duration, so
        parked time still shipped. ``ranges=None`` keeps the whole-file
        behaviour (callers that have no planner output).
        A piece is ``contiguous_with_previous`` when it starts within 1 s of
        the previous piece's end (file-to-file joins stay contiguous; a
        cut-out stop breaks contiguity and earns a crossfade).
        """
        composites = day_scores[:, COL_COMPOSITE]
        timestamps = day_scores[:, COL_TIMESTAMP]
        spans = [(float(r.start_sec), float(r.end_sec)) for r in ranges] if ranges else None
        segments: list[LFSegment] = []
        prev_end: Optional[float] = None
        for local_idx, (fb_start, fb_end) in enumerate(day_file_boundaries):
            file_duration = max(0.0, fb_end - fb_start)
            if file_duration <= 0:
                continue
            # 0.5s EOF guard matches _maybe_add_segment convention
            file_end = fb_start + max(0.0, file_duration - 0.5)
            if spans is None:
                pieces = [(fb_start, file_end)]
            else:
                pieces = [
                    (max(fb_start, s), min(file_end, e))
                    for s, e in spans
                    if e > fb_start and s < file_end
                ]
            global_idx = file_indices[local_idx]
            if global_idx < len(processed_files):
                src_path = processed_files[global_idx]
            else:
                src_path = Path(f"file_{global_idx:02d}.mp4")
            for o_start, o_end in pieces:
                if o_end - o_start <= 0:
                    continue
                # Per-piece score mask for accurate avg metadata
                mask = (timestamps >= o_start) & (timestamps < o_end)
                avg_score = float(np.mean(composites[mask])) if mask.any() else 0.0
                avg_motion = (
                    float(np.mean(day_scores[mask, COL_MOTION_ENERGY])) if mask.any() else 0.0
                )
                contiguous = prev_end is not None and abs(o_start - prev_end) <= 1.0
                segments.append(
                    LFSegment(
                        source_file_index=global_idx,
                        source_path=src_path,
                        start_time=o_start - fb_start,
                        end_time=o_end - fb_start,
                        original_start_time=o_start,
                        original_end_time=o_end,
                        composite_score=avg_score,
                        gps_center_lat=0.0,
                        gps_center_lon=0.0,
                        terrain_type="unknown",
                        contiguous_with_previous=contiguous,
                        avg_motion_energy=avg_motion,
                    )
                )
                prev_end = o_end
        return segments

    def _build_full_drive_peak_pool(
        self,
        day_scores: np.ndarray,
        day_file_boundaries: list[tuple[float, float]],
        file_indices: list[int],
        processed_files: list[Path],
        gps_per_episode: Optional[list],
    ) -> list[LFSegment]:
        """Score-derived candidate pool for montage peak selection.

        Full-drive LFs collapse the body to one giant segment, so the
        default ``peak_pool == segments`` path in ``_create_lf_plan``
        would degenerate to a single peak. Build a broader candidate
        set via the existing threshold-based scan at p50 so the montage
        intro still cuts from real high-score sub-windows.
        """
        composites = day_scores[:, COL_COMPOSITE]
        if composites.size == 0:
            return []
        threshold = float(np.percentile(composites, 50))
        # Temporarily shrink min_segment_duration so the pool isn't empty
        # for short single-file Insta360 drives.
        orig_min_seg = self.config.min_segment_duration
        self.config.min_segment_duration = 5.0
        try:
            pool = self.find_segments_above_threshold(
                day_scores,
                threshold,
                day_file_boundaries,
                gps_per_episode=gps_per_episode,
            )
        finally:
            self.config.min_segment_duration = orig_min_seg
        # Remap day-local source_file_index to global; populate paths
        for seg in pool:
            local_idx = seg.source_file_index
            if 0 <= local_idx < len(file_indices):
                seg.source_file_index = file_indices[local_idx]
            if seg.source_file_index < len(processed_files):
                seg.source_path = processed_files[seg.source_file_index]
        return pool

    def _build_full_drive_plan_per_day(
        self,
        scores: np.ndarray,
        processed_files: list[Path],
        file_boundaries: list[tuple[float, float]],
        episode_dates: list[str],
        gps_per_episode: Optional[list],
        region: str,
        skip_file_indices: set[int],
        stop_mask: Optional[np.ndarray] = None,
    ) -> DrivePlan:
        """Day-first full-drive: one LF per calendar day, spanning the
        day's entire post-stop-removal timeline."""
        from .regular_lf_planner import plan_full_drive_parts

        total_duration = float(scores[-1, COL_TIMESTAMP]) if len(scores) > 0 else 0.0

        day_groups: dict[str, list[int]] = {}
        for i, date in enumerate(episode_dates):
            if i not in skip_file_indices:
                day_groups.setdefault(date, []).append(i)

        all_lfs: list[RegularLFPlan] = []
        all_segments: list[LFSegment] = []
        global_lf_index = 1
        full_drive_max = int(getattr(self.config, "full_drive_max_duration_sec", 7200))
        part_min = getattr(self.config, "full_drive_part_min_sec", None)
        part_max = getattr(self.config, "full_drive_part_max_sec", None)

        for date in sorted(day_groups.keys()):
            file_indices = day_groups[date]
            day_boundaries = [file_boundaries[i] for i in file_indices]
            day_start = min(fb[0] for fb in day_boundaries)
            day_end = max(fb[1] for fb in day_boundaries)
            mask = (scores[:, COL_TIMESTAMP] >= day_start) & (scores[:, COL_TIMESTAMP] < day_end)
            day_scores = scores[mask]
            if len(day_scores) == 0:
                logger.warning("[full_drive] day=%s: no scores, skipping", date)
                continue

            # Slice the drive-wide stop mask to this day with the SAME
            # boolean selector used for day_scores, so rows stay aligned.
            day_stop_mask = stop_mask[mask] if stop_mask is not None else None

            # Raises ValueError on missing mask / under-min / over-max —
            # robust-by-construction. With the part bounds set the day is
            # split into balanced parts and min/ceiling apply per part.
            parts = plan_full_drive_parts(
                day_scores,
                min_duration_sec=self.config.min_lf_duration,
                full_drive_max_duration_sec=full_drive_max,
                stop_mask=day_stop_mask,
                part_min_sec=part_min,
                part_max_sec=part_max,
            )
            parts = [p for p in parts if p]
            if not parts:
                continue

            day_gps = None
            if gps_per_episode:
                day_gps = [gps_per_episode[i] for i in file_indices]

            # The pool is built once per day; _create_lf_plan scopes it to
            # each part's own time span, keeping peaks partitioned per part.
            peak_pool = self._build_full_drive_peak_pool(
                day_scores,
                day_boundaries,
                file_indices,
                processed_files,
                gps_per_episode=day_gps,
            )
            split = len(parts) > 1
            for part_no, part_ranges in enumerate(parts, 1):
                full_segs = self._build_full_drive_lf_segments(
                    day_scores,
                    day_boundaries,
                    file_indices,
                    processed_files,
                    ranges=part_ranges,
                )
                if not full_segs:
                    if split:
                        raise ValueError(
                            f"[full_drive] day={date}: part {part_no}/{len(parts)} "
                            "produced no segments (empty part); refusing to drop it"
                        )
                    continue
                plan = self._create_lf_plan(full_segs, global_lf_index, peak_pool)
                plan.region = region
                if split:
                    plan.part_index = part_no
                    plan.part_count = len(parts)
                global_lf_index += 1

                all_lfs.append(plan)
                all_segments.extend(full_segs)

                logger.info(
                    "[full_drive] day=%s: LF %d/%d spanning %.1fs (%d source files)",
                    date,
                    part_no,
                    len(parts),
                    plan.total_duration,
                    len({s.source_file_index for s in full_segs}),
                )

        highlight_duration = sum(s.duration for s in all_segments)
        logger.info(
            "Full-drive: %d LFs from %d days, %.0fs spanned / %.0fs total",
            len(all_lfs),
            len(day_groups),
            highlight_duration,
            total_duration,
        )

        return DrivePlan(
            source_files=processed_files,
            processed_files=processed_files,
            regular_lfs=all_lfs,
            score_threshold=0.0,
            total_source_duration=total_duration,
            total_highlight_duration=highlight_duration,
        )

    def _build_full_drive_plan(
        self,
        scores: np.ndarray,
        processed_files: list[Path],
        file_boundaries: list[tuple[float, float]],
        gps_per_episode: Optional[list],
        region: str,
        skip_file_indices: set[int],
        stop_mask: Optional[np.ndarray] = None,
    ) -> DrivePlan:
        """Non-day-first full-drive: collapse the entire drive into a
        single LF. Legacy path; day-first is the first-class invocation.
        """
        from .regular_lf_planner import plan_full_drive_parts

        total_duration = float(scores[-1, COL_TIMESTAMP]) if len(scores) > 0 else 0.0
        kept_file_indices = [i for i in range(len(file_boundaries)) if i not in skip_file_indices]
        if not kept_file_indices:
            return DrivePlan(
                source_files=processed_files,
                processed_files=processed_files,
                regular_lfs=[],
                score_threshold=0.0,
                total_source_duration=total_duration,
                total_highlight_duration=0.0,
            )

        full_drive_max = int(getattr(self.config, "full_drive_max_duration_sec", 7200))
        # Raises on missing mask / out-of-bounds
        part_min = getattr(self.config, "full_drive_part_min_sec", None)
        part_max = getattr(self.config, "full_drive_part_max_sec", None)
        parts = plan_full_drive_parts(
            scores,
            min_duration_sec=self.config.min_lf_duration,
            full_drive_max_duration_sec=full_drive_max,
            stop_mask=stop_mask,
            part_min_sec=part_min,
            part_max_sec=part_max,
        )
        if len(parts) > 1:
            raise ValueError(
                f"full_drive balanced parts: the drive splits into {len(parts)} parts "
                "but the legacy (non-day-first) path emits a single LF. Use the "
                "day-first path, or unset full_drive_part_min_sec/full_drive_part_max_sec."
            )
        ranges = parts[0]
        if not ranges:
            return DrivePlan(
                source_files=processed_files,
                processed_files=processed_files,
                regular_lfs=[],
                score_threshold=0.0,
                total_source_duration=total_duration,
                total_highlight_duration=0.0,
            )

        kept_boundaries = [file_boundaries[i] for i in kept_file_indices]
        full_segs = self._build_full_drive_lf_segments(
            scores, kept_boundaries, kept_file_indices, processed_files, ranges=ranges
        )
        peak_pool = self._build_full_drive_peak_pool(
            scores,
            kept_boundaries,
            kept_file_indices,
            processed_files,
            gps_per_episode=gps_per_episode,
        )
        plan = self._create_lf_plan(full_segs, 1, peak_pool)
        plan.region = region
        highlight_duration = sum(s.duration for s in full_segs)
        logger.info(
            "Full-drive: 1 LF spanning %.0fs (%d source files) / %.0fs total",
            plan.total_duration,
            len({s.source_file_index for s in full_segs}),
            total_duration,
        )
        return DrivePlan(
            source_files=processed_files,
            processed_files=processed_files,
            regular_lfs=[plan],
            score_threshold=0.0,
            total_source_duration=total_duration,
            total_highlight_duration=highlight_duration,
        )
