"""Segment scoring and selection for shorts extraction."""

import logging
from typing import Dict, List, Optional

from ..models import DeadSpot, HighMotionSegment
from .scenic_scorer import ScenicScorer
from .shorts_models import ScoringWeights, _haversine_m
from .shorts_time_mapping import VideoToGpsTimeMapper

logger = logging.getLogger(__name__)

# Dead-spot fallback threshold: when every candidate overlaps a dead spot
# (with buffer), only candidates spending less than this fraction of their
# duration actually inside dead spots are kept. Tightened from 0.5 to 0.1:
# a "mostly-moving" short may carry at most ~10% stopped footage — the old 50%
# tolerance shipped clips that were half parked. The real footage that survives
# is dominated by motion; a candidate more than a tenth inside a stop is dropped.
_MOSTLY_MOVING_MAX_OVERLAP = 0.1


class ShortsCandidateSelector:
    """Scores and selects the best non-overlapping shorts from candidate segments."""

    def __init__(
        self,
        scoring_weights: ScoringWeights,
        min_gap: float,
        min_duration: float,
        max_duration: float,
        target_duration: float,
        time_mapper: Optional[VideoToGpsTimeMapper] = None,
        scenic_scorer: Optional[ScenicScorer] = None,
        geo_dedup_meters: float = 500.0,
    ):
        self.scoring_weights = scoring_weights
        self.min_gap = min_gap
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_duration = target_duration
        self._time_mapper = time_mapper
        self._scenic_scorer = scenic_scorer
        self.geo_dedup_meters = geo_dedup_meters

    def _segment_overlaps_stop(
        self,
        segment: HighMotionSegment,
        dead_spots: List[DeadSpot],
        buffer: float = 5.0,
    ) -> bool:
        """Check if segment overlaps with any dead spot (with buffer)."""
        for stop in dead_spots:
            # Check if segment overlaps with stop (including buffer)
            if not (
                segment.end_time + buffer < stop.start_time
                or segment.start_time - buffer > stop.end_time
            ):
                return True
        return False

    def _dead_spot_overlap_fraction(
        self,
        segment: HighMotionSegment,
        dead_spots: List[DeadSpot],
    ) -> float:
        """
        Fraction of the segment's duration spent actually inside dead spots.

        Unlike _segment_overlaps_stop this uses NO proximity buffer: a segment
        merely near a stop (e.g. next to a crossfade transition) scores 0.0.
        Overlapping dead spots may double-count, so the result is capped at
        1.0 — it is a preference signal, not an exact measure.
        """
        duration = segment.end_time - segment.start_time
        if duration <= 0:
            return 1.0
        overlap = sum(
            max(
                0.0,
                min(segment.end_time, stop.end_time) - max(segment.start_time, stop.start_time),
            )
            for stop in dead_spots
        )
        return min(overlap / duration, 1.0)

    def _segments_too_close(
        self,
        seg1: HighMotionSegment,
        seg2: HighMotionSegment,
    ) -> bool:
        """Check if two segments are too close together."""
        gap = min(abs(seg1.start_time - seg2.end_time), abs(seg2.start_time - seg1.end_time))
        return gap < self.min_gap

    def _calculate_spread_score(
        self,
        segment: HighMotionSegment,
        selected_segments: List[HighMotionSegment],
        video_duration: float,
    ) -> float:
        """
        Calculate spread score based on distance from already-selected segments.

        Higher score = segment is far from already-selected segments.
        This encourages variety across the video timeline.

        Args:
            segment: Segment to score
            selected_segments: Already selected segments
            video_duration: Total video duration

        Returns:
            Score from 0-100 (100 = maximum spread)
        """
        if not selected_segments or video_duration <= 0:
            return 100.0  # First segment gets full spread score

        # Calculate midpoint of this segment
        seg_mid = (segment.start_time + segment.end_time) / 2

        # Find minimum distance to any selected segment
        min_distance = float("inf")
        for sel in selected_segments:
            sel_mid = (sel.start_time + sel.end_time) / 2
            distance = abs(seg_mid - sel_mid)
            min_distance = min(min_distance, distance)

        # Ideal spread: segments evenly distributed
        # If we have N selected, ideal gap is duration/(N+1)
        num_selected = len(selected_segments)
        ideal_gap = video_duration / (num_selected + 2)  # +2 for start/end buffer

        # Score based on how close to ideal gap
        # 100 if at ideal gap or further, decreases linearly below
        if min_distance >= ideal_gap:
            return 100.0
        else:
            return (min_distance / ideal_gap) * 100

    def _score_segment(
        self,
        segment: HighMotionSegment,
        video_duration: float,
        selected_segments: Optional[List[HighMotionSegment]] = None,
        visual_scores: Optional[Dict[int, float]] = None,
        segment_index: Optional[int] = None,
    ) -> float:
        """
        Score a segment for shorts suitability.

        Higher score = better short candidate.
        Uses configurable weights from self.scoring_weights.

        Args:
            segment: Segment to score
            video_duration: Total video duration
            selected_segments: Already selected segments (for spread calculation)
            visual_scores: Pre-computed visual interest scores by segment index
            segment_index: Index of this segment in the candidates list

        Returns:
            Weighted score
        """
        w = self.scoring_weights
        selected = selected_segments or []

        # Normalize each factor to roughly 0-100 scale, then apply weights

        # Motion energy: typical range 0.01-0.10, scale to 0-100
        motion_score = min(segment.avg_motion_energy * 1000, 100)

        # Peak motion: bonus for exciting moments
        peak_score = min(segment.peak_motion_energy * 800, 100)

        # Duration match: 100 if perfect, decreases with deviation
        duration_diff = abs(segment.duration - self.target_duration)
        max_diff = self.max_duration - self.min_duration
        duration_score = max(0, 100 - (duration_diff / max_diff) * 100)

        # Position: prefer middle of video (not very start/end)
        relative_pos = segment.start_time / video_duration if video_duration > 0 else 0.5
        # Score peaks at 0.5 (middle), lowest at 0 and 1
        position_score = 100 * (1 - abs(relative_pos - 0.5) * 2)
        # Boost middle range (0.1-0.9 gets full score)
        if 0.1 <= relative_pos <= 0.9:
            position_score = max(position_score, 80)

        # Spread score: higher when far from already-selected segments
        spread_score = self._calculate_spread_score(segment, selected, video_duration)

        # Visual interest score: from cache or neutral default
        if w.visual_interest > 0 and visual_scores and segment_index is not None:
            visual_score = visual_scores.get(segment_index, 50.0)
        else:
            visual_score = 50.0  # Neutral if not computed

        # Scenic location score: from GPS data or neutral default
        if (
            w.scenic_location > 0
            and self._scenic_scorer is not None
            and self._time_mapper is not None
        ):
            gps_start = self._time_mapper.video_to_gps_time(segment.start_time)
            gps_end = self._time_mapper.video_to_gps_time(segment.end_time)
            scenic_score = self._scenic_scorer.score_segment(gps_start, gps_end)
        else:
            scenic_score = 50.0  # Neutral if not computed

        # Weighted combination
        score = (
            w.motion_energy * motion_score
            + w.peak_motion * peak_score
            + w.duration_match * duration_score
            + w.position * position_score
            + w.spread * spread_score
            + w.visual_interest * visual_score
            + w.scenic_location * scenic_score
        )

        return score

    def select_best_shorts(
        self,
        candidates: List[HighMotionSegment],
        dead_spots: List[DeadSpot],
        video_duration: float,
        num_shorts: int = 5,
        visual_scores: Optional[Dict[int, float]] = None,
    ) -> List[HighMotionSegment]:
        """
        Select the best non-overlapping shorts from candidates.

        Uses iterative selection with re-scoring to account for spread:
        after each selection, remaining candidates are re-scored based
        on their distance from already-selected segments.

        Args:
            candidates: High motion segments to choose from
            dead_spots: Stops to avoid
            video_duration: Total video duration
            num_shorts: Number of shorts to extract
            visual_scores: Pre-computed visual interest scores by candidate index

        Returns:
            List of selected segments
        """
        logger.debug(
            f"Selecting {num_shorts} shorts from {len(candidates)} candidates, "
            f"avoiding {len(dead_spots)} dead spots"
        )

        # Filter out segments that overlap with stops
        valid_candidates = [
            seg for seg in candidates if not self._segment_overlaps_stop(seg, dead_spots)
        ]

        if not valid_candidates and candidates:
            # Fallback: ALL candidates overlap dead spots (common for
            # assembled highlight reels where crossfade transitions register
            # as dead spots and the proximity buffer taints neighbouring
            # segments). Instead of using every candidate blindly — which can
            # ship stopped footage — keep only mostly-moving candidates:
            # those spending less than half their duration actually inside a
            # dead spot (measured without buffer).
            fractions = {
                id(seg): self._dead_spot_overlap_fraction(seg, dead_spots) for seg in candidates
            }
            mostly_moving = [
                seg for seg in candidates if fractions[id(seg)] < _MOSTLY_MOVING_MAX_OVERLAP
            ]
            if mostly_moving:
                logger.warning(
                    f"All {len(candidates)} candidates overlap dead spots "
                    f"(buffered) — keeping {len(mostly_moving)} mostly-moving "
                    f"candidate(s) with <{_MOSTLY_MOVING_MAX_OVERLAP:.0%} of "
                    f"their duration inside a dead spot"
                )
                valid_candidates = mostly_moving
            else:
                logger.warning(
                    f"All {len(candidates)} candidates spend >="
                    f"{_MOSTLY_MOVING_MAX_OVERLAP:.0%} of their duration inside "
                    f"dead spots — refusing to ship stopped footage, "
                    f"selecting none"
                )

        # Build index mapping for valid candidates back to original indices
        # (needed for visual_scores lookup) - use id() since dataclasses aren't hashable
        candidate_indices = {id(seg): i for i, seg in enumerate(candidates)}

        selected: List[HighMotionSegment] = []
        remaining = valid_candidates.copy()

        # Pre-compute GPS centers for geographic deduplication
        _geo_centers: dict[int, tuple[float, float]] = {}
        _missing_gps_segments = 0
        if self._scenic_scorer is not None and self._time_mapper is not None:
            for seg in valid_candidates:
                _gps_start = self._time_mapper.video_to_gps_time(seg.start_time)
                _gps_end = self._time_mapper.video_to_gps_time(seg.end_time)
                _pts = self._scenic_scorer.gps_data.get_points_in_range(_gps_start, _gps_end)
                if _pts:
                    _geo_centers[id(seg)] = (
                        sum(p.lat for p in _pts) / len(_pts),
                        sum(p.lon for p in _pts) / len(_pts),
                    )
                else:
                    # Silent-fallback fix: no GPS points in range means this
                    # segment bypasses geo-dedup entirely (kept without spatial
                    # comparison). Surface it so unexpected duplicates can be
                    # traced to GPS coverage gaps rather than buggy dedup.
                    _missing_gps_segments += 1
            if _missing_gps_segments:
                logger.warning(
                    "Geo-dedup: %d/%d candidate segment(s) have no GPS points "
                    "in range — those segments bypass geo-dedup and may cluster "
                    "spatially",
                    _missing_gps_segments,
                    len(valid_candidates),
                )

        # Rejection accounting for shortfall diagnostics
        rejections = {
            "gap": 0,
            "geo_dedup": 0,
            "dead_spot_filter": len(candidates) - len(valid_candidates),
        }

        # Iteratively select best segment, re-scoring each time for spread
        while len(selected) < num_shorts and remaining:
            # Score remaining candidates (with current selection for spread calc)
            scored = []
            for seg in remaining:
                orig_idx = candidate_indices.get(id(seg))
                score = self._score_segment(
                    seg,
                    video_duration,
                    selected,
                    visual_scores=visual_scores,
                    segment_index=orig_idx,
                )
                scored.append((seg, score))

            # Sort by score (highest first)
            scored.sort(key=lambda x: x[1], reverse=True)

            # Find best segment that's not too close to already selected
            found = False
            for seg, score in scored:
                too_close = any(self._segments_too_close(seg, existing) for existing in selected)
                rejected_reason: Optional[str] = "gap" if too_close else None

                # Geographic dedup: segments within configurable distance are redundant
                if not too_close and _geo_centers:
                    seg_center = _geo_centers.get(id(seg))
                    if seg_center:
                        for existing in selected:
                            ex_center = _geo_centers.get(id(existing))
                            if ex_center:
                                _dist = _haversine_m(
                                    seg_center[0],
                                    seg_center[1],
                                    ex_center[0],
                                    ex_center[1],
                                )
                                if _dist < self.geo_dedup_meters:
                                    too_close = True
                                    rejected_reason = "geo_dedup"
                                    break

                if not too_close:
                    selected.append(seg)
                    remaining.remove(seg)
                    found = True
                    break
                if rejected_reason is not None:
                    rejections[rejected_reason] += 1

            # If no valid segment found, we're done
            if not found:
                break

        # Surface silent shortfall: if we couldn't meet the request, say why.
        if len(selected) < num_shorts:
            logger.warning(
                "Shorts shortfall: requested=%d selected=%d. "
                "Rejection breakdown: dead_spot_filter=%d, gap=%d, geo_dedup=%d. "
                "(geo_dedup_threshold_m=%.0f, min_gap_s=%.1f)",
                num_shorts,
                len(selected),
                rejections["dead_spot_filter"],
                rejections["gap"],
                rejections["geo_dedup"],
                self.geo_dedup_meters,
                self.min_gap,
            )

        # Sort selected by time order
        selected.sort(key=lambda s: s.start_time)

        # Assign ranks based on motion energy
        by_energy = sorted(selected, key=lambda s: s.avg_motion_energy, reverse=True)
        for rank, seg in enumerate(by_energy, start=1):
            seg.rank = rank

        logger.info(f"Selected {len(selected)}/{num_shorts} shorts")
        for seg in selected:
            msg = f"  Rank {seg.rank}: {seg.start_time:.1f}s - {seg.end_time:.1f}s (motion={seg.avg_motion_energy:.4f})"
            logger.debug(msg)

        return selected
