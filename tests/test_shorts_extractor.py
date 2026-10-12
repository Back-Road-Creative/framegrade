"""
TDD Tests for Shorts Extractor v2

v1 features (implemented):
- Motion-based segment detection
- Stop avoidance
- Basic scoring by motion energy

v2 features (to implement via TDD):
- Scene change detection (visual variety)
- Segment boundary refinement (cut on scene changes)
- Configurable scoring weights
- Visual interest scoring (colors, composition)
- Spread distribution (shorts from different parts of video)
- Resolve integration for markers/rendering
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gomoveshift.analysis.shorts_extractor import (
    ExtractedShort,
    ExtractionResult,
    InvalidVideoError,
    ScoringWeights,
    ShortsExtractor,
    VideoNotFoundError,
)
from gomoveshift.models import DeadSpot, HighMotionSegment

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_motion_data():
    """Generate sample motion energy data for a 10-minute video."""
    # Simulate 10 minutes at 5 fps sampling = 3000 data points
    timestamps = [i / 5.0 for i in range(3000)]

    # Base motion around 0.02 (typical driving)
    energies = [0.02 + np.random.normal(0, 0.005) for _ in range(3000)]

    # Add some high-motion peaks
    # Peak 1: 1:00-1:30 (indices 300-450)
    for i in range(300, 450):
        energies[i] = 0.05 + np.random.normal(0, 0.01)

    # Peak 2: 3:00-3:45 (indices 900-1125)
    for i in range(900, 1125):
        energies[i] = 0.06 + np.random.normal(0, 0.01)

    # Peak 3: 7:00-7:30 (indices 2100-2250)
    for i in range(2100, 2250):
        energies[i] = 0.045 + np.random.normal(0, 0.008)

    return list(zip(timestamps, energies))


@pytest.fixture
def sample_dead_spots():
    """Sample dead spots (stops) in a 10-minute video."""
    return [
        DeadSpot(start_time=120.0, end_time=150.0, motion_energy=0.002),  # 2:00-2:30
        DeadSpot(start_time=400.0, end_time=420.0, motion_energy=0.003),  # 6:40-7:00
    ]


@pytest.fixture
def sample_segments():
    """Sample high-motion segments."""
    return [
        HighMotionSegment(
            start_time=60.0, end_time=90.0, avg_motion_energy=0.05, peak_motion_energy=0.08
        ),
        HighMotionSegment(
            start_time=180.0, end_time=225.0, avg_motion_energy=0.06, peak_motion_energy=0.09
        ),
        HighMotionSegment(
            start_time=420.0, end_time=450.0, avg_motion_energy=0.045, peak_motion_energy=0.07
        ),
    ]


@pytest.fixture
def extractor():
    """Default shorts extractor instance."""
    return ShortsExtractor(
        min_duration=30.0,
        max_duration=45.0,
        target_duration=38.0,
        min_gap_between_shorts=60.0,
    )


# =============================================================================
# v1 Tests (existing functionality)
# =============================================================================


class TestShortsExtractorV1:
    """Tests for v1 functionality (motion-based extraction)."""

    def test_extractor_initialization(self):
        """Should initialize with configurable parameters."""
        extractor = ShortsExtractor(
            min_duration=20.0,
            max_duration=60.0,
            target_duration=40.0,
            min_gap_between_shorts=90.0,
        )
        assert extractor.min_duration == 20.0
        assert extractor.max_duration == 60.0
        assert extractor.target_duration == 40.0
        assert extractor.min_gap == 90.0

    def test_dead_spot_analyzer_uses_stop_detection_sensitivity(self):
        """The shorts dead-spot analyzer runs at the SENSITIVE stop_detection
        profile (5s / 0.008), not the coarse dead_spots preview (30s / 0.02), so a
        5-30s stop can't leak into a --skip-stops short."""
        from gomoveshift.config import get_config

        extractor = ShortsExtractor()
        cfg = get_config()
        assert extractor._stop_analyzer.motion_threshold == cfg.stop_detection.motion_threshold
        assert extractor._stop_analyzer.min_duration == int(cfg.stop_detection.min_stop_duration)
        # The high-motion analyzer keeps its own (dead_spots) threshold.
        assert extractor.motion_analyzer.motion_threshold == cfg.dead_spots.motion_threshold

    def test_segment_overlaps_stop(self, extractor, sample_dead_spots):
        """Should detect when segment overlaps with a stop."""
        # Segment that overlaps with stop at 120-150
        overlapping = HighMotionSegment(
            start_time=140.0, end_time=170.0, avg_motion_energy=0.05, peak_motion_energy=0.08
        )
        assert extractor._segment_overlaps_stop(overlapping, sample_dead_spots) is True

        # Segment that doesn't overlap
        clear = HighMotionSegment(
            start_time=200.0, end_time=230.0, avg_motion_energy=0.05, peak_motion_energy=0.08
        )
        assert extractor._segment_overlaps_stop(clear, sample_dead_spots) is False

    def test_segments_too_close(self, extractor):
        """Should detect segments that are too close together."""
        seg1 = HighMotionSegment(
            start_time=60.0, end_time=90.0, avg_motion_energy=0.05, peak_motion_energy=0.08
        )

        # Too close (within 60s gap)
        seg2_close = HighMotionSegment(
            start_time=120.0, end_time=150.0, avg_motion_energy=0.05, peak_motion_energy=0.08
        )
        assert extractor._segments_too_close(seg1, seg2_close) is True

        # Far enough apart
        seg2_far = HighMotionSegment(
            start_time=200.0, end_time=230.0, avg_motion_energy=0.05, peak_motion_energy=0.08
        )
        assert extractor._segments_too_close(seg1, seg2_far) is False

    def test_score_segment_prefers_target_duration(self, extractor):
        """Segments closer to target duration should score higher."""
        # Exactly target duration (38s)
        perfect = HighMotionSegment(
            start_time=0, end_time=38.0, avg_motion_energy=0.05, peak_motion_energy=0.08
        )

        # Too short (30s)
        short = HighMotionSegment(
            start_time=0, end_time=30.0, avg_motion_energy=0.05, peak_motion_energy=0.08
        )

        # Too long (45s)
        long = HighMotionSegment(
            start_time=0, end_time=45.0, avg_motion_energy=0.05, peak_motion_energy=0.08
        )

        score_perfect = extractor._score_segment(perfect, 600.0)
        score_short = extractor._score_segment(short, 600.0)
        score_long = extractor._score_segment(long, 600.0)

        assert score_perfect > score_short
        assert score_perfect > score_long

    def test_score_segment_prefers_high_motion(self, extractor):
        """Higher motion energy should score higher."""
        low_motion = HighMotionSegment(
            start_time=0, end_time=30.0, avg_motion_energy=0.03, peak_motion_energy=0.05
        )

        high_motion = HighMotionSegment(
            start_time=0, end_time=30.0, avg_motion_energy=0.06, peak_motion_energy=0.10
        )

        score_low = extractor._score_segment(low_motion, 600.0)
        score_high = extractor._score_segment(high_motion, 600.0)

        assert score_high > score_low

    def test_select_best_shorts_avoids_stops(self, extractor, sample_segments, sample_dead_spots):
        """Selection should avoid segments overlapping with stops."""
        # Add a segment that overlaps with a stop
        bad_segment = HighMotionSegment(
            start_time=125.0,
            end_time=155.0,  # Overlaps with stop at 120-150
            avg_motion_energy=0.10,
            peak_motion_energy=0.15,  # High energy to tempt selection
        )
        candidates = sample_segments + [bad_segment]

        selected = extractor.select_best_shorts(
            candidates=candidates,
            dead_spots=sample_dead_spots,
            video_duration=600.0,
            num_shorts=3,
        )

        # Should not select the overlapping segment
        selected_starts = [s.start_time for s in selected]
        assert 125.0 not in selected_starts

    def test_select_best_shorts_refuses_fully_stopped_candidates(self, extractor):
        """Candidates majority-inside dead spots are never shipped."""
        candidates = [
            HighMotionSegment(
                start_time=10.0,
                end_time=40.0,
                avg_motion_energy=0.05,
                peak_motion_energy=0.08,
            ),
            HighMotionSegment(
                start_time=100.0,
                end_time=130.0,
                avg_motion_energy=0.06,
                peak_motion_energy=0.09,
            ),
            HighMotionSegment(
                start_time=200.0,
                end_time=230.0,
                avg_motion_energy=0.04,
                peak_motion_energy=0.07,
            ),
        ]
        # Dead spots cover the entire video — all candidates overlap
        wall_to_wall_dead_spots = [
            DeadSpot(start_time=0.0, end_time=300.0, motion_energy=0.001, suggested_trim=True),
        ]
        selected = extractor.select_best_shorts(
            candidates=candidates,
            dead_spots=wall_to_wall_dead_spots,
            video_duration=300.0,
            num_shorts=3,
        )
        # All candidates spend 100% of their duration inside the dead spot:
        # shipping them would ship stopped footage, so nothing is selected.
        assert selected == []

    def test_select_best_shorts_respects_gap(self, extractor):
        """Selection should maintain minimum gap between shorts."""
        # Three segments, two are too close
        candidates = [
            HighMotionSegment(
                start_time=60.0, end_time=90.0, avg_motion_energy=0.05, peak_motion_energy=0.08
            ),
            HighMotionSegment(
                start_time=100.0, end_time=130.0, avg_motion_energy=0.06, peak_motion_energy=0.09
            ),  # Too close to first
            HighMotionSegment(
                start_time=300.0, end_time=330.0, avg_motion_energy=0.04, peak_motion_energy=0.07
            ),
        ]

        selected = extractor.select_best_shorts(
            candidates=candidates,
            dead_spots=[],
            video_duration=600.0,
            num_shorts=3,
        )

        # Should only select 2 (first two are too close)
        assert len(selected) <= 2

    def test_extracted_short_to_dict(self):
        """ExtractedShort should serialize to dict properly."""
        short = ExtractedShort(
            index=1,
            rank=2,
            start_time=60.5,
            end_time=90.5,
            duration=30.0,
            avg_motion_energy=0.05,
            peak_motion_energy=0.08,
            source_video="/path/to/video.mp4",
        )

        d = short.to_dict()
        assert d["index"] == 1
        assert d["rank"] == 2
        assert d["start_time"] == 60.5
        assert d["duration"] == 30.0
        assert "start_formatted" in d

    def test_extraction_result_save_load(self, tmp_path):
        """ExtractionResult should save to JSON."""
        result = ExtractionResult(
            source_video="/path/to/video.mp4",
            source_duration=600.0,
            num_shorts_requested=5,
            num_shorts_found=3,
            shorts=[
                ExtractedShort(
                    index=1,
                    rank=1,
                    start_time=60.0,
                    end_time=90.0,
                    duration=30.0,
                    avg_motion_energy=0.05,
                    peak_motion_energy=0.08,
                    source_video="/path/to/video.mp4",
                )
            ],
            dead_spots_avoided=2,
            processing_time=45.5,
        )

        output_path = tmp_path / "result.json"
        result.save(output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert data["num_shorts_found"] == 3


# =============================================================================
# v2 Tests (new functionality - TDD)
# =============================================================================


class TestSceneChangeDetection:
    """Tests for scene change detection (v2 feature)."""

    def test_detect_scene_changes_method_exists(self, extractor):
        """Should have detect_scene_changes method."""
        assert hasattr(extractor, "detect_scene_changes")
        assert callable(extractor.detect_scene_changes)

    def test_refine_segment_boundaries_snaps_to_scene_change(self, extractor):
        """Should adjust segment boundaries to align with scene changes."""
        segment = HighMotionSegment(
            start_time=60.0, end_time=90.0, avg_motion_energy=0.05, peak_motion_energy=0.08
        )

        # Scene changes near the boundaries
        scene_changes = [58.5, 62.0, 88.0, 92.5]

        refined = extractor.refine_segment_boundaries(segment, scene_changes)

        # Should snap start to 58.5 (before original) or stay at 60
        assert refined.start_time in [58.5, 60.0, 62.0]
        # Should snap end to 92.5 (after original) or stay at 90
        assert refined.end_time in [88.0, 90.0, 92.5]

    def test_refine_boundaries_no_scene_changes(self, extractor):
        """Should return original segment if no scene changes provided."""
        segment = HighMotionSegment(
            start_time=60.0, end_time=90.0, avg_motion_energy=0.05, peak_motion_energy=0.08
        )

        refined = extractor.refine_segment_boundaries(segment, [])

        assert refined.start_time == 60.0
        assert refined.end_time == 90.0

    def test_refine_boundaries_respects_max_adjustment(self, extractor):
        """Should not adjust beyond max_adjustment."""
        segment = HighMotionSegment(
            start_time=60.0, end_time=90.0, avg_motion_energy=0.05, peak_motion_energy=0.08
        )

        # Scene changes too far away
        scene_changes = [50.0, 100.0]  # 10s away from boundaries

        refined = extractor.refine_segment_boundaries(segment, scene_changes, max_adjustment=5.0)

        # Should keep original (scene changes > 5s away)
        assert refined.start_time == 60.0
        assert refined.end_time == 90.0

    def test_refine_boundaries_maintains_min_duration(self, extractor):
        """Should not refine if it would make segment too short."""
        segment = HighMotionSegment(
            start_time=60.0,
            end_time=95.0,  # 35s duration
            avg_motion_energy=0.05,
            peak_motion_energy=0.08,
        )

        # Scene changes that would shorten segment below min_duration (30s)
        scene_changes = [63.0, 80.0]  # Would make it 17s

        refined = extractor.refine_segment_boundaries(segment, scene_changes)

        # Should keep original (refinement would be too short)
        assert refined.end_time - refined.start_time >= extractor.min_duration

    def test_refine_all_segments(self, extractor):
        """Should refine all segments in a list."""
        segments = [
            HighMotionSegment(60.0, 90.0, 0.05, 0.08),
            HighMotionSegment(200.0, 230.0, 0.06, 0.09),
        ]

        scene_changes = [58.0, 92.0, 198.0, 232.0]

        refined = extractor.refine_all_segments(segments, scene_changes)

        assert len(refined) == 2
        assert all(isinstance(s, HighMotionSegment) for s in refined)


class TestVisualInterestScoring:
    """Tests for visual interest scoring (v2 feature)."""

    def test_score_frame_visual_interest_method_exists(self, extractor):
        """Should have score_frame_visual_interest method."""
        assert hasattr(extractor, "score_frame_visual_interest")
        assert callable(extractor.score_frame_visual_interest)

    def test_score_frame_visual_interest_vibrant_image(self, extractor):
        """Vibrant colorful image should score higher than gray."""
        import numpy as np

        # Create a vibrant colorful image (high saturation)
        vibrant = np.zeros((100, 100, 3), dtype=np.uint8)
        vibrant[:50, :50] = [255, 0, 0]  # Blue
        vibrant[:50, 50:] = [0, 255, 0]  # Green
        vibrant[50:, :50] = [0, 0, 255]  # Red
        vibrant[50:, 50:] = [255, 255, 0]  # Cyan

        # Create a gray monotone image (low saturation)
        gray = np.full((100, 100, 3), 128, dtype=np.uint8)

        vibrant_score = extractor.score_frame_visual_interest(vibrant)
        gray_score = extractor.score_frame_visual_interest(gray)

        assert 0 <= vibrant_score <= 100
        assert 0 <= gray_score <= 100
        assert vibrant_score > gray_score, (
            f"Vibrant ({vibrant_score}) should score higher than gray ({gray_score})"
        )

    def test_score_frame_visual_interest_high_contrast(self, extractor):
        """High contrast image should score higher than flat."""
        import numpy as np

        # High contrast image (black and white stripes)
        high_contrast = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(10):
            high_contrast[i * 10 : (i + 1) * 10, :] = 255 if i % 2 == 0 else 0

        # Flat image (all mid-gray)
        flat = np.full((100, 100, 3), 128, dtype=np.uint8)

        contrast_score = extractor.score_frame_visual_interest(high_contrast)
        flat_score = extractor.score_frame_visual_interest(flat)

        assert contrast_score > flat_score, (
            f"High contrast ({contrast_score}) should score higher than flat ({flat_score})"
        )

    def test_analyze_segment_visual_interest_method_exists(self, extractor):
        """Should have analyze_segment_visual_interest method."""
        assert hasattr(extractor, "analyze_segment_visual_interest")
        assert callable(extractor.analyze_segment_visual_interest)

    def test_score_segments_visual_interest_method_exists(self, extractor):
        """Should have score_segments_visual_interest method."""
        assert hasattr(extractor, "score_segments_visual_interest")
        assert callable(extractor.score_segments_visual_interest)


class TestVisualInterestInScoring:
    """Tests for visual interest integration in scoring weights."""

    def test_scoring_weights_has_visual_interest(self):
        """ScoringWeights should have visual_interest field."""
        weights = ScoringWeights()
        assert hasattr(weights, "visual_interest")

    def test_default_visual_interest_enabled(self):
        """Visual interest is on by default so dark/stopped footage is demoted."""
        weights = ScoringWeights()
        assert weights.visual_interest > 0.0
        weights.validate()  # defaults must still sum to 1.0

    def test_weights_with_visual_interest_sum_to_one(self):
        """Weights including visual_interest should sum to 1.0."""
        # Reduce other weights to make room for visual_interest
        weights = ScoringWeights(
            motion_energy=0.3,
            peak_motion=0.15,
            duration_match=0.1,
            position=0.05,
            spread=0.1,
            visual_interest=0.2,  # Added
            scenic_location=0.1,
        )
        weights.validate()  # Should not raise

    def test_extractor_accepts_visual_interest_weight(self):
        """Extractor should accept visual_interest in scoring weights."""
        weights = ScoringWeights(
            motion_energy=0.3,
            peak_motion=0.15,
            duration_match=0.15,
            position=0.1,
            spread=0.1,
            visual_interest=0.2,
        )
        extractor = ShortsExtractor(scoring_weights=weights)
        assert extractor.scoring_weights.visual_interest == 0.2

    def test_score_segment_uses_visual_interest_cache(self):
        """_score_segment should use cached visual interest scores."""
        weights = ScoringWeights(
            motion_energy=0.3,
            peak_motion=0.1,
            duration_match=0.2,
            position=0.1,
            spread=0.1,
            visual_interest=0.2,
        )
        extractor = ShortsExtractor(scoring_weights=weights)

        segment = HighMotionSegment(60.0, 90.0, 0.05, 0.08)

        # Without cache, visual interest contributes neutral score
        score_no_cache = extractor._score_segment(segment, 600.0)

        # With cache, visual interest uses cached value
        visual_cache = {0: 90.0}  # Segment index 0 has high visual score
        score_with_cache = extractor._score_segment(
            segment, 600.0, visual_scores=visual_cache, segment_index=0
        )

        # High visual interest should increase score
        assert score_with_cache > score_no_cache

    def test_high_visual_interest_boosts_score(self):
        """Segment with high visual interest should score higher."""
        weights = ScoringWeights(
            motion_energy=0.2,
            peak_motion=0.1,
            duration_match=0.1,
            position=0.1,
            spread=0.1,
            visual_interest=0.4,  # High visual weight
        )
        extractor = ShortsExtractor(scoring_weights=weights)

        # Same segment, different visual scores
        segment = HighMotionSegment(60.0, 90.0, 0.05, 0.08)

        low_visual_cache = {0: 20.0}
        high_visual_cache = {0: 90.0}

        score_low = extractor._score_segment(
            segment, 600.0, visual_scores=low_visual_cache, segment_index=0
        )
        score_high = extractor._score_segment(
            segment, 600.0, visual_scores=high_visual_cache, segment_index=0
        )

        assert score_high > score_low

    def test_zero_visual_weight_ignores_visual_scores(self):
        """With visual_interest=0, visual scores should not affect result."""
        weights = ScoringWeights(motion_energy=0.40, visual_interest=0.0)
        extractor = ShortsExtractor(scoring_weights=weights)

        segment = HighMotionSegment(60.0, 90.0, 0.05, 0.08)

        # Different visual caches
        score_no_cache = extractor._score_segment(segment, 600.0)
        score_with_cache = extractor._score_segment(
            segment, 600.0, visual_scores={0: 100.0}, segment_index=0
        )

        # Scores should be equal (visual interest ignored)
        assert abs(score_no_cache - score_with_cache) < 0.01

    def test_select_best_shorts_with_visual_scoring(self):
        """select_best_shorts should use visual scores when weight > 0."""
        weights = ScoringWeights(
            motion_energy=0.2,
            peak_motion=0.1,
            duration_match=0.1,
            position=0.1,
            spread=0.1,
            visual_interest=0.4,
        )
        extractor = ShortsExtractor(
            scoring_weights=weights,
            min_gap_between_shorts=30.0,
        )

        # Two segments with same motion but different visual interest
        segments = [
            HighMotionSegment(60.0, 90.0, 0.05, 0.08),  # Low visual
            HighMotionSegment(200.0, 230.0, 0.05, 0.08),  # High visual
        ]

        # Pre-computed visual scores
        visual_scores = {0: 20.0, 1: 95.0}

        selected = extractor.select_best_shorts(
            candidates=segments,
            dead_spots=[],
            video_duration=600.0,
            num_shorts=1,
            visual_scores=visual_scores,
        )

        # Should select the high visual interest segment
        assert len(selected) == 1
        assert selected[0].start_time == 200.0


class TestSpreadDistribution:
    """Tests for spread distribution (v2 feature)."""

    def test_calculate_spread_score_first_segment(self, extractor):
        """First segment should get full spread score."""
        segment = HighMotionSegment(
            start_time=100.0, end_time=130.0, avg_motion_energy=0.05, peak_motion_energy=0.08
        )

        score = extractor._calculate_spread_score(segment, [], 600.0)
        assert score == 100.0

    def test_calculate_spread_score_far_segment(self, extractor):
        """Segment far from selected should get high spread score."""
        selected = [
            HighMotionSegment(60.0, 90.0, 0.05, 0.08),  # Near start
        ]

        # Segment at end of video (far from selected)
        far_segment = HighMotionSegment(500.0, 530.0, 0.05, 0.08)
        far_score = extractor._calculate_spread_score(far_segment, selected, 600.0)

        # Segment near the selected one
        near_segment = HighMotionSegment(100.0, 130.0, 0.05, 0.08)
        near_score = extractor._calculate_spread_score(near_segment, selected, 600.0)

        assert far_score > near_score

    def test_prefer_spread_across_video(self):
        """Should prefer shorts from different parts of video when spread weight is high."""
        # All high-quality segments clustered at start
        clustered_segments = [
            HighMotionSegment(60.0, 90.0, 0.08, 0.12),  # Start - best motion
            HighMotionSegment(100.0, 130.0, 0.07, 0.11),  # Start - good motion
            HighMotionSegment(140.0, 170.0, 0.06, 0.10),  # Start - good motion
        ]

        # Lower quality segment at end
        end_segment = HighMotionSegment(500.0, 530.0, 0.04, 0.06)

        all_segments = clustered_segments + [end_segment]

        # With high spread weight, should select the end segment
        spread_extractor = ShortsExtractor(
            min_gap_between_shorts=30.0,  # Allow closer segments
            scoring_weights=ScoringWeights(
                motion_energy=0.2,
                peak_motion=0.1,
                duration_match=0.1,
                position=0.1,
                spread=0.5,  # High spread weight
            ),
        )

        selected = spread_extractor.select_best_shorts(
            candidates=all_segments,
            dead_spots=[],
            video_duration=600.0,
            num_shorts=2,
        )

        # Should have one from start and one from end
        start_times = [s.start_time for s in selected]
        has_early = any(t < 200 for t in start_times)
        has_late = any(t > 400 for t in start_times)

        assert has_early and has_late, f"Expected spread selection, got {start_times}"

    def test_low_spread_weight_prefers_quality(self):
        """With low spread weight, should prefer highest quality regardless of position."""
        # High quality at start
        high_quality = [
            HighMotionSegment(60.0, 90.0, 0.09, 0.15),
            HighMotionSegment(150.0, 180.0, 0.08, 0.14),
        ]

        # Lower quality at end
        low_quality = HighMotionSegment(500.0, 530.0, 0.03, 0.05)

        all_segments = high_quality + [low_quality]

        # With zero spread weight, should only select high quality
        no_spread_extractor = ShortsExtractor(
            min_gap_between_shorts=30.0,
            scoring_weights=ScoringWeights(
                motion_energy=0.6,
                peak_motion=0.3,
                duration_match=0.05,
                position=0.05,
                spread=0.0,  # No spread
            ),
        )

        selected = no_spread_extractor.select_best_shorts(
            candidates=all_segments,
            dead_spots=[],
            video_duration=600.0,
            num_shorts=2,
        )

        # Should select both high quality segments (both at start)
        assert all(s.start_time < 200 for s in selected)


class TestConfigurableScoring:
    """Tests for configurable scoring weights (v2 feature)."""

    def test_scoring_weights_configurable(self):
        """Should allow configuring scoring weights."""
        # Default weights
        extractor = ShortsExtractor()
        assert hasattr(extractor, "scoring_weights")
        assert extractor.scoring_weights.motion_energy == 0.33
        assert extractor.scoring_weights.peak_motion == 0.2

        # Custom weights
        custom_weights = ScoringWeights(
            motion_energy=0.6,
            peak_motion=0.1,
            duration_match=0.1,
            position=0.1,
            spread=0.1,
        )
        extractor2 = ShortsExtractor(scoring_weights=custom_weights)
        assert extractor2.scoring_weights.motion_energy == 0.6

    def test_weights_must_sum_to_one(self):
        """Weights should validate they sum to 1.0."""
        # Valid weights (all fields must be specified to sum to 1.0)
        valid = ScoringWeights(
            motion_energy=0.4,
            peak_motion=0.2,
            duration_match=0.1,
            position=0.05,
            spread=0.1,
            visual_interest=0.0,
            scenic_location=0.15,
        )
        valid.validate()  # Should not raise

        # Invalid weights
        invalid = ScoringWeights(
            motion_energy=0.5,
            peak_motion=0.5,
            duration_match=0.5,
            position=0.1,
            spread=0.1,
        )
        with pytest.raises(ValueError):
            invalid.validate()

    def test_weights_affect_scoring(self):
        """Different weights should produce different scores."""
        # Segment with high motion but poor duration match
        segment = HighMotionSegment(
            start_time=100.0,
            end_time=160.0,  # 60s - longer than target 30s
            avg_motion_energy=0.08,
            peak_motion_energy=0.12,
        )

        # Motion-heavy weights
        motion_extractor = ShortsExtractor(
            scoring_weights=ScoringWeights(
                motion_energy=0.7,
                peak_motion=0.2,
                duration_match=0.05,
                position=0.025,
                spread=0.025,
            )
        )

        # Duration-heavy weights
        duration_extractor = ShortsExtractor(
            scoring_weights=ScoringWeights(
                motion_energy=0.1, peak_motion=0.1, duration_match=0.6, position=0.1, spread=0.1
            )
        )

        motion_score = motion_extractor._score_segment(segment, 600.0)
        duration_score = duration_extractor._score_segment(segment, 600.0)

        # Motion-heavy should score this segment higher (good motion, bad duration)
        assert motion_score > duration_score

    def test_weights_affect_selection(self):
        """Different weights should produce different selections."""
        # Two segments: one with great motion, one with perfect duration
        high_motion_seg = HighMotionSegment(
            start_time=60.0,
            end_time=100.0,  # 40s - not ideal duration
            avg_motion_energy=0.09,
            peak_motion_energy=0.15,
        )
        perfect_duration_seg = HighMotionSegment(
            start_time=300.0,
            end_time=338.0,  # 38s - perfect duration (matches target)
            avg_motion_energy=0.04,
            peak_motion_energy=0.06,
        )

        candidates = [high_motion_seg, perfect_duration_seg]

        # Motion-heavy extractor should pick high motion first
        motion_extractor = ShortsExtractor(
            scoring_weights=ScoringWeights(
                motion_energy=0.8,
                peak_motion=0.1,
                duration_match=0.05,
                position=0.025,
                spread=0.025,
            )
        )
        motion_selected = motion_extractor.select_best_shorts(
            candidates=candidates, dead_spots=[], video_duration=600.0, num_shorts=1
        )
        assert motion_selected[0].start_time == 60.0  # High motion segment

        # Duration-heavy extractor should pick perfect duration first
        duration_extractor = ShortsExtractor(
            scoring_weights=ScoringWeights(
                motion_energy=0.1, peak_motion=0.05, duration_match=0.7, position=0.1, spread=0.05
            )
        )
        duration_selected = duration_extractor.select_best_shorts(
            candidates=candidates, dead_spots=[], video_duration=600.0, num_shorts=1
        )
        assert duration_selected[0].start_time == 300.0  # Perfect duration segment


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_video_not_found_raises_error(self, extractor):
        """Should raise VideoNotFoundError for missing file."""
        with pytest.raises(VideoNotFoundError) as exc_info:
            extractor.extract_shorts("/nonexistent/path/video.mp4")
        assert "not found" in str(exc_info.value).lower()

    def test_empty_video_raises_error(self, extractor, tmp_path):
        """Should raise InvalidVideoError for empty file."""
        empty_file = tmp_path / "empty.mp4"
        empty_file.write_bytes(b"")

        with pytest.raises(InvalidVideoError) as exc_info:
            extractor.extract_shorts(str(empty_file))
        assert "empty" in str(exc_info.value).lower()

    def test_non_video_file_raises_error(self, extractor, tmp_path):
        """Should raise InvalidVideoError for non-video file."""
        text_file = tmp_path / "not_a_video.mp4"
        text_file.write_text("This is not a video file")

        with pytest.raises(InvalidVideoError) as exc_info:
            extractor.extract_shorts(str(text_file))
        assert "invalid" in str(exc_info.value).lower()

    def test_directory_path_raises_error(self, extractor, tmp_path):
        """Should raise VideoNotFoundError for directory path."""
        with pytest.raises(VideoNotFoundError) as exc_info:
            extractor.extract_shorts(str(tmp_path))
        assert "not a file" in str(exc_info.value).lower()

    def test_no_candidates_returns_empty(self, extractor):
        """Should return empty list when no candidates available."""
        selected = extractor.select_best_shorts(
            candidates=[],
            dead_spots=[],
            video_duration=600.0,
            num_shorts=5,
        )
        assert selected == []

    def test_all_candidates_overlap_stops_uses_fallback(self, extractor):
        """When all candidates overlap stops, mostly-moving ones are still used."""
        candidates = [
            HighMotionSegment(60.0, 90.0, 0.05, 0.08),
            HighMotionSegment(120.0, 150.0, 0.06, 0.09),
        ]
        # Every candidate buffer-overlaps a stop, but only ~2s of its 30s (<10%)
        # is actually inside — mostly-moving fallback keeps them (tolerance).
        dead_spots = [
            DeadSpot(50.0, 62.0, 0.001),
            DeadSpot(110.0, 122.0, 0.001),
        ]

        selected = extractor.select_best_shorts(
            candidates=candidates,
            dead_spots=dead_spots,
            video_duration=600.0,
            num_shorts=5,
        )
        # Fallback: return mostly-moving candidates rather than empty
        assert len(selected) > 0
        assert len(selected) <= len(candidates)

    def test_request_more_shorts_than_available(self, extractor):
        """Should return all available when requesting more than exist."""
        candidates = [
            HighMotionSegment(60.0, 90.0, 0.05, 0.08),
            HighMotionSegment(200.0, 230.0, 0.06, 0.09),
        ]

        selected = extractor.select_best_shorts(
            candidates=candidates,
            dead_spots=[],
            video_duration=600.0,
            num_shorts=10,  # Requesting 10 but only 2 available
        )
        assert len(selected) == 2

    def test_zero_duration_video(self, extractor):
        """Should handle zero duration gracefully in scoring."""
        segment = HighMotionSegment(0.0, 30.0, 0.05, 0.08)
        # Should not raise - position calculation should handle 0 duration
        score = extractor._score_segment(segment, 0.0)
        assert score >= 0

    def test_validate_video_file_method_exists(self, extractor):
        """Should have _validate_video_file method."""
        assert hasattr(extractor, "_validate_video_file")
        assert callable(extractor._validate_video_file)


# =============================================================================
# Integration Tests
# =============================================================================


class TestShortsExtractionIntegration:
    """Integration tests for full extraction workflow."""

    @pytest.mark.slow
    def test_extract_from_real_video(self):
        """Test extraction from a real video file."""
        # This would test against a real test video
        pytest.skip("Requires test video file")

    def test_full_workflow_mocked(self, extractor, tmp_path):
        """Test full workflow with mocked video analysis."""
        # Create a fake video file to pass validation
        fake_video = tmp_path / "fake_video.mp4"
        fake_video.write_bytes(b"fake video content" * 100)

        with patch.object(extractor, "_validate_video_file", return_value=True):
            with patch.object(extractor, "analyze_video") as mock_analyze:
                mock_analyze.return_value = (
                    [
                        HighMotionSegment(60.0, 90.0, 0.05, 0.08),
                        HighMotionSegment(200.0, 230.0, 0.06, 0.09),
                        HighMotionSegment(400.0, 430.0, 0.04, 0.07),
                    ],
                    [DeadSpot(120.0, 150.0, 0.002)],
                    600.0,  # duration
                )

                result = extractor.extract_shorts(
                    video_path=str(fake_video),
                    num_shorts=3,
                    output_dir=str(tmp_path / "output"),
                    render_clips=False,
                )

                assert result.num_shorts_found == 3
                assert result.dead_spots_avoided >= 0
                assert len(result.shorts) == 3


# =============================================================================
# GPS Integration Tests
# =============================================================================


class TestGPSDataclass:
    """Tests for GPSTrackpoint and GPSData dataclasses."""

    def test_gps_trackpoint_creation(self):
        """Should create GPSTrackpoint with all fields."""
        from gomoveshift.analysis.shorts_extractor import GPSTrackpoint

        tp = GPSTrackpoint(
            lat=52.1,
            lon=-7.1,
            elevation=100.0,
            timestamp=1700000000.0,  # Unix timestamp
        )

        assert tp.lat == 52.1
        assert tp.lon == -7.1
        assert tp.elevation == 100.0
        assert tp.timestamp == 1700000000.0

    def test_gps_trackpoint_optional_elevation(self):
        """Should allow None elevation."""
        from gomoveshift.analysis.shorts_extractor import GPSTrackpoint

        tp = GPSTrackpoint(lat=52.0, lon=-7.0, elevation=None, timestamp=0.0)
        assert tp.elevation is None

    def test_gps_data_creation(self):
        """Should create GPSData with trackpoints."""
        from gomoveshift.analysis.shorts_extractor import GPSData, GPSTrackpoint

        trackpoints = [
            GPSTrackpoint(52.0, -7.0, 100.0, 0.0),
            GPSTrackpoint(52.1, -7.1, 150.0, 10.0),
            GPSTrackpoint(52.2, -7.2, 200.0, 20.0),
        ]

        gps_data = GPSData(trackpoints=trackpoints)

        assert len(gps_data.trackpoints) == 3
        assert gps_data.start_time == 0.0
        assert gps_data.end_time == 20.0

    def test_gps_data_empty_trackpoints(self):
        """Should handle empty trackpoints."""
        from gomoveshift.analysis.shorts_extractor import GPSData

        gps_data = GPSData(trackpoints=[])

        assert len(gps_data.trackpoints) == 0
        assert gps_data.start_time is None
        assert gps_data.end_time is None


class TestGPXParser:
    """Tests for GPX file parsing."""

    def test_parse_gpx_basic(self, tmp_path):
        """Should parse basic GPX file with trackpoints."""
        from gomoveshift.analysis.shorts_extractor import parse_gpx_file

        gpx_content = """<?xml version="1.0"?>
        <gpx xmlns="http://www.topografix.com/GPX/1/1">
            <trk><trkseg>
                <trkpt lat="52.1" lon="-7.1">
                    <ele>100.0</ele>
                    <time>2025-01-01T10:00:00Z</time>
                </trkpt>
                <trkpt lat="52.2" lon="-7.2">
                    <ele>50.0</ele>
                    <time>2025-01-01T10:00:02Z</time>
                </trkpt>
            </trkseg></trk>
        </gpx>"""

        gpx_file = tmp_path / "test.gpx"
        gpx_file.write_text(gpx_content)

        gps_data = parse_gpx_file(gpx_file)

        assert len(gps_data.trackpoints) == 2
        assert gps_data.trackpoints[0].lat == pytest.approx(52.1)
        assert gps_data.trackpoints[0].lon == pytest.approx(-7.1)
        assert gps_data.trackpoints[0].elevation == pytest.approx(100.0)

    def test_parse_gpx_missing_elevation(self, tmp_path):
        """Should handle trackpoints without elevation."""
        from gomoveshift.analysis.shorts_extractor import parse_gpx_file

        gpx_content = """<?xml version="1.0"?>
        <gpx xmlns="http://www.topografix.com/GPX/1/1">
            <trk><trkseg>
                <trkpt lat="52.0" lon="-7.0">
                    <time>2025-01-01T10:00:00Z</time>
                </trkpt>
            </trkseg></trk>
        </gpx>"""

        gpx_file = tmp_path / "test.gpx"
        gpx_file.write_text(gpx_content)

        gps_data = parse_gpx_file(gpx_file)

        assert len(gps_data.trackpoints) == 1
        assert gps_data.trackpoints[0].elevation is None

    def test_parse_gpx_nonexistent_file(self, tmp_path):
        """Should raise error for nonexistent file."""
        from gomoveshift.analysis.shorts_extractor import GPSError, parse_gpx_file

        with pytest.raises(GPSError):
            parse_gpx_file(tmp_path / "nonexistent.gpx")

    def test_parse_gpx_invalid_xml(self, tmp_path):
        """Should raise error for invalid XML."""
        from gomoveshift.analysis.shorts_extractor import GPSError, parse_gpx_file

        gpx_file = tmp_path / "invalid.gpx"
        gpx_file.write_text("not valid xml <><>")

        with pytest.raises(GPSError):
            parse_gpx_file(gpx_file)


class TestScenicScoring:
    """Tests for GPS-based scenic location scoring."""

    @pytest.fixture
    def climbing_track(self):
        """GPS track with significant elevation gain (climbing)."""
        from gomoveshift.analysis.shorts_extractor import GPSData, GPSTrackpoint

        # 5 minutes of climbing, gaining 200m (40m/min = steep climb)
        trackpoints = []
        for i in range(60):  # 60 points, 5 seconds apart = 5 minutes
            trackpoints.append(
                GPSTrackpoint(
                    lat=52.0 + i * 0.001,
                    lon=-7.0,
                    elevation=100.0 + i * 3.33,  # Gain ~200m over 60 points
                    timestamp=float(i * 5),
                )
            )
        return GPSData(trackpoints=trackpoints)

    @pytest.fixture
    def flat_track(self):
        """GPS track on flat terrain."""
        from gomoveshift.analysis.shorts_extractor import GPSData, GPSTrackpoint

        trackpoints = []
        for i in range(60):
            trackpoints.append(
                GPSTrackpoint(
                    lat=52.0 + i * 0.001,
                    lon=-7.0,
                    elevation=50.0,  # Constant elevation (truly flat)
                    timestamp=float(i * 5),
                )
            )
        return GPSData(trackpoints=trackpoints)

    @pytest.fixture
    def hairpin_track(self):
        """GPS track with tight turns (hairpin-like)."""

        from gomoveshift.analysis.shorts_extractor import GPSData, GPSTrackpoint

        # Zigzag pattern simulating hairpin turns
        trackpoints = []
        for i in range(60):
            # Reverse direction every 5 points
            direction = 1 if (i // 5) % 2 == 0 else -1
            trackpoints.append(
                GPSTrackpoint(
                    lat=52.0 + i * 0.0005,
                    lon=-7.0 + direction * (i % 5) * 0.001,
                    elevation=100.0 + i * 1.5,  # Also climbing
                    timestamp=float(i * 5),
                )
            )
        return GPSData(trackpoints=trackpoints)

    def test_calculate_elevation_gain_rate(self, climbing_track):
        """Should calculate elevation gain rate in m/min."""
        from gomoveshift.analysis.shorts_extractor import ScenicScorer

        scorer = ScenicScorer(climbing_track)

        # Get gain rate for first minute (0-60 seconds)
        rate = scorer.get_elevation_gain_rate(0.0, 60.0)

        # Should be ~40m/min (200m over 5 minutes)
        assert rate > 30.0  # At least 30m/min

    def test_elevation_gain_flat_terrain(self, flat_track):
        """Should return low elevation gain for flat terrain."""
        from gomoveshift.analysis.shorts_extractor import ScenicScorer

        scorer = ScenicScorer(flat_track)
        rate = scorer.get_elevation_gain_rate(0.0, 60.0)

        # Should be very low (< 5m/min)
        assert rate < 10.0

    def test_calculate_turn_intensity(self, hairpin_track):
        """Should detect high turn intensity in hairpin track."""
        from gomoveshift.analysis.shorts_extractor import ScenicScorer

        scorer = ScenicScorer(hairpin_track)
        intensity = scorer.get_turn_intensity(0.0, 60.0)

        # Higher intensity = more dramatic turns
        assert intensity > 20.0  # Arbitrary threshold for "turny"

    def test_turn_intensity_straight_road(self, flat_track):
        """Should return low turn intensity for straight road."""
        from gomoveshift.analysis.shorts_extractor import ScenicScorer

        scorer = ScenicScorer(flat_track)
        intensity = scorer.get_turn_intensity(0.0, 60.0)

        # Straight road = low turn intensity
        assert intensity < 10.0

    def test_score_segment_scenic(self, climbing_track):
        """Should score climbing segment higher for scenic-ness."""
        from gomoveshift.analysis.shorts_extractor import ScenicScorer

        scorer = ScenicScorer(climbing_track)
        score = scorer.score_segment(0.0, 60.0)

        # Score should be 0-100, climbing should score well
        assert 0 <= score <= 100
        assert score > 35  # Should be above average (weights redistributed for POI scoring)

    def test_score_segment_flat(self, flat_track):
        """Should score flat segment lower for scenic-ness."""
        from gomoveshift.analysis.shorts_extractor import ScenicScorer

        scorer = ScenicScorer(flat_track)
        score = scorer.score_segment(0.0, 60.0)

        # Flat terrain = less scenic
        assert 0 <= score <= 100
        assert score < 50  # Below average

    def test_scenic_scorer_no_data_in_range(self, climbing_track):
        """Should return neutral score when no GPS data in range."""
        from gomoveshift.analysis.shorts_extractor import ScenicScorer

        scorer = ScenicScorer(climbing_track)
        # Request time range outside the GPS data
        score = scorer.score_segment(10000.0, 10060.0)

        # Should return neutral (50)
        assert score == pytest.approx(50.0)


class TestGPSVideoCorrelation:
    """Tests for correlating video time with GPS time."""

    def test_set_gps_offset(self, extractor):
        """Should allow setting GPS time offset."""
        from gomoveshift.analysis.shorts_extractor import GPSData, GPSTrackpoint

        gps_data = GPSData([GPSTrackpoint(52.0, -7.0, 100.0, 1000.0)])

        # Video starts at GPS time 1000, so offset = 1000
        extractor.set_gps_data(gps_data, gps_offset=1000.0)

        assert extractor._gps_data is not None
        assert extractor._gps_offset == 1000.0

    def test_video_to_gps_time(self, extractor):
        """Should convert video time to GPS time."""
        from gomoveshift.analysis.shorts_extractor import GPSData, GPSTrackpoint

        gps_data = GPSData(
            [
                GPSTrackpoint(52.0, -7.0, 100.0, 1000.0),
                GPSTrackpoint(52.1, -7.1, 150.0, 1100.0),
            ]
        )

        # Video starts when GPS time = 1000
        extractor.set_gps_data(gps_data, gps_offset=1000.0)

        # Video time 30s should map to GPS time 1030
        gps_time = extractor._video_to_gps_time(30.0)
        assert gps_time == pytest.approx(1030.0)


class TestScoringWeightsWithGPS:
    """Tests for GPS integration in scoring weights."""

    def test_scoring_weights_has_scenic_field(self):
        """ScoringWeights should have scenic_location field."""
        weights = ScoringWeights()
        assert hasattr(weights, "scenic_location")
        assert weights.scenic_location == 0.15  # Enabled by default

    def test_scoring_weights_validation_with_scenic(self):
        """Weights should still validate to 1.0 with scenic enabled."""
        # Enable scenic by reducing other weights
        weights = ScoringWeights(
            motion_energy=0.3,
            peak_motion=0.2,
            duration_match=0.2,
            position=0.1,
            spread=0.1,
            visual_interest=0.0,
            scenic_location=0.1,  # New weight
        )

        # Should not raise
        weights.validate()

    def test_scoring_weights_validation_fails_over_one(self):
        """Weights validation should fail if sum > 1.0."""
        weights = ScoringWeights(
            motion_energy=0.4,
            peak_motion=0.2,
            duration_match=0.2,
            position=0.1,
            spread=0.1,
            visual_interest=0.0,
            scenic_location=0.1,  # This pushes total to 1.1
        )

        with pytest.raises(ValueError):
            weights.validate()


class TestScoringWithGPS:
    """Tests for segment scoring with GPS integration."""

    @pytest.fixture
    def scenic_extractor(self):
        """Extractor with GPS scoring enabled."""
        from gomoveshift.analysis.shorts_extractor import GPSData, GPSTrackpoint

        # Create GPS track with a scenic section (100-200s has elevation gain)
        trackpoints = []
        for i in range(60):
            t = float(i * 5)  # 0 to 295 seconds
            if 100 <= t <= 200:  # Scenic section
                ele = 100.0 + (t - 100) * 0.5  # Climbing
            else:
                ele = 100.0  # Flat
            trackpoints.append(GPSTrackpoint(52.0 + i * 0.001, -7.0, ele, t))

        gps_data = GPSData(trackpoints=trackpoints)

        extractor = ShortsExtractor(
            scoring_weights=ScoringWeights(
                motion_energy=0.3,
                peak_motion=0.2,
                duration_match=0.2,
                position=0.1,
                spread=0.1,
                visual_interest=0.0,
                scenic_location=0.1,
            )
        )
        extractor.set_gps_data(gps_data, gps_offset=0.0)

        return extractor

    def test_scenic_segment_scores_higher(self, scenic_extractor):
        """Segment in scenic area should score higher than flat area."""
        from gomoveshift.models import HighMotionSegment

        # Both segments have same motion characteristics
        scenic_seg = HighMotionSegment(120.0, 150.0, 0.05, 0.08)  # In scenic zone
        flat_seg = HighMotionSegment(20.0, 50.0, 0.05, 0.08)  # Outside scenic zone

        scenic_score = scenic_extractor._score_segment(scenic_seg, 300.0)
        flat_score = scenic_extractor._score_segment(flat_seg, 300.0)

        # Scenic segment should score higher (due to GPS boost)
        assert scenic_score > flat_score

    def test_scoring_without_gps_data(self, extractor):
        """Should work fine without GPS data (use neutral score)."""
        from gomoveshift.models import HighMotionSegment

        # Enable scenic weight but don't set GPS data
        extractor.scoring_weights.scenic_location = 0.1
        extractor.scoring_weights.motion_energy = 0.3  # Adjust to sum to 1.0

        segment = HighMotionSegment(60.0, 90.0, 0.05, 0.08)

        # Should not raise - uses neutral score
        score = extractor._score_segment(segment, 600.0)
        assert score >= 0


# =============================================================================
# YouTube Shorts Duration Compliance
# =============================================================================


class TestYouTubeShortsDurationCompliance:
    """Both extractors must enforce 30s minimum for YouTube Shorts."""

    def test_analysis_extractor_default_min_30s(self):
        """Analysis ShortsExtractor default min_duration must be >= 30s."""
        ext = ShortsExtractor()
        assert ext.min_duration >= 30.0, (
            f"Analysis extractor min_duration={ext.min_duration}s, YouTube Shorts requires >= 30s"
        )

    def test_analysis_extractor_default_target_gte_30s(self):
        """Analysis ShortsExtractor default target must be >= 30s."""
        ext = ShortsExtractor()
        assert ext.target_duration >= 30.0

    def test_pipeline_extractor_enforces_30s_floor(self):
        """PipelineShortsExtractor must enforce 30s floor even if given lower."""
        from gomoveshift.pipeline.shorts_extractor import PipelineShortsExtractor

        ext = PipelineShortsExtractor(min_duration=15.0)
        assert ext.min_duration >= 30.0, (
            f"Pipeline extractor accepted min_duration={ext.min_duration}s below 30s floor"
        )

    def test_analysis_extractor_explicit_low_value_still_works(self):
        """Callers can set min_duration < 30 for analysis (non-YouTube) use."""
        ext = ShortsExtractor(min_duration=10.0)
        assert ext.min_duration == 10.0, "Analysis extractor should accept explicit overrides"


# =============================================================================
# Terrain Classification on ExtractedShort
# =============================================================================


class TestExtractedShortTerrain:
    """ExtractedShort must carry terrain classification from GPS data."""

    def test_terrain_fields_in_to_dict(self):
        """terrain_type, avg_elevation, turn_intensity must appear in to_dict()."""
        short = ExtractedShort(
            index=1,
            rank=1,
            start_time=60.0,
            end_time=98.0,
            duration=38.0,
            avg_motion_energy=0.05,
            peak_motion_energy=0.08,
            source_video="test.mp4",
            terrain_type="coastal",
            avg_elevation=12.0,
            turn_intensity=18.5,
        )
        d = short.to_dict()
        assert d["terrain_type"] == "coastal"
        assert d["avg_elevation"] == 12.0
        assert d["turn_intensity"] == 18.5

    def test_terrain_fields_default_none(self):
        """Without terrain data, fields default to None."""
        short = ExtractedShort(
            index=1,
            rank=1,
            start_time=0.0,
            end_time=30.0,
            duration=30.0,
            avg_motion_energy=0.05,
            peak_motion_energy=0.08,
            source_video="test.mp4",
        )
        d = short.to_dict()
        assert d["terrain_type"] is None
        assert d["avg_elevation"] is None
        assert d["turn_intensity"] is None

    def test_extraction_result_load_backward_compat(self, tmp_path):
        """Loading old JSON without terrain fields must not crash."""
        old_data = {
            "source_video": "/v.mp4",
            "source_duration": 600.0,
            "num_shorts_requested": 5,
            "num_shorts_found": 1,
            "dead_spots_avoided": 0,
            "processing_time": 1.0,
            "shorts": [
                {
                    "index": 1,
                    "rank": 1,
                    "start_time": 60.0,
                    "end_time": 90.0,
                    "duration": 30.0,
                    "avg_motion_energy": 0.05,
                    "peak_motion_energy": 0.08,
                    "source_video": "/v.mp4",
                }
            ],
        }
        json_path = tmp_path / "shorts_analysis.json"
        json_path.write_text(json.dumps(old_data))

        result = ExtractionResult.load(json_path)
        assert result.shorts[0].terrain_type is None
        assert result.shorts[0].avg_elevation is None

    def test_extraction_result_load_with_terrain(self, tmp_path):
        """Loading JSON with terrain fields must populate them."""
        data = {
            "source_video": "/v.mp4",
            "source_duration": 600.0,
            "num_shorts_requested": 5,
            "num_shorts_found": 1,
            "dead_spots_avoided": 0,
            "processing_time": 1.0,
            "shorts": [
                {
                    "index": 1,
                    "rank": 1,
                    "start_time": 60.0,
                    "end_time": 90.0,
                    "duration": 30.0,
                    "avg_motion_energy": 0.05,
                    "peak_motion_energy": 0.08,
                    "source_video": "/v.mp4",
                    "terrain_type": "mountain",
                    "avg_elevation": 450.0,
                    "turn_intensity": 65.0,
                }
            ],
        }
        json_path = tmp_path / "shorts_analysis.json"
        json_path.write_text(json.dumps(data))

        result = ExtractionResult.load(json_path)
        assert result.shorts[0].terrain_type == "mountain"
        assert result.shorts[0].avg_elevation == 450.0
        assert result.shorts[0].turn_intensity == 65.0


class TestGeographicDiversity:
    """Shorts from the same location (<500m) should be deduplicated."""

    def test_gps_center_fields_in_dataclass(self):
        """ExtractedShort must have gps_center_lat/lon fields."""
        short = ExtractedShort(
            index=1,
            rank=1,
            start_time=60.0,
            end_time=98.0,
            duration=38.0,
            avg_motion_energy=0.05,
            peak_motion_energy=0.08,
            source_video="test.mp4",
            gps_center_lat=32.36,
            gps_center_lon=-80.44,
        )
        assert short.gps_center_lat == 32.36
        assert short.gps_center_lon == -80.44

    def test_gps_center_defaults_none(self):
        """Without GPS, center fields default to None."""
        short = ExtractedShort(
            index=1,
            rank=1,
            start_time=0.0,
            end_time=30.0,
            duration=30.0,
            avg_motion_energy=0.05,
            peak_motion_energy=0.08,
            source_video="test.mp4",
        )
        assert short.gps_center_lat is None
        assert short.gps_center_lon is None

    def test_gps_center_in_to_dict(self):
        """to_dict() must include GPS center fields."""
        short = ExtractedShort(
            index=1,
            rank=1,
            start_time=60.0,
            end_time=98.0,
            duration=38.0,
            avg_motion_energy=0.05,
            peak_motion_energy=0.08,
            source_video="test.mp4",
            gps_center_lat=32.36,
            gps_center_lon=-80.44,
        )
        d = short.to_dict()
        assert d["gps_center_lat"] == 32.36
        assert d["gps_center_lon"] == -80.44

    def test_load_backward_compat_gps_center(self, tmp_path):
        """Old JSON without GPS center fields must load with None."""
        old_data = {
            "source_video": "/v.mp4",
            "source_duration": 600.0,
            "num_shorts_requested": 5,
            "num_shorts_found": 1,
            "dead_spots_avoided": 0,
            "processing_time": 1.0,
            "shorts": [
                {
                    "index": 1,
                    "rank": 1,
                    "start_time": 60.0,
                    "end_time": 90.0,
                    "duration": 30.0,
                    "avg_motion_energy": 0.05,
                    "peak_motion_energy": 0.08,
                    "source_video": "/v.mp4",
                }
            ],
        }
        json_path = tmp_path / "shorts_analysis.json"
        json_path.write_text(json.dumps(old_data))

        result = ExtractionResult.load(json_path)
        assert result.shorts[0].gps_center_lat is None
        assert result.shorts[0].gps_center_lon is None

    def test_geographically_close_segments_deduplicated(self):
        """Two segments with GPS centers <500m apart — only one selected."""
        extractor = ShortsExtractor(target_duration=45.0, min_gap_between_shorts=60)

        # Two candidates at same GPS location, different times
        seg1 = HighMotionSegment(
            start_time=100,
            end_time=145,
            avg_motion_energy=0.08,
            peak_motion_energy=0.12,
        )
        seg2 = HighMotionSegment(
            start_time=300,
            end_time=345,
            avg_motion_energy=0.07,
            peak_motion_energy=0.10,
        )
        # Far enough apart in time (200s > 60s min_gap)
        assert not extractor._segments_too_close(seg1, seg2)

        # Mock GPS data: both segments map to nearly the same GPS location
        mock_gps = MagicMock()
        mock_point = MagicMock()
        mock_point.lat = 32.36
        mock_point.lon = -80.44
        mock_point.elevation = 10.0
        mock_gps.get_points_in_range.return_value = [mock_point]

        mock_scorer = MagicMock()
        mock_scorer.gps_data = mock_gps
        mock_scorer.score_segment.return_value = 50.0
        extractor._scenic_scorer = mock_scorer
        extractor._gps_offset = 0.0
        extractor._gps_duration = 600.0
        extractor._original_duration = 600.0
        extractor._keep_segments = None

        selected = extractor.select_best_shorts(
            candidates=[seg1, seg2],
            dead_spots=[],
            video_duration=600.0,
            num_shorts=2,
        )
        # Should only select 1 because they're at the same GPS location
        assert len(selected) == 1


# =============================================================================
# Candidate ceiling scales with num_shorts (shortfall fix)
# =============================================================================


class TestAnalyzeVideoCandidateCeiling:
    """analyze_video must request enough candidates to survive downstream rejection."""

    def _patch_cv_duration(self, duration: float = 600.0):
        """Make cv2.VideoCapture return a fake working video of `duration`s."""
        fake_cap = MagicMock()
        fake_cap.get.side_effect = lambda prop: {
            5: 30.0,  # cv2.CAP_PROP_FPS
            7: int(duration * 30),  # cv2.CAP_PROP_FRAME_COUNT
        }.get(prop, 0)
        fake_cap.release.return_value = None
        return fake_cap

    def test_top_n_floor_of_40_for_small_requests(self, extractor, tmp_path):
        """Requesting <=13 shorts still gets at least 40 candidates."""
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"fake" * 100)

        fake_cap = self._patch_cv_duration(600.0)
        with (
            patch("cv2.VideoCapture", return_value=fake_cap),
            patch.object(extractor._stop_analyzer, "analyze", return_value=[]),
            patch.object(
                extractor.motion_analyzer,
                "find_high_motion_segments",
                return_value=[],
            ) as mock_find,
        ):
            extractor.analyze_video(fake_video, num_shorts=5)

        _args, kwargs = mock_find.call_args
        assert kwargs["top_n"] == 40

    def test_top_n_scales_3x_with_num_shorts(self, extractor, tmp_path):
        """Requesting 14 shorts yields 42 candidates (3x ratio kicks in above floor)."""
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"fake" * 100)

        fake_cap = self._patch_cv_duration(600.0)
        with (
            patch("cv2.VideoCapture", return_value=fake_cap),
            patch.object(extractor._stop_analyzer, "analyze", return_value=[]),
            patch.object(
                extractor.motion_analyzer,
                "find_high_motion_segments",
                return_value=[],
            ) as mock_find,
        ):
            extractor.analyze_video(fake_video, num_shorts=14)

        _args, kwargs = mock_find.call_args
        assert kwargs["top_n"] == 42

    def test_top_n_large_request(self, extractor, tmp_path):
        """Requesting 20 shorts yields 60 candidates."""
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"fake" * 100)

        fake_cap = self._patch_cv_duration(600.0)
        with (
            patch("cv2.VideoCapture", return_value=fake_cap),
            patch.object(extractor._stop_analyzer, "analyze", return_value=[]),
            patch.object(
                extractor.motion_analyzer,
                "find_high_motion_segments",
                return_value=[],
            ) as mock_find,
        ):
            extractor.analyze_video(fake_video, num_shorts=20)

        _args, kwargs = mock_find.call_args
        assert kwargs["top_n"] == 60


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
