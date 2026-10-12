"""
Automatic Shorts Extraction from Long-Form Videos

Analyzes long-form POV driving videos and extracts the best segments
for vertical shorts (TikTok, YouTube Shorts, Reels).

v1 Strategy:
- Find high-motion segments (scenic driving, not stopped)
- Avoid detected stops (traffic lights, parking)
- Score by motion energy + variety
- Extract 15-60 second clips

Usage:
    from gomoveshift.analysis.shorts_extractor import ShortsExtractor

    extractor = ShortsExtractor()
    shorts = extractor.extract_shorts("long_form.mp4", num_shorts=5)

    for short in shorts:
        print(f"Short {short.rank}: {short.start_time:.1f}s - {short.end_time:.1f}s")
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..config import get_config
from ..models import DeadSpot, HighMotionSegment

# Re-exports for backward compatibility — all consumers import from this module
from .gps_parser import parse_gpx_file  # noqa: F401
from .motion import MotionAnalyzer
from .scenic_scorer import ScenicScorer  # noqa: F401
from .shorts_candidate_selector import ShortsCandidateSelector
from .shorts_models import (  # noqa: F401
    ExtractedShort,
    ExtractionError,
    ExtractionResult,
    GPSData,
    GPSError,
    GPSTrackpoint,
    InvalidVideoError,
    ScoringWeights,
    ShortsExtractorError,
    VideoNotFoundError,
    _haversine_m,
    footage_origin,
)
from .shorts_renderer import ShortsRenderer
from .shorts_time_mapping import VideoToGpsTimeMapper, map_from_edited_time

logger = logging.getLogger(__name__)


class ShortsExtractor:
    """
    Extracts the best short-form clips from long-form driving videos.

    Uses motion analysis to find engaging segments and avoids
    stopped periods (traffic, parking, etc.)

    Scoring is configurable via ScoringWeights:
        extractor = ShortsExtractor()
        extractor.scoring_weights.motion_energy = 0.5
        extractor.scoring_weights.duration_match = 0.3
    """

    # Default scoring weights
    DEFAULT_WEIGHTS = ScoringWeights()

    def __init__(
        self,
        min_duration: float = 30.0,
        max_duration: float = 45.0,
        target_duration: float = 38.0,
        min_gap_between_shorts: float = 60.0,
        motion_threshold_multiplier: float = 1.5,
        scoring_weights: Optional[ScoringWeights] = None,
    ):
        """
        Initialize shorts extractor.

        Args:
            min_duration: Minimum short duration in seconds
            max_duration: Maximum short duration in seconds
            target_duration: Ideal short duration in seconds
            min_gap_between_shorts: Minimum time between extracted shorts
            motion_threshold_multiplier: How much above baseline motion to require
            scoring_weights: Custom scoring weights (default: balanced)
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_duration = target_duration
        self.min_gap = min_gap_between_shorts
        self.motion_multiplier = motion_threshold_multiplier
        self.scoring_weights = scoring_weights or ScoringWeights()

        self.config = get_config()
        self.motion_analyzer = MotionAnalyzer(self.config)
        # Dead-spot avoidance for shorts must use the SENSITIVE stop_detection
        # profile (min_stop_duration 5s / motion_threshold 0.008), not the coarse
        # dead_spots preview profile (30s / 0.02). Under --skip-stops the shorts
        # derive straight from raw video, and a 5-30s stop the dead_spots profile
        # ignores would otherwise leak into a shipped clip. A separate analyzer
        # keeps find_high_motion_segments on the original threshold.
        _stop_cfg = self.config.model_copy(deep=True)
        _stop_cfg.dead_spots.motion_threshold = self.config.stop_detection.motion_threshold
        _stop_cfg.dead_spots.min_duration_seconds = int(
            self.config.stop_detection.min_stop_duration
        )
        self._stop_analyzer = MotionAnalyzer(_stop_cfg)

        # GPS integration (optional)
        self._gps_data: Optional[GPSData] = None
        self._gps_offset: float = 0.0  # Offset from video time to GPS time
        self._scenic_scorer: Optional[ScenicScorer] = None
        self._keep_segments: Optional[list[tuple[float, float]]] = None
        self._gps_duration: float = 0.0
        self._original_duration: float = 0.0
        self._footage_clock: bool = False

        # Sub-components
        self._time_mapper = VideoToGpsTimeMapper()
        # Stop removal stitches kept segments with a crossfade dissolve; the
        # mapper must subtract that per-seam overlap or edited→source time drifts.
        self._time_mapper.crossfade_duration = self.config.stop_detection.crossfade_duration
        self._renderer = ShortsRenderer()
        self._selector = ShortsCandidateSelector(
            scoring_weights=self.scoring_weights,
            min_gap=self.min_gap,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
            target_duration=self.target_duration,
            time_mapper=self._time_mapper,
            scenic_scorer=None,
            geo_dedup_meters=getattr(
                getattr(self.config, "drive", None),
                "shorts_geo_dedup_meters",
                500.0,
            ),
        )

        logger.debug(
            f"Initialized ShortsExtractor: duration={min_duration}-{max_duration}s, "
            f"target={target_duration}s, gap={min_gap_between_shorts}s"
        )

    def set_gps_data(
        self,
        gps_data: GPSData,
        gps_offset: float = 0.0,
    ) -> None:
        """
        Set GPS data for scenic scoring.

        Args:
            gps_data: Parsed GPS data
            gps_offset: Offset to add to video time to get GPS time
                       (e.g., if video starts when GPS time = 1000, offset = 1000)
        """
        self._gps_data = gps_data
        self._gps_offset = gps_offset
        self._scenic_scorer = ScenicScorer(gps_data)
        self._time_mapper.gps_offset = gps_offset
        self._footage_clock = footage_origin(gps_data) is not None
        self._time_mapper.footage_clock = self._footage_clock
        self._selector._scenic_scorer = self._scenic_scorer

        logger.info(
            f"GPS data loaded: {len(gps_data.trackpoints)} trackpoints, offset={gps_offset:.1f}s"
        )

    def set_time_mapping(
        self,
        keep_segments: Optional[list[tuple[float, float]]],
        gps_duration: float,
        original_duration: float,
    ) -> None:
        """
        Set piecewise time mapping for correct GPS lookup after stop removal.

        Args:
            keep_segments: List of (start, end) in original video time for kept segments.
            gps_duration: Total GPS track duration in seconds.
            original_duration: Original (pre-stop-removal) video duration in seconds.
        """
        self._keep_segments = keep_segments
        self._gps_duration = gps_duration
        self._original_duration = original_duration
        self._time_mapper.keep_segments = keep_segments
        self._time_mapper.gps_duration = gps_duration
        self._time_mapper.original_duration = original_duration
        logger.info(
            f"Time mapping set: {len(keep_segments) if keep_segments else 0} segments, "
            f"gps={gps_duration:.0f}s, original={original_duration:.0f}s"
        )

    @staticmethod
    def _map_from_edited_time(
        edited_time: float,
        keep_segments: Optional[list[tuple[float, float]]],
    ) -> float:
        """Inverse of _map_to_edited_time: edited video time -> original video time."""
        return map_from_edited_time(edited_time, keep_segments)

    def _video_to_gps_time(self, video_time: float) -> float:
        """Convert edited video timestamp to GPS timestamp using piecewise mapping."""
        return self._time_mapper.video_to_gps_time(video_time)

    def _validate_video_file(self, video_path: Path) -> bool:
        """
        Validate that a file is a readable video with a video track.

        Args:
            video_path: Path to video file

        Returns:
            True if valid video, False otherwise
        """
        import cv2

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return False

            # Check for video track
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cap.release()

            if fps <= 0 or frame_count <= 0:
                logger.warning(f"Video has no frames: fps={fps}, frames={frame_count}")
                return False

            if width <= 0 or height <= 0:
                logger.warning(f"Video has invalid dimensions: {width}x{height}")
                return False

            logger.debug(f"Valid video: {width}x{height} @ {fps}fps, {frame_count} frames")
            return True

        except Exception as e:
            logger.warning(f"Error validating video: {e}")
            return False

    def analyze_video(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        num_shorts: int = 5,
    ) -> tuple[List[HighMotionSegment], List[DeadSpot], float]:
        """
        Analyze video for motion and stops.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback(stage, progress)
            num_shorts: Target number of shorts (drives the candidate ceiling so
                we surface enough options for downstream dead-spot, gap, and
                geo-dedup rejection to still yield the target count).

        Returns:
            Tuple of (high_motion_segments, dead_spots, video_duration)
        """
        import cv2

        video_path = Path(video_path)
        logger.info(f"Analyzing video: {video_path.name}")

        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        logger.debug(f"Video info: {total_frames} frames @ {fps}fps = {duration:.1f}s")

        if duration <= 0:
            logger.error(f"Invalid video duration: {duration}")
            raise InvalidVideoError(f"Video has invalid duration: {video_path}")

        if progress_callback:
            progress_callback("analyzing_stops", 0.0)

        # Find dead spots (stops to avoid) at stop_detection sensitivity.
        logger.debug("Finding dead spots (stops)...")
        dead_spots = self._stop_analyzer.analyze(video_path)
        logger.info(f"Found {len(dead_spots)} dead spots (stops to avoid)")

        if progress_callback:
            progress_callback("analyzing_motion", 0.5)

        # Find high motion segments
        logger.debug("Finding high motion segments...")
        # Candidate ceiling scales with request: 14 shorts -> 42 candidates.
        # This buffer accounts for dead-spot overlap, min-gap, and geo-dedup
        # rejection that would otherwise starve selection and cause shortfall.
        top_n = max(40, num_shorts * 3)
        high_motion = self.motion_analyzer.find_high_motion_segments(
            video_path=video_path,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
            top_n=top_n,
        )
        logger.info(f"Found {len(high_motion)} high motion segment candidates")

        if progress_callback:
            progress_callback("complete", 1.0)

        return high_motion, dead_spots, duration

    def detect_scene_changes(
        self,
        video_path: Path,
        threshold: float = 0.4,
        sample_interval: float = 0.5,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> List[float]:
        """
        Detect scene changes in video using histogram comparison.

        Scene changes indicate significant visual transitions (entering towns,
        landscape changes, etc.) - useful for finding good cut points.

        Args:
            video_path: Path to video file
            threshold: Histogram difference threshold (0-1, higher = fewer detections)
            sample_interval: Seconds between sampled frames
            progress_callback: Optional callback(stage, progress)

        Returns:
            List of timestamps (seconds) where scene changes occur
        """
        import cv2

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * sample_interval)

        scene_changes: List[float] = []
        prev_hist = None
        frame_idx = 0

        if progress_callback:
            progress_callback("detecting_scenes", 0.0)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    # Convert to HSV for better color comparison
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    # Calculate histogram (H and S channels)
                    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

                    if prev_hist is not None:
                        # Compare histograms using correlation
                        # Lower correlation = more different = scene change
                        correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)

                        # Scene change if correlation drops below threshold
                        if correlation < (1 - threshold):
                            timestamp = frame_idx / fps
                            scene_changes.append(timestamp)

                    prev_hist = hist.copy()

                    if progress_callback and total_frames > 0:
                        progress_callback("detecting_scenes", frame_idx / total_frames)

                frame_idx += 1
        finally:
            cap.release()

        if progress_callback:
            progress_callback("detecting_scenes", 1.0)

        return scene_changes

    def refine_segment_boundaries(
        self,
        segment: HighMotionSegment,
        scene_changes: List[float],
        max_adjustment: float = 5.0,
    ) -> HighMotionSegment:
        """
        Refine segment boundaries to align with nearby scene changes.

        This creates cleaner cuts by snapping to natural visual transitions
        rather than arbitrary time points.

        Args:
            segment: Original segment
            scene_changes: List of scene change timestamps
            max_adjustment: Maximum seconds to adjust boundary

        Returns:
            New segment with refined boundaries
        """
        if not scene_changes:
            return segment

        # Find nearest scene change to start
        new_start = segment.start_time
        for sc in scene_changes:
            if abs(sc - segment.start_time) <= max_adjustment:
                # Prefer scene change slightly before segment start
                if sc <= segment.start_time:
                    new_start = sc
                    break
                elif new_start == segment.start_time:
                    new_start = sc

        # Find nearest scene change to end
        new_end = segment.end_time
        for sc in reversed(scene_changes):
            if abs(sc - segment.end_time) <= max_adjustment:
                # Prefer scene change slightly after segment end
                if sc >= segment.end_time:
                    new_end = sc
                    break
                elif new_end == segment.end_time:
                    new_end = sc

        # Ensure we still have a valid duration
        if new_end - new_start < self.min_duration:
            return segment  # Keep original if refinement makes it too short

        if new_end - new_start > self.max_duration:
            return segment  # Keep original if refinement makes it too long

        return HighMotionSegment(
            start_time=new_start,
            end_time=new_end,
            avg_motion_energy=segment.avg_motion_energy,
            peak_motion_energy=segment.peak_motion_energy,
            rank=segment.rank,
        )

    def refine_all_segments(
        self,
        segments: List[HighMotionSegment],
        scene_changes: List[float],
        max_adjustment: float = 5.0,
    ) -> List[HighMotionSegment]:
        """
        Refine boundaries for all segments.

        Args:
            segments: List of segments to refine
            scene_changes: List of scene change timestamps
            max_adjustment: Maximum seconds to adjust each boundary

        Returns:
            List of refined segments
        """
        return [
            self.refine_segment_boundaries(seg, scene_changes, max_adjustment) for seg in segments
        ]

    def score_frame_visual_interest(
        self,
        frame,  # numpy array (BGR)
    ) -> float:
        """
        Score a frame for visual interest.

        Combines multiple factors:
        - Color vibrancy (saturation)
        - Contrast (luminance range)
        - Edge complexity (visual detail)

        Args:
            frame: BGR image as numpy array

        Returns:
            Score from 0-100 (100 = highly interesting)
        """
        import cv2
        import numpy as np

        # Resize for faster processing
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Color vibrancy (saturation) - 0-100
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation)
        vibrancy_score = min(avg_saturation / 1.5, 100)  # Scale ~150 sat to 100

        # 2. Contrast (luminance std dev) - 0-100
        luminance_std = np.std(gray)
        contrast_score = min(luminance_std / 0.6, 100)  # Scale ~60 std to 100

        # 3. Edge complexity (Canny edge density) - 0-100
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255 * 100
        edge_score = min(edge_density * 5, 100)  # Scale ~20% density to 100

        # 4. Color variety (hue histogram spread) - 0-100
        hue = hsv[:, :, 0]
        hue_hist, _ = np.histogram(hue.flatten(), bins=18, range=(0, 180))
        hue_entropy = -np.sum(
            (hue_hist / hue_hist.sum() + 1e-10) * np.log(hue_hist / hue_hist.sum() + 1e-10)
        )
        color_variety_score = min(hue_entropy / 2.5 * 100, 100)

        # Weighted combination
        visual_interest = (
            0.35 * vibrancy_score
            + 0.25 * contrast_score
            + 0.20 * edge_score
            + 0.20 * color_variety_score
        )

        return float(visual_interest)

    def analyze_segment_visual_interest(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        num_samples: int = 5,
    ) -> float:
        """
        Analyze visual interest of a video segment.

        Samples multiple frames and returns average visual interest score.

        Args:
            video_path: Path to video file
            start_time: Segment start in seconds
            end_time: Segment end in seconds
            num_samples: Number of frames to sample

        Returns:
            Average visual interest score (0-100)
        """
        import cv2

        from .frame_range import align_cv2_frame_range

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 50.0  # Neutral score if can't open

        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = end_time - start_time

        # Sample frames evenly across segment
        sample_times = [start_time + (i + 0.5) * duration / num_samples for i in range(num_samples)]

        scores = []
        for t in sample_times:
            frame_num = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if ret:
                # Saturation, luminance spread and edge density are absolute
                # numbers off these pixels, and OpenCV decodes an untagged
                # stream as limited-range — biasing the score, and through it
                # which footage becomes a short.
                score = self.score_frame_visual_interest(align_cv2_frame_range(frame, video_path))
                scores.append(score)

        cap.release()

        return sum(scores) / len(scores) if scores else 50.0

    def score_segments_visual_interest(
        self,
        video_path: Path,
        segments: List[HighMotionSegment],
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[int, float]:
        """
        Score visual interest for multiple segments.

        Args:
            video_path: Path to video file
            segments: List of segments to score
            progress_callback: Optional progress callback

        Returns:
            Dict mapping segment index to visual interest score
        """
        scores = {}

        for i, seg in enumerate(segments):
            if progress_callback:
                progress_callback("scoring_visual", i / len(segments))

            score = self.analyze_segment_visual_interest(video_path, seg.start_time, seg.end_time)
            scores[i] = score

        if progress_callback:
            progress_callback("scoring_visual", 1.0)

        return scores

    # -- Delegation to ShortsCandidateSelector --

    def _segment_overlaps_stop(
        self,
        segment: HighMotionSegment,
        dead_spots: List[DeadSpot],
        buffer: float = 5.0,
    ) -> bool:
        """Check if segment overlaps with any dead spot (with buffer)."""
        return self._selector._segment_overlaps_stop(segment, dead_spots, buffer)

    def _segments_too_close(
        self,
        seg1: HighMotionSegment,
        seg2: HighMotionSegment,
    ) -> bool:
        """Check if two segments are too close together."""
        return self._selector._segments_too_close(seg1, seg2)

    def _calculate_spread_score(
        self,
        segment: HighMotionSegment,
        selected_segments: List[HighMotionSegment],
        video_duration: float,
    ) -> float:
        """Calculate spread score based on distance from already-selected segments."""
        return self._selector._calculate_spread_score(segment, selected_segments, video_duration)

    def _score_segment(
        self,
        segment: HighMotionSegment,
        video_duration: float,
        selected_segments: Optional[List[HighMotionSegment]] = None,
        visual_scores: Optional[Dict[int, float]] = None,
        segment_index: Optional[int] = None,
    ) -> float:
        """Score a segment for shorts suitability."""
        # Sync GPS state — tests may set _scenic_scorer directly
        self._selector._scenic_scorer = self._scenic_scorer
        self._selector._time_mapper = self._time_mapper
        self._time_mapper.gps_offset = self._gps_offset
        self._time_mapper.keep_segments = self._keep_segments
        self._time_mapper.gps_duration = self._gps_duration
        self._time_mapper.original_duration = self._original_duration
        self._time_mapper.footage_clock = self._footage_clock
        return self._selector._score_segment(
            segment, video_duration, selected_segments, visual_scores, segment_index
        )

    def select_best_shorts(
        self,
        candidates: List[HighMotionSegment],
        dead_spots: List[DeadSpot],
        video_duration: float,
        num_shorts: int = 5,
        visual_scores: Optional[Dict[int, float]] = None,
    ) -> List[HighMotionSegment]:
        """Select the best non-overlapping shorts from candidates."""
        # Sync GPS state — tests may set _scenic_scorer directly
        self._selector._scenic_scorer = self._scenic_scorer
        self._selector._time_mapper = self._time_mapper
        self._time_mapper.gps_offset = self._gps_offset
        self._time_mapper.keep_segments = self._keep_segments
        self._time_mapper.gps_duration = self._gps_duration
        self._time_mapper.original_duration = self._original_duration
        self._time_mapper.footage_clock = self._footage_clock
        return self._selector.select_best_shorts(
            candidates, dead_spots, video_duration, num_shorts, visual_scores
        )

    # -- Delegation to ShortsRenderer --

    def _render_clip(
        self,
        source: Path,
        output: Path,
        start_time: float,
        end_time: float,
    ) -> bool:
        """Render a clip using ffmpeg."""
        return self._renderer.render_clip(source, output, start_time, end_time)

    def _extract_thumbnail(
        self,
        source: Path,
        output: Path,
        timestamp: float,
    ) -> bool:
        """Extract a thumbnail frame."""
        return self._renderer.extract_thumbnail(source, output, timestamp)

    # -- Main orchestration --

    def extract_shorts(
        self,
        video_path: str,
        num_shorts: int = 5,
        output_dir: Optional[str] = None,
        render_clips: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        use_visual_scoring: Optional[bool] = None,
    ) -> ExtractionResult:
        """
        Extract shorts from a long-form video.

        Args:
            video_path: Path to source video
            num_shorts: Number of shorts to extract
            output_dir: Directory for output files (default: same as video)
            render_clips: Whether to render the clips (or just identify them)
            progress_callback: Optional callback(stage, progress)
            use_visual_scoring: Override for visual scoring (None = auto based on weight)

        Returns:
            ExtractionResult with identified/extracted shorts

        Raises:
            VideoNotFoundError: If video file doesn't exist
            InvalidVideoError: If video is corrupt or has no video track
            ExtractionError: If extraction fails
        """
        import time

        start_time = time.time()

        video_path = Path(video_path)

        # Input validation
        if not video_path.exists():
            raise VideoNotFoundError(f"Video not found: {video_path}")

        if not video_path.is_file():
            raise VideoNotFoundError(f"Path is not a file: {video_path}")

        # Check file is readable and has content
        try:
            file_size = video_path.stat().st_size
            if file_size == 0:
                raise InvalidVideoError(f"Video file is empty: {video_path}")
            logger.info(f"Processing video: {video_path.name} ({file_size / 1024 / 1024:.1f} MB)")
        except PermissionError:
            raise VideoNotFoundError(f"Permission denied reading: {video_path}")

        # Validate it's actually a video file
        if not self._validate_video_file(video_path):
            raise InvalidVideoError(f"Invalid or corrupt video file: {video_path}")

        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = video_path.parent / "shorts"

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise ExtractionError(f"Cannot create output directory: {output_dir}")

        # Analyze video
        if progress_callback:
            progress_callback("analyzing", 0.0)

        high_motion, dead_spots, duration = self.analyze_video(
            video_path,
            progress_callback=progress_callback,
            num_shorts=num_shorts,
        )

        # Compute visual scores if enabled
        visual_scores = None
        should_use_visual = (
            use_visual_scoring
            if use_visual_scoring is not None
            else (self.scoring_weights.visual_interest > 0)
        )

        if should_use_visual and high_motion:
            if progress_callback:
                progress_callback("scoring_visual", 0.6)
            visual_scores = self.score_segments_visual_interest(
                video_path,
                high_motion,
                progress_callback=progress_callback,
            )

        # Select best segments
        if progress_callback:
            progress_callback("selecting", 0.7)

        selected = self.select_best_shorts(
            candidates=high_motion,
            dead_spots=dead_spots,
            video_duration=duration,
            num_shorts=num_shorts,
            visual_scores=visual_scores,
        )

        # Create ExtractedShort objects with terrain classification
        shorts = []
        for i, seg in enumerate(selected, start=1):
            _terrain_type = None
            _avg_elev = None
            _turn_int = None
            _center_lat = None
            _center_lon = None
            if self._scenic_scorer is not None:
                _gps_start = self._video_to_gps_time(seg.start_time)
                _gps_end = self._video_to_gps_time(seg.end_time)
                _gain_rate = self._scenic_scorer.get_elevation_gain_rate(_gps_start, _gps_end)
                _turn_int = self._scenic_scorer.get_turn_intensity(_gps_start, _gps_end)
                _points = self._scenic_scorer.gps_data.get_points_in_range(_gps_start, _gps_end)
                if _points:
                    _center_lat = round(sum(p.lat for p in _points) / len(_points), 6)
                    _center_lon = round(sum(p.lon for p in _points) / len(_points), 6)
                _elevs = [p.elevation for p in _points if p.elevation is not None]
                _avg_elev = sum(_elevs) / len(_elevs) if _elevs else None
                if _avg_elev is not None:
                    from gomoveshift.pipeline.segment_context import classify_terrain

                    # Net elevation range (max-min) guards the mountain class
                    # against summed GPS jitter on a flat plateau. near_coast is
                    # left False here: no coastline signal is threaded into this
                    # path, so "coastal" is withheld rather than faked from a low
                    # elevation (see classify_terrain).
                    _net_range = (max(_elevs) - min(_elevs)) if _elevs else 0.0
                    _terrain_type = classify_terrain(
                        _gain_rate,
                        _avg_elev,
                        _turn_int,
                        net_elevation_range_m=_net_range,
                    )

            short = ExtractedShort(
                index=i,
                rank=seg.rank,
                start_time=seg.start_time,
                end_time=seg.end_time,
                duration=seg.duration,
                avg_motion_energy=seg.avg_motion_energy,
                peak_motion_energy=seg.peak_motion_energy,
                source_video=str(video_path),
                terrain_type=_terrain_type,
                avg_elevation=round(_avg_elev, 1) if _avg_elev is not None else None,
                turn_intensity=round(_turn_int, 1) if _turn_int is not None else None,
                gps_center_lat=_center_lat,
                gps_center_lon=_center_lon,
            )
            shorts.append(short)

        # Optionally render clips
        if render_clips and shorts:
            if progress_callback:
                progress_callback("rendering", 0.8)

            for short in shorts:
                clip_path = output_dir / f"{video_path.stem}_short{short.index}.mp4"
                thumb_path = output_dir / f"{video_path.stem}_short{short.index}_thumb.jpg"

                # Render clip
                if self._render_clip(video_path, clip_path, short.start_time, short.end_time):
                    short.output_path = str(clip_path)

                # Generate thumbnail
                if self._extract_thumbnail(video_path, thumb_path, short.start_time + 5):
                    short.thumbnail_path = str(thumb_path)

        processing_time = time.time() - start_time

        if progress_callback:
            progress_callback("complete", 1.0)

        # Count avoided stops
        avoided = len([seg for seg in high_motion if self._segment_overlaps_stop(seg, dead_spots)])

        return ExtractionResult(
            source_video=str(video_path),
            source_duration=duration,
            num_shorts_requested=num_shorts,
            num_shorts_found=len(shorts),
            shorts=shorts,
            dead_spots_avoided=avoided,
            processing_time=processing_time,
        )


def extract_shorts_cli(
    video_path: str,
    num_shorts: int = 5,
    output_dir: Optional[str] = None,
    render: bool = False,
) -> ExtractionResult:
    """
    CLI-friendly function for shorts extraction.

    Args:
        video_path: Path to source video
        num_shorts: Number of shorts to extract
        output_dir: Output directory (optional)
        render: Whether to render clips

    Returns:
        ExtractionResult
    """
    extractor = ShortsExtractor()

    def progress(stage: str, pct: float):
        stages = {
            "analyzing": "Analyzing video",
            "analyzing_stops": "Finding stops",
            "analyzing_motion": "Finding high motion",
            "selecting": "Selecting shorts",
            "rendering": "Rendering clips",
            "complete": "Complete",
        }
        status = stages.get(stage, stage)
        print(f"\r{status}... {pct * 100:.0f}%", end="", flush=True)
        if stage == "complete":
            print()

    result = extractor.extract_shorts(
        video_path=video_path,
        num_shorts=num_shorts,
        output_dir=output_dir,
        render_clips=render,
        progress_callback=progress,
    )

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python shorts_extractor.py <video_path> [num_shorts] [--render]")
        print("\nExample:")
        print("  python shorts_extractor.py long_form.mp4 5")
        print("  python shorts_extractor.py long_form.mp4 3 --render")
        sys.exit(1)

    video_path = sys.argv[1]
    num_shorts = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 5
    render = "--render" in sys.argv

    print(f"Extracting {num_shorts} shorts from: {video_path}")
    print(f"Render clips: {render}")
    print()

    result = extract_shorts_cli(
        video_path=video_path,
        num_shorts=num_shorts,
        render=render,
    )

    print(f"\n{'=' * 60}")
    print("EXTRACTION RESULTS")
    print(f"{'=' * 60}")
    print(f"Source: {result.source_video}")
    print(f"Duration: {timedelta(seconds=int(result.source_duration))}")
    print(f"Processing time: {result.processing_time:.1f}s")
    print(f"Dead spots avoided: {result.dead_spots_avoided}")
    print(f"\nFound {result.num_shorts_found}/{result.num_shorts_requested} shorts:")
    print()

    for short in result.shorts:
        start = str(timedelta(seconds=int(short.start_time)))
        end = str(timedelta(seconds=int(short.end_time)))
        print(f"  Short {short.index} (Rank #{short.rank}):")
        print(f"    Time: {start} - {end} ({short.duration:.1f}s)")
        print(f"    Motion: avg={short.avg_motion_energy:.4f}, peak={short.peak_motion_energy:.4f}")
        if short.output_path:
            print(f"    Output: {short.output_path}")
        print()

    # Save results
    output_path = Path(video_path).parent / f"{Path(video_path).stem}_shorts.json"
    result.save(output_path)
    print(f"Results saved to: {output_path}")
