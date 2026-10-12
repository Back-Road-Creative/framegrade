"""Main analysis pipeline that orchestrates all analysis components."""

from pathlib import Path
from typing import Callable, Optional

import cv2

from ..config import Config
from ..models import CutPoint, DetectedLocation, VideoAnalysis
from .motion import MotionAnalyzer
from .ocr import LocationDeduplicator, OCREngine
from .signs import SignDetector, extract_frames_at_intervals


class AnalysisPipeline:
    """
    Main pipeline that coordinates video analysis.

    Runs motion analysis and sign detection/OCR, then generates
    suggested cut points.
    """

    def __init__(self, config: Config):
        self.config = config
        self.motion_analyzer = MotionAnalyzer(config)
        self.sign_detector = SignDetector(config)

        self.ocr_engine = OCREngine(config)
        self.deduplicator = LocationDeduplicator()

    def analyze(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        save_frames: bool = True,
    ) -> VideoAnalysis:
        """
        Run full analysis on a video file.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback(stage, progress) for UI updates
            save_frames: Whether to save frames with detected signs

        Returns:
            VideoAnalysis object with all detected information
        """
        video_path = Path(video_path)

        # Get video metadata
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if fps <= 0:
            raise ValueError(f"Cannot read FPS from {video_path}")
        duration = total_frames / fps

        # Create output directory for frames
        frames_dir = None
        if save_frames:
            frames_dir = Path(self.config.paths.temp_folder) / video_path.stem / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Motion analysis
        if progress_callback:
            progress_callback("motion_analysis", 0.0)

        dead_spots = self.motion_analyzer.analyze(video_path)

        if progress_callback:
            progress_callback("motion_analysis", 1.0)

        # Stage 2: Sign detection and OCR
        if progress_callback:
            progress_callback("sign_detection", 0.0)

        locations = []
        processed_frames = 0

        # Sample frames at fixed intervals and OCR them
        for frame_idx, timestamp, frame in extract_frames_at_intervals(
            video_path,
            interval_seconds=0.5,  # 2 frames per second
        ):
            processed_frames += 1

            if progress_callback and processed_frames % 10 == 0:
                progress = timestamp / duration
                progress_callback("sign_detection", progress)

            # Save frame if requested
            frame_path = None
            if frames_dir:
                frame_path = str(frames_dir / f"frame_{frame_idx:06d}.jpg")

            # Run OCR on the frame
            location = self.ocr_engine.process_frame(frame, timestamp, frame_path)

            if location:
                # Save the frame if we found something
                if frames_dir and location.confidence_score >= 0.5:
                    cv2.imwrite(frame_path, frame)
                    location.frame_path = frame_path

                locations.append(location)

        if progress_callback:
            progress_callback("sign_detection", 1.0)

        # Deduplicate locations
        locations = self.deduplicator.deduplicate(locations)

        # Stage 3: Generate suggested cut points
        if progress_callback:
            progress_callback("segmentation", 0.0)

        suggested_cuts = self._generate_cut_points(duration, locations, dead_spots)

        if progress_callback:
            progress_callback("segmentation", 1.0)

        # Build analysis result
        analysis = VideoAnalysis(
            video_path=str(video_path),
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            locations=locations,
            dead_spots=dead_spots,
            suggested_cuts=suggested_cuts,
            processing_stats={
                "frames_processed": processed_frames,
                "locations_detected": len(locations),
                "dead_spots_detected": len(dead_spots),
            },
        )

        return analysis

    def _generate_cut_points(
        self,
        duration: float,
        locations: list[DetectedLocation],
        dead_spots: list,
    ) -> list[CutPoint]:
        """Generate suggested cut points based on analysis."""
        cuts = []
        target_duration = self.config.segmentation.target_duration_minutes * 60
        min_duration = self.config.segmentation.min_duration_minutes * 60
        max_duration = self.config.segmentation.max_duration_minutes * 60

        # If video is shorter than minimum, no cuts needed
        if duration <= max_duration:
            return []

        # Collect candidate cut points
        candidates = []

        # Location changes are good cut points
        for loc in locations:
            if loc.location_type.value in ("town_entry", "county_boundary"):
                candidates.append(
                    CutPoint(
                        timestamp=loc.timestamp,
                        reason="location_change",
                        priority=1,
                        location=loc,
                    )
                )

        # End of dead spots are natural cut points
        for ds in dead_spots:
            candidates.append(
                CutPoint(
                    timestamp=ds.end_time,
                    reason="dead_spot_end",
                    priority=2,
                )
            )

        # Add time-based candidates every target_duration
        time_point = target_duration
        while time_point < duration - min_duration:
            candidates.append(
                CutPoint(
                    timestamp=time_point,
                    reason="duration_target",
                    priority=3,
                )
            )
            time_point += target_duration

        # Sort candidates by timestamp
        candidates.sort(key=lambda x: x.timestamp)

        # Select best cut points to achieve target segment lengths
        last_cut = 0.0

        while last_cut + min_duration < duration:
            # Find candidates in the acceptable range
            range_start = last_cut + min_duration
            range_end = last_cut + max_duration

            # If we're near the end, don't cut
            if duration - range_start < min_duration:
                break

            # Find best candidate in range
            best_candidate = None
            for c in candidates:
                if range_start <= c.timestamp <= range_end:
                    if best_candidate is None or c.priority < best_candidate.priority:
                        best_candidate = c
                    elif c.priority == best_candidate.priority and abs(
                        c.timestamp - (last_cut + target_duration)
                    ) < abs(best_candidate.timestamp - (last_cut + target_duration)):
                        # Same priority, prefer closer to target duration
                        best_candidate = c

            if best_candidate:
                cuts.append(best_candidate)
                last_cut = best_candidate.timestamp
            else:
                # No good candidate, cut at target duration
                cuts.append(
                    CutPoint(
                        timestamp=last_cut + target_duration,
                        reason="duration_target",
                        priority=3,
                    )
                )
                last_cut += target_duration

        return cuts

    def quick_scan(self, video_path: Path) -> dict:
        """
        Quick scan of video for basic info without full analysis.

        Returns dict with duration, fps, resolution, and rough
        dead spot estimates.
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if fps <= 0:
            raise ValueError(f"Cannot read FPS from {video_path}")
        duration = total_frames / fps

        return {
            "video_path": str(video_path),
            "duration": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
        }
