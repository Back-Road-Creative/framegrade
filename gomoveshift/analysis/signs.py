"""Sign detection module - finds frames likely to contain road signs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

from ..config import Config
from .frame_range import align_cv2_frame_range


@dataclass
class SignCandidate:
    """A potential road sign detected in a frame."""

    frame_idx: int
    timestamp: float
    frame: np.ndarray  # The full frame
    roi: np.ndarray  # Cropped region of interest
    bbox: tuple[int, int, int, int]  # x, y, w, h in original frame
    confidence: float  # Detection confidence


class SignDetector:
    """
    Detects frames likely to contain road signs.

    Uses color filtering and shape detection rather than ML models
    for speed. The goal is to reduce the number of frames sent to OCR,
    not to perfectly detect signs.
    """

    # Road sign colors (in HSV ranges)
    SIGN_COLORS = {
        # Irish/EU road signs
        "green": ((35, 50, 50), (85, 255, 255)),  # Direction signs
        "blue": ((100, 50, 50), (130, 255, 255)),  # Motorway signs
        "brown": ((10, 50, 30), (20, 255, 200)),  # Tourist signs
        "white": ((0, 0, 200), (180, 30, 255)),  # Town signs, speed limits
        "yellow": ((20, 100, 100), (35, 255, 255)),  # Warning signs
    }

    def __init__(self, config: Config):
        self.config = config
        self.sample_divisor = config.extraction.sample_divisor
        self.roi_top = config.sign_detection.roi_top_percent
        self.roi_bottom = config.sign_detection.roi_bottom_percent
        self.min_area = config.sign_detection.min_sign_area_percent
        self.max_area = config.sign_detection.max_sign_area_percent

    def detect_candidates(
        self, video_path: Path, progress_callback: Optional[callable] = None
    ) -> Generator[SignCandidate, None, None]:
        """
        Scan video for frames containing potential road signs.

        Yields SignCandidate objects for frames that likely contain signs.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_area = width * height
        min_sign_pixels = frame_area * self.min_area
        max_sign_pixels = frame_area * self.max_area

        # Region of interest (where signs typically appear)
        roi_y_start = int(height * self.roi_top)
        roi_y_end = int(height * self.roi_bottom)

        frame_idx = 0
        last_detection_frame = -100  # Track to avoid duplicate detections

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # SIGN_COLORS below are ABSOLUTE HSV bounds, and OpenCV
                # decodes an untagged stream as limited-range. No range
                # control on VideoCapture, so correct right after the decode.
                frame = align_cv2_frame_range(frame, video_path)

                # Only process every Nth frame
                if frame_idx % self.sample_divisor == 0:
                    if progress_callback:
                        progress_callback(frame_idx, total_frames)

                    # Extract ROI
                    roi = frame[roi_y_start:roi_y_end, :]

                    # Look for sign-colored regions
                    candidates = self._find_sign_regions(roi, min_sign_pixels, max_sign_pixels)

                    for bbox, confidence in candidates:
                        # Avoid detecting same sign multiple times
                        if frame_idx - last_detection_frame < fps * 0.5:
                            # Within 0.5 seconds of last detection, skip
                            # unless this one is significantly better
                            if confidence < 0.8:
                                continue

                        # Adjust bbox to full frame coordinates
                        x, y, w, h = bbox
                        full_bbox = (x, y + roi_y_start, w, h)

                        # Extract the sign region with some padding
                        pad = 10
                        sign_y1 = max(0, full_bbox[1] - pad)
                        sign_y2 = min(height, full_bbox[1] + full_bbox[3] + pad)
                        sign_x1 = max(0, full_bbox[0] - pad)
                        sign_x2 = min(width, full_bbox[0] + full_bbox[2] + pad)
                        sign_roi = frame[sign_y1:sign_y2, sign_x1:sign_x2]

                        yield SignCandidate(
                            frame_idx=frame_idx,
                            timestamp=frame_idx / fps,
                            frame=frame.copy(),
                            roi=sign_roi.copy(),
                            bbox=full_bbox,
                            confidence=confidence,
                        )

                        last_detection_frame = frame_idx

                frame_idx += 1
        finally:
            cap.release()

    def _find_sign_regions(
        self, roi: np.ndarray, min_area: float, max_area: float
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Find rectangular regions that might be road signs.

        Returns list of (bbox, confidence) tuples.
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        results = []

        # Check each sign color
        for color_name, (lower, upper) in self.SIGN_COLORS.items():
            lower = np.array(lower)
            upper = np.array(upper)

            # Create mask for this color
            mask = cv2.inRange(hsv, lower, upper)

            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)

                # Size filter
                if area < min_area or area > max_area:
                    continue

                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Aspect ratio filter (signs are usually wider than tall, or square)
                aspect = w / h if h > 0 else 0
                if aspect < 0.3 or aspect > 5.0:
                    continue

                # Rectangularity check (how well does contour fill its bounding box)
                rect_area = w * h
                rectangularity = area / rect_area if rect_area > 0 else 0

                # Signs should be fairly rectangular
                if rectangularity < 0.4:
                    continue

                # Confidence based on rectangularity and color match
                confidence = rectangularity * 0.7 + 0.3

                # Boost confidence for typical sign colors
                if color_name in ("green", "blue", "white"):
                    confidence += 0.1

                results.append(((x, y, w, h), min(confidence, 1.0)))

        # Sort by confidence and return
        results.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate overlapping detections
        final_results = []
        for bbox, conf in results:
            # Check if this overlaps with any existing result
            dominated = False
            for existing_bbox, existing_conf in final_results:
                if self._bbox_overlap(bbox, existing_bbox) > 0.5:
                    dominated = True
                    break
            if not dominated:
                final_results.append((bbox, conf))

        return final_results[:5]  # Return top 5 candidates per frame

    def _bbox_overlap(
        self, bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]
    ) -> float:
        """Calculate IoU (intersection over union) of two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Find intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


def extract_frames_at_intervals(
    video_path: Path, interval_seconds: float = 1.0
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """
    Simple frame extraction at fixed intervals.

    Yields (frame_idx, timestamp, frame) tuples.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                yield frame_idx, frame_idx / fps, frame

            frame_idx += 1
    finally:
        cap.release()
