"""Tests for sign detection module."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from gomoveshift.analysis.signs import SignDetector, extract_frames_at_intervals
from gomoveshift.config import Config


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def detector(config):
    return SignDetector(config)


class TestFindSignRegions:
    """Tests for SignDetector._find_sign_regions (color/shape filtering)."""

    def test_no_signs_in_black_frame(self, detector):
        """Solid black frame yields no sign candidates."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        results = detector._find_sign_regions(frame, min_area=100, max_area=50000)
        assert results == []

    def test_green_rectangle_detected(self, detector):
        """A bright green rectangle (road sign color) is detected."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        # Draw a green rectangle (BGR: 0, 200, 0)
        cv2.rectangle(frame, (100, 50), (300, 100), (0, 200, 0), -1)

        results = detector._find_sign_regions(frame, min_area=50, max_area=100000)
        assert len(results) >= 1
        bbox, confidence = results[0]
        assert confidence > 0
        # Bbox should roughly match drawn rectangle area
        x, y, w, h = bbox
        assert w > 50
        assert h > 10

    def test_blue_rectangle_detected(self, detector):
        """A bright blue rectangle (motorway sign) is detected."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        # BGR blue in the sign detection HSV range
        cv2.rectangle(frame, (50, 30), (250, 80), (200, 100, 0), -1)

        results = detector._find_sign_regions(frame, min_area=50, max_area=100000)
        assert len(results) >= 1

    def test_area_filter_rejects_tiny_regions(self, detector):
        """Small colored regions below min_area are rejected."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        # Tiny 3x3 green dot
        cv2.rectangle(frame, (100, 100), (103, 103), (0, 200, 0), -1)

        results = detector._find_sign_regions(frame, min_area=500, max_area=100000)
        assert results == []

    def test_max_five_candidates(self, detector):
        """At most 5 candidates returned per frame."""
        frame = np.zeros((400, 800, 3), dtype=np.uint8)
        # Draw many green rectangles
        for i in range(10):
            x = i * 70
            cv2.rectangle(frame, (x, 50), (x + 60, 90), (0, 200, 0), -1)

        results = detector._find_sign_regions(frame, min_area=10, max_area=100000)
        assert len(results) <= 5


class TestBboxOverlap:
    """Tests for IoU overlap calculation."""

    def test_identical_boxes(self, detector):
        """Identical boxes have IoU = 1.0."""
        iou = detector._bbox_overlap((10, 10, 50, 50), (10, 10, 50, 50))
        assert iou == pytest.approx(1.0)

    def test_no_overlap(self, detector):
        """Non-overlapping boxes have IoU = 0.0."""
        iou = detector._bbox_overlap((0, 0, 10, 10), (100, 100, 10, 10))
        assert iou == pytest.approx(0.0)

    def test_partial_overlap(self, detector):
        """Partially overlapping boxes have 0 < IoU < 1."""
        iou = detector._bbox_overlap((0, 0, 20, 20), (10, 10, 20, 20))
        assert 0.0 < iou < 1.0

    def test_zero_area_box(self, detector):
        """Zero-area box returns 0.0 (no division error)."""
        iou = detector._bbox_overlap((0, 0, 0, 0), (10, 10, 50, 50))
        assert iou == pytest.approx(0.0)


class TestDetectCandidates:
    """Tests for detect_candidates with mocked video capture."""

    @patch("gomoveshift.analysis.signs.cv2.VideoCapture")
    def test_unopenable_video_raises(self, mock_cap_cls, detector, tmp_path):
        """Raises ValueError if video can't be opened."""
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cap_cls.return_value = cap

        with pytest.raises(ValueError, match="Could not open"):
            list(detector.detect_candidates(tmp_path / "missing.mp4"))

    @patch("gomoveshift.analysis.signs.cv2.VideoCapture")
    def test_empty_video_yields_nothing(self, mock_cap_cls, detector, tmp_path):
        """Video with zero frames yields no candidates."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        cap.read.return_value = (False, None)
        mock_cap_cls.return_value = cap

        results = list(detector.detect_candidates(tmp_path / "empty.mp4"))
        assert results == []
        cap.release.assert_called_once()


class TestExtractFramesAtIntervals:
    """Tests for the helper extract_frames_at_intervals."""

    @patch("gomoveshift.analysis.signs.cv2.VideoCapture")
    def test_unopenable_video(self, mock_cap_cls, tmp_path):
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cap_cls.return_value = cap

        with pytest.raises(ValueError, match="Could not open"):
            list(extract_frames_at_intervals(tmp_path / "bad.mp4"))
