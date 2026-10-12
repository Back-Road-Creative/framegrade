"""Tests for the weather hard-fail path in UnifiedSegmentScorer.score_file.

A scorer that silently falls back to DEFAULT_WEATHER_CLASS when
WeatherDetector returns None (degraded measurement) hides a failed
measurement. This test pins the contract: raise by default, fall
back only when allow_degraded_weather=True.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gomoveshift.analysis.unified_scorer import (
    DEFAULT_WEATHER_CLASS,
    UnifiedSegmentScorer,
)
from gomoveshift.pipeline.exceptions import WeatherMeasurementError


WEIGHTS = {
    "motion_energy": 0.20,
    "speed_variance": 0.15,
    "color_vibrancy": 0.10,
    "scene_change": 0.10,
    "sky_ratio": 0.05,
    "golden_hour": 0.10,
    "scenic_score": 0.20,
    "weather": 0.10,
}


def _frames(n: int = 10) -> list[tuple[float, np.ndarray]]:
    rng = np.random.RandomState(0)
    return [(float(i), rng.randint(0, 255, (360, 640, 3), dtype=np.uint8)) for i in range(n)]


class TestWeatherHardFail:
    def test_detector_none_raises_by_default(self):
        scorer = UnifiedSegmentScorer(WEIGHTS)
        detector = MagicMock()
        detector.detect_segment.return_value = None
        with patch.object(scorer, "_extract_frames_bgr", return_value=_frames()):
            with pytest.raises(WeatherMeasurementError, match="degraded"):
                scorer.score_file(Path("/fake/video.mp4"), weather_detector=detector)

    def test_detector_raises_propagates_as_measurement_error(self):
        scorer = UnifiedSegmentScorer(WEIGHTS)
        detector = MagicMock()
        detector.detect_segment.side_effect = TimeoutError("ffmpeg stalled")
        with patch.object(scorer, "_extract_frames_bgr", return_value=_frames()):
            with pytest.raises(WeatherMeasurementError):
                scorer.score_file(Path("/fake/video.mp4"), weather_detector=detector)

    def test_allow_degraded_weather_uses_default_class(self, caplog):
        scorer = UnifiedSegmentScorer(WEIGHTS)
        detector = MagicMock()
        detector.detect_segment.return_value = None
        with patch.object(scorer, "_extract_frames_bgr", return_value=_frames()):
            with caplog.at_level(logging.WARNING):
                scores = scorer.score_file(
                    Path("/fake/video.mp4"),
                    weather_detector=detector,
                    allow_degraded_weather=True,
                )
        assert scores is not None
        assert any(DEFAULT_WEATHER_CLASS in r.message for r in caplog.records), (
            "expected DEFAULT_WEATHER_CLASS WARNING when opting into degraded mode"
        )

    def test_explicit_weather_class_bypasses_detector(self):
        """If caller supplies an explicit weather_class, detector is not
        consulted — preserves the cached-result path used by tests and
        non-Phase-A callers."""
        scorer = UnifiedSegmentScorer(WEIGHTS)
        detector = MagicMock()
        detector.detect_segment.return_value = None
        with patch.object(scorer, "_extract_frames_bgr", return_value=_frames()):
            scores = scorer.score_file(
                Path("/fake/video.mp4"),
                weather_detector=detector,
                weather_class="sunny",
            )
        assert scores is not None
        detector.detect_segment.assert_not_called()

    def test_no_detector_no_raise(self):
        """Legacy callers that don't pass a detector still work — falls
        back to DEFAULT_WEATHER_CLASS without raising (no detector
        attempt was made, so there's no failure to hard-fail on)."""
        scorer = UnifiedSegmentScorer(WEIGHTS)
        with patch.object(scorer, "_extract_frames_bgr", return_value=_frames()):
            scores = scorer.score_file(Path("/fake/video.mp4"), weather_detector=None)
        assert scores is not None
