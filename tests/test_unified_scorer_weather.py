"""Weather signal in UnifiedSegmentScorer.

Construction-correctness tests: weather_class enters the composite as
the 8th signal. Mapping lives in WEATHER_SCORE_BY_CLASS as the single
source of truth.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gomoveshift.analysis.unified_scorer import (
    DEFAULT_WEATHER_CLASS,
    SIGNAL_NAMES,
    WEATHER_SCORE_BY_CLASS,
    UnifiedSegmentScorer,
)


def _weights_with_weather() -> dict:
    """Default 8-signal weights summing to 1.0."""
    return {
        "motion_energy": 0.20,
        "speed_variance": 0.15,
        "color_vibrancy": 0.10,
        "scene_change": 0.10,
        "sky_ratio": 0.05,
        "golden_hour": 0.10,
        "scenic_score": 0.20,
        "weather": 0.10,
    }


def _equal_signals(n: int, value: float = 0.5) -> dict:
    """Build a baseline signal dict with all non-weather signals at one value."""
    return {
        "motion_energy": np.full(n, value),
        "speed_variance": np.full(n, value),
        "color_vibrancy": np.full(n, value),
        "scene_change": np.full(n, value),
        "sky_ratio": np.full(n, value),
        "golden_hour": np.full(n, value),
        "scenic_score": np.full(n, value),
    }


class TestWeatherScoreByClass:
    """The mapping is the single source of truth for weather scoring."""

    def test_canonical_mapping(self):
        assert WEATHER_SCORE_BY_CLASS["sunny"] == 1.0
        assert WEATHER_SCORE_BY_CLASS["overcast"] == 0.6
        assert WEATHER_SCORE_BY_CLASS["rainy"] == 0.2

    def test_default_class_is_overcast(self):
        # Legacy code paths without weather data must not regress: default is
        # overcast (mid-band), not sunny (would inflate) or rainy (would deflate).
        assert DEFAULT_WEATHER_CLASS == "overcast"

    def test_weather_listed_in_signal_names(self):
        assert "weather" in SIGNAL_NAMES

    def test_strict_ordering(self):
        # Property the planner ordering tests rely on.
        assert (
            WEATHER_SCORE_BY_CLASS["rainy"]
            < WEATHER_SCORE_BY_CLASS["overcast"]
            < WEATHER_SCORE_BY_CLASS["sunny"]
        )


class TestCombineSignalsWithWeather:
    """Composite arithmetic includes weather as a weighted signal."""

    def test_strict_ordering_at_equal_scenic_inputs(self):
        scorer = UnifiedSegmentScorer(_weights_with_weather())
        n = 10
        ts = np.arange(n, dtype=np.float64)
        base = _equal_signals(n)

        results = {}
        for cls in ("rainy", "overcast", "sunny"):
            signals = dict(base)
            signals["weather"] = np.full(n, WEATHER_SCORE_BY_CLASS[cls])
            results[cls] = scorer.combine_signals(ts, signals)[:, 1]

        assert np.all(results["rainy"] < results["overcast"])
        assert np.all(results["overcast"] < results["sunny"])

    def test_mapping_is_single_source_of_truth(self):
        # Bypass the constants by reading them through the same import: a test
        # that hardcoded 0.2/0.6/1.0 would diverge if WEATHER_SCORE_BY_CLASS
        # were retuned. We use the constant directly so retuning propagates.
        scorer = UnifiedSegmentScorer(_weights_with_weather())
        n = 5
        ts = np.arange(n, dtype=np.float64)
        signals = _equal_signals(n)
        signals["weather"] = np.full(n, WEATHER_SCORE_BY_CLASS["sunny"])
        composite_sunny = scorer.combine_signals(ts, signals)[:, 1]

        # Composite from sunny weather alone, holding non-weather equal: same
        # contribution regardless of which constant value sunny holds.
        signals["weather"] = np.full(n, WEATHER_SCORE_BY_CLASS["overcast"])
        composite_oc = scorer.combine_signals(ts, signals)[:, 1]
        delta = composite_sunny - composite_oc
        expected_delta = _weights_with_weather()["weather"] * (
            WEATHER_SCORE_BY_CLASS["sunny"] - WEATHER_SCORE_BY_CLASS["overcast"]
        )
        np.testing.assert_allclose(delta, expected_delta, atol=1e-9)


class TestScoreFileWeatherDefault:
    """When no weather_detector is supplied, default class kicks in."""

    def _make_frames(self, n: int) -> list[tuple[float, np.ndarray]]:
        rng = np.random.RandomState(7)
        return [(float(i), rng.randint(0, 255, (360, 640, 3), dtype=np.uint8)) for i in range(n)]

    def test_legacy_call_path_no_regression(self):
        """Calling score_file without weather_detector yields the
        same composite as supplying the default 'overcast' class explicitly."""
        scorer = UnifiedSegmentScorer(_weights_with_weather())
        n = 12
        frames = self._make_frames(n)

        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            no_weather = scorer.score_file(Path("/fake/video.mp4"))

        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            overcast = scorer.score_file(
                Path("/fake/video.mp4"),
                weather_class=DEFAULT_WEATHER_CLASS,
            )

        np.testing.assert_allclose(no_weather[:, 1], overcast[:, 1], atol=1e-9)


class TestScoreFileWeatherDetector:
    """When a WeatherDetector is supplied, it is invoked once per file."""

    def _make_frames(self, n: int) -> list[tuple[float, np.ndarray]]:
        rng = np.random.RandomState(11)
        return [(float(i), rng.randint(0, 255, (360, 640, 3), dtype=np.uint8)) for i in range(n)]

    def test_detector_called_once_per_score_file(self):
        scorer = UnifiedSegmentScorer(_weights_with_weather())
        n = 20
        frames = self._make_frames(n)

        detector = MagicMock()
        # Return a sentinel WeatherResult-like object with .condition
        weather_result = MagicMock()
        weather_result.condition = MagicMock(value="sunny")
        detector.detect_segment.return_value = weather_result

        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            scorer.score_file(
                Path("/fake/video.mp4"),
                weather_detector=detector,
            )

        assert detector.detect_segment.call_count == 1

    def test_rainy_classification_lowers_composite(self):
        scorer = UnifiedSegmentScorer(_weights_with_weather())
        n = 20
        frames = self._make_frames(n)

        rainy_detector = MagicMock()
        rainy_result = MagicMock()
        rainy_result.condition = MagicMock(value="rainy")
        rainy_detector.detect_segment.return_value = rainy_result

        sunny_detector = MagicMock()
        sunny_result = MagicMock()
        sunny_result.condition = MagicMock(value="sunny")
        sunny_detector.detect_segment.return_value = sunny_result

        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            rainy_scores = scorer.score_file(
                Path("/fake/video.mp4"), weather_detector=rainy_detector
            )
        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            sunny_scores = scorer.score_file(
                Path("/fake/video.mp4"), weather_detector=sunny_detector
            )

        assert np.all(rainy_scores[:, 1] < sunny_scores[:, 1])

    def test_failed_detection_falls_back_only_when_opted_in(self):
        """Detector-None hard-fails by default. Fallback to
        DEFAULT_WEATHER_CLASS only when allow_degraded_weather=True.
        Original silent-fallback contract was a failure mode: a missing
        weather measurement must hard-fail.
        """
        from gomoveshift.pipeline.exceptions import WeatherMeasurementError

        scorer = UnifiedSegmentScorer(_weights_with_weather())
        n = 8
        frames = self._make_frames(n)

        detector = MagicMock()
        detector.detect_segment.return_value = None

        # Default path raises.
        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            with pytest.raises(WeatherMeasurementError):
                scorer.score_file(Path("/fake/video.mp4"), weather_detector=detector)

        # Opt-in path falls back to DEFAULT_WEATHER_CLASS.
        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            failed = scorer.score_file(
                Path("/fake/video.mp4"),
                weather_detector=detector,
                allow_degraded_weather=True,
            )

        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            default = scorer.score_file(
                Path("/fake/video.mp4"), weather_class=DEFAULT_WEATHER_CLASS
            )

        np.testing.assert_allclose(failed[:, 1], default[:, 1], atol=1e-9)


class TestUnknownWeatherClass:
    """Unknown class strings fall back to DEFAULT_WEATHER_CLASS, not crash."""

    def test_unknown_class_treated_as_default(self):
        scorer = UnifiedSegmentScorer(_weights_with_weather())
        n = 5

        # Path through score_file with unknown weather_class
        frames = [(float(i), np.zeros((360, 640, 3), dtype=np.uint8)) for i in range(n)]
        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            scores = scorer.score_file(Path("/fake/video.mp4"), weather_class="unknown")
        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            default = scorer.score_file(
                Path("/fake/video.mp4"), weather_class=DEFAULT_WEATHER_CLASS
            )

        # Weather column equal to the default mapping.
        from gomoveshift.analysis.unified_scorer import COL_WEATHER

        np.testing.assert_allclose(
            scores[:, COL_WEATHER],
            WEATHER_SCORE_BY_CLASS[DEFAULT_WEATHER_CLASS],
            atol=1e-9,
        )
        # And matches the composite produced by the explicit default path.
        np.testing.assert_allclose(scores[:, 1], default[:, 1], atol=1e-9)


@pytest.fixture
def weather_weights() -> dict:
    return _weights_with_weather()
