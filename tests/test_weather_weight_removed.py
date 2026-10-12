"""The dead 7% weather weight is gone from the composite scorer.

``WeatherDetector._classify`` has returned OVERCAST on every branch since the
brightness-classifier demotion — brightness cannot tell rain from a tunnel, or sun from
snow glare — so the weather signal fed the scorer a constant 0.6 for every
episode. A constant contributes nothing to a ranking; it just diluted the
signals that do discriminate.

The detector itself STAYS. It is a light-level reporter with other callers,
and pruning it is not part of this change.
"""

import inspect

import pytest

from gomoveshift.config import DriveConfig
from gomoveshift.pipeline.weather import WeatherCondition, WeatherDetector


class TestWeightRemoved:
    def test_no_weather_key_in_default_scoring_weights(self):
        assert "weather" not in DriveConfig().scoring_weights

    def test_brightness_absorbs_the_lighting_weight(self):
        """0.07 goes to brightness — the per-second version of the same signal."""
        assert DriveConfig().scoring_weights["brightness"] == pytest.approx(0.14, abs=1e-9)

    def test_default_weights_still_sum_to_one(self):
        cfg = DriveConfig()
        assert sum(cfg.scoring_weights.values()) == pytest.approx(1.0, abs=1e-9)
        cfg.validate_weights()  # must not raise

    def test_degraded_weather_no_longer_hard_fails_a_run(self):
        """A failed measurement of an unweighted signal must not block a drive."""
        assert DriveConfig().allow_degraded_weather is True


class TestDetectorApiSurvives:
    """Cross-lane guard: other code imports these. Removing the weight is not
    licence to prune the classifier."""

    def test_score_frame_brightness_is_still_public(self):
        assert callable(WeatherDetector.score_frame_brightness)
        params = inspect.signature(WeatherDetector.score_frame_brightness).parameters
        assert {"video_path", "timestamp", "sample_duration"} <= set(params)

    def test_classify_still_reports_a_light_level(self):
        det = WeatherDetector()
        for brightness in (10.0, 500.0, 990.0):
            condition, confidence = det._classify(brightness)
            assert condition is WeatherCondition.OVERCAST
            assert 0.0 <= confidence <= 1.0
