"""Weather signal effect on regular_lf_planner output."""

import numpy as np

from gomoveshift.analysis.unified_scorer import (
    WEATHER_SCORE_BY_CLASS,
    UnifiedSegmentScorer,
)
from gomoveshift.pipeline.regular_lf_planner import plan_regular_lfs


def _weights_with_weather() -> dict:
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


def _build_mixed_weather_lf_timeline() -> np.ndarray:
    """Three contiguous 1500s regions (rainy / overcast / sunny) so each
    can host one regular LF (target 1200s)."""
    scorer = UnifiedSegmentScorer(_weights_with_weather())
    region_len = 1500
    n = region_len * 3
    timestamps = np.arange(n, dtype=np.float64)
    base_value = 0.5

    weather_per_second = np.empty(n, dtype=np.float64)
    weather_per_second[0:region_len] = WEATHER_SCORE_BY_CLASS["rainy"]
    weather_per_second[region_len : 2 * region_len] = WEATHER_SCORE_BY_CLASS["overcast"]
    weather_per_second[2 * region_len : 3 * region_len] = WEATHER_SCORE_BY_CLASS["sunny"]

    signals = {
        "motion_energy": np.full(n, base_value),
        "speed_variance": np.full(n, base_value),
        "color_vibrancy": np.full(n, base_value),
        "scene_change": np.full(n, base_value),
        "sky_ratio": np.full(n, base_value),
        "golden_hour": np.full(n, base_value),
        "scenic_score": np.full(n, base_value),
        "weather": weather_per_second,
    }
    return scorer.combine_signals(timestamps, signals)


class TestRegularLFPlannerRespectsWeatherOrdering:
    def test_sunny_region_has_highest_integrated_score(self):
        scores = _build_mixed_weather_lf_timeline()
        ranges = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=480.0,
            max_duration_sec=1800.0,
            target_window_sec=1200.0,
            stride_sec=30.0,
            max_count=3,
        )
        assert len(ranges) >= 1

        # Returned chronologically; best score should be from sunny region.
        best = max(ranges, key=lambda r: r.integrated_score)
        assert best.start_sec >= 3000.0, (
            "Highest-scored LF should start in the sunny region "
            f"[3000, 4500). Got start={best.start_sec}"
        )

        # Among any picked ranges, the rainy region [0, 1500) must hold
        # the lowest integrated score.
        if len(ranges) == 3:
            worst = min(ranges, key=lambda r: r.integrated_score)
            assert worst.start_sec < 1500.0, (
                "Lowest-scored LF should start in the rainy region "
                f"[0, 1500). Got start={worst.start_sec}"
            )

    def test_chronological_return_order_preserved(self):
        scores = _build_mixed_weather_lf_timeline()
        ranges = plan_regular_lfs(
            scores,
            mode="best_moments",
            min_duration_sec=480.0,
            max_duration_sec=1800.0,
            target_window_sec=1200.0,
            stride_sec=30.0,
            max_count=3,
        )
        for a, b in zip(ranges, ranges[1:]):
            assert a.start_sec < b.start_sec


class TestRegularLFPlannerDoesNotCallWeatherDetector:
    def test_planner_accepts_no_detector_argument(self):
        import inspect

        sig = inspect.signature(plan_regular_lfs)
        assert "weather_detector" not in sig.parameters
        assert "weather_class" not in sig.parameters
