"""Weather signal effect on short_planner output.

The planner consumes precomputed composite scores. With weather wired
into the composite via UnifiedSegmentScorer, mixed-weather timelines
must yield rank ordering that puts sunny windows ahead of rainy ones,
all other scenic inputs equal.
"""

import numpy as np

from gomoveshift.analysis.unified_scorer import (
    WEATHER_SCORE_BY_CLASS,
    UnifiedSegmentScorer,
)
from gomoveshift.pipeline.short_planner import plan_shorts


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


def _build_mixed_weather_timeline() -> np.ndarray:
    """Build a (N,9)-shaped score array with three regions of distinct
    weather, all other signals equal at 0.5.

    Layout (each region 200s, 600s total — well above min_duration_sec):
      [0..200)   rainy
      [200..400) overcast
      [400..600) sunny
    """
    scorer = UnifiedSegmentScorer(_weights_with_weather())
    region_len = 200
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


class TestShortPlannerRespectsWeatherOrdering:
    def test_sunny_picked_before_rainy(self):
        scores = _build_mixed_weather_timeline()
        # Plan two shorts: with min_gap_sec well below region width, the
        # planner picks the highest-scoring window (sunny) first, then the
        # next highest from the remaining timeline (overcast — strictly
        # higher score than rainy at otherwise-equal scenic inputs).
        ranges = plan_shorts(
            scores,
            target_count=2,
            min_gap_sec=60.0,
            target_duration_sec=38.0,
            stride_sec=5.0,
        )
        assert len(ranges) == 2

        # Rank 1 (best) must come from the sunny region [400, 600).
        first = ranges[0]
        assert first.start_sec >= 400.0, (
            f"Best short should start in the sunny region, got start={first.start_sec}"
        )

        # Rank 2 must NOT come from the rainy region [0, 200) — overcast
        # outscores rainy and there is space for it given the gap.
        second = ranges[1]
        assert second.start_sec >= 200.0, (
            f"Second short should NOT start in the rainy region, got start={second.start_sec}"
        )

    def test_integrated_scores_descending_by_rank(self):
        scores = _build_mixed_weather_timeline()
        ranges = plan_shorts(
            scores,
            target_count=3,
            min_gap_sec=60.0,
            target_duration_sec=38.0,
            stride_sec=5.0,
        )
        for a, b in zip(ranges, ranges[1:]):
            assert a.integrated_score >= b.integrated_score


class TestShortPlannerDoesNotCallWeatherDetector:
    """The planner is a pure consumer of the precomputed composite. It
    must NOT invoke WeatherDetector itself — the upstream scorer is the
    single source of weather classification (no double-classification)."""

    def test_planner_accepts_no_detector_argument(self):
        # Signature contract: plan_shorts has no weather_detector keyword.
        import inspect

        sig = inspect.signature(plan_shorts)
        assert "weather_detector" not in sig.parameters
        assert "weather_class" not in sig.parameters
