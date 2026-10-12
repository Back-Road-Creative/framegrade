"""Brightness signal in UnifiedSegmentScorer (per-second visual signal).

Construction-correctness tests: brightness is the 11th column (column index
10), computed per-second from the already-decoded numpy frame, complementing
the per-episode weather signal. Where weather discriminates between
episodes, brightness discriminates within an episode.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from gomoveshift.analysis.unified_scorer import (
    COL_BRIGHTNESS,
    COL_COMPOSITE,
    COL_WEATHER,
    NUM_COLUMNS,
    SIGNAL_NAMES,
    WEATHER_SCORE_BY_CLASS,
    UnifiedSegmentScorer,
)
from gomoveshift.config import DriveConfig


def _brightness_era_weights() -> dict:
    """9-signal weights from when brightness was introduced (weather 0.07, brightness 0.07), summing to 1.0."""
    return {
        "motion_energy": 0.16,
        "speed_variance": 0.15,
        "color_vibrancy": 0.10,
        "scene_change": 0.10,
        "sky_ratio": 0.05,
        "golden_hour": 0.10,
        "scenic_score": 0.20,
        "weather": 0.07,
        "brightness": 0.07,
    }


def _equal_signals(n: int, value: float = 0.5) -> dict:
    """All non-weather, non-brightness signals at one value."""
    return {
        "motion_energy": np.full(n, value),
        "speed_variance": np.full(n, value),
        "color_vibrancy": np.full(n, value),
        "scene_change": np.full(n, value),
        "sky_ratio": np.full(n, value),
        "golden_hour": np.full(n, value),
        "scenic_score": np.full(n, value),
    }


class TestBrightnessColumnPresent:
    """Brightness lands as the 11th column (index 10) and stays in [0, 1]."""

    def test_num_columns_is_eleven(self):
        assert NUM_COLUMNS == 11

    def test_col_brightness_index(self):
        # COL_WEATHER is 9; brightness is the next column.
        assert COL_BRIGHTNESS == 10
        assert COL_BRIGHTNESS == COL_WEATHER + 1

    def test_brightness_in_signal_names(self):
        # Final entry by design; ordering in SIGNAL_NAMES matches column
        # layout (motion_energy=col2 ... brightness=col10).
        assert SIGNAL_NAMES[-1] == "brightness"
        assert "brightness" in SIGNAL_NAMES

    def test_combine_signals_shape_includes_brightness_column(self):
        scorer = UnifiedSegmentScorer(_brightness_era_weights())
        n = 8
        ts = np.arange(n, dtype=np.float64)
        signals = _equal_signals(n)
        signals["weather"] = np.full(n, WEATHER_SCORE_BY_CLASS["overcast"])
        signals["brightness"] = np.full(n, 0.5)

        result = scorer.combine_signals(ts, signals)

        assert result.shape == (n, NUM_COLUMNS)
        # Column 10 should reflect the brightness signal we passed in.
        assert np.all(result[:, COL_BRIGHTNESS] == 0.5)
        assert np.all(result[:, COL_BRIGHTNESS] >= 0.0)
        assert np.all(result[:, COL_BRIGHTNESS] <= 1.0)


class TestBrightFrameScoresHigherThanDark:
    """A bright frame's column-10 value must exceed a dark frame's."""

    def test_bright_vs_dark_column_value(self):
        scorer = UnifiedSegmentScorer(_brightness_era_weights())
        n = 2
        ts = np.arange(n, dtype=np.float64)
        signals = _equal_signals(n)
        signals["weather"] = np.full(n, WEATHER_SCORE_BY_CLASS["overcast"])
        signals["brightness"] = np.array([0.15, 0.85])  # dark, bright

        result = scorer.combine_signals(ts, signals)

        assert result[1, COL_BRIGHTNESS] > result[0, COL_BRIGHTNESS]
        # Composite for the bright row must exceed the dark row given the
        # 0.07 weight, since every other signal is identical.
        assert result[1, COL_COMPOSITE] > result[0, COL_COMPOSITE]


class TestWithinEpisodeDiscrimination:
    """Within a single episode (constant weather), brightness discriminates."""

    def test_brightness_drives_composite_within_episode(self):
        scorer = UnifiedSegmentScorer(_brightness_era_weights())
        n = 20
        ts = np.arange(n, dtype=np.float64)
        signals = _equal_signals(n)
        # Constant overcast for the whole episode.
        signals["weather"] = np.full(n, WEATHER_SCORE_BY_CLASS["overcast"])
        # First half dark (rainy-like 0.30); second half bright (sunbreak 0.75).
        brightness = np.empty(n)
        brightness[: n // 2] = 0.30
        brightness[n // 2 :] = 0.75
        signals["brightness"] = brightness

        result = scorer.combine_signals(ts, signals)

        first_half_mean = float(np.mean(result[: n // 2, COL_COMPOSITE]))
        second_half_mean = float(np.mean(result[n // 2 :, COL_COMPOSITE]))
        delta = second_half_mean - first_half_mean

        # 0.07 weight × (0.75 - 0.30) = 0.0315; require ≥ 0.025.
        assert delta >= 0.025, f"within-episode discrimination too weak: {delta:.4f}"


class TestWeightsStillSumToOne:
    """The current default weights still satisfy DriveConfig's validator."""

    def test_drive_config_validates_with_new_defaults(self):
        cfg = DriveConfig()
        # Validator raises if abs(sum - 1.0) > 0.01.
        cfg.validate_weights()

    def test_default_weights_contain_brightness(self):
        cfg = DriveConfig()
        assert "brightness" in cfg.scoring_weights
        # 0.07 at introduction, 0.14 since the dead weather weight folded in.
        assert cfg.scoring_weights["brightness"] == pytest.approx(0.14, abs=1e-9)

    def test_default_weight_sum_is_exactly_one(self):
        cfg = DriveConfig()
        total = sum(cfg.scoring_weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_weather_weight_is_gone(self):
        cfg = DriveConfig()
        # weather went 0.10 → 0.07 → removed (the classifier reports a
        # constant, so the signal ranked nothing). Mass went to brightness.
        assert "weather" not in cfg.scoring_weights

    def test_motion_energy_funded_by_four_hundredths(self):
        cfg = DriveConfig()
        # motion_energy went 0.20 → 0.16 to fund brightness.
        assert cfg.scoring_weights["motion_energy"] == pytest.approx(0.16, abs=1e-9)


class TestGpsRedistributionIncludesBrightness:
    """Brightness is a visual signal; GPS-absent redistribution scales it up."""

    def test_brightness_weight_scaled_when_gps_absent(self):
        weights = _brightness_era_weights()
        scorer = UnifiedSegmentScorer(weights)
        new_weights = scorer._redistribute_gps_weights()

        # GPS-only signals zeroed.
        assert new_weights["speed_variance"] == 0.0
        assert new_weights["scenic_score"] == 0.0

        # Brightness must be scaled up (not zeroed and not unchanged).
        assert new_weights["brightness"] > weights["brightness"]

        # Redistributed weights still sum to 1.0.
        assert sum(new_weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_brightness_proportional_scale(self):
        weights = _brightness_era_weights()
        scorer = UnifiedSegmentScorer(weights)
        new_weights = scorer._redistribute_gps_weights()

        # Visual budget pre-redistribution = 1 - (speed_variance + scenic_score)
        visual_budget = 1.0 - (weights["speed_variance"] + weights["scenic_score"])
        expected_brightness = weights["brightness"] / visual_budget
        assert new_weights["brightness"] == pytest.approx(expected_brightness, abs=1e-9)


class TestBrightnessOrthogonalToWeather:
    """Brightness and weather are independent dimensions of the composite."""

    def _composite(self, scorer, weather_score, brightness_value):
        n = 4
        ts = np.arange(n, dtype=np.float64)
        signals = _equal_signals(n)
        signals["weather"] = np.full(n, weather_score)
        signals["brightness"] = np.full(n, brightness_value)
        return float(np.mean(scorer.combine_signals(ts, signals)[:, COL_COMPOSITE]))

    def test_same_brightness_different_weather_differs_by_weather_only(self):
        scorer = UnifiedSegmentScorer(_brightness_era_weights())
        rainy = self._composite(scorer, WEATHER_SCORE_BY_CLASS["rainy"], 0.5)
        sunny = self._composite(scorer, WEATHER_SCORE_BY_CLASS["sunny"], 0.5)
        # Difference must equal the weather weight × delta in weather scores.
        expected = _brightness_era_weights()["weather"] * (
            WEATHER_SCORE_BY_CLASS["sunny"] - WEATHER_SCORE_BY_CLASS["rainy"]
        )
        assert (sunny - rainy) == pytest.approx(expected, abs=1e-9)

    def test_same_weather_different_brightness_differs_by_brightness_only(self):
        scorer = UnifiedSegmentScorer(_brightness_era_weights())
        dim = self._composite(scorer, WEATHER_SCORE_BY_CLASS["overcast"], 0.30)
        bright = self._composite(scorer, WEATHER_SCORE_BY_CLASS["overcast"], 0.75)
        expected = _brightness_era_weights()["brightness"] * (0.75 - 0.30)
        assert (bright - dim) == pytest.approx(expected, abs=1e-9)


class TestScoreFileComputesBrightnessFromFrame:
    """score_file must compute brightness inline from the decoded frame."""

    def test_score_file_populates_brightness_column(self, tmp_path):
        scorer = UnifiedSegmentScorer(_brightness_era_weights())
        # Three synthetic BGR frames at increasing brightness levels.
        # cv2.cvtColor BGR2GRAY uses 0.114*B + 0.587*G + 0.299*R.
        # Uniform-color frames mean grayscale value == that uniform value.
        h, w = 4, 4
        dark = np.full((h, w, 3), 30, dtype=np.uint8)
        mid = np.full((h, w, 3), 128, dtype=np.uint8)
        bright = np.full((h, w, 3), 230, dtype=np.uint8)
        frames = [(0.0, dark), (1.0, mid), (2.0, bright)]

        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            result = scorer.score_file(Path("dummy.mp4"))

        assert result.shape == (3, NUM_COLUMNS)
        # Column 10 monotonically increases with frame brightness.
        assert result[0, COL_BRIGHTNESS] < result[1, COL_BRIGHTNESS]
        assert result[1, COL_BRIGHTNESS] < result[2, COL_BRIGHTNESS]
        # Bright frame (uniform 230) → grayscale 230/255 ≈ 0.902.
        assert result[2, COL_BRIGHTNESS] == pytest.approx(230.0 / 255.0, abs=1e-6)
        # Dark frame (uniform 30) → 30/255 ≈ 0.118.
        assert result[0, COL_BRIGHTNESS] == pytest.approx(30.0 / 255.0, abs=1e-6)
