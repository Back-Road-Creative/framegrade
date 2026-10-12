"""Tests for visual scoring signals (color vibrancy, scene change, sky ratio, golden hour)."""

import numpy as np

from gomoveshift.analysis.visual_scorer import (
    score_color_vibrancy,
    score_golden_hour,
    score_scene_change,
    score_sky_ratio,
)


class TestColorVibrancy:
    """Color vibrancy via HSV saturation analysis."""

    def test_gray_frame_low_vibrancy(self):
        """A pure gray frame has zero saturation -> low vibrancy."""
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)  # BGR gray
        score = score_color_vibrancy(frame)
        assert 0.0 <= score <= 0.15

    def test_saturated_frame_high_vibrancy(self):
        """A bright red frame has high saturation -> high vibrancy."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # Red channel in BGR
        score = score_color_vibrancy(frame)
        assert score > 0.5

    def test_mixed_frame_moderate_vibrancy(self):
        """Half gray, half saturated -> moderate vibrancy."""
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        frame[:240, :, 2] = 255  # Top half red
        frame[:240, :, 0] = 0
        frame[:240, :, 1] = 0
        score = score_color_vibrancy(frame)
        assert 0.2 < score < 0.8

    def test_output_range(self):
        """Score always in [0, 1]."""
        for val in [0, 64, 128, 192, 255]:
            frame = np.full((100, 100, 3), val, dtype=np.uint8)
            score = score_color_vibrancy(frame)
            assert 0.0 <= score <= 1.0


class TestSceneChange:
    """Scene change magnitude via histogram distance."""

    def test_identical_histograms_no_change(self):
        """Same histogram -> 0 scene change."""
        hist = np.ones(256, dtype=np.float32) / 256
        score = score_scene_change(hist, hist)
        assert score < 0.05

    def test_opposite_histograms_max_change(self):
        """Completely different histograms -> high score."""
        hist_a = np.zeros(256, dtype=np.float32)
        hist_a[0] = 1.0
        hist_b = np.zeros(256, dtype=np.float32)
        hist_b[255] = 1.0
        score = score_scene_change(hist_a, hist_b)
        assert score > 0.5

    def test_similar_histograms_low_change(self):
        """Slightly shifted histogram -> low score."""
        hist_a = np.zeros(256, dtype=np.float32)
        hist_a[100:130] = 1.0 / 30
        hist_b = np.zeros(256, dtype=np.float32)
        hist_b[105:135] = 1.0 / 30
        score = score_scene_change(hist_a, hist_b)
        assert score < 0.3

    def test_output_range(self):
        """Score always in [0, 1]."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            h1 = rng.rand(256).astype(np.float32)
            h1 /= h1.sum()
            h2 = rng.rand(256).astype(np.float32)
            h2 /= h2.sum()
            score = score_scene_change(h1, h2)
            assert 0.0 <= score <= 1.0


class TestSkyRatio:
    """Sky ratio / openness from upper-frame brightness analysis."""

    def test_bright_sky_high_ratio(self):
        """Bright upper region -> high sky ratio."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:160, :] = 200  # Top third bright (sky)
        frame[160:, :] = 50  # Bottom dark (ground)
        score = score_sky_ratio(frame)
        assert score > 0.5

    def test_dark_overhead_low_ratio(self):
        """Dark upper region (tunnel, dense trees) -> low sky ratio."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:160, :] = 20  # Top third very dark
        frame[160:, :] = 100  # Bottom lighter
        score = score_sky_ratio(frame)
        assert score < 0.3

    def test_uniform_frame_moderate(self):
        """Uniform brightness -> moderate score."""
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        score = score_sky_ratio(frame)
        assert 0.2 < score < 0.8

    def test_output_range(self):
        """Score always in [0, 1]."""
        for val in [0, 64, 128, 255]:
            frame = np.full((100, 100, 3), val, dtype=np.uint8)
            score = score_sky_ratio(frame)
            assert 0.0 <= score <= 1.0


class TestGoldenHour:
    """Golden hour detection via warm color temperature analysis."""

    def test_warm_tones_high_score(self):
        """Warm orange/amber frame -> high golden hour score."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 2] = 220  # High red (BGR order)
        frame[:, :, 1] = 160  # Medium green
        frame[:, :, 0] = 60  # Low blue
        score = score_golden_hour(frame)
        assert score > 0.5

    def test_cool_blue_low_score(self):
        """Cool blue frame -> low golden hour score."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 220  # High blue
        frame[:, :, 1] = 180  # Medium green
        frame[:, :, 2] = 80  # Low red
        score = score_golden_hour(frame)
        assert score < 0.3

    def test_gray_overcast_low_score(self):
        """Neutral gray frame -> low golden hour score."""
        frame = np.full((480, 640, 3), 150, dtype=np.uint8)
        score = score_golden_hour(frame)
        assert score < 0.3

    def test_output_range(self):
        """Score always in [0, 1]."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            frame = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            score = score_golden_hour(frame)
            assert 0.0 <= score <= 1.0
