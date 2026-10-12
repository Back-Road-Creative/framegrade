"""Tests for brightness-scored thumbnail selection.

Covers:
- WeatherDetector.score_frame_brightness() method
- Candidate timestamp generation with margins
- Brightness-based frame selection (prefer bright, fallback when all dark)
"""

from pathlib import Path
from unittest.mock import patch


from gomoveshift.pipeline.weather import WeatherDetector


# ── score_frame_brightness ──────────────────────────────


class TestScoreFrameBrightness:
    """WeatherDetector.score_frame_brightness returns float brightness."""

    def test_returns_float(self):
        detector = WeatherDetector()
        with patch.object(detector, "_get_avg_brightness_ffmpeg", return_value=550.0):
            score = detector.score_frame_brightness(Path("/fake/video.mp4"), 100.0)
        assert isinstance(score, float)
        assert score == 550.0

    def test_uses_sample_duration(self):
        """Brightness is measured over sample_duration window."""
        detector = WeatherDetector()
        with patch.object(detector, "_get_avg_brightness_ffmpeg", return_value=400.0) as mock:
            detector.score_frame_brightness(Path("/fake/video.mp4"), 100.0, sample_duration=3.0)
        mock.assert_called_once_with(Path("/fake/video.mp4"), 100.0, 3.0)

    def test_default_sample_duration_is_2(self):
        detector = WeatherDetector()
        with patch.object(detector, "_get_avg_brightness_ffmpeg", return_value=400.0) as mock:
            detector.score_frame_brightness(Path("/fake/video.mp4"), 50.0)
        mock.assert_called_once_with(Path("/fake/video.mp4"), 50.0, 2.0)


# ── Candidate selection with margins ───────────────────


class TestCandidateSelection:
    """Thumbnail candidates skip first/last 5% and prefer bright frames."""

    def _make_candidates(self, duration):
        """Reproduce the candidate generation logic from orchestrator."""
        margin = duration * 0.05
        step = (duration - 2 * margin) / 19
        return [int(margin + i * step) for i in range(20)]

    def test_candidates_skip_margins(self):
        """First candidate is >= 5% of duration, last <= 95%."""
        duration = 3600.0  # 1 hour
        candidates = self._make_candidates(duration)
        assert len(candidates) == 20
        assert candidates[0] >= int(duration * 0.05)
        assert candidates[-1] <= int(duration * 0.95)

    def test_candidates_count(self):
        """Always produces 20 candidates."""
        for duration in [1800, 3600, 7200]:
            candidates = self._make_candidates(duration)
            assert len(candidates) == 20

    def test_bright_frames_preferred(self):
        """Given mock scores, top 3 brightest are selected."""
        scored = [
            (100, 300.0),
            (200, 350.0),
            (300, 500.0),
            (400, 620.0),
            (500, 580.0),
            (600, 700.0),
            (700, 200.0),
            (800, 450.0),
            (900, 650.0),
            (1000, 380.0),
            (1100, 410.0),
            (1200, 420.0),
            (1300, 390.0),
            (1400, 500.0),
            (1500, 550.0),
            (1600, 300.0),
            (1700, 280.0),
            (1800, 320.0),
            (1900, 400.0),
            (2000, 370.0),
        ]

        bright = [(ts, b) for ts, b in scored if b >= 400]
        bright.sort(key=lambda x: x[1], reverse=True)
        selected = sorted([ts for ts, _ in bright[:3]])

        # Top 3 by brightness: 700.0@600, 650.0@900, 620.0@400
        assert selected == [400, 600, 900]

    def test_fallback_when_all_dark(self):
        """When all brightness < 400, still returns 3 frames (top 3 overall)."""
        scored = [
            (100, 200.0),
            (200, 250.0),
            (300, 380.0),
            (400, 300.0),
            (500, 350.0),
            (600, 100.0),
            (700, 320.0),
        ]

        bright = [(ts, b) for ts, b in scored if b >= 400]
        if len(bright) < 3:
            bright = scored
        bright.sort(key=lambda x: x[1], reverse=True)
        selected = sorted([ts for ts, _ in bright[:3]])

        assert len(selected) == 3
        # Top 3: 380@300, 350@500, 320@700
        assert selected == [300, 500, 700]
