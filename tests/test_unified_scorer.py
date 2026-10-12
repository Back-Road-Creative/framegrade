"""Tests for ScenicScorer.get_speed_variance and UnifiedSegmentScorer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gomoveshift.analysis.shorts_extractor import (
    GPSData,
    GPSTrackpoint,
    ScenicScorer,
)
from gomoveshift.analysis.unified_scorer import NUM_COLUMNS, UnifiedSegmentScorer


# ── ScenicScorer.get_speed_variance ──────────────────────────────────


class TestSpeedVariance:
    """GPS speed variance scoring."""

    def _make_gps(self, speeds_kmh: list[float], interval: float = 1.0) -> GPSData:
        """Create GPS track from speed list. Each speed held for `interval` seconds."""
        points = []
        lat, lon = 51.77, -9.98
        t = 1000.0  # arbitrary start timestamp

        for speed in speeds_kmh:
            points.append(GPSTrackpoint(lat, lon, 100.0, t))
            # Move lat based on speed (rough: 1 deg lat ~111km)
            distance_km = speed * (interval / 3600.0)
            lat += distance_km / 111.0
            t += interval

        # Add final point
        points.append(GPSTrackpoint(lat, lon, 100.0, t))
        return GPSData(trackpoints=points)

    def test_constant_speed_low_variance(self):
        """Constant highway speed -> near-zero variance."""
        gps = self._make_gps([100.0] * 30)  # 100 km/h constant
        scorer = ScenicScorer(gps)
        start = gps.trackpoints[0].timestamp
        end = gps.trackpoints[-1].timestamp
        variance = scorer.get_speed_variance(start, end)
        assert variance < 5.0  # Very low

    def test_varying_speed_high_variance(self):
        """Speed changes between 20 and 80 -> high variance."""
        speeds = [20, 80, 30, 70, 25, 75, 20, 80, 35, 65] * 3
        gps = self._make_gps(speeds)
        scorer = ScenicScorer(gps)
        start = gps.trackpoints[0].timestamp
        end = gps.trackpoints[-1].timestamp
        variance = scorer.get_speed_variance(start, end)
        assert variance > 15.0

    def test_too_few_points_returns_zero(self):
        """Single point -> 0 variance."""
        gps = GPSData(trackpoints=[GPSTrackpoint(51.77, -9.98, 100.0, 1000.0)])
        scorer = ScenicScorer(gps)
        assert scorer.get_speed_variance(999.0, 1001.0) == 0.0

    def test_stopped_vehicle_low_variance(self):
        """Vehicle at ~0 speed -> near-zero variance."""
        gps = self._make_gps([0.5, 0.3, 0.1, 0.2, 0.4] * 5)
        scorer = ScenicScorer(gps)
        start = gps.trackpoints[0].timestamp
        end = gps.trackpoints[-1].timestamp
        variance = scorer.get_speed_variance(start, end)
        assert variance < 1.0


# ── UnifiedSegmentScorer ─────────────────────────────────────────────


class TestUnifiedSegmentScorer:
    """Unified scorer combining GPS + visual signals."""

    def _default_weights(self) -> dict:
        return {
            "motion_energy": 0.25,
            "speed_variance": 0.15,
            "color_vibrancy": 0.10,
            "scene_change": 0.10,
            "sky_ratio": 0.05,
            "golden_hour": 0.10,
            "scenic_score": 0.25,
        }

    def test_weights_validation_good(self):
        scorer = UnifiedSegmentScorer(self._default_weights())
        # Should not raise
        assert scorer is not None

    def test_weights_validation_bad_sum(self):
        bad_weights = self._default_weights()
        bad_weights["motion_energy"] = 0.90
        with pytest.raises(ValueError, match="sum to 1.0"):
            UnifiedSegmentScorer(bad_weights)

    def test_combine_signals_produces_composite(self):
        """combine_signals returns proper shape with composite in column 1."""
        scorer = UnifiedSegmentScorer(self._default_weights())

        n = 100
        timestamps = np.arange(n, dtype=np.float64)
        signals = {
            "motion_energy": np.random.rand(n),
            "speed_variance": np.random.rand(n),
            "color_vibrancy": np.random.rand(n),
            "scene_change": np.random.rand(n),
            "sky_ratio": np.random.rand(n),
            "golden_hour": np.random.rand(n),
            "scenic_score": np.random.rand(n),
        }

        result = scorer.combine_signals(timestamps, signals)

        # Shape: (N, NUM_COLUMNS) = [timestamp, composite, K signals]
        assert result.shape == (n, NUM_COLUMNS)
        # Composite scores should be in [0, 1]
        composites = result[:, 1]
        assert np.all(composites >= 0.0)
        assert np.all(composites <= 1.0)

    def test_combine_signals_respects_weights(self):
        """If only one signal is nonzero, composite = weight * signal."""
        weights = {k: 0.0 for k in self._default_weights()}
        weights["motion_energy"] = 1.0

        scorer = UnifiedSegmentScorer(weights)

        n = 50
        timestamps = np.arange(n, dtype=np.float64)
        signals = {k: np.zeros(n) for k in weights}
        signals["motion_energy"] = np.full(n, 0.8)

        result = scorer.combine_signals(timestamps, signals)
        composites = result[:, 1]
        np.testing.assert_allclose(composites, 0.8, atol=1e-6)

    def test_merge_file_scores(self):
        """Merging two file score arrays concatenates with offset timestamps."""
        scorer = UnifiedSegmentScorer(self._default_weights())

        # File A: 60 seconds
        scores_a = np.zeros((60, 9))
        scores_a[:, 0] = np.arange(60)  # timestamps 0-59
        scores_a[:, 1] = 0.5  # composite

        # File B: 40 seconds
        scores_b = np.zeros((40, 9))
        scores_b[:, 0] = np.arange(40)  # timestamps 0-39
        scores_b[:, 1] = 0.8  # composite

        merged = scorer.merge_file_scores(
            [scores_a, scores_b],
            [60.0, 40.0],
        )

        # Total: 100 rows
        assert merged.shape[0] == 100
        # File B timestamps offset by 60
        assert merged[60, 0] == pytest.approx(60.0)
        assert merged[99, 0] == pytest.approx(99.0)
        # Composites preserved
        assert merged[30, 1] == pytest.approx(0.5)
        assert merged[70, 1] == pytest.approx(0.8)

    def test_save_and_load_scores(self, tmp_path):
        """Scores round-trip through .npz save/load."""
        scorer = UnifiedSegmentScorer(self._default_weights())

        scores = np.random.rand(100, 9)
        metadata = {"total_duration": 100.0, "num_files": 2}

        path = tmp_path / "scores.npz"
        scorer.save_scores(scores, path, metadata)
        loaded_scores, loaded_meta = scorer.load_scores(path)

        np.testing.assert_array_equal(loaded_scores, scores)
        assert loaded_meta["total_duration"] == 100.0
        assert loaded_meta["num_files"] == 2


# ── UnifiedSegmentScorer.score_file ────────────────────────────────


class TestScoreFile:
    """Tests for score_file: FFmpeg frame extraction + multi-signal scoring."""

    def _default_weights(self) -> dict:
        return {
            "motion_energy": 0.25,
            "speed_variance": 0.15,
            "color_vibrancy": 0.10,
            "scene_change": 0.10,
            "sky_ratio": 0.05,
            "golden_hour": 0.10,
            "scenic_score": 0.25,
        }

    def _make_frames(self, n: int = 30) -> list[tuple[float, np.ndarray]]:
        """Create synthetic BGR frames with some variation."""
        rng = np.random.RandomState(42)
        frames = []
        for i in range(n):
            # Each frame slightly different (simulates real video)
            frame = rng.randint(0, 255, (360, 640, 3), dtype=np.uint8)
            frames.append((float(i), frame))
        return frames

    def test_score_file_output_shape(self):
        """score_file returns (N, NUM_COLUMNS) array matching frame count."""
        scorer = UnifiedSegmentScorer(self._default_weights())
        n = 30
        frames = self._make_frames(n)

        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            result = scorer.score_file(Path("/fake/video.mp4"))

        assert result.shape == (n, NUM_COLUMNS)
        # Timestamps match frame timestamps
        np.testing.assert_array_equal(result[:, 0], np.arange(n, dtype=np.float64))
        # All signal values in [0, 1]
        assert np.all(result[:, 1:] >= 0.0)
        assert np.all(result[:, 1:] <= 1.0)

    def test_score_file_first_frame_zero_motion(self):
        """First frame has 0 motion energy and 0 scene change (no previous frame)."""
        scorer = UnifiedSegmentScorer(self._default_weights())
        frames = self._make_frames(10)

        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            result = scorer.score_file(Path("/fake/video.mp4"))

        assert result[0, 2] == 0.0  # COL_MOTION_ENERGY
        assert result[0, 5] == 0.0  # COL_SCENE_CHANGE

    def test_score_file_no_gps_zeros_gps_signals(self):
        """Without GPS data, speed_variance and scenic_score columns are 0."""
        scorer = UnifiedSegmentScorer(self._default_weights())
        frames = self._make_frames(15)

        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            result = scorer.score_file(Path("/fake/video.mp4"))

        np.testing.assert_array_equal(result[:, 3], 0.0)  # speed_variance
        np.testing.assert_array_equal(result[:, 8], 0.0)  # scenic_score

    def test_score_file_with_gps_populates_gps_signals(self):
        """GPS data produces nonzero speed_variance and scenic_score."""
        scorer = UnifiedSegmentScorer(self._default_weights())
        n = 20
        frames = self._make_frames(n)

        # GPS with elevation gain and varying position → nonzero scenic score
        gps = GPSData(
            trackpoints=[
                GPSTrackpoint(
                    lat=51.77 + i * 0.002,
                    lon=-9.98,
                    elevation=100.0 + i * 10,
                    timestamp=1000.0 + i,
                )
                for i in range(n + 5)
            ]
        )

        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            result = scorer.score_file(
                Path("/fake/video.mp4"),
                gps_data=gps,
                gps_offset=1000.0,
            )

        assert result.shape == (n, NUM_COLUMNS)
        # scenic_score should have nonzero values (elevation gain + turns)
        assert np.any(result[:, 8] > 0.0)

    def test_score_file_empty_frames_returns_empty(self):
        """No extracted frames produces empty (0, NUM_COLUMNS) array."""
        scorer = UnifiedSegmentScorer(self._default_weights())

        with patch.object(scorer, "_extract_frames_bgr", return_value=[]):
            result = scorer.score_file(Path("/fake/video.mp4"))

        assert result.shape == (0, NUM_COLUMNS)


class TestGPSLessScoring:
    """GPS-absent scoring must redistribute weights, not drop to near-zero."""

    def _default_weights(self):
        return {
            "motion_energy": 0.25,
            "speed_variance": 0.15,
            "color_vibrancy": 0.10,
            "scene_change": 0.10,
            "sky_ratio": 0.05,
            "golden_hour": 0.10,
            "scenic_score": 0.25,
        }

    def test_gps_less_weight_redistribution_sums_to_one(self):
        """Redistributed weights must sum to 1.0."""
        scorer = UnifiedSegmentScorer(self._default_weights())
        new_weights = scorer._redistribute_gps_weights()
        total = sum(new_weights.values())
        assert abs(total - 1.0) < 0.01, f"Redistributed weights sum to {total}"

    def test_gps_less_scores_not_near_zero(self):
        """Without GPS, composite scores should still be reasonable (not all near-zero)."""
        scorer = UnifiedSegmentScorer(self._default_weights())
        n = 10
        # Simulate frames with good visual signals
        frames = [
            (float(i), np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)) for i in range(n)
        ]

        with patch.object(scorer, "_extract_frames_bgr", return_value=frames):
            result = scorer.score_file(Path("/fake/video.mp4"), gps_data=None)

        composites = result[:, 1]  # COL_COMPOSITE
        # At least some composites should be > 0.1 (not all near-zero)
        assert np.mean(composites) > 0.05, (
            f"GPS-less scores too low: mean={np.mean(composites):.3f}"
        )

    def test_gps_signals_get_zero_weight(self):
        """GPS signal weights must be zero when redistributed."""
        scorer = UnifiedSegmentScorer(self._default_weights())
        new_weights = scorer._redistribute_gps_weights()
        assert new_weights["speed_variance"] == 0.0
        assert new_weights["scenic_score"] == 0.0


# ── GPU-accelerated frame extraction ────────────────────────────────


class TestGPUFrameExtraction:
    """_extract_frames_bgr should attempt GPU decode then fall back to CPU."""

    def _default_weights(self) -> dict:
        return {
            "motion_energy": 0.25,
            "speed_variance": 0.15,
            "color_vibrancy": 0.10,
            "scene_change": 0.10,
            "sky_ratio": 0.05,
            "golden_hour": 0.10,
            "scenic_score": 0.25,
        }

    @patch("subprocess.run")
    def test_gpu_cmd_includes_hwaccel_cuda(self, mock_run, tmp_path):
        """First ffmpeg attempt should include -hwaccel cuda -c:v hevc_cuvid."""
        scorer = UnifiedSegmentScorer(self._default_weights())

        # Probe returns duration
        probe_result = MagicMock(returncode=0, stdout="60.0")
        # GPU decode succeeds with 1 frame of raw BGR
        frame_bytes = bytes(640 * 360 * 3)
        gpu_result = MagicMock(returncode=0, stdout=frame_bytes, stderr=b"")

        mock_run.side_effect = [probe_result, gpu_result]

        frames = scorer._extract_frames_bgr(tmp_path / "test.mp4")

        # Second call (after probe) should be the GPU ffmpeg command
        gpu_cmd = mock_run.call_args_list[1][0][0]
        assert "-hwaccel" in gpu_cmd
        assert "cuda" in gpu_cmd
        assert "hevc_cuvid" in gpu_cmd
        assert len(frames) == 1

    @patch("subprocess.run")
    def test_falls_back_to_cpu_on_gpu_failure(self, mock_run, tmp_path):
        """If GPU decode fails, should retry without -hwaccel."""
        scorer = UnifiedSegmentScorer(self._default_weights())

        probe_result = MagicMock(returncode=0, stdout="60.0")
        gpu_fail = MagicMock(returncode=1, stdout=b"", stderr=b"cuda error")
        frame_bytes = bytes(640 * 360 * 3)
        cpu_result = MagicMock(returncode=0, stdout=frame_bytes, stderr=b"")

        mock_run.side_effect = [probe_result, gpu_fail, cpu_result]

        frames = scorer._extract_frames_bgr(tmp_path / "test.mp4")

        # Should have 3 calls: probe, GPU attempt, CPU fallback
        assert mock_run.call_count == 3
        cpu_cmd = mock_run.call_args_list[2][0][0]
        assert "-hwaccel" not in cpu_cmd
        assert len(frames) == 1
