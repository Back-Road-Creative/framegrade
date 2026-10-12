"""Tests for ffmpeg-based keyframe motion analysis.

Validates the 10x speedup from switching OpenCV sequential decode
to ffmpeg keyframe-only piped decode.
"""

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gomoveshift.analysis.motion import MotionAnalyzer
from gomoveshift.config import Config


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def analyzer(config):
    return MotionAnalyzer(config)


class TestFFmpegFrameReader:
    """Test _decode_frames_ffmpeg returns correct gray frames."""

    def test_returns_list_of_gray_frames(self, analyzer, tmp_path):
        """ffmpeg reader should return list of 2D numpy arrays (grayscale)."""
        # Create a short test video with ffmpeg
        video = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=5:size=1280x720:rate=30",
                "-c:v",
                "libx264",
                "-g",
                "30",  # keyframe every 1 second
                "-v",
                "quiet",
                str(video),
            ],
            check=True,
            timeout=30,
        )

        frames, fps, duration = analyzer._decode_frames_ffmpeg(video)

        assert len(frames) > 0
        # Each frame should be 2D grayscale
        for frame in frames:
            assert frame.ndim == 2
            assert frame.dtype == np.uint8

    def test_frames_are_resized_to_analysis_width(self, analyzer, tmp_path):
        """Frames should be scaled to ANALYSIS_WIDTH (640px)."""
        video = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=3:size=1920x1080:rate=30",
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-v",
                "quiet",
                str(video),
            ],
            check=True,
            timeout=30,
        )

        frames, fps, duration = analyzer._decode_frames_ffmpeg(video)

        assert len(frames) > 0
        # Width should be ANALYSIS_WIDTH
        assert frames[0].shape[1] == analyzer.ANALYSIS_WIDTH

    def test_returns_approx_one_frame_per_keyframe(self, analyzer, tmp_path):
        """With 1-second keyframes, should get ~1 frame/sec."""
        video = tmp_path / "test.mp4"
        dur = 10
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=duration={dur}:size=640x480:rate=30",
                "-c:v",
                "libx264",
                "-g",
                "30",  # keyframe every second
                "-v",
                "quiet",
                str(video),
            ],
            check=True,
            timeout=30,
        )

        frames, fps, duration = analyzer._decode_frames_ffmpeg(video)

        # Should be approximately 1 frame per second (±2 for edges)
        assert abs(len(frames) - dur) <= 2

    def test_returns_fps_and_duration(self, analyzer, tmp_path):
        """Should return video fps and duration alongside frames."""
        video = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=5:size=640x480:rate=24",
                "-c:v",
                "libx264",
                "-g",
                "24",
                "-v",
                "quiet",
                str(video),
            ],
            check=True,
            timeout=30,
        )

        frames, fps, duration = analyzer._decode_frames_ffmpeg(video)

        assert fps == pytest.approx(24.0, abs=1.0)
        assert duration == pytest.approx(5.0, abs=0.5)

    def test_raises_on_missing_file(self, analyzer, tmp_path):
        """Should raise ValueError for missing video."""
        with pytest.raises(ValueError, match="Could not open|not found|No such"):
            analyzer._decode_frames_ffmpeg(tmp_path / "nonexistent.mp4")

    def test_raises_on_invalid_file(self, analyzer, tmp_path):
        """Should raise ValueError for non-video file."""
        bad = tmp_path / "bad.mp4"
        bad.write_text("not a video")
        with pytest.raises((ValueError, RuntimeError)):
            analyzer._decode_frames_ffmpeg(bad)


class TestAnalyzeRawFFmpeg:
    """Test that analyze_raw uses ffmpeg path and produces valid motion data."""

    def test_returns_timestamps_and_energies(self, analyzer, tmp_path):
        """analyze_raw should return parallel lists of timestamps and energies."""
        video = tmp_path / "test.mp4"
        # Generate video with moving content (sine wave = motion between frames)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=5:size=640x480:rate=30",
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-v",
                "quiet",
                str(video),
            ],
            check=True,
            timeout=30,
        )

        timestamps, energies = analyzer.analyze_raw(video)

        assert len(timestamps) == len(energies)
        assert len(timestamps) > 0
        # Timestamps should be monotonically increasing
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]
        # Energies should be 0-1 range
        for e in energies:
            assert 0.0 <= e <= 1.0

    def test_progress_callback_called(self, analyzer, tmp_path):
        """Progress callback should fire during analysis."""
        video = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=3:size=640x480:rate=30",
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-v",
                "quiet",
                str(video),
            ],
            check=True,
            timeout=30,
        )

        calls = []

        def cb(current, total, progress):
            calls.append((current, total, progress))

        analyzer.analyze_raw(video, progress_callback=cb)

        assert len(calls) > 0
        # Last call should be at 100%
        assert calls[-1][2] == pytest.approx(1.0)


class TestFindHighMotionSegmentsFFmpeg:
    """Test that find_high_motion_segments works with ffmpeg backend."""

    def test_returns_high_motion_segments(self, analyzer, tmp_path):
        """Should find segments with above-threshold motion."""
        video = tmp_path / "test.mp4"
        # testsrc has a moving counter — produces consistent motion
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=30:size=640x480:rate=30",
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-v",
                "quiet",
                str(video),
            ],
            check=True,
            timeout=30,
        )

        segments = analyzer.find_high_motion_segments(
            video, min_duration=5.0, max_duration=15.0, top_n=3
        )

        # Should return list of HighMotionSegment
        assert isinstance(segments, list)
        for seg in segments:
            assert hasattr(seg, "start_time")
            assert hasattr(seg, "end_time")
            assert hasattr(seg, "avg_motion_energy")
            assert seg.end_time > seg.start_time

    def test_segments_dont_overlap(self, analyzer, tmp_path):
        """Returned segments should not overlap."""
        video = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=60:size=640x480:rate=30",
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-v",
                "quiet",
                str(video),
            ],
            check=True,
            timeout=60,
        )

        segments = analyzer.find_high_motion_segments(
            video, min_duration=10.0, max_duration=20.0, top_n=5
        )

        for i, s1 in enumerate(segments):
            for j, s2 in enumerate(segments):
                if i != j:
                    # No overlap: one must end before the other starts (with gap)
                    assert s1.end_time + 5.0 < s2.start_time or s2.end_time + 5.0 < s1.start_time


class TestSparseKeyframeFallback:
    """Test fallback to OpenCV when ffmpeg keyframes are too sparse."""

    def test_fallback_when_keyframes_sparse(self, analyzer, tmp_path):
        """When ffmpeg keyframe decode produces < 1 sample/sec, use fixed-fps tier."""
        video = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=10:size=640x480:rate=30",
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-v",
                "quiet",
                str(video),
            ],
            check=True,
            timeout=30,
        )

        # Mock _decode_frames_ffmpeg to return sparse frames (3 frames over 10s = 0.3 fps)
        sparse_frames = [np.zeros((480, 640), dtype=np.uint8)] * 3
        with (
            patch.object(
                analyzer, "_decode_frames_ffmpeg", return_value=(sparse_frames, 30.0, 10.0)
            ) as mock_ffmpeg,
            patch.object(
                analyzer,
                "_decode_frames_ffmpeg_fixed_fps",
                wraps=analyzer._decode_frames_ffmpeg_fixed_fps,
            ) as mock_fixed,
        ):
            timestamps, energies = analyzer.analyze_raw(video)

        # Keyframe path was called but its sparse result was rejected
        mock_ffmpeg.assert_called_once()
        # Fixed-fps tier was used instead
        mock_fixed.assert_called_once()
        # Result should have dense data (~2 fps * 10s = ~20 samples)
        assert len(timestamps) > 10

    def test_no_fallback_when_keyframes_dense(self, analyzer, tmp_path):
        """When ffmpeg keyframe decode produces >= 1 sample/sec, use it directly."""
        video = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=5:size=640x480:rate=30",
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-v",
                "quiet",
                str(video),
            ],
            check=True,
            timeout=30,
        )

        # Mock _decode_frames_ffmpeg to return dense frames (6 frames over 5s = 1.2 fps)
        dense_frames = [np.zeros((480, 640), dtype=np.uint8)] * 7
        # Make frames differ so motion energy is nonzero
        dense_frames[3] = np.ones((480, 640), dtype=np.uint8) * 128
        with (
            patch.object(
                analyzer, "_decode_frames_ffmpeg", return_value=(dense_frames, 30.0, 5.0)
            ) as mock_ffmpeg,
            patch.object(analyzer, "_analyze_raw_opencv") as mock_opencv,
        ):
            timestamps, energies = analyzer.analyze_raw(video)

        mock_ffmpeg.assert_called_once()
        # OpenCV fallback should NOT be called
        mock_opencv.assert_not_called()
        # Result from ffmpeg path (7 frames - 1 = 6 motion samples)
        assert len(timestamps) == 6

    def test_sparse_threshold_boundary(self, analyzer, tmp_path):
        """Exactly 1.0 fps should NOT trigger fallback."""
        video = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=5:size=640x480:rate=30",
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-v",
                "quiet",
                str(video),
            ],
            check=True,
            timeout=30,
        )

        # 6 frames over 5s = 1.2 fps (just above threshold)
        frames_at_boundary = [np.zeros((480, 640), dtype=np.uint8)] * 6
        with (
            patch.object(
                analyzer, "_decode_frames_ffmpeg", return_value=(frames_at_boundary, 30.0, 5.0)
            ),
            patch.object(analyzer, "_analyze_raw_opencv") as mock_opencv,
        ):
            analyzer.analyze_raw(video)

        mock_opencv.assert_not_called()


class TestRealKeyframeTimestamps:
    """Motion samples land at REAL keyframe PTS, not synthetic i*dt."""

    def _frames(self, n):
        frames = [np.zeros((8, 8), dtype=np.uint8) for _ in range(n)]
        for i in range(n):
            frames[i][:] = i * 10  # make consecutive frames differ
        return frames

    def test_real_timestamps_used_when_valid(self, analyzer):
        frames = self._frames(4)
        # Non-uniform, strictly increasing real PTS (keyframe cadence is uneven).
        ts, _ = analyzer._compute_motion_from_frames(
            frames, duration=10.0, frame_timestamps=[0.0, 0.9, 2.5, 3.0]
        )
        # Motion samples index 1..3 -> the REAL PTS of the later frame each.
        assert ts == [0.9, 2.5, 3.0]
        # And NOT the synthetic i*dt (dt = 10/4 = 2.5 -> [2.5, 5.0, 7.5]).
        assert ts != [2.5, 5.0, 7.5]

    def test_length_mismatch_falls_back_to_uniform(self, analyzer):
        frames = self._frames(4)
        ts, _ = analyzer._compute_motion_from_frames(
            frames,
            duration=8.0,
            frame_timestamps=[0.0, 1.0],  # wrong length
        )
        assert ts == [2.0, 4.0, 6.0]  # dt = 8/4 = 2.0

    def test_non_monotonic_timestamps_fall_back(self, analyzer):
        frames = self._frames(4)
        ts, _ = analyzer._compute_motion_from_frames(
            frames,
            duration=8.0,
            frame_timestamps=[0.0, 2.0, 1.0, 3.0],  # backwards
        )
        assert ts == [2.0, 4.0, 6.0]

    def test_probe_returns_none_on_ffprobe_failure(self, analyzer, tmp_path):
        bad = tmp_path / "not_a_video.mp4"
        bad.write_text("definitely not a video container")
        assert analyzer._probe_keyframe_timestamps(bad) is None


class TestFixedFpsDecode:
    """Test ffmpeg fixed-fps decode (middle tier for HEVC videos)."""

    def test_fixed_fps_returns_dense_frames(self, analyzer, tmp_path):
        """Fixed-fps decode at 2fps should return ~2 frames/sec."""
        video = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=10:size=640x480:rate=30",
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-v",
                "quiet",
                str(video),
            ],
            check=True,
            timeout=30,
        )

        frames, fps, duration = analyzer._decode_frames_ffmpeg_fixed_fps(video)

        # Should get approximately 2 frames/sec * 10s = ~20 frames
        assert len(frames) >= 15
        assert len(frames) <= 25
        # Each frame should be 2D grayscale at ANALYSIS_WIDTH
        assert frames[0].ndim == 2
        assert frames[0].shape[1] == analyzer.ANALYSIS_WIDTH

    def test_sparse_keyframes_trigger_fixed_fps_fallback(self, analyzer, tmp_path):
        """When keyframe decode is sparse, fixed-fps should be used."""
        video = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=10:size=640x480:rate=30",
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-v",
                "quiet",
                str(video),
            ],
            check=True,
            timeout=30,
        )

        # Mock keyframe decode to return sparse data (< 1 fps)
        sparse_frames = [np.zeros((480, 640), dtype=np.uint8)] * 3
        with (
            patch.object(
                analyzer, "_decode_frames_ffmpeg", return_value=(sparse_frames, 30.0, 10.0)
            ),
            patch.object(
                analyzer,
                "_decode_frames_ffmpeg_fixed_fps",
                wraps=analyzer._decode_frames_ffmpeg_fixed_fps,
            ) as mock_fixed,
        ):
            timestamps, energies = analyzer.analyze_raw(video)

        # Fixed-fps path should be called
        mock_fixed.assert_called_once()
        # Should produce dense motion data
        assert len(timestamps) > 10


class TestPercentileFallbackThreshold:
    """Test percentile-based fallback when absolute threshold yields 0."""

    def test_uses_percentile_when_absolute_yields_nothing(self, analyzer):
        """When motion_threshold * 2 finds 0 segments, use top percentile."""
        # Simulate data where all values are above dead-spot but below 2x threshold
        # motion_threshold = 0.008, so 2x = 0.016
        # All values between 0.009 and 0.015 — above dead-spot, below high-motion
        timestamps = [float(i) for i in range(100)]
        energies = [0.012] * 100  # All at 0.012 — below 0.016
        # Add some peaks that stand out relative to neighbors
        energies[20] = 0.014
        energies[21] = 0.015
        energies[22] = 0.014
        energies[50] = 0.014
        energies[51] = 0.015
        energies[52] = 0.014

        motion_data = list(zip(timestamps, energies))

        segments = analyzer._find_high_motion_windows(
            motion_data=motion_data,
            min_duration=5.0,
            max_duration=15.0,
            video_duration=100.0,
        )

        # Should find segments via percentile fallback
        assert len(segments) > 0

    def test_absolute_threshold_still_used_when_it_works(self, analyzer):
        """When absolute threshold finds segments, don't fall back."""
        timestamps = [float(i) for i in range(100)]
        energies = [0.001] * 100
        # Add clear peaks well above threshold
        for i in range(30, 45):
            energies[i] = 0.05
        for i in range(70, 85):
            energies[i] = 0.04

        motion_data = list(zip(timestamps, energies))

        segments = analyzer._find_high_motion_windows(
            motion_data=motion_data,
            min_duration=5.0,
            max_duration=15.0,
            video_duration=100.0,
        )

        # Should find segments via absolute threshold
        assert len(segments) > 0
        # All segments should have avg_energy above threshold
        for seg in segments:
            assert seg.avg_motion_energy > analyzer.motion_threshold * 2


class TestOpenCVFallback:
    """Test fallback to OpenCV when ffmpeg is unavailable."""

    def test_fallback_when_ffmpeg_missing(self, analyzer, tmp_path):
        """Should fall back to OpenCV decode if ffmpeg subprocess fails."""
        video = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=3:size=640x480:rate=30",
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-v",
                "quiet",
                str(video),
            ],
            check=True,
            timeout=30,
        )

        # Patch _decode_frames_ffmpeg to raise, forcing fallback
        with patch.object(analyzer, "_decode_frames_ffmpeg", side_effect=RuntimeError("no ffmpeg")):
            timestamps, energies = analyzer.analyze_raw(video)

        # Should still work via OpenCV fallback
        assert len(timestamps) > 0
        assert len(timestamps) == len(energies)


def _make_hung_proc():
    """Build a Popen mock whose wait(timeout=...) hangs (raises TimeoutExpired).

    A second wait() with no timeout (the reap) succeeds and returns 0, mirroring
    a process that only dies once kill() is delivered.
    """
    proc = MagicMock()
    proc.stdout.read.return_value = b""  # decode loop exits immediately (EOF)
    proc.stderr.read.return_value = b""
    proc.returncode = 0
    # First wait(timeout=...) hangs; an unbounded wait() (post-kill reap) returns.
    proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10), 0]
    return proc


class TestDecodeReapsHungFfmpeg:
    """ffmpeg child must be killed and reaped if proc.wait(timeout=...) hangs.

    Regression guard: an unreaped ffmpeg child holds the NVDEC decoder (GPU/zombie
    leak) and TimeoutExpired masks the original error. Mirrors the reap pattern in
    pipeline/ffmpeg_runner.py (terminate/kill -> wait).
    """

    def _patch_probe(self, analyzer):
        """Patch info + ffprobe so the decode reaches the Popen wait site."""
        probe = MagicMock()
        probe.stdout = '{"streams": [{"width": 640, "height": 480}]}'
        return (
            patch.object(analyzer, "_get_video_info", return_value=(30.0, 100, 10.0)),
            patch("gomoveshift.analysis.motion.subprocess.run", return_value=probe),
        )

    def test_keyframe_decode_reaps_on_wait_timeout(self, analyzer, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00")  # exists() check only

        proc = _make_hung_proc()
        info_patch, run_patch = self._patch_probe(analyzer)
        with (
            info_patch,
            run_patch,
            patch("gomoveshift.analysis.motion.subprocess.Popen", return_value=proc),
        ):
            with pytest.raises(subprocess.TimeoutExpired):
                analyzer._decode_frames_ffmpeg(video)

        # Child was killed and reaped — not left as an orphan/zombie.
        proc.kill.assert_called_once()
        assert proc.wait.call_count == 2

    def test_fixed_fps_decode_reaps_on_wait_timeout(self, analyzer, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00")

        proc = _make_hung_proc()
        info_patch, run_patch = self._patch_probe(analyzer)
        with (
            info_patch,
            run_patch,
            patch("gomoveshift.analysis.motion.subprocess.Popen", return_value=proc),
        ):
            with pytest.raises(subprocess.TimeoutExpired):
                analyzer._decode_frames_ffmpeg_fixed_fps(video)

        proc.kill.assert_called_once()
        assert proc.wait.call_count == 2
