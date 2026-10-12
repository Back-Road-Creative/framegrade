"""
TDD tests for pipeline timeout calculations and audio measurement commands.

Regression guard: a long 4K video could time out on audio loudness measurement because:
  1. the loudness measurement command was missing the -vn flag, decoding 4K video for audio analysis
  2. Timeout multipliers were inflated to compensate for video decode overhead

These tests ensure:
  - SinglePassProcessor uses -vn for audio-only measurement
  - Timeout multipliers are appropriate for audio-only decode (not video decode)
  - Normalization pass (with -c:v copy) gets at least as much time as measurement

Reference workloads:
  - Short: 5 minutes (300s), 1080p
  - Medium: 44 minutes (2640s), 4K 60fps (typical GoMoveShift)
  - Long: 71 minutes (4263s), 4K 60fps
  - Max: 120 minutes (7200s), 4K 60fps (theoretical worst case)
"""

from unittest.mock import MagicMock, patch

# ── Audio Processor: -vn Flag ────────────────────────────────────────────


class TestAudioMeasurementCommand:
    """Verify _measure_audio uses -vn to skip video decode."""

    def test_single_pass_measure_audio_includes_vn_flag(self):
        """SinglePassProcessor._measure_audio must include -vn."""
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        proc = SinglePassProcessor.__new__(SinglePassProcessor)
        with patch("gomoveshift.pipeline.single_pass.run_ffmpeg_quick") as mock_quick:
            mock_quick.return_value = MagicMock(returncode=0, stderr='{\n"input_i": "-20.0"\n}')
            proc._measure_audio("/fake/video.mp4", 4263.0, 2160)

            cmd = mock_quick.call_args[0][0]
            assert "-vn" in cmd, (
                f"_measure_audio command missing -vn flag. Command: {' '.join(cmd)}"
            )


# The former ``TestAudioProcessorTimeouts`` class was removed with its subject,
# the dead ``gomoveshift/pipeline/audio.py`` ``AudioProcessor`` (since deleted).
# The live audio-measurement timeout is ``SinglePassProcessor._calc_audio_measurement_timeout`` —
# covered below.


# ── Single-Pass Processor Timeouts ──────────────────────────────────────


class TestSinglePassTimeouts:
    """Tests for single_pass.py timeout calculations."""

    def test_audio_measurement_71min_4k(self):
        """71-min 4K audio measurement with -vn should be reasonable."""
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        proc = SinglePassProcessor.__new__(SinglePassProcessor)
        duration = 4263
        timeout = proc._calc_audio_measurement_timeout(duration, 2160)
        # 1.5x * 4263 = 6394s — audio-only decode is fast
        assert timeout >= 6000, f"Single-pass audio measurement timeout {timeout}s too low"
        assert timeout <= 10000, (
            f"Single-pass audio measurement timeout {timeout}s too high with -vn"
        )

    def test_audio_measurement_minimum_floor(self):
        """Very short videos should hit the minimum floor."""
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        proc = SinglePassProcessor.__new__(SinglePassProcessor)
        timeout = proc._calc_audio_measurement_timeout(30, 1080)
        assert timeout >= 300, "Minimum floor should be 300s"

    def test_encode_nvenc_71min_4k(self):
        """NVENC encode timeout for 71-min 4K should have plenty of headroom."""
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        proc = SinglePassProcessor.__new__(SinglePassProcessor)
        duration = 4263
        timeout = proc._calc_encode_timeout(duration, use_nvenc=True)
        assert timeout >= 14000, f"NVENC encode timeout {timeout}s too low for 71-min 4K"

    def test_encode_nvenc_4k_grade_blur_chain_has_real_headroom(self):
        """The 4K nvenc single-pass encode runs its grade (gbrp16le lut3d) and
        blur overlay on the CPU at ~3-4 fps, roughly 0.11x wall speed, so a
        ~2500 s source takes on the order of 23,000 s of wall time. Stall
        detection (600 s) already kills a hung encode, so
        the ceiling only guards a runaway — it must never fire on a progressing
        4K encode. Require ≥ 2x that worst-case wall time."""
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        proc = SinglePassProcessor.__new__(SinglePassProcessor)
        duration = 2501
        worst_case_wall = 23200
        timeout = proc._calc_encode_timeout(duration, use_nvenc=True)
        assert timeout >= 2 * worst_case_wall, (
            f"NVENC 4K encode timeout {timeout}s < 2x worst-case wall {worst_case_wall}s"
        )

    def test_encode_cpu_71min_4k(self):
        """CPU encode timeout for 71-min 4K needs huge headroom."""
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        proc = SinglePassProcessor.__new__(SinglePassProcessor)
        duration = 4263
        timeout = proc._calc_encode_timeout(duration, use_nvenc=False)
        assert timeout >= 25000, f"CPU encode timeout {timeout}s too low for 71-min 4K"

    def test_encode_minimum_floor(self):
        """Short videos should hit the 600s minimum floor."""
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        proc = SinglePassProcessor.__new__(SinglePassProcessor)
        timeout = proc._calc_encode_timeout(30, use_nvenc=True)
        assert timeout >= 600, "Minimum floor should be 600s"

    def test_encode_nvenc_faster_than_cpu(self):
        """NVENC timeout should be shorter than CPU timeout."""
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        proc = SinglePassProcessor.__new__(SinglePassProcessor)
        duration = 4263
        nvenc = proc._calc_encode_timeout(duration, use_nvenc=True)
        cpu = proc._calc_encode_timeout(duration, use_nvenc=False)
        assert nvenc < cpu, "NVENC should be faster than CPU"

    def test_audio_measurement_4k_longer_than_1080p(self):
        """4K audio measurement should get more time than 1080p."""
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        proc = SinglePassProcessor.__new__(SinglePassProcessor)
        duration = 4263
        timeout_4k = proc._calc_audio_measurement_timeout(duration, 2160)
        timeout_1080 = proc._calc_audio_measurement_timeout(duration, 1080)
        assert timeout_4k >= timeout_1080, (
            "4K files are larger — audio measurement should get at least as much time"
        )


# The former ``TestTimeoutSync`` cross-module drift guard was removed: it
# compared ``audio.py``'s ``AudioProcessor`` against ``single_pass.py``, and
# ``AudioProcessor`` has since been deleted.
# ``single_pass.py`` is now the sole audio-measurement timeout owner, so there
# is no second implementation to drift from.


# ── FFmpeg Runner Integration ─────────────────────────────────────────


class TestFfmpegRunnerIntegration:
    """Verify that pipeline modules use ffmpeg_runner for encode operations."""

    def test_single_pass_uses_run_ffmpeg_encode(self):
        """SinglePassProcessor.process() should call run_ffmpeg_encode for the main encode."""
        from pathlib import Path

        from gomoveshift.config import Config
        from gomoveshift.pipeline.ffmpeg_runner import FFmpegResult, VideoValidation
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        proc = SinglePassProcessor(Config())

        mock_result = FFmpegResult(
            success=True,
            returncode=0,
            wall_time=10.0,
            stderr_tail="",
            last_speed=1.0,
        )
        mock_validation = VideoValidation(
            valid=True,
            duration=100.0,
            has_video=True,
            has_audio=True,
            video_codec="hevc",
            file_size=1000000,
        )

        with (
            patch(
                "gomoveshift.pipeline.single_pass.run_ffmpeg_encode", return_value=mock_result
            ) as mock_encode,
            patch("gomoveshift.pipeline.single_pass.run_ffmpeg_quick") as mock_quick,
            patch("gomoveshift.pipeline.single_pass.validate_video", return_value=mock_validation),
            patch.object(proc, "_has_nvenc", return_value=False),
            patch.object(proc, "_analyze_color", return_value=("null", "none", [], None)),
            patch.object(
                proc,
                "_get_video_info",
                return_value={"duration": 100.0, "height": 1080, "width": 1920, "fps": 30.0},
            ),
        ):
            # Mock audio measurement
            mock_quick.return_value = MagicMock(
                returncode=0,
                stderr='{\n"input_i": "-20.0",\n"input_tp": "-1.0",\n"input_lra": "10.0",\n"input_thresh": "-30.0",\n"target_offset": "0.0"\n}',
            )

            proc.process(
                video_path=Path("/fake/input.mp4"),
                output_path=Path("/tmp/test_out.mp4"),
                location="Test",
                video_duration=100.0,
                video_info={"duration": 100.0, "height": 1080, "width": 1920, "fps": 30.0},
            )

        # Verify run_ffmpeg_encode was called (not subprocess.run)
        assert mock_encode.called, "SinglePassProcessor should use run_ffmpeg_encode for encoding"
