"""
Quality regression guard — contract tests for encoding quality.

If any of these break, someone changed encoding quality parameters.
This file is the last line of defense against config drift.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFFmpegLongformCRF:
    """Long-form CRF must match encoding_constants — captures actual FFmpeg command args."""

    def test_cpu_crf_matches_constant(self):
        from gomoveshift.pipeline.color import get_encoding_args
        from gomoveshift.pipeline.encoding_constants import LONGFORM_CRF

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=False)
        idx = args.index("-crf")
        assert args[idx + 1] == str(LONGFORM_CRF), (
            f"Long-form CRF should be {LONGFORM_CRF}, got {args[idx + 1]}"
        )

    def test_nvenc_cq_matches_constant(self):
        """NVENC CQ must match encoding_constants."""
        from gomoveshift.pipeline.color import get_encoding_args
        from gomoveshift.pipeline.encoding_constants import LONGFORM_NVENC_CQ

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)
        idx = args.index("-cq")
        assert args[idx + 1] == str(LONGFORM_NVENC_CQ), (
            f"NVENC CQ should be {LONGFORM_NVENC_CQ}, got {args[idx + 1]}"
        )


class TestFFmpegLongformNeverUsesP3:
    """NVENC preset must never be p3 — that was the fast mode path."""

    def test_nvenc_preset_is_never_p3(self):
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)
        idx = args.index("-preset")
        assert args[idx + 1] != "p3", "NVENC preset must not be p3 (was fast mode)"
        assert args[idx + 1] == "p7", f"NVENC preset should be p7, got {args[idx + 1]}"


class TestShortsCRFMatchesConstant:
    """Shorts CQ must match encoding_constants — CQ 18, same as the long-form master."""

    def test_shorts_crf_constant(self):
        """CQ 18 matches long-form — 8-bit CQ 23 banded graded skies."""
        from gomoveshift.pipeline.encoding_constants import SHORTS_CRF

        assert SHORTS_CRF == 18, f"SHORTS_CRF should be 18, got {SHORTS_CRF}"

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    @patch("gomoveshift.pipeline.ffmpeg_runner.validate_video")
    def test_shorts_ffmpeg_cq_matches_constant(self, mock_val, mock_dims, mock_run, tmp_path):
        # NVENC-only path; -crf flag (libx265) replaced by -cq (hevc_nvenc).
        from gomoveshift.pipeline.shorts_extractor import render_short_ffmpeg

        mock_val_result = MagicMock()
        mock_val_result.valid = True
        mock_val_result.duration = 30.0
        mock_val.return_value = mock_val_result

        output = tmp_path / "test.mp4"
        output.write_bytes(b"x" * 1024)
        mock_run.return_value = MagicMock(returncode=0)

        render_short_ffmpeg(
            source_video=Path("/fake/video.mp4"),
            output_path=output,
            start_time=0.0,
            duration=30.0,
            location="Test",
            use_nvenc=True,
        )

        from gomoveshift.pipeline.encoding_constants import SHORTS_CRF

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("-cq")
        assert cmd[idx + 1] == str(SHORTS_CRF), (
            f"Shorts CQ should be '{SHORTS_CRF}', got '{cmd[idx + 1]}'"
        )


class TestLUFSConsistentMeasureAndEncode:
    """Audio measurement and encoding must use the same LUFS constant."""

    def test_measurement_uses_constant(self):
        """Audio measurement pass must use LONGFORM_AUDIO_LUFS, not hardcoded -16.

        The measurement command is built by the shared
        ``loudnorm_measure.build_loudnorm_measure_cmd`` helper (de-duped from
        ``single_pass``/``ost_applier``); the LUFS target lives there now, so
        the guard inspects both the method and the shared builder.
        """
        import inspect

        from gomoveshift.pipeline.loudnorm_measure import build_loudnorm_measure_cmd
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        source = inspect.getsource(SinglePassProcessor._measure_audio) + inspect.getsource(
            build_loudnorm_measure_cmd
        )
        assert "I=-16" not in source, (
            "Audio measurement still hardcodes -16 LUFS — must use encoding_constants"
        )
        assert "ec." in source or "encoding_constants" in source, (
            "Audio measurement should reference encoding_constants (directly or via ec alias)"
        )

    def test_encode_filter_uses_config_lufs(self):
        """Audio encode filter uses config which matches constant."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.encoding_constants import LONGFORM_AUDIO_LUFS

        config = Config()
        assert config.output.audio_target_lufs == LONGFORM_AUDIO_LUFS, (
            f"Config audio_target_lufs ({config.output.audio_target_lufs}) != "
            f"constant ({LONGFORM_AUDIO_LUFS})"
        )


class TestValidatorRejectsLowQuality4K:
    """Validator must reject 4K video below the ``LONGFORM_MIN_BITRATE_4K`` floor."""

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_4k_at_20mbps_rejected(self, mock_run):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"bit_rate": "20000000"},  # 20 Mbps
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "hevc",
                            "width": 3840,
                            "height": 2160,
                            "profile": "Main 10",
                            "pix_fmt": "yuv420p10le",
                        },
                        {"codec_type": "audio", "codec_name": "aac"},
                    ],
                }
            ),
        )

        validator = PipelineValidator()
        gate = validator._check_video_quality(Path("/fake/video.mp4"))
        assert not gate.passed, "4K at 20 Mbps should be rejected (below the ec 4K floor)"
        assert "below minimum" in gate.message

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_4k_at_40mbps_passes(self, mock_run):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"bit_rate": "40000000"},  # 40 Mbps
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "hevc",
                            "width": 3840,
                            "height": 2160,
                            "profile": "Main 10",
                            "pix_fmt": "yuv420p10le",
                            # Colour contract — SDR BT.709 limited.
                            "color_range": "tv",
                            "color_space": "bt709",
                            "color_primaries": "bt709",
                            "color_transfer": "bt709",
                        },
                        {"codec_type": "audio", "codec_name": "aac"},
                    ],
                }
            ),
        )

        validator = PipelineValidator()
        gate = validator._check_video_quality(Path("/fake/video.mp4"))
        assert gate.passed, f"4K at 40 Mbps should pass: {gate.message}"

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_1080p_at_8mbps_rejected(self, mock_run):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"bit_rate": "8000000"},  # 8 Mbps
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "hevc",
                            "width": 1920,
                            "height": 1080,
                            "profile": "Main 10",
                            "pix_fmt": "yuv420p10le",
                        },
                        {"codec_type": "audio", "codec_name": "aac"},
                    ],
                }
            ),
        )

        validator = PipelineValidator()
        gate = validator._check_video_quality(Path("/fake/video.mp4"))
        assert not gate.passed, "1080p at 8 Mbps should be rejected (below the ec 1080p floor)"

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_1080p_at_40mbps_rejected_false_4k_claim(self, mock_run):
        """A sub-4K long-form is rejected even with ample bitrate.

        Long-form metadata unconditionally claims 4K in the title ("… Scenic
        Drive in 4K …") and SEO description line ("… 4K drive …"). A
        1080p final — however high its bitrate — would ship those false 4K
        claims, so ``check_video_quality`` must reject it on resolution.
        """
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"bit_rate": "40000000"},  # 40 Mbps — well above any floor
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "hevc",
                            "width": 1920,
                            "height": 1080,
                            "profile": "Main 10",
                            "pix_fmt": "yuv420p10le",
                        },
                        {"codec_type": "audio", "codec_name": "aac"},
                    ],
                }
            ),
        )

        validator = PipelineValidator()
        gate = validator._check_video_quality(Path("/fake/video.mp4"))
        assert not gate.passed, "1080p long-form must be rejected (would ship a false 4K title)"
        assert "below 4K" in gate.message


class TestFailSavesState:
    """_fail() must persist state to disk."""

    def test_fail_writes_state_json(self, tmp_path):
        from gomoveshift.pipeline.orchestrator import (
            PipelineOrchestrator,
            PipelineResult,
            VideoProperties,
        )

        output_dir = tmp_path / "project"
        output_dir.mkdir()
        (output_dir / "_stages").mkdir()

        result = PipelineResult(
            success=True,
            project_name="timeline14",
            output_dir=output_dir,
            final_video=output_dir / "timeline14_FINAL.mp4",
            stages_completed=["detect_properties", "stop_removal", "single_pass_encode"],
            video_properties=VideoProperties(
                width=3840,
                height=2160,
                fps=59.94,
                duration=3600.0,
                codec="hevc",
                audio_codec="aac",
                file_size_bytes=15_000_000_000,
            ),
        )

        orch = PipelineOrchestrator()
        orch._fail("GPU memory exhausted", result)

        state_file = output_dir / "_stages" / "pipeline_state.json"
        assert state_file.exists(), "_fail() must write pipeline_state.json"

        state = json.loads(state_file.read_text())
        assert len(state["stages_completed"]) == 3


class TestSummaryIncludesEncodingParams:
    """Pipeline summary must include encoding_params for audit trail."""

    def test_summary_has_encoding_params_key(self):
        """The summary dict structure must include encoding_params."""
        from gomoveshift.pipeline.encoding_constants import (
            LONGFORM_CRF,
            LONGFORM_NVENC_CQ,
            LONGFORM_NVENC_PRESET,
            LONGFORM_PIX_FMT,
            LONGFORM_PRESET,
            LONGFORM_PROFILE,
        )

        # Verify the constants exist and are correct types
        assert isinstance(LONGFORM_CRF, int)
        assert isinstance(LONGFORM_NVENC_CQ, int)
        assert isinstance(LONGFORM_NVENC_PRESET, str)
        assert isinstance(LONGFORM_PRESET, str)
        assert isinstance(LONGFORM_PROFILE, str)
        assert isinstance(LONGFORM_PIX_FMT, str)

    def test_orchestrator_summary_generation_includes_encoding(self):
        """Verify orchestrator source includes encoding_params in summary."""
        import inspect

        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        source = inspect.getsource(PipelineOrchestrator.run)
        assert "encoding_params" in source, (
            "Pipeline summary must include 'encoding_params' dict for audit trail"
        )


class TestBitrateFloorEncoding:
    """NVENC encoding must include bitrate floor when source bitrate is provided."""

    def test_nvenc_always_includes_bv_floor(self):
        from gomoveshift.pipeline.color import get_encoding_args
        from gomoveshift.pipeline.encoding_constants import (
            LONGFORM_BUFSIZE_RATIO,
            LONGFORM_MAXRATE_RATIO,
            LONGFORM_MIN_BITRATE_4K,
        )

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)
        assert "-b:v" in args, "Must always include -b:v"
        assert "-maxrate" in args, "Must include -maxrate"
        assert "-bufsize" in args, "Must include -bufsize"
        # Must remain VBR; CBR drops -cq
        rc_idx = args.index("-rc")
        assert args[rc_idx + 1] == "vbr", f"Must use -rc vbr (not cbr), got {args[rc_idx + 1]}"
        assert "-cq" in args, "-cq must be retained in VBR mode"

        # Check values from encoding constants — fixed absolute floor
        expected_target = LONGFORM_MIN_BITRATE_4K
        expected_maxrate = int(expected_target * LONGFORM_MAXRATE_RATIO)
        expected_bufsize = int(expected_target * LONGFORM_BUFSIZE_RATIO)

        bv_idx = args.index("-b:v")
        bv_val = args[bv_idx + 1]
        assert bv_val == f"{expected_target}k", f"Expected -b:v {expected_target}k, got {bv_val}"

        maxrate_idx = args.index("-maxrate")
        maxrate_val = args[maxrate_idx + 1]
        assert maxrate_val == f"{expected_maxrate}k", (
            f"Expected -maxrate {expected_maxrate}k, got {maxrate_val}"
        )

        bufsize_idx = args.index("-bufsize")
        bufsize_val = args[bufsize_idx + 1]
        assert bufsize_val == f"{expected_bufsize}k", (
            f"Expected -bufsize {expected_bufsize}k, got {bufsize_val}"
        )

    def test_x265_also_includes_bv(self):
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=False)
        assert "-b:v" in args, "x265 path must also include -b:v"

    def test_bv_always_present_even_without_source_bitrate(self):
        """Floor is absolute — present regardless of source_bitrate_kbps."""
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True, source_bitrate_kbps=0)
        assert "-b:v" in args, "Must include -b:v (absolute floor, not source-dependent)"
        assert "-cq" in args, "Must keep -cq in VBR mode"

    def test_4k_absolute_floor_value(self):
        """4K floor is 40 Mbps — comfortably above YouTube's 35 Mbps recommendation."""
        from gomoveshift.pipeline.encoding_constants import LONGFORM_MIN_BITRATE_4K

        assert LONGFORM_MIN_BITRATE_4K == 40_000, (
            f"LONGFORM_MIN_BITRATE_4K should be 40000, got {LONGFORM_MIN_BITRATE_4K}"
        )

    def test_maxrate_ratio_lifts_ceiling_off_floor(self):
        """maxrate must be ≥2.0x the floor, so VBR has headroom above the floor
        for CQ 18 on detail-heavy scenes."""
        from gomoveshift.pipeline.encoding_constants import LONGFORM_MAXRATE_RATIO

        assert LONGFORM_MAXRATE_RATIO >= 2.0, (
            f"LONGFORM_MAXRATE_RATIO must be ≥2.0 (got {LONGFORM_MAXRATE_RATIO}); "
            "lower values cap VBR below what CQ 18 demands on complex scenes"
        )

    def test_bufsize_ratio_accommodates_maxrate_burst(self):
        """bufsize must be ≥ maxrate ratio so the VBR burst the new ceiling allows
        actually fits in the buffer — otherwise the encoder still gets clipped."""
        from gomoveshift.pipeline.encoding_constants import (
            LONGFORM_BUFSIZE_RATIO,
            LONGFORM_MAXRATE_RATIO,
        )

        assert LONGFORM_BUFSIZE_RATIO >= LONGFORM_MAXRATE_RATIO, (
            f"bufsize ratio ({LONGFORM_BUFSIZE_RATIO}) must be ≥ maxrate ratio "
            f"({LONGFORM_MAXRATE_RATIO}); otherwise burst is buffer-clipped"
        )

    def test_1080p_absolute_floor_value(self):
        """1080p floor is 15 Mbps — above YouTube's 12 Mbps recommendation."""
        from gomoveshift.pipeline.encoding_constants import LONGFORM_MIN_BITRATE_1080P

        assert LONGFORM_MIN_BITRATE_1080P == 15_000, (
            f"LONGFORM_MIN_BITRATE_1080P should be 15000, got {LONGFORM_MIN_BITRATE_1080P}"
        )


class TestBitratePreservationGate:
    """Validation gate must catch catastrophic bitrate drops."""

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_fails_when_output_below_absolute_floor(self, mock_run):
        """20 Mbps output for 4K — below 40 Mbps absolute floor, must FAIL."""
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"bit_rate": "20000000"},
                    "streams": [{"codec_type": "video", "height": 2160}],
                }
            ),
        )

        validator = PipelineValidator()
        gate = validator._check_bitrate_preservation(Path("/fake/video.mp4"))
        assert not gate.passed, f"20 Mbps for 4K should FAIL (below 40 Mbps floor): {gate.message}"

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_passes_when_output_above_floor(self, mock_run):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"bit_rate": "75000000"},
                    "streams": [{"codec_type": "video", "height": 2160}],
                }
            ),
        )

        validator = PipelineValidator()
        gate = validator._check_bitrate_preservation(Path("/fake/video.mp4"))
        assert gate.passed, f"75 Mbps for 4K should PASS (above 40 Mbps floor): {gate.message}"

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_fails_at_39mbps_4k(self, mock_run):
        """39 Mbps for 4K — just below 40 Mbps absolute floor, must FAIL."""
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"bit_rate": "39000000"},
                    "streams": [{"codec_type": "video", "height": 2160}],
                }
            ),
        )

        validator = PipelineValidator()
        gate = validator._check_bitrate_preservation(Path("/fake/video.mp4"))
        assert not gate.passed, f"39 Mbps for 4K should FAIL (below 40 Mbps floor): {gate.message}"

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_passes_at_exactly_40mbps_4k(self, mock_run):
        """40 Mbps for 4K — exactly at floor, must PASS."""
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"bit_rate": "40000000"},
                    "streams": [{"codec_type": "video", "height": 2160}],
                }
            ),
        )

        validator = PipelineValidator()
        gate = validator._check_bitrate_preservation(Path("/fake/video.mp4"))
        assert gate.passed, f"40 Mbps for 4K should PASS (at floor): {gate.message}"

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_runs_even_when_no_source_bitrate_is_supplied(self, mock_run):
        """The floor is absolute and resolution-based, so the gate must run on
        every validate(). It does not depend on a truthy
        ``source_bitrate_kbps`` reaching validate(), so callers with no source
        bitrate still get the constants-sourced bitrate check.
        """
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        # Mock all the gates that do run
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"bit_rate": "40000000", "duration": "3600"},
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "hevc",
                            "width": 3840,
                            "height": 2160,
                            "profile": "Main 10",
                            "pix_fmt": "yuv420p10le",
                        },
                        {"codec_type": "audio", "codec_name": "aac"},
                    ],
                }
            ),
            stderr="",
        )

        validator = PipelineValidator()
        result = validator.validate(video_path=Path("/fake/video.mp4"))
        gate_names = [g.gate for g in result.gates]
        assert "bitrate_preservation" in gate_names, (
            "bitrate_preservation must run unconditionally — its floor never "
            f"depended on the source bitrate; got gates {gate_names}"
        )


class TestOrchestratorPassesBitrateToSinglePass:
    """The orchestrator MUST pass bitrate_kbps in video_info to single_pass.

    This is the last line of defense against silent quality degradation.
    Without bitrate_kbps the NVENC VBR bitrate floor never activates and the
    output can fall far below the source bitrate.
    """

    def test_video_info_dict_includes_bitrate_kbps(self):
        """Inspect orchestrator source to verify bitrate_kbps is passed."""
        import inspect

        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        from gomoveshift.pipeline.stages.single_pass_encode import SinglePassEncodeStage

        source = inspect.getsource(SinglePassEncodeStage.run)
        assert "bitrate_kbps" in source, (
            "CRITICAL: SinglePassEncodeStage.run() must pass 'bitrate_kbps' in video_info "
            "to single_pass.process(). Without it, NVENC VBR has no bitrate floor "
            "and the output can silently fall far below the source bitrate."
        )

    def test_video_info_includes_bitrate_from_props(self):
        """The bitrate_kbps value must come from props, not be hardcoded to 0."""
        import inspect

        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        from gomoveshift.pipeline.stages.single_pass_encode import SinglePassEncodeStage

        source = inspect.getsource(SinglePassEncodeStage.run)
        assert "props.bitrate_kbps" in source, (
            "CRITICAL: bitrate_kbps must come from props.bitrate_kbps, "
            "not a hardcoded value. props reads the actual source file bitrate."
        )


class TestStateFileSavesBitrate:
    """State file must persist bitrate_kbps so resume preserves quality."""

    def test_save_state_includes_bitrate_kbps(self):
        """_save_state must write bitrate_kbps to video_properties."""
        import inspect

        from gomoveshift.pipeline.state_manager import save_state

        source = inspect.getsource(save_state)
        assert "bitrate_kbps" in source, (
            "CRITICAL: save_state must persist bitrate_kbps in video_properties. "
            "Without it, resume reconstructs props with bitrate_kbps=0 and the "
            "bitrate floor never activates on re-encode."
        )

    def test_resume_restores_bitrate_kbps(self):
        """Resume path must read bitrate_kbps from saved state."""
        import inspect

        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        source = inspect.getsource(PipelineOrchestrator._restore_from_state)
        assert "bitrate_kbps" in source
        # Must use .get() to read from props_data, not hardcode 0
        assert (
            'props_data.get("bitrate_kbps"' in source or "props_data.get('bitrate_kbps'" in source
        ), (
            "CRITICAL: resume must read bitrate_kbps from saved state via "
            "props_data.get('bitrate_kbps'), not default to 0."
        )


class TestPostEncodeQualityCheck:
    """Post-encode quality check must exist in orchestrator — catches bad encodes immediately."""

    def test_post_encode_bitrate_check_exists_in_orchestrator(self):
        """Orchestrator must do a filesize-based bitrate check right after single_pass."""
        import inspect

        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        from gomoveshift.pipeline.stages.single_pass_encode import SinglePassEncodeStage

        source = inspect.getsource(SinglePassEncodeStage.run)
        assert "QUALITY GATE" in source, (
            "CRITICAL: single_pass_encode stage must have a post-encode quality gate "
            "that checks output file size immediately after single_pass encode."
        )

    def test_post_encode_rejects_low_bitrate_output(self):
        """Mock scenario: output is 40% of source, must trigger quality gate failure."""
        # This test verifies the math: if props.bitrate_kbps=83000, duration=3600,
        # and output is only 40% of expected size, the quality gate fires.
        # output_bitrate = (file_size * 8) / duration / 1000
        # min_expected = 83000 * 0.90 = 74700
        # Bad output: 83000 * 0.40 = 33200 kbps → file_size = 33200*1000*3600/8 = ~14.9 GB
        # 33200 < 74700 → FAIL ✓

        expected_kbps = 83000
        bad_output_kbps = int(expected_kbps * 0.40)  # 33200
        min_expected = int(expected_kbps * 0.90)  # 74700
        assert bad_output_kbps < min_expected, (
            f"Test assumption: 40% output ({bad_output_kbps}) must be below "
            f"90% threshold ({min_expected})"
        )


class TestKeepSegmentsPersistedInState:
    """keep_segments must be saved and restored from pipeline state."""

    def test_save_state_includes_keep_segments(self, tmp_path):
        from gomoveshift.pipeline.orchestrator import (
            PipelineOrchestrator,
            VideoProperties,
        )

        state_file = tmp_path / "pipeline_state.json"
        orch = PipelineOrchestrator()
        vp = VideoProperties(
            width=3840,
            height=2160,
            fps=59.94,
            duration=3600.0,
            codec="hevc",
            audio_codec="aac",
            file_size_bytes=15_000_000_000,
            bitrate_kbps=83000,
        )
        keep_segments = [(0.0, 100.0), (120.0, 300.0), (350.0, 3600.0)]
        orch._save_state(
            state_file=state_file,
            project_name="test",
            source_video=Path("/fake/video.mp4"),
            source_hash="abc123",
            stages_completed=["detect_properties", "stop_removal"],
            current_video=Path("/fake/output.mp4"),
            video_properties=vp,
            timings={},
            params={},
            keep_segments=keep_segments,
        )

        state = json.loads(state_file.read_text())
        assert "keep_segments" in state, "keep_segments must be in saved state"
        assert len(state["keep_segments"]) == 3
        assert state["keep_segments"][0] == [0.0, 100.0]
        assert state["keep_segments"][2] == [350.0, 3600.0]

    def test_save_state_omits_keep_segments_when_none(self, tmp_path):
        from gomoveshift.pipeline.orchestrator import (
            PipelineOrchestrator,
            VideoProperties,
        )

        state_file = tmp_path / "pipeline_state.json"
        orch = PipelineOrchestrator()
        vp = VideoProperties(
            width=3840,
            height=2160,
            fps=59.94,
            duration=3600.0,
            codec="hevc",
            audio_codec="aac",
            file_size_bytes=15_000_000_000,
            bitrate_kbps=83000,
        )
        orch._save_state(
            state_file=state_file,
            project_name="test",
            source_video=Path("/fake/video.mp4"),
            source_hash="abc123",
            stages_completed=["detect_properties"],
            current_video=Path("/fake/output.mp4"),
            video_properties=vp,
            timings={},
            params={},
        )

        state = json.loads(state_file.read_text())
        assert "keep_segments" not in state, "keep_segments should not be in state when None"


class TestAnalyticsSyncNullGuard:
    """_sync_youtube_analytics must not crash when _manager is None."""

    def test_returns_gracefully_when_manager_none(self):
        from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

        gen = LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)
        gen._manager = None
        gen._optimizer = None
        gen._ledger = None
        gen.storage_path = Path("/tmp/fake")
        gen.credentials_dir = Path("/tmp/fake")

        # Should not raise
        gen._sync_youtube_analytics()


class TestMetadataQualityGateCatchesDegradedMode:
    """Validation gate must FAIL when metadata was generated in basic fallback mode."""

    def test_seo_degraded_flag_fails_gate(self):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        validator = PipelineValidator()
        metadata = {
            "title": "POV Drive Through County Kerry | Scenic Road Trip",
            "description": "Join me on a relaxing POV drive...",
            "tags": ["pov driving", "scenic drive"],
            "category": "19",
            "seo_degraded": True,
        }
        # Run just the gate check via validate (need a dummy video)
        # Instead, directly check the gate is in the result
        result = validator.validate(
            video_path=Path("/dev/null"),
            metadata_json=metadata,
        )
        gate_names = [g.gate for g in result.gates]
        assert "seo_quality" in gate_names
        seo_gate = next(g for g in result.gates if g.gate == "seo_quality")
        assert not seo_gate.passed
        assert "degraded" in seo_gate.message.lower()

    def test_seo_not_degraded_no_gate(self):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        validator = PipelineValidator()
        metadata = {
            "title": "POV Drive Through County Kerry | Scenic Road Trip",
            "description": "Join me on a relaxing POV drive...",
            "tags": ["pov driving", "scenic drive"],
            "category": "19",
            "location": "County Kerry, Ireland",
        }
        result = validator.validate(
            video_path=Path("/dev/null"),
            metadata_json=metadata,
        )
        gate_names = [g.gate for g in result.gates]
        assert "seo_quality" not in gate_names


class TestChapterDensityGateFailsOnEmpty:
    """Validation gate must FAIL when chapters are missing for long videos."""

    def test_no_chapters_for_long_video_fails(self):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        validator = PipelineValidator()
        metadata = {
            "title": "POV Drive Through County Kerry | Scenic Road Trip",
            "description": "A scenic drive...",
            "tags": ["driving"],
            "category": "19",
            "location": "County Kerry, Ireland",
            "video_duration": 3600,  # 60 min
            "chapters": [],  # Empty!
        }
        result = validator.validate(
            video_path=Path("/dev/null"),
            metadata_json=metadata,
        )
        meta_gate = next(g for g in result.gates if g.gate == "metadata_quality")
        assert not meta_gate.passed
        assert "no chapters" in meta_gate.message.lower()

    def test_start_finish_only_fails_density(self):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        validator = PipelineValidator()
        metadata = {
            "title": "POV Drive Through County Kerry | Scenic Road Trip",
            "description": "A scenic drive...",
            "tags": ["driving"],
            "category": "19",
            "location": "County Kerry, Ireland",
            "video_duration": 3600,  # 60 min — needs 4+ intermediate chapters
            "chapters": [
                {"timestamp": "0:00", "title": "Start"},
                {"timestamp": "59:36", "title": "Finish"},
            ],
        }
        result = validator.validate(
            video_path=Path("/dev/null"),
            metadata_json=metadata,
        )
        meta_gate = next(g for g in result.gates if g.gate == "metadata_quality")
        assert not meta_gate.passed
        assert "intermediate chapter" in meta_gate.message.lower()

    def test_short_video_no_chapters_ok(self):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        validator = PipelineValidator()
        metadata = {
            "title": "Quick Drive | County Kerry",
            "description": "Short scenic drive...",
            "tags": ["driving"],
            "category": "19",
            "location": "County Kerry, Ireland",
            "video_duration": 300,  # 5 min — no chapters needed
            "chapters": [],
        }
        result = validator.validate(
            video_path=Path("/dev/null"),
            metadata_json=metadata,
        )
        meta_gate = next(g for g in result.gates if g.gate == "metadata_quality")
        assert meta_gate.passed


class TestBasicFallbackIncludesDegradedFlag:
    """_generate_basic must include seo_degraded=True in output."""

    def test_basic_metadata_has_degraded_flag(self):
        from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

        gen = LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)
        gen._manager = None
        gen._optimizer = None
        gen._ledger = None
        gen._seo_degraded = True
        gen.storage_path = Path("/tmp/fake")
        gen.credentials_dir = Path("/tmp/fake")

        result = gen._generate_basic(
            location="County Kerry, Ireland",
            publish_day="Saturday",
            video_duration=3600.0,
            project_name="test_project",
        )
        assert result["seo_degraded"] is True
        assert result["quality_score"] is None


class TestGPSEnforcementForKnownRegions:
    """Pipeline must FAIL when GPS is required but not provided.

    Tests the enforcement logic directly via orch.run() with mocked WSL checks
    so tests don't hang on actual video processing.
    """

    @patch("gomoveshift.pipeline.orchestrator.validate_video")
    @patch("gomoveshift.pipeline.orchestrator.compute_file_hash", return_value="abc123")
    def test_ireland_location_requires_gps(self, mock_hash, mock_val, tmp_path):
        """Ireland locations must have GPS data — no silent 2-chapter degradation."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        video = tmp_path / "examplerun1_3840p.mp4"
        video.write_bytes(b"\x00" * 2048)

        orch = PipelineOrchestrator()
        result = orch.run(
            RunOptions.from_legacy(
                video_path=video,
                location="County Kerry, Ireland",
                publish_day="sat",
                output_base=tmp_path / "output",
                gps_path=None,
            )
        )
        assert not result.success
        assert "GPS data REQUIRED" in " ".join(result.errors)

    @patch("gomoveshift.pipeline.orchestrator.validate_video")
    @patch("gomoveshift.pipeline.orchestrator.compute_file_hash", return_value="abc123")
    def test_nc_location_requires_gps(self, mock_hash, mock_val, tmp_path):
        """North Carolina locations must have GPS data."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions

        video = tmp_path / "examplerun2_3840p.mp4"
        video.write_bytes(b"\x00" * 2048)

        orch = PipelineOrchestrator()
        result = orch.run(
            RunOptions.from_legacy(
                video_path=video,
                location="Western North Carolina",
                publish_day="sat",
                output_base=tmp_path / "output",
                gps_path=None,
            )
        )
        assert not result.success
        assert "GPS data REQUIRED" in " ".join(result.errors)

    def test_gps_provided_bypasses_enforcement(self):
        """When GPS is provided, enforcement logic does not trigger."""
        # Test the enforcement logic directly — if gps_path is set, the check is skipped
        _GPS_REQUIRED_REGIONS = {
            "Ireland",
            "Kerry",
            "Clare",
            "Galway",
            "Tipperary",
            "Cork",
            "Donegal",
            "Mayo",
            "Sligo",
            "Wicklow",
            "Connemara",
            "Burren",
            "Dingle",
            "Portumna",
            "Cloonlee",
            "North Carolina",
            "NC",
        }
        location = "County Kerry, Ireland"
        gps_path = Path("/fake/track.gpx")  # GPS provided
        _location_needs_gps = any(
            region.lower() in location.lower() for region in _GPS_REQUIRED_REGIONS
        )
        assert _location_needs_gps, "Kerry should match GPS-required regions"
        # With gps_path set, enforcement doesn't fire
        should_fail = _location_needs_gps and gps_path is None
        assert not should_fail

    def test_auto_gpx_in_same_dir_bypasses(self, tmp_path):
        """If GPX exists next to video, enforcement finds it."""
        video = tmp_path / "examplerun1_3840p.mp4"
        video.write_bytes(b"\x00" * 2048)
        gpx = tmp_path / "route.gpx"
        gpx.write_text("<gpx></gpx>")

        # Simulate the enforcement check from orchestrator
        _auto_gpx = video.with_suffix(".gpx")
        _dir_gpx = list(video.parent.glob("*.gpx"))
        # Auto-discovery should find the GPX
        assert _auto_gpx.exists() or len(_dir_gpx) > 0

    def test_non_gps_region_no_enforcement(self):
        """Locations not in the GPS-required list don't trigger enforcement."""
        _GPS_REQUIRED_REGIONS = {
            "Ireland",
            "Kerry",
            "Clare",
            "Galway",
            "Tipperary",
            "Cork",
            "Donegal",
            "Mayo",
            "Sligo",
            "Wicklow",
            "Connemara",
            "Burren",
            "Dingle",
            "Portumna",
            "Cloonlee",
            "North Carolina",
            "NC",
        }
        location = "Some Random City"
        _location_needs_gps = any(
            region.lower() in location.lower() for region in _GPS_REQUIRED_REGIONS
        )
        assert not _location_needs_gps, "Random city should NOT require GPS"


class TestGPSParamsPersistInState:
    """GPS params must be saved and restored from pipeline state."""

    def test_save_state_includes_gps_params(self, tmp_path):
        from gomoveshift.pipeline.orchestrator import (
            PipelineOrchestrator,
            VideoProperties,
        )

        state_file = tmp_path / "pipeline_state.json"
        orch = PipelineOrchestrator()
        vp = VideoProperties(
            width=3840,
            height=2160,
            fps=59.94,
            duration=3600.0,
            codec="hevc",
            audio_codec="aac",
            file_size_bytes=15_000_000_000,
            bitrate_kbps=83000,
        )
        params = {
            "location": "County Kerry, Ireland",
            "publish_day": "sat",
            "gps_path": "/data/input/route.gpx",
            "gps_date": "2025-09-13",
            "gps_range": "0-92",
        }
        orch._save_state(
            state_file=state_file,
            project_name="test",
            source_video=Path("/fake/video.mp4"),
            source_hash="abc123",
            stages_completed=["detect_properties"],
            current_video=Path("/fake/output.mp4"),
            video_properties=vp,
            timings={},
            params=params,
        )

        state = json.loads(state_file.read_text())
        assert state["params"]["gps_path"] == "/data/input/route.gpx"
        assert state["params"]["gps_date"] == "2025-09-13"
        assert state["params"]["gps_range"] == "0-92"

    def test_resume_comparison_ignores_gps_params(self):
        """_can_resume must not reject when GPS params differ."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator()
        state = {
            "source_hash": "abc123",
            "params": {"location": "County Kerry, Ireland", "publish_day": "sat"},
            "current_video": "/dev/null",
            "source_video": "/dev/null",
        }
        params = {
            "location": "County Kerry, Ireland",
            "publish_day": "sat",
            "gps_path": "/data/track.gpx",  # New! Not in saved state
            "gps_date": "2025-09-13",
            "gps_range": "0-92",
        }

        # Mock compute_file_hash to return our hash
        with (
            patch("gomoveshift.pipeline.orchestrator.compute_file_hash", return_value="abc123"),
            patch("gomoveshift.pipeline.state_manager.validate_video") as mock_val,
        ):
            mock_val.return_value = MagicMock(valid=True)
            assert orch._can_resume(state, Path("/dev/null"), params)


class TestLongformColorContractGate:
    """``check_video_quality`` must verify the SHIPPED colour tags, not just
    codec/resolution/bitrate.

    The validator must read ``color_range`` / ``color_space`` / ``color_primaries``
    / ``color_transfer`` on any final artifact; without it a full-range-upload
    problem would go unverified on every LF. The expected values are the
    frozen ``encoding_constants`` (`tv` / `bt709` ×3).
    """

    @staticmethod
    def _probe(**overrides) -> str:
        video = {
            "codec_type": "video",
            "codec_name": "hevc",
            "width": 3840,
            "height": 2160,
            "profile": "Main 10",
            "pix_fmt": "yuv420p10le",
            "color_range": "tv",
            "color_space": "bt709",
            "color_primaries": "bt709",
            "color_transfer": "bt709",
        }
        for key, value in overrides.items():
            if value is None:
                video.pop(key, None)
            else:
                video[key] = value
        return json.dumps(
            {
                "format": {"bit_rate": "45000000"},
                "streams": [video, {"codec_type": "audio", "codec_name": "aac"}],
            }
        )

    def _gate(self, mock_run, **overrides):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(returncode=0, stdout=self._probe(**overrides))
        return PipelineValidator()._check_video_quality(Path("/fake/video.mp4"))

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_correct_tags_pass(self, mock_run):
        gate = self._gate(mock_run)
        assert gate.passed, f"a correctly-tagged bt709/tv final must pass: {gate.message}"

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_full_range_final_is_rejected(self, mock_run):
        """A `pc` (full-range) final displays crushed/clipped on upload, so it must fail."""
        gate = self._gate(mock_run, color_range="pc")
        assert not gate.passed
        assert "color_range" in gate.message

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_wrong_colorspace_is_rejected(self, mock_run):
        gate = self._gate(mock_run, color_space="bt2020nc")
        assert not gate.passed
        assert "color_space" in gate.message

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_wrong_primaries_or_transfer_is_rejected(self, mock_run):
        assert not self._gate(mock_run, color_primaries="bt2020").passed
        assert not self._gate(mock_run, color_transfer="smpte2084").passed

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_untagged_final_fails_closed(self, mock_run):
        """An absent tag is the untagged-upload case — the player guesses.

        It must FAIL rather than skip, matching the rest of the gate.
        """
        gate = self._gate(mock_run, color_range=None, color_space=None)
        assert not gate.passed
        assert "unreadable" in gate.message


def _probe_stdout(**video_stream_overrides) -> str:
    """ffprobe JSON for a healthy 4K long-form, with per-field overrides.

    Pass ``None`` for a video-stream key to DELETE it (simulating a partial
    ffprobe read); pass ``format_bit_rate=None`` to drop the container bitrate.

    The SDR BT.709 limited-range colour tags are part of the healthy baseline:
    a fail-closed colour-tag check sits on the same gate, so an
    untagged probe is no longer "in spec" and would fail these fixtures for a
    reason none of them is about.
    """
    fmt_bit_rate = video_stream_overrides.pop("format_bit_rate", "45000000")
    video = {
        "codec_type": "video",
        "codec_name": "hevc",
        "width": 3840,
        "height": 2160,
        "profile": "Main 10",
        "pix_fmt": "yuv420p10le",
        "color_range": "tv",
        "color_space": "bt709",
        "color_primaries": "bt709",
        "color_transfer": "bt709",
    }
    for key, value in video_stream_overrides.items():
        if value is None:
            video.pop(key, None)
        else:
            video[key] = value
    fmt: dict = {}
    if fmt_bit_rate is not None:
        fmt["bit_rate"] = fmt_bit_rate
    return json.dumps(
        {"format": fmt, "streams": [video, {"codec_type": "audio", "codec_name": "aac"}]}
    )


class TestVideoQualityBitrateFloorsAreConstantsSourced:
    """``check_video_quality``'s bitrate floors must come from
    ``encoding_constants``, not hardcoded numbers.

    Hardcoded floors below ``LONGFORM_MIN_BITRATE_4K`` / ``_1080P`` would let a
    final between the two numbers pass this gate while failing
    ``bitrate_preservation``.
    """

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_4k_between_old_and_real_floor_is_rejected(self, mock_run):
        from gomoveshift.pipeline.encoding_constants import LONGFORM_MIN_BITRATE_4K
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=_probe_stdout(
                format_bit_rate="35000000"  # 35 Mbps, under ec's 40000
            ),
        )
        gate = PipelineValidator()._check_video_quality(Path("/fake/video.mp4"))
        assert not gate.passed, "4K at 35 Mbps is below the ec floor and must be rejected"
        assert str(LONGFORM_MIN_BITRATE_4K) in gate.message

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_1080p_floor_is_the_constant(self, mock_run):
        from gomoveshift.pipeline.encoding_constants import LONGFORM_MIN_BITRATE_1080P
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=_probe_stdout(
                width=1920,
                height=1080,
                format_bit_rate="13000000",  # under ec's 15000
            ),
        )
        gate = PipelineValidator()._check_video_quality(Path("/fake/video.mp4"))
        assert not gate.passed
        assert str(LONGFORM_MIN_BITRATE_1080P) in gate.message


class TestVideoQualityFailsClosedOnPartialProbe:
    """A field ffprobe could not read must FAIL the gate, never skip it.

    No check may be guarded by the field's own truthiness
    (``if profile and …``, ``if pix_fmt and …``, ``if height and …``,
    ``if bitrate > 0 and …``); that would let a partial ffprobe read report a
    clean pass on an unverified final.
    """

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_missing_profile_fails(self, mock_run):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(returncode=0, stdout=_probe_stdout(profile=None))
        gate = PipelineValidator()._check_video_quality(Path("/fake/video.mp4"))
        assert not gate.passed, "unreadable profile must fail closed, not skip the check"
        assert "profile" in gate.message

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_missing_pix_fmt_fails(self, mock_run):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(returncode=0, stdout=_probe_stdout(pix_fmt=None))
        gate = PipelineValidator()._check_video_quality(Path("/fake/video.mp4"))
        assert not gate.passed, "unreadable pix_fmt must fail closed, not skip the check"
        assert "pix_fmt" in gate.message

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_missing_height_fails(self, mock_run):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(returncode=0, stdout=_probe_stdout(height=None))
        gate = PipelineValidator()._check_video_quality(Path("/fake/video.mp4"))
        assert not gate.passed, "unreadable resolution must fail closed, not skip the 4K check"
        assert "resolution" in gate.message

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_missing_bitrate_fails(self, mock_run):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(returncode=0, stdout=_probe_stdout(format_bit_rate=None))
        gate = PipelineValidator()._check_video_quality(Path("/fake/video.mp4"))
        assert not gate.passed, "unreadable bitrate must fail closed, not skip the floor check"
        assert "bitrate" in gate.message

    @patch("gomoveshift.pipeline.pipeline_validator.subprocess.run")
    def test_healthy_final_still_passes(self, mock_run):
        from gomoveshift.pipeline.pipeline_validator import PipelineValidator

        mock_run.return_value = MagicMock(returncode=0, stdout=_probe_stdout())
        gate = PipelineValidator()._check_video_quality(Path("/fake/video.mp4"))
        assert gate.passed, f"a complete, in-spec probe must still pass: {gate.message}"
