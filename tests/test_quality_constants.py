"""
Tests for encoding constants — verify frozen quality params are exported
and that get_encoding_args() ignores config values.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEncodingConstantsExist:
    """encoding_constants.py must export frozen LONGFORM_* and SHORTS_* constants."""

    def test_longform_crf(self):
        from gomoveshift.pipeline.encoding_constants import LONGFORM_CRF

        assert LONGFORM_CRF == 18, f"YouTube-optimized CRF should be 18, got {LONGFORM_CRF}"

    def test_longform_preset(self):
        from gomoveshift.pipeline.encoding_constants import LONGFORM_PRESET

        assert LONGFORM_PRESET == "slow"

    def test_longform_nvenc_preset(self):
        from gomoveshift.pipeline.encoding_constants import LONGFORM_NVENC_PRESET

        assert LONGFORM_NVENC_PRESET == "p7"

    def test_longform_nvenc_cq(self):
        """LONGFORM_NVENC_CQ must be 18 (YouTube-optimized)."""
        from gomoveshift.pipeline.encoding_constants import LONGFORM_NVENC_CQ

        assert LONGFORM_NVENC_CQ == 18, (
            f"YouTube-optimized CQ should be 18, got {LONGFORM_NVENC_CQ}"
        )

    def test_longform_profile(self):
        from gomoveshift.pipeline.encoding_constants import LONGFORM_PROFILE

        assert LONGFORM_PROFILE == "main10"

    def test_longform_pix_fmt(self):
        from gomoveshift.pipeline.encoding_constants import LONGFORM_PIX_FMT

        assert LONGFORM_PIX_FMT == "yuv420p10le"

    def test_longform_audio_bitrate(self):
        from gomoveshift.pipeline.encoding_constants import LONGFORM_AUDIO_BITRATE

        assert LONGFORM_AUDIO_BITRATE == "384k"

    def test_longform_audio_lufs(self):
        """Near-silent policy: LF attenuate-only ceiling is -45.0 LUFS (attenuate, never boost)."""
        from gomoveshift.pipeline.encoding_constants import LONGFORM_AUDIO_LUFS

        assert LONGFORM_AUDIO_LUFS == -45.0, (
            f"Near-silent LF ceiling should be -45.0, got {LONGFORM_AUDIO_LUFS}"
        )

    def test_longform_audio_tp(self):
        """True-peak cap after attenuation is -10.0 dBTP (near-silent policy)."""
        from gomoveshift.pipeline.encoding_constants import LONGFORM_AUDIO_TP

        assert LONGFORM_AUDIO_TP == -10.0, (
            f"LF true-peak cap should be -10.0, got {LONGFORM_AUDIO_TP}"
        )

    def test_longform_audio_max_lufs(self):
        """Validation ceiling: reject an LF LOUDER than -43.0 LUFS (2 dB slack)."""
        from gomoveshift.pipeline.encoding_constants import LONGFORM_AUDIO_MAX_LUFS

        assert LONGFORM_AUDIO_MAX_LUFS == -43.0, (
            f"LF loudness gate ceiling should be -43.0, got {LONGFORM_AUDIO_MAX_LUFS}"
        )

    def test_shorts_audio_lufs(self):
        """Shorts ship muted; the constant tracks the near-silent ceiling for
        gate/constant parity."""
        from gomoveshift.pipeline.encoding_constants import SHORTS_AUDIO_LUFS

        assert SHORTS_AUDIO_LUFS == -45.0, (
            f"Shorts audio LUFS should be -45.0, got {SHORTS_AUDIO_LUFS}"
        )

    def test_audio_silence_ceiling_lufs(self):
        """New near-silent integrated ceiling — attenuate-only, never boost above."""
        from gomoveshift.pipeline.encoding_constants import AUDIO_SILENCE_CEILING_LUFS

        assert AUDIO_SILENCE_CEILING_LUFS == -45.0, (
            f"Silence ceiling should be -45.0, got {AUDIO_SILENCE_CEILING_LUFS}"
        )

    def test_audio_true_peak_cap(self):
        """New true-peak hard cap applied after attenuation."""
        from gomoveshift.pipeline.encoding_constants import AUDIO_TRUE_PEAK_CAP

        assert AUDIO_TRUE_PEAK_CAP == -10.0, (
            f"True-peak cap should be -10.0, got {AUDIO_TRUE_PEAK_CAP}"
        )

    def test_shorts_crf(self):
        """Shorts CQ matches long-form (18) — 8-bit CQ 23 banded graded skies."""
        from gomoveshift.pipeline.encoding_constants import LONGFORM_NVENC_CQ, SHORTS_CRF

        assert SHORTS_CRF == 18, f"Shorts CQ should be 18, got {SHORTS_CRF}"
        assert SHORTS_CRF == LONGFORM_NVENC_CQ, "Shorts CQ must match long-form CQ"

    def test_shorts_preset(self):
        """Shorts NVENC preset matches long-form p7 (shorts are NVENC-only)."""
        from gomoveshift.pipeline.encoding_constants import SHORTS_PRESET

        assert SHORTS_PRESET == "p7"

    def test_shorts_profile(self):
        """10-bit main10 — matches the graded master's depth."""
        from gomoveshift.pipeline.encoding_constants import SHORTS_PROFILE

        assert SHORTS_PROFILE == "main10"

    def test_shorts_pix_fmt(self):
        """p010le — NVENC 10-bit input surface, same as the long-form NVENC path."""
        from gomoveshift.pipeline.encoding_constants import SHORTS_PIX_FMT

        assert SHORTS_PIX_FMT == "p010le"

    def test_shorts_dimensions(self):
        from gomoveshift.pipeline.encoding_constants import SHORTS_HEIGHT, SHORTS_WIDTH

        assert SHORTS_WIDTH == 1080
        assert SHORTS_HEIGHT == 1920

    def test_shorts_audio_bitrate(self):
        from gomoveshift.pipeline.encoding_constants import SHORTS_AUDIO_BITRATE

        assert SHORTS_AUDIO_BITRATE == "128k"


class TestGetEncodingArgsIgnoresConfig:
    """get_encoding_args() must use constants, not config values."""

    def test_cpu_crf_uses_constant(self):
        from gomoveshift.pipeline.color import get_encoding_args
        from gomoveshift.pipeline.encoding_constants import LONGFORM_CRF

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=False)
        idx = args.index("-crf")
        assert args[idx + 1] == str(LONGFORM_CRF), (
            f"CRF should be '{LONGFORM_CRF}' from constants, got '{args[idx + 1]}'"
        )

    def test_nvenc_cq_uses_constant(self):
        from gomoveshift.pipeline.color import get_encoding_args
        from gomoveshift.pipeline.encoding_constants import LONGFORM_NVENC_CQ

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)
        idx = args.index("-cq")
        assert args[idx + 1] == str(LONGFORM_NVENC_CQ), (
            f"CQ should be '{LONGFORM_NVENC_CQ}' from constants, got '{args[idx + 1]}'"
        )

    def test_cpu_preset_uses_constant(self):
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=False)
        idx = args.index("-preset")
        assert args[idx + 1] == "slow"

    def test_nvenc_preset_uses_constant(self):
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)
        idx = args.index("-preset")
        assert args[idx + 1] == "p7"


class TestOutputFpsNormalization:
    """get_encoding_args() must always output at 29.97fps regardless of source fps."""

    def test_nvenc_output_fps_normalized(self):
        from gomoveshift.pipeline.color import get_encoding_args

        # 60fps source must produce 29.97fps output
        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)
        assert "-r" in args, "-r flag missing from NVENC encoding args"
        r_idx = args.index("-r")
        assert args[r_idx + 1] == "30000/1001", (
            f"NVENC output fps should be 30000/1001, got '{args[r_idx + 1]}'"
        )

    def test_cpu_output_fps_normalized(self):
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=False)
        assert "-r" in args, "-r flag missing from CPU encoding args"
        r_idx = args.index("-r")
        assert args[r_idx + 1] == "30000/1001", (
            f"CPU output fps should be 30000/1001, got '{args[r_idx + 1]}'"
        )

    def test_gop_size_based_on_output_fps(self):
        from gomoveshift.pipeline.color import get_encoding_args

        # GOP must be 60 (2s at 29.97fps output), not 120 (2s at 59.94fps source)
        args_nvenc = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)
        g_idx = args_nvenc.index("-g")
        assert args_nvenc[g_idx + 1] == "60", (
            f"GOP should be 60 for 29.97fps output, got '{args_nvenc[g_idx + 1]}'"
        )

    def test_29fps_source_unchanged(self):
        from gomoveshift.pipeline.color import get_encoding_args

        # 29.97fps source stays at 29.97fps — no redundant frame drop
        args = get_encoding_args(3840, 2160, 29.97, has_nvenc=True)
        assert "-r" in args
        r_idx = args.index("-r")
        assert args[r_idx + 1] == "30000/1001"


class TestFastFlagRemoved:
    """--fast flag must be removed (replaced by --draft)."""

    def test_get_encoding_args_has_no_fast_param(self):
        import inspect

        from gomoveshift.pipeline.color import get_encoding_args

        sig = inspect.signature(get_encoding_args)
        assert "fast" not in sig.parameters, "get_encoding_args still has 'fast' parameter"

    def test_single_pass_processor_has_no_fast_param(self):
        import inspect

        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        sig = inspect.signature(SinglePassProcessor.__init__)
        assert "fast" not in sig.parameters, "SinglePassProcessor still has 'fast' parameter"

    def test_orchestrator_run_has_draft_not_fast(self):
        import inspect

        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        sig = inspect.signature(PipelineOrchestrator().run)
        assert "fast" not in sig.parameters, "orchestrator.run still has 'fast' parameter"
        assert "draft" in sig.parameters, "orchestrator.run missing 'draft' parameter"
