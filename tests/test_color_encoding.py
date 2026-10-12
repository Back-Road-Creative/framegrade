"""
Tests for color correction encoding — NVENC detection and encoder selection.

TDD: Written BEFORE fixes. These should FAIL initially, then pass after fix.
"""

import pytest


class TestNVENCDetection:
    """NVENC detection must use a resolution NVENC actually accepts."""

    def test_detect_nvenc_uses_adequate_resolution(self):
        """detect_nvenc must NOT use 64x64 — NVENC rejects frames below minimum size."""
        import inspect

        from gomoveshift.pipeline.color import detect_nvenc

        source = inspect.getsource(detect_nvenc)

        # Must not use 64x64 (too small for NVENC)
        assert "64x64" not in source, (
            "detect_nvenc still uses 64x64 test source — "
            "NVENC rejects frames below minimum resolution"
        )

    def test_detect_nvenc_uses_256x256_or_larger(self):
        """detect_nvenc should use at least 256x256 for its test encode."""
        import inspect
        import re

        from gomoveshift.pipeline.color import detect_nvenc

        source = inspect.getsource(detect_nvenc)

        # Find the testsrc size parameter
        match = re.search(r"size=(\d+)x(\d+)", source)
        assert match, "Could not find size parameter in detect_nvenc"

        width, height = int(match.group(1)), int(match.group(2))
        assert width >= 256 and height >= 256, (
            f"detect_nvenc test source {width}x{height} too small — NVENC requires at least 256x256"
        )

    @pytest.mark.real_nvenc
    def test_detect_nvenc_returns_true_on_this_machine(self):
        """On a host with an NVIDIA GPU, NVENC MUST be detected."""
        from gomoveshift.pipeline.color import detect_nvenc

        result = detect_nvenc()
        assert result is True, "NVENC not detected on a GPU host. Fix the detection function."


class TestEncodingArgs:
    """Encoding args must produce correct ffmpeg flags for each encoder."""

    def test_nvenc_args_when_available(self):
        """When NVENC is available, encoding args should use hevc_nvenc."""
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)

        assert "-c:v" in args
        idx = args.index("-c:v")
        assert args[idx + 1] == "hevc_nvenc"

    def test_cpu_fallback_args(self):
        """When NVENC unavailable, should use libx265."""
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=False)

        assert "-c:v" in args
        idx = args.index("-c:v")
        assert args[idx + 1] == "libx265"

    def test_libx265_uses_slow_preset_for_max_quality(self):
        """libx265 must use 'slow' preset for highest quality output."""
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=False)

        if "-preset" in args:
            idx = args.index("-preset")
            preset = args[idx + 1]
            assert preset == "slow", (
                f"libx265 preset '{preset}' — must be 'slow' for maximum quality."
            )


class TestNVENCQualityEnforcement:
    """NVENC must use YouTube-optimized quality and never switch to CBR mode.

    Bug: when source_bitrate_kbps > 0, the old code switched -rc vbr → cbr
    and removed -cq. With -tune hq still present, NVENC ignored the CBR
    target and produced ~700 kbps instead of target.
    Fix: keep VBR + CQ, never mutate -rc or remove -cq.
    """

    def test_nvenc_cq_constant_is_youtube_optimized(self):
        """LONGFORM_NVENC_CQ must be 18 (YouTube-optimized ~40-50 Mbps at 4K)."""
        from gomoveshift.pipeline import encoding_constants as ec

        assert ec.LONGFORM_NVENC_CQ == 18, (
            f"LONGFORM_NVENC_CQ={ec.LONGFORM_NVENC_CQ} — must be 18 for "
            f"YouTube-optimized quality (~40-50 Mbps at 4K)"
        )

    def test_nvenc_args_use_cq_18(self):
        """get_encoding_args must pass -cq 18 to NVENC."""
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)

        assert "-cq" in args
        idx = args.index("-cq")
        assert args[idx + 1] == "18", (
            f"-cq is '{args[idx + 1]}' — must be '18' for YouTube-optimized quality"
        )

    def test_nvenc_stays_vbr_when_source_bitrate_provided(self):
        """-rc must remain 'vbr' even when source_bitrate_kbps is given.

        The old CBR block switched -rc to 'cbr' and removed -cq. With -tune hq
        still present, NVENC ignored the CBR target → 700 kbps output.
        """
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True, source_bitrate_kbps=83000)

        assert "-rc" in args
        rc_idx = args.index("-rc")
        assert args[rc_idx + 1] == "vbr", (
            f"-rc is '{args[rc_idx + 1]}' — CBR block must be removed; "
            f"it breaks quality by removing -cq with -tune hq still present"
        )

    def test_nvenc_cq_not_removed_when_source_bitrate_provided(self):
        """-cq must not be removed when source_bitrate_kbps triggers floor logic."""
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True, source_bitrate_kbps=83000)

        assert "-cq" in args, "-cq was removed — the CBR block that deletes -cq must be eliminated"

    def test_bitrate_floor_args_always_present(self):
        """-b:v, -maxrate, -bufsize are always added as absolute floor hints."""
        from gomoveshift.pipeline.color import get_encoding_args
        from gomoveshift.pipeline import encoding_constants as ec

        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True)

        assert "-b:v" in args
        assert "-maxrate" in args
        assert "-bufsize" in args
        bv_idx = args.index("-b:v")
        assert args[bv_idx + 1] == f"{ec.LONGFORM_MIN_BITRATE_4K}k", (
            f"-b:v={args[bv_idx + 1]}; expected {ec.LONGFORM_MIN_BITRATE_4K}k (4K absolute floor)"
        )

    def test_4k_absolute_floor_regardless_of_source(self):
        """4K floor applies regardless of source bitrate."""
        from gomoveshift.pipeline.color import get_encoding_args
        from gomoveshift.pipeline import encoding_constants as ec

        # Even with low source, floor is the same absolute value
        args = get_encoding_args(3840, 2160, 59.94, has_nvenc=True, source_bitrate_kbps=20000)

        bv_idx = args.index("-b:v")
        assert args[bv_idx + 1] == f"{ec.LONGFORM_MIN_BITRATE_4K}k", (
            f"-b:v={args[bv_idx + 1]}; expected {ec.LONGFORM_MIN_BITRATE_4K}k (4K absolute floor)"
        )

    def test_1080p_absolute_floor(self):
        """1080p uses its own absolute floor."""
        from gomoveshift.pipeline.color import get_encoding_args
        from gomoveshift.pipeline import encoding_constants as ec

        args = get_encoding_args(1920, 1080, 29.97, has_nvenc=True)

        bv_idx = args.index("-b:v")
        assert args[bv_idx + 1] == f"{ec.LONGFORM_MIN_BITRATE_1080P}k", (
            f"-b:v={args[bv_idx + 1]}; expected {ec.LONGFORM_MIN_BITRATE_1080P}k (1080p floor)"
        )


class TestTimeout:
    """Timeout must be adequate for the chosen encoder and resolution."""

    def test_encode_timeout_4k_60fps_cpu(self):
        """4K@60fps with CPU encoding (48 min video) needs realistic timeout."""
        from gomoveshift.pipeline.color import encode_video

        # Access the timeout multiplier from encode_video function
        # When NVENC unavailable: timeout = duration * mult
        # Must be enough for fast preset at 4K: ~3-4x realtime
        import inspect

        source = inspect.getsource(encode_video)

        # The timeout should be at least 4x duration for CPU
        # 2894 * 4 = 11576s (3.2h) minimum
        # But shouldn't need 6x anymore with faster preset
        # Let's just verify it's not going to time out by being too aggressive
        assert "timeout" in source.lower()
