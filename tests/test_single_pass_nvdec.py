"""NVDEC opt-in path for ``SinglePassProcessor._run_encode_step``."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.single_pass import SinglePassProcessor, SinglePassResult


def _capture_encode_cmd(processor, *, vf: str, use_nvenc: bool = True, pix_fmt: str = "yuv420p"):
    """Run ``_run_encode_step`` with ``run_ffmpeg_encode`` mocked; return the cmd list."""
    captured = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = list(cmd)
        return MagicMock(success=True, stderr_tail="", stdout="")

    fake_validation = MagicMock(
        valid=True,
        reason="",
        file_size=1024 * 1024 * 100,  # 100 MB
        video_codec="hevc",
        duration=60.0,
    )

    with (
        patch("gomoveshift.pipeline.single_pass.run_ffmpeg_encode", side_effect=fake_run),
        patch(
            "gomoveshift.pipeline.single_pass.validate_video",
            return_value=fake_validation,
        ),
        patch(
            "gomoveshift.pipeline.single_pass.SinglePassProcessor._calc_encode_timeout",
            return_value=3600,
        ),
    ):
        result = processor._run_encode_step(
            video_path=Path("/fake/input.mp4"),
            output_path=Path("/fake/out.mp4"),
            vf=vf,
            measured={
                "input_i": -20.0,
                "input_tp": -1.5,
                "input_lra": 11.0,
                "input_thresh": -30.0,
                "target_offset": 0.0,
            },
            skip_audio=True,
            skip_ost=True,
            use_nvenc=use_nvenc,
            info={
                "width": 3840,
                "height": 2160,
                "fps": 30.0,
                "bitrate_kbps": 70000,
                "pix_fmt": pix_fmt,
            },
            duration=60.0,
            font=None,
        )
    assert isinstance(result, SinglePassResult)
    assert "cmd" in captured, "_run_encode_step did not invoke run_ffmpeg_encode"
    return captured["cmd"]


@pytest.fixture
def processor_default():
    """Explicit opt-out: use_nvdec_decode=False (regression guard for the
    legacy CPU-decode path). The package default is True;
    these tests still exercise the off-branch so operators who set
    ``use_nvdec_decode: false`` in user config keep working behavior."""
    cfg = Config()
    cfg.drive.use_nvdec_decode = False
    return SinglePassProcessor(cfg)


@pytest.fixture
def processor_nvdec_on():
    """Package default: use_nvdec_decode=True."""
    cfg = Config()
    assert cfg.drive.use_nvdec_decode is True, "package default for use_nvdec_decode should be True"
    return SinglePassProcessor(cfg)


CPU_VF = "curves=m='0/0.03 1/0.98',hqdn3d=2.0:2.0:1.5:1.5,scale=in_range=full:out_range=limited"


class TestNvdecOptInOff:
    def test_no_hwaccel_when_off_and_filters_present(self, processor_default):
        """Default (off): CPU filter chain → no -hwaccel, no hwdownload prefix."""
        cmd = _capture_encode_cmd(processor_default, vf=CPU_VF)
        assert "-hwaccel" not in cmd
        # vf is untouched (no hwdownload prefix)
        vf_idx = cmd.index("-vf")
        assert cmd[vf_idx + 1] == CPU_VF

    def test_legacy_nullvf_hwaccel_preserved(self, processor_default):
        """Legacy fast-path for null vf still adds -hwaccel cuda (regression guard)."""
        cmd = _capture_encode_cmd(processor_default, vf="null")
        assert cmd[cmd.index("-hwaccel") + 1] == "cuda"
        # No -hwaccel_output_format on the legacy path
        assert "-hwaccel_output_format" not in cmd


class TestNvdecOptInOn:
    def test_hwaccel_cuda_with_output_format(self, processor_nvdec_on):
        """Opt-in + filters: -hwaccel cuda -hwaccel_output_format cuda added before -i."""
        cmd = _capture_encode_cmd(processor_nvdec_on, vf=CPU_VF)
        ha_idx = cmd.index("-hwaccel")
        assert cmd[ha_idx + 1] == "cuda"
        of_idx = cmd.index("-hwaccel_output_format")
        assert cmd[of_idx + 1] == "cuda"
        # Both must precede -i
        i_idx = cmd.index("-i")
        assert ha_idx < i_idx and of_idx < i_idx

    def test_hwdownload_8bit_uses_nv12(self, processor_nvdec_on):
        """8-bit source (yuv420p): hwdownload format is nv12."""
        cmd = _capture_encode_cmd(processor_nvdec_on, vf=CPU_VF, pix_fmt="yuv420p")
        vf_idx = cmd.index("-vf")
        effective_vf = cmd[vf_idx + 1]
        assert effective_vf.startswith("hwdownload,format=nv12,")
        # Original chain still intact after the prefix
        assert effective_vf.endswith(CPU_VF)

    def test_hwdownload_10bit_uses_p010le(self, processor_nvdec_on):
        """10-bit source (yuv420p10le, GoPro Main 10): hwdownload format is
        p010le — NOT nv12 (the bug that aborted every 10-bit day master with
        '[hwdownload] Invalid output format nv12 for hwframe download')."""
        cmd = _capture_encode_cmd(processor_nvdec_on, vf=CPU_VF, pix_fmt="yuv420p10le")
        vf_idx = cmd.index("-vf")
        effective_vf = cmd[vf_idx + 1]
        assert effective_vf.startswith("hwdownload,format=p010le,")
        # No bare nv12 alternation that ffmpeg constrains to 8-bit
        assert "nv12" not in effective_vf.split(",")[1]
        assert effective_vf.endswith(CPU_VF)

    def test_hwdownload_p010le_pixfmt_uses_p010le(self, processor_nvdec_on):
        """A source already reporting pix_fmt=p010le → p010le download."""
        cmd = _capture_encode_cmd(processor_nvdec_on, vf=CPU_VF, pix_fmt="p010le")
        vf_idx = cmd.index("-vf")
        assert cmd[vf_idx + 1].startswith("hwdownload,format=p010le,")

    def test_nvenc_disabled_no_nvdec(self, processor_nvdec_on):
        """Opt-in but use_nvenc=False → no NVDEC either (encoder is software)."""
        cmd = _capture_encode_cmd(processor_nvdec_on, vf=CPU_VF, use_nvenc=False)
        assert "-hwaccel" not in cmd

    def test_null_vf_uses_legacy_fastpath_not_output_format(self, processor_nvdec_on):
        """vf=null with NVDEC opt-in still uses the legacy `-hwaccel cuda` shape
        (no `-hwaccel_output_format`) — frames go straight encoder, no filter chain
        to hand them off to."""
        cmd = _capture_encode_cmd(processor_nvdec_on, vf="null")
        assert cmd[cmd.index("-hwaccel") + 1] == "cuda"
        assert "-hwaccel_output_format" not in cmd


class TestHwdownloadFormatForPixFmt:
    """Pure bit-depth → hwdownload format mapping (robust by construction)."""

    @pytest.mark.parametrize(
        "pix_fmt,expected",
        [
            ("yuv420p10le", "p010le"),
            ("yuv422p10le", "p010le"),
            ("p010le", "p010le"),
            ("yuv420p12le", "p010le"),  # >8-bit → 10-bit surface
            ("yuv420p", "nv12"),
            ("nv12", "nv12"),
            ("yuvj420p", "nv12"),
            ("", "nv12"),  # unknown → 8-bit default
            (None, "nv12"),
        ],
    )
    def test_format_for_pix_fmt(self, pix_fmt, expected):
        from gomoveshift.pipeline.single_pass import _hwdownload_format_for_pix_fmt

        assert _hwdownload_format_for_pix_fmt(pix_fmt) == expected
