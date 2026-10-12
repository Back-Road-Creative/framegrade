"""Privacy-blur WIRING into the stage-9 encode.

When ``blur_mask_path`` (the blur mask) reaches ``process``, the encode swaps
its bare ``-vf`` for a ``-filter_complex`` that takes the mask as a second
input and maps the blurred composite ``[vout]`` to output — NVDEC hand-off
preserved, audio mapped straight off input 0. No mask → the legacy ``-vf``
path is byte-for-byte unchanged.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.single_pass import SinglePassProcessor, SinglePassResult

MASK = Path("/fake/mask.mkv")
FC = (
    "[0:v]scale=in_range=full:out_range=limited[base];[base]split[b1][b2];"
    "[b2]boxblur=24:2[bl];[1:v]format=gray[m];[bl][m]alphamerge[bla];[b1][bla]overlay[vout]"
)


@pytest.fixture
def proc():
    return SinglePassProcessor(Config())


def _capture_cmd(
    processor,
    *,
    vf="null",
    filter_complex=None,
    blur_mask_path=None,
    skip_audio=True,
    pix_fmt="yuv420p",
):
    """Drive ``_run_encode_step`` with ffmpeg mocked; return the assembled cmd."""
    cap = {}
    val = MagicMock(valid=True, reason="", file_size=10**8, video_codec="hevc", duration=60.0)
    with (
        patch(
            "gomoveshift.pipeline.single_pass.run_ffmpeg_encode",
            side_effect=lambda cmd, **_: (
                cap.update(cmd=list(cmd)) or MagicMock(success=True, stderr_tail="", stdout="")
            ),
        ),
        patch("gomoveshift.pipeline.single_pass.validate_video", return_value=val),
        patch.object(SinglePassProcessor, "_calc_encode_timeout", return_value=3600),
    ):
        processor._run_encode_step(
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
            skip_audio=skip_audio,
            skip_ost=True,
            use_nvenc=True,
            info={
                "width": 3840,
                "height": 2160,
                "fps": 30.0,
                "bitrate_kbps": 70000,
                "pix_fmt": pix_fmt,
            },
            duration=60.0,
            font=None,
            filter_complex=filter_complex,
            blur_mask_path=blur_mask_path,
        )
    assert "cmd" in cap, "_run_encode_step did not invoke run_ffmpeg_encode"
    return cap["cmd"]


class TestEncodeCommandBlur:
    def test_filter_complex_replaces_vf_with_mask_input(self, proc):
        cmd = _capture_cmd(proc, filter_complex=FC, blur_mask_path=MASK)
        assert "-filter_complex" in cmd and "-vf" not in cmd
        inputs = [cmd[k + 1] for k, a in enumerate(cmd) if a == "-i"]
        assert inputs == ["/fake/input.mp4", str(MASK)]  # mask is input 1
        assert cmd[cmd.index("-filter_complex") + 1].rstrip().endswith("overlay[vout]")

    def test_maps_composite_and_audio(self, proc):
        cmd = _capture_cmd(proc, filter_complex=FC, blur_mask_path=MASK, skip_audio=False)
        assert [cmd[k + 1] for k, a in enumerate(cmd) if a == "-map"] == ["[vout]", "0:a"]

    def test_no_audio_map_when_skip_audio(self, proc):
        cmd = _capture_cmd(proc, filter_complex=FC, blur_mask_path=MASK, skip_audio=True)
        assert [cmd[k + 1] for k, a in enumerate(cmd) if a == "-map"] == ["[vout]"]
        assert "-an" in cmd

    @pytest.mark.parametrize("pix,fmt", [("yuv420p", "nv12"), ("yuv420p10le", "p010le")])
    def test_nvdec_injects_hwdownload(self, proc, pix, fmt):
        assert proc.config.drive.use_nvdec_decode is True  # default
        cmd = _capture_cmd(proc, filter_complex=FC, blur_mask_path=MASK, pix_fmt=pix)
        injected = cmd[cmd.index("-filter_complex") + 1]
        assert injected.startswith(f"[0:v]hwdownload,format={fmt},")
        assert cmd[cmd.index("-hwaccel") + 1] == "cuda"

    def test_nvdec_off_leaves_graph_untouched(self):
        cfg = Config()
        cfg.drive.use_nvdec_decode = False
        cmd = _capture_cmd(SinglePassProcessor(cfg), filter_complex=FC, blur_mask_path=MASK)
        assert cmd[cmd.index("-filter_complex") + 1] == FC
        assert "-hwaccel" not in cmd

    def test_no_blur_path_unchanged(self, proc):
        cmd = _capture_cmd(proc, vf="curves=x,scale=in_range=full:out_range=limited")
        assert "-filter_complex" not in cmd and "-vf" in cmd
        assert len([a for a in cmd if a == "-i"]) == 1 and "-map" not in cmd


class TestProcessThreadsMask:
    def _run(self, processor, blur_mask_path, out_path):
        cap = {}

        def fake_encode(*a, **kw):
            cap.update(filter_complex=kw.get("filter_complex"), mask=kw.get("blur_mask_path"))
            return SinglePassResult(success=True, output_path=out_path)

        with (
            patch.object(processor, "_run_color_step", return_value=("null", "skip", None)),
            patch.object(processor, "_run_audio_step", return_value={"input_i": -20.0}),
            patch.object(processor, "_run_encode_step", side_effect=fake_encode),
            patch.object(processor, "_has_nvenc", return_value=True),
            patch.object(processor, "_find_font", return_value=None),
        ):
            processor.process(
                video_path=Path("/fake/in.mp4"),
                output_path=out_path,
                location="County Kerry, Ireland",
                skip_color=True,
                skip_ost=True,
                skip_audio=True,
                video_info={
                    "duration": 600.0,
                    "height": 2160,
                    "width": 3840,
                    "fps": 30.0,
                    "bitrate_kbps": 70000,
                },
                blur_mask_path=blur_mask_path,
            )
        return cap

    def test_mask_set_builds_filter_complex(self, proc, tmp_path):
        cap = self._run(proc, MASK, tmp_path / "out.mp4")
        assert cap["mask"] == MASK and "alphamerge" in cap["filter_complex"]

    def test_mask_none_keeps_legacy_path(self, proc, tmp_path):
        cap = self._run(proc, None, tmp_path / "out.mp4")
        assert cap["mask"] is None and cap["filter_complex"] is None
