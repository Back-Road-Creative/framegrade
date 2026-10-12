"""Temp per-span LUT dir is cleaned up after the encode.

The CPU adaptive path bakes one .cube per lighting span into a temp dir and
returns it as ``temp_lut``; ``process`` must remove that dir once the encode
finishes — success or failure. (This was the libplacebo GPU path's cleanup
regression net; it now guards the only temp-LUT artifact that survives.)
"""

from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.single_pass import SinglePassProcessor


def _proc():
    return SinglePassProcessor(Config())


_VIDEO_INFO = {"duration": 60, "height": 1080, "width": 1920, "fps": 30.0}


def test_temp_lut_dir_cleaned_up_after_process(tmp_path):
    """Temp LUT dir should not exist after process() completes."""
    proc = _proc()
    lut_dir = tmp_path / "gms_span_luts_test"
    lut_dir.mkdir()
    (lut_dir / "span_000.cube").write_text("TITLE test\nLUT_3D_SIZE 2\n0 0 0\n")

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"\x00")
    fake_output = tmp_path / "out.mp4"

    with (
        patch.object(
            proc,
            "_run_color_step",
            return_value=("lut3d='span'", "auto_adaptive", lut_dir),
        ),
        patch.object(proc, "_run_audio_step", return_value={}),
        patch.object(proc, "_build_video_filter", return_value="null"),
        patch.object(
            proc,
            "_run_encode_step",
            return_value=MagicMock(
                success=True,
                encoding_time=10.0,
                color_method="auto_adaptive",
                total_time=15.0,
                output_path=fake_output,
            ),
        ),
        patch.object(proc, "_get_video_info", return_value=_VIDEO_INFO),
        patch.object(proc, "_has_nvenc", return_value=False),
        patch.object(proc, "_find_font", return_value=None),
    ):
        proc.process(video_path=fake_video, output_path=fake_output, location="Test")

    assert not lut_dir.exists(), "Temp LUT dir should be cleaned up"


def test_temp_lut_dir_cleaned_up_on_encode_failure(tmp_path):
    """Temp LUT dir should be cleaned up even if the encode raises."""
    proc = _proc()
    lut_dir = tmp_path / "gms_span_luts_fail"
    lut_dir.mkdir()
    (lut_dir / "span_000.cube").write_text("TITLE test\nLUT_3D_SIZE 2\n0 0 0\n")

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"\x00")
    fake_output = tmp_path / "out.mp4"

    with (
        patch.object(
            proc,
            "_run_color_step",
            return_value=("null", "auto_adaptive", lut_dir),
        ),
        patch.object(proc, "_run_audio_step", side_effect=RuntimeError("boom")),
        patch.object(proc, "_get_video_info", return_value=_VIDEO_INFO),
        patch.object(proc, "_has_nvenc", return_value=False),
        pytest.raises(RuntimeError, match="boom"),
    ):
        proc.process(video_path=fake_video, output_path=fake_output, location="Test")

    assert not lut_dir.exists(), "Temp LUT dir should be cleaned up on failure"
