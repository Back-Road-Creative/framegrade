"""The GOP log line must report the GOP actually emitted.

``encode_video`` logged ``GOP: round(fps * 2) frames (2s)`` while
``get_encoding_args`` emits a FIXED ``-g 60`` — every output is normalised to
29.97 fps, so 60 frames really is 2 seconds regardless of the source. For any
non-29.97 source the log therefore stated a GOP the encoder never used: a
59.94 fps source was reported as 120.

The log line now reads the ``-g`` value back out of the arguments being passed
to ffmpeg, so it cannot drift from the emitted arg again.
"""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline import color_filters


LOGGER_NAME = "gomoveshift.pipeline.color_filters"


def _emitted_gop(cmd: list[str]) -> str:
    """The -g value ffmpeg actually received."""
    assert "-g" in cmd, f"no -g in emitted command: {cmd}"
    return cmd[cmd.index("-g") + 1]


def _run_encode(caplog, tmp_path, *, fps: float, has_nvenc: bool):
    vid = tmp_path / "in.mp4"
    vid.write_bytes(b"\x00")
    out = tmp_path / "out.mp4"
    captured: dict[str, list[str]] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return SimpleNamespace(success=True, stderr_tail="")

    with patch.object(color_filters, "run_ffmpeg_encode", side_effect=_fake_run):
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            color_filters.encode_video(
                video_path=vid,
                output_path=out,
                filter_str="null",
                width=3840,
                height=2160,
                fps=fps,
                duration=10.0,
                has_nvenc=has_nvenc,
            )

    gop_lines = [
        r.getMessage() for r in caplog.records if r.name == LOGGER_NAME and "GOP:" in r.getMessage()
    ]
    assert gop_lines, f"no GOP log line emitted; records={[r.getMessage() for r in caplog.records]}"
    return captured["cmd"], gop_lines[0]


@pytest.mark.parametrize("fps", [29.97, 59.94, 25.0, 23.976, 120.0])
@pytest.mark.parametrize("has_nvenc", [False, True])
def test_gop_log_matches_the_emitted_arg(caplog, tmp_path, fps, has_nvenc):
    """Whatever the source fps, the logged GOP is the one ffmpeg was given."""
    cmd, line = _run_encode(caplog, tmp_path, fps=fps, has_nvenc=has_nvenc)

    logged = re.search(r"GOP:\s*(\d+)\s*frames", line)
    assert logged, f"could not parse GOP frames from log line: {line!r}"
    assert logged.group(1) == _emitted_gop(cmd), (
        f"log claims GOP {logged.group(1)} but ffmpeg was given "
        f"-g {_emitted_gop(cmd)} (source fps={fps})"
    )


def test_high_fps_source_is_not_reported_as_double(caplog, tmp_path):
    """The regression this guards: 59.94 fps used to log 120."""
    _, line = _run_encode(caplog, tmp_path, fps=59.94, has_nvenc=False)
    assert "120" not in line, f"GOP log still derived from source fps: {line!r}"
    assert "60" in line


def test_gop_log_states_the_real_output_cadence(caplog, tmp_path):
    """Output is normalised to 29.97 fps, so 60 frames is genuinely 2s.

    The '(2s)' claim is only true because of that normalisation — pin it
    alongside the frame count so the two cannot disagree.
    """
    cmd, line = _run_encode(caplog, tmp_path, fps=59.94, has_nvenc=False)
    assert "30000/1001" in cmd, "output fps normalisation disappeared from the encode args"
    assert "2s" in line
