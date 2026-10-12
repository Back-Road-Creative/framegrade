"""Shared time-lapse core: streaming stills/frames to ffmpeg, daylight stills
timelapse, and video frame-decimation. ``test_astro_timelapse.py`` covers the
night-specific layer built on top of this module; this file covers the
generic core alone."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import gomoveshift.pipeline.timelapse as timelapse
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser


def _write_png(path, color, size=(16, 12)):
    Image.new("RGB", size, color=color).save(path)


def test_render_stills_timelapse_encodes_a_real_clip_in_capture_order(tmp_path):
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        _write_png(tmp_path / f"IMG_{i}.png", color)
    out = tmp_path / "day.mp4"

    result_out, used, rejected = timelapse.render_stills_timelapse(
        [tmp_path / "IMG_2.png", tmp_path / "IMG_0.png", tmp_path / "IMG_1.png"], out, fps=6
    )

    assert (result_out, used, rejected) == (out, 3, ())
    assert out.is_file() and out.stat().st_size > 0


def test_render_stills_timelapse_refuses_empty_input(tmp_path):
    with pytest.raises(timelapse.TimelapseError, match="at least one still"):
        timelapse.render_stills_timelapse([], tmp_path / "day.mp4")


def test_render_stills_timelapse_names_an_unreadable_still_and_never_pads(tmp_path):
    _write_png(tmp_path / "IMG_0.png", (10, 10, 10))
    bad = tmp_path / "IMG_1.png"
    bad.write_bytes(b"not a png")
    _write_png(tmp_path / "IMG_2.png", (20, 20, 20))
    out = tmp_path / "day.mp4"

    result_out, used, rejected = timelapse.render_stills_timelapse(
        [tmp_path / "IMG_0.png", bad, tmp_path / "IMG_2.png"], out, fps=6
    )

    assert used == 2
    assert [p.name for p, _ in rejected] == ["IMG_1.png"]


def test_render_stills_timelapse_all_rejected_raises_and_leaves_no_stub(tmp_path):
    bad = tmp_path / "IMG_0.png"
    bad.write_bytes(b"not a png")
    out = tmp_path / "day.mp4"

    with pytest.raises(timelapse.TimelapseError, match="no frames"):
        timelapse.render_stills_timelapse([bad], out)

    assert not out.exists()


def test_scale_height_shrinks_every_frame(tmp_path):
    for i in range(2):
        _write_png(tmp_path / f"IMG_{i}.png", (i * 40, i * 40, i * 40), size=(16, 12))
    out = tmp_path / "day.mp4"

    result_out, used, rejected = timelapse.render_stills_timelapse(
        [tmp_path / "IMG_0.png", tmp_path / "IMG_1.png"], out, fps=6, scale_height=6
    )

    assert (used, rejected) == (2, ())
    assert out.is_file()


def test_stream_frames_to_video_reports_the_encode_failure(tmp_path, monkeypatch):
    def _bad_encode_cmd(out, fps, size):
        return ["ffmpeg", "-not-a-real-flag"]

    with pytest.raises(timelapse.TimelapseError, match="timelapse render failed"):
        timelapse.stream_frames_to_video(
            iter([b"\x00" * (4 * 4 * 3)]),
            tmp_path / "out.mp4",
            6,
            (4, 4),
            encode_cmd_builder=_bad_encode_cmd,
        )


def test_decimate_video_keep_every_builds_the_select_filter_and_encodes(tmp_path, monkeypatch):
    calls = {}

    class _Result:
        returncode = 0
        stderr = b""

    def _fake_run(cmd, capture_output, timeout):
        calls["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"stub")
        return _Result()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    out = tmp_path / "decimated.mp4"
    src = tmp_path / "source.mp4"
    src.write_bytes(b"stub-source")

    result_out, cmd = timelapse.decimate_video(src, out, keep_every=4)

    assert result_out == out
    assert "select='not(mod(n\\,4))'" in cmd[cmd.index("-vf") + 1]
    assert cmd[cmd.index("-i") + 1] == str(src)


def test_decimate_video_needs_exactly_one_of_keep_every_or_speed_factor(tmp_path):
    with pytest.raises(timelapse.TimelapseError, match="exactly one"):
        timelapse.decimate_video(tmp_path / "s.mp4", tmp_path / "o.mp4")
    with pytest.raises(timelapse.TimelapseError, match="exactly one"):
        timelapse.decimate_video(
            tmp_path / "s.mp4", tmp_path / "o.mp4", keep_every=2, speed_factor=2
        )


def test_decimate_video_raises_on_a_failed_encode_and_leaves_no_stub(tmp_path, monkeypatch):
    class _Result:
        returncode = 1
        stderr = b"boom"

    def _fake_run(cmd, capture_output, timeout):
        return _Result()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    out = tmp_path / "decimated.mp4"

    with pytest.raises(timelapse.TimelapseError, match="decimation failed"):
        timelapse.decimate_video(tmp_path / "s.mp4", out, speed_factor=2)

    assert not out.exists()


def test_order_stills_is_filename_order():
    paths = [Path("b.png"), Path("a.png"), Path("c.png")]
    assert timelapse.order_stills(paths) == [Path("a.png"), Path("b.png"), Path("c.png")]


def test_cli_daylight_timelapse_renders_and_reports(tmp_path, capsys):
    for i, color in enumerate([(255, 0, 0), (0, 255, 0)]):
        _write_png(tmp_path / f"IMG_{i}.png", color)
    out = tmp_path / "day.mp4"
    argv = (
        f"photos daylight-timelapse {tmp_path / 'IMG_0.png'} {tmp_path / 'IMG_1.png'} "
        f"--out {out} --fps 6"
    )

    assert photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None) == 0

    printed = capsys.readouterr().out
    assert "2 still(s) at 6 fps" in printed and out.is_file()
