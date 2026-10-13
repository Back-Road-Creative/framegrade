"""``full4k``: the 3840x2160 HEVC route-story preset, delivered through the rendition table."""

from __future__ import annotations

import subprocess
from pathlib import Path

from gomoveshift.pipeline.encoding_constants import RENDITIONS
from gomoveshift.pipeline.route_story import encode as encode_module
from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story.encode import encode_final
from gomoveshift.pipeline.route_story.presets import PRESETS

_BT709_AND_X264 = [
    "-c:v",
    "libx264",
    "-crf",
    "18",
    "-preset",
    "slow",
    "-pix_fmt",
    "yuv420p",
    "-color_range",
    "tv",
    "-colorspace",
    "bt709",
    "-color_primaries",
    "bt709",
    "-color_trc",
    "bt709",
    "-x264-params",
    "colorprim=bt709:transfer=bt709:colormatrix=bt709",
]


def _capture(monkeypatch, tmp_path: Path, **kwargs) -> list[str]:
    captured: dict = {}

    def fake_run(cmd, **kw):
        captured.setdefault("cmds", []).append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(encode_module.subprocess, "run", fake_run)
    monkeypatch.setattr(encode_module, "detect_nvenc", lambda: True)
    encode_final(tmp_path / "in.mp4", tmp_path / "out.mp4", **kwargs)
    return captured["cmds"][-1]


def test_full4k_preset_delivers_2160p_hevc(monkeypatch, tmp_path):
    preset = PRESETS["full4k"]
    full = PRESETS["full"]
    assert (preset.width, preset.height) == (3840, 2160)
    assert preset.rendition == "route_story_full4k"
    assert preset.hevc_delivery is True
    assert (preset.intro_seconds, preset.recap_seconds, preset.photo_seconds) == (
        full.intro_seconds,
        full.recap_seconds,
        full.photo_seconds,
    )
    assert (preset.map_seconds, preset.max_photos, preset.max_seconds) == (None, None, None)

    row = RENDITIONS["route_story_full4k"]
    assert (row.width, row.height, row.codec) == (3840, 2160, "hevc")
    assert row.audio is True and row.faststart is True and row.max_seconds is None
    assert row.min_bitrate_kbps == RENDITIONS["drive_longform_4k"].min_bitrate_kbps

    cmd = _capture(monkeypatch, tmp_path, rendition=preset.rendition)
    assert cmd[cmd.index("-c:v") + 1] == "hevc_nvenc"
    assert cmd[cmd.index("-cq") + 1] == "18"
    assert cmd[cmd.index("-pix_fmt") + 1] == "p010le"
    assert cmd[cmd.index("-profile:v") + 1] == "main10"
    assert cmd[cmd.index("-colorspace") + 1] == "bt709"
    assert "libx264" not in cmd and "-x264-params" not in cmd

    ok = {"codec": "hevc", "width": 3840, "height": 2160}
    check = film_module.build_rendition_check(
        preset, {"video": ok, "audio": {"codec": "aac"}, "duration_s": 60.0}
    )
    assert check == {
        "audio_stream": True,
        "size_ok": True,
        "within_ceiling": True,
        "codec_ok": True,
    }
    bad = film_module.build_rendition_check(
        preset,
        {
            "video": {"codec": "h264", "width": 1920, "height": 1080},
            "audio": {"codec": "aac"},
            "duration_s": 60.0,
        },
    )
    assert bad["size_ok"] is False and bad["codec_ok"] is False


def test_full4k_falls_back_to_libx265_when_nvenc_is_unavailable(monkeypatch, tmp_path):
    def no_nvenc():
        raise encode_module.NVENCUnavailableError("no gpu")

    _seen: list = []

    def fake_run(cmd, **kw):
        _seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(encode_module.subprocess, "run", fake_run)
    monkeypatch.setattr(encode_module, "detect_nvenc", no_nvenc)
    encode_final(tmp_path / "in.mp4", tmp_path / "out.mp4", rendition="route_story_full4k")
    cmd = _seen[-1]
    assert cmd[cmd.index("-c:v") + 1] == "libx265"
    assert "hevc_nvenc" not in cmd


def test_full_and_social_encode_args_unchanged(monkeypatch, tmp_path):
    for name in ("full", "social"):
        preset = PRESETS[name]
        assert preset.hevc_delivery is False
        rendition = preset.rendition if preset.hevc_delivery else None
        cmd = _capture(monkeypatch, tmp_path, rendition=rendition)
        start = cmd.index("-c:v")
        assert cmd[start : start + len(_BT709_AND_X264)] == _BT709_AND_X264
        assert cmd[:4] == ["ffmpeg", "-y", "-i", str(tmp_path / "in.mp4")]
    # No rendition argument at all is the pre-preset call and is the same command.
    assert _capture(monkeypatch, tmp_path) == _capture(monkeypatch, tmp_path, rendition=None)


def _hevc_probe() -> dict:
    stream = {
        "codec_type": "video",
        "codec_name": "hevc",
        "width": 3840,
        "height": 2160,
        "pix_fmt": "yuv420p10le",
        "r_frame_rate": "30/1",
        "nb_frames": "300",
        "color_range": "tv",
        "color_space": "bt709",
        "color_primaries": "bt709",
        "color_transfer": "bt709",
    }
    audio = {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2}
    return {"streams": [stream, audio], "format": {"duration": "10.0"}}


def _run_probe(cmd, **kwargs):
    import json
    from types import SimpleNamespace

    out = json.dumps(_hevc_probe()) if cmd[0] == "ffprobe" else ""
    return SimpleNamespace(returncode=0, stdout=out, stderr="")


def test_check_delivery_takes_the_expected_codec_as_a_parameter():
    from gomoveshift.pipeline.route_story import delivery

    spec = {"width": 3840, "height": 2160, "fps": 30, "runner": _run_probe}
    ok = delivery.check_delivery(
        Path("f.mp4"),
        expected_codec="hevc",
        expected_pix_fmt=delivery.HEVC_PIXEL_FORMAT,
        **spec,
    )
    assert ok["issues"] == []
    default = delivery.check_delivery(Path("f.mp4"), **spec)
    assert "video codec is hevc, expected h264" in default["issues"]
