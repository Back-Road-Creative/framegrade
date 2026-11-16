"""The Instagram Reels in-spec delivery file and its upstream gate (#399)."""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.publishing import instagram_reels_spec as spec

_GOOD_PROBE = {
    "streams": [
        {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p",
         "avg_frame_rate": "30000/1001", "field_order": "progressive"},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
    ],
    "format": {"bit_rate": "18000000"},
}  # fmt: skip


def _box(kind: bytes, payload: bytes = b"") -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _mp4(*, elst: bool = False, moov_first: bool = True) -> bytes:
    edts = _box(b"edts", _box(b"elst", b"\0" * 12)) if elst else b""
    moov = _box(b"moov", _box(b"trak", edts + _box(b"mdia")))
    mdat = _box(b"mdat", b"x" * 32)
    return _box(b"ftyp", b"isom") + (moov + mdat if moov_first else mdat + moov)


def test_ffmpeg_argv_carries_the_reels_spec_flags(tmp_path):
    argv = spec.build_ffmpeg_argv(tmp_path / "a.mp4", tmp_path / "b.mp4")
    joined = " ".join(argv)
    for flag in (
        "-c:v libx264", "-profile:v high", "-pix_fmt yuv420p", "-maxrate 20M", "-bufsize",
        "open-gop=0", "-c:a aac", "-ar 48000", "-ac 2", "-movflags +faststart",
        "-use_editlist 0",
    ):  # fmt: skip
        assert flag in joined
    assert argv[-1].endswith("b.mp4")


def test_layout_reads_edit_list_and_moov_position(tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(_mp4(elst=True, moov_first=False))
    assert spec.mp4_layout(f) == (False, True)
    f.write_bytes(_mp4())
    assert spec.mp4_layout(f) == (True, False)


def test_a_good_file_passes_the_check():
    assert spec.check_spec(_GOOD_PROBE, moov_first=True, has_elst=False, size_bytes=10) == []


@pytest.mark.parametrize(
    ("mutate", "layout", "size", "needle"),
    [
        (lambda p: None, {"has_elst": True}, 10, "edit list"),
        (lambda p: None, {"moov_first": False}, 10, "moov"),
        (lambda p: p["format"].update(bit_rate="27300000"), {}, 10, "bitrate 27.3 Mbps"),
        (lambda p: p["streams"][0].update(pix_fmt="yuv420p10le"), {}, 10, "pix_fmt"),
        (lambda p: p["streams"][0].update(codec_name="vp9"), {}, 10, "codec"),
        (lambda p: p["streams"][0].update(avg_frame_rate="120/1"), {}, 10, "frame rate"),
        (lambda p: p["streams"][1].update(sample_rate="96000"), {}, 10, "sample rate"),
        (lambda p: None, {}, 301 * 1024 * 1024, "300 MB"),
    ],
)
def test_the_check_names_each_broken_rule(mutate, layout, size, needle):
    import copy

    probe = copy.deepcopy(_GOOD_PROBE)
    mutate(probe)
    kw = {"moov_first": True, "has_elst": False, **layout}
    fails = spec.check_spec(probe, size_bytes=size, **kw)
    assert any(needle in f for f in fails), fails


def test_the_source_is_transcoded_gated_and_cleaned_up(tmp_path):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[0] == "ffmpeg":
            Path(cmd[-1]).write_bytes(_mp4())
            return SimpleNamespace(stdout="")
        import json

        return SimpleNamespace(stdout=json.dumps(_GOOD_PROBE))

    out = spec.make_in_spec_bytes(b"source", run=run, scratch_dir=str(tmp_path))
    assert out == _mp4() and out != b"source"
    assert [c[0] for c in calls] == ["ffmpeg", "ffprobe"]
    assert list(tmp_path.iterdir()) == []


def test_an_out_of_spec_transcode_raises_and_cleans_up(tmp_path):
    def run(cmd, **kw):
        if cmd[0] == "ffmpeg":
            Path(cmd[-1]).write_bytes(_mp4(elst=True))
            return SimpleNamespace(stdout="")
        import json

        return SimpleNamespace(stdout=json.dumps(_GOOD_PROBE))

    with pytest.raises(spec.ReelsSpecError, match="edit list"):
        spec.make_in_spec_bytes(b"source", run=run, scratch_dir=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_a_failed_ffmpeg_is_a_spec_error(tmp_path):
    def run(cmd, **kw):
        raise RuntimeError("boom")

    with pytest.raises(spec.ReelsSpecError, match="transcode failed"):
        spec.make_in_spec_bytes(b"source", run=run, scratch_dir=str(tmp_path))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent")
def test_real_ffmpeg_turns_a_10bit_hevc_clip_into_an_in_spec_file(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=320x568:rate=30:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=44100",
         "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-tag:v", "hev1", "-c:a", "aac",
         "-shortest", str(src)],
        check=True,
    )  # fmt: skip
    out = tmp_path / "out.mp4"
    out.write_bytes(spec.make_in_spec_bytes(src.read_bytes(), scratch_dir=str(tmp_path)))
    spec.gate_file(out)
    video = next(s for s in spec.probe_file(out)["streams"] if s["codec_type"] == "video")
    assert (video["codec_name"], video["pix_fmt"]) == ("h264", "yuv420p")
