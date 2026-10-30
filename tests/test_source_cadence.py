"""Opt-in source-cadence output (Q9b): 59.94/60 fps sources keep their rate.

The default stays the frozen 29.97 publish cadence, byte-identical. An explicit
output rate threads through the master filter chain, the encode args and the OST
pass; unsupported rates refuse. All media here is synthetic (ffmpeg lavfi).
"""

import subprocess

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import ost_applier
from gomoveshift.pipeline.color_filters import (
    OUTPUT_FPS,
    SUPPORTED_OUTPUT_FPS,
    cfr_filter,
    check_output_fps,
    get_encoding_args,
    source_output_fps,
)
from gomoveshift.pipeline.single_pass import SinglePassProcessor

KW = dict(
    color_filter="null",
    location="",
    font_path="",
    duration=2.0,
    fade_video=False,
    video_width=128,
    skip_ost=True,
)


@pytest.fixture
def proc():
    return SinglePassProcessor(Config())


def _arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


# ── default path unchanged ───────────────────────────────────────────


def test_default_filter_strings_are_unchanged(proc):
    assert cfr_filter() == "fps=30000/1001"
    assert cfr_filter(OUTPUT_FPS) == cfr_filter()
    assert proc._build_video_filter(**KW) == proc._build_video_filter(**KW, output_fps=OUTPUT_FPS)
    assert proc._build_video_filter(**KW).endswith("fps=30000/1001")
    blur = dict(KW, blur_filter="boxblur=8:2")
    assert proc._build_blur_filter_complex(**blur) == proc._build_blur_filter_complex(
        **blur, output_fps=OUTPUT_FPS
    )


@pytest.mark.parametrize("nvenc", [True, False])
def test_default_encoding_args_are_unchanged(nvenc):
    default = get_encoding_args(3840, 2160, 59.94, has_nvenc=nvenc)
    explicit = get_encoding_args(3840, 2160, 59.94, has_nvenc=nvenc, output_fps=OUTPUT_FPS)
    assert default == explicit
    assert _arg(default, "-r") == "30000/1001"
    assert _arg(default, "-g") == "60"


# ── explicit rate ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("rate", "gop"), [("30000/1001", 60), ("30", 60), ("60000/1001", 120), ("60", 120)]
)
@pytest.mark.parametrize("nvenc", [True, False])
def test_gop_is_two_seconds_at_the_chosen_rate(rate, gop, nvenc):
    args = get_encoding_args(1920, 1080, 59.94, has_nvenc=nvenc, output_fps=rate)
    assert _arg(args, "-r") == rate
    assert _arg(args, "-g") == str(gop)


def test_cfr_filter_takes_the_rate_and_stays_the_chain_tail(proc):
    assert cfr_filter("60000/1001") == "fps=60000/1001"
    vf = proc._build_video_filter(**KW, output_fps="60000/1001")
    assert vf.endswith("fps=60000/1001") and "fps=30000/1001" not in vf
    fc = proc._build_blur_filter_complex(**KW, blur_filter="boxblur=8:2", output_fps="60000/1001")
    body = fc.rsplit("[vout]", 1)[0]
    assert body.endswith("fps=60000/1001")
    assert body.index("overlay") < body.index("fps=60000/1001")


@pytest.mark.parametrize("bad", ["24", "25", "50", "120", "60000/1000", "abc", ""])
def test_unsupported_rates_refuse(bad):
    with pytest.raises(ValueError, match="output rate"):
        cfr_filter(bad)
    with pytest.raises(ValueError, match="output rate"):
        get_encoding_args(1920, 1080, 59.94, has_nvenc=False, output_fps=bad)
    with pytest.raises(ValueError, match="output rate"):
        check_output_fps(bad, 59.94)


def test_supported_set_is_exactly_the_four_rates():
    assert set(SUPPORTED_OUTPUT_FPS) == {"30000/1001", "30", "60000/1001", "60"}


def test_non_default_rate_must_be_the_probed_source_rate():
    check_output_fps("60000/1001", 60000 / 1001)
    check_output_fps("60", 60.0)
    check_output_fps(OUTPUT_FPS, 24.0)  # the default needs no match
    with pytest.raises(ValueError, match="source"):
        check_output_fps("60000/1001", 30000 / 1001)  # would upsample
    with pytest.raises(ValueError, match="source"):
        check_output_fps("60", 59.94)


def test_source_output_fps_maps_probed_rates_and_refuses_others():
    assert source_output_fps(60000 / 1001) == "60000/1001"
    assert source_output_fps(60.0) == "60"
    assert source_output_fps(30000 / 1001) == "30000/1001"
    assert source_output_fps(30.0) == "30"
    with pytest.raises(ValueError, match="source"):
        source_output_fps(24.0)


# ── OST pass keeps the master's rate ─────────────────────────────────


def test_ost_encoding_args_default_and_source_rate(monkeypatch):
    monkeypatch.setattr("gomoveshift.pipeline.color.detect_nvenc", lambda: False)
    info = dict(width=1920, height=1080, fps=60000 / 1001, bitrate_kbps=0)
    default = ost_applier._resolve_encoding_args(info, False, "x.mp4")
    assert _arg(default, "-r") == "30000/1001"
    src = ost_applier._resolve_encoding_args(info, False, "x.mp4", output_fps="60000/1001")
    assert _arg(src, "-r") == "60000/1001" and _arg(src, "-g") == "120"
    with pytest.raises(ValueError, match="source"):
        ost_applier._resolve_encoding_args(info, False, "x.mp4", output_fps="60")


# ── real ffmpeg on a synthetic 59.94 source ──────────────────────────


def _ffmpeg(args: list[str]) -> None:
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", *args], capture_output=True, text=True, timeout=180
    )
    assert p.returncode == 0, p.stderr[-2000:]


def _probe(path) -> tuple[int, str]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,r_frame_rate",
            "-of",
            "default=nw=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    fields = dict(line.split("=", 1) for line in out.splitlines())
    return int(fields["nb_read_frames"]), fields["r_frame_rate"]


@pytest.fixture(scope="module")
def synthetic_5994(tmp_path_factory):
    """SYNTHETIC 59.94 fps clip (lavfi testsrc2); never real footage."""
    src = tmp_path_factory.mktemp("cadence") / "synthetic_5994.nut"
    _ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=128x72:d=2:r=60000/1001", "-c:v", "ffv1", str(src)])
    return src


def _encode(proc, src, dst, output_fps):
    kw = {} if output_fps is None else {"output_fps": output_fps}
    vf = proc._build_video_filter(**KW, **kw)
    enc = get_encoding_args(128, 72, 60000 / 1001, has_nvenc=False, cpu_by_choice=True, **kw)
    _ffmpeg(["-i", str(src), "-vf", vf, "-an", *enc, "-c:a", "copy", str(dst)])


def test_real_ffmpeg_default_vs_source_cadence(proc, synthetic_5994, tmp_path):
    src_frames, src_rate = _probe(synthetic_5994)
    assert src_rate == "60000/1001"

    dflt = tmp_path / "default.mp4"
    _encode(proc, synthetic_5994, dflt, None)
    d_frames, d_rate = _probe(dflt)
    assert d_rate == "30000/1001"
    assert abs(d_frames - src_frames / 2) <= 1

    srcm = tmp_path / "source.mp4"
    _encode(proc, synthetic_5994, srcm, "60000/1001")
    s_frames, s_rate = _probe(srcm)
    assert s_rate == "60000/1001"
    assert s_frames == src_frames  # every source frame survives
