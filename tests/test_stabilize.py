"""Stabilise / deflicker pass.

Filter-probe parse, argv builders and the vidstab refusal are pure-Python
unit tests with a fake ``run``. One test exercises real ffmpeg deflicker on a
tiny synthetic alternating-brightness clip and checks the jitter gate — it is
skipped only when ffmpeg is genuinely absent.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from gomoveshift.pipeline import stabilize

FFMPEG = shutil.which("ffmpeg")


class _FakeCompleted:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _filters_stdout(names):
    header = " T.. abuffer            A->A       Buffer audio frames.\n"
    body = "".join(f" T.. {n:<18} V->V       fake filter.\n" for n in names)
    return header + body


# ── probe_ffmpeg_filters ──────────────────────────────────────────────────


def test_probe_parses_present_filters():
    caps = stabilize.probe_ffmpeg_filters(
        run=lambda *a, **k: _FakeCompleted(stdout=_filters_stdout(["deflicker", "scale"]))
    )
    assert caps.has_deflicker
    assert not caps.has_vidstab


def test_probe_needs_both_vidstab_filters():
    caps = stabilize.probe_ffmpeg_filters(
        run=lambda *a, **k: _FakeCompleted(stdout=_filters_stdout(["vidstabdetect"]))
    )
    assert not caps.has_vidstab

    caps_both = stabilize.probe_ffmpeg_filters(
        run=lambda *a, **k: _FakeCompleted(
            stdout=_filters_stdout(["vidstabdetect", "vidstabtransform"])
        )
    )
    assert caps_both.has_vidstab


def test_probe_matches_expected_ffmpeg_capabilities():
    """The real probe agrees with this ffmpeg's own filter list.

    Builds differ on libvidstab (the host build lacks it, stock and static
    builds ship it), so the expectation is read from ``ffmpeg -filters``
    independently rather than pinned; the refusal path is covered by the
    fake-``run`` tests above whichever build is installed.
    """
    caps = stabilize.probe_ffmpeg_filters()
    listed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, check=True
    ).stdout
    names = {m.group(1) for m in re.finditer(r"^ \S+ +(\S+) ", listed, re.M)}
    assert caps.has_deflicker
    assert caps.has_vidstab == ({"vidstabdetect", "vidstabtransform"} <= names)


# ── require_vidstab / refusal ─────────────────────────────────────────────


def test_require_vidstab_raises_named_reason_when_missing():
    caps = stabilize.FfmpegFilterCapabilities(frozenset({"deflicker"}))
    with pytest.raises(stabilize.StabilizeUnavailableError, match="libvidstab"):
        stabilize.require_vidstab(caps)


def test_require_vidstab_passes_when_present():
    caps = stabilize.FfmpegFilterCapabilities(frozenset({"vidstabdetect", "vidstabtransform"}))
    stabilize.require_vidstab(caps)  # no raise


def test_vidstabdetect_argv_refuses_without_vidstab():
    caps = stabilize.FfmpegFilterCapabilities(frozenset({"deflicker"}))
    with pytest.raises(stabilize.StabilizeUnavailableError):
        stabilize.build_vidstabdetect_argv("in.mp4", "transforms.trf", caps)


def test_vidstabtransform_argv_refuses_without_vidstab():
    caps = stabilize.FfmpegFilterCapabilities(frozenset({"deflicker"}))
    with pytest.raises(stabilize.StabilizeUnavailableError):
        stabilize.build_vidstabtransform_argv("in.mp4", "transforms.trf", "out.mp4", caps)


# ── argv builders ──────────────────────────────────────────────────────────


def test_build_deflicker_argv():
    argv = stabilize.build_deflicker_argv("in.mp4", "out.mp4", mode="pm", size=5)
    assert argv[0] == "ffmpeg"
    assert "in.mp4" in argv
    assert "out.mp4" in argv
    assert "-vf" in argv
    vf = argv[argv.index("-vf") + 1]
    assert vf == "deflicker=mode=pm:size=5"


def test_build_vidstabdetect_argv_when_available():
    caps = stabilize.FfmpegFilterCapabilities(frozenset({"vidstabdetect", "vidstabtransform"}))
    argv = stabilize.build_vidstabdetect_argv("in.mp4", "t.trf", caps, shakiness=6, accuracy=10)
    vf = argv[argv.index("-vf") + 1]
    assert vf == "vidstabdetect=shakiness=6:accuracy=10:result=t.trf"


def test_build_vidstabtransform_argv_when_available():
    caps = stabilize.FfmpegFilterCapabilities(frozenset({"vidstabdetect", "vidstabtransform"}))
    argv = stabilize.build_vidstabtransform_argv(
        "in.mp4", "t.trf", "out.mp4", caps, smoothing=15, zoom=0
    )
    vf = argv[argv.index("-vf") + 1]
    assert vf == "vidstabtransform=input=t.trf:smoothing=15:zoom=0"


# ── jitter gate ─────────────────────────────────────────────────────────────


def test_gate_on_jitter_keeps_when_output_lower():
    result = stabilize.gate_on_jitter(input_jitter=5.0, output_jitter=1.0)
    assert result.passed
    assert result.reason == ""


def test_gate_on_jitter_discards_when_output_not_lower():
    result = stabilize.gate_on_jitter(input_jitter=1.0, output_jitter=5.0)
    assert not result.passed
    assert "did not drop" in result.reason


def test_measure_luma_jitter_parses_yavg_and_needs_two_samples():
    stderr = (
        "frame=1 lavfi.signalstats.YAVG=10.0\n"
        "frame=2 lavfi.signalstats.YAVG=50.0\n"
        "frame=3 lavfi.signalstats.YAVG=10.0\n"
    )
    jitter = stabilize.measure_luma_jitter(
        "clip.mp4", run=lambda *a, **k: _FakeCompleted(stderr=stderr)
    )
    assert jitter > 0

    with pytest.raises(ValueError):
        stabilize.measure_luma_jitter(
            "clip.mp4", run=lambda *a, **k: _FakeCompleted(stderr="no samples here")
        )


def test_run_deflicker_raises_on_encode_failure():
    with pytest.raises(RuntimeError):
        stabilize.run_deflicker(
            "in.mp4",
            "out.mp4",
            run=lambda *a, **k: _FakeCompleted(returncode=1, stderr="boom"),
        )


# ── real ffmpeg: deflicker on a synthetic flickering clip ─────────────────


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_deflicker_reduces_measured_jitter_on_real_ffmpeg(tmp_path):
    import numpy as np

    width, height, fps, n_frames = 64, 64, 24, 40
    input_path = tmp_path / "flicker_in.mp4"

    argv = [
        FFMPEG,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-pix_fmt",
        "yuv420p",
        str(input_path),
    ]
    frames = []
    for i in range(n_frames):
        brightness = 30 if i % 2 == 0 else 225
        frames.append(np.full((height, width, 3), brightness, dtype=np.uint8))
    raw = b"".join(f.tobytes() for f in frames)

    proc = subprocess.run(argv, input=raw, capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")

    output_path = tmp_path / "flicker_out.mp4"
    gate = stabilize.run_deflicker(input_path, output_path)

    assert gate.passed, gate.reason
    assert gate.output_jitter < gate.input_jitter
    assert output_path.exists()


# ── explicit intermediate encoding ──────────────────────────────────────────


def _tenbit():
    return stabilize.StreamSpec(
        pix_fmt="yuv420p10le",
        color_range="tv",
        color_space="bt709",
        color_primaries="bt709",
        color_transfer="bt709",
    )


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_deflicker_argv_is_explicit_about_codec_depth_and_tags():
    from gomoveshift.pipeline import encoding_constants as ec

    argv = stabilize.build_deflicker_argv("in.mp4", "out.mp4", stream=_tenbit())
    assert _flag(argv, "-c:v") == "libx265"
    assert _flag(argv, "-preset") == ec.LONGFORM_PRESET
    assert _flag(argv, "-crf") == str(ec.LONGFORM_CRF)
    assert _flag(argv, "-pix_fmt") == "yuv420p10le"
    assert _flag(argv, "-profile:v") == ec.LONGFORM_PROFILE
    assert _flag(argv, "-color_range") == "tv"
    assert _flag(argv, "-colorspace") == "bt709"
    assert _flag(argv, "-color_primaries") == "bt709"
    assert _flag(argv, "-color_trc") == "bt709"
    # libx265 writes primaries / transfer only through x265-params
    assert _flag(argv, "-x265-params") == "colorprim=bt709:transfer=bt709:colormatrix=bt709"
    assert argv[-1] == "out.mp4"
    assert argv.index("-c:v") < argv.index("-c:a") < len(argv) - 1


def test_vidstabtransform_argv_is_explicit_too():
    caps = stabilize.FfmpegFilterCapabilities(frozenset({"vidstabdetect", "vidstabtransform"}))
    argv = stabilize.build_vidstabtransform_argv(
        "in.mp4", "t.trf", "out.mp4", caps, stream=_tenbit()
    )
    assert _flag(argv, "-c:v") == "libx265"
    assert _flag(argv, "-pix_fmt") == "yuv420p10le"
    assert _flag(argv, "-color_trc") == "bt709"


def test_eight_bit_input_stays_eight_bit_and_unset_tags_are_not_invented():
    spec = stabilize.StreamSpec(pix_fmt="yuv420p", color_range="tv", color_space="unknown")
    args = stabilize.intermediate_video_args(spec)
    assert _flag(args, "-pix_fmt") == "yuv420p"
    assert "-profile:v" not in args
    assert "-colorspace" not in args and "-color_trc" not in args
    assert "-x265-params" not in args
    assert _flag(args, "-color_range") == "tv"


def test_unprobed_or_unsupported_pix_fmt_falls_back_to_ten_bit():
    for spec in (stabilize.StreamSpec(), stabilize.StreamSpec(pix_fmt="cuda")):
        assert _flag(stabilize.intermediate_video_args(spec), "-pix_fmt") == "yuv420p10le"


def test_probe_stream_spec_parses_ffprobe_and_survives_failure():
    out = '{"streams":[{"pix_fmt":"yuv420p10le","color_range":"tv","color_space":"bt709"}]}'
    spec = stabilize.probe_stream_spec("c.mp4", run=lambda *a, **k: _FakeCompleted(stdout=out))
    assert spec.pix_fmt == "yuv420p10le" and spec.color_space == "bt709"
    assert spec.color_transfer == ""
    bad = stabilize.probe_stream_spec("c.mp4", run=lambda *a, **k: _FakeCompleted(returncode=1))
    assert bad == stabilize.StreamSpec()


def _probe_out(path):
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries"]
        + ["stream=codec_name,pix_fmt,color_range,color_space,color_primaries,color_transfer"]
        + ["-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    import json

    return json.loads(res.stdout)["streams"][0]


def _tenbit_flicker_clip(path):
    src = (
        "color=c=gray:s=128x128:r=24:d=2,format=yuv420p10le,"
        "geq=lum='if(mod(N,2),300,800)':cb=512:cr=512"
    )
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", src, "-c:v", "libx265"]
        + ["-pix_fmt", "yuv420p10le", "-crf", "10", "-color_range", "tv"]
        + ["-x265-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709", str(path)],
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-400:]


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_deflicker_keeps_ten_bit_and_colour_tags_on_real_ffmpeg(tmp_path):
    src = tmp_path / "in10.mp4"
    _tenbit_flicker_clip(src)
    tagged = _probe_out(src)
    assert tagged["pix_fmt"] == "yuv420p10le" and tagged["color_transfer"] == "bt709"
    out = tmp_path / "out10.mp4"
    gate = stabilize.run_deflicker(src, out)
    assert gate.passed, gate.reason
    got = _probe_out(out)
    # ffmpeg's own default for .mp4 is h264 at crf 23; the codec must be chosen, not defaulted
    assert got["codec_name"] == "hevc"
    assert got["pix_fmt"] == "yuv420p10le"
    assert (got["color_range"], got["color_space"]) == ("tv", "bt709")
    assert (got["color_primaries"], got["color_transfer"]) == ("bt709", "bt709")


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_vidstab_keeps_ten_bit_and_colour_tags_on_real_ffmpeg(tmp_path):
    caps = stabilize.probe_ffmpeg_filters()
    if not caps.has_vidstab:
        pytest.skip(stabilize.VIDSTAB_MISSING_REASON)
    src = tmp_path / "in10.mp4"
    _tenbit_flicker_clip(src)
    trf, out = tmp_path / "t.trf", tmp_path / "o.mp4"
    detect = stabilize.build_vidstabdetect_argv(src, trf, caps)
    assert subprocess.run(detect, capture_output=True, timeout=120).returncode == 0
    spec = stabilize.probe_stream_spec(src)
    argv = stabilize.build_vidstabtransform_argv(src, trf, out, caps, stream=spec)
    assert subprocess.run(argv, capture_output=True, timeout=120).returncode == 0
    got = _probe_out(out)
    assert got["pix_fmt"] == "yuv420p10le" and got["color_transfer"] == "bt709"


# --- encoder selection: NVENC first, x265 fallback (no GPU needed) -----------


def _uhd(**kw):
    base = dict(
        pix_fmt="yuv420p10le",
        color_range="tv",
        color_space="bt709",
        color_primaries="bt709",
        color_transfer="bt709",
        height=2160,
    )
    base.update(kw)
    return stabilize.StreamSpec(**base)


def test_nvenc_branch_uses_longform_nvenc_constants_and_keeps_timing():
    from gomoveshift.pipeline import encoding_constants as ec

    args = stabilize.intermediate_video_args(_uhd(), has_nvenc=True)
    assert _flag(args, "-c:v") == "hevc_nvenc"
    assert _flag(args, "-preset") == ec.LONGFORM_NVENC_PRESET
    assert _flag(args, "-cq") == str(ec.LONGFORM_NVENC_CQ)
    assert _flag(args, "-rc") == "vbr"
    assert _flag(args, "-pix_fmt") == "p010le"
    assert _flag(args, "-profile:v") == ec.LONGFORM_PROFILE
    assert _flag(args, "-b:v") == f"{ec.LONGFORM_MIN_BITRATE_4K}k"
    assert "-r" not in args and "-crf" not in args and "-x265-params" not in args
    assert _flag(args, "-color_range") == "tv"
    assert _flag(args, "-colorspace") == "bt709"
    assert _flag(args, "-color_primaries") == "bt709"
    assert _flag(args, "-color_trc") == "bt709"


def test_nvenc_branch_eight_bit_stays_eight_bit_and_other_depths_go_p010():
    eight = stabilize.intermediate_video_args(_uhd(pix_fmt="yuv420p", height=1080), has_nvenc=True)
    assert _flag(eight, "-pix_fmt") == "yuv420p"
    assert _flag(eight, "-profile:v") == "main"
    assert _flag(eight, "-b:v") == "15000k"
    for pf in ("", "yuv444p12le"):
        args = stabilize.intermediate_video_args(_uhd(pix_fmt=pf), has_nvenc=True)
        assert _flag(args, "-pix_fmt") == "p010le"


def test_x265_branch_is_the_default_and_unchanged():
    assert stabilize.intermediate_video_args(_uhd()) == stabilize.intermediate_video_args(
        _uhd(), has_nvenc=False
    )
    assert _flag(stabilize.intermediate_video_args(_uhd()), "-c:v") == "libx265"


def test_build_argv_threads_has_nvenc():
    caps = stabilize.FfmpegFilterCapabilities(frozenset({"vidstabdetect", "vidstabtransform"}))
    d = stabilize.build_deflicker_argv("i.mp4", "o.mp4", stream=_uhd(), has_nvenc=True)
    v = stabilize.build_vidstabtransform_argv(
        "i.mp4", "t.trf", "o.mp4", caps, stream=_uhd(), has_nvenc=True
    )
    assert _flag(d, "-c:v") == "hevc_nvenc" and _flag(v, "-c:v") == "hevc_nvenc"


def test_run_deflicker_injects_has_nvenc_without_a_gpu():
    seen = []
    probe = '{"streams":[{"pix_fmt":"yuv420p10le","height":2160}]}'

    def fake(argv, *a, **k):
        seen.append(list(argv))
        if "ffprobe" in str(argv[0]):
            return _FakeCompleted(stdout=probe)
        return _FakeCompleted(stderr="")

    try:
        stabilize.run_deflicker("i.mp4", "o.mp4", run=fake, has_nvenc=True)
    except Exception:  # the jitter measurement of a fake run is not the point
        pass
    encode = [a for a in seen if "-c:v" in a]
    assert encode and _flag(encode[0], "-c:v") == "hevc_nvenc"


def test_nvenc_available_reads_unavailable_as_false(monkeypatch):
    from gomoveshift.pipeline import color_filters

    monkeypatch.setattr(color_filters, "detect_nvenc", lambda: True)
    assert stabilize.nvenc_available() is True

    def boom():
        raise color_filters.NVENCUnavailableError("no gpu")

    monkeypatch.setattr(color_filters, "detect_nvenc", boom)
    assert stabilize.nvenc_available() is False
