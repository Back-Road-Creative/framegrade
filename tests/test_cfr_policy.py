"""CFR conversion must be an explicit filter, not the output-side `-r`.

`-r 30000/1001` on the OUTPUT lets the muxer decide frame selection, and on
this ffmpeg it over-produces: a 10.000 s / 600-frame 60 fps source comes out
301 frames / 10.043 s instead of the correct 300 / 10.010 s, and the frames it
picks differ from the deterministic filter's (PSNR ~26 dB between the two).
`-fps_mode cfr` / `-vsync cfr` do NOT change that — only the `fps` filter does.

Policy: frames are DROPPED or DUPLICATED, never blended. Rationale:
docs/pipeline-internals.md.
"""

import re
import subprocess

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.color_filters import OUTPUT_FPS, cfr_filter
from gomoveshift.pipeline.single_pass import SinglePassProcessor

KW = dict(
    color_filter="null",
    location="",
    font_path="",
    duration=6.0,
    fade_video=False,
    video_width=1920,
    skip_ost=True,
)


@pytest.fixture
def proc():
    return SinglePassProcessor(Config())


def _ffmpeg(args: list[str]) -> str:
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", *args], capture_output=True, text=True, timeout=120
    )
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stderr


def test_cfr_filter_is_the_frozen_output_cadence():
    assert cfr_filter() == f"fps={OUTPUT_FPS}"
    assert OUTPUT_FPS == "30000/1001"


@pytest.mark.parametrize("builder", ["_build_video_filter", "_build_blur_filter_complex"])
def test_chains_end_with_the_explicit_cfr_filter(proc, builder):
    """Last in the chain, always.

    It must not run before the blur composite: the privacy mask is
    materialised one frame per SOURCE frame and `alphamerge` pairs them 1:1,
    so decimating the base first would desync the mask from the footage.
    """
    extra = {"blur_filter": "boxblur=8:2"} if "blur" in builder else {}
    out = getattr(proc, builder)(**KW, **extra)
    body = out.rsplit("[vout]", 1)[0] if "blur" in builder else out
    assert body.rstrip().endswith(cfr_filter()), out
    if "blur" in builder:
        assert body.index("overlay") < body.index(cfr_filter())


def test_filter_beats_output_r_on_frame_count_and_duration(tmp_path):
    """The defect this chunk exists for, measured end to end."""
    src = tmp_path / "s60.mkv"
    _ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=64x64:d=10:r=60", "-c:v", "ffv1", str(src)])

    def counts(args: list[str], out: str) -> tuple[int, float]:
        dst = tmp_path / out
        _ffmpeg(["-i", str(src), *args, "-c:v", "ffv1", str(dst)])
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames:format=duration",
                "-of",
                "csv=p=0",
                str(dst),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        nums = re.findall(r"[\d.]+", probe.stdout)
        return int(nums[0]), float(nums[1])

    r_frames, r_dur = counts(["-r", OUTPUT_FPS], "by_r.mkv")
    f_frames, f_dur = counts(["-vf", cfr_filter()], "by_filter.mkv")

    # 10 s at 29.97 is 299.7 frames; the filter lands on 300 / 10.010 s.
    assert f_frames == 300, f_frames
    assert f_dur == pytest.approx(10.010, abs=0.002)
    # The output-side -r over-produces and stretches the duration.
    assert r_frames > f_frames, f"-r no longer over-produces ({r_frames}); re-check the premise"
    assert r_dur > f_dur


def test_policy_is_drop_or_duplicate_never_blend(tmp_path):
    """A blended frame would be an average of two sources and match neither."""
    src = tmp_path / "ramp.mkv"
    # Each source frame is a flat, distinct grey; a blend would land between two.
    _ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=black:s=64x64:d=2:r=60",
            "-vf",
            "geq=lum='40+2*N':cb=128:cr=128,format=yuv420p",
            "-c:v",
            "ffv1",
            str(src),
        ]
    )
    dst = tmp_path / "cfr.mkv"
    _ffmpeg(["-i", str(src), "-vf", cfr_filter(), "-c:v", "ffv1", str(dst)])

    log = _ffmpeg(["-i", str(dst), "-vf", "signalstats,metadata=print", "-f", "null", "-"])
    lums = [float(v) for v in re.findall(r"YAVG=([\d.]+)", log)]
    assert lums, log[-1000:]
    # Every kept frame must still be one of the source's exact even integers.
    for lum in lums[:20]:
        nearest = round((lum - 40) / 2) * 2 + 40
        assert abs(lum - nearest) < 0.75, f"frame blended: YAVG={lum} is between source levels"
