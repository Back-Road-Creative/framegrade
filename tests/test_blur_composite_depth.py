"""The blur composite must not drag the whole frame through 8 bits.

`alphamerge` is 8-bit-only in this ffmpeg's default negotiation, so the
production graph settled on `yuva420p` and `auto_scale` converted the
UNTOUCHED base branch 10-bit -> 8-bit as well. Blur is default-on, so every
day master paid that on every pixel, not just the masked regions, immediately
before a 10-bit master encode.

The existing composite test string-matches `"alphamerge" in fc` — which is
exactly how this shipped despite the written-down PSNR discipline. These tests
run the REAL graph through the REAL ffmpeg instead.

Rationale: docs/pipeline-internals.md.
"""

import re
import subprocess

import pytest

from gomoveshift.config import Config
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


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    """A 10-bit source plus an all-black and a half-white mask."""
    d = tmp_path_factory.mktemp("blur")
    src, black, box = d / "src.mp4", d / "black.mkv", d / "box.mkv"
    _ffmpeg(
        ["-f", "lavfi", "-i", "testsrc2=s=320x180:d=1:r=5,format=yuv420p10le"]
        + ["-c:v", "libx265", "-x265-params", "lossless=1", "-pix_fmt", "yuv420p10le", str(src)]
    )
    _ffmpeg(
        ["-f", "lavfi", "-i", "color=black:s=320x180:d=1:r=5,format=gray", "-c:v", "ffv1"]
        + [str(black)]
    )
    # Left half white => blur applies there, right half must stay pristine.
    _ffmpeg(
        ["-f", "lavfi", "-i", "color=black:s=320x180:d=1:r=5"]
        + ["-vf", "drawbox=x=0:y=0:w=160:h=180:color=white:t=fill,format=gray"]
        + ["-c:v", "ffv1", str(box)]
    )
    return src, black, box


def _ffmpeg(args: list[str]) -> str:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", *args], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stderr


def _run_graph(src, mask, fc: str, extra: list[str], loglevel="info") -> str:
    return _ffmpeg(
        ["-loglevel", loglevel, "-i", str(src), "-i", str(mask), "-filter_complex", fc, *extra]
    )


def _frame_count(path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames"]
        + ["-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return int(out.stdout.strip())


# The chain tail converts to a fixed 29.97 CFR, so [vout] carries a different
# frame timeline from a raw decode of the same source. `psnr` pairs its two
# lanes through framesync, which by default REPEATS the shorter lane's last
# frame against every remaining frame of the longer one and scores those
# invented pairs — reading ~35 dB on a composite whose every real frame is
# pixel-exact. `shortest=1` stops at the first lane's EOF so only frames both
# lanes actually produced are compared. It removes fabricated comparisons, not
# real ones: every frame the composite emits is still scored, and an aggregate
# of `inf` is reachable only when each of them is individually exact.
PSNR_PAIRING = "shortest=1"


def test_base_leg_never_negotiates_down_to_eight_bit(proc, clips):
    """The base must stay 10-bit through the split.

    ffmpeg negotiates formats BACKWARDS, so alphamerge's 8-bit-only surface
    used to propagate up through overlay and split into the base `scale`,
    flattening the untouched frame before the blur ever ran. The blur branch
    itself still crosses alphamerge's 8-bit surface — unavoidable here, and
    immaterial, since those pixels are boxblurred anyway.
    """
    src, black, _ = clips
    fc = proc._build_blur_filter_complex(blur_filter="boxblur=8:2", **KW)
    log = _run_graph(
        src,
        black,
        fc,
        ["-map", "[vout]", "-frames:v", "1", "-pix_fmt", "yuv420p10le", "-f", "null", "-"],
        loglevel="debug",
    )
    # The base chain's own scale must not be talked into an 8-bit output.
    picked = re.findall(r"\[Parsed_scale_0 @ [^\]]+\] picking (\S+) ", log)
    collapsed = [f for f in picked if not re.search(r"(9|10|12|14|16)(le|be)$", f)]
    assert not collapsed, f"base chain negotiated down to {sorted(set(collapsed))}"


def test_all_black_mask_is_a_pixel_exact_no_op(proc, clips, tmp_path):
    """The discriminating gate: an empty mask must reproduce the base exactly.

    Measured 54.2 dB before this fix and inf after, so it catches the depth
    collapse as well as the `maskedmerge` chroma-blend regression class. The
    reference MUST be an independent decode — deriving it from the same
    `split` makes both sides degrade together and reads inf either way.

    Every frame is asserted individually via `stats_file=`, which is stricter
    than the old aggregate-only read: one damaged frame used to be able to hide
    inside the average. The count floor keeps `shortest=1` honest — a
    comparison that silently shrank to a handful of frames would otherwise
    report `inf` vacuously.
    """
    src, black, _ = clips
    stats = tmp_path / "psnr.txt"
    fc = proc._build_blur_filter_complex(blur_filter="boxblur=8:2", **KW)
    # Re-derive the untouched base from a second decode and compare.
    fc += ";[2:v]scale=in_range=full:out_range=limited[ref]"
    fc += f";[vout][ref]psnr={PSNR_PAIRING}:stats_file={stats}"
    log = _ffmpeg(
        ["-i", str(src), "-i", str(black), "-i", str(src), "-filter_complex", fc, "-f", "null", "-"]
    )
    rows = [ln for ln in stats.read_text().splitlines() if ln.startswith("n:")]
    assert len(rows) >= _frame_count(src), f"only {len(rows)} frames compared\n{log[-1500:]}"
    damaged = [ln for ln in rows if re.search(r"psnr_[yuv]:(?!inf)", ln)]
    assert not damaged, f"empty mask altered {len(damaged)}/{len(rows)} frames: {damaged[:2]}"


def _half_psnr(proc, src, mask, crop: str) -> str:
    """PSNR of one half of the composite against an independent decode.

    One crop per ffmpeg run on purpose: two `psnr` sinks in a single graph
    leaves one unmapped, and its numbers never appear.
    """
    fc = proc._build_blur_filter_complex(blur_filter="boxblur=8:2", **KW)
    fc += f";[vout]crop={crop}[a]"
    fc += f";[2:v]scale=in_range=full:out_range=limited,crop={crop}[b]"
    fc += f";[a][b]psnr={PSNR_PAIRING}"
    log = _ffmpeg(
        ["-i", str(src), "-i", str(mask), "-i", str(src), "-filter_complex", fc, "-f", "null", "-"]
    )
    m = re.search(r"PSNR y:(\S+)", log)
    assert m, log[-1500:]
    return m.group(1)


def test_box_mask_blurs_inside_and_spares_outside(proc, clips):
    """A gate that only checked "unchanged" would pass with blur disabled."""
    src, _, box = clips
    inside = _half_psnr(proc, src, box, "160:180:0:0")
    outside = _half_psnr(proc, src, box, "160:180:160:0")
    assert inside != "inf" and float(inside) < 40.0, f"masked half was not blurred (y={inside})"
    assert outside == "inf", f"unmasked half was altered (y={outside})"
