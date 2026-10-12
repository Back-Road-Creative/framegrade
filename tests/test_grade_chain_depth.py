"""No 8-bit stage may appear anywhere in an emitted grade chain.

Nothing pinned a working format, so with a standard 8-bit GoPro source the
whole grade — curve bending, gamma stretch, range squeeze — ran at source
depth via ffmpeg's own negotiation (an 8-bit source picks `rgb24`), quantising
every intermediate. Grading at >=10 bits with a dithered downconvert at the
encoder is the standard anti-banding fix.

The gate here is not a string check: it runs the REAL emitted chain through
ffmpeg and reads back every format ffmpeg negotiated for it. That is what
catches the next filter edit that silently reintroduces a collapse — the way
`eq` did. Rationale: docs/pipeline-internals.md.
"""

import re
import subprocess
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.single_pass import GRADE_WORKING_FORMAT, SinglePassProcessor

GRADE = "curves=m='0/0.03 1/0.98'"

# A pixel format is >=10-bit only if its name says so. Everything else --
# yuv420p, rgb24, gbrp, yuva420p -- is 8 bits per component.
_HIGH_DEPTH = re.compile(r"(9|10|12|14|16)(le|be)$|^p0(1|1[026])")


def _is_high_depth(fmt: str) -> bool:
    return bool(_HIGH_DEPTH.search(fmt))


def negotiated_formats(chain: str, source_fmt: str = "yuv420p10le") -> list[str]:
    """Every format ffmpeg CONVERTS INTO while running `chain`.

    ffmpeg's debug log records both ends of each conversion
    (`... fmt:X ... -> w:.. h:.. fmt:Y ...`). Only the destinations are read:
    the source's own depth is a given, but every format the graph converts
    *into* is a working depth we chose. The chain's final output format alone
    would not reveal a mid-chain collapse that later widens back — which is
    exactly how `eq` hid.

    Probed against a 10-bit output (`-pix_fmt`), matching the real encode, so
    any 8-bit destination is a genuine collapse rather than the tail convert.
    """
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "debug",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=s=64x64:d=1:r=1,format={source_fmt}",
            "-vf",
            chain,
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuv420p10le",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"chain failed to run: {chain}\n{proc.stderr[-1500:]}"
    return re.findall(r"-> w:\d+ h:\d+ fmt:(\S+)", proc.stderr)


def _processor():
    config = MagicMock()
    config.output.audio_bitrate = "192k"
    return SinglePassProcessor(config)


def _kwargs(color_filter=GRADE):
    return dict(
        color_filter=color_filter,
        location="",
        font_path="",
        duration=60.0,
        fade_video=False,
        skip_ost=True,
    )


def test_the_depth_gate_can_actually_fail():
    """A gate that cannot go red is decoration. `eq` is the known collapser."""
    fmts = negotiated_formats(f"{GRADE},eq=saturation=1.1")
    assert any(not _is_high_depth(f) for f in fmts), (
        f"detector missed a known 8-bit stage; formats seen: {sorted(set(fmts))}"
    )


def test_working_format_is_high_depth():
    assert _is_high_depth(GRADE_WORKING_FORMAT)


@pytest.mark.parametrize("builder", ["_build_video_filter", "_build_blur_filter_complex"])
def test_grade_chain_pins_the_working_format_ahead_of_the_grade(builder):
    extra = {"blur_filter": "boxblur=4:2"} if "blur" in builder else {}
    out = getattr(_processor(), builder)(**_kwargs(), **extra)
    assert out.index(f"format={GRADE_WORKING_FORMAT}") < out.index("curves=")


def test_no_pin_when_no_grade_runs():
    """A 16-bit RGB working copy of a 4K frame is not free — skip it when
    there is no grade to protect."""
    out = _processor()._build_video_filter(**_kwargs(color_filter="null"))
    assert f"format={GRADE_WORKING_FORMAT}" not in out


def test_emitted_chain_has_no_8bit_stage():
    """The real gate: run what we emit and read back every negotiated format."""
    chain = _processor()._build_video_filter(**_kwargs())
    collapsed = [f for f in negotiated_formats(chain) if not _is_high_depth(f)]
    assert not collapsed, f"8-bit stage in the emitted grade chain: {sorted(set(collapsed))}"


def test_eight_bit_source_is_graded_at_high_depth():
    """The point of the pin: an 8-bit GoPro source must not grade at 8 bits."""
    chain = _processor()._build_video_filter(**_kwargs())
    collapsed = [
        f for f in negotiated_formats(chain, source_fmt="yuv420p") if not _is_high_depth(f)
    ]
    assert not collapsed, f"8-bit source still grades at 8 bits: {sorted(set(collapsed))}"


def test_pin_does_not_defeat_the_limited_range_squeeze():
    """Guard the interaction: the working format must not turn the
    full->limited squeeze into a no-op (that ships a full-range upload)."""
    chain = _processor()._build_video_filter(**_kwargs(color_filter="null"))
    pinned = f"format={GRADE_WORKING_FORMAT}," + chain
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-f",
            "lavfi",
            "-i",
            "gradients=s=256x64:d=1:r=1:c0=black:c1=white,format=gbrp",
            "-vf",
            f"{pinned},signalstats,metadata=print",
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    ymin = float(re.search(r"YMIN=([0-9.]+)", proc.stderr).group(1))
    ymax = float(re.search(r"YMAX=([0-9.]+)", proc.stderr).group(1))
    assert (ymin, ymax) == (16.0, 235.0), f"squeeze broken: {ymin}..{ymax}"
