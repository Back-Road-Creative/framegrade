"""Ken Burns reel: the pure motion planner, its upstream gates, one real render.

The planner tests need neither ffmpeg nor a fixture — that is the point of
splitting the motion out of the renderer. Only the last test shells out.
"""

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from gomoveshift.pipeline.encoding_constants import (
    LONGFORM_COLOR_PRIMARIES,
    LONGFORM_COLOR_RANGE,
    LONGFORM_COLOR_TRC,
    LONGFORM_COLORSPACE,
)
from gomoveshift.pipeline.photos.reel import (
    REEL_FPS,
    REEL_PAN_FROM,
    REEL_PAN_TO,
    REEL_SIZE,
    REEL_ZOOM,
    ReelError,
    build_encode_cmd,
    plan_motion,
    render_reel,
)

SRC = (6000, 4000)
OUT = (1080, 1920)
ASPECT = OUT[0] / OUT[1]


def _plan(index=0, source=SRC, duration=2.0, fps=10):
    return plan_motion(source, OUT, duration, fps, index=index)


@pytest.mark.parametrize("source", [SRC, (3000, 4500), (900, 4000)])
def test_every_rect_carries_the_output_aspect_and_fits_the_source(source):
    """Aspect and bounds are arithmetic, not tolerance: a landscape, a portrait
    and a source narrower than 9:16 all crop inside themselves at 9:16."""
    src_w, src_h = source
    for index in (0, 1):
        for r in _plan(index=index, source=source):
            assert r.width == r.height * ASPECT
            assert 0.0 <= r.x and r.x + r.width <= src_w
            assert 0.0 <= r.y and r.y + r.height <= src_h


def test_motion_is_monotonic_and_alternates_direction_per_still():
    """Every still drifts one way only; consecutive stills push in and pull out
    so a reel of six does not drone."""
    push = _plan(index=0)
    pull = _plan(index=1)
    assert all(b.height < a.height for a, b in zip(push, push[1:]))
    assert all(b.x > a.x for a, b in zip(push, push[1:]))
    assert all(b.height > a.height for a, b in zip(pull, pull[1:]))
    assert all(b.x < a.x for a, b in zip(pull, pull[1:]))


def test_endpoints_are_the_planned_zoom_and_pan():
    rects = _plan()
    assert rects[0].height == pytest.approx(4000.0)
    assert rects[-1].height == pytest.approx(4000.0 / (1.0 + REEL_ZOOM))
    assert rects[0].x == pytest.approx(REEL_PAN_FROM * (SRC[0] - rects[0].width))
    assert rects[-1].x == pytest.approx(REEL_PAN_TO * (SRC[0] - rects[-1].width))


def test_frame_count_follows_duration_and_never_degenerates():
    assert len(_plan(duration=3.0, fps=30)) == 90
    assert len(_plan(duration=0.01, fps=30)) == 2  # two frames is the floor


def test_planned_motion_is_deterministic():
    assert _plan() == _plan()


def test_reel_refuses_bad_inputs_before_decoding_anything(tmp_path):
    still = tmp_path / "a.jpg"
    cv2.imwrite(str(still), np.zeros((40, 60, 3), np.uint8))
    with pytest.raises(ReelError, match="at least one"):
        render_reel([], tmp_path / "out.mp4")
    with pytest.raises(ReelError, match="unreadable"):
        render_reel([still, tmp_path / "gone.jpg"], tmp_path / "out.mp4")
    with pytest.raises(ReelError, match="writable"):
        render_reel([still], tmp_path / "nope" / "out.mp4")


def test_encode_cmd_states_the_rgb_to_yuv_range_and_stamps_it():
    """The reel feeds ffmpeg true full-range rgb24, so the RGB->YUV hop is the one
    conversion whose range convention must be *stated* rather than inherited from
    swscale's defaults — and the stream must be tagged to match what was written."""
    cmd = build_encode_cmd(Path("/tmp/reel.mp4"), REEL_FPS, REEL_SIZE)
    assert "-vf" in cmd, "the RGB->YUV conversion must be an explicit filter"
    vf = cmd[cmd.index("-vf") + 1]
    assert "in_range=full" in vf, f"source rgb24 is full-range: {vf!r}"
    assert "out_range=tv" in vf, f"output must be limited-range, explicitly: {vf!r}"
    assert "out_color_matrix=bt709" in vf, f"matrix must be stated: {vf!r}"
    for flag, value in (
        ("-color_range", LONGFORM_COLOR_RANGE),
        ("-colorspace", LONGFORM_COLORSPACE),
        ("-color_primaries", LONGFORM_COLOR_PRIMARIES),
        ("-color_trc", LONGFORM_COLOR_TRC),
    ):
        assert flag in cmd, f"{flag} is not stamped on the reel encode"
        assert cmd[cmd.index(flag) + 1] == value


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_actually_converts_to_the_range_it_tags(tmp_path):
    """Verify the conversion, not just the tag: a black/white still must come out
    coded 16-235. A tag that disagrees with the pixels is worse than no tag."""
    still = tmp_path / "split.png"
    frame = np.zeros((2000, 1000, 3), np.uint8)
    frame[1000:, :, :] = 255
    cv2.imwrite(str(still), frame)

    out = render_reel([still], tmp_path / "reel.mp4", seconds=0.4, fps=10)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=color_range,color_space,color_primaries,color_transfer",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True,
    )  # fmt: skip
    assert probe.stdout.strip() == "tv,bt709,bt709,bt709"

    # Decode the coded Y plane with no conversion (the stream already IS yuv420p),
    # so these are the numbers libx264 was handed.
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(out), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        capture_output=True,
    )  # fmt: skip
    luma = np.frombuffer(raw.stdout[: 1080 * 1920], dtype=np.uint8)
    lo, hi = np.percentile(luma, 1), np.percentile(luma, 99)
    assert lo > 8, f"black crushed below the limited-range floor (full-range write?): {lo}"
    assert hi < 247, f"white pushed above the limited-range ceiling: {hi}"
    assert lo < 40 and hi > 210, f"degenerate frame, not a black/white split: {lo}/{hi}"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_writes_a_deterministic_vertical_mp4(tmp_path):
    """Two runs over the same stills give a byte-identical container here, so
    determinism is asserted at that level rather than on decoded frames."""
    stills = []
    for i in range(2):
        path = tmp_path / f"still_{i}.jpg"
        frame = np.tile(np.arange(600, dtype=np.uint8).reshape(1, 600, 1), (400, 1, 3))
        cv2.imwrite(str(path), frame + i)
        stills.append(path)

    outs = [render_reel(stills, tmp_path / f"reel_{i}.mp4", seconds=0.4, fps=10) for i in range(2)]
    args = "-v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of csv=p=0"
    probe = subprocess.run(["ffprobe", *args.split(), str(outs[0])], capture_output=True, text=True)
    assert probe.stdout.strip() == "1080,1920,8"
    assert outs[0].read_bytes() == outs[1].read_bytes()
