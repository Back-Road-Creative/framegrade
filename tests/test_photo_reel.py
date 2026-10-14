"""Ken Burns reel: the pure motion planner, its upstream gates, one real render.

The planner tests need neither ffmpeg nor a fixture — that is the point of
splitting the motion out of the renderer. Only the last test shells out.
"""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

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
    # White rims: the developed still crops a black border, so black must be inside.
    frame = np.full((2000, 1000, 3), 255, np.uint8)
    frame[500:1500, 100:900, :] = 0
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


def test_reel_and_still_card_share_developed_frame(tmp_path, monkeypatch):
    """The reel takes its pixels from ``hud._load_still`` -- the one graded,
    cached still the route-story cards and inserts use -- not a private decode."""
    from PIL import Image

    from gomoveshift.pipeline import inserts
    from gomoveshift.pipeline.photos import reel
    from gomoveshift.pipeline.route_story import hud

    developed = Image.new("RGB", (400, 300), (10, 200, 30))
    loads = []

    def fake_load(path, *, source="developed"):
        loads.append(Path(path))
        return developed

    monkeypatch.setattr(hud, "_load_still", fake_load)
    monkeypatch.setattr(
        reel, "decode", lambda *_a, **_k: pytest.fail("reel decoded itself"), raising=False
    )

    class _Sink:
        def __init__(self):
            self.data = bytearray()

        def write(self, b):
            self.data += b

        def close(self):
            pass

    class _Proc:
        def __init__(self, cmd, stdin=None, stderr=None):
            self.stdin = _Sink()
            procs.append(self)
            Path(cmd[-1]).write_bytes(b"x")

        def wait(self, timeout=None):
            return 0

    procs = []
    monkeypatch.setattr(reel.subprocess, "Popen", _Proc)
    still = tmp_path / "a.jpg"
    still.write_bytes(b"x")
    reel.render_reel([still], tmp_path / "o.mp4", seconds=0.2, fps=10, size=(90, 160))

    assert loads == [still]
    first = np.frombuffer(bytes(procs[0].stdin.data[: 90 * 160 * 3]), np.uint8)
    assert set(first.reshape(-1, 3)[0]) == {10, 200, 30}
    assert inserts._load_still_default(still) is developed


def _gradient_jpeg(path, *, orientation=None, icc=None):
    from PIL import Image

    ramp = np.linspace(40, 215, 600, dtype=np.uint8)
    frame = np.stack([np.tile(ramp, (300, 1))] * 3, axis=2)
    exif = Image.Exif()
    if orientation:
        exif[274] = orientation
    kwargs = {"exif": exif}
    if icc is not None:
        kwargs["icc_profile"] = icc
    Image.fromarray(frame).save(path, "JPEG", **kwargs)


def test_shared_developed_still_is_upright_for_exif_oriented_photos(tmp_path):
    """Every consumer of the shared loader gets the decode()'s orientation: a
    phone photo tagged Orientation=6 comes out portrait, not sideways."""
    from gomoveshift.pipeline.route_story import hud

    still = tmp_path / "phone.jpg"
    _gradient_jpeg(still, orientation=6)

    graded = hud._grade_still_from_file(still, "developed")

    assert graded.height > graded.width


def test_shared_developed_still_refuses_an_unusable_icc_profile(tmp_path):
    """...and its ICC gate: a profile ImageCms cannot convert is refused, never
    passed through as sRGB numbers."""
    from gomoveshift.pipeline.photos.develop import UnusableColorProfileError
    from gomoveshift.pipeline.route_story import hud

    still = tmp_path / "bad.jpg"
    _gradient_jpeg(still, icc=b"not an icc profile")

    with pytest.raises(UnusableColorProfileError):
        hud._grade_still_from_file(still, "developed")


def test_route_story_card_renders_an_exif_rotated_still_upright(tmp_path, monkeypatch):
    """The route-story card path takes its still from the shared loader, so a
    phone photo tagged Orientation=6 reaches the card portrait, not sideways."""
    from gomoveshift.pipeline.route_story import assembly

    still = tmp_path / "phone.jpg"
    _gradient_jpeg(still, orientation=6)
    seen = []

    def capture(card_still, width, height, progress):
        seen.append(card_still)
        return card_still.resize((width, height))

    monkeypatch.setattr(assembly, "still_card_frame", capture)
    monkeypatch.setattr(assembly, "_save_frame", lambda *_a, **_k: None)

    assembly._append_photo(tmp_path, 0, SimpleNamespace(source=still), 2, 0.5, 90, 160, mapped=True)

    assert seen and seen[0].height > seen[0].width


def test_insert_engine_still_loader_keeps_webp_upright(tmp_path):
    """.webp is an accepted insert still suffix: it must still load, with its
    EXIF orientation applied, through the shared loader."""
    from PIL import Image

    from gomoveshift.pipeline import inserts

    assert ".webp" in inserts.STILL_SUFFIXES
    ramp = np.linspace(40, 215, 600, dtype=np.uint8)
    frame = np.stack([np.tile(ramp, (300, 1))] * 3, axis=2)
    exif = Image.Exif()
    exif[274] = 6
    still = tmp_path / "phone.webp"
    Image.fromarray(frame).save(still, "WEBP", exif=exif)

    loaded = inserts._load_still_default(still)

    assert loaded.height > loaded.width


def test_shared_loader_refuses_an_unusable_icc_profile_on_a_webp(tmp_path):
    """A suffix outside develop.SUPPORTED_EXTENSIONS (an insert's .webp) takes the
    Pillow branch of the loader; it must hold the same ICC gate, not pass raw numbers."""
    from PIL import Image

    from gomoveshift.pipeline.photos.develop import UnusableColorProfileError
    from gomoveshift.pipeline.route_story import hud

    ramp = np.linspace(40, 215, 600, dtype=np.uint8)
    frame = np.stack([np.tile(ramp, (300, 1))] * 3, axis=2)
    still = tmp_path / "bad.webp"
    Image.fromarray(frame).save(still, "WEBP", icc_profile=b"not an icc profile")

    with pytest.raises(UnusableColorProfileError):
        hud._grade_still_from_file(still, "developed")
