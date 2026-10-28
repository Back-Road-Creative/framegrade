"""Insert still push-in motion and clip upscale disclosure (synthetic media, ffmpeg mocked)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from gomoveshift.pipeline import inserts as ins_mod
from gomoveshift.pipeline.encoding_constants import RENDITIONS
from gomoveshift.pipeline.inserts import Insert, MediaInfo, _encode_argv, plan_inserts
from gomoveshift.pipeline.route_story import hud
from gomoveshift.publishing.brand_policy import MediaProfile

LF = Path("/work/lf.mp4")
KEYFRAMES = [float(t) for t in range(0, 100, 2)]
W, H = 320, 180  # synthetic LF frame
RENDITION = replace(RENDITIONS["youtube_longform"], width=W, height=H)
CLIP_SIZES = {"small.mp4": (160, 90), "big.mp4": (640, 360), "tall.mp4": (90, 160)}


def _probe(path: Path) -> MediaInfo:
    if Path(path) == LF:
        return MediaInfo(width=W, height=H, fps=10.0, duration=100.0)
    w, h = CLIP_SIZES[Path(path).name]
    return MediaInfo(width=w, height=h, fps=30.0, duration=6.0)


def _still(path: Path) -> Image.Image:  # synthetic landscape still with detail
    img = Image.new("RGB", (640, 360), (40, 80, 120))
    d = ImageDraw.Draw(img)
    for x in range(0, 640, 40):
        d.line([(x, 0), (x, 360)], fill=(255, 255, 255), width=3)
    return img


def _plan(tmp_path, inserts, **kw):
    return plan_inserts(
        LF,
        inserts,
        profile=MediaProfile(upscale="resample"),
        rendition=RENDITION,
        probe=_probe,
        keyframes=lambda p: KEYFRAMES,
        load_still=_still,
        work_dir=tmp_path,
        **kw,
    )


def test_default_motion_is_byte_identical(tmp_path):
    ins = [Insert(10.0, Path("a.jpg"), hold_s=2.0)]
    default = _plan(tmp_path / "d", ins)[0]
    explicit = _plan(tmp_path / "e", ins, still_motion="none")[0]
    assert default.card_frames == 0 and default.card_path.name == "insert_00_card.png"
    assert default.card_path.read_bytes() == explicit.card_path.read_bytes()
    static = ins_mod._render_card(_still(Path("a")), W, H, "full")
    assert Image.open(default.card_path).tobytes() == static.tobytes()
    argv = _encode_argv(LF, default, Path("out.mp4"))
    assert argv[argv.index("-loop") :][:6] == ["-loop", "1", "-framerate", "10.0", "-t", "2.0"]
    assert argv[argv.index("-i", argv.index("-loop")) + 1] == str(default.card_path)
    out = Path("out.mp4")
    assert _encode_argv(LF, replace(default, card_path=Path("c.png")), out) == _encode_argv(
        LF, replace(explicit, card_path=Path("c.png")), out
    )


def test_unknown_motion_rejected(tmp_path):
    with pytest.raises(ValueError, match="still_motion"):
        _plan(tmp_path, [Insert(10.0, Path("a.jpg"))], still_motion="zoom")


def test_push_in_progress_is_linear_and_eased_by_card(tmp_path, monkeypatch):
    seen = []
    real = hud.still_card_frame
    monkeypatch.setattr(hud, "still_card_frame", lambda *a: (seen.append(a[3]), real(*a))[1])
    plan = _plan(tmp_path, [Insert(10.0, Path("a.jpg"), hold_s=2.0)], still_motion="push_in")[0]
    assert plan.card_frames == 20  # 2 s x 10 fps
    assert seen[0] == 0.0 and seen[-1] == 1.0
    assert all(b > a for a, b in zip(seen, seen[1:]))
    # the recipe's easing is smoothstep and capped at the existing max scale
    scales = [1.0 + (hud.STILL_PUSH_IN_MAX_SCALE - 1.0) * hud._ease_in_out(p) for p in seen]
    assert all(b >= a for a, b in zip(scales, scales[1:]))
    assert scales[-1] == pytest.approx(hud.STILL_PUSH_IN_MAX_SCALE)
    assert scales[1] - scales[0] < scales[10] - scales[9]  # eased in, fastest mid-hold


def test_push_in_frames_zoom_in_and_encode_reads_sequence(tmp_path):
    plan = _plan(tmp_path, [Insert(10.0, Path("a.jpg"), hold_s=1.0)], still_motion="push_in")[0]
    frames = [Path(str(plan.card_path) % (k + 1)) for k in range(plan.card_frames)]
    assert all(f.exists() for f in frames) and not Path(str(plan.card_path) % 0).exists()
    first, last = (Image.open(f).convert("RGB") for f in (frames[0], frames[-1]))
    assert first.size == last.size == (W, H)
    assert first.tobytes() != last.tobytes()
    # grid lines spread outward from the centre as the frame pushes in: the line
    # nearest the left edge sits further left in the first frame than in the last
    row = lambda im: [im.getpixel((x, H // 2)) for x in range(W)]  # noqa: E731
    bright = lambda im: [x for x, p in enumerate(row(im)) if p[0] > 200]  # noqa: E731
    assert bright(first) != bright(last)
    argv = _encode_argv(LF, plan, Path("out.mp4"))
    assert "-loop" not in argv
    assert argv[argv.index("-framerate") : argv.index("-framerate") + 4] == [
        "-framerate", "10.0", "-i", str(plan.card_path),
    ]  # fmt: skip
    assert "overlay=eof_action=pass" in argv[argv.index("-filter_complex") + 1]


@pytest.mark.parametrize(
    ("clip", "upscaled", "factor"),
    [("small.mp4", True, 2.0), ("big.mp4", False, 1.0), ("tall.mp4", True, 3.5555555)],
)
def test_clip_upscale_disclosure(tmp_path, clip, upscaled, factor):
    plan = _plan(tmp_path, [Insert(10.0, Path(clip))])[0]
    assert plan.kind == "clip"
    assert plan.upscaled is upscaled
    assert plan.upscale_factor == pytest.approx(factor)
    assert (plan.upscaled_from is not None) is upscaled


def test_still_plan_reports_no_upscale(tmp_path):
    plan = _plan(tmp_path, [Insert(10.0, Path("a.jpg"))])[0]
    assert plan.upscaled is False and plan.upscale_factor == 1.0
