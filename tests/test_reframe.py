"""Tracked vertical reframe (Q9a): planner, filter wiring, real-ffmpeg proof.

All media here is synthetic (ffmpeg lavfi sources, generated arrays).
"""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from gomoveshift.pipeline import shorts_extractor
from gomoveshift.pipeline.ffmpeg_filters import build_crop_filter, crop_upscaled_from
from gomoveshift.pipeline.reframe import CropTrack, frame_centre, plan_crop_track

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)

SRC_W, CROP_W = 3840, 1214
LO, HI = 0.0, float(SRC_W - CROP_W)
MID = HI / 2


def _plan(targets, dt=1.0, **kw):
    return plan_crop_track(
        [i * dt for i in range(len(targets))], targets, crop_w=CROP_W, src_w=SRC_W, **kw
    )


# ── planner on synthetic signals ───────────────────────────────────────────


def test_static_subject_stays_put_within_dead_zone():
    # Subject wobbles +-20 px around the middle; the dead zone is far wider.
    rng = np.random.default_rng(1)
    centres = MID + CROP_W / 2 + rng.uniform(-20, 20, 40)
    track = _plan(list(centres))
    assert max(track.xs) - min(track.xs) < 1e-6


def test_step_move_is_speed_bounded_without_overshoot():
    target = [MID + CROP_W / 2] * 10 + [MID + CROP_W / 2 + 800] * 40
    track = _plan(target, max_speed_frac=0.10)
    xs = np.array(track.xs)
    assert np.abs(np.diff(xs)).max() <= 0.10 * CROP_W + 1e-6
    assert xs.max() <= MID + 800 + 1e-6  # never past the new target
    assert np.all(np.diff(xs) >= -1e-9)  # monotone: no overshoot or ringing
    assert xs[-1] > MID + 500  # it did follow


def test_noisy_jitter_does_not_oscillate():
    rng = np.random.default_rng(7)
    centres = MID + CROP_W / 2 + rng.normal(0, 150, 120)
    track = _plan(list(centres))
    d = np.diff(track.xs)
    d = d[np.abs(d) > 1e-6]
    flips = int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))
    assert flips <= 2


def test_edges_clamp_inside_the_frame():
    far_left = _plan([-5000.0] * 30)
    far_right = _plan([SRC_W + 5000.0] * 30)
    for t in (far_left, far_right):
        assert min(t.xs) >= LO and max(t.xs) <= HI
    assert far_left.xs[-1] == LO
    assert far_right.xs[-1] == HI


def test_planner_is_deterministic():
    t = [MID + 500 + 40 * (i % 5) for i in range(30)]
    assert _plan(t) == _plan(t)


def test_planner_refuses_bad_input():
    with pytest.raises(ValueError):
        plan_crop_track([], [], crop_w=CROP_W, src_w=SRC_W)
    with pytest.raises(ValueError):
        plan_crop_track([0.0, 1.0], [1.0], crop_w=CROP_W, src_w=SRC_W)
    with pytest.raises(ValueError):
        plan_crop_track([0.0], [1.0], crop_w=SRC_W + 2, src_w=SRC_W)


def test_frame_centre_follows_the_textured_side():
    flat = np.full((90, 160), 128, dtype=np.uint8)
    left = flat.copy()
    left[:, :40] = np.tile(np.array([0, 255], dtype=np.uint8), (90, 20))
    right = left[:, ::-1].copy()
    assert abs(frame_centre(flat) - 0.5) < 0.01
    assert frame_centre(left) < 0.4
    assert frame_centre(right) > 0.6


# ── filter string and argv ─────────────────────────────────────────────────


def test_default_crop_filter_unchanged():
    legacy = "crop=1214:2160,zscale=w=1080:h=1920:filter=spline36,setsar=1"
    assert build_crop_filter(3840, 2160) == legacy
    assert build_crop_filter(3840, 2160, track=None) == legacy


def test_tracked_filter_keeps_size_scaler_and_upscale_declaration():
    track = CropTrack(times=(0.0, 1.0, 2.0), xs=(100.0, 100.0, 300.0))
    f = build_crop_filter(3840, 2160, track=track)
    assert f.startswith("crop=1214:2160:x='")
    assert f.endswith(",zscale=w=1080:h=1920:filter=spline36,setsar=1")
    assert crop_upscaled_from(3840, 2160) is None
    assert crop_upscaled_from(1920, 1080) == [606, 1080]


def _capture_render(monkeypatch, tmp_path, **kw):
    source = tmp_path / "master.mp4"
    source.touch()
    out = tmp_path / "short.mp4"
    captured: list[list[str]] = []

    def fake_run(cmd, *a, **k):
        captured.append(list(cmd))
        out.write_bytes(b"x" * 4096)
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(shorts_extractor.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "gomoveshift.pipeline.ffmpeg_runner.validate_video",
        lambda *a, **k: MagicMock(valid=True, reason="", duration=30.0),
    )
    ok = shorts_extractor.render_short_ffmpeg(
        source_video=source,
        output_path=out,
        start_time=42.0,
        duration=30.0,
        location="Somewhere",
        src_width=3840,
        src_height=2160,
        use_nvenc=True,
        **kw,
    )
    assert ok is True
    return captured


def test_default_reframe_argv_is_identical_to_center(monkeypatch, tmp_path):
    default = _capture_render(monkeypatch, tmp_path)
    center = _capture_render(monkeypatch, tmp_path, reframe="center")
    assert default == center
    vf = default[0][default[0].index("-vf") + 1]
    assert vf.startswith(build_crop_filter(3840, 2160) + ",")


def test_tracked_reframe_changes_only_the_crop_and_stays_muted(monkeypatch, tmp_path):
    track = CropTrack(times=(0.0, 1.0), xs=(600.0, 900.0))
    monkeypatch.setattr(shorts_extractor, "plan_tracked_crop", lambda *a, **k: track)
    base = _capture_render(monkeypatch, tmp_path)[0]
    tracked = _capture_render(monkeypatch, tmp_path, reframe="tracked")[0]
    i = base.index("-vf")
    assert tracked[:i] == base[:i] and tracked[i + 2 :] == base[i + 2 :]
    assert "x='" in tracked[i + 1]
    assert "-an" in tracked and "-c:a" not in tracked and "-af" not in tracked


def test_unknown_reframe_mode_refused(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        _capture_render(monkeypatch, tmp_path, reframe="zoom")


# ── real ffmpeg (synthetic source) ─────────────────────────────────────────

W, H, FPS, DUR = 320, 180, 10, 3
OUT_W, OUT_H = 90, 160


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    assert r.returncode == 0, r.stderr.decode()[-800:]
    return r


def _make_source(path: Path, blurred: bool | None) -> None:
    graph = f"testsrc2=s={W}x{H}:r={FPS}:d={DUR}"
    if blurred is not None:
        # High-frequency synthetic noise, so blur is unmistakable in the gradient.
        graph = f"nullsrc=s={W}x{H}:r={FPS}:d={DUR},format=yuv420p,geq=lum='random(1)*255':cb=128:cr=128"
    if blurred:
        graph += ",split=2[a][b];[b]crop=60:60:130:60,boxblur=12[bl];[a][bl]overlay=130:60"
    _run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", graph]
        + ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "8", str(path)]
    )


def _render(src: Path, dst: Path, track) -> np.ndarray:
    vf = build_crop_filter(W, H, OUT_W, OUT_H, track=track)
    _run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf, "-an"]
        + ["-pix_fmt", "yuv420p", "-c:v", "libx264", str(dst)]
    )
    raw = _run(
        ["ffmpeg", "-v", "error", "-i", str(dst), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, OUT_H, OUT_W)


@needs_ffmpeg
def test_time_varying_crop_runs_and_output_has_declared_size(tmp_path):
    src = tmp_path / "synthetic_src.mp4"
    _make_source(src, blurred=None)
    track = CropTrack(times=(0.0, 1.0, 2.0, 3.0), xs=(0.0, 0.0, 100.0, 100.0))
    out = tmp_path / "out.mp4"
    frames = _render(src, out, track)
    assert frames.shape[1:] == (OUT_H, OUT_W)
    assert frames.shape[0] >= FPS * DUR - 2
    # The crop really moves: early and late frames differ.
    assert np.abs(frames[2].astype(int) - frames[-3].astype(int)).mean() > 2
    probe = _run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height"]
        + ["-of", "csv=p=0", str(out)]
    )
    assert probe.stdout.decode().split() == [f"{OUT_W},{OUT_H}"]


@needs_ffmpeg
def test_blurred_region_stays_blurred_inside_tracked_crop(tmp_path):
    sharp_src, blur_src = tmp_path / "sharp.mp4", tmp_path / "blur.mp4"
    _make_source(sharp_src, blurred=False)
    _make_source(blur_src, blurred=True)
    # Static track (still the time-varying expression path): crop spans x=100..200.
    track = CropTrack(times=(0.0, 3.0), xs=(100.0, 100.0))
    sharp = _render(sharp_src, tmp_path / "o1.mp4", track)
    blur = _render(blur_src, tmp_path / "o2.mp4", track)
    # Source box x130..190,y60..120 maps into the 100-wide crop scaled to 90 wide.
    sc = OUT_W / 100.0
    x0, x1 = int((130 - 100) * sc) + 2, int((190 - 100) * sc) - 2
    y0, y1 = int(60 * OUT_H / H) + 2, int(120 * OUT_H / H) - 2

    def grad(a):
        return np.abs(np.diff(a[5, y0:y1, x0:x1].astype(float), axis=1)).mean()

    assert grad(blur) < 0.5 * grad(sharp)
