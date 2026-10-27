"""The stabilise step is judged on camera motion, not luma jitter.

Synthetic frames are windows cropped from one numpy texture, so a window that
moves is a camera that moved. No video and no ffmpeg: the frame reader is
injected.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gomoveshift.pipeline import stabilize

H, W = 72, 128
N = 48


def _texture(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    big = rng.random((H + 80, W + 160)).astype(np.float32)
    # a little blur so phase correlation sees structure, not pure noise
    k = np.ones(3, dtype=np.float32) / 3
    big = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, big)
    big = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, big)
    return (40 + 180 * big).astype(np.float32)


def _frames(offsets, *, brightness=None, border=0):
    tex = _texture()
    out = []
    for i, (dx, dy) in enumerate(offsets):
        y, x = 40 + int(dy), 80 + int(dx)
        f = tex[y : y + H, x : x + W].copy()
        if brightness is not None:
            f = np.clip(f + brightness[i], 0, 255)
        if border:
            f[:border, :] = 0
            f[-border:, :] = 0
            f[:, :border] = 0
            f[:, -border:] = 0
        out.append(f.astype(np.uint8))
    return out


def _jitter(seed=1, amp=3):
    rng = np.random.default_rng(seed)
    return [(rng.integers(-amp, amp + 1), rng.integers(-amp, amp + 1)) for _ in range(N)]


def _judge(inp, out):
    def reader(path, **kw):
        return inp if str(path) == "in" else out

    return stabilize.judge_stabilisation("in", "out", read_frames=reader)


def test_removed_jitter_is_kept():
    gate = _judge(_frames(_jitter()), _frames([(0, 0)] * N))
    assert gate.input_residual > 1.0
    assert gate.output_residual < 0.2
    assert gate.border_coverage < stabilize.MOTION_MAX_BORDER_COVERAGE
    assert gate.passed, gate.reason


def test_luma_only_change_is_not_accepted():
    rng = np.random.default_rng(3)
    flicker = rng.normal(0, 12, N)
    inp = _frames([(0, 0)] * N, brightness=flicker)
    out = _frames([(0, 0)] * N, brightness=flicker * 0.1)
    means = lambda fr: [float(f.mean()) for f in fr]  # noqa: E731
    deltas = lambda m: np.diff(m)  # noqa: E731
    # the old luma gate would keep this: output luma jitter is far lower
    assert stabilize.gate_on_jitter(
        float(np.std(deltas(means(inp)))), float(np.std(deltas(means(out))))
    ).passed
    gate = _judge(inp, out)
    assert gate.input_residual < 0.1  # the camera never moved
    assert not gate.passed
    assert "shake" in gate.reason


def test_smooth_pan_is_not_shake():
    pan = [(i, 0) for i in range(N)]
    residual = stabilize.shake_residual(_frames(pan))
    assert residual < 0.15


def test_pan_survives_stabilisation_of_jitter():
    pan = [(i, 0) for i in range(N)]
    shaky = [(px + jx, py + jy) for (px, py), (jx, jy) in zip(pan, _jitter(amp=3))]
    gate = _judge(_frames(shaky), _frames(pan))
    # residual of the shaky input reflects the jitter, not the 1 px/frame pan
    assert 0.8 < gate.input_residual < 5.0
    assert gate.output_residual < 0.2
    assert gate.passed, gate.reason


def test_black_borders_are_rejected():
    gate = _judge(_frames(_jitter()), _frames([(0, 0)] * N, border=14))
    assert gate.output_residual < gate.input_residual * 0.5
    assert gate.border_coverage > stabilize.MOTION_MAX_BORDER_COVERAGE
    assert not gate.passed
    assert "border" in gate.reason


def test_shake_not_reduced_enough_is_rejected():
    jit = _jitter()
    gate = _judge(_frames(jit), _frames(jit))
    assert not gate.passed
    assert "shake" in gate.reason


def test_too_few_frames_raises():
    with pytest.raises(ValueError):
        stabilize.shake_residual(_frames([(0, 0)] * 3))


def test_read_grey_frames_decodes_the_pipe():
    size = stabilize.MOTION_FRAME_W * stabilize.MOTION_FRAME_H
    raw = bytes(range(256)) * (size // 256 + 1)
    seen = []

    def fake_run(argv, **kw):
        seen.append(argv)
        return SimpleNamespace(returncode=0, stdout=raw[:size] * 5 + b"xx", stderr=b"")

    frames = stabilize.read_grey_frames("clip.mp4", run=fake_run)
    assert len(frames) == 5
    assert frames[0].shape == (stabilize.MOTION_FRAME_H, stabilize.MOTION_FRAME_W)
    assert "gray" in seen[0] and "rawvideo" in seen[0]


def test_read_grey_frames_failure_raises():
    def fake_run(argv, **kw):
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"boom")

    with pytest.raises(RuntimeError):
        stabilize.read_grey_frames("clip.mp4", run=fake_run)


def _steps(monkeypatch, tmp_path, verdict):
    """apply_picture_steps(stabilize only) with ffmpeg faked and the judge stubbed."""

    def fake_run(cmd, *a, **k):
        cmd = list(cmd)
        if cmd and str(cmd[-1]).endswith(".mp4"):
            Path(cmd[-1]).write_bytes(b"x")
        return SimpleNamespace(returncode=0, stdout=b"", stderr="")

    def boom(*a, **k):
        raise AssertionError("luma jitter must not judge the stabilise step")

    monkeypatch.setattr(stabilize, "measure_luma_jitter", boom)
    monkeypatch.setattr(stabilize, "judge_stabilisation", verdict)
    monkeypatch.setattr(
        stabilize, "probe_stream_spec", lambda *a, **k: stabilize.StreamSpec("yuv420p")
    )
    src = tmp_path / "raw.mp4"
    src.write_bytes(b"s")
    caps = stabilize.FfmpegFilterCapabilities(
        frozenset({stabilize.VIDSTABDETECT_FILTER, stabilize.VIDSTABTRANSFORM_FILTER})
    )
    return stabilize.apply_picture_steps(
        src, tmp_path, "d", deflicker=False, stabilize=True, capabilities=caps, run=fake_run
    )


def test_wiring_keeps_on_motion_pass(monkeypatch, tmp_path):
    gate = stabilize.MotionGateResult(4.0, 0.5, 0.0, True)
    res = _steps(monkeypatch, tmp_path, lambda *a, **k: gate)
    assert res.record["stabilize_applied"] is True
    assert res.record["steps"]["stabilize"]["output_residual"] == 0.5
    assert res.path.name == "d.stabilize.mp4"


def test_wiring_falls_back_with_logged_reason(monkeypatch, tmp_path, caplog):
    gate = stabilize.MotionGateResult(4.0, 3.9, 0.0, False, "shake residual not reduced enough")
    with caplog.at_level(logging.WARNING):
        res = _steps(monkeypatch, tmp_path, lambda *a, **k: gate)
    assert res.record["stabilize_applied"] is False
    assert res.path.name == "raw.mp4"
    assert not (tmp_path / "d.stabilize.mp4").exists()
    assert "not reduced enough" in caplog.text


def test_wiring_never_raises_when_judge_fails(monkeypatch, tmp_path):
    def bad(*a, **k):
        raise ValueError("no frames")

    res = _steps(monkeypatch, tmp_path, bad)
    assert res.record["stabilize_applied"] is False
    assert "no frames" in res.record["steps"]["stabilize"]["error"]
    assert res.path.name == "raw.mp4"
