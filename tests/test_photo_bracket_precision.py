"""``fuse_bracket`` keeps a 16-bit bracket's precision through the merge.

Synthetic: a smooth, low-amplitude gradient at three exposure offsets. Eight bits cannot hold
it (a few levels across the frame); sixteen can (hundreds). No card or RAW file is needed."""

from pathlib import Path

import cv2
import numpy as np

from gomoveshift.pipeline.photos import develop as dev

H, W = 64, 1024
GAINS = (0.8, 1.0, 1.25)


def _ramp():
    # 4% of full scale across the frame: ~10 eight-bit levels, ~2600 sixteen-bit levels.
    return np.tile(np.linspace(0.40, 0.44, W, dtype=np.float64), (H, 1))


def _frames(wide, shifts=None):
    out = []
    for i, gain in enumerate(GAINS):
        disp = np.clip(_ramp() * gain, 0.0, 1.0)
        if shifts:
            dx, dy = shifts[i]
            disp = np.roll(disp, (dy, dx), (0, 1))
        peak = 65535.0 if wide else 255.0
        grey = np.rint(disp * peak).astype(np.uint16 if wide else np.uint8)
        out.append(np.stack([grey, grey, grey], 2))
    return out


def _fuse(monkeypatch, frames, size=(H, W), **kw):
    by_path = {Path(f"/synthetic/P{i}.jpg"): f for i, f in enumerate(frames)}

    def fake(path):
        rgb = by_path[Path(path)]
        wide = rgb.dtype == np.uint16
        return dev.DecodedPhoto(
            rgb=rgb, path=Path(path), decoder="fake", width=size[1], height=size[0],
            bit_depth=16 if wide else 8, is_raw=wide,
        )  # fmt: skip

    monkeypatch.setattr(dev, "decode", fake)
    return dev.fuse_bracket(list(by_path), **kw)


def _row(rgb):
    return rgb[H // 2, 64 : W - 64, 1].astype(np.float64)


def _tread(rgb):
    """Largest jump between neighbouring columns of the fused ramp, in 16-bit levels."""
    return float(np.abs(np.diff(_row(rgb))).max())


def test_a_16_bit_bracket_keeps_more_than_256_levels_in_a_gradient(monkeypatch):
    fused = _fuse(monkeypatch, _frames(wide=True), align=False)
    assert fused.rgb.dtype == np.uint16
    assert len(np.unique(_row(fused.rgb))) > 256
    # An 8-bit hop leaves ~256-level treads on the ramp; 16-bit keeps it smooth.
    assert _tread(fused.rgb) < 64


def test_an_8_bit_bracket_is_byte_identical_to_the_8_bit_merge(monkeypatch):
    frames = _frames(wide=False)
    got = _fuse(monkeypatch, frames, align=False)
    pool = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        ref = cv2.createMergeMertens().process([np.ascontiguousarray(f[..., ::-1]) for f in frames])
    finally:
        cv2.setNumThreads(pool)
    low, high = min(0.0, float(ref.min())), max(1.0, float(ref.max()))
    want = np.rint(np.clip((ref[:, :, ::-1] - low) / (high - low), 0.0, 1.0) * 255.0)
    assert got.rgb.dtype == np.uint8
    assert np.array_equal(got.rgb, want.astype(np.uint8))


def test_a_shifted_16_bit_bracket_still_aligns_and_fuses(monkeypatch):
    h, w = 240, 320
    rng = np.random.default_rng(3)
    tex = sum(
        cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), s) * s for s in (1.5, 4, 10)
    )
    tex = 0.1 + 0.9 * (tex - tex.min()) / (tex.max() - tex.min())
    frames = []
    for gain, (dx, dy) in zip((0.5, 1.0, 2.0), [(6, -4), (0, 0), (-5, 3)], strict=True):
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        moved = cv2.warpAffine(tex, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        g = np.rint(np.clip(moved * gain, 0.0, 1.0) ** (1 / 2.2) * 65535.0).astype(np.uint16)
        frames.append(np.stack([g, g, g], 2))
    fused = _fuse(monkeypatch, frames, size=(h, w))
    assert fused.rgb.dtype == np.uint16
    assert [s.applied for s in fused.bracket_shifts] == [True, False, True]
    assert len(np.unique(fused.rgb)) > 256
