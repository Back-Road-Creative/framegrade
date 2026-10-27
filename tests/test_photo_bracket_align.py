"""``fuse_bracket`` aligns a handheld bracket's frames to the middle exposure before the merge.

Synthetic: one textured scene rendered at three exposures, then the outer frames moved by known
shifts (integer and sub-pixel). No card or RAW file is needed."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from gomoveshift.pipeline.photos import develop as dev

H, W = 240, 320
CROP = (slice(24, H - 24), slice(24, W - 24))  # past the replicated border
GAINS = (0.5, 1.0, 2.0)  # under, middle, over


def _scene():
    rng = np.random.default_rng(3)
    # Three octaves of texture: MTB needs structure at more than one scale to lock on.
    t = sum(
        cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), s) * s for s in (1.5, 4, 10)
    )
    t = (t - t.min()) / (t.max() - t.min())
    return 0.1 + 0.9 * t


def _shift(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _frames(shifts, wide=False):
    """Three exposures of the scene; frame i is displaced by ``shifts[i]`` (dx, dy)."""
    scene, out = _scene(), []
    peak = 65535.0 if wide else 255.0
    for gain, (dx, dy) in zip(GAINS, shifts, strict=True):
        disp = np.clip(_shift(scene, dx, dy) * gain, 0.0, 1.0) ** (1 / 2.2)
        grey = np.rint(disp * peak).astype(np.uint16 if wide else np.uint8)
        out.append(np.stack([grey, grey, grey], 2))
    return out


def _patch_decode(monkeypatch, frames):
    by_path = {Path(f"/synthetic/F{i}.jpg"): f for i, f in enumerate(frames)}

    def fake(path):
        rgb = by_path[Path(path)]
        wide = rgb.dtype == np.uint16
        return dev.DecodedPhoto(
            rgb=rgb, path=Path(path), decoder="fake", width=W, height=H,
            bit_depth=16 if wide else 8, is_raw=wide,
        )  # fmt: skip

    monkeypatch.setattr(dev, "decode", fake)
    return list(by_path)


def _mse(a, b):
    d = a[CROP].astype(np.float64) - b[CROP].astype(np.float64)
    return float(np.mean(d * d))


def _fuse(monkeypatch, shifts, **kw):
    paths = _patch_decode(monkeypatch, _frames(shifts, kw.pop("wide", False)))
    return dev.fuse_bracket(paths, **kw)


def test_integer_shifts_are_recovered_and_the_fusion_gets_closer(monkeypatch):
    ref = _fuse(monkeypatch, [(0, 0)] * 3, align=False)
    moved = [(6, -4), (0, 0), (-5, 3)]
    aligned = _fuse(monkeypatch, moved)
    plain = _fuse(monkeypatch, moved, align=False)
    got = {s.path.name: (s.dx, s.dy) for s in aligned.bracket_shifts}
    # The frame is moved BY (dx, dy); alignment moves it back by the negation.
    assert got["F0.jpg"] == pytest.approx((-6, 4), abs=1)
    assert got["F2.jpg"] == pytest.approx((5, -3), abs=1)
    assert [s.applied for s in aligned.bracket_shifts] == [True, False, True]  # middle = reference
    assert _mse(aligned.rgb, ref.rgb) < 0.25 * _mse(plain.rgb, ref.rgb)


def test_subpixel_shifts_land_within_a_pixel_and_still_help(monkeypatch):
    ref = _fuse(monkeypatch, [(0, 0)] * 3, align=False)
    moved = [(3.5, -2.5), (0, 0), (-2.5, 1.5)]
    aligned = _fuse(monkeypatch, moved)
    plain = _fuse(monkeypatch, moved, align=False)
    got = {s.path.name: (s.dx, s.dy) for s in aligned.bracket_shifts}
    assert got["F0.jpg"] == pytest.approx((-3.5, 2.5), abs=1)
    assert got["F2.jpg"] == pytest.approx((2.5, -1.5), abs=1)
    assert _mse(aligned.rgb, ref.rgb) < _mse(plain.rgb, ref.rgb)


def test_a_tripod_bracket_is_bit_identical_to_the_unaligned_merge(monkeypatch):
    on = _fuse(monkeypatch, [(0, 0)] * 3)
    off = _fuse(monkeypatch, [(0, 0)] * 3, align=False)
    assert np.array_equal(on.rgb, off.rgb)
    assert all((s.dx, s.dy) == (0, 0) and not s.applied for s in on.bracket_shifts)


def test_opt_out_is_today_s_path_and_records_nothing(monkeypatch):
    off = _fuse(monkeypatch, [(6, -4), (0, 0), (-5, 3)], align=False)
    assert off.bracket_shifts == ()


def test_a_shift_over_the_bound_is_skipped_and_recorded(monkeypatch):
    monkeypatch.setattr(dev, "BRACKET_ALIGN_MAX_FRACTION", 0.02)  # 4.8 px of a 240 px side
    moved = [(8, -6), (0, 0), (-2, 1)]  # F0 is measured at ~(-8, 6): over; F2 at ~(2, -1): under
    aligned = _fuse(monkeypatch, moved)
    plain = _fuse(monkeypatch, moved, align=False)
    by = {s.path.name: s for s in aligned.bracket_shifts}
    assert not by["F0.jpg"].applied and max(abs(by["F0.jpg"].dx), abs(by["F0.jpg"].dy)) > 4.8
    assert by["F2.jpg"].applied
    assert not np.array_equal(aligned.rgb, plain.rgb)  # F2 was still moved


def test_sixteen_bit_input_stays_sixteen_bit_through_alignment(monkeypatch):
    fused = _fuse(monkeypatch, [(5, -3), (0, 0), (-4, 2)], wide=True)
    assert fused.rgb.dtype == np.uint16
    assert fused.rgb.shape == (H, W, 3)
    assert fused.rgb.max() > 255  # real 16-bit codes, not 8-bit values in a wide container
    assert any(s.applied for s in fused.bracket_shifts)


def _spied_run(tmp_path, monkeypatch, extra, photos_cfg=None):
    """Run ``photos develop --hdr`` over a three-frame bracket; return the ``align`` it passed."""
    from PIL import Image

    import gomoveshift.config as config_module
    from gomoveshift.cli.handlers import photos as photos_handler
    from gomoveshift.cli.parser import _build_parser
    from gomoveshift.config import Config, PhotosConfig

    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    cfg = Config(photos=photos_cfg or PhotosConfig())
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: cfg)
    paths = []
    for i, (at, denom) in enumerate([("12:00:08", 4000), ("12:00:20", 500), ("12:00:32", 62.5)]):
        exif = Image.Exif()
        exif[0x8769] = {0x9003: f"2000:01:01 {at}", 0x8827: 100, 0x829A: 1 / denom}
        paths.append(tmp_path / f"DSC_{i:04d}.jpg")
        Image.fromarray(_frames([(0, 0)] * 3)[i], "RGB").save(paths[-1], exif=exif)
    seen = []
    import gomoveshift.pipeline.photos.develop as develop_mod

    real = develop_mod.fuse_bracket
    monkeypatch.setattr(
        develop_mod, "fuse_bracket", lambda p, align=True: (seen.append(align), real(p, align))[1]
    )
    argv = ["photos", "develop", *map(str, paths), "--styles", "natural", "--hdr"]
    args = _build_parser().parse_args([*argv, "--out", str(tmp_path / "o"), *extra])
    assert photos_handler.cmd_photos(args, None) == 0
    return seen


def test_alignment_is_on_by_default_and_the_flag_and_config_turn_it_off(tmp_path, monkeypatch):
    for d in "abc":
        (tmp_path / d).mkdir()
    from gomoveshift.config import PhotosConfig

    assert PhotosConfig().develop_hdr_align is True
    assert _spied_run(tmp_path / "a", monkeypatch, []) == [True]
    assert _spied_run(tmp_path / "b", monkeypatch, ["--no-bracket-align"]) == [False]
    off = PhotosConfig(develop_hdr_align=False)
    assert _spied_run(tmp_path / "c", monkeypatch, [], off) == [False]
