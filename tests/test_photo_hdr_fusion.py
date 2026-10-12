"""``photos develop --hdr``: one exposure bracket fuses to ONE frame, before the grade.

CI has no card, so the bracket is synthesised from one radiance map — a blown-out window over a
dim interior, ~14 stops, far more than a single 8-bit frame holds — rendered at five exposures
three stops apart with the EXIF ``detect_brackets`` groups on. The real-card test is optional and enabled with ``GMS_TEST_RAW_CARD_DIR``."""

import inspect
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import bracket as bracket_mod
from gomoveshift.pipeline.photos import develop as dev

#: A directory holding a real six-frame NEF bracket; unset -> the real-card test is skipped.
CARD = Path(os.environ.get("GMS_TEST_RAW_CARD_DIR") or "/nonexistent-raw-card")
#: (clock time, shutter denominator): 8x steps in ONE direction, ISO held, 12 s apart.
SEQUENCE = [(f"12:00:{8 + 12 * i:02d}", 4000 / 8**i) for i in range(5)]
WINDOW = (slice(6, 20), slice(40, 60))  # blown out in every frame but the darkest
INTERIOR = (slice(26, 34), slice(4, 24))  # crushed in every frame but the brightest
#: Every ``bracket`` threshold that decides which frames merge, and so which pixels.
THRESHOLDS = ("MAX_GAP_S", "MAX_SPAN_S", "MIN_STEP_RATIO", "MIN_BRACKET_FRAMES")
STOCK = Config(photos=PhotosConfig())  # every CLI test runs on the stock, all-off config


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)


def _render(exposure_s):  # the scene at one exposure, gamma-encoded like a camera JPEG
    texture = 0.6 + 0.8 * np.random.default_rng(7).random((48, 64))
    scene = 0.004 * texture
    scene[WINDOW] = 12.0 * texture[WINDOW]
    scene[INTERIOR] = 0.05 * texture[INTERIOR]
    disp = np.clip(scene * exposure_s * 240.0, 0.0, 1.0) ** (1 / 2.2)
    return np.rint(np.stack([disp, disp * 0.97, disp * 0.94], 2) * 255).astype(np.uint8)


def _bracket(tmp_path):
    paths = []
    for i, (at, denom) in enumerate(SEQUENCE, 1):
        exif = Image.Exif()
        exif[0x8769] = {0x9003: f"2000:01:01 {at}", 0x8827: 100, 0x829A: 1 / denom}
        paths.append(tmp_path / f"DSC_{i:04d}.jpg")
        Image.fromarray(_render(1 / denom), "RGB").save(paths[-1], "JPEG", quality=95, exif=exif)
    return paths


def _levels(rgb, box):  # distinct codes in one region: how much detail survived there
    return len(np.unique(rgb[box]))


def _clipped(rgb):
    return float(np.mean(rgb.max(axis=2) >= 254.0 / 255.0 * dev._peak(rgb)))


def _run(argv):
    return photos_handler.cmd_photos(_build_parser().parse_args(argv), None)


def test_fusing_recovers_both_ends_keeps_the_depth_and_refuses_a_mismatch(tmp_path, monkeypatch):
    """Range no single frame holds: the brightest blows the window down to 3 levels and the
    darkest crushes the interior to 8, so the fused frame carries both while clipping no more
    than the best input. That pins the range map too (this merge reaches 1.48, and clamping
    collapses the window to ONE level) and the depth plumbing, since a true-raw bracket must
    not drop a tiff16 master to 8 bits. Frames of unequal size — the card mixes a 3712x5568
    preview fallback with a 3728x5600 DNG decode — are refused, never resampled."""
    paths = _bracket(tmp_path)
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    fused = dev.fuse_bracket(paths)
    assert _levels(fused.rgb, WINDOW) > _levels(frames[-1], WINDOW)
    assert _levels(fused.rgb, INTERIOR) > _levels(frames[0], INTERIOR)
    assert _clipped(fused.rgb) <= min(_clipped(f) for f in frames)
    Image.fromarray(_render(1 / 500)[:40], "RGB").save(tmp_path / "odd.jpg", "JPEG")
    with pytest.raises(ValueError, match="different sizes"):
        dev.fuse_bracket(paths + [tmp_path / "odd.jpg"])
    real = dev.decode  # what a true-raw decode hands over: 16-bit pixels, sensor depth
    deep = lambda p: replace(real(p), rgb=real(p).rgb.astype(np.uint16) * 257, bit_depth=14)  # noqa: E731
    monkeypatch.setattr(dev, "decode", deep)
    wide = dev.fuse_bracket(paths)
    assert (wide.rgb.dtype, wide.bit_depth) == (np.uint16, 14)
    assert wide.rgb.max() > 255, "a 16-bit fuse must fill the range, not sit in the low byte"


def test_the_merge_is_pinned_single_threaded_so_two_runs_match_byte_for_byte(tmp_path, monkeypatch):
    """Measured on a 20 MP bracket: under OpenCV's 12-thread default two
    ``MergeMertens`` runs differ on 17% of pixels (max 4.2e-7) — the parallel stripes move the
    float rounding. Pinned they are bit-identical, 7.7 s against 4.1 s on three 20 MP frames."""
    paths = _bracket(tmp_path)
    assert np.array_equal(dev.fuse_bracket(paths).rgb, dev.fuse_bracket(paths).rgb)
    seen, real = [], cv2.createMergeMertens()
    spy = SimpleNamespace(process=lambda f: (seen.append(cv2.getNumThreads()), real.process(f))[1])
    monkeypatch.setattr(cv2, "createMergeMertens", lambda: spy)
    pool = cv2.getNumThreads()
    dev.fuse_bracket(paths)
    assert seen == [1], "the merge must run single-threaded or it is not reproducible"
    assert cv2.getNumThreads() == pool, "the thread pool must be handed back"


def test_the_flag_consumes_a_bracket_and_leaves_the_unflagged_develop_byte_identical(
    tmp_path, monkeypatch
):
    """A bracket's frames are CONSUMED — one export under the first stem plus ``_hdr``, never also
    one by one — while a photo in no bracket develops as it does today. OFF (the default) that same
    bracket's masters are byte-identical to a direct ``develop_variants`` render, one per frame."""
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: STOCK)
    paths = _bracket(tmp_path)
    loose = tmp_path / "IMG_9999.jpg"  # no EXIF at all — never part of a bracket
    Image.fromarray(_render(1 / 500), "RGB").save(loose, "JPEG")
    out, off = tmp_path / "hdr", tmp_path / "off"
    argv = ["photos", "develop", *map(str, [*paths, loose]), "--styles", "natural"]
    assert _run([*argv, "--out", str(out), "--hdr"]) == 0
    assert sorted(p.name for p in out.glob("*.jpg")) == [
        "DSC_0001_hdr_natural.jpg",
        "IMG_9999_natural.jpg",
    ]
    assert _run([*argv, "--out", str(off)]) == 0
    assert len(list(off.glob("*.jpg"))) == len(paths) + 1
    ((_, developed),) = dev.develop_variants(paths[0], ["natural"])
    reference = dev.export_jpeg(developed.rgb, tmp_path / "reference.jpg")
    assert (off / f"{paths[0].stem}_natural.jpg").read_bytes() == reference.read_bytes()


def test_hdr_is_refused_before_any_decode_when_no_bracket_can_exist(tmp_path, monkeypatch, capsys):
    """Upstream gate: fewer photos than ``MIN_BRACKET_FRAMES`` can never form a bracket."""
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: STOCK)
    paths = _bracket(tmp_path)[:2]
    monkeypatch.setattr(dev, "decode", lambda p: pytest.fail(f"decoded {p} behind the gate"))
    gated = tmp_path / "gated"
    assert _run(["photos", "develop", *map(str, paths), "--out", str(gated), "--hdr"]) == 1
    assert "--hdr" in capsys.readouterr().err and not gated.exists()


def test_the_fusion_stage_carries_its_own_cache_key_input(monkeypatch):
    """The contract a later grouping step relies on. The grouping rule and its thresholds decide WHICH frames
    merge, so with fusion wired they decide output pixels — from another module, outside the
    ALL_CAPS sweep of ``develop.py``'s globals. Left there, a retune replays a photo cached from
    a different set of merged frames, with no error anywhere."""
    grouping = (bracket_mod.detect_brackets, bracket_mod._step, bracket_mod._read_frame)
    inputs = dev._develop_algo_inputs()
    for fn in (dev.fuse_bracket, *grouping):
        assert inspect.getsource(fn) in inputs
    for knob in THRESHOLDS:
        before = dev._develop_algo_version()
        monkeypatch.setattr(bracket_mod, knob, getattr(bracket_mod, knob) + 0.5)
        assert dev._develop_algo_version() != before, knob


@pytest.mark.skipif(
    len(sorted(CARD.glob("*.NEF"))) < 6,
    reason="set GMS_TEST_RAW_CARD_DIR to a real NEF bracket directory to run",
)
def test_a_real_hand_stepped_bracket_fuses_without_introducing_clipping():
    """Fuses the 2nd, 4th and 6th raw files of the directory (alternate frames, since a
    neighbour may fall back to the embedded preview, a different size, and so be
    refused). A hand-stepped sweep in even light
    is not a range-limited scene, so no input clips or crushes; fusion buys tone blending
    there, not rail recovery. It must never ADD clipping."""
    frames = sorted(CARD.glob("*.NEF"))
    fused = dev.fuse_bracket([frames[i] for i in (1, 3, 5)])
    assert _clipped(fused.rgb) == 0.0
    assert fused.decoder.endswith("+mertens3")
