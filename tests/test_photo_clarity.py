"""Phase-3 spatial stage: midtone-weighted clarity (broad local contrast).

Clarity is the wide "pop" pass — a LARGE-radius luma unsharp fed back as a
per-pixel ratio gain, weighted to the midtones so highlights and shadows are
protected. It rides the same reusable ``_apply_gain_mask`` seam the vignette
introduced, and runs BEFORE the fine ``_sharpen`` detail pass (broad contrast
first, fine detail after). The hard invariant is that amount 0 is a
byte-identical no-op, so the default develop is exactly what it was before
clarity existed.
"""

import cv2
import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.photos import develop as develop_mod


def _write_photo(path, size=(80, 100)):
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _local_contrast(gray, radius=8.0):
    """Std of the high-pass (a − blur): how much local structure a frame carries."""
    a = np.asarray(gray, dtype=np.float64)
    return float(np.std(a - cv2.GaussianBlur(a, (0, 0), radius)))


def _textured(value=128, amp=12, size=(200, 260)):
    """A flat-toned gray frame at ``value`` carrying a fine sinusoidal texture (± amp)."""
    h, w = size
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    tex = amp * np.sin(xx / 4.0) * np.cos(yy / 4.0)
    frame = np.clip(value + tex, 0, 255).astype(np.uint8)
    return np.repeat(frame[:, :, np.newaxis], 3, axis=2)


# ── the clarity stage ────────────────────────────────────────────────


def test_clarity_zero_amount_is_byte_identical():
    """The load-bearing no-op: amount 0 returns the exact input bytes."""
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    out = develop_mod._clarity(rgb, 0.0)
    assert out.dtype == np.uint8
    assert np.array_equal(out, rgb)


def test_clarity_raises_midtone_local_contrast():
    rgb = _textured(value=128)
    before = _local_contrast(develop_mod._luma(rgb))
    out = develop_mod._clarity(rgb, 0.8)
    after = _local_contrast(develop_mod._luma(out))
    assert after > before * 1.05, "clarity must lift local contrast on a midtone patch"


def test_clarity_preserves_global_mean():
    """Local contrast, not exposure: the frame's overall brightness barely moves."""
    rgb = _textured(value=128)
    out = develop_mod._clarity(rgb, 0.8)
    assert float(np.mean(out)) == pytest.approx(float(np.mean(rgb)), abs=3.0)


def test_clarity_protects_highlights_and_shadows():
    """Near-clipped pixels move far less than midtones for the same texture."""

    def moved(value):
        rgb = _textured(value=value)
        out = develop_mod._clarity(rgb, 1.0)
        return float(np.mean(np.abs(out.astype(np.int16) - rgb.astype(np.int16))))

    mid = moved(128)
    highlight = moved(250)
    shadow = moved(5)
    assert mid > highlight * 2.0, "highlights must be protected"
    assert mid > shadow * 2.0, "shadows must be protected"


def test_clarity_negative_amount_softens_midtone_local_contrast():
    """The other direction: a negative amount REDUCES midtone local contrast."""
    rgb = _textured(value=128)
    before = _local_contrast(develop_mod._luma(rgb))
    after = _local_contrast(develop_mod._luma(develop_mod._clarity(rgb, -0.8)))
    assert after < before * 0.95, "negative clarity must soften local contrast"


def test_clarity_negative_and_positive_are_different_frames():
    """Same magnitude, opposite sign: two distinct renders, neither raising."""
    rgb = _textured(value=128)
    soft = develop_mod._clarity(rgb, -0.6)
    hard = develop_mod._clarity(rgb, 0.6)
    assert soft.dtype == hard.dtype == np.uint8
    assert not np.array_equal(soft, hard)
    assert not np.array_equal(soft, rgb)


def test_negative_clarity_is_a_valid_look_knob():
    """The config range carries the sign too, so a softening look loads."""
    from gomoveshift.config import PhotosConfig

    assert PhotosConfig(looks={"portrait": {"clarity": -0.4}}).looks["portrait"]["clarity"] == -0.4
    with pytest.raises(ValueError):
        PhotosConfig(looks={"portrait": {"clarity": -1.5}})


def test_clarity_radius_participates_in_the_algo_version(monkeypatch):
    """The radius knob is an ALL_CAPS tunable, so a retune self-invalidates the cache."""
    baseline = develop_mod._develop_algo_version()
    monkeypatch.setattr(develop_mod, "CLARITY_RADIUS", develop_mod.CLARITY_RADIUS + 8.0)
    assert develop_mod._develop_algo_version() != baseline


# ── end to end through the develop stage ─────────────────────────────


def test_develop_variants_default_is_byte_identical_to_no_clarity(tmp_path):
    photo = _write_photo(tmp_path / "IMG_A.jpg")
    plain = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    off = develop_mod.DevelopOptions(clarity=0.0)
    zero = dict(develop_mod.develop_variants(photo, ["natural"], options=off))["natural"].rgb
    assert np.array_equal(plain, zero)


def test_develop_variants_clarity_raises_local_contrast(tmp_path):
    photo = _write_photo(tmp_path / "IMG_B.jpg")
    base = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    on = develop_mod.DevelopOptions(clarity=0.8)
    clr = dict(develop_mod.develop_variants(photo, ["natural"], options=on))["natural"].rgb
    assert not np.array_equal(base, clr)
    assert _local_contrast(develop_mod._luma(clr)) > _local_contrast(develop_mod._luma(base))


def test_develop_single_photo_clarity_raises_local_contrast(tmp_path):
    photo = _write_photo(tmp_path / "IMG_C.jpg")
    base = develop_mod.develop(photo).rgb
    clr = develop_mod.develop(photo, clarity=0.8).rgb
    assert _local_contrast(develop_mod._luma(clr)) > _local_contrast(develop_mod._luma(base))
