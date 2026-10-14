"""ML upscale for stills: measured acceptance, resampled fallback, preflight."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline import preflight, quality
from gomoveshift.pipeline.photos import resizer as resizer_mod
from gomoveshift.pipeline.photos.resizer import PhotoResizer


def _scene(w: int, h: int) -> np.ndarray:
    img = np.full((h, w, 3), 90, np.uint8)
    cv2.rectangle(img, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (200, 180, 160), -1)
    cv2.circle(img, (w // 2, h // 2), h // 6, (40, 60, 80), -1)
    return img


def _resampled(small: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


class _Sess:
    def __init__(self, fn):
        self.fn = fn

    def run(self, _names, feeds):
        (arr,) = feeds.values()
        return [self.fn(arr)]


def _factory(fn):
    return lambda: _Sess(fn)


def _model_returning(image_fn):
    """A fake session whose output is image_fn(target size) in NCHW float 0..1."""

    def fn(arr):
        out = image_fn()
        return (out.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    return fn


TARGET = (128, 96)


def _setup():
    small = _scene(64, 48)
    res = _resampled(small, *TARGET)
    return small, res


def test_ml_upscale_accepted_when_better(caplog):
    small, res = _setup()
    sharper = np.clip(
        res.astype(np.int16) + 0.3 * (res.astype(np.int16) - cv2.GaussianBlur(res, (0, 0), 1.2)),
        0,
        255,
    ).astype(np.uint8)
    # sanity: the fixture really is sharper and not ringing
    assert quality.sharpness_score(sharper) >= quality.sharpness_score(res)
    out, rep = quality.ml_upscale_still(
        small, *TARGET, resampled=res, session_factory=_factory(_model_returning(lambda: sharper))
    )
    if not rep.used:  # fixture guard: report names why
        pytest.fail(rep.reason)
    assert np.array_equal(out, sharper)
    assert rep.sharpness_ml >= rep.sharpness_resampled
    assert rep.ringing_ml <= quality.RINGING_MAX
    assert rep.ssim_to_resampled >= quality.ML_UPSCALE_SSIM_FLOOR


def test_ml_upscale_falls_back_when_worse(caplog):
    small, res = _setup()
    soft = cv2.GaussianBlur(res, (0, 0), 3.0)
    with caplog.at_level(logging.WARNING):
        out, rep = quality.ml_upscale_still(
            small, *TARGET, resampled=res, session_factory=_factory(_model_returning(lambda: soft))
        )
    assert rep.used is False
    assert "sharpness" in rep.reason
    assert np.array_equal(out, res)
    assert any("sharpness" in r.getMessage() for r in caplog.records)


def test_ml_upscale_falls_back_on_ringing_or_dissimilar():
    small, res = _setup()
    wrong = 255 - res  # sharp but nothing like the resampled frame
    out, rep = quality.ml_upscale_still(
        small, *TARGET, resampled=res, session_factory=_factory(_model_returning(lambda: wrong))
    )
    assert rep.used is False and np.array_equal(out, res)
    assert "ssim" in rep.reason or "ringing" in rep.reason


def test_ml_upscale_falls_back_on_model_exception():
    small, res = _setup()

    def boom():
        raise RuntimeError("bad shape")

    out, rep = quality.ml_upscale_still(small, *TARGET, resampled=res, session_factory=boom)
    assert rep.used is False and "bad shape" in rep.reason
    assert np.array_equal(out, res)


def test_ml_upscale_refused_without_weights(tmp_path, monkeypatch):
    small, res = _setup()
    monkeypatch.setenv(quality.ML_UPSCALE_WEIGHTS_ENV, str(tmp_path / "absent.onnx"))
    out, rep = quality.ml_upscale_still(small, *TARGET, resampled=res)
    assert rep.used is False
    assert "weights" in rep.reason
    assert np.array_equal(out, res)


def test_weights_hash_mismatch_refused(tmp_path):
    f = tmp_path / "w.onnx"
    f.write_bytes(b"not the model")
    with pytest.raises(quality.MlUpscaleRefused, match="sha256"):
        quality.load_ml_upscaler(f, providers=["CUDAExecutionProvider"], onnxruntime=object())


def test_preflight_ml_upscale_off_probes_nothing():
    def explode(*a, **k):
        raise AssertionError("probed")

    for policy in ("off", "resample"):
        res = preflight.check_ml_upscale(policy, weights_path=explode, providers=explode)
        assert res.ok and not res.checked


def test_preflight_ml_upscale_reports_missing_weights_and_cuda(tmp_path):
    res = preflight.check_ml_upscale(
        "ml", weights_path=tmp_path / "x.onnx", providers=["CPUExecutionProvider"]
    )
    assert not res.ok and res.checked
    assert "weights" in res.reason and "CUDA" in res.reason


def _jpeg(path: Path, w: int, h: int) -> Path:
    Image.fromarray(_scene(w, h)).save(path, "JPEG", quality=95)
    return path


def _resizer(tmp_path, **kw):
    return PhotoResizer(target_width=200, quality=90, output_base=tmp_path / "out", **kw)


@pytest.fixture
def _no_blur(monkeypatch):
    from gomoveshift.pipeline.photos import face_filter

    monkeypatch.setattr(face_filter, "blur_faces_for_publish", lambda f, n: (f, False))
    monkeypatch.setattr(resizer_mod, "_write_privacy_attestation", lambda *a, **k: None)


def _entries(out_dir):
    import json

    return json.loads((out_dir / resizer_mod.PHOTO_PROVENANCE_NAME).read_text())["photos"]


def test_still_export_record_names_scaler(tmp_path, monkeypatch, _no_blur):
    src = _jpeg(tmp_path / "small.jpg", 100, 75)
    calls = []

    def fake(image, w, h, *, resampled, session_factory=None):
        calls.append((w, h))
        rep = quality.MlUpscaleReport(True, "accepted", 2.0, 1.0, 0.0, 0.99)
        return resampled, rep

    monkeypatch.setattr(resizer_mod.quality, "ml_upscale_still", fake)
    r = _resizer(tmp_path, upscale="ml")
    r.resize_batch([str(src)], "2026-10-05")
    rec = _entries(r.output_base / "2026-10-05")[0]["upscale"]
    assert calls == [(200, 150)]
    assert rec["scaler"] == "ml" and rec["upscaled_from"] == [100, 75]
    assert rec["sharpness_ml"] == 2.0


def test_still_export_record_names_resample_on_fallback(tmp_path, monkeypatch, _no_blur):
    src = _jpeg(tmp_path / "small.jpg", 100, 75)

    def fake(image, w, h, *, resampled, session_factory=None):
        return resampled, quality.MlUpscaleReport(False, "sharpness lower", 1.0, 2.0, 0.0, 0.9)

    monkeypatch.setattr(resizer_mod.quality, "ml_upscale_still", fake)
    r = _resizer(tmp_path, upscale="ml")
    r.resize_batch([str(src)], "2026-10-05")
    rec = _entries(r.output_base / "2026-10-05")[0]["upscale"]
    assert rec["scaler"] == "resample" and rec["reason"] == "sharpness lower"


@pytest.mark.parametrize("policy", ["ml", "off"])
def test_native_size_export_unchanged(tmp_path, monkeypatch, _no_blur, policy):
    src = _jpeg(tmp_path / "big.jpg", 300, 225)

    def never(*a, **k):
        raise AssertionError("ml called on a non-upscale")

    monkeypatch.setattr(resizer_mod.quality, "ml_upscale_still", never)
    r = _resizer(tmp_path, upscale=policy)
    out = r.resize_batch([str(src)], "2026-10-05")
    base = _resizer(tmp_path / "b")
    ref = base.resize_batch([str(src)], "2026-10-05")
    assert out[0].read_bytes() == ref[0].read_bytes()
    assert "upscale" not in _entries(r.output_base / "2026-10-05")[0]


# ── CUDA preload + tiled inference ──


class _X4Sess:
    """Deterministic x4 nearest-neighbour model that records every input shape."""

    def __init__(self):
        self.shapes = []

    def run(self, _names, feeds):
        (arr,) = feeds.values()
        self.shapes.append(arr.shape)
        return [np.repeat(np.repeat(arr, 4, axis=2), 4, axis=3)]


@pytest.mark.parametrize("size", [(300, 200), (256, 256), (257, 513), (600, 330), (64, 48)])
def test_tiled_run_matches_untiled_and_bounds_input(size):
    w, h = size
    rng = np.random.default_rng(w * 1000 + h)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    sess = _X4Sess()
    out = quality._run_model(sess, img, w * 4, h * 4)
    expected = np.repeat(np.repeat(img, 4, axis=0), 4, axis=1)
    assert np.array_equal(out, expected)
    limit = quality.ML_UPSCALE_TILE + 2 * quality.ML_UPSCALE_TILE_PAD
    assert sess.shapes and all(s[2] <= limit and s[3] <= limit for s in sess.shapes)
    if max(w, h) > quality.ML_UPSCALE_TILE:
        assert len(sess.shapes) > 1


class _FakeOrt:
    def __init__(self, with_preload=True):
        self.calls = []
        if with_preload:
            self.preload_dlls = lambda: self.calls.append("preload")

    def InferenceSession(self, path, providers=None):  # noqa: N802 -- ort's API name
        self.calls.append("session")

        class _S:
            def get_providers(self):
                return list(providers)

        return _S()


def _pinned_weights(tmp_path, monkeypatch):
    f = tmp_path / "w.onnx"
    f.write_bytes(b"x")
    monkeypatch.setattr(
        quality, "ML_UPSCALE_WEIGHTS_SHA256", quality.hashlib.sha256(b"x").hexdigest()
    )
    return f


def test_load_preloads_cuda_dlls_before_session(tmp_path, monkeypatch):
    ort = _FakeOrt()
    quality.load_ml_upscaler(_pinned_weights(tmp_path, monkeypatch), onnxruntime=ort)
    assert ort.calls == ["preload", "session"]


def test_load_tolerates_onnxruntime_without_preload(tmp_path, monkeypatch):
    ort = _FakeOrt(with_preload=False)
    quality.load_ml_upscaler(_pinned_weights(tmp_path, monkeypatch), onnxruntime=ort)
    assert ort.calls == ["session"]
