"""Local CLIP aesthetic scorer: preprocessing, head, pinned-weight fetch, scorer."""

import hashlib
import io
import os
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.photos import aesthetic as ae

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


# --- hand-built torch-format zip (no torch needed) -------------------------


def _s(text: str) -> bytes:
    raw = text.encode()
    return b"X" + struct.pack("<I", len(raw)) + raw


def _tuple(*items: bytes) -> bytes:
    return b"(" + b"".join(items) + b"t"


def _int(n: int) -> bytes:
    return b"J" + struct.pack("<i", n)


def _tensor(key: str, shape: tuple[int, ...], stride: tuple[int, ...], n: int) -> bytes:
    persid = _tuple(_s("storage"), b"ctorch\nFloatStorage\n", _s(key), _s("cpu"), _int(n))
    return (
        b"ctorch._utils\n_rebuild_tensor_v2\n"
        + _tuple(
            persid + b"Q",
            _int(0),
            _tuple(*[_int(d) for d in shape]),
            _tuple(*[_int(d) for d in stride]),
            b"\x89",
            b"ccollections\nOrderedDict\n)R",
        )
        + b"R"
    )


def _torch_zip(path: Path, tensors: dict[str, np.ndarray], *, stride_bad=False) -> None:
    pkl = b"\x80\x02ccollections\nOrderedDict\n)R("
    blobs = {}
    for i, (name, arr) in enumerate(tensors.items()):
        arr = np.asarray(arr, dtype="<f4")
        stride = tuple(int(s // 4) for s in arr.strides)
        if stride_bad:
            stride = tuple(s + 1 for s in stride)
        pkl += _s(name) + _tensor(str(i), arr.shape, stride, arr.size)
        blobs[str(i)] = arr.tobytes()
    pkl += b"u."
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", pkl)
        for k, v in blobs.items():
            z.writestr(f"archive/data/{k}", v)


def _tiny_net(rng):
    shapes = [(3, 2), (2, 3), (1, 2)]
    return [(rng.normal(size=s).astype("f4"), rng.normal(size=s[0]).astype("f4")) for s in shapes]


# --- preprocess ------------------------------------------------------------


def test_preprocess_shape_dtype_and_normalisation():
    out = ae.preprocess(np.full((224, 224, 3), 255, dtype=np.uint8))
    assert out.shape == (1, 3, 224, 224) and out.dtype == np.float32
    for c in range(3):
        assert out[0, c, 5, 7] == pytest.approx((1.0 - MEAN[c]) / STD[c], rel=1e-5)


def test_preprocess_centre_crops_non_square():
    img = np.zeros((224, 448, 3), dtype=np.uint8)
    img[:, 112:336] = 255
    out = ae.preprocess(img)
    expect = (1.0 - MEAN) / STD
    assert np.allclose(out[0, :, 0, 0], expect, rtol=1e-5)
    assert np.allclose(out[0, :, 223, 223], expect, rtol=1e-5)


def test_preprocess_resizes_shortest_edge():
    out = ae.preprocess(np.full((448, 896, 3), 255, dtype=np.uint8))
    assert out.shape == (1, 3, 224, 224)


# --- head ------------------------------------------------------------------


def test_head_forward_matches_hand_computed_net():
    w1 = np.array([[1.0, 0.0], [0.0, 2.0]], dtype="f4")
    b1 = np.array([0.5, -0.5], dtype="f4")
    w2 = np.array([[1.0, 1.0]], dtype="f4")
    b2 = np.array([0.25], dtype="f4")
    emb = np.array([3.0, 4.0], dtype="f4")  # unit: 0.6, 0.8
    # layer1: [0.6+0.5, 1.6-0.5] = [1.1, 1.1]; layer2: 2.2 + 0.25
    assert ae.head_forward(emb, [(w1, b1), (w2, b2)]) == pytest.approx(2.45, rel=1e-5)


def test_head_forward_l2_normalises_embedding():
    layers = _tiny_net(np.random.default_rng(1))
    emb = np.random.default_rng(2).normal(size=2).astype("f4")
    assert ae.head_forward(emb * 17.0, layers) == pytest.approx(
        ae.head_forward(emb, layers), rel=1e-4
    )


def test_head_forward_zero_embedding_is_finite():
    layers = _tiny_net(np.random.default_rng(1))
    assert np.isfinite(ae.head_forward(np.zeros(2, dtype="f4"), layers))


def test_load_head_reads_synthetic_torch_zip(tmp_path):
    rng = np.random.default_rng(3)
    tensors = {}
    for idx, shape in zip((0, 2, 4, 6, 7), [(4, 3), (2, 4), (2, 2), (2, 2), (1, 2)], strict=True):
        tensors[f"layers.{idx}.weight"] = rng.normal(size=shape).astype("f4")
        tensors[f"layers.{idx}.bias"] = rng.normal(size=shape[0]).astype("f4")
    p = tmp_path / "h.pth"
    _torch_zip(p, tensors)
    layers = ae.load_head(p)
    assert len(layers) == 5
    for (w, b), idx in zip(layers, (0, 2, 4, 6, 7), strict=True):
        assert np.array_equal(w, tensors[f"layers.{idx}.weight"]) and w.dtype == np.float32
        assert np.array_equal(b, tensors[f"layers.{idx}.bias"])


def test_load_head_refuses_non_contiguous_tensor(tmp_path):
    p = tmp_path / "h.pth"
    _torch_zip(
        p, {"layers.0.weight": np.ones((2, 2)), "layers.0.bias": np.ones(2)}, stride_bad=True
    )
    with pytest.raises(ae.AestheticModelUnavailable, match="stride"):
        ae.load_head(p)


def test_load_head_missing_key_raises(tmp_path):
    p = tmp_path / "h.pth"
    _torch_zip(p, {"layers.0.weight": np.ones((2, 2)), "layers.0.bias": np.ones(2)})
    with pytest.raises(ae.AestheticModelUnavailable, match="layers"):
        ae.load_head(p)


def test_load_head_without_pickle_member_raises(tmp_path):
    p = tmp_path / "h.pth"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("archive/other", b"x")
    with pytest.raises(ae.AestheticModelUnavailable, match="data.pkl"):
        ae.load_head(p)


def test_unpickler_refuses_os_system(tmp_path):
    p = tmp_path / "evil.pth"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("archive/data.pkl", b"cos\nsystem\n(S'echo pwned'\ntR.")
    with pytest.raises(ae.pickle.UnpicklingError):
        ae.load_head(p)


def test_unpickler_refuses_unknown_persistent_id():
    up = ae._HeadUnpickler(io.BytesIO(b""), {})
    with pytest.raises(ae.pickle.UnpicklingError):
        up.persistent_load(("bogus",))


# --- ensure_models ---------------------------------------------------------


def _specs(monkeypatch, vision: bytes, head: bytes):
    v = ae.ModelSpec(
        "v.onnx", "https://example.invalid/v", len(vision), hashlib.sha256(vision).hexdigest()
    )
    h = ae.ModelSpec(
        "h.pth", "https://example.invalid/h", len(head), hashlib.sha256(head).hexdigest()
    )
    monkeypatch.setattr(ae, "VISION_SPEC", v)
    monkeypatch.setattr(ae, "HEAD_SPEC", h)


def test_model_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMEGRADE_AESTHETIC_DIR", str(tmp_path / "x"))
    assert ae.model_dir() == tmp_path / "x"


def test_model_dir_default_under_data_home(monkeypatch, tmp_path):
    monkeypatch.delenv("FRAMEGRADE_AESTHETIC_DIR", raising=False)
    monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path))
    assert ae.model_dir() == tmp_path / "models" / "aesthetic"


def test_pinned_constants():
    assert (
        ae.VISION_SPEC.sha256 == "9a658de8944a2001719beabf6c5030cfe7b3dc80e85983f69fc648d6b4d7ec0f"
    )
    assert ae.VISION_SPEC.size == 307001500
    assert ae.HEAD_SPEC.sha256 == "21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0"
    assert ae.HEAD_SPEC.size == 3714759
    assert ae.VISION_SPEC.url.startswith("https://huggingface.co/Xenova/clip-vit-large-patch14/")
    assert ae.HEAD_SPEC.url.startswith("https://github.com/christophschuhmann/")


def test_ensure_models_present_with_good_hash_does_not_fetch(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMEGRADE_AESTHETIC_DIR", str(tmp_path))
    _specs(monkeypatch, b"vision", b"head")
    (tmp_path / "v.onnx").write_bytes(b"vision")
    (tmp_path / "h.pth").write_bytes(b"head")

    def boom(url, dest):
        raise AssertionError("fetched")

    assert ae.ensure_models(allow_download=True, fetch=boom) == (
        tmp_path / "v.onnx",
        tmp_path / "h.pth",
    )


def test_ensure_models_missing_without_download_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMEGRADE_AESTHETIC_DIR", str(tmp_path))
    _specs(monkeypatch, b"vision", b"head")
    with pytest.raises(ae.AestheticModelUnavailable, match="fetch_models"):
        ae.ensure_models()


def test_ensure_models_wrong_hash_without_download_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMEGRADE_AESTHETIC_DIR", str(tmp_path))
    _specs(monkeypatch, b"vision", b"head")
    (tmp_path / "v.onnx").write_bytes(b"tampered")
    with pytest.raises(ae.AestheticModelUnavailable):
        ae.ensure_models()


def test_ensure_models_download_places_verified_files_atomically(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMEGRADE_AESTHETIC_DIR", str(tmp_path / "new"))
    _specs(monkeypatch, b"vision", b"head")
    payload = {"https://example.invalid/v": b"vision", "https://example.invalid/h": b"head"}
    seen = []

    def fetch(url, dest):
        seen.append(Path(dest).name)
        Path(dest).write_bytes(payload[url])

    v, h = ae.ensure_models(allow_download=True, fetch=fetch)
    assert seen == ["v.onnx.part", "h.pth.part"]
    assert v.read_bytes() == b"vision" and h.read_bytes() == b"head"
    assert sorted(p.name for p in (tmp_path / "new").iterdir()) == ["h.pth", "v.onnx"]


def test_ensure_models_bad_hash_removes_part_and_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMEGRADE_AESTHETIC_DIR", str(tmp_path))
    _specs(monkeypatch, b"vision", b"head")

    # six bytes = the right size for "vision", wrong content
    with pytest.raises(ae.AestheticModelUnavailable, match="sha256"):
        ae.ensure_models(allow_download=True, fetch=lambda u, d: Path(d).write_bytes(b"XXXXXX"))
    assert list(tmp_path.iterdir()) == []


def test_ensure_models_wrong_size_removes_part(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMEGRADE_AESTHETIC_DIR", str(tmp_path))
    _specs(monkeypatch, b"vision", b"head")
    with pytest.raises(ae.AestheticModelUnavailable, match="size"):
        ae.ensure_models(allow_download=True, fetch=lambda u, d: Path(d).write_bytes(b"x"))
    assert list(tmp_path.iterdir()) == []


def test_fetch_models_allows_download(monkeypatch):
    calls = {}
    monkeypatch.setattr(ae, "ensure_models", lambda **kw: calls.update(kw) or ("v", "h"))
    assert ae.fetch_models() == ("v", "h")
    assert calls == {"allow_download": True}


def test_default_fetch_streams_to_dest(monkeypatch, tmp_path):
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout):
        seen["url"], seen["timeout"] = url, timeout
        return Resp(b"abc" * 10)

    monkeypatch.setattr(ae.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "d.part"
    ae._default_fetch("https://example.invalid/x", dest)
    assert dest.read_bytes() == b"abc" * 10
    assert seen["timeout"] > 0


# --- scorer ----------------------------------------------------------------


class _Meta:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    def __init__(self, embedding, out_names=("image_embeds",)):
        self.embedding = embedding
        self.out_names = out_names
        self.feeds = None

    def get_inputs(self):
        return [_Meta("pixel_values")]

    def get_outputs(self):
        return [_Meta(n) for n in self.out_names]

    def run(self, names, feeds):
        self.feeds = (names, feeds)
        return [np.asarray(self.embedding, dtype="f4").reshape(1, -1)] * len(self.out_names)


def _img():
    return np.full((300, 400, 3), 128, dtype=np.uint8)


def test_scorer_returns_head_of_fake_embedding():
    layers = _tiny_net(np.random.default_rng(5))
    emb = np.array([0.3, -0.7], dtype="f4")
    sess = _FakeSession(emb)
    scorer = ae.AestheticScorer(session=sess, head=layers)
    assert scorer.score(_img()) == pytest.approx(ae.head_forward(emb, layers), rel=1e-5)
    names, feeds = sess.feeds
    assert names == ["image_embeds"] and feeds["pixel_values"].shape == (1, 3, 224, 224)


def test_scorer_uses_first_output_when_preferred_name_absent():
    layers = _tiny_net(np.random.default_rng(5))
    sess = _FakeSession(np.array([1.0, 2.0]), out_names=("pooled",))
    ae.AestheticScorer(session=sess, head=layers).score(_img())
    assert sess.feeds[0] == ["pooled"]


def test_score_unit_clamps(monkeypatch):
    scorer = ae.AestheticScorer(session=_FakeSession([1.0, 0.0]), head=[])
    for raw, want in ((5.5, 0.5), (0.2, 0.0), (12.0, 1.0), (10.0, 1.0), (1.0, 0.0)):
        monkeypatch.setattr(scorer, "score", lambda rgb, raw=raw: raw)
        assert scorer.score_unit(_img()) == pytest.approx(want)


def _patch_session(monkeypatch, tmp_path):
    import onnxruntime

    made = {}

    class Opts:
        pass

    def fake_session(path, sess_options=None, providers=None):
        made.update(path=path, opts=sess_options, providers=providers)
        return _FakeSession([1.0, 0.0])

    monkeypatch.setattr(onnxruntime, "InferenceSession", fake_session)
    monkeypatch.setattr(onnxruntime, "SessionOptions", Opts)
    monkeypatch.setattr(ae, "ensure_models", lambda **kw: (tmp_path / "v.onnx", tmp_path / "h.pth"))
    monkeypatch.setattr(ae, "load_head", lambda p: [])
    return made


def test_scorer_builds_cpu_session_with_thread_count(monkeypatch, tmp_path):
    made = _patch_session(monkeypatch, tmp_path)
    ae.AestheticScorer(threads=3)
    assert made["providers"] == ["CPUExecutionProvider"]
    assert made["opts"].intra_op_num_threads == 3 and made["opts"].inter_op_num_threads == 1
    assert made["path"] == str(tmp_path / "v.onnx")


def test_scorer_use_gpu_prefers_cuda_with_cpu_fallback(monkeypatch, tmp_path):
    made = _patch_session(monkeypatch, tmp_path)
    ae.AestheticScorer(use_gpu=True)
    assert made["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_module_import_does_not_import_onnxruntime():
    import subprocess
    import sys

    code = "import sys; import gomoveshift.pipeline.photos.aesthetic; print('onnxruntime' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    assert out.stdout.strip() == "False"
