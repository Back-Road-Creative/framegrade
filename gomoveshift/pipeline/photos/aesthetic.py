"""Local aesthetic scorer: CLIP ViT-L/14 image encoder plus the LAION head.

Judges "most striking" without a network call or the GPU: the int8 ONNX image
encoder runs on CPU, its 768-d embedding is L2-normalised and fed through the
LAION improved-aesthetic-predictor MLP (about 1-10). Both files are pinned by
size and sha256 and fetched only by an explicit ``fetch_models()``.

torch is not a core dependency, so the head's ``.pth`` (a torch zip) is read
with a restricted unpickler that admits only the names a plain state dict uses.
"""

from __future__ import annotations

import collections
import hashlib
import io
import os
import pickle
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gomoveshift import paths

INPUT_SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
PREFERRED_OUTPUT = "image_embeds"
FETCH_TIMEOUT_S = 60
_HEAD_LAYER_IDS = (0, 2, 4, 6, 7)


class AestheticModelUnavailable(RuntimeError):  # noqa: N818 - name fixed by the interface
    """The pinned weights are missing, corrupt or unreadable."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    size: int
    sha256: str


VISION_SPEC = ModelSpec(
    "vision_model_quantized.onnx",
    "https://huggingface.co/Xenova/clip-vit-large-patch14/resolve/main/onnx/vision_model_quantized.onnx",
    307001500,
    "9a658de8944a2001719beabf6c5030cfe7b3dc80e85983f69fc648d6b4d7ec0f",  # pragma: allowlist secret (public model checksum, not a credential)
)
HEAD_SPEC = ModelSpec(
    "sac+logos+ava1-l14-linearMSE.pth",
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth",
    3714759,
    "21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0",  # pragma: allowlist secret (public model checksum, not a credential)
)


# --- weights ---------------------------------------------------------------


def model_dir() -> Path:
    override = os.environ.get("FRAMEGRADE_AESTHETIC_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return paths.gms_data_home() / "models" / "aesthetic"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_S) as resp, open(dest, "wb") as out:
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            out.write(chunk)


def _ensure_one(spec: ModelSpec, directory: Path, allow_download: bool, fetch) -> Path:
    target = directory / spec.name
    if target.is_file() and _sha256(target) == spec.sha256:
        return target
    if not allow_download:
        raise AestheticModelUnavailable(
            f"{spec.name} is missing or fails its sha256 check in {directory}; fetch the pinned "
            "weights with: python -c 'from gomoveshift.pipeline.photos.aesthetic import "
            "fetch_models; fetch_models()'"
        )
    directory.mkdir(parents=True, exist_ok=True)
    part = directory / (spec.name + ".part")
    try:
        fetch(spec.url, part)
        if part.stat().st_size != spec.size:
            raise AestheticModelUnavailable(
                f"{spec.name}: downloaded size {part.stat().st_size} != pinned {spec.size}"
            )
        if _sha256(part) != spec.sha256:
            raise AestheticModelUnavailable(f"{spec.name}: downloaded sha256 differs from the pin")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, target)
    return target


def ensure_models(*, allow_download: bool = False, fetch=None) -> tuple[Path, Path]:
    """Return ``(vision_onnx, head_pth)``, downloading only when allowed."""
    fetch = fetch or _default_fetch
    directory = model_dir()
    vision = _ensure_one(VISION_SPEC, directory, allow_download, fetch)
    head = _ensure_one(HEAD_SPEC, directory, allow_download, fetch)
    return vision, head


def fetch_models() -> tuple[Path, Path]:
    return ensure_models(allow_download=True)


# --- head (torch zip, no torch) -------------------------------------------


class _FloatStorage:
    """Stand-in for ``torch.FloatStorage``; never instantiated."""


class _HeadUnpickler(pickle.Unpickler):
    def __init__(self, file, storages: dict[str, np.ndarray]):
        super().__init__(file)
        self._storages = storages

    def find_class(self, module: str, name: str):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return self._rebuild
        if (module, name) == ("torch", "FloatStorage"):
            return _FloatStorage
        raise pickle.UnpicklingError(f"refused global {module}.{name}")

    def persistent_load(self, pid):
        if not (
            isinstance(pid, tuple)
            and len(pid) >= 3
            and pid[0] == "storage"
            and pid[1] is _FloatStorage
        ):
            raise pickle.UnpicklingError(f"unsupported persistent id {pid!r}")
        return self._storages[str(pid[2])]

    @staticmethod
    def _rebuild(storage, offset, size, stride, *_rest):
        size, stride = tuple(size), tuple(stride)
        expected = tuple(int(np.prod(size[i + 1 :], dtype=np.int64)) for i in range(len(size)))
        if stride != expected:
            raise AestheticModelUnavailable(
                f"non-contiguous tensor stride {stride} for size {size}"
            )
        count = int(np.prod(size, dtype=np.int64))
        return storage[offset : offset + count].reshape(size)


def load_head(pth_path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read the LAION head ``layers.{0,2,4,6,7}`` as ``(weight, bias)`` float32 pairs."""
    with zipfile.ZipFile(pth_path) as zf:
        pkl_names = [n for n in zf.namelist() if n.endswith("data.pkl")]
        if not pkl_names:
            raise AestheticModelUnavailable(f"{pth_path}: no data.pkl member")
        prefix = pkl_names[0][: -len("data.pkl")]
        storages = {
            n[len(prefix) + len("data/") :]: np.frombuffer(zf.read(n), dtype="<f4")
            for n in zf.namelist()
            if n.startswith(prefix + "data/") and not n.endswith("/")
        }
        state = _HeadUnpickler(io.BytesIO(zf.read(pkl_names[0])), storages).load()
    try:
        return [
            (
                np.asarray(state[f"layers.{i}.weight"], dtype=np.float32),
                np.asarray(state[f"layers.{i}.bias"], dtype=np.float32),
            )
            for i in _HEAD_LAYER_IDS
        ]
    except KeyError as exc:
        raise AestheticModelUnavailable(f"{pth_path}: missing head key layers.* ({exc})") from exc


def head_forward(embedding: np.ndarray, layers) -> float:
    """L2-normalise ``embedding`` and run the activation-free MLP."""
    x = np.asarray(embedding, dtype=np.float32).reshape(-1)
    x = x / max(float(np.linalg.norm(x)), 1e-12)
    for weight, bias in layers:
        x = weight @ x + bias
    return float(x.reshape(-1)[0])


# --- image -> embedding ----------------------------------------------------


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> (1,3,224,224) float32: bicubic shortest-edge resize, centre crop, CLIP norm."""
    from PIL import Image

    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), "RGB")
    w, h = img.size
    scale = INPUT_SIZE / min(w, h)
    nw, nh = max(INPUT_SIZE, round(w * scale)), max(INPUT_SIZE, round(h * scale))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - INPUT_SIZE) // 2, (nh - INPUT_SIZE) // 2
    img = img.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)


class AestheticScorer:
    def __init__(self, *, session: Any = None, head=None, use_gpu: bool = False, threads: int = 2):
        if session is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            providers = ["CPUExecutionProvider"]
            if use_gpu:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            session = ort.InferenceSession(
                str(ensure_models()[0]), sess_options=opts, providers=providers
            )
        self._session = session
        self._head = load_head(ensure_models()[1]) if head is None else head
        self._input = session.get_inputs()[0].name
        outputs = [o.name for o in session.get_outputs()]
        self._output = PREFERRED_OUTPUT if PREFERRED_OUTPUT in outputs else outputs[0]

    def score(self, rgb) -> float:
        out = self._session.run([self._output], {self._input: preprocess(rgb)})
        return head_forward(np.asarray(out[0]).reshape(-1), self._head)

    def score_unit(self, rgb) -> float:
        return min(1.0, max(0.0, (self.score(rgb) - 1.0) / 9.0))
