"""Receipt-bound adapters for the four private night product contracts.

The existing astro engines remain authoritative. This module supplies a common
typed result and durable aggregate receipt; it does not implement a second stack,
timelapse, trail, or deep-combine algorithm.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from gomoveshift.pipeline._file_lock import lock_exclusive

from ..media_admission import media_admission
from .night_collection import NightProduct
from .night_project import (
    NightStore,
    StorageWaitError,
    _mark_cancelled,
    available_memory_bytes,
    boot_id,
    build_identity,
    digest,
    generated_bytes,
    process_start,
    runtime_identity,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductOutput:
    path: Path
    role: str
    primary: bool = False


@dataclass(frozen=True)
class RenderedProduct:
    outputs: tuple[ProductOutput, ...]
    evidence: dict = field(default_factory=dict)
    rejected: tuple[dict, ...] = ()


class ProductWaitError(RuntimeError):
    """A transient product resource shortage, distinct from invalid inputs."""

    def __init__(self, message: str, *, kind: str):
        super().__init__(message)
        self.kind = kind


@contextmanager
def _worker_heartbeat(
    store: NightStore,
    brand: str,
    project: str,
    worker: dict,
    *,
    interval_s: float = 30.0,
):
    """Persist liveness during one opaque engine call without claiming progress."""
    stopped = threading.Event()
    store.heartbeat(brand, project, worker)

    def beat() -> None:
        while not stopped.wait(interval_s):
            try:
                current = store.heartbeat(brand, project, worker)
            except (OSError, sqlite3.Error):
                logger.warning("Night worker heartbeat could not be persisted", exc_info=True)
                return
            if not current:
                return

    thread = threading.Thread(target=beat, name=f"night-heartbeat-{project[:8]}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=1)


def _raw_megapixels(path: Path) -> float:
    import rawpy

    from .astro_calibrate import AstroCalibrationError

    try:
        with rawpy.imread(str(path)) as raw:
            sizes = raw.sizes
    except rawpy.LibRawError as exc:
        raise AstroCalibrationError(f"{path}: not a readable RAW frame ({exc})") from exc
    return (int(sizes.width) * int(sizes.height)) / 1_000_000.0


def _render_deep(linears: list[Path], output: Path):
    from .astro_combine import deep_combine
    from .astro_combine_output import publish_deep_product

    return publish_deep_product(deep_combine(linears), output)


def render_master_preview(master: Path, output: Path) -> Path:
    """Export a browser JPEG from the real 16-bit master without changing its pixels."""
    import numpy as np
    import tifffile

    from .develop import export_jpeg

    pixels = tifffile.imread(master)
    if pixels.dtype != np.uint16 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("night product preview requires an RGB uint16 master")
    preview = np.rint(pixels.astype(np.float32) / 257.0).astype(np.uint8)
    return export_jpeg(preview, output)


def default_engines() -> dict[str, Callable | int]:
    """Fresh engine table so tests and workers can replace explicit seams."""
    from .astro_timelapse import (
        TIMELAPSE_FPS,
        TIMELAPSE_HEIGHT_DEFAULT,
        preflight_timelapse,
        render_timelapse,
    )
    from .astro_trails import preflight_trail, render_trail
    from .night_render import render_artifact

    return {
        "render_still": render_artifact,
        "raw_megapixels": _raw_megapixels,
        "preflight_timelapse": preflight_timelapse,
        "render_timelapse": render_timelapse,
        "timelapse_fps": TIMELAPSE_FPS,
        "timelapse_height": TIMELAPSE_HEIGHT_DEFAULT,
        "preflight_trail": preflight_trail,
        "render_trail": render_trail,
        "render_deep": _render_deep,
        "render_preview": render_master_preview,
    }


def _paths(row: dict, kind: str) -> list[Path]:
    return [Path(item["path"]) for item in row["inputs"][kind]]


def _rejections(rejected, row: dict) -> tuple[dict, ...]:
    captured = {item["path"]: item.get("captured_at") for item in row["inputs"]["lights"]}
    return tuple(
        {
            "path": str(path),
            "captured_at": captured.get(str(path)),
            "reason": str(reason),
        }
        for path, reason in rejected
    )


def _preflight_wait(exc: RuntimeError) -> ProductWaitError:
    message = str(exc)
    lowered = message.lower()
    kind = "disk" if "disk" in lowered or "space" in lowered else "memory"
    return ProductWaitError(f"Waiting for product resources: {message}", kind=kind)


def _best_stills(row: dict, attempt: Path, engines: dict) -> RenderedProduct:
    outputs: list[ProductOutput] = []
    rejected: list[dict] = []
    sources = _paths(row, "linear")
    for index, source in enumerate(sources):
        destination = attempt / f"best-{index}"
        try:
            engines["render_still"](source, destination, row["recipe"])
            outputs.extend(
                (
                    ProductOutput(destination / "master.tif", "master", True),
                    ProductOutput(destination / "preview.jpg", "preview"),
                    ProductOutput(destination / "receipt.json", "lineage"),
                )
            )
        except Exception as exc:  # preserve other verified stills and name the missing one
            rejected.append({"path": str(source), "reason": str(exc)})
        finally:
            if engines.get("progress") is not None:
                engines["progress"](index + 1, len(sources))
    if not any(output.primary for output in outputs):
        raise ValueError("best_stills produced no validated master")
    return RenderedProduct(
        tuple(outputs),
        {"stacks_requested": len(row["inputs"]["linear"]), "stacks_rendered": len(outputs) // 3},
        tuple(rejected),
    )


def _timelapse(row: dict, attempt: Path, engines: dict) -> RenderedProduct:
    lights, darks = _paths(row, "lights"), _paths(row, "darks")
    megapixels = engines["raw_megapixels"](lights[0])
    try:
        engines["preflight_timelapse"](attempt, megapixels=megapixels, dark_count=len(darks))
    except RuntimeError as exc:
        raise _preflight_wait(exc) from exc
    fps = int(engines["timelapse_fps"])
    output = attempt / "timelapse.mp4"
    kwargs = {"fps": fps, "scale_height": engines["timelapse_height"]}
    if engines.get("progress") is not None:
        kwargs["progress"] = engines["progress"]
    path, used, rejected = engines["render_timelapse"](lights, output, darks, **kwargs)
    missing = _rejections(rejected, row)
    return RenderedProduct(
        (ProductOutput(Path(path), "video", True),),
        {
            "frames_requested": len(lights),
            "frames_used": int(used),
            "fps": fps,
            "duration_s": int(used) / fps,
            "capture_started_at": row["capture"]["started_at"],
            "capture_ended_at": row["capture"]["ended_at"],
            "missing_intervals": list(missing),
        },
        missing,
    )


def _trails(row: dict, attempt: Path, engines: dict) -> RenderedProduct:
    lights, darks = _paths(row, "lights"), _paths(row, "darks")
    megapixels = engines["raw_megapixels"](lights[0])
    try:
        engines["preflight_trail"](attempt, megapixels=megapixels, dark_count=len(darks))
    except RuntimeError as exc:
        raise _preflight_wait(exc) from exc
    kwargs = {"progress": engines["progress"]} if engines.get("progress") is not None else {}
    path, used, rejected = engines["render_trail"](lights, attempt / "trails.tif", darks, **kwargs)
    path = Path(path)
    preview = engines["render_preview"](path, attempt / "preview.jpg")
    missing = _rejections(rejected, row)
    return RenderedProduct(
        (
            ProductOutput(path, "master", True),
            ProductOutput(Path(preview), "preview"),
        ),
        {
            "frames_requested": len(lights),
            "frames_used": int(used),
            "capture_started_at": row["capture"]["started_at"],
            "capture_ended_at": row["capture"]["ended_at"],
            "missing_intervals": list(missing),
        },
        missing,
    )


def _deep_still(row: dict, attempt: Path, engines: dict) -> RenderedProduct:
    paths = engines["render_deep"](_paths(row, "linear"), attempt / "deep.tif")
    if engines.get("progress") is not None:
        engines["progress"](1, 1)
    preview = engines["render_preview"](Path(paths.image), attempt / "preview.jpg")
    return RenderedProduct(
        (
            ProductOutput(Path(paths.image), "master", True),
            ProductOutput(Path(preview), "preview"),
            ProductOutput(Path(paths.coverage), "coverage"),
            ProductOutput(Path(paths.sidecar), "lineage"),
        ),
        {"linear_artifacts": len(row["inputs"]["linear"])},
    )


_DEFAULT_RENDERERS = {
    NightProduct.BEST_STILLS.value: _best_stills,
    NightProduct.TIMELAPSE.value: _timelapse,
    NightProduct.STAR_TRAILS.value: _trails,
    NightProduct.DEEP_STILL.value: _deep_still,
}


def _inside(path: Path, roots: list[Path]) -> Path:
    if path.is_symlink() or not path.exists():
        raise ValueError(f"night product input changed or disappeared: {path}")
    resolved = path.resolve(strict=True)
    if not any(resolved.is_relative_to(root.resolve(strict=True)) for root in roots):
        raise ValueError(f"night product input escaped configured roots: {path}")
    return resolved


def validate_inputs(row: dict) -> None:
    """Recheck the exact lightweight identities captured when the request was made."""
    roots = [Path(path) for path in row["source_roots"]]
    for kind in ("lights", "darks"):
        for saved in row["inputs"][kind]:
            path = _inside(Path(saved["path"]), roots)
            if not path.is_file():
                raise ValueError(f"night product input is no longer a regular file: {path}")
            info = path.stat()
            actual = {
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "device": info.st_dev,
                "inode": info.st_ino,
            }
            if any(saved[name] != value for name, value in actual.items()):
                raise ValueError(f"night product input changed: {path}")
    for saved in row["inputs"]["linear"]:
        path = _inside(Path(saved["path"]), roots)
        manifest = path / "manifest.json"
        if not path.is_dir() or manifest.is_symlink() or not manifest.is_file():
            raise ValueError(f"linear product input changed: {path}")
        if digest(manifest) != saved["manifest_sha256"]:
            raise ValueError(f"linear product input changed: {path}")


def _input_digest(row: dict) -> str:
    encoded = json.dumps(row["inputs"], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _output_spec(output: ProductOutput, attempt: Path, evidence: dict) -> dict:
    path = Path(output.path)
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(attempt.resolve())
    ):
        raise ValueError(f"product output is not a regular file inside its attempt: {path}")
    spec = {
        "path": str(path.relative_to(attempt)),
        "role": output.role,
        "primary": output.primary,
        "size": path.stat().st_size,
        "sha256": digest(path),
    }
    if output.primary and path.suffix.lower() == ".mp4":
        spec["duration_s"] = evidence.get("duration_s")
    return spec


def _media_type(product: NightProduct, specs: list[dict]) -> None:
    primary = [spec for spec in specs if spec["primary"]]
    if not primary:
        raise ValueError(f"{product.value} produced no primary output")
    suffixes = {Path(spec["path"]).suffix.lower() for spec in primary}
    if product is NightProduct.TIMELAPSE and suffixes != {".mp4"}:
        raise ValueError("timelapse requires an MP4 primary output")
    if product is not NightProduct.TIMELAPSE and not suffixes.issubset({".tif", ".tiff"}):
        raise ValueError(f"{product.value} requires TIFF primary output")


# Matches ``publishing.media_fingerprint.DEFAULT_FFPROBE_TIMEOUT_SEC``.
_PROBE_TIMEOUT_SEC = 30


def _validate_media(product: NightProduct, path: Path, spec: dict) -> bool | dict:
    suffix = path.suffix.lower()
    try:
        if suffix == ".mp4":
            from ..ffmpeg_runner import validate_video
            from ..validation.ffprobe_client import probe_video_summary

            result = validate_video(
                path, expected_duration=spec.get("duration_s"), require_audio=False
            )
            if not result.valid or result.duration <= 0:
                return False
            duration_s, width, height, codec = probe_video_summary(
                path, timeout=_PROBE_TIMEOUT_SEC, label="night media probe"
            )
            return {"duration_s": duration_s, "width": width, "height": height, "codec": codec}
        if suffix in {".tif", ".tiff"}:
            import tifffile

            with tifffile.TiffFile(path) as image:
                if len(image.pages) != 1 or len(image.pages[0].shape) not in {2, 3}:
                    return False
                return {
                    "shape": list(image.pages[0].shape),
                    "dtype": str(image.pages[0].dtype),
                }
        if suffix in {".jpg", ".jpeg"}:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
                return {"width": image.width, "height": image.height, "format": image.format}
        if suffix == ".npy":
            import numpy as np

            value = np.load(path, mmap_mode="r", allow_pickle=False)
            return {"shape": list(value.shape), "dtype": str(value.dtype)} if value.size else False
        if suffix == ".json":
            payload = json.loads(path.read_text())
            return {"json_object": True} if isinstance(payload, dict) else False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return False


def _check_media(media_validator: Callable, product: NightProduct, path: Path, spec: dict) -> bool:
    observed = media_validator(product, path, spec)
    if not observed:
        return False
    if isinstance(observed, dict):
        recorded = spec.get("media")
        if recorded is not None and recorded != observed:
            return False
        spec["media"] = observed
    return True


def _atomic_json(path: Path, payload: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def render_product(
    row: dict,
    task: dict,
    attempt: Path,
    *,
    renderers: dict[str, Callable] | None = None,
    engines: dict | None = None,
    media_validator: Callable = _validate_media,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Render one requested product and publish its aggregate receipt last."""
    if row.get("schema") != 2 or row.get("kind") != "night_products":
        raise ValueError("typed night product project required")
    if row.get("build") != build_identity() or row.get("runtime") != runtime_identity():
        raise ValueError("night product build or dependency identity changed")
    validate_inputs(row)
    product = NightProduct(task["product"])
    attempt = Path(attempt)
    if attempt.is_symlink() or (attempt.exists() and any(attempt.iterdir())):
        raise ValueError("product attempt must be a new empty directory")
    attempt.mkdir(parents=True, exist_ok=True)
    engines = default_engines() if engines is None else dict(engines)
    engines["progress"] = progress
    if renderers and product.value in renderers:
        rendered = renderers[product.value](row, attempt)
    else:
        rendered = _DEFAULT_RENDERERS[product.value](row, attempt, engines)
    specs = [_output_spec(output, attempt, rendered.evidence) for output in rendered.outputs]
    _media_type(product, specs)
    for spec in specs:
        if not _check_media(media_validator, product, attempt / spec["path"], spec):
            raise ValueError(f"product output failed media validation: {spec['path']}")
    receipt = {
        "schema": 1,
        "state": "partial" if rendered.rejected else "complete",
        "project_id": row["id"],
        "product": product.value,
        "recipe": row["recipe"],
        "build": row["build"],
        "runtime": row["runtime"],
        "input_digest": _input_digest(row),
        "quality_review_required": True,
        "outputs": specs,
        "evidence": rendered.evidence,
        "rejected": list(rendered.rejected),
    }
    _atomic_json(attempt / "product-receipt.json", receipt)
    validated = validate_product_receipt(row, task, attempt, media_validator=media_validator)
    if validated is None:
        raise ValueError("published product receipt did not validate")
    return validated


def validate_product_receipt(
    row: dict,
    task: dict,
    attempt: Path,
    *,
    media_validator: Callable = _validate_media,
) -> dict | None:
    """Bounded receipt and byte validation for crash adoption and reuse."""
    try:
        attempt = Path(attempt)
        receipt_path = attempt / "product-receipt.json"
        if (
            attempt.is_symlink()
            or receipt_path.is_symlink()
            or receipt_path.stat().st_size > 1024**2
        ):
            return None
        receipt = json.loads(receipt_path.read_text())
        product = NightProduct(task["product"])
        if (
            receipt["schema"] != 1
            or receipt["state"] not in {"complete", "partial"}
            or receipt["project_id"] != row["id"]
            or receipt["product"] != product.value
            or receipt["recipe"] != row["recipe"]
            or receipt["build"] != row["build"]
            or receipt["runtime"] != row["runtime"]
            or receipt["input_digest"] != _input_digest(row)
            or receipt["quality_review_required"] is not True
        ):
            return None
        specs = receipt["outputs"]
        _media_type(product, specs)
        for spec in specs:
            path = attempt / spec["path"]
            if (
                path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(attempt.resolve())
                or path.stat().st_size != spec["size"]
                or digest(path) != spec["sha256"]
                or not _check_media(media_validator, product, path, spec)
            ):
                return None
        return receipt
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _linear_pixels(row: dict) -> list[int]:
    pixels = []
    for item in row["inputs"]["linear"]:
        try:
            shape = json.loads((Path(item["path"]) / "manifest.json").read_text())["payloads"][
                "sky_sum.npy"
            ]["shape"]
            if (
                not isinstance(shape, list)
                or len(shape) != 3
                or shape[2] != 3
                or any(type(value) is not int or value <= 0 for value in shape)
            ):
                raise ValueError
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid linear dimensions: {item['path']}") from exc
        pixels.append(shape[0] * shape[1])
    return pixels


def estimate_product_bytes(row: dict, task: dict, *, engines: dict | None = None) -> int:
    """Conservative additional output reservation; engine preflights still apply."""
    product = NightProduct(task["product"])
    if product is NightProduct.BEST_STILLS:
        return sum(pixels * 12 + 16 * 1024**2 for pixels in _linear_pixels(row))
    if product is NightProduct.DEEP_STILL:
        return max(_linear_pixels(row)) * 16 + 64 * 1024**2
    engines = default_engines() if engines is None else engines
    megapixels = float(engines["raw_megapixels"](_paths(row, "lights")[0]))
    pixels = int(megapixels * 1_000_000)
    if product is NightProduct.TIMELAPSE:
        return pixels * 4 + 512 * 1024**2
    return pixels * 12 + 16 * 1024**2


def product_resource_preflight(row: dict, task: dict) -> None:
    """Cheap linear-product memory gate; RAW engines apply their exact gates."""
    product = NightProduct(task["product"])
    if product in {NightProduct.TIMELAPSE, NightProduct.STAR_TRAILS}:
        return
    pixels = max(_linear_pixels(row))
    available = available_memory_bytes()
    required = pixels * (300 if product is NightProduct.DEEP_STILL else 200) + 2 * 1024**3
    if available < required:
        raise ProductWaitError(
            f"Waiting for memory: {available} available bytes, {required} required.",
            kind="memory",
        )


def _source_paths(row: dict) -> list[Path]:
    return [
        Path(item["path"]) for kind in ("lights", "darks", "linear") for item in row["inputs"][kind]
    ]


def _storage_preflight(row: dict, target: Path, required: int) -> None:
    used = generated_bytes(target, _source_paths(row))
    budget = row.get("storage_budget_bytes")
    if budget is not None and used + required > budget:
        raise StorageWaitError(
            f"Waiting for project storage budget: {used} used + {required} reserved bytes "
            f"exceeds {budget}; preserve files or create a request with a larger budget.",
            kind="budget",
        )
    if required and shutil.disk_usage(target).free < required + 1024**3:
        raise StorageWaitError(
            f"Waiting for output free space: {required} reserved bytes plus 1 GiB headroom required.",
            kind="disk",
        )


def _adopt(task: dict, candidate: Path, receipt: dict) -> None:
    task.update(
        status=receipt["state"],
        output=str(candidate),
        receipt=str(candidate / "product-receipt.json"),
        evidence=receipt.get("evidence", {}),
        rejected=receipt.get("rejected", []),
        error=None,
    )
    task.pop("wait_kind", None)


def process_product_project(
    store: NightStore,
    brand: str,
    project: str,
    *,
    render: Callable = render_product,
    validate: Callable = validate_product_receipt,
    estimate: Callable = estimate_product_bytes,
    resource_check: Callable = product_resource_preflight,
    admission_context: Callable = media_admission,
    prepare_video: Callable | None = None,
    estimate_video: Callable | None = None,
) -> int:
    """One resumable typed-product pass: 0 complete, 1 partial/failed, 2 waiting."""
    row = store.get(brand, project)
    if row.get("schema") != 2 or row.get("kind") != "night_products":
        raise ValueError("typed night product project required")
    if (
        row.get("build") != build_identity()
        or row.get("runtime") != runtime_identity()
        or row.get("interpreter") != sys.executable
    ):
        row.update(status="blocked", reason="Processing build changed; create a new request.")
        store.save(row)
        return 1
    if row.get("cancel_requested"):
        _mark_cancelled(row)
        store.save(row)
        return 1

    with admission_context() as host_admitted, (store.root / "heavy.lock").open("a") as lock:
        try:
            if not host_admitted:
                raise BlockingIOError("Another media worker holds admission")
            lock_exclusive(lock, blocking=False)
        except BlockingIOError:
            current = store.get(brand, project)
            if current["status"] not in {"running", "complete", "paused"}:
                current.update(
                    status="waiting",
                    wait_kind="admission",
                    reason="Another media worker holds admission.",
                )
                store.save(current)
            return 2
        if row.get("cancel_requested"):
            _mark_cancelled(row)
            store.save(row)
            return 1
        if row.get("pause_requested"):
            row["status"] = "paused"
            store.save(row)
            return 2
        row["worker"] = {
            "pid": os.getpid(),
            "started_at": process_start(os.getpid()),
            "boot_id": boot_id(),
        }
        row["status"] = "running"
        row["last_progress_at"] = time.time()
        row.pop("reason", None)
        row.pop("wait_kind", None)
        store.save(row)
        target = store.root / project
        if target.is_symlink():
            row.update(status="blocked", reason="Project output directory must not be a symlink.")
            store.save(row)
            return 1
        target.mkdir(exist_ok=True)

        for index, task in enumerate(row["tasks"]):
            controls = store.get(brand, project)
            if controls.get("cancel_requested"):
                _mark_cancelled(row)
                store.save(row)
                return 1
            if controls.get("pause_requested"):
                row["status"] = "paused"
                store.save(row)
                return 2
            if task.get("status") == "waiting" and task.get("wait_kind") == "budget":
                row.update(status="waiting", wait_kind="budget")
                store.save(row)
                return 2

            candidates = []
            if task.get("output"):
                candidates.append(Path(task["output"]))
            candidates.extend(
                candidate
                for candidate in sorted(target.glob(f"{index}-{task['product']}-*"), reverse=True)
                if candidate not in candidates
            )
            reusable = None
            for candidate in candidates:
                receipt = validate(row, task, candidate)
                if receipt is not None:
                    reusable = (candidate, receipt)
                    break
            if reusable is not None:
                _adopt(task, *reusable)
                row["last_progress_at"] = time.time()
                store.save(row)
                continue

            try:
                resource_check(row, task)
                required = int(estimate(row, task))
                if required < 0:
                    raise ValueError("product storage estimate must not be negative")
                _storage_preflight(row, target, required)
                destination = target / f"{index}-{task['product']}-{uuid.uuid4().hex}"
                task["attempts"] += 1
                task.update(status="running", error=None, last_attempt=str(destination))
                started = time.time()
                row.update(
                    current_stage=task["product"],
                    stage_started_at=started,
                    last_progress_at=started,
                )
                row.pop("stage_progress", None)
                store.save(row)

                last_reported_at: list[float | None] = [None]

                def report_progress(completed: int, total: int) -> None:
                    now = time.monotonic()
                    if (
                        completed not in {1, total}
                        and last_reported_at[0] is not None
                        and now - last_reported_at[0] < 15
                    ):
                        return
                    try:
                        persisted = store.progress(
                            brand,
                            project,
                            row["worker"],
                            task["product"],
                            completed,
                            total,
                        )
                    except (OSError, sqlite3.Error):
                        logger.warning(
                            "Night product progress could not be persisted", exc_info=True
                        )
                        return
                    if persisted:
                        last_reported_at[0] = now

                with _worker_heartbeat(store, brand, project, row["worker"]):
                    if render is render_product:
                        render(row, task, destination, progress=report_progress)
                    else:
                        render(row, task, destination)
                if row["build"] != build_identity() or row["runtime"] != runtime_identity():
                    raise ValueError("Processing build changed during product render")
                receipt = validate(row, task, destination)
                if receipt is None:
                    raise ValueError("render returned without a validated product receipt")
                _adopt(task, destination, receipt)
                row["last_progress_at"] = time.time()
                _storage_preflight(row, target, 0)
            except (StorageWaitError, ProductWaitError) as exc:
                task.update(status="waiting", wait_kind=exc.kind)
                row.update(status="waiting", reason=str(exc), wait_kind=exc.kind)
                store.save(row)
                return 2
            except Exception as exc:  # noqa: BLE001 - preserve sibling products and evidence
                task.update(status="failed", error=str(exc), output=None, receipt=None)
            store.save(row)

        row = store.get(brand, project)
        for request in row.get("private_export_requests", []):
            if request.get("status") == "complete":
                continue
            if request.get("status") == "waiting" and request.get("wait_kind") == "budget":
                row.update(status="waiting", wait_kind="budget")
                store.save(row)
                return 2
            if request.get("status") not in {"queued", "running", "waiting"}:
                continue
            controls = store.get(brand, project)
            if controls.get("cancel_requested"):
                _mark_cancelled(row)
                store.save(row)
                return 1
            if controls.get("pause_requested"):
                row["status"] = "paused"
                store.save(row)
                return 2
            if prepare_video is None or estimate_video is None:
                from .night_private_video import (
                    estimate_private_video_bytes,
                    prepare_private_video,
                )

                prepare_video = prepare_video or prepare_private_video
                estimate_video = estimate_video or estimate_private_video_bytes
            try:
                required = int(estimate_video(row, request["task_index"]))
                if required < 0:
                    raise ValueError("private video storage estimate must not be negative")
                _storage_preflight(row, target, required)
                request["status"] = "running"
                request["attempts"] += 1
                request.pop("error", None)
                request.pop("wait_kind", None)
                started = time.time()
                row.update(
                    current_stage="private_video",
                    stage_started_at=started,
                    last_progress_at=started,
                )
                row.pop("stage_progress", None)
                store.save(row)
                with _worker_heartbeat(store, brand, project, row["worker"]):
                    record = prepare_video(row, request["task_index"], store.root)
                store.add_private_export(brand, project, record)
                row = store.get(brand, project)
                saved_request = next(
                    item for item in row["private_export_requests"] if item["id"] == request["id"]
                )
                saved_request.update(status="complete", export_id=record["export_id"])
                saved_request.pop("error", None)
                _storage_preflight(row, target, 0)
            except StorageWaitError as exc:
                row = store.get(brand, project)
                saved_request = next(
                    item for item in row["private_export_requests"] if item["id"] == request["id"]
                )
                saved_request.update(status="waiting", wait_kind=exc.kind)
                row.update(status="waiting", reason=str(exc), wait_kind=exc.kind)
                store.save(row)
                return 2
            except Exception as exc:  # noqa: BLE001 - preserve the validated timelapse
                row = store.get(brand, project)
                saved_request = next(
                    item for item in row["private_export_requests"] if item["id"] == request["id"]
                )
                saved_request.update(status="failed", error=str(exc))
            store.save(row)

        statuses = [task["status"] for task in row["tasks"]]
        row["current_stage"] = None
        row.pop("stage_progress", None)
        row["last_progress_at"] = time.time()
        row["status"] = (
            "complete"
            if statuses and all(status == "complete" for status in statuses)
            else "failed"
            if statuses and all(status == "failed" for status in statuses)
            else "partial"
        )
        store.save(row)
        return 0 if row["status"] == "complete" else 1
