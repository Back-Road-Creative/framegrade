"""Receipt-bound private keeper exports from validated night still products."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from gomoveshift.pipeline._file_lock import lock_exclusive

from ..day_master_encoder import PRIVACY_VERIFIED, REQUIRED_PRIVACY_CLASSES
from .develop import export_jpeg
from .face_filter import attestable_privacy_classes, blur_faces_for_publish
from .night_products import validate_product_receipt

_GPS_IFD_TAG = 34853


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _validated_jpeg(path: Path, expected_sha: str, expected_size: int) -> bool:
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected_size
            or _sha256(path) != expected_sha
        ):
            return False
        with Image.open(path) as image:
            image.load()
            return (
                image.format == "JPEG"
                and image.mode == "RGB"
                and bool(image.info.get("icc_profile"))
                and _GPS_IFD_TAG not in image.getexif()
            )
    except (OSError, ValueError):
        return False


def _existing(project_root: Path, target: Path, expected: dict) -> dict | None:
    receipt_path = target / "export-receipt.json"
    try:
        if (
            target.is_symlink()
            or not target.is_dir()
            or not target.resolve().is_relative_to(project_root)
            or receipt_path.is_symlink()
            or receipt_path.stat().st_size > 1024**2
        ):
            return None
        receipt = json.loads(receipt_path.read_text())
        if any(receipt.get(key) != value for key, value in expected.items()):
            return None
        if receipt.get("gps_exif_absent") is not True or receipt.get("publishable") is not False:
            return None
        classes = receipt["privacy_classes"]
        if any(classes[name]["state"] != PRIVACY_VERIFIED for name in REQUIRED_PRIVACY_CLASSES):
            return None
        output = project_root / receipt["path"]
        if not output.resolve().is_relative_to(target.resolve()):
            return None
        if not _validated_jpeg(output, receipt["sha256"], receipt["size"]):
            return None
        return receipt
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def validate_private_export(row: dict, record: dict, workspace: Path) -> Path | None:
    """Validate a saved keeper record and return its private JPEG path."""
    try:
        workspace = Path(workspace).resolve(strict=True)
        project_path = workspace / row["id"]
        project_root = project_path.resolve(strict=True)
        if (
            project_path.is_symlink()
            or not project_root.is_relative_to(workspace)
            or record["project_id"] != row["id"]
        ):
            return None
        receipt_path = project_root / record["receipt"]
        if receipt_path.is_symlink() or not receipt_path.resolve().is_relative_to(project_root):
            return None
        expected = {
            key: record[key]
            for key in (
                "schema",
                "project_id",
                "task_index",
                "product",
                "source_sha256",
                "build",
                "runtime",
                "state",
            )
        }
        validated = _existing(project_root, receipt_path.parent, expected)
        if validated != record:
            return None
        return project_root / record["path"]
    except (OSError, ValueError, TypeError, KeyError):
        return None


def prepare_private_export(
    row: dict,
    task_index: int,
    workspace: Path,
    *,
    receipt_validator: Callable = validate_product_receipt,
    privacy_reporter: Callable = attestable_privacy_classes,
    blur_filter: Callable = blur_faces_for_publish,
    jpeg_exporter: Callable = export_jpeg,
) -> dict:
    """Prepare one unpublished, privacy-checked JPEG keeper from a real preview."""
    if row.get("schema") != 2 or row.get("kind") != "night_products":
        raise ValueError("typed night product project required")
    tasks = row.get("tasks", [])
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError("unknown night product task")
    task = tasks[task_index]
    if task.get("status") not in {"complete", "partial"} or not task.get("output"):
        raise ValueError("night product is not ready for selection")

    workspace = Path(workspace).resolve(strict=True)
    project_path = workspace / row["id"]
    project_root = project_path.resolve(strict=True)
    if project_path.is_symlink() or not project_root.is_relative_to(workspace):
        raise ValueError("invalid night project workspace")
    attempt = Path(task["output"])
    resolved_attempt = attempt.resolve(strict=True)
    if attempt.is_symlink() or not resolved_attempt.is_relative_to(project_root):
        raise ValueError("invalid night product attempt")
    receipt = receipt_validator(row, task, resolved_attempt)
    if receipt is None:
        raise ValueError("validated night product receipt required")
    preview = next(
        (
            output
            for output in receipt.get("outputs", [])
            if output.get("role") == "preview"
            and Path(output.get("path", "")).suffix.lower() in {".jpg", ".jpeg"}
        ),
        None,
    )
    if preview is None:
        raise ValueError("a validated still preview is required for private export")
    source = resolved_attempt / preview["path"]
    if (
        source.is_symlink()
        or not source.is_file()
        or not source.resolve().is_relative_to(resolved_attempt)
        or source.stat().st_size != preview["size"]
        or _sha256(source) != preview["sha256"]
    ):
        raise ValueError("night product preview changed after validation")

    identity = {
        "schema": 1,
        "project_id": row["id"],
        "task_index": task_index,
        "product": task["product"],
        "source_sha256": preview["sha256"],
        "build": row["build"],
        "runtime": row["runtime"],
        "state": "private_review",
    }
    export_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    exports = project_root / "exports"
    exports.mkdir(exist_ok=True)
    if exports.is_symlink():
        raise ValueError("night export directory must not be a symlink")
    target = exports / export_id
    lock_descriptor = os.open(
        workspace / f".{row['id']}.export.lock",
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(lock_descriptor, "a") as lock:
        lock_exclusive(lock)
        if target.exists():
            existing = _existing(project_root, target, identity)
            if existing is None:
                raise ValueError("existing private export failed receipt validation")
            return existing

        privacy = privacy_reporter(blur=True)
        unavailable = [
            name
            for name in REQUIRED_PRIVACY_CLASSES
            if privacy.get(name, {}).get("state") != PRIVACY_VERIFIED
        ]
        if unavailable:
            raise ValueError(
                "privacy models are not verified for required classes: " + ", ".join(unavailable)
            )
        temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=exports))
        try:
            with Image.open(source) as image:
                image.load()
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            filtered, regions = blur_filter(pixels, source.name, blur=True)
            output = jpeg_exporter(filtered, temporary / "selected.jpg")
            with Image.open(output) as exported:
                if _GPS_IFD_TAG in exported.getexif():
                    raise ValueError("private export retained GPS metadata")
            relative_output = Path("exports") / export_id / output.name
            record = {
                **identity,
                "export_id": export_id,
                "path": str(relative_output),
                "receipt": str(Path("exports") / export_id / "export-receipt.json"),
                "sha256": _sha256(output),
                "size": output.stat().st_size,
                "privacy_classes": privacy,
                "privacy_regions": int(regions),
                "gps_exif_absent": True,
                "quality_review_required": True,
                "publishable": False,
            }
            if not _validated_jpeg(output, record["sha256"], record["size"]):
                raise ValueError("private export failed JPEG or metadata validation")
            _atomic_json(temporary / "export-receipt.json", record)
            os.rename(temporary, target)
            return record
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
