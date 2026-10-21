"""Receipt-bound private timelapse exports through the video privacy pipeline."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from ..day_master_encoder import PRIVACY_VERIFIED, REQUIRED_PRIVACY_CLASSES
from .face_filter import attestable_privacy_classes
from .night_products import validate_product_receipt
from .night_project import build_identity, runtime_identity


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


def _location_metadata_absent(path: Path) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format_tags:stream_tags",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        payload = json.loads(result.stdout)
        groups = [payload.get("format", {}).get("tags", {})]
        groups.extend(stream.get("tags", {}) for stream in payload.get("streams", []))
        sensitive = ("location", "gps", "latitude", "longitude", "iso6709")
        return not any(
            any(marker in str(key).lower() for marker in sensitive)
            for tags in groups
            for key in tags
        )
    except (OSError, ValueError, TypeError, subprocess.SubprocessError, json.JSONDecodeError):
        return False


def _validated_video(path: Path, expected_sha: str, expected_size: int) -> bool:
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected_size
            or _sha256(path) != expected_sha
            or not _location_metadata_absent(path)
        ):
            return False
        from ..ffmpeg_runner import validate_video

        result = validate_video(path, require_audio=False)
        return result.valid and result.duration > 0
    except (OSError, ValueError, TypeError):
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
        if (
            receipt.get("location_metadata_absent") is not True
            or receipt.get("cabin_coverage_verified") is not True
            or receipt.get("publishable") is not False
        ):
            return None
        classes = receipt["privacy_classes"]
        if any(classes[name]["state"] != PRIVACY_VERIFIED for name in REQUIRED_PRIVACY_CLASSES):
            return None
        output = project_root / receipt["path"]
        if not output.resolve().is_relative_to(target.resolve()):
            return None
        if not _validated_video(output, receipt["sha256"], receipt["size"]):
            return None
        return receipt
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def validate_private_video(row: dict, record: dict, workspace: Path) -> Path | None:
    """Validate a saved private-video record and return its MP4 path."""
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
                "media_type",
                "source_sha256",
                "build",
                "runtime",
                "export_build",
                "export_runtime",
                "state",
            )
        }
        validated = _existing(project_root, receipt_path.parent, expected)
        if validated != record:
            return None
        return project_root / record["path"]
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _verify_cabin_mask(source: Path, mask: Path, drive, vehicle_model, plate_model) -> dict:
    """Fail closed when a corroborated vehicle cabin is not covered by the mask."""
    from ...analysis import verify_cabin_coverage

    if vehicle_model is None:
        raise ValueError("verified vehicle-occupant model required for cabin coverage")
    report = verify_cabin_coverage(
        source,
        mask,
        vehicle_model,
        plate_model=plate_model,
        cabin_frac=drive.blur_vehicle_cabin_frac,
        conf=drive.blur_vehicle_conf,
        sample_fps=getattr(drive, "blur_cabin_gate_sample_fps", 4.0),
        min_coverage=getattr(drive, "blur_cabin_gate_min_coverage", 0.5),
        min_height_frac=drive.blur_vehicle_min_height_frac,
        ignore_bottom_frac=drive.blur_vehicle_ignore_bottom_frac,
        plate_max_area_frac=drive.blur_plate_max_area_frac,
    )
    if not report.ok:
        raise ValueError(
            f"private video cabin coverage failed for {len(report.leaks)} of "
            f"{report.vehicle_frames} corroborated vehicle frames"
        )
    return {
        "cabin_coverage_verified": True,
        "cabin_vehicle_frames": int(report.vehicle_frames),
    }


def _render_private_video(source: Path, output: Path, mask: Path) -> dict:
    """Detect, track, mask, and blur with the service's established video settings."""
    from redactcam import apply_blur, detect_and_track, materialize_mask_video
    from redactcam import timeline as blur_timeline

    from ...analysis import (
        resolve_face_model,
        resolve_person_model,
        resolve_plate_model,
        resolve_vehicle_model,
    )
    from ...config import get_config
    from .. import blur_gate

    drive = get_config().drive
    person_on = bool(getattr(drive, "blur_person_enabled", False))
    person_model = resolve_person_model(drive) if person_on else None
    plate_model = resolve_plate_model(drive)
    vehicle_model = resolve_vehicle_model(drive)
    fps, frame_count, width, height = blur_timeline.probe_video(source)
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("timelapse video properties are incomplete")
    boxes, decoded_count, faces, plates = detect_and_track(
        source,
        sample_fps=drive.blur_sample_fps,
        conf=drive.blur_conf,
        face_conf=drive.blur_face_conf,
        face_sample_fps=drive.blur_face_sample_fps,
        face_confirm_min_detections=drive.blur_face_confirm_min_detections,
        pre_roll_frames=drive.blur_pre_roll_frames,
        plate_model_path=plate_model,
        face_model_path=resolve_face_model(drive),
        tile_px=drive.blur_tile_px,
        tile_overlap=drive.blur_tile_overlap,
        plate_min_aspect=drive.blur_plate_min_aspect,
        plate_max_aspect=drive.blur_plate_max_aspect,
        plate_max_area_frac=drive.blur_plate_max_area_frac,
        plate_boost_speed_px=drive.blur_plate_boost_speed_px,
        plate_boost_max_run=drive.blur_plate_boost_max_run,
        track_face_margin_frac=drive.blur_face_margin_frac,
        track_face_motion_gain=drive.blur_face_motion_gain,
        track_cabin_frac=drive.blur_vehicle_cabin_frac,
        vehicle_model_path=vehicle_model,
        vehicle_conf=drive.blur_vehicle_conf,
        vehicle_min_height_frac=drive.blur_vehicle_min_height_frac,
        vehicle_ignore_bottom_frac=drive.blur_vehicle_ignore_bottom_frac,
        track_person=person_on and person_model is not None,
        person_model_path=person_model,
        person_conf=getattr(drive, "blur_person_conf", 0.35),
        person_min_height_frac=getattr(drive, "blur_person_min_height_frac", 0.0),
    )
    # A detector miss must not unmask an object still in frame: hold it per object.
    boxes = blur_gate.hold_missed_detections(
        boxes,
        sample_interval_frames=math.ceil(fps / max(drive.blur_sample_fps, 1e-6)),
    )
    boxes = blur_gate.fill_detection_gaps(
        boxes, max_gap_frames=math.ceil(fps / max(drive.blur_sample_fps, 1e-6)) + 1
    )
    boxes = blur_gate.backfill_track_births(boxes, pre_roll_frames=drive.blur_pre_roll_frames)
    decoded_count = decoded_count or frame_count
    timeline = blur_timeline.build_timeline(
        boxes,
        source_hash=_sha256(source),
        source_fps=fps,
        source_frame_count=decoded_count,
        frame_width=width,
        frame_height=height,
        box_dilation_px=drive.blur_box_dilation_px,
        sample_fps=drive.blur_sample_fps,
        conf=drive.blur_conf,
    )
    materialize_mask_video(timeline, mask)
    coverage = _verify_cabin_mask(source, mask, drive, vehicle_model, plate_model)
    apply_blur(
        source,
        mask,
        output,
        video_width=width,
        strength=drive.blur_strength,
        extra_output_args=["-map_metadata", "-1", "-movflags", "+faststart", "-pix_fmt", "yuv420p"],
    )
    return {
        **coverage,
        "decoded_frames": int(decoded_count),
        "masked_frames": sum(1 for regions in boxes.values() if regions),
        "face_detections": int(faces),
        "plate_detections": int(plates),
    }


def estimate_private_video_bytes(
    row: dict,
    task_index: int,
    *,
    receipt_validator: Callable = validate_product_receipt,
) -> int:
    """Reserve output, mask, and encoding headroom from the validated source size."""
    tasks = row.get("tasks", [])
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError("unknown night product task")
    task = tasks[task_index]
    if task.get("product") != "timelapse" or not task.get("output"):
        raise ValueError("timelapse product required for private video estimate")
    receipt = receipt_validator(row, task, Path(task["output"]))
    if receipt is None:
        raise ValueError("validated night product receipt required")
    sizes = [
        output.get("size")
        for output in receipt.get("outputs", [])
        if output.get("role") == "video" and Path(output.get("path", "")).suffix.lower() == ".mp4"
    ]
    if len(sizes) != 1 or type(sizes[0]) is not int or sizes[0] <= 0:
        raise ValueError("validated timelapse size required for private video estimate")
    return sizes[0] * 3 + 64 * 1024**2


def prepare_private_video(
    row: dict,
    task_index: int,
    workspace: Path,
    *,
    receipt_validator: Callable = validate_product_receipt,
    privacy_reporter: Callable = attestable_privacy_classes,
    renderer: Callable = _render_private_video,
) -> dict:
    """Create an unpublished, privacy-checked MP4 copy of one timelapse product."""
    if row.get("schema") != 2 or row.get("kind") != "night_products":
        raise ValueError("typed night product project required")
    tasks = row.get("tasks", [])
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError("unknown night product task")
    task = tasks[task_index]
    if (
        task.get("product") != "timelapse"
        or task.get("status") not in {"complete", "partial"}
        or not task.get("output")
    ):
        raise ValueError("completed timelapse product required for private video export")

    workspace = Path(workspace).resolve(strict=True)
    project_path = workspace / row["id"]
    project_root = project_path.resolve(strict=True)
    attempt = Path(task["output"])
    resolved_attempt = attempt.resolve(strict=True)
    if (
        project_path.is_symlink()
        or not project_root.is_relative_to(workspace)
        or attempt.is_symlink()
        or not resolved_attempt.is_relative_to(project_root)
    ):
        raise ValueError("invalid night product workspace")
    receipt = receipt_validator(row, task, resolved_attempt)
    if receipt is None:
        raise ValueError("validated night product receipt required")
    video = next(
        (
            output
            for output in receipt.get("outputs", [])
            if output.get("role") == "video"
            and Path(output.get("path", "")).suffix.lower() == ".mp4"
        ),
        None,
    )
    if video is None:
        raise ValueError("validated timelapse video required")
    source = resolved_attempt / video["path"]
    if (
        source.is_symlink()
        or not source.is_file()
        or not source.resolve().is_relative_to(resolved_attempt)
        or source.stat().st_size != video["size"]
        or _sha256(source) != video["sha256"]
    ):
        raise ValueError("night timelapse changed after validation")

    identity = {
        "schema": 1,
        "project_id": row["id"],
        "task_index": task_index,
        "product": "timelapse",
        "media_type": "video",
        "source_sha256": video["sha256"],
        "build": row["build"],
        "runtime": row["runtime"],
        "export_build": build_identity(),
        "export_runtime": runtime_identity(),
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
    descriptor = os.open(
        workspace / f".{row['id']}.export.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            existing = _existing(project_root, target, identity)
            if existing is None:
                raise ValueError("existing private video export failed receipt validation")
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
        temporary = Path(tempfile.mkdtemp(prefix=".export-video-", dir=exports))
        try:
            output = temporary / "selected.mp4"
            mask = temporary / "privacy-mask.mkv"
            evidence = renderer(source, output, mask)
            mask.unlink(missing_ok=True)
            output_sha = _sha256(output)
            output_size = output.stat().st_size
            if not _validated_video(output, output_sha, output_size):
                raise ValueError("private video failed media or location-metadata validation")
            relative = Path("exports") / export_id / output.name
            record = {
                **identity,
                "export_id": export_id,
                "path": str(relative),
                "receipt": str(Path("exports") / export_id / "export-receipt.json"),
                "sha256": output_sha,
                "size": output_size,
                "privacy_classes": privacy,
                "privacy_regions": int(evidence.get("masked_frames", 0)),
                "masked_frames": int(evidence.get("masked_frames", 0)),
                "decoded_frames": int(evidence.get("decoded_frames", 0)),
                "face_detections": int(evidence.get("face_detections", 0)),
                "plate_detections": int(evidence.get("plate_detections", 0)),
                "cabin_coverage_verified": evidence.get("cabin_coverage_verified") is True,
                "cabin_vehicle_frames": int(evidence.get("cabin_vehicle_frames", 0)),
                "location_metadata_absent": True,
                "quality_review_required": True,
                "publishable": False,
            }
            _atomic_json(temporary / "export-receipt.json", record)
            os.rename(temporary, target)
            return record
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
