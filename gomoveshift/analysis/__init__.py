"""Video analysis modules.

The privacy-blur chain (detection, optical-flow tracking, region timeline, mask
rasterisation and the independent cabin-coverage verifier) now ships as the
standalone ``redactcam`` package — it used to live here as ``privacy_detect``,
``blur_track``, ``blur_regions``, ``blur_mask``, ``blur_coverage`` and
``model_cache``. Only the seam stays behind: the four ``resolve_*_model``
helpers below adapt this service's ``config.drive`` model knobs onto redactcam's
``ModelSpec``, and pin the model cache to ``<cache home>/models`` so a warm cache is never
re-downloaded.
"""

from __future__ import annotations

import logging
from pathlib import Path

from redactcam import FrameRegions, ModelSpec, detect_regions, resolve_model, verify_cabin_coverage

from ..paths import gms_cache_home
from .motion import MotionAnalyzer
from .ocr import OCREngine
from .pipeline import AnalysisPipeline
from .shorts_extractor import ExtractedShort, ExtractionResult, ShortsExtractor, extract_shorts_cli
from .signs import SignDetector

logger = logging.getLogger(__name__)

# Pinned, NOT redactcam's ``DEFAULT_CACHE_DIR`` (``~/.cache/redactcam/models``):
# the models live under the gomoveshift cache home (``<cache home>/models``),
# so a package-default fetch never orphans the warm cache. ``None`` means "resolve
# the cache home at call time" (it follows ``GMS_CACHE_HOME`` / ``XDG_CACHE_HOME``);
# set it to a path to pin the directory.
MODEL_CACHE_DIR: Path | None = None


def _model_cache_dir() -> Path:
    return MODEL_CACHE_DIR if MODEL_CACHE_DIR is not None else gms_cache_home() / "models"


# What the caller falls back to when a model cannot be resolved — named in the
# WARN so an operator sees which capability just went missing.
_DEGRADE = {
    "plate": "faces only",
    "face": "CenterFace fallback",
    "vehicle": "occupant cabin blur off",
    "person": "whole-person blur off",
}


def _resolve_drive_model(drive, prefix: str) -> Path | None:
    """Resolve a ``config.drive`` ``{prefix}_model_*`` triple to a local ONNX path.

    ``{prefix}_model_path`` (operator override) wins; otherwise the model is
    fetched once from ``{prefix}_model_url``, SHA256-verified against
    ``{prefix}_model_sha256`` and cached. Never raises — an unresolvable model
    logs a WARN naming the caller's fallback and returns ``None``.
    """
    spec = ModelSpec(
        name=prefix,
        url=(getattr(drive, f"{prefix}_model_url", "") or "").strip(),
        sha256=(getattr(drive, f"{prefix}_model_sha256", "") or "").strip(),
        path=(getattr(drive, f"{prefix}_model_path", "") or "").strip(),
    )
    resolved = resolve_model(spec, cache_dir=_model_cache_dir())
    if resolved is None:
        logger.warning("privacy blur: %s model unresolvable — %s", prefix, _DEGRADE[prefix])
    return resolved


def resolve_plate_model(drive) -> Path | None:
    """Plate ONNX path, or None — the detector then runs faces-only (WARN)."""
    return _resolve_drive_model(drive, "plate")


def resolve_face_model(drive) -> Path | None:
    """YOLO face ONNX path, or None — the detector falls back to CenterFace."""
    return _resolve_drive_model(drive, "face")


def resolve_vehicle_model(drive) -> Path | None:
    """COCO vehicle ONNX path, or None — the occupant cabin blur is disabled."""
    return _resolve_drive_model(drive, "vehicle")


def resolve_person_model(drive) -> Path | None:
    """COCO person ONNX path, or None — the whole-person blur is disabled."""
    return _resolve_drive_model(drive, "person")


__all__ = [
    "MotionAnalyzer",
    "SignDetector",
    "OCREngine",
    "AnalysisPipeline",
    "ShortsExtractor",
    "ExtractedShort",
    "ExtractionResult",
    "extract_shorts_cli",
    "FrameRegions",
    "detect_regions",
    "verify_cabin_coverage",
    "MODEL_CACHE_DIR",
    "resolve_plate_model",
    "resolve_face_model",
    "resolve_vehicle_model",
    "resolve_person_model",
]
