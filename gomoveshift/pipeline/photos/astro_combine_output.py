from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .astro_combine import DeepCombineResult
from .astro_develop import AstroDevelopResult, develop_stack
from .develop import export_tiff16
from .preflight import check_disk_space

DEEP_SIDECAR_SCHEMA = 1
MATERIAL_THIN_COVERAGE_FRACTION = 0.01
_OUTPUT_BYTES_PER_PIXEL = 10  # float32 coverage + RGB uint16 TIFF
_OUTPUT_SAFETY_FACTOR = 1.10
_BYTES_PER_GIB = 1024**3


class DeepOutputError(ValueError):
    pass


@dataclass(frozen=True)
class DeepOutputPaths:
    image: Path
    coverage: Path
    sidecar: Path


def deep_output_paths(image: Path) -> DeepOutputPaths:
    image = Path(image)
    if image.suffix.lower() not in {".tif", ".tiff"}:
        raise DeepOutputError("deep combine output must be a .tif or .tiff")
    return DeepOutputPaths(
        image=image,
        coverage=image.with_suffix(".coverage.npy"),
        sidecar=image.with_suffix(".deep.json"),
    )


def estimate_deep_output_disk_gb(shape: tuple[int, ...]) -> float:
    pixels = int(shape[0]) * int(shape[1])
    return pixels * _OUTPUT_BYTES_PER_PIXEL * _OUTPUT_SAFETY_FACTOR / _BYTES_PER_GIB


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _photometry_payload(value) -> dict:
    return {"offset": value.offset, "scale": value.scale}


def _gradient_residual_payload(residual) -> dict:
    if residual is None:
        return {"center": None, "corners": None}
    return {"center": residual.center, "corners": residual.corners}


def deep_sidecar_payload(
    result: DeepCombineResult, developed: AstroDevelopResult, paths: DeepOutputPaths
) -> dict:
    chunks = []
    for index, (artifact, transform, sky_photo, ground_photo) in enumerate(
        zip(
            result.artifacts.artifacts,
            result.transforms,
            result.sky.photometry,
            result.ground.photometry,
        )
    ):
        manifest = artifact.manifest
        chunks.append(
            {
                "artifact": str(artifact.path),
                "payload_hashes": {
                    name: spec.get("sha256") for name, spec in manifest["payloads"].items()
                },
                "reference_raw": manifest["reference_raw"],
                "source_frames": manifest["source_frames"],
                "capture_identity": manifest["capture_identity"],
                "n_frames": int(manifest["n_frames"]),
                "sky_photometry": _photometry_payload(sky_photo),
                "ground_photometry": _photometry_payload(ground_photo),
                "transform": {
                    "matrix": np.asarray(transform.transform).tolist(),
                    "method": transform.method,
                    "links": [list(link) for link in transform.links],
                    "diagnostics": list(transform.diagnostics),
                },
                "is_reference": index == result.artifacts.reference_index,
            }
        )
    coverage = result.coverage
    warnings = []
    if coverage.below_floor_fraction >= MATERIAL_THIN_COVERAGE_FRACTION:
        warnings.append(
            f"{coverage.below_floor_fraction:.2%} of sky pixels are below the "
            f"{coverage.floor:g}-contributor floor"
        )
    reference = result.artifacts.reference
    return {
        "schema": DEEP_SIDECAR_SCHEMA,
        "meteor_mode": "off",
        "reference": {
            "artifact": str(reference.path),
            "index": result.artifacts.reference_index,
            "raw": reference.manifest["reference_raw"],
        },
        "mask": reference.manifest["mask_provenance"],
        "scope": {
            # `requested_scope` is None only for a result built without going through
            # `deep_combine` (older callers/tests) -- fall back to the resolved scope so
            # this key is never missing/null for a genuinely explicit request.
            "requested": result.requested_scope
            if result.requested_scope is not None
            else result.scope,
            "resolved": result.scope,
            "decision": result.scope_decision,
        },
        "chunks": chunks,
        "rejection": {
            "rejected_chunks": [],
            "sky_pixel_fraction": result.sky.rejected_pixel_fraction,
            "ground_pixel_fraction": result.ground.rejected_pixel_fraction,
        },
        "coverage": {
            "map": str(paths.coverage),
            "nominal_contributors": coverage.nominal_contributors,
            "floor": coverage.floor,
            "min_contributors": coverage.min_contributors,
            "median_contributors": coverage.median_contributors,
            "below_floor_fraction": coverage.below_floor_fraction,
        },
        "develop": {
            "black_point": developed.black_point,
            "m_values": developed.m_values,
            "background_before": developed.background_before,
            "background_after": developed.background_after,
            "background_gains": list(developed.background_gains),
            "peak": developed.peak,
            "gradient_residual": _gradient_residual_payload(developed.gradient_residual),
        },
        "output": {"image": str(paths.image), "sha256": _sha256(paths.image)},
        "warnings": warnings,
    }


def _write_npy_atomic(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def publish_deep_product(
    result: DeepCombineResult,
    image: Path,
    *,
    develop: Callable = develop_stack,
    export: Callable = export_tiff16,
    disk_check: Callable = check_disk_space,
) -> DeepOutputPaths:
    paths = deep_output_paths(image)
    collision = next((path for path in paths.__dict__.values() if path.exists()), None)
    if collision is not None:
        raise DeepOutputError(f"deep combine output already exists: {collision}")
    paths.image.parent.mkdir(parents=True, exist_ok=True)
    disk_check(paths.image.parent, min_free_gb=estimate_deep_output_disk_gb(result.rgb.shape))
    developed = develop(result.rgb, sky_mask=result.mask > 0.5)
    export(developed.rgb_uint16, paths.image)
    _write_npy_atomic(paths.coverage, result.sky.sky_count)
    _write_json_atomic(paths.sidecar, deep_sidecar_payload(result, developed, paths))
    return paths
