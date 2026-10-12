"""Atomic, memory-mappable persistence for deep-linear stack components.

This module deliberately has no CLI or orchestration dependency.  It is the small
on-disk boundary between a completed chunk stack and a later deep combine.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from gomoveshift.pipeline.photos.astro_stack import StackResult

LINEAR_ARTIFACT_SCHEMA = 4
# schema 1: original manifest, no per-frame triage/radial-residual provenance.
# schema 2: registration_provenance entries MAY additionally carry a "triage" block
# (star_count, background_median, fwhm_px, shape major/minor/ellipticity split),
# "radial_inlier_residuals_px_in_sample" (center/middle/outer RANSAC-inlier residuals),
# "exposure_s" and "captured_at" -- additive only, every schema-1 key unchanged.
# schema 3: `sky_count.npy` is float32, a per-pixel QUALITY-WEIGHTED contribution sum
# (`sky_count = raw_count * chunk_sky_weight(registration)`, see `chunk_sky_weight` below)
# instead of a literal uint16 per-pixel frame count; a "sky_weight" manifest key records the
# scalar applied. Schema 1/2 artifacts keep the literal uint16 count (implicit weight 1.0) and
# are still readable -- every downstream reader already treats `sky_count` as a generic
# non-negative numeric weight, never a literal integer.
# schema 4: registration_provenance entries MAY additionally carry "match_yield_by_stage"
# (``match_progressive``'s per-stage match counts: seed/wide/tighten_10px/tighten_4px/final) --
# additive only, every schema-1/2/3 key and shape unchanged.
_MIN_SUPPORTED_LINEAR_ARTIFACT_SCHEMA = 1
_SKY_COUNT_WEIGHTED_SCHEMA = 3
_LINEAR_ALGORITHM_SOURCES = (
    "astro_calibrate.py",
    "astro_orchestrator.py",
    "astro_register.py",
    "astro_stack.py",
)
_PAYLOAD_DTYPES = {
    "sky_sum.npy": np.dtype(np.float32),
    "sky_count.npy": np.dtype(np.uint16),
    "ground.npy": np.dtype(np.float32),
}


def _sky_count_dtype(schema: int) -> np.dtype:
    """The on-disk dtype of ``sky_count.npy`` for a given manifest schema."""
    if schema >= _SKY_COUNT_WEIGHTED_SCHEMA:
        return np.dtype(np.float32)
    return _PAYLOAD_DTYPES["sky_count.npy"]


def _finite_positive_values(entries: Sequence, key: str) -> list[float]:
    values = []
    for entry in entries:
        triage = entry.get("triage") if isinstance(entry, dict) else None
        if not isinstance(triage, dict):
            continue
        value = triage.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
            values.append(float(value))
    return values


def _finite_values(entries: Sequence, key: str) -> list[float]:
    values = []
    for entry in entries:
        triage = entry.get("triage") if isinstance(entry, dict) else None
        if not isinstance(triage, dict):
            continue
        value = triage.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    return values


def chunk_sky_weight(registration_provenance: Sequence[Mapping[str, object]]) -> float:
    """A stacked chunk's relative sky-contribution weight in ``(0, 1]``.

    Derived entirely from THIS chunk's own persisted per-frame triage (``fwhm_px``,
    ``background_median``, recorded from schema 2): the more its own frames' focus and sky
    background level varied from each other, the less that chunk's frames actually agreed
    with one another, so the whole chunk contributes less to the deep stack. Each term is a
    coefficient of variation (stddev / mean), so the weight is dimensionless and bounded in
    ``(0, 1]`` -- it can only ever reduce a chunk's raw physical frame-count scale, never
    invent an absolute physical reference this codebase has never measured, and every
    downstream reader that assumes ``sky_count`` tracks something on the order of
    `n_frames` (the `MIN_STACK_CONTRIBUTORS` floor, the rejection kappa table) keeps working.
    Fewer than two triage-bearing frames -- schema-1/2 manifests, or a schema-2 manifest with
    no triage recorded -- return 1.0: contribute exactly as before, no invented signal.
    """
    fwhms = _finite_positive_values(registration_provenance, "fwhm_px")
    backgrounds = _finite_values(registration_provenance, "background_median")
    if len(fwhms) < 2 or len(backgrounds) < 2:
        return 1.0
    fwhm_mean = float(np.mean(fwhms))
    background_mean = float(np.mean(backgrounds))
    if fwhm_mean <= 0 or background_mean == 0:
        return 1.0
    fwhm_cv = float(np.std(fwhms)) / fwhm_mean
    background_cv = abs(float(np.std(backgrounds)) / background_mean)
    denominator = (1.0 + fwhm_cv) * (1.0 + background_cv)
    if not np.isfinite(denominator) or denominator <= 0:
        return 1.0
    return 1.0 / denominator


class LinearArtifactError(ValueError):
    """A deep-linear artifact is incomplete, incompatible, or corrupted."""


def linear_algorithm_hash() -> str:
    """Content hash of the stages that produce a chunk's persisted linear values."""
    digest = hashlib.sha256()
    module_dir = Path(__file__).parent
    for filename in _LINEAR_ALGORITHM_SOURCES:
        digest.update(filename.encode("utf-8"))
        digest.update((module_dir / filename).read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LinearArtifact:
    """Validated, read-only component arrays loaded with ``numpy.memmap``."""

    sky_sum: np.ndarray
    sky_count: np.ndarray
    ground: np.ndarray
    manifest: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_components(result: StackResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    named = {
        "sky_sum": result.sky_sum,
        "sky_count": result.sky_count,
        "ground": result.ground,
    }
    for name, value in named.items():
        if value is None:
            raise LinearArtifactError(f"linear artifact requires StackResult.{name}")
    sky_sum = np.asarray(named["sky_sum"])
    sky_count = np.asarray(named["sky_count"])
    ground = np.asarray(named["ground"])
    if sky_sum.dtype != np.float32:
        raise LinearArtifactError(f"sky_sum must be float32, got {sky_sum.dtype}")
    if sky_count.dtype != np.uint16:
        raise LinearArtifactError(f"sky_count must be uint16, got {sky_count.dtype}")
    if ground.dtype != np.float32:
        raise LinearArtifactError(f"ground must be float32, got {ground.dtype}")
    if sky_sum.ndim != 3 or sky_sum.shape[-1] != 3:
        raise LinearArtifactError(f"sky_sum must be HxWx3, got {sky_sum.shape}")
    if ground.shape != sky_sum.shape or sky_count.shape != sky_sum.shape[:2]:
        raise LinearArtifactError("sky_sum, sky_count, and ground shapes are inconsistent")
    return sky_sum, sky_count, ground


def write_linear_artifact(
    destination: Path,
    result: StackResult,
    *,
    reference_raw: str,
    capture_identity: Mapping[str, object],
    mask_provenance: Mapping[str, object],
    source_frames: Sequence[str],
    registration_provenance: Sequence[Mapping[str, object]],
    photometry_provenance: Sequence[Mapping[str, object]],
    algorithm_version: str,
) -> None:
    """Write a validated artifact directory atomically, refusing replacement.

    Payloads stay as uncompressed ``.npy`` files so later combines can map them
    instead of materializing every chunk.  Nothing is visible at ``destination``
    until all three payloads and their manifest have been written and validated.
    """
    destination = Path(destination)
    sky_sum, sky_count, ground = _validate_components(result)
    if destination.exists():
        raise LinearArtifactError(f"linear artifact already exists: {destination}")
    try:
        provenance = json.loads(json.dumps(mask_provenance, sort_keys=True))
        identity = json.loads(json.dumps(capture_identity, sort_keys=True))
        frames = json.loads(json.dumps(list(source_frames)))
        registration = json.loads(json.dumps(list(registration_provenance), sort_keys=True))
        photometry = json.loads(json.dumps(list(photometry_provenance), sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise LinearArtifactError("linear artifact provenance must be JSON serializable") from exc
    if not frames or not registration or not photometry:
        raise LinearArtifactError(
            "linear artifact requires source-frame, registration, and photometry provenance"
        )
    sky_weight = chunk_sky_weight(registration)
    weighted_sky_count = (sky_count.astype(np.float32) * sky_weight).astype(np.float32)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir()
        payloads = {
            "sky_sum.npy": sky_sum,
            "sky_count.npy": weighted_sky_count,
            "ground.npy": ground,
        }
        manifest_payloads = {}
        for filename, array in payloads.items():
            path = temporary / filename
            np.save(path, array, allow_pickle=False)
            manifest_payloads[filename] = {
                "sha256": _sha256(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        manifest = {
            "schema": LINEAR_ARTIFACT_SCHEMA,
            "algorithm_version": algorithm_version,
            "reference_raw": reference_raw,
            "capture_identity": identity,
            "mask_provenance": provenance,
            "source_frames": frames,
            "registration": registration,
            "photometry": photometry,
            "n_frames": result.n_frames,
            "mean_rejected_fraction": result.mean_rejected_fraction,
            "rejection_mode": result.rejection_mode,
            "sky_weight": sky_weight,
            "payloads": manifest_payloads,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _load_validated(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_validated(root: Path, *, verify_hashes: bool = True) -> LinearArtifact:
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LinearArtifactError(f"invalid linear artifact manifest: {manifest_path}") from exc
    schema = manifest.get("schema")
    if not isinstance(schema, int) or not (
        _MIN_SUPPORTED_LINEAR_ARTIFACT_SCHEMA <= schema <= LINEAR_ARTIFACT_SCHEMA
    ):
        raise LinearArtifactError(f"unsupported linear artifact schema: {manifest.get('schema')!r}")
    payload_specs = manifest.get("payloads")
    if not isinstance(payload_specs, dict):
        raise LinearArtifactError("linear artifact manifest has no payloads")
    arrays = {}
    expected_dtypes = dict(_PAYLOAD_DTYPES)
    expected_dtypes["sky_count.npy"] = _sky_count_dtype(schema)
    for filename, dtype in expected_dtypes.items():
        spec = payload_specs.get(filename)
        if not isinstance(spec, dict):
            raise LinearArtifactError(f"linear artifact manifest is missing {filename}")
        path = root / filename
        if not path.is_file():
            raise LinearArtifactError(f"linear artifact payload is missing: {filename}")
        if verify_hashes and spec.get("sha256") != _sha256(path):
            raise LinearArtifactError(f"linear artifact payload sha256 mismatch: {filename}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise LinearArtifactError(f"invalid linear artifact payload: {filename}") from exc
        if (
            array.dtype != dtype
            or list(array.shape) != spec.get("shape")
            or spec.get("dtype") != str(dtype)
        ):
            raise LinearArtifactError(f"linear artifact payload metadata mismatch: {filename}")
        arrays[filename] = array
    if (
        arrays["sky_sum.npy"].ndim != 3
        or arrays["sky_sum.npy"].shape[-1] != 3
        or arrays["ground.npy"].shape != arrays["sky_sum.npy"].shape
        or arrays["sky_count.npy"].shape != arrays["sky_sum.npy"].shape[:2]
    ):
        raise LinearArtifactError("linear artifact component shapes are inconsistent")
    return LinearArtifact(
        sky_sum=arrays["sky_sum.npy"],
        sky_count=arrays["sky_count.npy"],
        ground=arrays["ground.npy"],
        manifest=manifest,
    )


def load_linear_artifact(destination: Path, *, verify_hashes: bool = True) -> LinearArtifact:
    """Load an artifact, verifying payload hashes unless already trusted by the caller."""
    return _load_validated(Path(destination), verify_hashes=verify_hashes)
