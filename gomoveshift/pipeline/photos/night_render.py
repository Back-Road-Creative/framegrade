"""Private, conventional derivatives of verified linear stacks, without RAW decoding."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .astro_linear import LinearArtifact, load_linear_artifact
from .astro_recipe import recipe_for, render_linear
from .astro_stack import blend, horizon_mask
from .develop import export_jpeg, export_tiff16

OUTPUTS = ("master.tif", "preview.jpg")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build() -> dict:
    root = Path(__file__).parent
    return {
        name: _sha(root / name)
        for name in (
            "night_render.py",
            "astro_recipe.py",
            "astro_linear.py",
            "astro_stack.py",
            "develop.py",
        )
    }


def _load(source: Path) -> tuple[LinearArtifact, str]:
    for filename in ("manifest.json", "sky_sum.npy", "sky_count.npy", "ground.npy"):
        if (source / filename).is_symlink():
            raise ValueError("linear artifact members must not be symlinks")
    before = _sha(source / "manifest.json")
    artifact = load_linear_artifact(source)
    if _sha(source / "manifest.json") != before:
        raise ValueError("linear manifest changed during validation")
    return artifact, before


def recompose_chunk(artifact: LinearArtifact) -> np.ndarray:
    """Undo persisted quality weighting; reconstruct the recorded horizon boundary.

    Recomposition here is horizon-only: the sky/ground split is rebuilt from the
    recorded boundary row. Silhouette masking is not implemented, so tree edges
    require review.
    """
    weight = float(artifact.manifest.get("sky_weight", 1.0))
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("unusable sky_weight")
    count = np.asarray(artifact.sky_count, dtype=np.float32) / weight
    sky = np.zeros(artifact.sky_sum.shape, dtype=np.float32)
    if not np.isfinite(count).all() or (count < 0).any():
        raise ValueError("invalid sky contribution count")
    np.divide(artifact.sky_sum, count[..., None], out=sky, where=count[..., None] > 0)
    try:
        boundary = float(artifact.manifest["mask_provenance"]["resolved_boundary_row"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("missing reproducible horizon boundary") from exc
    mask = horizon_mask(sky.shape[:2], horizon_frac=boundary / sky.shape[0])
    # A pixel with no aligned contribution must retain its captured ground sample.
    mask = np.where(count > 0, mask, 0)
    return blend(sky, artifact.ground, mask)


def render_artifact(source: Path, destination: Path, recipe_name: str = "natural") -> dict:
    """Write an empty attempt directory. A receipt exists only after BOTH exports.

    The caller owns admission, exclusive attempt allocation and atomic publication.
    A failed preview leaves a bound master-only receipt for preview-only recovery.
    """
    source, destination = Path(source), Path(destination)
    build = _build()
    recipe = recipe_for(recipe_name)
    artifact, manifest_sha = _load(source)
    if destination.is_symlink() or (destination.exists() and any(destination.iterdir())):
        raise FileExistsError("render destination must be an empty new attempt directory")
    destination.mkdir(parents=True, exist_ok=True)
    linear = render_linear(recompose_chunk(artifact), recipe)
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
    master = np.rint(np.clip(srgb, 0, 1) * 65535).astype(np.uint16)
    export_tiff16(master, destination / "master.tif")
    if not _valid_image(destination / "master.tif", master.shape[1], master.shape[0]):
        raise ValueError("master export is not a valid profiled TIFF")
    if _sha(source / "manifest.json") != manifest_sha:
        raise ValueError("linear manifest changed during render")
    if _build() != build:
        raise ValueError("render build changed during processing")
    receipt = {
        "schema": 1,
        "state": "preview_pending",
        "quality_review_required": True,
        "source_manifest_sha256": manifest_sha,
        "source_payloads": artifact.manifest["payloads"],
        "recipe_name": recipe_name,
        "recipe": asdict(recipe),
        "build": build,
        "encoding": "sRGB",
        "crop": None,
        "limitations": [
            "Horizon-only composition; trees may retain alignment ghosts.",
            "No white-balance calibration or spatial color denoising.",
        ],
        "outputs": {
            name: {
                "sha256": _sha(destination / name),
                "width": master.shape[1],
                "height": master.shape[0],
            }
            for name in ("master.tif",)
        },
    }
    _sync(destination / "master.tif")
    _atomic_receipt(destination, "preview-pending.json", receipt)
    return _finish_preview(source, destination, receipt, master)


def _sync(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _atomic_receipt(destination: Path, name: str, receipt: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination / name)
        directory_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _finish_preview(source: Path, destination: Path, receipt: dict, master: np.ndarray) -> dict:
    preview = destination / "preview.jpg"
    if preview.is_symlink():
        raise ValueError("preview must not be a symlink")
    # TIFF is already sRGB encoded. Quantize only, never redevelop or encode twice.
    export_jpeg(np.rint(master.astype(np.float32) / 257).astype(np.uint8), preview)
    if _sha(source / "manifest.json") != receipt["source_manifest_sha256"]:
        raise ValueError("linear manifest changed during render")
    if _build() != receipt["build"]:
        raise ValueError("render build changed during processing")
    if not _valid_image(preview, master.shape[1], master.shape[0]):
        raise ValueError("preview export is not a valid profiled JPEG")
    _sync(preview)
    receipt["outputs"]["preview.jpg"] = {
        "sha256": _sha(preview),
        "width": master.shape[1],
        "height": master.shape[0],
    }
    receipt["state"] = "complete"
    _atomic_receipt(destination, "receipt.json", receipt)
    return receipt


def recover_preview(source: Path, destination: Path, recipe_name: str = "natural") -> bool:
    """Recover only a provenance-verified master, under caller admission/exclusion.

    Corrupt or mismatched partials remain untouched. False means no validated
    completion; failed export may still leave its diagnostic partial JPEG.
    """
    import tifffile

    source, destination = Path(source), Path(destination)
    if (destination / "receipt.json").exists():
        return validate_render(source, destination, recipe_name)
    if not _validate(
        source, destination, recipe_name, "preview-pending.json", "preview_pending", ("master.tif",)
    ):
        return False
    try:
        receipt = json.loads((destination / "preview-pending.json").read_text())
        master = tifffile.imread(destination / "master.tif")
        _finish_preview(source, destination, receipt, master)
        return validate_render(source, destination, recipe_name)
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _valid_image(path: Path, width: int, height: int) -> bool:
    """Decode pixels as well as headers; matching digests alone are insufficient."""
    if path.suffix == ".tif":
        import tifffile

        with tifffile.TiffFile(path) as image:
            rgb = image.asarray()
            return (
                len(image.pages) == 1
                and rgb.shape == (height, width, 3)
                and rgb.dtype == np.uint16
                and bool(image.pages[0].tags.get(34675))
            )
    from PIL import Image

    with Image.open(path) as preview:
        preview.load()
        return (
            preview.format == "JPEG"
            and preview.size == (width, height)
            and preview.mode == "RGB"
            and bool(preview.info.get("icc_profile"))
        )


def validate_render(source: Path, destination: Path, recipe_name: str = "natural") -> bool:
    """Conservative reuse: source, recipe, producing build and both outputs must match."""
    return _validate(source, destination, recipe_name, "receipt.json", "complete", OUTPUTS)


def _validate(
    source: Path,
    destination: Path,
    recipe_name: str,
    receipt_name: str,
    state: str,
    outputs: tuple[str, ...],
) -> bool:
    try:
        source, destination = Path(source), Path(destination)
        if destination.is_symlink() or (destination / receipt_name).is_symlink():
            return False
        receipt = json.loads((destination / receipt_name).read_text())
        artifact, manifest_sha = _load(source)
        if (
            receipt["schema"] != 1
            or receipt["state"] != state
            or receipt["source_manifest_sha256"] != manifest_sha
            or receipt["source_payloads"] != artifact.manifest["payloads"]
            or receipt["recipe_name"] != recipe_name
            or receipt["recipe"] != asdict(recipe_for(recipe_name))
            or receipt["build"] != _build()
            or receipt["encoding"] != "sRGB"
            or receipt["crop"] is not None
            or receipt["quality_review_required"] is not True
        ):
            return False
        height, width = artifact.ground.shape[:2]
        for name in outputs:
            path = destination / name
            spec = receipt["outputs"][name]
            if (
                path.is_symlink()
                or spec["sha256"] != _sha(path)
                or spec["width"] != width
                or spec["height"] != height
                or not _valid_image(path, width, height)
            ):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False
