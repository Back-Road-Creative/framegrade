"""Versioned, conventional luminance rendering of measured scene-linear RGB.

Outputs remain display-linear: the export adapter must apply sRGB encoding once.
No spatial filtering, saturation, sky synthesis, or automatic percentile clipping.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_VERSION = "conventional-v1"
Crop = tuple[int, int, int, int]


@dataclass(frozen=True)
class NaturalRecipe:
    """Exposure multiplier, luminance contrast around 0.18, and linear black point."""

    version: str = _VERSION
    exposure: float = 1.0
    contrast: float = 1.0
    black_point: float = 0.0


def recipe_for(name: str) -> NaturalRecipe:
    """Resolve explicit presets; Natural remains the conservative default."""
    presets = {
        "natural": NaturalRecipe(),
        "accent": NaturalRecipe(exposure=2.0, contrast=1.15),
        "bold": NaturalRecipe(exposure=3.0, contrast=1.3),
    }
    if name not in presets:
        raise ValueError(f"Unknown conventional recipe: {name}")
    return presets[name]


def _validate(recipe: NaturalRecipe) -> None:
    if recipe.version != _VERSION:
        raise ValueError("Unsupported recipe version")
    for name, lower, upper in (
        ("exposure", 1e-6, 1e6),
        ("contrast", 0.1, 10),
        ("black_point", 0, 1e6),
    ):
        value = getattr(recipe, name)
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"Invalid recipe {name}")


def _crop(crop: Crop) -> None:
    if len(crop) != 4 or any(type(value) is not int for value in crop):
        raise ValueError("crop must contain four integer coordinates")
    left, top, right, bottom = crop
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError("crop must have positive area and nonnegative coordinates")


def render_linear(
    rgb: NDArray, recipe: NaturalRecipe, *, crop: Crop | None = None
) -> NDArray[np.float32]:
    """Return new bounded display-linear RGB, preserving per-pixel color ratios.

    Input is finite, nonnegative, scene-linear RGB in linear sRGB primaries. A
    nonzero black point explicitly clips dark luminance; it never occurs by default.
    Crop is (left, top, right, bottom), with exclusive right/bottom coordinates.
    """
    _validate(recipe)
    data = np.asarray(rgb, dtype=np.float64)
    if data.ndim != 3 or data.shape[2] != 3 or min(data.shape[:2]) == 0:
        raise ValueError("Expected nonempty H x W x 3 scene-linear RGB")
    if not np.isfinite(data).all() or np.any(data < 0):
        raise ValueError("Scene-linear RGB must be finite and nonnegative")
    if crop is not None:
        _crop(crop)
        left, top, right, bottom = crop
        if right > data.shape[1] or bottom > data.shape[0]:
            raise ValueError("crop exceeds image dimensions")
        data = data[top:bottom, left:right]
    luminance = data @ np.array([0.2126, 0.7152, 0.0722])
    signal = np.maximum(luminance - recipe.black_point, 0)
    # Log-domain curve avoids overflowing on bright, finite scene-linear inputs.
    log_signal = np.full_like(signal, -np.inf)
    np.log(signal, out=log_signal, where=signal > 0)
    log_curve = math.log(0.18) + recipe.contrast * (
        log_signal + math.log(recipe.exposure) - math.log(0.18)
    )
    mapped = np.exp(-np.logaddexp(0, -log_curve))
    scale = np.divide(mapped, luminance, out=np.zeros_like(mapped), where=luminance > 0)
    result = data * scale[..., None]
    # Shared RGB scale compresses out-of-gamut highlights without channel clipping.
    result /= np.maximum(result.max(axis=2, keepdims=True), 1)
    return result.astype(np.float32)


def _identity(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"name": path.name, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def derivative_receipt(
    source: Path, output: Path, recipe: NaturalRecipe, *, crop: Crop | None = None
) -> dict:
    """Build a deterministic private receipt; caller persists it atomically.

    Source may be a validated linear-artifact manifest. Its referenced payloads
    must already have been validated by the loader. This helper does not load them.
    """
    _validate(recipe)
    if crop is not None:
        _crop(crop)
    parameters = {"recipe": asdict(recipe), "crop": list(crop) if crop else None}
    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "schema_version": 1,
        **parameters,
        "recipe_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "source": _identity(source),
        "output": _identity(output),
        "generative": False,
        "working_space": "linear-sRGB",
        "operations": ["luminance_curve", "color_preserving_highlight_compression"],
    }
