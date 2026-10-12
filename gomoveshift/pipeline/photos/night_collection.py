"""Read-only discovery and typed output contracts for a night collection.

Discovery reads filesystem entries and EXIF headers only. It never decodes,
moves, publishes, or deletes source media. Dark classification is deliberately
conservative: only an explicitly named dark/calibration directory or filename
is treated as a likely dark; pixel darkness is not guessed from a header.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .astro_sequence import MAX_FRAME_GAP_S, MAX_HOLD_RATIO, MIN_SEQUENCE_FRAMES
from .develop import RAW_EXTENSIONS
from .exif_gps import (
    extract_exif_capture_metadata,
    extract_exif_capture_timestamp,
    extract_exif_dimensions,
)

_DARK_NAMES = frozenset({"dark", "darks", "darkframe", "darkframes", "calibration"})
_STILL_EXTENSIONS = frozenset({".tif", ".tiff", ".jpg", ".jpeg", ".png"})


class NightProduct(StrEnum):
    BEST_STILLS = "best_stills"
    TIMELAPSE = "timelapse"
    STAR_TRAILS = "star_trails"
    DEEP_STILL = "deep_still"


@dataclass(frozen=True)
class ProductResult:
    product: NightProduct
    path: Path
    receipt: Path

    def __post_init__(self) -> None:
        suffix = self.path.suffix.lower()
        if self.product is NightProduct.TIMELAPSE and suffix != ".mp4":
            raise ValueError("timelapse result must be an MP4 video")
        if self.product is not NightProduct.TIMELAPSE and suffix not in _STILL_EXTENSIONS:
            raise ValueError(f"{self.product.value} result must be a supported still image")
        if self.receipt.suffix.lower() != ".json":
            raise ValueError("product result receipt must be JSON")


@dataclass(frozen=True)
class ProductRequest:
    products: tuple[NightProduct, ...]

    @classmethod
    def create(cls, products) -> ProductRequest:
        converted: list[NightProduct] = []
        try:
            for value in products:
                product = NightProduct(value)
                if product not in converted:
                    converted.append(product)
        except ValueError as exc:
            supported = ", ".join(product.value for product in NightProduct)
            raise ValueError(f"requested product is unsupported; supported: {supported}") from exc
        if not converted:
            raise ValueError("at least one requested product is required")
        return cls(tuple(converted))

    def missing(self, results) -> tuple[NightProduct, ...]:
        delivered = {result.product for result in results}
        return tuple(product for product in self.products if product not in delivered)

    def complete(self, results) -> bool:
        return not self.missing(results)


@dataclass(frozen=True)
class NightFrame:
    path: Path
    captured_at: dt.datetime
    iso: float
    exposure_s: float
    capture_utc: dt.datetime | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class NightSequence:
    frames: tuple[NightFrame, ...]
    break_before: dict | None = None

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def started_at(self) -> dt.datetime:
        return self.frames[0].captured_at

    @property
    def ended_at(self) -> dt.datetime:
        return self.frames[-1].captured_at

    @property
    def timezone_confidence(self) -> str:
        known = sum(frame.capture_utc is not None for frame in self.frames)
        if known == len(self.frames):
            return "recorded_offset"
        return "partial_offset" if known else "camera_clock_only"

    @property
    def geometry(self) -> dict:
        first = self.frames[0]
        return {
            "width": first.width,
            "height": first.height,
            "confidence": "recorded" if first.width is not None else "unknown",
        }


@dataclass(frozen=True)
class NightCollection:
    root: Path
    lights: tuple[NightFrame, ...]
    likely_darks: tuple[Path, ...]
    unreadable: tuple[Path, ...]
    sequences: tuple[NightSequence, ...]
    existing_linear: tuple[Path, ...]
    invalid_linear: tuple[Path, ...]
    refused: tuple[Path, ...]
    unsequenced: tuple[dict, ...] = ()

    @property
    def light_count(self) -> int:
        return len(self.lights)

    @property
    def likely_dark_count(self) -> int:
        return len(self.likely_darks)

    @property
    def capture_span_s(self) -> float | None:
        if not self.lights:
            return None
        return (self.lights[-1].captured_at - self.lights[0].captured_at).total_seconds()

    def as_dict(self) -> dict:
        """Stable JSON-ready summary; paths remain local and no pixels are read."""
        return {
            "root": str(self.root),
            "light_count": self.light_count,
            "likely_dark_count": self.likely_dark_count,
            "capture_span_s": self.capture_span_s,
            "unreadable": [str(path) for path in self.unreadable],
            "likely_darks": [str(path) for path in self.likely_darks],
            "existing_linear": [str(path) for path in self.existing_linear],
            "invalid_linear": [str(path) for path in self.invalid_linear],
            "refused": [str(path) for path in self.refused],
            "unsequenced": list(self.unsequenced),
            "sequences": [
                {
                    "frame_count": sequence.frame_count,
                    "started_at": sequence.started_at.isoformat(),
                    "ended_at": sequence.ended_at.isoformat(),
                    "iso": sequence.frames[0].iso,
                    "exposure_s": sequence.frames[0].exposure_s,
                    "timezone_confidence": sequence.timezone_confidence,
                    "geometry": sequence.geometry,
                    "break_before": sequence.break_before,
                    "frames": [str(frame.path) for frame in sequence.frames],
                }
                for sequence in self.sequences
            ],
        }


def _likely_dark(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    words = {part.lower().replace("_", "").replace("-", "") for part in relative.parts[:-1]}
    stem = relative.stem.lower().replace("_", "").replace("-", "")
    return bool(words & _DARK_NAMES) or stem.startswith("dark")


def _capture_gap(previous: NightFrame, current: NightFrame) -> float:
    if previous.capture_utc is not None and current.capture_utc is not None:
        return (current.capture_utc - previous.capture_utc).total_seconds()
    return (current.captured_at - previous.captured_at).total_seconds()


def _break_reason(previous: NightFrame, current: NightFrame) -> dict:
    gap = _capture_gap(previous, current)
    if gap < 0:
        return {"kind": "clock_reset", "gap_s": gap}
    if gap > MAX_FRAME_GAP_S:
        return {"kind": "capture_gap", "gap_s": gap}
    previous_geometry = (previous.width, previous.height)
    current_geometry = (current.width, current.height)
    if current_geometry != previous_geometry:
        if None in previous_geometry or None in current_geometry:
            return {
                "kind": "geometry_availability_change",
                "from": list(previous_geometry),
                "to": list(current_geometry),
            }
        return {
            "kind": "geometry_change",
            "from": list(previous_geometry),
            "to": list(current_geometry),
        }
    if current.iso != previous.iso:
        return {"kind": "iso_change", "from": previous.iso, "to": current.iso}
    return {
        "kind": "exposure_change",
        "from_s": previous.exposure_s,
        "to_s": current.exposure_s,
    }


def _holds(previous: NightFrame, current: NightFrame) -> bool:
    gap = _capture_gap(previous, current)
    pair = (current.exposure_s, previous.exposure_s)
    return (
        0 <= gap <= MAX_FRAME_GAP_S
        and (current.width, current.height) == (previous.width, previous.height)
        and current.iso == previous.iso
        and max(pair) / min(pair) < MAX_HOLD_RATIO
    )


def _omitted_run(
    run: list[NightFrame], boundary_before: dict | None, boundary_after: dict | None
) -> dict:
    return {
        "frames": [str(frame.path) for frame in run],
        "frame_count": len(run),
        "reason": "fewer_than_minimum_compatible_frames",
        "boundary_before": boundary_before,
        "boundary_after": boundary_after,
    }


def _group(frames: list[NightFrame]) -> tuple[tuple[NightSequence, ...], tuple[dict, ...]]:
    sequences: list[NightSequence] = []
    unsequenced: list[dict] = []
    run: list[NightFrame] = []
    break_before = None
    for frame in sorted(frames, key=lambda item: (item.captured_at, str(item.path))):
        if run and not _holds(run[-1], frame):
            boundary = _break_reason(run[-1], frame)
            if len(run) >= MIN_SEQUENCE_FRAMES:
                sequences.append(NightSequence(tuple(run), break_before))
            else:
                unsequenced.append(_omitted_run(run, break_before, boundary))
            break_before = boundary
            run = []
        run.append(frame)
    if len(run) >= MIN_SEQUENCE_FRAMES:
        sequences.append(NightSequence(tuple(run), break_before))
    elif run:
        unsequenced.append(_omitted_run(run, break_before, None))
    return tuple(sequences), tuple(unsequenced)


def _walk(root: Path, *, max_entries: int):
    count = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        safe_dirs = []
        for name in sorted(dirs):
            path = base / name
            count += 1
            if count > max_entries:
                raise ValueError(f"collection exceeds {max_entries} entries")
            if path.is_symlink():
                yield "refused", path
            else:
                safe_dirs.append(name)
        dirs[:] = safe_dirs
        for name in sorted(files):
            path = base / name
            count += 1
            if count > max_entries:
                raise ValueError(f"collection exceeds {max_entries} entries")
            if path.is_symlink() or not path.is_file():
                yield "refused", path
            else:
                yield "file", path


def scan_collection(root: Path, *, max_entries: int = 20_000) -> NightCollection:
    """Inspect one source tree without decoding any image payload."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("night collection root must be a real directory")
    root = root.resolve(strict=True)
    lights: list[NightFrame] = []
    darks: list[Path] = []
    unreadable: list[Path] = []
    existing: list[Path] = []
    invalid: list[Path] = []
    refused: list[Path] = []
    seen_linear: set[Path] = set()

    for kind, path in _walk(root, max_entries=max_entries):
        if kind == "refused":
            refused.append(path)
            continue
        if path.name == "manifest.json" and path.parent.suffix == ".linear":
            seen_linear.add(path.parent)
            try:
                manifest = json.loads(path.read_text())
                if not isinstance(manifest, dict) or "schema" not in manifest:
                    raise ValueError
            except (OSError, ValueError, TypeError):
                invalid.append(path.parent)
            else:
                existing.append(path.parent)
            continue
        if path.suffix.lower() not in RAW_EXTENSIONS:
            continue
        if _likely_dark(path, root):
            darks.append(path)
            continue
        captured_at = extract_exif_capture_timestamp(path)
        metadata = extract_exif_capture_metadata(path)
        dimensions = extract_exif_dimensions(path)
        if captured_at is None or not metadata.iso or not metadata.exposure_s:
            unreadable.append(path)
            continue
        lights.append(
            NightFrame(
                path,
                captured_at,
                float(metadata.iso),
                float(metadata.exposure_s),
                getattr(metadata, "capture_utc", None),
                *(dimensions or (None, None)),
            )
        )

    # A .linear directory without a regular manifest is visible as invalid too.
    for path in sorted(root.rglob("*.linear")):
        if not path.is_symlink() and path.is_dir() and path not in seen_linear:
            invalid.append(path)
    lights.sort(key=lambda item: (item.captured_at, str(item.path)))
    sequences, unsequenced = _group(lights)
    return NightCollection(
        root=root,
        lights=tuple(lights),
        likely_darks=tuple(sorted(darks)),
        unreadable=tuple(sorted(unreadable)),
        sequences=sequences,
        existing_linear=tuple(sorted(existing)),
        invalid_linear=tuple(sorted(set(invalid))),
        refused=tuple(sorted(refused)),
        unsequenced=unsequenced,
    )
