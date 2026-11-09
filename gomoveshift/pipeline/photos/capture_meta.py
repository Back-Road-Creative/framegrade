"""Camera facts read with exiftool: body serial, shutter count, focus distance.

Pillow (``exif_gps``) cannot read the Nikon MakerNote, where the body serial and
``ShutterCount`` live; exiftool can. Paths travel to exiftool in a temp argfile
(one per line), never on the command line, and never through a shell. A missing
exiftool, a timed-out batch or a bad value degrades to ``None`` fields plus a
warning; a per-file problem never raises.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MISSING_WARNING = "exiftool not found: same-shot dedupe and close-up detection disabled"

_TAGS = (
    "-Make",
    "-Model",
    "-SerialNumber",
    "-InternalSerialNumber",
    "-ShutterCount",
    "-DateTimeOriginal",
    "-SubSecTimeOriginal",
    "-FocusDistance",
    "-LensModel",
    "-Lens",
    "-LensID",
    "-FocalLength",
    "-FNumber",
    "-ISO",
    "-ExposureTime",
)


@dataclass(frozen=True)
class CaptureRecord:
    path: Path
    make: Optional[str] = None
    model: Optional[str] = None
    serial: Optional[str] = None
    shutter_count: Optional[int] = None
    datetime_original: Optional[str] = None
    subsec: Optional[str] = None
    focus_distance_m: Optional[float] = None
    lens_model: Optional[str] = None
    focal_length_mm: Optional[float] = None
    f_number: Optional[float] = None
    iso: Optional[float] = None
    exposure_s: Optional[float] = None


def exiftool_path() -> Optional[str]:
    return shutil.which("exiftool")


def _text(value) -> Optional[str]:
    if value is None or isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    return text or None


def _number(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _count(value) -> Optional[int]:
    number = _number(value)
    return int(number) if number is not None and number >= 0 else None


def _distance(value) -> Optional[float]:
    if isinstance(value, str) and value.strip().lower() in ("inf", "infinity"):
        return math.inf
    number = _number(value)
    return number if number is not None and number >= 0 else None


def _first_text(row: dict, *names: str) -> Optional[str]:
    for name in names:
        text = _text(row.get(name))
        if text:
            return text
    return None


def _record_from_row(path: Path, row: dict) -> CaptureRecord:
    return CaptureRecord(
        path=path,
        make=_text(row.get("Make")),
        model=_text(row.get("Model")),
        serial=_first_text(row, "SerialNumber", "InternalSerialNumber"),
        shutter_count=_count(row.get("ShutterCount")),
        datetime_original=_text(row.get("DateTimeOriginal")),
        subsec=_text(row.get("SubSecTimeOriginal")),
        focus_distance_m=_distance(row.get("FocusDistance")),
        lens_model=_first_text(row, "LensModel", "Lens", "LensID"),
        focal_length_mm=_number(row.get("FocalLength")),
        f_number=_number(row.get("FNumber")),
        iso=_number(row.get("ISO")),
        exposure_s=_number(row.get("ExposureTime")),
    )


def _fallback_record(path: Path) -> CaptureRecord:
    """Pillow-only facts: iso, exposure, focal length, f-number."""
    from .exif_gps import (
        extract_exif_aperture,
        extract_exif_capture_metadata,
        extract_exif_focal_length,
    )

    exif = extract_exif_capture_metadata(path)
    return CaptureRecord(
        path=path,
        iso=exif.iso,
        exposure_s=exif.exposure_s,
        focal_length_mm=extract_exif_focal_length(path),
        f_number=extract_exif_aperture(path),
    )


def _run_batch(batch, exiftool, run, timeout_s):
    """Parsed exiftool rows for one batch; raises on timeout, exit or bad JSON."""
    fd, argfile = tempfile.mkstemp(prefix="exiftool-args-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(str(p) for p in batch) + "\n")
        argv = [
            exiftool,
            "-json",
            "-n",
            "-fast",
            "-charset",
            "filename=utf8",
            "-@",
            argfile,
            *_TAGS,
        ]
        proc = run(argv, capture_output=True, text=True, timeout=timeout_s, check=False)
    finally:
        os.unlink(argfile)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode}")
    rows = json.loads(proc.stdout or "[]")
    if not isinstance(rows, list):
        raise ValueError("exiftool output is not a list")
    return rows


def _resolve(path) -> Path:
    try:
        return Path(path).resolve()
    except OSError:
        return Path(path)


def read_capture_records(
    paths,
    *,
    exiftool: Optional[str] = ...,
    batch_size: int = 200,
    run=subprocess.run,
    timeout_s: float = 300.0,
) -> tuple[dict[Path, CaptureRecord], list[str]]:
    """Records for every input path, plus warnings."""
    paths = [Path(p) for p in paths]
    warnings: list[str] = []
    records: dict[Path, CaptureRecord] = {}
    if exiftool is ...:
        exiftool = exiftool_path()
    if exiftool is None:
        if paths:
            warnings.append(MISSING_WARNING)
        return {p: _fallback_record(p) for p in paths}, warnings
    size = max(1, batch_size)
    for index in range(0, len(paths), size):
        batch = paths[index : index + size]
        label = f"batch {index // size + 1} ({len(batch)} files)"
        try:
            rows = _run_batch(batch, exiftool, run, timeout_s)
        except (subprocess.TimeoutExpired, OSError, RuntimeError, ValueError) as exc:
            warnings.append(f"exiftool {label} failed: {exc}; Pillow fallback used")
            records.update({p: _fallback_record(p) for p in batch})
            continue
        by_source = {
            _resolve(row["SourceFile"]): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("SourceFile"), str)
        }
        for path in batch:
            row = by_source.get(_resolve(path))
            if row is None:
                warnings.append(f"exiftool returned nothing for {path}")
                records[path] = CaptureRecord(path=path)
            else:
                records[path] = _record_from_row(path, row)
    return records, warnings
