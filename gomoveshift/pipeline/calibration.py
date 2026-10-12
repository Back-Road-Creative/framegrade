"""Measure the day-master chain on THIS machine and keep the rate for preflight.

The preflight ETA used to come only from a fixed model calibrated once, on one
box (``preflight_plan._RES_FACTOR`` and friends). On any other
machine that number is a guess. Here a short slice of the operator's own
footage goes through the REAL chain (grade + OST + audio + optional blur
composite + encode, via ``SinglePassProcessor.process``) once; the measured
wall-seconds-per-source-second is saved to ``calibration.json`` under
``gms_intermediate_base()`` and preflight prefers it over the model. A wrong
model cannot be printed when a measurement exists.

A record carries ``chain_version`` (= ``MASTER_ENCODE_ALGO_VERSION``, a
fingerprint of the frozen encode constants): changing an encode constant
invalidates the measurement by construction, so a stale rate is never used.

Scope: one backend today (CPU filters + ``hevc_nvenc``, the only supported
encode path -- see ``single_pass.process``). Other backends live in
``encoders.LADDER``; calibrating across them is not built yet.

Calibration is advisory: :func:`ensure_calibrated` never blocks a run.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ..paths import gms_intermediate_base
from . import encoders

logger = logging.getLogger(__name__)

#: Seconds of source pushed through the chain; long enough to amortise start-up.
DEFAULT_SLICE_SECONDS = 15.0
#: The only encoder the pipeline supports today (single_pass.process: NVENC only).
_ENCODER = encoders.NVENC.name
_FFMPEG_TIMEOUT_S = 600


def _current_chain_version() -> str:
    from .day_master_encoder import MASTER_ENCODE_ALGO_VERSION

    return str(MASTER_ENCODE_ALGO_VERSION)


@dataclass(frozen=True)
class CalibrationRecord:
    measured_at: str
    chain_version: str
    encoder: str
    width: int
    height: int
    source_fps: float
    slice_seconds: float
    wall_seconds: float
    blur: bool
    ffmpeg_version: str
    cpu_count: int

    @property
    def realtime_factor(self) -> float:
        """Wall-seconds spent per second of source."""
        return self.wall_seconds / self.slice_seconds

    @property
    def chain_fps(self) -> float:
        """Output frames per wall-second the chain sustained."""
        return self.source_fps * self.slice_seconds / self.wall_seconds


def calibration_path() -> Path:
    return gms_intermediate_base() / "calibration.json"


def load_calibration(path: Optional[Path] = None) -> Optional[CalibrationRecord]:
    """The saved record, or ``None`` (missing / unreadable / wrong shape / stale)."""
    path = Path(path) if path is not None else calibration_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rec = CalibrationRecord(**data)
        if rec.slice_seconds <= 0 or rec.wall_seconds <= 0:
            raise ValueError("non-positive timing")
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("calibration record %s unusable (%s); ignoring it", path, exc)
        return None
    current = _current_chain_version()
    if rec.chain_version != current:
        logger.warning(
            "calibration record %s is stale: measured on chain %s, current chain is %s; "
            "ignoring it",
            path,
            rec.chain_version,
            current,
        )
        return None
    return rec


def save_calibration(record: CalibrationRecord, path: Optional[Path] = None) -> Path:
    """Atomically write ``record`` (tmp + ``os.replace``)."""
    path = Path(path) if path is not None else calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(asdict(record), fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT_S)
    if result.returncode != 0:
        tail = (result.stderr or "")[-600:]
        raise RuntimeError(f"calibration ffmpeg step failed (rc={result.returncode}): {tail}")


def _cut_slice(source: Path, slice_path: Path, seconds: float) -> None:
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            "0",
            "-t",
            f"{seconds:g}",
            "-i",
            str(source),
            "-c",
            "copy",
            "-map",
            "0:v",
            "-map",
            "0:a?",
            str(slice_path),
        ]  # fmt: skip
    )


def _make_mask(mask_path: Path, width: int, height: int, fps: float, seconds: float) -> None:
    """Synthetic privacy mask: black frame, two white boxes (~5 % of the area).

    Same format the composite consumes and ``redactcam.mask.materialize_mask_video``
    writes: lossless gray FFV1 in matroska; ``single_pass._build_blur_filter_complex``
    reads it as ``[1:v]format=gray``.
    """
    bw, bh = max(2, int(width * 0.16)), max(2, int(height * 0.16))
    boxes = (
        f"drawbox=x={int(width * 0.10)}:y={int(height * 0.10)}:w={bw}:h={bh}:color=white:t=fill,"
        f"drawbox=x={int(width * 0.60)}:y={int(height * 0.60)}:w={bw}:h={bh}:color=white:t=fill"
    )
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=black:size={width}x{height}:rate={fps:g}:duration={seconds:g}",
            "-vf",
            boxes,
            "-pix_fmt",
            "gray",
            "-c:v",
            "ffv1",
            str(mask_path),
        ]  # fmt: skip
    )


def _ffmpeg_version() -> str:
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=15)
        return (out.stdout or "").splitlines()[0].strip() if out.stdout else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def calibrate_master_chain(
    config: Any,
    source: Path,
    *,
    seconds: float = DEFAULT_SLICE_SECONDS,
    blur: bool,
    workdir: Path,
    run_chain: Optional[Callable[[Path, Path, Optional[Path]], Any]] = None,
    probe: Optional[Callable[[Path], dict]] = None,
    cut_slice: Optional[Callable[[Path, Path, float], None]] = None,
    make_mask: Optional[Callable[[Path, int, int, float, float], None]] = None,
) -> CalibrationRecord:
    """Time the real day-master chain on the first ``seconds`` of ``source``.

    Raises ``RuntimeError`` (with the ffmpeg stderr tail) when a step fails;
    scratch files are removed either way.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    slice_path = workdir / "calib_slice.mp4"
    out_path = workdir / "calib_out.mp4"
    mask_path = workdir / "calib_mask.mkv"
    cut = cut_slice or _cut_slice
    mask_maker = make_mask or _make_mask

    processor = None
    if run_chain is None or probe is None:
        from .single_pass import SinglePassProcessor

        processor = SinglePassProcessor(config)

    if run_chain is None:

        def run_chain(slc: Path, out: Path, mask: Optional[Path]) -> Any:
            result = processor.process(slc, out, location="Calibration", blur_mask_path=mask)
            if not getattr(result, "success", False):
                raise RuntimeError("calibration encode reported failure")
            return result

    if probe is None:
        probe = processor._get_video_info

    try:
        cut(Path(source), slice_path, seconds)
        info = probe(slice_path)
        slice_seconds = float(info.get("duration") or 0) or float(seconds)
        width, height, fps = int(info["width"]), int(info["height"]), float(info["fps"])
        mask: Optional[Path] = None
        if blur:
            mask_maker(mask_path, width, height, fps, slice_seconds)
            mask = mask_path
        started = time.monotonic()
        run_chain(slice_path, out_path, mask)
        wall = max(time.monotonic() - started, 1e-6)
        return CalibrationRecord(
            measured_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            chain_version=_current_chain_version(),
            encoder=_ENCODER,
            width=width,
            height=height,
            source_fps=fps,
            slice_seconds=slice_seconds,
            wall_seconds=wall,
            blur=bool(blur),
            ffmpeg_version=_ffmpeg_version(),
            cpu_count=os.cpu_count() or 1,
        )
    finally:
        for leftover in (slice_path, out_path, mask_path):
            try:
                leftover.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.debug("calibration: could not remove %s: %s", leftover, exc)


def ensure_calibrated(
    config: Any,
    source_files: list[Path],
    *,
    blur: bool,
    workdir: Path,
    calibrate: Optional[Callable[..., CalibrationRecord]] = None,
    load: Optional[Callable[[], Optional[CalibrationRecord]]] = None,
    save: Optional[Callable[[CalibrationRecord], Any]] = None,
) -> Optional[CalibrationRecord]:
    """A valid record, measuring one now if needed. Advisory: never raises.

    ``calibrate`` / ``load`` / ``save`` default to this module's functions,
    resolved at call time so tests can substitute them.
    """
    calibrate = calibrate or calibrate_master_chain
    load = load or load_calibration
    save = save or save_calibration
    existing = load()
    if existing is not None or not source_files:
        return existing
    source = source_files[0]
    try:
        logger.info(
            "Calibrating the day-master chain on this machine (%d s slice of %s)...",
            int(DEFAULT_SLICE_SECONDS),
            source.name,
        )
        record = calibrate(config, source, blur=blur, workdir=workdir)
        save(record)
        return record
    except Exception as exc:  # noqa: BLE001 - advisory, must never block a run
        logger.warning("calibration failed (%s); ETA falls back to the reference model", exc)
        return None
