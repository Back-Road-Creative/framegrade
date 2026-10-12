"""Offline astro benchmark. Forks prepare_frames + stack_run in a CHILD PROCESS that
self-measures its own peak RSS (per-run, not RUSAGE_CHILDREN's cross-call max). The child
resets its kernel high-water mark first, so nothing it held before its run (fork, runtime
bootstrap) can floor the figure; ``ru_maxrss`` is the labelled fallback if it cannot."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import resource
import time
from pathlib import Path

import cv2
import numpy as np

from ... import config as _config_mod
from . import astro_orchestrator as _orch
from .astro_sidecar import coverage_payload
from .preflight import estimate_astro_stack_ram_gb

BENCHMARK_SCHEMA_VERSION = "1"
_RU_MAXRSS_KIB_TO_BYTES = 1024  # ru_maxrss is KiB on Linux (getrusage(2)); Linux-only here.
_HASH_CHUNK_BYTES = 1024 * 1024
_CLEAR_REFS = "/proc/self/clear_refs"  # writing "5" resets VmHWM to the current RSS (proc(5))
_PROC_STATUS = "/proc/self/status"
_RSS_NOTE_RESET = "VmHWM (KiB) reset at child start; see ru_maxrss_to_bytes"
_RSS_NOTE_UNRESET = (
    "ru_maxrss (KiB), peak NOT reset at child start (clear_refs/VmHWM unavailable): "
    "may include memory held before the run"
)
PANEL_PX = 512  # 100% center-crop side length AND contact-sheet longest side, downscaled

UNAVAILABLE_PANELS = {  # panels the pipeline can't produce yet -- honest, never faked
    "background_model": "astro_develop returns only before/after scalars, not the map",
    # astro_stack now computes real per-pixel coverage/thin_coverage (surfaced in
    # payload["stack"]["coverage"] below) once the caller supplies sky_validity --
    # but nothing here renders it as an image panel yet.
    "coverage": "StackResult.coverage/thin_coverage are numeric only, not rendered as a panel",
    "rejection": "astro_stack's sigma-clip mask is discarded after mean_rejected_fraction",
    "sky_mask": "stack_run's horizon_mask is local to the call, never returned",
}


class SourceDirectoryWriteError(ValueError):
    """Refused: output_dir is inside a light/dark source directory."""


def ru_maxrss_to_bytes(ru_maxrss_kib) -> int:
    return int(ru_maxrss_kib) * _RU_MAXRSS_KIB_TO_BYTES


def _reset_peak_rss() -> bool:
    """Reset this process's RSS high-water mark to its current RSS; False if the kernel refuses."""
    try:
        with open(_CLEAR_REFS, "w") as fh:
            fh.write("5")
    except OSError:
        return False
    return True


def _read_vmhwm_kib() -> int | None:
    try:
        with open(_PROC_STATUS) as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _measure_peak_rss_kib(reset_ok: bool) -> tuple[int, str]:
    """(peak KiB, source). A figure is trusted as per-run only when the reset took AND VmHWM
    read back; otherwise the un-reset ru_maxrss is returned, labelled as such."""
    vmhwm = _read_vmhwm_kib() if reset_ok else None
    if vmhwm is not None:
        return vmhwm, "vmhwm"
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "ru_maxrss"


def _hash_file(path: Path) -> str:  # hashes bytes, never copies the file
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_output_not_in_source(light_paths, dark_paths, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    for source_path in (*light_paths, *dark_paths):
        source_dir = Path(source_path).resolve().parent
        if output_dir.is_relative_to(source_dir):
            raise SourceDirectoryWriteError(f"{output_dir} is inside source dir {source_dir}")


def _pinned_get_config(real_get_config, output_dir: Path):
    def _pinned(config_path=None):
        cfg = real_get_config(config_path)
        cfg.photos.he_dng_cache_dir = str(output_dir)
        return cfg

    return _pinned


def _panel_uint8(arr: np.ndarray, crop: int | None = None) -> np.ndarray:
    arr = np.asarray(arr)
    if crop:
        h, w = arr.shape[:2]
        s = min(crop, h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        arr = arr[y0 : y0 + s, x0 : x0 + s]
    if arr.dtype == np.uint16:
        return (arr / 257).astype(np.uint8)
    arr = np.nan_to_num(arr.astype(np.float32))
    lo, hi = np.percentile(arr, (0.5, 99.5))
    hi = max(hi, lo + 1e-6)
    return (np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)


def _write_panels(report, output_dir: Path) -> dict:
    full = _panel_uint8(report.develop.rgb_uint16)
    h, w = full.shape[:2]
    scale = PANEL_PX / max(h, w, 1)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    panels = {
        "linear_stack_crop": _panel_uint8(report.stack.rgb, crop=PANEL_PX),
        "final_render_crop": _panel_uint8(report.develop.rgb_uint16, crop=PANEL_PX),
        "contact_sheet": cv2.resize(full, size),
    }
    for name, arr in panels.items():
        cv2.imwrite(str(output_dir / f"panel_{name}.png"), arr[..., ::-1])
    return {name: str(output_dir / f"panel_{name}.png") for name in panels}


def _child_worker(light_paths, dark_paths, out_path, horizon_frac, meteors, queue) -> None:
    reset_ok = _reset_peak_rss()  # first: drop any peak held from before this run
    try:
        t0 = time.monotonic()
        prepared = _orch.prepare_frames(light_paths, dark_paths=dark_paths)
        t1 = time.monotonic()
        report = _orch.stack_run(prepared, out_path, horizon_frac=horizon_frac, meteors=meteors)
        t2 = time.monotonic()
        h, w = prepared.frames[0].rgb.shape[:2]
        payload = {
            "ok": True,
            "frames": {
                "decoded_count": len(prepared.frames),
                "failed_count": len(prepared.failed),
                "megapixels": (h * w) / 1_000_000,
            },
            "stack": {
                "n_frames": int(report.stack.n_frames),
                "mean_rejected_fraction": float(report.stack.mean_rejected_fraction),
                "meteors_reinserted": int(report.meteors_reinserted),
                "coverage": coverage_payload(report.stack.coverage),
            },
            "image_path": str(report.image_path),
            "sidecar_path": str(report.sidecar_path) if report.sidecar_path else None,
            "panels": _write_panels(report, Path(out_path).parent),
            "wall_clock_s": {"prepare": t1 - t0, "stack_run": t2 - t1},
        }
    except Exception as exc:  # the queue is the only channel back to the parent
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    payload["rss_self_kib"], payload["rss_source"] = _measure_peak_rss_kib(reset_ok)
    queue.put(payload)


def run_benchmark(
    light_paths, output_dir, *, dark_paths=(), horizon_frac=None, meteors=None, mp_context=None
) -> dict:
    light_paths = [Path(p) for p in light_paths]
    dark_paths = [Path(p) for p in dark_paths]
    output_dir = Path(output_dir)
    _check_output_not_in_source(light_paths, dark_paths, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    light_hashes = {str(p): _hash_file(p) for p in light_paths}
    dark_hashes = {str(p): _hash_file(p) for p in dark_paths}
    capture = _orch.extract_exif_capture_metadata(light_paths[0]) if light_paths else None

    real_get_config = _config_mod.get_config
    _config_mod.get_config = _pinned_get_config(real_get_config, output_dir)  # inherited by fork
    ctx = mp_context or multiprocessing.get_context("fork")
    queue = ctx.Queue()
    stack_out_path = output_dir / "benchmark_stack.tiff"
    proc = ctx.Process(
        target=_child_worker,
        args=(light_paths, dark_paths, stack_out_path, horizon_frac, meteors, queue),
    )
    proc.start()
    _config_mod.get_config = real_get_config  # restore -- the child already forked its copy
    result = queue.get()  # read before join -- a full pipe would otherwise deadlock both
    proc.join()

    measured_peak_bytes = ru_maxrss_to_bytes(result["rss_self_kib"])
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "ok": result["ok"],
        "inputs": {
            "light_hashes": light_hashes,
            "dark_hashes": dark_hashes,
            "light_count": len(light_paths),
            "dark_count": len(dark_paths),
        },
        "capture_identity": {
            "iso": capture.iso if capture else None,
            "exposure_s": capture.exposure_s if capture else None,
            "camera_model": None,  # unmeasured: no Make/Model tag exposed today
            "focal_length_mm": None,  # unmeasured: no FocalLength tag exposed today
        },
        "rss": {
            "measured_peak_bytes": measured_peak_bytes,
            "unit_note": _RSS_NOTE_RESET if result["rss_source"] == "vmhwm" else _RSS_NOTE_UNRESET,
        },
        "unavailable_panels": UNAVAILABLE_PANELS,
    }
    if not result["ok"]:
        payload["error"] = result["error"]
        (output_dir / "benchmark.json").write_text(json.dumps(payload, indent=2))
        return payload

    # Shape the estimate from THIS run, never the estimator's defaults: an un-passed
    # dark_count biases ratio_estimate_over_measured low by exactly the dark term, and
    # this module exists to calibrate those constants against the measured figure.
    # meteors_reinserted is the measured count, not the requested posture -- stack_run
    # allocates a meteor layer only when it composites one.
    frame_count, mp = result["frames"]["decoded_count"], result["frames"]["megapixels"]
    estimate_gb = estimate_astro_stack_ram_gb(
        frame_count,
        mp,
        dark_count=len(dark_paths),
        meteor_reinsertion_enabled=bool(result["stack"]["meteors_reinserted"]),
    )
    estimate_bytes = int(estimate_gb * 1024**3)
    payload["rss"]["estimate_bytes"] = estimate_bytes
    payload["rss"]["ratio_estimate_over_measured"] = (
        estimate_bytes / measured_peak_bytes if measured_peak_bytes else None
    )
    payload["frames"] = result["frames"]
    payload["stack"] = result["stack"]
    payload["wall_clock_s"] = result["wall_clock_s"]
    artifacts = {"image": result["image_path"], **result["panels"]}
    if result["sidecar_path"]:
        artifacts["sidecar"] = result["sidecar_path"]
    payload["output_artifacts"] = {
        name: {"path": path, "sha256": _hash_file(Path(path))} for name, path in artifacts.items()
    }
    (output_dir / "benchmark.json").write_text(json.dumps(payload, indent=2))
    return payload
