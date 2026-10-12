"""Star trails: the sky's rotation drawn as ARCS, the opposite of a stack.

Registration is what REMOVES trailing, so this path never registers and never averages: it
folds each frame into a running per-pixel MAXIMUM, so every frame contributes its stars where
they stood at that moment and the arcs build up. That fold is STREAMING, this path's biggest
advantage -- it holds only the accumulator plus the frame being folded in, so peak memory is
O(1) in frame count. ``estimate_astro_stack_ram_gb`` charges the stack path ~0.583 GB/frame
over ~11.4 GB fixed at a ~20.9 MP body (18.35 GB for a 15-frame chunk), which
is why THAT path chunks a night; frames reach this one as an ITERATOR, never a list.

Dark + hot-pixel calibration matters more here than anywhere else in the pipeline: a maximum
keeps the brightest value each pixel ever took, so one hot photosite in one frame is a
permanent bright dot and defects accumulate instead of averaging away. The correction is
``astro_calibrate``'s own, through the same decode + HE->DNG seam the stack path uses.

Arcs come out DASHED, and that is honest output: the unrecorded gap per cycle is ``interval
- exposure`` -- 17 s of every 32 s on a measured capture (15 s exposures at ISO
6400, 18 mm, on a realised 32 s interval across 29 of 29 measured intervals, a 16 s
intervalometer setting whose every alternate trigger the 15 s exposure plus HE raw write
misses). Nothing here interpolates or blurs it away; continuous arcs come from an interval
matching its exposure, never invented pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..preflight import check_disk_space, check_ram
from . import preflight as pf  # the astro RAM constants, read here so they cannot drift
from .astro_calibrate import hot_pixel_map
from .astro_develop import develop_stack
from .astro_orchestrator import (
    AstroOrchestratorError,
    build_validated_master_dark,
    decode_with_he_fallback,
)

MIN_TRAIL_FRAMES = 2  # below two, a "maximum across the night" is one frame and no arc
# Live float32 RGB frame-equivalents at the blend's peak, frame-count-INDEPENDENT by
# construction: accumulator, frame being folded in, decode transient -- 12 bytes/px each.
_TRAIL_LIVE_FRAME_EQUIVALENTS = 3
_TRAIL_BYTES_PER_MP = 3 * 4 * 1_000_000


class TrailInputError(AstroOrchestratorError):
    """Too few frames, or frames of mixed shape, handed to the maximum blend -- the same
    "this run has no usable frames" family the orchestrator raises, so callers catch one."""


def estimate_trail_ram_gb(megapixels: float, *, dark_count: int = 0) -> float:
    """Estimated peak RAM (GiB) for a star-trail render of ANY length -- deliberately no
    ``frame_count`` parameter, since a streaming maximum's peak has none. Terms: the live
    frame-equivalents above, the frame-count-independent decode/develop terms
    ``estimate_astro_stack_ram_gb`` charges (both run here unchanged), plus the darks."""
    per_mp = pf._ASTRO_DEVELOP_GIB_PER_MP + pf._ASTRO_RAM_DECODE_OVERHEAD_GIB_PER_MP
    live_bytes = megapixels * _TRAIL_BYTES_PER_MP * _TRAIL_LIVE_FRAME_EQUIVALENTS
    dark_bytes = dark_count * megapixels * 1_000_000 * pf._ASTRO_DARK_BYTES_PER_MP_PER_DARK
    return (live_bytes + dark_bytes) / pf._BYTES_PER_GIB + megapixels * per_mp


def preflight_trail(output_dir, *, megapixels: float, dark_count: int = 0) -> None:
    """Disk/RAM gate before the first decode, sized to what THIS path actually holds."""
    check_disk_space(Path(output_dir), min_free_gb=pf._ASTRO_MIN_FREE_GB)
    check_ram(min_free_gb=estimate_trail_ram_gb(megapixels, dark_count=dark_count))


def accumulate_max(frames) -> tuple[np.ndarray, int]:
    """``(per-pixel maximum, frame count)`` over ``frames``, folded in ONE AT A TIME: any
    iterable (a generator is the shipped caller), never listed, stacked or indexed, so only
    the accumulator and the frame being folded in are ever resident."""
    accumulator = None
    count = 0
    for frame in frames:
        array = np.asarray(frame, dtype=np.float32)
        if accumulator is None:
            accumulator = array.copy()
        elif array.shape != accumulator.shape:
            raise TrailInputError(
                f"accumulate_max requires one shape, got {accumulator.shape} then {array.shape}"
            )
        else:
            np.maximum(accumulator, array, out=accumulator)
        count += 1
    if count < MIN_TRAIL_FRAMES:
        raise TrailInputError(f"accumulate_max needs >= {MIN_TRAIL_FRAMES} frames, got {count}")
    return accumulator, count


def render_trail(light_paths, out_path, dark_paths=(), sky_mask=None, progress=None):
    """One night's lights -> one star-trail master, as ``(image_path, frames_used,
    rejected)``: calibrated, UNALIGNED, maximum-blended, then developed and exported through
    the stack path's own chain. ``dark_paths`` builds the master dark + hot-pixel map exactly
    as ``prepare_frames`` does -- through the same ``build_validated_master_dark`` seam, so a
    dark that can't be proven to match the lights refuses this render too (a maximum keeps an
    under-subtracted pixel forever, so a mismatched dark is even less recoverable here than in
    a mean); omitting it leaves hot photosites to burn in permanently."""
    lights = tuple(map(Path, light_paths))
    master_dark = hotmap = None
    if dark_paths:
        master_dark = build_validated_master_dark(lights, dark_paths)
        hotmap = hot_pixel_map(master_dark)
    rejected: list[tuple[Path, str]] = []

    def _frames():  # one at a time, never a list -- the memory argument above
        for completed, light_path in enumerate(lights, start=1):
            try:
                decoded, reason = decode_with_he_fallback(light_path, master_dark, hotmap)
                if decoded is None:
                    rejected.append((light_path, reason))
                    continue
                yield decoded.rgb
            finally:
                if progress is not None:
                    progress(completed, len(lights))

    trail, n_frames = accumulate_max(_frames())
    developed = develop_stack(trail, sky_mask=sky_mask)

    from .develop import export_tiff16  # lazy: no develop<->astro import cycle

    return export_tiff16(developed.rgb_uint16, out_path), n_frames, tuple(rejected)
