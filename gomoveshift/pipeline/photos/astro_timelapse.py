"""Night timelapse: an interval sequence as the sky MOVING -- one video frame per captured
frame, the third render a night run is for, beside a stack and trails.

ONE TONE FOR THE WHOLE SEQUENCE is the design, and why this is not a loop over
``develop_stack``: per-frame auto-stretch is what makes an amateur night timelapse strobe. The
MIDDLE frame develops normally and every frame renders through ``develop_stack_with_params``
against it, so gain, black point, peak normalisation and every stretch ``m`` are the
sequence's, not the frame's; gradient removal deliberately stays per-frame (that function's
contract). Frames reach the encoder one at a time and nothing pads or interpolates: 90 frames
at 24 fps is 3.75 s of video, said before the render. The middle frame, the per-frame
gradient's cost, the measured flicker, the frame rate, memory: docs/pipeline-internals.md.
"""

from pathlib import Path

import numpy as np

from ..preflight import check_disk_space, check_ram
from ..timelapse import TimelapseError as _SharedTimelapseError
from ..timelapse import stream_frames_to_video
from . import preflight as pf  # the astro RAM constants, read here so they cannot drift
from .astro_calibrate import build_master_dark, hot_pixel_map, read_raw_mosaic
from .astro_develop import develop_stack, develop_stack_with_params
from .astro_orchestrator import AstroOrchestratorError, decode_with_he_fallback

TIMELAPSE_FPS = 24  # not reel's 30: no in-frame motion to judder, and 25% more clip
TIMELAPSE_TIMEOUT_S = 1800  # bounds ffmpeg's exit after the last frame is piped, not the render

# Default output height cap. 4K rather than the sensor's own, because develop fits a gradient
# polynomial over EVERY pixel, so the render cost tracks pixel count and not much else:
# on a 20 MP body the uncapped render is roughly seven times slower per frame than at 2160
# -- an overnight-length job instead of a short one -- for a file no display can show. This is
# a CEILING (see ``_scale_to_height``), so a shorter body renders at its own height untouched;
# ``--timelapse-height 0`` asks for the full sensor back.
TIMELAPSE_HEIGHT_DEFAULT = 2160

# Live bytes/px at peak, frame-count-INDEPENDENT by construction: the frame's own linear RGB
# (float32, 12) + the reference master on the base result (6) + the frame just developed (6)
# + its rgb24 pipe copy (3), summed not maxed for headroom.
_TIMELAPSE_LIVE_BYTES_PER_PX = 27


class TimelapseError(AstroOrchestratorError):
    """No tone, too few frames, or a failed encode -- the orchestrator's own family."""


def preflight_timelapse(output_dir, *, megapixels: float, dark_count: int = 0) -> None:
    """Disk/RAM gate before the first decode, sized to what THIS path holds: the live bytes
    above (no ``frame_count`` term), plus the stack model's decode/develop terms and darks."""
    per_mp = pf._ASTRO_DEVELOP_GIB_PER_MP + pf._ASTRO_RAM_DECODE_OVERHEAD_GIB_PER_MP
    live = megapixels * 1_000_000 * _TIMELAPSE_LIVE_BYTES_PER_PX
    darks = dark_count * megapixels * 1_000_000 * pf._ASTRO_DARK_BYTES_PER_MP_PER_DARK
    check_disk_space(Path(output_dir), min_free_gb=pf._ASTRO_MIN_FREE_GB)
    check_ram(min_free_gb=(live + darks) / pf._BYTES_PER_GIB + megapixels * per_mp)


def _scale_to_height(array, scale_height, *, interpolation=None):
    """``array`` reduced so its height is ``scale_height``, aspect preserved -- or returned
    untouched when no cap is asked for or the source is already that short.

    A CEILING, never a target: a request taller than the frame is a no-op rather than an
    enlargement (pipeline/quality.py HONEST_RESOLUTION_RULE: never claim a native resolution
    the source lacked). The default
    ``INTER_AREA`` is the correct reduction filter for pixels -- it averages the source pixels
    a destination pixel covers, which keeps a downscaled star field from aliasing into a
    different star field. A boolean mask must override it (see the caller): averaging a mask
    yields fractions that re-threshold into a dilated mask, not the same region smaller.
    """
    if scale_height is None:
        return array
    height = array.shape[0]
    if height <= scale_height:
        return array
    import cv2

    width = max(1, int(round(array.shape[1] * scale_height / height)))
    if interpolation is None:
        interpolation = cv2.INTER_AREA
    return cv2.resize(array, (width, scale_height), interpolation=interpolation)


def _developed_frames(
    paths,
    master_dark,
    hotmap,
    base,
    size,
    rejected,
    scale_height=None,
    progress=None,
):
    """Each light as one rgb24 pipe frame, through ``base``'s tone, yielded ONE AT A TIME --
    never listed, so only the frame in flight is resident. One that cannot decode, or is not
    the reference's shape, is ``rejected``: a frame shorter, never one duplicated.

    Scaling happens BEFORE develop, not after, and that ordering is the whole point: at full
    sensor size ``develop_stack_with_params`` fits a gradient polynomial over every pixel and
    stretches through float64 temporaries, several times slower per frame than the decode. The
    shape check runs on the SCALED frame so a differently-shaped light is still caught -- a
    fixed target height gives it a different width.
    """
    width, height = size
    total = len(paths)
    for completed, path in enumerate(paths, start=1):
        try:
            decoded, reason = decode_with_he_fallback(path, master_dark, hotmap)
            if decoded is None:
                rejected.append((path, reason))
                continue
            rgb = _scale_to_height(decoded.rgb, scale_height)
            del decoded  # the full-size decode is spent the moment its scaled copy exists
            if rgb.shape[:2] != base.rgb_uint16.shape[:2]:
                rejected.append((path, f"shape {rgb.shape[:2]} != the reference frame's"))
                continue
            developed = develop_stack_with_params(rgb, base)
            del rgb  # released before the (blocking) pipe write, not after it
            yield (developed[:height, :width] >> 8).astype(np.uint8)
        finally:
            if progress is not None:
                progress(completed, total)


def render_timelapse(
    light_paths,
    out_path,
    dark_paths=(),
    sky_mask=None,
    fps=TIMELAPSE_FPS,
    scale_height=None,
    progress=None,
):
    """One night's lights -> one MP4, as ``(video_path, frames_used, rejected)``: calibrated,
    developed through the middle frame's tone, streamed to ffmpeg. ``dark_paths`` builds the
    master dark + hot-pixel map as ``prepare_frames`` does; ``sky_mask`` rides ``base``.

    ``scale_height`` caps the output height (aspect preserved; see quality.py
    HONEST_RESOLUTION_RULE). The reference
    frame and ``sky_mask`` scale with it, because ``base.sky_mask`` is what every later frame's
    gradient is fitted under -- a full-size mask against a scaled frame is a shape error.
    """
    paths = [Path(p) for p in light_paths]
    master_dark = hotmap = None
    if dark_paths:
        master_dark = build_master_dark([read_raw_mosaic(p) for p in dark_paths])
        hotmap = hot_pixel_map(master_dark)

    ref_path = paths[len(paths) // 2]  # the MIDDLE frame's tone, for all of them
    decoded, reason = decode_with_he_fallback(ref_path, master_dark, hotmap)
    if decoded is None:
        raise TimelapseError(
            f"the reference frame {ref_path} failed to decode ({reason}) -- every frame's tone "
            "comes from it, so there is nothing to develop the sequence against"
        )
    ref_rgb = _scale_to_height(decoded.rgb, scale_height)
    del decoded  # its linear pixels are spent; holding them would cost a frame all render
    if sky_mask is not None and np.shape(sky_mask)[:2] != ref_rgb.shape[:2]:
        import cv2

        # Nearest-neighbour by construction: a boolean mask has no meaningful average, and
        # INTER_AREA would return fractions that re-threshold into a DILATED mask -- the
        # gradient would then be fitted over ground pixels the full-size mask excluded.
        sky_mask = _scale_to_height(
            np.asarray(sky_mask, dtype=np.uint8),
            scale_height,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    base = develop_stack(ref_rgb, sky_mask=sky_mask)
    del ref_rgb
    height, width = base.rgb_uint16.shape[:2]
    size = (width - width % 2, height - height % 2)  # yuv420p needs even axes: crop, never scale

    # ffmpeg runs via the shared ``timelapse.stream_frames_to_video`` -- one house
    # encode, colour contract included; only the frame source and geometry differ here.
    rejected: list[tuple[Path, str]] = []
    frames = (
        frame.tobytes()
        for frame in _developed_frames(
            paths,
            master_dark,
            hotmap,
            base,
            size,
            rejected,
            scale_height,
            progress,
        )
    )
    try:
        out, used = stream_frames_to_video(
            frames, out_path, fps, size, timeout_s=TIMELAPSE_TIMEOUT_S
        )
    except _SharedTimelapseError as exc:
        raise TimelapseError(str(exc)) from exc
    # `code == 0` and a file on disk are NOT evidence of a video: ffmpeg writes the
    # container header the moment it starts, so a run whose every frame was rejected
    # leaves a valid ftyp + empty mdat -- 48 bytes that `out.is_file()` accepts and an
    # operator finds indistinguishable from an output until they try to play it.
    if not used:
        first = "; ".join(f"{p.name}: {reason}" for p, reason in rejected[:3])
        out.unlink(missing_ok=True)
        raise TimelapseError(
            f"timelapse produced no frames for {out}: all {len(paths)} light(s) were rejected"
            + (f" ({first}{', ...' if len(rejected) > 3 else ''})" if rejected else "")
        )
    return out, used, tuple(rejected)
