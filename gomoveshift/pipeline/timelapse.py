"""Shared time-lapse core: streaming frames to ffmpeg, common to every source --
night RAW sequences (``photos/astro_timelapse.py``), daylight stills, and video
frame-decimation.

The night module owns its own decode/develop/dark-calibration pipeline (a
sequence's shared tone is astro-specific and stays there); this module only
knows how to turn an already-decoded stream of rgb24 frames into one MP4
(``stream_frames_to_video``), plus the two source-specific readers that feed
it: ordered daylight stills, and a decimated video source.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .photos.develop import PhotoDecodeError, decode
from .photos.reel import build_encode_cmd

#: Frame rate for a daylight stills timelapse. Distinct from the night module's
#: 24 (chosen there to avoid sky-motion judder); this is an ordinary encode.
TIMELAPSE_FPS = 24
TIMELAPSE_TIMEOUT_S = 1800


class TimelapseError(Exception):
    """No frames given, an unreadable still, a bad decimation request, or a
    failed encode -- every refusal this module raises."""


def _scale_to_height(rgb, height):
    """``rgb`` reduced so its height is ``height``, aspect preserved -- a
    CEILING, never a target; ``None`` or an already-shorter frame is untouched.
    Same contract as ``astro_timelapse._scale_to_height``, generalised."""
    if height is None:
        return rgb
    h = rgb.shape[0]
    if h <= height:
        return rgb
    import cv2

    width = max(1, int(round(rgb.shape[1] * height / h)))
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def order_stills(paths):
    """Stills in capture order. No EXIF capture-time reader exists elsewhere in
    this codebase to reuse, so this is filename order -- the convention an
    operator naming frames sequentially (IMG_0001, IMG_0002, ...) already relies
    on. A future EXIF-aware reader can replace this body without touching callers."""
    return sorted(Path(p) for p in paths)


def stream_frames_to_video(
    frames,
    out_path,
    fps,
    size,
    *,
    timeout_s=TIMELAPSE_TIMEOUT_S,
    encode_cmd_builder=build_encode_cmd,
):
    """Pipe ``frames`` (rgb24 bytes, one per video frame) to ffmpeg via
    ``encode_cmd_builder`` -- one house encode regardless of the frame source.
    Returns ``(out_path, used)``; a failed encode raises ``TimelapseError``.

    Deliberately does NOT check for zero frames written: "empty" means something
    different per caller (every light rejected, no stills found, fewer source
    frames than a decimation step) so that refusal -- and its wording -- is each
    caller's own, the same split ``astro_timelapse.render_timelapse`` already made.
    """
    out = Path(out_path)
    used = 0
    cmd = encode_cmd_builder(out, fps, size)
    with tempfile.TemporaryFile() as errfile:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errfile)
        try:
            for frame in frames:
                proc.stdin.write(frame)
                used += 1
            proc.stdin.close()
            code, why = proc.wait(timeout=timeout_s), ""
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            proc.kill()
            code, why = -1, f"{exc}: "
        if code != 0 or not out.is_file():
            errfile.seek(0)
            tail = errfile.read().decode(errors="replace").strip()[-400:]
            out.unlink(missing_ok=True)  # ffmpeg's stub, not an output
            raise TimelapseError(f"timelapse render failed for {out}: {why}exit {code}: {tail}")
    return out, used


def render_stills_timelapse(paths, out_path, fps=TIMELAPSE_FPS, scale_height=None):
    """Daylight stills -> one MP4, ordered by ``order_stills``. No astro-specific
    step runs here (no dark/hot-pixel calibration, no shared-tone develop --
    daylight stills already carry their own exposure); an unreadable or
    differently-shaped still is named and skipped, never padded over, mirroring
    ``astro_timelapse.render_timelapse``'s own contract.

    Returns ``(out_path, frames_used, rejected)``. Raises ``TimelapseError`` on
    empty input, or when every still was rejected -- and leaves no stub file.
    """
    stills = order_stills(paths)
    if not stills:
        raise TimelapseError("timelapse needs at least one still: none were given")

    rejected: list[tuple[Path, str]] = []
    decoded: list[np.ndarray] = []
    ref_shape = None  # (height, width), even on both axes -- yuv420p's requirement
    for path in stills:
        try:
            rgb = decode(path).rgb
        except PhotoDecodeError as exc:
            rejected.append((path, str(exc)))
            continue
        if rgb.dtype != np.uint8:
            rgb = (rgb >> 8).astype(np.uint8)
        rgb = _scale_to_height(rgb, scale_height)
        height, width = rgb.shape[:2]
        even_shape = (height - height % 2, width - width % 2)  # yuv420p needs even axes
        if ref_shape is None:
            ref_shape = even_shape
        if even_shape != ref_shape:
            rejected.append((path, f"shape {rgb.shape[:2]} != the first frame's {ref_shape}"))
            continue
        decoded.append(rgb[: ref_shape[0], : ref_shape[1]])

    if not decoded:
        first = "; ".join(f"{p.name}: {reason}" for p, reason in rejected[:3])
        raise TimelapseError(
            f"timelapse produced no frames for {out_path}: all {len(stills)} still(s) "
            "were rejected"
            + (f" ({first}{', ...' if len(rejected) > 3 else ''})" if rejected else "")
        )

    height, width = ref_shape
    out, used = stream_frames_to_video(
        (frame.tobytes() for frame in decoded), out_path, fps, (width, height)
    )
    return out, used, tuple(rejected)


def decimate_video(source, out_path, *, keep_every=None, speed_factor=None):
    """Video source -> a shorter MP4 by dropping frames -- exactly one of:

    ``keep_every=N``: keep every Nth frame, a hard frame-count decimation
    (timestamps re-derived so the result plays at the SAME apparent rate).
    ``speed_factor=N``: keep every Nth frame too, but the intent is playback N
    times faster rather than a frame-count target -- same filter either way.

    Uses ffmpeg's ``select``/``setpts`` filters, the same filter family the rest
    of this pipeline already reaches for to drop frames (``build_encode_cmd``'s
    own house encode is for a raw rgb24 pipe, which a video source isn't).
    """
    if (keep_every is None) == (speed_factor is None):
        raise TimelapseError("decimate_video needs exactly one of keep_every/speed_factor")
    step = keep_every if keep_every is not None else speed_factor
    if step < 1:
        raise TimelapseError(f"decimate step must be >= 1 (got {step})")

    out = Path(out_path)
    vf = f"select='not(mod(n\\,{step}))',setpts=N/FRAME_RATE/TB"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-vf", vf, "-an", str(out)]
    result = subprocess.run(cmd, capture_output=True, timeout=TIMELAPSE_TIMEOUT_S)
    if result.returncode != 0 or not out.is_file():
        tail = result.stderr.decode(errors="replace").strip()[-400:]
        out.unlink(missing_ok=True)
        raise TimelapseError(f"decimation failed for {out}: exit {result.returncode}: {tail}")
    return out, cmd
