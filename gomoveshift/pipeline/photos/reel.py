"""Ken Burns pan-zoom reel: N stills -> one vertical MP4, a cut between each.

Split in two on purpose. ``plan_motion`` is pure arithmetic — no ffmpeg, no I/O
— so the motion (the part with taste in it) is exhaustively unit-testable.
``render_reel`` crops+resizes each frame and pipes it to ffmpeg over stdin
rather than using the ``zoompan`` filter, which keeps the easing ours and
reproducible from the planner alone. ``assembly.encoder.VideoEncoder`` is not
reused: every one of its methods takes a video *file*, never a raw stream; only
its invocation style (and ShortsRenderer's) carries over. Local working output
like ``photos develop`` — not the blur-gated publish path. Each still is the
route-story developed still (``hud._load_still``): one grade for the reel, the
still cards and the inserts.
"""

import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from ..encoding_constants import (
    LONGFORM_COLOR_PRIMARIES,
    LONGFORM_COLOR_RANGE,
    LONGFORM_COLOR_TRC,
    LONGFORM_COLORSPACE,
    SHORTS_HEIGHT,
    SHORTS_WIDTH,
)
from .develop import PhotoDecodeError

logger = logging.getLogger(__name__)

#: Vertical social format, shared with the shorts encode — one house geometry.
REEL_SIZE = (SHORTS_WIDTH, SHORTS_HEIGHT)
#: Seconds one still holds: long enough to read a landscape, short enough that
#: six stills stay inside a 20s reel. Under ~2s the eye never settles.
REEL_SECONDS_PER_STILL = 3.0
#: Output cadence. 30 keeps a slow pan smooth; 24 judders on straight lines.
REEL_FPS = 30
#: How far the window shrinks (or grows) across one still — 12% over 3s = 4%/s.
#: Below ~8% the move is invisible on a phone; above ~20% it reads as an effect.
REEL_ZOOM = 0.12
#: Where the window sits in the slack the zoom opens, start -> end. Centred is
#: 0.5, so 0.35 -> 0.65 drifts through centre instead of hugging an edge.
REEL_PAN_FROM = 0.35
REEL_PAN_TO = 0.65
#: A held still shows banding a moving frame hides, so quality over size; the
#: content is near-static, which makes ``slow`` cheap.
REEL_CRF = 18
REEL_PRESET = "slow"
REEL_TIMEOUT_S = 900
#: The one colour conversion in this module. Frames arrive as **true full-range**
#: rgb24 (``develop.decode`` hands back 0-255 RGB), and the encode is 4:2:0 YUV, so
#: something must decide where 0 and 255 land. Leaving it to swscale's defaults made
#: a shipped artefact's range convention implicit — the class of bug the rest of the
#: pipeline stamps its way out of. State it: full-range in, limited-range BT.709 out,
#: which is what the ``-color_*`` flags below then declare. ``format=yuv420p``
#: downstream is what makes ``scale`` negotiate YUV as its output, so the range and
#: matrix options apply to a real RGB->YUV hop rather than an RGB->RGB no-op. The
#: trailing ``setparams`` is not decoration: filtered-frame properties beat the
#: ``-color_*`` output options at the encoder, so without it primaries and transfer
#: ship ``unknown`` no matter what the command line says (measured, not assumed).
REEL_RANGE_FILTER = (
    f"scale=in_range=full:out_range={LONGFORM_COLOR_RANGE}"
    f":out_color_matrix={LONGFORM_COLORSPACE},format=yuv420p"
    f",setparams=range={LONGFORM_COLOR_RANGE}:colorspace={LONGFORM_COLORSPACE}"
    f":color_primaries={LONGFORM_COLOR_PRIMARIES}:color_trc={LONGFORM_COLOR_TRC}"
)


def _load_still(path: Path):
    """``route_story.hud._load_still``, imported late (hud imports this package)."""
    from ..route_story import hud

    return hud._load_still(path)


class ReelError(RuntimeError):
    """A reel could not be rendered. Always names the offending input."""


class CropRect(NamedTuple):
    """One frame's source-pixel crop window, in float pixels."""

    x: float
    y: float
    width: float
    height: float


def plan_motion(
    source_size, output_size=REEL_SIZE, duration=REEL_SECONDS_PER_STILL, fps=REEL_FPS, index=0
) -> list[CropRect]:
    """Per-frame crop rectangles for one still. Pure: no I/O, no encoder.

    Odd ``index`` stills pull out while even ones push in, so a reel alternates
    instead of droning. Every rect carries the output aspect exactly and sits
    inside the source by construction, not by a later clamp.
    """
    src_w, src_h = source_size
    aspect = output_size[0] / output_size[1]
    # The largest window of the OUTPUT aspect that fits the source; every rect
    # is this scaled down and slid inside the slack. ``src_w / aspect * aspect``
    # can land a hair over src_w in binary floating point, so step it down until
    # it cannot — "the crop fits" stays arithmetic rather than a tolerance.
    base = min(float(src_h), src_w / aspect)
    while base * aspect > src_w:
        base = math.nextafter(base, 0.0)
    zoomed = 1.0 / (1.0 + REEL_ZOOM)
    push = index % 2 == 0
    scale_from, scale_to = (1.0, zoomed) if push else (zoomed, 1.0)
    pan_from, pan_to = (REEL_PAN_FROM, REEL_PAN_TO) if push else (REEL_PAN_TO, REEL_PAN_FROM)
    frames = max(2, round(duration * fps))
    rects = []
    for i in range(frames):
        t = i / (frames - 1)
        eased = t * t * (3.0 - 2.0 * t)  # smoothstep: the move starts and stops from rest
        height = base * (scale_from + (scale_to - scale_from) * eased)
        width = height * aspect
        pan = pan_from + (pan_to - pan_from) * eased
        rects.append(CropRect(pan * (src_w - width), pan * (src_h - height), width, height))
    return rects


def _frame(rgb: np.ndarray, rect: CropRect, size: tuple[int, int]) -> np.ndarray:
    """Crop ``rect`` out of ``rgb`` and resample it to ``size``."""
    x, y = int(rect.x), int(rect.y)
    h = min(max(1, round(rect.height)), rgb.shape[0] - y)
    w = min(max(1, round(rect.width)), rgb.shape[1] - x)
    crop = np.ascontiguousarray(rgb[y : y + h, x : x + w])
    interp = cv2.INTER_AREA if w >= size[0] else cv2.INTER_LANCZOS4
    return cv2.resize(crop, size, interpolation=interp)


def _check(paths, output) -> tuple[list[Path], Path]:
    """Upstream gate: refuse at the door, before a byte is decoded or encoded."""
    stills = [Path(p) for p in paths]
    if not stills:
        raise ReelError("no stills given: a reel needs at least one photo")
    if missing := [str(p) for p in stills if not p.is_file()]:
        raise ReelError(f"unreadable still(s): {', '.join(missing)}")
    out = Path(output).expanduser()
    if not (out.parent.is_dir() and os.access(out.parent, os.W_OK)):
        raise ReelError(f"cannot write {out}: {out.parent} is not a writable directory")
    return stills, out


def build_encode_cmd(out: Path, fps: int, size: tuple[int, int]) -> list[str]:
    """The ffmpeg invocation ``render_reel`` pipes frames into. Pure, so the colour
    contract (explicit RGB->YUV range + the tags that describe it) is unit-testable
    without a render — the same split that keeps ``plan_motion`` testable."""
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{size[0]}x{size[1]}", "-framerate", str(fps), "-i", "-",
        "-an", "-vf", REEL_RANGE_FILTER,
        "-c:v", "libx264", "-preset", REEL_PRESET, "-crf", str(REEL_CRF),
        "-pix_fmt", "yuv420p",
        "-color_range", LONGFORM_COLOR_RANGE, "-colorspace", LONGFORM_COLORSPACE,
        "-color_primaries", LONGFORM_COLOR_PRIMARIES, "-color_trc", LONGFORM_COLOR_TRC,
        "-movflags", "+faststart", str(out),
    ]  # fmt: skip


def render_reel(
    paths, output, seconds=REEL_SECONDS_PER_STILL, fps=REEL_FPS, size=REEL_SIZE
) -> Path:
    """Render ``paths`` as one vertical Ken Burns reel and return the output."""
    stills, out = _check(paths, output)
    cmd = build_encode_cmd(out, fps, size)
    pool = cv2.getNumThreads()
    with tempfile.TemporaryFile() as errfile:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errfile)
        try:
            # cv2's thread pool is not deterministic across runs; pin it for the
            # whole render, exactly as the Mertens fuse does.
            cv2.setNumThreads(1)
            for index, path in enumerate(stills):
                # The one developed still: the same graded, disk-cached pixels the
                # route-story cards and the insert engine hold, never a private decode.
                rgb = np.asarray(_load_still(path))
                logger.info("Reel still %d/%d: %s", index + 1, len(stills), path.name)
                for rect in plan_motion((rgb.shape[1], rgb.shape[0]), size, seconds, fps, index):
                    proc.stdin.write(_frame(rgb, rect, size).tobytes())
            proc.stdin.close()
            code, why = proc.wait(timeout=REEL_TIMEOUT_S), ""
        except (OSError, PhotoDecodeError, ValueError, subprocess.TimeoutExpired) as exc:
            proc.kill()
            code, why = -1, f"{exc}: "
        finally:
            cv2.setNumThreads(pool)
        if code != 0 or not out.is_file():
            errfile.seek(0)
            tail = errfile.read().decode(errors="replace").strip()[-400:]
            raise ReelError(f"reel render failed for {out}: {why}ffmpeg exit {code}: {tail}")
    return out
