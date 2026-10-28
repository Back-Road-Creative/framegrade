"""Tracked vertical reframe: where the 9:16 crop window sits, over time.

The default short is a fixed centre crop. This module plans a horizontal crop
position per second from a cheap frame signal (where the horizontal-gradient
energy of the road/horizon band sits), then smooths it so the window never
chases: a low-pass, a dead zone it must leave before moving at all, and a
maximum pan speed. The plan is a ``CropTrack`` that ``build_crop_filter`` turns
into a time-varying ``crop`` x expression. Crop size and scaler are the
caller's and are not touched here, so the declared upscale stays true.

``plan_crop_track`` and ``frame_centre`` are pure; only ``plan_tracked_crop``
runs ffmpeg (one low-resolution gray decode).
"""

from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_S = 1.0
SAMPLE_WIDTH = 160
# Rows (fraction of frame height) holding road, verge and horizon; sky and
# bonnet are left out of the energy.
BAND_TOP, BAND_BOTTOM = 0.35, 0.80
# Smoothing time constant, dead zone and pan speed, the latter two as fractions
# of the crop width (speed per second).
SMOOTH_TAU_S = 3.0
DEAD_ZONE_FRAC = 0.10
MAX_SPEED_FRAC = 0.08


@dataclass(frozen=True)
class CropTrack:
    """Crop left edge ``xs[i]`` (source pixels) at ``times[i]`` seconds into the clip."""

    times: tuple[float, ...]
    xs: tuple[float, ...]


def frame_centre(gray: np.ndarray) -> float:
    """Normalised (0..1) horizontal centre of detail in one gray frame.

    Column energy of the horizontal gradient inside the road band, plus a flat
    prior equal to its mean, so an empty or uniform frame reads as centred and
    one strong edge cannot fling the window to the side.
    """
    g = np.asarray(gray, dtype=np.float32)
    h, w = g.shape
    band = g[int(h * BAND_TOP) : max(int(h * BAND_BOTTOM), int(h * BAND_TOP) + 1)]
    energy = np.abs(np.diff(band, axis=1)).sum(axis=0)
    energy = np.append(energy, energy[-1]) + energy.mean()
    total = float(energy.sum())
    if total <= 0.0:
        return 0.5
    return float(((np.arange(w) + 0.5) * energy).sum() / total / w)


def plan_crop_track(
    times: Sequence[float],
    centres: Sequence[float],
    *,
    crop_w: int,
    src_w: int,
    smooth_tau_s: float = SMOOTH_TAU_S,
    dead_zone_frac: float = DEAD_ZONE_FRAC,
    max_speed_frac: float = MAX_SPEED_FRAC,
) -> CropTrack:
    """Smooth per-sample subject centres (source px) into a crop-left-edge track.

    Low-pass the centres, then move the window only by the part of the error
    that lies outside the dead zone, at most ``max_speed_frac * crop_w`` per
    second, and never past the target (no overshoot). Clamped inside the frame.
    """
    if len(times) == 0 or len(times) != len(centres):
        raise ValueError("times and centres must be non-empty and equal length")
    if crop_w <= 0 or crop_w > src_w:
        raise ValueError(f"crop_w {crop_w} does not fit source width {src_w}")
    hi = float(src_w - crop_w)

    def clamp(x: float) -> float:
        return min(max(x, 0.0), hi)

    dead = dead_zone_frac * crop_w
    filt = float(centres[0])
    pos = clamp(filt - crop_w / 2)
    xs = [pos]
    for i in range(1, len(times)):
        dt = float(times[i]) - float(times[i - 1])
        if dt <= 0:
            raise ValueError("times must be strictly increasing")
        filt += (1.0 - math.exp(-dt / smooth_tau_s)) * (float(centres[i]) - filt)
        err = clamp(filt - crop_w / 2) - pos
        excess = abs(err) - dead
        if excess > 0:
            step = min(excess, max_speed_frac * crop_w * dt)
            pos = clamp(pos + math.copysign(step, err))
        xs.append(pos)
    return CropTrack(times=tuple(float(t) for t in times), xs=tuple(xs))


def crop_x_expression(track: CropTrack, src_w: int, crop_w: int) -> str:
    """ffmpeg ``crop`` x expression: the track linearly interpolated over ``t``.

    A flat sum of clipped ramps (no nested ``if``), so a long clip stays short.
    Commas are bare: the caller single-quotes the expression.
    """
    hi = src_w - crop_w
    xs = [min(max(x, 0.0), float(hi)) for x in track.xs]
    expr = f"{xs[0]:.2f}"
    for i in range(len(xs) - 1):
        delta = xs[i + 1] - xs[i]
        if abs(delta) < 0.005:
            continue
        t0, t1 = track.times[i], track.times[i + 1]
        expr += f"{delta:+.2f}*clip((t-{t0:.3f})/{t1 - t0:.3f},0,1)"
    return expr


def sample_centres(
    video: Path,
    start: float,
    duration: float,
    src_w: int,
    src_h: int,
    interval: float = SAMPLE_INTERVAL_S,
) -> tuple[list[float], list[float]]:
    """Decode one small gray frame per ``interval`` and return (times, centres in px)."""
    w = SAMPLE_WIDTH
    h = max(2, round(w * src_h / src_w) // 2 * 2)
    cmd = [
        "ffmpeg", "-v", "error", "-ss", str(start), "-i", str(video), "-t", str(duration),
        "-vf", f"fps=1/{interval},scale={w}:{h},format=gray",
        "-f", "rawvideo", "-",
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"reframe sampling failed: {r.stderr.decode(errors='replace')[-300:]}")
    frames = np.frombuffer(r.stdout, dtype=np.uint8)
    n = frames.size // (w * h)
    if n == 0:
        raise RuntimeError("reframe sampling produced no frames")
    frames = frames[: n * w * h].reshape(n, h, w)
    return [i * interval for i in range(n)], [frame_centre(f) * src_w for f in frames]


def plan_tracked_crop(
    video: Path,
    start: float,
    duration: float,
    src_w: int,
    src_h: int,
    crop_w: int,
) -> CropTrack:
    """Sample the clip and plan its crop track. Raises ``RuntimeError`` on no signal."""
    times, centres = sample_centres(video, start, duration, src_w, src_h)
    return plan_crop_track(times, centres, crop_w=crop_w, src_w=src_w)
