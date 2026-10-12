"""Phase D thumbnail generator.

Thin wrapper around :mod:`gomoveshift.pipeline.thumbnail_text_overlay` that:

  * picks 3 timestamps from the body of a video (skipping head/tail 5%);
  * probes per-frame QUALITY — sharpness/exposure/clipping (10-bit safe);
  * generates 3 styles × 3 frames = 9 candidate thumbnails;
  * returns the *bold*-styled thumbnail for the best-scoring frame.

Design note — HEVC 10-bit brightness
------------------------------------
The prior implementation used ``ffmpeg -vf signalstats`` on the raw stream,
which returns ``N/A`` for ``yuv420p10le`` input because the default stats
filter expects 8-bit YUV. We fix this by extracting a single JPEG frame
(ffmpeg converts to 8-bit RGB on output) and measuring brightness with PIL.
Option (b) from the spec — empirically reliable across pix_fmts, avoids a
second decode pass just to force ``-pix_fmt rgb24``.

Why not brightest-wins: it ships blown skies and glare on driving footage,
and measured focus not at all. :func:`quality_score` replaces it, and BOTH
thumbnail surfaces (drive path + legacy ``ThumbnailGenerationStage``) rank
through it, so they cannot drift apart again.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..encoding_constants import LONGFORM_COLOR_RANGE
from ..quality import sharpness_score

logger = logging.getLogger(__name__)

# Mid-grey on the 0..255 JPEG scale; exposure falls off linearly from here.
TARGET_BRIGHTNESS = 128.0
# A well-focused 8-bit frame clears this Laplacian variance comfortably and
# motion-blurred driving frames land an order of magnitude below; saturating
# here stops a hyper-detailed frame outranking exposure on its own.
SHARPNESS_SATURATION = 500.0
BLOWN_LEVEL = 250  # luma at/above this is clipped white (sky, glare)
MAX_BLOWN_FRACTION = 0.15  # share of blown pixels that zeroes the term
SHARPNESS_WEIGHT = 0.5
EXPOSURE_WEIGHT = 0.3
HIGHLIGHT_WEIGHT = 0.2
# Bounds the Laplacian pass on a 4K still; one fixed working size also keeps
# every candidate measured on the same footing (ranking is comparative).
SHARPNESS_MAX_DIM = 960


@dataclass(frozen=True)
class FrameQuality:  # measured thumbnail-worthiness of one extracted frame
    brightness: float  # mean luma, 0..255
    sharpness: float  # Laplacian variance — higher is sharper
    blown_fraction: float  # share of pixels at/above ``BLOWN_LEVEL``


def quality_score(q: FrameQuality) -> float:
    """Rank a frame in [0, 1]: sharp, well exposed, not blown out."""
    exposure = max(0.0, 1.0 - abs(q.brightness - TARGET_BRIGHTNESS) / TARGET_BRIGHTNESS)
    sharp = min(max(q.sharpness, 0.0) / SHARPNESS_SATURATION, 1.0)
    highlight = max(0.0, 1.0 - q.blown_fraction / MAX_BLOWN_FRACTION)
    return SHARPNESS_WEIGHT * sharp + EXPOSURE_WEIGHT * exposure + HIGHLIGHT_WEIGHT * highlight


def rank_by_quality(scored: Sequence[Tuple[Any, Optional[FrameQuality]]]) -> List[Any]:
    """Return the keys of *scored* best-first; unmeasurable frames sort last."""
    measured = [(k, q) for k, q in scored if q is not None]
    measured.sort(key=lambda kq: quality_score(kq[1]), reverse=True)
    return [k for k, _ in measured] + [k for k, q in scored if q is None]


def _evenly_spaced_body(duration: float, count: int) -> List[float]:
    """Legacy pick: *count* timestamps inside the central 90% of a video.

    For very short clips (<1s), clamps into (0, duration)."""
    if duration <= 0:
        return [0.0] * count
    if duration < 1.0:
        # Degenerate: spread evenly within the clip avoiding exact endpoints.
        step = duration / (count + 1)
        return [step * (i + 1) for i in range(count)]
    margin = duration * 0.05
    usable = duration - 2 * margin
    if usable <= 0:
        return [duration / 2] * count
    if count == 1:
        return [margin + usable / 2]
    step = usable / (count - 1) if count > 1 else usable
    return [margin + step * i for i in range(count)]


def pick_body_timestamps(
    duration: float,
    count: int = 3,
    *,
    is_masked: Optional[Callable[[float], bool]] = None,
    unmasked_intervals: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[float]:
    """Return *count* timestamps inside the central 90% of a video.

    Stop-aware: when ``is_masked`` (a ``t -> bool`` predicate, ``True`` at a
    STOPPED second) or ``unmasked_intervals`` (kept ``(start, end)`` ranges, both
    in the SAME clock as ``duration`` — LF-FINAL-local seconds) is supplied, only
    UNMASKED seconds of the body are sampled, so a parked roadside frame can never
    become the thumbnail even though it is often the brightest. Belt-and-suspenders
    over the LF's structural stop removal.

    When neither is supplied (legacy / GPS-less), behaviour is byte-for-byte the
    prior evenly-spaced pick. If the mask leaves no unmasked body second (or the
    clip is sub-second), it also falls back to the legacy pick rather than crash.
    """
    base = _evenly_spaced_body(duration, count)
    if is_masked is None and unmasked_intervals is None:
        return base  # legacy path — unchanged
    if duration < 1.0:
        return base  # mask-awareness is meaningless on a sub-second clip
    margin = duration * 0.05
    usable = duration - 2 * margin
    if usable <= 0:
        return base
    lo = margin

    def _allowed(t: float) -> bool:
        if is_masked is not None and is_masked(t):
            return False
        if unmasked_intervals is not None and not any(s <= t <= e for s, e in unmasked_intervals):
            return False
        return True

    # Sample the body on a ~1 Hz grid (mask resolution) and keep only unmasked
    # points, capped so a multi-hour LF can't explode the grid.
    n_grid = min(int(usable) + 1, 2000)
    step = usable / (n_grid - 1) if n_grid > 1 else usable
    allowed = [lo + step * i for i in range(n_grid) if _allowed(lo + step * i)]
    if not allowed:
        return base  # nothing unmasked in the body → legacy fallback
    if count <= 1:
        return [allowed[len(allowed) // 2]]
    idxs = [round(i * (len(allowed) - 1) / (count - 1)) for i in range(count)]
    return [allowed[j] for j in idxs]


def probe_frame_quality(video_path: str, timestamp: float) -> Optional[FrameQuality]:
    """Measure sharpness/exposure/clipping of a single frame at *timestamp*.

    Works for HEVC 10-bit sources because ffmpeg writes an 8-bit JPEG on
    output (pix_fmt conversion happens during encode).
    """
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "frame.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            # The measured frame must land in the same numeric domain as the
            # shipped thumbnail: limited-range video → full-range JPEG, stated
            # rather than left to implicit swscale behaviour.
            "-vf",
            f"scale=in_range={LONGFORM_COLOR_RANGE}:out_range=pc",
            "-q:v",
            "3",
            str(out),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out probing %s @ %s", video_path, timestamp)
            return None
        if r.returncode != 0 or not out.exists() or out.stat().st_size < 128:
            logger.warning(
                "ffmpeg frame probe failed (%s @ %s): %s",
                video_path,
                timestamp,
                (r.stderr or "")[:200],
            )
            return None
        try:
            with Image.open(out) as im:
                gs = im.convert("L")
                w, h = gs.size
                if max(w, h) > SHARPNESS_MAX_DIM:
                    k = SHARPNESS_MAX_DIM / float(max(w, h))
                    size = (max(1, int(w * k)), max(1, int(h * k)))
                    gs = gs.resize(size, Image.Resampling.LANCZOS)
                a = np.asarray(gs, dtype=np.float64)
        except Exception as e:
            logger.warning("PIL failed to read probed frame: %s", e)
            return None

    if a.size == 0:
        return None
    sharpness = sharpness_score(a) if a.shape[0] > 2 and a.shape[1] > 2 else 0.0
    return FrameQuality(
        brightness=float(a.mean()),
        sharpness=sharpness,
        blown_fraction=float((a >= BLOWN_LEVEL).mean()),
    )


def probe_frame_brightness(video_path: str, timestamp: float) -> Optional[float]:
    """Return mean brightness [0..255] of a single frame at *timestamp*."""
    q = probe_frame_quality(video_path, timestamp)
    return None if q is None else q.brightness


def select_best_timestamps(
    video_path: str,
    candidates: Sequence[float],
    count: int = 3,
    probe: Optional[Callable[[str, float], Optional[FrameQuality]]] = None,
) -> List[float]:
    """Score *candidates* and return the best *count*, in chronological order.

    Returns ``[]`` when nothing could be measured, so callers keep a fallback.
    """
    probe_fn = probe if probe is not None else probe_frame_quality
    scored = [(t, probe_fn(str(video_path), t)) for t in candidates]
    best = rank_by_quality(scored)[:count]
    measurable = {t for t, q in scored if q is not None}
    return sorted(t for t in best if t in measurable)


def generate_lf_thumbnail(
    video_path: Path,
    title: str,
    subtitle: Optional[str],
    output_dir: Path,
    duration: float,
    *,
    is_masked: Optional[Callable[[float], bool]] = None,
    unmasked_intervals: Optional[Sequence[Tuple[float, float]]] = None,
) -> Optional[Path]:
    """Generate 9 candidate thumbnails, return the *bold* variant for the best frame.

    "Best" is :func:`quality_score`, not the brightest sample.

    ``is_masked`` / ``unmasked_intervals`` (both in LF-FINAL-local seconds) are
    forwarded to :func:`pick_body_timestamps` so a stopped/parked second is never
    sampled; ``None`` on both keeps the legacy evenly-spaced pick.

    Returns None on failure.
    """
    from ..thumbnail_text_overlay import generate_thumbnails_from_video

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = pick_body_timestamps(
        duration, count=3, is_masked=is_masked, unmasked_intervals=unmasked_intervals
    )
    quality: List[Optional[FrameQuality]] = [
        probe_frame_quality(str(video_path), ts) for ts in timestamps
    ]

    try:
        results = generate_thumbnails_from_video(
            video_path=str(video_path),
            title=title,
            subtitle=subtitle,
            output_dir=str(output_dir),
            frame_timestamps=[int(round(t)) for t in timestamps],
        )
    except Exception as e:
        logger.warning("Thumbnail generation failed for %s: %s", video_path, e)
        return None

    if not results:
        return None

    # Pick the best-scoring frame; fall back to the first available frame.
    ranked = rank_by_quality(list(enumerate(quality)))
    best_idx = ranked[0] if any(q is not None for q in quality) else 0
    frame_key = f"frame{best_idx + 1}"
    frame_entry = results.get(frame_key) or next(iter(results.values()))
    bold_path = frame_entry.get("bold") if isinstance(frame_entry, dict) else None
    if not bold_path:
        return None
    return Path(bold_path)
