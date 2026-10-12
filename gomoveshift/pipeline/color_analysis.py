"""Color analysis — frame sampling, histogram statistics, segment aggregation.

Split from color.py.
"""

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from gomoveshift.pipeline._json_safe import safe_json_loads
from gomoveshift.pipeline.quality import measure_detail, measure_noise

logger = logging.getLogger(__name__)


def scale_range_token(color_range: str) -> str:
    """Map an ffprobe ``color_range`` to an ffmpeg range token.

    ffprobe reports ``pc`` (full / 0-255), ``tv`` (limited / 16-235), or an
    empty/``unknown`` value for untagged streams. Raw GoPro footage is
    full-range by construction, so unknown/empty defaults to ``full``.

    Feeds both ``scale``'s ``in_range`` (a limited source
    becomes a no-op squeeze instead of being double-compressed) and the
    ``setrange`` tag at the head of every chain reaching an RGB filter.
    Lives here, not in ``single_pass``, so ``analysis/`` can
    import it without the encoder. See docs/pipeline-internals.md.
    """
    cr = (color_range or "").strip().lower()
    if cr in ("tv", "mpeg", "limited"):
        return "limited"
    # pc / jpeg / full / unknown / empty -> full (GoPro default).
    return "full"


# Operability: ffmpeg sample-frames runs silently for 60–90 minutes on 4K
# HEVC sources. We surface a wall-clock heartbeat so operators can tell
# "still working" from "deadlocked" without inspecting /proc/PID/io. Tests
# monkeypatch this to a sub-second value to keep CI fast.
_HEARTBEAT_INTERVAL_SEC = 30.0


def _emit_color_analysis_heartbeat(stop_event: threading.Event, started_at: float) -> None:
    """Daemon-thread target: log a heartbeat every _HEARTBEAT_INTERVAL_SEC.

    Exits as soon as ``stop_event`` is set. Sleeps in short slices so the
    thread terminates quickly when the parent ffmpeg subprocess returns.
    """
    # Wait one full interval before the first emission — short runs
    # (cache hits, small videos) should not produce a heartbeat at all.
    if stop_event.wait(timeout=_HEARTBEAT_INTERVAL_SEC):
        return
    while not stop_event.is_set():
        elapsed = time.monotonic() - started_at
        elapsed_min = elapsed / 60.0
        if elapsed_min >= 1.0:
            logger.info(
                "  Progress: color analysis still running, elapsed %.1fm",
                elapsed_min,
            )
        else:
            logger.info(
                "  Progress: color analysis still running, elapsed %.0fs",
                elapsed,
            )
        if stop_event.wait(timeout=_HEARTBEAT_INTERVAL_SEC):
            return


# ── Data Classes ──────────────────────────────────────────────────────


@dataclass
class FrameStats:
    """Histogram statistics for a single frame."""

    black_point: float  # 5th percentile luminance (0-255)
    white_point: float  # 95th percentile luminance (0-255)
    mid_tone: float  # Median luminance (0-255)
    saturation: float  # Average saturation (0-1 normalized)
    r_mean: float = 0.5  # Mean red channel (0-1 normalized)
    g_mean: float = 0.5  # Mean green channel (0-1 normalized)
    b_mean: float = 0.5  # Mean blue channel (0-1 normalized)
    r_mean_high: float = 0.5  # Mean red over highlight pixels (0-1)
    g_mean_high: float = 0.5  # Mean green over highlight pixels (0-1)
    b_mean_high: float = 0.5  # Mean blue over highlight pixels (0-1)
    highlight_frac: float = 0.0  # Fraction of frame in the highlight region
    sky_fraction: float = 0.0  # Area share of the bright upper-band sky region
    fg_white_point: float = 235.0  # 95th-pct luminance, sky excluded (0-255)
    fg_mid_tone: float = 128.0  # Median luminance, sky excluded (0-255)
    fg_r_mean: float = 0.5  # Mean red over the non-sky region (0-1)
    fg_g_mean: float = 0.5  # Mean green over the non-sky region (0-1)
    fg_b_mean: float = 0.5  # Mean blue over the non-sky region (0-1)
    detail_score: float = 0.0  # Variance of Laplacian, bounded luma view (0 = unmeasured)
    noise_score: float = 0.0  # Noise sigma in luma levels (0 = unmeasured)


@dataclass
class SegmentStats:
    """Aggregated statistics for a video segment (multiple frames)."""

    black_point: float
    white_point: float
    mid_tone: float
    saturation: float
    r_mean: float = 0.5
    g_mean: float = 0.5
    b_mean: float = 0.5
    r_mean_high: float = 0.5
    g_mean_high: float = 0.5
    b_mean_high: float = 0.5
    highlight_frac: float = 0.0
    sky_fraction: float = 0.0
    fg_white_point: float = 235.0
    fg_mid_tone: float = 128.0
    fg_r_mean: float = 0.5
    fg_g_mean: float = 0.5
    fg_b_mean: float = 0.5
    detail_score: float = 0.0
    noise_score: float = 0.0


# ── Frame Analysis ────────────────────────────────────────────────────

# Highlight region for the highlight white-balance term. A pixel is a highlight
# when its BT.709 luminance sits above the colorbalance highlight knee (0.75 of
# full scale) — the same band the rh/bh shift acts on — so the cast is measured
# exactly where it is corrected.
HIGHLIGHT_LUMA_THRESHOLD = 0.75

# Sky region-of-interest for the exposure/white-balance meter. The sky lives in
# the top of the frame, so only this upper band is eligible to be treated as sky
# (mirrors the visual scorer's upper-third convention) — a white car or snow on
# the road below never counts. The band is a fixed geometric ROI, not a grading
# strength: the correction magnitude is governed entirely by the measured sky
# fraction, so this constant is not a tuned on/off threshold.
SKY_ROI_UPPER_FRACTION = 0.33


def analyze_frame(frame: np.ndarray) -> FrameStats:
    """
    Analyze a single RGB frame's luminance and chroma statistics.

    Args:
        frame: numpy array of shape (H, W, 3) in uint8 RGB

    Returns:
        FrameStats with black_point, white_point, mid_tone, saturation
    """
    # Compute luminance using BT.709 coefficients
    # Y = 0.2126*R + 0.7152*G + 0.0722*B
    r = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    b = frame[:, :, 2].astype(np.float32)

    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    black_point = float(np.percentile(luminance, 5))
    white_point = float(np.percentile(luminance, 95))
    mid_tone = float(np.median(luminance))

    # Compute saturation: max(R,G,B) - min(R,G,B) / max(R,G,B)
    # Avoid division by zero
    rgb_max = np.maximum(np.maximum(r, g), b)
    rgb_min = np.minimum(np.minimum(r, g), b)

    # Saturation = chroma / max, where chroma = max - min
    with np.errstate(divide="ignore", invalid="ignore"):
        sat_per_pixel = np.where(
            rgb_max > 0,
            (rgb_max - rgb_min) / rgb_max,
            0.0,
        )

    saturation = float(np.mean(sat_per_pixel))

    # Per-channel means for white balance detection
    r_mean = float(np.mean(r)) / 255.0
    g_mean = float(np.mean(g)) / 255.0
    b_mean = float(np.mean(b)) / 255.0

    # Highlight-region per-channel means + area share, for a white-balance term
    # that targets a cast living only in the bright end (a blue sky, a warm
    # blown window). Whole-frame means average such a cast away; measuring over
    # just the highlight pixels sees it. The area share travels with the means
    # so the correction can be weighted by how much evidence the frame offers.
    hl_mask = luminance > HIGHLIGHT_LUMA_THRESHOLD * 255.0
    highlight_frac = float(np.mean(hl_mask))
    if highlight_frac > 0.0:
        r_mean_high = float(np.mean(r[hl_mask])) / 255.0
        g_mean_high = float(np.mean(g[hl_mask])) / 255.0
        b_mean_high = float(np.mean(b[hl_mask])) / 255.0
    else:
        r_mean_high = g_mean_high = b_mean_high = 0.5

    # Sky-ROI metering. A bright sky in the upper band drags the whole-frame
    # exposure meter (its clipping inflates the white point and raises the mean)
    # and pulls the white balance toward sky blue, so the road/foreground grades
    # too dark or too warm. Meter the foreground by excluding the sky: bright
    # pixels — above THIS frame's own median — sitting in the upper band. The
    # measured area share (sky_fraction) travels with the foreground stats so the
    # correction downstream is weighted by how much sky the frame actually holds:
    # a frame with little sky is barely affected, a frame with none is untouched.
    # The upper-band ROI assumes landscape framing — a dashcam horizon puts sky
    # and only sky in the top band. A portrait still breaks that assumption (its
    # bright top is usually the subject: a face against sky, a building top), so
    # a taller-than-wide frame skips sky exclusion and meters whole-frame. Video
    # frames are always landscape samples, so the guard never fires for video.
    sky_mask = np.zeros(luminance.shape, dtype=bool)
    upper_rows = int(luminance.shape[0] * SKY_ROI_UPPER_FRACTION)
    if upper_rows >= 1 and luminance.shape[1] >= luminance.shape[0]:
        sky_mask[:upper_rows] = luminance[:upper_rows] > mid_tone
    fg_mask = ~sky_mask
    sky_fraction = float(np.mean(sky_mask))
    if sky_fraction > 0.0 and bool(fg_mask.any()):
        fg_lum = luminance[fg_mask]
        fg_white_point = float(np.percentile(fg_lum, 95))
        fg_mid_tone = float(np.median(fg_lum))
        fg_r_mean = float(np.mean(r[fg_mask])) / 255.0
        fg_g_mean = float(np.mean(g[fg_mask])) / 255.0
        fg_b_mean = float(np.mean(b[fg_mask])) / 255.0
    else:
        sky_fraction = 0.0
        fg_white_point, fg_mid_tone = white_point, mid_tone
        fg_r_mean, fg_g_mean, fg_b_mean = r_mean, g_mean, b_mean

    return FrameStats(
        black_point=black_point,
        white_point=white_point,
        mid_tone=mid_tone,
        saturation=saturation,
        r_mean=r_mean,
        g_mean=g_mean,
        b_mean=b_mean,
        r_mean_high=r_mean_high,
        g_mean_high=g_mean_high,
        b_mean_high=b_mean_high,
        highlight_frac=highlight_frac,
        sky_fraction=sky_fraction,
        fg_white_point=fg_white_point,
        fg_mid_tone=fg_mid_tone,
        fg_r_mean=fg_r_mean,
        fg_g_mean=fg_g_mean,
        fg_b_mean=fg_b_mean,
        detail_score=measure_detail(frame),
        noise_score=measure_noise(frame),
    )


# ── Segment Aggregation ──────────────────────────────────────────────


def _lighting_feature(s) -> np.ndarray:
    """Normalized [black, white, mid, saturation] vector for a frame or segment.

    The one distance metric the whole module grades by: span splitting,
    file clustering and the short-event guard all measure lighting difference
    in this space, so "different enough to need its own LUT" means the same
    thing everywhere.
    """
    return np.array(
        [s.black_point / 255.0, s.white_point / 255.0, s.mid_tone / 255.0, s.saturation]
    )


# Two lighting states farther apart than this need separate corrections — it is
# the span-splitting threshold, reused as the short-event guard so the guard
# cannot be tuned independently of the thing it feeds.
SPAN_SIMILARITY_THRESHOLD = 0.3


def aggregate_segment_stats(frames: list[FrameStats]) -> SegmentStats:
    """
    Aggregate multiple frame stats into a single segment summary.

    Uses median for robustness against outlier frames (e.g., flash, tunnel entry).

    A lighting event shorter than SAMPLE_INTERVAL_SEC shows up in a minority of
    a segment's samples, and the median discards it outright — a short tunnel or
    an overpass got smeared into the surrounding grade. When some sample sits
    farther from the median than SPAN_SIMILARITY_THRESHOLD (i.e. far enough that
    span splitting would refuse to merge the two), aggregate by mean instead so
    the excursion survives. The substitution is self-limiting by construction:
    the mean moves by the excursion's time share, so a 1-of-3 tunnel shifts the
    segment by a third of its delta while a rare outlier in a long aggregate
    barely moves it at all. Ordinary noise and scene drift stay below the
    threshold and keep the robust median.
    """
    if not frames:
        return SegmentStats(black_point=16, white_point=235, mid_tone=128, saturation=0.3)

    def _build(reduce) -> SegmentStats:
        return SegmentStats(
            black_point=float(reduce([f.black_point for f in frames])),
            white_point=float(reduce([f.white_point for f in frames])),
            mid_tone=float(reduce([f.mid_tone for f in frames])),
            saturation=float(reduce([f.saturation for f in frames])),
            r_mean=float(reduce([f.r_mean for f in frames])),
            g_mean=float(reduce([f.g_mean for f in frames])),
            b_mean=float(reduce([f.b_mean for f in frames])),
            r_mean_high=float(reduce([f.r_mean_high for f in frames])),
            g_mean_high=float(reduce([f.g_mean_high for f in frames])),
            b_mean_high=float(reduce([f.b_mean_high for f in frames])),
            highlight_frac=float(reduce([f.highlight_frac for f in frames])),
            sky_fraction=float(reduce([f.sky_fraction for f in frames])),
            fg_white_point=float(reduce([f.fg_white_point for f in frames])),
            fg_mid_tone=float(reduce([f.fg_mid_tone for f in frames])),
            fg_r_mean=float(reduce([f.fg_r_mean for f in frames])),
            fg_g_mean=float(reduce([f.fg_g_mean for f in frames])),
            fg_b_mean=float(reduce([f.fg_b_mean for f in frames])),
            detail_score=float(reduce([f.detail_score for f in frames])),
            noise_score=float(reduce([f.noise_score for f in frames])),
        )

    median = _build(np.median)
    center = _lighting_feature(median)
    if any(
        float(np.linalg.norm(_lighting_feature(f) - center)) >= SPAN_SIMILARITY_THRESHOLD
        for f in frames
    ):
        return _build(np.mean)
    return median


# Chroma-variance calibration. The white-balance correction clamps its
# per-channel cast at ±WB_CAST_CLAMP, so cast spread is measured in that same
# clamped space: two segments already pinned to the rail get the SAME white
# balance and must contribute no variance. Spread at/above the clamp means the
# two ends of the day would be graded at opposite rails — one averaged white
# balance mis-serves both. Below WB_CAST_NOISE_FLOOR the spread is ordinary
# sensor/scene noise (measured ~0.011 on a synthetic steady-illuminant day) and
# contributes exactly zero, so normal footage cannot creep into adaptive mode.
WB_CAST_CLAMP = 0.05
WB_CAST_NOISE_FLOOR = 0.01
# Weight chosen so a genuine cast swing trips the 0.3 threshold on its own
# (overcast→golden-hour scores ~0.38-0.50) while foliage-vs-road scene drift
# under one illuminant lands ~0.16 and steady light ~0.01 — a ~3.7x margin
# between "real cast change" and the worst realistic false positive.
CHROMA_VARIANCE_WEIGHT = 0.5


def _cast_spread(segments: list[SegmentStats]) -> float:
    """Spread of the white balance these segments would actually receive.

    Mirrors ``compute_correction``'s cast math (channel mean minus the green
    reference, clamped) and returns the magnitude of the per-axis standard
    deviations. Measuring the CLAMPED cast — the correction that would really
    be applied — rather than the raw channel means keeps content-driven
    outliers that are already saturated from inflating the score.
    """
    r_cast = np.clip(
        np.array([s.r_mean - s.g_mean for s in segments]), -WB_CAST_CLAMP, WB_CAST_CLAMP
    )
    b_cast = np.clip(
        np.array([s.b_mean - s.g_mean for s in segments]), -WB_CAST_CLAMP, WB_CAST_CLAMP
    )
    return float(np.hypot(np.std(r_cast), np.std(b_cast)))


def compute_variance(segments: list[SegmentStats]) -> float:
    """
    Compute normalized variance across segments to detect lighting changes.

    Combines luminance variation (black/white/mid points) with colour-cast
    variation, because a day can hold steady luma while its white balance
    swings hard — overcast into golden hour, or in and out of sodium-lit
    tunnels. A luma-only score leaves those days in single-correction mode and
    ships ONE averaged white balance across visibly different lighting.

    Returns a 0-1 score where:
      < 0.3 = stable lighting (use global correction)
      >= 0.3 = variable lighting (use adaptive per-segment)
    """
    if len(segments) < 2:
        return 0.0

    # Normalize each metric to 0-1 range then compute coefficient of variation
    black_points = np.array([s.black_point for s in segments])
    white_points = np.array([s.white_point for s in segments])
    mid_tones = np.array([s.mid_tone for s in segments])

    def _cv(arr: np.ndarray) -> float:
        """Coefficient of variation, bounded to [0, 1]."""
        mean = np.mean(arr)
        if mean < 1.0:
            return 0.0
        return min(float(np.std(arr) / mean), 1.0)

    # Weight mid_tone variance highest (most perceptually relevant)
    cv_black = _cv(black_points)
    cv_white = _cv(white_points)
    cv_mid = _cv(mid_tones)

    luma_score = 0.3 * cv_black + 0.3 * cv_white + 0.4 * cv_mid

    # Additive by design: the chroma term can only RAISE the score, so every
    # day that trips adaptive on luma today keeps tripping it.
    spread = _cast_spread(segments)
    normalized = (spread - WB_CAST_NOISE_FLOOR) / (WB_CAST_CLAMP - WB_CAST_NOISE_FLOOR)
    chroma_score = CHROMA_VARIANCE_WEIGHT * min(max(normalized, 0.0), 1.0)

    return min(luma_score + chroma_score, 1.0)


def should_use_adaptive(variance: float, threshold: float = 0.3) -> bool:
    """Decide whether to use per-segment adaptive correction."""
    return variance >= threshold


# ── Frame Sampling ───────────────────────────────────────────────────

SAMPLE_INTERVAL_SEC = 10  # Sample one frame every N seconds
# 30s segments catch tunnel/cloud transitions (5–30s timescale). Per the
# measured calibration, the legacy 45s value smeared transitions; 30s with
# 1.0s xfade stitching keeps boundary YAVG delta at/below JND (~1.0 luma
# unit). Below 15s the per-segment median collapses to 1–2 samples and
# is no longer robust at the current SAMPLE_INTERVAL_SEC=10.
SEGMENT_DURATION_SEC = 30  # Aggregate frames into segments of N seconds


def get_video_properties(video_path: Path) -> tuple[float, int, int, float]:
    """
    Get video properties via ffprobe.

    Returns:
        Tuple of (duration_sec, height, width, fps)
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-select_streams",
        "v:0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return (3600.0, 1080, 1920, 30.0)

    data = safe_json_loads(result.stdout, "ffprobe video properties (color analysis)")
    duration = float(data.get("format", {}).get("duration", 3600))

    streams = data.get("streams", [])
    if not streams:
        return (duration, 1080, 1920, 30.0)

    stream = streams[0]
    height = int(stream.get("height", 1080))
    width = int(stream.get("width", 1920))

    fps_str = stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str)

    return (duration, height, width, fps)


def sample_frames(
    video_path: Path,
    duration: float,
    interval: float = SAMPLE_INTERVAL_SEC,
    has_nvenc: bool = False,
    source_range: str = "full",
    tonemap_filter: str = "",
) -> list[np.ndarray]:
    """
    Sample frames from video at regular intervals using a single ffmpeg process.

    Uses fps filter to extract frames in one batch rather than spawning
    N separate ffmpeg processes. Falls back to sequential on failure.

    ``source_range`` (see ``scale_range_token``) tags the frames before the
    rgb24 hop so every histogram statistic feeding ``compute_correction`` is
    measured in the domain the grade is later applied in.

    ``tonemap_filter`` (an HDR source's BT.709 tonemap chain) runs first so the
    statistics come from the post-tonemap SDR frames the grade will act on.

    Returns list of numpy arrays (H, W, 3) in uint8.
    """
    if duration <= 0:
        return []

    num_samples = max(1, int(duration / interval))
    # Fixed output dimensions — eliminates height-guessing from raw bytes.
    # Color analysis only needs representative samples, not source resolution.
    w, h = 160, 90

    # ── Batch mode: single ffmpeg process with fps filter ──
    try:
        hwaccel_args: list[str] = []
        if has_nvenc:
            hwaccel_args = ["-hwaccel", "cuda"]

        cmd = [
            "ffmpeg",
            *hwaccel_args,
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{interval},{tonemap_filter + ',' if tonemap_filter else ''}"
            f"setrange={source_range},scale={w}:{h}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-v",
            "quiet",
            "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wall-clock heartbeat: ffmpeg emits no progress on stderr (-v quiet)
        # and we cannot use -progress pipe:1 because stdout is consumed by the
        # rawvideo decoder. A daemon thread logging every _HEARTBEAT_INTERVAL_SEC
        # gives operators a "still alive" signal without touching the data path.
        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=_emit_color_analysis_heartbeat,
            args=(heartbeat_stop, time.monotonic()),
            name="color-analysis-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()

        try:
            try:
                raw_data, _ = proc.communicate(timeout=max(60, duration * 0.5))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()  # reap zombie
                raise
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5.0)

        if proc.returncode == 0 and raw_data:
            frame_size = w * h * 3
            total_bytes = len(raw_data)
            if total_bytes >= frame_size:
                frames = []
                for offset in range(0, total_bytes - frame_size + 1, frame_size):
                    chunk = raw_data[offset : offset + frame_size]
                    frame = np.frombuffer(chunk, dtype=np.uint8).reshape(h, w, 3)
                    frames.append(frame)

                logger.info(f"  Sampled {len(frames)}/{num_samples} frames (batch)")
                return frames

    except Exception as e:
        logger.debug(f"  Batch frame sampling failed: {e}, falling back to sequential")

    # ── Sequential fallback ──
    # ``source_range`` MUST forward: a fallback that drops the tag re-creates
    # the exact domain error on the path taken when batch mode fails.
    return _sample_frames_sequential(
        video_path, duration, interval, source_range=source_range, tonemap_filter=tonemap_filter
    )


def _sample_frames_sequential(
    video_path: Path,
    duration: float,
    interval: float = SAMPLE_INTERVAL_SEC,
    source_range: str = "full",
    tonemap_filter: str = "",
) -> list[np.ndarray]:
    """
    Sequential per-frame sampling fallback.

    Spawns one ffmpeg process per sample frame. Slower but more robust.
    ``source_range`` carries the same meaning as in ``sample_frames``.
    """
    num_samples = max(1, int(duration / interval))
    timestamps = [i * interval for i in range(num_samples)]
    # Fixed output dimensions — matches batch mode
    w, h = 160, 90

    frames = []
    for ts in timestamps:
        try:
            cmd = [
                "ffmpeg",
                "-ss",
                f"{ts:.2f}",
                "-i",
                str(video_path),
                "-vframes",
                "1",
                "-vf",
                f"{tonemap_filter + ',' if tonemap_filter else ''}"
                f"setrange={source_range},scale={w}:{h}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-v",
                "quiet",
                "-",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=15,
            )
            if result.returncode == 0 and result.stdout:
                raw = result.stdout
                if len(raw) == w * h * 3:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                    frames.append(frame)

        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"  Failed to sample frame at {ts:.1f}s: {e}")
            continue

    logger.info(f"  Sampled {len(frames)}/{num_samples} frames (sequential)")
    return frames


def analyze_segments(
    frames: list[np.ndarray],
    segment_size: Optional[int] = None,
) -> tuple[list[SegmentStats], bool]:
    """
    Analyze frames grouped into segments.

    Returns:
        Tuple of (segment_stats_list, is_adaptive)
    """
    if not frames:
        return ([], False)

    if segment_size is None:
        # How many frames per segment
        segment_size = max(1, SEGMENT_DURATION_SEC // SAMPLE_INTERVAL_SEC)

    # Analyze each frame
    frame_stats = [analyze_frame(f) for f in frames]

    # Group into segments
    segments = []
    for i in range(0, len(frame_stats), segment_size):
        chunk = frame_stats[i : i + segment_size]
        segments.append(aggregate_segment_stats(chunk))

    variance = compute_variance(segments)
    is_adaptive = should_use_adaptive(variance)

    logger.info(f"  Analyzed {len(segments)} segments, variance={variance:.3f}")
    logger.info(f"  Mode: {'adaptive (per-segment)' if is_adaptive else 'global (uniform)'}")

    return (segments, is_adaptive)


def merge_adjacent_segments(
    segments: list[SegmentStats],
    segment_duration: float = float(SEGMENT_DURATION_SEC),
    similarity_threshold: float = SPAN_SIMILARITY_THRESHOLD,
    max_spans: int = 12,
    total_duration: float = 0.0,
) -> tuple[list[SegmentStats], list[float]]:
    """Merge consecutive segments with similar lighting into LUT spans.

    A 5h day master at SEGMENT_DURATION_SEC=30 yields ~600 segments; baking
    one LUT per segment is untenable. This folds consecutive similar
    segments (same feature-distance approach as ``cluster_files_by_lighting``,
    constrained to temporally adjacent runs) into a handful of spans, each
    of which gets one LUT in the adaptive filtergraph.

    Returns:
        (span_stats, span_durations) — median-aggregated stats per span and
        each span's duration in seconds (member count * segment_duration).
        When ``total_duration`` (the real media length) is given, the final
        span is clamped to the real remaining duration so a partial trailing
        segment can't inflate the last span past EOF (the adaptive
        filtergraph's last xfade would otherwise run off the end).
    """
    if not segments:
        return ([], [])

    def _centroid(run: list[SegmentStats]) -> np.ndarray:
        return np.mean([_lighting_feature(s) for s in run], axis=0)

    runs: list[list[SegmentStats]] = [[segments[0]]]
    for seg in segments[1:]:
        if float(np.linalg.norm(_lighting_feature(seg) - _centroid(runs[-1]))) <= (
            similarity_threshold
        ):
            runs[-1].append(seg)
        else:
            runs.append([seg])

    # Pathological oscillating lighting: fold away the WEAKEST boundary — the
    # adjacent pair whose lighting is most alike — until the span count is
    # encodable. Raising the threshold globally instead (the old behaviour) let
    # one oscillating stretch coarsen the entire day, so hours that were merging
    # cleanly lost precision to pay for a busy afternoon. Merging weakest-first
    # keeps the coarsening local to the region that overflowed, and still never
    # exceeds max_spans, so encode cost is unchanged.
    while len(runs) > max_spans:
        centroids = [_centroid(r) for r in runs]
        gaps = [
            float(np.linalg.norm(centroids[i + 1] - centroids[i])) for i in range(len(runs) - 1)
        ]
        i = int(np.argmin(gaps))
        runs[i : i + 2] = [runs[i] + runs[i + 1]]

    span_stats = [
        SegmentStats(
            black_point=float(np.median([s.black_point for s in r])),
            white_point=float(np.median([s.white_point for s in r])),
            mid_tone=float(np.median([s.mid_tone for s in r])),
            saturation=float(np.median([s.saturation for s in r])),
            r_mean=float(np.median([s.r_mean for s in r])),
            g_mean=float(np.median([s.g_mean for s in r])),
            b_mean=float(np.median([s.b_mean for s in r])),
            r_mean_high=float(np.median([s.r_mean_high for s in r])),
            g_mean_high=float(np.median([s.g_mean_high for s in r])),
            b_mean_high=float(np.median([s.b_mean_high for s in r])),
            highlight_frac=float(np.median([s.highlight_frac for s in r])),
            sky_fraction=float(np.median([s.sky_fraction for s in r])),
            fg_white_point=float(np.median([s.fg_white_point for s in r])),
            fg_mid_tone=float(np.median([s.fg_mid_tone for s in r])),
            fg_r_mean=float(np.median([s.fg_r_mean for s in r])),
            fg_g_mean=float(np.median([s.fg_g_mean for s in r])),
            fg_b_mean=float(np.median([s.fg_b_mean for s in r])),
        )
        for r in runs
    ]
    durations = [len(r) * segment_duration for r in runs]
    # A partial final segment makes the nominal (count * segment_duration) last
    # span longer than the real remaining media. Clamp it so sum(durations)
    # equals the true media length and the filtergraph lands the last xfade
    # inside the footage instead of past EOF.
    if total_duration > 0 and durations:
        durations[-1] = total_duration - sum(durations[:-1])
    return (span_stats, durations)


# ── Drive Pipeline: Color Clustering ──────────────────────────────────

# Chroma weight for the file-cluster feature. Per-channel mean deltas live in
# a ~±0.15 band while the luma terms span 0-1, so unweighted chroma could never
# cross the cluster threshold on its own — golden hour and blue hour at equal
# luminance would share one averaged white balance. 2.0 puts a strongly opposed
# cast pair (~±0.1 per channel) past the default 0.3 threshold while same-scene
# chroma wobble (~±0.02) stays negligible.
CLUSTER_CHROMA_WEIGHT = 2.0


def _cluster_feature(s: SegmentStats) -> np.ndarray:
    """File-cluster feature: the shared lighting vector plus a weighted cast.

    The cast terms are the r-g and b-g deltas of the per-channel means, so two
    files whose luminance matches but whose white balance opposes still land
    in separate clusters. Neutral defaults (0.5 each) contribute zero, so
    stats without measured channel means cluster exactly as before.
    """
    return np.concatenate(
        (
            _lighting_feature(s),
            (
                CLUSTER_CHROMA_WEIGHT * (s.r_mean - s.g_mean),
                CLUSTER_CHROMA_WEIGHT * (s.b_mean - s.g_mean),
            ),
        )
    )


def cluster_files_by_lighting(
    file_stats: list[tuple[Path, SegmentStats]],
    variance_threshold: float = 0.3,
) -> list[list[int]]:
    """Group files into clusters with similar lighting conditions.

    Distance is measured in ``_cluster_feature`` space (lighting plus chroma
    cast); files farther apart than the threshold go into separate clusters.

    Files are processed in a canonical order — sorted by feature vector, never
    input order — because greedy centroid assignment is order-dependent and a
    card's enumeration order is arbitrary: the same file set in any order
    yields the same clusters.

    Args:
        file_stats: List of (path, SegmentStats) per file, in order.
        variance_threshold: Maximum feature-space distance within a cluster
            (0-1 scale on the luma axes, where 1.0 = full 255 range).

    Returns:
        List of clusters, each a list of file indices — indices ascending
        within a cluster, clusters ordered by their first member.
    """
    if not file_stats:
        return []

    n = len(file_stats)
    if n == 1:
        return [[0]]

    features = np.array([_cluster_feature(s) for _, s in file_stats])

    # Simple agglomerative clustering: greedily assign to nearest existing
    # cluster centroid, or start a new cluster if distance exceeds threshold.
    order = sorted(range(n), key=lambda i: tuple(features[i]))
    clusters: list[list[int]] = []
    centroids: list[np.ndarray] = []

    for i in order:
        best_cluster = -1
        best_dist = float("inf")

        for c_idx, centroid in enumerate(centroids):
            dist = float(np.linalg.norm(features[i] - centroid))
            if dist < best_dist:
                best_dist = dist
                best_cluster = c_idx

        if best_cluster >= 0 and best_dist <= variance_threshold:
            clusters[best_cluster].append(i)
            # Update centroid as running mean
            members = clusters[best_cluster]
            centroids[best_cluster] = np.mean(features[members], axis=0)
        else:
            clusters.append([i])
            centroids.append(features[i].copy())

    for cluster in clusters:
        cluster.sort()
    clusters.sort(key=lambda c: c[0])
    return clusters
