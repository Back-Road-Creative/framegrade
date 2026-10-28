"""
Pipeline Shorts Extractor — wraps existing motion analysis + FFmpeg rendering.

Takes a processed long form video, identifies high-motion segments using
the existing ShortsExtractor, then renders each as a 1080x1920 vertical
short with center crop, OST overlays, and NO audio (muted — near-silent
operator policy; see encoding_constants.py).


Usage:
    from gomoveshift.pipeline.shorts_extractor import PipelineShortsExtractor

    extractor = PipelineShortsExtractor()
    result = extractor.extract_and_render(
        video_path="/path/to/long_form_FINAL.mp4",
        output_dir="/path/to/output/shorts",
        location="County Tipperary, Ireland",
        num_shorts=9,
    )
"""

import hashlib
import json
import logging
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from keycut import keyframe_at_or_before, probe_keyframes

from ..config import Config, get_config
from .color import detect_nvenc
from .encoding_constants import (
    SHORTS_BUFSIZE,
    SHORTS_MAXRATE,
    SHORTS_PIX_FMT,
    SHORTS_PROFILE,
)
from .ffmpeg_filters import (
    DEFAULT_CRF,
    DEFAULT_LUFS,
    DEFAULT_OUTPUT_HEIGHT,
    DEFAULT_OUTPUT_WIDTH,
    DEFAULT_PRESET,
    _crop_size,
    build_crop_filter,
    build_ost_filter,
    get_video_dimensions,
    shorten_ost_for_shorts,
)
from .reframe import plan_tracked_crop

logger = logging.getLogger(__name__)

# ── Batching constants ───────────────────────────────────────────────────
DEFAULT_BATCH_SIZE = 5  # segments per stream-copy batch

# ── Module-level NVENC cache ─────────────────────────────────────────────
_nvenc_available: Optional[bool] = None


def _detect_nvenc_cached() -> bool:
    """Check NVENC availability once and cache the result."""
    global _nvenc_available
    if _nvenc_available is None:
        _nvenc_available = detect_nvenc()
        logger.info(f"NVENC detection: {'available' if _nvenc_available else 'not available'}")
    return _nvenc_available


def _short_ost_fingerprint(
    location: str, subscribe_text: str, start_time: float, end_time: float
) -> str:
    """Stable hash of the material render inputs for one short.

    Persisted next to ``short_NN_rankM.mp4`` as ``short_NN_rankM.ost.sha256``
    after a successful render (mirrors ``_lf_ost_fingerprint`` /
    ``lf_NN.ost.sha256`` in day_pipeline). On resume the cache pre-filter
    reuses a rendered short only when the stored fingerprint still matches —
    a corrected location override mismatches and re-renders instead of
    shipping the old WRONG burned text.

    The clip's WINDOW (``start_time``/``end_time``) is part of the hash so a
    re-planned short at the SAME rank/index but a DIFFERENT (possibly parked)
    window re-renders instead of reusing the old render of the prior window.
    """
    raw = (
        f"{location or ''}\x00{subscribe_text or ''}"
        f"\x00{float(start_time):.4f}:{float(end_time):.4f}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _short_fp_path(output_path: Path) -> Path:
    """Sidecar fingerprint path for a rendered short (``*.ost.sha256``)."""
    return output_path.with_suffix(".ost.sha256")


@dataclass
class RenderedShort:
    """A fully rendered short video."""

    index: int
    rank: int
    start_time: float
    end_time: float
    duration: float
    avg_motion_energy: float
    peak_motion_energy: float
    source_video: str
    output_path: str
    location_text: str
    subscribe_text: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "rank": self.rank,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "avg_motion_energy": round(self.avg_motion_energy, 6),
            "peak_motion_energy": round(self.peak_motion_energy, 6),
            "source_video": self.source_video,
            "output_path": self.output_path,
            "location_text": self.location_text,
            "subscribe_text": self.subscribe_text,
        }


@dataclass
class ShortsExtractionResult:
    """Result of the full extraction + rendering pipeline."""

    source_video: str
    output_dir: str
    num_requested: int
    num_rendered: int
    shorts: List[RenderedShort]
    errors: List[str]
    processing_time: float

    def to_dict(self) -> dict:
        return {
            "source_video": self.source_video,
            "output_dir": self.output_dir,
            "num_requested": self.num_requested,
            "num_rendered": self.num_rendered,
            "shorts": [s.to_dict() for s in self.shorts],
            "errors": self.errors,
            "processing_time": round(self.processing_time, 2),
        }

    def save(self, path: Path) -> None:
        """Save results to JSON.

        Before writing, reconciles ``num_rendered`` with the actual
        ``short_*.mp4`` file count in ``output_dir``. Prior runs may have
        left extra files that the current analysis didn't iterate over,
        causing the manifest to under-count and fail DoD check 11.
        """
        out = Path(self.output_dir)
        actual_count = (
            sum(
                1
                for f in out.iterdir()
                if f.is_file() and f.name.startswith("short_") and f.name.endswith(".mp4")
            )
            if out.is_dir()
            else self.num_rendered
        )
        data = self.to_dict()
        if actual_count != data["num_rendered"]:
            data["num_rendered"] = actual_count
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# FFmpeg utilities imported from gomoveshift.pipeline.ffmpeg_filters:
# escape_ffmpeg_text, get_video_dimensions, build_crop_filter,
# build_ost_filter, wrap_text_for_ffmpeg, and all rendering constants.


def _probe_video_stream_duration(path: Path) -> float:
    """Probe video stream duration via ffprobe.

    Returns the video stream's own duration (not format/container duration).
    Used to guard shorts extraction against A/V drift in the source LF.

    Returns:
        Video stream duration in seconds, or 0.0 on error.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration",
        "-print_format",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return 0.0
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if streams:
            return float(streams[0].get("duration", 0))
        return 0.0
    except Exception:
        return 0.0


def reframe_kwargs(config) -> dict:
    """``{"reframe": "tracked"}`` when ``--short-reframe tracked`` is on, else ``{}``.

    Empty by default so every short render call receives exactly what it did before.
    """
    mode = getattr(getattr(config, "drive", None), "short_reframe", "center")
    return {} if mode == "center" else {"reframe": mode}


def render_short_ffmpeg(
    source_video: Path,
    output_path: Path,
    start_time: float,
    duration: float,
    location: str,
    subscribe_text: str = "Subscribe for more",
    src_width: Optional[int] = None,
    src_height: Optional[int] = None,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    target_lufs: float = DEFAULT_LUFS,
    skip_audio_norm: bool = False,
    skip_audio: bool = False,
    use_nvenc: Optional[bool] = None,
    reframe: str = "center",
) -> bool:
    """
    Render a single short with center crop, OST, and optional audio normalization.

    This is the core FFmpeg rendering function used by the pipeline.
    Combines patterns from render_shorts_ffmpeg.py and create_shorts_ffmpeg_full.py.

    Args:
        source_video: Path to processed long form video
        output_path: Where to write the rendered short
        start_time: Segment start in seconds
        duration: Segment duration in seconds
        location: Location text for OST
        subscribe_text: CTA text for OST
        src_width: Source video width (auto-detected if None)
        src_height: Source video height (auto-detected if None)
        crf: NVENC CQ quality (lower = better)
        preset: NVENC encoding preset (p1–p7)
        target_lufs: Target loudness in LUFS
        skip_audio_norm: Skip loudnorm filter (default False — always normalize
            since source audio is not always pre-normalized)
        use_nvenc: Use NVENC GPU encoding. None = auto-detect.
        reframe: "center" (default, fixed centre crop) or "tracked" (opt-in:
            the crop follows the road with a dead zone and a pan-speed cap;
            see ``reframe.py``). Crop size and scaler are the same either way.

    Returns:
        True if rendering succeeded
    """
    source_video = Path(source_video)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect NVENC if not specified
    if use_nvenc is None:
        use_nvenc = _detect_nvenc_cached()

    # Get source dimensions if not provided
    if src_width is None or src_height is None:
        src_width, src_height = get_video_dimensions(source_video)

    if reframe not in ("center", "tracked"):
        raise ValueError(f"reframe must be 'center' or 'tracked', got {reframe!r}")
    track = None
    if reframe == "tracked":
        try:
            track = plan_tracked_crop(
                source_video,
                start_time,
                duration,
                src_width,
                src_height,
                _crop_size(src_width, src_height, DEFAULT_OUTPUT_WIDTH, DEFAULT_OUTPUT_HEIGHT)[0],
            )
        except (RuntimeError, OSError, subprocess.SubprocessError) as e:
            logger.warning(
                "tracked reframe unavailable for %s (%s) — using the centre crop",
                output_path.name,
                e,
            )

    # Build filter chain: crop → OST
    if track is None:
        crop_filter = build_crop_filter(src_width, src_height)
    else:
        crop_filter = build_crop_filter(src_width, src_height, track=track)
    ost_filter = build_ost_filter(location, subscribe_text, duration)

    video_filter = f"{crop_filter},{ost_filter}"

    cmd = ["ffmpeg", "-y"]

    # Hardware-accelerated decode when NVENC available
    if use_nvenc:
        cmd.extend(["-hwaccel", "cuda"])

    cmd.extend(
        [
            "-ss",
            str(start_time),
            "-i",
            str(source_video),
            "-t",
            str(duration),
            "-vf",
            video_filter,
        ]
    )

    # Audio: shorts ALWAYS ship muted. The
    # operator's near-silent rule is absolute for shorts, and vertical platforms
    # (TikTok/IG) may re-normalize quiet audio back UP on their side — a stripped
    # stream cannot be undone, so `-an` is the robust choice (see encoding_
    # constants.py). `skip_audio` / `skip_audio_norm` / `target_lufs` are retained
    # in the signature for back-compat but no longer gate audio: there is no path
    # that keeps a short's audio. The `-an` flag is added at the encode below.

    # Color metadata: limited-range BT.709 for YouTube/social SDR
    shorts_color_flags = [
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
    ]

    # hevc_nvenc only — _detect_nvenc_cached() raises
    # NVENCUnavailableError on probe failure, so reaching here implies
    # NVENC is available. The libx265 branch is deleted.
    # Quality matches the graded long-form master (p7 / CQ 18 / 10-bit
    # main10) via encoding_constants — the prior hardcoded p5 / CQ 23 /
    # 8-bit yuv420p visibly banded graded skies.
    # The ENCODER CONFIGURATION matches too: the
    # lookahead / adaptive-QP / B-frame-ref / two-pass flags below mirror
    # color_filters._nvenc_args. Without them "quality matches long-form"
    # held only for preset, CQ, profile and pixel format — the shorts encode
    # ran single-pass with no lookahead and no adaptive quantisation, which
    # is exactly where graded skies and shadow texture lose bits.
    cmd.extend(
        [
            "-c:v",
            "hevc_nvenc",
            "-preset",
            preset,
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(crf),
            "-profile:v",
            SHORTS_PROFILE,
            "-pix_fmt",
            SHORTS_PIX_FMT,
            "-rc-lookahead",
            "32",  # 32-frame lookahead for rate allocation
            "-spatial-aq",
            "1",  # Adaptive QP based on spatial complexity
            "-temporal-aq",
            "1",  # Adaptive QP based on temporal complexity
            "-b_ref_mode",
            "2",  # Use B-frames as references (middle)
            "-bf",
            "4",  # 4 B-frames for better inter-frame prediction
            "-multipass",
            "2",  # Full two-pass (qres+fullres) within NVENC
            "-maxrate",
            SHORTS_MAXRATE,
            "-bufsize",
            SHORTS_BUFSIZE,
            *shorts_color_flags,
        ]
    )

    # Near-silent policy: shorts are always muted — no audio stream is encoded.
    cmd.extend(["-an", "-movflags", "+faststart", str(output_path)])

    logger.debug(f"FFmpeg command: {' '.join(cmd)}")

    # NVENC-only path → 600s suffices (no CPU 1800s branch).
    timeout = 600

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timed out rendering {output_path.name}")
        return False

    if result.returncode != 0:
        # Capture the full stderr and exact command line so the next diagnostic
        # session has raw data (no truncation, no masking via retries). Written
        # to <output_parent>/_short_failures/<stem>.stderr.log and .cmd.log.
        try:
            failures_dir = output_path.parent / "_short_failures"
            failures_dir.mkdir(parents=True, exist_ok=True)
            stderr_log = failures_dir / f"{output_path.stem}.stderr.log"
            cmd_log = failures_dir / f"{output_path.stem}.cmd.log"
            stderr_log.write_text(result.stderr or "")
            cmd_log.write_text(" ".join(str(a) for a in cmd) + "\n")
        except OSError as e:
            logger.warning(f"Could not write short failure logs for {output_path.name}: {e}")

        logger.error(
            "FFmpeg failed rendering %s (full stderr at %s)",
            output_path.name,
            output_path.parent / "_short_failures" / f"{output_path.stem}.stderr.log",
        )
        return False

    if not output_path.exists() or output_path.stat().st_size == 0:
        logger.error(f"FFmpeg produced empty output: {output_path}")
        return False

    # Validate the rendered MP4 is a playable file (catches moov atom missing, etc.)
    from .ffmpeg_runner import validate_video

    # Shorts are ALWAYS muted (-an) under the near-silent policy, so a missing
    # audio stream is expected — never require one here (would hard-fail every
    # short). Intent-derived: shorts are mute-by-design.
    vr = validate_video(output_path, expected_duration=duration, require_audio=False)
    if not vr.valid:
        logger.error(f"Rendered MP4 is CORRUPT: {output_path.name} — {vr.reason}")
        output_path.unlink(missing_ok=True)
        return False

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Rendered {output_path.name}: {size_mb:.1f} MB ({vr.duration:.1f}s)")
    return True


def _keyframe_lead_in(start_time: float, keyframes: Optional[list]) -> float:
    """Seconds a stream copy at ``start_time`` would replay from before it.

    ``-ss t -i src -c copy`` cannot start on a non-keyframe, so it lands on the
    keyframe at or before ``t`` and replays everything in between. Returning
    that distance makes the displacement explicit: the caller extends the copied
    range by it and trims it back off in the (accurately-seeking) re-encode.

    Returns ``0.0`` when the keyframe positions are unknown or ``start_time``
    precedes the first keyframe — the pre-existing behaviour, never a guess.
    """
    if not keyframes:
        return 0.0
    kf = keyframe_at_or_before(start_time, keyframes)
    if kf is None:
        return 0.0
    return max(0.0, start_time - kf)


def _probe_source_keyframes(source_video: Path) -> Optional[list]:
    """Keyframe timestamps for the batch's source, or ``None`` when unknown.

    Wrapped so that a probe that raises (rather than returning ``None``) still
    degrades to the pre-existing snap-back behaviour instead of aborting a
    whole batch of shorts.
    """
    try:
        return probe_keyframes(source_video)
    except Exception as e:  # noqa: BLE001 — a probe must never abort the batch
        logger.warning("Keyframe probe raised for %s (%s)", source_video.name, e)
        return None


def extract_segment_streamcopy(
    source_video: Path,
    output_path: Path,
    start_time: float,
    duration: float,
) -> bool:
    """
    Extract a segment from the source video via stream copy (no re-encode).

    This is extremely fast — it just copies the compressed bitstream for the
    requested time range. The output is a rough-cut segment that still needs
    encoding with crop/OST/audio filters, but it avoids re-reading the large
    source file during the expensive encode step.

    ``start_time`` should be a KEYFRAME timestamp. A stream copy cannot start
    anywhere else, so an arbitrary value is silently snapped back to the
    previous keyframe and the emitted segment does not begin where the caller
    asked. ``render_shorts_batched`` computes that keyframe with
    ``_keyframe_lead_in``; a caller passing an unaligned time gets the legacy
    snap-back.

    Args:
        source_video: Path to the large source video
        output_path: Where to write the stream-copied segment
        start_time: Segment start in seconds (keyframe-aligned by the caller)
        duration: Segment duration in seconds

    Returns:
        True if extraction succeeded
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time),
        "-i",
        str(source_video),
        "-t",
        str(duration),
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    logger.debug(f"Stream-copy extract: {output_path.name}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        logger.error(f"Stream-copy timed out for {output_path.name}")
        return False

    if result.returncode != 0:
        logger.error(f"Stream-copy failed for {output_path.name}: {result.stderr[-300:]}")
        return False

    if not output_path.exists() or output_path.stat().st_size == 0:
        logger.error(f"Stream-copy produced empty output: {output_path}")
        return False

    return True


def render_shorts_batched(
    source_video: Path,
    shorts_specs: List[dict],
    output_dir: Path,
    src_width: int,
    src_height: int,
    use_nvenc: bool = False,
    skip_audio: bool = False,
    max_workers: int = 3,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    reframe: str = "center",
) -> dict[int, bool]:
    """
    Batch-render shorts using a two-phase approach to reduce source reads.

    Phase 1 — Segment extraction:
        Stream-copy all requested segments from the large source video.
        Each segment becomes a small intermediate file (~20-50 MB vs 2-3 GB source).
        This is I/O-bound and very fast (seconds per segment).

    Phase 2 — Parallel encode:
        Encode each extracted segment with crop, OST, and audio normalization.
        Each encode reads its small segment file instead of the full source.

    Compared to individual rendering (15 x full-source reads), this approach:
    - Reads the source ~15 times via stream-copy (fast: no decode, just byte copy)
    - But the expensive decode+encode reads only small segment files
    - Net result: ~10-15x less I/O during the CPU-heavy encode phase

    Args:
        source_video: Path to the large source video
        shorts_specs: List of dicts with keys:
            index, start_time, duration, location, subscribe_text, output_path
        output_dir: Directory for output and intermediate files
        src_width: Source video width
        src_height: Source video height
        use_nvenc: Use GPU encoding
        skip_audio: Skip audio track
        max_workers: Parallel encode workers
        batch_size: Not used in current impl (kept for API compatibility)
        progress_callback: Optional callback(completed_count, total_count)
        reframe: 9:16 crop placement forwarded to ``render_short_ffmpeg``

    Returns:
        Dict mapping short index -> success (True/False)
    """
    source_video = Path(source_video)
    output_dir = Path(output_dir)
    segments_dir = output_dir / "_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, bool] = {}
    total = len(shorts_specs)

    if total == 0:
        return results

    # Guard: probe the video stream duration and drop any shorts that
    # extend beyond it.  When an LF has A/V drift (audio longer than
    # video), shorts near the end would get zero video frames → encoder
    # EOF.  This check prevents that failure mode.
    video_dur = _probe_video_stream_duration(source_video)
    if video_dur > 0:
        valid_specs = []
        for spec in shorts_specs:
            end = spec["start_time"] + spec["duration"]
            if end > video_dur + 1.0:  # 1s tolerance for keyframe snap
                logger.warning(
                    "Short #%d extends past video stream (%.1fs > %.1fs) — skipping",
                    spec["index"],
                    end,
                    video_dur,
                )
                results[spec["index"]] = False
            else:
                valid_specs.append(spec)
        if len(valid_specs) < len(shorts_specs):
            logger.warning(
                "%d/%d shorts exceed video stream duration (%.1fs)",
                len(shorts_specs) - len(valid_specs),
                len(shorts_specs),
                video_dur,
            )
        shorts_specs = valid_specs
        total = len(shorts_specs)

    # Keyframe positions for the whole batch, probed once. A stream copy can
    # only start on a keyframe, so without this every segment silently began up
    # to one GOP (~2s) before the scored moment and lost the same off its tail.
    # `None` means the probe failed — the legacy snap-back, logged not silent.
    keyframes = _probe_source_keyframes(source_video)
    if not keyframes:
        logger.warning(
            "Keyframe positions unknown for %s — batched shorts fall back to "
            "stream-copy snap-back, so each clip may start up to one GOP before "
            "its scored moment",
            source_video.name,
        )
    lead_ins: dict[int, float] = {
        spec["index"]: _keyframe_lead_in(spec["start_time"], keyframes) for spec in shorts_specs
    }

    # ── Phase 1: Stream-copy extract all segments ────────────────────────
    logger.info(f"Batch render phase 1: extracting {total} segments via stream-copy...")
    segment_paths: dict[int, Path] = {}
    extract_failures: list[int] = []

    for spec in shorts_specs:
        idx = spec["index"]
        seg_path = segments_dir / f"segment_{idx:02d}.mp4"
        lead_in = lead_ins[idx]
        # Start ON the keyframe the copy would have snapped to anyway, and add
        # the lead-in to the duration so the scored window survives intact.
        ok = extract_segment_streamcopy(
            source_video=source_video,
            output_path=seg_path,
            start_time=spec["start_time"] - lead_in,
            duration=spec["duration"] + lead_in,
        )
        if ok:
            segment_paths[idx] = seg_path
        else:
            extract_failures.append(idx)
            results[idx] = False
            logger.warning(f"  Segment extract failed for short #{idx}, will skip encode")

    if extract_failures:
        logger.warning(
            f"Phase 1 complete: {len(segment_paths)}/{total} segments extracted "
            f"({len(extract_failures)} failed)"
        )
    else:
        logger.info(f"Phase 1 complete: all {total} segments extracted")

    # ── Phase 2: Encode segments in parallel ─────────────────────────────
    logger.info(f"Batch render phase 2: encoding {len(segment_paths)} segments...")
    completed_count = len(extract_failures)

    def _encode_segment(spec: dict) -> tuple[int, bool]:
        idx = spec["index"]
        seg_path = segment_paths.get(idx)
        if seg_path is None:
            return idx, False

        final_path = Path(spec["output_path"])

        # Encode from the small segment file (not the full source). The segment
        # starts on the keyframe at or before the scored moment, so seek past
        # the lead-in here: this pass re-encodes, and an input-side `-ss` on a
        # re-encode is frame-accurate (unlike the stream copy that produced the
        # segment). The lead-in is 0.0 when the keyframe probe was unavailable.
        success = render_short_ffmpeg(
            source_video=seg_path,
            output_path=final_path,
            start_time=lead_ins[idx],
            duration=spec["duration"],
            location=spec["location"],
            subscribe_text=spec["subscribe_text"],
            src_width=src_width,
            src_height=src_height,
            use_nvenc=use_nvenc,
            skip_audio=skip_audio,
            **({} if reframe == "center" else {"reframe": reframe}),
        )
        return idx, success

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        specs_to_encode = [s for s in shorts_specs if s["index"] in segment_paths]
        futures = {executor.submit(_encode_segment, spec): spec for spec in specs_to_encode}

        for future in as_completed(futures):
            idx, success = future.result()
            results[idx] = success
            completed_count += 1

            if progress_callback:
                progress_callback(completed_count, total)

            if not success:
                logger.error(f"  Batch encode failed for short #{idx}")

    # ── Cleanup: remove intermediate segment files ───────────────────────
    for seg_path in segment_paths.values():
        try:
            seg_path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        segments_dir.rmdir()
    except OSError:
        pass  # Not empty or already gone

    succeeded = sum(1 for v in results.values() if v)
    logger.info(
        f"Batch render complete: {succeeded}/{total} shorts rendered ({total - succeeded} failed)"
    )
    return results


def _derive_short_location_map(
    shorts,
    gpx_path: Optional[Path],
    keep_segments: Optional[list[tuple[float, float]]],
    gps_duration: float,
    original_duration: float,
    config: Config,
) -> dict[int, str]:
    """Per-clip location labels for legacy-path shorts.

    Mirrors the day-first derivation: map each short's edited-video window back
    to GPS time (inverting stop removal via ``VideoToGpsTimeMapper``), take the
    trackpoints on screen during the clip, and vote the modal county through the
    shared geocoder (``gps_enrichment._make_short_ost_text_fn``) — so an
    end-of-drive short is never labeled with the start-of-drive town. A clip
    whose derivation fails (no fixes / geocode quorum miss) is absent from the
    map; the render-site OST gate cascades it to the drive-wide label, matching
    the day-first ``_gate_short_ost_text`` contract. Returns ``{}`` when the GPX
    or the geocoder is unavailable (operator-``--location`` fallback).
    """
    if gpx_path is None or not Path(gpx_path).exists():
        return {}
    try:
        from ..analysis.gps_parser import parse_gpx_file
        from ..analysis.shorts_models import effective_offset, footage_origin
        from ..analysis.shorts_time_mapping import VideoToGpsTimeMapper
        from .drive_phases.phase_bc_day_first import _forward_geocode_places
        from .gps_enrichment import _build_shared_geocoder, _make_short_ost_text_fn

        gps_data = parse_gpx_file(Path(gpx_path))
        if not gps_data.trackpoints:
            return {}
        geocoder = _build_shared_geocoder(str(config.drive.resolved_geocode_cache_dir()))
        if geocoder is None:
            return {}
        # Skip a leading cold-start anomaly (null-island / stale lock) so
        # a short's GPS window opens at the first real fix, not the garbage lead.
        mapper = VideoToGpsTimeMapper(
            gps_offset=effective_offset(gps_data, gps_data.start_time or 0.0),
            keep_segments=keep_segments,
            gps_duration=gps_duration,
            original_duration=original_duration,
            crossfade_duration=config.stop_detection.crossfade_duration,
            footage_clock=footage_origin(gps_data) is not None,
        )
        ost_text_fn = _make_short_ost_text_fn(geocoder, places_fn=_forward_geocode_places)
        location_map: dict[int, str] = {}
        for short in shorts:
            gps_t0 = mapper.video_to_gps_time(short.start_time)
            gps_t1 = mapper.video_to_gps_time(short.end_time)
            pts = [
                (tp.timestamp, tp.lat, tp.lon)
                for tp in gps_data.get_points_in_range(gps_t0, gps_t1)
            ]
            label = ost_text_fn(pts, short)
            if label:
                location_map[short.index] = label
        return location_map
    except Exception as exc:  # noqa: BLE001 — derivation must never kill the render
        logger.warning("per-short location derivation failed: %s", exc)
        return {}


class PipelineShortsExtractor:
    """
    Full pipeline for extracting and rendering shorts from long form video.

    Wraps the existing ShortsExtractor for motion analysis and uses
    FFmpeg for vertical rendering with OST + audio normalization.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        min_duration: float = 30.0,
        max_duration: float = 45.0,
        target_duration: float = 38.0,
        min_gap_between_shorts: float = 60.0,
    ):
        self.config = config or get_config()
        # Enforce minimum 30 seconds for YouTube Shorts compliance
        self.min_duration = max(min_duration, 30.0)
        self.max_duration = max_duration
        self.target_duration = target_duration
        self.min_gap = min_gap_between_shorts

    def extract_and_render(
        self,
        video_path: str,
        output_dir: str,
        location: str,
        num_shorts: int = 9,
        subscribe_text: str = "Subscribe for more",
        render: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        motion_cache_dir: Optional[Path] = None,
        max_workers: int = 5,
        gpx_path: Optional[Path] = None,
        keep_segments: Optional[list[tuple[float, float]]] = None,
        gps_duration: float = 0.0,
        original_duration: float = 0.0,
        route_pois: Optional[list[dict]] = None,
        location_map: Optional[dict[int, str]] = None,
    ) -> ShortsExtractionResult:
        """
        Extract high-motion segments and render as vertical shorts.

        Args:
            video_path: Path to processed long form video
            output_dir: Directory for rendered shorts
            location: Location text for OST overlay
            num_shorts: Number of shorts to extract
            subscribe_text: CTA text for OST overlay
            render: If True, render shorts. If False, only identify segments.
            progress_callback: Optional callback(stage, progress)

        Returns:
            ShortsExtractionResult with all rendered shorts
        """
        t0 = time.time()
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        errors: List[str] = []
        rendered_shorts: List[RenderedShort] = []

        # ── Step 1: Motion analysis using existing ShortsExtractor ────
        from ..analysis.shorts_extractor import ExtractionResult, ShortsExtractor
        from ..analysis.shorts_models import SHORTS_SELECTOR_VERSION
        from .cache import is_analysis_cached, load_cached_json

        analysis_path = output_dir / "shorts_analysis.json"

        # Check if analysis is already cached and fresh. Keyed on the selector
        # version so a selection-rule change (stop-aware windows) re-analyzes
        # instead of shipping pre-fix single-video shorts windows.
        if is_analysis_cached(video_path, analysis_path, expected_version=SHORTS_SELECTOR_VERSION):
            logger.info(f"Loading cached motion analysis from {analysis_path.name}")
            cached_data = load_cached_json(analysis_path)
            if cached_data:
                try:
                    extraction_result = ExtractionResult.load(analysis_path)
                    logger.info(
                        f"Loaded {extraction_result.num_shorts_found} cached segments "
                        f"(skipping {extraction_result.source_duration:.0f}s analysis)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load cached analysis: {e}")
                    extraction_result = None
            else:
                extraction_result = None
        else:
            extraction_result = None

        if extraction_result is None:
            logger.info(f"Analyzing {video_path.name} for high-motion segments...")

            if progress_callback:
                progress_callback("analyzing", 0.0)

            extractor = ShortsExtractor(
                min_duration=self.min_duration,
                max_duration=self.max_duration,
                target_duration=self.target_duration,
                min_gap_between_shorts=self.min_gap,
            )

            # Wire GPS data for scenic scoring if available
            if gpx_path and gpx_path.exists():
                try:
                    from ..analysis.shorts_extractor import parse_gpx_file

                    gps_data = parse_gpx_file(gpx_path)
                    from ..analysis.shorts_models import effective_offset, footage_origin

                    gps_start = effective_offset(gps_data, gps_data.start_time or 0.0)
                    extractor.set_gps_data(gps_data, gps_offset=gps_start)
                    on_footage_clock = footage_origin(gps_data) is not None
                    if keep_segments and (
                        on_footage_clock or (gps_duration > 0 and original_duration > 0)
                    ):
                        extractor.set_time_mapping(keep_segments, gps_duration, original_duration)
                    # Set POI timestamps for proximity scoring
                    if route_pois and extractor._scenic_scorer:
                        poi_gps_times = []
                        for poi in route_pois:
                            # POI lat/lon → find closest GPS trackpoint time
                            for tp in gps_data.trackpoints[::10]:  # sample every 10th
                                from ..pipeline.chapters import _haversine

                                if _haversine(poi["lat"], poi["lon"], tp.lat, tp.lon) < 500:
                                    poi_gps_times.append(tp.timestamp)
                                    break
                        if poi_gps_times:
                            extractor._scenic_scorer.set_poi_timestamps(poi_gps_times)
                            logger.info(
                                f"Set {len(poi_gps_times)} POI timestamps for scenic scoring"
                            )
                except Exception as e:
                    logger.warning(f"GPS data wiring failed: {e}")

            extraction_result = extractor.extract_shorts(
                video_path=str(video_path),
                num_shorts=num_shorts,
                output_dir=str(output_dir / "_analysis"),
                render_clips=False,  # We do our own rendering
                progress_callback=progress_callback,
            )

            logger.info(
                f"Found {extraction_result.num_shorts_found} segments from "
                f"{extraction_result.source_duration:.0f}s video "
                f"(avoided {extraction_result.dead_spots_avoided} dead spots)"
            )

            # Save analysis results
            extraction_result.save(analysis_path)

        if not extraction_result.shorts:
            logger.warning("No suitable shorts segments found!")
            return ShortsExtractionResult(
                source_video=str(video_path),
                output_dir=str(output_dir),
                num_requested=num_shorts,
                num_rendered=0,
                shorts=[],
                errors=["No suitable shorts segments found"],
                processing_time=time.time() - t0,
            )

        if not render:
            # Return analysis-only result
            shorts_data = [
                RenderedShort(
                    index=s.index,
                    rank=s.rank,
                    start_time=s.start_time,
                    end_time=s.end_time,
                    duration=s.duration,
                    avg_motion_energy=s.avg_motion_energy,
                    peak_motion_energy=s.peak_motion_energy,
                    source_video=str(video_path),
                    output_path="",
                    location_text=location,
                    subscribe_text=subscribe_text,
                )
                for s in extraction_result.shorts
            ]
            return ShortsExtractionResult(
                source_video=str(video_path),
                output_dir=str(output_dir),
                num_requested=num_shorts,
                num_rendered=0,
                shorts=shorts_data,
                errors=[],
                processing_time=time.time() - t0,
            )

        # ── Step 1.6: per-clip labels + OST burn gate ──
        # When the caller supplied no location_map, derive one from the GPX so
        # each short burns ITS OWN footage's location instead of inheriting the
        # drive-wide label. Every label (derived or fallback) then passes the
        # day-first _gate_short_ost_text cascade (shorten → validate_ost →
        # hard-fit → drive-location fallback), which hard-fails rather than
        # burning garbage into the irreversible 9:16 encode.
        if location_map is None:
            location_map = _derive_short_location_map(
                extraction_result.shorts,
                gpx_path,
                keep_segments,
                gps_duration,
                original_duration,
                self.config,
            )
        from .day_pipeline import _gate_short_ost_text  # runtime import: day_pipeline imports us

        gated_labels: dict[int, str] = {}
        for short in extraction_result.shorts:
            candidate = location_map.get(short.index) if location_map else None
            gated_labels[short.index] = _gate_short_ost_text(
                candidate,
                None,
                short.rank,
                video_path.stem,
                drive_location=location,
            )

        # ── Step 2: Get source video dimensions ──────────────────────
        src_width, src_height = get_video_dimensions(video_path)
        logger.info(f"Source video: {src_width}x{src_height}")

        # ── Step 3: Render each short as vertical ─────────────────────
        total_shorts = len(extraction_result.shorts)
        logger.info(f"Rendering {total_shorts} shorts as 1080x1920 vertical...")

        from .cache import is_output_fresh

        # Detect NVENC once before the loop
        use_nvenc = _detect_nvenc_cached()
        effective_workers = max_workers if use_nvenc else min(max_workers, 2)

        # Pre-filter: check cache sequentially (filesystem reads).
        # Freshness is keyed on the burned OST text via an .ost.sha256
        # sidecar (mirrors day_pipeline's lf_NN.ost.sha256) — mtime alone
        # would reuse a short whose location text was since corrected,
        # shipping the old WRONG burned text on resume.
        cached_shorts = []
        pending_shorts = []
        for short in extraction_result.shorts:
            output_name = f"short_{short.index:02d}_rank{short.rank}.mp4"
            output_path = output_dir / output_name
            # Effective burned text: per-short override > drive-wide, then
            # shortened — must match the pending-spec construction below.
            short_location = location
            if location_map and short.index in location_map:
                short_location = location_map[short.index]
            short_location = shorten_ost_for_shorts(short_location)
            fp_path = _short_fp_path(output_path)
            if (
                is_output_fresh(video_path, output_path, min_size_bytes=100 * 1024)
                and fp_path.is_file()
                and fp_path.read_text().strip()
                == _short_ost_fingerprint(
                    short_location, subscribe_text, short.start_time, short.end_time
                )
            ):
                logger.info(f"  Short #{short.index} — already rendered (OST unchanged), skipping")
                cached_shorts.append((short, output_path, short_location))
            else:
                # Drop any stale fingerprint so a failed re-render can't
                # leave a matching sidecar next to a wrong/partial output.
                fp_path.unlink(missing_ok=True)
                pending_shorts.append((short, output_path))

        skipped_count = len(cached_shorts)

        # Add cached shorts to results — stamp the effective per-short OST
        # text (override-applied), not the drive-wide default, so resumed
        # runs report the text actually burned into the reused file.
        for short, output_path, short_location in cached_shorts:
            rendered_shorts.append(
                RenderedShort(
                    index=short.index,
                    rank=short.rank,
                    start_time=short.start_time,
                    end_time=short.end_time,
                    duration=short.duration,
                    avg_motion_energy=short.avg_motion_energy,
                    peak_motion_energy=short.peak_motion_energy,
                    source_video=str(video_path),
                    output_path=str(output_path),
                    location_text=gated_labels[short.index],
                    subscribe_text=subscribe_text,
                )
            )

        # Build specs for pending shorts
        pending_specs = []
        pending_map = {}  # index -> (short, output_path)
        for short, output_path in pending_shorts:
            short_location = gated_labels[short.index]

            logger.info(
                f"  Short #{short.index} "
                f"({short.start_time:.1f}s - {short.end_time:.1f}s, "
                f"rank={short.rank}, loc={short_location})"
            )
            pending_specs.append(
                {
                    "index": short.index,
                    "start_time": short.start_time,
                    "end_time": short.end_time,
                    "duration": short.duration,
                    "location": short_location,
                    "subscribe_text": subscribe_text,
                    "output_path": str(output_path),
                }
            )
            pending_map[short.index] = (short, output_path)

        # Try batched rendering first, fall back to individual on failure
        batch_results = None
        if pending_specs:
            try:
                completed_count = skipped_count

                def _batch_progress(done: int, total: int) -> None:
                    nonlocal completed_count
                    completed_count = skipped_count + done
                    if progress_callback:
                        progress_callback("rendering", completed_count / total_shorts)

                batch_results = render_shorts_batched(
                    source_video=video_path,
                    shorts_specs=pending_specs,
                    output_dir=output_dir,
                    src_width=src_width,
                    src_height=src_height,
                    use_nvenc=use_nvenc,
                    skip_audio=self.config.drive.skip_audio,
                    max_workers=effective_workers,
                    progress_callback=_batch_progress,
                    **reframe_kwargs(self.config),
                )
            except Exception as e:
                logger.warning(f"Batched rendering failed ({e}), falling back to individual")
                batch_results = None

        # If batched rendering succeeded (even partially), collect results
        # and fall back to individual for any failures
        fallback_specs = []
        if batch_results is not None:
            for spec in pending_specs:
                idx = spec["index"]
                short, output_path = pending_map[idx]
                if batch_results.get(idx):
                    # Persist the burned-OST fingerprint so a later resume
                    # can detect this short is fresh and skip the re-render.
                    _short_fp_path(output_path).write_text(
                        _short_ost_fingerprint(
                            spec["location"],
                            spec["subscribe_text"],
                            spec["start_time"],
                            spec["end_time"],
                        )
                    )
                    rendered_shorts.append(
                        RenderedShort(
                            index=short.index,
                            rank=short.rank,
                            start_time=short.start_time,
                            end_time=short.end_time,
                            duration=short.duration,
                            avg_motion_energy=short.avg_motion_energy,
                            peak_motion_energy=short.peak_motion_energy,
                            source_video=str(video_path),
                            output_path=str(output_path),
                            location_text=spec["location"],
                            subscribe_text=subscribe_text,
                        )
                    )
                else:
                    fallback_specs.append(spec)
        else:
            # Full fallback — batched rendering not attempted or crashed
            fallback_specs = pending_specs

        # Individual fallback for any shorts that failed batched rendering
        if fallback_specs:
            if batch_results is not None:
                logger.info(
                    f"Falling back to individual rendering for {len(fallback_specs)} shorts"
                )
            completed_count = skipped_count + (len(pending_specs) - len(fallback_specs))
            results_lock = threading.Lock()

            def _render_one(spec):
                idx = spec["index"]
                short, output_path = pending_map[idx]
                success = render_short_ffmpeg(
                    source_video=video_path,
                    output_path=output_path,
                    start_time=short.start_time,
                    duration=short.duration,
                    location=spec["location"],
                    subscribe_text=spec["subscribe_text"],
                    src_width=src_width,
                    src_height=src_height,
                    use_nvenc=use_nvenc,
                    skip_audio=self.config.drive.skip_audio,
                    **reframe_kwargs(self.config),
                )
                if success:
                    # Persist the burned-OST fingerprint (see batched path).
                    _short_fp_path(output_path).write_text(
                        _short_ost_fingerprint(
                            spec["location"],
                            spec["subscribe_text"],
                            spec["start_time"],
                            spec["end_time"],
                        )
                    )
                return short, output_path, success

            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {executor.submit(_render_one, spec): spec for spec in fallback_specs}
                for future in as_completed(futures):
                    short, output_path, success = future.result()
                    completed_count += 1

                    if progress_callback:
                        progress_callback("rendering", completed_count / total_shorts)

                    if success:
                        with results_lock:
                            rendered_shorts.append(
                                RenderedShort(
                                    index=short.index,
                                    rank=short.rank,
                                    start_time=short.start_time,
                                    end_time=short.end_time,
                                    duration=short.duration,
                                    avg_motion_energy=short.avg_motion_energy,
                                    peak_motion_energy=short.peak_motion_energy,
                                    source_video=str(video_path),
                                    output_path=str(output_path),
                                    location_text=gated_labels[short.index],
                                    subscribe_text=subscribe_text,
                                )
                            )
                    else:
                        error_msg = f"Failed to render short #{short.index}"
                        with results_lock:
                            errors.append(error_msg)
                        logger.error(f"  {error_msg}")

        # Sort rendered shorts by index for consistent output
        rendered_shorts.sort(key=lambda s: s.index)

        # ── Step 4: Save results ──────────────────────────────────────
        result = ShortsExtractionResult(
            source_video=str(video_path),
            output_dir=str(output_dir),
            num_requested=num_shorts,
            num_rendered=len(rendered_shorts),
            shorts=rendered_shorts,
            errors=errors,
            processing_time=time.time() - t0,
        )

        result_path = output_dir / "shorts_rendered.json"
        result.save(result_path)

        logger.info(
            f"Shorts extraction complete: {result.num_rendered}/{num_shorts} rendered "
            f"({skipped_count} cached, skipped) "
            f"in {result.processing_time:.1f}s"
        )
        if errors:
            logger.warning(f"  Errors: {errors}")

        return result
