"""
Weather detection per video segment via brightness analysis.

Categorizes segments as rainy/overcast/sunny based on average frame brightness.
Used to generate weather-aware social posts and schedule decisions.


Thresholds:
  - brightness < 400  → rainy
  - 400 ≤ brightness ≤ 580 → overcast
  - brightness > 580  → sunny

Usage:
    from gomoveshift.pipeline.weather import WeatherDetector

    detector = WeatherDetector()
    weather = detector.detect_segment(video_path, start_time=30.0, end_time=60.0)
    # weather = WeatherResult(condition='sunny', brightness=612.3, confidence=0.85)
"""

import logging
import math
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from .exceptions import WeatherMeasurementError

logger = logging.getLogger(__name__)


# Process-level cache of ffmpeg capability probe results. Populated on
# first call to ``ffmpeg_supports_cuda_decode`` so the per-chunk hot path
# does not re-shell ffmpeg. Reset in tests via ``reset_ffmpeg_capability_cache``.
_FFMPEG_CAPABILITY_CACHE: dict[str, object] = {}


def reset_ffmpeg_capability_cache() -> None:
    """Clear the cached ffmpeg capability probe (tests only)."""
    _FFMPEG_CAPABILITY_CACHE.clear()


_YAVG_RE = re.compile(
    r"""
    (?:                                # legacy ffmpeg < 5.x emitted
      \bYAVG:\s*(?P<old>-?\d+(?:\.\d+)?)   # `[Parsed_signalstats_*] YAVG:128.00`
    )
    |
    (?:                                # modern ffmpeg (≥ 5.x) emits frame
      \blavfi\.signalstats\.YAVG\s*=   # metadata via the `metadata=print`
      \s*(?P<new>-?\d+(?:\.\d+)?)      # filter: `lavfi.signalstats.YAVG=132.295`
    )
    """,
    re.VERBOSE,
)


def _parse_yavg_lines(stderr: str) -> list[float]:
    """Extract YAVG values from an ffmpeg stderr buffer.

    Accepts both the legacy ``YAVG:128.00`` shape (ffmpeg < 5.x) and the
    modern ``lavfi.signalstats.YAVG=128.00`` metadata shape (ffmpeg ≥ 5.x
    via the ``metadata=print`` filter). Returns values in encounter order;
    invalid floats are skipped, not raised — the chunk aggregator decides
    whether the surviving count is enough.
    """
    if not stderr:
        return []
    values: list[float] = []
    for match in _YAVG_RE.finditer(stderr):
        raw = match.group("old") or match.group("new")
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return values


def ffmpeg_supports_cuda_decode(decoder: Optional[str] = None) -> bool:
    """Return True iff the active ``ffmpeg`` build advertises both
    ``cuda`` in ``-hwaccels`` and the requested cuvid ``decoder`` (or any
    cuvid decoder when ``decoder`` is None) in ``-decoders``.

    Result is cached for the process. ffmpeg missing or probe failure
    returns False — callers decide whether that is a hard fail (preflight
    raises) or a soft fall-through (``hwaccel="none"`` software path).
    """
    cached = _FFMPEG_CAPABILITY_CACHE.get("probed")
    if not cached:
        hwaccels: set[str] = set()
        decoders: set[str] = set()
        try:
            hw_res = subprocess.run(
                ["ffmpeg", "-hide_banner", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in (hw_res.stdout or "").splitlines():
                name = line.strip()
                if name and not name.endswith(":"):
                    hwaccels.add(name.lower())
            dec_res = subprocess.run(
                ["ffmpeg", "-hide_banner", "-decoders"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in (dec_res.stdout or "").splitlines():
                parts = line.strip().split()
                # Format: " V..... hevc_cuvid          Nvidia CUVID HEVC decoder (codec hevc)"
                for tok in parts:
                    if tok.endswith("_cuvid"):
                        decoders.add(tok)
                        break
        except (subprocess.SubprocessError, FileNotFoundError):
            hwaccels = set()
            decoders = set()
        _FFMPEG_CAPABILITY_CACHE["hwaccels"] = hwaccels
        _FFMPEG_CAPABILITY_CACHE["decoders"] = decoders
        _FFMPEG_CAPABILITY_CACHE["probed"] = True

    hwaccels = _FFMPEG_CAPABILITY_CACHE.get("hwaccels", set())  # type: ignore[assignment]
    decoders = _FFMPEG_CAPABILITY_CACHE.get("decoders", set())  # type: ignore[assignment]
    if "cuda" not in hwaccels:
        return False
    if decoder is None:
        return bool(decoders)
    return decoder in decoders


class WeatherCondition(str, Enum):
    """Weather condition detected from brightness."""

    RAINY = "rainy"
    OVERCAST = "overcast"
    SUNNY = "sunny"


# Default thresholds from project spec. NOTE: brightness
# alone cannot verify weather, so `_classify` no longer asserts a specific
# condition from it — the dark band (< RAINY) and the bright band (> SUNNY) both
# now report OVERCAST (see `_classify`), not RAINY/SUNNY. These constants mark
# the light-level band edges only; the RAINY and SUNNY enum members are retained
# for compatibility but are never emitted from brightness.
DEFAULT_RAINY_THRESHOLD = 400.0
DEFAULT_SUNNY_THRESHOLD = 580.0


@dataclass
class WeatherResult:
    """Weather detection result for a single segment."""

    condition: WeatherCondition
    brightness: float
    confidence: float  # 0-1 how far from threshold boundaries
    segment_start: float
    segment_end: float

    @property
    def emoji(self) -> str:
        """Emoji representation for social posts."""
        return {
            WeatherCondition.RAINY: "🌧️",
            WeatherCondition.OVERCAST: "☁️",
            WeatherCondition.SUNNY: "☀️",
        }[self.condition]

    @property
    def adjective(self) -> str:
        """Human-readable adjective for social posts."""
        return {
            WeatherCondition.RAINY: "rainy",
            WeatherCondition.OVERCAST: "overcast",
            WeatherCondition.SUNNY: "sunny",
        }[self.condition]

    @property
    def mood(self) -> str:
        """Mood descriptor for social posts."""
        return {
            WeatherCondition.RAINY: "moody",
            WeatherCondition.OVERCAST: "atmospheric",
            WeatherCondition.SUNNY: "beautiful",
        }[self.condition]

    def to_dict(self) -> dict:
        return {
            "condition": self.condition.value,
            "brightness": round(self.brightness, 1),
            "confidence": round(self.confidence, 3),
            "segment_start": self.segment_start,
            "segment_end": self.segment_end,
            "emoji": self.emoji,
        }


class WeatherDetector:
    """
    Detects weather conditions from video segment brightness.

    Uses FFmpeg signalstats filter to measure average brightness across
    sampled frames in a segment, then classifies by threshold.
    """

    def __init__(
        self,
        rainy_threshold: float = DEFAULT_RAINY_THRESHOLD,
        sunny_threshold: float = DEFAULT_SUNNY_THRESHOLD,
        sample_frames: int = 10,
        chunk_max_sec: float = 120.0,
        min_success_ratio: float = 0.5,
        chunk_timeout_cap_sec: float = 180.0,
        hwaccel: str = "auto",
    ):
        """
        Args:
            rainy_threshold: Brightness below this = rainy
            sunny_threshold: Brightness above this = sunny
            sample_frames: Number of frames to sample per segment
            chunk_max_sec: Maximum span in seconds for a single ffmpeg
                invocation. The full segment is split into chunks of at
                most this duration; each chunk gets its own timeout.
                Defaults to 120s — large enough that small segments stay
                one-shot, small enough that a 30-min Insta360 episode on
                slow IO (WSL2 NTFS) splits into ~15 chunks that each
                comfortably fit a 30-60s timeout.
            min_success_ratio: Minimum fraction of chunks that must
                successfully return YAVG data for the aggregate result
                to count as a measurement. Below this, the method
                returns ``None`` and logs ERROR.
        """
        if rainy_threshold >= sunny_threshold:
            raise ValueError(
                f"rainy_threshold ({rainy_threshold}) must be less than "
                f"sunny_threshold ({sunny_threshold})"
            )
        if not (0.0 < min_success_ratio <= 1.0):
            raise ValueError(f"min_success_ratio must be in (0, 1] (got {min_success_ratio})")
        if chunk_max_sec <= 0:
            raise ValueError(f"chunk_max_sec must be > 0 (got {chunk_max_sec})")
        if chunk_timeout_cap_sec <= 0:
            raise ValueError(f"chunk_timeout_cap_sec must be > 0 (got {chunk_timeout_cap_sec})")

        self.rainy_threshold = rainy_threshold
        self.sunny_threshold = sunny_threshold
        self.sample_frames = sample_frames
        self.chunk_max_sec = chunk_max_sec
        self.min_success_ratio = min_success_ratio
        self.chunk_timeout_cap_sec = chunk_timeout_cap_sec
        self.hwaccel = hwaccel
        # Per-detector cache so the codec probe runs once per source file
        # rather than once per chunk (a 30-min Insta360 file produces 15
        # chunks; without the cache that's 15 redundant ffprobe shellouts).
        # Sentinel ``_UNPROBED`` separates "not yet probed" from "probed,
        # not cuvid-mappable".
        self._decoder_for_path: dict[str, Optional[str]] = {}

    @staticmethod
    def _plan_chunks(
        start_time: float, duration: float, chunk_max_sec: float
    ) -> list[tuple[float, float]]:
        """Split ``[start_time, start_time + duration)`` into chunks of
        at most ``chunk_max_sec`` seconds each. Returns list of
        ``(chunk_start, chunk_duration)``."""
        if duration <= 0:
            return []
        if duration <= chunk_max_sec:
            return [(start_time, duration)]
        chunks: list[tuple[float, float]] = []
        t = start_time
        remaining = duration
        while remaining > 0:
            d = min(chunk_max_sec, remaining)
            chunks.append((t, d))
            t += d
            remaining -= d
        return chunks

    # ffmpeg codec name (from ffprobe ``codec_name``) → CUDA cuvid decoder.
    # Covers the codecs the pipeline actually ingests: HEVC (Insta360,
    # GoPro 10+), H.264 (legacy GoPro, dashcam), AV1 (forward-looking).
    # Unknown codecs fall back to software decode (no decoder flag).
    _CUVID_BY_CODEC = {
        "hevc": "hevc_cuvid",
        "h264": "h264_cuvid",
        "av1": "av1_cuvid",
    }

    def _cuvid_decoder_for(self, video_path: Path) -> Optional[str]:
        """Return the matching cuvid decoder for ``video_path``'s codec,
        or ``None`` when the codec isn't supported by NVDEC. Failures
        (ffprobe missing, codec probe errors) return ``None`` so the
        caller falls through to software decode rather than raising."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
        codec = (result.stdout or "").strip().lower()
        return self._CUVID_BY_CODEC.get(codec)

    def _measure_chunk(
        self,
        video_path: Path,
        start_time: float,
        duration: float,
        timeout_sec: float,
    ) -> list[float]:
        """Run a single ffmpeg signalstats pass for ``[start_time, +duration]``
        and return the list of per-sampled-frame YAVG values. Empty list
        on timeout or no data. Caller decides aggregation policy."""
        fps_rate = max(1, int(self.sample_frames / max(duration, 1)))
        # Weather measurement is GPU-accelerated when
        # available. The rest of the pipeline (BGR frame extraction in
        # unified_scorer, master encode, OST burn) uses `-hwaccel cuda
        # -c:v <codec>_cuvid`; weather was the lone software-decode path.
        # On 4K HEVC sources (Insta360, GoPro 12+) software signalstats
        # throughput drops below realtime, breaching the per-chunk
        # timeout cap and degrading the whole measurement. With CUDA
        # decode the same chunk runs at ~3.3× realtime — comfortably
        # under any reasonable cap.
        #
        # signalstats requires CPU-side frames, so we pair `-hwaccel cuda`
        # with an explicit cuvid decoder (NOT
        # `-hwaccel_output_format cuda`) so frames land in system memory
        # for the filter graph.
        cmd = ["ffmpeg"]
        # hwaccel selection contract:
        #   "none"  → no hwaccel flag at all (software decode, explicit).
        #   "cuda"  → require cuvid for this codec. Raise WeatherMeasurementError
        #             if codec is unmappable OR ffmpeg lacks cuda/cuvid support.
        #             NO silent software fallback: operator asked for CUDA.
        #   "auto"  → probe codec; if cuvid-mappable AND ffmpeg supports it,
        #             emit -hwaccel cuda -c:v <cuvid>. Otherwise software.
        #             The pre-fix bug: "auto" emitted bare `-hwaccel auto`
        #             with no -c:v decoder, leaving signalstats on software
        #             HEVC decode that times out on 4K Insta360 sources.
        if self.hwaccel and self.hwaccel != "none":
            key = str(video_path)
            if key in self._decoder_for_path:
                decoder = self._decoder_for_path[key]
            else:
                decoder = self._cuvid_decoder_for(video_path)
                self._decoder_for_path[key] = decoder
            if self.hwaccel == "cuda":
                if decoder is None:
                    raise WeatherMeasurementError(
                        f"hwaccel='cuda' requested but no cuvid decoder is "
                        f"mappable for {video_path.name} (codec probe failed "
                        f"or codec unsupported by NVDEC). Either install a "
                        f"cuvid-capable codec source or set "
                        f"config.drive.weather_hwaccel='auto'."
                    )
                if not ffmpeg_supports_cuda_decode(decoder):
                    raise WeatherMeasurementError(
                        f"hwaccel='cuda' requested but the active ffmpeg "
                        f"build does not advertise hwaccel=cuda and/or "
                        f"decoder {decoder}. Rebuild ffmpeg with "
                        f"--enable-cuda-llvm --enable-nvdec, or set "
                        f"config.drive.weather_hwaccel='none' to accept "
                        f"software decode."
                    )
                cmd += ["-hwaccel", "cuda", "-c:v", decoder]
            else:
                # "auto" (or any unrecognised opt-in value): probe + use cuvid
                # when both the codec and the ffmpeg build support it.
                if decoder is not None and ffmpeg_supports_cuda_decode(decoder):
                    cmd += ["-hwaccel", "cuda", "-c:v", decoder]
                # else: software decode (no -hwaccel flag). Codec is not
                # cuvid-mappable (e.g. prores) or ffmpeg lacks cuda — both
                # are legitimate auto-mode fall-throughs.
        cmd += [
            "-ss",
            str(start_time),
            "-i",
            str(video_path),
            "-t",
            str(duration),
            "-vf",
            # Filter chain contract:
            # signalstats writes its statistics as per-frame metadata under the
            # ``lavfi.signalstats.*`` namespace. Modern ffmpeg builds (≥ 5.x)
            # NO LONGER emit `[Parsed_signalstats_*] YAVG:xxx` formatted
            # lines on stderr; the legacy ``YAVG:`` substring parser silently
            # found zero matches and the chunk reported as degraded even
            # though decode succeeded at 3.3× realtime. Chaining
            # ``metadata=print`` forces those metadata keys onto stderr as
            # ``lavfi.signalstats.YAVG=<value>`` lines, which is the parser's
            # contract. This shape is stable across ffmpeg versions.
            f"fps={fps_rate},signalstats,metadata=print",
            "-an",  # discard audio — we only need the video frame metadata
            "-f",
            "null",
            "-",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                f"FFmpeg timed out ({timeout_sec}s) measuring brightness chunk "
                f"at {start_time}s+{duration}s on {video_path.name}"
            )
            return []

        yavg_values = _parse_yavg_lines(result.stderr)
        if not yavg_values and result.returncode == 0:
            # ffmpeg ran clean but the parser found zero YAVG samples. This
            # is NOT a slow-IO degradation — it's a filter-chain or parser
            # contract mismatch (e.g. the filter graph dropped frames, or
            # an ffmpeg upgrade changed the metadata key format). Surface
            # it loudly so the operator does not chase a non-existent IO
            # problem. The legacy code path silently returned [] here and
            # let the chunk aggregator misclassify as "degraded".
            raise WeatherMeasurementError(
                f"FFmpeg signalstats produced no YAVG samples for chunk "
                f"{start_time}s+{duration}s on {video_path.name} despite "
                f"a clean exit (returncode={result.returncode}). This "
                f"means the filter chain or metadata parser is out of "
                f"sync with the active ffmpeg build — check the "
                f"`signalstats,metadata=print` filter contract in "
                f"gomoveshift.pipeline.weather._measure_chunk and the "
                f"matching `_parse_yavg_lines` regex against `ffmpeg "
                f"-version` output. Tail of ffmpeg stderr (last 500 chars):\n"
                f"  {(result.stderr or '')[-500:]!r}"
            )
        return yavg_values

    def _get_avg_brightness_ffmpeg(
        self,
        video_path: Path,
        start_time: float,
        duration: float,
    ) -> Optional[float]:
        """
        Measure average brightness of a video segment using FFmpeg.

        The segment is split into chunks of at most ``self.chunk_max_sec``
        seconds; each chunk runs as its own ffmpeg invocation with a
        proportional timeout. This stops a single slow demux on a
        high-bitrate source (e.g. 27 GB Insta360 file via WSL2 NTFS)
        from blowing a 120s monolithic timeout and silently losing the
        whole weather signal.

        At least ``self.min_success_ratio`` of chunks must succeed for
        the aggregate to count as a measurement. Below the floor the
        method returns ``None`` and logs ERROR (not WARNING) so the
        downstream silent-overcast fallback is at least operator-
        visible.

        Returns:
            Brightness on 0-1000 scale, or None if too few chunks
            succeeded. Callers handle None.
        """
        chunks = self._plan_chunks(start_time, duration, self.chunk_max_sec)
        if not chunks:
            return None

        # Per-chunk timeout: 2x chunk duration, clamped to
        # [30s, self.chunk_timeout_cap_sec]. 30s floor: short chunks still
        # tolerate cold-cache demux jitter. Cap defaults to 180s — large
        # enough to absorb 4K HEVC software-decode fallback on a 120s
        # chunk when CUDA isn't available, configurable for hardware that
        # genuinely needs more wall budget.
        all_yavg: list[float] = []
        chunks_succeeded = 0
        for c_start, c_dur in chunks:
            c_timeout = max(30.0, min(self.chunk_timeout_cap_sec, c_dur * 2.0))
            yavg = self._measure_chunk(video_path, c_start, c_dur, c_timeout)
            if yavg:
                all_yavg.extend(yavg)
                chunks_succeeded += 1

        min_required = max(1, math.ceil(len(chunks) * self.min_success_ratio))
        if chunks_succeeded < min_required:
            logger.error(
                f"Weather brightness measurement degraded for segment "
                f"{start_time:.1f}-{start_time + duration:.1f}s on "
                f"{video_path.name}: only {chunks_succeeded}/{len(chunks)} "
                f"chunks returned data (need >= {min_required}). "
                "Downstream scorer will fall back to DEFAULT_WEATHER_CLASS — "
                "fix the source IO or raise "
                "config.drive.weather_chunk_max_sec if this repeats."
            )
            return None

        avg = sum(all_yavg) / len(all_yavg)
        scaled = avg * (1000.0 / 255.0)

        logger.debug(
            f"Segment {start_time:.1f}-{start_time + duration:.1f}s: "
            f"raw_yavg={avg:.1f}, scaled={scaled:.1f}, "
            f"frames_sampled={len(all_yavg)}, "
            f"chunks_ok={chunks_succeeded}/{len(chunks)}"
        )

        return scaled

    def _classify(self, brightness: float) -> tuple[WeatherCondition, float]:
        """
        Classify brightness into weather condition with confidence.

        Confidence is higher the further from threshold boundaries.

        Every branch returns
        OVERCAST, so this reports a LIGHT LEVEL, not weather. The composite
        scorer's 7% weather weight has been deleted rather than the classifier
        restored — a constant cannot rank episodes, and honest weather needs a
        weather API, not pixel brightness. Confidence still varies with
        distance from the thresholds, so callers that surface a light-level
        reading keep a usable number. Reinstating a real classifier means
        reinstating the weight in ``config.DriveConfig.scoring_weights``.
        """
        midpoint = (self.rainy_threshold + self.sunny_threshold) / 2.0
        range_half = (self.sunny_threshold - self.rainy_threshold) / 2.0

        if brightness < self.rainy_threshold:
            # Very-low-light band. Demoted: brightness CANNOT detect
            # precipitation (a tunnel, dusk, or dense overcast all read dark),
            # so this no longer claims RAINY — it reports OVERCAST (a light-level
            # statement, not a weather-as-fact claim).
            distance = self.rainy_threshold - brightness
            confidence = min(distance / self.rainy_threshold, 1.0)
            return WeatherCondition.OVERCAST, max(0.5, confidence)

        elif brightness > self.sunny_threshold:
            # High-brightness band. Demoted: brightness CANNOT verify
            # clear skies (snow glare, a bright-but-cloudy sky, or a washed-out
            # sensor all read bright), so this no longer claims SUNNY — it
            # reports OVERCAST (the same unremarkable-sky fallback as the dark
            # band). With both extremes demoted, `_classify` never asserts a
            # specific weather from pixel brightness alone. Note: ☁️/
            # "overcast" can now surface on genuinely-sunny footage; the clean
            # fix is to drop brightness-weather or verify via a weather API.
            distance = brightness - self.sunny_threshold
            confidence = min(distance / (1000.0 - self.sunny_threshold), 1.0)
            return WeatherCondition.OVERCAST, max(0.5, confidence)

        else:
            # Overcast — confidence is highest at midpoint
            if not range_half:
                # Thresholds were set equal post-construction (the __init__
                # guard can't see mutated attrs); no band to scale within.
                return WeatherCondition.OVERCAST, 0.5
            distance_from_midpoint = abs(brightness - midpoint)
            confidence = 1.0 - (distance_from_midpoint / range_half)
            return WeatherCondition.OVERCAST, max(0.3, confidence)

    def score_frame_brightness(
        self,
        video_path: Path,
        timestamp: float,
        sample_duration: float = 2.0,
    ) -> Optional[float]:
        """
        Score the brightness of a single frame/moment in the video.

        Wraps _get_avg_brightness_ffmpeg with a short sample window.

        Args:
            video_path: Path to source video
            timestamp: Time position in seconds
            sample_duration: Duration of sample window (default 2s)

        Returns:
            Brightness score on 0-1000 scale, or None if measurement failed.
        """
        return self._get_avg_brightness_ffmpeg(video_path, timestamp, sample_duration)

    def detect_segment(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
    ) -> Optional[WeatherResult]:
        """
        Detect weather for a single video segment.

        Args:
            video_path: Path to source video
            start_time: Segment start in seconds
            end_time: Segment end in seconds

        Returns:
            WeatherResult with condition, brightness, confidence, or None if
            brightness measurement failed (FFmpeg timeout or no data extracted).
            Callers should omit weather from output when None is returned.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        duration = end_time - start_time
        if duration <= 0:
            raise ValueError(f"Invalid segment: {start_time}s to {end_time}s")

        brightness = self._get_avg_brightness_ffmpeg(video_path, start_time, duration)
        if brightness is None:
            logger.warning(
                f"Weather detection skipped for {start_time:.1f}-{end_time:.1f}s "
                f"(brightness measurement failed)"
            )
            return None

        condition, confidence = self._classify(brightness)

        result = WeatherResult(
            condition=condition,
            brightness=brightness,
            confidence=confidence,
            segment_start=start_time,
            segment_end=end_time,
        )

        logger.info(
            f"Weather detection: {start_time:.1f}-{end_time:.1f}s → "
            f"{result.condition.value} (brightness={result.brightness:.1f}, "
            f"confidence={result.confidence:.2f})"
        )

        return result

    def detect_all_segments(
        self,
        video_path: Path,
        segments: List[dict],
    ) -> List[WeatherResult]:
        """
        Detect weather for multiple segments.

        Segments where brightness measurement fails (FFmpeg timeout or no data)
        are silently skipped — they are not included in the returned list.
        Callers receive only segments with valid measurements.

        Args:
            video_path: Path to source video
            segments: List of dicts with 'start_time' and 'end_time' keys

        Returns:
            List of WeatherResult objects for successfully-measured segments only.
            May be shorter than the input segments list if some measurements failed.
        """
        results = []
        failed_count = 0
        for seg in segments:
            result = self.detect_segment(
                video_path,
                start_time=seg["start_time"],
                end_time=seg["end_time"],
            )
            if result is not None:
                results.append(result)
            else:
                failed_count += 1

        if failed_count:
            logger.warning(
                f"Weather detection: {failed_count}/{len(segments)} segments "
                f"skipped (brightness measurement failed)"
            )

        # Log summary
        conditions = [r.condition.value for r in results]
        logger.info(
            f"Weather summary for {len(results)} segments: "
            f"sunny={conditions.count('sunny')}, "
            f"overcast={conditions.count('overcast')}, "
            f"rainy={conditions.count('rainy')}"
        )

        return results
