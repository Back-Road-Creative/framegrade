"""Unified segment scorer for drive pipeline.

Combines GPS-based signals (elevation, turns, speed variance, POI proximity)
with visual signals (motion energy, color vibrancy, scene change, sky ratio,
golden hour) into a single per-second composite score across an entire file.

The composite score drives both long-form assembly (which segments to include)
and shorts extraction (which peaks to use as shorts).

Usage:
    scorer = UnifiedSegmentScorer(weights)
    file_scores = scorer.score_file(video_path, gps_data, gps_offset)
    merged = scorer.merge_file_scores([scores_a, scores_b], [dur_a, dur_b])
    scorer.save_scores(merged, output_path, metadata)
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..pipeline.color_analysis import scale_range_token
from ..pipeline.exceptions import WeatherMeasurementError
from .shorts_extractor import GPSData, ScenicScorer
from .visual_scorer import (
    compute_grayscale_histogram,
    score_color_vibrancy,
    score_golden_hour,
    score_scene_change,
    score_sky_ratio,
)

logger = logging.getLogger(__name__)

# Column indices in the score array
COL_TIMESTAMP = 0
COL_COMPOSITE = 1
COL_MOTION_ENERGY = 2
COL_SPEED_VARIANCE = 3
COL_COLOR_VIBRANCY = 4
COL_SCENE_CHANGE = 5
COL_SKY_RATIO = 6
COL_GOLDEN_HOUR = 7
COL_SCENIC_SCORE = 8
COL_WEATHER = 9
COL_BRIGHTNESS = 10
NUM_COLUMNS = 11

SIGNAL_NAMES = [
    "motion_energy",
    "speed_variance",
    "color_vibrancy",
    "scene_change",
    "sky_ratio",
    "golden_hour",
    "scenic_score",
    "weather",
    "brightness",
]

# Explicit weather signal in the composite scorer.
# WeatherDetector classifies each video segment as one of the keys below;
# the score is the canonical mapping from class → [0, 1] used everywhere
# weather contributes to scoring. Single source of truth: planners and
# tests must consume this constant rather than hardcoding the values.
WEATHER_SCORE_BY_CLASS: dict[str, float] = {
    "sunny": 1.0,
    "overcast": 0.6,
    "rainy": 0.2,
}

# When weather data is unavailable (legacy code paths, ffmpeg measurement
# failure, GPS-less episodes), default to "overcast" — the mid-band class.
# Choosing rainy or sunny would systematically deflate or inflate every
# unmeasured segment; overcast is the construction-correct neutral.
DEFAULT_WEATHER_CLASS: str = "overcast"


def weather_score_for_class(weather_class: Optional[str]) -> float:
    """Resolve a weather class string to its scalar score.

    Unknown / None values fall back to the score for DEFAULT_WEATHER_CLASS.
    Centralised here so every call site goes through the same lookup.
    """
    if weather_class is None:
        weather_class = DEFAULT_WEATHER_CLASS
    return WEATHER_SCORE_BY_CLASS.get(weather_class, WEATHER_SCORE_BY_CLASS[DEFAULT_WEATHER_CLASS])


#: GPS-derived signals that go to zero when ``gps_data`` is unavailable.
#: Module-level so tests and callers without a scorer instance can refer
#: to the canonical set.
GPS_SIGNAL_NAMES: frozenset[str] = frozenset({"speed_variance", "scenic_score"})


def redistribute_gps_weights(
    weights: dict[str, float],
    has_gps: bool,
    gps_signal_names: Optional[frozenset[str] | set[str]] = None,
) -> dict[str, float]:
    """Pro-rata redistribute GPS signal mass across visual/motion signals.

    Construction-correct GPS-optional path: when ``has_gps`` is False the
    GPS-only signals (``speed_variance`` + ``scenic_score``, summing to
    0.35 under the default weight schedule) are zeroed and their mass is
    distributed across the remaining keys in proportion to their existing
    weight. The result sums to exactly 1.0 (within float epsilon).

    Args:
        weights: Original weight dict (must sum to ~1.0 to be meaningful;
            this helper does not validate the sum).
        has_gps: True if the caller has GPS data — returns ``weights``
            unchanged. False triggers the redistribution path.
        gps_signal_names: Optional override of the GPS signal name set.
            Defaults to ``GPS_SIGNAL_NAMES``.

    Returns:
        New dict with the same keys as ``weights``; GPS keys at 0.0,
        non-GPS keys scaled by ``1 / visual_weight``.
    """
    gps_names = gps_signal_names if gps_signal_names is not None else GPS_SIGNAL_NAMES
    if has_gps:
        return dict(weights)
    visual_weight = sum(w for name, w in weights.items() if name not in gps_names)
    new_weights: dict[str, float] = {}
    for name, w in weights.items():
        if name in gps_names:
            new_weights[name] = 0.0
        elif visual_weight > 0:
            new_weights[name] = w * (1.0 / visual_weight)
        else:
            new_weights[name] = w
    return new_weights


class UnifiedSegmentScorer:
    """Produces per-second composite scores across video files.

    Combines 9 signals with configurable weights into a normalized 0-1
    composite score suitable for segment selection and shorts extraction.

    The 8th and 9th signals — ``weather`` and ``brightness`` — capture
    macro and micro lighting respectively. ``weather`` is sourced from
    :class:`gomoveshift.pipeline.weather.WeatherDetector` (one
    classification per scored file) and broadcast across the per-second
    arrays; see ``WEATHER_SCORE_BY_CLASS`` for the canonical mapping and
    ``DEFAULT_WEATHER_CLASS`` for the unmeasured fallback. ``brightness``
    is computed per-second from the already-decoded numpy frame
    (mean grayscale / 255), so it discriminates *within* an episode where
    the constant per-episode weather class cannot — e.g. picking the
    sun-break window inside an otherwise-overcast file.
    """

    def __init__(self, weights: dict[str, float]):
        """Initialize with scoring weights.

        Args:
            weights: Dict mapping signal name to weight (must sum to 1.0).
                     Keys: motion_energy, speed_variance, color_vibrancy,
                     scene_change, sky_ratio, golden_hour, scenic_score,
                     weather, brightness. Missing keys default to 0.0
                     (treated as not contributing to composite); legacy
                     configs therefore continue to load but score with
                     zero contribution from any unknown key.

        Raises:
            ValueError: If weights don't sum to 1.0 (within 0.01 tolerance).
        """
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total:.3f}")
        self.weights = weights

    GPS_SIGNAL_NAMES = {"speed_variance", "scenic_score"}

    def _redistribute_gps_weights(
        self, weights: Optional[dict[str, float]] = None, has_gps: bool = False
    ) -> dict[str, float]:
        """Redistribute GPS signal weights proportionally to visual signals.

        Returns new weight dict that sums to 1.0 with GPS signals zeroed out
        when ``has_gps`` is False; otherwise returns ``weights`` unchanged.

        Args:
            weights: Optional weight dict to redistribute. Defaults to
                ``self.weights`` so existing callers keep working.
            has_gps: When True, return ``weights`` as-is (no redistribution).
                When False, zero out GPS signals and pro-rata distribute
                their mass across the remaining (visual / motion) keys.
        """
        return redistribute_gps_weights(
            weights if weights is not None else self.weights,
            has_gps=has_gps,
            gps_signal_names=self.GPS_SIGNAL_NAMES,
        )

    def combine_signals(
        self,
        timestamps: np.ndarray,
        signals: dict[str, np.ndarray],
        weights_override: Optional[dict[str, float]] = None,
    ) -> np.ndarray:
        """Combine per-second signal arrays into a composite score array.

        Args:
            timestamps: 1D array of timestamps (seconds).
            signals: Dict mapping signal name to 1D array of scores (0-1).
                     Each array must be same length as timestamps.
            weights_override: Optional alternative weights dict (e.g. GPS-redistributed).

        Returns:
            Array of shape (N, 9):
            [timestamp, composite, motion, speed_var, vibrancy, scene_change,
             sky, golden_hour, scenic_score]
        """
        weights = weights_override or self.weights
        n = len(timestamps)
        result = np.zeros((n, NUM_COLUMNS), dtype=np.float64)
        result[:, COL_TIMESTAMP] = timestamps

        # Fill signal columns and compute weighted composite
        composite = np.zeros(n, dtype=np.float64)
        for i, name in enumerate(SIGNAL_NAMES):
            col_idx = COL_MOTION_ENERGY + i
            signal = signals.get(name, np.zeros(n))
            # Clamp to [0, 1]
            signal = np.clip(signal, 0.0, 1.0)
            result[:, col_idx] = signal
            weight = weights.get(name, 0.0)
            composite += weight * signal

        result[:, COL_COMPOSITE] = np.clip(composite, 0.0, 1.0)
        return result

    def score_file(
        self,
        video_path: Path,
        gps_data: Optional[GPSData] = None,
        gps_offset: float = 0.0,
        sample_interval: float = 1.0,
        weather_detector=None,
        weather_class: Optional[str] = None,
        allow_degraded_weather: bool = False,
    ) -> np.ndarray:
        """Score an entire video file, producing per-second composite scores.

        Extracts BGR frames at sample_interval, computes visual and motion
        signals per frame, optionally adds GPS-based signals, and
        broadcasts a single weather classification across the file.

        Args:
            video_path: Path to video file.
            gps_data: Optional GPS data for speed_variance and scenic_score.
            gps_offset: GPS timestamp at video t=0 (adds to video timestamps
                        to get GPS timestamps).
            sample_interval: Seconds between frame samples (default 1.0).
            weather_detector: Optional ``WeatherDetector``. When supplied,
                ``detect_segment`` is called exactly once per ``score_file``
                invocation to classify the file as a single segment. The
                resulting class is broadcast across every per-second row.
                If detection fails (returns None), falls back to
                ``DEFAULT_WEATHER_CLASS``.
            weather_class: Optional explicit class override (one of
                ``"sunny" | "overcast" | "rainy"``). Takes precedence over
                ``weather_detector`` and is the path used by callers that
                already have a classification (e.g. cached results, tests).
                Unknown values fall back to ``DEFAULT_WEATHER_CLASS``.

        Returns:
            Array of shape (N, 11) where N is the number of extracted frames.
            Column layout: [timestamp, composite, motion, speed_var,
            vibrancy, scene_change, sky, golden_hour, scenic_score,
            weather, brightness].
        """
        frames = self._extract_frames_bgr(video_path, sample_interval)

        if not frames:
            return np.zeros((0, NUM_COLUMNS), dtype=np.float64)

        n = len(frames)
        timestamps = np.array([t for t, _ in frames], dtype=np.float64)

        # Compute per-frame visual signals
        motion_energy = np.zeros(n, dtype=np.float64)
        color_vibrancy = np.zeros(n, dtype=np.float64)
        scene_change_arr = np.zeros(n, dtype=np.float64)
        sky_ratio = np.zeros(n, dtype=np.float64)
        golden_hour = np.zeros(n, dtype=np.float64)
        brightness_arr = np.zeros(n, dtype=np.float64)

        prev_hist = None
        prev_gray = None

        for i, (_, frame) in enumerate(frames):
            # Visual signals from BGR frame
            color_vibrancy[i] = score_color_vibrancy(frame)
            sky_ratio[i] = score_sky_ratio(frame)
            golden_hour[i] = score_golden_hour(frame)

            # Grayscale histogram for scene change
            curr_hist = compute_grayscale_histogram(frame)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Per-second brightness — mean grayscale luminance, normalized
            # to [0, 1]. Free signal: the cv2.cvtColor call already runs
            # below for motion_energy/scene_change; we just hoisted it.
            # Sunny moments ~ 0.65–0.85, overcast ~ 0.45–0.60, rainy ~
            # 0.25–0.45. Complements the per-episode weather class.
            brightness_arr[i] = float(np.mean(curr_gray)) / 255.0

            # Motion energy and scene change require previous frame
            if i > 0:
                # Scene change from histogram comparison
                scene_change_arr[i] = score_scene_change(prev_hist, curr_hist)

                # Motion energy from grayscale frame diff
                diff = cv2.absdiff(prev_gray, curr_gray)
                motion_energy[i] = float(np.mean(diff)) / 255.0
            # else: first frame stays 0 for motion and scene change

            prev_hist = curr_hist
            prev_gray = curr_gray

        # GPS-based signals
        speed_variance = np.zeros(n, dtype=np.float64)
        scenic_score = np.zeros(n, dtype=np.float64)

        if gps_data is not None:
            scorer = ScenicScorer(gps_data)
            window = max(5.0, sample_interval * 5)

            # Correct a cold-start offset here too (scoring
            # runs before the builder; shared helper keeps scoring + labels on
            # the same time base).
            from .shorts_models import effective_offset

            gps_offset = effective_offset(gps_data, gps_offset)

            for i, ts in enumerate(timestamps):
                gps_ts = ts + gps_offset
                gps_start = gps_ts - window / 2
                gps_end = gps_ts + window / 2

                # Speed variance normalized to 0-1 (50 km/h std = 1.0)
                sv = scorer.get_speed_variance(gps_start, gps_end)
                speed_variance[i] = min(sv / 50.0, 1.0)

                # Scenic score normalized to 0-1 (already 0-100)
                ss = scorer.score_segment(gps_start, gps_end)
                scenic_score[i] = min(ss / 100.0, 1.0)

        # Resolve weather class for this file.
        # Priority: explicit weather_class arg > weather_detector call >
        # DEFAULT_WEATHER_CLASS. WeatherDetector is invoked at most once
        # per score_file call.
        #
        # Degraded weather (detector returns
        # None, or raises) now hard-fails by default. The silent fallback
        # to DEFAULT_WEATHER_CLASS masked slow-IO and baked wrong scores
        # into drive_scores.npz. Operator opts into the degraded path
        # via ``allow_degraded_weather=True`` (config flag /
        # ``--allow-degraded-weather`` CLI).
        resolved_weather_class: Optional[str] = weather_class
        weather_failed = False
        if resolved_weather_class is None and weather_detector is not None:
            try:
                file_duration = float(timestamps[-1] - timestamps[0] + sample_interval)
                result = weather_detector.detect_segment(
                    video_path,
                    start_time=float(timestamps[0]),
                    end_time=float(timestamps[0]) + file_duration,
                )
                if result is None:
                    weather_failed = True
                else:
                    # WeatherResult.condition is a WeatherCondition enum;
                    # use its .value (lowercase string) to look up the score.
                    cond = getattr(result, "condition", None)
                    resolved_weather_class = getattr(cond, "value", None) or str(cond)
            except Exception as e:  # noqa: BLE001
                weather_failed = True
                logger.error("Weather classification raised for %s: %s", video_path, e)

        if weather_failed:
            if allow_degraded_weather:
                logger.warning(
                    "Weather measurement degraded for %s — proceeding with "
                    "DEFAULT_WEATHER_CLASS=%r (allow_degraded_weather=True).",
                    video_path,
                    DEFAULT_WEATHER_CLASS,
                )
                resolved_weather_class = None
            else:
                raise WeatherMeasurementError(
                    f"Weather measurement degraded for {video_path}. "
                    f"Cause is usually slow source IO (cross-filesystem "
                    f"source — copy to ~/.cache/gomoveshift/sources/). "
                    f"Pass --allow-degraded-weather to bypass (not "
                    f"recommended — bakes overcast into drive_scores.npz)."
                )

        weather_value = weather_score_for_class(resolved_weather_class)
        weather_signal = np.full(n, weather_value, dtype=np.float64)

        signals = {
            "motion_energy": motion_energy,
            "speed_variance": speed_variance,
            "color_vibrancy": color_vibrancy,
            "scene_change": scene_change_arr,
            "sky_ratio": sky_ratio,
            "golden_hour": golden_hour,
            "scenic_score": scenic_score,
            "weather": weather_signal,
            "brightness": brightness_arr,
        }

        # When GPS is absent, redistribute GPS signal weights to visual signals
        gps_absent = gps_data is None
        weights = self.weights
        if gps_absent:
            weights = self._redistribute_gps_weights()
            logger.info("GPS unavailable — scoring with visual signals only (weight redistributed)")

        return self.combine_signals(timestamps, signals, weights_override=weights)

    @staticmethod
    def _parse_frame_probe(probe_result) -> tuple[float, str]:
        """Split the combined duration+color_range probe into its two values.

        ``csv=p=0`` emits one bare value per line: the v:0 stream's
        ``color_range`` (empty when untagged) then the format duration. An
        unreadable probe falls back to the historical 600 s budget and the
        GoPro full-range default rather than aborting a fine scoring run.
        """
        duration = 600.0
        color_range = ""
        if getattr(probe_result, "returncode", 1) != 0:
            return duration, color_range
        lines = [ln.strip() for ln in str(probe_result.stdout or "").splitlines()]
        for line in lines:
            try:
                duration = float(line)
                break
            except ValueError:
                if line:
                    color_range = line
        return duration, color_range

    def _extract_frames_bgr(
        self,
        video_path: Path,
        sample_interval: float = 1.0,
        width: int = 640,
        height: int = 360,
    ) -> list[tuple[float, np.ndarray]]:
        """Extract BGR frames from video at given interval using FFmpeg.

        Attempts GPU-accelerated decode (CUDA/NVDEC) first, falling back to
        CPU decode on failure.  GPU decode is ~3-5x faster for 4K HEVC.

        These frames feed ``drive_scores.npz`` — the array that SELECTS which
        footage ships — so the probed ``setrange`` tag ahead of the bgr24 hop
        is what keeps colour_vibrancy / sky_ratio / golden_hour / brightness
        out of the wrong numeric domain.

        Args:
            video_path: Path to video file.
            sample_interval: Seconds between frames.
            width: Output frame width.
            height: Output frame height.

        Returns:
            List of (timestamp, BGR frame) tuples.
        """
        sample_fps = 1.0 / sample_interval

        try:
            # Get duration (timeout budget) and colour range (domain tag) in
            # one probe. ``-show_entries`` emits stream rows before format rows.
            probe_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=color_range:format=duration",
                "-of",
                "csv=p=0",
                str(video_path),
            ]
            probe_result = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            duration, probed_range = self._parse_frame_probe(probe_result)
            source_range = scale_range_token(probed_range)
            timeout = max(300, int(duration * 5))
            # setrange only writes metadata; scale/format do the real work.
            vf = f"fps={sample_fps},setrange={source_range},scale={width}:{height}"

            # GPU-accelerated command (NVDEC for HEVC/H.265)
            gpu_cmd = [
                "ffmpeg",
                "-hwaccel",
                "cuda",
                "-c:v",
                "hevc_cuvid",
                "-i",
                str(video_path),
                "-vf",
                vf,
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "-v",
                "quiet",
                "pipe:1",
            ]

            result = subprocess.run(gpu_cmd, capture_output=True, timeout=timeout)

            if result.returncode != 0:
                logger.info(
                    "GPU decode failed for %s, falling back to CPU",
                    video_path.name,
                )
                cpu_cmd = [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-vf",
                    vf,
                    "-pix_fmt",
                    "bgr24",
                    "-f",
                    "rawvideo",
                    "-v",
                    "quiet",
                    "pipe:1",
                ]
                result = subprocess.run(cpu_cmd, capture_output=True, timeout=timeout)

            if result.returncode != 0:
                logger.error(
                    "FFmpeg frame extraction failed: %s",
                    result.stderr[:500] if result.stderr else "unknown",
                )
                return []

            frame_size = width * height * 3  # BGR = 3 channels
            raw = result.stdout
            num_frames = len(raw) // frame_size

            frames = []
            for i in range(num_frames):
                offset = i * frame_size
                data = raw[offset : offset + frame_size]
                frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                timestamp = i * sample_interval
                frames.append((timestamp, frame))

            logger.info("Extracted %d BGR frames from %s", num_frames, video_path.name)
            return frames

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg frame extraction timed out for %s", video_path)
            return []
        except Exception as e:
            logger.error("Frame extraction error for %s: %s", video_path, e)
            return []

    def merge_file_scores(
        self,
        file_scores: list[np.ndarray],
        file_durations: list[float],
    ) -> np.ndarray:
        """Stitch per-file score arrays into a drive-wide time series.

        File B timestamps are offset by file A's duration, etc.

        Args:
            file_scores: List of per-file score arrays (shape (N_i, 9)).
            file_durations: Duration of each file in seconds.

        Returns:
            Merged score array with continuous timestamps.
        """
        if not file_scores:
            return np.zeros((0, NUM_COLUMNS), dtype=np.float64)

        offset = 0.0
        adjusted = []
        for scores, duration in zip(file_scores, file_durations):
            chunk = scores.copy()
            chunk[:, COL_TIMESTAMP] += offset
            adjusted.append(chunk)
            offset += duration

        return np.concatenate(adjusted, axis=0)

    def save_scores(
        self,
        scores: np.ndarray,
        path: Path,
        metadata: dict,
        extra: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Save scores to .npz with JSON metadata sidecar.

        Args:
            scores: Score array (shape (N, 9)).
            path: Output path (should end in .npz).
            metadata: Dict of metadata (total_duration, num_files, etc.).
            extra: Optional named arrays written beside ``scores`` in the same
                ``np.savez_compressed`` call (e.g. ``source_sharpness``).
        """
        np.savez_compressed(path, scores=scores, **(extra or {}))

        # Save metadata sidecar
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved {len(scores)} score samples to {path}")

    def load_scores(
        self,
        path: Path,
    ) -> tuple[np.ndarray, dict]:
        """Load scores from .npz with JSON metadata.

        Args:
            path: Path to .npz file.

        Returns:
            Tuple of (scores array, metadata dict).
        """
        data = np.load(path)
        scores = data["scores"]

        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                logger.warning(
                    "Corrupt scorer metadata JSON at %s — using empty defaults", meta_path
                )
                metadata = {}
        else:
            metadata = {}

        return scores, metadata
