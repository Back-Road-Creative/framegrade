"""Motion analysis for dead spot detection."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from ..config import Config
from ..models import DeadSpot, HighMotionSegment
from .frame_range import setrange_filter

logger = logging.getLogger(__name__)


class MotionAnalyzer:
    """Analyzes video for motion to detect stopped vehicle periods."""

    # Target width for motion analysis (4K not needed for motion detection)
    ANALYSIS_WIDTH = 640

    def __init__(self, config: Config):
        self.config = config
        self.motion_threshold = config.dead_spots.motion_threshold
        self.min_duration = config.dead_spots.min_duration_seconds
        self.sample_fps = config.extraction.motion_sample_fps

        # Hysteresis for stop classification: enter "stopped" below
        # motion_threshold, exit only above motion_threshold * exit_multiplier.
        # Disabled by default; exit_multiplier 1.0 collapses the exit band onto
        # the enter band, reproducing the single-threshold behavior exactly.
        hyst = config.stop_detection.hysteresis
        self.hysteresis_enabled = hyst.enabled
        self.hysteresis_exit_multiplier = hyst.exit_multiplier

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for motion analysis (resize + grayscale + blur)."""
        # Resize for faster processing (motion detection doesn't need 4K)
        h, w = frame.shape[:2]
        if w > self.ANALYSIS_WIDTH:
            scale = self.ANALYSIS_WIDTH / w
            frame = cv2.resize(frame, (self.ANALYSIS_WIDTH, int(h * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        return gray

    def analyze(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> list[DeadSpot]:
        """
        Analyze video for dead spots (periods of low motion).

        Args:
            video_path: Path to the video file.
            progress_callback: Optional callback function called with
                (current_frame, total_frames, progress) where progress is 0.0-1.0.

        Returns list of DeadSpot objects representing stopped periods.
        """
        timestamps, motion_energies = self.analyze_raw(
            video_path, progress_callback=progress_callback
        )
        return self._find_dead_spots(timestamps, motion_energies)

    def _get_video_info(self, video_path: Path) -> tuple[float, int, float]:
        """Get video fps, total frames, and duration via ffprobe.

        Returns:
            Tuple of (fps, total_frames, duration).
        """
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate,nb_frames,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise ValueError(f"Could not probe video: {video_path}")

        try:
            info = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse ffprobe output for {video_path}: {exc}") from exc
        streams = info.get("streams", [])
        if not streams:
            raise ValueError(f"No video stream found in {video_path}")
        stream = streams[0]
        fmt = info.get("format", {})

        # Parse fps from fraction like "30/1" or "60000/1001"
        r_frame_rate = stream.get("r_frame_rate", "30/1")
        num, den = (int(x) for x in r_frame_rate.split("/"))
        fps = num / den

        # Duration from stream or format
        dur_str = stream.get("duration") or fmt.get("duration", "0")
        duration = float(dur_str)

        total_frames = int(stream.get("nb_frames", 0))
        if total_frames == 0:
            total_frames = int(fps * duration)

        return fps, total_frames, duration

    def _decode_frames_ffmpeg(
        self,
        video_path: Path,
    ) -> tuple[list[np.ndarray], float, float]:
        """Decode keyframes only via ffmpeg pipe.

        Uses -skip_frame nokey to only decode I-frames, avoiding the
        massive cost of decoding all P/B frames. For a 64-min 4K HEVC
        file with 1-second keyframes, this is ~10x faster than OpenCV
        sequential decode (~4 min vs ~42 min).

        Returns:
            Tuple of (frames, fps, duration) where frames are 2D uint8
            grayscale arrays at ANALYSIS_WIDTH resolution.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video not found: {video_path}")

        fps, total_frames, duration = self._get_video_info(video_path)

        # Calculate output height maintaining aspect ratio
        # We need to probe the actual dimensions for correct scaling
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        try:
            probe_info = json.loads(probe.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse ffprobe output for {video_path}: {exc}") from exc
        streams = probe_info.get("streams", [])
        if not streams:
            raise ValueError(f"No video stream found in {video_path}")
        src_w = int(streams[0]["width"])
        src_h = int(streams[0]["height"])
        out_w = self.ANALYSIS_WIDTH
        out_h = int(src_h * (out_w / src_w))
        # Ensure even dimensions for ffmpeg
        out_h = out_h + (out_h % 2)

        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-skip_frame",
                "nokey",
                "-i",
                str(video_path),
                # State the source range before the gray conversion:
                # untagged GoPro footage is full-range, so without the tag this
                # works from swscale's limited-range guess instead.
                "-vf",
                f"{setrange_filter(video_path)},scale={out_w}:{out_h},format=gray",
                "-vsync",
                "vfr",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-v",
                "quiet",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        frame_size = out_w * out_h
        frames = []
        try:
            while True:
                data = proc.stdout.read(frame_size)
                if len(data) < frame_size:
                    break
                frame = np.frombuffer(data, dtype=np.uint8).reshape(out_h, out_w)
                frames.append(frame)
        finally:
            proc.stdout.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # ffmpeg hung — kill and reap so we don't leak an orphan
                # NVDEC decoder (GPU lock) or a zombie. Mirrors ffmpeg_runner.
                proc.kill()
                proc.wait()
                raise

        if proc.returncode != 0 and len(frames) == 0:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            proc.stderr.close()
            raise RuntimeError(f"ffmpeg decode failed (rc={proc.returncode}): {stderr[:200]}")
        if proc.stderr:
            proc.stderr.close()

        if len(frames) == 0:
            raise ValueError(f"No frames decoded from {video_path}")

        return frames, fps, duration

    def _decode_frames_ffmpeg_fixed_fps(
        self,
        video_path: Path,
        target_fps: int = 2,
    ) -> tuple[list[np.ndarray], float, float]:
        """Decode video at a fixed fps via ffmpeg pipe.

        More reliable than keyframe-only for videos with sparse I-frames
        (e.g., HEVC-encoded assembled long forms). Slower than keyframe-only
        but much faster than OpenCV sequential decode, and handles HEVC
        reliably unlike OpenCV's internal decoder.

        Returns:
            Tuple of (frames, fps, duration) where frames are 2D uint8
            grayscale arrays at ANALYSIS_WIDTH resolution.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video not found: {video_path}")

        fps, total_frames, duration = self._get_video_info(video_path)

        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        try:
            probe_info = json.loads(probe.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse ffprobe output for {video_path}: {exc}") from exc
        streams = probe_info.get("streams", [])
        if not streams:
            raise ValueError(f"No video stream found in {video_path}")
        src_w = int(streams[0]["width"])
        src_h = int(streams[0]["height"])
        out_w = self.ANALYSIS_WIDTH
        out_h = int(src_h * (out_w / src_w))
        out_h = out_h + (out_h % 2)

        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                # Same range statement as the keyframe decode above.
                "-vf",
                f"{setrange_filter(video_path)},fps={target_fps},scale={out_w}:{out_h},format=gray",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-v",
                "quiet",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        frame_size = out_w * out_h
        frames = []
        try:
            while True:
                data = proc.stdout.read(frame_size)
                if len(data) < frame_size:
                    break
                frame = np.frombuffer(data, dtype=np.uint8).reshape(out_h, out_w)
                frames.append(frame)
        finally:
            proc.stdout.close()
            try:
                proc.wait(timeout=max(30, int(duration / 5)))
            except subprocess.TimeoutExpired:
                # ffmpeg hung — kill and reap so we don't leak an orphan
                # NVDEC decoder (GPU lock) or a zombie. Mirrors ffmpeg_runner.
                proc.kill()
                proc.wait()
                raise

        if proc.returncode != 0 and len(frames) == 0:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            proc.stderr.close()
            raise RuntimeError(
                f"ffmpeg fixed-fps decode failed (rc={proc.returncode}): {stderr[:200]}"
            )
        if proc.stderr:
            proc.stderr.close()

        if len(frames) == 0:
            raise ValueError(f"No frames decoded from {video_path}")

        logger.info(
            f"ffmpeg fixed-fps decode: {len(frames)} frames at {target_fps}fps "
            f"from {duration:.0f}s video"
        )
        return frames, fps, duration

    def _probe_keyframe_timestamps(
        self, video_path: Path, timeout: float = 120.0
    ) -> Optional[list[float]]:
        """REAL presentation timestamps (seconds) of each keyframe.

        Mirrors ``_decode_frames_ffmpeg``'s ``-skip_frame nokey`` selection so the
        returned list aligns 1:1 with the decoded keyframes. Returns ``None`` on
        any probe failure or a missing/``N/A`` PTS — the caller then synthesizes
        uniform timestamps, so a degraded probe never fabricates cadence.
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-select_streams",
                    "v:0",
                    "-skip_frame",
                    "nokey",
                    "-show_entries",
                    "frame=best_effort_timestamp_time",
                    "-of",
                    "csv=p=0",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if result.returncode != 0:
            return None
        stamps: list[float] = []
        for line in result.stdout.splitlines():
            token = line.strip().rstrip(",")
            if not token or token.upper() == "N/A":
                return None  # a missing PTS makes 1:1 alignment unsafe
            try:
                stamps.append(float(token))
            except ValueError:
                return None
        return stamps or None

    def _compute_motion_from_frames(
        self,
        frames: list[np.ndarray],
        duration: float,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        frame_timestamps: Optional[list[float]] = None,
    ) -> tuple[list[float], list[float]]:
        """Compute motion energy between consecutive frames.

        Args:
            frames: List of grayscale frames (already resized).
            duration: Video duration in seconds.
            progress_callback: Optional callback(current, total, progress).
            frame_timestamps: REAL per-frame PTS (seconds), aligned 1:1 with
                ``frames``. When present, valid (length-matched and strictly
                increasing) it is used verbatim so a motion sample lands at the
                frame's ACTUAL time — keyframe cadence is NOT uniform, so the old
                synthetic ``i*dt`` misattributed a stop boundary by up to a whole
                GOP, not just ±1s quantization. Falls back to ``i*dt`` when absent
                or unusable.

        Returns:
            Tuple of (timestamps, motion_energies).
        """
        if len(frames) < 2:
            return [], []

        total = len(frames)
        # Prefer REAL frame timestamps; only trust them when they align 1:1 and
        # are strictly increasing (else a duplicate/backwards PTS would break the
        # monotonic-timestamp contract downstream).
        use_real = (
            frame_timestamps is not None
            and len(frame_timestamps) == total
            and all(frame_timestamps[i] > frame_timestamps[i - 1] for i in range(1, total))
        )
        # Approximate uniform spacing from duration and frame count (fallback).
        dt = duration / total if total > 0 else 1.0

        timestamps = []
        motion_energies = []

        for i in range(1, total):
            if progress_callback and i % 50 == 0:
                progress_callback(i, total, i / total)

            # Apply Gaussian blur for noise reduction (same as _preprocess_frame)
            prev = cv2.GaussianBlur(frames[i - 1], (21, 21), 0)
            curr = cv2.GaussianBlur(frames[i], (21, 21), 0)

            diff = cv2.absdiff(prev, curr)
            motion_energy = np.mean(diff) / 255.0
            motion_energies.append(motion_energy)
            timestamps.append(frame_timestamps[i] if use_real else i * dt)

        if progress_callback:
            progress_callback(total, total, 1.0)

        return timestamps, motion_energies

    def analyze_raw(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> tuple[list[float], list[float]]:
        """
        Analyze video and return raw motion data.

        Uses ffmpeg keyframe-only decode for ~10x speedup on long videos.
        Falls back to OpenCV sequential decode if ffmpeg fails.

        Args:
            video_path: Path to the video file.
            progress_callback: Optional callback function called with
                (current_frame, total_frames, progress) where progress is 0.0-1.0.

        Returns:
            Tuple of (timestamps, motion_energies).
        """
        # Try fast ffmpeg keyframe-only path first
        try:
            frames, fps, duration = self._decode_frames_ffmpeg(video_path)
            if len(frames) >= 2:
                sample_density = len(frames) / duration if duration > 0 else 0
                if sample_density < 1.0:
                    logger.warning(
                        f"ffmpeg keyframe decode too sparse: {len(frames)} frames / "
                        f"{duration:.0f}s = {sample_density:.3f} fps — "
                        f"trying ffmpeg fixed-fps decode"
                    )
                else:
                    logger.info(
                        f"ffmpeg keyframe decode: {len(frames)} frames from "
                        f"{duration:.0f}s video ({sample_density:.1f} fps)"
                    )
                    # Keyframe cadence is NOT uniform — place each motion sample
                    # at the keyframe's REAL PTS so a stop boundary isn't shifted
                    # by up to a GOP (falls back to i*dt if the probe degrades).
                    real_ts = self._probe_keyframe_timestamps(
                        video_path, timeout=max(30.0, duration / 2)
                    )
                    return self._compute_motion_from_frames(
                        frames, duration, progress_callback, frame_timestamps=real_ts
                    )
            else:
                logger.warning(
                    f"ffmpeg keyframe decode got only {len(frames)} frames "
                    f"(need ≥2), trying ffmpeg fixed-fps decode"
                )
        except (RuntimeError, FileNotFoundError, OSError) as e:
            logger.warning(f"ffmpeg keyframe decode failed: {e}")

        # Tier 2: ffmpeg fixed-fps decode (reliable for HEVC, faster than OpenCV)
        try:
            frames, fps, duration = self._decode_frames_ffmpeg_fixed_fps(video_path)
            if len(frames) >= 2:
                return self._compute_motion_from_frames(frames, duration, progress_callback)
            logger.warning(
                f"ffmpeg fixed-fps decode got only {len(frames)} frames, falling back to OpenCV"
            )
        except (RuntimeError, FileNotFoundError, OSError) as e:
            logger.warning(f"ffmpeg fixed-fps decode failed, falling back to OpenCV: {e}")

        # Tier 3: OpenCV sequential decode (slow, may struggle with HEVC)
        return self._analyze_raw_opencv(video_path, progress_callback)

    def _analyze_raw_opencv(
        self,
        video_path: Path,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> tuple[list[float], list[float]]:
        """OpenCV-based sequential decode (fallback path)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame skip for desired sample rate
        frame_skip = max(1, int(fps / self.sample_fps))

        motion_energies = []
        timestamps = []
        prev_gray = None
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if progress_callback and frame_idx % 30 == 0 and total_frames > 0:
                    progress = frame_idx / total_frames
                    progress_callback(frame_idx, total_frames, progress)

                if frame_idx % frame_skip == 0:
                    gray = self._preprocess_frame(frame)

                    if prev_gray is not None:
                        diff = cv2.absdiff(prev_gray, gray)
                        motion_energy = np.mean(diff) / 255.0
                        motion_energies.append(motion_energy)
                        timestamps.append(frame_idx / fps)

                    prev_gray = gray

                frame_idx += 1

            if progress_callback and total_frames > 0:
                progress_callback(total_frames, total_frames, 1.0)
        finally:
            cap.release()

        return timestamps, motion_energies

    def _find_dead_spots(
        self, timestamps: list[float], motion_energies: list[float]
    ) -> list[DeadSpot]:
        """Find contiguous periods of low motion."""
        if not timestamps:
            return []

        # Hysteresis state machine: while moving, enter "stopped" the moment
        # energy drops below motion_threshold; while stopped, stay stopped
        # until energy rises above the (possibly higher) exit threshold so
        # near-threshold flicker can't re-open/close the span. With hysteresis
        # DISABLED the exit test is ``energy >= motion_threshold`` — byte-for-byte
        # the original single-threshold ``else`` branch (brief creep splits a
        # span just as before).
        exit_threshold = self.motion_threshold * self.hysteresis_exit_multiplier

        dead_spots = []
        in_dead_spot = False
        spot_start = 0.0
        spot_energies = []

        for ts, energy in zip(timestamps, motion_energies):
            if not in_dead_spot:
                # Enter "stopped" — unchanged enter condition either way.
                still_stopped = energy < self.motion_threshold
            elif self.hysteresis_enabled:
                # Stay stopped until energy clears the higher exit band.
                still_stopped = energy <= exit_threshold
            else:
                # Original behavior: exit the instant energy reaches threshold.
                still_stopped = energy < self.motion_threshold

            if still_stopped:
                if not in_dead_spot:
                    in_dead_spot = True
                    spot_start = ts
                    spot_energies = [energy]
                else:
                    spot_energies.append(energy)
            else:
                if in_dead_spot:
                    # End of dead spot
                    duration = ts - spot_start
                    if duration >= self.min_duration:
                        dead_spots.append(
                            DeadSpot(
                                start_time=spot_start,
                                end_time=ts,
                                motion_energy=np.mean(spot_energies),
                                suggested_trim=True,
                            )
                        )
                    in_dead_spot = False
                    spot_energies = []

        # Handle dead spot at end of video
        if in_dead_spot and timestamps:
            duration = timestamps[-1] - spot_start
            if duration >= self.min_duration:
                dead_spots.append(
                    DeadSpot(
                        start_time=spot_start,
                        end_time=timestamps[-1],
                        motion_energy=np.mean(spot_energies),
                        suggested_trim=True,
                    )
                )

        return dead_spots

    def find_high_motion_segments(
        self,
        video_path: Path,
        min_duration: float = 15.0,
        max_duration: float = 60.0,
        top_n: int = 3,
    ) -> list[HighMotionSegment]:
        """
        Find the highest motion segments in a video for shorts extraction.

        Uses ffmpeg keyframe-only decode for speed, falls back to OpenCV.

        Args:
            video_path: Path to the video file.
            min_duration: Minimum segment duration in seconds.
            max_duration: Maximum segment duration in seconds.
            top_n: Number of top segments to return.

        Returns:
            List of HighMotionSegment objects sorted by motion energy (highest first).
        """
        # Get motion data using shared analyze_raw (ffmpeg fast path)
        timestamps, energies = self.analyze_raw(video_path)

        if not timestamps:
            return []

        duration = timestamps[-1] if timestamps else 0
        motion_data = list(zip(timestamps, energies))

        # Find high-motion segments using sliding window approach
        segments = self._find_high_motion_windows(
            motion_data=motion_data,
            min_duration=min_duration,
            max_duration=max_duration,
            video_duration=duration,
        )

        # Sort by average motion energy and take top N
        segments.sort(key=lambda s: s.avg_motion_energy, reverse=True)

        # Remove overlapping segments (keep higher energy ones)
        non_overlapping = []
        for segment in segments:
            overlaps = False
            for existing in non_overlapping:
                if self._segments_overlap(segment, existing):
                    overlaps = True
                    break
            if not overlaps:
                non_overlapping.append(segment)
            if len(non_overlapping) >= top_n:
                break

        # Assign ranks
        for i, segment in enumerate(non_overlapping, start=1):
            segment.rank = i

        return non_overlapping

    def _find_high_motion_windows(
        self,
        motion_data: list[tuple[float, float]],
        min_duration: float,
        max_duration: float,
        video_duration: float,
    ) -> list[HighMotionSegment]:
        """Find windows with high motion using a sliding window approach."""
        if not motion_data:
            return []

        segments = []
        timestamps = [m[0] for m in motion_data]
        energies = [m[1] for m in motion_data]

        # Derive actual sample rate from the data (handles both OpenCV
        # and ffmpeg keyframe paths correctly)
        if len(timestamps) >= 2:
            actual_fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
        else:
            actual_fps = self.sample_fps

        # Use sliding windows of varying sizes
        for target_duration in [min_duration, (min_duration + max_duration) / 2, max_duration]:
            window_size = max(2, int(target_duration * actual_fps))
            if window_size < 2:
                continue

            for i in range(len(energies) - window_size + 1):
                window_energies = energies[i : i + window_size]
                avg_energy = np.mean(window_energies)
                peak_energy = np.max(window_energies)

                start_time = timestamps[i]
                end_time = timestamps[min(i + window_size - 1, len(timestamps) - 1)]

                # Only consider windows with significant motion
                if avg_energy > self.motion_threshold * 2:
                    segments.append(
                        HighMotionSegment(
                            start_time=start_time,
                            end_time=end_time,
                            avg_motion_energy=avg_energy,
                            peak_motion_energy=peak_energy,
                        )
                    )

        # Percentile fallback: when absolute threshold yields nothing (common
        # for curated highlight reels where all footage is "moving"), use the
        # top 20th percentile of window energies as the threshold instead.
        if not segments and len(energies) >= 10:
            mid_duration = (min_duration + max_duration) / 2
            window_size = max(2, int(mid_duration * actual_fps))
            if window_size <= len(energies):
                window_avgs = []
                for i in range(len(energies) - window_size + 1):
                    window_avgs.append(np.mean(energies[i : i + window_size]))

                if window_avgs:
                    p80 = float(np.percentile(window_avgs, 80))
                    logger.info(
                        f"Absolute threshold ({self.motion_threshold * 2:.4f}) "
                        f"found 0 segments — using 80th percentile ({p80:.6f})"
                    )
                    for i in range(len(energies) - window_size + 1):
                        we = energies[i : i + window_size]
                        avg_energy = np.mean(we)
                        if avg_energy >= p80:
                            segments.append(
                                HighMotionSegment(
                                    start_time=timestamps[i],
                                    end_time=timestamps[
                                        min(i + window_size - 1, len(timestamps) - 1)
                                    ],
                                    avg_motion_energy=avg_energy,
                                    peak_motion_energy=np.max(we),
                                )
                            )

        return segments

    def _segments_overlap(
        self,
        seg1: HighMotionSegment,
        seg2: HighMotionSegment,
        min_gap: float = 5.0,
    ) -> bool:
        """Check if two segments overlap or are too close."""
        # Check for overlap or segments within min_gap of each other
        return not (
            seg1.end_time + min_gap < seg2.start_time or seg2.end_time + min_gap < seg1.start_time
        )

    def get_motion_graph(
        self, video_path: Path, resolution: int = 100
    ) -> list[tuple[float, float]]:
        """
        Get motion energy over time for visualization.

        Returns list of (timestamp, motion_energy) tuples, downsampled
        to approximately 'resolution' points.
        """
        timestamps, energies = self.analyze_raw(video_path)

        if not timestamps:
            return []

        # Downsample to requested resolution
        data = list(zip(timestamps, energies))
        if len(data) <= resolution:
            return data

        step = max(1, len(data) // resolution)
        return data[::step]
