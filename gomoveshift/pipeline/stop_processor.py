"""
Stop Processor — detect stops and remove them with crossfade transitions.

Uses the existing MotionAnalyzer for stop detection, then FFmpeg for
cutting and stitching segments with crossfade transitions (or hard cuts
when crossfade_duration=0).

"""

import hashlib
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ..analysis.motion import MotionAnalyzer
from ..config import Config
from ..models import DeadSpot
from ..paths import gms_cache_home
from . import encoding_constants as ec
from .color import detect_nvenc
from .exceptions import ProbeFailedError
from .ffmpeg_runner import run_ffmpeg_quick

logger = logging.getLogger(__name__)


# ============================================================================
# Motion Cache Functions
# ============================================================================

# Default motion cache: under the per-user cache home (``GMS_CACHE_HOME``, else
# ``$XDG_CACHE_HOME/gomoveshift``, else ``~/.cache/gomoveshift``), bound when
# this module is imported. Callers pass ``cache_dir`` to use another location.
DEFAULT_CACHE_DIR = gms_cache_home() / "motion_cache"


def _ensure_cache_dir(cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Create cache directory with group-writable mode, falling back to tempdir.

    Uses mode 0o775 (rwxrwxr-x) so every member of the owning group can read
    and write cache files without requiring world-writable permissions.
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.chmod(0o775)
        return cache_dir
    except PermissionError:
        fallback = Path(tempfile.mkdtemp(prefix="gomoveshift_motion_cache_"))
        logger.warning("Cannot create %s — using %s", cache_dir, fallback)
        return fallback


def compute_file_hash(video_path: Path) -> str:
    """
    Compute stable file hash from first+last 64KB + file size.

    This is the same pattern used by the shorts extractor for motion caching.
    Using first+last chunks instead of full file hash makes this fast even
    for large video files (45GB+).

    Args:
        video_path: Path to video file

    Returns:
        32-character hex MD5 hash string
    """
    video_path = Path(video_path)
    hasher = hashlib.md5()

    # Add file size
    file_size = video_path.stat().st_size
    hasher.update(str(file_size).encode())

    # Read first 64KB
    chunk_size = 64 * 1024
    with open(video_path, "rb") as f:
        first_chunk = f.read(chunk_size)
        hasher.update(first_chunk)

        # Read last 64KB if file is large enough
        if file_size > chunk_size * 2:
            f.seek(-chunk_size, 2)  # Seek to last 64KB
            last_chunk = f.read(chunk_size)
            hasher.update(last_chunk)

    return hasher.hexdigest()


def get_motion_cache_path(video_path: Path, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """
    Get path to cached motion data for a video file.

    Args:
        video_path: Path to video file
        cache_dir: Directory for motion cache (default: ``<cache home>/motion_cache``)

    Returns:
        Path to .npz cache file
    """
    file_hash = compute_file_hash(video_path)
    cache_filename = f"{video_path.stem}_{file_hash}_motion.npz"
    return cache_dir / cache_filename


def load_motion_cache(
    video_path: Path, cache_dir: Path = DEFAULT_CACHE_DIR
) -> Optional[Tuple[list, list]]:
    """
    Load cached motion data if available.

    Args:
        video_path: Path to video file
        cache_dir: Directory for motion cache

    Returns:
        Tuple of (timestamps, motion_energies) or None if cache miss
    """
    cache_path = get_motion_cache_path(video_path, cache_dir)

    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path)
        timestamps = data["timestamps"].tolist()
        motion_energies = data["motion_energies"].tolist()
        logger.debug(f"  Loaded motion cache: {len(timestamps)} samples")
        return timestamps, motion_energies
    except Exception as e:
        logger.warning(f"  Corrupted motion cache, will re-analyze: {e}")
        # Delete corrupted cache
        cache_path.unlink(missing_ok=True)
        return None


def save_motion_cache(
    video_path: Path, timestamps: list, motion_energies: list, cache_dir: Path = DEFAULT_CACHE_DIR
):
    """
    Save motion data to cache.

    Args:
        video_path: Path to video file
        timestamps: List of frame timestamps
        motion_energies: List of motion energy values
        cache_dir: Directory for motion cache
    """
    cache_dir = _ensure_cache_dir(cache_dir)
    cache_path = get_motion_cache_path(video_path, cache_dir)

    np.savez_compressed(
        cache_path,
        timestamps=np.array(timestamps, dtype=np.float32),
        motion_energies=np.array(motion_energies, dtype=np.float32),
    )
    logger.debug(f"  Saved motion cache: {cache_path.name}")


def build_stop_crossfade_filter(
    durations: list[float], crossfade: float, fps: int = 30
) -> tuple[str, float]:
    """Single-pass ``xfade``+``acrossfade`` filter_complex joining ``N`` segments
    with a ``crossfade``-second dissolve per seam, plus the output duration
    (``Σ durations − (N−1)·crossfade``). Each seam overlaps the running chain's
    last ``crossfade``s with the next head so every segment BODY survives — the
    k-th xfade offset is the running chain duration minus ``crossfade``. (The old
    per-pair loop kept only the final transition, collapsing output to one
    dissolve.) Requires every segment longer than ``crossfade``.
    """
    n = len(durations)
    if n < 2:
        raise ValueError("crossfade stitch needs >= 2 segments")
    tb = fps * 1000
    v = [f"[{i}:v]fps={fps},settb=1/{tb},setpts=PTS-STARTPTS[v{i}]" for i in range(n)]
    a: list[str] = []
    prev_v, prev_a = "v0", "0:a"
    cum = durations[0]
    for k in range(1, n):
        offset = cum - crossfade
        v_out = "vraw" if k == n - 1 else f"x{k}"
        a_out = "aud" if k == n - 1 else f"a{k}"
        v.append(
            f"[{prev_v}][v{k}]xfade=transition=fade:duration={crossfade}:offset={offset:.6f}[{v_out}]"
        )
        a.append(f"[{prev_a}][{k}:a]acrossfade=d={crossfade}[{a_out}]")
        prev_v, prev_a = v_out, a_out
        cum += durations[k] - crossfade
    v.append(f"[vraw]format={ec.LONGFORM_PIX_FMT}[v]")
    return ";".join(v + a), cum


class StopProcessor:
    """Detect stops in video and remove them with smooth crossfade transitions."""

    def __init__(
        self,
        config: Config,
        crossfade_duration: float = 0.5,
        cache_dir: Optional[Path] = None,
        tracker=None,
        video_height: Optional[int] = None,
        gps_data=None,
    ):
        """
        Args:
            config: Application configuration (uses stop_detection settings)
            crossfade_duration: Crossfade duration in seconds between segments (0 = hard cuts).
            cache_dir: Directory for motion cache (default: ``<cache home>/motion_cache``)
            tracker: Optional ProcessTracker instance for progress reporting
            video_height: Pre-detected video height (skips ffprobe in _calc_timeout)
            gps_data: Optional ``GPSData`` for this LF. When present (and
                stop_detection.gps_speed_corroboration.enabled is True), each
                optical stop is cross-checked against GPS speed over the same
                source-time span. ``None`` (GPS-absent LF) keeps the optical-only
                path exactly — no corroboration runs.
        """
        self.config = config
        self.crossfade_duration = crossfade_duration
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.tracker = tracker
        self._video_height = video_height
        self.gps_data = gps_data

        # Use stop_detection config for more sensitive detection than dead_spots
        self.motion_threshold = config.stop_detection.motion_threshold
        self.min_stop_duration = config.stop_detection.min_stop_duration

    def detect_stops(self, video_path: Path) -> list[DeadSpot]:
        """
        Detect stops using cached motion data or fresh analysis.

        Returns list of DeadSpot objects representing stopped periods.
        """
        logger.info("  Analyzing motion to find stops...")
        video_path = Path(video_path)

        # Try to load from cache first
        cached = load_motion_cache(video_path, self.cache_dir)

        if cached is not None:
            logger.info("  Using cached motion analysis")
            timestamps, motion_energies = cached
        else:
            logger.info("  Running motion analysis (no cache)")
            timestamps, motion_energies = self._analyze_motion_from_video(video_path)

            # Save to cache for next time
            save_motion_cache(video_path, timestamps, motion_energies, self.cache_dir)

        # Get video duration for accurate trailing stop end_time. A probe
        # failure is tolerated here only because a video WITHOUT a trailing
        # stop doesn't need the duration; _find_stops_in_motion_data raises
        # loudly (ProbeFailedError) if a real trailing stop actually needs it.
        try:
            video_duration = self._get_duration(video_path)
        except (ProbeFailedError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "  Could not probe video duration (%s) — a trailing stop, if any, "
                "will hard-fail rather than silently truncate",
                exc,
            )
            video_duration = None

        # Find stops in motion data
        stops = self._find_stops_in_motion_data(
            timestamps, motion_energies, video_duration=video_duration
        )

        # GPS-speed corroboration: drop optical stops the GPS trace
        # shows were still moving (false dark/tunnel stops). No-op for GPS-absent
        # LFs and when disabled — optical-only behavior is preserved exactly.
        stops = self._corroborate_stops_with_gps(stops)
        # ADDITIVE GPS source: emit stops the optics missed (parked with passing
        # traffic — optical energy stays high, GPS shows ~0). Strict no-op for
        # GPS-absent LFs / when disabled, so the optical-only path is unchanged.
        gps_stops = self._gps_derived_stops()
        if gps_stops:
            stops = self._merge_stop_spans(stops + gps_stops)
        return stops

    def _analyze_motion_from_video(self, video_path: Path) -> Tuple[list, list]:
        """
        Analyze video and extract motion data in a single pass.

        Uses MotionAnalyzer.analyze_raw() to get timestamps and energies
        directly, avoiding a redundant second read of the video.

        Returns:
            Tuple of (timestamps, motion_energies)
        """
        # Create a temporary config with stop-detection sensitivity
        stop_config = self.config.model_copy(deep=True)
        stop_config.dead_spots.motion_threshold = self.motion_threshold
        stop_config.dead_spots.min_duration_seconds = int(self.min_stop_duration)

        analyzer = MotionAnalyzer(stop_config)

        tracker = self.tracker

        def progress_cb(current_frame, total_frames, progress):
            if tracker:
                tracker.tick(
                    max(1, current_frame - (tracker._stage.items_done if tracker._stage else 0))
                )
            if int(progress * 100) % 10 == 0:
                logger.info(f"    Motion analysis: {progress * 100:.0f}%")

        # Single-pass: get raw timestamps and energies directly
        timestamps, motion_energies = analyzer.analyze_raw(
            video_path, progress_callback=progress_cb
        )

        return timestamps, motion_energies

    def _find_stops_in_motion_data(
        self,
        timestamps: list,
        motion_energies: list,
        video_duration: Optional[float] = None,
    ) -> list[DeadSpot]:
        """
        Find stops in pre-computed motion data.

        Args:
            timestamps: List of frame timestamps
            motion_energies: List of motion energy values
            video_duration: Total video duration in seconds; required for an
                accurate trailing-stop end_time. If a trailing stop exists and
                this is None (probe failed), raises ProbeFailedError rather than
                silently truncating to the last motion sample.

        Returns:
            List of DeadSpot objects

        Raises:
            ProbeFailedError: a trailing stop exists but video_duration is None.
        """
        if not timestamps:
            return []

        stops = []
        in_stop = False
        stop_start = 0.0
        stop_energies = []

        for ts, energy in zip(timestamps, motion_energies):
            if energy < self.motion_threshold:
                if not in_stop:
                    in_stop = True
                    stop_start = ts
                    stop_energies = [energy]
                else:
                    stop_energies.append(energy)
            else:
                if in_stop:
                    # End of stop
                    duration = ts - stop_start
                    if duration >= self.min_stop_duration:
                        stops.append(
                            DeadSpot(
                                start_time=float(stop_start),
                                end_time=float(ts),
                                motion_energy=float(np.mean(stop_energies)),
                                suggested_trim=True,
                            )
                        )
                    in_stop = False
                    stop_energies = []

        # Handle stop at end of video — use the true media duration so trailing
        # frames beyond the last motion sample are included. There is NO silent
        # fallback to the last motion sample: that would truncate the trailing
        # stop (and leave dead footage in the cut) with no signal. If the probe
        # failed upstream, a real trailing stop hard-fails loudly instead.
        if in_stop and timestamps:
            if video_duration is None:
                logger.error(
                    "Trailing stop detected (starts %.1fs) but video duration is "
                    "unavailable — refusing to fall back to last motion sample %.1fs",
                    stop_start,
                    timestamps[-1],
                )
                raise ProbeFailedError(
                    "<motion-data trailing stop>",
                    f"trailing stop at {stop_start:.1f}s needs true media duration; "
                    "ffprobe failed and there is no safe fallback",
                )
            trail_end = video_duration
            duration = trail_end - stop_start
            if duration >= self.min_stop_duration:
                stops.append(
                    DeadSpot(
                        start_time=float(stop_start),
                        end_time=float(trail_end),
                        motion_energy=float(np.mean(stop_energies)),
                        suggested_trim=True,
                    )
                )

        return stops

    def _gps_speed_over_span(self, gps_offset: float, start: float, end: float) -> Optional[float]:
        """Median GPS speed (km/h) over the optical stop's source-time span.

        Maps the video-relative span ``[start, end]`` to GPS time with the same
        linear ``gps_offset`` the scorers use (``video_time + gps_offset``), then
        derives per-step speeds from consecutive trackpoints via the shared
        ``_haversine_m``. Returns ``None`` when the span can't be measured
        (< 2 timestamped points / no positive dt) so a degraded or partial trace
        leaves the optical stop untouched rather than corrupting it. Median (not
        mean) so a single glitch fix can't swing the verdict.
        """
        from ..analysis.shorts_models import _haversine_m

        points = self.gps_data.get_points_in_range(start + gps_offset, end + gps_offset)
        if len(points) < 2:
            return None
        speeds: list[float] = []
        for a, b in zip(points, points[1:]):
            dt = b.timestamp - a.timestamp
            if dt <= 0:
                continue
            dist_m = _haversine_m(a.lat, a.lon, b.lat, b.lon)
            speeds.append((dist_m / dt) * 3.6)
        if not speeds:
            return None
        return float(np.median(speeds))

    def _corroborate_stops_with_gps(self, stops: list[DeadSpot]) -> list[DeadSpot]:
        """Cross-check optical stops against GPS speed.

        Drops an optical stop whose GPS speed over the same source-time span is
        clearly above ``stop_speed_kmh`` (a false dark/tunnel stop the camera saw
        as motionless). A span where GPS speed is ~0 but optics flagged a stop is
        kept and logged. Guarded so it is a strict no-op for GPS-absent LFs, when
        disabled, or when speed can't be derived for a span.
        """
        cfg = self.config.stop_detection.gps_speed_corroboration
        if not cfg.enabled or self.gps_data is None:
            return stops
        trackpoints = getattr(self.gps_data, "trackpoints", None)
        if not trackpoints:
            return stops

        from ..analysis.shorts_models import effective_offset

        gps_offset = effective_offset(self.gps_data, self.gps_data.start_time)
        # Clearly above the threshold = drop; a small band avoids dropping stops
        # whose measured pace barely brushes the knob from GPS jitter.
        drop_threshold = cfg.stop_speed_kmh * 1.5

        kept: list[DeadSpot] = []
        for stop in stops:
            speed = self._gps_speed_over_span(gps_offset, stop.start_time, stop.end_time)
            if speed is None:
                kept.append(stop)  # degraded/partial span — leave optical as-is
                continue
            if speed > drop_threshold:
                logger.info(
                    "  GPS corroboration: dropping optical stop %.1fs-%.1fs "
                    "(GPS speed %.1f km/h > %.1f km/h — false dark/tunnel stop)",
                    stop.start_time,
                    stop.end_time,
                    speed,
                    drop_threshold,
                )
                continue
            if speed <= cfg.stop_speed_kmh:
                logger.debug(
                    "  GPS corroboration: optical stop %.1fs-%.1fs confirmed (GPS speed %.1f km/h)",
                    stop.start_time,
                    stop.end_time,
                    speed,
                )
            kept.append(stop)
        return kept

    def _gps_derived_stops(self) -> list[DeadSpot]:
        """ADDITIVE GPS-speed stop source.

        Scans the GPS trace for maximal runs whose per-step ground speed stays
        below ``stop_speed_kmh`` AND whose median speed is at or below
        ``parked_speed_kmh`` (≈ 0, tolerating parked-GPS jitter). Each such run
        at least ``min_stop_duration`` long is emitted as a stop in VIDEO time
        (``gps_time − gps_offset``, clamped to ≥ 0) EVEN when optical energy over
        the same span stayed above threshold — the "parked with passing traffic"
        class the optical detector cannot see. Strict no-op when disabled, for a
        GPS-absent LF, or when speeds can't be derived (< 2 timestamped points /
        no positive dt) so a degraded trace never fabricates stops.
        """
        cfg = self.config.stop_detection.gps_speed_corroboration
        if not cfg.enabled or self.gps_data is None:
            return []
        trackpoints = getattr(self.gps_data, "trackpoints", None)
        if not trackpoints or len(trackpoints) < 2:
            return []

        from ..analysis.shorts_models import _haversine_m, effective_offset

        gps_offset = effective_offset(self.gps_data, self.gps_data.start_time)
        # Per-step (t_start_gps, t_end_gps, speed_kmh) from consecutive fixes.
        steps: list[tuple[float, float, float]] = []
        for a, b in zip(trackpoints, trackpoints[1:]):
            dt = b.timestamp - a.timestamp
            if dt <= 0:
                continue
            dist_m = _haversine_m(a.lat, a.lon, b.lat, b.lon)
            steps.append((a.timestamp, b.timestamp, (dist_m / dt) * 3.6))
        if not steps:
            return []

        derived: list[DeadSpot] = []
        n = len(steps)
        i = 0
        while i < n:
            if steps[i][2] < cfg.stop_speed_kmh:
                j = i
                while j < n and steps[j][2] < cfg.stop_speed_kmh:
                    j += 1
                run = steps[i:j]
                run_start_gps, run_end_gps = run[0][0], run[-1][1]
                duration = run_end_gps - run_start_gps
                median_speed = float(np.median([s[2] for s in run]))
                if duration >= self.min_stop_duration and median_speed <= cfg.parked_speed_kmh:
                    start_v = max(0.0, run_start_gps - gps_offset)
                    end_v = run_end_gps - gps_offset
                    if end_v - start_v >= self.min_stop_duration:
                        logger.info(
                            "  GPS corroboration: ADDING parked stop %.1fs-%.1fs "
                            "(median GPS speed %.1f km/h — optics missed it)",
                            start_v,
                            end_v,
                            median_speed,
                        )
                        derived.append(
                            DeadSpot(
                                start_time=start_v,
                                end_time=end_v,
                                motion_energy=0.0,
                                suggested_trim=True,
                            )
                        )
                i = j
            else:
                i += 1
        return derived

    @staticmethod
    def _merge_stop_spans(stops: list[DeadSpot]) -> list[DeadSpot]:
        """Sort by start and coalesce overlapping/touching spans into one.

        Union of the optical + GPS-derived stop sets: overlapping spans (an
        optical dark-stop that also reads parked on GPS) must not survive as two
        DeadSpots or ``_calculate_keep_segments`` would rewind ``current_pos`` on
        the nested span and re-admit footage between them.
        """
        if not stops:
            return []
        ordered = sorted(stops, key=lambda s: s.start_time)
        merged = [ordered[0]]
        for s in ordered[1:]:
            last = merged[-1]
            if s.start_time <= last.end_time:
                if s.end_time > last.end_time:
                    merged[-1] = DeadSpot(
                        start_time=last.start_time,
                        end_time=s.end_time,
                        motion_energy=min(last.motion_energy, s.motion_energy),
                        suggested_trim=True,
                    )
            else:
                merged.append(s)
        return merged

    def process(
        self,
        video_path: Path,
        output_path: Path,
        stops: Optional[list[DeadSpot]] = None,
    ) -> dict:
        """
        Detect stops and produce a video with stops removed (hard cuts between segments).

        Args:
            video_path: Input video path
            output_path: Output video path (stops removed)
            stops: Pre-computed stops (if None, will detect automatically)

        Returns:
            Dict with processing results:
              stops_found, stops, total_removed_duration, output_path
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        # Check cache: skip if output is fresh. A cache hit must still return the
        # REAL stops/keep_segments persisted beside the output on the run that
        # produced the cut — otherwise a resumed run reads stops_found=0, the
        # stage skips switching current_video to the cut file and skips
        # STOP-VERIFY, and the encode silently consumes the stops-included
        # original while masking any prior verify failure.
        from .cache import is_output_fresh

        if is_output_fresh(video_path, output_path, min_size_bytes=1024 * 1024):
            persisted = self._load_stops_result(output_path)
            if persisted is not None:
                # Only reuse when the detection params that produced the cut
                # (motion_threshold / min_stop_duration) still match the current
                # config. A knob change (or a legacy sidecar with no recorded
                # params) means the cut may no longer reflect what we'd detect —
                # re-derive when the source is available; only trust a mismatched
                # cache when the source is gone (nothing left to re-derive from).
                if self._detection_params_match(persisted):
                    logger.info(
                        "  Output already exists and is newer than input — reusing "
                        "persisted stop result (%d stop(s))",
                        persisted["stops_found"],
                    )
                    persisted["cached"] = True
                    return persisted
                if not video_path.exists():
                    logger.warning(
                        "  Cached stop result was produced with different detection "
                        "params and the source is gone — trusting cached output as-is"
                    )
                    persisted["cached"] = True
                    return persisted
                logger.info(
                    "  Cached stop result's detection params (motion_threshold/"
                    "min_stop_duration) changed — re-deriving stops"
                )
                # fall through to detection/cut below (re-writes the sidecar)
            elif not video_path.exists():
                # No provenance sidecar and the source is gone — can't re-derive.
                logger.warning(
                    "  Cached output has no persisted stop result and input is gone "
                    "— trusting output as-is"
                )
                return {
                    "stops_found": 0,
                    "stops": [],
                    "total_removed_duration": 0.0,
                    "output_path": str(output_path),
                    "cached": True,
                }
            else:
                logger.info(
                    "  Cached output has no persisted stop result — re-deriving so the "
                    "resume still switches to the cut file and runs STOP-VERIFY"
                )
            # fall through to detection/cut below (re-writes the sidecar)

        # Step 1: Detect stops if not provided
        if stops is None:
            stops = self.detect_stops(video_path)

        if not stops:
            # No stops — symlink input to expected output so downstream paths
            # are consistent (all point into stops_removed/).
            if not output_path.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.symlink_to(video_path.resolve())
                logger.debug("  No stops — symlinked %s → %s", output_path.name, video_path.name)
            result = {
                "stops_found": 0,
                "stops": [],
                "total_removed_duration": 0.0,
                "output_path": str(output_path),
            }
            self._save_stops_result(output_path, result)
            return result

        # Step 2: Calculate keep segments (inverse of stops)
        keep_segments = self._calculate_keep_segments(stops, video_path)

        if not keep_segments:
            raise RuntimeError("No segments to keep after stop removal — all video is stops?")

        # Removed footage is source − kept: this INCLUDES the boundary-buffer
        # trim around each cut (dead_spots.trim_buffer_seconds), so the reported
        # figure matches the real output shrink rather than the raw stop spans.
        kept_duration = sum(end - start for start, end in keep_segments)
        total_removed = max(0.0, self._get_duration(video_path) - kept_duration)

        # Step 3: Cut segments and stitch with hard cuts (no re-encoding)
        self._cut_and_stitch(video_path, output_path, keep_segments)

        stop_details = [
            {
                "start": s.start_time,
                "end": s.end_time,
                "duration": s.duration,
                "motion_energy": s.motion_energy,
            }
            for s in stops
        ]

        result = {
            "stops_found": len(stops),
            "stops": stop_details,
            "total_removed_duration": total_removed,
            "output_path": str(output_path),
            "keep_segments": keep_segments,
        }
        self._save_stops_result(output_path, result)
        return result

    def _stops_result_path(self, output_path: Path) -> Path:
        """Sidecar JSON path holding this run's stops/keep_segments, beside the
        output. Persisted on every completed ``process`` so a cache-hit resume
        returns the real result (switching current_video + running STOP-VERIFY)
        instead of a masking stops_found=0."""
        return output_path.with_name(output_path.name + ".stops.json")

    def _detection_params(self) -> dict:
        """The detection knobs that determine which stops a cut reflects."""
        return {
            "motion_threshold": self.motion_threshold,
            "min_stop_duration": self.min_stop_duration,
        }

    def _detection_params_match(self, persisted: dict) -> bool:
        """True only when the sidecar records detection params equal to current.

        A missing ``detection_params`` block (a legacy sidecar) never matches —
        we can't prove the cut still reflects the current knobs, so it is treated
        as a mismatch and re-derived when the source is available.
        """
        params = persisted.get("detection_params")
        if not isinstance(params, dict):
            return False
        return params == self._detection_params()

    def _save_stops_result(self, output_path: Path, result: dict) -> None:
        """Persist the process result beside the output (best-effort)."""
        from .cache import save_cache_json

        payload = {k: v for k, v in result.items() if k != "cached"}
        # Record the detection knobs so a cache-hit resume can invalidate
        # the reuse when motion_threshold / min_stop_duration have changed.
        payload["detection_params"] = self._detection_params()
        try:
            save_cache_json(self._stops_result_path(output_path), payload)
        except OSError as exc:  # a sidecar we can't write must not fail the cut
            logger.warning("  Could not persist stop-result sidecar: %s", exc)

    def _load_stops_result(self, output_path: Path) -> Optional[dict]:
        """Load the persisted process result, or None when absent/unusable."""
        from .cache import load_cached_json

        data = load_cached_json(self._stops_result_path(output_path))
        if not data or "stops_found" not in data:
            return None
        if data.get("keep_segments") is not None:
            # JSON round-trips tuples as lists; restore (start, end) tuples.
            data["keep_segments"] = [tuple(seg) for seg in data["keep_segments"]]
        return data

    def _calculate_keep_segments(
        self, stops: list[DeadSpot], video_path: Path
    ) -> list[tuple[float, float]]:
        """
        Calculate segments to keep (everything that isn't a stop).

        Each stop cut is widened by ``dead_spots.trim_buffer_seconds`` on BOTH
        sides — the keep segment ends ``buffer`` seconds before the stop starts
        and resumes ``buffer`` seconds after it ends. This trims the ugly
        deceleration-into-stop / acceleration-out-of-stop footage that abuts a
        stop (previously kept verbatim, a hard zero-padding cut). Buffers are
        clamped to ``[0, duration]`` and can't rewind ``current_pos``, so a
        stop whose padded start precedes the prior cut just extends it.

        Returns list of (start_time, end_time) tuples.
        """
        # Get video duration via ffprobe
        duration = self._get_duration(video_path)
        buffer = float(getattr(self.config.dead_spots, "trim_buffer_seconds", 0) or 0)

        # Sort stops by start time
        sorted_stops = sorted(stops, key=lambda s: s.start_time)

        segments = []
        current_pos = 0.0

        for stop in sorted_stops:
            # Keep everything from current position up to the buffered stop start.
            seg_end = max(current_pos, stop.start_time - buffer)
            if seg_end > current_pos + 0.1:  # at least 100ms
                segments.append((current_pos, seg_end))
            # Resume after the buffered stop end (never rewind current_pos).
            current_pos = max(current_pos, min(duration, stop.end_time + buffer))

        # Keep everything after the last stop
        if current_pos < duration - 0.1:
            segments.append((current_pos, duration))

        return segments

    def _get_video_resolution(self, video_path: Path) -> tuple[int, int]:
        """Get video width and height via ffprobe."""
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v:0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return (1920, 1080)  # safe fallback
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if streams:
            return (int(streams[0].get("width", 1920)), int(streams[0].get("height", 1080)))
        return (1920, 1080)

    def _get_color_range(self, video_path: Path) -> str:
        """Probe the source ``color_range`` for the crossfade re-encode tag.

        The crossfade dissolve does NOT convert pixel values, so the output
        must carry the SOURCE's range. The old hardcoded ``tv`` (limited) tag
        mislabeled full-range GoPro pixels as limited, which mis-scaled the
        downstream rgb24 color-analysis sampling in single_pass and misled
        tag-trusting players. The hard-cut path already
        preserves the source tag via ``-c copy``; this makes the crossfade path
        consistent. Returns ``pc``/``tv`` as reported, defaulting to ``pc``
        (full) for untagged GoPro footage.
        """
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v:0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return "pc"
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        cr = str(streams[0].get("color_range", "")).strip().lower() if streams else ""
        return "tv" if cr in ("tv", "mpeg", "limited") else "pc"

    def _calc_timeout(self, duration_sec: float, video_path: Path, is_copy: bool = False) -> int:
        """
        Calculate a dynamic timeout based on segment duration and resolution.

        Args:
            duration_sec: Duration of the segment in seconds
            video_path: Path to the video (used to check resolution)
            is_copy: If True, this is a stream-copy operation (fast)

        Returns:
            Timeout in seconds
        """
        if is_copy:
            # Stream copy is fast but cross-mount I/O (WSL→NTFS) can be
            # very slow for large 4K files; allow generous headroom.
            return max(600, int(duration_sec * 2.0))

        # Use cached height if available (avoids ffprobe call)
        if self._video_height is not None:
            height = self._video_height
        else:
            _, height = self._get_video_resolution(video_path)
        if height >= 2160:
            multiplier = 4.0  # 4K re-encode
        elif height >= 1080:
            multiplier = 2.0  # 1080p
        else:
            multiplier = 1.5

        return max(300, int(duration_sec * multiplier))

    def _cut_and_stitch(
        self,
        video_path: Path,
        output_path: Path,
        segments: list[tuple[float, float]],
    ):
        """
        Cut video into segments and stitch them together.

        When crossfade_duration > 0 and multiple segments exist, applies
        crossfade transitions between adjacent segments. Otherwise uses
        stream copy (-c copy) for fast concatenation with hard cuts.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if len(segments) == 1:
            # Only one segment — just trim with stream copy
            start, end = segments[0]
            timeout = self._calc_timeout(end - start, video_path, is_copy=True)
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-i",
                str(video_path),
                "-t",
                str(end - start),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            self._run_ffmpeg(cmd, "trim single segment", timeout=timeout)
            if self.tracker:
                self.tracker.tick()
            return

        # Multiple segments — use ffmpeg concat demuxer:
        # 1. Stream-copy each segment to temp file (near-instant, no re-encoding)
        # 2. Write concat list file with all segment paths
        # 3. Use ffmpeg concat demuxer to join segments with hard cuts

        # Scratch dir for the stream-copied segments (~source size, multi-GB
        # 4K) must live under the operator-controlled staging base, never the
        # default tmpfs /tmp (ENOSPC / reboot-wipe mid-run). output_path.parent
        # is the run's staging dir, so co-locating keeps the workspace on the
        # same never-/tmp filesystem as the final cut by construction.
        with tempfile.TemporaryDirectory(prefix="gms_stops_", dir=output_path.parent) as tmpdir:
            tmpdir = Path(tmpdir)
            total_segs = len(segments)
            segment_files = []

            # Step 1: Extract all segments with stream copy (near-instant)
            for i, (start, end) in enumerate(segments):
                seg_path = tmpdir / f"seg_{i:03d}.mp4"
                duration = end - start
                timeout = self._calc_timeout(duration, video_path, is_copy=True)

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start),
                    "-i",
                    str(video_path),
                    "-t",
                    str(duration),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(seg_path),
                ]
                self._run_ffmpeg(cmd, f"stream-copy segment {i + 1}/{total_segs}", timeout=timeout)
                segment_files.append(seg_path)
                if self.tracker:
                    self.tracker.tick()

            # A segment shorter than the crossfade can't host the dissolve —
            # fall back to hard cuts (the xfade offset would go negative). Probe
            # the ACTUAL extracted durations (stream-copy keyframe-snaps the cut)
            # only when crossfade is on; the hard-cut path needs no probe.
            use_crossfade = False
            seg_durations: list[float] = []
            if self.crossfade_duration > 0:
                seg_durations = [self._get_duration(p) for p in segment_files]
                use_crossfade = min(seg_durations) > self.crossfade_duration
                if not use_crossfade:
                    logger.warning(
                        "  Crossfade %.2fs >= shortest kept segment (%.2fs) — "
                        "using hard cuts for this stitch",
                        self.crossfade_duration,
                        min(seg_durations),
                    )

            if use_crossfade:
                # Step 2: ONE encode crossfading ALL segments — every body survives.
                self._crossfade_stitch(segment_files, seg_durations, output_path, video_path)
                if self.tracker:
                    self.tracker.tick()
            else:
                # Step 2: Write concat list file (hard cuts)
                concat_list_path = tmpdir / "concat_list.txt"
                with open(concat_list_path, "w") as f:
                    for seg_file in segment_files:
                        escaped = str(seg_file).replace("'", "'\\''")
                        f.write(f"file '{escaped}'\n")

                # Step 3: Use ffmpeg concat demuxer to join all segments with hard cuts
                total_duration = sum(end - start for start, end in segments)
                timeout = self._calc_timeout(total_duration, video_path, is_copy=True)

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-fflags",
                    "+genpts",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_list_path),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]
                self._run_ffmpeg(cmd, "concat segments with hard cuts", timeout=timeout)
                if self.tracker:
                    self.tracker.tick()

    def _crossfade_stitch(
        self,
        segment_files: list[Path],
        seg_durations: list[float],
        output_path: Path,
        video_path: Path,
    ) -> None:
        """Crossfade-join ALL segments in ONE encode (every body preserved).

        Replaces the old per-pair ``_render_crossfade`` loop, which kept only
        the final transition and collapsed the output to ~``crossfade_duration``
        seconds. Output duration is ``Σ seg_durations − (N−1)·crossfade``.
        """
        # detect_nvenc() raises NVENCUnavailableError if the probe fails.
        # Single hevc_nvenc path; libx265 branch deleted.
        detect_nvenc()
        filter_complex, total_duration = build_stop_crossfade_filter(
            seg_durations, self.crossfade_duration
        )
        # hevc_nvenc at the frozen LONGFORM NVENC preset/CQ/profile + color tags
        # (matches single_pass). This stitch is a real extra generation on the way
        # to the master, so it takes the same quality knobs as the master encode:
        # the previous hardcoded ``p5`` silently encoded it a preset below frozen,
        # and ``LONGFORM_CRF`` is the x265 fallback knob — it landed on the right
        # NVENC ``-cq`` only because the two constants happen to share the value 18.
        codec = (
            f"-c:v hevc_nvenc -preset {ec.LONGFORM_NVENC_PRESET} -rc vbr "
            f"-cq {ec.LONGFORM_NVENC_CQ} -profile:v {ec.LONGFORM_PROFILE} "
            f"-pix_fmt {ec.LONGFORM_PIX_FMT}"
        ).split()
        # Range tag follows the SOURCE (no pixel conversion happens here) — the
        # stop-processed file is an intermediate that single_pass later squeezes
        # full->limited. bt709 primaries/trc are correct for GoPro either way.
        src_range = self._get_color_range(video_path)
        color = f"-color_range {src_range} -colorspace {ec.LONGFORM_COLORSPACE} -color_primaries {ec.LONGFORM_COLOR_PRIMARIES} -color_trc {ec.LONGFORM_COLOR_TRC}".split()
        cmd = ["ffmpeg", "-y"]
        for seg in segment_files:
            cmd += ["-i", str(seg)]
        cmd += ["-filter_complex", filter_complex, "-map", "[v]", "-map", "[aud]", "-c:a", "aac"]
        cmd += codec + color + [str(output_path)]
        timeout = self._calc_timeout(total_duration, video_path, is_copy=False)
        logger.debug(
            "Crossfade-stitching %d segments (NVENC), output ~%.1fs",
            len(segment_files),
            total_duration,
        )
        self._run_ffmpeg(cmd, "crossfade stitch all segments", timeout=timeout)

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration via ffprobe.

        Probes container format first, then recomputes from the video stream's
        own ``duration`` if the container lacks one (common for raw/remuxed
        streams). If neither path yields a positive duration, raises
        :class:`ProbeFailedError` rather than returning 0 — a fabricated 0 would
        silently truncate a trailing stop. There is no silent fallback.
        """
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise ProbeFailedError(
                video_path, f"ffprobe exited {result.returncode}: {result.stderr}"
            )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ProbeFailedError(video_path, f"ffprobe returned invalid JSON: {exc}") from exc

        # Prefer container duration; recompute from the video stream when absent.
        duration = float(data.get("format", {}).get("duration", 0) or 0)
        if duration <= 0:
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    duration = float(stream.get("duration", 0) or 0)
                    if duration > 0:
                        break

        if duration <= 0:
            raise ProbeFailedError(
                video_path, "ffprobe reported no positive duration in format or streams"
            )
        return duration

    def _run_ffmpeg(self, cmd: list[str], description: str, timeout: int = 3600):
        """Run an FFmpeg command with error handling and dynamic timeout."""
        logger.debug(f"  FFmpeg: {description} (timeout={timeout}s)")
        # Determine output path from last argument (stream copy ops always end with output path)
        output_path = Path(cmd[-1]) if cmd else None
        run_ffmpeg_quick(cmd, output_path=output_path, timeout=timeout, description=description)
