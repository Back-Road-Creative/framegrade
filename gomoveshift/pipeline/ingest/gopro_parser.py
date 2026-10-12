"""GoPro filename parsing, data models, and directory scanning.

GoPro Hero cameras split continuous recordings into ~17min chapter files with naming:
  GXccNNNN.MP4  where cc=chapter (01,02,...), NNNN=recording session number

This module:
1. Parses filenames -> groups chapters by recording number
2. Scans directories for GoPro MP4 files
3. Finds GPS-based split points for long recordings
4. Reads camera model/firmware for the per-clip record
"""

import json
import logging
import math
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# -- Container-tag keys GoPro (and other cameras) use for model and
# firmware. ffprobe surfaces MP4 udta atoms as ``format.tags``; the exact key
# varies by camera/firmware generation, so several known spellings are tried
# in order. Absent every key, the field stays ``None`` -- never invented.
_CAMERA_MODEL_TAG_KEYS = ("com.gopro.model", "model", "com.apple.quicktime.model")
_CAMERA_FIRMWARE_TAG_KEYS = ("firmware", "com.gopro.firmware", "com.apple.quicktime.software")

# -- GoPro filename pattern --
# GXccNNNN.MP4 — cc = chapter (2 digits), NNNN = recording number (4 digits)
# Also handles GH (HEVC) prefix
GOPRO_RE = re.compile(r"^G[XH](\d{2})(\d{4})\.(MP4|mp4)$")

# -- Duration rules (seconds) --
MIN_CLIP_DURATION = 10  # Filter accidental presses
MIN_EPISODE_DURATION = 20 * 60  # 20 min — shortest publishable episode
MAX_EPISODE_DURATION = 75 * 60  # 75 min — try to split above this
TARGET_EPISODE_DURATION = 60 * 60  # 60 min — ideal target

# -- Grouping rules --
MAX_GAP_BETWEEN_RECORDINGS = 120 * 60  # 2 hours — combine if closer
GPS_RETURN_RADIUS_KM = 0.5  # 500m — "returned to start"
GPS_STOP_DURATION = 120  # 2 min stationary by GPS = natural stop


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class GoProChapter:
    """A single GoPro chapter file."""

    path: Path
    chapter_num: int  # 1-based chapter index
    recording_num: int  # 4-digit recording session number
    duration: float  # seconds
    creation_time: Optional[datetime] = None
    # Per-clip camera model/firmware, read from ffprobe container
    # tags. None when the container carries no recognized tag -- never
    # invented, never blocks ingest.
    camera_model: Optional[str] = None
    firmware: Optional[str] = None


@dataclass
class GoProRecording:
    """A continuous GoPro recording (multiple chapter files)."""

    recording_num: int
    chapters: list[GoProChapter] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return sum(ch.duration for ch in self.chapters)

    @property
    def creation_time(self) -> Optional[datetime]:
        if self.chapters:
            return self.chapters[0].creation_time
        return None

    @property
    def end_time(self) -> Optional[datetime]:
        """Estimated end time = creation_time + total_duration."""
        ct = self.creation_time
        if ct:
            return ct + timedelta(seconds=self.total_duration)
        return None


@dataclass
class Episode:
    """A group of recordings concatenated into one publishable video.

    Two construction shapes:

    1. **Chapter-derived** (the common case): ``recordings`` is populated
       by the parser, and ``total_duration`` / ``creation_time`` are
       derived from the chapter metadata.

    2. **Slice-derived sub-episode** (post-GPS-split): produced by
       ``_split_episode_at_points`` after a long episode is sliced into
       smaller chunks. The slice is the new source of truth — the
       original chapter recordings no longer describe its bytes — so
       ``recordings`` MUST be empty and ``_explicit_total_duration`` /
       ``_explicit_creation_time`` carry the slice's properties.
       ``concat_path`` and ``gpx_path`` are set up-front by the splitter.

    The properties below resolve in this priority: ``_explicit_*`` field
    if set, else derive from ``recordings``. This lets sub-episodes
    survive every downstream consumer that calls ``ep.total_duration`` /
    ``ep.creation_time`` / ``ep.total_duration_min`` without the
    consumer needing to know which shape it has.
    """

    index: int  # 1-based episode number
    recordings: list[GoProRecording] = field(default_factory=list)
    location: str = ""
    date: str = ""  # YYYY-MM-DD
    concat_path: Optional[Path] = None  # Output path after concatenation
    gpx_path: Optional[Path] = None  # Merged GPX path
    # GPX provenance. ``"gpmf"`` = extracted from THIS episode's own
    # GoPro GPMF chapter telemetry — frame-aligned by construction, so the
    # correspondence gate skips it at zero cost. ``"external"`` = a GPX set
    # up-front (operator-supplied generic-source sidecar, or a resume state
    # pairing) whose temporal/spatial alignment to the video is NOT
    # guaranteed — exactly the swap surface the gate targets. ``"gpmf_slice"``
    # = a sub-episode slice cut from the parent's own GPMF (never a swapped
    # track, but a gross anchoring/slicing error can mis-window it), so
    # Phase A runs a slice-tuned gate (start-time + one-sided over-span only).
    gpx_source: str = "gpmf"
    # Slice-derived sub-episode overrides. None on chapter-derived
    # episodes so the property derivations (sum of recordings) fire.
    # Set by ``_split_episode_at_points`` to the slice's actual extent.
    _explicit_total_duration: Optional[float] = None
    _explicit_creation_time: Optional[datetime] = None

    @property
    def total_duration(self) -> float:
        if self._explicit_total_duration is not None:
            return self._explicit_total_duration
        return sum(r.total_duration for r in self.recordings)

    @property
    def total_duration_min(self) -> float:
        return self.total_duration / 60

    @property
    def num_chapters(self) -> int:
        return sum(len(r.chapters) for r in self.recordings)

    @property
    def creation_time(self) -> Optional[datetime]:
        if self._explicit_creation_time is not None:
            return self._explicit_creation_time
        if self.recordings:
            return self.recordings[0].creation_time
        return None

    def chapter_files(self) -> list[Path]:
        """All chapter file paths in playback order.

        Empty on slice-derived sub-episodes — the slice file at
        ``concat_path`` IS the source-of-truth bytes; downstream
        ``concat_episode`` short-circuits on the existing ``concat_path``
        instead of re-concatenating from chapters.
        """
        files = []
        for rec in self.recordings:
            for ch in sorted(rec.chapters, key=lambda c: c.chapter_num):
                files.append(ch.path)
        return files


@dataclass
class IngestPlan:
    """Complete ingestion plan for a directory of GoPro clips."""

    source_dir: Path
    episodes: list[Episode] = field(default_factory=list)
    filtered_clips: list[GoProChapter] = field(default_factory=list)
    unrecognized_files: list[Path] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "IngestPlan":
        """Reconstruct a minimal ``IngestPlan`` from a serialized dict.

        Used by ``PipelineState.from_dict`` to re-hydrate the ``ingest_plan``
        field on resume. ``source_dir`` is coerced back to ``Path``; nested
        dataclass collections (``episodes``, ``filtered_clips``,
        ``unrecognized_files``) are **not** deep-rehydrated — they pass
        through as the raw ``list[dict]`` they were serialized as, because
        deep rehydration would require matching ``from_dict`` plumbing on
        four nested dataclasses (``Episode``, ``Recording``, ``Chapter``,
        ``GoProChapter``).

        **Caller responsibility.** Any consumer that performs attribute
        access on ``episodes`` (e.g. ``ep.date``, ``ep.creation_time``)
        MUST detect the dict-children case and synthesize from the parallel
        state arrays (``episode_files`` / ``episode_dates`` / etc.) instead
        — see ``phase_bc_day_first._resolve_ingest_plan``. ``getattr`` on a
        dict silently returns the default, so the symptom of forgetting
        this is "every episode skipped, 0 day batches, phase reports
        success on no work".
        """
        source_dir = d.get("source_dir")
        if source_dir is not None and not isinstance(source_dir, Path):
            source_dir = Path(source_dir)
        return cls(
            source_dir=source_dir,
            episodes=d.get("episodes", []) or [],
            filtered_clips=d.get("filtered_clips", []) or [],
            unrecognized_files=d.get("unrecognized_files", []) or [],
        )

    @property
    def total_duration_min(self) -> float:
        return sum(ep.total_duration_min for ep in self.episodes)

    @property
    def total_source_files(self) -> int:
        return sum(ep.num_chapters for ep in self.episodes)

    def summary(self) -> str:
        lines = [
            f"Ingest Plan: {self.source_dir}",
            f"  {self.total_source_files} source files → {len(self.episodes)} episodes "
            f"({self.total_duration_min:.0f} min total)",
        ]
        if self.filtered_clips:
            lines.append(f"  {len(self.filtered_clips)} clips filtered (< {MIN_CLIP_DURATION}s)")
        if self.unrecognized_files:
            lines.append(f"  {len(self.unrecognized_files)} non-GoPro files skipped")

        lines.append("")
        for ep in self.episodes:
            rec_nums = [f"{r.recording_num:04d}" for r in ep.recordings]
            lines.append(
                f"  Episode {ep.index}: {ep.total_duration_min:.0f}min "
                f"({ep.num_chapters} files, recordings: {'+'.join(rec_nums)})"
            )
            lines.append(f"    Date: {ep.date}  Location: {ep.location or '(pending GPS)'}")
            for rec in ep.recordings:
                lines.append(
                    f"    Rec {rec.recording_num:04d}: "
                    f"{len(rec.chapters)} chapters, {rec.total_duration / 60:.1f}min"
                )

        return "\n".join(lines)


# ============================================================================
# GoPro Filename Parsing
# ============================================================================


def parse_gopro_filename(filename: str) -> Optional[tuple[int, int]]:
    """Parse GoPro filename into (chapter_num, recording_num).

    Args:
        filename: e.g. "GX010001.MP4"

    Returns:
        (chapter_num, recording_num) or None if not a GoPro file
    """
    m = GOPRO_RE.match(filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def extract_camera_metadata(video_path: "str | Path") -> tuple[Optional[str], Optional[str]]:
    """Read ``(camera_model, firmware)`` from the container's ffprobe format
    tags, or ``(None, None)`` on any failure or absent tags.

    This is a container-tag read, not a GPMF binary parse -- ffprobe's
    ``format.tags`` already surfaces the MP4 udta atoms most cameras (GoPro
    included) write, and every existing ffprobe call in this pipeline
    already goes through subprocess the same way. Never raises: a clip with
    no recognized tag is a legitimate ``(None, None)``, not an error.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None, None
        probe = json.loads(result.stdout)
        tags = probe.get("format", {}).get("tags", {}) or {}
        # Tag keys are case-insensitive in practice across muxers.
        lower_tags = {str(k).lower(): v for k, v in tags.items()}

        model = None
        for key in _CAMERA_MODEL_TAG_KEYS:
            if key.lower() in lower_tags:
                model = lower_tags[key.lower()]
                break

        firmware = None
        for key in _CAMERA_FIRMWARE_TAG_KEYS:
            if key.lower() in lower_tags:
                firmware = lower_tags[key.lower()]
                break

        return model, firmware
    except Exception as e:
        logger.debug("extract_camera_metadata failed for %s: %s", video_path, e)
        return None, None


def get_clip_duration(video_path: Path) -> float:
    """Get video duration via ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Failed to get duration for {video_path.name}: {e}")
    return 0.0


# ============================================================================
# Scanning & Grouping
# ============================================================================


def _is_gopro_named(name: str) -> bool:
    """Return True if ``name`` matches the GoPro chapter filename pattern."""
    return GOPRO_RE.match(name) is not None


def _ffprobe_creation_time(video_path: Path) -> Optional[datetime]:
    """Read ``format.tags.creation_time`` via ffprobe; None on any failure."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags=creation_time",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        raw = (result.stdout or "").strip()
        if result.returncode != 0 or not raw:
            return None
        # ffmpeg writes ISO-8601 with a trailing 'Z' for UTC. Python's
        # fromisoformat accepts the 'Z' suffix from 3.11 on, but we support
        # 3.10 — normalize it to the explicit '+00:00' offset instead of
        # stripping it, so the result stays TIMEZONE-AWARE. Generic-mode
        # Insta360/DJI/phone sources stamp true UTC; a naive parse silently
        # dropped the zone and let a UTC evening drive key into the next
        # calendar day. Offset-bearing strings (e.g. the GoPro '-05:00' form)
        # are already tz-aware and pass through unchanged.
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except Exception as e:
        logger.debug("ffprobe creation_time failed for %s: %s", video_path.name, e)
        return None


def scan_video_directory(
    directory: Path,
    extensions: list[str] | tuple[str, ...] | None = None,
    mode: str = "gopro",
    min_clip_duration: float = MIN_CLIP_DURATION,
    strict_mode: bool = True,
) -> tuple[dict[int, GoProRecording], list[GoProChapter], list[Path]]:
    """Scan ``directory`` for video files and group into ``GoProRecording``s.

    The accepted-extension list and the grouping strategy are explicit.
    ``mode`` is total — every
    accepted file is dispatched by mode:

    - ``mode="gopro"``: existing chapter+session regex grouping. Files
      whose name doesn't match ``GOPRO_RE`` AND aren't filtered for
      duration land in ``unrecognized_files``.
    - ``mode="generic"``: each accepted file becomes its own singleton
      "recording" (one chapter, one recording). Episode start time is
      ``ffprobe creation_time`` when present; when absent, a GPX sidecar
      (``<name>.gpx`` beside the file) must date the recording, else the
      scan hard-fails — file ``st_mtime`` is copy time, never the
      recording day. No chapter/session grouping.

    Mixed mode is refused: a GoPro-named file under ``mode="generic"``,
    or a non-GoPro-named accepted-extension file under ``mode="gopro"``,
    raises ``ValueError`` naming the conflicting files. Keeps the mode
    contract total — operators must commit to one source layout per run.

    Returns:
        (recordings_dict, filtered_clips, unrecognized_files)
    """
    from ..gopro_gps import extract_recording_date

    if mode not in ("gopro", "generic"):
        raise ValueError(f"scan_video_directory: mode must be 'gopro' or 'generic', got {mode!r}")

    directory = Path(directory)
    recordings: dict[int, GoProRecording] = {}
    filtered: list[GoProChapter] = []
    unrecognized: list[Path] = []

    if extensions is None:
        # Historical default — preserves callers who haven't been
        # plumbed for the new contract yet.
        exts_lower = {".mp4"}
    else:
        exts_lower = {e.lower() for e in extensions}

    # Find all accepted-extension files (case-insensitive).
    mp4_files = sorted(
        [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in exts_lower],
        key=lambda f: f.name,
    )

    # Mixed-mode refusal: keeps the mode contract total. Operator
    # must commit to one source layout per run. ``strict_mode=False``
    # exists only for the legacy ``scan_gopro_directory`` shim which
    # historically routed non-GoPro names into ``unrecognized_files``
    # rather than raising.
    if strict_mode:
        if mode == "generic":
            bad = [f for f in mp4_files if _is_gopro_named(f.name)]
            if bad:
                names = ", ".join(b.name for b in bad)
                raise ValueError(
                    "scan_video_directory(mode='generic') refuses GoPro-named "
                    f"file(s) in source dir: {names}. Run with --ingest-mode "
                    "gopro or remove the conflicting files."
                )
        elif mode == "gopro":
            bad = [f for f in mp4_files if not _is_gopro_named(f.name)]
            if bad:
                names = ", ".join(b.name for b in bad)
                raise ValueError(
                    "scan_video_directory(mode='gopro') refuses non-GoPro-named "
                    f"accepted-extension file(s) in source dir: {names}. Run "
                    "with --ingest-mode generic or remove the conflicting files."
                )

    if not mp4_files:
        logger.warning(f"No video files found in {directory}")
        return {}, [], []

    logger.info(f"Found {len(mp4_files)} video files in {directory} (mode={mode})")

    if mode == "generic":
        # Generic mode: each accepted file is its own singleton episode.
        # No chapter/session grouping. Recording-number key is the
        # 1-based discovery order so downstream dict-keyed lookups
        # remain stable.
        for idx, vid in enumerate(mp4_files, start=1):
            try:
                duration = get_clip_duration(vid)
            except Exception as e:
                logger.warning(f"  Skipping {vid.name}: failed to get duration: {e}")
                unrecognized.append(vid)
                continue

            creation_time = _ffprobe_creation_time(vid)
            if creation_time is None:
                # No container creation_time. The recording day must come
                # from a real clock — ffprobe's creation_time or the GPX
                # first fix (day_grouper derives the longitude-local day
                # from it). File mtime is *copy* time, not recording time:
                # a day-N drive offloaded on day M would silently group into
                # day M's LF/journal slot, mixing a host-local-tz-naive clock
                # with the UTC-naive ffprobe base. Require a GPX sidecar to
                # date the recording, or hard-fail loudly.
                sidecar_gpx = vid.with_suffix(".gpx")
                if not (sidecar_gpx.exists() and sidecar_gpx.stat().st_size > 0):
                    raise ValueError(
                        f"generic-mode ingest: {vid.name} has no container "
                        "creation_time (ffprobe) and no GPX sidecar "
                        f"({sidecar_gpx.name}) to date the recording. File "
                        "mtime is copy time, not the recording day — refusing "
                        "to stamp it. Provide a GPX sidecar next to the file "
                        "or repair the file's creation_time metadata."
                    )
                # GPX sidecar present: leave creation_time None so the day
                # key derives from the GPX first fix (longitude-local),
                # never the file's copy time.

            chapter = GoProChapter(
                path=vid,
                chapter_num=1,
                recording_num=idx,
                duration=duration,
                creation_time=creation_time,
            )

            if duration < min_clip_duration:
                filtered.append(chapter)
                logger.debug(f"  Filtered {vid.name}: {duration:.1f}s < {min_clip_duration}s")
                continue

            recordings[idx] = GoProRecording(recording_num=idx, chapters=[chapter])

        logger.info(
            f"  generic-mode: {len(recordings)} singleton recordings, "
            f"{len(filtered)} filtered, {len(unrecognized)} unrecognized"
        )
        return recordings, filtered, unrecognized

    # mode == "gopro" — historical chapter/session regex grouping.
    for mp4 in mp4_files:
        parsed = parse_gopro_filename(mp4.name)
        if parsed is None:
            unrecognized.append(mp4)
            continue

        chapter_num, rec_num = parsed
        try:
            duration = get_clip_duration(mp4)
        except Exception as e:
            logger.warning(f"  Skipping {mp4.name}: failed to get duration: {e}")
            unrecognized.append(mp4)
            continue

        # Extract creation time
        creation_time = extract_recording_date(mp4)
        camera_model, firmware = extract_camera_metadata(mp4)

        chapter = GoProChapter(
            path=mp4,
            chapter_num=chapter_num,
            recording_num=rec_num,
            duration=duration,
            creation_time=creation_time,
            camera_model=camera_model,
            firmware=firmware,
        )

        if duration < min_clip_duration:
            filtered.append(chapter)
            logger.debug(f"  Filtered {mp4.name}: {duration:.1f}s < {min_clip_duration}s")
            continue

        if rec_num not in recordings:
            recordings[rec_num] = GoProRecording(recording_num=rec_num)
        recordings[rec_num].chapters.append(chapter)

    # Sort chapters within each recording by chapter number
    for rec in recordings.values():
        rec.chapters.sort(key=lambda c: c.chapter_num)

    logger.info(
        f"  {len(recordings)} recordings, "
        f"{sum(len(r.chapters) for r in recordings.values())} clips, "
        f"{len(filtered)} filtered, {len(unrecognized)} unrecognized"
    )

    return recordings, filtered, unrecognized


def scan_gopro_directory(
    directory: Path,
    min_clip_duration: float = MIN_CLIP_DURATION,
) -> tuple[dict[int, GoProRecording], list[GoProChapter], list[Path]]:
    """Backwards-compatibility shim — delegates to ``scan_video_directory``.

    Preserves the historical contract for callers (tests, ad-hoc tools)
    that pre-date the mode/extensions plumbing: GoPro mode, ``.MP4``
    extension only. New code should call ``scan_video_directory``
    directly with explicit ``mode`` + ``extensions``.

    Note: the legacy shim used ``f.suffix.upper() == ".MP4"`` which
    accepted both ``.mp4`` and ``.MP4``. ``scan_video_directory`` does
    case-insensitive matching against the supplied extension list,
    yielding the same set of files.
    """
    return scan_video_directory(
        directory=directory,
        extensions=[".mp4"],
        mode="gopro",
        min_clip_duration=min_clip_duration,
        strict_mode=False,
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates in km."""
    earth_radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


# Maximum allowed gap between consecutive valid waypoints. GoPro GPS
# loss followed by re-acquisition with a stale/wrong system clock can
# drop a waypoint with a timestamp days/years off — we treat any gap
# beyond this as a hard signal that the later waypoint is corrupt and
# truncate the track there. 2h covers any realistic real-world stop
# (lunch, fuel) without partitioning a clean recording.
GPS_MAX_INTERWAYPOINT_GAP_SEC: float = 2 * 60 * 60


def _is_invalid_coord(lat: float, lon: float) -> bool:
    """Return True for GPS sentinels that should be filtered out.

    Catches: null-island (0,0) and physically impossible coords
    (|lat| > 90 or |lon| > 180). Mirrors the validity heuristic in
    ``gps_journey.core.models`` so the two services agree on what an
    invalid waypoint looks like.
    """
    if lat is None or lon is None:
        return True
    if abs(lat) > 90 or abs(lon) > 180:
        return True
    if abs(lat) < 0.001 and abs(lon) < 0.001:
        return True
    return False


def _sanitize_waypoints(waypoints: list) -> list:
    """Drop invalid-coord waypoints and truncate at large time gaps.

    Two failure modes seen on real GoPro data:

    1. GPS lock loss writes (0,0) coords for one or more waypoints. The
       splitter's "return-to-start" check then matches on every (0,0)
       waypoint when start is also (0,0), or when ``_haversine_km``
       returns 0 because the start happens to coincide with null island
       in the projected math. Filter these out before scanning.

    2. After GPS re-acquisition with a stale system clock, a waypoint
       can land days/years away from its neighbours. The first such
       waypoint inflates ``waypoints[-1] - waypoints[0]`` to absurd
       values (years of drift have been observed),
       which then produces split candidates at elapsed timestamps
       beyond the actual file duration. Truncating the track at the
       first inter-waypoint gap > ``GPS_MAX_INTERWAYPOINT_GAP_SEC``
       keeps the splitter inside the real recording window.
    """
    valid = [wp for wp in waypoints if not _is_invalid_coord(wp.latitude, wp.longitude)]
    if len(valid) < 2:
        return valid
    truncated = [valid[0]]
    for wp in valid[1:]:
        gap = (wp.timestamp - truncated[-1].timestamp).total_seconds()
        if gap > GPS_MAX_INTERWAYPOINT_GAP_SEC:
            logger.warning(
                "  Truncating GPS track: %.0fh gap between waypoints "
                "(%s → %s); later points discarded as corrupt.",
                gap / 3600,
                truncated[-1].timestamp.isoformat(),
                wp.timestamp.isoformat(),
            )
            break
        truncated.append(wp)
    return truncated


def find_gps_split_points(
    gpx_path: Path,
    max_episode_duration: float = MAX_EPISODE_DURATION,
    min_episode_duration: float = MIN_EPISODE_DURATION,
    return_radius_km: float = GPS_RETURN_RADIUS_KM,
    stop_duration: float = GPS_STOP_DURATION,
    max_total_duration: Optional[float] = None,
) -> list[float]:
    """Find natural GPS-based split points within a recording.

    Looks for:
    1. Return to within return_radius_km of recording start
    2. Extended stops (GPS speed ~ 0 for > stop_duration seconds)

    Only considers splits that would produce segments >= min_episode_duration.

    Args:
        max_total_duration: Optional hard upper bound on total elapsed
            seconds (typically the source MP4 duration). Any candidate
            split with elapsed beyond this bound is discarded — defends
            against corrupt GPX timestamps producing physically
            impossible split offsets. Stale-clock waypoints can otherwise yield
            candidates at hours or years past a source's real length, so
            ffmpeg slices past EOF and times out.

    Returns:
        List of timestamps (seconds from start) where natural splits occur.
        Empty if no good splits found or recording is under max_episode_duration.
    """
    try:
        from gps_journey.core.gpx_parser import GPXParser

        parser = GPXParser()
        journey = parser.parse(gpx_path)
    except Exception as e:
        logger.warning(f"  GPS split analysis failed: {e}")
        return []

    waypoints = _sanitize_waypoints(journey.waypoints)
    if len(waypoints) < 10:
        return []

    total_duration = (waypoints[-1].timestamp - waypoints[0].timestamp).total_seconds()
    if total_duration <= max_episode_duration:
        return []
    # Defence in depth: even after sanitization a single corrupt waypoint
    # that survives both filters could push total_duration past the
    # source MP4 length. Cap to the caller-supplied bound.
    if max_total_duration is not None and total_duration > max_total_duration:
        total_duration = max_total_duration

    start_lat = waypoints[0].latitude
    start_lon = waypoints[0].longitude
    start_time = waypoints[0].timestamp

    candidates: list[tuple[float, str]] = []  # (seconds_from_start, reason)

    # Scan for return-to-start and extended stops
    stationary_start: Optional[datetime] = None
    prev_wp = waypoints[0]

    for wp in waypoints[1:]:
        elapsed = (wp.timestamp - start_time).total_seconds()

        # Skip waypoints whose elapsed exceeds the validated total
        # duration — defends against any corrupt waypoint that slipped
        # past _sanitize_waypoints (e.g. a single bad point inside an
        # otherwise-clean cluster, less than the gap threshold from
        # both neighbours but with an absurd elapsed when measured from
        # start_time).
        if elapsed > total_duration:
            prev_wp = wp
            continue

        # Skip if too early or too late for a useful split
        if elapsed < min_episode_duration:
            prev_wp = wp
            continue
        if total_duration - elapsed < min_episode_duration:
            break

        # Check 1: Return to start
        dist = _haversine_km(start_lat, start_lon, wp.latitude, wp.longitude)
        if dist <= return_radius_km:
            candidates.append((elapsed, f"return-to-start ({dist:.2f}km)"))
            # Don't flood with consecutive return points
            prev_wp = wp
            continue

        # Check 2: Extended stop (low speed between consecutive waypoints)
        inter_dist = _haversine_km(prev_wp.latitude, prev_wp.longitude, wp.latitude, wp.longitude)
        inter_time = (wp.timestamp - prev_wp.timestamp).total_seconds()
        if inter_time > 0:
            speed_kmh = (inter_dist / inter_time) * 3600
            if speed_kmh < 2:  # Nearly stationary
                if stationary_start is None:
                    stationary_start = prev_wp.timestamp
                stationary_elapsed = (wp.timestamp - stationary_start).total_seconds()
                if stationary_elapsed >= stop_duration:
                    candidates.append((elapsed, f"extended-stop ({stationary_elapsed:.0f}s)"))
            else:
                stationary_start = None
        else:
            stationary_start = None

        prev_wp = wp

    if not candidates:
        return []

    # Pick the best split point(s) — prefer splits closest to target duration
    splits = []
    remaining = total_duration
    cursor = 0.0

    for ts, reason in sorted(candidates, key=lambda x: x[0]):
        segment_len = ts - cursor
        if segment_len >= min_episode_duration and remaining - segment_len >= min_episode_duration:
            # Prefer splits near the target duration
            if segment_len >= TARGET_EPISODE_DURATION * 0.8:
                splits.append(ts)
                logger.info(f"  GPS split at {ts / 60:.1f}min: {reason}")
                cursor = ts
                remaining = total_duration - cursor

    return splits
