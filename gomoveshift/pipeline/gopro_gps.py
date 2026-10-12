"""
GoPro GPMF GPS extraction.

Auto-detects GoPro telemetry streams in MP4 files and extracts GPS data
to GPX format for use by the existing chapter generation pipeline.

Uses gopro-overlay library for GPMF parsing (maintained, Python 3.12+).
"""

import datetime
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from ._atomic_write import atomic_write_text

logger = logging.getLogger(__name__)

# -- GPX↔video correspondence gate tolerances --
# Start-time skew: a corresponding GPX starts within an hour of the video's
# container creation_time (timezone offsets, GPS warm-up, clock drift).
# Beyond that the GPX almost certainly belongs to a DIFFERENT drive.
GPX_START_TIME_TOLERANCE_SEC: float = 60 * 60
# Span skew: the GPX should span roughly the video's duration. Allow a
# generous floor + relative slack for sampling tails / a few dropped fixes,
# but a 30-min track on a 10-min video is a swap.
GPX_SPAN_ABS_TOLERANCE_SEC: float = 120.0
GPX_SPAN_REL_TOLERANCE: float = 0.25
# First-fix vs embedded ISO-6709 tag: same drive ⇒ within a few km.
GPX_EMBEDDED_COORD_RADIUS_KM: float = 25.0


class GpxCorrespondenceError(RuntimeError):
    """The GPX does not temporally/spatially correspond to the video.

    Raised by ``assert_gpx_video_correspondence`` on the resume +
    generic-external-GPX paths when a swapped/mismatched GPX is detected
    BEFORE it can stamp a wrong-drive location onto the video. GoPro GPMF
    is frame-aligned by construction and never reaches this gate.
    """


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two coords."""
    import math

    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def assert_gpx_video_correspondence(
    gps_data,
    video_creation_time: Optional["datetime.datetime"],
    video_duration: float,
    embedded_coord: Optional[tuple[float, float]] = None,
) -> None:
    """Assert the GPX temporally/spatially corresponds to its video.

    Targets the resume + generic-external-GPX paths only (GoPro GPMF is
    frame-aligned → caller skips). Gates, in order:

    1. **Start-time** — ``|gpx.start − video.creation_time| <``
       ``GPX_START_TIME_TOLERANCE_SEC``. Skipped when either timestamp is
       unavailable (degrade, never raise on missing data).
    2. **Span** — ``|gpx.span − video.duration|`` within an absolute floor
       OR a relative fraction of the video duration.
    3. **Embedded coord** — when the video carries an ISO-6709 ``location``
       tag, the GPX first fix must be within
       ``GPX_EMBEDDED_COORD_RADIUS_KM`` of it.

    **Degrade-not-fail:** missing GPS points, missing creation_time, or no
    embedded tag never raise — but the span/start-time checks STILL run
    whenever their inputs exist (the no-embedded-tag external-GPX case is
    exactly where a swapped GPX slips through, so it must not be a free
    pass). Raises ``GpxCorrespondenceError`` on a genuine mismatch.
    """
    if gps_data is None:
        return
    trackpoints = getattr(gps_data, "trackpoints", None) or []
    if not trackpoints:
        logger.debug("GPX correspondence: no trackpoints — skipping gate")
        return

    gpx_start = trackpoints[0].timestamp
    gpx_end = trackpoints[-1].timestamp
    gpx_span = gpx_end - gpx_start

    # 1. Start-time correspondence (only when we have a video clock).
    if video_creation_time is not None:
        video_start = video_creation_time.timestamp()
        start_delta = abs(gpx_start - video_start)
        if start_delta > GPX_START_TIME_TOLERANCE_SEC:
            raise GpxCorrespondenceError(
                f"GPX-CORRESPONDENCE: GPX start-time off by {start_delta / 60:.0f} min "
                f"(> {GPX_START_TIME_TOLERANCE_SEC / 60:.0f} min) from video creation_time "
                f"— GPX likely belongs to a different drive."
            )

    # 2. Span correspondence.
    if video_duration and video_duration > 0:
        span_delta = abs(gpx_span - video_duration)
        span_tol = max(GPX_SPAN_ABS_TOLERANCE_SEC, GPX_SPAN_REL_TOLERANCE * video_duration)
        if span_delta > span_tol:
            raise GpxCorrespondenceError(
                f"GPX-CORRESPONDENCE: GPX span {gpx_span:.0f}s diverges from video duration "
                f"{video_duration:.0f}s by {span_delta:.0f}s (> {span_tol:.0f}s tolerance) "
                f"— GPX likely belongs to a different drive."
            )

    # 3. Embedded ISO-6709 coord anchor (only when present — degrade else).
    if embedded_coord is not None:
        elat, elon = embedded_coord
        flat = trackpoints[0].lat
        flon = trackpoints[0].lon
        if flat is not None and flon is not None:
            dist_km = _haversine_km(flat, flon, elat, elon)
            if dist_km > GPX_EMBEDDED_COORD_RADIUS_KM:
                raise GpxCorrespondenceError(
                    f"GPX-CORRESPONDENCE: GPX first fix is {dist_km:.0f} km from the video's "
                    f"embedded location tag (> {GPX_EMBEDDED_COORD_RADIUS_KM:.0f} km) "
                    f"— GPX likely belongs to a different drive."
                )
    else:
        logger.debug("GPX correspondence: no embedded coord — temporal gate only")


def assert_slice_gpx_correspondence(
    gps_data,
    video_creation_time: Optional["datetime.datetime"],
    video_duration: float,
) -> None:
    """Correspondence gate for a GPMF *sub-slice*.

    A sub-slice's GPX is cut from the parent's own merged GPMF, so it can
    never be a *swapped* track — but a gross anchoring/slicing error can
    still hand a slice a GPX window that doesn't match its video. This
    variant of the external gate is tuned to a slice's realities:

    * **Start-time** — identical to the external gate: the sliced GPX's
      first fix must be within ``GPX_START_TIME_TOLERANCE_SEC`` of the
      slice's video creation_time. A leading GPS-lock delay (unlocked
      prefix) stays inside the hour tolerance; a wrong-epoch pairing does
      not.
    * **Span (one-sided)** — a slice may legitimately *under*-cover its
      video (a lock-delay prefix or a mid-clip dropout leaves the GPX
      shorter than the footage), so under-coverage NEVER raises. Only an
      *over*-covering GPX (span exceeds the video by more than tolerance)
      raises — that means the slice window grabbed points from beyond its
      own extent, a real anchoring/slicing bug.
    * **No embedded-coord check** — a stream-copy slice inherits the
      *parent's* container ISO-6709 location tag (the drive's start), which
      does not correspond to a later slice's position; checking it would
      false-positive on any drive that moved.

    Degrade-not-fail on missing trackpoints / creation_time, mirroring
    ``assert_gpx_video_correspondence``. Raises ``GpxCorrespondenceError``
    on a genuine mismatch.
    """
    if gps_data is None:
        return
    trackpoints = getattr(gps_data, "trackpoints", None) or []
    if not trackpoints:
        logger.debug("GPX slice correspondence: no trackpoints — skipping gate")
        return

    gpx_start = trackpoints[0].timestamp
    gpx_span = trackpoints[-1].timestamp - gpx_start

    if video_creation_time is not None:
        start_delta = abs(gpx_start - video_creation_time.timestamp())
        if start_delta > GPX_START_TIME_TOLERANCE_SEC:
            raise GpxCorrespondenceError(
                f"GPX-CORRESPONDENCE(slice): GPX start-time off by {start_delta / 60:.0f} min "
                f"(> {GPX_START_TIME_TOLERANCE_SEC / 60:.0f} min) from slice creation_time "
                f"— sliced GPX likely mis-anchored to its video."
            )

    if video_duration and video_duration > 0:
        over = gpx_span - video_duration
        span_tol = max(GPX_SPAN_ABS_TOLERANCE_SEC, GPX_SPAN_REL_TOLERANCE * video_duration)
        if over > span_tol:
            raise GpxCorrespondenceError(
                f"GPX-CORRESPONDENCE(slice): GPX span {gpx_span:.0f}s over-covers video "
                f"duration {video_duration:.0f}s by {over:.0f}s (> {span_tol:.0f}s tolerance) "
                f"— slice window grabbed points beyond its own extent."
            )


def gps_coverage_window(
    gps_data,
    video_creation_time: Optional["datetime.datetime"],
) -> Optional[tuple[float, float]]:
    """Per-clip GPS coverage window in **video time** (read-only, metadata).

    Returns ``(first_fix_offset_s, last_fix_offset_s)`` — the in-video
    seconds (relative to the container ``creation_time``, i.e. video
    ``t=0``) spanned by real GPS fixes. Anchoring on ``creation_time`` (NOT
    the GPX first fix) is deliberate: a nonzero ``first_fix_offset_s``
    exposes an **unlocked GPS prefix** — the clip's opening seconds carry no
    fix, so any first-fix-based footage→GPS mapping is shifted by that much
    (the failure mode).

    Leading GoPro cold-start anomalies (null-island ``(0,0)`` and stale-lock
    runs) are skipped via :func:`effective_offset`, so the window opens at
    the first *real* fix. Purely diagnostic — it derives no labels and
    mutates nothing. Degrades to ``None`` when the creation_time anchor or
    the fixes are unavailable.
    """
    if gps_data is None or video_creation_time is None:
        return None
    trackpoints = getattr(gps_data, "trackpoints", None) or []
    if not trackpoints:
        return None

    from ..analysis.shorts_models import effective_offset

    anchor = video_creation_time.timestamp()
    first_valid_ts = effective_offset(gps_data, trackpoints[0].timestamp)
    last_ts = trackpoints[-1].timestamp
    return (first_valid_ts - anchor, last_ts - anchor)


def resolve_corresponding_gpx(
    video_path: Path,
    explicit_gpx: Optional[Path] = None,
    video_duration: float = 0.0,
) -> Optional[Path]:
    """Resolve which GPX a standalone stage may adopt for ``video_path``.

    Standalone stages (chapters/shorts/metadata) historically did
    ``glob("*.gpx")[0]`` — adopting an arbitrary sidecar by filesystem
    order, unvalidated. A wrong-drive GPX left in the folder then silently
    drove chapters, the title overwrite, per-clip shorts labels and the
    recording date. The drive path already gates external GPX with
    ``assert_gpx_video_correspondence``; this reuses that gate. Precedence:

    1. **Explicit** ``explicit_gpx`` (operator ``--gps`` / ``ctx.gps_path``)
       — authoritative; returned when it exists, else ``None`` (no silent
       fall-through to a glob).
    2. **Exact ``<video-stem>.gpx`` sidecar** — trusted by name.
    3. **Arbitrary ``*.gpx``** in the video's folder — adopted ONLY when it
       passes ``assert_gpx_video_correspondence`` against the video;
       otherwise refused. First candidate that passes wins.

    Returns an existing, trustworthy GPX ``Path`` or ``None``.
    """
    if explicit_gpx is not None:
        explicit_gpx = Path(explicit_gpx)
        return explicit_gpx if explicit_gpx.exists() else None

    sidecar = video_path.with_suffix(".gpx")
    if sidecar.exists():
        return sidecar

    candidates = sorted(video_path.parent.glob("*.gpx"))
    if not candidates:
        return None

    from ..analysis.gps_parser import parse_gpx_file

    creation_time = extract_recording_date(video_path)
    embedded = extract_location_coords(video_path)
    for candidate in candidates:
        try:
            gps_data = parse_gpx_file(candidate)
        except Exception as exc:
            logger.warning("  Skipping unparseable GPX %s: %s", candidate.name, exc)
            continue
        try:
            assert_gpx_video_correspondence(
                gps_data, creation_time, video_duration, embedded_coord=embedded
            )
        except GpxCorrespondenceError as exc:
            logger.warning(
                "  Refusing arbitrary GPX %s (no video correspondence): %s", candidate.name, exc
            )
            continue
        return candidate
    return None


def detect_gpmf_stream(video_path: Path) -> bool:
    """Check if an MP4 file contains a GoPro GPMF telemetry stream."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False

        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            codec_tag = stream.get("codec_tag_string", "")
            codec_name = stream.get("codec_name", "")
            if codec_tag == "gpmd" or codec_name == "gpmd":
                return True

        return False
    except Exception as e:
        logger.debug(f"GPMF detection failed for {video_path}: {e}")
        return False


def iter_stamped_samples(fm, media_offset_s: float, keep, step=None):
    """Yield (media_s, dt, lat, lon, ele_m) per kept fix of a FrameMeta.

    Same selection as framemeta_to_gpx: sample at step FIRST, then apply
    keep (locked-fix filter). media_s is the fix's pts within its own MP4
    (Entry.timestamp, milliseconds) plus media_offset_s — that MP4's start
    in the episode concat. None when the entry carries no timestamp.
    """
    step = step if step is not None else datetime.timedelta(seconds=1)
    for e in fm.items(step=step):
        if not keep(e):
            continue
        ts = e.timestamp
        media_s = None if ts is None else media_offset_s + float(ts.magnitude) / 1000.0
        yield (media_s, e.dt, e.point.lat, e.point.lon, e.alt.to("m").magnitude)


def build_stamped_gpx(samples):
    """Build a GPX whose trackpoints carry <extensions><gms_media_s>.

    samples: iterable of (media_s | None, dt, lat, lon, ele_m). The
    wall-clock <time> stays as-is; media_s records the fix's footage
    second in the episode concat timeline so consumers map footage -> GPS time
    without assuming the first fix arrives at record start. The tag is
    namespace-free (see shorts_models.GPX_MEDIA_S_TAG).
    """
    import xml.etree.ElementTree as ET  # noqa: N817  building our own extension element

    import gpxpy
    import gpxpy.gpx

    from ..analysis.shorts_models import GPX_MEDIA_S_TAG

    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)
    for media_s, dt, lat, lon, ele in samples:
        pt = gpxpy.gpx.GPXTrackPoint(time=dt, latitude=lat, longitude=lon, elevation=ele)
        if media_s is not None:
            el = ET.Element(GPX_MEDIA_S_TAG)
            el.text = f"{media_s:.3f}"
            pt.extensions.append(el)
        segment.points.append(pt)
    return gpx


def extract_gopro_gpx(
    video_path: Path, output_dir: Path, media_offset_s: float = 0.0
) -> Optional[Path]:
    """Extract GoPro GPMF telemetry to GPX format.

    Uses gopro-overlay library to parse GPMF data from GoPro MP4 files.
    Handles GoPro's dual audio streams by manually selecting the default.

    Args:
        video_path: Path to the GoPro video file
        output_dir: Directory to write the extracted GPX file
        media_offset_s: Start of this MP4 in the episode concat timeline; added
            to each fix's pts to give its gms_media_s

    Returns:
        Path to the extracted GPX file, or None on failure
    """
    try:
        from gopro_overlay import gpmd_filters
        from gopro_overlay.counter import ReasonCounter
        from gopro_overlay.dimensions import Dimension
        from gopro_overlay.ffmpeg import FFMPEG
        from gopro_overlay.ffmpeg_gopro import (
            AudioStream,
            DataStream,
            FFMPEGGoPro,
            GoproRecording,
            VideoStream,
            filestat,
        )
        from gopro_overlay.framemeta_gpmd import parse_gopro
        from gopro_overlay.gpmf import GPS_FIXED_VALUES
        from gopro_overlay.timeunits import timeunits
        from gopro_overlay.units import units
    except ImportError:
        logger.error("  gopro-overlay not installed — install with: pip install gopro-overlay")
        return None

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        gpx_path = output_dir / f"{video_path.stem}_gopro.gpx"

        ffmpeg = FFMPEG(None)

        # Probe streams
        output = (
            ffmpeg.ffprobe()
            .invoke(["-hide_banner", "-print_format", "json", "-show_streams", str(video_path)])
            .stdout
        )
        streams = json.loads(str(output))["streams"]

        # Select streams — handle GoPro's dual audio by filtering disposition.default
        video_streams = [
            s for s in streams if s["codec_type"] == "video" and s["disposition"]["default"] == 1
        ]
        audio_streams = [
            s for s in streams if s["codec_type"] == "audio" and s["disposition"]["default"] == 1
        ]
        data_streams = [s for s in streams if s.get("codec_tag_string") == "gpmd"]

        if not video_streams or not data_streams:
            logger.warning(f"  Missing video or GPMF stream in {video_path.name}")
            return None

        if not audio_streams:
            # Fall back to first audio stream if none marked default
            audio_streams = [s for s in streams if s["codec_type"] == "audio"]

        video = video_streams[0]
        audio = audio_streams[0] if audio_streams else None
        data = data_streams[0]

        avg_fr = video["avg_frame_rate"].split("/")
        video_stream = VideoStream(
            stream=int(video["index"]),
            dimension=Dimension(video["width"], video["height"]),
            duration=timeunits(seconds=float(video["duration"])),
            frame_count=int(video["nb_frames"]),
            frame_rate_numerator=int(avg_fr[0]),
            frame_rate_denominator=int(avg_fr[1]),
        )

        ffmpeg_gopro = FFMPEGGoPro(ffmpeg)
        data_stream = DataStream(
            stream=int(data["index"]),
            frame_duration=ffmpeg_gopro.find_frame_duration(video_path, int(data["index"])),
            frame_count=int(video["nb_frames"]),
            timebase=int(data.get("time_base", "1/1000").split("/")[1]),
        )

        audio_stream = AudioStream(stream=int(audio["index"])) if audio else AudioStream(stream=1)

        recording = GoproRecording(
            ffmpeg=ffmpeg,
            location=video_path,
            file=filestat(video_path),
            video=video_stream,
            audio=audio_stream,
            data=data_stream,
        )

        # Parse GPMF and convert to GPX
        counter = ReasonCounter()
        fm = parse_gopro(
            recording.load_data(),
            units,
            recording.data,
            gps_lock_filter=gpmd_filters.standard(
                dop_max=10,
                # Glitch bound, not a cruising cap: driving footage routinely
                # exceeds the library's 60 kph cycling default, which relabelled
                # every highway fix NO-lock and dropped it from the GPX. ~200 kph
                # still discards teleport readings without losing real highway.
                speed_max=units.Quantity(200, "kph"),
                report=counter.because,
            ),
        )
        # Only LOCKED fixes reach the canonical GPX. The lock filter above
        # merely RELABELS a bad reading's gpsfix (it never drops the point), so
        # without this filter pre-lock stale-cache coordinates from the camera's
        # previous session — non-(0,0), invisible to null-island guards — would
        # pollute county voting / chapters / endpoints. GPX has no fix channel,
        # so no downstream consumer could recover the classification.
        gpx = build_stamped_gpx(
            iter_stamped_samples(
                fm,
                media_offset_s,
                lambda e: e.gpsfix in GPS_FIXED_VALUES,
                step=datetime.timedelta(seconds=1),
            )
        )

        # Surface the otherwise write-only ReasonCounter: report how many
        # readings the lock filter rejected and why.
        dropped = sum(counter.values()) if counter else 0
        if counter:
            reasons = "; ".join(f"{reason}: {n}" for reason, n in sorted(counter.items()))
            logger.info(
                f"  GoPro GPS lock filter rejected {dropped} unlocked reading(s), "
                f"excluded from GPX ({reasons})"
            )

        # A recording that never achieved a GPS lock (every reading gpsfix=0)
        # survives the filter with zero trackpoints. Writing that GPX would
        # look like a real, GPS-having LF downstream (lf_has_gps=True) and
        # trip the GPS-required chapter-density gate instead of the
        # GPS-less skip path. Refuse to write it and report None instead.
        if gpx.get_points_no() == 0:
            logger.warning(
                f"  No GPS lock achieved in {video_path.name} — "
                f"{dropped} reading(s) rejected by lock filter, no GPX written"
            )
            if gpx_path.exists():
                gpx_path.unlink()
            return None

        atomic_write_text(gpx_path, gpx.to_xml())

        if gpx_path.exists() and gpx_path.stat().st_size > 0:
            logger.info(f"  Extracted GoPro GPS: {gpx_path.name}")
            return gpx_path

        logger.warning(f"  No GPS data extracted from {video_path.name}")
        return None

    except Exception as e:
        logger.warning(f"  GoPro GPS extraction failed for {video_path.name}: {e}")
        return None


def extract_location_coords(video_path: Path) -> Optional[tuple[float, float]]:
    """Extract GPS coordinates from video file metadata via ffprobe.

    GoPro embeds a `location` tag in the container metadata in ISO 6709 format
    (e.g., "+32.4231-080.4675/"). Quick way to get approximate coordinates
    without parsing the full GPMF telemetry stream.

    Args:
        video_path: Path to the video file

    Returns:
        Tuple of (latitude, longitude), or None if not found
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags=location,location-eng",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})
        loc_str = tags.get("location") or tags.get("location-eng")

        if not loc_str:
            return None

        match = re.match(r"([+-]\d+\.?\d*?)([+-]\d+\.?\d*)", loc_str)
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            return (lat, lon)

        return None
    except Exception as e:
        logger.debug(f"Location extraction failed for {video_path}: {e}")
        return None


def extract_recording_date(video_path: Path) -> Optional[datetime.datetime]:
    """Extract recording date from video file metadata via ffprobe.

    Reads the creation_time tag from the container format metadata.
    Works with GoPro, DJI, and most modern cameras.

    Args:
        video_path: Path to the video file

    Returns:
        Recording datetime, or None if not found
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags=creation_time",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        creation_time = data.get("format", {}).get("tags", {}).get("creation_time")

        if creation_time:
            for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"]:
                try:
                    parsed = datetime.datetime.strptime(creation_time, fmt)
                except ValueError:
                    continue
                # ffmpeg writes the container creation_time tag in UTC (the
                # trailing 'Z'). The 'Z' is a *literal* in the format strings
                # above (not %z), so strptime returns a NAIVE datetime, and
                # the zone-less space form is UTC too. Attach UTC so
                # ``.timestamp()`` yields the true epoch on any host TZ — the
                # GPX↔video correspondence gate compares it against the GPX's
                # UTC start, and a naive value would skew by the host offset.
                return parsed.replace(tzinfo=datetime.timezone.utc)

        return None
    except Exception as e:
        logger.debug(f"Recording date extraction failed for {video_path}: {e}")
        return None
