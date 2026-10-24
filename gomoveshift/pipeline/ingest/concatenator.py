"""Episode concatenation and GPS extraction.

Handles FFmpeg-based stream-copy concatenation of GoPro chapter files,
GPX extraction and merging, and episode splitting at GPS-determined points.
"""

import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from keycut import probe_keyframes, snap_start_to_keyframe

from ...analysis.shorts_models import GPX_MEDIA_S_TAG
from .._atomic_write import atomic_write_text
from ..ffmpeg_runner import run_ffmpeg_retry_enomem
from .gopro_parser import Episode, _is_invalid_coord, get_clip_duration

logger = logging.getLogger(__name__)

#: A cached concat must probe to at least this share of the episode's summed
#: chapter durations. A stream copy matches within a second; a run killed
#: mid-concat leaves an MP4 with no moov atom, which probes to 0.
_CACHED_CONCAT_MIN_FRACTION = 0.98


def _align_splits_to_keyframes(
    source: Path,
    points: list[float],
    total_duration: float,
) -> list[float]:
    """Move every interior split point FORWARD onto a real source keyframe.

    ``-ss <t> -i src … -c copy`` cannot start on a non-keyframe, so ffmpeg
    silently rewinds to the previous one. Every slice below is cut by its own
    independent ffmpeg call against the same source, so slice N stops at the
    point it was given while slice N+1 *begins* up to one GOP earlier — and
    after the day concat that overlap plays twice, at every splice. Snapping the
    shared boundary forward onto a keyframe makes the rewind a no-op, so the
    slices tile exactly: slice N ends precisely where slice N+1 starts.

    Forward, never backward — backward is the bug. Cost is at most one GOP of
    footage moved from one side of a splice to the other; nothing is dropped.

    ``probe_keyframes`` returning ``None`` means "unknown", not "no keyframes":
    the points come back untouched (the pre-fix behaviour) rather than refusing
    to split a multi-hour episode.
    """
    keyframes = probe_keyframes(source)
    if not keyframes:
        logger.warning(
            "  Keyframe positions unknown for %s — split points left unaligned; "
            "consecutive slices may overlap by up to one GOP.",
            source.name,
        )
        return points

    aligned = [points[0]]
    for p in points[1:-1]:
        aligned.append(round(snap_start_to_keyframe(p, total_duration, keyframes), 3))
    aligned.append(points[-1])

    # Two split points inside one GOP snap onto the same keyframe — keep the
    # sequence strictly increasing so no zero-length slice is emitted.
    deduped: list[float] = []
    for p in aligned:
        if not deduped or p > deduped[-1]:
            deduped.append(p)
    return deduped


def concat_episode(
    episode: Episode,
    output_dir: Path,
) -> Path:
    """Concatenate all chapter files in an episode into one MP4 (stream copy).

    Uses FFmpeg concat demuxer — no re-encoding.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Splitter-set concat_path takes priority: a
    # slice-derived sub-Episode arrives with concat_path already pointing
    # at its sliced bytes (e.g. ``episode_04_a.mp4``). The slice IS the
    # episode's source-of-truth. Re-running ffmpeg concat would attempt
    # to rebuild from chapter_files, which is empty on sub-episodes →
    # ffmpeg fails with no input. Honour the pre-set path and skip.
    if (
        episode.concat_path is not None
        and episode.concat_path.exists()
        and episode.concat_path.stat().st_size > 0
    ):
        logger.info(
            "  Episode %s: using pre-set concat_path → %s",
            episode.index,
            episode.concat_path.name,
        )
        return episode.concat_path

    output_path = output_dir / f"episode_{episode.index:02d}_{episode.date}.mp4"

    # Skip if already concatenated (restart safety) — only a whole, readable concat.
    if output_path.exists() and output_path.stat().st_size > 0:
        if _cached_concat_is_whole(output_path, episode):
            logger.info(f"  Episode {episode.index}: concat already exists → {output_path.name}")
            episode.concat_path = output_path
            return output_path
        output_path.replace(output_path.with_name(output_path.name + ".invalid"))

    files = episode.chapter_files()
    if len(files) == 1:
        # Single file — symlink to avoid copying 10-17GB unnecessarily
        output_path.symlink_to(files[0])
        logger.info(f"  Episode {episode.index}: single file linked → {output_path.name}")
        episode.concat_path = output_path
        return output_path

    # ffmpeg writes a partial name, renamed only on success: a run stopped
    # mid-concat never leaves a truncated file at the name the next run reuses.
    partial_path = output_dir / f"episode_{episode.index:02d}_{episode.date}.partial.mp4"

    # Write concat list
    concat_list = output_dir / f"episode_{episode.index:02d}_concat.txt"
    with open(concat_list, "w") as f:
        for fp in files:
            escaped = str(fp).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(partial_path),
    ]

    logger.info(
        f"  Episode {episode.index}: concatenating {len(files)} files "
        f"({episode.total_duration_min:.0f}min) → {output_path.name}"
    )

    # 3600s timeout — GoPro 4K files can be several GB per chapter,
    # and WSL2 cross-mount I/O is slow even with stream copy
    result = run_ffmpeg_retry_enomem(cmd, partial_path, timeout=3600)
    if result.returncode != 0:
        partial_path.unlink(missing_ok=True)
        # The cause is at the end of stderr; its head is the version banner.
        raise RuntimeError(
            f"FFmpeg concat failed for episode {episode.index}: {result.stderr[-1500:]}"
        )
    partial_path.replace(output_path)

    # Clean up concat list
    concat_list.unlink(missing_ok=True)

    episode.concat_path = output_path
    return output_path


def _cached_concat_is_whole(path: Path, episode: Episode) -> bool:
    """True when a cached concat probes to (nearly) the episode's full length."""
    expected = episode.total_duration
    actual = get_clip_duration(path)
    floor = expected * _CACHED_CONCAT_MIN_FRACTION if expected > 0 else 0.0
    if actual > 0 and actual >= floor:
        return True
    logger.warning(
        "CONCAT-CACHE-INVALID: episode %s cached %s probes to %.1fs (expected ~%.1fs); "
        "setting it aside as %s.invalid and rebuilding",
        episode.index,
        path.name,
        actual,
        expected,
        path.name,
    )
    return False


def _gpx_has_trackpoints(gpx_path: Path) -> bool:
    """True if ``gpx_path`` parses as well-formed GPX with ≥1 trackpoint.

    Distinguishes a truncated/corrupt file (passes an exists+nonzero-size
    check but is missing its closing tags or track data) from a genuinely
    usable one — the exact shape a mid-write crash leaves behind.
    """
    try:
        import gpxpy

        with open(gpx_path, encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
    except Exception:
        return False
    return any(seg.points for track in gpx.tracks for seg in track.segments)


def extract_and_merge_gpx(
    episode: Episode,
    output_dir: Path,
) -> Optional[Path]:
    """Extract GPS from each chapter file and merge into one GPX per episode.

    Each chapter's GPMF telemetry is extracted via gopro2gpx, then the resulting
    GPX track segments are concatenated into a single GPX file.
    """
    from ..gopro_gps import detect_gpmf_stream, extract_gopro_gpx

    output_dir.mkdir(parents=True, exist_ok=True)

    # Splitter-set gpx_path takes priority: a sub-
    # Episode arrives with gpx_path already pointing at its sliced GPX
    # derived from the parent's merged GPX. Re-extracting from chapter
    # files would fail (sub-eps have empty recordings). Honour and skip.
    is_sub_episode = not episode.recordings and episode._explicit_total_duration is not None
    gpx_preset = (
        episode.gpx_path is not None
        and episode.gpx_path.exists()
        and episode.gpx_path.stat().st_size > 0
    )
    # exists+nonzero-size is too weak — a crash-truncated write (missing
    # closing tags / no trackpoints) still passes that check and would be
    # adopted forever. Parse it (the module already parses GPX at every
    # other site, so this adds no dependency; an episode-scoped GPX is
    # minutes of 1Hz fixes, negligible next to the ffmpeg work already
    # here). A sub-episode has no chapter recordings to fall back to, so a
    # corrupt pre-set GPX is a hard failure naming the bad path rather than
    # a silent downstream GPS corruption; a chapter-derived episode simply
    # discards it and re-extracts below.
    if gpx_preset and not _gpx_has_trackpoints(episode.gpx_path):
        if is_sub_episode:
            raise RuntimeError(
                f"Episode {episode.index}: pre-set gpx_path {episode.gpx_path} is "
                "truncated or corrupt (fails GPX parse or has no trackpoints) and "
                "cannot be re-extracted — sub-episodes have no chapter recordings "
                "to fall back to."
            )
        logger.warning(
            "  Episode %s: pre-set gpx_path %s is truncated/corrupt (nonzero size, "
            "no usable trackpoints) — discarding and re-extracting from chapters",
            episode.index,
            episode.gpx_path.name,
        )
        gpx_preset = False

    if gpx_preset:
        logger.info(
            "  Episode %s: using pre-set gpx_path → %s",
            episode.index,
            episode.gpx_path.name,
        )
        # A pre-set GPX on a *chapter-derived* episode is an
        # operator-supplied (or resume-state) sidecar — NOT extracted from
        # this episode's own chapter GPMF, so its temporal/spatial
        # alignment to the video is not guaranteed. Mark it ``external`` so
        # Phase A runs the full correspondence gate. A sub-episode slice
        # (``recordings == []``, ``_explicit_*`` set) was cut from the
        # parent's own merged GPMF track — it can't be a swapped track, but
        # a gross anchoring/slicing error can still mis-window it, so
        # tag it ``gpmf_slice`` for Phase A's slice-tuned gate (this mirrors
        # the born-tag the splitter sets on first ingest, for the resume
        # path where the sub-episode re-enters here with its GPX pre-set).
        episode.gpx_source = "gpmf_slice" if is_sub_episode else "external"
        return episode.gpx_path

    gpx_parts: list[Path] = []

    # Footage second where each chapter starts in the concat (same order as
    # Episode.chapter_files). Advanced for EVERY chapter — including ones
    # skipped for no GPMF or no GPS lock — because the concat still contains
    # their footage; skipping the advance would shift every later chapter's fixes.
    media_offset = 0.0
    for rec in episode.recordings:
        for chapter in sorted(rec.chapters, key=lambda c: c.chapter_num):
            chapter_offset = media_offset
            media_offset += chapter.duration
            if not detect_gpmf_stream(chapter.path):
                logger.debug(f"    No GPMF in {chapter.path.name}")
                continue

            gpx = extract_gopro_gpx(chapter.path, output_dir, media_offset_s=chapter_offset)
            if gpx:
                gpx_parts.append(gpx)

    if not gpx_parts:
        logger.warning(f"  Episode {episode.index}: no GPS data extracted")
        return None

    if len(gpx_parts) == 1:
        merged_path = output_dir / f"episode_{episode.index:02d}_{episode.date}.gpx"
        gpx_parts[0].rename(merged_path)
        episode.gpx_path = merged_path
        return merged_path

    # Merge multiple GPX files: parse all, combine track segments
    merged_path = _merge_gpx_files(gpx_parts, output_dir, episode)
    episode.gpx_path = merged_path

    # Clean up individual GPX parts
    for part in gpx_parts:
        if part.exists() and part != merged_path:
            part.unlink(missing_ok=True)

    return merged_path


def _merge_gpx_files(
    gpx_files: list[Path],
    output_dir: Path,
    episode: Episode,
) -> Path:
    """Merge multiple GPX files into one by combining track segments."""
    import gpxpy
    import gpxpy.gpx

    merged = gpxpy.gpx.GPX()
    merged_track = gpxpy.gpx.GPXTrack()
    merged.tracks.append(merged_track)

    for gpx_file in gpx_files:
        try:
            with open(gpx_file, encoding="utf-8") as f:
                gpx = gpxpy.parse(f)
            for track in gpx.tracks:
                for segment in track.segments:
                    merged_track.segments.append(segment)
        except Exception as e:
            logger.warning(f"    Failed to parse {gpx_file.name}: {e}")
            continue

    merged_path = output_dir / f"episode_{episode.index:02d}_{episode.date}.gpx"
    atomic_write_text(merged_path, merged.to_xml())

    logger.info(
        f"  Episode {episode.index}: merged {len(gpx_files)} GPX files → "
        f"{merged_path.name} ({len(merged_track.segments)} segments)"
    )

    return merged_path


def _first_valid_point_time(points) -> Optional[datetime]:
    """Time of the first track-order point with a valid coord.

    Mirrors the origin ``find_gps_split_points`` uses
    (``_sanitize_waypoints(...)[0]``): a leading invalid-coord fix (GPS-lock
    null-island) is skipped, so this never anchors on a corrupt point the
    way a raw ``min()`` over timestamps would.
    """
    for p in points:
        if p.time is not None and not _is_invalid_coord(p.latitude, p.longitude):
            return p.time
    return None


def _point_media_s(point) -> Optional[float]:
    """The gms_media_s stamp of a gpxpy point, or None when unstamped."""
    for el in getattr(point, "extensions", None) or []:
        if str(el.tag).rsplit("}", 1)[-1] == GPX_MEDIA_S_TAG:
            try:
                return float(el.text)
            except (TypeError, ValueError):
                return None
    return None


def _rebase_point_media_s(point, delta: float) -> None:
    for el in point.extensions:
        if str(el.tag).rsplit("}", 1)[-1] == GPX_MEDIA_S_TAG:
            el.text = f"{float(el.text) - delta:.3f}"


def _first_valid_media_s(points) -> Optional[float]:
    """Footage second of the first valid fix when EVERY point is stamped."""
    if not points or any(_point_media_s(p) is None for p in points):
        return None
    for p in points:
        if p.time is not None and not _is_invalid_coord(p.latitude, p.longitude):
            return _point_media_s(p)
    return None


def _read_first_fix_media_s(gpx_path: Path) -> Optional[float]:
    """Footage second of the first sanitized fix (stamped GPX); None otherwise."""
    try:
        import gpxpy

        with open(gpx_path, encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
        points = [pt for track in gpx.tracks for seg in track.segments for pt in seg.points]
        return _first_valid_media_s(points)
    except Exception as e:
        logger.debug("  first-fix media read failed for %s: %s", gpx_path, e)
        return None


def _read_first_fix_time(gpx_path: Path) -> Optional[datetime]:
    """First sanitized fix time in ``gpx_path``; None on any parse failure."""
    try:
        import gpxpy

        with open(gpx_path, encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
        points = [pt for track in gpx.tracks for seg in track.segments for pt in seg.points]
        return _first_valid_point_time(points)
    except Exception as e:
        logger.debug("  first-fix read failed for %s: %s", gpx_path, e)
        return None


def _split_episode_at_points(
    episode: Episode,
    split_points: list[float],
    output_dir: Path,
    stages_dir: Path,
) -> list[Episode]:
    """Split a concatenated episode at GPS-determined timestamps.

    Uses FFmpeg stream copy to split without re-encoding.

    Returns:
        List of new Episode objects with updated paths, or empty list on failure.
    """
    if not episode.concat_path or not episode.concat_path.exists():
        return []

    source = episode.concat_path
    # Single origin: split_points come from find_gps_split_points anchored
    # to the GPX FIRST FIX, but -ss/-t below index the video whose t=0 is
    # record start (creation_time). Fold the GPS lock-delay
    # D = first_fix − creation_time into every split so each cut lands where
    # the GPS event actually was (else every cut is D early). D degrades to 0
    # when the clock or GPX is unavailable — record-start == first-fix frame.
    creation_time = getattr(episode, "creation_time", None)
    lock_delay = 0.0
    stamped_delay = (
        _read_first_fix_media_s(episode.gpx_path) if getattr(episode, "gpx_path", None) else None
    )
    if stamped_delay is not None:
        # Stamped GPX: the first fix's footage second IS the lock delay; no
        # creation_time arithmetic (only >75 min single recordings split, so
        # wall elapsed == media elapsed there anyway).
        lock_delay = max(0.0, stamped_delay)
    elif creation_time is not None and getattr(episode, "gpx_path", None):
        first_fix = _read_first_fix_time(episode.gpx_path)
        if first_fix is not None:
            try:
                lock_delay = max(0.0, (first_fix - creation_time).total_seconds())
            except TypeError:
                lock_delay = 0.0

    # Slice-site backstop: even with the
    # GPS-side filters in find_gps_split_points / _sanitize_waypoints,
    # a corrupt split offset reaching here would ask ffmpeg for a slice
    # past EOF — stream-copy then runs to the end of the source and
    # blows the 300s timeout on cross-mount I/O. Clamp + dedupe to
    # guarantee every slice fits inside the source.
    safe_points = sorted(
        {
            round(min(max(0.0, float(p) + lock_delay), episode.total_duration), 3)
            for p in split_points
        }
        | {0.0, round(episode.total_duration, 3)}
    )
    if len(safe_points) < 2:
        logger.warning(
            "  Episode %d: all split points collapsed after clamping; skipping split.",
            episode.index,
        )
        return []
    # Splice correctness: land every interior boundary on a real keyframe so the
    # stream copies below tile instead of overlapping. See the helper's docstring.
    points = _align_splits_to_keyframes(source, safe_points, episode.total_duration)
    sub_episodes: list[Episode] = []

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        duration = end - start

        sub_path = output_dir / f"episode_{episode.index:02d}_{chr(ord('a') + i)}.mp4"

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(source),
            "-t",
            str(duration),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(sub_path),
        ]

        # 3600s timeout — matches the concat path above. A single sub-slice
        # from a multi-hour day master can be tens of GB, and WSL2
        # cross-mount stream-copy I/O runs at ~100 MB/s, so a multi-GB slice
        # can outlast the legacy 300s ceiling before EOF even though the
        # splits themselves are correct.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            logger.warning(f"  Split failed at {start:.0f}s: {result.stderr[:200]}")
            return []

        # Build a self-consistent sub-Episode:
        #   * recordings=[] for ALL sub-eps — the slice file at sub_path
        #     IS the source-of-truth bytes; the parent's chapter recordings
        #     no longer describe these bytes, so leaving them attached
        #     produced wrong total_duration (the whole parent's length on a short slice)
        #     and a broken downstream concat re-attempt.
        #   * _explicit_total_duration = the slice's actual extent so
        #     ep.total_duration / total_duration_min report correctly.
        #   * _explicit_creation_time = parent_creation_time + start so
        #     day_grouper's chronological ordering still works.
        #   * concat_path = the freshly-written slice (no re-concat).
        #   * gpx_path = sliced GPX from parent's merged GPX, time-windowed
        #     to [start, end] (set below if parent had a GPX).
        parent_creation_time = episode.creation_time if episode.creation_time is not None else None
        sub_creation_time = (
            parent_creation_time + timedelta(seconds=start)
            if parent_creation_time is not None
            else None
        )
        sub_ep = Episode(
            index=episode.index * 100 + i + 1,  # Temp numbering, renumbered later
            recordings=[],
            date=episode.date,
            concat_path=sub_path,
            # Cut from the parent's own GPMF (never a swapped track), but
            # a gross anchoring/slicing error can still mis-window it — tag it
            # so Phase A runs the slice-tuned correspondence gate.
            gpx_source="gpmf_slice",
            _explicit_total_duration=duration,
            _explicit_creation_time=sub_creation_time,
        )

        # Extract GPS for the time range of this sub-episode from the
        # parent's already-merged GPX. Always attempt — even when the
        # absolute timestamp window doesn't yield waypoints, the helper
        # returns None and the sub-ep proceeds without GPS (location
        # derivation simply skips).
        if episode.gpx_path:
            sub_gpx = _extract_gpx_time_range(
                episode.gpx_path,
                start,
                end,
                stages_dir,
                f"episode_{episode.index:02d}_{chr(ord('a') + i)}.gpx",
                anchor_time=creation_time,
            )
            sub_ep.gpx_path = sub_gpx

        sub_episodes.append(sub_ep)
        logger.info(
            f"  Split {episode.index}{chr(ord('a') + i)}: "
            f"{start / 60:.1f}–{end / 60:.1f}min ({duration / 60:.1f}min) → {sub_path.name}"
        )

    # Remove the original unsplit file
    if sub_episodes and source.exists():
        source.unlink()

    return sub_episodes


def _extract_gpx_time_range(
    gpx_path: Path,
    start_seconds: float,
    end_seconds: float,
    output_dir: Path,
    output_name: str,
    anchor_time: Optional[datetime] = None,
) -> Optional[Path]:
    """Extract waypoints from a GPX file within a time range.

    ``start_seconds``/``end_seconds`` are offsets from a single origin:
    ``anchor_time`` — the video's record-start ``creation_time`` — when
    supplied, else the first SANITIZED fix. Anchoring on ``creation_time``
    keeps the sliced GPX window in the SAME frame as the ffmpeg ``-ss`` cut.
    The fallback origin is the first VALID fix (via ``_first_valid_point_time``),
    never ``min()`` over raw points, so a leading invalid-coord/stale-clock
    timestamp can't shift or empty the window. The upper bound is EXCLUSIVE so a
    boundary fix lands in exactly one of two adjacent slices.
    """
    try:
        import gpxpy
        import gpxpy.gpx

        with open(gpx_path, encoding="utf-8") as f:
            gpx = gpxpy.parse(f)

        if not gpx.tracks:
            return None

        all_points = []
        for track in gpx.tracks:
            for segment in track.segments:
                all_points.extend(segment.points)

        if not all_points:
            return None

        # Stamped GPX: slice on the footage clock and rebase so the sub-episode
        # footage starts at 0 (wall-clock anchoring is the unstamped fallback).
        stamped = all(_point_media_s(p) is not None for p in all_points)
        base_time = None
        if not stamped:
            base_time = (
                anchor_time if anchor_time is not None else _first_valid_point_time(all_points)
            )
            if base_time is None:
                return None

        filtered = gpxpy.gpx.GPX()
        filtered_track = gpxpy.gpx.GPXTrack()
        filtered.tracks.append(filtered_track)
        filtered_seg = gpxpy.gpx.GPXTrackSegment()
        filtered_track.segments.append(filtered_seg)

        for point in all_points:
            if stamped:
                if start_seconds <= _point_media_s(point) < end_seconds:
                    _rebase_point_media_s(point, start_seconds)
                    filtered_seg.points.append(point)
            elif point.time:
                offset = (point.time - base_time).total_seconds()
                if start_seconds <= offset < end_seconds:
                    filtered_seg.points.append(point)

        if not filtered_seg.points:
            return None

        out_path = output_dir / output_name
        atomic_write_text(out_path, filtered.to_xml())

        return out_path

    except Exception as e:
        logger.warning(f"  GPX time range extraction failed: {e}")
        return None
