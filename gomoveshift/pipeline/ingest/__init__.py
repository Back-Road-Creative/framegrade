"""GoPro multi-clip ingestion — reassemble chaptered files into pipeline-ready episodes.

This package decomposes the monolithic ingest module into focused submodules:
  - gopro_parser: GoPro filename parsing, data models, directory scanning
  - episode_grouper: Recording-to-episode grouping logic
  - concatenator: FFmpeg concat, GPX merge, episode splitting
  - geocoding: GPS-based location derivation
  - ost_formatter: On-screen text formatting from route analysis
  - batch_runner: Batch pipeline execution across episodes
"""

import logging
from pathlib import Path

# Re-export everything for backward compatibility with:
#   from gomoveshift.pipeline.ingest import <name>
from ..geo_constants import US_STATE_ABBREVS as _US_STATE_ABBREVS
from .batch_runner import run_batch_pipeline
from .concatenator import (
    _extract_gpx_time_range,
    _merge_gpx_files,
    _split_episode_at_points,
    concat_episode,
    extract_and_merge_gpx,
)
from .episode_grouper import group_recordings_into_episodes
from .geocoding import (
    _get_route_analyzer,
    _get_shared_geocoder,
    derive_location_from_gpx,
)
from .gopro_parser import (
    GOPRO_RE,
    GPS_RETURN_RADIUS_KM,
    GPS_STOP_DURATION,
    MAX_EPISODE_DURATION,
    MAX_GAP_BETWEEN_RECORDINGS,
    MIN_CLIP_DURATION,
    MIN_EPISODE_DURATION,
    TARGET_EPISODE_DURATION,
    Episode,
    GoProChapter,
    GoProRecording,
    IngestPlan,
    _haversine_km,
    find_gps_split_points,
    get_clip_duration,
    parse_gopro_filename,
    scan_gopro_directory,
    scan_video_directory,
)
from .ost_formatter import (
    _drive_ost_from_episode_locations,
    _format_ost_from_route,
    _geocode_town_state,
    _is_junk_town,
    _parse_route_ost_endpoints,
    _select_endpoint_towns,
    _state_abbrev,
    derive_drive_ost_text,
    derive_episode_ost_text,
)

_logger = logging.getLogger(__name__)


# ============================================================================
# Master Ingest Pipeline (kept here as it coordinates across submodules)
# ============================================================================


def create_ingest_plan(
    source_dir: Path,
    min_clip_duration: float = MIN_CLIP_DURATION,
    max_episode_duration: float = MAX_EPISODE_DURATION,
    max_gap: float = MAX_GAP_BETWEEN_RECORDINGS,
    mode: str = "gopro",
    extensions: list[str] | tuple[str, ...] | None = None,
) -> IngestPlan:
    """Scan a video source directory and create an episode plan.

    ``mode`` and ``extensions`` are first-class. ``mode="generic"``
    produces one singleton episode
    per accepted file (Insta360, drone, phone exports). ``mode="gopro"``
    runs the chapter/session regex grouping. ``extensions`` is the
    single source of truth for accepted file suffixes; callers should
    plumb it from ``config.drive.video_extensions``.

    Performs no file operations (no concat, no GPS extraction) — this
    is a dry-run for review before committing to the full ingest.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise ValueError(f"Not a directory: {source_dir}")

    recordings, filtered, unrecognized = scan_video_directory(
        source_dir,
        extensions=extensions if extensions is not None else [".mp4"],
        mode=mode,
        min_clip_duration=min_clip_duration,
    )

    if mode == "generic":
        # Each singleton recording becomes its own episode. Skip the
        # GoPro grouping pass entirely — generic-mode files have no
        # chapter/session relationship.
        episodes: list[Episode] = []
        for idx, rec_num in enumerate(sorted(recordings.keys()), start=1):
            rec = recordings[rec_num]
            ep = Episode(index=idx, recordings=[rec])
            ct = rec.creation_time
            if ct is not None:
                ep.date = ct.strftime("%Y-%m-%d")
            # Attach the operator-supplied sidecar GPX (``<stem>.gpx``
            # beside the source). GoPro episodes get GPS from GPMF extraction;
            # a generic (Insta360/DJI/phone) source has no GPMF, so the sidecar
            # is its only GPS. Attaching it makes ``extract_and_merge_gpx`` tag
            # it ``external`` (running the Phase-A correspondence gate) and lets
            # ``day_grouper`` derive the longitude-local day key — without it a
            # creation_time-less generic clip is silently dropped from every day
            # batch.
            if rec.chapters:
                sidecar_gpx = rec.chapters[0].path.with_suffix(".gpx")
                if sidecar_gpx.exists() and sidecar_gpx.stat().st_size > 0:
                    ep.gpx_path = sidecar_gpx
            episodes.append(ep)
    else:
        episodes = group_recordings_into_episodes(
            recordings,
            max_episode_duration=max_episode_duration,
            max_gap=max_gap,
        )

    return IngestPlan(
        source_dir=source_dir,
        episodes=episodes,
        filtered_clips=filtered,
        unrecognized_files=unrecognized,
    )


def execute_ingest(
    plan: IngestPlan,
    output_dir: Path,
    derive_locations: bool = True,
) -> IngestPlan:
    """Execute an ingest plan: concatenate chapters, extract GPS, derive locations.

    Modifies episodes in-place with concat_path, gpx_path, location.

    Args:
        plan: The ingest plan from create_ingest_plan()
        output_dir: Directory for concatenated outputs and GPX files
        derive_locations: Whether to reverse-geocode locations from GPS

    Returns:
        The same plan with paths and locations filled in.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stages_dir = output_dir / "_ingest_stages"
    stages_dir.mkdir(exist_ok=True)

    # Snapshot the list before iteration so mid-loop splits don't cause the
    # iterator to visit the newly-inserted sub-episodes a second time.
    # Replacements are applied in reverse-index order after the loop so
    # earlier insertions don't shift the indices recorded for later ones.
    replacements: list[tuple[int, list]] = []

    for idx, episode in enumerate(list(plan.episodes)):
        _logger.info(f"\n{'=' * 60}")
        _logger.info(f"  Episode {episode.index}: {episode.total_duration_min:.0f}min")
        _logger.info(f"{'=' * 60}")

        # 1. Concatenate chapter files
        concat_episode(episode, output_dir)

        # 2. Extract and merge GPS
        gpx = extract_and_merge_gpx(episode, stages_dir)

        # 3. Check for GPS-based splits on long episodes
        if gpx and episode.total_duration > MAX_EPISODE_DURATION:
            splits = find_gps_split_points(gpx, max_total_duration=episode.total_duration)
            if splits:
                _logger.info(
                    f"  Episode {episode.index} exceeds {MAX_EPISODE_DURATION / 60:.0f}min — "
                    f"found {len(splits)} GPS split point(s)"
                )
                sub_episodes = _split_episode_at_points(episode, splits, output_dir, stages_dir)
                if sub_episodes:
                    for sub_ep in sub_episodes:
                        if sub_ep.gpx_path and derive_locations and not sub_ep.location:
                            sub_ep.location = derive_location_from_gpx(sub_ep.gpx_path)
                    replacements.append((idx, sub_episodes))
                    continue

        # 4. Auto-derive location from GPS
        if gpx and derive_locations and not episode.location:
            episode.location = derive_location_from_gpx(gpx)

    for idx, sub_episodes in reversed(replacements):
        plan.episodes[idx : idx + 1] = sub_episodes

    # Re-number episodes after any splits
    for i, ep in enumerate(plan.episodes, 1):
        ep.index = i

    return plan


__all__ = [
    # Data models
    "GoProChapter",
    "GoProRecording",
    "Episode",
    "IngestPlan",
    # Constants
    "GOPRO_RE",
    "MIN_CLIP_DURATION",
    "MIN_EPISODE_DURATION",
    "MAX_EPISODE_DURATION",
    "TARGET_EPISODE_DURATION",
    "MAX_GAP_BETWEEN_RECORDINGS",
    "GPS_RETURN_RADIUS_KM",
    "GPS_STOP_DURATION",
    # GoPro parsing
    "parse_gopro_filename",
    "get_clip_duration",
    "scan_gopro_directory",
    "scan_video_directory",
    "find_gps_split_points",
    # Episode grouping
    "group_recordings_into_episodes",
    # Concatenation
    "concat_episode",
    "extract_and_merge_gpx",
    "_split_episode_at_points",
    "_merge_gpx_files",
    "_extract_gpx_time_range",
    # Geocoding
    "derive_location_from_gpx",
    "_get_shared_geocoder",
    "_get_route_analyzer",
    # OST formatting
    "derive_episode_ost_text",
    "derive_drive_ost_text",
    "_state_abbrev",
    "_format_ost_from_route",
    "_is_junk_town",
    "_select_endpoint_towns",
    "_haversine_km",
    "_US_STATE_ABBREVS",
    "_drive_ost_from_episode_locations",
    "_geocode_town_state",
    "_parse_route_ost_endpoints",
    # Master ingest
    "create_ingest_plan",
    "execute_ingest",
    # Batch runner
    "run_batch_pipeline",
]
