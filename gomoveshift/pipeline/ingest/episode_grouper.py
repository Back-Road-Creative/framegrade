"""Episode grouping logic — combine recordings into publishable episodes.

Groups GoPro recordings into episodes based on time proximity and duration
limits. Single recordings over the maximum duration remain standalone.
GPS-based splitting is handled separately after GPS extraction.
"""

import logging
from datetime import datetime

from .gopro_parser import (
    MAX_EPISODE_DURATION,
    MAX_GAP_BETWEEN_RECORDINGS,
    MIN_EPISODE_DURATION,
    Episode,
    GoProRecording,
)

logger = logging.getLogger(__name__)


def group_recordings_into_episodes(
    recordings: dict[int, GoProRecording],
    max_episode_duration: float = MAX_EPISODE_DURATION,
    max_gap: float = MAX_GAP_BETWEEN_RECORDINGS,
) -> list[Episode]:
    """Group recordings into episodes based on time proximity and duration limits.

    Rules:
    - Same-date recordings with gap < max_gap -> candidates for combining
    - Only combine if total stays under max_episode_duration
    - Single recordings over max_episode_duration remain as standalone episodes
      (GPS-based splitting handled separately after GPS extraction)
    """
    if not recordings:
        return []

    # Sort recordings chronologically. ``creation_time`` is now tz-aware
    # (UTC) when present but ``None`` when ffprobe couldn't read the tag;
    # segregate the missing ones with a boolean so the sort never compares a
    # tz-aware datetime against the naive ``datetime.min`` fallback (which
    # would raise ``TypeError``). Missing-time recordings still sort first,
    # preserving the prior ordering.
    sorted_recs = sorted(
        recordings.values(),
        key=lambda r: (r.creation_time is not None, r.creation_time or datetime.min),
    )

    episodes: list[Episode] = []
    current_group: list[GoProRecording] = [sorted_recs[0]]

    for rec in sorted_recs[1:]:
        prev = current_group[-1]
        prev_end = prev.end_time
        curr_start = rec.creation_time

        can_combine = True

        # Check time gap
        if prev_end and curr_start:
            gap = (curr_start - prev_end).total_seconds()
            if gap > max_gap:
                can_combine = False
        else:
            can_combine = False

        # Check combined duration
        group_duration = sum(r.total_duration for r in current_group)
        if group_duration + rec.total_duration > max_episode_duration:
            can_combine = False

        if can_combine:
            current_group.append(rec)
        else:
            # Finalize current group
            episodes.append(_make_episode(len(episodes) + 1, current_group))
            current_group = [rec]

    # Finalize last group
    if current_group:
        episodes.append(_make_episode(len(episodes) + 1, current_group))

    # Filter out episodes shorter than minimum publishable duration
    kept = []
    for ep in episodes:
        if ep.total_duration < MIN_EPISODE_DURATION:
            logger.info(
                f"  Dropping episode {ep.index} ({ep.total_duration:.0f}s) "
                f"— under {MIN_EPISODE_DURATION}s minimum"
            )
        else:
            kept.append(ep)

    # Re-number after filtering
    for i, ep in enumerate(kept, 1):
        ep.index = i

    return kept


def _make_episode(index: int, recordings: list[GoProRecording]) -> Episode:
    """Create an Episode from a group of recordings."""
    ct = recordings[0].creation_time
    return Episode(
        index=index,
        recordings=recordings,
        date=ct.strftime("%Y-%m-%d") if ct else "",
    )
