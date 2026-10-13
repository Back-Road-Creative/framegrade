"""Episode-local trim windows for an LF plan (engine side).

``_pick_representative_episode`` and ``_build_included_segments_for_lf`` are
pure functions of an LF plan's segments. The engine (``post_encoding_invariants``)
and Phase D both need them; they live here so the engine never imports
``drive_phases.phase_d`` (``tests/test_media_boundary.py``). ``phase_d``
re-exports both names, so ``phase_d._build_included_segments_for_lf`` keeps
resolving for its callers and tests.
"""

from __future__ import annotations

from typing import Optional


def _pick_representative_episode(segments, gpx_files=None) -> Optional[int]:
    """The single representative episode for an LF: the source episode
    contributing the MOST segments (ties → lowest global episode index).

    When ``gpx_files`` is supplied, only GPX-BEARING episodes are eligible
    (an episode whose ``gpx_files`` entry is ``None`` or out of range is
    skipped). This is the ONE selection shared by both pickers —
    ``_build_per_lf_gpx_path_map`` (which resolves the representative GPX)
    and ``_build_included_segments_for_lf`` (which builds the trim window) —
    so they cannot disagree: without the shared filter, an LF whose
    segment-majority episode lacked a GPX got its window built from episode
    A while the GPX came from episode B, trimming B's trace with A's windows
    (wrong ``Through:`` counties, chapters naming cut footage, region
    checks false-failing a correct title). ``None`` when no eligible
    episode exists.
    """
    seg_counts: dict[int, int] = {}
    for seg in segments or []:
        gi = int(getattr(seg, "source_file_index", 0))
        if gpx_files is not None and (gi < 0 or gi >= len(gpx_files) or gpx_files[gi] is None):
            continue
        seg_counts[gi] = seg_counts.get(gi, 0) + 1
    if not seg_counts:
        return None
    return min(seg_counts, key=lambda g: (-seg_counts[g], g))


def _build_included_segments_for_lf(lf_plan, gpx_files=None) -> list[tuple[float, float]]:
    """Build ``(start_sec, end_sec)`` ranges in the representative GPX's
    EPISODE-LOCAL time base from an LF plan.

    Chapter generation charts each LF against ONE representative GPX — the
    episode contributing the most segments (ties → lowest source-file
    index), matching ``_build_per_lf_gpx_path_map``. That GPX's waypoints
    are keyed from the episode's own local second 0, so the trim window
    must be episode-local too. ``LFSegment.start_time`` / ``end_time`` are
    already episode-local (seconds within the source file); the drive-global
    ``original_start_time`` / ``original_end_time`` are offset by every
    prior episode's duration, so an LF past the drive's first episode would
    otherwise get a window shifted out of the GPX's range — silently keeping
    the wrong stretch or none at all. Segments from other episodes live
    in a different GPX time base and are dropped: their footage isn't charted
    by this GPX. The chapter generator uses this list to trim chapters to the surviving
    footage so titles don't advertise roads that got cut.

    Pass ``gpx_files`` (the drive-wide ``state.gpx_files`` list, indices still
    GLOBAL) so the representative is chosen among GPX-BEARING episodes ONLY —
    the SAME episode ``_build_per_lf_gpx_path_map`` resolves the GPX from, so
    the window base and the GPX trace can't come from different episodes. Omit
    it (GPS-less / legacy callers) to pick among all episodes.
    """
    segments = list(getattr(lf_plan, "segments", []) or [])
    if not segments:
        return []
    rep_idx = _pick_representative_episode(segments, gpx_files)
    if rep_idx is None:
        return []
    ranges: list[tuple[float, float]] = []
    for seg in segments:
        if int(getattr(seg, "source_file_index", 0)) != rep_idx:
            continue
        start = float(getattr(seg, "start_time", 0.0))
        end = float(getattr(seg, "end_time", start))
        if end > start:
            ranges.append((start, end))
    return ranges
